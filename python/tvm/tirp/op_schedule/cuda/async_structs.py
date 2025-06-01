# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Implementation of async structs schedules."""

import copy
from typing import Optional, List, Union
from enum import Enum

import tvm
from tvm.script import tir as T
from tvm.script import tirp as Tp
from tvm.tirp.op_schedule import ScheduleContext
from tvm.tir import Buffer, PrimFunc, BufferRegion
from tvm.tir.async_structs import CopyPipeline
from tvm.tir.stmt import OpCall
from tvm.tir.layout import SwizzleLayout, TileLayout
from .common import CopyInstType, copy_g2s_s2g_cta_vec_load_impl, target_cuda, validate_copy_op
from ..registry import register_schedule


class PipelineOp(Enum):
    """The pipeline operation."""

    PRODUCER_COMMIT = 0
    CONSUMER_WAIT = 1
    COPY = 2
    INIT = 3


class SwizzleMode(Enum):
    """The swizzle mode of the TMA"""

    SWIZZLE_NONE = 0
    SWIZZLE_32B_ATOM = 1
    SWIZZLE_64B_ATOM = 2
    SWIZZLE_128B_ATOM = 3


def tma_atom_layout(dtype: str, swizzle_mode: Union[SwizzleMode, int]):
    """Generate the TMA atom layout given dtype and swizzle mode."""
    bits = tvm.DataType(dtype).bits
    if isinstance(swizzle_mode, int):
        swizzle_mode = SwizzleMode(swizzle_mode)
    return SwizzleLayout(
        per_element=(128 // bits).bit_length() - 1,
        swizzle_len=swizzle_mode.value,
        atom_len=3,
    )


def tma_atom_shape(
    dtype: str, swizzle_mode: Union[SwizzleMode, int], shape: Optional[List[int]] = None
):
    """Generate the TMA atom shape given dtype and swizzle mode."""
    bits = tvm.DataType(dtype).bits
    if isinstance(swizzle_mode, int):
        swizzle_mode = SwizzleMode(swizzle_mode)
    atom_shape = {
        SwizzleMode.SWIZZLE_32B_ATOM: [8, 256],
        SwizzleMode.SWIZZLE_64B_ATOM: [8, 512],
        SwizzleMode.SWIZZLE_128B_ATOM: [8, 1024],
    }[swizzle_mode]
    atom_shape[-1] //= bits
    if shape is None:
        return atom_shape
    atom_shape = [1] * (len(shape) - len(atom_shape)) + atom_shape
    return atom_shape


def tma_shared_layout(dtype: str, swizzle_mode: Union[SwizzleMode, int], shape):
    """Generate the TMA layout for the shared memory given shape and dtype.
    It uses a default tiling strategy to tile the TMA atom layout into the shared memory.
    """
    if isinstance(swizzle_mode, int):
        swizzle_mode = SwizzleMode(swizzle_mode)
    if swizzle_mode == SwizzleMode.SWIZZLE_NONE:
        return TileLayout(shape).normalize()
    atom_shape = tma_atom_shape(dtype, swizzle_mode, shape)
    layout = tma_atom_layout(dtype, swizzle_mode)
    tile_to_shape = copy.copy(atom_shape)
    tile_to_shape[-2] = shape[-2]
    return layout.tile_to(tile_to_shape, atom_shape).tile_to(shape, tile_to_shape).normalize()


def tma_atom_compatible(dst_shape, dst_st, dst_extent, atom_shape):
    """Check if the copy region in dst is compatible with the TMA atom shape."""
    for i, _ in enumerate(dst_st):
        if any(x % atom_shape[i] != 0 for x in [dst_shape[i], dst_st[i], dst_extent[i]]):
            return False
    return True


def copy_tma_impl(
    pipeline: "CopyPipeline",
    dst_buffer_region: "BufferRegion",
    src_buffer_region: "BufferRegion",
    sctx: "ScheduleContext",
) -> Optional["PrimFunc"]:
    """Schedule a copy between global <‑> shared memory using CUDA TMA.

    This is a unified replacement for the previous
    ``copy_g2s_cta_tma_impl`` and ``copy_s2g_cta_tma_impl`` helpers.  The
    direction is inferred from the scopes of *src* and *dst* buffers:

    * **global → shared**  ⇒  ``cp_async.bulk.tensor.g2c``
    * **shared → global**  ⇒  ``cp_async.bulk.tensor.s2g``

    If neither pattern matches, the function returns *None* so that other
    schedule rules may attempt to handle the copy.
    """

    # ---------------------------------------------------------------------
    # Guard: CTA scope only
    # ---------------------------------------------------------------------
    if sctx.exec_scope.name != "cta":
        return None

    # ---------------------------------------------------------------------
    # Identify direction & basic legality checks
    # ---------------------------------------------------------------------
    src: "Buffer" = src_buffer_region.buffer
    dst: "Buffer" = dst_buffer_region.buffer

    src_scope, dst_scope = src.scope(), dst.scope()
    if src_scope == "global" and dst_scope.startswith("shared"):
        direction = "g2s"  # global → shared
        s_buf, g_buf = dst, src
        shared_region, global_region = dst_buffer_region, src_buffer_region
    elif src_scope.startswith("shared") and dst_scope == "global":
        direction = "s2g"  # shared → global
        s_buf, g_buf = src, dst
        shared_region, global_region = src_buffer_region, dst_buffer_region
    else:
        # Unsupported combination (e.g. global→global, shared→shared, etc.)
        return None

    # For now, we require that the global side layout is trivial.
    # TODO(bohan): support strided global memory in the future.
    if not g_buf.layout.is_trivial():
        return None

    # ---------------------------------------------------------------------
    # Region metadata & axis‑matching between global and shared coordinates
    # ---------------------------------------------------------------------
    g_st = [r.min for r in global_region.region]
    g_ext = [r.extent for r in global_region.region]
    s_st = [r.min for r in shared_region.region]
    s_ext = [r.extent for r in shared_region.region]

    def dim_match(a_ext: List[int], b_ext: List[int], rank: int) -> List[int]:
        """Return a map from *a*'s logical axes to *b*'s axes ignoring unit dims."""
        a_nu = [(i, e) for i, e in enumerate(a_ext) if e != 1]
        b_nu = [(i, e) for i, e in enumerate(b_ext) if e != 1]
        axis_map = [-1] * rank
        for (ai, _), (bi, _) in zip(a_nu, b_nu):
            axis_map[ai] = bi
        return axis_map

    # Map *global axis* → *shared axis*
    axis_map = dim_match(g_ext, s_ext, len(g_buf.shape))

    # ---------------------------------------------------------------------
    # Determine swizzle mode for the *shared* side (if any)
    # ---------------------------------------------------------------------
    swizzle_mode = None  # type: Optional["SwizzleMode"]
    box_dim: List[int]

    if s_buf.layout.is_trivial():
        # No swizzling – straightforward copy of a rectangular box.
        swizzle_mode = SwizzleMode.SWIZZLE_NONE
        box_dim = g_ext
    else:
        # Try the standard TMA atom swizzles in decreasing size.
        for mode in (
            SwizzleMode.SWIZZLE_128B_ATOM,
            SwizzleMode.SWIZZLE_64B_ATOM,
            SwizzleMode.SWIZZLE_32B_ATOM,
        ):
            swizzle_atom = tma_atom_layout(s_buf.dtype, mode)
            atom_shape = tma_atom_shape(s_buf.dtype, mode, s_buf.shape)
            outer = swizzle_atom.is_tile_inner(s_buf.layout, s_buf.shape, atom_shape)
            if outer is None:
                continue

            # Check the region is compatible with the atom shape.
            if not tma_atom_compatible(s_buf.shape, s_st, s_ext, atom_shape):
                continue

            # Swizzle mode selected
            swizzle_mode = mode
            outer_shape = [s // a for s, a in zip(s_buf.shape, atom_shape)]
            outer, seps = outer.normalize().group_by_shape(outer_shape)

            # -------------- iterator derivation (mostly unchanged) ----------
            def derive_iters(outer, seps):
                iters_ = list(enumerate(outer.shard))
                iter_ranges_ = [0] * len(iters_)
                for i in range(len(seps) - 1):
                    st_i, ext_i = s_st[i], s_ext[i]
                    for j in reversed(range(seps[i], seps[i + 1])):
                        if st_i % outer.shard[j].extent == 0:
                            assert ext_i % outer.shard[j].extent == 0
                            iter_ranges_[j] = outer.shard[j].extent
                            st_i //= outer.shard[j].extent
                            ext_i //= outer.shard[j].extent
                        else:
                            # Region falls within a partial tile
                            assert st_i < outer.shard[j].extent
                            assert st_i + ext_i <= outer.shard[j].extent
                            iter_ranges_[j] = ext_i
                            break
                iters_.sort(key=lambda x: x[1].stride, reverse=True)
                return iters_, iter_ranges_

            iters, iter_ranges = derive_iters(outer, seps)

            # -------- derive box_dim (how many atoms per cp.async) ----------
            if outer.shard[seps[-2] - 1].stride == 1:
                box_dim = copy.copy(atom_shape)
                box_dim[-2] *= outer.shard[seps[-2] - 1].extent
                iter_ranges.pop(iters[-1][0])
                iters.pop()
            else:
                box_dim = atom_shape

            # Convert box_dim to *global* axis order for tensor‑map encode.
            box_dim_global = [1] * len(g_buf.shape)
            for i in range(len(g_buf.shape)):
                if axis_map[i] != -1:
                    box_dim_global[i] = box_dim[axis_map[i]]
            box_dim = box_dim_global
            break  # swizzle mode found

        if swizzle_mode is None:
            raise ValueError(
                "No valid swizzle mode found for TMA copy. Shared layout is " f"{s_buf.layout}"
            )

    # ---------------------------------------------------------------------
    # Launch configuration & common symbols
    # ---------------------------------------------------------------------
    tx = sctx.launch_params["threadIdx.x"]
    assert "threadIdx.y" not in sctx.launch_params and "threadIdx.z" not in sctx.launch_params

    tensor_map = T.Var(g_buf.data.name + "_tensormap", dtype=T.handle("tensormap").type_annotation)

    # ---------------------------------------------------------------------
    # Coordinate helpers (captures *axis_map*)
    # ---------------------------------------------------------------------
    def make_global_coord(shared_coord):
        """Project *shared_coord* → global‑space coord (apply inverse map)."""
        coord = copy.copy(g_st)
        for g_ax in range(len(g_buf.shape)):
            s_ax = axis_map[g_ax]
            if s_ax != -1:
                coord[g_ax] += shared_coord[s_ax] - s_st[s_ax]
        return coord

    def make_shared_coord(st, lvs):
        if isinstance(lvs, tvm.tir.Var):
            lvs = [lvs]
        lv_shuffled = [0] * len(outer.shard)
        for (idx, data_iter), lv in zip(iters, lvs):
            lv_shuffled[idx] = lv
        coord = copy.copy(st)
        for i in range(len(s_buf.shape)):
            grouped_shape = [outer.shard[j].extent for j in range(seps[i], seps[i + 1])]
            grouped_outer = TileLayout(grouped_shape)
            coord[i] += grouped_outer.apply(
                *lv_shuffled[seps[i] : seps[i + 1]], shape=grouped_shape
            )["m"]
            coord[i] *= atom_shape[i]
        return coord

    # ---------------------------------------------------------------------
    # Device‑side TIR implementation
    # ---------------------------------------------------------------------
    # fmt: off
    @T.prim_func(tirp=True, check_well_formed=False)
    def impl():
        rank = len(g_buf.shape)
        # TODO(@bohan): reconsider this under warp specialized scenario. Q: should we place the fence here?
        # make sure smem write is visible to tma proxy
        T.ptx.fence.proxy("shared")
        T.tvm_storage_sync("shared")
        for tid_x in T.thread_binding(tx, "threadIdx.x"):
            with T.thread()[tid_x == 0]:
                if swizzle_mode == SwizzleMode.SWIZZLE_NONE:
                    if direction == "g2s":
                        T.ptx.cp_async.bulk.tensor.g2c(
                            rank,
                            s_buf.access_ptr("w", offset=s_buf.offset_of_p(s_st)),
                            pipeline.workspace.get("mbarrier", None).access_ptr("rw", offset=0)  # type: ignore
                            if direction == "g2s" else 0,  # dummy when not needed
                            tensor_map,
                            *reversed(g_st),
                        )
                    else:
                        T.ptx.cp_async.bulk.tensor.s2g(
                            rank,
                            s_buf.access_ptr("r", offset=s_buf.offset_of_p(s_st)),
                            tensor_map,
                            *reversed(g_st),
                        )
                else:
                    for lvs in T.grid(*iter_ranges):
                        if direction == "g2s":
                            s_coord = T.meta_var(make_shared_coord(s_st, lvs))
                            g_coord = T.meta_var(make_global_coord(s_coord))
                            T.ptx.cp_async.bulk.tensor.g2c(
                                rank,
                                s_buf.access_ptr("w", offset=s_buf.offset_of_p(s_coord)),
                                pipeline.workspace.get("mbarrier", None).access_ptr("rw", offset=0),  # type: ignore
                                tensor_map,
                                *reversed(g_coord),
                            )
                        else:
                            s_coord = T.meta_var(make_shared_coord(s_st, lvs))
                            g_coord = T.meta_var(make_global_coord(s_coord))
                            T.ptx.cp_async.bulk.tensor.s2g(
                                rank,
                                s_buf.access_ptr("r", offset=s_buf.offset_of_p(s_coord)),
                                tensor_map,
                                *reversed(g_coord),
                            )
    # fmt: on
    # ---------------------------------------------------------------------
    # Host‑side tensor‑map creation
    # ---------------------------------------------------------------------
    element_strides = [1] * len(g_buf.shape)
    dtype_bytes = tvm.DataType(g_buf.dtype).bits // 8
    g_strides = [g_buf.layout.shard[i].stride * dtype_bytes for i in range(len(g_buf.shape))]
    # fmt: off
    @T.prim_func(tirp=True, check_well_formed=False)
    def create_tensor_map():
        with T.LetStmt(T.tvm_stack_alloca("tensormap", 1), var=tensor_map):
            T.call_packed(
                "runtime.cuTensorMapEncodeTiled",
                tensor_map,
                g_buf.dtype,
                len(g_buf.shape),
                g_buf.data,
                *reversed(g_buf.shape),
                *reversed(g_strides[:-1]),
                *reversed(box_dim),
                *element_strides,
                0,  # CU_TENSOR_MAP_INTERLEAVE_NONE
                swizzle_mode.value,
                0,  # CU_TENSOR_MAP_L2PROMOTION_NONE
                0,  # CU_TENSOR_MAP_FLOAT_OOBFILL_NONE
            )
            Tp.tvm_kernel_replace_point()
    # fmt: on
    # Insert host‑side initialization
    sctx.add_init_stmt(create_tensor_map.body, host=True)

    return impl


def copy_pipeline_cta_impl(
    op_type: PipelineOp,
    sctx: ScheduleContext,
    *args,
) -> Optional[PrimFunc]:
    """Copy pipeline implementation."""
    pipeline = args[0]

    impl = pipeline.schedule_config.get(CopyPipeline.StrategyKind.IMPL, None)
    assert impl is not None, "Copy pipeline implementation is not found"

    if impl == CopyPipeline.Impl.VEC_LOAD:
        # async-group-like completion mechanism
        assert pipeline.depth == 0 and not pipeline.separate_pc
        if op_type == PipelineOp.PRODUCER_COMMIT:
            # cp.async.commit_group()
            @T.prim_func
            def func():
                T.evaluate(T.ptx.cp_async.commit_group())

            return func

        if op_type == PipelineOp.CONSUMER_WAIT:
            # cp.async.wait_group(num_stages)
            num_stages = args[1]

            @T.prim_func
            def func():  # pylint: disable=function-redefined
                T.evaluate(T.ptx.cp_async.wait_group(num_stages))
                T.tvm_storage_sync("shared")

            return func

        if op_type == PipelineOp.COPY:
            dst, src = args[1], args[2]
            # copy the data from src to dst
            for schedule in [copy_g2s_s2g_cta_vec_load_impl]:
                res = schedule(dst, src, sctx, CopyInstType.CP_ASYNC)
                if res is not None:
                    return res
            raise ValueError(
                f"No valid implementation found for copy pipeline with strategy={impl}"
            )

        if op_type == PipelineOp.INIT:
            # no-op
            @T.prim_func
            def func():  # pylint: disable=function-redefined
                T.evaluate(0)

            return func
        # other ops are not supported
        raise ValueError(f"Copy pipeline {op_type} is not supported for strategy={impl}")

    if impl == CopyPipeline.Impl.TMA_LOAD:
        # TODO(@bohan): support private memory allocation
        mbarrier: Optional[Buffer] = pipeline.workspace.get("mbarrier", None)
        phase: Optional[Buffer] = pipeline.workspace.get("phase", None)
        assert mbarrier is not None, "mbarrier is not found in the workspace"
        assert phase is not None, "phase is not found in the workspace"
        # TODO(@bohan): support other pipeline configurations
        assert pipeline.depth == 0 and not pipeline.separate_pc
        # TODO(@bohan): consider cluster multicasting cases
        # TODO(@bohan): support other launch parameters
        tx = sctx.launch_params["threadIdx.x"]
        assert "threadIdx.y" not in sctx.launch_params and "threadIdx.z" not in sctx.launch_params

        # TMA-based copy pipeline
        if op_type == PipelineOp.PRODUCER_COMMIT:
            # mbarrier.arrive_expect_tx.shared::cta.b64
            tma_bytes = args[1]

            @T.prim_func(check_well_formed=False, tirp=True)
            def func():  # pylint: disable=function-redefined
                # TODO(@bohan): fetch env variables from the launch parameters
                for tid in T.thread_binding(tx, "threadIdx.x"):
                    with T.thread()[tid == 0]:
                        T.ptx.mbarrier.arrive.expect_tx(mbarrier.access_ptr("rw"), tma_bytes)

            return func
        if op_type == PipelineOp.CONSUMER_WAIT:
            # wait mbarrier to flip the phase
            @T.prim_func(check_well_formed=False, tirp=True)
            def func():  # pylint: disable=function-redefined
                T.ptx.mbarrier.try_wait(mbarrier.access_ptr("rw"), phase[0])
                T.tvm_storage_sync("shared")
                phase[0] = phase[0] ^ 1

            return func
        if op_type == PipelineOp.COPY:
            dst, src = args[1], args[2]
            for schedule in [copy_tma_impl]:
                res = schedule(pipeline, dst, src, sctx)
                if res is not None:
                    return res
            raise ValueError(
                f"No valid implementation found for copy pipeline with strategy={impl}"
            )
        if op_type == PipelineOp.INIT:
            # initialize the mbarrier, make sure the initialization is visible to all threads
            # and the phase is initialized to 0
            @T.prim_func(check_well_formed=False, tirp=True)
            def func():  # pylint: disable=function-redefined
                for tid in T.thread_binding(tx, "threadIdx.x"):
                    with T.thread()[tid == 0]:
                        T.ptx.mbarrier.init(mbarrier.access_ptr("rw"), 1)
                        T.ptx.fence.proxy("shared")
                    phase[0] = 0
                T.tvm_storage_sync("shared")

            return func
        # other ops are not supported
        raise ValueError(f"Copy pipeline {op_type} is not supported for strategy={impl}")

    if impl == CopyPipeline.Impl.TMA_STORE:
        if op_type == PipelineOp.PRODUCER_COMMIT:

            @T.prim_func(check_well_formed=False, tirp=True)
            def func():  # pylint: disable=function-redefined
                T.ptx.cp_async.bulk.commit_group()

            return func
        if op_type == PipelineOp.CONSUMER_WAIT:
            n = args[1]

            @T.prim_func(check_well_formed=False, tirp=True)
            def func():  # pylint: disable=function-redefined
                T.ptx.cp_async.bulk.wait_group(n)

            return func
        if op_type == PipelineOp.COPY:
            dst, src = args[1], args[2]
            for schedule in [copy_tma_impl]:
                res = schedule(pipeline, dst, src, sctx)
                if res is not None:
                    return res
            raise ValueError(
                f"No valid implementation found for copy pipeline with strategy={impl}"
            )

        if op_type == PipelineOp.INIT:

            @T.prim_func(check_well_formed=False, tirp=True)
            def func():  # pylint: disable=function-redefined
                pass

            return func
        # other ops are not supported
        raise ValueError(f"Copy pipeline {op_type} is not supported for strategy={impl}")


@register_schedule("pipeline_init", "cuda")
@target_cuda
def pipeline_init(op: OpCall, sctx: ScheduleContext) -> Optional[PrimFunc]:
    """Schedule pipeline initialization."""
    if sctx.exec_scope.name != "cta":
        return None
    pipeline = op.args[0]
    if isinstance(pipeline, CopyPipeline):
        return copy_pipeline_cta_impl(PipelineOp.INIT, sctx, *op.args)

    return None


@register_schedule("pipeline_producer_commit", "cuda")
@target_cuda
def pipeline_producer_commit(op: OpCall, sctx: ScheduleContext) -> Optional[PrimFunc]:
    """Schedule pipeline producer commit."""
    if sctx.exec_scope.name != "cta":
        return None
    pipeline = op.args[0]
    if isinstance(pipeline, CopyPipeline):
        return copy_pipeline_cta_impl(PipelineOp.PRODUCER_COMMIT, sctx, *op.args)

    return None


@register_schedule("pipeline_consumer_wait", "cuda")
@target_cuda
def pipeline_consumer_wait(op: OpCall, sctx: ScheduleContext) -> Optional[PrimFunc]:
    """Schedule pipeline consumer wait."""
    if sctx.exec_scope.name != "cta":
        return None
    pipeline = op.args[0]
    if isinstance(pipeline, CopyPipeline):
        return copy_pipeline_cta_impl(PipelineOp.CONSUMER_WAIT, sctx, *op.args)

    return None


@register_schedule("pipeline_copy", "cuda")
@target_cuda
def pipeline_copy(op: OpCall, sctx: ScheduleContext) -> Optional[PrimFunc]:
    """Schedule pipeline copy."""
    pipeline, dst_buffer_region, src_buffer_region = op.args
    if not validate_copy_op(dst_buffer_region, src_buffer_region, sctx):
        return None

    if isinstance(pipeline, CopyPipeline):
        return copy_pipeline_cta_impl(PipelineOp.COPY, sctx, *op.args)
    return None
