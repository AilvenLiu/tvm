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
from tvm.arith import Analyzer
from .common import InstType, copy_g2s_s2g_cta_vec_load_impl, target_cuda
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
        return TileLayout.from_tuple(shape).normalize()
    atom_shape = tma_atom_shape(dtype, swizzle_mode, shape)
    layout = tma_atom_layout(dtype, swizzle_mode)
    tile_to_shape = copy.deepcopy(atom_shape)
    tile_to_shape[-2] = shape[-2]
    return layout.tile_to(tile_to_shape, atom_shape).tile_to(shape, tile_to_shape).normalize()


def tma_atom_compatible(dst_shape, dst_st, dst_extent, atom_shape):
    """Check if the copy region in dst is compatible with the TMA atom shape."""
    for i, _ in enumerate(dst_st):
        if any(x % atom_shape[i] != 0 for x in [dst_shape[i], dst_st[i], dst_extent[i]]):
            return False
    return True


def copy_g2s_cta_tma_impl(
    pipeline: CopyPipeline,
    dst_buffer_region: BufferRegion,
    src_buffer_region: BufferRegion,
    sctx: ScheduleContext,
) -> Optional[PrimFunc]:
    """Schedule copy operation between global and shared memory on CUDA."""
    if sctx.exec_scope.name != "cta":
        return None

    src: Buffer = src_buffer_region.buffer
    dst: Buffer = dst_buffer_region.buffer
    if not all(
        [
            src.layout and dst.layout,
            src.dtype == dst.dtype,
            src.layout.is_trivial(),  # TODO(@bohan): support strided global memory
            src.scope() == "global" and dst.scope().startswith("shared"),
        ]
    ):
        return None

    # Extract regions and validate dimensions
    analyzer = Analyzer()
    src_region, dst_region = src_buffer_region.region, dst_buffer_region.region
    src_st = [r.min for r in src_region]
    dst_st = [r.min for r in dst_region]
    src_extent = [r.extent for r in src_region]
    dst_extent = [r.extent for r in dst_region]

    # Validate non-unit dimensions match
    def dim_match(src_extent, dst_extent) -> List[int]:
        src_extent_ = [(i, e) for i, e in enumerate(src_extent) if e != 1]
        dst_extent_ = [(i, e) for i, e in enumerate(dst_extent) if e != 1]
        if len(src_extent_) == len(dst_extent_) and all(
            analyzer.can_prove_equal(s[1], d[1]) for s, d in zip(src_extent_, dst_extent_)
        ):
            axis_map = [-1] * len(src.shape)
            for s, d in zip(src_extent_, dst_extent_):
                axis_map[s[0]] = d[0]
            return axis_map
        return None

    axis_map = dim_match(src_extent, dst_extent)
    if axis_map is None:
        return None

    # Determine the swizzle mode
    swizzle_mode = None
    box_dim = None
    if dst.layout.is_trivial():
        swizzle_mode = SwizzleMode.SWIZZLE_NONE
        box_dim = src_extent
    else:
        for mode in (
            SwizzleMode.SWIZZLE_128B_ATOM,
            SwizzleMode.SWIZZLE_64B_ATOM,
            SwizzleMode.SWIZZLE_32B_ATOM,
        ):
            swizzle_atom = tma_atom_layout(src.dtype, mode)
            atom_shape = tma_atom_shape(src.dtype, mode, dst.shape)
            outer = swizzle_atom.is_tile_inner(dst.layout, dst.shape, atom_shape)
            if outer is None:
                continue

            # check if copy region in dst is compatible with the TMA atom shape
            if not tma_atom_compatible(dst.shape, dst_st, dst_extent, atom_shape):
                continue

            # swizzle mode found
            swizzle_mode = mode
            outer_shape = [s // a for s, a in zip(dst.shape, atom_shape)]
            outer, seps = outer.normalize().group_by_logical_shape(outer_shape)

            # Derive the iters to traverse the TMA atoms
            def derive_iters(outer, seps):
                iters = list(enumerate(outer.data_iter_array))
                # given dst_st, dst_extent, derive the iter ranges
                iter_ranges = [0] * len(iters)  # extent for each iterator
                for i in range(len(seps) - 1):
                    dst_st_, dst_extent_ = dst_st[i], dst_extent[i]
                    for j in reversed(range(seps[i], seps[i + 1])):
                        if dst_st_ % outer.data_iter_array[j].extent == 0:
                            # TODO(@bohan): let each schedule report the failure reason
                            assert dst_extent_ % outer.data_iter_array[j].extent == 0
                            iter_ranges[j] = outer.data_iter_array[j].extent
                            dst_st_ //= outer.data_iter_array[j].extent
                            dst_extent_ //= outer.data_iter_array[j].extent
                        else:
                            assert dst_st_ < outer.data_iter_array[j].extent
                            assert dst_st_ + dst_extent_ <= outer.data_iter_array[j].extent
                            iter_ranges[j] = dst_extent_
                            break
                # set iterators to traverse the TMA atom in the descending order of the strides
                iters.sort(key=lambda x: x[1].stride, reverse=True)
                return iters, iter_ranges

            iters, iter_ranges = derive_iters(outer, seps)
            # set the box dimension
            if outer.data_iter_array[seps[-2] - 1].stride == 1:
                # can load multiple TMA atoms at once,
                box_dim = copy.deepcopy(atom_shape)
                box_dim[-2] *= outer.data_iter_array[seps[-2] - 1].extent
                # remove the last iterator
                iter_ranges.pop(iters[-1][0])
                iters.pop()
            else:
                # can only load one TMA atom at a time
                box_dim = atom_shape
            # turn box_dim from dst_wise to src_wise
            box_dim_src = [1] * len(src.shape)
            for i in range(len(src.shape)):
                if axis_map[i] != -1:
                    box_dim_src[i] = box_dim[axis_map[i]]
            box_dim = box_dim_src
            break

        if swizzle_mode is None:
            raise ValueError(
                f"No valid swizzle mode found for TMA copy. The destination layout is {dst.layout}"
            )

    tx = sctx.launch_params["threadIdx.x"]
    assert "threadIdx.y" not in sctx.launch_params and "threadIdx.z" not in sctx.launch_params

    tensor_map = T.Var(src.data.name + "_tensormap", dtype=T.handle("tensormap").type_annotation)
    mbarrier: Optional[Buffer] = pipeline.workspace.get("mbarrier", None)

    # Device-side implementation
    def get_src_coord(src_st, dst_st, dst_coord):
        """Convert the coordinates to the global coordinates."""
        coord = copy.deepcopy(src_st)
        for i in range(len(src.shape)):
            if axis_map[i] != -1:
                coord[i] += dst_coord[axis_map[i]] - dst_st[axis_map[i]]
        return coord

    def get_dst_coord(st, lvs):
        """Convert the coordinates to the smem coordinates."""
        if isinstance(lvs, tvm.tir.Var):
            lvs = [lvs]
        lv_shuffled = [0] * len(outer.data_iter_array)
        for iter_, lv in zip(iters, lvs):
            lv_shuffled[iter_[0]] = lv
        coord = copy.deepcopy(st)
        for i in range(len(dst.shape)):
            grouped_shape = [outer.data_iter_array[j].extent for j in range(seps[i], seps[i + 1])]
            grouped_outer = TileLayout.from_tuple(grouped_shape)
            coord[i] += grouped_outer.apply(
                *lv_shuffled[seps[i] : seps[i + 1]], shape=grouped_shape
            )[0]
            coord[i] *= atom_shape[i]
        return coord

    # fmt: off
    @T.prim_func(tirp=True, check_well_formed=False)
    def impl():
        """Implement copy operation with TMA."""
        for tid_x in T.thread_binding(tx, "threadIdx.x"):
            with T.thread()[tid_x == 0]:
                if swizzle_mode == SwizzleMode.SWIZZLE_NONE:
                    T.cp_async_bulk_tensor_global_to_cluster(
                        len(src.shape), # rank of global coordinate
                        dst.access_ptr("w", offset=dst.offset_of_p(dst_st)), # dst pointer
                        mbarrier.access_ptr("rw", offset=0), # mbarrier pointer
                        tensor_map, # tensor map
                        *reversed(src_st), # global coordinate
                    )
                else:
                    # TODO(@bohan): enhance T.grid to support non-zero start
                    for lvs in T.grid(*iter_ranges):
                        dst_coord = T.meta_var(get_dst_coord(dst_st, lvs))
                        src_coord = T.meta_var(get_src_coord(src_st, dst_st, dst_coord))
                        T.cp_async_bulk_tensor_global_to_cluster(
                            len(src.shape), # rank of global coordinate
                            dst.access_ptr("w", offset=dst.offset_of_p(dst_coord)), # dst pointer
                            mbarrier.access_ptr("rw", offset=0), # mbarrier pointer
                            tensor_map, # tensor map
                            *reversed(src_coord), # global coordinate
                        )
    # fmt: on
    element_strides = [1 for _ in range(len(src.shape))]
    # TODO(@bohan): make better APIs to access extents/strides of layouts
    dtype_bytes = tvm.DataType(src.dtype).bits // 8
    src_strides = [
        src.layout.data_iter_array[i].stride * dtype_bytes for i in range(len(src.shape))
    ]

    # Host-side implementation
    @T.prim_func(tirp=True, check_well_formed=False)
    def create_tensor_map():
        """Create the tensor map."""
        with T.LetStmt(T.tvm_stack_alloca("tensormap", 1), var=tensor_map):
            T.call_packed(
                "runtime.cuTensorMapEncodeTiled",
                tensor_map,  # pointer to the tensor map
                src.dtype,  # tensor_dtype
                len(src.shape),  # global_rank
                src.data,  # global_ptr
                *reversed(src.shape),  # global_shape
                *reversed(src_strides[:-1]),  # global_strides
                *reversed(box_dim),  # boxDim
                *element_strides,  # element_strides
                0,  # CU_TENSOR_MAP_INTERLEAVE_NONE
                swizzle_mode.value,  # CU_TENSOR_MAP_SWIZZLE_NONE
                0,  # CU_TENSOR_MAP_L2PROMOTION_NONE
                0,  # CU_TENSOR_MAP_FLOAT_OOBFILL_NONE
            )
            Tp.tvm_kernel_replace_point()

    # insert these codes to host code before the kernel
    sctx.add_init_stmt(create_tensor_map.body, host=True)
    impl.show()
    create_tensor_map.show()
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
                T.evaluate(T.ptx_commit_group())

            return func

        if op_type == PipelineOp.CONSUMER_WAIT:
            # cp.async.wait_group(num_stages)
            num_stages = args[1]

            @T.prim_func
            def func():  # pylint: disable=function-redefined
                T.evaluate(T.ptx_wait_group(num_stages))
                T.tvm_storage_sync("shared")

            return func

        if op_type == PipelineOp.COPY:
            dst, src = args[1], args[2]
            # copy the data from src to dst
            for schedule in [copy_g2s_s2g_cta_vec_load_impl]:
                res = schedule(dst, src, sctx, InstType.CP_ASYNC)
                if res is not None:
                    return res
            raise ValueError(
                f"No valid implementation found for copy pipeline with strategy={pipeline.strategy}"
            )

        if op_type == PipelineOp.INIT:
            # no-op
            @T.prim_func
            def func():  # pylint: disable=function-redefined
                T.evaluate(0)

            return func
        # other ops are not supported
        raise ValueError(
            f"Copy pipeline {op_type} is not supported for strategy={pipeline.strategy}"
        )

    if impl == CopyPipeline.Impl.TMA:
        # TODO(@bohan): support private memory allocation
        mbarrier: Optional[Buffer] = pipeline.workspace.get("mbarrier", None)
        assert mbarrier is not None, "mbarrier is not found in the workspace"
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
                        T.mbarrier_arrive_expect_tx(mbarrier.access_ptr("rw"), tma_bytes)

            return func
        if op_type == PipelineOp.CONSUMER_WAIT:
            # wait mbarrier to flip
            @T.prim_func(check_well_formed=False, tirp=True)
            def func():  # pylint: disable=function-redefined
                # TODO(@bohan): fix the phase issue
                T.mbarrier_wait(mbarrier.access_ptr("rw"), 0)
                T.tvm_storage_sync("shared")

            return func
        if op_type == PipelineOp.COPY:
            dst, src = args[1], args[2]
            for schedule in [copy_g2s_cta_tma_impl]:
                res = schedule(pipeline, dst, src, sctx)
                if res is not None:
                    return res
            raise ValueError(
                f"No valid implementation found for copy pipeline with strategy={pipeline.strategy}"
            )
        if op_type == PipelineOp.INIT:
            # initialize the mbarrier, make sure the initialization is visible to all threads
            @T.prim_func(check_well_formed=False, tirp=True)
            def func():  # pylint: disable=function-redefined
                for tid in T.thread_binding(tx, "threadIdx.x"):
                    with T.thread()[tid == 0]:
                        T.mbarrier_init(mbarrier.access_ptr("rw"), 1)
                        T.cuda_fence_proxy_async("shared")
                T.tvm_storage_sync("shared")

            return func
        # other ops are not supported
        raise ValueError(
            f"Copy pipeline {op_type} is not supported for strategy={pipeline.strategy}"
        )


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
    if sctx.exec_scope.name != "cta":
        return None
    pipeline = op.args[0]
    if isinstance(pipeline, CopyPipeline):
        return copy_pipeline_cta_impl(PipelineOp.COPY, sctx, *op.args)
    return None
