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

from typing import Optional, Union, List, Dict, Any
from enum import Enum
import copy
import tvm
import functools

from tvm.script import tir as T
from tvm.script import tirp as Tp
from tvm.tir import PrimFunc, Buffer
from tvm.tir.stmt import OpCall
from tvm.tirp.op_schedule import (
    ScheduleContext,
    register_dispatch,
    predicate,
)
from .common import target_cuda
from tvm.tir import BufferRegion
from tvm.tir.event import SemaphoreEventTensor, EventImpl, BulkGroupEvent, SemaphoreEventTensorItem
from .common import CopyInstType, copy_vec_load_impl, validate_copy_op, thread_selector
from tvm.tir.layout import TileLayout, SwizzleLayout


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
    op_call: OpCall,
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
    # Identify direction & basic legality checks
    # ---------------------------------------------------------------------
    dst_buffer_region, src_buffer_region, evt = op_call.args
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
                            iter_ranges_[j] = outer.shard[j].extent
                            st_i //= outer.shard[j].extent
                            ext_i //= outer.shard[j].extent
                        else:
                            # Region falls within a partial tile
                            iter_ranges_[j] = ext_i
                            break
                return iters_, iter_ranges_

            iters, iter_ranges = derive_iters(outer, seps)

            # -------- derive box_dim (how many atoms per cp.async) ----------
            if outer.shard[seps[-2] - 1].stride == 1:
                box_dim = copy.copy(atom_shape)
                box_dim[-2] *= outer.shard[seps[-2] - 1].extent
                iters.pop(seps[-2] - 1)
                iter_ranges.pop(seps[-2] - 1)
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
    if sctx.target.arch == "sm_100a":
        cta_group = 1
    else:
        cta_group = -1

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

    if direction == "g2s":
        mbar, phase, tx_cnt = evt.get_state()

    # ---------------------------------------------------------------------
    # Device‑side TIR implementation
    # ---------------------------------------------------------------------

    def total_bytes():
        return (
            functools.reduce(lambda acc, extent: acc * extent, s_ext, 1)
            * tvm.DataType(s_buf.dtype).bits
            // 8
        )

    rank = len(g_buf.shape)

    # fmt: off
    @T.macro
    def inner_impl():
        # TODO(@bohan): reconsider this under warp specialized scenario. Q: should we place the fence here?
        # make sure smem write is visible to tma proxy
        if swizzle_mode == SwizzleMode.SWIZZLE_NONE:
            if direction == "g2s":
                tx_cnt[0] += total_bytes()
                T.ptx.cp_async.bulk.tensor.g2c(
                    rank,
                    s_buf.ptr_to(s_st),
                    mbar.ptr_to(evt.indices)  # type: ignore
                    if direction == "g2s" else 0,  # dummy when not needed
                    tensor_map,
                    *reversed(g_st),
                    cta_group=cta_group,
                    cache_hint=op_call.config.get("cache_hint", ""),
                )
            else:
                T.ptx.cp_async.bulk.tensor.s2g(
                    rank,
                    s_buf.ptr_to(s_st),
                    tensor_map,
                    *reversed(g_st),
                )
        else:
            if direction == "g2s":
                tx_cnt[0] += total_bytes()
            for lvs in T.grid(*iter_ranges):
                if direction == "g2s":
                    s_coord = T.meta_var(make_shared_coord(s_st, lvs))
                    g_coord = T.meta_var(make_global_coord(s_coord))
                    T.ptx.cp_async.bulk.tensor.g2c(
                        rank,
                        s_buf.ptr_to(s_coord),
                        mbar.ptr_to(evt.indices),
                        tensor_map,
                        *reversed(g_coord),
                        cta_group=cta_group,
                        cache_hint=op_call.config.get("cache_hint", ""),
                    )
                else:
                    s_coord = T.meta_var(make_shared_coord(s_st, lvs))
                    g_coord = T.meta_var(make_global_coord(s_coord))
                    T.ptx.cp_async.bulk.tensor.s2g(
                        rank,
                        s_buf.ptr_to(s_coord),
                        tensor_map,
                        *reversed(g_coord),
                    )
    
    impl = thread_selector(sctx, inner_impl)

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
    # create_tensor_map.show()

    # Insert host‑side initialization
    sctx.add_init_stmt(create_tensor_map.body, host=True)

    return impl


def _is_cp_async_event(op: OpCall, sctx: ScheduleContext):
    evt = op.args[2]
    return (
        isinstance(evt, BulkGroupEvent) and evt.get_impl() == EventImpl.kCpAsync
    ), "event is not cp.async bulk"


def _is_tma_event(op: OpCall, sctx: ScheduleContext):
    evt = op.args[2]
    return (
        (isinstance(evt, BulkGroupEvent) and evt.get_impl() == EventImpl.kTMAStore)
        or (
            isinstance(evt, SemaphoreEventTensorItem)
            and evt.get_impl() in [EventImpl.kTMALoad, EventImpl.kTMALoadOnly]
        )
    ), "event is not TMA (load/store)"


@register_dispatch(
    "copy_async",
    "cuda",
    variant="cp_async",
    priority=20,
    when=[
        predicate(
            "validate_copy_op",
            lambda op, sctx: (validate_copy_op(op, sctx), "validate_copy_op failed"),
        ),
        predicate("event", _is_cp_async_event),
    ],
)
def copy_async_dispatch_cp_async(op: OpCall, sctx: ScheduleContext) -> Optional[PrimFunc]:
    return copy_vec_load_impl(op, sctx, CopyInstType.CP_ASYNC)


@register_dispatch(
    "copy_async",
    "cuda",
    variant="tma",
    priority=10,
    when=[
        predicate(
            "validate_copy_op",
            lambda op, sctx: (validate_copy_op(op, sctx), "validate_copy_op failed"),
        ),
        predicate("event", _is_tma_event),
    ],
)
def copy_async_dispatch_tma(op: OpCall, sctx: ScheduleContext) -> Optional[PrimFunc]:
    return copy_tma_impl(op, sctx)
