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

from typing import Optional, Union, List
from enum import Enum
import copy
import tvm

from tvm.script import tir as T
from tvm.script import tirx as Tx
from tvm.tir import PrimFunc, Buffer
from tvm.tir.stmt import OpCall
from tvm.tirx.op_schedule import (
    ScheduleContext,
    register_dispatch,
    predicate,
)
from .common import (
    CopyInstType,
    copy_vec_load_impl,
    validate_copy_op,
    single_thread,
)
from tvm.tir.layout import TileLayout, SwizzleLayout, TLayout


class SwizzleMode(Enum):
    """The swizzle mode of the TMA"""

    SWIZZLE_NONE = 0
    SWIZZLE_32B_ATOM = 1
    SWIZZLE_64B_ATOM = 2
    SWIZZLE_128B_ATOM = 3


def tma_atom_layout(dtype: str, swizzle_mode: Union[SwizzleMode, int]) -> SwizzleLayout:
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


def tma_shared_layout(dtype: str, swizzle_mode: Union[SwizzleMode, int], shape) -> TLayout:
    """Generate the TMA layout for the shared memory given shape and dtype.
    It uses a default tiling strategy to tile the TMA atom layout into the shared memory.
    """
    if isinstance(swizzle_mode, int):
        swizzle_mode = SwizzleMode(swizzle_mode)
    if swizzle_mode == SwizzleMode.SWIZZLE_NONE:
        return TileLayout(shape).canonicalize()
    atom_shape = tma_atom_shape(dtype, swizzle_mode, shape)
    layout = tma_atom_layout(dtype, swizzle_mode)
    tile_to_shape = copy.copy(atom_shape)
    tile_to_shape[-2] = shape[-2]
    return layout.tile_to(tile_to_shape, atom_shape).tile_to(shape, tile_to_shape).canonicalize()


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
    """
    # ---------------------------------------------------------------------
    # Identify direction & basic legality checks
    # ---------------------------------------------------------------------
    dst_buffer_region, src_buffer_region = op_call.args
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
        raise ValueError(
            f"Unsupported combination of src and dst scopes: src={src_scope} dst={dst_scope}"
        )

    rank = len(g_buf.shape)

    # For now, we require that the global side layout is trivial.
    # TODO(bohan): support strided global memory in the future.
    if not g_buf.layout.is_trivial():
        raise ValueError(f"Global buffer layout is not trivial: {g_buf.layout}")

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
        axis_map = dict()
        for (ai, _), (bi, _) in zip(a_nu, b_nu):
            axis_map[ai] = bi
        return axis_map

    # Map *global axis* → *shared axis*
    axis_map = dim_match(g_ext, s_ext, rank)

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
            atom_shape_shared = tma_atom_shape(s_buf.dtype, mode, s_buf.shape)
            atom_shape_global = tma_atom_shape(s_buf.dtype, mode, g_buf.shape)
            outer_shared = swizzle_atom.is_tile_inner(s_buf.layout, s_buf.shape, atom_shape_shared)
            if outer_shared is None:
                continue

            # Check the region is compatible with the atom shape.
            if not tma_atom_compatible(s_buf.shape, s_st, s_ext, atom_shape_shared):
                continue

            # Swizzle mode selected
            swizzle_mode = mode
            outer_shape_shared = [s // a for s, a in zip(s_buf.shape, atom_shape_shared)]
            outer_shared, seps = outer_shared.canonicalize().group(outer_shape_shared)

            # copy box starts at atom shape, clamped to actual copy extent
            # (can't copy more than g_ext in any dimension)
            box_dim = [min(a, e) for a, e in zip(atom_shape_global, g_ext)]

            # Try to enlarge each dimension where shared layout allows it
            # Skip innermost dimension to respect TMA swizzle limits
            for i in range(rank - 1):
                if box_dim[i] >= g_ext[i]:
                    continue  # Already at extent limit
                s_dim = axis_map.get(i, i)
                if s_dim < len(outer_shared.shard):
                    if outer_shared.shard[s_dim].stride == 1:
                        enlarge_factor = outer_shared.shard[s_dim].extent
                        new_box_dim = atom_shape_shared[s_dim] * enlarge_factor
                        # Only enlarge if it fits within g_ext and divides evenly
                        if new_box_dim <= g_ext[i] and g_ext[i] % new_box_dim == 0:
                            box_dim[i] = new_box_dim

            # iterator over global space, each element is a box in global space
            iters_global = [(g_st[i], g_ext[i] // box_dim[i]) for i in range(rank)]
            break  # swizzle mode found

        if swizzle_mode is None:
            raise ValueError(
                "No valid swizzle mode found for TMA copy. Shared layout is "
                f"{s_buf.layout} and global layout is {g_buf.layout}"
            )

    # ---------------------------------------------------------------------
    # Launch configuration & common symbols
    # ---------------------------------------------------------------------
    cta_group = op_call.config.get("cta_group", None)
    if cta_group is None:
        cta_group = 1 if sctx.target.arch == "sm_100a" else -1

    tensor_map = T.Var(g_buf.data.name + "_tensormap", dtype=T.handle("tensormap").type_annotation)

    # ---------------------------------------------------------------------
    # Coordinate helpers
    # ---------------------------------------------------------------------
    def make_global_coord(lvs):
        return [g_st[i] + lvs[i] * box_dim[i] for i in range(rank)]

    def make_shared_coord(lvs):
        shared_coord = copy.copy(s_st)
        for g_axis, s_axis in axis_map.items():
            shared_coord[s_axis] += lvs[g_axis] * box_dim[g_axis]
        return shared_coord

    if direction == "g2s":
        # get mbar from config
        mbar = op_call.config.get("mbar", None)
        if mbar is None:
            raise ValueError("mbar is not set in config")

    # ---------------------------------------------------------------------
    # Device‑side TIR implementation
    # ---------------------------------------------------------------------

    # fmt: off
    @T.prim_func(tirx=True, check_well_formed=False)
    def impl():
        # TODO(@bohan): reconsider this under warp specialized scenario. Q: should we place the fence here?
        # make sure smem write is visible to tma proxy
        if swizzle_mode == SwizzleMode.SWIZZLE_NONE:
            if direction == "g2s":
                T.ptx.cp_async.bulk.tensor.g2c(
                    rank,
                    s_buf.ptr_to(s_st),
                    mbar,
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
                    cache_hint=op_call.config.get("cache_hint", ""),
                )
        else:
            for lvs in T.grid(*[it[1] for it in iters_global]):
                if direction == "g2s":
                    g_coord = T.meta_var(make_global_coord(lvs))
                    s_coord = T.meta_var(make_shared_coord(lvs))
                    T.ptx.cp_async.bulk.tensor.g2c(
                        rank,
                        s_buf.ptr_to(s_coord),
                        mbar,
                        tensor_map,
                        *reversed(g_coord),
                        cta_group=cta_group,
                        cache_hint=op_call.config.get("cache_hint", ""),
                    )
                else:
                    g_coord = T.meta_var(make_global_coord(lvs))
                    s_coord = T.meta_var(make_shared_coord(lvs))
                    T.ptx.cp_async.bulk.tensor.s2g(
                        rank,
                        s_buf.ptr_to(s_coord),
                        tensor_map,
                        *reversed(g_coord),
                        cache_hint=op_call.config.get("cache_hint", ""),
                    )

    # fmt: on
    # ---------------------------------------------------------------------
    # Host‑side tensor‑map creation
    # ---------------------------------------------------------------------
    element_strides = [1] * rank
    dtype_bytes = tvm.DataType(g_buf.dtype).bits // 8
    g_strides = [shard.stride * dtype_bytes for shard in TileLayout(shard=list(g_buf.shape)).shard]
    # fmt: off
    @T.prim_func(tirx=True, check_well_formed=False)
    def create_tensor_map():
        with T.LetStmt(T.tvm_stack_alloca("tensormap", 1), var=tensor_map):
            T.call_packed(
                "runtime.cuTensorMapEncodeTiled",
                tensor_map,
                g_buf.dtype,
                rank,
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
            Tx.tvm_kernel_replace_point()
    # fmt: on
    # create_tensor_map.show()

    # Insert host‑side initialization
    sctx.add_init_stmt(create_tensor_map.body, host=True)

    return impl


@register_dispatch(
    "copy_async",
    "cuda",
    variant="non-bulk-copy",
    priority=20,
    when=[
        predicate(
            "validate_copy_op",
            lambda op, sctx: (validate_copy_op(op, sctx), "not a valid copy op"),
        ),
    ],
)
def copy_async_dispatch_cp_async(op: OpCall, sctx: ScheduleContext) -> PrimFunc:
    return copy_vec_load_impl(op, sctx, CopyInstType.CP_ASYNC)


@register_dispatch(
    "copy_async",
    "cuda",
    variant="tma",
    priority=10,
    when=[
        predicate(
            "validate_copy_op",
            lambda op, sctx: (validate_copy_op(op, sctx), "not a valid copy op"),
        ),
        predicate(
            "single_thread",
            lambda op, sctx: (
                single_thread(op, sctx),
                f"unsupported exec_scope {sctx.exec_scope}, expected single thread",
            ),
        ),
    ],
)
def copy_async_dispatch_tma(op: OpCall, sctx: ScheduleContext) -> PrimFunc:
    return copy_tma_impl(op, sctx)
