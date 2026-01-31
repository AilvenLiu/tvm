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
import functools
import tvm

from tvm.arith import Analyzer
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
from tvm.tir.layout import TileLayout, SwizzleLayout, ComposeLayout, TLayout


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
    analyzer = Analyzer()
    for i, _ in enumerate(dst_st):
        if any(
            not analyzer.can_prove_equal(x % atom_shape[i], 0)
            for x in [dst_shape[i], dst_st[i], dst_extent[i]]
        ):
            return False
    return True


def get_swizzle_mode_from_layout(layout: TLayout) -> Optional[SwizzleMode]:
    """Extract swizzle mode from a shared memory layout.

    Parameters
    ----------
    layout : TLayout
        The shared memory layout (ComposeLayout, SwizzleLayout, or TileLayout)

    Returns
    -------
    Optional[SwizzleMode]
        The swizzle mode if recognized, None otherwise
    """
    if isinstance(layout, ComposeLayout):
        swizzle = layout.swizzle  # SwizzleLayout is named 'swizzle' in ComposeLayout
        swizzle_len = swizzle.swizzle_len
    elif isinstance(layout, SwizzleLayout):
        swizzle_len = layout.swizzle_len
    elif isinstance(layout, TileLayout):
        if layout.is_trivial():
            return SwizzleMode.SWIZZLE_NONE
        return None
    else:
        return None

    # Map swizzle_len to SwizzleMode
    return {
        0: SwizzleMode.SWIZZLE_NONE,
        1: SwizzleMode.SWIZZLE_32B_ATOM,
        2: SwizzleMode.SWIZZLE_64B_ATOM,
        3: SwizzleMode.SWIZZLE_128B_ATOM,
    }.get(swizzle_len)


def find_contiguous_region(layout: TileLayout) -> tuple:
    """Find the contiguous region of a layout.

    NOTE: Do NOT canonicalize the layout - this would break the shard correspondence
    with the grouped global layout.

    Algorithm:
    1. Start from the lowest-stride shard, verify stride == 1
    2. Go to second-lowest-stride shard, verify stride == extent of lowest-stride shard
    3. Continue: each shard's stride must equal the product of extents of all
       previously collected shards
    4. Collect indices of all shards satisfying this condition

    Parameters
    ----------
    layout : TileLayout
        The layout to analyze

    Returns
    -------
    Tuple[List[int], int]
        - contiguous_indices: List of shard indices in the original layout
          (ordered from fastest to slowest stride)
        - contiguous_element_count: Total number of elements in the contiguous region
    """
    # Filter to memory-only shards with their original indices
    indexed_shards = [(i, s) for i, s in enumerate(layout.shard) if s.axis.is_memory()]

    if not indexed_shards:
        return [], 1

    # Sort by stride ascending (lowest stride first), keeping original indices
    sorted_indexed = sorted(indexed_shards, key=lambda x: int(x[1].stride))

    # First shard must have stride == 1
    if int(sorted_indexed[0][1].stride) != 1:
        return [], 0  # Not contiguous from the start

    contiguous_indices = [sorted_indexed[0][0]]
    contiguous_extent = sorted_indexed[0][1].extent
    expected_stride = contiguous_extent  # Next shard must have this stride

    for idx, shard in sorted_indexed[1:]:
        stride = int(shard.stride)
        extent = shard.extent

        # reserve 1 dimension for OOB detection
        # we may be able to make this number configurable in the future
        if len(contiguous_indices) == 4:
            break
        if stride != expected_stride:
            break  # Non-contiguous, stop collecting

        contiguous_indices.append(idx)
        contiguous_extent *= extent
        expected_stride = contiguous_extent

    return contiguous_indices, contiguous_extent


def sort_strides_partial_order(strides: List, analyzer: Analyzer) -> List[int]:
    """Sort indices by stride using partial ordering.

    For symbolic strides: s0 % s1 == 0 means s0 >= s1.
    Returns indices sorted from smallest to largest stride.
    """
    n = len(strides)
    if n == 0:
        return []

    # Simple selection sort using divisibility checks
    result = []
    remaining = list(range(n))

    while remaining:
        # Find minimum: element that all others are >= (i.e., all others divide by it)
        min_idx = remaining[0]
        for i in remaining[1:]:
            # Check if strides[i] <= strides[min_idx]
            # i.e., strides[min_idx] % strides[i] == 0
            if analyzer.can_prove_equal(tvm.tir.floormod(strides[min_idx], strides[i]), 0):
                min_idx = i
        result.append(min_idx)
        remaining.remove(min_idx)

    return result


def assert_compact_layout(shards: List, total_size, analyzer: Analyzer):
    """Assert layout is compact: for each shard, stride * extent == next shard's stride."""
    stride_set = [s for s, _ in shards]

    for stride, extent in shards:
        expected_next = stride * extent
        # For the largest stride shard, expected_next should equal total_size
        # For others, expected_next should match another shard's stride
        if not analyzer.can_prove_equal(expected_next, total_size):
            # Check if expected_next matches any other shard's stride
            found = False
            for other_stride in stride_set:
                if analyzer.can_prove_equal(expected_next, other_stride):
                    found = True
                    break
            if not found:
                raise ValueError(
                    f"Global layout is not compact: stride={stride} * extent={extent} = {expected_next} "
                    f"does not match any other stride or total_size={total_size}"
                )


def compute_box_dim(grouped_shared: TileLayout) -> tuple:
    """Compute box dimensions from a grouped shared layout.

    Algorithm:
    1. Find contiguous shard indices in grouped shared layout
    2. box_dim = extents of contiguous shards (ordered outermost to innermost)

    Parameters
    ----------
    grouped_shared : TileLayout
        The shared memory layout already grouped by global extents

    Returns
    -------
    Tuple[List[int], List[int], int]
        - box_dim: The box dimensions for TMA copy (outermost to innermost, 1s removed)
        - contiguous_indices: Indices of contiguous shards in grouped_shared
          (ordered from fastest to slowest stride)
        - contiguous_extent: Total number of elements in the contiguous region
    """
    # Step 1: Find contiguous shard indices (ordered from highest to lowest stride)
    contiguous_indices, contiguous_extent = find_contiguous_region(grouped_shared)
    contiguous_indices = list(reversed(contiguous_indices))

    # Step 2: box_dim = extents of contiguous shards (reversed: outermost to innermost)
    box_dim = [grouped_shared.shard[i].extent for i in contiguous_indices]

    # Step 3: Remove 1s from box_dim
    box_dim, contiguous_indices = zip(
        *[(d, i) for d, i in zip(box_dim, contiguous_indices) if d != 1]
    )

    return list(box_dim), list(contiguous_indices), contiguous_extent


def to_tile_layout(layout: TLayout, shape: List[int]) -> TileLayout:
    """Convert a layout to a tile layout."""
    if isinstance(layout, ComposeLayout):
        return layout.tile_layout
    elif isinstance(layout, SwizzleLayout):
        return TileLayout(shard=(list(shape), TLayout._get_default_strides(list(shape))))
    else:
        return layout


def _decide_box_dim(
    s_buf: "Buffer",
    g_buf: "Buffer",
    g_ext: List,
    s_st: List,
    s_ext: List,
    g_st: List,
) -> tuple:
    """Decide box dimensions for TMA copy.

    This function handles:
    - Get swizzle mode from shared buffer layout
    - Get tile layout from shared buffer (handle ComposeLayout/SwizzleLayout)
    - Slice shared and global layouts to get copy region
    - Group layouts by extents
    - Compute box_dim from contiguous shards via compute_box_dim()
    - Verify TMA atom divisibility

    Parameters
    ----------
    s_buf : Buffer
        The shared memory buffer
    g_buf : Buffer
        The global memory buffer
    g_ext : List
        Global region extents
    s_st : List
        Shared region starts
    s_ext : List
        Shared region extents
    g_st : List
        Global region starts

    Returns
    -------
    tuple
        (swizzle_mode, box_dim, contiguous_indices, contiguous_extent,
         grouped_shared, grouped_global)
    """
    # Step 1: Get swizzle mode from shared buffer layout
    swizzle_mode = get_swizzle_mode_from_layout(s_buf.layout)
    if swizzle_mode is None:
        raise ValueError(f"Cannot determine swizzle mode from layout: {s_buf.layout}")

    # Step 2: Get the tile layout from shared buffer (handle ComposeLayout)
    shared_tile_layout = to_tile_layout(s_buf.layout, s_buf.shape)

    # Step 3: Slice the shared layout to get copy region layout
    s_region_tuples = [(s_st[i], s_st[i] + s_ext[i]) for i in range(len(s_st))]
    sliced_shared = shared_tile_layout.slice([s for s in s_buf.shape], s_region_tuples)
    if sliced_shared is None:
        raise ValueError("Cannot slice shared memory layout for TMA copy")

    # Step 4: Slice the global layout to get copy region layout
    g_region_tuples = [(g_st[i], g_st[i] + g_ext[i]) for i in range(len(g_st))]
    sliced_global = g_buf.layout.slice([e for e in g_buf.shape], g_region_tuples)
    if sliced_global is None:
        raise ValueError("Cannot slice global memory layout for TMA copy")

    # Step 5: Group shared layout by global extents
    grouped_shared, _ = sliced_shared.canonicalize().group(g_ext)

    # Step 6: Group global layout by grouped_shared extents (1:1 shard mapping)
    grouped_extents = [s.extent for s in grouped_shared.shard]
    grouped_global, _ = sliced_global.canonicalize().group(grouped_extents)

    # Step 7: Compute box_dim from contiguous shards (same algorithm for all swizzle modes)
    box_dim, contiguous_indices, contiguous_extent = compute_box_dim(grouped_shared)

    # Step 8: Verify atom divisibility (only for swizzled layouts)
    if swizzle_mode != SwizzleMode.SWIZZLE_NONE:
        atom_shape = tma_atom_shape(s_buf.dtype, swizzle_mode)
        atom_total = atom_shape[0] * atom_shape[-1]
        if contiguous_extent % atom_total != 0:
            raise ValueError(
                f"Contiguous region {contiguous_extent} not divisible by TMA atom size {atom_total}. "
                f"box_dim={box_dim}, atom_shape={atom_shape}"
            )

    return (
        swizzle_mode,
        box_dim,
        contiguous_indices,
        contiguous_extent,
        grouped_shared,
        grouped_global,
    )


def _decide_tma_global_strides_and_shape(
    g_buf: "Buffer",
    grouped_global: TileLayout,
    contiguous_indices: List[int],
    box_dim: List[int],
    g_st: List,
    g_ext: List,
    dtype: str,
) -> tuple:
    """Decide TMA global strides and shape.

    This function handles:
    - Assert compact layout of g_buf
    - Extract raw strides from grouped_global using contiguous_indices
    - OOB detection and stride insertion
    - Sort strides using partial ordering
    - Compute shapes from sorted strides
    - Reorder shapes to match raw_strides order

    Note: box_dim is modified in place for OOB detection.

    Parameters
    ----------
    g_buf : Buffer
        The global memory buffer
    grouped_global : TileLayout
        The global layout grouped by shared extents
    contiguous_indices : List[int]
        Indices of contiguous shards
    box_dim : List[int]
        Box dimensions (modified in place for OOB)
    g_st : List
        Global region starts
    g_ext : List
        Global region extents
    dtype : str
        Data type string

    Returns
    -------
    tuple
        (tma_g_strides, tma_g_shape, tma_rank, tma_global_strides,
         g_buf_grouped, g_buf_separators)
    """
    dtype_bytes = tvm.DataType(dtype).bits // 8
    analyzer = Analyzer()

    # Step 9: Assert compactness of g_buf.layout using stride * extent check
    total_size = functools.reduce(lambda x, y: x * y, g_buf.shape, 1)
    g_buf_shards_data = [(shard.stride, shard.extent) for shard in g_buf.layout.shard]
    assert_compact_layout(g_buf_shards_data, total_size, analyzer)

    # Step 10: Get strides from grouped_global using contiguous_indices
    raw_strides = [grouped_global.shard[idx].stride for idx in contiguous_indices]

    # Step 11: OOB Detection - Group g_buf.layout by g_buf.shape
    g_buf_grouped, g_buf_separators = g_buf.layout.group(list(g_buf.shape))

    # For each dimension, check if the highest shard could be OOB
    oob_info = []  # List of (stride, extent) tuples
    for dim in range(len(g_buf.shape)):
        # Get the logically highest (first) shard for this dimension
        highest_shard_idx = g_buf_separators[dim]
        highest_shard = g_buf_grouped.shard[highest_shard_idx]

        # Compute product of all other shards' extents in this dimension (excluding the highest)
        other_extents_product = 1
        for shard_idx in range(highest_shard_idx + 1, g_buf_separators[dim + 1]):
            other_extents_product *= g_buf_grouped.shard[shard_idx].extent

        # Compute the decomposed coordinate at the end of the copy region
        end_coord = g_st[dim] + g_ext[dim]
        decomposed_coord = tvm.tir.floordiv(end_coord, other_extents_product)

        # If we cannot prove it's in bounds, mark as potential OOB
        if not analyzer.can_prove(decomposed_coord <= highest_shard.extent):
            oob_info.append((highest_shard.stride, highest_shard.extent))

    # Add OOB strides to front (with deduplication against existing raw_strides)
    for oob_stride, oob_extent in oob_info:
        # Insert oob_stride
        is_duplicate = any(
            analyzer.can_prove_equal(oob_stride, existing) for existing in raw_strides
        )
        if not is_duplicate:
            raw_strides.insert(0, oob_stride)
            box_dim.insert(0, 1)

        # Insert oob_stride * oob_extent
        second_stride = oob_stride * oob_extent
        is_duplicate = any(
            analyzer.can_prove_equal(second_stride, existing)
            for existing in (*raw_strides, total_size)
        )
        if not is_duplicate:
            raw_strides.insert(0, second_stride)
            box_dim.insert(0, 1)
    assert len(box_dim) <= 5, f"only support up to 5 dimensions for TMA copy"

    # Step 12: Sort strides for shape computation
    sorted_order = sort_strides_partial_order(raw_strides, analyzer)
    sorted_strides = [raw_strides[i] for i in sorted_order]
    tma_g_strides = raw_strides  # Keep original order for TMA

    # Step 13: Compute shapes from sorted strides
    sorted_shapes = []
    for i in range(len(sorted_strides)):
        if i == len(sorted_strides) - 1:
            next_stride = total_size
        else:
            next_stride = sorted_strides[i + 1]
        sorted_shapes.append(tvm.tir.floordiv(next_stride, sorted_strides[i]))

    # Reorder shapes to match raw_strides order (tma_g_strides)
    inverse_order = [0] * len(sorted_order)
    for k, i in enumerate(sorted_order):
        inverse_order[i] = k
    tma_g_shape = [sorted_shapes[inverse_order[i]] for i in range(len(sorted_shapes))]

    tma_rank = len(box_dim)
    tma_global_strides = [s * dtype_bytes for s in tma_g_strides]

    return (
        tma_g_strides,
        tma_g_shape,
        tma_rank,
        tma_global_strides,
        g_buf_grouped,
        g_buf_separators,
    )


def _build_iteration_space(
    grouped_shared: TileLayout,
    grouped_global: TileLayout,
    contiguous_indices: List[int],
) -> List[dict]:
    """Build iteration space from non-contiguous shards.

    This function identifies non-contiguous shard indices and builds
    iteration info for each non-contiguous shard.

    Parameters
    ----------
    grouped_shared : TileLayout
        The shared layout grouped by global extents
    grouped_global : TileLayout
        The global layout grouped by shared extents
    contiguous_indices : List[int]
        Indices of contiguous shards

    Returns
    -------
    List[dict]
        List of dicts with 'extent', 's_stride', 'g_stride' for each
        non-contiguous shard
    """
    # Step 14: Identify non-contiguous shards for iteration
    all_indices = set(range(len(grouped_shared.shard)))
    non_contiguous_indices = sorted(
        all_indices - set(contiguous_indices),
        key=lambda i: grouped_shared.shard[i].stride,
        reverse=True,  # outermost first (largest stride first)
    )

    # Step 15: Build iteration space from non-contiguous shards
    iter_info = []
    for idx in non_contiguous_indices:
        iter_info.append(
            {
                "extent": grouped_shared.shard[idx].extent,
                "s_stride": grouped_shared.shard[idx].stride,
                "g_stride": grouped_global.shard[idx].stride,
            }
        )

    return iter_info


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

    # For now, we require that the global side layout is trivial.
    # TODO(bohan): support strided global memory in the future.
    if not g_buf.layout.is_trivial():
        raise ValueError(f"Global buffer layout is not trivial: {g_buf.layout}")

    # ---------------------------------------------------------------------
    # Region metadata
    # ---------------------------------------------------------------------
    g_st = [r.min for r in global_region.region]
    g_ext = [r.extent for r in global_region.region]
    s_st = [r.min for r in shared_region.region]
    s_ext = [r.extent for r in shared_region.region]

    # =========================================================================
    # PHASE 1: Deciding box_dim
    # =========================================================================
    (swizzle_mode, box_dim, contiguous_indices, _, grouped_shared, grouped_global) = (
        _decide_box_dim(s_buf, g_buf, g_ext, s_st, s_ext, g_st)
    )

    # =========================================================================
    # PHASE 2: Deciding TMA global strides/shape
    # =========================================================================
    (tma_g_strides, tma_g_shape, tma_rank, tma_global_strides, g_buf_grouped, g_buf_separators) = (
        _decide_tma_global_strides_and_shape(
            g_buf, grouped_global, contiguous_indices, box_dim, g_st, g_ext, s_buf.dtype
        )
    )

    # =========================================================================
    # Build iteration space
    # =========================================================================
    iter_info = _build_iteration_space(grouped_shared, grouped_global, contiguous_indices)

    # ---------------------------------------------------------------------
    # Launch configuration & common symbols
    # ---------------------------------------------------------------------
    cta_group = op_call.config.get("cta_group", None)
    if cta_group is None:
        cta_group = 1 if sctx.target.arch == "sm_100a" else -1

    cta_mask = op_call.config.get("cta_mask", None)
    if cta_mask is not None:
        assert direction == "g2s", "cta_mask is only supported for global to shared copy"
    else:
        cta_mask = 0

    tensor_map = T.Var(g_buf.data.name + "_tensormap", dtype=T.handle("tensormap").type_annotation)

    if direction == "g2s":
        # get mbar from config
        mbar = op_call.config.get("mbar", None)
        if mbar is None:
            raise ValueError("mbar is not set in config")
    use_tma_reduce = op_call.config.get("use_tma_reduce", None)

    # ---------------------------------------------------------------------
    # Device‑side TIR implementation
    # ---------------------------------------------------------------------
    analyzer = Analyzer()

    # Filter to only iter_info entries with extent > 1
    active_iter_info = [(i, info) for i, info in enumerate(iter_info) if info["extent"] > 1]
    # If no active iterations, use a single iteration with extent 1
    loop_extents = [info["extent"] for _, info in active_iter_info] if active_iter_info else [1]
    loop_s_strides = [info["s_stride"] for _, info in active_iter_info]
    loop_g_strides = [info["g_stride"] for _, info in active_iter_info]

    # Helper function to find matching TMA index for shard-based decomposition
    def find_matching_tma_idx(shard_stride, tma_strides, ana):
        """Find the TMA index whose stride is:
        (a) in tma_strides
        (b) smaller than shard_stride (shard_stride % tma_stride == 0 and not equal)
        (c) largest among all satisfying (a) and (b)
        """
        best_tma_idx = None
        best_tma_stride = None
        for tma_idx, tma_stride in enumerate(tma_strides):
            # Check shard_stride >= tma_stride via divisibility
            is_smaller = ana.can_prove_equal(tvm.tir.floormod(shard_stride, tma_stride), 0)

            if is_smaller:
                if best_tma_stride is None:
                    best_tma_idx = tma_idx
                    best_tma_stride = tma_stride
                elif ana.can_prove_equal(tvm.tir.floormod(tma_stride, best_tma_stride), 0):
                    # tma_stride >= best_tma_stride, update
                    best_tma_idx = tma_idx
                    best_tma_stride = tma_stride
        return best_tma_idx, best_tma_stride

    # Build shard-to-TMA mapping for g_st decomposition
    shard_to_tma = {}  # (dim, local_shard_idx) -> (tma_idx, multiplier)
    for dim in range(len(g_buf.shape)):
        for shard_idx in range(g_buf_separators[dim], g_buf_separators[dim + 1]):
            shard = g_buf_grouped.shard[shard_idx]
            local_idx = shard_idx - g_buf_separators[dim]
            tma_idx, tma_stride = find_matching_tma_idx(shard.stride, tma_g_strides, analyzer)
            if tma_idx is not None:
                multiplier = tvm.tir.floordiv(shard.stride, tma_stride)
                shard_to_tma[(dim, local_idx)] = (tma_idx, multiplier)

    def compute_offsets_and_tma_coords(loop_vars):
        """Compute s_offset and tma_coords from loop variables.

        Args:
            loop_vars: Loop variables from T.grid (single var if 1D, tuple if multi-D)

        Returns:
            Tuple of (s_offset, reversed tma_coords)
        """
        # Compute s_offset (unchanged)
        s_offset = 0
        for i in range(len(active_iter_info)):
            loop_var = loop_vars[i] if len(loop_extents) > 1 else loop_vars
            s_offset = s_offset + loop_var * loop_s_strides[i]

        # Initialize TMA coordinates
        tma_coords = [0] * tma_rank

        # Decompose each g_st[dim] into shard coordinates
        for dim in range(len(g_buf.shape)):
            num_shards = g_buf_separators[dim + 1] - g_buf_separators[dim]
            coord_value = g_st[dim]

            # Compute divisor for multi-radix decomposition (shards ordered highest-stride first)
            divisor = 1
            for local_idx in range(num_shards - 1, -1, -1):
                global_shard_idx = g_buf_separators[dim] + local_idx
                shard = g_buf_grouped.shard[global_shard_idx]

                # Multi-radix decomposition
                shard_coord = (
                    tvm.tir.floormod(tvm.tir.floordiv(coord_value, divisor), shard.extent)
                    if local_idx > 0
                    else tvm.tir.floordiv(coord_value, divisor)
                )

                # Accumulate to matching TMA coordinate
                if (dim, local_idx) in shard_to_tma:
                    tma_idx, multiplier = shard_to_tma[(dim, local_idx)]
                    tma_coords[tma_idx] = tma_coords[tma_idx] + shard_coord * multiplier

                divisor = divisor * shard.extent

        # Handle loop offset contribution
        for i in range(len(active_iter_info)):
            loop_var = loop_vars[i] if len(loop_extents) > 1 else loop_vars
            g_stride = loop_g_strides[i]

            tma_idx, tma_stride = find_matching_tma_idx(g_stride, tma_g_strides, analyzer)
            if tma_idx is not None:
                multiplier = tvm.tir.floordiv(g_stride, tma_stride)
                tma_coords[tma_idx] = tma_coords[tma_idx] + loop_var * multiplier

        return s_offset, reversed(tma_coords)

    # fmt: off
    @T.prim_func(tirx=True, check_well_formed=False)
    def impl():
        for loop_vars in T.grid(*loop_extents):
            s_offset, tma_coords = T.meta_var(compute_offsets_and_tma_coords(loop_vars))
            s_buf_w_offset = T.decl_buffer(s_buf.shape, s_buf.dtype, s_buf.data, elem_offset=s_buf.elem_offset + s_offset, scope=s_buf.scope(), layout=to_tile_layout(s_buf.layout, s_buf.shape))

            # Emit TMA copy with computed offsets and coordinates
            if direction == "g2s":
                T.ptx.cp_async.bulk.tensor.g2c(
                    tma_rank,
                    s_buf_w_offset.ptr_to(s_st),
                    mbar,
                    tensor_map,
                    *tma_coords,
                    cta_mask=cta_mask,
                    cta_group=cta_group,
                    cache_hint=op_call.config.get("cache_hint", ""),
                )
            else:
                if use_tma_reduce is None:
                    T.ptx.cp_async.bulk.tensor.s2g(
                        tma_rank,
                        s_buf_w_offset.ptr_to(s_st),
                        tensor_map,
                        *tma_coords,
                        cache_hint=op_call.config.get("cache_hint", ""),
                    )
                else:
                    T.ptx.cp_async.bulk.tensor.s2g_reduce(
                        tma_rank,
                        s_buf_w_offset.ptr_to(s_st),
                        tensor_map,
                        *tma_coords,
                        cache_hint=op_call.config.get("cache_hint", ""),
                        red_op=use_tma_reduce,
                    )

    # fmt: on
    # ---------------------------------------------------------------------
    # Host‑side tensor‑map creation
    # ---------------------------------------------------------------------
    element_strides = [1] * tma_rank

    # Use the tensor map shape and strides computed from the layout analysis
    # This works for both SWIZZLE_NONE and swizzled layouts
    # tma_g_shape and tma_g_strides are already computed above
    tma_g_strides_for_map = (
        tma_global_strides[:-1] if tma_rank > 1 else []
    )  # Exclude innermost stride

    # fmt: off
    @T.prim_func(tirx=True, check_well_formed=False)
    def create_tensor_map():
        with T.LetStmt(T.tvm_stack_alloca("tensormap", 1), var=tensor_map):
            T.call_packed(
                "runtime.cuTensorMapEncodeTiled",
                tensor_map,
                g_buf.dtype,
                tma_rank,
                g_buf.data,
                *reversed(tma_g_shape),
                *reversed(tma_g_strides_for_map) if tma_rank > 1 else [],
                *reversed(box_dim),
                *element_strides,
                0,  # CU_TENSOR_MAP_INTERLEAVE_NONE
                swizzle_mode.value,
                2,  # CU_TENSOR_MAP_L2_PROMOTION_L2_128B
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
