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

"""Implementation of copy_async operator dispatchs for CUDA targets.

Registered op: copy_async (2 variants here; 2 more in copy.py).
See the @register_dispatch blocks below for detailed documentation with
before/after IR examples.
"""

import functools
from typing import Optional

import tvm
from tvm.arith import Analyzer
from tvm.script import tirx as Tx
from tvm.tir import Buffer, PrimFunc
from tvm.tir.layout import ComposeLayout, S, SwizzleLayout, TileLayout, TLayout
from tvm.tir.stmt import OpCall
from tvm.tirx.op_dispatch import (
    DispatchContext,
    fail,
    predicate,
    register_dispatch,
)

from .common import (
    CopyInstType,
    copy_vec_load_impl,
    validate_copy_op,
)
from .exec_scope_utils import single_thread
from .tma_utils import SwizzleMode, get_swizzle_mode_from_layout, tma_atom_shape


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
    indexed_shards = [
        (i, s) for i, s in enumerate(layout.shard) if s.axis.is_memory() and s.extent != 1
    ]

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


def sort_strides_partial_order(strides: list, analyzer: Analyzer) -> list[int]:
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


def assert_compact_layout(shards: list, total_size, analyzer: Analyzer):
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
                    f"Global layout is not compact: "
                    f"stride={stride} * extent={extent} = {expected_next} "
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


def to_tile_layout(layout: TLayout, shape: list[int]) -> TileLayout:
    """Convert a layout to a tile layout."""
    if isinstance(layout, ComposeLayout):
        return layout.tile_layout
    elif isinstance(layout, SwizzleLayout):
        return TileLayout(S[tuple(shape)])
    else:
        return layout


def _decide_box_dim(
    s_buf: "Buffer",
    g_buf: "Buffer",
    g_ext: list,
    s_st: list,
    s_ext: list,
    g_st: list,
) -> tuple:
    """Decide box dimensions for TMA copy.

    Handles transpose (smem column-major, gmem row-major) and multi-dim partition
    buffers by validating that global strides ascend in the shared contiguity order.

    Algorithm:
    1. Group shared by g_ext, group global by shared shard extents (1:1 mapping)
    2. Find contiguous region in shared (inner->outer by stride)
    3. Extract corresponding global strides
    4. Validate ascending stride order; trim box if violated
    5. Build box_dim from remaining contiguous shards

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
    analyzer = Analyzer()

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
    grouped_shared, s_seps = sliced_shared.canonicalize().group(g_ext)

    # Step 6: Group global layout by grouped_shared extents (1:1 shard mapping)
    grouped_extents = [s.extent for s in grouped_shared.shard]
    grouped_global, _ = sliced_global.canonicalize().group(grouped_extents)

    # Step 7a: Find contiguous region in shared (inner->outer by stride)
    box_indices, _ = find_contiguous_region(grouped_shared)
    # box_indices is in inner->outer order (lowest stride first)

    # Step 7b: Validate ascending global strides at the g_ext group level.
    # Shards within the same g_ext group may have non-ascending individual strides
    # but together form a contiguous block. We validate that the *effective* stride
    # (min global stride per g_ext group) ascends across groups.

    # Map each shard index to its g_ext group
    def _shard_to_group(shard_idx):
        for g in range(len(s_seps) - 1):
            if s_seps[g] <= shard_idx < s_seps[g + 1]:
                return g
        return -1

    # Collect unique g_ext groups in inner→outer order, with their effective strides
    seen_groups = {}  # group_id -> (min_global_stride, first_position_in_box)
    for pos, idx in enumerate(box_indices):
        g = _shard_to_group(idx)
        g_stride = grouped_global.shard[idx].stride
        if g not in seen_groups:
            seen_groups[g] = (g_stride, pos)
        else:
            prev_stride, first_pos = seen_groups[g]
            # Use the minimum stride within the group
            if analyzer.can_prove_equal(tvm.tir.floormod(prev_stride, g_stride), 0):
                seen_groups[g] = (g_stride, first_pos)

    # Sort groups by their first appearance (inner→outer)
    group_order = sorted(seen_groups.keys(), key=lambda g: seen_groups[g][1])
    group_strides = [seen_groups[g][0] for g in group_order]

    # Find the last valid group (ascending effective strides)
    valid_group_count = len(group_strides)
    for j in range(1, len(group_strides)):
        prev_s = group_strides[j - 1]
        cur_s = group_strides[j]
        is_multiple = analyzer.can_prove_equal(tvm.tir.floormod(cur_s, prev_s), 0)
        is_equal = analyzer.can_prove_equal(cur_s, prev_s)
        if not is_multiple or is_equal:
            valid_group_count = j
            break

    # Trim box_indices: keep only shards belonging to valid groups
    valid_groups = set(group_order[:valid_group_count])
    box_indices = [idx for idx in box_indices if _shard_to_group(idx) in valid_groups]

    # Step 7c: Build box_dim (outer->inner) and contiguous_indices
    box_indices_outer_to_inner = list(reversed(box_indices))
    box_dim = [grouped_shared.shard[i].extent for i in box_indices_outer_to_inner]
    contiguous_indices = box_indices_outer_to_inner

    # Remove extent-1 entries
    pairs = [(d, i) for d, i in zip(box_dim, contiguous_indices) if d != 1]
    if pairs:
        box_dim, contiguous_indices = [list(x) for x in zip(*pairs)]
    else:
        box_dim, contiguous_indices = [], []

    contiguous_extent = functools.reduce(lambda a, b: a * b, box_dim, 1)

    # Step 8: Verify atom divisibility (only for swizzled layouts)
    if swizzle_mode != SwizzleMode.SWIZZLE_NONE:
        atom_shape = tma_atom_shape(s_buf.dtype, swizzle_mode)
        atom_total = atom_shape[0] * atom_shape[-1]
        if contiguous_extent % atom_total != 0:
            raise ValueError(
                f"Contiguous region {contiguous_extent} not divisible by "
                f"TMA atom size {atom_total}. "
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
    contiguous_indices: list[int],
    box_dim: list[int],
    g_st: list,
    g_ext: list,
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
        # Skip dimensions with extent=1 — they can't be OOB
        if analyzer.can_prove_equal(g_ext[dim], 1):
            continue

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
    # If the box doesn't include a stride=1 (contiguous) global dimension, TMA cannot
    # handle this configuration. This happens for transpose layouts (e.g., column-major
    # shared with row-major global) where the box only covers non-innermost global dims.
    # TMA deposits data linearly in shared memory and cannot reorder elements, so we must
    # fall back to non-bulk-copy.
    if raw_strides and not any(analyzer.can_prove_equal(s, 1) for s in raw_strides):
        fail(
            "TMA box does not include the contiguous (stride=1) global dimension. "
            "Transpose layouts require non-bulk-copy."
        )

    assert len(box_dim) <= 5, "only support up to 5 dimensions for TMA copy"

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
    contiguous_indices: list[int],
) -> list[dict]:
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
    sctx: "DispatchContext",
) -> Optional["PrimFunc"]:
    """Schedule a copy between global <-> shared memory using CUDA TMA.

    This is a unified replacement for the previous
    ``copy_g2s_cta_tma_impl`` and ``copy_s2g_cta_tma_impl`` helpers.  The
    direction is inferred from the scopes of *src* and *dst* buffers:

    * **global → shared**  ⇒  ``cp_async.bulk.tensor.g2c``
    * **shared → global**  ⇒  ``cp_async.bulk.tensor.s2g``
    """
    # ---------------------------------------------------------------------
    # Identify direction & basic legality checks
    # ---------------------------------------------------------------------
    op_call = OpCall.downcast(op_call)
    dst_buffer_region, src_buffer_region = op_call.dst, op_call.src
    src: Buffer = src_buffer_region.buffer
    dst: Buffer = dst_buffer_region.buffer

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

    # Build tensormap cache key from cuTensorMapEncodeTiled construction params.
    # tma_g_strides_for_map needs to be computed here for the cache key
    # (also used later in host-side code).
    _element_strides = [1] * tma_rank
    _tma_g_strides_for_map = tma_global_strides[:-1] if tma_rank > 1 else []

    def _val_key(v):
        """Convert a value (int, IntImm, or PrimExpr) to a cache-key-safe string."""
        return str(v)

    _tensormap_cache_key = (
        f"tensormap:{hash(g_buf.data)}:{g_buf.dtype}:{_val_key(tma_rank)}"
        f":{tuple(_val_key(v) for v in tma_g_shape)}"
        f":{tuple(_val_key(v) for v in _tma_g_strides_for_map)}"
        f":{tuple(_val_key(v) for v in box_dim)}"
        f":{_val_key(swizzle_mode.value)}"
    )
    _cached_tensormap = sctx.cache_get(_tensormap_cache_key)
    if _cached_tensormap is not None:
        tensor_map = _cached_tensormap
        _tensormap_is_cached = True
    else:
        tensor_map = Tx.Var(
            g_buf.data.name + "_tensormap", dtype=Tx.handle("tensormap").type_annotation
        )
        _tensormap_is_cached = False

    if direction == "g2s":
        # get mbar from config
        mbar = op_call.config.get("mbar", None)
        if mbar is None:
            raise ValueError("mbar is not set in config")
    use_tma_reduce = op_call.config.get("use_tma_reduce", None)

    # ---------------------------------------------------------------------
    # Device-side TIR implementation
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

    # When swizzle or OOB detection creates TMA dimensions that don't
    # directly match any shard in g_buf_grouped, split the containing
    # shard so every TMA stride has a corresponding shard entry.  This
    # keeps the shard-based coordinate decomposition working for all cases.
    _matched_tma = set(tma_idx for tma_idx, _ in shard_to_tma.values())
    _unmatched_tma = [i for i in range(tma_rank) if i not in _matched_tma]

    if _unmatched_tma:
        # Build mutable shard lists per dimension: [[stride, extent], ...]
        _dim_shards = []
        for dim in range(len(g_buf.shape)):
            shards = []
            for si in range(g_buf_separators[dim], g_buf_separators[dim + 1]):
                s = g_buf_grouped.shard[si]
                shards.append([int(s.stride), int(s.extent)])
            _dim_shards.append(shards)

        # Split shards to cover each unmatched TMA stride
        for ui in _unmatched_tma:
            ts = int(tma_g_strides[ui])
            for dim in range(len(g_buf.shape)):
                for si, (s, e) in enumerate(_dim_shards[dim]):
                    if s < ts < s * e and ts % s == 0 and e % (ts // s) == 0:
                        factor = ts // s
                        _dim_shards[dim][si : si + 1] = [[ts, e // factor], [s, factor]]
                        break
                else:
                    continue
                break

        # Rebuild separators
        g_buf_separators = [0]
        for dim in range(len(g_buf.shape)):
            g_buf_separators.append(g_buf_separators[-1] + len(_dim_shards[dim]))

        # Build refined shard list with .stride / .extent attributes
        class _Shard:
            __slots__ = ("extent", "stride")

            def __init__(self, stride, extent):
                self.stride, self.extent = stride, extent

        _refined_shards = []
        for dim in range(len(g_buf.shape)):
            for s, e in _dim_shards[dim]:
                _refined_shards.append(_Shard(s, e))

        # Replace shard accessor for coordinate decomposition
        g_buf_grouped_shards = _refined_shards

        # Rebuild shard_to_tma with refined shards
        shard_to_tma = {}
        for dim in range(len(g_buf.shape)):
            for local_idx in range(len(_dim_shards[dim])):
                s = _dim_shards[dim][local_idx][0]
                tma_idx, tma_stride = find_matching_tma_idx(s, tma_g_strides, analyzer)
                if tma_idx is not None:
                    multiplier = tvm.tir.floordiv(s, tma_stride)
                    shard_to_tma[(dim, local_idx)] = (tma_idx, multiplier)
    else:
        g_buf_grouped_shards = g_buf_grouped.shard

    def compute_offsets_and_tma_coords(loop_vars):
        """Compute s_offset and tma_coords from loop variables.

        Args:
            loop_vars: Loop variables from Tx.grid (single var if 1D, tuple if multi-D)

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
                shard = g_buf_grouped_shards[global_shard_idx]

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
    @Tx.prim_func(tirx=True, check_well_formed=False)
    def impl():
        for loop_vars in Tx.grid(*loop_extents):
            s_offset, tma_coords = Tx.meta_var(compute_offsets_and_tma_coords(loop_vars))
            s_buf_w_offset = Tx.decl_buffer(
                s_buf.shape, s_buf.dtype, s_buf.data,
                elem_offset=s_buf.elem_offset + s_offset,
                scope=s_buf.scope(),
                layout=to_tile_layout(s_buf.layout, s_buf.shape),
            )

            # Emit TMA copy with computed offsets and coordinates
            if direction == "g2s":
                Tx.ptx.cp_async.bulk.tensor.g2c(
                    tma_rank,
                    s_buf_w_offset.ptr_to(s_st),
                    mbar,
                    tensor_map,
                    cta_mask,
                    cta_group,
                    op_call.config.get("cache_hint", ""),
                    *tma_coords,
                )
            else:
                if use_tma_reduce is None:
                    Tx.ptx.cp_async.bulk.tensor.s2g(
                        tma_rank,
                        s_buf_w_offset.ptr_to(s_st),
                        tensor_map,
                        op_call.config.get("cache_hint", ""),
                        *tma_coords,
                    )
                else:
                    Tx.ptx.cp_async.bulk.tensor.s2g_reduce(
                        tma_rank,
                        s_buf_w_offset.ptr_to(s_st),
                        tensor_map,
                        op_call.config.get("cache_hint", ""),
                        use_tma_reduce,
                        *tma_coords,
                    )

    # fmt: on
    # ---------------------------------------------------------------------
    # Host-side tensor-map creation (skipped on cache hit)
    # ---------------------------------------------------------------------
    if not _tensormap_is_cached:
        element_strides = _element_strides
        tma_g_strides_for_map = _tma_g_strides_for_map

        # fmt: off
        @Tx.prim_func(tirx=True, check_well_formed=False)
        def create_tensor_map():
            Tx.Bind(Tx.tvm_stack_alloca("tensormap", 1), var=tensor_map)
            Tx.call_packed(
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

        sctx.add_init_stmt(create_tensor_map.body, host=True)
        sctx.cache_set(_tensormap_cache_key, tensor_map)

    return impl


# === Variant: copy_async/non-bulk-copy (priority=20) ===
#
# When: any valid async copy. Highest priority — tried first before TMA.
# Succeeds for global↔shared copies where vectorization works; fails back
# to TMA for single-thread scope or when cp.async doesn't apply.
#
# Before (OpCall):
#     with Tx.cta():
#         Tx.copy_async(A_smem[0:64, 0:64], A[0:64, 0:64])
#
# After (uses cp.async PTX instead of regular load/store):
#     for s in Tx.serial(ceildiv(4096, 8 * 128)):
#         for vec in Tx.vectorized(8):
#             fused = s * 1024 + threadIdx.x * 8 + vec
#             if fused < 4096:
#                 # emitted as cp.async.bulk.shared.global [smem_addr], [gmem_addr], 16
#                 A_smem[idx] = A[idx]
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
def copy_async_dispatch_cp_async(op: OpCall, sctx: DispatchContext) -> PrimFunc:
    return copy_vec_load_impl(op, sctx, CopyInstType.CP_ASYNC)


# === Variant: copy_async/tma (priority=10) ===
#
# When: valid async copy at single-thread exec scope. Applies to global↔shared
# copies on Hopper+ (SM90+) where TMA hardware is available.
# Falls back (DispatchFail) for scope pairs TMA can't handle.
#
# Before (OpCall):
#     Tx.copy_async(A_smem[0:128, 0:64], A[row:row+128, 0:64])
#     # A: global float16, A_smem: shared float16 with 128B swizzle layout
#
# After (generates host-side tensormap + device-side TMA instruction):
#   Host init:
#     tensor_map = Tx.cuda.tma_create_tensormap(
#         tensor_rank=2, gmem_ptr=A.data,
#         global_shape=[M, 64], global_strides=[64, 1],
#         box_dim=[128, 64], element_strides=[1, 1],
#         swizzle_mode="SWIZZLE_128B", ...)
#   Device:
#     Tx.ptx.cp_async.bulk.tensor.g2c(
#         rank=2, smem_ptr=A_smem.ptr_to(0),
#         mbar=barrier, tensor_map=tensor_map,
#         coord0=0, coord1=row)
#
# Key internal algorithm (copy_tma_impl):
#   1. Infer swizzle mode from smem layout (none/32B/64B/128B)
#   2. Group layout axes, find contiguous regions
#   3. Compute box_dim — the tile size per TMA instruction
#   4. Sort global strides to determine TMA addressing order
#   5. Build iteration space if tensor > one TMA box
#   6. Create tensormap descriptor on host
#   7. Emit cp_async.bulk.tensor.g2c (load) or s2g (store)
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
def copy_async_dispatch_tma(op: OpCall, sctx: DispatchContext) -> PrimFunc:
    return copy_tma_impl(op, sctx)


# ---------------------------------------------------------------------------
# Blackwell tmem/smem async variants (moved from copy.py for colocation).
# The impl functions remain in copy.py; only dispatch registration lives here.
# ---------------------------------------------------------------------------
from .copy import (  # noqa: E402
    _is_valid_copy,
    _is_valid_smem_tmem_copy,
    _scope_allowed,
    _single_thread_exec,
    copy_smem_tmem_impl,
    copy_tmem_local_impl,
)
from .exec_scope_utils import exec_scope_ok  # noqa: E402


# === Variant: copy_async/tmem<->local (priority=10) ===
#
# Same as copy/tmem<->local but async (async_op=True).
# After: Tx.ptx.tcgen05.ld/st with async completion signaling.
@register_dispatch(
    "copy_async",
    "cuda",
    variant="tmem<->local",
    priority=10,
    when=[
        predicate("validate_copy_op", _is_valid_copy),
        predicate("exec_scope", exec_scope_ok, expected_scopes=["warpgroup"]),
        predicate(
            "storage_scope", _scope_allowed, allowed_pairs=[("tmem", "local"), ("local", "tmem")]
        ),
    ],
)
def copy_async_schedule_tmem_local_async(op_call: OpCall, sctx: DispatchContext) -> PrimFunc:
    return copy_tmem_local_impl(op_call, sctx, async_op=True)


# === Variant: copy_async/smem->tmem (priority=10) ===
#
# Same as copy/smem->tmem but async (async_op=True).
@register_dispatch(
    "copy_async",
    "cuda",
    variant="smem->tmem",
    priority=10,
    when=[
        predicate("validate_smem_tmem_copy", _is_valid_smem_tmem_copy),
        predicate("exec_scope", _single_thread_exec),
    ],
)
def copy_async_schedule_smem_tmem(op_call: OpCall, sctx: DispatchContext) -> PrimFunc:
    return copy_smem_tmem_impl(op_call, sctx, async_op=True)
