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

"""Common utilities for operator scheduling."""

from collections import namedtuple, defaultdict
from typing import Tuple, Optional, Dict, Callable, List
from functools import wraps, reduce
import operator
import functools
import itertools
from dataclasses import dataclass
from tvm.arith.analyzer import Analyzer
from tvm.script import tir as T
from tvm.ir import Range
from tvm.tir import BufferRegion, Buffer, PrimFunc, Var, PrimExpr
from tvm.tir.stmt import OpCall
from tvm.tir.expr_functor import ExprMutator
from tvm._ffi import get_global_func
from tvm.tirp.op_schedule import ScheduleContext

f_normalize_trn_layout_with_shape = get_global_func("tir.NormalizeTrainiumLayoutWithShape")
f_normalize_tile_layout_with_shape = get_global_func("tir.NormalizeTileLayoutWithShape")

# Used to generate the correct [:, None] for mask/predicate
nki_dim = "nki_dim"

# Represents the part of data iter covered by the buffer region
RangeInfo = namedtuple(
    "RangeInfo", ["start", "extent", "dim_in_data_iter", "dim_in_shape", "dim_type"]
)


def normalize_layout_with_shape(layout, shape):
    """Normalize a layout with a given shape.

    Parameters
    ----------
    layout : Union[T.TrainiumLayout, T.TileLayout]
        The layout to normalize
    shape : List[int]
        The shape to normalize with

    Returns
    -------
    Tuple[Union[T.TrainiumLayout, T.TileLayout], List[int]] :
        Normalized layout and separators

    Raises
    ------
    ValueError :
        If layout is not a valid layout type
    """
    if isinstance(layout, T.TrainiumLayout):
        ret = f_normalize_trn_layout_with_shape(layout, shape)
        return ret[0], ret[1]
    elif isinstance(layout, T.TileLayout):
        ret = f_normalize_tile_layout_with_shape(layout, shape)
        return ret[0], ret[1]
    else:
        raise ValueError("Invalid layout")


def get_layout_data_iters(layout):
    """Get the data iterators from a layout.

    Parameters
    ----------
    layout : Union[T.TrainiumLayout, T.TileLayout]
        The layout to get data iterators from

    Returns
    -------
    List :
        Data iterators

    Raises
    ------
    ValueError :
        If layout is not a valid layout type
    """
    if isinstance(layout, T.TrainiumLayout):
        return layout.combined_1d_layout.data_iter_array
    elif isinstance(layout, T.TileLayout):
        return layout.data_iter_array
    else:
        raise ValueError("Invalid layout")


# TODO: refactor all instruction-generation logic to use InstructionGenerator


def bound_buffer_region(buffer_region: BufferRegion, analyzer: Analyzer):
    """Relax the bounds of the buffer region to the maximum possible constant value.
        bound_buffer_region must be called on the arguments of any function involving range_info
    (e.g. generate_axes_in_region, find_max_inst_size_*, etc.)
    Parameters
    ----------
    buffer_region : BufferRegion
        The buffer region to bound
    analyzer : Analyzer
        The analyzer to use

    Returns
    -------
    BufferRegion :
        The bounded buffer region
    """
    if not isinstance(buffer_region, BufferRegion):
        return buffer_region
    region = []
    for r in buffer_region.region:
        bound = analyzer.const_int_bound(r.extent)
        region.append(Range.from_min_extent(r.min, bound.max_value))
    return BufferRegion(buffer_region.buffer, region)


def make_guard(buffer_region: BufferRegion, analyzer: Analyzer):
    """Make a guard function for the buffer region.
    The guard function takes in the axes of the buffer region and
    checks if the buffer region is accessed within the bounds.

    Parameters
    ----------
    buffer_region : BufferRegion
        The buffer region to make a guard for
    analyzer : Analyzer
        The analyzer to use

    Returns
    -------
    Callable :
        A function that takes axes and returns a guard expression
    """
    bound_region = bound_buffer_region(buffer_region, analyzer)
    relaxed_dims = [
        i
        for i, (r1, r2) in enumerate(zip(bound_region.region, buffer_region.region))
        if not analyzer.can_prove(r1.extent == r2.extent)
    ]
    guard = lambda axes: reduce(
        T.And,
        [axes[i] < r.extent for i, r in enumerate(buffer_region.region) if i in relaxed_dims],
        True,
    )
    return guard


def infer_range_info(buffer_region: BufferRegion, analyzer: Analyzer):
    """Infer the range information for a buffer region.
    The range information is a list of RangeInfo, which contains information of
    data iter that is covered by the buffer region.


    Parameters
    ----------
    buffer_region : BufferRegion
        The buffer region to infer range information for
    analyzer : Analyzer
        The analyzer to use

    Returns
    -------
    Tuple[List[RangeInfo], Union[T.TrainiumLayout, T.TileLayout], List[int]] :
        The range information, normalized layout, and separators

    Raises
    ------
    AssertionError :
        If the layout is invalid
    """
    layout = buffer_region.buffer.layout
    layout, seps = normalize_layout_with_shape(layout, buffer_region.buffer.shape)
    data_iters = get_layout_data_iters(layout)
    tiled_range_infos_per_dim = []

    for i in range(len(seps) - 1):
        r = buffer_region.region[i]
        st = r.min
        ext = r.extent
        for j in reversed(range(seps[i], seps[i + 1])):
            dim_type = layout.dimension_types[j] if isinstance(layout, T.TrainiumLayout) else -1
            if analyzer.can_prove_equal(ext, 1):
                break
            if dim_type == T.TrainiumLayout.Partition and (
                not analyzer.can_prove(st % data_iters[j].extent == 0)
                or not analyzer.can_prove(ext % data_iters[j].extent == 0)
            ):
                assert False, "Invalid layout"
            if analyzer.can_prove(ext % data_iters[j].extent == 0) and analyzer.can_prove(
                st % data_iters[j].extent == 0
            ):
                st = st // data_iters[j].extent
                ext = ext // data_iters[j].extent
                tiled_range_infos_per_dim.append(RangeInfo(0, data_iters[j].extent, j, i, dim_type))
                continue
            if analyzer.can_prove(st + ext <= data_iters[j].extent):
                tiled_range_infos_per_dim.append(RangeInfo(st, ext, j, i, dim_type))
                break
            assert False, f"Cannot analyze physical tensor region for: {buffer_region}"

    # Put partition axis at front, then free axis with lower stride at front
    tiled_range_infos_per_dim = sorted(
        tiled_range_infos_per_dim, key=lambda x: (x.dim_type, data_iters[x.dim_in_data_iter].stride)
    )
    return tiled_range_infos_per_dim, layout, seps


def generate_axes_in_region(
    buffer_region: BufferRegion,
    inst_stride,
    inst_data_iters,
    analyzer,
):
    """Generate axes function for a buffer region.
    The function takes in the block loops, the f loop, the p loop, and the dim2block_var
    (for each shape dim which block var it should use as part of the axes).

    Parameters
    ----------
    buffer_region : BufferRegion
        The buffer region to generate axes for
    inst_stride : int
        The instruction stride
    inst_data_iters : Dict[int, int]
        The instruction data iterators
    analyzer : Analyzer
        The analyzer to use

    Returns
    -------
    Callable :
        A function that takes block loops, f loop, p loop, and dim2block_var and returns axes
    """
    range_info, tiled_layout, seps = infer_range_info(buffer_region, analyzer)
    dim_in_region = {
        range_info[i].dim_in_data_iter: range_info[i].extent for i in range(len(range_info))
    }
    data_iters = get_layout_data_iters(tiled_layout)

    # dim2block_var: shape dim -> block var
    def f(b_loops, f_loop, p_loop, dim2block_var=None):
        b_loops = list(b_loops)

        def extract_block_loop(extent, dim_in_shape):
            nonlocal b_loops
            idx = 0
            if dim2block_var is not None:
                assert dim_in_shape in dim2block_var
                idx = dim2block_var[dim_in_shape]
            b_loop, b_extent = b_loops[idx]
            ret = b_loop % b_extent // (b_extent // extent)
            b_extent = b_extent // extent
            b_loops[idx] = (b_loop, b_extent)
            return ret

        region_indices = []
        for i in range(len(seps) - 1):
            index_in_this_dim = 0
            for j in range(seps[i], seps[i + 1]):
                index_in_this_dim *= data_iters[j].extent
                dim_type = tiled_layout.dimension_types[j]
                if dim_type == T.TrainiumLayout.Partition:
                    index_in_this_dim += (p_loop // data_iters[j].stride) % data_iters[j].extent
                elif j in inst_data_iters:
                    if analyzer.can_prove(data_iters[j].stride < inst_stride):
                        assert analyzer.can_prove(inst_stride % data_iters[j].stride == 0)
                        assert analyzer.can_prove(
                            dim_in_region[j]
                            % (inst_data_iters[j] * inst_stride // data_iters[j].stride)
                            == 0
                        )
                        index_in_this_dim += (
                            extract_block_loop(
                                dim_in_region[j]
                                // (inst_data_iters[j] * inst_stride // data_iters[j].stride),
                                i,
                            )
                            * inst_data_iters[j]
                            * inst_stride
                            // data_iters[j].stride
                        )
                        index_in_this_dim += (f_loop % inst_data_iters[j]) * (
                            inst_stride // data_iters[j].stride
                        )
                        index_in_this_dim += extract_block_loop(
                            inst_stride // data_iters[j].stride, i
                        )
                    else:
                        assert analyzer.can_prove_equal(dim_in_region[j] % inst_data_iters[j], 0)
                        index_in_this_dim += (
                            extract_block_loop(dim_in_region[j] // inst_data_iters[j], i)
                            * inst_data_iters[j]
                        )
                        index_in_this_dim += (
                            f_loop // (data_iters[j].stride // inst_stride)
                        ) % inst_data_iters[j]
                elif j in dim_in_region:
                    index_in_this_dim += extract_block_loop(dim_in_region[j], i)
            region_indices.append(index_in_this_dim)
        return region_indices

    return f


def get_ewise_dim_map(
    buffer_region: BufferRegion, second_buffer_region: BufferRegion, analyzer: Analyzer
):
    """Get the dimension map between two elementwise buffer regions.

    Parameters
    ----------
    buffer_region : BufferRegion
        The first buffer region
    second_buffer_region : BufferRegion
        The second buffer region
    analyzer : Analyzer
        The analyzer to use

    Returns
    -------
    Dict[int, int] :
        A dimension map from first to second buffer region

    Raises
    ------
    AssertionError :
        If dimensions do not match
    """
    extent_1 = [r.extent for r in buffer_region.region]
    extent_2 = [r.extent for r in second_buffer_region.region]
    extent_1_non_unit = [e for e in extent_1 if e != 1]
    extent_2_non_unit = [e for e in extent_2 if e != 1]
    assert all(
        [
            len(extent_1_non_unit) == len(extent_2_non_unit),
            all(
                analyzer.can_prove_equal(s, d) for s, d in zip(extent_1_non_unit, extent_2_non_unit)
            ),
        ]
    )
    dim_map = {}
    i = 0
    j = 0
    while i < len(extent_1) and j < len(extent_2):
        if analyzer.can_prove_equal(extent_1[i], 1):
            i += 1
            continue
        if analyzer.can_prove_equal(extent_2[j], 1):
            j += 1
            continue
        dim_map[i] = j
        i += 1
        j += 1
    return dim_map


def get_reduction_dim_map(
    src_buffer_region: BufferRegion,
    dst_buffer_region: BufferRegion,
    axes: Tuple[int],
    analyzer: Analyzer,
):
    """Get the dimension map between source and destination buffer regions for reduction.

    Parameters
    ----------
    src_buffer_region : BufferRegion
        The source buffer region
    dst_buffer_region : BufferRegion
        The destination buffer region
    axes : Tuple[int]
        The reduction axes
    analyzer : Analyzer
        The analyzer to use

    Returns
    -------
    Dict[int, int] :
        A dimension map from source to destination buffer region

    Raises
    ------
    AssertionError :
        If dimensions do not match
    """
    dst_region = dst_buffer_region.region
    dst_extent = [r.extent for r in dst_region]
    dst_non_unit_extent_ = [(i, e) for i, e in enumerate(dst_extent) if e != 1]
    src_region = src_buffer_region.region
    src_extent = [r.extent for r in src_region]
    src_non_unit_extent_ = [(i, e) for i, e in enumerate(src_extent) if e != 1]
    src_non_reduction_extents = [(i, e) for i, e in src_non_unit_extent_ if i not in axes]
    assert len(src_non_reduction_extents) == len(
        dst_non_unit_extent_
    ), f"Source and destination must have the same number of non-reduction extents: {len(src_non_reduction_extents)} != {len(dst_non_unit_extent_)}"
    for i in range(len(src_non_reduction_extents)):
        assert analyzer.can_prove_equal(
            src_non_reduction_extents[i][1], dst_non_unit_extent_[i][1]
        ), f"Source and destination must have the same extent for non-reduction axes: {src_non_reduction_extents[i][1]} != {dst_non_unit_extent_[i][1]}"
    dim_map = {s[0]: d[0] for s, d in zip(src_non_reduction_extents, dst_non_unit_extent_)}
    return dim_map


def _find_corresponding_data_iter(
    range_info,
    tiled_layout,
    seps,
    second_tiled_layout,
    second_seps,
    buffer_dim_map,
    analyzer,
    extra_logical_stride=1,
):
    st, ext, dim_in_data_iter, dim_in_shape, dim_type = range_info
    data_iters = get_layout_data_iters(tiled_layout)
    second_data_iters = get_layout_data_iters(second_tiled_layout)
    stride_in_logical_dim = extra_logical_stride
    for i in range(dim_in_data_iter + 1, seps[dim_in_shape + 1]):
        stride_in_logical_dim *= data_iters[i].extent
    second_stride_in_logical_dim = 1
    second_extent = None
    second_data_iter = None
    second_buffer_dim = buffer_dim_map[dim_in_shape]
    # find the corresponding data iter in the second buffer region
    for i in reversed(range(second_seps[second_buffer_dim], second_seps[second_buffer_dim + 1])):
        second_stride_in_logical_dim *= second_data_iters[i].extent
        if second_stride_in_logical_dim > stride_in_logical_dim:
            assert analyzer.can_prove(
                second_stride_in_logical_dim % stride_in_logical_dim == 0
            ), "Invalid layout"
            second_extent = second_stride_in_logical_dim // stride_in_logical_dim
            second_data_iter = i
            second_new_stride = (
                second_data_iters[i].stride * second_data_iters[i].extent // second_extent
            )
            break
        elif second_stride_in_logical_dim == stride_in_logical_dim:
            assert i - 1 >= second_seps[second_buffer_dim]
            second_extent = second_data_iters[i - 1].extent
            second_data_iter = i - 1
            second_new_stride = second_data_iters[i - 1].stride
            break
    assert second_extent is not None
    return second_extent, second_new_stride, second_data_iter


# FIXME: this function tends to use the lowest-stride axis from the first buffer region, which
#       might leads to smaller instruction size. We should also try using the lowest-stride axis
#       from the second buffer region and choose the best one.
# infer an instruction size from buffer_region's access pattern that is compatible on second_buffer_region
def _refine_inst_tile(
    buffer_region: BufferRegion,
    second_buffer_region: BufferRegion,
    inst_size: int,
    inst_stride: int,
    analyzer: Analyzer,
    allowed_f_dim_1: Optional[Tuple[int]] = None,
    allowed_f_dim_2: Optional[Tuple[int]] = None,
    buffer_dim_map: Optional[Dict[int, int]] = None,
    check_partition=True,
):
    if allowed_f_dim_1 is None:
        allowed_f_dim_1 = tuple(range(len(buffer_region.buffer.shape)))
    if allowed_f_dim_2 is None:
        allowed_f_dim_2 = tuple(range(len(second_buffer_region.buffer.shape)))
    tiled_range_infos_per_dim, tiled_layout, seps = infer_range_info(buffer_region, analyzer)
    data_iters = get_layout_data_iters(tiled_layout)
    _, second_tiled_layout, second_seps = infer_range_info(second_buffer_region, analyzer)
    second_not_hbm = isinstance(second_buffer_region.buffer.layout, T.TrainiumLayout)
    second_data_iters = get_layout_data_iters(second_tiled_layout)
    new_inst_size = None
    new_first_inst_stride = None
    new_second_inst_stride = None
    new_inst_data_iters = {}
    buffer_dim_map = (
        get_ewise_dim_map(buffer_region, second_buffer_region, analyzer)
        if buffer_dim_map is None
        else buffer_dim_map
    )
    while len(tiled_range_infos_per_dim) > 0:
        range_info = tiled_range_infos_per_dim[0]
        tiled_range_infos_per_dim = tiled_range_infos_per_dim[1:]
        st, ext, dim_in_data_iter, dim_in_shape, dim_type = range_info
        if dim_in_shape not in buffer_dim_map:
            if dim_type == T.TrainiumLayout.Partition and check_partition:
                raise ValueError(
                    f"Partition dimension {dim_in_shape} in {buffer_region} not mapped to the second buffer region: {second_buffer_region}"
                )
            # skip if the dimension is not mapped to the second buffer region
            continue
        extra_logical_stride = 1
        if dim_type == T.TrainiumLayout.Free:
            # this range is not covered by the inst tile
            if analyzer.can_prove(ext * data_iters[dim_in_data_iter].stride <= inst_stride):
                continue
            if analyzer.can_prove(data_iters[dim_in_data_iter].stride >= inst_stride * inst_size):
                continue
            if new_inst_size is not None and not analyzer.can_prove(
                new_first_inst_stride * new_inst_size == data_iters[dim_in_data_iter].stride
            ):
                # the stride of the found data iter is not compatible with previous data iters
                break
            if (
                dim_in_shape not in allowed_f_dim_1
                or buffer_dim_map[dim_in_shape] not in allowed_f_dim_2
            ):
                # do not include this dimension if the dimension is not allowed to be free
                if new_inst_size is not None:
                    break
                else:
                    continue
            if analyzer.can_prove(
                ext * data_iters[dim_in_data_iter].stride > inst_stride * inst_size
            ):
                ext = (inst_stride * inst_size) // data_iters[dim_in_data_iter].stride
            if analyzer.can_prove(data_iters[dim_in_data_iter].stride < inst_stride):
                extra_logical_stride = inst_stride // data_iters[dim_in_data_iter].stride
        second_extent, second_stride, second_data_iter = _find_corresponding_data_iter(
            range_info,
            tiled_layout,
            seps,
            second_tiled_layout,
            second_seps,
            buffer_dim_map,
            analyzer,
            extra_logical_stride,
        )
        if dim_type == T.TrainiumLayout.Partition:
            if not second_not_hbm or not check_partition:
                continue
            if not analyzer.can_prove(st == 0):
                raise ValueError(f"Partition dimension not starting from 0. Start: {st}")
            if second_tiled_layout.dimension_types[second_data_iter] == T.TrainiumLayout.Free:
                raise ValueError(f"Partition dimension mismatch. Cannot perform ewise operation.")
            else:
                if not analyzer.can_prove(
                    second_data_iters[second_data_iter].stride
                    == data_iters[dim_in_data_iter].stride
                    and second_extent == data_iters[dim_in_data_iter].extent
                ):
                    # mismatch P dim
                    raise ValueError(
                        f"Partition dimension mismatch. Cannot perform ewise operation."
                    )
            continue
        # find max inst size for unary instruction
        if new_inst_size is not None and not analyzer.can_prove(
            new_second_inst_stride * new_inst_size == second_stride
        ):
            # the stride of the found data iter is not compatible with previous data iters
            break
        if new_inst_size is None:
            if not second_not_hbm and not analyzer.can_prove(
                second_data_iters[second_data_iter].stride == 1
            ):
                # stride of hbm tensor access must be 1
                break
            new_first_inst_stride = data_iters[dim_in_data_iter].stride
            new_second_inst_stride = second_data_iters[second_data_iter].stride
        if analyzer.can_prove(second_extent >= ext):
            new_inst_size = ext if new_inst_size is None else new_inst_size * ext
            new_inst_data_iters[dim_in_data_iter] = ext
            if not analyzer.can_prove(st == 0) or not analyzer.can_prove(
                ext == data_iters[dim_in_data_iter].extent
            ):
                break
        elif analyzer.can_prove(ext % second_extent == 0):
            new_inst_size = (
                second_extent if new_inst_size is None else new_inst_size * second_extent
            )
            new_inst_data_iters[dim_in_data_iter] = second_extent
            break
        else:
            assert False, "Invalid layout"
    if new_inst_size is None:
        return 1, 1, 1, {}
    return new_inst_size, new_first_inst_stride, new_second_inst_stride, new_inst_data_iters


def find_max_inst_size_from_one_region(
    buffer_region: BufferRegion,
    analyzer: Analyzer,
    allowed_f_dim: Optional[Tuple[int]] = None,
):
    """Find the maximum possible instruction size for an operation on a single buffer region.

    Parameters
    ----------
    buffer_region : BufferRegion
        The buffer region to analyze
    analyzer : Analyzer
        The analyzer to use
    allowed_f_dim : Optional[Tuple[int]]
        The allowed free dimensions

    Returns
    -------
    Tuple[int, int, Dict[int, int]] :
        Instruction size, stride, and data iterators
    """
    tiled_range_infos_per_dim, layout, seps = infer_range_info(buffer_region, analyzer)
    data_iters = get_layout_data_iters(layout)
    if allowed_f_dim is None:
        allowed_f_dim = tuple(range(len(buffer_region.buffer.shape)))
    # check largest inst size
    inst_size = 1
    inst_stride = 1
    inst_data_iters = {}
    prod_p_size = 1
    while len(tiled_range_infos_per_dim) > 0:
        range_info = tiled_range_infos_per_dim[0]
        tiled_range_infos_per_dim = tiled_range_infos_per_dim[1:]
        st, ext, dim_in_data_iter, dim_in_shape, dim_type = range_info
        if dim_type == T.TrainiumLayout.Partition:
            prod_p_size *= ext
            continue
        if dim_in_shape not in allowed_f_dim:
            if inst_size != 1:
                break
            else:
                continue
        if inst_size != 1 and not analyzer.can_prove(
            inst_stride * inst_size == data_iters[dim_in_data_iter].stride
        ):
            # the stride of the found data iter is not compatible with previous data iters
            break
        if inst_size == 1:
            inst_stride = data_iters[dim_in_data_iter].stride
        inst_data_iters[dim_in_data_iter] = ext
        inst_size *= ext
    # check p_dim covers whole partition
    assert analyzer.can_prove(
        prod_p_size == layout.partition_size
    ), "Partition size of the instruction must match that of the buffer region"

    return inst_size, inst_stride, inst_data_iters


def find_max_inst_size_unary(
    buffer_region: BufferRegion,
    second_buffer_region: BufferRegion,
    analyzer: Analyzer,
    allowed_f_dim_1: Optional[Tuple[int]] = None,
    allowed_f_dim_2: Optional[Tuple[int]] = None,
):
    """Find the maximum possible instruction size for a unary operation.

    Parameters
    ----------
    buffer_region : BufferRegion
        The destination buffer region
    second_buffer_region : BufferRegion
        The source buffer region
    analyzer : Analyzer
        The analyzer to use
    allowed_f_dim_1 : Optional[Tuple[int]]
        Allowed free dimensions for destination
    allowed_f_dim_2 : Optional[Tuple[int]]
        Allowed free dimensions for source

    Returns
    -------
    Tuple[int, int, Dict[int, int]] :
        Instruction size, first stride, and first data iterators
    """
    inst_size, inst_stride, _ = find_max_inst_size_from_one_region(
        buffer_region=buffer_region, analyzer=analyzer, allowed_f_dim=allowed_f_dim_1
    )
    inst_size, first_inst_stride, _, first_inst_data_iters = _refine_inst_tile(
        buffer_region,
        second_buffer_region,
        inst_size,
        inst_stride,
        analyzer,
        allowed_f_dim_1,
        allowed_f_dim_2,
    )
    return inst_size, first_inst_stride, first_inst_data_iters


def find_max_inst_size_transpose(
    buffer_region: BufferRegion, second_buffer_region: BufferRegion, analyzer: Analyzer
):
    """Find the maximum possible instruction size for a transpose operation.

    Parameters
    ----------
    buffer_region : BufferRegion
        The destination buffer region
    second_buffer_region : BufferRegion
        The source buffer region
    analyzer : Analyzer
        The analyzer to use

    Returns
    -------
    Tuple[int, Dict[int, int], int, Dict[int, int]] :
        First stride, first data iterators, second stride, second data iterators
    """
    inst_size, second_inst_stride, second_inst_data_iters = _find_max_inst_size_transpose(
        buffer_region, second_buffer_region, analyzer
    )
    assert inst_size == buffer_region.buffer.layout.partition_size
    inst_size, first_inst_stride, first_inst_data_iters = _find_max_inst_size_transpose(
        second_buffer_region, buffer_region, analyzer
    )
    assert inst_size == second_buffer_region.buffer.layout.partition_size
    return first_inst_stride, first_inst_data_iters, second_inst_stride, second_inst_data_iters


def _find_max_inst_size_transpose(
    buffer_region: BufferRegion,
    second_buffer_region: BufferRegion,
    analyzer: Analyzer,
    allowed_f_dim_2: Optional[Tuple[int]] = None,
):
    if allowed_f_dim_2 is None:
        allowed_f_dim_2 = tuple(range(len(second_buffer_region.buffer.shape)))
    tiled_range_infos_per_dim, tiled_layout, seps = infer_range_info(buffer_region, analyzer)
    data_iters = get_layout_data_iters(tiled_layout)
    _, second_tiled_layout, second_seps = infer_range_info(second_buffer_region, analyzer)
    inst_size = 1
    second_inst_stride = 1
    inst_data_iters = {}
    buffer_dim_map = get_ewise_dim_map(buffer_region, second_buffer_region, analyzer)
    while len(tiled_range_infos_per_dim) > 0:
        range_info = tiled_range_infos_per_dim[0]
        tiled_range_infos_per_dim = tiled_range_infos_per_dim[1:]
        st, ext, dim_in_data_iter, dim_in_shape, dim_type = range_info
        second_extent, second_new_stride, second_data_iter = _find_corresponding_data_iter(
            range_info,
            tiled_layout,
            seps,
            second_tiled_layout,
            second_seps,
            buffer_dim_map,
            analyzer,
        )
        if dim_type == T.TrainiumLayout.Partition:
            if not analyzer.can_prove(st == 0):
                raise ValueError(f"Partition dimension not starting from 0. Start: {st}")
            if second_tiled_layout.dimension_types[second_data_iter] == T.TrainiumLayout.Free:
                # The P dim on the first buffer must be mapped to a contiguous F dim on the second buffer
                if not analyzer.can_prove(
                    second_extent >= ext and ext == data_iters[dim_in_data_iter].extent
                ):
                    raise ValueError(
                        f"Cannot perform transpose due to corresponding F dim not contiguous"
                    )
                if buffer_dim_map[dim_in_shape] not in allowed_f_dim_2:
                    raise ValueError(
                        f"Cannot perform transpose due to corresponding F dim not allowed"
                    )
                if inst_size == 1:
                    second_inst_stride = second_new_stride
                elif not analyzer.can_prove(second_inst_stride * inst_size == second_new_stride):
                    raise ValueError(f"Transpose must be applied to contiguous F dim")
                inst_size *= ext
                inst_data_iters[second_data_iter] = ext
            else:
                # Cannot transpose 2 buffers with overlapping partition dimensions
                raise ValueError(
                    f"Cannot perform transpose due to overlapping partition dimensions"
                )
            continue
        return inst_size, second_inst_stride, inst_data_iters


def find_max_inst_size_matmul(
    rhs_buffer_region: BufferRegion,
    output_buffer_region: BufferRegion,
    analyzer: Analyzer,
    allowed_f_dim_rhs: Optional[Tuple[int]] = None,
    allowed_f_dim_output: Optional[Tuple[int]] = None,
    f_dim_map: Optional[Dict[int, int]] = None,
):
    """Find the maximum possible instruction size for a matrix multiplication operation.
    The instruction must be linear, which means the access pattern must be a * f_loop + b.

    Parameters
    ----------
    rhs_buffer_region : BufferRegion
        The right-hand-side buffer region
    output_buffer_region : BufferRegion
        The output buffer region
    analyzer : Analyzer
        The analyzer to use
    allowed_f_dim_rhs : Optional[Tuple[int]]
        Allowed free dimensions for RHS
    allowed_f_dim_output : Optional[Tuple[int]]
        Allowed free dimensions for output
    f_dim_map : Optional[Dict[int, int]]
        Dimension map from RHS to output

    Returns
    -------
    Tuple[int, int, int, Dict[int, int]] :
        Instruction size, RHS stride, output stride, and RHS data iterators
    """
    inst_size, inst_stride, _ = find_max_inst_size_from_one_region(
        buffer_region=rhs_buffer_region, analyzer=analyzer, allowed_f_dim=allowed_f_dim_rhs
    )
    return _refine_inst_tile(
        rhs_buffer_region,
        output_buffer_region,
        inst_size,
        inst_stride,
        analyzer,
        allowed_f_dim_rhs,
        allowed_f_dim_output,
        f_dim_map,
        check_partition=False,
    )


def bound_inst_with_limit(inst_size, inst_size_limit, analyzer):
    """Bound the instruction size with a limit.

    Parameters
    ----------
    inst_size : int
        The instruction size
    inst_size_limit : Optional[int]
        The instruction size limit (default: 512)
    analyzer : Analyzer
        The analyzer to use

    Returns
    -------
    Tuple[int, int] :
        The actual instruction size and additional block size
    """
    if inst_size_limit is None:
        return inst_size, 1
    if not analyzer.can_prove(inst_size <= inst_size_limit):
        # FIXME: this constraint can be relaxed if we support mask
        assert analyzer.can_prove(inst_size % inst_size_limit == 0)
        actual_inst_size = inst_size_limit
        additional_b_size = T.ceildiv(inst_size, inst_size_limit)
    else:
        actual_inst_size = inst_size
        additional_b_size = 1
    return actual_inst_size, additional_b_size


def bound_inst_data_iter_with_limit(buffer_region, inst_data_iters, inst_size_limit, analyzer):
    """Bound the instruction data iterators with a limit.

    Parameters
    ----------
    buffer_region : BufferRegion
        The buffer region
    inst_data_iters : Dict[int, int]
        The instruction data iterators
    inst_size_limit : Optional[int]
        The instruction size limit (default: 512)
    analyzer : Analyzer
        The analyzer to use

    Returns
    -------
    Tuple[int, Dict[int, int]] :
        The actual instruction size and data iterators
    """
    # default instruction size limit is 512
    if inst_size_limit is None:
        return (
            functools.reduce(operator.mul, [ext for ext in inst_data_iters.values()], 1),
            inst_data_iters,
        )
    _, layout, _ = infer_range_info(buffer_region, analyzer)
    data_iter_array = layout.combined_1d_layout.data_iter_array
    sorted_data_iters = sorted(
        ((idx, ext, data_iter_array[idx].stride) for (idx, ext) in inst_data_iters.items()),
        key=lambda tup: tup[-1],
    )
    actual_inst_size = 1
    actual_inst_data_iters = {}
    for idx, ext, stride in sorted_data_iters:
        if analyzer.can_prove(actual_inst_size * ext <= inst_size_limit):
            actual_inst_data_iters[idx] = ext
            actual_inst_size *= ext
        elif analyzer.can_prove(
            (actual_inst_size * ext) % inst_size_limit == 0
            and inst_size_limit % actual_inst_size == 0
        ):
            actual_inst_data_iters[idx] = inst_size_limit // actual_inst_size
            actual_inst_size = inst_size_limit
        else:
            break
    return actual_inst_size, actual_inst_data_iters


def f_gen_idx_anchor(buffer_region: BufferRegion, f_gen_axes):
    """Generate index function for a buffer region based on axes function.

    Parameters
    ----------
    buffer_region : BufferRegion
        The buffer region
    f_gen_axes : Callable
        The function to generate axes

    Returns
    -------
    Callable :
        A function to generate indices
    """

    def f(b_loops, f_loop, p_loop, dim2block_var=None):
        region_axes = f_gen_axes(b_loops, f_loop, p_loop, dim2block_var)
        return [buffer_region.region[i].min + region_axes[i] for i in range(len(region_axes))]

    return f


def f_gen_idx_mapped(buffer_region: BufferRegion, f_gen_axes, dim_map):
    """Generate index function for a buffer region based on axes function from another region.

    Parameters
    ----------
    buffer_region : BufferRegion
        The buffer region
    f_gen_axes : Callable
        The function to generate axes from another region
    dim_map : Dict[int, int]
        The dimension map from other region to this region

    Returns
    -------
    Callable :
        A function to generate indices
    """

    def f(b_loops, f_loop, p_loop, dim2block_var=None):
        region_axes = f_gen_axes(b_loops, f_loop, p_loop, dim2block_var)
        indices = [buffer_region.region[i].min for i in range(len(buffer_region.region))]
        for i, j in dim_map.items():
            indices[j] += region_axes[i]
        return indices

    return f


def check_partition_dim_match(
    buffer_region: BufferRegion,
    second_buffer_region: BufferRegion,
    dim_map: Dict[int, int],
    analyzer: Analyzer,
):
    """Check if partition dimensions match between two buffer regions.

    Parameters
    ----------
    buffer_region : BufferRegion
        The first buffer region
    second_buffer_region : BufferRegion
        The second buffer region
    dim_map : Dict[int, int]
        The dimension map from first to second buffer region
    analyzer : Analyzer
        The analyzer to use

    Returns
    -------
    bool :
        Whether partition dimensions match
    """
    if buffer_region.buffer.scope() == "global" or second_buffer_region.buffer.scope() == "global":
        return True
    src1_range_info, src1_layout, src1_seps = infer_range_info(buffer_region, analyzer)
    src2_range_info, src2_layout, src2_seps = infer_range_info(second_buffer_region, analyzer)

    def get_partition_logical_data_iters(range_info, layout, seps):
        # (dim_in_shape, extent, stride)
        partition_logical_data_iters = []
        logical_stride_map = {}
        for i in range(len(seps) - 1):
            logical_stride_in_dim = 1
            for j in reversed(range(seps[i], seps[i + 1])):
                logical_stride_map[j] = logical_stride_in_dim
                logical_stride_in_dim *= layout.combined_1d_layout.data_iter_array[j].extent
        for i in range(len(range_info)):
            if range_info[i].dim_type == T.TrainiumLayout.Partition:
                partition_logical_data_iters.append(
                    (
                        range_info[i].dim_in_shape,
                        range_info[i].extent,
                        logical_stride_map[range_info[i].dim_in_data_iter],
                    )
                )
        return partition_logical_data_iters

    src1_partition_logical_data_iters = get_partition_logical_data_iters(
        src1_range_info, src1_layout, src1_seps
    )
    src2_partition_logical_data_iters = get_partition_logical_data_iters(
        src2_range_info, src2_layout, src2_seps
    )
    src1_partition_logical_data_iters.sort(key=lambda x: (x[0], x[2]))
    src2_partition_logical_data_iters.sort(key=lambda x: (x[0], x[2]))
    if not all(x[0] in dim_map for x in src1_partition_logical_data_iters):
        return False
    src1_partition_logical_data_iters = [
        (dim_map[x[0]], x[1], x[2]) for x in src1_partition_logical_data_iters
    ]
    return src1_partition_logical_data_iters == src2_partition_logical_data_iters


largest_psum_per_bank = 512
max_psum_banks = 8

def check_workspace_buffer(buffer: Buffer, shape: Tuple[int], scope: str):
    """Check if a workspace buffer is valid.

    Parameters
    ----------
    buffer : Buffer
        The workspace buffer to check
    shape : Tuple[int]
        The required shape
    scope : str
        The required scope

    Raises
    ------
    AssertionError :
        If the buffer is invalid
    """
    assert buffer.scope() == scope, f"workspace buffer must be a {scope} buffer"
    assert buffer.layout is None, "workspace buffer must not have a layout"
    if scope == "trn.psum":
        # the number of psum banks used is inferred from the shape
        # only check p and f dims
        assert all(
            x >= y for x, y in zip(buffer.shape[1:], shape)
        ), f"workspace buffer must have enough size, {buffer.shape[1:]} cannot cover {shape}"
    else:
        assert all(
            x >= y for x, y in zip(buffer.shape, shape)
        ), f"workspace buffer must have enough size, {buffer.shape} cannot cover {shape}"


def init_analyzer(sctx: ScheduleContext):
    """Initialize an analyzer with the schedule context.

    Parameters
    ----------
    sctx : ScheduleContext
        The schedule context

    Returns
    -------
    Analyzer :
        The initialized analyzer
    """
    analyzer = Analyzer()
    for v, r in sctx.var_range_map.items():
        analyzer.bind(v, r)
    return analyzer


def target_trn(fn: Callable[[OpCall, ScheduleContext], Optional[PrimFunc]]):
    """Decorator that ensures a function is only executed for TRN targets.

    Parameters
    ----------
    fn : Callable[[OpCall, ScheduleContext], Optional[PrimFunc]]
        The function to decorate

    Returns
    -------
    Callable :
        The decorated function
    """

    @wraps(fn)
    def wrapper(*args, **kwargs):
        sctx = kwargs.get("sctx", None)
        if sctx is None:
            assert len(args) == 2 and isinstance(
                args[1], ScheduleContext
            ), "The target_cuda() needs to annotate a function with signature (op_call, sctx)"
            sctx = args[1]
        if not sctx.is_trn():
            return None
        return fn(*args, **kwargs)

    return wrapper


class DimensionMapper:
    """
    A class to manage dimension mappings between tensors.

    A dimension mapping (dim_map) has type Dict[int, int]. dim_map[i] = j means
    dimension i in the first tensor should be mapped to dimension j in the second tensor.
    """

    def __init__(self):
        self.mappings = {}  # Dictionary to store mappings between tensors

    def register_dim_map(self, first_tensor, second_tensor, dim_map):
        """
        Register a dimension mapping between two tensors.

        Args:
            first_tensor: The first tensor
            second_tensor: The second tensor
            dim_map: A dictionary mapping dimensions from first_tensor to second_tensor
        """
        # Initialize dictionaries if they don't exist
        if first_tensor not in self.mappings:
            self.mappings[first_tensor] = {}

        # Register the mapping
        self.mappings[first_tensor][second_tensor] = dim_map

        # Register the reverse mapping
        reverse_dim_map = {dim_map[i]: i for i in dim_map}

        if second_tensor not in self.mappings:
            self.mappings[second_tensor] = {}

        self.mappings[second_tensor][first_tensor] = reverse_dim_map

    def compose_mappings(self, map1, map2):
        """
        Compose two mappings: map1 followed by map2.

        Args:
            map1: The first mapping
            map2: The second mapping

        Returns:
            A composition of the two mappings, or None if the composition is empty
        """
        result = {}
        for i, j in map1.items():
            if j in map2:
                result[i] = map2[j]

        # If the result is empty, return None
        return result if result else None

    def get_dim_map(self, first_tensor, second_tensor):
        """
        Get the dimension mapping between two tensors.

        Args:
            first_tensor: The first tensor
            second_tensor: The second tensor

        Returns:
            A dictionary mapping dimensions from first_tensor to second_tensor,
            or {} if no mapping exists
        """
        # Check if there is a direct mapping
        if first_tensor in self.mappings and second_tensor in self.mappings[first_tensor]:
            return self.mappings[first_tensor][second_tensor]

        # No direct mapping, try to find a path using BFS
        visited = {first_tensor}
        queue = []

        # Add all direct neighbors of the first tensor to the queue
        if first_tensor in self.mappings:
            for neighbor, direct_mapping in self.mappings[first_tensor].items():
                visited.add(neighbor)
                queue.append((neighbor, direct_mapping))

        while queue:
            current_tensor, mapping_from_first = queue.pop(0)

            if current_tensor == second_tensor:
                # Found a path to the second tensor
                self.register_dim_map(first_tensor, second_tensor, mapping_from_first)
                return mapping_from_first

            if current_tensor not in self.mappings:
                continue

            for neighbor, direct_mapping in self.mappings[current_tensor].items():
                if neighbor not in visited:
                    visited.add(neighbor)

                    # Compose the mappings: first_tensor -> current_tensor -> neighbor
                    composed_mapping = self.compose_mappings(mapping_from_first, direct_mapping)

                    # Only add to the queue if the composed mapping is not None
                    if composed_mapping is not None:
                        queue.append((neighbor, composed_mapping))

        # No mapping found
        return {}


@dataclass
class LogicalIterDim:
    logical_stride: int
    extent: int
    bind_expr: PrimExpr

    @staticmethod
    def default():
        return LogicalIterDim(1, 1, T.int32(0))


LogicalIterList = Tuple[Tuple[Tuple[LogicalIterDim]]]


class VarReplacer(ExprMutator):
    def __init__(self, var_map: Dict[Var, PrimExpr]):
        super().__init__()
        self.var_map = var_map

    def visit_var_(self, op):
        if op in self.var_map:
            return self.var_map[op]
        return op

    @staticmethod
    def replace_vars(expr: PrimExpr, var_map: Dict[Var, PrimExpr]) -> PrimExpr:
        return VarReplacer(var_map).visit_expr(expr)


@dataclass
class InstructionRepr:
    buffer_region: BufferRegion
    size: int
    stride: int
    selected_data_iter_ids: List[int]

    def __init__(
        self,
        buffer_region: BufferRegion,
        inst_size: int,
        inst_stride: int,
        selected_data_iter_ids: List[int],
    ):
        self.buffer_region = buffer_region
        self.size = inst_size if inst_size is not None else 1
        self.stride = inst_stride if inst_stride is not None else 1
        self.selected_data_iter_ids = selected_data_iter_ids

    def bound_inst_size(self, max_inst_size: int, analyzer: Analyzer):
        if analyzer.can_prove(self.size <= max_inst_size):
            return
        assert analyzer.can_prove(
            self.size % max_inst_size == 0
        ), f"The instruction size {self.size} is not a multiple of the max instruction size {max_inst_size}"
        self.size = max_inst_size
        self.selected_data_iter_ids = None


class InstructionGenerator:
    def __init__(self, buffer_regions: Tuple[BufferRegion], analyzer: Analyzer):
        self.buffer_regions = buffer_regions
        self.analyzer = analyzer
        self.split_shape_views = {}
        self.split_layout_views = {}
        self.seps = {}
        self.bound_regions = {}
        self.bind_iters: Dict[BufferRegion, LogicalIterList] = None
        self.bind_maps: Dict[BufferRegion, Dict[Var, PrimExpr]] = {}
        for buffer_region in buffer_regions:
            bound_buffer_region = self._bound_buffer_region(buffer_region)
            layout, seps = self._get_sub_layout(bound_buffer_region)
            self.split_shape_views[buffer_region] = self._get_flattened_shape_view_from_layout_seps(
                layout, seps
            )
            self.split_layout_views[buffer_region] = layout
            self.seps[buffer_region] = seps
        self.dim_mapper = DimensionMapper()

    def _bound_buffer_region(self, buffer_region: BufferRegion):
        region = []
        changed = False
        for r in buffer_region.region:
            bound = self.analyzer.const_int_bound(r.extent)
            if not self.analyzer.can_prove_equal(bound.max_value, r.extent):
                changed = True
            region.append(Range.from_min_extent(r.min, bound.max_value))
        if changed:
            bound_region = BufferRegion(buffer_region.buffer, region)
            self.bound_regions[buffer_region] = bound_region
            return bound_region
        return buffer_region

    def _get_sub_layout(self, buffer_region: BufferRegion):
        layout = buffer_region.buffer.layout
        layout, seps = normalize_layout_with_shape(layout, buffer_region.buffer.shape)
        tile_layout = layout.combined_1d_layout if isinstance(layout, T.TrainiumLayout) else layout
        data_iters = tile_layout.data_iter_array
        tiled_range_infos_per_dim = []
        new_data_iters = []
        new_dim_types = []
        new_seps = [0]
        for i in range(len(seps) - 1):
            r = buffer_region.region[i]
            st = r.min
            ext = r.extent
            reversed_data_iters = []
            reversed_dim_types = []
            for j in reversed(range(seps[i], seps[i + 1])):
                dim_type = layout.dimension_types[j] if isinstance(layout, T.TrainiumLayout) else -1
                if self.analyzer.can_prove_equal(ext, 1):
                    break
                if dim_type == T.TrainiumLayout.Partition and (
                    not self.analyzer.can_prove(st % data_iters[j].extent == 0)
                    or not self.analyzer.can_prove(ext % data_iters[j].extent == 0)
                ):
                    assert False, "Invalid layout"
                if self.analyzer.can_prove(
                    ext % data_iters[j].extent == 0
                ) and self.analyzer.can_prove(st % data_iters[j].extent == 0):
                    st = st // data_iters[j].extent
                    ext = ext // data_iters[j].extent
                    tiled_range_infos_per_dim.append(
                        RangeInfo(0, data_iters[j].extent, j, i, dim_type)
                    )
                    reversed_data_iters.append(data_iters[j])
                    reversed_dim_types.append(dim_type)
                    continue
                if self.analyzer.can_prove(st + ext <= data_iters[j].extent):
                    tiled_range_infos_per_dim.append(RangeInfo(st, ext, j, i, dim_type))
                    reversed_data_iters.append(T.DataIterAttr(ext, data_iters[j].stride))
                    reversed_dim_types.append(dim_type)
                    break
                assert False, f"Cannot analyze physical tensor region for: {buffer_region}"
            new_data_iters += reversed(reversed_data_iters)
            new_dim_types += reversed(reversed_dim_types)
            new_seps.append(len(reversed_data_iters) + new_seps[-1])
        # FIXME: device_iter_array and from_scope/to_scope are not set
        new_tile_layout = T.TileLayout(new_data_iters)
        if isinstance(layout, T.TrainiumLayout):
            return T.TrainiumLayout(new_dim_types, new_tile_layout), new_seps
        else:
            return new_tile_layout, new_seps

    def _init_bind_iters(self):
        self.bind_iters = {}
        for buffer_region in self.buffer_regions:
            seps = self.seps[buffer_region]
            self.bind_iters[buffer_region] = [
                [[] for _ in range(seps[i], seps[i + 1])] for i in range(len(buffer_region.region))
            ]

    def _normalize_bind_iters(self):
        for buffer_region in self.buffer_regions:
            seps = self.seps[buffer_region]
            self.bind_iters[buffer_region] = [
                [
                    sorted(
                        self.bind_iters[buffer_region][i][j - seps[i]],
                        key=lambda x: (x.logical_stride, x.extent),
                    )
                    for j in range(seps[i], seps[i + 1])
                ]
                for i in range(len(buffer_region.region))
            ]

    def _get_flattened_shape_view_from_layout_seps(self, layout, seps):
        data_iters = get_layout_data_iters(layout)
        return [
            [data_iters[j].extent for j in range(seps[i], seps[i + 1])]
            for i in range(len(seps) - 1)
        ]

    def _link_buffer_regions(
        self, buffer_region: BufferRegion, to_link: BufferRegion, dim_map: Dict[int, int]
    ):
        split_shape_view_1 = self.split_shape_views[buffer_region]
        split_layout_view_1 = self.split_layout_views[buffer_region]
        split_shape_view_2 = self.split_shape_views[to_link]

        # adapt to the shape view of the to_link buffer region
        new_split_shape_view_1 = [
            split_shape_view_2[dim_map[i]] if i in dim_map else split_shape_view_1[i]
            for i in range(len(buffer_region.region))
        ]
        flattened_shape_view_1 = list(itertools.chain(*new_split_shape_view_1))
        layout, tiled_seps = normalize_layout_with_shape(
            split_layout_view_1, flattened_shape_view_1
        )
        actual_seps = [0]
        ptr = 0
        for i in range(len(buffer_region.region)):
            ptr += len(new_split_shape_view_1[i])
            actual_seps.append(tiled_seps[ptr])
        self.split_shape_views[buffer_region] = self._get_flattened_shape_view_from_layout_seps(
            layout, actual_seps
        )
        self.split_layout_views[buffer_region] = layout
        self.seps[buffer_region] = actual_seps

    def _get_reverse_dim_map(self, dim_map: Dict[int, int]) -> Dict[int, int]:
        return {dim_map[i]: i for i in dim_map}

    def link_buffer_regions(
        self, buffer_region: BufferRegion, to_link: BufferRegion, dim_map: Dict[int, int]
    ):
        self.dim_mapper.register_dim_map(buffer_region, to_link, dim_map)
        for r in self.buffer_regions:
            if r == to_link:
                continue
            dim_map = self.dim_mapper.get_dim_map(r, to_link)
            reverse_dim_map = self._get_reverse_dim_map(dim_map)
            self._link_buffer_regions(r, to_link, dim_map)
            self._link_buffer_regions(to_link, r, reverse_dim_map)
            seps_1 = self.seps[r]
            seps_2 = self.seps[to_link]
            for i, j in dim_map.items():
                assert (
                    seps_1[i + 1] - seps_1[i] == seps_2[j + 1] - seps_2[j]
                ), f"The number of data iters at dim {i} of {buffer_region.buffer.name} is not equal to the number of data iters at dim {j} of {to_link.buffer.name}"

    def bind_inst_iter(
        self,
        buffer_region: BufferRegion,
        bind: Var,
        inst_size: int,
        inst_stride: int,
        is_free_dim: bool,
        no_propagate: bool = False,
    ):
        logical_iter_list = self._get_inst_logical_iter_list(
            buffer_region, bind, inst_stride, inst_size, is_free_dim
        )
        self._add_bind_iter_list(buffer_region, logical_iter_list)
        if no_propagate:
            return
        self._propagate_bind_iter(buffer_region, logical_iter_list)

    def _propagate_bind_iter(self, buffer_region: BufferRegion, logical_iter_list: LogicalIterList):
        for to_propagate in self.buffer_regions:
            if to_propagate == buffer_region:
                continue
            dim_map = self.dim_mapper.get_dim_map(buffer_region, to_propagate)
            reverse_dim_map = self._get_reverse_dim_map(dim_map)
            seps = self.seps[to_propagate]
            propagated_logical_iter = [
                (
                    logical_iter_list[reverse_dim_map[i]]
                    if i in reverse_dim_map
                    else [[] for _ in range(seps[i], seps[i + 1])]
                )
                for i in range(len(to_propagate.region))
            ]
            self._add_bind_iter_list(to_propagate, propagated_logical_iter)

    def _add_bind_iter_list(self, buffer_region: BufferRegion, bind_iter_list: LogicalIterList):
        if self.bind_iters is None:
            self._init_bind_iters()
        seps = self.seps[buffer_region]
        for i in range(len(buffer_region.region)):
            for j in range(seps[i], seps[i + 1]):
                self.bind_iters[buffer_region][i][j - seps[i]].extend(
                    bind_iter_list[i][j - seps[i]]
                )

    def fill_in_block_dim(
        self, buffer_region: BufferRegion, bind: Var, dims: Optional[List[int]] = None
    ):
        # fixme: be cautious of the min of buffer region. This implementation is not correct.
        #        we need to first take a view of sub-layout (keep strides, but reduce the extent
        #        then we analyze the relationship between data iter of sub-layout
        dims = dims or list(range(len(buffer_region.buffer.shape)))
        layout = self.split_layout_views[buffer_region]
        data_iters = get_layout_data_iters(layout)
        self._normalize_bind_iters()
        bind_iters = self.bind_iters[buffer_region]
        seps = self.seps[buffer_region]
        logical_iter_list_block = [
            [[] for _ in range(seps[i], seps[i + 1])] for i in range(len(buffer_region.region))
        ]
        acc_block_ext = 1
        for i in reversed(dims):
            for j in reversed(range(seps[i], seps[i + 1])):
                data_iter = data_iters[j]
                is_partition = (
                    layout.dimension_types[j] == T.TrainiumLayout.Partition
                    if isinstance(layout, T.TrainiumLayout)
                    else False
                )
                logical_iter_dims = bind_iters[i][j - seps[i]]
                for d in range(-1, len(logical_iter_dims)):
                    next_logical_stride = (
                        logical_iter_dims[d + 1].logical_stride
                        if d + 1 < len(logical_iter_dims)
                        else data_iter.extent
                    )
                    cur = (
                        logical_iter_dims[d].logical_stride * logical_iter_dims[d].extent
                        if d >= 0
                        else 1
                    )
                    assert (
                        next_logical_stride % cur == 0
                    ), f"Fail to infer block dim for {buffer_region.buffer.name} at dim {i}"
                    gap = next_logical_stride // cur
                    if is_partition:
                        assert (
                            gap == 1
                        ), f"Fail to propagate partition dim. The propagated dim does not cover the whole partition on {buffer_region.buffer.name} at dim {i}"
                    elif gap > 1:
                        new_acc_block_ext = acc_block_ext * gap
                        logical_iter_list_block[i][j - seps[i]].append(
                            LogicalIterDim(cur, gap, bind % new_acc_block_ext // acc_block_ext)
                        )
                        acc_block_ext = new_acc_block_ext
        self._add_bind_iter_list(buffer_region, logical_iter_list_block)
        self._propagate_bind_iter(buffer_region, logical_iter_list_block)
        return acc_block_ext

    def _check_bind_iter_coverage(self, buffer_region: BufferRegion):
        self._normalize_bind_iters()
        seps = self.seps[buffer_region]
        data_iters = get_layout_data_iters(self.split_layout_views[buffer_region])
        bind_iters = self.bind_iters[buffer_region]
        for i in range(len(buffer_region.region)):
            for j in range(seps[i], seps[i + 1]):
                data_iter = data_iters[j]
                logical_iter_dims = bind_iters[i][j - seps[i]]
                for d in range(len(logical_iter_dims)):
                    next_logical_stride = (
                        logical_iter_dims[d + 1].logical_stride
                        if d + 1 < len(logical_iter_dims)
                        else data_iter.extent
                    )
                    assert (
                        next_logical_stride
                        % (logical_iter_dims[d].logical_stride * logical_iter_dims[d].extent)
                        == 0
                    ), f"Fail to infer block dim for {buffer_region.buffer.name} at dim {i}"
                    gap = next_logical_stride // (
                        logical_iter_dims[d].logical_stride * logical_iter_dims[d].extent
                    )
                    assert gap == 1, f"Call fill_in_block_dim() before calling generate_indices()"

    def set_bind_map(self, buffer_region: BufferRegion, bind_map: Dict[Var, PrimExpr]):
        self.bind_maps[buffer_region] = bind_map

    def generate_axes(self, buffer_region: BufferRegion) -> List[PrimExpr]:
        self._check_bind_iter_coverage(buffer_region)
        layout = self.split_layout_views[buffer_region]
        data_iters = get_layout_data_iters(layout)
        bind_iters = self.bind_iters[buffer_region]
        seps = self.seps[buffer_region]
        axes = []
        for i in range(len(bind_iters)):
            index = 0
            acc_logical_stride = 1
            for j in reversed(range(seps[i], seps[i + 1])):
                logical_iter_dims = bind_iters[i][j - seps[i]]
                for d in reversed(logical_iter_dims):
                    if d.extent == 1:
                        continue
                    index += (
                        d.logical_stride
                        * VarReplacer.replace_vars(d.bind_expr, self.bind_maps[buffer_region])
                        * acc_logical_stride
                    )
                acc_logical_stride *= data_iters[j].extent
            axes.append(index)
        return axes

    def generate_indices(self, buffer_region: BufferRegion) -> List[PrimExpr]:
        axes = self.generate_axes(buffer_region)
        return [axes[i] + r.min for i, r in enumerate(buffer_region.region)]

    def _get_inst_logical_iter_list(
        self,
        buffer_region: BufferRegion,
        bind: Var,
        stride: int,
        size: int,
        is_free_dim: bool = True,
    ) -> LogicalIterList:
        layout = self.split_layout_views[buffer_region]
        assert isinstance(
            layout, T.TrainiumLayout
        ), " Cannot propagate instruction information from HBM tensor"
        data_iters = get_layout_data_iters(layout)
        seps = self.seps[buffer_region]
        ret = [[[] for _ in range(seps[i], seps[i + 1])] for i in range(len(buffer_region.region))]
        for i in range(len(buffer_region.region)):
            for j in range(seps[i], seps[i + 1]):
                if (layout.dimension_types[j] == T.TrainiumLayout.Free) ^ is_free_dim:
                    continue
                data_iter = data_iters[j]
                if (
                    data_iter.stride * data_iter.extent <= stride
                    or data_iter.stride >= size * stride
                ):
                    continue
                if (
                    data_iter.stride * data_iter.extent < size * stride
                    and stride <= data_iter.stride
                ):
                    assert (size * stride) % (
                        data_iter.stride * data_iter.extent
                    ) == 0 and data_iter.stride % stride == 0
                    ret[i][j - seps[i]].append(
                        LogicalIterDim(
                            1,
                            data_iter.extent,
                            bind
                            % (data_iter.stride * data_iter.extent // stride)
                            // (data_iter.stride // stride),
                        )
                    )
                elif (
                    data_iter.stride * data_iter.extent < size * stride
                    and stride > data_iter.stride
                ):
                    assert (size * stride) % (
                        data_iter.stride * data_iter.extent
                    ) == 0 and stride % data_iter.stride == 0
                    ret[i][j - seps[i]].append(
                        LogicalIterDim(
                            stride // data_iter.stride,
                            data_iter.stride * data_iter.extent // stride,
                            bind % (data_iter.stride * data_iter.extent // stride),
                        )
                    )
                elif (
                    data_iter.stride * data_iter.extent >= size * stride
                    and stride <= data_iter.stride
                ):
                    assert (data_iter.stride * data_iter.extent) % (
                        size * stride
                    ) == 0 and data_iter.stride % stride == 0
                    ret[i][j - seps[i]].append(
                        LogicalIterDim(
                            1,
                            size * stride // data_iter.stride,
                            bind // (data_iter.stride // stride),
                        )
                    )
        return ret

    def make_guard(self, buffer_region: BufferRegion):
        if buffer_region not in self.bound_regions:
            return True
        bound_region = self.bound_regions[buffer_region]
        relaxed_dims = [
            i
            for i, (r1, r2) in enumerate(zip(bound_region.region, buffer_region.region))
            if not self.analyzer.can_prove(r1.extent == r2.extent)
        ]
        axes = self.generate_axes(buffer_region)
        guard = reduce(
            T.And,
            [axes[i] < r.extent for i, r in enumerate(buffer_region.region) if i in relaxed_dims],
            True,
        )
        return guard

    def _find_max_linear_inst(self, indexed_data_iters, min_stride: Optional[int] = None):
        min_stride = min_stride or 1
        indexed_data_iters = sorted(indexed_data_iters, key=lambda x: x[1].stride)
        inst_size = 1
        inst_stride = None
        idx_list = []
        for idx, data_iter in indexed_data_iters:
            if data_iter.extent == 1 or data_iter.stride * data_iter.extent < min_stride:
                continue
            assert (
                data_iter.stride % min_stride == 0 or min_stride % data_iter.stride == 0
            ), f"Invalid instruction stride {min_stride}"
            if inst_stride is not None and inst_stride * inst_size != data_iter.stride:
                # the stride of the found data iter is not compatible with previous data iters
                break
            elif inst_stride is None:
                inst_stride = max(min_stride, data_iter.stride)
            if min_stride % data_iter.stride == 0:
                inst_size = data_iter.extent * data_iter.stride // inst_stride
            else:
                inst_size *= data_iter.extent
            idx_list.append(idx)
        return inst_size, inst_stride, idx_list

    def find_max_inst_size_from_one_region(
        self,
        buffer_region: BufferRegion,
        allowed_f_dim: Optional[Tuple[int]] = None,
        min_stride: Optional[int] = None,
    ):
        allowed_f_dim = allowed_f_dim or tuple(range(len(buffer_region.region)))
        layout = self.split_layout_views[buffer_region]
        data_iters = get_layout_data_iters(layout)
        seps = self.seps[buffer_region]
        allowed_data_iter_idx = itertools.chain.from_iterable(
            range(seps[dim], seps[dim + 1]) for dim in allowed_f_dim
        )
        filtered_data_iters = [
            (i, data_iters[i])
            for i in allowed_data_iter_idx
            if layout.dimension_types[i] == T.TrainiumLayout.Free
        ]
        inst_size, inst_stride, idx_list = self._find_max_linear_inst(
            filtered_data_iters, min_stride
        )
        return InstructionRepr(buffer_region, inst_size, inst_stride, idx_list)

    def fit_inst_tile_to_region(
        self,
        inst_repr: InstructionRepr,
        to_region: BufferRegion,
        allowed_to_f_dim: Optional[Tuple[int]] = None,
    ):
        allowed_to_f_dim = allowed_to_f_dim or tuple(range(len(to_region.region)))
        from_region = inst_repr.buffer_region
        from_layout = self.split_layout_views[from_region]
        from_data_iters = get_layout_data_iters(from_layout)
        to_layout = self.split_layout_views[to_region]
        to_data_iters = get_layout_data_iters(to_layout)
        from_seps = self.seps[from_region]
        to_seps = self.seps[to_region]
        dim_map = self.dim_mapper.get_dim_map(from_region, to_region)
        dim_map = {i: j for i, j in dim_map.items() if j in allowed_to_f_dim}
        data_iter_map = {
            from_seps[i] + idx: to_seps[j] + idx
            for i, j in dim_map.items()
            for idx in range(from_seps[i + 1] - from_seps[i])
        }
        indexed_selected_data_iters = [
            (i, from_data_iters[i]) for i in inst_repr.selected_data_iter_ids
        ]
        indexed_selected_data_iters = sorted(indexed_selected_data_iters, key=lambda x: x[1].stride)
        inst_size = 1
        inst_stride_from = None
        inst_stride_to = None
        idx_list = []
        for i, data_iter in indexed_selected_data_iters:
            if i not in data_iter_map:
                if inst_stride_from is None:
                    continue
                break
            mapped_data_iter = to_data_iters[data_iter_map[i]]
            if inst_stride_from is None:
                inst_stride_from = data_iter.stride
                if not isinstance(to_layout, T.TrainiumLayout) and mapped_data_iter.stride != 1:
                    # dma copy must be contiguous on hbm
                    break
                inst_stride_to = mapped_data_iter.stride
            elif inst_stride_to * inst_size != mapped_data_iter.stride:
                break
            inst_size *= data_iter.extent
            idx_list.append(i)
        return InstructionRepr(from_region, inst_size, inst_stride_from, idx_list)

    def check_partition_dim_match(
        self, buffer_region_1: BufferRegion, buffer_region_2: BufferRegion
    ):
        dim_map = self.dim_mapper.get_dim_map(buffer_region_1, buffer_region_2)
        layout_1 = self.split_layout_views[buffer_region_1]
        layout_2 = self.split_layout_views[buffer_region_2]
        if not isinstance(layout_1, T.TrainiumLayout) or not isinstance(layout_2, T.TrainiumLayout):
            return True
        data_iters_1 = get_layout_data_iters(layout_1)
        data_iters_2 = get_layout_data_iters(layout_2)
        seps_1 = self.seps[buffer_region_1]
        seps_2 = self.seps[buffer_region_2]
        for i, j in dim_map.items():
            for k in range(seps_1[i + 1] - seps_1[i]):
                if (
                    layout_1.dimension_types[seps_1[i] + k]
                    != layout_2.dimension_types[seps_2[j] + k]
                ):
                    return False
                if layout_1.dimension_types[seps_1[i] + k] == T.TrainiumLayout.Free:
                    continue
                if data_iters_1[seps_1[i] + k].stride != data_iters_2[seps_2[j] + k].stride:
                    return False
                if data_iters_1[seps_1[i] + k].extent != data_iters_2[seps_2[j] + k].extent:
                    return False
        return True

    def find_max_inst_size_transpose(
        self, buffer_region_1: BufferRegion, buffer_region_2: BufferRegion
    ):
        dim_map = self.dim_mapper.get_dim_map(buffer_region_1, buffer_region_2)
        layout_1 = self.split_layout_views[buffer_region_1]

        layout_2 = self.split_layout_views[buffer_region_2]
        data_iters_1 = get_layout_data_iters(layout_1)
        data_iters_2 = get_layout_data_iters(layout_2)
        seps_1 = self.seps[buffer_region_1]
        seps_2 = self.seps[buffer_region_2]
        indexed_data_iters_1 = []
        indexed_data_iters_2 = []
        for i, j in dim_map.items():
            for k in range(seps_1[i + 1] - seps_1[i]):
                if (
                    layout_1.dimension_types[seps_1[i] + k]
                    == layout_2.dimension_types[seps_2[j] + k]
                ):
                    if layout_1.dimension_types[seps_1[i] + k] == T.TrainiumLayout.Free:
                        continue
                    raise ValueError(f"Transpose only part of P dimension is not supported")
                if layout_1.dimension_types[seps_1[i] + k] == T.TrainiumLayout.Partition:
                    indexed_data_iters_2.append((seps_2[j] + k, data_iters_2[seps_2[j] + k]))
                else:
                    indexed_data_iters_1.append((seps_1[i] + k, data_iters_1[seps_1[i] + k]))
        inst_repr_1 = InstructionRepr(
            buffer_region_1, *self._find_max_linear_inst(indexed_data_iters_1)
        )
        inst_repr_2 = InstructionRepr(
            buffer_region_2, *self._find_max_linear_inst(indexed_data_iters_2)
        )
        assert (
            inst_repr_1.size == layout_2.partition_size
        ), f"The instruction size of {buffer_region_1.buffer.name} does not match the partition size of {buffer_region_2.buffer.name}"
        assert (
            inst_repr_2.size == layout_1.partition_size
        ), f"The instruction size of {buffer_region_2.buffer.name} does not match the partition size of {buffer_region_1.buffer.name}"
        return inst_repr_1, inst_repr_2
