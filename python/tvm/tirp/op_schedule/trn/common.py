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

from collections import namedtuple
from typing import Tuple, Optional, Dict, Callable
from functools import wraps

from tvm.arith.analyzer import Analyzer
from tvm.script import tir as T
from tvm.tir import BufferRegion, Buffer
from tvm._ffi import get_global_func
from tvm.tir import PrimFunc
from tvm.tir.stmt import OpCall
from tvm.tirp.op_schedule import ScheduleContext

f_normalize_trn_layout_with_shape = get_global_func("tir.NormalizeTrainiumLayoutWithShape")
f_normalize_tile_layout_with_shape = get_global_func("tir.NormalizeTileLayoutWithShape")


def normalize_layout_with_shape(layout, shape):
    if isinstance(layout, T.TrainiumLayout):
        ret = f_normalize_trn_layout_with_shape(layout, shape)
        return ret[0], ret[1]
    elif isinstance(layout, T.TileLayout):
        ret = f_normalize_tile_layout_with_shape(layout, shape)
        return ret[0], ret[1]
    else:
        raise ValueError("Invalid layout")


def get_layout_data_iters(layout):
    if isinstance(layout, T.TrainiumLayout):
        return layout.combined_1d_layout.data_iter_array
    elif isinstance(layout, T.TileLayout):
        return layout.data_iter_array
    else:
        raise ValueError("Invalid layout")


RangeInfo = namedtuple(
    "RangeInfo", ["start", "extent", "dim_in_data_iter", "dim_in_shape", "dim_type"]
)


def infer_range_info(buffer_region: BufferRegion, analyzer: Analyzer):
    layout = buffer_region.buffer.layout
    layout, seps = normalize_layout_with_shape(layout, buffer_region.buffer.shape)
    # (start, extent, dim_in_data_iter, dim_in_shape, dim_type)
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
            assert False, f"Cannot analyze the physical tensor region for: {buffer_region}"
    # put partition axis at the front
    # then put free axis with lower stride at the front
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


# src -> dst
def get_reduction_dim_map(
    src_buffer_region: BufferRegion, dst_buffer_region: BufferRegion, axes: Tuple[int]
):
    dst_region = dst_buffer_region.region
    dst_extent = [r.extent for r in dst_region]
    dst_non_unit_extent_ = [(i, e) for i, e in enumerate(dst_extent) if e != 1]
    src_region = src_buffer_region.region
    src_extent = [r.extent for r in src_region]
    src_non_unit_extent_ = [(i, e) for i, e in enumerate(src_extent) if e != 1]
    src_non_reduction_extents = [(i, e) for i, e in src_non_unit_extent_ if i not in axes]
    assert len(src_non_reduction_extents) == len(
        dst_non_unit_extent_
    ), "Source and destination must have the same number of non-reduction extents"
    for i in range(len(src_non_reduction_extents)):
        assert (
            src_non_reduction_extents[i][1] == dst_non_unit_extent_[i][1]
        ), "Source and destination must have the same extent for non-reduction axes"
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
                    print(
                        second_data_iters[second_data_iter].stride,
                        data_iters[dim_in_data_iter].stride,
                        second_extent,
                        data_iters[dim_in_data_iter].extent,
                    )
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

        if analyzer.can_prove(second_extent >= st + ext):
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


# use the first buffer region's P dim to find the corresponding F dim on the second buffer region
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


def find_max_inst_size_unary(
    buffer_region: BufferRegion,
    second_buffer_region: BufferRegion,
    analyzer: Analyzer,
    allowed_f_dim_1: Optional[Tuple[int]] = None,
    allowed_f_dim_2: Optional[Tuple[int]] = None,
):
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
    inst_size, second_inst_stride, second_inst_data_iters = _find_max_inst_size_transpose(
        buffer_region, second_buffer_region, analyzer
    )
    assert inst_size == buffer_region.buffer.layout.partition_size
    inst_size, first_inst_stride, first_inst_data_iters = _find_max_inst_size_transpose(
        second_buffer_region, buffer_region, analyzer
    )
    assert inst_size == second_buffer_region.buffer.layout.partition_size
    return first_inst_stride, first_inst_data_iters, second_inst_stride, second_inst_data_iters


def find_max_inst_size_matmul(
    rhs_buffer_region: BufferRegion,
    output_buffer_region: BufferRegion,
    analyzer: Analyzer,
    allowed_f_dim_rhs: Optional[Tuple[int]] = None,
    allowed_f_dim_output: Optional[Tuple[int]] = None,
    f_dim_map: Optional[Dict[int, int]] = None,
):
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


def get_hardware_inst_size_limit(is_dma: bool) -> int:
    return 1e9 if is_dma else 512


def bound_inst_with_limit(inst_size, inst_size_limit, analyzer):
    if not analyzer.can_prove(inst_size <= inst_size_limit):
        # FIXME: this constraint can be relaxed if we support mask
        assert analyzer.can_prove(inst_size % inst_size_limit == 0)
        actual_inst_size = inst_size_limit
        additional_b_size = T.ceildiv(inst_size, inst_size_limit)
    else:
        actual_inst_size = inst_size
        additional_b_size = 1
    return actual_inst_size, additional_b_size


def find_max_inst_size_from_one_region(
    buffer_region: BufferRegion,
    analyzer: Analyzer,
    allowed_f_dim: Optional[Tuple[int]] = None,
):
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


def init_analyzer(sctx: ScheduleContext):
    analyzer = Analyzer()
    for v, r in sctx.var_range_map.items():
        analyzer.bind(v, r)
    return analyzer


def f_gen_idx_anchor(buffer_region: BufferRegion, f_gen_axes):
    def f(b_loops, f_loop, p_loop, dim2block_var=None):
        region_axes = f_gen_axes(b_loops, f_loop, p_loop, dim2block_var)
        return [buffer_region.region[i].min + region_axes[i] for i in range(len(region_axes))]

    return f


def f_gen_idx_mapped(buffer_region: BufferRegion, f_gen_axes, dim_map):
    def f(b_loops, f_loop, p_loop, dim2block_var=None):
        region_axes = f_gen_axes(b_loops, f_loop, p_loop, dim2block_var)
        indices = [buffer_region.region[i].min for i in range(len(buffer_region.region))]
        for i, j in dim_map.items():
            indices[j] += region_axes[i]
        return indices

    return f


def check_partition_dim_match(
    buffer_region: BufferRegion, second_buffer_region: BufferRegion, analyzer: Analyzer
):
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
    src1_partition_logical_data_iters = [x[1:] for x in src1_partition_logical_data_iters]
    src2_partition_logical_data_iters = [x[1:] for x in src2_partition_logical_data_iters]
    return src1_partition_logical_data_iters == src2_partition_logical_data_iters


def get_largest_psum_per_bank():
    return 512


def get_max_psum_banks():
    return 8


def check_workspace_buffer(buffer: Buffer, shape: Tuple[int], scope: str):
    assert buffer.scope() == scope, f"workspace buffer must be a {scope} buffer"
    assert buffer.layout is None, "workspace buffer must not have a layout"
    if scope == "trn.psum":
        # the number of psum banks used is inferred from the shape
        # only check p and f dims
        assert tuple(buffer.shape[1:]) == tuple(
            shape
        ), f"workspace buffer must have the correct shape, {buffer.shape[1:]} != {shape[1:]}"
    else:
        assert tuple(buffer.shape) == tuple(
            shape
        ), f"workspace buffer must have the correct shape, {buffer.shape} != {shape}"


def target_trn(fn: Callable[[OpCall, ScheduleContext], Optional[PrimFunc]]):
    """Decorator that ensures the function is only executed for CUDA targets."""

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
