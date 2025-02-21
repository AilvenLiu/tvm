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
from typing import Tuple

from tvm.arith.analyzer import Analyzer
from tvm.script import tir as T
from tvm.tir import BufferRegion
from tvm._ffi import get_global_func

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
        tiled_range_infos_per_dim, key=lambda x: data_iters[x[2]].stride + int(dim_type * 1e9)
    )
    return tiled_range_infos_per_dim, layout, seps


def generate_axes_in_region(
    buffer_region: BufferRegion,
    inst_stride,
    inst_data_iters,
    analyzer,
    dim2block_var=None,  # shape dim -> block var
):
    range_info, tiled_layout, seps = infer_range_info(buffer_region, analyzer)
    dim_in_region = {
        range_info[i].dim_in_data_iter: range_info[i].extent for i in range(len(range_info))
    }
    data_iters = get_layout_data_iters(tiled_layout)

    def f(b_loops, f_loop, p_loop):
        b_loops = list(b_loops)

        def extract_block_loop(extent, dim_in_shape):
            nonlocal b_loops
            idx = 0
            if dim2block_var is not None:
                assert dim_in_shape in dim2block_var
                idx = dim2block_var[dim_in_shape]
            b_loop, b_extent = b_loops[idx]
            ret = b_loop // (b_extent // extent)
            b_loop = b_loop % (b_extent // extent)
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


# infer an instruction size from buffer_region's access pattern that is compatible on second_buffer_region
def find_max_inst_size_unary(
    buffer_region: BufferRegion, second_buffer_region: BufferRegion, analyzer: Analyzer
):
    tiled_range_infos_per_dim, tiled_layout, seps = infer_range_info(buffer_region, analyzer)
    data_iters = get_layout_data_iters(tiled_layout)
    _, second_tiled_layout, second_seps = infer_range_info(second_buffer_region, analyzer)
    second_is_sbuf = isinstance(second_buffer_region.buffer.layout, T.TrainiumLayout)
    second_data_iters = get_layout_data_iters(second_tiled_layout)
    inst_size = 1
    first_inst_stride = 1
    second_inst_stride = 1
    inst_data_iters = {}
    buffer_dim_map = get_ewise_dim_map(buffer_region, second_buffer_region, analyzer)
    while len(tiled_range_infos_per_dim) > 0:
        range_info = tiled_range_infos_per_dim[0]
        tiled_range_infos_per_dim = tiled_range_infos_per_dim[1:]
        st, ext, dim_in_data_iter, dim_in_shape, dim_type = range_info
        if (
            dim_type == T.TrainiumLayout.Free
            and inst_size != 1
            and not analyzer.can_prove(
                first_inst_stride * inst_size == data_iters[dim_in_data_iter].stride
            )
        ):
            # the stride of the found data iter is not compatible with previous data iters
            break
        stride_in_logical_dim = 1
        for i in range(dim_in_data_iter + 1, seps[dim_in_shape + 1]):
            stride_in_logical_dim *= data_iters[i].extent
        second_stride_in_logical_dim = 1
        leftover = None
        second_data_iter = None
        second_buffer_dim = buffer_dim_map[dim_in_shape]
        # find the corresponding data iter in the second buffer region
        for i in reversed(
            range(second_seps[second_buffer_dim], second_seps[second_buffer_dim + 1])
        ):
            second_stride_in_logical_dim *= second_data_iters[i].extent
            if second_stride_in_logical_dim > stride_in_logical_dim:
                assert analyzer.can_prove(
                    second_stride_in_logical_dim % stride_in_logical_dim == 0
                ), "Invalid layout"
                leftover = second_stride_in_logical_dim // stride_in_logical_dim
                second_data_iter = i
                second_new_stride = (
                    second_data_iters[i].stride * second_data_iters[i].extent // leftover
                )
                break
            elif second_stride_in_logical_dim == stride_in_logical_dim:
                assert i - 1 >= second_seps[second_buffer_dim]
                leftover = second_data_iters[i - 1].extent
                second_data_iter = i - 1
                second_new_stride = second_data_iters[i - 1].stride
                break
        assert leftover is not None
        if dim_type == T.TrainiumLayout.Partition:
            if (
                second_is_sbuf
                and second_buffer_region.buffer.layout.dimension_types[second_data_iter]
                == T.TrainiumLayout.Free
            ):
                assert (
                    False
                ), "partition dimension in the first buffer must be partition dimension in the second buffer"
            continue
        if inst_size != 1 and not analyzer.can_prove(
            second_inst_stride * inst_size == second_new_stride
        ):
            # the stride of the found data iter is not compatible with previous data iters
            break
        if inst_size == 1:
            if not second_is_sbuf and not analyzer.can_prove(
                second_data_iters[second_data_iter].stride == 1
            ):
                # stride of hbm tensor access must be 1
                break
            first_inst_stride = data_iters[dim_in_data_iter].stride
            second_inst_stride = second_data_iters[second_data_iter].stride

        if analyzer.can_prove(leftover >= st + ext):
            inst_size *= ext
            inst_data_iters[dim_in_data_iter] = ext
            if not analyzer.can_prove(st == 0) or not analyzer.can_prove(
                ext == data_iters[dim_in_data_iter].extent
            ):
                break
        elif analyzer.can_prove(ext % leftover == 0):
            inst_size *= leftover
            inst_data_iters[dim_in_data_iter] = leftover
            break
        else:
            assert False, "Invalid layout"
    return inst_size, first_inst_stride, inst_data_iters


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
    allowed_f_dim: Tuple[int],
    analyzer: Analyzer,
):
    tiled_range_infos_per_dim, layout, seps = infer_range_info(buffer_region, analyzer)
    data_iters = get_layout_data_iters(layout)
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
            break
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
