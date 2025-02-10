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


from typing import Optional, Set, List, Dict
import operator

from tvm.arith.analyzer import Analyzer
from tvm.script import tir as T
from tvm.tir import BufferRegion, PrimFunc, PrimExpr, Var
from tvm.ir import Range
from tvm.tirp.op_schedule import ScheduleContext, register_schedule

from functools import reduce
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


def normalize_buffer_region_with_layout(buffer_region: BufferRegion, analyzer: Analyzer):
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
                tiled_range_infos_per_dim.append((0, data_iters[j].extent, j, i, dim_type))
                continue
            if analyzer.can_prove(st + ext <= data_iters[j].extent):
                tiled_range_infos_per_dim.append((st, ext, j, i, dim_type))
                break
            assert False, "Invalid layout"
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
    range_info, tiled_layout, seps = normalize_buffer_region_with_layout(buffer_region, analyzer)
    dim_in_region = {range_info[i][2]: range_info[i][1] for i in range(len(range_info))}
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
                    assert analyzer.can_prove_equal(data_iters[j].extent % inst_data_iters[j], 0)
                    index_in_this_dim += (
                        extract_block_loop(data_iters[j].extent // inst_data_iters[j], i)
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
