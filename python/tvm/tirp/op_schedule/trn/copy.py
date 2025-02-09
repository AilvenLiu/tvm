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

"""Implementation of copy operator schedules."""

from typing import Optional
import operator

from tvm.arith.analyzer import Analyzer
from tvm.script import tir as T
from tvm.tir import BufferRegion, PrimFunc
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


# infer an instruction size from buffer_region's access pattern that is compatible on second_buffer_region
def find_max_inst_size(
    buffer_region: BufferRegion, second_buffer_region: BufferRegion, analyzer: Analyzer
):
    tiled_range_infos_per_dim, tiled_layout, seps = normalize_buffer_region_with_layout(
        buffer_region, analyzer
    )
    data_iters = get_layout_data_iters(tiled_layout)
    _, second_tiled_layout, second_seps = normalize_buffer_region_with_layout(
        second_buffer_region, analyzer
    )
    second_is_sbuf = isinstance(second_buffer_region.buffer.layout, T.TrainiumLayout)
    second_data_iters = get_layout_data_iters(second_tiled_layout)
    inst_size = 1
    first_inst_stride = 1
    second_inst_stride = 1
    inst_data_iters = {}
    while True:
        if len(tiled_range_infos_per_dim) == 0:
            break
        range_info = tiled_range_infos_per_dim[0]
        tiled_range_infos_per_dim = tiled_range_infos_per_dim[1:]
        dim_in_shape = range_info[3]
        dim_in_data_iter = range_info[2]
        dim_type = range_info[4]
        if (
            dim_type == T.TrainiumLayout.Free
            and inst_size != 1
            and not analyzer.can_prove(
                first_inst_stride * inst_size == data_iters[dim_in_data_iter].stride
            )
        ):
            # the stride of the found data iter is not compatible with previous data iters
            break
        st = range_info[0]
        ext = range_info[1]
        stride_in_logical_dim = 1
        for i in range(dim_in_data_iter + 1, seps[dim_in_shape + 1]):
            stride_in_logical_dim *= data_iters[i].extent
        second_stride_in_logical_dim = 1
        leftover = None
        second_data_iter = None
        # find the corresponding data iter in the second buffer region
        for i in reversed(range(second_seps[dim_in_shape], second_seps[dim_in_shape + 1])):
            second_stride_in_logical_dim *= second_data_iters[i].extent
            if second_stride_in_logical_dim > stride_in_logical_dim:
                assert analyzer.can_prove(
                    second_stride_in_logical_dim % stride_in_logical_dim == 0
                ), "Invalid layout"
                leftover = second_stride_in_logical_dim // stride_in_logical_dim
                second_data_iter = i
                break
            elif second_stride_in_logical_dim == stride_in_logical_dim:
                assert i - 1 >= second_seps[dim_in_shape]
                leftover = second_data_iters[i - 1].extent
                second_data_iter = i - 1
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
            second_inst_stride * inst_size == second_data_iters[second_data_iter].stride
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
        elif analyzer.can_prove(st == 0) and analyzer.can_prove(ext % leftover == 0):
            inst_size *= leftover
            inst_data_iters[dim_in_data_iter] = leftover
            break
        else:
            assert False, "Invalid layout"
    return inst_size, first_inst_stride, inst_data_iters


def generate_indices(
    buffer_region: BufferRegion,
    second_buffer_region: BufferRegion,
    inst_size,
    inst_stride,
    inst_data_iters,
    analyzer,
):
    range_info, tiled_layout, seps = normalize_buffer_region_with_layout(buffer_region, analyzer)
    dim_in_region = [range_info[i][2] for i in range(len(range_info))]
    data_iters = get_layout_data_iters(tiled_layout)

    def f(b_loop, b_extent, f_loop, p_loop):
        def extract_block_loop(extent):
            nonlocal b_loop, b_extent
            ret = b_loop // (b_extent // extent)
            b_loop = b_loop % (b_extent // extent)
            b_extent = b_extent // extent
            return ret

        region_indices = []
        second_region_indices = []
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
                        extract_block_loop(data_iters[j].extent // inst_data_iters[j])
                        * inst_data_iters[j]
                    )
                    index_in_this_dim += (
                        f_loop // (data_iters[j].stride // inst_stride)
                    ) % inst_data_iters[j]
                elif j in dim_in_region:
                    index_in_this_dim += extract_block_loop(data_iters[j].extent)
            region_indices.append(index_in_this_dim + buffer_region.region[i].min)
            second_region_indices.append(index_in_this_dim + second_buffer_region.region[i].min)
        return region_indices, second_region_indices

    return f


def get_p_size(buffer_region: BufferRegion, analyzer: Analyzer):
    tiled_range_infos_per_dim, tiled_layout, seps = normalize_buffer_region_with_layout(
        buffer_region, analyzer
    )
    data_iters = get_layout_data_iters(tiled_layout)
    p_size = 1
    for range_info in tiled_range_infos_per_dim:
        dim_in_data_iter = range_info[2]
        if tiled_layout.dimension_types[dim_in_data_iter] == T.TrainiumLayout.Partition:
            p_size *= data_iters[dim_in_data_iter].extent
    return p_size


@register_schedule("copy")
def copy_trn(
    dst_buffer_region: BufferRegion,
    src_buffer_region: BufferRegion,
    sctx: ScheduleContext,
) -> Optional[PrimFunc]:
    """Schedule copy operation between global and shared memory on CUDA."""
    # Basic validation checks
    if not (sctx.is_trn() and sctx.exec_scope.name == "kernel"):
        return None

    src, dst = src_buffer_region.buffer, dst_buffer_region.buffer
    if not all(
        [
            src.layout and dst.layout,
            src.dtype == dst.dtype,
            (src.scope() == "global" and dst.scope() == "trn.sbuf")
            or (src.scope() == "trn.sbuf" and dst.scope() == "global")
            or (src.scope() == "trn.sbuf" and dst.scope() == "trn.sbuf"),
            (src.scope() == "global" and isinstance(src.layout, T.TileLayout))
            or (src.scope() == "trn.sbuf" and isinstance(src.layout, T.TrainiumLayout)),
            (dst.scope() == "global" and isinstance(dst.layout, T.TileLayout))
            or (dst.scope() == "trn.sbuf" and isinstance(dst.layout, T.TrainiumLayout)),
        ]
    ):
        return None

    # Extract regions and validate dimensions
    analyzer = Analyzer()
    for v, r in sctx.var_range_map.items():
        analyzer.bind(v, r)
    src_region, dst_region = src_buffer_region.region, dst_buffer_region.region
    src_extent = [r.extent for r in src_region]
    dst_extent = [r.extent for r in dst_region]

    # Validate non-unit dimensions match
    src_extent_ = [e for e in src_extent if e != 1]
    dst_extent_ = [e for e in dst_extent if e != 1]
    if not (
        len(src_extent_) == len(dst_extent_)
        and all(analyzer.can_prove_equal(s, d) for s, d in zip(src_extent_, dst_extent_))
    ):
        return None

    inst_size_1 = inst_size_2 = -1
    if isinstance(src.layout, T.TrainiumLayout):
        inst_size_1, inst_stride_1, inst_data_iters_1 = find_max_inst_size(
            src_buffer_region, dst_buffer_region, analyzer
        )
    if isinstance(dst.layout, T.TrainiumLayout):
        inst_size_2, inst_stride_2, inst_data_iters_2 = find_max_inst_size(
            dst_buffer_region, src_buffer_region, analyzer
        )

    if inst_size_1 > inst_size_2:
        inst_size = inst_size_1
        inst_stride = inst_stride_1
        inst_data_iters = inst_data_iters_1
        p_size = get_p_size(src_buffer_region, analyzer)
        f_gen_idx = generate_indices(
            src_buffer_region, dst_buffer_region, inst_size, inst_stride, inst_data_iters, analyzer
        )
        f_gen_src_idx = lambda b_loop, b_extent, f_loop, p_loop: f_gen_idx(
            b_loop, b_extent, f_loop, p_loop
        )[0]
        f_gen_dst_idx = lambda b_loop, b_extent, f_loop, p_loop: f_gen_idx(
            b_loop, b_extent, f_loop, p_loop
        )[1]

    else:
        inst_size = inst_size_2
        inst_stride = inst_stride_2
        inst_data_iters = inst_data_iters_2
        p_size = get_p_size(dst_buffer_region, analyzer)
        f_gen_idx = generate_indices(
            dst_buffer_region, src_buffer_region, inst_size, inst_stride, inst_data_iters, analyzer
        )
        f_gen_src_idx = lambda b_loop, b_extent, f_loop, p_loop: f_gen_idx(
            b_loop, b_extent, f_loop, p_loop
        )[1]
        f_gen_dst_idx = lambda b_loop, b_extent, f_loop, p_loop: f_gen_idx(
            b_loop, b_extent, f_loop, p_loop
        )[0]

    b_extent = reduce(operator.mul, src_extent, 1) // p_size // inst_size
    # fmt: off
    @T.prim_func(tirp=True)
    def impl():
        for b_loop in T.serial(0, b_extent):
            with T.attr(0, "tensorized_nki_instruction", 1):
                for p_loop in T.serial(0, p_size):
                    for f_loop in T.serial(0, inst_size):
                        src_indices = T.meta_var(f_gen_src_idx(b_loop, b_extent, f_loop, p_loop))
                        dst_indices = T.meta_var(f_gen_dst_idx(b_loop, b_extent, f_loop, p_loop))
                        dst[*dst_indices] = src[*src_indices]
    # fmt: on

    return impl
