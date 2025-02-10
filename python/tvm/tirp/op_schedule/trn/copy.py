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
from .common import get_layout_data_iters, normalize_buffer_region_with_layout, generate_axes_in_region


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
        f_gen_axes = generate_axes_in_region(
            src_buffer_region, inst_stride, inst_data_iters, analyzer
        )

    else:
        inst_size = inst_size_2
        inst_stride = inst_stride_2
        inst_data_iters = inst_data_iters_2
        p_size = get_p_size(dst_buffer_region, analyzer)
        f_gen_axes = generate_axes_in_region(
            dst_buffer_region, inst_stride, inst_data_iters, analyzer
        )

    def f_gen_src_idx(b_loop, b_extent, f_loop, p_loop):
        axes = f_gen_axes(((b_loop, b_extent),), f_loop, p_loop)
        return [src_buffer_region.region[i].min + axes[i] for i in range(len(axes))]
    def f_gen_dst_idx(b_loop, b_extent, f_loop, p_loop):
        axes = f_gen_axes(((b_loop, b_extent),), f_loop, p_loop)
        return [dst_buffer_region.region[i].min + axes[i] for i in range(len(axes))]

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
