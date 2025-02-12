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

"""Implementation of binary operator schedules."""

from typing import Optional, Union, List
import operator
import functools

from tvm.arith.analyzer import Analyzer
from tvm.script import tir as T
from tvm.tir import BufferRegion, PrimFunc, FloatImm
from tvm.tirp.op_schedule import ScheduleContext

from .common import (
    generate_axes_in_region,
    find_max_inst_size_unary,
    infer_range_info,
    get_ewise_dim_map,
)
from ..common import MapOpType

binary_map_ops = {MapOpType.ADD: "add", MapOpType.SUB: "sub", MapOpType.MUL: "mul"}


# check if src1's partition dimension matches src2's partition dimension
def check_broadcast_match_partition(src1: BufferRegion, src2: BufferRegion, analyzer: Analyzer):
    src1_range_info, src1_layout, src1_seps = infer_range_info(src1, analyzer)
    src2_range_info, src2_layout, src2_seps = infer_range_info(src2, analyzer)

    def get_partition_logical_data_iters(range_info, layout, seps):
        # (dim_in_shape, extent, stride)
        partition_logical_data_iters = []
        logical_stride_map = {}
        for i in range(len(src1_seps) - 1):
            logical_stride_in_dim = 1
            for j in reversed(range(src1_seps[i], src1_seps[i + 1])):
                logical_stride_map[j] = logical_stride_in_dim
                logical_stride_in_dim *= src1_layout.combined_1d_layout.data_iter_array[j].extent
        for i in range(len(range_info)):
            if range_info[i].dim_type == T.TrainiumLayout.Partition:
                partition_logical_data_iters.append(
                    (range_info[i].dim_in_shape, range_info[i].extent, logical_stride_map[i])
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


# return inst tile size, inst stride, inst data iters, and is_tensor_tensor
def get_inst_tile_with_const(dst: BufferRegion, src1: BufferRegion, analyzer: Analyzer):
    inst_size, inst_stride, inst_data_iters = find_max_inst_size_unary(dst, src1, analyzer)
    return inst_size, inst_stride, inst_data_iters, False


def get_inst_tile_with_no_broadcast(
    dst: BufferRegion, src1: BufferRegion, src2: BufferRegion, analyzer: Analyzer
):
    inst_size_1, inst_stride_1, inst_data_iters_1 = find_max_inst_size_unary(dst, src1, analyzer)
    inst_size_2, inst_stride_2, inst_data_iters_2 = find_max_inst_size_unary(dst, src2, analyzer)
    if inst_size_1 < inst_size_2:
        (
            inst_size_1,
            inst_stride_1,
            inst_data_iters_1,
            inst_size_2,
            inst_stride_2,
            inst_data_iters_2,
        ) = (
            inst_size_2,
            inst_stride_2,
            inst_data_iters_2,
            inst_size_1,
            inst_stride_1,
            inst_data_iters_1,
        )
    assert (
        inst_size_1 % inst_size_2 == 0 and inst_stride_1 == inst_stride_2
    ), "src1 and src2 not compatible for tensortensor"
    return inst_size_2, inst_stride_2, inst_data_iters_2, True


def get_inst_tile_with_broadcast(
    dst: BufferRegion,
    src1: BufferRegion,
    src2: BufferRegion,
    broadcast_dims: List[int],
    analyzer: Analyzer,
):
    inst_size, inst_stride, inst_data_iters = find_max_inst_size_unary(dst, src1, analyzer)
    dst_range_info, dst_layout, dst_seps = infer_range_info(dst, analyzer)
    f_data_iters_in_broadcast_dim = []
    for i in range(len(dst_seps) - 1):
        if i not in broadcast_dims:
            continue
        for j in range(dst_seps[i], dst_seps[i + 1]):
            # F dimension in broadcast dim
            if j in inst_data_iters:
                f_data_iters_in_broadcast_dim.append((j, inst_data_iters[j]))
    # broadcast dim is B dimension
    # fallback to tensortensor
    if len(f_data_iters_in_broadcast_dim) == 0:
        return get_inst_tile_with_no_broadcast(dst, src1, src2, analyzer)

    # todo: consider using tensortensor if F dim is small when using tensorscalar
    # broadcast dim contains F dimension
    # adjust F size to fit in broadcast dim
    f_data_iters_in_broadcast_dim.sort(
        key=lambda x: dst_layout.combined_1d_layout.data_iter_array[x[0]].stride
    )
    new_inst_size = 1
    new_data_iters = {}
    while len(f_data_iters_in_broadcast_dim) > 0:
        dim_in_data_iter, extent = f_data_iters_in_broadcast_dim[0]
        f_data_iters_in_broadcast_dim = f_data_iters_in_broadcast_dim[1:]
        iter = dst_layout.combined_1d_layout.data_iter_array[dim_in_data_iter]
        if iter.extent == 1:
            continue
        if new_inst_size * inst_stride != iter.stride:
            break
        new_inst_size *= extent
        new_data_iters[dim_in_data_iter] = extent
    return new_inst_size, inst_stride, new_data_iters, False


def binary_trn(
    _dst: BufferRegion,
    _src1: Union[BufferRegion, FloatImm],
    _src2: Union[BufferRegion, FloatImm],
    binary_op: MapOpType,
    sctx: ScheduleContext,
) -> Optional[PrimFunc]:
    if not (sctx.is_trn() and sctx.exec_scope.name == "kernel"):
        return None
    assert binary_op in binary_map_ops, f"Unsupported binary operation {binary_op}"

    CONST = None
    reorder = False
    # Type checks
    if isinstance(_src1, FloatImm) and isinstance(_src2, FloatImm):
        return None
    if isinstance(_src1, FloatImm):
        reorder = True
        _src1, _src2 = _src2, _src1
    if isinstance(_src2, FloatImm):
        CONST = _src2

        # Basic region checks
    dst, src1, src2 = _dst.buffer, _src1.buffer, None if CONST is not None else _src2.buffer
    dst_region, src1_region, src2_region = (
        _dst.region,
        _src1.region,
        None if CONST is not None else _src2.region,
    )

    dst_st = [r.min for r in dst_region]
    src1_st = [r.min for r in src1_region]
    src2_st = [r.min for r in src2_region] if src2_region else None
    dst_extent = [r.extent for r in dst_region]
    src1_extent = [r.extent for r in src1_region]
    src2_extent = [r.extent for r in src2_region] if src2_region else None

    if not all(
        [
            dst.layout and src1.layout and (src2.layout if src2 else True),
            isinstance(dst.layout, T.TrainiumLayout),
            isinstance(src1.layout, T.TrainiumLayout),
            isinstance(src2.layout, T.TrainiumLayout) if src2 else True,
            dst.scope() == "trn.sbuf",
            src1.scope() == "trn.sbuf" or src1.scope() == "trn.psum",
            (src2.scope() == "trn.sbuf" or src2.scope() == "trn.psum") if src2 else True,
        ]
    ):
        return None

    # Switch broadcasting
    analyzer = Analyzer()
    for v, r in sctx.var_range_map.items():
        analyzer.bind(v, r)
    NUM_ELEMENTS = functools.reduce(operator.mul, dst_extent, 1)

    if CONST is None:
        src1_num = functools.reduce(operator.mul, src1_extent, 1)
        if NUM_ELEMENTS > src1_num:
            reorder = True
            (
                _src1,
                _src2,
                src1,
                src2,
                src1_region,
                src2_region,
                src1_st,
                src2_st,
                src1_extent,
                src2_extent,
            ) = (
                _src2,
                _src1,
                src2,
                src1,
                src2_region,
                src1_region,
                src2_st,
                src1_st,
                src2_extent,
                src1_extent,
            )
    # Check dst and src1 have the same shape
    dst_extent_ = [e for e in dst_extent if e != 1]
    src1_extent_ = [e for e in src1_extent if e != 1]
    if not all(
        [
            len(src1_extent_) == len(dst_extent_),
            all(analyzer.can_prove_equal(s, d) for s, d in zip(src1_extent_, dst_extent_)),
        ]
    ):
        return None

    # Check src2 is broadcastable to src1
    broadcast_dims = []
    if CONST is None:
        for i in range(1, len(src2_extent) + 1):
            if src2_extent[-i] != 1 and src2_extent[-i] != src1_extent[-i]:
                return None
            elif src2_extent[-i] != src1_extent[-i]:
                broadcast_dims.append(len(src1_extent) - i)
        broadcast_dims += list(range(0, len(src1_extent) - len(src2_extent)))
    if len(broadcast_dims) > 0:
        assert check_broadcast_match_partition(_src1, _src2, analyzer)
    # find inst tile compatible for dst and src
    if CONST is not None:
        inst_size, inst_stride, inst_data_iters, is_tensor_tensor = get_inst_tile_with_const(
            _dst, _src1, analyzer
        )
    elif len(broadcast_dims) == 0:
        inst_size, inst_stride, inst_data_iters, is_tensor_tensor = get_inst_tile_with_no_broadcast(
            _dst, _src1, _src2, analyzer
        )
    else:
        inst_size, inst_stride, inst_data_iters, is_tensor_tensor = get_inst_tile_with_broadcast(
            _dst, _src1, _src2, broadcast_dims, analyzer
        )

    assert not (
        reorder and is_tensor_tensor
    ), "reorder not supported for TensorTensor. Consider manually switch the order and apply minus operator somewhere."

    p_size = dst.layout.partition_size
    f_gen_axes = generate_axes_in_region(_dst, inst_stride, inst_data_iters, analyzer)

    def f_gen_dst_idx(b_loop, b_extent, f_loop, p_loop):
        axes = f_gen_axes(((b_loop, b_extent),), f_loop, p_loop)
        return [_dst.region[i].min + axes[i] for i in range(len(axes))]

    def f_gen_src1_idx(b_loop, b_extent, f_loop, p_loop):
        indices = [_src1.region[i].min for i in range(len(_src1.region))]
        axes = f_gen_axes(((b_loop, b_extent),), f_loop, p_loop)
        dim_map = get_ewise_dim_map(_dst, _src1, analyzer)
        for i, j in dim_map.items():
            indices[j] += axes[i]
        return indices

    def f_gen_src2_idx(b_loop, b_extent, f_loop, p_loop):
        indices = [_src2.region[i].min for i in range(len(_src2.region))]
        axes = f_gen_axes(((b_loop, b_extent),), f_loop, p_loop)
        dim_map = get_ewise_dim_map(_dst, _src1, analyzer)
        offset = len(src1_extent) - len(src2_extent)
        for i, j in dim_map.items():
            if j not in broadcast_dims:
                indices[j - offset] += axes[i]
        return indices

    b_extent = NUM_ELEMENTS // p_size // inst_size
    opcode = binary_map_ops[binary_op]
    # fmt: off
    _func = T.nki_tensortensor if is_tensor_tensor else T.nki_tensorscalar
    func = lambda *args: _func(*args, reorder) if not is_tensor_tensor else _func(*args)
    @T.prim_func(tirp=True)
    def impl():
        for b_loop in T.serial(0, b_extent):
            with T.attr(0, "tensorized_nki_instruction", 1):
                for p_loop, f_loop in T.grid(p_size, inst_size):
                        dst_indices = T.meta_var(f_gen_dst_idx(b_loop, b_extent, f_loop, p_loop))
                        src1_indices = T.meta_var(f_gen_src1_idx(b_loop, b_extent, f_loop, p_loop))
                        if CONST is None:
                            src2_indices = T.meta_var(f_gen_src2_idx(b_loop, b_extent, f_loop, p_loop))
                            T.evaluate(func(dst[*dst_indices], src1[*src1_indices], src2[*src2_indices], opcode))
                        else:
                            T.evaluate(func(dst[*dst_indices], src1[*src1_indices], CONST, opcode))

    # fmt: on
    return impl
