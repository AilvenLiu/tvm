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

from typing import Optional, Union, List, Tuple, Dict
import operator
import functools
from enum import Enum
from tvm.arith.analyzer import Analyzer
from tvm.script import tir as T
from tvm.tir import BufferRegion, PrimFunc, FloatImm
from tvm.tir.stmt import OpCall
from tvm.tirp.op_schedule import ScheduleContext

from .common import (
    generate_axes_in_region,
    find_max_inst_size_unary,
    infer_range_info,
    get_ewise_dim_map,
    bound_inst_with_limit,
    init_analyzer,
    f_gen_idx_anchor,
    f_gen_idx_mapped,
    check_partition_dim_match,
    _refine_inst_tile,
    bound_buffer_region,
    make_guard,
    nki_dim,
)
from ..common import MapOpType

binary_map_ops = {
    MapOpType.ADD: "add",
    MapOpType.SUB: "sub",
    MapOpType.MUL: "mul",
    MapOpType.MAX: "max",
    MapOpType.MIN: "min",
}


class InstType(Enum):
    TENSOR_TENSOR = 0
    TENSOR_SCALAR = 1


def get_inst_tile_with_const(
    dst: BufferRegion,
    src1: BufferRegion,
    analyzer: Analyzer,
    allowed_f_dim_dst: Optional[Tuple[int]] = None,
    allowed_f_dim_src1: Optional[Tuple[int]] = None,
):
    """Find instruction tile size for tensor-scalar operations."""
    inst_size, inst_stride, inst_data_iters = find_max_inst_size_unary(
        dst, src1, analyzer, allowed_f_dim_dst, allowed_f_dim_src1
    )
    return inst_size, inst_stride, inst_data_iters, InstType.TENSOR_SCALAR


def get_inst_tile_with_broadcast(
    dst: BufferRegion,
    src1: BufferRegion,
    src2: BufferRegion,
    broadcast_dims: List[int],
    analyzer: Analyzer,
    allow_tensortensor: bool = True,
    allowed_f_dim_dst: Optional[Tuple[int]] = None,
    allowed_f_dim_src1: Optional[Tuple[int]] = None,
    allowed_f_dim_src2: Optional[Tuple[int]] = None,
    refine_inst: Optional[Tuple[int, int]] = None,
):
    """Find instruction tile size for operations with broadcasting."""
    if refine_inst is not None:
        inst_size, inst_stride, inst_data_iters = refine_inst
    else:
        inst_size, inst_stride, inst_data_iters = find_max_inst_size_unary(
            dst, src1, analyzer, allowed_f_dim_dst, allowed_f_dim_src1
        )

    # Extract range info and categorize iterators by broadcast dimensions
    dst_range_info, dst_layout, dst_seps = infer_range_info(dst, analyzer)
    f_data_iters_in_broadcast_dim = []
    f_data_iters_in_non_broadcast_dim = []

    for i in range(len(dst_seps) - 1):
        for j in range(dst_seps[i], dst_seps[i + 1]):
            if j not in inst_data_iters:
                continue
            if i in broadcast_dims:
                f_data_iters_in_broadcast_dim.append((j, inst_data_iters[j]))
            elif allowed_f_dim_dst is None or i in allowed_f_dim_src2:
                f_data_iters_in_non_broadcast_dim.append((j, inst_data_iters[j]))

    def try_f_data_iters(f_data_iters):
        """Find maximum possible instruction size from given data iterators."""
        f_data_iters.sort(key=lambda x: dst_layout.combined_1d_layout.data_iter_array[x[0]].stride)
        new_inst_size, new_data_iters, new_inst_stride = 1, {}, 1

        for dim_in_data_iter, extent in f_data_iters:
            if extent == 1:
                continue

            iter = dst_layout.combined_1d_layout.data_iter_array[dim_in_data_iter]
            if new_inst_size != 1 and new_inst_size * new_inst_stride != iter.stride:
                break

            if new_inst_size == 1:
                new_inst_stride = max(inst_stride, iter.stride)

            new_inst_size *= extent
            new_data_iters[dim_in_data_iter] = extent

        return new_inst_size, new_inst_stride, new_data_iters

    # Find best tile sizes for tensor-tensor and tensor-scalar
    tensortensor_inst_size, tensortensor_inst_stride, _ = try_f_data_iters(
        f_data_iters_in_non_broadcast_dim
    )

    # Map dimensions between buffers
    dst_to_src1_dim_map = get_ewise_dim_map(dst, src1, analyzer)
    src1_src2_offset = len(src1.region) - len(src2.region)
    dst_to_src2_dim_map = {
        d: s - src1_src2_offset for d, s in dst_to_src1_dim_map.items() if s not in broadcast_dims
    }

    # Refine instruction tiles
    tensortensor_inst_size, tensortensor_inst_stride, _, tensortensor_inst_data_iters = (
        _refine_inst_tile(
            dst,
            src2,
            tensortensor_inst_size,
            tensortensor_inst_stride,
            analyzer,
            allowed_f_dim_dst,
            allowed_f_dim_src2,
            dst_to_src2_dim_map,
        )
    )

    tensorscalar_inst_size, tensorscalar_inst_stride, tensorscalar_inst_data_iters = (
        try_f_data_iters(f_data_iters_in_broadcast_dim)
    )

    # Return tensor-scalar if tensor-tensor not allowed
    if not allow_tensortensor:
        return (
            tensorscalar_inst_size,
            tensorscalar_inst_stride,
            tensorscalar_inst_data_iters,
            InstType.TENSOR_SCALAR,
        )

    # Select best instruction type based on size and stride
    if tensortensor_inst_size == 1:
        option_chosen = "tensorscalar"
    elif tensorscalar_inst_size == 1:
        option_chosen = "tensortensor"
    elif tensortensor_inst_stride == 1:
        option_chosen = "tensortensor"
    elif tensorscalar_inst_stride == 1:
        option_chosen = "tensorscalar"
    elif tensortensor_inst_size > tensorscalar_inst_size:
        option_chosen = "tensortensor"
    else:
        option_chosen = "tensorscalar"

    if option_chosen == "tensortensor":
        return (
            tensortensor_inst_size,
            tensortensor_inst_stride,
            tensortensor_inst_data_iters,
            InstType.TENSOR_TENSOR,
        )

    return (
        tensorscalar_inst_size,
        tensorscalar_inst_stride,
        tensorscalar_inst_data_iters,
        InstType.TENSOR_SCALAR,
    )


def try_find_inst_binary(
    _dst: BufferRegion,
    _src1: Union[BufferRegion, FloatImm],
    _src2: Union[BufferRegion, FloatImm],
    analyzer: Analyzer,
    allowed_f_dim_dst: Optional[Tuple[int]] = None,
    allowed_f_dim_src1: Optional[Tuple[int]] = None,
    allowed_f_dim_src2: Optional[Tuple[int]] = None,
    allow_tensortensor: bool = True,
    allow_reverse: bool = True,
):
    """Find instruction parameters for binary operations."""
    result = try_find_inst_nary(
        _dst,
        [_src1, _src2],
        analyzer,
        allowed_f_dim_dst,
        (allowed_f_dim_src1, allowed_f_dim_src2),
        allow_tensortensor,
    )

    inst_size, inst_stride, inst_data_iters, inst_types, reverse, broadcast_dims = result

    if inst_size is None:
        return None, None, None, None, None, None

    if inst_types[0] == InstType.TENSOR_TENSOR and not allow_tensortensor:
        return None, None, None, None, None, None

    if reverse[0] and not allow_reverse:
        return None, None, None, None, None, None

    return inst_size, inst_stride, inst_data_iters, inst_types[0], reverse[0], broadcast_dims[0]


def try_find_inst_nary(
    _dst: BufferRegion,
    _srcs: List[Union[BufferRegion, FloatImm]],
    analyzer: Analyzer,
    allowed_f_dim_dst: Optional[Tuple[int]] = None,
    allowed_f_dim_srcs: Optional[Tuple[Tuple[int]]] = None,
    allow_first_op_tensortensor: bool = True,
):
    """Find instruction parameters for n-ary operations."""
    # Validate inputs and handle source swapping if needed
    assert not (
        isinstance(_srcs[0], FloatImm) and isinstance(_srcs[1], FloatImm)
    ), "Nary operation does not support taking all FloatImm sources"
    assert 2 <= len(_srcs) <= 3, "Only 2-3 sources are supported for nary operation"

    if isinstance(_srcs[0], FloatImm):
        _srcs[0], _srcs[1] = _srcs[1], _srcs[0]
        reverse = [True] + [False] * (len(_srcs) - 2)
    else:
        reverse = [False] * (len(_srcs) - 1)

    # Extract buffers and validate properties
    dst, srcs = _dst.buffer, [
        _src.buffer if isinstance(_src, BufferRegion) else None for _src in _srcs
    ]
    dst_region = _dst.region

    valid_buffers = all(
        [
            dst.layout and all(src.layout for src in srcs if src is not None),
            isinstance(dst.layout, T.TrainiumLayout),
            all(isinstance(src.layout, T.TrainiumLayout) for src in srcs if src is not None),
            dst.scope() == "trn.sbuf",
            all(src.scope() in ["trn.sbuf", "trn.psum"] for src in srcs if src is not None),
        ]
    )

    if not valid_buffers:
        raise ValueError(f"Invalid buffer region: dst: {_dst}, srcs: {_srcs}")

    # Check non-unit extents
    dst_non_unit_extent = [r.extent for r in dst_region if r.extent != 1]

    # Handle broadcasting between first two sources
    if not isinstance(_srcs[1], FloatImm):
        src0_extent = [r.extent for r in _srcs[0].region]
        src1_extent = [r.extent for r in _srcs[1].region]
        shared_dim_num = min(len(src0_extent), len(src1_extent))

        # Check for various broadcasting patterns and swap sources if needed
        dims_equal = all(
            analyzer.can_prove(e0 == e1)
            for e0, e1 in zip(src0_extent[-shared_dim_num:], src1_extent[-shared_dim_num:])
        )

        if dims_equal:
            if len(src0_extent) < len(src1_extent) and not all(
                analyzer.can_prove(e1 == 1) for e1 in src1_extent[:-shared_dim_num]
            ):
                _srcs[0], _srcs[1] = _srcs[1], _srcs[0]
                reverse[0] = True
        elif all(
            analyzer.can_prove(e0 == e1 or e0 == 1)
            for e0, e1 in zip(src0_extent[-shared_dim_num:], src1_extent[-shared_dim_num:])
        ):
            _srcs[0], _srcs[1] = _srcs[1], _srcs[0]
            reverse[0] = True
            assert shared_dim_num == len(src0_extent) or all(
                analyzer.can_prove(e0 == 1) for e0 in src0_extent[:-shared_dim_num]
            ), f"Shape mismatch: src0: {_srcs[0]}, src1: {_srcs[1]}"
        elif all(
            analyzer.can_prove(e0 == e1 or e1 == 1)
            for e0, e1 in zip(src0_extent[-shared_dim_num:], src1_extent[-shared_dim_num:])
        ):
            assert shared_dim_num == len(src1_extent) or all(
                analyzer.can_prove(e1 == 1) for e1 in src1_extent[:-shared_dim_num]
            ), f"Shape mismatch: src0: {_srcs[0]}, src1: {_srcs[1]}"
        else:
            raise ValueError(f"Shape mismatch: src0: {_srcs[0]}, src1: {_srcs[1]}")

    # Verify src0 and dst have matching non-unit dimensions
    src0_non_unit_extent = [r.extent for r in _srcs[0].region if r.extent != 1]
    valid_shapes = all(
        [
            len(src0_non_unit_extent) == len(dst_non_unit_extent),
            all(
                analyzer.can_prove_equal(s, d)
                for s, d in zip(src0_non_unit_extent, dst_non_unit_extent)
            ),
        ]
    )

    assert valid_shapes, "the larger between src0 and src1 must have the same shape as dst"

    # Identify broadcast dimensions for each source after src0
    src0_extent = [r.extent for r in _srcs[0].region]
    broadcast_dims = []

    for src in _srcs[1:]:
        if isinstance(src, FloatImm):
            broadcast_dims.append(None)
            continue

        src_extent = [r.extent for r in src.region]

        # Check extra dimensions
        assert len(src_extent) <= len(src0_extent) or all(
            analyzer.can_prove(src_extent[i] == 1)
            for i in range(len(src_extent) - len(src0_extent))
        )

        # Find broadcast dimensions
        local_broadcast_dims = []
        for i in range(1, min(len(src_extent), len(src0_extent)) + 1):
            if analyzer.can_prove(src_extent[-i] != 1) and analyzer.can_prove(
                src_extent[-i] != src0_extent[-i]
            ):
                raise ValueError(f"Shape mismatch: src0: {_srcs[0]}, src: {src}")
            elif analyzer.can_prove(src_extent[-i] != src0_extent[-i]):
                local_broadcast_dims.append(len(src0_extent) - i)

        # Add leading dimensions
        local_broadcast_dims += list(range(0, len(src0_extent) - len(src_extent)))

        # Create dimension mapping and verify partition
        src0_to_src_dim_map = {
            i: i + len(src_extent) - len(src0_extent)
            for i in range(len(src0_extent))
            if i not in local_broadcast_dims
        }

        assert check_partition_dim_match(
            _srcs[0], src, src0_to_src_dim_map, analyzer
        ), f"partition dimension mismatch: src0: {_srcs[0]}, src: {src}"

        broadcast_dims.append(local_broadcast_dims)

    # Find instruction parameters for each source
    inst_size, inst_stride, inst_data_iters, inst_types = None, None, None, []
    allowed_f_dim_srcs = [None] * len(_srcs) if allowed_f_dim_srcs is None else allowed_f_dim_srcs

    for i, src in enumerate(_srcs[1:]):
        if isinstance(src, FloatImm):
            if inst_size is None:
                inst_size, inst_stride, inst_data_iters, inst_type = get_inst_tile_with_const(
                    _dst, _srcs[0], analyzer, allowed_f_dim_dst, allowed_f_dim_srcs[0]
                )
            inst_types.append(InstType.TENSOR_SCALAR)
            continue

        refine_inst = (inst_size, inst_stride, inst_data_iters) if inst_size is not None else None
        allow_tt = allow_first_op_tensortensor or i != 0
        src_dim = allowed_f_dim_srcs[i + 1] if i + 1 < len(allowed_f_dim_srcs) else None

        inst_size, inst_stride, inst_data_iters, inst_type = get_inst_tile_with_broadcast(
            _dst,
            _srcs[0],
            src,
            broadcast_dims[i],
            analyzer,
            allow_tt,
            allowed_f_dim_dst,
            allowed_f_dim_srcs[0],
            src_dim,
            refine_inst,
        )
        inst_types.append(inst_type)

    return inst_size, inst_stride, inst_data_iters, inst_types, reverse, broadcast_dims


def binary_trn(
    op: OpCall,
    binary_op: MapOpType,
    sctx: ScheduleContext,
) -> Optional[PrimFunc]:
    """Generate a binary operation schedule for Trainium."""
    if not (sctx.is_trn() and sctx.exec_scope.name == "kernel"):
        return None

    assert binary_op in binary_map_ops, f"Unsupported binary operation {binary_op}"

    # Initialize analyzer and buffer regions
    analyzer = init_analyzer(sctx)
    _dst, _src1, _src2 = op.args
    bound_dst = bound_buffer_region(_dst, analyzer)
    bound_src1 = bound_buffer_region(_src1, analyzer)
    bound_src2 = bound_buffer_region(_src2, analyzer)

    # Find instruction parameters
    inst_params = try_find_inst_binary(bound_dst, bound_src1, bound_src2, analyzer)
    inst_size, inst_stride, inst_data_iters, inst_type, reverse, broadcast_dims = inst_params
    f_gen_axes = generate_axes_in_region(bound_dst, inst_stride, inst_data_iters, analyzer)

    # Handle operand swapping if needed
    if reverse:
        _src1, _src2, bound_src1, bound_src2 = _src2, _src1, bound_src2, bound_src1

    # Generate index functions
    f_gen_dst_idx = f_gen_idx_anchor(_dst, f_gen_axes)
    dst_to_src1_dim_map = get_ewise_dim_map(_dst, _src1, analyzer)
    f_gen_src1_idx = f_gen_idx_mapped(_src1, f_gen_axes, dst_to_src1_dim_map)

    if isinstance(_src2, BufferRegion):
        src1_src2_offset = len(_src1.region) - len(_src2.region)
        dst_to_src2_dim_map = {
            d: s - src1_src2_offset
            for d, s in dst_to_src1_dim_map.items()
            if s not in broadcast_dims
        }
        f_gen_src2_idx = f_gen_idx_mapped(_src2, f_gen_axes, dst_to_src2_dim_map)

    # Extract buffers and constants
    CONST = _src2 if isinstance(_src2, FloatImm) else None
    dst, src1 = _dst.buffer, _src1.buffer
    src2 = None if CONST is not None else _src2.buffer

    # Setup execution parameters
    p_size = dst.layout.partition_size
    b_extent = (
        functools.reduce(operator.mul, [r.extent for r in bound_dst.region], 1)
        // p_size
        // inst_size
    )
    opcode = binary_map_ops[binary_op]

    # Select appropriate NKI function based on instruction type
    _func = T.nki_tensortensor if inst_type == InstType.TENSOR_TENSOR else T.nki_tensorscalar
    func = lambda *args: (
        _func(*args, reverse) if inst_type == InstType.TENSOR_SCALAR else _func(*args)
    )

    # Handle instruction size limits
    inst_size_limit = op.schedule_config.get("max_inst_size", None)
    actual_inst_size, additional_b_size = bound_inst_with_limit(
        inst_size, inst_size_limit, analyzer
    )
    f_guard = make_guard(_dst, analyzer)

    # Define the implementation function
    @T.prim_func(tirp=True)
    def impl():
        for b_loop, additional_b_loop in T.grid(b_extent, additional_b_size):
            with T.attr(0, "tensorized_nki_instruction", 1):
                for p_loop in T.serial(0, p_size, annotations={nki_dim: "P"}):
                    for f_loop in T.serial(0, actual_inst_size, annotations={nki_dim: "F"}):
                        f_loop_wo_limit = T.meta_var(f_loop + additional_b_loop * actual_inst_size)
                        if f_guard(f_gen_axes(((b_loop, b_extent),), f_loop_wo_limit, p_loop)):
                            dst_indices = T.meta_var(
                                f_gen_dst_idx(((b_loop, b_extent),), f_loop_wo_limit, p_loop)
                            )
                            src1_indices = T.meta_var(
                                f_gen_src1_idx(((b_loop, b_extent),), f_loop_wo_limit, p_loop)
                            )
                            if CONST is None:
                                src2_indices = T.meta_var(
                                    f_gen_src2_idx(((b_loop, b_extent),), f_loop_wo_limit, p_loop)
                                )
                                T.evaluate(
                                    func(
                                        dst[*dst_indices],
                                        src1[*src1_indices],
                                        src2[*src2_indices],
                                        opcode,
                                    )
                                )
                            else:
                                T.evaluate(
                                    func(dst[*dst_indices], src1[*src1_indices], CONST, opcode)
                                )

    return impl
