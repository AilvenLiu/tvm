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
    get_ewise_dim_map,
    init_analyzer,
    InstructionGenerator,
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


def try_find_inst_nary(
    _dst: BufferRegion,
    _srcs: List[Union[BufferRegion, FloatImm]],
    analyzer: Analyzer,
    inst_gen: InstructionGenerator,
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
            dst.layout.is_trainium(),
            all(src.layout.is_trainium() for src in srcs if src is not None),
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
            analyzer.can_prove(e0 == e1) or analyzer.can_prove(e0 == 1)
            for e0, e1 in zip(src0_extent[-shared_dim_num:], src1_extent[-shared_dim_num:])
        ):
            _srcs[0], _srcs[1] = _srcs[1], _srcs[0]
            reverse[0] = True
            assert shared_dim_num == len(src0_extent) or all(
                analyzer.can_prove(e0 == 1) for e0 in src0_extent[:-shared_dim_num]
            ), f"Shape mismatch: src0: {_srcs[0]}, src1: {_srcs[1]}"
        elif all(
            analyzer.can_prove(e0 == e1) or analyzer.can_prove(e1 == 1)
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
    dst_to_src0_dim_map = get_ewise_dim_map(_dst, _srcs[0], analyzer)
    inst_gen.link_buffer_regions(_dst, _srcs[0], dst_to_src0_dim_map)

    for src in _srcs[1:]:
        if isinstance(src, FloatImm):
            continue

        src_extent = [r.extent for r in src.region]

        # Check extra dimensions
        assert len(src_extent) <= len(src0_extent) or all(
            analyzer.can_prove(src_extent[i] == 1)
            for i in range(len(src_extent) - len(src0_extent))
        )

        # Find broadcast dimensions
        broadcast_dims = []
        for i in range(1, min(len(src_extent), len(src0_extent)) + 1):
            if analyzer.can_prove(src_extent[-i] != 1) and analyzer.can_prove(
                src_extent[-i] != src0_extent[-i]
            ):
                raise ValueError(f"Shape mismatch: src0: {_srcs[0]}, src: {src}")
            elif analyzer.can_prove(src_extent[-i] != src0_extent[-i]):
                broadcast_dims.append(len(src0_extent) - i)

        # Add leading dimensions
        broadcast_dims += list(range(0, len(src0_extent) - len(src_extent)))

        # Create dimension mapping and verify partition
        src0_to_src_dim_map = {
            i: i + len(src_extent) - len(src0_extent)
            for i in range(len(src0_extent))
            if i not in broadcast_dims
        }
        inst_gen.link_buffer_regions(_srcs[0], src, src0_to_src_dim_map)
        assert inst_gen.check_partition_dim_match(
            _srcs[0], src
        ), f"partition dimension mismatch: src0: {_srcs[0]}, src: {src}"

    # Find instruction parameters for each source
    inst_types = []
    allowed_f_dim_srcs = [None] * len(_srcs) if allowed_f_dim_srcs is None else allowed_f_dim_srcs
    inst_repr = inst_gen.find_max_inst_size_from_one_region(_dst, allowed_f_dim_dst)
    for i, src in enumerate(_srcs):
        if isinstance(src, FloatImm):
            inst_types.append(InstType.TENSOR_SCALAR)
            continue

        allow_tt = allow_first_op_tensortensor or i != 0
        inst_repr_non_bcast = inst_gen.fit_inst_tile_to_region(
            inst_repr, src, allowed_f_dim_srcs[i]
        )
        inst_repr_bcast = inst_gen.fit_inst_tile_to_region(
            inst_repr, src, allowed_f_dim_srcs[i], broadcast=True
        )
        if i == 0:
            inst_repr = inst_repr_non_bcast
            continue
        plan = None
        if not allow_tt:
            plan = "tensorscalar"
        else:

            if (
                inst_repr_bcast.stride == 1
                and inst_repr_non_bcast.stride > 1
                and inst_repr_bcast.size > 1
            ):
                plan = "tensorscalar"
            elif (
                inst_repr_bcast.stride > 1
                and inst_repr_non_bcast.stride == 1
                and inst_repr_non_bcast.size > 1
            ):
                plan = "tensortensor"
            elif inst_repr_bcast.size > inst_repr_non_bcast.size:
                plan = "tensorscalar"
            else:
                plan = "tensortensor"
        if plan == "tensorscalar":
            inst_type = InstType.TENSOR_SCALAR
            inst_repr = inst_repr_bcast
        else:
            inst_type = InstType.TENSOR_TENSOR
            inst_repr = inst_repr_non_bcast
        inst_types.append(inst_type)

    return inst_repr, inst_types, reverse


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

    # Find instruction parameters
    inst_gen = InstructionGenerator([_dst, _src1, _src2], analyzer)
    inst_repr, inst_types, reverse = try_find_inst_nary(_dst, [_src1, _src2], analyzer, inst_gen)
    # Handle operand swapping if needed
    if reverse[0]:
        _src1, _src2 = _src2, _src1

    # Extract buffers and constants
    CONST = _src2 if isinstance(_src2, FloatImm) else None
    dst, src1 = _dst.buffer, _src1.buffer
    src2 = None if CONST is not None else _src2.buffer

    p_var = T.var("int32", name="P")
    b_var = T.var("int32", name="B")
    f_var = T.var("int32", name="F")
    p_size = dst.layout.size("P")
    inst_size_limit = op.config.get("max_inst_size", 512)
    inst_repr.bound_inst_size(inst_size_limit, analyzer)
    inst_gen.bind_inst_iter(_dst, p_var, p_size, 1, False)
    inst_gen.bind_inst_iter(_dst, f_var, inst_repr.size, inst_repr.stride, True)
    b_extent = inst_gen.fill_in_block_dim(_dst, b_var)
    # Setup execution parameters
    opcode = binary_map_ops[binary_op]

    # Select appropriate NKI function based on instruction type
    _func = T.nki.tensortensor if inst_types[0] == InstType.TENSOR_TENSOR else T.nki.tensorscalar
    func = lambda *args: (
        _func(*args, reverse[0]) if inst_types[0] == InstType.TENSOR_SCALAR else _func(*args)
    )

    # Define the implementation function
    @T.prim_func(tirp=True)
    def impl():
        for b_loop in T.serial(0, b_extent):
            with T.attr(0, "tensorized_nki_instruction", 1):
                for p_loop in T.serial(0, p_size, annotations={nki_dim: "P"}):
                    for f_loop in T.serial(0, inst_repr.size, annotations={nki_dim: "F"}):
                        inst_gen.set_bind_map_all({p_var: p_loop, f_var: f_loop, b_var: b_loop})

                        if inst_gen.make_guard(_dst):
                            dst_indices = T.meta_var(inst_gen.generate_indices(_dst))
                            src1_indices = T.meta_var(inst_gen.generate_indices(_src1))
                            if CONST is None:
                                src2_indices = T.meta_var(inst_gen.generate_indices(_src2))
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
