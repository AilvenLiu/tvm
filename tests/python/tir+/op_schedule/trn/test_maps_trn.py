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
import pytest

import tvm
from tvm.tir.layout import TrainiumLayout, TileLayout
import numpy as np
import tvm.testing
from tvm.script import ir as I
from tvm.script import tir as T
from tvm.script import tirp as Tp
from tvm.ir import assert_structural_equal

target = tvm.target.Target("aws/trn1/trn1.2xlarge")

Tp_func_map = {
    "reciprocal": Tp.reciprocal,
    "sqrt": Tp.sqrt,
    "memset": Tp.memset,
    "exp": Tp.exp,
    "add": Tp.add,
    "sub": Tp.sub,
    "mul": Tp.mul,
    "min": Tp.minimum,
    "max": Tp.maximum,
}


@pytest.mark.parametrize("op_type", ["reciprocal", "sqrt", "memset", "exp"])
def test_simple_unary(op_type):
    src_shape = [128, 512]
    src_layout = TrainiumLayout(
        dimension_types="PF", combined_1d_layout=T.TileLayout.from_tuple((128, 512), (1, 1))
    )
    dst_shape = [128, 512]
    dst_layout = TrainiumLayout(
        dimension_types="PF", combined_1d_layout=T.TileLayout.from_tuple((128, 512), (1, 1))
    )
    tp_func = Tp_func_map[op_type]

    # fmt: off
    @T.prim_func(tirp=True)
    def unary() -> None:
        with T.kernel():
            A_sbuf = T.alloc_buffer(src_shape, "float32", scope="trn.sbuf", layout=src_layout)
            B_sbuf = T.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
            if op_type == "memset":
                tp_func(B_sbuf, T.float32(0.0))
            else:
                tp_func(B_sbuf, A_sbuf)

    @T.prim_func(tirp=True)
    def expected():
        T.func_attr({"global_symbol": "unary"})
        with T.kernel():
            A_sbuf = T.alloc_buffer((128, 512), scope="trn.sbuf", logical_scope="kernel")
            B_sbuf = T.alloc_buffer((128, 512), scope="trn.sbuf", logical_scope="kernel")
            for b_loop, additional_b_loop in T.grid(1, 1):
                T.attr(0, "tensorized_nki_instruction", 1)
                for p_loop in T.serial(0, 128, annotations={"nki_dim":"P"}):
                    for f_loop in T.serial(0, 512, annotations={"nki_dim":"F"}):
                        if op_type == "reciprocal":
                            T.nki_reciprocal(
                                B_sbuf[p_loop, f_loop], A_sbuf[p_loop, f_loop]
                            )
                        elif op_type in ["sqrt", "exp"]:
                            T.nki_activation(
                                B_sbuf[p_loop, f_loop], A_sbuf[p_loop, f_loop], op_type
                            )
                        elif op_type == "memset":
                            T.nki_memset(B_sbuf[p_loop, f_loop], 0.0)
    # fmt: on
    with target:
        mod = tvm.IRModule({"main": unary})
        mod = tvm.tir.transform.LowerTIRp()(mod)
        assert_structural_equal(mod["main"], expected)


@pytest.mark.parametrize("op_type", ["reciprocal", "sqrt", "memset", "exp"])
def test_unary_in_a_loop(op_type):
    src_shape = [1024, 512]
    src_layout = TrainiumLayout(
        dimension_types="PF", combined_1d_layout=T.TileLayout.from_tuple((128, 4096), (1, 1))
    )
    dst_shape = [512, 512]
    dst_layout = TrainiumLayout(
        dimension_types="PF", combined_1d_layout=T.TileLayout.from_tuple((128, 2048), (1, 1))
    )

    Tp_func = Tp_func_map[op_type]
    # fmt: off
    @T.prim_func(tirp=True)
    def unary() -> None:
        with T.kernel():
            A_sbuf = T.alloc_buffer(src_shape, "float32", scope="trn.sbuf", layout=src_layout)
            B_sbuf = T.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
            A_sbuf_view = T.view(A_sbuf, A_sbuf.layout, (128, 8, 512))
            B_sbuf_view = T.view(B_sbuf, B_sbuf.layout, (128, 4, 512))
            for i in range(4):
                if op_type == "memset":
                    Tp_func(B_sbuf_view[:, i, :], T.float32(0.0))
                else:
                    Tp_func(B_sbuf_view[:, i, :], A_sbuf_view[:, i * 2, :])

    @T.prim_func(tirp=True)
    def expected():
        T.func_attr({"global_symbol": "unary"})
        with T.kernel():
            A_sbuf = T.alloc_buffer((128, 4096), scope="trn.sbuf", logical_scope="kernel")
            B_sbuf = T.alloc_buffer((128, 2048), scope="trn.sbuf", logical_scope="kernel")
            for i, b_loop, additional_b_loop in T.grid(4, 1, 1):
                T.attr(0, "tensorized_nki_instruction", 1)
                for p_loop in T.serial(0, 128, annotations={"nki_dim":"P"}):
                    for f_loop in T.serial(0, 512, annotations={"nki_dim":"F"}):
                        if op_type == "reciprocal":
                            T.nki_reciprocal(B_sbuf[p_loop, i * 512 + f_loop], A_sbuf[p_loop, i * 1024 + f_loop])
                        elif op_type in ["sqrt", "exp"]:
                            T.nki_activation(B_sbuf[p_loop, i * 512 + f_loop], A_sbuf[p_loop, i * 1024 + f_loop], op_type)
                        elif op_type == "memset":
                            T.nki_memset(B_sbuf[p_loop, i * 512 + f_loop], 0.0)
    # fmt: on
    with target:
        mod = tvm.IRModule({"main": unary})
        mod = tvm.tir.transform.LowerTIRp()(mod)
        assert_structural_equal(mod["main"], expected)


@pytest.mark.parametrize("op_type", ["add", "sub", "mul", "min", "max"])
@pytest.mark.parametrize(
    "operands_type",
    [
        "region_region",
        "const_region",
        "region_const",
        "region_broadcast_lhs",
        "region_broadcast_rhs",
    ],
)
def test_simple_binary(op_type, operands_type):
    const = T.float32(3.0)
    src1_shape = [128, 512] if operands_type != "region_broadcast_lhs" else [128, 1]
    src1_layout = TrainiumLayout(
        dimension_types="PF", combined_1d_layout=T.TileLayout.from_tuple(src1_shape, (1, 1))
    )
    src2_shape = [128, 512] if operands_type != "region_broadcast_rhs" else [128, 1]
    src2_layout = TrainiumLayout(
        dimension_types="PF", combined_1d_layout=T.TileLayout.from_tuple(src2_shape, (1, 1))
    )
    dst_shape = [128, 512]
    dst_layout = TrainiumLayout(
        dimension_types="PF", combined_1d_layout=T.TileLayout.from_tuple((128, 512), (1, 1))
    )
    Tp_func = Tp_func_map[op_type]

    # fmt: off
    @T.prim_func(tirp=True)
    def binary() ->None:
        with T.kernel():
            A_sbuf = T.alloc_buffer(src1_shape, "float32", scope="trn.sbuf", layout=src1_layout)
            B_sbuf = T.alloc_buffer(src2_shape, "float32", scope="trn.sbuf", layout=src2_layout)
            C_sbuf = T.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
            if operands_type == "region_region" or operands_type.startswith("region_broadcast"):
                Tp_func(C_sbuf, A_sbuf, B_sbuf)
            elif operands_type == "const_region":
                Tp_func(C_sbuf, const, A_sbuf)
            elif operands_type == "region_const":
                Tp_func(C_sbuf, A_sbuf, const)
                
    @T.prim_func(tirp=True)
    def expected():
        T.func_attr({"global_symbol": "binary"})
        with T.kernel():
            A_sbuf = T.alloc_buffer(src1_shape, scope="trn.sbuf", logical_scope="kernel")
            B_sbuf = T.alloc_buffer(src2_shape, scope="trn.sbuf", logical_scope="kernel")
            C_sbuf = T.alloc_buffer(dst_shape, scope="trn.sbuf", logical_scope="kernel")
            for b_loop, additional_b_loop in T.grid(1, 1):
                T.attr(0, "tensorized_nki_instruction", 1)
                for p_loop in T.serial(0, 128, annotations={"nki_dim":"P"}):
                    for f_loop in T.serial(0, 512, annotations={"nki_dim":"F"}):
                        if operands_type == "region_region":
                            T.nki_tensortensor(C_sbuf[p_loop, f_loop], A_sbuf[p_loop, f_loop], B_sbuf[p_loop, f_loop], op_type)
                        elif operands_type == "region_const":
                            T.nki_tensorscalar(C_sbuf[p_loop, f_loop], A_sbuf[p_loop, f_loop], T.float32(3.0), op_type, T.bool(False))
                        elif operands_type == "const_region":
                            T.nki_tensorscalar(C_sbuf[p_loop, f_loop], A_sbuf[p_loop, f_loop], T.float32(3.0), op_type, T.bool(True))
                        elif operands_type == "region_broadcast_rhs":
                            T.nki_tensorscalar(C_sbuf[p_loop, f_loop], A_sbuf[p_loop, f_loop], B_sbuf[p_loop, 0], op_type, T.bool(False))
                        elif operands_type == "region_broadcast_lhs":
                            T.nki_tensorscalar(C_sbuf[p_loop, f_loop], B_sbuf[p_loop, f_loop], A_sbuf[p_loop, 0], op_type, T.bool(True))
    # fmt: on
    with target:
        mod = tvm.IRModule({"main": binary})
        mod = tvm.tir.transform.LowerTIRp()(mod)
        assert_structural_equal(mod["main"], expected)


@pytest.mark.parametrize("op_type", ["add", "sub", "mul", "min", "max"])
@pytest.mark.parametrize(
    "operands_type",
    [
        "region_region",
        "const_region",
        "region_const",
        "region_broadcast_lhs",
        "region_broadcast_rhs",
    ],
)
def test_binary_complex(op_type, operands_type):
    src1_shape = [1024, 512] if operands_type != "region_broadcast_lhs" else [1024, 4]
    src1_layout_data_iter = (128, 4096) if operands_type != "region_broadcast_lhs" else (128, 32)
    src1_layout = TrainiumLayout(
        dimension_types="PF",
        combined_1d_layout=T.TileLayout.from_tuple(src1_layout_data_iter, (1, 1)),
    )
    src2_shape = [512, 512] if operands_type != "region_broadcast_rhs" else [128, 512]
    src2_layout_data_iter = (128, 2048) if operands_type != "region_broadcast_rhs" else (128, 512)
    src2_layout = TrainiumLayout(
        dimension_types="PF",
        combined_1d_layout=T.TileLayout.from_tuple(src2_layout_data_iter, (1, 1)),
    )

    dst_shape = [512, 512]
    dst_layout = TrainiumLayout(
        dimension_types="PF", combined_1d_layout=T.TileLayout.from_tuple((128, 2048), (1, 1))
    )
    const = T.float32(3.0)
    Tp_func = Tp_func_map[op_type]

    src1_view_shape = [128, 8, 512]
    src2_view_shape = [128, 4, 512] if operands_type != "region_broadcast_rhs" else [128, 1, 512]
    dst_view_shape = [128, 4, 512]
    if operands_type == "region_broadcast_lhs":
        src1_view_shape = [128, 8, 4, 1]
        src2_view_shape = [128, 4, 4, 128]
        dst_view_shape = [128, 4, 4, 128]

    # fmt: off
    @T.prim_func(tirp=True)
    def binary() -> None:
        with T.kernel():
            A_sbuf = T.alloc_buffer(src1_shape, "float32", scope="trn.sbuf", layout=src1_layout)
            B_sbuf = T.alloc_buffer(src2_shape, "float32", scope="trn.sbuf", layout=src2_layout)
            C_sbuf = T.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
            A_sbuf_view = T.view(A_sbuf, A_sbuf.layout, src1_view_shape)
            B_sbuf_view = T.view(B_sbuf, B_sbuf.layout, src2_view_shape)
            C_sbuf_view = T.view(C_sbuf, C_sbuf.layout, dst_view_shape)
            for i in range(4):
                if operands_type == "region_region":
                    Tp_func(C_sbuf_view[:, i, :], A_sbuf_view[:, i * 2, :], B_sbuf_view[:, i, :])
                elif operands_type == "region_const":
                    Tp_func(C_sbuf_view[:, i, :], A_sbuf_view[:, i * 2, :], const)
                elif operands_type == "const_region":
                    Tp_func(C_sbuf_view[:, i, :], const, A_sbuf_view[:, i * 2, :])
                elif operands_type == "region_broadcast_rhs":
                    Tp_func(C_sbuf_view[:, i, :], A_sbuf_view[:, i * 2, :], B_sbuf_view[:, 0, :])
                elif operands_type == "region_broadcast_lhs":
                    Tp_func(C_sbuf_view[:, i, :, :], A_sbuf_view[:, i*2,:, :], B_sbuf_view[:, i, :, :])
    
    f_extent = 128 if operands_type == "region_broadcast_lhs" else 512
    b_extent = 4 if operands_type == "region_broadcast_lhs" else 1
    
    @T.prim_func(tirp=True)
    def expected():
        T.func_attr({"global_symbol": "binary"})
        with T.kernel():
            A_sbuf = T.alloc_buffer(src1_layout_data_iter, scope="trn.sbuf", logical_scope="kernel")
            B_sbuf = T.alloc_buffer(src2_layout_data_iter, scope="trn.sbuf", logical_scope="kernel")
            C_sbuf = T.alloc_buffer((128, 2048), scope="trn.sbuf", logical_scope="kernel")
            for i, b_loop, additional_b_loop in T.grid(4, b_extent, 1):
                T.attr(0, "tensorized_nki_instruction", 1)
                for p_loop in T.serial(0, 128, annotations={"nki_dim":"P"}):
                    for f_loop in T.serial(0, f_extent, annotations={"nki_dim":"F"}):
                        if operands_type == "region_region":
                            T.nki_tensortensor(C_sbuf[p_loop, i * 512 + f_loop], A_sbuf[p_loop, i * 1024 + f_loop], B_sbuf[p_loop, i * 512 + f_loop], op_type)
                        elif operands_type == "const_region":
                            T.nki_tensorscalar(C_sbuf[p_loop, i * 512 + f_loop], A_sbuf[p_loop, i * 1024 + f_loop], T.float32(3.0), op_type, T.bool(True))
                        elif operands_type == "region_const":
                            T.nki_tensorscalar(C_sbuf[p_loop, i * 512 + f_loop], A_sbuf[p_loop, i * 1024 + f_loop], T.float32(3.0), op_type, T.bool(False))
                        elif operands_type == "region_broadcast_lhs":
                            T.nki_tensorscalar(C_sbuf[p_loop, i * 512 + b_loop * 128 + f_loop], B_sbuf[p_loop, i * 512 + b_loop * 128 + f_loop], A_sbuf[p_loop, i * 8 + b_loop], op_type, T.bool(True))
                        elif operands_type == "region_broadcast_rhs":
                            T.nki_tensortensor(C_sbuf[p_loop, i * 512 + f_loop], A_sbuf[p_loop, i * 1024 + f_loop], B_sbuf[p_loop, f_loop], op_type)

    # fmt: on

    with target:
        mod = tvm.IRModule({"main": binary})
        mod = tvm.tir.transform.LowerTIRp()(mod)
        assert_structural_equal(mod["main"], expected)


def test_binary_broadcast1():
    src1_shape = [32, 128, 512]
    src1_layout = TrainiumLayout(
        dimension_types="FFFP",
        combined_1d_layout=T.TileLayout.from_tuple((32, 128, 4, 128), (1, 32, 32 * 128, 1)),
    )
    src2_shape = [128, 512]
    src2_layout = TrainiumLayout(
        dimension_types="FP", combined_1d_layout=T.TileLayout.from_tuple((512, 128), (1, 1))
    )
    dst_shape = src1_shape
    dst_layout = src1_layout

    # fmt: off
    @T.prim_func(tirp=True)
    def binary() -> None:
        with T.kernel():
            A_sbuf = T.alloc_buffer(src1_shape, "float32", scope="trn.sbuf", layout=src1_layout)
            B_sbuf = T.alloc_buffer(src2_shape, "float32", scope="trn.sbuf", layout=src2_layout)
            C_sbuf = T.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
            Tp.add(C_sbuf, A_sbuf, B_sbuf)

    @T.prim_func(tirp=True)
    def expected():
        T.func_attr({"global_symbol": "binary"})
        with T.kernel():
            A_sbuf = T.alloc_buffer((128, 16384), scope="trn.sbuf", logical_scope="kernel")
            B_sbuf = T.alloc_buffer((128, 512), scope="trn.sbuf", logical_scope="kernel")
            C_sbuf = T.alloc_buffer((128, 16384), scope="trn.sbuf", logical_scope="kernel")
            for b_loop, additional_b_loop in T.grid(512, 1):
                T.attr(0, "tensorized_nki_instruction", 1)
                for p_loop in T.serial(0, 128, annotations={"nki_dim":"P"}):
                    for f_loop in T.serial(0, 32, annotations={"nki_dim":"F"}):
                        T.nki_tensorscalar(C_sbuf[p_loop, b_loop % 4 * 4096 + b_loop // 4 * 32 + f_loop], A_sbuf[p_loop, b_loop % 4 * 4096 + b_loop // 4 * 32 + f_loop], B_sbuf[p_loop, b_loop], "add", T.bool(False))
    # fmt: on

    with target:
        mod = tvm.IRModule({"main": binary})
        mod = tvm.tir.transform.LowerTIRp()(mod)
        assert_structural_equal(mod["main"], expected)


def test_binary_broadcast2():
    src1_shape = [32, 128, 512]
    src1_layout = TrainiumLayout(
        dimension_types="FFFP",
        combined_1d_layout=T.TileLayout.from_tuple((32, 128, 4, 128), (128, 1, 32 * 128, 1)),
    )
    src2_shape = [128, 512]
    src2_layout = TrainiumLayout(
        dimension_types="FFP",
        combined_1d_layout=T.TileLayout.from_tuple((128, 4, 128), (1, 128, 1)),
    )
    dst_shape = src1_shape
    dst_layout = src1_layout

    # fmt: off
    @T.prim_func(tirp=True)
    def binary() -> None:
        with T.kernel():
            A_sbuf = T.alloc_buffer(src1_shape, "float32", scope="trn.sbuf", layout=src1_layout)
            B_sbuf = T.alloc_buffer(src2_shape, "float32", scope="trn.sbuf", layout=src2_layout)
            C_sbuf = T.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
            Tp.add(C_sbuf, A_sbuf, B_sbuf)

    @T.prim_func(tirp=True)
    def expected():
        T.func_attr({"global_symbol": "binary"})
        with T.kernel():
            A_sbuf = T.alloc_buffer((128, 16384), scope="trn.sbuf", logical_scope="kernel")
            B_sbuf = T.alloc_buffer((128, 512), scope="trn.sbuf", logical_scope="kernel")
            C_sbuf = T.alloc_buffer((128, 16384), scope="trn.sbuf", logical_scope="kernel")
            for b_loop, additional_b_loop in T.grid(128, 1):
                T.attr(0, "tensorized_nki_instruction", 1)
                for p_loop in T.serial(0, 128, annotations={"nki_dim":"P"}):
                    for f_loop in T.serial(0, 128, annotations={"nki_dim":"F"}):
                        T.nki_tensortensor(C_sbuf[p_loop, b_loop % 4 * 4096 + b_loop // 4 * 128 + f_loop], A_sbuf[p_loop, b_loop % 4 * 4096 + b_loop // 4 * 128 + f_loop], B_sbuf[p_loop, b_loop % 4 * 128 + f_loop], "add")
    # fmt: on

    with target:
        mod = tvm.IRModule({"main": binary})
        mod = tvm.tir.transform.LowerTIRp()(mod)
        assert_structural_equal(mod["main"], expected)


def test_unary_complex1():
    dst_layout = TrainiumLayout(
        dimension_types="FPF",
        combined_1d_layout=T.TileLayout.from_tuple((32, 128, 256), (256, 1, 1)),
    )
    dst_shape = [4096, 256]
    # fmt: off
    @T.prim_func(tirp=True)
    def unary() -> None:
        with T.kernel():
            A_sbuf = T.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
            Tp.memset(A_sbuf, T.float32(0.0))
            
    @T.prim_func(tirp=True)
    def expected():
        T.func_attr({"global_symbol": "unary"})
        with T.kernel():
            A_sbuf = T.alloc_buffer((128, 8192), scope="trn.sbuf", logical_scope="kernel")
            for b_loop, additional_b_loop in T.grid(1, 16):
                T.attr(0, "tensorized_nki_instruction", 1)
                for p_loop in T.serial(0, 128, annotations={"nki_dim":"P"}):
                    for f_loop in T.serial(0, 512, annotations={"nki_dim":"F"}):
                        T.nki_memset(A_sbuf[p_loop, additional_b_loop * 512 + f_loop], T.float32(0.0))       
    # fmt: on
    with target:
        mod = tvm.IRModule({"main": unary})
        mod = tvm.tir.transform.LowerTIRp()(mod)
        assert_structural_equal(mod["main"], expected)


def test_binary_broadcast3():
    src1_shape = [128, 512]
    src1_layout = TrainiumLayout(
        dimension_types="FFP",
        combined_1d_layout=T.TileLayout.from_tuple((128, 4, 128), (1, 128, 1)),
    )
    src2_shape = [32, 128, 512]
    src2_layout = TrainiumLayout(
        dimension_types="FFFP",
        combined_1d_layout=T.TileLayout.from_tuple((32, 128, 4, 128), (128, 1, 32 * 128, 1)),
    )
    dst_shape = src1_shape
    dst_layout = src1_layout

    # fmt: off
    @T.prim_func(tirp=True)
    def binary() -> None:
        with T.kernel():
            A_sbuf = T.alloc_buffer(src1_shape, "float32", scope="trn.sbuf", layout=src1_layout)
            B_sbuf = T.alloc_buffer(src2_shape, "float32", scope="trn.sbuf", layout=src2_layout)
            C_sbuf = T.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
            Tp.add(C_sbuf, A_sbuf, B_sbuf[0])

    @T.prim_func(tirp=True)
    def expected():
        T.func_attr({"global_symbol": "binary"})
        with T.kernel():
            A_sbuf = T.alloc_buffer((128, 512), scope="trn.sbuf", logical_scope="kernel")
            B_sbuf = T.alloc_buffer((128, 16384), scope="trn.sbuf", logical_scope="kernel")
            C_sbuf = T.alloc_buffer((128, 512), scope="trn.sbuf", logical_scope="kernel")
            for b_loop, additional_b_loop in T.grid(4, 1):
                T.attr(0, "tensorized_nki_instruction", 1)
                for p_loop in T.serial(0, 128, annotations={"nki_dim":"P"}):
                    for f_loop in T.serial(0, 128, annotations={"nki_dim":"F"}):
                        T.nki_tensortensor(C_sbuf[p_loop, b_loop * 128 + f_loop], A_sbuf[p_loop, b_loop * 128 + f_loop], B_sbuf[p_loop, b_loop * 4096 + f_loop], "add")
    # fmt: on

    with target:
        mod = tvm.IRModule({"main": binary})
        mod = tvm.tir.transform.LowerTIRp()(mod)
        assert_structural_equal(mod["main"], expected)


@pytest.mark.parametrize("op_type", ["sqrt"])
def test_unary_with_bias_scale(op_type):
    src_shape = [512, 1024]
    src_layout = TrainiumLayout(
        dimension_types="PF",
        combined_1d_layout=T.TileLayout.from_tuple((128, 4096), (1, 1)),
    )
    dst_shape = src_shape
    dst_layout = src_layout
    bias_shape = [512, 1]
    bias_layout = TrainiumLayout(
        dimension_types="PF",
        combined_1d_layout=T.TileLayout.from_tuple((128, 4), (1, 1)),
    )
    scale = T.float32(2.0)

    # fmt: off
    @T.prim_func(tirp=True)
    def unary() -> None:
        with T.kernel():
            A_sbuf = T.alloc_buffer(src_shape, "float32", scope="trn.sbuf", layout=src_layout)
            B_sbuf = T.alloc_buffer(bias_shape, "float32", scope="trn.sbuf", layout=bias_layout)
            C_sbuf = T.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
            Tp.sqrt(C_sbuf, A_sbuf, bias=B_sbuf, scale=scale)

    @T.prim_func(tirp=True)
    def expected():
        T.func_attr({"global_symbol": "unary"})
        with T.kernel():
            A_sbuf = T.alloc_buffer((128, 4096), scope="trn.sbuf", logical_scope="kernel")
            B_sbuf = T.alloc_buffer((128, 4), scope="trn.sbuf", logical_scope="kernel")
            C_sbuf = T.alloc_buffer((128, 4096), scope="trn.sbuf", logical_scope="kernel")
            for b_loop, additional_b_loop in T.grid(4, 2):
                T.attr(0, "tensorized_nki_instruction", 1)
                for p_loop in T.serial(0, 128, annotations={"nki_dim":"P"}):
                    for f_loop in T.serial(0, 512, annotations={"nki_dim":"F"}):
                        T.nki_activation(C_sbuf[p_loop, b_loop * 1024 + additional_b_loop * 512 + f_loop], A_sbuf[p_loop, b_loop * 1024 + additional_b_loop * 512 + f_loop], "sqrt", B_sbuf[p_loop, b_loop], T.float32(2.0))
    # fmt: off
    with target:
        mod = tvm.IRModule({"main": unary})
        mod = tvm.tir.transform.LowerTIRp()(mod)
        assert_structural_equal(mod["main"], expected)


@pytest.mark.parametrize("op_type", ["sqrt"])
def test_unary_with_bias_scale_2(op_type):
    src_shape = [512, 1024]
    src_layout = TrainiumLayout(
        dimension_types="PF",
        combined_1d_layout=T.TileLayout.from_tuple((128, 4096), (1, 1)),
    )
    dst_shape = src_shape
    dst_layout = src_layout
    bias = T.float32(1.0)
    scale = T.float32(2.0)

    # fmt: off
    @T.prim_func(tirp=True)
    def unary() -> None:
        with T.kernel():
            A_sbuf = T.alloc_buffer(src_shape, "float32", scope="trn.sbuf", layout=src_layout)
            C_sbuf = T.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
            Tp.sqrt(C_sbuf, A_sbuf, bias=bias, scale=scale)

    @T.prim_func(tirp=True)
    def expected():
        T.func_attr({"global_symbol": "unary"})
        with T.kernel():
            A_sbuf = T.alloc_buffer((128, 4096), scope="trn.sbuf", logical_scope="kernel")
            C_sbuf = T.alloc_buffer((128, 4096), scope="trn.sbuf", logical_scope="kernel")
            for b_loop, additional_b_loop in T.grid(1, 8):
                T.attr(0, "tensorized_nki_instruction", 1)
                for p_loop in T.serial(0, 128, annotations={"nki_dim":"P"}):
                    for f_loop in T.serial(0, 512, annotations={"nki_dim":"F"}):
                        T.nki_activation(C_sbuf[p_loop, additional_b_loop * 512 + f_loop], A_sbuf[p_loop, additional_b_loop * 512 + f_loop], "sqrt", bias, scale)
    # fmt: off
    with target:
        mod = tvm.IRModule({"main": unary})
        mod = tvm.tir.transform.LowerTIRp()(mod)
        assert_structural_equal(mod["main"], expected)


def test_binary_with_guard():
    src1_shape = [32, 128, 512]
    src1_layout = TrainiumLayout(
        dimension_types="FFFP",
        combined_1d_layout=T.TileLayout.from_tuple((32, 128, 4, 128), (128, 1, 32 * 128, 1)),
    )
    src2_shape = [128, 512]
    src2_layout = TrainiumLayout(
        dimension_types="FFP",
        combined_1d_layout=T.TileLayout.from_tuple((128, 4, 128), (1, 128, 1)),
    )
    dst_shape = src1_shape
    dst_layout = src1_layout

    # fmt: off
    @T.prim_func(tirp=True)
    def binary() -> None:
        with T.kernel():
            A_sbuf = T.alloc_buffer(src1_shape, "float32", scope="trn.sbuf", layout=src1_layout)
            B_sbuf = T.alloc_buffer(src2_shape, "float32", scope="trn.sbuf", layout=src2_layout)
            C_sbuf = T.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
            for j in range(4):
                Tp.add(C_sbuf[:, :, 0:j*128], A_sbuf[:, :, 0:j*128], B_sbuf[:, 0:j*128])

    @T.prim_func(tirp=True)
    def expected():
        T.func_attr({"global_symbol": "binary"})
        with T.kernel():
            A_sbuf = T.alloc_buffer((128, 16384), scope="trn.sbuf", logical_scope="kernel")
            B_sbuf = T.alloc_buffer((128, 512), scope="trn.sbuf", logical_scope="kernel")
            C_sbuf = T.alloc_buffer((128, 16384), scope="trn.sbuf", logical_scope="kernel")
            for j, b_loop, additional_b_loop in T.grid(4, 96, 1):
                T.attr(0, "tensorized_nki_instruction", 1)
                for p_loop in T.serial(0, 128, annotations={"nki_dim":"P"}):
                    for f_loop in T.serial(0, 128, annotations={"nki_dim":"F"}):
                        if b_loop % 3 - j < 0:
                            T.nki_tensortensor(C_sbuf[p_loop, b_loop % 3 * 4096 + b_loop // 3 * 128 + f_loop], A_sbuf[p_loop, b_loop % 3 * 4096 + b_loop // 3 * 128 + f_loop], B_sbuf[p_loop, b_loop % 3 * 128 + f_loop], "add")

    # fmt: on
    with target:
        mod = tvm.IRModule({"main": binary})
        mod = tvm.tir.transform.LowerTIRp()(mod)
        mod = tvm.tir.transform.Simplify()(mod)
        assert_structural_equal(mod["main"], expected)


def test_unary_with_guard():
    src_shape = [512, 1024]
    src_layout = TrainiumLayout(
        dimension_types="FPF",
        combined_1d_layout=T.TileLayout.from_tuple((4, 128, 1024), (1024, 1, 1)),
    )
    dst_shape = src_shape
    dst_layout = src_layout
    bias_shape = [512, 1]
    bias_layout = TrainiumLayout(
        dimension_types="FP",
        combined_1d_layout=T.TileLayout.from_tuple((4, 128), (1, 1)),
    )
    scale = T.float32(2.0)

    # fmt: off
    @T.prim_func(tirp=True)
    def unary() -> None:
        with T.kernel():
            A_sbuf = T.alloc_buffer(src_shape, "float32", scope="trn.sbuf", layout=src_layout)
            B_sbuf = T.alloc_buffer(bias_shape, "float32", scope="trn.sbuf", layout=bias_layout)
            C_sbuf = T.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
            for i in range(4):
                for j in range(4):
                    Tp.sqrt(C_sbuf[0: (i+1) * 128, 0: (j+1)*256], A_sbuf[0: (i+1) * 128, 0: (j+1)*256], bias=B_sbuf[0: (i+1) * 128, 0], scale=scale)

    @T.prim_func(tirp=True)
    def expected():
        T.func_attr({"global_symbol": "unary"})
        with T.kernel():
            A_sbuf = T.alloc_buffer((128, 4096), scope="trn.sbuf", logical_scope="kernel")
            B_sbuf = T.alloc_buffer((128, 4), scope="trn.sbuf", logical_scope="kernel")
            C_sbuf = T.alloc_buffer((128, 4096), scope="trn.sbuf", logical_scope="kernel")
            for i, j, b_loop, additional_b_loop in T.grid(4, 4, 4, 2):
                T.attr(0, "tensorized_nki_instruction", 1)
                for p_loop in T.serial(0, 128, annotations={"nki_dim":"P"}):
                    for f_loop in T.serial(0, 512, annotations={"nki_dim":"F"}):
                        if b_loop - i < 1 and additional_b_loop * 512 + f_loop < j * 256 + 256:
                            T.nki_activation(C_sbuf[p_loop, b_loop * 1024 + additional_b_loop * 512 + f_loop], A_sbuf[p_loop, b_loop * 1024 + additional_b_loop * 512 + f_loop], "sqrt", B_sbuf[p_loop, b_loop], T.float32(2.0))
     # fmt: off
    with target:
        mod = tvm.IRModule({"main": unary})
        mod = tvm.tir.transform.LowerTIRp()(mod)
        mod = tvm.tir.transform.Simplify()(mod)
        assert_structural_equal(mod["main"], expected)


if __name__ == "__main__":
    tvm.testing.main()
