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


def test_simple_gemm():
    A_layout = T.TrainiumLayout(
        dimension_types="FP", combined_1d_layout=T.TileLayout.from_tuple((128, 128), (1, 1))
    )
    B_layout = T.TrainiumLayout(
        dimension_types="PF", combined_1d_layout=T.TileLayout.from_tuple((128, 128), (1, 1))
    )

    C_layout = T.TrainiumPSUMLayout(
        dimension_types="PF", combined_1d_layout=T.TileLayout.from_tuple((128, 128), (1, 1))
    )
    # fmt: off
    @T.prim_func(tirp=True)
    def gemm() -> None:
        with T.kernel():
            A_sbuf = T.alloc_buffer((128, 128), "float32", scope="trn.sbuf", layout=A_layout)
            B_sbuf = T.alloc_buffer((128, 128), "float32", scope="trn.sbuf", layout=B_layout)
            C_psum = T.alloc_buffer((128, 128), "float32", scope="trn.psum", layout=C_layout)
            Tp.gemm(C_psum, A_sbuf, B_sbuf, C_psum)
            
    @T.prim_func(tirp=True)
    def expected():
        T.func_attr({"global_symbol": "gemm"})
        with T.kernel():
            A_sbuf = T.alloc_buffer((128, 128), scope="trn.sbuf", logical_scope="kernel")
            B_sbuf = T.alloc_buffer((128, 128), scope="trn.sbuf", logical_scope="kernel")
            C_psum = T.alloc_buffer((1, 128, 128), scope="trn.psum", logical_scope="kernel")
            for lhs_b_loop, rhs_b_loop, reduction_b_loop, additional_lhs_b_loop, additional_rhs_b_loop in T.grid(1, 1, 1, 1, 1):
                T.attr(0, "tensorized_nki_instruction", 1)
                for p_loop, lhs_f_loop, rhs_f_loop in T.grid(128, 128, 128):
                    T.nki_matmul(C_psum[0, lhs_f_loop, rhs_f_loop], A_sbuf[p_loop, lhs_f_loop], B_sbuf[p_loop, rhs_f_loop], T.bool(True))
    # fmt: on
    with target:
        mod = tvm.IRModule({"main": gemm})
        mod = tvm.tir.transform.LowerTIRp()(mod)
        assert_structural_equal(mod["main"], expected)


def test_larger_gemm():
    A_layout = T.TrainiumLayout(
        dimension_types="FFFP",
        combined_1d_layout=T.TileLayout.from_tuple((2, 128, 4, 128), (512, 1, 128, 1)),
    )
    B_layout = T.TrainiumLayout(
        dimension_types="FPFF",
        combined_1d_layout=T.TileLayout.from_tuple((4, 128, 2, 128), (256, 1, 128, 1)),
    )

    C_layout = T.TrainiumPSUMLayout(
        dimension_types="FPFF",
        combined_1d_layout=T.TileLayout.from_tuple((2, 128, 2, 128), (256, 1, 128, 1)),
    )
    # fmt: off
    @T.prim_func(tirp=True)
    def gemm() -> None:
        with T.kernel():
            A_sbuf = T.alloc_buffer((256, 512), "float32", scope="trn.sbuf", layout=A_layout)
            B_sbuf = T.alloc_buffer((512, 256), "float32", scope="trn.sbuf", layout=B_layout)
            C_psum = T.alloc_buffer((256, 256), "float32", scope="trn.psum", layout=C_layout)
            Tp.gemm(C_psum, A_sbuf, B_sbuf, C_psum)

    @T.prim_func(tirp=True)
    def expected():
        T.func_attr({"global_symbol": "gemm"})
        with T.kernel():
            A_sbuf = T.alloc_buffer((128, 1024), scope="trn.sbuf", logical_scope="kernel")
            B_sbuf = T.alloc_buffer((128, 1024), scope="trn.sbuf", logical_scope="kernel")
            C_psum = T.alloc_buffer((1, 128, 512), scope="trn.psum", logical_scope="kernel")
            for lhs_b_loop, rhs_b_loop, reduction_b_loop, additional_lhs_b_loop, additional_rhs_b_loop in T.grid(2, 1, 4, 1, 1):
                T.attr(0, "tensorized_nki_instruction", 1)
                for p_loop, lhs_f_loop, rhs_f_loop in T.grid(128, 128, 256):
                    T.nki_matmul(C_psum[0, lhs_f_loop, lhs_b_loop * 256 + rhs_f_loop], A_sbuf[p_loop, lhs_b_loop * 512 + reduction_b_loop * 128 + lhs_f_loop], B_sbuf[p_loop, reduction_b_loop * 256 + rhs_f_loop], T.bool(True))
    # fmt: on
    with target:
        mod = tvm.IRModule({"main": gemm})
        mod = tvm.tir.transform.LowerTIRp()(mod)
        assert_structural_equal(mod["main"], expected)


def test_gemm_in_a_loop():
    A_layout = T.TrainiumLayout(
        dimension_types="FFFP",
        combined_1d_layout=T.TileLayout.from_tuple((4, 128, 8, 128), (1024, 1, 128, 1)),
    )
    B_layout = T.TrainiumLayout(
        dimension_types="FPFF",
        combined_1d_layout=T.TileLayout.from_tuple((8, 128, 2, 128), (256, 1, 128, 1)),
    )

    C_layout = T.TrainiumPSUMLayout(
        dimension_types="FPFF",
        combined_1d_layout=T.TileLayout.from_tuple((4, 128, 2, 128), (256, 1, 128, 1)),
    )
    # fmt: off
    @T.prim_func(tirp=True)
    def gemm() -> None:
        with T.kernel():
            A_sbuf = T.alloc_buffer((512, 1024), "float32", scope="trn.sbuf", layout=A_layout)
            B_sbuf = T.alloc_buffer((1024, 256), "float32", scope="trn.sbuf", layout=B_layout)
            C_psum = T.alloc_buffer((512, 256), "float32", scope="trn.psum", layout=C_layout)
            for i in range(2):
                for k in range(2):
                    Tp.gemm(
                        C_psum[256 * i : 256 * i + 256, :],
                        A_sbuf[256 * i : 256 * i + 256, 512 * k : 512 * k + 512],
                        B_sbuf[512 * k : 512 * k + 512, :],
                        C_psum[256 * i : 256 * i + 256, :],
                    )
                    
    @T.prim_func(tirp=True)
    def expected():
        T.func_attr({"global_symbol": "gemm"})
        with T.kernel():
            A_sbuf = T.alloc_buffer((128, 4096), scope="trn.sbuf", logical_scope="kernel")
            B_sbuf = T.alloc_buffer((128, 2048), scope="trn.sbuf", logical_scope="kernel")
            C_psum = T.alloc_buffer((2, 128, 512), scope="trn.psum", logical_scope="kernel")
            for i, k, lhs_b_loop, rhs_b_loop, reduction_b_loop, additional_lhs_b_loop, additional_rhs_b_loop in T.grid(2, 2, 2, 1, 4, 1, 1):
                T.attr(0, "tensorized_nki_instruction", 1)
                for p_loop, lhs_f_loop, rhs_f_loop in T.grid(128, 128, 256):
                    T.nki_matmul(C_psum[i, lhs_f_loop, lhs_b_loop * 256 + rhs_f_loop], A_sbuf[p_loop, i * 2048 + lhs_b_loop * 1024 + k * 512 + reduction_b_loop * 128 + lhs_f_loop], B_sbuf[p_loop, k * 1024 + reduction_b_loop * 256 + rhs_f_loop], T.bool(True))
    # fmt: on
    with target:
        mod = tvm.IRModule({"main": gemm})
        mod = tvm.tir.transform.LowerTIRp()(mod)
        assert_structural_equal(mod["main"], expected)


def test_gemm_with_stride():
    A_layout = T.TrainiumLayout(
        dimension_types="FFPF",
        combined_1d_layout=T.TileLayout.from_tuple((4, 128, 128, 8), (1024, 1, 1, 128)),
    )
    B_layout = T.TrainiumLayout(
        dimension_types="PFFF",
        combined_1d_layout=T.TileLayout.from_tuple((128, 8, 2, 128), (1, 512, 256, 2)),
    )

    C_layout = T.TrainiumPSUMLayout(
        dimension_types="FPFF",
        combined_1d_layout=T.TileLayout.from_tuple((4, 128, 2, 128), (256, 1, 128, 1)),
    )
    # fmt: off
    @T.prim_func(tirp=True)
    def gemm() -> None:
        with T.kernel():
            A_sbuf = T.alloc_buffer((512, 512, 2), "float32", scope="trn.sbuf", layout=A_layout)
            B_sbuf = T.alloc_buffer((512, 2, 256), "float32", scope="trn.sbuf", layout=B_layout)
            C_psum = T.alloc_buffer((512, 256), "float32", scope="trn.psum", layout=C_layout)
            for i in range(2):
                for k in range(2):
                    Tp.gemm(
                        C_psum[256 * i : 256 * i + 256, :],
                        A_sbuf[256 * i : 256 * i + 256, :, k],
                        B_sbuf[:, k, :],
                        C_psum[256 * i : 256 * i + 256, :],
                    )
                    
    @T.prim_func(tirp=True)
    def expected():
        T.func_attr({"global_symbol": "gemm"})
        with T.kernel():
            A_sbuf = T.alloc_buffer((128, 4096), scope="trn.sbuf", logical_scope="kernel")
            B_sbuf = T.alloc_buffer((128, 4095), scope="trn.sbuf", logical_scope="kernel")
            C_psum = T.alloc_buffer((2, 128, 512), scope="trn.psum", logical_scope="kernel")
            for i, k, lhs_b_loop, rhs_b_loop, reduction_b_loop, additional_lhs_b_loop, additional_rhs_b_loop in T.grid(2, 2, 2, 1, 4, 1, 1):
                T.attr(0, "tensorized_nki_instruction", 1)
                for p_loop, lhs_f_loop, rhs_f_loop in T.grid(128, 128, 256):
                    T.nki_matmul(C_psum[i, lhs_f_loop, lhs_b_loop * 256 + rhs_f_loop], A_sbuf[p_loop, i * 2048 + lhs_b_loop * 1024 + reduction_b_loop * 256 + k * 128 + lhs_f_loop], B_sbuf[p_loop, reduction_b_loop * 1024 + k * 512 + rhs_f_loop * 2], T.bool(True))
    # fmt: on

    with target:
        mod = tvm.IRModule({"main": gemm})
        mod = tvm.tir.transform.LowerTIRp()(mod)
        assert_structural_equal(mod["main"], expected)


def test_gemm_swap_lhs_rhs():
    A_layout = T.TrainiumLayout(
        dimension_types="FFFP",
        combined_1d_layout=T.TileLayout.from_tuple((4, 128, 8, 128), (1024, 1, 128, 1)),
    )
    B_layout = T.TrainiumLayout(
        dimension_types="FPFF",
        combined_1d_layout=T.TileLayout.from_tuple((8, 128, 2, 128), (256, 1, 128, 1)),
    )

    C_layout = T.TrainiumPSUMLayout(
        dimension_types="FFFP",
        combined_1d_layout=T.TileLayout.from_tuple((4, 128, 2, 128), (256, 1, 128, 1)),
    )
    # fmt: off
    @T.prim_func(tirp=True)
    def gemm() -> None:
        with T.kernel():
            A_sbuf = T.alloc_buffer((512, 1024), "float32", scope="trn.sbuf", layout=A_layout)
            B_sbuf = T.alloc_buffer((1024, 256), "float32", scope="trn.sbuf", layout=B_layout)
            C_psum = T.alloc_buffer((512, 256), "float32", scope="trn.psum", layout=C_layout)
            for i in range(2):
                for k in range(2):
                    Tp.gemm(
                        C_psum[256 * i : 256 * i + 256, :],
                        A_sbuf[256 * i : 256 * i + 256, 512 * k : 512 * k + 512],
                        B_sbuf[512 * k : 512 * k + 512, :],
                        C_psum[256 * i : 256 * i + 256, :],
                    )
                    
    @T.prim_func(tirp=True)
    def expected():
        T.func_attr({"global_symbol": "gemm"})
        with T.kernel():
            A_sbuf = T.alloc_buffer((128, 4096), scope="trn.sbuf", logical_scope="kernel")
            B_sbuf = T.alloc_buffer((128, 2048), scope="trn.sbuf", logical_scope="kernel")
            C_psum = T.alloc_buffer((2, 128, 512), scope="trn.psum", logical_scope="kernel")
            for i, k, lhs_b_loop, rhs_b_loop, reduction_b_loop, additional_lhs_b_loop, additional_rhs_b_loop in T.grid(2, 2, 1, 2, 4, 2, 1):
                T.attr(0, "tensorized_nki_instruction", 1)
                for p_loop, lhs_f_loop, rhs_f_loop in T.grid(128, 128, 128):
                    T.nki_matmul(C_psum[i, lhs_f_loop, rhs_b_loop * 256 + additional_lhs_b_loop * 128 + rhs_f_loop], B_sbuf[p_loop, k * 1024 + reduction_b_loop * 256 + additional_lhs_b_loop * 128 + lhs_f_loop], A_sbuf[p_loop, i * 2048 + rhs_b_loop * 1024 + k * 512 + reduction_b_loop * 128 + rhs_f_loop], T.bool(True))
    # fmt: on
    with target:
        mod = tvm.IRModule({"main": gemm})
        mod = tvm.tir.transform.LowerTIRp()(mod)
        assert_structural_equal(mod["main"], expected)


def test_gemm_with_sbuf_output():
    A_layout = T.TrainiumLayout(
        dimension_types="FFFP",
        combined_1d_layout=T.TileLayout.from_tuple((4, 128, 8, 128), (1024, 1, 128, 1)),
    )
    B_layout = T.TrainiumLayout(
        dimension_types="FPFF",
        combined_1d_layout=T.TileLayout.from_tuple((8, 128, 2, 128), (256, 1, 128, 1)),
    )

    C_layout = T.TrainiumLayout(
        dimension_types="FFFP",
        combined_1d_layout=T.TileLayout.from_tuple((4, 128, 2, 128), (256, 1, 128, 1)),
    )
    # fmt: off
    @T.prim_func(tirp=True)
    def gemm() -> None:
        with T.kernel():
            A_sbuf = T.alloc_buffer((512, 1024), "float32", scope="trn.sbuf", layout=A_layout)
            B_sbuf = T.alloc_buffer((1024, 256), "float32", scope="trn.sbuf", layout=B_layout)
            C_sbuf = T.alloc_buffer((512, 256), "float32", scope="trn.sbuf", layout=C_layout)
            for i in range(2):
                for k in range(2):
                    Tp.gemm(
                        C_sbuf[256 * i : 256 * i + 256, :],
                        A_sbuf[256 * i : 256 * i + 256, 512 * k : 512 * k + 512],
                        B_sbuf[512 * k : 512 * k + 512, :],
                        C_sbuf[256 * i : 256 * i + 256, :],
                    )
    @T.prim_func(tirp=True)
    def expected():
        T.func_attr({"global_symbol": "gemm"})
        with T.kernel():
            A_sbuf = T.alloc_buffer((128, 4096), scope="trn.sbuf", logical_scope="kernel")
            B_sbuf = T.alloc_buffer((128, 2048), scope="trn.sbuf", logical_scope="kernel")
            C_sbuf = T.alloc_buffer((128, 1024), scope="trn.sbuf", logical_scope="kernel")
            buffer = T.alloc_buffer((8, 128, 512), scope="trn.psum",logical_scope="kernel", allocated_addr=[0, 0])
            for i, k, lhs_b_loop, rhs_b_loop, additional_lhs_b_loop, additional_rhs_b_loop in T.grid(2, 2, 1, 2, 2, 1):
                for reduction_b_loop in range(4):
                    T.attr(0, "tensorized_nki_instruction", 1)
                    for p_loop, lhs_f_loop, rhs_f_loop in T.grid(128, 128, 128):
                        T.nki_matmul(buffer[0, lhs_f_loop, additional_lhs_b_loop * 256 + rhs_b_loop * 128 + rhs_f_loop], B_sbuf[p_loop, k * 1024 + reduction_b_loop * 256 + additional_lhs_b_loop * 128 + lhs_f_loop], A_sbuf[p_loop, i * 2048 + rhs_b_loop * 1024 + k * 512 + reduction_b_loop * 128 + rhs_f_loop], T.bool(True))
                T.attr(0, "tensorized_nki_instruction", 1)
                for lhs_f_loop, rhs_f_loop in T.grid(128, 128):
                    T.nki_tensor_copy(C_sbuf[lhs_f_loop, i * 512 + rhs_b_loop * 256 + additional_lhs_b_loop * 128 + rhs_f_loop], buffer[0, lhs_f_loop, additional_lhs_b_loop * 256 + rhs_b_loop * 128 + rhs_f_loop])
    # fmt: on
    with target:
        mod = tvm.IRModule({"main": gemm})
        mod = tvm.tir.transform.LowerTIRp()(mod)
        mod = tvm.tir.transform.Simplify()(mod)
        assert_structural_equal(mod["main"], expected)


def test_gemm_different_shape():
    A_layout = T.TrainiumLayout(
        dimension_types="FFFFP",
        combined_1d_layout=T.TileLayout.from_tuple((2, 4, 128, 8, 128), (4096, 1024, 1, 128, 1)),
    )
    B_layout = T.TrainiumLayout(
        dimension_types="FPFF",
        combined_1d_layout=T.TileLayout.from_tuple((8, 128, 2, 128), (256, 1, 128, 1)),
    )

    C_layout = T.TrainiumPSUMLayout(
        dimension_types="FFFP",
        combined_1d_layout=T.TileLayout.from_tuple((4, 128, 2, 128), (256, 1, 128, 1)),
    )
    # fmt: off
    @T.prim_func(tirp=True)
    def gemm() -> None:
        with T.kernel():
            A_sbuf = T.alloc_buffer((2, 512, 1024), "float32", scope="trn.sbuf", layout=A_layout)
            B_sbuf = T.alloc_buffer((1024, 256), "float32", scope="trn.sbuf", layout=B_layout)
            C_psum = T.alloc_buffer((512, 256), "float32", scope="trn.psum", layout=C_layout)
            for i in range(2):
                for k in range(2):
                    Tp.gemm(
                        C_psum[256 * i : 256 * i + 256, :],
                        A_sbuf[1, 256 * i : 256 * i + 256, 512 * k : 512 * k + 512],
                        B_sbuf[512 * k : 512 * k + 512, :],
                        C_psum[256 * i : 256 * i + 256, :],
                    )
                    
    @T.prim_func(tirp=True)
    def expected():
        T.func_attr({"global_symbol": "gemm"})
        with T.kernel():
            A_sbuf = T.alloc_buffer((128, 8192), scope="trn.sbuf", logical_scope="kernel")
            B_sbuf = T.alloc_buffer((128, 2048), scope="trn.sbuf", logical_scope="kernel")
            C_psum = T.alloc_buffer((2, 128, 512), scope="trn.psum", logical_scope="kernel")
            for i, k, lhs_b_loop, rhs_b_loop, reduction_b_loop, additional_lhs_b_loop, additional_rhs_b_loop in T.grid(2, 2, 1, 2, 4, 2, 1):
                T.attr(0, "tensorized_nki_instruction", 1)
                for p_loop, lhs_f_loop, rhs_f_loop in T.grid(128, 128, 128):
                    T.nki_matmul(C_psum[i, lhs_f_loop, rhs_b_loop * 256 + additional_lhs_b_loop * 128 + rhs_f_loop], B_sbuf[p_loop, k * 1024 + reduction_b_loop * 256 + additional_lhs_b_loop * 128 + lhs_f_loop], A_sbuf[p_loop, i * 2048 + rhs_b_loop * 1024 + k * 512 + reduction_b_loop * 128 + rhs_f_loop + 4096], T.bool(True))
    # fmt: on
    with target:
        mod = tvm.IRModule({"main": gemm})
        mod = tvm.tir.transform.LowerTIRp()(mod)
        assert_structural_equal(mod["main"], expected)


def test_gemm_too_large_f_size():
    A_layout = T.TrainiumLayout(
        dimension_types="FP",
        combined_1d_layout=T.TileLayout.from_tuple((256, 128), (1, 1)),
    )
    B_layout = T.TrainiumLayout(
        dimension_types="PF",
        combined_1d_layout=T.TileLayout.from_tuple((128, 1024), (1, 1)),
    )

    C_layout = T.TrainiumPSUMLayout(
        dimension_types="FPF",
        combined_1d_layout=T.TileLayout.from_tuple((2, 128, 1024), (1024, 1, 1)),
    )

    # fmt: off
    @T.prim_func(tirp=True)
    def gemm() -> None:
        with T.kernel():
            A_sbuf = T.alloc_buffer((256, 128), "float32", scope="trn.sbuf", layout=A_layout)
            B_sbuf = T.alloc_buffer((128, 1024), "float32", scope="trn.sbuf", layout=B_layout)
            C_psum = T.alloc_buffer((256, 1024), "float32", scope="trn.psum", layout=C_layout)
            Tp.gemm(C_psum, A_sbuf, B_sbuf, C_psum)
            
    @T.prim_func(tirp=True)
    def expected():
        T.func_attr({"global_symbol": "gemm"})
        with T.kernel():
            A_sbuf = T.alloc_buffer((128, 256), scope="trn.sbuf", logical_scope="kernel")
            B_sbuf = T.alloc_buffer((128, 1024), scope="trn.sbuf", logical_scope="kernel")
            C_psum = T.alloc_buffer((4, 128, 512), scope="trn.psum", logical_scope="kernel")
            for lhs_b_loop, rhs_b_loop, reduction_b_loop, additional_lhs_b_loop, additional_rhs_b_loop in T.grid(1, 1, 1, 2, 2):
                T.attr(0, "tensorized_nki_instruction", 1)
                for p_loop, lhs_f_loop, rhs_f_loop in T.grid(128, 128, 512):
                    T.nki_matmul(C_psum[additional_lhs_b_loop * 2 + additional_rhs_b_loop, lhs_f_loop, rhs_f_loop], A_sbuf[p_loop, additional_lhs_b_loop * 128 + lhs_f_loop], B_sbuf[p_loop, additional_rhs_b_loop * 512 + rhs_f_loop], T.bool(True))
    # fmt: on
    with target:
        mod = tvm.IRModule({"main": gemm})
        mod = tvm.tir.transform.LowerTIRp()(mod)
        assert_structural_equal(mod["main"], expected)


def test_gemm_sbuf_output_with_workspace():
    A_layout = T.TrainiumLayout(
        dimension_types="FFFP",
        combined_1d_layout=T.TileLayout.from_tuple((4, 128, 8, 128), (1024, 1, 128, 1)),
    )
    B_layout = T.TrainiumLayout(
        dimension_types="FPFF",
        combined_1d_layout=T.TileLayout.from_tuple((8, 128, 2, 128), (256, 1, 128, 1)),
    )

    C_layout = T.TrainiumLayout(
        dimension_types="FFFP",
        combined_1d_layout=T.TileLayout.from_tuple((4, 128, 2, 128), (256, 1, 128, 1)),
    )
    # fmt: off
    @T.prim_func(tirp=True)
    def gemm() -> None:
        with T.kernel():
            A_sbuf = T.alloc_buffer((512, 1024), "float32", scope="trn.sbuf", layout=A_layout)
            B_sbuf = T.alloc_buffer((1024, 256), "float32", scope="trn.sbuf", layout=B_layout)
            C_sbuf = T.alloc_buffer((512, 256), "float32", scope="trn.sbuf", layout=C_layout)
            C_psum = T.alloc_buffer((1, 128, 512), "float32", scope="trn.psum", allocated_addr=(0, 0))
            for i in range(2):
                for k in range(2):
                    Tp.gemm(
                        C_sbuf[256 * i : 256 * i + 256, :],
                        A_sbuf[256 * i : 256 * i + 256, 512 * k : 512 * k + 512],
                        B_sbuf[512 * k : 512 * k + 512, :],
                        C_sbuf[256 * i : 256 * i + 256, :],
                        workspace={"acc_psum": C_psum}
                    )
    @T.prim_func(tirp=True)
    def expected():
        T.func_attr({"global_symbol": "gemm"})
        with T.kernel():
            A_sbuf = T.alloc_buffer((128, 4096), scope="trn.sbuf", logical_scope="kernel")
            B_sbuf = T.alloc_buffer((128, 2048), scope="trn.sbuf", logical_scope="kernel")
            C_sbuf = T.alloc_buffer((128, 1024), scope="trn.sbuf", logical_scope="kernel")
            C_psum = T.alloc_buffer((1, 128, 512), scope="trn.psum", logical_scope="kernel", allocated_addr=[0, 0])
            for i, k, lhs_b_loop, rhs_b_loop, additional_lhs_b_loop, additional_rhs_b_loop in T.grid(2, 2, 1, 2, 2, 1):
                for reduction_b_loop in range(4):
                    T.attr(0, "tensorized_nki_instruction", 1)
                    for p_loop, lhs_f_loop, rhs_f_loop in T.grid(128, 128, 128):
                        T.nki_matmul(C_psum[0, lhs_f_loop, additional_lhs_b_loop * 256 + rhs_b_loop * 128 + rhs_f_loop], B_sbuf[p_loop, k * 1024 + reduction_b_loop * 256 + additional_lhs_b_loop * 128 + lhs_f_loop], A_sbuf[p_loop, i * 2048 + rhs_b_loop * 1024 + k * 512 + reduction_b_loop * 128 + rhs_f_loop], T.bool(True))
                T.attr(0, "tensorized_nki_instruction", 1)
                for lhs_f_loop, rhs_f_loop in T.grid(128, 128):
                    T.nki_tensor_copy(C_sbuf[lhs_f_loop, i * 512 + rhs_b_loop * 256 + additional_lhs_b_loop * 128 + rhs_f_loop], C_psum[0, lhs_f_loop, additional_lhs_b_loop * 256 + rhs_b_loop * 128 + rhs_f_loop])
    # fmt: on
    with target:
        mod = tvm.IRModule({"main": gemm})
        mod = tvm.tir.transform.LowerTIRp()(mod)
        mod = tvm.tir.transform.Simplify()(mod)
        assert_structural_equal(mod["main"], expected)
        
def test_gemm_pf_mismatch_fail():
    A_layout = T.TrainiumLayout(
        dimension_types="FFFP",
        combined_1d_layout=T.TileLayout.from_tuple((4, 128, 8, 128), (1024, 1, 128, 1)),
    )
    B_layout = T.TrainiumLayout(
        dimension_types="FFFP",
        combined_1d_layout=T.TileLayout.from_tuple(( 2, 128, 8, 128), (128, 1, 256, 1)),
    )

    C_layout = T.TrainiumPSUMLayout(
        dimension_types="FPFF",
        combined_1d_layout=T.TileLayout.from_tuple((4, 128, 2, 128), (256, 1, 128, 1)),
    )
    # fmt: off
    @T.prim_func(tirp=True)
    def gemm() -> None:
        with T.kernel():
            A_sbuf = T.alloc_buffer((512, 1024), "float32", scope="trn.sbuf", layout=A_layout)
            B_sbuf = T.alloc_buffer((256, 1024), "float32", scope="trn.sbuf", layout=B_layout)
            C_psum = T.alloc_buffer((512, 256), "float32", scope="trn.psum", layout=C_layout)
            for i in range(2):
                for k in range(2):
                    Tp.gemm(
                        C_psum[256 * i : 256 * i + 256, :],
                        A_sbuf[256 * i : 256 * i + 256, 512 * k : 512 * k + 512],
                        B_sbuf[:, 512 * k : 512 * k + 512],
                        C_psum[256 * i : 256 * i + 256, :],
                    )
    # fmt: on
    with pytest.raises(Exception):
        with target:
            mod = tvm.IRModule({"main": gemm})
            mod = tvm.tir.transform.LowerTIRp()(mod)
    

def test_gemm_transpose_AB():
    A_layout = T.TrainiumLayout(
        dimension_types="FPFF",
        combined_1d_layout=T.TileLayout.from_tuple((8, 128, 4, 128), (128, 1, 1024, 1)),
    )
    B_layout = T.TrainiumLayout(
        dimension_types="FFFP",
        combined_1d_layout=T.TileLayout.from_tuple(( 2, 128, 8, 128), (128, 1, 256, 1)),
    )

    C_layout = T.TrainiumPSUMLayout(
        dimension_types="FPFF",
        combined_1d_layout=T.TileLayout.from_tuple((4, 128, 2, 128), (256, 1, 128, 1)),
    )
    # fmt: off
    @T.prim_func(tirp=True)
    def gemm() -> None:
        with T.kernel():
            A_sbuf = T.alloc_buffer((1024, 512), "float32", scope="trn.sbuf", layout=A_layout)
            B_sbuf = T.alloc_buffer((256, 1024), "float32", scope="trn.sbuf", layout=B_layout)
            C_psum = T.alloc_buffer((512, 256), "float32", scope="trn.psum", layout=C_layout)
            for i in range(2):
                for k in range(2):
                    Tp.gemm(
                        C_psum[256 * i : 256 * i + 256, :],
                        A_sbuf[512 * k : 512 * k + 512, 256 * i : 256 * i + 256],
                        B_sbuf[:, 512 * k : 512 * k + 512],
                        C_psum[256 * i : 256 * i + 256, :],
                        transpose_A=True,
                        transpose_B=True,
                    )
    
    @T.prim_func(tirp=True)
    def expected():
        T.func_attr({"global_symbol": "gemm"})
        with T.kernel():
            A_sbuf = T.alloc_buffer((128, 4096), scope="trn.sbuf", logical_scope="kernel")
            B_sbuf = T.alloc_buffer((128, 2048), scope="trn.sbuf", logical_scope="kernel")
            C_psum = T.alloc_buffer((2, 128, 512), scope="trn.psum", logical_scope="kernel")
            for i, k, lhs_b_loop, rhs_b_loop, reduction_b_loop, additional_lhs_b_loop, additional_rhs_b_loop in T.grid(2, 2, 2, 1, 4, 1, 1):
                T.attr(0, "tensorized_nki_instruction", 1)
                for p_loop, lhs_f_loop, rhs_f_loop in T.grid(128, 128, 256):
                    T.nki_matmul(C_psum[i, lhs_f_loop, lhs_b_loop * 256 + rhs_f_loop], A_sbuf[p_loop, i * 2048 + lhs_b_loop * 1024 + k * 512 + reduction_b_loop * 128 + lhs_f_loop], B_sbuf[p_loop, k * 1024 + reduction_b_loop * 256 + rhs_f_loop], T.bool(True))
 
    #fmt: off
    with target:
        mod = tvm.IRModule({"main": gemm})
        mod = tvm.tir.transform.LowerTIRp()(mod)
        assert_structural_equal(mod["main"], expected)


if __name__ == "__main__":
    tvm.testing.main()
