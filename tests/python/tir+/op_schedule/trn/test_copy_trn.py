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


def test_simple_copy():
    src_shape = [128, 512]
    src_layout = T.TileLayout.from_tuple((128, 512), (512, 1))
    dst_shape = [128, 512]
    dst_layout = TrainiumLayout(
        dimension_types="PF", combined_1d_layout=T.TileLayout.from_tuple((128, 512), (1, 1))
    )

    @T.prim_func(tirp=True)
    def copy(A_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, src_shape, "float32", layout=src_layout)
        with T.kernel():
            A_sbuf = T.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
            Tp.copy(A_sbuf, A)

    @T.prim_func(tirp=True)
    def expected(A_ptr: T.handle):
        T.func_attr({"global_symbol": "copy"})
        A = T.match_buffer(
            A_ptr,
            (128, 512),
            logical_scope="kernel",
            layout=T.TileLayout.from_tuple(data=(128, 512), strides=(512, 1)),
        )
        with T.kernel():
            A_sbuf = T.alloc_buffer((128, 512), scope="trn.sbuf", logical_scope="kernel")
            for b_loop, additional_b_loop in T.grid(1, 1):
                T.attr(0, "tensorized_nki_instruction", 1)
                for p_loop, f_loop in T.grid(128, 512):
                    A_1 = T.Buffer((65536,), data=A.data, logical_scope="kernel")
                    T.nki_load(A_sbuf[p_loop, f_loop], A_1[p_loop * 512 + f_loop])

    with target:
        mod = tvm.IRModule({"main": copy})
        mod = tvm.tir.transform.LowerTIRp()(mod)
        assert_structural_equal(mod["main"], expected)


def test_simple_copy_2():
    src_shape = [128, 512]
    src_layout = T.TileLayout.from_tuple((128, 4, 128), (512, 128, 1))

    dst_shape = [128, 512]
    dst_layout = TrainiumLayout(
        dimension_types="FFP",
        combined_1d_layout=T.TileLayout.from_tuple((128, 4, 128), (4, 1, 1)),
    )

    @T.prim_func(tirp=True)
    def copy(A_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, src_shape, "float32", layout=src_layout)
        with T.kernel():
            A_sbuf = T.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
            Tp.copy(A_sbuf, A)

    @T.prim_func(tirp=True)
    def expected(A_ptr: T.handle):
        T.func_attr({"global_symbol": "copy"})
        A = T.match_buffer(
            A_ptr,
            (128, 512),
            logical_scope="kernel",
            layout=T.TileLayout.from_tuple(data=(128, 4, 128), strides=(512, 128, 1)),
        )
        with T.kernel():
            A_sbuf = T.alloc_buffer((128, 512), scope="trn.sbuf", logical_scope="kernel")
            for b_loop, additional_b_loop in T.grid(512, 1):
                T.attr(0, "tensorized_nki_instruction", 1)
                for p_loop, f_loop in T.grid(128, 1):
                    A_1 = T.Buffer((65536,), data=A.data, logical_scope="kernel")
                    T.nki_load(A_sbuf[p_loop, b_loop], A_1[b_loop * 128 + p_loop])

    with target:
        mod = tvm.IRModule({"main": copy})
        mod = tvm.tir.transform.LowerTIRp()(mod)
        assert_structural_equal(mod["main"], expected)


def test_copy_in_a_loop():
    src_shape = [512, 512]
    src_layout = T.TileLayout.from_tuple((4, 128, 512), (512 * 128, 512, 1))
    dst_shape = [512, 512]
    dst_layout = TrainiumLayout(
        dimension_types="FPF",
        combined_1d_layout=T.TileLayout.from_tuple((4, 128, 512), (512, 1, 1)),
    )

    @T.prim_func(tirp=True)
    def copy(A_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, src_shape, "float32", layout=src_layout)
        with T.kernel():
            A_sbuf = T.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
            for i in range(4):
                Tp.copy(A_sbuf[i * 128 : i * 128 + 128, :], A[i * 128 : i * 128 + 128, :])

    @T.prim_func(tirp=True)
    def expected(A_ptr: T.handle):
        T.func_attr({"global_symbol": "copy"})
        A = T.match_buffer(
            A_ptr,
            (512, 512),
            logical_scope="kernel",
            layout=T.TileLayout.from_tuple(data=(4, 128, 512), strides=(65536, 512, 1)),
        )
        with T.kernel():
            A_sbuf = T.alloc_buffer((128, 2048), scope="trn.sbuf", logical_scope="kernel")
            for i, b_loop, additional_b_loop in T.grid(4, 1, 1):
                T.attr(0, "tensorized_nki_instruction", 1)
                for p_loop, f_loop in T.grid(128, 512):
                    A_1 = T.Buffer((262144,), data=A.data, logical_scope="kernel")
                    T.nki_load(A_sbuf[p_loop, i * 512 + f_loop], A_1[i * 65536 + p_loop * 512 + f_loop])

    with target:
        mod = tvm.IRModule({"main": copy})
        mod = tvm.tir.transform.LowerTIRp()(mod)
        assert_structural_equal(mod["main"], expected)


def test_copy_in_a_loop_2():
    src_shape = [512, 512]
    src_layout = T.TileLayout.from_tuple((128, 2048), (2048, 1))
    dst_shape = [512, 512]
    dst_layout = TrainiumLayout(
        dimension_types="PF", combined_1d_layout=T.TileLayout.from_tuple((128, 2048), (1, 1))
    )

    @T.prim_func(tirp=True)
    def copy(A_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, src_shape, "float32", layout=src_layout)
        with T.kernel():
            A_sbuf = T.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
            A_sbuf_view = T.view(A_sbuf, A_sbuf.layout, (128, 4, 512))
            A_view = T.view(A, A.layout, (128, 4, 512))
            for i in range(4):
                Tp.copy(A_sbuf_view[:, i, :], A_view[:, i, :])

    @T.prim_func(tirp=True)
    def expected(A_ptr: T.handle):
        T.func_attr({"global_symbol": "copy"})
        A = T.match_buffer(
            A_ptr,
            (512, 512),
            logical_scope="kernel",
            layout=T.TileLayout.from_tuple(data=(128, 2048), strides=(2048, 1)),
        )
        with T.kernel():
            A_sbuf = T.alloc_buffer((128, 2048), scope="trn.sbuf", logical_scope="kernel")
            for i, b_loop, additional_b_loop in T.grid(4, 1, 1):
                T.attr(0, "tensorized_nki_instruction", 1)
                for p_loop, f_loop in T.grid(128, 512):
                    A_1 = T.Buffer((262144,), data=A.data, logical_scope="kernel")
                    T.nki_load(A_sbuf[p_loop, i * 512 + f_loop], A_1[p_loop * 2048 + i * 512 + f_loop])

    with target:
        mod = tvm.IRModule({"main": copy})
        mod = tvm.tir.transform.LowerTIRp()(mod)
        assert_structural_equal(mod["main"], expected)


# transpose is not a copy
def test_copy_transpose():
    src_shape = [512, 512]
    src_layout = TrainiumLayout(
        dimension_types="PF", combined_1d_layout=T.TileLayout.from_tuple((128, 2048), (1, 1))
    )
    dst_shape = [512, 512]
    dst_layout = TrainiumLayout(
        dimension_types="FP", combined_1d_layout=T.TileLayout.from_tuple((2048, 128), (1, 1))
    )

    #fmt: off
    @T.prim_func(tirp=True)
    def copy() -> None:
        with T.kernel():
            A_sbuf = T.alloc_buffer(src_shape, "float32", scope="trn.sbuf", layout=src_layout)
            B_sbuf = T.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
            Tp.copy(B_sbuf, A_sbuf)
            
    @T.prim_func(tirp=True)
    def expected():
        T.func_attr({"global_symbol": "copy"})
        with T.kernel():
            A_sbuf = T.alloc_buffer((128, 2048), scope="trn.sbuf", logical_scope="kernel")
            B_sbuf = T.alloc_buffer((128, 2048), scope="trn.sbuf", logical_scope="kernel")
            buffer = T.alloc_buffer((128, 128), scope="trn.sbuf", logical_scope="kernel")
            buffer_1 = T.alloc_buffer((8, 128, 512), scope="trn.psum", logical_scope="kernel", allocated_addr=[0, 0])
            with T.attr(0, "tensorized_nki_instruction", 1):
                for p_loop, rhs_f_loop in T.grid(128, 128):
                    T.nki_identity(buffer[p_loop, rhs_f_loop], 128)
            with T.kernel():
                for b_loop in range(16):
                    with T.attr(0, "tensorized_nki_instruction", 1):
                        for p_loop, lhs_f_loop, rhs_f_loop in T.grid(128, 128, 128):
                            T.nki_matmul(buffer_1[b_loop // 4, lhs_f_loop, b_loop % 4 * 128 + rhs_f_loop], A_sbuf[p_loop, b_loop * 128 + lhs_f_loop], buffer[p_loop, rhs_f_loop], T.bool(True))
                    T.attr(0, "tensorized_nki_instruction", 1)
                    for p_loop, f_loop in T.grid(128, 128):
                        T.nki_tensor_copy(B_sbuf[p_loop, f_loop * 16 + b_loop], buffer_1[b_loop // 4, p_loop, b_loop % 4 * 128 + f_loop])    
    #fmt: on

    with target:
        mod = tvm.IRModule({"main": copy})
        mod = tvm.tir.transform.LowerTIRp()(mod)
        mod = tvm.tir.transform.Simplify()(mod)
        assert_structural_equal(mod["main"], expected)

def test_copy_transpose_2():
    src_shape = [65536]
    src_layout = TrainiumLayout(
        dimension_types="PF", combined_1d_layout=T.TileLayout.from_tuple((128, 512), (1, 1))
    )
    dst_shape = [4, 65536]
    dst_layout = TrainiumLayout(
        dimension_types="FFPF", combined_1d_layout=T.TileLayout.from_tuple((4, 128, 128, 4), (4,16,1,1))
    )
    #fmt: off
    @T.prim_func(tirp=True)
    def copy() -> None:
        with T.kernel():
            A_sbuf = T.alloc_buffer(src_shape, "float32", scope="trn.sbuf", layout=src_layout)
            B_sbuf = T.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
            for i in range(4):
                Tp.copy(B_sbuf[i, :], A_sbuf)
                
    @T.prim_func(tirp=True)
    def expected():
        T.func_attr({"global_symbol": "copy"})
        with T.kernel():
            A_sbuf = T.alloc_buffer((128, 512), scope="trn.sbuf", logical_scope="kernel")
            B_sbuf = T.alloc_buffer((128, 2048), scope="trn.sbuf", logical_scope="kernel")
            buffer = T.alloc_buffer((128, 128), scope="trn.sbuf", logical_scope="kernel")
            dst_psum = T.alloc_buffer((8, 128, 512), scope="trn.psum", logical_scope="kernel", allocated_addr=[0, 0])
            with T.attr(0, "tensorized_nki_instruction", 1):
                for p_loop, rhs_f_loop in T.grid(128, 128):
                    T.nki_identity(buffer[p_loop, rhs_f_loop], 128)
            for i in range(4):
                with T.kernel():
                    for b_loop in range(4):
                        with T.attr(0, "tensorized_nki_instruction", 1):
                            for p_loop, lhs_f_loop, rhs_f_loop in T.grid(128, 128, 128):
                                T.nki_matmul(dst_psum[0, lhs_f_loop, b_loop * 128 + rhs_f_loop], A_sbuf[p_loop, lhs_f_loop * 4 + b_loop], buffer[p_loop, rhs_f_loop], T.bool(True))
                        T.attr(0, "tensorized_nki_instruction", 1)
                        for p_loop, f_loop in T.grid(128, 128):
                            T.nki_tensor_copy(B_sbuf[p_loop, f_loop * 16 + i * 4 + b_loop], dst_psum[0, p_loop, b_loop * 128 + f_loop])
    #fmt: on
    with target:
        mod = tvm.IRModule({"main": copy})
        mod = tvm.tir.transform.LowerTIRp()(mod)
        mod = tvm.tir.transform.Simplify()(mod)
        assert_structural_equal(mod["main"], expected)

def test_copy_different_f():
    src_shape = [512, 64]
    src_layout = TrainiumLayout(
        dimension_types="FPFFF",
        combined_1d_layout=T.TileLayout.from_tuple((4, 128, 4, 4, 4), (64, 1, 16, 4, 1)),
    )
    dst_shape = [512, 64]
    dst_layout = TrainiumLayout(
        dimension_types="FPFFF",
        combined_1d_layout=T.TileLayout.from_tuple((4, 128, 4, 4, 4), (64, 1, 4, 16, 1)),
    )

    @T.prim_func(tirp=True)
    def copy() -> None:
        with T.kernel():
            A_sbuf = T.alloc_buffer(src_shape, "float32", scope="trn.sbuf", layout=src_layout)
            B_sbuf = T.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
            Tp.copy(B_sbuf, A_sbuf)

    @T.prim_func(tirp=True)
    def expected():
        T.func_attr({"global_symbol": "copy"})
        with T.kernel():
            A_sbuf = T.alloc_buffer((128, 256), scope="trn.sbuf", logical_scope="kernel")
            B_sbuf = T.alloc_buffer((128, 256), scope="trn.sbuf", logical_scope="kernel")
            for b_loop, additional_b_loop in T.grid(64, 1):
                T.attr(0, "tensorized_nki_instruction", 1)
                for p_loop, f_loop in T.grid(128, 4):
                    T.nki_tensor_copy(B_sbuf[
                        p_loop, b_loop // 16 * 64 + b_loop % 4 * 16 + b_loop % 16 // 4 * 4 + f_loop
                    ], A_sbuf[p_loop, b_loop * 4 + f_loop])

    with target:
        mod = tvm.IRModule({"main": copy})
        mod = tvm.tir.transform.LowerTIRp()(mod)
        assert_structural_equal(mod["main"], expected)


def test_copy_different_shape():
    src_shape = [512, 64]
    src_layout = TrainiumLayout(
        dimension_types="FPFFF",
        combined_1d_layout=T.TileLayout.from_tuple((4, 128, 4, 4, 4), (64, 1, 16, 4, 1)),
    )
    dst_shape = [4, 128, 4]
    dst_layout = TrainiumLayout(
        dimension_types="FPF",
        combined_1d_layout=T.TileLayout.from_tuple((4, 128, 4), (4, 1, 1)),
    )

    @T.prim_func(tirp=True)
    def copy() -> None:
        with T.kernel():
            A_sbuf = T.alloc_buffer(src_shape, "float32", scope="trn.sbuf", layout=src_layout)
            B_sbuf = T.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
            B_sbuf_view = T.view(B_sbuf, B_sbuf.layout, (512, 4))
            Tp.copy(B_sbuf_view, A_sbuf[:, 0:4])

    @T.prim_func(tirp=True)
    def expected():
        T.func_attr({"global_symbol": "copy"})
        with T.kernel():
            A_sbuf = T.alloc_buffer((128, 256), scope="trn.sbuf", logical_scope="kernel")
            B_sbuf = T.alloc_buffer((128, 16), scope="trn.sbuf", logical_scope="kernel")
            for b_loop, additional_b_loop in T.grid(4, 1):
                T.attr(0, "tensorized_nki_instruction", 1)
                for p_loop, f_loop in T.grid(128, 4):
                    T.nki_tensor_copy(B_sbuf[p_loop, b_loop * 4 + f_loop], A_sbuf[p_loop, b_loop * 64 + f_loop])

    with target:
        mod = tvm.IRModule({"main": copy})
        mod = tvm.tir.transform.LowerTIRp()(mod)
        assert_structural_equal(mod["main"], expected)


def test_copy_irregular_shape():
    src_shape = [128, 10000]
    src_layout = T.TileLayout.from_tuple((128, 10000), (10000, 1))
    dst_shape = [128, 512]
    dst_layout = TrainiumLayout(
        dimension_types="PF", combined_1d_layout=T.TileLayout.from_tuple((128, 512), (1, 1))
    )

    @T.prim_func(tirp=True)
    def copy(A_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, src_shape, "float32", layout=src_layout)
        with T.kernel():
            A_sbuf = T.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
            for i in range(4):
                Tp.copy(A[:, i * 512 : i * 512 + 512], A_sbuf)

    @T.prim_func(tirp=True)
    def expected(A_ptr: T.handle):
        T.func_attr({"global_symbol": "copy"})
        A = T.match_buffer(
            A_ptr,
            (128, 10000),
            logical_scope="kernel",
            layout=T.TileLayout.from_tuple(data=(128, 10000), strides=(10000, 1)),
        )
        with T.kernel():
            A_sbuf = T.alloc_buffer((128, 512), scope="trn.sbuf", logical_scope="kernel")
            for i, b_loop, additional_b_loop in T.grid(4, 1, 1):
                T.attr(0, "tensorized_nki_instruction", 1)
                for p_loop, f_loop in T.grid(128, 512):
                    A_1 = T.Buffer((1280000,), data=A.data, logical_scope="kernel")
                    T.nki_store(A_1[p_loop * 10000 + i * 512 + f_loop], A_sbuf[p_loop, f_loop])

    with target:
        mod = tvm.IRModule({"main": copy})
        mod = tvm.tir.transform.LowerTIRp()(mod)
        assert_structural_equal(mod["main"], expected)


def test_copy_different_shape_dim():
    src_shape = [32, 128, 512]
    src_layout = T.TileLayout.from_tuple((32, 128, 512), (128 * 512, 128, 1))
    dst_shape = [128, 512]
    dst_layout = TrainiumLayout(
        dimension_types="PF", combined_1d_layout=T.TileLayout.from_tuple((128, 512), (1, 1))
    )
    # fmt: off
    @T.prim_func(tirp=True)
    def copy(A_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, src_shape, "float32", layout=src_layout)
        with T.kernel():
            A_sbuf = T.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
            for i in range(32):
                Tp.copy(A_sbuf, A[i, :, :])
        
    @T.prim_func(tirp=True)
    def expected(A_ptr: T.handle):
        T.func_attr({"global_symbol": "copy"})
        A = T.match_buffer(A_ptr, (32, 128, 512), logical_scope="kernel", layout=T.TileLayout.from_tuple(data=(32, 128, 512), strides=(65536, 128, 1)))
        with T.kernel():
            A_sbuf = T.alloc_buffer((128, 512), scope="trn.sbuf", logical_scope="kernel")
            for i, b_loop, additional_b_loop in T.grid(32, 1, 1):
                T.attr(0, "tensorized_nki_instruction", 1)
                for p_loop, f_loop in T.grid(128, 512):
                    A_1 = T.Buffer((2097152,), data=A.data, logical_scope="kernel")
                    T.nki_load(A_sbuf[p_loop, f_loop], A_1[i * 65536 + p_loop * 128 + f_loop])        
    # fmt: on
    with target:
        mod = tvm.IRModule({"main": copy})
        mod = tvm.tir.transform.LowerTIRp()(mod)
        assert_structural_equal(mod["main"], expected)

def test_copy_with_offset():
    src_shape = [256, 512]
    src_layout = T.TileLayout.from_tuple((256, 512), (512, 1))
    dst_shape = [512, 512]
    dst_layout = TrainiumLayout(
        dimension_types="FPF", combined_1d_layout=T.TileLayout.from_tuple((4, 128, 512), (512, 1, 1))
    )

    @T.prim_func(tirp=True)
    def copy(A_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, src_shape, "float32", layout=src_layout)
        with T.kernel():
            A_sbuf = T.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
            for i in range(2):
                Tp.copy(A_sbuf[i * 256 : i * 256 + 256, :], A)

    @T.prim_func(tirp=True)
    def expected(A_ptr: T.handle):
        T.func_attr({"global_symbol": "copy"})
        A = T.match_buffer(
            A_ptr,
            (256, 512),
            logical_scope="kernel",
            layout=T.TileLayout.from_tuple(data=(256, 512), strides=(512, 1)),
        )
        with T.kernel():
            A_sbuf = T.alloc_buffer((128, 2048), scope="trn.sbuf", logical_scope="kernel")
            for i, b_loop, additional_b_loop in T.grid(2, 2, 1):
                T.attr(0, "tensorized_nki_instruction", 1)
                for p_loop, f_loop in T.grid(128, 512):
                    A_1 = T.Buffer((131072,), data=A.data, logical_scope="kernel")
                    T.nki_load(
                        A_sbuf[p_loop, i * 1024 + b_loop * 512 + f_loop],
                        A_1[b_loop * 65536 + p_loop * 512 + f_loop],
                    )

    with target:
        mod = tvm.IRModule({"main": copy})
        mod = tvm.tir.transform.LowerTIRp()(mod)
        assert_structural_equal(mod["main"], expected)

def test_large_dma_copy():
    src_shape = [512, 4096]
    src_layout = T.TileLayout.from_tuple((4, 128, 4096), (4096 * 128, 4096, 1))
    dst_shape = [512, 4096]
    dst_layout = TrainiumLayout(
        dimension_types="FPF",
        combined_1d_layout=T.TileLayout.from_tuple((4, 128, 4096), (4096, 1, 1)),
    )

    @T.prim_func(tirp=True)
    def copy(A_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, src_shape, "float32", layout=src_layout)
        with T.kernel():
            A_sbuf = T.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
            for i in range(4):
                Tp.copy(A_sbuf[i * 128 : i * 128 + 128, :], A[i * 128 : i * 128 + 128, :])

    @T.prim_func(tirp=True)
    def expected(A_ptr: T.handle):
        T.func_attr({"global_symbol": "copy"})
        A = T.match_buffer(
            A_ptr,
            (512, 4096),
            logical_scope="kernel",
            layout=T.TileLayout.from_tuple(data=(4, 128, 4096), strides=(524288, 4096, 1)),
        )
        with T.kernel():
            A_sbuf = T.alloc_buffer((128, 16384), scope="trn.sbuf", logical_scope="kernel")
            for i, b_loop, additional_b_loop in T.grid(4, 1, 1):
                T.attr(0, "tensorized_nki_instruction", 1)
                for p_loop, f_loop in T.grid(128, 4096):
                    A_1 = T.Buffer((2097152,), data=A.data, logical_scope="kernel")
                    T.nki_load(
                        A_sbuf[p_loop, i * 4096 + f_loop], A_1[i * 524288 + p_loop * 4096 + f_loop]
                    )

    with target:
        mod = tvm.IRModule({"main": copy})
        mod = tvm.tir.transform.LowerTIRp()(mod)
        assert_structural_equal(mod["main"], expected)

def test_copy_with_inst_size_limit():
    src_shape = [512, 4096]
    src_layout = dst_layout = TrainiumLayout(
        dimension_types="FPF",
        combined_1d_layout=T.TileLayout.from_tuple((4, 128, 4096), (4096, 1, 1)),
    )
    dst_shape = src_shape
    dst_layout = src_layout
    @T.prim_func(tirp=True)
    def copy(A_ptr: T.handle) -> None:
        with T.kernel():
            B_sbuf = T.alloc_buffer(src_shape, "float32", scope="trn.sbuf", layout=src_layout)
            A_sbuf = T.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
            for i in range(4):
                Tp.copy(A_sbuf[i * 128 : i * 128 + 128, :], B_sbuf[i * 128 : i * 128 + 128, :])

    @T.prim_func(tirp=True)
    def expected(A_ptr: T.handle):
        T.func_attr({"global_symbol": "copy"})
        with T.kernel():
            B_sbuf = T.alloc_buffer((128, 16384), scope="trn.sbuf", logical_scope="kernel")
            A_sbuf = T.alloc_buffer((128, 16384), scope="trn.sbuf", logical_scope="kernel")
            for i, b_loop, additional_b_loop in T.grid(4, 1, 8):
                T.attr(0, "tensorized_nki_instruction", 1)
                for p_loop, f_loop in T.grid(128, 512):
                    T.nki_tensor_copy(
                        A_sbuf[p_loop, i * 4096 + additional_b_loop * 512 + f_loop],
                        B_sbuf[p_loop, i * 4096 + additional_b_loop * 512 + f_loop],
                    )

    with target:
        mod = tvm.IRModule({"main": copy})
        mod = tvm.tir.transform.LowerTIRp()(mod)
        assert_structural_equal(mod["main"], expected)

def test_copy_with_complex_index():
    A_shape = [4096, 4096]
    A_layout = T.TileLayout.from_tuple((4096, 4096), (1, 4096))
    A_sbuf_shape = (2, 2048, 1024)
    A_sbuf_layout = T.TrainiumLayout("FFFP", T.TileLayout.from_tuple((2, 2048,8, 128), (16384, 1, 2048, 1)))
    
    # fmt: off
    @T.prim_func(tirp=True)
    def copy(A_ptr: T.handle, ) -> None:
        A = T.match_buffer(A_ptr, A_shape, "float32", layout=A_layout)
        with T.kernel():
            A_sbuf = T.alloc_buffer(A_sbuf_shape, "float32", scope="trn.sbuf", layout=A_sbuf_layout)
            Tp.copy(A_sbuf[1, 0:2048, 0:1024], A[2048: 4096, 3072:4096])
            
    @T.prim_func(tirp=True)
    def expected(A_ptr: T.handle):
        T.func_attr({"global_symbol": "copy"})
        A = T.match_buffer(A_ptr, (4096, 4096), logical_scope="kernel", layout=T.TileLayout.from_tuple(data=(4096, 4096), strides=(1, 4096)))
        with T.kernel():
            A_sbuf = T.alloc_buffer((128, 32768), scope="trn.sbuf", logical_scope="kernel")
            for b_loop, additional_b_loop in T.grid(8, 1):
                T.attr(0, "tensorized_nki_instruction", 1)
                for p_loop, f_loop in T.grid(128, 2048):
                    A_1 = T.Buffer((16777216,), data=A.data, logical_scope="kernel")
                    T.nki_load(A_sbuf[p_loop, b_loop * 2048 + f_loop + 16384], A_1[b_loop * 524288 + p_loop * 4096 + f_loop + 12584960])
    # fmt: on
    with target:
        mod = tvm.IRModule({"main": copy})
        mod = tvm.tir.transform.LowerTIRp()(mod)
        assert_structural_equal(mod["main"], expected)
        
def test_copy_with_complex_index_2():
    A_sbuf_shape = [4096, 4096]
    A_sbuf_layout = T.TrainiumLayout("FFP", T.TileLayout.from_tuple((4096, 32, 128), (1, 4096, 1)))
    A_shape = (2, 2048, 1024)
    A_layout = T.TileLayout.from_tuple((2, 2048,1024), (2048*1024, 1, 2048,))
    
    # fmt: off
    @T.prim_func(tirp=True)
    def copy(A_ptr: T.handle, ) -> None:
        A = T.match_buffer(A_ptr, A_shape, "float32", layout=A_layout)
        with T.kernel():
            A_sbuf = T.alloc_buffer(A_sbuf_shape, "float32", scope="trn.sbuf", layout=A_sbuf_layout)
            Tp.copy(A_sbuf[2048: 4096, 3072:4096], A[1, 0:2048, 0:1024])
            
    @T.prim_func(tirp=True)
    def expected(A_ptr: T.handle):
        T.func_attr({"global_symbol": "copy"})
        A = T.match_buffer(A_ptr, (2, 2048, 1024), logical_scope="kernel", layout=T.TileLayout.from_tuple(data=(2, 2048, 1024), strides=(2097152, 1, 2048)))
        with T.kernel():
            A_sbuf = T.alloc_buffer((128, 131072), scope="trn.sbuf", logical_scope="kernel")
            for b_loop, additional_b_loop in T.grid(8, 1):
                T.attr(0, "tensorized_nki_instruction", 1)
                for p_loop, f_loop in T.grid(128, 2048):
                    A_1 = T.Buffer((4194304,), data=A.data, logical_scope="kernel")
                    T.nki_load(A_sbuf[p_loop, b_loop * 4096 + f_loop + 100352], A_1[b_loop * 262144 + p_loop * 2048 + f_loop + 2097152])
    # fmt: on
    
    with target:
        mod = tvm.IRModule({"main": copy})
        mod = tvm.tir.transform.LowerTIRp()(mod)
        assert_structural_equal(mod["main"], expected)
    
def test_copy_transpose_with_workspace():
    src_shape = [512, 512]
    src_layout = TrainiumLayout(
        dimension_types="PF", combined_1d_layout=T.TileLayout.from_tuple((128, 2048), (1, 1))
    )
    dst_shape = [512, 512]
    dst_layout = TrainiumLayout(
        dimension_types="FP", combined_1d_layout=T.TileLayout.from_tuple((2048, 128), (1, 1))
    )

    #fmt: off
    @T.prim_func(tirp=True)
    def copy() -> None:
        with T.kernel():
            A_sbuf = T.alloc_buffer(src_shape, "float32", scope="trn.sbuf", layout=src_layout)
            B_sbuf = T.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
            identity = T.alloc_buffer((128, 128), "float32", scope="trn.sbuf")
            acc_psum = T.alloc_buffer((1, 128, 512), "float32", scope="trn.psum", allocated_addr=(0, 0))
            with T.attr(0, "tensorized_nki_instruction", 1):
                for p_loop in range(128):
                    for rhs_f_loop in range(128):
                        T.nki_identity(identity[p_loop, rhs_f_loop], 128)
            Tp.copy(B_sbuf, A_sbuf, workspace={"identity": identity, "acc_psum": acc_psum})
    
    @T.prim_func(tirp=True)
    def expected():
        T.func_attr({"global_symbol": "copy"})
        with T.kernel():
            A_sbuf = T.alloc_buffer((128, 2048), scope="trn.sbuf", logical_scope="kernel")
            B_sbuf = T.alloc_buffer((128, 2048), scope="trn.sbuf", logical_scope="kernel")
            identity = T.alloc_buffer((128, 128), scope="trn.sbuf", logical_scope="kernel")
            acc_psum = T.alloc_buffer((1, 128, 512), scope="trn.psum", logical_scope="kernel", allocated_addr=[0, 0])
            with T.attr(0, "tensorized_nki_instruction", 1):
                for p_loop, rhs_f_loop in T.grid(128, 128):
                    T.nki_identity(identity[p_loop, rhs_f_loop], 128)
            with T.kernel():
                for b_loop in range(16):
                    with T.attr(0, "tensorized_nki_instruction", 1):
                        for p_loop, lhs_f_loop, rhs_f_loop in T.grid(128, 128, 128):
                            T.nki_matmul(acc_psum[0, lhs_f_loop, b_loop % 4 * 128 + rhs_f_loop], A_sbuf[p_loop, b_loop * 128 + lhs_f_loop], identity[p_loop, rhs_f_loop], T.bool(True))
                    T.attr(0, "tensorized_nki_instruction", 1)
                    for p_loop, f_loop in T.grid(128, 128):
                        T.nki_tensor_copy(B_sbuf[p_loop, f_loop * 16 + b_loop], acc_psum[0, p_loop, b_loop % 4 * 128 + f_loop])
    #fmt: on
    with target:
        mod = tvm.IRModule({"main": copy})
        mod = tvm.tir.transform.LowerTIRp()(mod)
        assert_structural_equal(mod["main"], expected)
    
if __name__ == "__main__":
    tvm.testing.main()
