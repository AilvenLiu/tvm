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
    src_layout = T.TileLayout.from_nested_tuple((128, 512), (512, 1))
    dst_shape = [128, 512]
    dst_layout = TrainiumLayout(
        dimension_types="PF", combined_1d_layout=T.TileLayout.from_nested_tuple((128, 512), (1, 1))
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
            layout=T.TileLayout.from_nested_tuple(data=(128, 512), strides=(512, 1)),
        )
        with T.kernel():
            A_sbuf = T.alloc_buffer((128, 512), scope="trn.sbuf", logical_scope="kernel")
            for b_loop in range(1):
                T.attr(0, "tensorized_nki_instruction", 1)
                for p_loop, f_loop in T.grid(128, 512):
                    A_1 = T.Buffer((65536,), data=A.data, logical_scope="kernel")
                    A_sbuf[p_loop, f_loop] = A_1[p_loop * 512 + f_loop]

    with target:
        mod = tvm.IRModule({"main": copy})
        mod = tvm.tir.transform.LowerTIRp()(mod)
        assert_structural_equal(mod["main"], expected)


def test_simple_copy_2():
    src_shape = [128, 512]
    src_layout = T.TileLayout.from_nested_tuple((128, 4, 128), (512, 128, 1))

    dst_shape = [128, 512]
    dst_layout = TrainiumLayout(
        dimension_types="FFP",
        combined_1d_layout=T.TileLayout.from_nested_tuple((128, 4, 128), (4, 1, 1)),
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
            layout=T.TileLayout.from_nested_tuple(data=(128, 4, 128), strides=(512, 128, 1)),
        )
        with T.kernel():
            A_sbuf = T.alloc_buffer((128, 512), scope="trn.sbuf", logical_scope="kernel")
            for b_loop in range(512):
                T.attr(0, "tensorized_nki_instruction", 1)
                for p_loop, f_loop in T.grid(128, 1):
                    A_1 = T.Buffer((65536,), data=A.data, logical_scope="kernel")
                    A_sbuf[p_loop, b_loop] = A_1[b_loop * 128 + p_loop]

    with target:
        mod = tvm.IRModule({"main": copy})
        mod = tvm.tir.transform.LowerTIRp()(mod)
        assert_structural_equal(mod["main"], expected)


def test_copy_in_a_loop():
    src_shape = [512, 512]
    src_layout = T.TileLayout.from_nested_tuple((4, 128, 512), (512 * 128, 512, 1))
    dst_shape = [512, 512]
    dst_layout = TrainiumLayout(
        dimension_types="FPF",
        combined_1d_layout=T.TileLayout.from_nested_tuple((4, 128, 512), (512, 1, 1)),
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
            layout=T.TileLayout.from_nested_tuple(data=(4, 128, 512), strides=(65536, 512, 1)),
        )
        with T.kernel():
            A_sbuf = T.alloc_buffer((128, 2048), scope="trn.sbuf", logical_scope="kernel")
            for i, b_loop in T.grid(4, 1):
                T.attr(0, "tensorized_nki_instruction", 1)
                for p_loop, f_loop in T.grid(128, 512):
                    A_1 = T.Buffer((262144,), data=A.data, logical_scope="kernel")
                    A_sbuf[p_loop, i * 512 + f_loop] = A_1[i * 65536 + p_loop * 512 + f_loop]

    with target:
        mod = tvm.IRModule({"main": copy})
        mod = tvm.tir.transform.LowerTIRp()(mod)
        assert_structural_equal(mod["main"], expected)


def test_copy_in_a_loop_2():
    src_shape = [512, 512]
    src_layout = T.TileLayout.from_nested_tuple((128, 2048), (2048, 1))
    dst_shape = [512, 512]
    dst_layout = TrainiumLayout(
        dimension_types="PF", combined_1d_layout=T.TileLayout.from_nested_tuple((128, 2048), (1, 1))
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
            layout=T.TileLayout.from_nested_tuple(data=(128, 2048), strides=(2048, 1)),
        )
        with T.kernel():
            A_sbuf = T.alloc_buffer((128, 2048), scope="trn.sbuf", logical_scope="kernel")
            for i, b_loop in T.grid(4, 1):
                T.attr(0, "tensorized_nki_instruction", 1)
                for p_loop, f_loop in T.grid(128, 512):
                    A_1 = T.Buffer((262144,), data=A.data, logical_scope="kernel")
                    A_sbuf[p_loop, i * 512 + f_loop] = A_1[p_loop * 2048 + i * 512 + f_loop]

    with target:
        mod = tvm.IRModule({"main": copy})
        mod = tvm.tir.transform.LowerTIRp()(mod)
        assert_structural_equal(mod["main"], expected)


# transpose is not a copy
def test_copy_transpose_fail():
    src_shape = [512, 512]
    src_layout = TrainiumLayout(
        dimension_types="PF", combined_1d_layout=T.TileLayout.from_nested_tuple((128, 2048), (1, 1))
    )
    dst_shape = [512, 512]
    dst_layout = TrainiumLayout(
        dimension_types="FP", combined_1d_layout=T.TileLayout.from_nested_tuple((2048, 128), (1, 1))
    )

    @T.prim_func(tirp=True)
    def copy() -> None:
        with T.kernel():
            A_sbuf = T.alloc_buffer(src_shape, "float32", scope="trn.sbuf", layout=src_layout)
            B_sbuf = T.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
            Tp.copy(B_sbuf, A_sbuf)

    with pytest.raises(Exception):
        with target:
            mod = tvm.IRModule({"main": copy})
            mod = tvm.tir.transform.LowerTIRp()(mod)


def test_copy_different_f():
    src_shape = [512, 64]
    src_layout = TrainiumLayout(
        dimension_types="FPFFF",
        combined_1d_layout=T.TileLayout.from_nested_tuple((4, 128, 4, 4, 4), (64, 1, 16, 4, 1)),
    )
    dst_shape = [512, 64]
    dst_layout = TrainiumLayout(
        dimension_types="FPFFF",
        combined_1d_layout=T.TileLayout.from_nested_tuple((4, 128, 4, 4, 4), (64, 1, 4, 16, 1)),
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
            for b_loop in range(64):
                T.attr(0, "tensorized_nki_instruction", 1)
                for p_loop, f_loop in T.grid(128, 4):
                    B_sbuf[
                        p_loop, b_loop // 16 * 64 + b_loop % 4 * 16 + b_loop % 16 // 4 * 4 + f_loop
                    ] = A_sbuf[p_loop, b_loop * 4 + f_loop]

    with target:
        mod = tvm.IRModule({"main": copy})
        mod = tvm.tir.transform.LowerTIRp()(mod)
        assert_structural_equal(mod["main"], expected)


def test_copy_different_shape():
    src_shape = [512, 64]
    src_layout = TrainiumLayout(
        dimension_types="FPFFF",
        combined_1d_layout=T.TileLayout.from_nested_tuple((4, 128, 4, 4, 4), (64, 1, 16, 4, 1)),
    )
    dst_shape = [4, 128, 4]
    dst_layout = TrainiumLayout(
        dimension_types="FPF",
        combined_1d_layout=T.TileLayout.from_nested_tuple((4, 128, 4), (4, 1, 1)),
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
            for b_loop in range(4):
                T.attr(0, "tensorized_nki_instruction", 1)
                for p_loop, f_loop in T.grid(128, 4):
                    B_sbuf[p_loop, b_loop * 4 + f_loop] = A_sbuf[p_loop, b_loop * 64 + f_loop]

    with target:
        mod = tvm.IRModule({"main": copy})
        mod = tvm.tir.transform.LowerTIRp()(mod)
        assert_structural_equal(mod["main"], expected)


def test_copy_irregular_shape():
    src_shape = [128, 10000]
    src_layout = T.TileLayout.from_nested_tuple((128, 10000), (10000, 1))
    dst_shape = [128, 512]
    dst_layout = TrainiumLayout(
        dimension_types="PF", combined_1d_layout=T.TileLayout.from_nested_tuple((128, 512), (1, 1))
    )

    @T.prim_func(tirp=True)
    def copy(A_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, src_shape, "float32", layout=src_layout)
        with T.kernel():
            A_sbuf = T.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
            for i in range(4):
                Tp.copy(A_sbuf, A[:, i * 512 : i * 512 + 512])

    @T.prim_func(tirp=True)
    def expected(A_ptr: T.handle):
        T.func_attr({"global_symbol": "copy"})
        A = T.match_buffer(
            A_ptr,
            (128, 10000),
            logical_scope="kernel",
            layout=T.TileLayout.from_nested_tuple(data=(128, 10000), strides=(10000, 1)),
        )
        with T.kernel():
            A_sbuf = T.alloc_buffer((128, 512), scope="trn.sbuf", logical_scope="kernel")
            for i, b_loop in T.grid(4, 1):
                T.attr(0, "tensorized_nki_instruction", 1)
                for p_loop, f_loop in T.grid(128, 512):
                    A_1 = T.Buffer((1280000,), data=A.data, logical_scope="kernel")
                    A_sbuf[p_loop, f_loop] = A_1[p_loop * 10000 + i * 512 + f_loop]

    with target:
        mod = tvm.IRModule({"main": copy})
        mod = tvm.tir.transform.LowerTIRp()(mod)
        assert_structural_equal(mod["main"], expected)


if __name__ == "__main__":
    tvm.testing.main()
