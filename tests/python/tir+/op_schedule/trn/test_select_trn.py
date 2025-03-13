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


def test_select():
    src_shape = [128, 512]
    src_layout = TrainiumLayout(
        dimension_types="PF", combined_1d_layout=T.TileLayout.from_tuple((128, 512), (1, 1))
    )
    dst_shape = [128, 512]
    dst_layout = TrainiumLayout(
        dimension_types="PF", combined_1d_layout=T.TileLayout.from_tuple((128, 512), (1, 1))
    )

    # fmt: off
    @T.prim_func(tirp=True)
    def select() -> None:
        with T.kernel():
            A_sbuf = T.alloc_buffer(src_shape, "float32", scope="trn.sbuf", layout=src_layout)
            B_sbuf = T.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
            Tp.select(B_sbuf, A_sbuf, 0.0, lambda i, j: i < j)
            
    @T.prim_func(tirp=True)
    def expected():
        T.func_attr({"global_symbol": "select"})
        with T.kernel():
            A_sbuf = T.alloc_buffer((128, 512), scope="trn.sbuf", logical_scope="kernel")
            B_sbuf = T.alloc_buffer((128, 512), scope="trn.sbuf", logical_scope="kernel")
            for b_loop, additional_b_loop in T.grid(1, 1):
                T.attr(0, "tensorized_nki_instruction", 1)
                for p_loop, f_loop in T.grid(128, 512):
                    T.nki_affine_select(B_sbuf[p_loop, f_loop], p_loop < f_loop, A_sbuf[p_loop, f_loop], T.float32(0.0))
    # fmt: on

    with target:
        mod = tvm.IRModule({"main": select})
        mod = tvm.tir.transform.LowerTIRp()(mod)
        mod = tvm.tir.transform.Simplify()(mod)
        assert_structural_equal(mod["main"], expected)


def test_select_in_loop():
    src_shape = [32, 128, 512]
    src_layout = TrainiumLayout(
        dimension_types="FPF",
        combined_1d_layout=T.TileLayout.from_tuple((32, 128, 512), (512, 1, 1)),
    )
    dst_shape = [128, 512]
    dst_layout = TrainiumLayout(
        dimension_types="PF", combined_1d_layout=T.TileLayout.from_tuple((128, 512), (1, 1))
    )

    # fmt: off
    @T.prim_func(tirp=True)
    def select() -> None:
        with T.kernel():
            A_sbuf = T.alloc_buffer(src_shape, "float32", scope="trn.sbuf", layout=src_layout)
            B_sbuf = T.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
            for i in range(2):
                Tp.select(B_sbuf, A_sbuf[i*16, :, :], 0.0, lambda a, b: (i+1)* a < b)
    
    @T.prim_func(tirp=True)
    def expected():
        T.func_attr({"global_symbol": "select"})
        with T.kernel():
            A_sbuf = T.alloc_buffer((128, 16384), scope="trn.sbuf", logical_scope="kernel")
            B_sbuf = T.alloc_buffer((128, 512), scope="trn.sbuf", logical_scope="kernel")
            for i, b_loop, additional_b_loop in T.grid(2, 1, 1):
                T.attr(0, "tensorized_nki_instruction", 1)
                for p_loop, f_loop in T.grid(128, 512):
                    T.nki_affine_select(B_sbuf[p_loop, f_loop], (i + 1) * p_loop < f_loop, A_sbuf[p_loop, i * 8192 + f_loop], T.float32(0.0))

    # fmt: on
    with target:
        mod = tvm.IRModule({"main": select})
        mod = tvm.tir.transform.LowerTIRp()(mod)
        mod = tvm.tir.transform.Simplify()(mod)
        assert_structural_equal(mod["main"], expected)


def test_select_expr_affine():
    src_shape = [512, 512]
    src_layout = TrainiumLayout(
        dimension_types="FPF",
        combined_1d_layout=T.TileLayout.from_tuple((4, 128, 512), (512, 1, 1)),
    )
    dst_shape = src_shape
    dst_layout = src_layout
    # fmt: off
    @T.prim_func(tirp=True)
    def select() -> None:
        with T.kernel():
            A_sbuf = T.alloc_buffer(src_shape, "float32", scope="trn.sbuf", layout=src_layout)
            B_sbuf = T.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
            Tp.select(B_sbuf, A_sbuf, 0.0, lambda i, j: i < j)           
            
    @T.prim_func(tirp=True)
    def expected():
        T.func_attr({"global_symbol": "select"})
        with T.kernel():
            A_sbuf = T.alloc_buffer((128, 2048), scope="trn.sbuf", logical_scope="kernel")
            B_sbuf = T.alloc_buffer((128, 2048), scope="trn.sbuf", logical_scope="kernel")
            for b_loop, additional_b_loop in T.grid(4, 1):
                T.attr(0, "tensorized_nki_instruction", 1)
                for p_loop, f_loop in T.grid(128, 512):
                    T.nki_affine_select(B_sbuf[p_loop, b_loop * 512 + f_loop], b_loop * 128 + p_loop < f_loop, A_sbuf[p_loop, b_loop * 512 + f_loop], T.float32(0.0))
    # fmt: on
    with target:
        mod = tvm.IRModule({"main": select})
        mod = tvm.tir.transform.LowerTIRp()(mod)
        mod = tvm.tir.transform.Simplify()(mod)
        assert_structural_equal(mod["main"], expected)


def test_select_with_guard():
    src_shape = [512, 512]
    src_layout = TrainiumLayout(
        dimension_types="FPF",
        combined_1d_layout=T.TileLayout.from_tuple((4, 128, 512), (512, 1, 1)),
    )
    dst_shape = src_shape
    dst_layout = src_layout
    # fmt: off
    @T.prim_func(tirp=True)
    def select() -> None:
        with T.kernel():
            A_sbuf = T.alloc_buffer(src_shape, "float32", scope="trn.sbuf", layout=src_layout)
            B_sbuf = T.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
            for i in range(4):
                for j in range(4):
                    Tp.select(B_sbuf[0: (i+1) * 128, 0: (j+1) * 128], A_sbuf[0: (i+1) * 128, 0: (j+1) * 128], 0.0, lambda a, b: a < b)  
         
    @T.prim_func(tirp=True)
    def expected():
        T.func_attr({"global_symbol": "select"})
        with T.kernel():
            A_sbuf = T.alloc_buffer((128, 2048), scope="trn.sbuf", logical_scope="kernel")
            B_sbuf = T.alloc_buffer((128, 2048), scope="trn.sbuf", logical_scope="kernel")
            for i, j, b_loop, additional_b_loop in T.grid(4, 4, 4, 1):
                T.attr(0, "tensorized_nki_instruction", 1)
                for p_loop, f_loop in T.grid(128, 512):
                    if b_loop - i < 1 and f_loop < j * 128 + 128:
                        T.nki_affine_select(B_sbuf[p_loop, b_loop * 512 + f_loop], b_loop * 128 + p_loop < f_loop, A_sbuf[p_loop, b_loop * 512 + f_loop], T.float32(0.0))         
    # fmt: on
    with target:
        mod = tvm.IRModule({"main": select})
        mod = tvm.tir.transform.LowerTIRp()(mod)
        mod = tvm.tir.transform.Simplify()(mod)
        assert_structural_equal(mod["main"], expected)


if __name__ == "__main__":
    tvm.testing.main()
