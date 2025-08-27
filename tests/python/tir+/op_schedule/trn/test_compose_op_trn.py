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
import tvm.testing
from tvm.ir import assert_structural_equal
from tvm.script import tir as T
from tvm.script import tirp as Tp
from tvm.tir.layout import TileLayout

target = tvm.target.Target("aws/trn1/trn1.2xlarge")


def test_simple_activation_reduce():
    A_shape = (128, 512)
    A_layout = TileLayout(shard=([128, 512], [(1, "P"), (1, "F")]))
    B_shape = (128, 512)
    B_layout = TileLayout(shard=([128, 512], [(1, "P"), (1, "F")]))
    C_shape = (128, 1)
    C_layout = TileLayout(shard=([128, 1], [(1, "P"), (1, "F")]))
    # fmt: off
    @T.prim_func(tirp=True)
    def activation_reduce():
        with T.kernel():
            A = T.alloc_buffer(A_shape, dtype="float32", scope="trn.sbuf", layout=A_layout)
            B = T.alloc_buffer(B_shape, dtype="float32", scope="trn.sbuf", layout=B_layout)
            C = T.alloc_buffer(C_shape, dtype="float32", scope="trn.sbuf", layout=C_layout)
            Tp.unary_reduce(B, C, A, "sqrt", "sum", reduce_axes=1)
    
                
    @T.prim_func(tirp=True)
    def expected():
        T.func_attr({"global_symbol": "activation_reduce"})
        with T.kernel():
            const_bias = T.alloc_buffer((128, 512), scope="trn.sbuf")
            A = T.alloc_buffer((128, 512), scope="trn.sbuf")
            B = T.alloc_buffer((128, 512), scope="trn.sbuf")
            C = T.alloc_buffer((128, 1), scope="trn.sbuf")
            with T.attr(0, "tensorized_nki_instruction", 1):
                for p_loop in T.serial(128, annotations={"nki_dim": "P"}):
                    for f_loop in T.serial(512, annotations={"nki_dim": "F"}):
                        T.nki.memset(const_bias[p_loop, f_loop], T.float32(0.0))
            for b_loop in range(1):
                T.attr(0, "tensorized_nki_instruction", 1)
                for p_loop in T.serial(0, 128, annotations={"nki_dim":"P"}):
                    for f_loop in T.serial(0, 512, annotations={"nki_dim":"F"}):
                        T.nki.activation_reduce(C[p_loop, 0], B[p_loop, f_loop], A[p_loop, f_loop], "sqrt", "add", bias=const_bias[p_loop, f_loop])
    # fmt: on
    with target:
        mod = tvm.IRModule({"main": activation_reduce})
        mod = tvm.tirp.transform.PrivateBufferAlloc()(mod)
        mod = tvm.tir.transform.LowerTIRp()(mod)
        assert_structural_equal(mod["main"], expected)


def test_activation_reduce_in_loop():
    A_shape = (32, 512, 128)
    A_layout = TileLayout(shard=([16 * 1024, 128], [(1, "F"), (1, "P")]))
    B_shape = (16, 512, 128)
    B_layout = TileLayout(shard=([2, 4, 1024, 128], [(1024, "F"), (2048, "F"), (1, "F"), (1, "P")]))
    C_shape = (16, 128)
    C_layout = TileLayout(shard=([2, 4, 2, 128], [(2, "F"), (4, "F"), (1, "F"), (1, "P")]))
    # fmt: off
    @T.prim_func(tirp=True)
    def activation_reduce():
        with T.kernel():
            A = T.alloc_buffer(A_shape, dtype="float32", scope="trn.sbuf", layout=A_layout)
            B = T.alloc_buffer(B_shape, dtype="float32", scope="trn.sbuf", layout=B_layout)
            C = T.alloc_buffer(C_shape, dtype="float32", scope="trn.sbuf", layout=C_layout)
            for i in range(2):
                Tp.unary_reduce(B, C, A[i*16:i*16+16], "sqrt", "sum", reduce_axes=1)
                    
    @T.prim_func(tirp=True)
    def expected():
        T.func_attr({"global_symbol": "activation_reduce"})
        with T.kernel():
            const_bias = T.alloc_buffer((128, 512), scope="trn.sbuf")
            A = T.alloc_buffer((128, 16384), scope="trn.sbuf")
            B = T.alloc_buffer((128, 8192), scope="trn.sbuf")
            C = T.alloc_buffer((128, 16), scope="trn.sbuf")
            with T.attr(0, "tensorized_nki_instruction", 1):
                for p_loop in T.serial(128, annotations={"nki_dim": "P"}):
                    for f_loop in T.serial(512, annotations={"nki_dim": "F"}):
                        T.nki.memset(const_bias[p_loop, f_loop], T.float32(0.0))
            for i, b_loop in T.grid(2, 16):
                T.attr(0, "tensorized_nki_instruction", 1)
                for p_loop in T.serial(0, 128, annotations={"nki_dim":"P"}):
                    for f_loop in T.serial(0, 512, annotations={"nki_dim":"F"}):
                        T.nki.activation_reduce(C[p_loop, b_loop % 8 // 2 * 4 + b_loop // 8 * 2 + b_loop % 2], B[p_loop, b_loop % 8 // 2 * 2048 + b_loop // 8 * 1024 + b_loop % 2 * 512 + f_loop], A[p_loop, i * 8192 + b_loop * 512 + f_loop], "sqrt", "add", bias=const_bias[p_loop, f_loop])
    # fmt: off
    with target:
        mod = tvm.IRModule({"main": activation_reduce})
        mod = tvm.tirp.transform.PrivateBufferAlloc()(mod)
        mod = tvm.tir.transform.LowerTIRp()(mod)
        assert_structural_equal(mod["main"], expected)


def test_activation_reduce_in_loop2():
    A_shape = (32, 512, 128)
    A_layout = TileLayout(shard=([16 * 1024, 128], [(1, "F"), (1, "P")]))
    B_shape = (16, 512, 128)
    B_layout = TileLayout(shard=([16 * 512, 128], [(1, "F"), (1, "P")]))
    C_shape = (16, 128)
    C_layout = TileLayout(shard=([2, 4, 2, 128], [(2, "F"), (4, "F"), (1, "F"), (1, "P")]))
    # fmt: off
    @T.prim_func(tirp=True)
    def activation_reduce():
        with T.kernel():
            A = T.alloc_buffer(A_shape, dtype="float32", scope="trn.sbuf", layout=A_layout)
            B = T.alloc_buffer(B_shape, dtype="float32", scope="trn.sbuf", layout=B_layout)
            C = T.alloc_buffer(C_shape, dtype="float32", scope="trn.sbuf", layout=C_layout)
            for i in range(2):
                Tp.unary_reduce(B, C, A[i*16:i*16+16], "sqrt", "sum", reduce_axes=1)

    @T.prim_func(tirp=True)
    def expected():
        T.func_attr({"global_symbol": "activation_reduce"})
        with T.kernel():
            const_bias = T.alloc_buffer((128, 512), scope="trn.sbuf")
            A = T.alloc_buffer((128, 16384), scope="trn.sbuf")
            B = T.alloc_buffer((128, 8192), scope="trn.sbuf")
            C = T.alloc_buffer((128, 16), scope="trn.sbuf")
            with T.attr(0, "tensorized_nki_instruction", 1):
                for p_loop in T.serial(128, annotations={"nki_dim": "P"}):
                    for f_loop in T.serial(512, annotations={"nki_dim": "F"}):
                        T.nki.memset(const_bias[p_loop, f_loop], T.float32(0.0))
            for i, b_loop in T.grid(2, 16):
                T.attr(0, "tensorized_nki_instruction", 1)
                for p_loop in T.serial(0, 128, annotations={"nki_dim":"P"}):
                    for f_loop in T.serial(0, 512, annotations={"nki_dim":"F"}):
                        T.nki.activation_reduce(C[p_loop, b_loop % 8 // 2 * 4 + b_loop // 8 * 2 + b_loop % 2], B[p_loop, b_loop * 512 + f_loop], A[p_loop, i * 8192 + b_loop * 512 + f_loop], "sqrt", "add", bias=const_bias[p_loop, f_loop])
    # fmt: off
    with target:
        mod = tvm.IRModule({"main": activation_reduce})
        mod = tvm.tirp.transform.PrivateBufferAlloc()(mod)
        mod = tvm.tir.transform.LowerTIRp()(mod)
        assert_structural_equal(mod["main"], expected)


def test_activation_reduce_two_stage():
    A_shape = (32, 512, 128)
    A_layout = TileLayout(shard=([16 * 1024, 128], [(1, "F"), (1, "P")]))
    B_shape = (16, 512, 128)
    B_layout = TileLayout(shard=([2, 4, 1024, 128], [(1024, "F"), (2048, "F"), (1, "F"), (1, "P")]))
    C_shape = (1, 128)
    C_layout = TileLayout(shard=([1, 128], [(1, "F"), (1, "P")]))
    # fmt: off
    @T.prim_func(tirp=True)
    def activation_reduce():
        with T.kernel():
            A = T.alloc_buffer(A_shape, dtype="float32", scope="trn.sbuf", layout=A_layout)
            B = T.alloc_buffer(B_shape, dtype="float32", scope="trn.sbuf", layout=B_layout)
            C = T.alloc_buffer(C_shape, dtype="float32", scope="trn.sbuf", layout=C_layout)
            for i in range(2):
                Tp.unary_reduce(B, C, A[i*16:i*16+16], "sqrt", "sum", reduce_axes=(0,1))

    @T.prim_func(tirp=True)
    def expected():
        T.func_attr({"global_symbol": "activation_reduce"})
        with T.kernel():
            partial_reduce = T.alloc_buffer((128, 8), scope="trn.sbuf")
            const_bias = T.alloc_buffer((128, 1024), scope="trn.sbuf")
            A = T.alloc_buffer((128, 16384), scope="trn.sbuf")
            B = T.alloc_buffer((128, 8192), scope="trn.sbuf")
            C = T.alloc_buffer((128, 1), scope="trn.sbuf")
            with T.attr(0, "tensorized_nki_instruction", 1):
                for p_loop in T.serial(128, annotations={"nki_dim": "P"}):
                    for f_loop in T.serial(1024, annotations={"nki_dim": "F"}):
                        T.nki.memset(const_bias[p_loop, f_loop], T.float32(0.0))
            for i, b_loop in T.grid(2, 1):
                for reduction_b_loop in range(8):
                    T.attr(0, "tensorized_nki_instruction", 1)
                    for p_loop in T.serial(128, annotations={"nki_dim": "P"}):
                        for f_loop in T.serial(1024, annotations={"nki_dim": "F"}):
                            T.nki.activation_reduce(partial_reduce[p_loop, reduction_b_loop], B[p_loop, reduction_b_loop % 4 * 2048 + reduction_b_loop // 4 * 1024 + f_loop], A[p_loop, i * 8192 + reduction_b_loop * 1024 + f_loop], "sqrt", "add", const_bias[p_loop, f_loop], T.float32(1.0))
                T.attr(0, "tensorized_nki_instruction", 1)
                for p_loop in T.serial(128, annotations={"nki_dim": "P"}):
                    for f_loop in T.serial(8, annotations={"nki_dim": "F"}):
                        T.nki.tensorreduce(C[p_loop, 0], partial_reduce[p_loop, f_loop], "add", T.bool(False), -1)
    # fmt: off
    with target:
        mod = tvm.IRModule({"main": activation_reduce})
        mod = tvm.tirp.transform.PrivateBufferAlloc()(mod)
        mod = tvm.tir.transform.LowerTIRp()(mod)
        assert_structural_equal(mod["main"], expected)


def test_activation_reduce_with_bias_scale():
    A_shape = (32, 512, 128)
    A_layout = TileLayout(shard=([16 * 1024, 128], [(1, "F"), (1, "P")]))
    B_shape = (16, 512, 128)
    B_layout = TileLayout(shard=([16 * 512, 128], [(1, "F"), (1, "P")]))
    C_shape = (16, 128)
    C_layout = TileLayout(shard=([2, 4, 2, 128], [(2, "F"), (4, "F"), (1, "F"), (1, "P")]))
    bias_shape = 128
    bias_layout = TileLayout(shard=([128, 1], [(1, "P"), (1, "F")]))
    # fmt: off
    @T.prim_func(tirp=True)
    def activation_reduce():
        with T.kernel():
            A = T.alloc_buffer(A_shape, dtype="float32", scope="trn.sbuf", layout=A_layout)
            B = T.alloc_buffer(B_shape, dtype="float32", scope="trn.sbuf", layout=B_layout)
            C = T.alloc_buffer(C_shape, dtype="float32", scope="trn.sbuf", layout=C_layout)
            bias = T.alloc_buffer(bias_shape, dtype="float32", scope="trn.sbuf", layout=bias_layout)
            for i in range(2):
                Tp.unary_reduce(B, C, A[i*16:i*16+16], "sqrt", "sum", reduce_axes=1, bias=bias, scale=2.0)
                    
    @T.prim_func(tirp=True)
    def expected():
        T.func_attr({"global_symbol": "activation_reduce"})
        with T.kernel():
            A = T.alloc_buffer((128, 16384), scope="trn.sbuf")
            B = T.alloc_buffer((128, 8192), scope="trn.sbuf")
            C = T.alloc_buffer((128, 16), scope="trn.sbuf")
            bias = T.alloc_buffer((128, 1), scope="trn.sbuf")
            for i, b_loop in T.grid(2, 16):
                T.attr(0, "tensorized_nki_instruction", 1)
                for p_loop in T.serial(0, 128, annotations={"nki_dim":"P"}):
                    for f_loop in T.serial(0, 512, annotations={"nki_dim":"F"}):
                        T.nki.activation_reduce(C[p_loop, b_loop % 8 // 2 * 4 + b_loop // 8 * 2 + b_loop % 2], B[p_loop, b_loop * 512 + f_loop], A[p_loop, i * 8192 + b_loop * 512 + f_loop], "sqrt", "add", bias[p_loop, 0], T.float32(2.0))
    # fmt: on
    with target:
        mod = tvm.IRModule({"main": activation_reduce})
        mod = tvm.tir.transform.LowerTIRp()(mod)
        assert_structural_equal(mod["main"], expected)


def test_simple_tensor_scalar_reduce():
    A_shape = (128, 512)
    A_layout = TileLayout(shard=([128, 512], [(1, "P"), (1, "F")]))
    B_shape = (128, 512)
    B_layout = TileLayout(shard=([128, 512], [(1, "P"), (1, "F")]))
    C_shape = (128, 1)
    C_layout = TileLayout(shard=([128, 1], [(1, "P"), (1, "F")]))
    # fmt: off
    @T.prim_func(tirp=True)
    def tensor_scalar_reduce():
        with T.kernel():
            A = T.alloc_buffer(A_shape, dtype="float32", scope="trn.sbuf", layout=A_layout)
            B = T.alloc_buffer(B_shape, dtype="float32", scope="trn.sbuf", layout=B_layout)
            C = T.alloc_buffer(C_shape, dtype="float32", scope="trn.sbuf", layout=C_layout)
            Tp.binary_reduce(B, C, A, 1.0, "add", "sum", reduce_axes=1)
                
    @T.prim_func(tirp=True)
    def expected():
        T.func_attr({"global_symbol": "tensor_scalar_reduce"})
        with T.kernel():
            A = T.alloc_buffer((128, 512), scope="trn.sbuf")
            B = T.alloc_buffer((128, 512), scope="trn.sbuf")
            C = T.alloc_buffer((128, 1), scope="trn.sbuf")
            for b_loop in range(1):
                T.attr(0, "tensorized_nki_instruction", 1)
                for p_loop in T.serial(0, 128, annotations={"nki_dim":"P"}):
                    for f_loop in T.serial(0, 512, annotations={"nki_dim":"F"}):
                        T.nki.tensorscalar_reduce(C[p_loop, 0], B[p_loop, f_loop], A[p_loop, f_loop], T.float32(1.0), "add", "add", T.bool(False))
    # fmt: off
    with target:
        mod = tvm.IRModule({"main": tensor_scalar_reduce})
        mod = tvm.tir.transform.LowerTIRp()(mod)
        assert_structural_equal(mod["main"], expected)


def test_tensor_tensor_reduce_fail():
    A_shape = (128, 512)
    A_layout = TileLayout(shard=([128, 512], [(1, "P"), (1, "F")]))
    B_shape = (128, 512)
    B_layout = TileLayout(shard=([128, 512], [(1, "P"), (1, "F")]))
    D_shape = (128, 512)
    D_layout = TileLayout(shard=([128, 512], [(1, "P"), (1, "F")]))
    C_shape = (128, 1)
    C_layout = TileLayout(shard=([128, 1], [(1, "P"), (1, "F")]))
    # fmt: off
    @T.prim_func(tirp=True)
    def tensor_scalar_reduce():
        with T.kernel():
            A = T.alloc_buffer(A_shape, dtype="float32", scope="trn.sbuf", layout=A_layout)
            B = T.alloc_buffer(B_shape, dtype="float32", scope="trn.sbuf", layout=B_layout)
            C = T.alloc_buffer(C_shape, dtype="float32", scope="trn.sbuf", layout=C_layout)
            D = T.alloc_buffer(D_shape, dtype="float32", scope="trn.sbuf", layout=D_layout)
            Tp.binary_reduce(B, C, A, D, "add", "sum", reduce_axes=1)
                
    # fmt: off
    with pytest.raises(Exception):
        with target:
            mod = tvm.IRModule({"main": tensor_scalar_reduce})
            mod = tvm.tir.transform.LowerTIRp()(mod)


def test_tensor_scalar_reduce_complex():
    src1_shape = [32, 128, 512]
    src1_layout = TileLayout(
        shard=([32, 128, 4, 128], [(128, "F"), (1, "F"), (32 * 128, "F"), (1, "P")])
    )
    src2_shape = [128, 512]
    src2_layout = TileLayout(shard=([128, 4, 128], [(1, "F"), (128, "F"), (1, "P")]))
    dst_shape = src1_shape
    dst_layout = src1_layout
    reduce_dst_shape = [128, 512]
    reduce_dst_layout = TileLayout(shard=([128, 4, 128], [(1, "F"), (128, "F"), (1, "P")]))
    # fmt: off
    @T.prim_func(tirp=True)
    def tensor_scalar_reduce() -> None:
        with T.kernel():
            A_sbuf = T.alloc_buffer(src1_shape, "float32", scope="trn.sbuf", layout=src1_layout)
            B_sbuf = T.alloc_buffer(src2_shape, "float32", scope="trn.sbuf", layout=src2_layout)
            C_sbuf = T.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
            D_sbuf = T.alloc_buffer(reduce_dst_shape, "float32", scope="trn.sbuf", layout=reduce_dst_layout)
            Tp.binary_reduce(C_sbuf, D_sbuf, B_sbuf, A_sbuf, "add", "sum", reduce_axes=0)
                
    @T.prim_func(tirp=True)
    def expected():
        T.func_attr({"global_symbol": "tensor_scalar_reduce"})
        with T.kernel():
            A_sbuf = T.alloc_buffer((128, 16384), scope="trn.sbuf")
            B_sbuf = T.alloc_buffer((128, 512), scope="trn.sbuf")
            C_sbuf = T.alloc_buffer((128, 16384), scope="trn.sbuf")
            D_sbuf = T.alloc_buffer((128, 512), scope="trn.sbuf")
            for b_loop in range(512):
                T.attr(0, "tensorized_nki_instruction", 1)
                for p_loop in T.serial(0, 128, annotations={"nki_dim":"P"}):
                    for f_loop in T.serial(0, 32, annotations={"nki_dim":"F"}):
                        T.nki.tensorscalar_reduce(D_sbuf[p_loop, b_loop % 4 * 128 + b_loop // 4], C_sbuf[p_loop, b_loop % 4 * 4096 + f_loop * 128 + b_loop // 4], A_sbuf[p_loop, b_loop % 4 * 4096 + f_loop * 128 + b_loop // 4], B_sbuf[p_loop, b_loop % 4 * 128 + b_loop // 4], "add", "add", T.bool(True))
    # fmt: off
    with target:
        mod = tvm.IRModule({"main": tensor_scalar_reduce})
        mod = tvm.tir.transform.LowerTIRp()(mod)
        assert_structural_equal(mod["main"], expected)


def test_tensor_scalar_reduce_two_stage():
    src1_shape = [512, 1024, 4]
    src1_layout = TileLayout(shard=([128, 4096, 4], [(1, "P"), (1, "F"), (4096, "F")]))
    dst1_shape = src1_shape
    dst1_layout = src1_layout
    reduce_dst_shape = [512]
    reduce_dst_layout = TileLayout(shard=([128, 4], [(1, "P"), (1, "F")]))
    # fmt: off
    @T.prim_func(tirp=True)
    def tensor_scalar_reduce() -> None:
        with T.kernel():
            A_sbuf = T.alloc_buffer(src1_shape, "float32", scope="trn.sbuf", layout=src1_layout)
            B_sbuf = T.alloc_buffer(dst1_shape, "float32", scope="trn.sbuf", layout=dst1_layout)
            C_sbuf = T.alloc_buffer(reduce_dst_shape, "float32", scope="trn.sbuf", layout=reduce_dst_layout)
            Tp.binary_reduce(B_sbuf, C_sbuf, A_sbuf, 1.0, "add", "sum", reduce_axes=(1, 2))
                
    @T.prim_func(tirp=True)
    def expected():
        T.func_attr({"global_symbol": "tensor_scalar_reduce"})
        with T.kernel():
            partial_reduce = T.alloc_buffer((128, 4), scope="trn.sbuf")
            A_sbuf = T.alloc_buffer((128, 16384), scope="trn.sbuf")
            B_sbuf = T.alloc_buffer((128, 16384), scope="trn.sbuf")
            C_sbuf = T.alloc_buffer((128, 4), scope="trn.sbuf")
            for b_loop in range(4):
                for reduction_b_loop in range(4):
                    T.attr(0, "tensorized_nki_instruction", 1)
                    for p_loop in T.serial(128, annotations={"nki_dim": "P"}):
                        for f_loop in T.serial(1024, annotations={"nki_dim": "F"}):
                            T.nki.tensorscalar_reduce(partial_reduce[p_loop, reduction_b_loop], B_sbuf[p_loop, reduction_b_loop * 4096 + b_loop * 1024 + f_loop], A_sbuf[p_loop, reduction_b_loop * 4096 + b_loop * 1024 + f_loop], T.float32(1.0), "add", "add", T.bool(False))
                T.attr(0, "tensorized_nki_instruction", 1)
                for p_loop in T.serial(128, annotations={"nki_dim": "P"}):
                    for f_loop in T.serial(4, annotations={"nki_dim": "F"}):
                        T.nki.tensorreduce(C_sbuf[p_loop, b_loop], partial_reduce[p_loop, f_loop], "add", T.bool(False), -1)
    # fmt: on
    with target:
        mod = tvm.IRModule({"main": tensor_scalar_reduce})
        mod = tvm.tirp.transform.PrivateBufferAlloc()(mod)
        mod = tvm.tir.transform.LowerTIRp()(mod)
        assert_structural_equal(mod["main"], expected)


def test_vector_chain():
    src1_shape = [32, 128, 512]
    src1_layout = TileLayout(
        shard=([32, 128, 4, 128], [(1, "F"), (32, "F"), (32 * 128, "F"), (1, "P")])
    )
    src2_shape = [128, 512]
    src2_layout = TileLayout(shard=([512, 128], [(1, "F"), (1, "P")]))
    src3_shape = [512]
    src3_layout = TileLayout(shard=([4, 128], [(1, "F"), (1, "P")]))
    dst_shape = src1_shape
    dst_layout = src1_layout

    # fmt: off
    @T.prim_func(tirp=True)
    def binary() -> None:
        with T.kernel():
            A_sbuf = T.alloc_buffer(src1_shape, "float32", scope="trn.sbuf", layout=src1_layout)
            B_sbuf = T.alloc_buffer(src2_shape, "float32", scope="trn.sbuf", layout=src2_layout)
            C_sbuf = T.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
            D_sbuf = T.alloc_buffer(src3_shape, "float32", scope="trn.sbuf", layout=src3_layout)
            E_sbuf = T.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
            Tp.binary_chain(E_sbuf, A_sbuf, B_sbuf, D_sbuf, "add", "add", reverse1=True)

    @T.prim_func(tirp=True)
    def expected():
        T.func_attr({"global_symbol": "binary"})
        with T.kernel():
            A_sbuf = T.alloc_buffer((128, 16384), scope="trn.sbuf")
            B_sbuf = T.alloc_buffer((128, 512), scope="trn.sbuf")
            C_sbuf = T.alloc_buffer((128, 16384), scope="trn.sbuf")
            D_sbuf = T.alloc_buffer((128, 4), scope="trn.sbuf")
            E_sbuf = T.alloc_buffer((128, 16384), scope="trn.sbuf")
            for b_loop in T.serial(0, 512):
                T.attr(0, "tensorized_nki_instruction", 1)
                for p_loop in T.serial(0, 128, annotations={"nki_dim":"P"}):
                    for f_loop in T.serial(0, 32, annotations={"nki_dim":"F"}):
                        T.nki.scalar_tensor_scalar(E_sbuf[p_loop, b_loop % 4 * 4096 + b_loop // 4 * 32 + f_loop], A_sbuf[p_loop, b_loop % 4 * 4096 + b_loop // 4 * 32 + f_loop], B_sbuf[p_loop, b_loop], D_sbuf[p_loop, b_loop % 4], "add", "add", T.bool(False), T.bool(True))
    # fmt: on

    with target:
        mod = tvm.IRModule({"main": binary})
        mod = tvm.tir.transform.LowerTIRp()(mod)
        assert_structural_equal(mod["main"], expected)


def test_vector_chain_2():
    src1_shape = [32, 128, 512]
    src1_layout = TileLayout(
        shard=([32, 128, 4, 128], [(1, "F"), (32, "F"), (32 * 128, "F"), (1, "P")])
    )
    src2_shape = [128, 512]
    src2_layout = TileLayout(shard=([512, 128], [(1, "F"), (1, "P")]))
    src3_shape = src1_shape
    src3_layout = src1_layout
    dst_shape = src1_shape
    dst_layout = src1_layout

    # fmt: off
    @T.prim_func(tirp=True)
    def binary() -> None:
        with T.kernel():
            A_sbuf = T.alloc_buffer(src1_shape, "float32", scope="trn.sbuf", layout=src1_layout)
            B_sbuf = T.alloc_buffer(src2_shape, "float32", scope="trn.sbuf", layout=src2_layout)
            C_sbuf = T.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
            D_sbuf = T.alloc_buffer(src3_shape, "float32", scope="trn.sbuf", layout=src3_layout)
            E_sbuf = T.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
            Tp.binary_chain(E_sbuf, A_sbuf, B_sbuf, D_sbuf, "add", "add", reverse1=True)
    
    @T.prim_func(tirp=True)
    def expected():
        T.func_attr({"global_symbol": "binary"})
        with T.kernel():
            A_sbuf = T.alloc_buffer((128, 16384), scope="trn.sbuf")
            B_sbuf = T.alloc_buffer((128, 512), scope="trn.sbuf")
            C_sbuf = T.alloc_buffer((128, 16384), scope="trn.sbuf")
            D_sbuf = T.alloc_buffer((128, 16384), scope="trn.sbuf")
            E_sbuf = T.alloc_buffer((128, 16384), scope="trn.sbuf")
            for b_loop in T.serial(0, 512):
                T.attr(0, "tensorized_nki_instruction", 1)
                for p_loop in T.serial(0, 128, annotations={"nki_dim":"P"}):
                    for f_loop in T.serial(0, 32, annotations={"nki_dim":"F"}):
                        T.nki.scalar_tensor_tensor(E_sbuf[p_loop, b_loop % 4 * 4096 + b_loop // 4 * 32 + f_loop], A_sbuf[p_loop, b_loop % 4 * 4096 + b_loop // 4 * 32 + f_loop], B_sbuf[p_loop, b_loop], D_sbuf[p_loop, b_loop % 4 * 4096 + b_loop // 4 * 32 + f_loop], "add", "add", T.bool(False), T.bool(True))
    # fmt: on

    with target:
        mod = tvm.IRModule({"main": binary})
        mod = tvm.tir.transform.LowerTIRp()(mod)
        assert_structural_equal(mod["main"], expected)


def test_reduce_negate():
    src_shape = [128, 512, 4]
    src_layout = TileLayout(shard=([128, 512, 4], [(1, "P"), (4, "F"), (1, "F")]))
    dst_shape = [128, 4]
    dst_layout = TileLayout(shard=([128, 4], [(1, "P"), (1, "F")]))

    # fmt: off
    @T.prim_func(tirp=True)
    def reduction():
        with T.kernel():
            A_sbuf = T.alloc_buffer(src_shape, "float32", scope="trn.sbuf", layout=src_layout)
            B_sbuf = T.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
            for i in range(4):
                Tp.reduce_negate(B_sbuf[:, i], A_sbuf[:, :, i], reduce_op="sum", reduce_axes=-2)

    @T.prim_func(tirp=True)
    def expected():
        T.func_attr({"global_symbol": "reduction"})
        with T.kernel():
            A_sbuf = T.alloc_buffer((128, 2048), scope="trn.sbuf")
            B_sbuf = T.alloc_buffer((128, 4), scope="trn.sbuf")
            for i, b_loop in T.grid(4, 1):
                T.attr(0, "tensorized_nki_instruction", 1)
                for p_loop in T.serial(0, 128, annotations={"nki_dim":"P"}):
                    for f_loop in T.serial(0, 512, annotations={"nki_dim":"F"}):
                        T.nki.tensorreduce(B_sbuf[p_loop, i], A_sbuf[p_loop, f_loop * 4 + i], "add", True, -1)
    # fmt: on
    with target:
        mod = tvm.IRModule({"main": reduction})
        mod = tvm.tir.transform.LowerTIRp()(mod)
        assert_structural_equal(mod["main"], expected)


def test_binary_reduce_guard():
    src_shape = [512, 512]
    src_layout = TileLayout(shard=([4, 128, 512], [(512, "F"), (1, "P"), (1, "F")]))
    dst_shape = src_shape
    dst_layout = src_layout
    reduce_dst_shape = [512]
    reduce_dst_layout = TileLayout(shard=([4, 128], [(1, "F"), (1, "P")]))
    # fmt: off
    @T.prim_func(tirp=True)
    def binary_reduce() -> None:
        with T.kernel():
            A_sbuf = T.alloc_buffer(src_shape, "float32", scope="trn.sbuf", layout=src_layout)
            B_sbuf = T.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
            C_sbuf = T.alloc_buffer(reduce_dst_shape, "float32", scope="trn.sbuf", layout=reduce_dst_layout)
            for j in range(4):
                for i in range(4):
                    Tp.binary_reduce(B_sbuf[0:128*(j+1), 0:128*(i+1)], C_sbuf[0:128*(j+1)], A_sbuf[0:128*(j+1), 0:128*(i+1)], 0.0, "add", "sum", [-1])

    @T.prim_func(tirp=True)
    def expected():
        T.func_attr({"global_symbol": "binary_reduce"})
        with T.kernel():
            A_sbuf = T.alloc_buffer((128, 2048), scope="trn.sbuf")
            B_sbuf = T.alloc_buffer((128, 2048), scope="trn.sbuf")
            C_sbuf = T.alloc_buffer((128, 4), scope="trn.sbuf")
            for j, i, b_loop in T.grid(4, 4, 4):
                T.attr(0, "tensorized_nki_instruction", 1)
                for p_loop in T.serial(0, 128, annotations={"nki_dim":"P"}):
                    for f_loop in T.serial(0, 512, annotations={"nki_dim":"F"}):
                        if b_loop - j < 1 and f_loop < i * 128 + 128:
                            T.nki.tensorscalar_reduce(C_sbuf[p_loop, b_loop], B_sbuf[p_loop, b_loop * 512 + f_loop], A_sbuf[p_loop, b_loop * 512 + f_loop], T.float32(0.0), "add", "add", T.bool(False))
    # fmt: on
    with target:
        mod = tvm.IRModule({"main": binary_reduce})
        mod = tvm.tir.transform.LowerTIRp()(mod)
        mod = tvm.tir.transform.Simplify()(mod)
        assert_structural_equal(mod["main"], expected)


def test_unary_reduce_guard():
    src_shape = [512, 512]
    src_layout = TileLayout(shard=([4, 128, 512], [(512, "F"), (1, "P"), (1, "F")]))
    dst_shape = src_shape
    dst_layout = src_layout
    reduce_dst_shape = [512]
    reduce_dst_layout = TileLayout(shard=([4, 128], [(1, "F"), (1, "P")]))

    # fmt: off
    @T.prim_func(tirp=True)
    def unary_reduce() -> None:
        with T.kernel():
            A_sbuf = T.alloc_buffer(src_shape, "float32", scope="trn.sbuf", layout=src_layout)
            B_sbuf = T.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
            C_sbuf = T.alloc_buffer(reduce_dst_shape, "float32", scope="trn.sbuf", layout=reduce_dst_layout)
            for j in range(4):
                for i in range(4):
                    Tp.unary_reduce(B_sbuf[0:128*(j+1), 0:128*(i+1)], C_sbuf[0:128*(j+1)], A_sbuf[0:128*(j+1), 0:128*(i+1)], "sqrt", "sum", reduce_axes=[-1])
    
    @T.prim_func(tirp=True)
    def expected():
        T.func_attr({"global_symbol": "unary_reduce"})
        with T.kernel():
            const_bias = T.alloc_buffer((128, 512), scope="trn.sbuf")
            A_sbuf = T.alloc_buffer((128, 2048), scope="trn.sbuf")
            B_sbuf = T.alloc_buffer((128, 2048), scope="trn.sbuf")
            C_sbuf = T.alloc_buffer((128, 4), scope="trn.sbuf")
            with T.attr(0, "tensorized_nki_instruction", 1):
                for p_loop in T.serial(128, annotations={"nki_dim": "P"}):
                    for f_loop in T.serial(512, annotations={"nki_dim": "F"}):
                        T.nki.memset(const_bias[p_loop, f_loop], T.float32(0.0))
            for j, i, b_loop in T.grid(4, 4, 4):
                T.attr(0, "tensorized_nki_instruction", 1)
                for p_loop in T.serial(128, annotations={"nki_dim": "P"}):
                    for f_loop in T.serial(512, annotations={"nki_dim": "F"}):
                        if b_loop - j < 1 and f_loop < i * 128 + 128:
                            T.nki.activation_reduce(C_sbuf[p_loop, b_loop], B_sbuf[p_loop, b_loop * 512 + f_loop], A_sbuf[p_loop, b_loop * 512 + f_loop], "sqrt", "add", const_bias[p_loop, f_loop], T.float32(1.0))

    # fmt: on
    with target:
        mod = tvm.IRModule({"main": unary_reduce})
        mod = tvm.tirp.transform.PrivateBufferAlloc()(mod)
        mod = tvm.tir.transform.LowerTIRp()(mod)
        mod = tvm.tir.transform.Simplify()(mod)
        assert_structural_equal(mod["main"], expected)


def test_binary_chain_guard():
    src_shape = [512, 512]
    src_layout = TileLayout(shard=([4, 128, 512], [(512, "F"), (1, "P"), (1, "F")]))
    dst_shape = src_shape
    dst_layout = src_layout
    src2_shape = [512, 1]
    src2_layout = TileLayout(shard=([4, 128], [(1, "F"), (1, "P")]))

    # fmt: off
    @T.prim_func(tirp=True)
    def binary_chain() -> None:
        with T.kernel():
            A_sbuf = T.alloc_buffer(src_shape, "float32", scope="trn.sbuf", layout=src_layout)
            B_sbuf = T.alloc_buffer(src2_shape, "float32", scope="trn.sbuf", layout=src2_layout)
            C_sbuf = T.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
            for j in range(4):
                for i in range(4):
                    Tp.binary_chain(C_sbuf[0:128*(j+1), 0:128*(i+1)], A_sbuf[0:128*(j+1), 0:128*(i+1)], B_sbuf[0:128*(j+1), 0], 1.0, "add", "sub", reverse1=True)
    @T.prim_func(tirp=True)
    def expected():
        T.func_attr({"global_symbol": "binary_chain"})
        with T.kernel():
            A_sbuf = T.alloc_buffer((128, 2048), scope="trn.sbuf")
            B_sbuf = T.alloc_buffer((128, 4), scope="trn.sbuf")
            C_sbuf = T.alloc_buffer((128, 2048), scope="trn.sbuf")
            for j, i, b_loop in T.grid(4, 4, 4):
                T.attr(0, "tensorized_nki_instruction", 1)
                for p_loop in T.serial(0, 128, annotations={"nki_dim":"P"}):
                    for f_loop in T.serial(0, 512, annotations={"nki_dim":"F"}):
                        if b_loop - j < 1 and f_loop < i * 128 + 128:
                            T.nki.scalar_tensor_scalar(C_sbuf[p_loop, b_loop * 512 + f_loop], A_sbuf[p_loop, b_loop * 512 + f_loop], B_sbuf[p_loop, b_loop], T.float32(1.0), "add", "sub", T.bool(False), T.bool(True))
    # fmt: on
    with target:
        mod = tvm.IRModule({"main": binary_chain})
        mod = tvm.tir.transform.LowerTIRp()(mod)
        mod = tvm.tir.transform.Simplify()(mod)
        assert_structural_equal(mod["main"], expected)


def test_activation_reduce_two_stage_workspace():
    A_shape = (32, 512, 128)
    A_layout = TileLayout(shard=([16 * 1024, 128], [(1, "F"), (1, "P")]))
    B_shape = (16, 512, 128)
    B_layout = TileLayout(shard=([2, 4, 1024, 128], [(1024, "F"), (2048, "F"), (1, "F"), (1, "P")]))
    C_shape = (1, 128)
    C_layout = TileLayout(shard=([1, 128], [(1, "F"), (1, "P")]))
    # fmt: off
    @T.prim_func(tirp=True)
    def activation_reduce():
        with T.kernel():
            intermediate_buffer = T.alloc_buffer((128, 16), scope="trn.sbuf")
            A = T.alloc_buffer(A_shape, dtype="float32", scope="trn.sbuf", layout=A_layout)
            B = T.alloc_buffer(B_shape, dtype="float32", scope="trn.sbuf", layout=B_layout)
            C = T.alloc_buffer(C_shape, dtype="float32", scope="trn.sbuf", layout=C_layout)
            for i in range(2):
                Tp.unary_reduce(B, C, A[i*16:i*16+16], "sqrt", "sum", reduce_axes=(0,1), workspace={"partial_reduce": intermediate_buffer})

    @T.prim_func(tirp=True)
    def expected():
        T.func_attr({"global_symbol": "activation_reduce"})
        with T.kernel():
            const_bias = T.alloc_buffer((128, 1024), scope="trn.sbuf")
            intermediate_buffer = T.alloc_buffer((128, 16), scope="trn.sbuf")
            A = T.alloc_buffer((128, 16384), scope="trn.sbuf")
            B = T.alloc_buffer((128, 8192), scope="trn.sbuf")
            C = T.alloc_buffer((128, 1), scope="trn.sbuf")
            with T.attr(0, "tensorized_nki_instruction", 1):
                for p_loop in T.serial(128, annotations={"nki_dim": "P"}):
                    for f_loop in T.serial(1024, annotations={"nki_dim": "F"}):
                        T.nki.memset(const_bias[p_loop, f_loop], T.float32(0.0))
            for i, b_loop in T.grid(2, 1):
                for reduction_b_loop in range(8):
                    T.attr(0, "tensorized_nki_instruction", 1)
                    for p_loop in T.serial(128, annotations={"nki_dim": "P"}):
                        for f_loop in T.serial(1024, annotations={"nki_dim": "F"}):
                            T.nki.activation_reduce(intermediate_buffer[p_loop, reduction_b_loop], B[p_loop, reduction_b_loop % 4 * 2048 + reduction_b_loop // 4 * 1024 + f_loop], A[p_loop, i * 8192 + reduction_b_loop * 1024 + f_loop], "sqrt", "add", const_bias[p_loop, f_loop], T.float32(1.0))
                T.attr(0, "tensorized_nki_instruction", 1)
                for p_loop in T.serial(128, annotations={"nki_dim": "P"}):
                    for f_loop in T.serial(8, annotations={"nki_dim": "F"}):
                        T.nki.tensorreduce(C[p_loop, 0], intermediate_buffer[p_loop, f_loop], "add", T.bool(False), -1)

    # fmt: on
    with target:
        mod = tvm.IRModule({"main": activation_reduce})
        mod = tvm.tirp.transform.PrivateBufferAlloc()(mod)
        mod = tvm.tir.transform.LowerTIRp()(mod)
        assert_structural_equal(mod["main"], expected)


def test_tensor_scalar_reduce_two_stage_workspace():
    src1_shape = [512, 1024, 4]
    src1_layout = TileLayout(shard=([128, 4096, 4], [(1, "P"), (1, "F"), (4096, "F")]))
    dst1_shape = src1_shape
    dst1_layout = src1_layout
    reduce_dst_shape = [512]
    reduce_dst_layout = TileLayout(shard=([128, 4], [(1, "P"), (1, "F")]))
    # fmt: off
    @T.prim_func(tirp=True)
    def tensor_scalar_reduce() -> None:
        with T.kernel():
            intermediate_buffer = T.alloc_buffer((128, 8), scope="trn.sbuf")
            A_sbuf = T.alloc_buffer(src1_shape, "float32", scope="trn.sbuf", layout=src1_layout)
            B_sbuf = T.alloc_buffer(dst1_shape, "float32", scope="trn.sbuf", layout=dst1_layout)
            C_sbuf = T.alloc_buffer(reduce_dst_shape, "float32", scope="trn.sbuf", layout=reduce_dst_layout)
            Tp.binary_reduce(B_sbuf, C_sbuf, A_sbuf, 1.0, "add", "sum", reduce_axes=(1, 2), workspace={"partial_reduce": intermediate_buffer})
                
    @T.prim_func(tirp=True)
    def expected():
        T.func_attr({"global_symbol": "tensor_scalar_reduce"})
        with T.kernel():
            intermediate_buffer = T.alloc_buffer((128, 8), scope="trn.sbuf")
            A_sbuf = T.alloc_buffer((128, 16384), scope="trn.sbuf")
            B_sbuf = T.alloc_buffer((128, 16384), scope="trn.sbuf")
            C_sbuf = T.alloc_buffer((128, 4), scope="trn.sbuf")
            for b_loop in range(4):
                for reduction_b_loop in range(4):
                    T.attr(0, "tensorized_nki_instruction", 1)
                    for p_loop in T.serial(128, annotations={"nki_dim": "P"}):
                        for f_loop in T.serial(1024, annotations={"nki_dim": "F"}):
                            T.nki.tensorscalar_reduce(intermediate_buffer[p_loop, reduction_b_loop], B_sbuf[p_loop, reduction_b_loop * 4096 + b_loop * 1024 + f_loop], A_sbuf[p_loop, reduction_b_loop * 4096 + b_loop * 1024 + f_loop], T.float32(1.0), "add", "add", T.bool(False))
                T.attr(0, "tensorized_nki_instruction", 1)
                for p_loop in T.serial(128, annotations={"nki_dim": "P"}):
                    for f_loop in T.serial(4, annotations={"nki_dim": "F"}):
                        T.nki.tensorreduce(C_sbuf[p_loop, b_loop], intermediate_buffer[p_loop, f_loop], "add", T.bool(False), -1)
    # fmt: on
    with target:
        mod = tvm.IRModule({"main": tensor_scalar_reduce})
        mod = tvm.tir.transform.LowerTIRp()(mod)
        assert_structural_equal(mod["main"], expected)


def test_unary_reduce_complex():
    # fmt: off
    @T.prim_func(tirp=True)
    def unary_reduce():
        with T.kernel():
            p = T.alloc_buffer((128, 8192), "float16", scope="trn.sbuf", layout="PF")
            rowsum_p = T.alloc_buffer((2, 128, 1), scope="trn.sbuf", layout="FPF")
            qk = T.alloc_buffer((2, 128, 8192), scope="trn.sbuf", layout="FPF")
            running_max = T.alloc_buffer((16384, 1), dtype="float32", scope="trn.sbuf", layout="PF")
            for i in range(4):
                Tp.unary_reduce(p[0:128, 0:8192], rowsum_p[i % 2, 0:128, 0], qk[i % 2, 0:128, 0:8192], "exp", "sum", bias=running_max[i * 128:i * 128 + 128, 0])

    @T.prim_func(tirp=True)
    def expected():
        T.func_attr({"global_symbol": "unary_reduce"})
        with T.kernel():
            p = T.alloc_buffer((128, 8192), "float16", scope="trn.sbuf")
            rowsum_p = T.alloc_buffer((128, 2), scope="trn.sbuf")
            qk = T.alloc_buffer((128, 16384), scope="trn.sbuf")
            running_max = T.alloc_buffer((128, 128), scope="trn.sbuf")
            for i, b_loop in T.grid(4, 1):
                T.attr(0, "tensorized_nki_instruction", 1)
                for p_loop in T.serial(128, annotations={"nki_dim": "P"}):
                    for f_loop in T.serial(8192, annotations={"nki_dim": "F"}):
                        T.nki.activation_reduce(rowsum_p[p_loop, i % 2], p[p_loop, f_loop], qk[p_loop, i % 2 * 8192 + f_loop], "exp", "add", running_max[p_loop, i], T.float32(1.0))
    # fmt: on
    with target:
        mod = tvm.IRModule({"main": unary_reduce})
        mod = tvm.tir.transform.LowerTIRp()(mod)
        assert_structural_equal(mod["main"], expected)


if __name__ == "__main__":
    tvm.testing.main()
