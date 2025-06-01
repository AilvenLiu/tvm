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
from tvm.tir.layout import TileLayout
import numpy as np
import tvm.testing
from tvm.script import ir as I
from tvm.script import tir as T
from tvm.script import tirp as Tp
from tvm.ir import assert_structural_equal
from tvm.tirp.transform import PrivateBufferAlloc

target = tvm.target.Target("aws/trn1/trn1.2xlarge")


def test_copy_transpose():
    src_shape = [512, 512]
    src_layout = TileLayout(([128, 2048], [(1, "P", 1), (1, "F")]))
    dst_shape = [512, 512]
    dst_layout = TileLayout(([2048, 128], [(1, "F"), (1, "P")]))

    # fmt: off
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
            identity = T.alloc_buffer((128, 128), scope="trn.sbuf", logical_scope="kernel")
            acc_psum = T.alloc_buffer((8, 128, 512), scope="trn.psum", logical_scope="kernel", allocated_addr=[0, 0])
            A_sbuf = T.alloc_buffer((512, 512), scope="trn.sbuf", logical_scope="kernel", 
                                    layout=T.TileLayout(([128, 2048], [(1, "P", 1), (1, "F")])))
            B_sbuf = T.alloc_buffer((512, 512), scope="trn.sbuf", logical_scope="kernel", 
                                    layout=T.TileLayout(([2048, 128], [(1, "F"), (1, "P")])))
            with T.attr(0, "tensorized_nki_instruction", 1):
                for p_loop in T.serial(128, annotations={"nki_dim": "P"}):
                    for rhs_f_loop in T.serial(128, annotations={"nki_dim": "F"}):
                        T.nki.identity(identity[p_loop, rhs_f_loop], 128)
            Tp.copy(B_sbuf[0:512, 0:512], A_sbuf[0:512, 0:512], workspace={"acc_psum": acc_psum, "identity": identity})

    # fmt: on
    with target:
        mod = tvm.IRModule({"main": copy})
        mod = PrivateBufferAlloc()(mod)
        assert_structural_equal(mod["main"], expected)


def test_normal_copy():
    src_shape = [128, 512]
    src_layout = TileLayout(([128, 512], [512, 1]))
    dst_shape = [128, 512]
    dst_layout = TileLayout(([128, 512], [(1, "P"), (1, "F")]))
    # fmt: off
    @T.prim_func(tirp=True)
    def copy(A_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, src_shape, "float32", layout=src_layout)
        with T.kernel():
            A_sbuf = T.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
            Tp.copy(A_sbuf, A)
    # fmt: on
    with target:
        mod = tvm.IRModule({"main": copy})
        mod = PrivateBufferAlloc()(mod)
        assert_structural_equal(mod["main"], copy)


def test_unary_with_bias_scale():
    src_shape = [512, 1024]
    src_layout = TileLayout(([128, 4096], [(1, "P"), (1, "F")]))
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
            Tp.exp(C_sbuf, A_sbuf, bias=bias, scale=scale)
            
    @T.prim_func(tirp=True)
    def expected():
        T.func_attr({"global_symbol": "unary"})
        with T.kernel():
            const_bias = T.alloc_buffer((128, 512), scope="trn.sbuf", logical_scope="kernel")
            A_sbuf = T.alloc_buffer((512, 1024), scope="trn.sbuf", logical_scope="kernel", 
                                    layout=T.TileLayout(([128, 4096], [(1, "P"), (1, "F")])))
            C_sbuf = T.alloc_buffer((512, 1024), scope="trn.sbuf", logical_scope="kernel", 
                                    layout=T.TileLayout(([128, 4096], [(1, "P"), (1, "F")])))
            with T.attr(0, "tensorized_nki_instruction", 1):
                for p_loop in T.serial(128, annotations={"nki_dim": "P"}):
                    for f_loop in T.serial(512, annotations={"nki_dim": "F"}):
                        T.nki.memset(const_bias[p_loop, f_loop], T.float32(1.0))
            Tp.exp(C_sbuf[0:512, 0:1024], A_sbuf[0:512, 0:1024], T.float32(1.0), T.float32(2.0), workspace={"const_bias": const_bias})
    # fmt: on
    with target:
        mod = tvm.IRModule({"main": unary})
        mod = PrivateBufferAlloc()(mod)
        assert_structural_equal(mod["main"], expected)


def test_reduction_two_stage():
    src_shape = [128, 32, 4, 32]
    src_layout = TileLayout(([128, 32 * 32 * 4], [(1, "P"), (1, "F")]))
    dst_shape = [128, 4]
    dst_layout = TileLayout(([128, 4], [(1, "P"), (1, "F")]))

    # fmt: off
    @T.prim_func(tirp=True)
    def reduction():
        with T.kernel():
            A_sbuf = T.alloc_buffer(src_shape, "float32", scope="trn.sbuf", layout=src_layout)
            B_sbuf = T.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
            Tp.sum(B_sbuf, A_sbuf, axes=(1, 3))
            
    @T.prim_func(tirp=True)
    def expected():
        T.func_attr({"global_symbol": "reduction"})
        with T.kernel():
            partial_reduce = T.alloc_buffer((128, 32), scope="trn.sbuf", logical_scope="kernel")
            A_sbuf = T.alloc_buffer((128, 32, 4, 32), scope="trn.sbuf", logical_scope="kernel", 
                                    layout=T.TileLayout(([128, 32 * 32 * 4], [(1, "P"), (1, "F")])))
            B_sbuf = T.alloc_buffer((128, 4), scope="trn.sbuf", logical_scope="kernel", 
                                    layout=T.TileLayout(([128, 4], [(1, "P"), (1, "F")])))
            Tp.sum(B_sbuf[0:128, 0:4], A_sbuf[0:128, 0:32, 0:4, 0:32], [1, 3], False, workspace={"partial_reduce": partial_reduce})

    # fmt: on
    with target:
        mod = tvm.IRModule({"main": reduction})
        mod = PrivateBufferAlloc()(mod)
        assert_structural_equal(mod["main"], expected)


def test_gemm():
    A_layout = TileLayout(([4, 128, 8, 128], [(1024, "F"), (1, "F"), (1, "F"), (1, "P")]))
    B_layout = TileLayout(([8, 128, 2, 128], [(256, "F"), (1, "P"), (128, "F"), (1, "F")]))

    C_layout = TileLayout(([4, 128, 2, 128], [(256, "F"), (1, "F"), (128, "F"), (1, "P")]))
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
            acc_psum = T.alloc_buffer((8, 128, 512), scope="trn.psum", logical_scope="kernel", allocated_addr=[0, 0])
            A_sbuf = T.alloc_buffer((512, 1024), scope="trn.sbuf", logical_scope="kernel", 
                                    layout=T.TileLayout(([4, 128, 8, 128], [(1024, "F"), (1, "F"), (1, "F"), (1, "P")])))
            B_sbuf = T.alloc_buffer((1024, 256), scope="trn.sbuf", logical_scope="kernel", 
                                    layout=T.TileLayout(([8, 128, 2, 128], [(256, "F"), (1, "P"), (128, "F"), (1, "F")])))
            C_sbuf = T.alloc_buffer((512, 256), scope="trn.sbuf", logical_scope="kernel", 
                                    layout=T.TileLayout(([4, 128, 2, 128], [(256, "F"), (1, "F"), (128, "F"), (1, "P")])))
            for i, k in T.grid(2, 2):
                Tp.gemm(C_sbuf[256 * i:256 * i + 256, 0:256], A_sbuf[256 * i:256 * i + 256, 512 * k:512 * k + 512], B_sbuf[512 * k:512 * k + 512, 0:256], C_sbuf[256 * i:256 * i + 256, 0:256], False, False, T.float32(1.0), T.float32(0.0), workspace={"acc_psum": acc_psum})
    # fmt: on
    with target:
        mod = tvm.IRModule({"main": gemm})
        mod = PrivateBufferAlloc()(mod)
        assert_structural_equal(mod["main"], expected)


def test_binary_reduce_two_stage():
    src1_shape = [512, 1024, 4]
    src1_layout = TileLayout(([128, 4096, 4], [(1, "P"), (1, "F"), (4096, "F")]))
    dst1_shape = src1_shape
    dst1_layout = src1_layout
    reduce_dst_shape = [512]
    reduce_dst_layout = TileLayout(([128, 4], [(1, "P"), (1, "F")]))
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
            partial_reduce = T.alloc_buffer((128, 4), scope="trn.sbuf", logical_scope="kernel")
            A_sbuf = T.alloc_buffer((512, 1024, 4), scope="trn.sbuf", logical_scope="kernel", 
                                    layout=T.TileLayout(([128, 4096, 4], [(1, "P"), (1, "F"), (4096, "F")])))
            B_sbuf = T.alloc_buffer((512, 1024, 4), scope="trn.sbuf", logical_scope="kernel", 
                                    layout=T.TileLayout(([128, 4096, 4], [(1, "P"), (1, "F"), (4096, "F")])))
            C_sbuf = T.alloc_buffer((512,), scope="trn.sbuf", logical_scope="kernel", 
                                    layout=T.TileLayout(([128, 4], [(1, "P"), (1, "F")])))
            Tp.binary_reduce(B_sbuf[0:512, 0:1024, 0:4], C_sbuf[0:512], A_sbuf[0:512, 0:1024, 0:4], T.float32(1.0), "add", "sum", [1, 2], workspace={"partial_reduce": partial_reduce})
    # fmt: on
    with target:
        mod = tvm.IRModule({"main": tensor_scalar_reduce})
        mod = PrivateBufferAlloc()(mod)
        assert_structural_equal(mod["main"], expected)


def test_activation_reduce_two_stage():
    A_shape = (32, 512, 128)
    A_layout = TileLayout(([16 * 1024, 128], [(1, "F"), (1, "P")]))
    B_shape = (16, 512, 128)
    B_layout = TileLayout(([2, 4, 1024, 128], [(1024, "F"), (2048, "F"), (1, "F"), (1, "P")]))
    C_shape = (1, 128)
    C_layout = TileLayout(([1, 128], [(1, "F"), (1, "P")]))
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
            partial_reduce = T.alloc_buffer((128, 8), scope="trn.sbuf", logical_scope="kernel")
            const_bias = T.alloc_buffer((128, 1024), scope="trn.sbuf", logical_scope="kernel")
            A = T.alloc_buffer((32, 512, 128), scope="trn.sbuf", logical_scope="kernel", 
                               layout=T.TileLayout(([16 * 1024, 128], [(1, "F"), (1, "P")])))
            B = T.alloc_buffer((16, 512, 128), scope="trn.sbuf", logical_scope="kernel", 
                               layout=T.TileLayout(([2, 4, 1024, 128], [(1024, "F"), (2048, "F"), (1, "F"), (1, "P")])))
            C = T.alloc_buffer((1, 128), scope="trn.sbuf", logical_scope="kernel", 
                               layout=T.TileLayout(([1, 128], [(1, "F"), (1, "P")])))
            with T.attr(0, "tensorized_nki_instruction", 1):
                for p_loop in T.serial(128, annotations={"nki_dim": "P"}):
                    for f_loop in T.serial(1024, annotations={"nki_dim": "F"}):
                        T.nki.memset(const_bias[p_loop, f_loop], T.float32(0.0))
            for i in range(2):
                Tp.unary_reduce(B[0:16, 0:512, 0:128], C[0, 0:128], A[i * 16:i * 16 + 16, 0:512, 0:128], "sqrt", "sum", None, None, [0, 1], workspace={"const_bias": const_bias, "partial_reduce": partial_reduce})
    # fmt: on
    with target:
        mod = tvm.IRModule({"main": activation_reduce})
        mod = PrivateBufferAlloc()(mod)
        assert_structural_equal(mod["main"], expected)


def test_partial_workspace_specify():
    A_shape = (32, 512, 128)
    A_layout = TileLayout(([16 * 1024, 128], [(1, "F"), (1, "P")]))
    B_shape = (16, 512, 128)
    B_layout = TileLayout(([2, 4, 1024, 128], [(1024, "F"), (2048, "F"), (1, "F"), (1, "P")]))
    C_shape = (1, 128)
    C_layout = TileLayout(([1, 128], [(1, "F"), (1, "P")]))
    # fmt: off
    @T.prim_func(tirp=True)
    def activation_reduce():
        with T.kernel():
            partial_reduce = T.alloc_buffer((128, 16), scope="trn.sbuf", logical_scope="kernel")
            A = T.alloc_buffer(A_shape, dtype="float32", scope="trn.sbuf", layout=A_layout)
            B = T.alloc_buffer(B_shape, dtype="float32", scope="trn.sbuf", layout=B_layout)
            C = T.alloc_buffer(C_shape, dtype="float32", scope="trn.sbuf", layout=C_layout)
            for i in range(2):
                Tp.unary_reduce(B, C, A[i*16:i*16+16], "sqrt", "sum", reduce_axes=(0,1), workspace={"partial_reduce": partial_reduce})
    
    @T.prim_func(tirp=True)
    def expected():
        T.func_attr({"global_symbol": "activation_reduce"})
        with T.kernel():
            const_bias = T.alloc_buffer((128, 1024), scope="trn.sbuf", logical_scope="kernel")
            partial_reduce = T.alloc_buffer((128, 16), scope="trn.sbuf", logical_scope="kernel")
            A = T.alloc_buffer((32, 512, 128), scope="trn.sbuf", logical_scope="kernel", 
                               layout=T.TileLayout(([16 * 1024, 128], [(1, "F"), (1, "P")])))
            B = T.alloc_buffer((16, 512, 128), scope="trn.sbuf", logical_scope="kernel", 
                               layout=T.TileLayout(([2, 4, 1024, 128], [(1024, "F"), (2048, "F"), (1, "F"), (1, "P")])))
            C = T.alloc_buffer((1, 128), scope="trn.sbuf", logical_scope="kernel", 
                               layout=T.TileLayout(([1, 128], [(1, "F"), (1, "P")])))
            with T.attr(0, "tensorized_nki_instruction", 1):
                for p_loop in T.serial(128, annotations={"nki_dim": "P"}):
                    for f_loop in T.serial(1024, annotations={"nki_dim": "F"}):
                        T.nki.memset(const_bias[p_loop, f_loop], T.float32(0.0))
            for i in range(2):
                Tp.unary_reduce(B[0:16, 0:512, 0:128], C[0, 0:128], A[i * 16:i * 16 + 16, 0:512, 0:128], "sqrt", "sum", None, None, [0, 1], workspace={"const_bias": const_bias, "partial_reduce": partial_reduce})
    # fmt: on
    with target:
        mod = tvm.IRModule({"main": activation_reduce})
        mod = PrivateBufferAlloc()(mod)
        assert_structural_equal(mod["main"], expected)


def test_workspace_reuse():
    src_shape = [512, 1024]
    src_layout = TileLayout(([128, 4096], [(1, "P"), (1, "F")]))
    dst_shape = src_shape
    dst_layout = src_layout
    scale = T.float32(2.0)
    # fmt: off
    @T.prim_func(tirp=True)
    def unary() -> None:
        with T.kernel():
            A_sbuf = T.alloc_buffer(src_shape, "float32", scope="trn.sbuf", layout=src_layout)
            C_sbuf = T.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
            Tp.exp(C_sbuf, A_sbuf, bias=0.0, scale=scale, schedule_config={"max_inst_size": 1024})
            Tp.exp(C_sbuf, C_sbuf)
            
    @T.prim_func(tirp=True)
    def expected():
        T.func_attr({"global_symbol": "unary"})
        with T.kernel():
            const_bias = T.alloc_buffer((128, 1024), scope="trn.sbuf", logical_scope="kernel")
            A_sbuf = T.alloc_buffer((512, 1024), scope="trn.sbuf", logical_scope="kernel", 
                                    layout=T.TileLayout(([128, 4096], [(1, "P"), (1, "F")])))
            C_sbuf = T.alloc_buffer((512, 1024), scope="trn.sbuf", logical_scope="kernel", 
                                    layout=T.TileLayout(([128, 4096], [(1, "P"), (1, "F")])))
            with T.attr(0, "tensorized_nki_instruction", 1):
                for p_loop in T.serial(128, annotations={"nki_dim": "P"}):
                    for f_loop in T.serial(1024, annotations={"nki_dim": "F"}):
                        T.nki.memset(const_bias[p_loop, f_loop], T.float32(0.0))
            Tp.exp(C_sbuf[0:512, 0:1024], A_sbuf[0:512, 0:1024], T.float32(0.0), T.float32(2.0), workspace={"const_bias": const_bias}, schedule_config={"max_inst_size": 1024})
            Tp.exp(C_sbuf[0:512, 0:1024], C_sbuf[0:512, 0:1024], None, None, workspace={"const_bias": const_bias})

    # fmt: on

    with target:
        mod = tvm.IRModule({"main": unary})
        mod = PrivateBufferAlloc()(mod)
        assert_structural_equal(mod["main"], expected)


def test_no_rewrite_with_existing_workspace():
    src_shape = [128, 32, 4, 32]
    src_layout = TileLayout(([128, 32 * 32 * 4], [(1, "P"), (1, "F")]))
    dst_shape = [128, 4]
    dst_layout = TileLayout(([128, 4], [(1, "P"), (1, "F")]))

    # fmt: off
    @T.prim_func(tirp=True)
    def reduction():
        with T.kernel():
            intermediate_buffer = T.alloc_buffer((128, 64), logical_scope="kernel", scope="trn.sbuf")
            A_sbuf = T.alloc_buffer(src_shape, "float32", scope="trn.sbuf", layout=src_layout)
            B_sbuf = T.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
            Tp.sum(B_sbuf, A_sbuf, axes=(1, 3), workspace={"partial_reduce": intermediate_buffer})
    # fmt: on
    with target:
        mod = tvm.IRModule({"main": reduction})
        mod = PrivateBufferAlloc()(mod)
        assert_structural_equal(mod["main"], reduction)


def test_no_rewrite_with_psum_output():
    A_layout = TileLayout(([128, 128], [(1, "F"), (1, "P")]))
    B_layout = TileLayout(([128, 128], [(1, "P"), (1, "F")]))

    C_layout = TileLayout(([128, 128], [(1, "P"), (1, "F")]))
    # fmt: off
    @T.prim_func(tirp=True)
    def gemm() -> None:
        with T.kernel():
            A_sbuf = T.alloc_buffer((128, 128), "float32", scope="trn.sbuf", layout=A_layout)
            B_sbuf = T.alloc_buffer((128, 128), "float32", scope="trn.sbuf", layout=B_layout)
            C_psum = T.alloc_buffer((128, 128), "float32", scope="trn.psum", layout=C_layout)
            Tp.gemm(C_psum, A_sbuf, B_sbuf, C_psum)
    # fmt: on
    with target:
        mod = tvm.IRModule({"main": gemm})
        mod = PrivateBufferAlloc()(mod)
        assert_structural_equal(mod["main"], gemm)


if __name__ == "__main__":
    tvm.testing.main()
