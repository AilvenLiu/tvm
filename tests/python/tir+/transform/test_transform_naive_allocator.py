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
import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.ir import assert_structural_equal
from tvm.script import ir as I
from tvm.script import tir as T
from tvm.script import tirp as Tp
from tvm.tir.layout import TileLayout
from tvm.tirp.transform import NaiveAllocator


def test_one_alloc():
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
            
    @T.prim_func(tirp=True)
    def expected(A_ptr: T.handle) -> None:
        T.func_attr({"global_symbol": "copy"})
        A = T.match_buffer(A_ptr, src_shape, "float32", layout=src_layout)
        with T.kernel():
            A_sbuf = T.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout, allocated_addr=[0])
            Tp.copy(A_sbuf, A)
    # fmt: on

    mod = tvm.IRModule({"copy": copy})
    mod = NaiveAllocator()(mod)
    assert_structural_equal(mod["copy"], expected)


def test_two_alloc():
    # fmt: off
    @T.prim_func(tirp=True)
    def copy(A_ptr: T.handle) -> None:
        with T.kernel():
            A_sbuf = T.alloc_buffer([256, 512], "float32", scope="trn.sbuf", layout="PF")
            B_sbuf = T.alloc_buffer([512, 512], "float32", scope="trn.sbuf", layout="PF")
            Tp.copy(B_sbuf[0:256, :], A_sbuf)
            
    @T.prim_func(tirp=True)
    def expected(A_ptr: T.handle) -> None:
        T.func_attr({"global_symbol": "copy"})
        with T.kernel():
            A_sbuf = T.alloc_buffer([256, 512], "float32", scope="trn.sbuf", layout="PF", allocated_addr=[0])
            B_sbuf = T.alloc_buffer([512, 512], "float32", scope="trn.sbuf", layout="PF", allocated_addr=[2*512*4])
            Tp.copy(B_sbuf[0:256, :], A_sbuf)
    # fmt: on

    mod = tvm.IRModule({"copy": copy})
    mod = NaiveAllocator()(mod)
    assert_structural_equal(mod["copy"], expected)


def test_existing_alloc():
    # fmt: off
    @T.prim_func(tirp=True)
    def copy(A_ptr: T.handle) -> None:
        with T.kernel():
            A_sbuf = T.alloc_buffer([256, 512], "float32", scope="trn.sbuf", layout="PF")
            B_sbuf = T.alloc_buffer([512, 512], "float32", scope="trn.sbuf", layout="PF", allocated_addr=[1])
            Tp.copy(B_sbuf[0:256, :], A_sbuf)
            
    @T.prim_func(tirp=True)
    def expected(A_ptr: T.handle) -> None:
        T.func_attr({"global_symbol": "copy"})
        with T.kernel():
            A_sbuf = T.alloc_buffer([256, 512], "float32", scope="trn.sbuf", layout="PF", allocated_addr=[4*512*4+1])
            B_sbuf = T.alloc_buffer([512, 512], "float32", scope="trn.sbuf", layout="PF", allocated_addr=[1])
            Tp.copy(B_sbuf[0:256, :], A_sbuf)
    # fmt: on

    mod = tvm.IRModule({"copy": copy})
    mod = NaiveAllocator()(mod)
    assert_structural_equal(mod["copy"], expected)


def test_workspace():
    # fmt: off
    @T.prim_func(tirp=True)
    def copy(A_ptr: T.handle) -> None:
        with T.kernel():
            A_sbuf = T.alloc_buffer([256, 512], "float32", scope="trn.sbuf", layout="PF")
            B_sbuf = T.alloc_buffer([512, 512], "float32", scope="trn.sbuf", layout="PF")
            C_sbuf = T.alloc_buffer([128, 1024], "float32", scope="trn.sbuf")
            Tp.copy(B_sbuf[0:256, :], A_sbuf, workspace={"C": C_sbuf})

    @T.prim_func(tirp=True)
    def expected(A_ptr: T.handle) -> None:
        T.func_attr({"global_symbol": "copy"})
        with T.kernel():
            A_sbuf = T.alloc_buffer([256, 512], "float32", scope="trn.sbuf", layout="PF", allocated_addr=[0])
            B_sbuf = T.alloc_buffer([512, 512], "float32", scope="trn.sbuf", layout="PF", allocated_addr=[2*512*4])
            C_sbuf = T.alloc_buffer([128, 1024], "float32", scope="trn.sbuf", allocated_addr=[2*512*4+4*512*4])
            Tp.copy(B_sbuf[0:256, :], A_sbuf, workspace={"C": C_sbuf})
    # fmt: on

    mod = tvm.IRModule({"copy": copy})
    mod = NaiveAllocator()(mod)
    assert_structural_equal(mod["copy"], expected)


def test_other_scope_alloc():
    # fmt: off
    @T.prim_func(tirp=True)
    def copy(A_ptr: T.handle) -> None:
        with T.kernel():
            A_sbuf = T.alloc_buffer([256, 512], "float32", scope="trn.sbuf", layout="PF")
            B_sbuf = T.alloc_buffer([512, 512], "float32", scope="trn.sbuf", layout="PF")
            C_sbuf = T.alloc_buffer([8, 128, 512], "float32", scope="global")
            Tp.copy(B_sbuf[0:256, :], A_sbuf, workspace={"C": C_sbuf})

    @T.prim_func(tirp=True)
    def expected(A_ptr: T.handle) -> None:
        T.func_attr({"global_symbol": "copy"})
        with T.kernel():
            A_sbuf = T.alloc_buffer([256, 512], "float32", scope="trn.sbuf", layout="PF", allocated_addr=[0])
            B_sbuf = T.alloc_buffer([512, 512], "float32", scope="trn.sbuf", layout="PF", allocated_addr=[2*512*4])
            C_sbuf = T.alloc_buffer([8, 128, 512], "float32", scope="global")
            Tp.copy(B_sbuf[0:256, :], A_sbuf, workspace={"C": C_sbuf})
    # fmt: on

    mod = tvm.IRModule({"copy": copy})
    mod = NaiveAllocator()(mod)
    assert_structural_equal(mod["copy"], expected)


def test_buffer_views():
    # fmt: off
    @T.prim_func(tirp=True)
    def copy(A_ptr: T.handle) -> None:
        with T.kernel():
            A_sbuf = T.alloc_buffer([256, 512], "float32", scope="trn.sbuf", layout="PF")
            B_sbuf = T.alloc_buffer([512, 512], "float32", scope="trn.sbuf", layout="PF")
            B_view = B_sbuf.view(2, 256, 512)
            Tp.copy(B_view[0], A_sbuf)

    @T.prim_func(tirp=True)
    def expected(A_ptr: T.handle) -> None:
        T.func_attr({"global_symbol": "copy"})
        with T.kernel():
            A_sbuf = T.alloc_buffer([256, 512], "float32", scope="trn.sbuf", layout="PF", allocated_addr=[0])
            B_sbuf = T.alloc_buffer([512, 512], "float32", scope="trn.sbuf", layout="PF", allocated_addr=[2*512*4])
            B_view = B_sbuf.view(2, 256, 512)
            Tp.copy(B_view[0], A_sbuf)
    # fmt: on

    mod = tvm.IRModule({"copy": copy})
    mod = NaiveAllocator()(mod)
    assert_structural_equal(mod["copy"], expected)


if __name__ == "__main__":
    tvm.testing.main()
