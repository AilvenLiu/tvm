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

from tvm.script import tir as T
from tvm.tir.analysis import verify_tirp_well_formed as verify


def test_root_scope():
    # fmt: off
    @T.prim_func(tirp=True, check_well_formed=False)
    def test1() -> None:
        with T.thread():
            pass

    @T.prim_func(tirp=True, check_well_formed=False)
    def test2() -> None:
        with T.warp():
            with T.thread():
                pass

    @T.prim_func(tirp=True, check_well_formed=False)
    def test3() -> None:
        with T.cta():
            with T.warp():
                with T.thread():
                    pass

    @T.prim_func(tirp=True, check_well_formed=False)
    def test4() -> None:
        with T.kernel():
            with T.cta():
                with T.warp():
                    with T.thread():
                        pass

    @T.prim_func(tirp=True, check_well_formed=False)
    def test5() -> None:
        with T.world():
            with T.kernel():
                with T.cta():
                    with T.warp():
                        with T.thread():
                            pass
    # fmt: on

    with pytest.raises(Exception, match="invalid exec_scope thread as root"):
        verify(test1)
    with pytest.raises(Exception, match="invalid exec_scope warp as root"):
        verify(test2)
    with pytest.raises(Exception, match="invalid exec_scope cta as root"):
        verify(test3)
    verify(test4)
    verify(test5)


def test_nested_scope():
    # fmt: off
    @T.prim_func(tirp=True, check_well_formed=False)
    def test1() -> None:
        with T.kernel():
            with T.cta():
                with T.warp():
                    with T.thread():
                        pass
                with T.thread():
                    pass

    @T.prim_func(tirp=True, check_well_formed=False)
    def test2() -> None:
        with T.kernel():
            with T.thread():
                with T.cta():
                    with T.thread():
                        pass

    @T.prim_func(tirp=True, check_well_formed=False)
    def test3() -> None:
        with T.kernel():
            with T.warp():
                with T.thread():
                    with T.cta():
                        with T.thread():
                            pass
    # fmt: on

    verify(test1)
    verify(test2)
    with pytest.raises(Exception, match="invalid exec_scope cta under warp"):
        verify(test3)


def test_invalid_stmt():
    # fmt: off
    @T.prim_func(tirp=True, check_well_formed=False)
    def test1() -> None:
        with T.kernel():
            with T.cta():
                with T.warp():
                    with T.thread():
                        pass
                with T.thread():
                    pass

    @T.prim_func(tirp=True, check_well_formed=False)
    def test2() -> None:
        with T.kernel():
            with T.cta():
                for i in range(3):
                    with T.warp():
                        with T.thread():
                            pass
                    with T.thread():
                        pass


    # fmt: on

    verify(test1)
    verify(test2)


def test_inconsistent_scope_id():
    # fmt: off
    @T.prim_func(tirp=True, check_well_formed=False)
    def test1():
        with T.kernel():
            bx = T.cta_id([32], parent="kernel")
            wid = T.warp_id([4], parent="cta")
            lane = T.thread_id([32], parent="warp")

            with T.thread():
                pass

    @T.prim_func(tirp=True, check_well_formed=False)
    def test2():
        with T.kernel():
            bx = T.cta_id([32], parent="kernel")
            wid = T.warp_id([4], parent="cta")
            lane = T.thread_id([32], parent="warp")
            tid = T.thread_id([128], parent="cta")

            with T.thread():
                pass

    @T.prim_func(tirp=True, check_well_formed=False)
    def test3():
        with T.kernel():
            bx = T.cta_id([32], parent="kernel")
            wid = T.warp_id([2], parent="cta")
            lane = T.thread_id([32], parent="warp")
            tid = T.thread_id([128], parent="cta")

            with T.thread():
                pass
    # fmt: on

    verify(test1)
    verify(test2)
    with pytest.raises(Exception, match="Inconsistent extents for scope"):
        verify(test3)


def test_layout():
    ### TileLayout
    # fmt: off
    @T.prim_func(tirp=True, check_well_formed=False)
    def test1():
        with T.kernel():
            bx = T.cta_id([32], parent="kernel")
            wid = T.warp_id([4], parent="cta")
            lane = T.thread_id([32], parent="warp")

            with T.thread():
                A = T.alloc_buffer((2,), layout=T.TileLayout.from_nested_tuple(2, 1))

                A[0] = 0

    @T.prim_func(tirp=True, check_well_formed=False)
    def test2():
        with T.kernel():
            bx = T.cta_id([32], parent="kernel")
            wid = T.warp_id([4], parent="cta")
            lane = T.thread_id([32], parent="warp")

            with T.thread():
                A = T.alloc_buffer((2,), layout=T.TileLayout.from_nested_tuple(3, 1))

                A[0] = 0
    @T.prim_func(tirp=True, check_well_formed=False)
    def test3():
        with T.kernel():
            bx = T.cta_id([32], parent="kernel")
            wid = T.warp_id([4], parent="cta")
            lane = T.thread_id([32], parent="warp")

            with T.thread():
                A = T.alloc_buffer((2,), layout=T.TileLayout.from_nested_tuple(3, -1))

                A[0] = 0
    # fmt: on

    verify(test1)
    with pytest.raises(Exception, match="not compatible with shape"):
        verify(test2)
        verify(test3)

    ### SwizzleLayout
    # fmt: off
    @T.prim_func(tirp=True, check_well_formed=False)
    def test4():
        with T.kernel():
            bx = T.cta_id([32], parent="kernel")
            wid = T.warp_id([4], parent="cta")
            lane = T.thread_id([32], parent="warp")

            with T.thread():
                A = T.alloc_buffer((512,), scope="shared", layout=T.SwizzleLayout(3, 3, 3))

                A[0] = 0
    @T.prim_func(tirp=True, check_well_formed=False)
    def test5():
        with T.kernel():
            bx = T.cta_id([32], parent="kernel")
            wid = T.warp_id([4], parent="cta")
            lane = T.thread_id([32], parent="warp")

            with T.thread():
                A = T.alloc_buffer((513,), scope="shared", layout=T.SwizzleLayout(3, 3, 3))

                A[0] = 0
    # fmt: on
    verify(test4)
    with pytest.raises(Exception, match="not compatible with shape"):
        verify(test5)


def test_host():
    # fmt: off
    @T.prim_func(tirp=True, check_well_formed=False)
    def test1(A_ptr: T.handle):
        A = T.match_buffer(A_ptr, (16, 16), dtype="float32", align=16, logical_scope="kernel")

        A_map: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
        T.call_packed("runtime.cuTensorMapInit", A_map, "float32", 2, A.data, 16, 16, 64, 16, 16, 1, 1, 0, 0, 0, 0)

        with T.kernel():
            for blockIdx in T.thread_binding(1, thread="blockIdx.x"):
                for threadIdx in T.thread_binding(128, thread="threadIdx.x"):
                    with T.thread():
                        bar = T.alloc_buffer((1,), "uint64", scope="shared", logical_scope="cta", align=8)
                        phase = T.alloc_buffer((1,), "int32", scope="local", logical_scope="thread")
                        A_smem = T.alloc_buffer((16, 16), "float32", scope="shared", logical_scope="cta", align=128)

                        phase[0] = 0
                        if threadIdx == 0:
                            T.mbarrier_init(bar.data, 1)
                            T.cuda_fence_proxy_async("shared")
                            T.cp_async_bulk_tensor_global_to_cluster(2, A_smem.data, bar.data, A_map, 0, 0)
                            T.mbarrier_arrive_expect_tx(bar.data, 16*16*4)
                        T.mbarrier_wait(bar.data, phase[0])
                        phase[0] = phase[0] ^ 1
                        T.print_buffer(A_smem.data, "float32", 2, 16*16)
    # fmt: on
    verify(test1)


if __name__ == "__main__":
    test_root_scope()
    test_nested_scope()
    test_invalid_stmt()
    test_inconsistent_scope_id()
    test_layout()
    test_host()
