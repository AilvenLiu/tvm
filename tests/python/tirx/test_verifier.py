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
from tvm.script import tirx as Tx
from tvm.tir.analysis import verify_tirx_well_formed as verify


def test_root_scope():
    # fmt: off
    @T.prim_func(tirx=True, check_well_formed=False)
    def test1() -> None:
        with T.thread():
            pass

    @T.prim_func(tirx=True, check_well_formed=False)
    def test2() -> None:
        with T.warp():
            with T.thread():
                pass

    @T.prim_func(tirx=True, check_well_formed=False)
    def test3() -> None:
        with T.cta():
            with T.warp():
                with T.thread():
                    pass

    @T.prim_func(tirx=True, check_well_formed=False)
    def test4() -> None:
        with T.kernel():
            with T.cta():
                with T.warp():
                    with T.thread():
                        pass

    @T.prim_func(tirx=True, check_well_formed=False)
    def test5() -> None:
        with T.world():
            with T.kernel():
                with T.cta():
                    with T.warp():
                        with T.thread():
                            pass

    @T.prim_func(tirx=True, check_well_formed=False)
    def test6() -> None:
        with T.world():
            with T.kernel():
                with T.cta():
                    with T.warp():
                        with T.cluster():
                            pass

    # fmt: on

    verify(test1)
    verify(test2)
    verify(test3)
    verify(test4)
    verify(test5)
    verify(test6)


def test_nested_scope():
    # fmt: off
    @T.prim_func(tirx=True, check_well_formed=False)
    def test1() -> None:
        with T.kernel():
            with T.cta():
                with T.warp():
                    with T.thread():
                        pass
                with T.thread():
                    pass

    @T.prim_func(tirx=True, check_well_formed=False)
    def test2() -> None:
        with T.kernel():
            with T.thread():
                with T.cta():
                    with T.thread():
                        pass

    @T.prim_func(tirx=True, check_well_formed=False)
    def test3() -> None:
        with T.kernel():
            with T.warp():
                with T.thread():
                    with T.cta():
                        with T.thread():
                            pass
    @T.prim_func(tirx=True, check_well_formed=False)
    def test4() -> None:
        with T.kernel():
            with T.thread():
                with T.warpgroup():
                    with T.warp():
                        with T.thread():
                            pass
                with T.warpgroup():
                    with T.warp():
                        with T.thread():
                            pass

    @T.prim_func(tirx=True, check_well_formed=False)
    def test5() -> None:
        with T.kernel():
            with T.world():
                pass
    # fmt: on

    verify(test1)
    verify(test2)
    verify(test3)
    verify(test4)
    with pytest.raises(Exception, match="has invalid exec_scope world under kernel"):
        verify(test5)


def test_scope_slice():
    # fmt: off
    @T.prim_func(tirx=True, check_well_formed=False)
    def test1() -> None:
        with T.kernel():
            with T.cta():
                with T.warpgroup()[0:1]:
                    with T.thread():
                        pass

    @T.prim_func(tirx=True, check_well_formed=False)
    def test2() -> None:
        with T.kernel():
            with T.cta():
                with T.warpgroup()[0:3]:
                    with T.warpgroup()[1:2]:
                        pass
                    with T.warpgroup()[2:3]:
                        with T.thread()[T.ptx.elect_sync(0xFFFFFFFF)]:
                            pass
    @T.prim_func(tirx=True, check_well_formed=False)
    def test3() -> None:
        with T.kernel():
            with T.cta():
                with T.warpgroup()[0:3]:
                    with T.warpgroup()[1:5]:
                        pass
    @T.prim_func(tirx=True, check_well_formed=False)
    def test4() -> None:
        with T.kernel():
            with T.cta():
                with T.thread()[0:3]:
                    with T.thread()[T.ptx.elect_sync(0xFFFFFFFF)]:
                        pass
    @T.prim_func(tirx=True, check_well_formed=False)
    def test5() -> None:
        with T.kernel():
            with T.cta():
                with T.thread()[T.ptx.elect_sync(0xFFFFFFFF)]:
                    with T.thread()[T.ptx.elect_sync(0xFFFFFFFF)]:
                        pass
    @T.prim_func(tirx=True, check_well_formed=False)
    def test6() -> None:
        with T.kernel():
            with T.cta():
                with T.thread()[T.ptx.elect_sync(0xFFFFFFFF)]:
                    with T.thread()[T.int32(0)]:
                        pass
    # fmt: on
    verify(test1)
    verify(test2)
    with pytest.raises(Exception, match="is inconsistent with"):
        verify(test3)
    with pytest.raises(Exception, match="has both slices and select_cond"):
        verify(test4)
    # we don't check select_cond for now
    verify(test5)
    verify(test6)


def test_scope_partition():
    # fmt: off
    @T.prim_func(tirx=True, check_well_formed=False)
    def test1():
        with T.kernel():
            bx, by, bz = T.cta_id([3, 4, 5], parent="kernel")
            warp_id = T.warp_id([4], parent="cta")
            tx = T.thread_id([128], parent="cta")

            with T.cta():
                T.block_attr({"tirx.scope_partition": True})
                with T.thread()[0:30]:
                    T.evaluate(tx)
                with T.thread()[30:128]:
                    T.evaluate(tx)
    @T.prim_func(tirx=True, check_well_formed=False)
    def test2():
        with T.kernel():
            bx, by, bz = T.cta_id([3, 4, 5], parent="kernel")
            warp_id = T.warp_id([4], parent="cta")
            tx = T.thread_id([128], parent="cta")

            with T.cta():
                T.block_attr({"tirx.scope_partition": True})
                with T.thread()[0:30]:
                    T.evaluate(tx)
                with T.thread():
                    T.evaluate(tx)
    @T.prim_func(tirx=True, check_well_formed=False)
    def test3():
        with T.kernel():
            bx, by, bz = T.cta_id([3, 4, 5], parent="kernel")
            warp_id = T.warp_id([4], parent="cta")
            tx = T.thread_id([128], parent="cta")

            with T.thread():
                T.block_attr({"tirx.scope_partition": True})
                with T.thread()[0:30]:
                    T.evaluate(tx)
                T.evaluate(tx)
    @T.prim_func(tirx=True, check_well_formed=False)
    def test4():
        with T.kernel():
            bx, by, bz = T.cta_id([3, 4, 5], parent="kernel")
            warp_id = T.warp_id([4], parent="cta")
            tx = T.thread_id([128], parent="cta")

            with T.thread():
                T.block_attr({"tirx.scope_partition": True})
                with T.thread()[0:30]:
                    T.evaluate(tx)
                with T.warp()[30:128]:
                    T.evaluate(tx)
    # fmt: on

    verify(test1)
    with pytest.raises(Exception, match="has invalid exec_scope"):
        verify(test2)
    with pytest.raises(Exception, match="has invalid body"):
        verify(test3)
    with pytest.raises(Exception, match="has invalid exec_scope"):
        verify(test4)


def test_scope_id_consistency():
    # fmt: off
    @T.prim_func(tirx=True, check_well_formed=False)
    def test1():
        with T.kernel():
            bx = T.cta_id([32], parent="kernel")
            wid = T.warp_id([4], parent="cta")
            lane = T.thread_id([32], parent="warp")

            with T.thread():
                pass

    @T.prim_func(tirx=True, check_well_formed=False)
    def test2():
        with T.kernel():
            bx = T.cta_id([32], parent="kernel")
            wid = T.warp_id([4], parent="cta")
            lane = T.thread_id([32], parent="warp")
            tid = T.thread_id([128], parent="cta")

            with T.thread():
                pass

    @T.prim_func(tirx=True, check_well_formed=False)
    def test3():
        with T.kernel():
            bx = T.cta_id([32], parent="kernel")
            wid = T.warp_id([2], parent="cta")
            lane = T.thread_id([32], parent="warp")
            tid = T.thread_id([128], parent="cta")

            with T.thread():
                pass

    @T.prim_func(tirx=True, check_well_formed=False)
    def test4():
        with T.kernel():
            bx, by, bz = T.cta_id([8, 10, 12], parent="kernel")
            cbx, cby, cbz = T.cta_id([2, 2, 1], parent="cluster")
            clx, cly, clz = T.cluster_id([4, 5, 12], parent="kernel")
            with T.cta():
                with T.warp():
                    with T.thread():
                        T.evaluate(bx + by + bz)
                        T.evaluate(cbx + cby + cbz)
                        T.evaluate(clx + cly + clz)

    @T.prim_func(tirx=True, check_well_formed=False)
    def test5():
        with T.kernel():
            bx, by, bz = T.cta_id([8, 10, 12], parent="kernel")
            cbx, cby, cbz = T.cta_id([2, 2, 1], parent="cluster")
            clx, cly, clz = T.cluster_id([3, 5, 12], parent="kernel")
            with T.cta():
                with T.warp():
                    with T.thread():
                        T.evaluate(bx + by + bz)
                        T.evaluate(cbx + cby + cbz)
                        T.evaluate(clx + cly + clz)

    @T.prim_func(tirx=True, check_well_formed=False)
    def test6():
        with T.kernel():
            clx, cly, clz = T.cluster_id([4, 5, 12], parent="kernel")
            bx, by, bz = T.cta_id([8, 10, 12], parent="kernel")
            with T.cluster():
                cbx, cby, cbz = T.cta_id([2, 2, 1], parent="cluster")
                with T.warp():
                    with T.thread():
                        T.evaluate(bx + by + bz)
                        T.evaluate(cbx + cby + cbz)
                        T.evaluate(clx + cly + clz)

    @T.prim_func(tirx=True, check_well_formed=False)
    def test7():
        with T.kernel():
            clx, cly, clz = T.cluster_id([3, 5, 12], parent="kernel")
            bx, by, bz = T.cta_id([8, 10, 12], parent="kernel")
            with T.cluster():
                cbx, cby, cbz = T.cta_id([2, 2, 1], parent="cluster")
                with T.warp():
                    with T.thread():
                        T.evaluate(bx + by + bz)
                        T.evaluate(cbx + cby + cbz)
                        T.evaluate(clx + cly + clz)

    # fmt: on

    verify(test1)
    verify(test2)
    with pytest.raises(Exception, match="Inconsistent extents for scope"):
        verify(test3)
    verify(test4)
    with pytest.raises(Exception, match="Inconsistent extents for scope"):
        verify(test5)
    verify(test6)
    with pytest.raises(Exception, match="Inconsistent extents for scope"):
        verify(test7)


def test_layout():
    ### TileLayout
    # fmt: off
    @T.prim_func(tirx=True, check_well_formed=False)
    def test1():
        with T.kernel():
            bx = T.cta_id([32], parent="kernel")
            wid = T.warp_id([4], parent="cta")
            lane = T.thread_id([32], parent="warp")

            with T.thread():
                A = T.alloc_buffer((2,), layout=T.TileLayout((2, 1)))

                A[0] = 0
    # fmt: on
    verify(test1)

    ### SwizzleLayout
    # fmt: off
    @T.prim_func(tirx=True, check_well_formed=False)
    def test2():
        with T.kernel():
            bx = T.cta_id([32], parent="kernel")
            wid = T.warp_id([4], parent="cta")
            lane = T.thread_id([32], parent="warp")

            with T.thread():
                A = T.alloc_buffer((512,), scope="shared", layout=T.SwizzleLayout(3, 3, 3))

                A[0] = 0
    # fmt: on
    verify(test2)


def test_host():
    # fmt: off
    @T.prim_func(tirx=True, check_well_formed=False)
    def test1(A_ptr: T.handle):
        A = T.match_buffer(A_ptr, (16, 16), dtype="float32", align=16)

        A_map: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
        T.call_packed("runtime.cuTensorMapEncodeTiled", A_map, "float32", 2, A.data, 16, 16, 64, 16, 16, 1, 1, 0, 0, 0, 0)

        with T.kernel():
            for blockIdx in T.thread_binding(1, thread="blockIdx.x"):
                for threadIdx in T.thread_binding(128, thread="threadIdx.x"):
                    with T.thread():
                        bar = T.alloc_buffer((1,), "uint64", scope="shared", align=8)
                        phase = T.alloc_buffer((1,), "int32", scope="local")
                        A_smem = T.alloc_buffer((16, 16), "float32", scope="shared", align=128)

                        phase[0] = 0
                        if threadIdx == 0:
                            T.ptx.mbarrier.init(bar.data, 1)
                            T.ptx.fence.proxy("shared")
                            T.ptx.cp_async.bulk.tensor.g2c(2, A_smem.data, bar.data, A_map, 0, 0)
                            T.ptx.mbarrier.arrive.expect_tx(bar.data, 16*16*4)
                        T.ptx.mbarrier.try_wait(bar.data, phase[0])
                        phase[0] = phase[0] ^ 1
                        T.print_buffer(A_smem.data, "float32", False, False, 2, 16*16)
    # fmt: on
    verify(test1)


def test_device_func():
    # fmt: off
    @T.prim_func(tirx=True, check_well_formed=False)
    def test1(A: T.Buffer((128,), "float32")):
        with T.cta():
            thread_id = T.thread_id([128], parent="cta")
            Tx.fill(A, 0.)

    @T.prim_func(tirx=True, check_well_formed=False)
    def test2(A: T.Buffer((128,), "float32")):
        with T.kernel():
            cta_id = T.cta_id([128], parent="kernel")
            thread_id = T.thread_id([128], parent="cta")
            Tx.fill(A, 0.)

    @T.prim_func(tirx=True, check_well_formed=False)
    def test3(A: T.Buffer((128,), "float32")):
        with T.cta():
            thread_id = T.thread_id([128], parent="cta")
            Tx.fill(A, 0.)
        with T.cta():
            thread_id = T.thread_id([128], parent="cta")
            Tx.fill(A, 0.)
    # fmt: on
    verify(test1, device_func=True)
    with pytest.raises(Exception, match="higher than kernel scope"):
        verify(test2, device_func=True)
    with pytest.raises(Exception, match="Only one root scope is allowed in device function"):
        verify(test3, device_func=True)


if __name__ == "__main__":
    test_root_scope()
    test_nested_scope()
    test_scope_slice()
    test_scope_id_consistency()
    test_layout()
    test_host()
    test_device_func()
