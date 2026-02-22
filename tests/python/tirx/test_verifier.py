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

from tvm.script import tirx as Tx
from tvm.tir.analysis import verify_tirx_well_formed as verify


def test_root_scope():
    # fmt: off
    @Tx.prim_func(tirx=True, check_well_formed=False)
    def test1() -> None:
        with Tx.thread():
            pass

    @Tx.prim_func(tirx=True, check_well_formed=False)
    def test2() -> None:
        with Tx.warp():
            with Tx.thread():
                pass

    @Tx.prim_func(tirx=True, check_well_formed=False)
    def test3() -> None:
        with Tx.cta():
            with Tx.warp():
                with Tx.thread():
                    pass

    @Tx.prim_func(tirx=True, check_well_formed=False)
    def test4() -> None:
        with Tx.kernel():
            with Tx.cta():
                with Tx.warp():
                    with Tx.thread():
                        pass

    @Tx.prim_func(tirx=True, check_well_formed=False)
    def test5() -> None:
        with Tx.world():
            with Tx.kernel():
                with Tx.cta():
                    with Tx.warp():
                        with Tx.thread():
                            pass

    @Tx.prim_func(tirx=True, check_well_formed=False)
    def test6() -> None:
        with Tx.world():
            with Tx.kernel():
                with Tx.cta():
                    with Tx.warp():
                        with Tx.cluster():
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
    @Tx.prim_func(tirx=True, check_well_formed=False)
    def test1() -> None:
        with Tx.kernel():
            with Tx.cta():
                with Tx.warp():
                    with Tx.thread():
                        pass
                with Tx.thread():
                    pass

    @Tx.prim_func(tirx=True, check_well_formed=False)
    def test2() -> None:
        with Tx.kernel():
            with Tx.thread():
                with Tx.cta():
                    with Tx.thread():
                        pass

    @Tx.prim_func(tirx=True, check_well_formed=False)
    def test3() -> None:
        with Tx.kernel():
            with Tx.warp():
                with Tx.thread():
                    with Tx.cta():
                        with Tx.thread():
                            pass
    @Tx.prim_func(tirx=True, check_well_formed=False)
    def test4() -> None:
        with Tx.kernel():
            with Tx.thread():
                with Tx.warpgroup():
                    with Tx.warp():
                        with Tx.thread():
                            pass
                with Tx.warpgroup():
                    with Tx.warp():
                        with Tx.thread():
                            pass

    @Tx.prim_func(tirx=True, check_well_formed=False)
    def test5() -> None:
        with Tx.kernel():
            with Tx.world():
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
    @Tx.prim_func(tirx=True, check_well_formed=False)
    def test1() -> None:
        with Tx.kernel():
            with Tx.cta():
                with Tx.warpgroup()[0:1]:
                    with Tx.thread():
                        pass

    @Tx.prim_func(tirx=True, check_well_formed=False)
    def test2() -> None:
        with Tx.kernel():
            with Tx.cta():
                with Tx.warpgroup()[0:3]:
                    with Tx.warpgroup()[1:2]:
                        pass
                    with Tx.warpgroup()[2:3]:
                        with Tx.thread()[Tx.ptx.elect_sync()]:
                            pass
    @Tx.prim_func(tirx=True, check_well_formed=False)
    def test3() -> None:
        with Tx.kernel():
            with Tx.cta():
                with Tx.warpgroup()[0:3]:
                    with Tx.warpgroup()[1:5]:
                        pass
    @Tx.prim_func(tirx=True, check_well_formed=False)
    def test4() -> None:
        with Tx.kernel():
            with Tx.cta():
                with Tx.thread()[0:3]:
                    with Tx.thread()[Tx.ptx.elect_sync()]:
                        pass
    @Tx.prim_func(tirx=True, check_well_formed=False)
    def test5() -> None:
        with Tx.kernel():
            with Tx.cta():
                with Tx.thread()[Tx.ptx.elect_sync()]:
                    with Tx.thread()[Tx.ptx.elect_sync()]:
                        pass
    @Tx.prim_func(tirx=True, check_well_formed=False)
    def test6() -> None:
        with Tx.kernel():
            with Tx.cta():
                with Tx.thread()[Tx.ptx.elect_sync()]:
                    with Tx.thread()[Tx.int32(0)]:
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
    @Tx.prim_func(tirx=True, check_well_formed=False)
    def test1():
        with Tx.kernel():
            bx, by, bz = Tx.cta_id([3, 4, 5], parent="kernel")
            warp_id = Tx.warp_id([4], parent="cta")
            tx = Tx.thread_id([128], parent="cta")

            with Tx.cta():
                Tx.attr({"tirx.scope_partition": True})
                with Tx.thread()[0:30]:
                    Tx.evaluate(tx)
                with Tx.thread()[30:128]:
                    Tx.evaluate(tx)
    @Tx.prim_func(tirx=True, check_well_formed=False)
    def test2():
        with Tx.kernel():
            bx, by, bz = Tx.cta_id([3, 4, 5], parent="kernel")
            warp_id = Tx.warp_id([4], parent="cta")
            tx = Tx.thread_id([128], parent="cta")

            with Tx.cta():
                Tx.attr({"tirx.scope_partition": True})
                with Tx.thread()[0:30]:
                    Tx.evaluate(tx)
                with Tx.thread():
                    Tx.evaluate(tx)
    @Tx.prim_func(tirx=True, check_well_formed=False)
    def test3():
        with Tx.kernel():
            bx, by, bz = Tx.cta_id([3, 4, 5], parent="kernel")
            warp_id = Tx.warp_id([4], parent="cta")
            tx = Tx.thread_id([128], parent="cta")

            with Tx.thread():
                Tx.attr({"tirx.scope_partition": True})
                with Tx.thread()[0:30]:
                    Tx.evaluate(tx)
                Tx.evaluate(tx)
    @Tx.prim_func(tirx=True, check_well_formed=False)
    def test4():
        with Tx.kernel():
            bx, by, bz = Tx.cta_id([3, 4, 5], parent="kernel")
            warp_id = Tx.warp_id([4], parent="cta")
            tx = Tx.thread_id([128], parent="cta")

            with Tx.thread():
                Tx.attr({"tirx.scope_partition": True})
                with Tx.thread()[0:30]:
                    Tx.evaluate(tx)
                with Tx.warp()[30:128]:
                    Tx.evaluate(tx)
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
    @Tx.prim_func(tirx=True, check_well_formed=False)
    def test1():
        with Tx.kernel():
            bx = Tx.cta_id([32], parent="kernel")
            wid = Tx.warp_id([4], parent="cta")
            lane = Tx.thread_id([32], parent="warp")

            with Tx.thread():
                pass

    @Tx.prim_func(tirx=True, check_well_formed=False)
    def test2():
        with Tx.kernel():
            bx = Tx.cta_id([32], parent="kernel")
            wid = Tx.warp_id([4], parent="cta")
            lane = Tx.thread_id([32], parent="warp")
            tid = Tx.thread_id([128], parent="cta")

            with Tx.thread():
                pass

    @Tx.prim_func(tirx=True, check_well_formed=False)
    def test3():
        with Tx.kernel():
            bx = Tx.cta_id([32], parent="kernel")
            wid = Tx.warp_id([2], parent="cta")
            lane = Tx.thread_id([32], parent="warp")
            tid = Tx.thread_id([128], parent="cta")

            with Tx.thread():
                pass

    @Tx.prim_func(tirx=True, check_well_formed=False)
    def test4():
        with Tx.kernel():
            bx, by, bz = Tx.cta_id([8, 10, 12], parent="kernel")
            cbx, cby, cbz = Tx.cta_id([2, 2, 1], parent="cluster")
            clx, cly, clz = Tx.cluster_id([4, 5, 12], parent="kernel")
            with Tx.cta():
                with Tx.warp():
                    with Tx.thread():
                        Tx.evaluate(bx + by + bz)
                        Tx.evaluate(cbx + cby + cbz)
                        Tx.evaluate(clx + cly + clz)

    @Tx.prim_func(tirx=True, check_well_formed=False)
    def test5():
        with Tx.kernel():
            bx, by, bz = Tx.cta_id([8, 10, 12], parent="kernel")
            cbx, cby, cbz = Tx.cta_id([2, 2, 1], parent="cluster")
            clx, cly, clz = Tx.cluster_id([3, 5, 12], parent="kernel")
            with Tx.cta():
                with Tx.warp():
                    with Tx.thread():
                        Tx.evaluate(bx + by + bz)
                        Tx.evaluate(cbx + cby + cbz)
                        Tx.evaluate(clx + cly + clz)

    @Tx.prim_func(tirx=True, check_well_formed=False)
    def test6():
        with Tx.kernel():
            clx, cly, clz = Tx.cluster_id([4, 5, 12], parent="kernel")
            bx, by, bz = Tx.cta_id([8, 10, 12], parent="kernel")
            with Tx.cluster():
                cbx, cby, cbz = Tx.cta_id([2, 2, 1], parent="cluster")
                with Tx.warp():
                    with Tx.thread():
                        Tx.evaluate(bx + by + bz)
                        Tx.evaluate(cbx + cby + cbz)
                        Tx.evaluate(clx + cly + clz)

    @Tx.prim_func(tirx=True, check_well_formed=False)
    def test7():
        with Tx.kernel():
            clx, cly, clz = Tx.cluster_id([3, 5, 12], parent="kernel")
            bx, by, bz = Tx.cta_id([8, 10, 12], parent="kernel")
            with Tx.cluster():
                cbx, cby, cbz = Tx.cta_id([2, 2, 1], parent="cluster")
                with Tx.warp():
                    with Tx.thread():
                        Tx.evaluate(bx + by + bz)
                        Tx.evaluate(cbx + cby + cbz)
                        Tx.evaluate(clx + cly + clz)

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
    @Tx.prim_func(tirx=True, check_well_formed=False)
    def test1():
        with Tx.kernel():
            bx = Tx.cta_id([32], parent="kernel")
            wid = Tx.warp_id([4], parent="cta")
            lane = Tx.thread_id([32], parent="warp")

            with Tx.thread():
                A = Tx.alloc_buffer((2,), layout=Tx.TileLayout(Tx.S[2, 1]))

                A[0] = 0
    # fmt: on
    verify(test1)

    ### SwizzleLayout
    # fmt: off
    @Tx.prim_func(tirx=True, check_well_formed=False)
    def test2():
        with Tx.kernel():
            bx = Tx.cta_id([32], parent="kernel")
            wid = Tx.warp_id([4], parent="cta")
            lane = Tx.thread_id([32], parent="warp")

            with Tx.thread():
                A = Tx.alloc_buffer((512,), scope="shared", layout=Tx.SwizzleLayout(3, 3, 3))

                A[0] = 0
    # fmt: on
    verify(test2)


def test_host():
    # fmt: off
    @Tx.prim_func(tirx=True, check_well_formed=False)
    def test1(A_ptr: Tx.handle):
        A = Tx.match_buffer(A_ptr, (16, 16), dtype="float32", align=16)

        A_map: Tx.let[Tx.handle("tensormap")] = Tx.tvm_stack_alloca("tensormap", 1)
        Tx.call_packed("runtime.cuTensorMapEncodeTiled", A_map, "float32", 2, A.data, 16, 16, 64, 16, 16, 1, 1, 0, 0, 0, 0)

        with Tx.kernel():
            for blockIdx in Tx.thread_binding(1, thread="blockIdx.x"):
                for threadIdx in Tx.thread_binding(128, thread="threadIdx.x"):
                    with Tx.thread():
                        bar = Tx.alloc_buffer((1,), "uint64", scope="shared", align=8)
                        phase = Tx.alloc_buffer((1,), "int32", scope="local")
                        A_smem = Tx.alloc_buffer((16, 16), "float32", scope="shared", align=128)

                        phase[0] = 0
                        if threadIdx == 0:
                            Tx.ptx.mbarrier.init(bar.data, 1)
                            Tx.ptx.fence.proxy("shared")
                            Tx.ptx.cp_async.bulk.tensor.g2c(2, A_smem.data, bar.data, A_map, 0, 0)
                            Tx.ptx.mbarrier.arrive.expect_tx(bar.data, 16*16*4)
                        Tx.ptx.mbarrier.try_wait(bar.data, phase[0])
                        phase[0] = phase[0] ^ 1
                        Tx.print_buffer(A_smem.data, "float32", False, False, 2, 16*16)
    # fmt: on
    verify(test1)


def test_device_func():
    # fmt: off
    @Tx.prim_func(tirx=True, check_well_formed=False)
    def test1(A: Tx.Buffer((128,), "float32")):
        with Tx.cta():
            thread_id = Tx.thread_id([128], parent="cta")
            Tx.fill(A, 0.)

    @Tx.prim_func(tirx=True, check_well_formed=False)
    def test2(A: Tx.Buffer((128,), "float32")):
        with Tx.kernel():
            cta_id = Tx.cta_id([128], parent="kernel")
            thread_id = Tx.thread_id([128], parent="cta")
            Tx.fill(A, 0.)

    @Tx.prim_func(tirx=True, check_well_formed=False)
    def test3(A: Tx.Buffer((128,), "float32")):
        with Tx.cta():
            thread_id = Tx.thread_id([128], parent="cta")
            Tx.fill(A, 0.)
        with Tx.cta():
            thread_id = Tx.thread_id([128], parent="cta")
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
