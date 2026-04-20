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
from tvm.script import tirx as Tx
from tvm.tirx.function import PrimFunc
from tvm.tirx.layout import laneid, warpid, wg_local_layout
from tvm.tirx.stmt import ExecScopeStmt
from tvm.tirx.stmt_functor import post_order_visit
from tvm.tirx.transform import LowerTIRx


def _contains_exec_scope(mod):
    found = [False]

    def _visit(node):
        if isinstance(node, ExecScopeStmt):
            found[0] = True

    for _gv, base_func in mod.functions.items():
        if isinstance(base_func, PrimFunc):
            post_order_visit(base_func.body, _visit)
    return found[0]


def compare(before, after, transform):
    """Compare lowered output against expected ``after`` IR."""
    if isinstance(before, PrimFunc):
        before = tvm.IRModule({"main": before})
    if isinstance(after, PrimFunc):
        after = tvm.IRModule({"main": after})
    assert isinstance(before, tvm.IRModule)
    assert isinstance(after, tvm.IRModule)

    with tvm.target.Target("cuda"):
        lowered = transform()(before)
        lowered.show()
        assert not _contains_exec_scope(lowered)
        tvm.ir.assert_structural_equal(lowered, after, map_free_vars=False)


L_LANE = Tx.TileLayout(Tx.S[32 : 1 @ laneid])


def test_lower_view_get():
    # fmt: off
    @Tx.prim_func(private=True, tirx=True)
    def before1(in_buf: Tx.Buffer(64, "float32"), out: Tx.Buffer(64, "float32")) -> None:
        with Tx.kernel():
            bx, by, bz = Tx.cta_id([1, 1, 1], parent="kernel")
            Tx.warp_id([1], parent="cta")
            lane_id = Tx.thread_id([32], parent="warp")

            with Tx.thread():
                A = Tx.alloc_buffer([2], dtype="float16", scope="local", layout=Tx.TileLayout(Tx.S[2 : 1]))  # noqa: E501
                B_layout = A.layout.tile(L_LANE, (32,), (2,))
                # load in_buf into A
                with Tx.warp():
                    # warp view of this load
                    B = A.view(64, layout=B_layout)
                    # B[i] = in_buf[i]
                    with Tx.thread():
                        # done by each thread
                        A_local = B.local(2)
                        for i in Tx.vectorized(2):
                            A_local[i] = Tx.float32(in_buf[lane_id * 2 + i])
                # write A into out
                with Tx.warp():
                    # warp view of this write
                    B = A.view(64, layout=B_layout)
                    # out[i] = B[i]
                    with Tx.thread():
                        # done by each thread
                        A_local = B.local(2)
                        for i in Tx.vectorized(2):
                            out[lane_id * 2 + i] = Tx.float32(A_local[i])

    @Tx.prim_func(private=True, tirx=True)
    def after1(in_buf_handle: Tx.handle, out_handle: Tx.handle):
        in_buf = Tx.match_buffer(in_buf_handle, (64,), layout=None)
        out = Tx.match_buffer(out_handle, (64,), layout=None)
        out_1 = Tx.decl_buffer((64,), data=out.data, layout=None)
        in_buf_1 = Tx.decl_buffer((64,), data=in_buf.data, layout=None)
        blockIdx_x = Tx.launch_thread("blockIdx.x", 1)
        threadIdx_x = Tx.launch_thread("threadIdx.x", 32)
        blockIdx_y = Tx.launch_thread("blockIdx.y", 1)
        blockIdx_z = Tx.launch_thread("blockIdx.z", 1)
        warp_id_in_cta: Tx.let[Tx.int32] = Tx.tvm_warp_shuffle(Tx.uint32(4294967295), threadIdx_x // 32, 0, 32, 32)  # noqa: E501
        bx: Tx.let[Tx.int32] = blockIdx_x  # noqa: F841
        by: Tx.let[Tx.int32] = blockIdx_y  # noqa: F841
        bz: Tx.let[Tx.int32] = blockIdx_z  # noqa: F841
        v: Tx.let[Tx.int32] = warp_id_in_cta
        lane_id: Tx.let[Tx.int32] = threadIdx_x % 32  # noqa: F841
        Tx.evaluate(v)
        A = Tx.alloc_local((2,), "float16", layout=None)
        B = Tx.decl_buffer((64,), "float16", data=A.data, scope="local", layout=None)  # noqa: F841
        A_local = Tx.decl_buffer((2,), "float16", data=A.data, scope="local", layout=None)
        for i in Tx.vectorized(2):
            A_local[i] = Tx.Cast("float16", in_buf_1[threadIdx_x * 2 + i])
        B_1 = Tx.decl_buffer((64,), "float16", data=A.data, scope="local", layout=None)  # noqa: F841
        A_local_1 = Tx.decl_buffer((2,), "float16", data=A.data, scope="local", layout=None)
        for i in Tx.vectorized(2):
            out_1[threadIdx_x * 2 + i] = Tx.Cast("float32", A_local_1[i])
    # fmt: on

    compare(before1, after1, LowerTIRx)

    # fmt: off
    @Tx.prim_func(private=True, tirx=True)
    def before2(in_buf: Tx.Buffer((16, 16), "float32"), out: Tx.Buffer((16, 16), "float32")) -> None:  # noqa: E501
        with Tx.kernel():
            bx, by, bz = Tx.cta_id([1, 1, 1], parent="kernel")
            Tx.warp_id([1], parent="cta")
            lane_id = Tx.thread_id([32], parent="warp")

            with Tx.thread():
                atom = Tx.TileLayout(Tx.S[(1, 2) : (2, 1)])
                tile = Tx.TileLayout(Tx.S[(2, 2) : (2, 1)])
                warp_atom = atom.tile(L_LANE, (8, 4), (1, 2))

                A = Tx.alloc_buffer([4, 2], dtype="float32", scope="local", layout=atom.tile(tile, (2, 2), (1, 2)))  # noqa: E501
                B_layout = warp_atom.tile(tile, (2, 2), (8, 8))

                # load in_buf into A
                with Tx.warp():
                    # warp view of this load
                    B = A.view(16, 16, layout=B_layout)
                    with Tx.thread():
                        # done by each thread
                        A_local = B.local(2, 2, 2)
                        for i in Tx.unroll(4):
                            for j in Tx.vectorized(2):
                                A_local[i // 2, i % 2, j] = in_buf[i // 2 * 8 + lane_id // 4, i % 2 * 8 + lane_id % 4 + j]  # noqa: E501
                # write A into out
                with Tx.warp():
                    # warp view of this write
                    B = A.view(16, 16, layout=B_layout)
                    with Tx.thread():
                        # done by each thread
                        A_local = B.local(8)
                        for i in Tx.vectorized(2):
                            out[lane_id // 4 * 8 + i // 2 * 8 + lane_id % 4, lane_id % 4 * 2 + i % 2] = A_local[i]  # noqa: E501

    @Tx.prim_func(private=True, tirx=True)
    def after2(in_buf_handle: Tx.handle, out_handle: Tx.handle):
        in_buf = Tx.match_buffer(in_buf_handle, (16, 16), layout=None)
        out = Tx.match_buffer(out_handle, (16, 16), layout=None)
        out_1 = Tx.decl_buffer((256,), data=out.data, layout=None)
        in_buf_1 = Tx.decl_buffer((256,), data=in_buf.data, layout=None)
        blockIdx_x = Tx.launch_thread("blockIdx.x", 1)
        threadIdx_x = Tx.launch_thread("threadIdx.x", 32)
        blockIdx_y = Tx.launch_thread("blockIdx.y", 1)
        blockIdx_z = Tx.launch_thread("blockIdx.z", 1)
        warp_id_in_cta: Tx.let[Tx.int32] = Tx.tvm_warp_shuffle(Tx.uint32(4294967295), threadIdx_x // 32, 0, 32, 32)  # noqa: E501
        bx: Tx.let[Tx.int32] = blockIdx_x  # noqa: F841
        by: Tx.let[Tx.int32] = blockIdx_y  # noqa: F841
        bz: Tx.let[Tx.int32] = blockIdx_z  # noqa: F841
        v: Tx.let[Tx.int32] = warp_id_in_cta
        lane_id: Tx.let[Tx.int32] = threadIdx_x % 32  # noqa: F841
        Tx.evaluate(v)
        A = Tx.alloc_local((8,), layout=None)
        B = Tx.decl_buffer((256,), data=A.data, scope="local", layout=None)  # noqa: F841
        A_local = Tx.decl_buffer((8,), data=A.data, scope="local", layout=None)
        for i in Tx.unroll(4):
            for j in Tx.vectorized(2):
                A_local[i * 2 + j] = in_buf_1[i // 2 * 128 + threadIdx_x // 4 * 16 + i % 2 * 8 + j + threadIdx_x % 4]  # noqa: E501
        B_1 = Tx.decl_buffer((256,), data=A.data, scope="local", layout=None)  # noqa: F841
        A_local_1 = Tx.decl_buffer((8,), data=A.data, scope="local", layout=None)
        for i in Tx.vectorized(2):
            out_1[threadIdx_x // 4 * 128 + threadIdx_x % 4 * 18 + i] = A_local_1[i]
    # fmt: on

    compare(before2, after2, LowerTIRx)

    # fmt: off
    @Tx.prim_func(private=True, tirx=True)
    def before3_wgmma_layout(in_buf: Tx.Buffer((128, 128), "float32"), out: Tx.Buffer((128, 128), "float32")) -> None:  # noqa: E501
        with Tx.kernel():
            bx, by, bz = Tx.cta_id([1, 1, 1], parent="kernel")
            wg_id = Tx.warpgroup_id([2], parent="cta")
            warp_id_in_wg = Tx.warp_id([4], parent="warpgroup")
            lane_id = Tx.thread_id([32], parent="warp")

            with Tx.thread():
                # shard from thread to warp
                atom = Tx.TileLayout(Tx.S[1, 2])
                warp_atom = atom.tile(L_LANE, (8, 4), (1, 2))
                # tile
                tile = Tx.TileLayout(Tx.S[(2, 128 // 8) : (1, 2)]) # column-major
                warp_layout = warp_atom.tile(tile, (2, 128 // 8), (8, 8))
                # shard from warp to cta
                L_warp = Tx.TileLayout(Tx.S[8 : 1@warpid])
                layout = warp_layout.tile(L_warp, (8, 1), (16, 128))
                # alloc
                acc = Tx.alloc_buffer([64,], dtype="float32", scope="local", layout=atom.tile(tile, (2, 128 // 8), (1, 2)))  # noqa: E501

                # load in_buf into acc
                with Tx.cta():
                    # cta view of this load
                    A = acc.view(128, 128, layout=layout)
                    with Tx.thread():
                        # done by each thread
                        acc_local = A.local(16, 2, 2, layout=atom.tile(tile, (2, 128 // 8), (1, 2)))
                        for i in Tx.serial(128 // 8):
                            for j in Tx.unroll(2):
                                for vec in Tx.vectorized(2):
                                    acc_local[i, j, vec] = in_buf[wg_id * 64 + warp_id_in_wg * 16 + j * 8 + lane_id // 4, i * 8 + lane_id % 4 * 2 + vec]  # noqa: E501

                # write acc into out
                with Tx.cta():
                    # cta view of this write
                    A = acc.view(128, 128, layout=layout)
                    with Tx.thread():
                        # done by each thread
                        acc_local = A.local(64, layout=atom.tile(tile, (2, 128 // 8), (1, 2)))
                        for i in Tx.serial(128 // 8):
                            for j in Tx.unroll(2):
                                for vec in Tx.vectorized(2):
                                    out[wg_id * 64 + warp_id_in_wg * 16 + j * 8 + lane_id // 4, i * 8 + lane_id % 4 * 2 + vec] = acc_local[i * 4 + j * 2 + vec]  # noqa: E501

    @Tx.prim_func(private=True, tirx=True)
    def after3_wgmma_layout(in_buf_handle: Tx.handle, out_handle: Tx.handle):
        in_buf = Tx.match_buffer(in_buf_handle, (128, 128), layout=None)
        out = Tx.match_buffer(out_handle, (128, 128), layout=None)
        out_1 = Tx.decl_buffer((16384,), data=out.data, layout=None)
        in_buf_1 = Tx.decl_buffer((16384,), data=in_buf.data, layout=None)
        blockIdx_x = Tx.launch_thread("blockIdx.x", 1)
        threadIdx_x = Tx.launch_thread("threadIdx.x", 256)
        blockIdx_y = Tx.launch_thread("blockIdx.y", 1)
        blockIdx_z = Tx.launch_thread("blockIdx.z", 1)
        warp_id_in_cta: Tx.let[Tx.int32] = Tx.tvm_warp_shuffle(Tx.uint32(4294967295), threadIdx_x // 32, 0, 32, 32)  # noqa: E501
        bx: Tx.let[Tx.int32] = blockIdx_x  # noqa: F841
        by: Tx.let[Tx.int32] = blockIdx_y  # noqa: F841
        bz: Tx.let[Tx.int32] = blockIdx_z  # noqa: F841
        wg_id: Tx.let[Tx.int32] = warp_id_in_cta // 4  # noqa: F841
        warp_id_in_wg: Tx.let[Tx.int32] = warp_id_in_cta % 4  # noqa: F841
        lane_id: Tx.let[Tx.int32] = threadIdx_x % 32  # noqa: F841
        acc = Tx.alloc_local((64,), layout=None)
        B = Tx.decl_buffer((16384,), data=acc.data, scope="local", layout=None)  # noqa: F841
        acc_local = Tx.decl_buffer((64,), data=acc.data, scope="local", layout=None)
        for i in range(16):
            for j in Tx.unroll(2):
                for vec in Tx.vectorized(2):
                    acc_local[i % 8 * 8 + j * 4 + i // 8 * 2 + vec] = in_buf_1[warp_id_in_cta * 2048 + j * 1024 + threadIdx_x % 32 // 4 * 128 + i * 8 + threadIdx_x % 4 * 2 + vec]  # noqa: E501
        B_1 = Tx.decl_buffer((16384,), data=acc.data, scope="local", layout=None)  # noqa: F841
        acc_local_1 = Tx.decl_buffer((64,), data=acc.data, scope="local", layout=None)
        for i in range(16):
            for j in Tx.unroll(2):
                for vec in Tx.vectorized(2):
                    out_1[warp_id_in_cta * 2048 + j * 1024 + threadIdx_x % 32 // 4 * 128 + i * 8 + threadIdx_x % 4 * 2 + vec] = acc_local_1[i % 8 * 8 + j * 4 + i // 8 * 2 + vec]  # noqa: E501
    # fmt: on

    compare(before3_wgmma_layout, after3_wgmma_layout, LowerTIRx)

    # fmt: off
    @Tx.prim_func(private=True, tirx=True)
    def before4_multi_view_get(in_buf: Tx.Buffer(64, "float32"), out: Tx.Buffer(64, "float32")) -> None:  # noqa: E501
        with Tx.kernel():
            bx, by, bz = Tx.cta_id([1, 1, 1], parent="kernel")
            Tx.warp_id([1], parent="cta")
            lane_id = Tx.thread_id([32], parent="warp")

            with Tx.thread():
                A = Tx.alloc_buffer([2], dtype="float16", scope="local", layout=Tx.TileLayout(Tx.S[2 : 1]))  # noqa: E501
                B_layout = A.layout.tile(L_LANE, (32,), (2,))
                with Tx.warp():
                    # warp view of this load
                    B = A.view(64, layout=B_layout) # TODO(@bohan): consider making view API directly accepts shard parameters  # noqa: E501
                    B_1 = A.view(64, layout=B_layout) # TODO(@bohan): consider making view API directly accepts shard parameters  # noqa: E501
                    # B[i] = in_buf[i]
                    with Tx.thread():
                        # done by each thread
                        A_local = B.local(2)
                        A_local[0] = Tx.float32(in_buf[lane_id * 2])
                        # done by each thread
                        A_local_1 = B_1.local(2)
                        A_local_1[1] = Tx.float32(in_buf[lane_id * 2 + 1])
                """
                write A into out
                """
                with Tx.warp():
                    # warp view of this write
                    B = A.view(64, layout=B_layout)
                    B_1 = A.view(64, layout=B_layout)
                    # out[i] = B[i]
                    with Tx.thread():
                        # done by each thread
                        A_local = B.local(2)
                        out[lane_id * 2] = Tx.float32(A_local[0])
                        # done by each thread
                        A_local_1 = B_1.local(2)
                        out[lane_id * 2 + 1] = Tx.float32(A_local_1[1])

    @Tx.prim_func(private=True, tirx=True)
    def after4_multi_view_get(in_buf_handle: Tx.handle, out_handle: Tx.handle):
        in_buf = Tx.match_buffer(in_buf_handle, (64,), layout=None)
        out = Tx.match_buffer(out_handle, (64,), layout=None)
        out_1 = Tx.decl_buffer((64,), data=out.data, layout=None)
        in_buf_1 = Tx.decl_buffer((64,), data=in_buf.data, layout=None)
        blockIdx_x = Tx.launch_thread("blockIdx.x", 1)
        threadIdx_x = Tx.launch_thread("threadIdx.x", 32)
        blockIdx_y = Tx.launch_thread("blockIdx.y", 1)
        blockIdx_z = Tx.launch_thread("blockIdx.z", 1)
        warp_id_in_cta: Tx.let[Tx.int32] = Tx.tvm_warp_shuffle(Tx.uint32(4294967295), threadIdx_x // 32, 0, 32, 32)  # noqa: E501
        bx: Tx.let[Tx.int32] = blockIdx_x  # noqa: F841
        by: Tx.let[Tx.int32] = blockIdx_y  # noqa: F841
        bz: Tx.let[Tx.int32] = blockIdx_z  # noqa: F841
        v: Tx.let[Tx.int32] = warp_id_in_cta
        lane_id: Tx.let[Tx.int32] = threadIdx_x % 32  # noqa: F841
        Tx.evaluate(v)
        A = Tx.alloc_local((2,), "float16", layout=None)
        B = Tx.decl_buffer((64,), "float16", data=A.data, scope="local", layout=None)  # noqa: F841
        B_1 = Tx.decl_buffer((64,), "float16", data=A.data, scope="local", layout=None)  # noqa: F841
        A_local = Tx.decl_buffer((2,), "float16", data=A.data, scope="local", layout=None)
        A_local[0] = Tx.Cast("float16", in_buf_1[threadIdx_x * 2])
        A_local_1 = Tx.decl_buffer((2,), "float16", data=A.data, scope="local", layout=None)
        A_local_1[1] = Tx.Cast("float16", in_buf_1[threadIdx_x * 2 + 1])
        B_2 = Tx.decl_buffer((64,), "float16", data=A.data, scope="local", layout=None)  # noqa: F841
        B_3 = Tx.decl_buffer((64,), "float16", data=A.data, scope="local", layout=None)  # noqa: F841
        A_local_2 = Tx.decl_buffer((2,), "float16", data=A.data, scope="local", layout=None)
        out_1[threadIdx_x * 2] = Tx.Cast("float32", A_local_2[0])
        A_local_3 = Tx.decl_buffer((2,), "float16", data=A.data, scope="local", layout=None)
        out_1[threadIdx_x * 2 + 1] = Tx.Cast("float32", A_local_3[1])
    # fmt: on

    compare(before4_multi_view_get, after4_multi_view_get, LowerTIRx)


def test_lower_scope_id():
    # fmt: off
    @Tx.prim_func(private=True, tirx=True)
    def before1() -> None:
        with Tx.kernel():
            bx, by, bz = Tx.cta_id([3, 4, 5], parent="kernel")
            tx = Tx.thread_id([32], parent="cta")

            with Tx.thread():
                Tx.evaluate(bx + by + bz + tx)

    @Tx.prim_func(private=True, tirx=True)
    def after1() -> None:
        blockIdx_x = Tx.launch_thread("blockIdx.x", 3)
        threadIdx_x = Tx.launch_thread("threadIdx.x", 32)
        blockIdx_y = Tx.launch_thread("blockIdx.y", 4)
        blockIdx_z = Tx.launch_thread("blockIdx.z", 5)
        warp_id_in_cta: Tx.let[Tx.int32] = Tx.tvm_warp_shuffle(Tx.uint32(4294967295), threadIdx_x // 32, 0, 32, 32)  # noqa: F841,E501
        bx: Tx.let[Tx.int32] = blockIdx_x
        by: Tx.let[Tx.int32] = blockIdx_y
        bz: Tx.let[Tx.int32] = blockIdx_z
        tx: Tx.let[Tx.int32] = threadIdx_x
        Tx.evaluate(bx + by + bz + tx)
    # fmt: on
    compare(before1, after1, LowerTIRx)

    # fmt: off
    @Tx.prim_func(private=True, tirx=True)
    def before2() -> None:
        with Tx.kernel():
            cbx, cby, cbz = Tx.cta_id([2, 2, 2], parent="cluster")
            bx, by, bz = Tx.cta_id([8, 8, 8], parent="kernel")
            warp_id = Tx.warp_id([4], parent="cta")
            lane_id = Tx.thread_id([32], parent="warp")

            with Tx.thread():
                Tx.evaluate(bx + by + bz + warp_id + lane_id + cbx + cby + cbz)

    @Tx.prim_func(private=True, tirx=True)
    def after2() -> None:
        clusterCtaIdx_x = Tx.launch_thread("clusterCtaIdx.x", 2)
        blockIdx_z = Tx.launch_thread("blockIdx.z", 8)
        clusterCtaIdx_y = Tx.launch_thread("clusterCtaIdx.y", 2)
        clusterCtaIdx_z = Tx.launch_thread("clusterCtaIdx.z", 2)
        blockIdx_x = Tx.launch_thread("blockIdx.x", 8)
        threadIdx_x = Tx.launch_thread("threadIdx.x", 128)
        blockIdx_y = Tx.launch_thread("blockIdx.y", 8)
        warp_id_in_cta: Tx.let[Tx.int32] = Tx.tvm_warp_shuffle(Tx.uint32(4294967295), threadIdx_x // 32, 0, 32, 32)  # noqa: E501
        cbx: Tx.let[Tx.int32] = clusterCtaIdx_x
        cby: Tx.let[Tx.int32] = clusterCtaIdx_y
        cbz: Tx.let[Tx.int32] = clusterCtaIdx_z
        bx: Tx.let[Tx.int32] = blockIdx_x
        by: Tx.let[Tx.int32] = blockIdx_y
        bz: Tx.let[Tx.int32] = blockIdx_z
        warp_id: Tx.let[Tx.int32] = warp_id_in_cta
        lane_id: Tx.let[Tx.int32] = threadIdx_x % 32
        Tx.evaluate(bx + by + bz + warp_id + lane_id + cbx + cby + cbz)

    # fmt: on
    compare(before2, after2, LowerTIRx)

    # fmt: off
    @Tx.prim_func(private=True, tirx=True)
    def before3() -> None:
        with Tx.kernel():
            bx, by, bz = Tx.cta_id([8, 10, 12], parent="kernel")
            cbx, cby, cbz = Tx.cta_id([2, 2, 1], parent="cluster")
            clx, cly, clz = Tx.cluster_id([4, 5, 12], parent="kernel")
            wg_id = Tx.warpgroup_id([3], parent="cta")
            warp_id = Tx.warp_id([4], parent="warpgroup")
            lane_id = Tx.thread_id([32], parent="warp")
            tid_in_wg = Tx.thread_id([128], parent="warpgroup")
            with Tx.cta():
                with Tx.warpgroup():
                    with Tx.thread():
                        Tx.evaluate(bx + by + bz)
                        Tx.evaluate(cbx + cby + cbz)
                        Tx.evaluate(clx + cly + clz)
                        Tx.evaluate(wg_id + warp_id + lane_id + tid_in_wg)

    @Tx.prim_func(private=True, tirx=True)
    def after3() -> None:
        clusterCtaIdx_x = Tx.launch_thread("clusterCtaIdx.x", 2)
        blockIdx_z = Tx.launch_thread("blockIdx.z", 12)
        clusterCtaIdx_y = Tx.launch_thread("clusterCtaIdx.y", 2)
        clusterCtaIdx_z = Tx.launch_thread("clusterCtaIdx.z", 1)
        blockIdx_x = Tx.launch_thread("blockIdx.x", 8)
        threadIdx_x = Tx.launch_thread("threadIdx.x", 384)
        blockIdx_y = Tx.launch_thread("blockIdx.y", 10)
        warp_id_in_cta: Tx.let[Tx.int32] = Tx.tvm_warp_shuffle(Tx.uint32(4294967295), threadIdx_x // 32, 0, 32, 32)  # noqa: E501
        bx: Tx.let[Tx.int32] = blockIdx_x
        by: Tx.let[Tx.int32] = blockIdx_y
        bz: Tx.let[Tx.int32] = blockIdx_z
        cbx: Tx.let[Tx.int32] = clusterCtaIdx_x
        cby: Tx.let[Tx.int32] = clusterCtaIdx_y
        cbz: Tx.let[Tx.int32] = clusterCtaIdx_z
        clx: Tx.let[Tx.int32] = Tx.ptx.fetch_register(32, "clusterid.x")
        cly: Tx.let[Tx.int32] = Tx.ptx.fetch_register(32, "clusterid.y")
        clz: Tx.let[Tx.int32] = Tx.ptx.fetch_register(32, "clusterid.z")
        wg_id: Tx.let[Tx.int32] = warp_id_in_cta // 4
        warp_id: Tx.let[Tx.int32] = warp_id_in_cta % 4
        lane_id: Tx.let[Tx.int32] = threadIdx_x % 32
        tid_in_wg: Tx.let[Tx.int32] = threadIdx_x % 128
        Tx.evaluate(bx + by + bz)
        Tx.evaluate(cbx + cby + cbz)
        Tx.evaluate(clx + cly + clz)
        Tx.evaluate(wg_id + warp_id + lane_id + tid_in_wg)
    # fmt: on

    compare(before3, after3, LowerTIRx)


def test_lower_scope_id2():
    # fmt: off
    @Tx.inline
    def func(warp_id, tx):
        with Tx.cta():
            wg_id = Tx.warpgroup_id([2], parent="cta")
            with Tx.thread():
                Tx.evaluate(wg_id + warp_id + tx)

    @Tx.prim_func(private=True, tirx=True)
    def before():
        with Tx.kernel():
            bx, by, bz = Tx.cta_id([3, 4, 5], parent="kernel")
            warp_id = Tx.warp_id([8], parent="cta")
            tx = Tx.thread_id([256], parent="cta")

            func(warp_id, tx)
    # fmt: on

    # fmt: off
    @Tx.prim_func(private=True, tirx=True)
    def after():
        blockIdx_x = Tx.launch_thread("blockIdx.x", 3)
        threadIdx_x = Tx.launch_thread("threadIdx.x", 256)
        blockIdx_y = Tx.launch_thread("blockIdx.y", 4)
        blockIdx_z = Tx.launch_thread("blockIdx.z", 5)
        warp_id_in_cta: Tx.let[Tx.int32] = Tx.tvm_warp_shuffle(Tx.uint32(4294967295), threadIdx_x // 32, 0, 32, 32)  # noqa: E501
        wg_id: Tx.let[Tx.int32] = warp_id_in_cta // 4
        bx: Tx.let[Tx.int32] = blockIdx_x  # noqa: F841
        by: Tx.let[Tx.int32] = blockIdx_y  # noqa: F841
        bz: Tx.let[Tx.int32] = blockIdx_z  # noqa: F841
        warp_id: Tx.let[Tx.int32] = warp_id_in_cta
        tx: Tx.let[Tx.int32] = threadIdx_x
        Tx.evaluate(wg_id + warp_id + tx)
    # fmt: on

    compare(before, after, LowerTIRx)


def test_lower_scope_id3():
    # fmt: off
    @Tx.prim_func(private=True, tirx=True)
    def before():
        with Tx.kernel():
            bx, by, bz = Tx.cta_id([3, 4, 5], parent="kernel")
            warp_id = Tx.warp_id([4], parent="cta")
            tx = Tx.thread_id([128], parent="cta")

            with Tx.cta():
                with Tx.thread():
                    Tx.evaluate(bx + by + bz + warp_id + tx)
        with Tx.kernel():
            bx, by, bz = Tx.cta_id([6, 7, 8], parent="kernel")
            warp_id = Tx.warp_id([8], parent="cta")
            tx = Tx.thread_id([256], parent="cta")

            with Tx.cta():
                with Tx.thread():
                    Tx.evaluate(bx + by + bz + warp_id + tx)
    # fmt: on

    # fmt: off
    @Tx.prim_func(private=True, tirx=True)
    def after():
        with Tx.launch_thread("blockIdx.x", 3) as blockIdx_x:
            threadIdx_x = Tx.launch_thread("threadIdx.x", 128)
            blockIdx_y = Tx.launch_thread("blockIdx.y", 4)
            blockIdx_z = Tx.launch_thread("blockIdx.z", 5)
            warp_id_in_cta: Tx.let[Tx.int32] = Tx.tvm_warp_shuffle(Tx.uint32(4294967295), threadIdx_x // 32, 0, 32, 32)  # noqa: E501
            bx: Tx.let[Tx.int32] = blockIdx_x
            by: Tx.let[Tx.int32] = blockIdx_y
            bz: Tx.let[Tx.int32] = blockIdx_z
            warp_id: Tx.let[Tx.int32] = warp_id_in_cta
            tx: Tx.let[Tx.int32] = threadIdx_x
            Tx.evaluate(bx + by + bz + warp_id + tx)
        blockIdx_x = Tx.launch_thread("blockIdx.x", 6)
        threadIdx_x = Tx.launch_thread("threadIdx.x", 256)
        blockIdx_y = Tx.launch_thread("blockIdx.y", 7)
        blockIdx_z = Tx.launch_thread("blockIdx.z", 8)
        warp_id_in_cta: Tx.let[Tx.int32] = Tx.tvm_warp_shuffle(Tx.uint32(4294967295), threadIdx_x // 32, 0, 32, 32)  # noqa: E501
        bx: Tx.let[Tx.int32] = blockIdx_x
        by: Tx.let[Tx.int32] = blockIdx_y
        bz: Tx.let[Tx.int32] = blockIdx_z
        warp_id: Tx.let[Tx.int32] = warp_id_in_cta
        tx: Tx.let[Tx.int32] = threadIdx_x
        Tx.evaluate(bx + by + bz + warp_id + tx)
    # fmt: on

    compare(before, after, LowerTIRx)


def test_lower_scope_slice():
    # fmt: off
    @Tx.prim_func(private=True, tirx=True)
    def before():
        with Tx.kernel():
            bx, by, bz = Tx.cta_id([3, 4, 5], parent="kernel")
            warp_id = Tx.warp_id([4], parent="cta")
            tx = Tx.thread_id([128], parent="cta")

            with Tx.cta()[0:1, 0:2, 0:3]:
                with Tx.thread()[0:64]:
                    Tx.evaluate(tx)
                    Tx.evaluate(warp_id)
                with Tx.thread()[Tx.ptx.elect_sync()]:
                    Tx.evaluate(tx)
                with Tx.thread()[tx == 0]:
                    Tx.evaluate(tx)

    @Tx.prim_func(private=True, tirx=True)
    def after():
        blockIdx_x = Tx.launch_thread("blockIdx.x", 3)
        threadIdx_x = Tx.launch_thread("threadIdx.x", 128)
        blockIdx_y = Tx.launch_thread("blockIdx.y", 4)
        blockIdx_z = Tx.launch_thread("blockIdx.z", 5)
        warp_id_in_cta: Tx.let[Tx.int32] = Tx.tvm_warp_shuffle(Tx.uint32(4294967295), threadIdx_x // 32, 0, 32, 32)  # noqa: E501
        bx: Tx.let[Tx.int32] = blockIdx_x  # noqa: F841
        by: Tx.let[Tx.int32] = blockIdx_y  # noqa: F841
        bz: Tx.let[Tx.int32] = blockIdx_z  # noqa: F841
        warp_id: Tx.let[Tx.int32] = warp_id_in_cta
        tx: Tx.let[Tx.int32] = threadIdx_x
        if blockIdx_x >= 0 and blockIdx_x < 1 and blockIdx_y >= 0 and blockIdx_y < 2 and blockIdx_z >= 0 and blockIdx_z < 3:  # noqa: E501
            if threadIdx_x >= 0 and threadIdx_x < 64:
                Tx.evaluate(tx)
                Tx.evaluate(warp_id)
            if Tx.ptx.elect_sync():
                Tx.evaluate(tx)
            if tx == 0:
                Tx.evaluate(tx)
    # fmt: on

    compare(before, after, LowerTIRx)


def test_lower_scope_partition1():
    # fmt: off
    @Tx.prim_func(private=True, tirx=True)
    def before():
        with Tx.kernel():
            bx, by, bz = Tx.cta_id([3, 4, 5], parent="kernel")
            Tx.warp_id([4], parent="cta")
            tx = Tx.thread_id([128], parent="cta")

            with Tx.cta():
                Tx.attr({"tirx.scope_partition": True})
                with Tx.thread()[0:32]:
                    Tx.evaluate(tx)
                with Tx.thread()[32:64]:
                    Tx.evaluate(tx)
                with Tx.thread()[64:96]:
                    Tx.evaluate(tx)
                with Tx.thread()[96:128]:
                    Tx.evaluate(tx)

    @Tx.prim_func(private=True, tirx=True)
    def main():
        blockIdx_x = Tx.launch_thread("blockIdx.x", 3)
        threadIdx_x = Tx.launch_thread("threadIdx.x", 128)
        blockIdx_y = Tx.launch_thread("blockIdx.y", 4)
        blockIdx_z = Tx.launch_thread("blockIdx.z", 5)
        warp_id_in_cta: Tx.let[Tx.int32] = Tx.tvm_warp_shuffle(Tx.uint32(4294967295), threadIdx_x // 32, 0, 32, 32)  # noqa: E501
        bx: Tx.let[Tx.int32] = blockIdx_x  # noqa: F841
        by: Tx.let[Tx.int32] = blockIdx_y  # noqa: F841
        bz: Tx.let[Tx.int32] = blockIdx_z  # noqa: F841
        v: Tx.let[Tx.int32] = warp_id_in_cta
        tx: Tx.let[Tx.int32] = threadIdx_x
        Tx.evaluate(v)
        if threadIdx_x >= 0 and threadIdx_x < 32:
            Tx.evaluate(tx)
        else:
            if threadIdx_x >= 32 and threadIdx_x < 64:
                Tx.evaluate(tx)
            else:
                if threadIdx_x >= 64 and threadIdx_x < 96:
                    Tx.evaluate(tx)
                else:
                    if threadIdx_x >= 96 and threadIdx_x < 128:
                        Tx.evaluate(tx)
    # fmt: on

    compare(before, main, LowerTIRx)


def test_lower_scope_partition2():
    # fmt: off
    @Tx.prim_func(private=True, tirx=True)
    def before():
        with Tx.kernel():
            cbx, cby = Tx.cta_id([2, 1], parent="cluster")
            Tx.cta_id([148], parent="kernel")
            Tx.warpgroup_id([2], parent="cta")
            warp_id = Tx.warp_id([4], parent="warpgroup")
            Tx.thread_id([32], parent="warp")
            with Tx.cta():
                Tx.attr({"tirx.scope_partition": True})
                with Tx.warpgroup()[1:2]:
                    Tx.attr({"tirx.scope_partition": True})
                    with Tx.warp(parent="warpgroup")[3:4]:
                        Tx.evaluate(warp_id)
                    with Tx.warp(parent="warpgroup")[2:3]:
                        Tx.evaluate(warp_id)
                    with Tx.warp(parent="warpgroup")[0:1]:
                        Tx.evaluate(warp_id)
                with Tx.warpgroup()[0:1]:
                    with Tx.thread():
                        Tx.evaluate(warp_id)
    # fmt: on

    # fmt: off
    @Tx.prim_func(private=True, tirx=True)
    def after():
        clusterCtaIdx_x = Tx.launch_thread("clusterCtaIdx.x", 2)
        clusterCtaIdx_y = Tx.launch_thread("clusterCtaIdx.y", 1)
        blockIdx_x = Tx.launch_thread("blockIdx.x", 148)
        threadIdx_x = Tx.launch_thread("threadIdx.x", 256)
        warp_id_in_cta: Tx.let[Tx.int32] = Tx.tvm_warp_shuffle(Tx.uint32(4294967295), threadIdx_x // 32, 0, 32, 32)  # noqa: E501
        cbx: Tx.let[Tx.int32] = clusterCtaIdx_x  # noqa: F841
        cby: Tx.let[Tx.int32] = clusterCtaIdx_y  # noqa: F841
        v: Tx.let[Tx.int32] = blockIdx_x
        v_1: Tx.let[Tx.int32] = warp_id_in_cta // 4
        warp_id: Tx.let[Tx.int32] = warp_id_in_cta % 4
        v_2: Tx.let[Tx.int32] = threadIdx_x % 32
        Tx.evaluate(v)
        Tx.evaluate(v_1)
        Tx.evaluate(v_2)
        if warp_id_in_cta // 4 >= 1 and warp_id_in_cta // 4 < 2:
            if warp_id_in_cta % 4 >= 3 and warp_id_in_cta % 4 < 4:
                Tx.evaluate(warp_id)
            else:
                if warp_id_in_cta % 4 >= 2 and warp_id_in_cta % 4 < 3:
                    Tx.evaluate(warp_id)
                else:
                    if warp_id_in_cta % 4 >= 0 and warp_id_in_cta % 4 < 1:
                        Tx.evaluate(warp_id)
        else:
            if warp_id_in_cta // 4 >= 0 and warp_id_in_cta // 4 < 1:
                Tx.evaluate(warp_id)
    # fmt: on

    compare(before, after, LowerTIRx)


def test_lower_layout():
    # fmt: off
    @Tx.prim_func(private=True, tirx=True)
    def before(A: Tx.Buffer((128, 32), "float16")) -> None:
        with Tx.kernel():
            bx, by, bz = Tx.cta_id([1, 1, 1], parent="kernel")
            Tx.warp_id([4], parent="cta")
            Tx.thread_id([32], parent="warp")
            tid = Tx.thread_id([128], parent="cta")

            with Tx.cta():
                A_smem = Tx.alloc_buffer([128, 32], dtype="float16", scope="shared", layout=Tx.SwizzleLayout(3, 3, 3))  # noqa: E501

                with Tx.thread():
                    thread_col = Tx.meta_var(4)
                    thread_row = Tx.meta_var(32)

                    for tile in Tx.serial(128 // thread_row):
                        row = Tx.meta_var(tile * thread_row + tid // thread_col)
                        col = Tx.meta_var(tid % thread_col * 8)
                        for vec in Tx.vectorized(8):
                            A_smem[row, col + vec] = A[bx * 128 + row, col + vec]

    @Tx.prim_func(private=True, tirx=True)
    def after(A_handle: Tx.handle) -> None:
        A = Tx.match_buffer(A_handle, (128, 32), "float16", layout=None)
        A_1 = Tx.decl_buffer((4096,), "float16", data=A.data, layout=None)
        blockIdx_x = Tx.launch_thread("blockIdx.x", 1)
        threadIdx_x = Tx.launch_thread("threadIdx.x", 128)
        blockIdx_y = Tx.launch_thread("blockIdx.y", 1)
        blockIdx_z = Tx.launch_thread("blockIdx.z", 1)
        warp_id_in_cta: Tx.let[Tx.int32] = Tx.tvm_warp_shuffle(Tx.uint32(4294967295), threadIdx_x // 32, 0, 32, 32)  # noqa: E501
        bx: Tx.let[Tx.int32] = blockIdx_x  # noqa: F841
        by: Tx.let[Tx.int32] = blockIdx_y  # noqa: F841
        bz: Tx.let[Tx.int32] = blockIdx_z  # noqa: F841
        v: Tx.let[Tx.int32] = warp_id_in_cta
        v_1: Tx.let[Tx.int32] = threadIdx_x % 32
        tid: Tx.let[Tx.int32] = threadIdx_x  # noqa: F841
        Tx.evaluate(v)
        Tx.evaluate(v_1)
        A_smem = Tx.alloc_shared((4096,), "float16", layout=None)
        for tile in range(4):
            for vec in Tx.vectorized(8):
                A_smem[Tx.shift_left(Tx.bitwise_xor(tile * 128 + threadIdx_x, Tx.shift_right(Tx.bitwise_and(tile * 128 + threadIdx_x, 56), 3)), 3) + vec] = A_1[tile * 1024 + threadIdx_x * 8 + vec]  # noqa: E501
    # fmt: on

    compare(before, after, LowerTIRx)


def test_lower_opcall_fail():
    # fmt: off
    @Tx.prim_func(tirx=True)
    def test(A_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, (64,), "float32", scope="global")

        with Tx.kernel():
            bx, by, bz = Tx.cta_id([1, 1, 1], parent="kernel")
            Tx.warp_id([1], parent="cta")
            Tx.thread_id([32], parent="warp")
            with Tx.cta():
                A_smem = Tx.alloc_buffer([64], dtype="float32", scope="shared")

                Tx.copy(A[0:64], A_smem[0:64])
                for i in range(10):
                    Tx.fill(A_smem[0:64], Tx.float32(0))
                    Tx.gemm(A_smem, A_smem, A_smem, A_smem)
                Tx.copy(A_smem[0:64], A[0:64])
    # fmt: on

    with pytest.raises(Exception):
        LowerTIRx()(tvm.IRModule({"main": test}))


def test_lower_decl_buffer_access_ptr():
    # fmt: off
    @Tx.prim_func(private=True, tirx=True)
    def before():
        with Tx.kernel():
            Tx.cta_id([1], parent="kernel")
            Tx.thread_id([128], parent="cta")
            with Tx.cta():
                buf = Tx.alloc_buffer([1024], "uint8", scope="shared.dyn")
                A = Tx.decl_buffer([128], "float16", buf.data, elem_offset=32)

                with Tx.thread():
                    Tx.evaluate(A.access_ptr("rw", offset=A.elem_offset_of([64])))

    @Tx.prim_func(private=True, tirx=True)
    def after():
        blockIdx_x = Tx.launch_thread("blockIdx.x", 1)
        threadIdx_x = Tx.launch_thread("threadIdx.x", 128)
        warp_id_in_cta: Tx.let[Tx.int32] = Tx.tvm_warp_shuffle(Tx.uint32(4294967295), threadIdx_x // 32, 0, 32, 32)  # noqa: F841,E501
        v: Tx.let[Tx.int32] = blockIdx_x
        v_1: Tx.let[Tx.int32] = threadIdx_x
        Tx.evaluate(v)
        Tx.evaluate(v_1)
        buf = Tx.alloc_buffer((1024,), "uint8", scope="shared.dyn", layout=None)
        A = Tx.decl_buffer((128,), "float16", data=buf.data, elem_offset=32, scope="shared.dyn", layout=None)  # noqa: F841,E501
        Tx.tvm_access_ptr(Tx.type_annotation("float16"), buf.data, Tx.Add(32, 64), Tx.Sub(128, 64), 3)  # noqa: E501
    # fmt: on

    compare(before, after, LowerTIRx)


def test_lower_separate_scope_id_def():
    # fmt: off
    @Tx.prim_func(private=True, tirx=True)
    def before():
        with Tx.kernel():
            Tx.cta_id([1], parent="kernel")
            with Tx.cta():
                tx = Tx.thread_id([128], parent="cta")
                with Tx.thread()[tx == 0]:
                    Tx.evaluate(tx)

    @Tx.prim_func(private=True, tirx=True)
    def after():
        blockIdx_x = Tx.launch_thread("blockIdx.x", 1)
        threadIdx_x = Tx.launch_thread("threadIdx.x", 128)
        warp_id_in_cta: Tx.let[Tx.int32] = Tx.tvm_warp_shuffle(Tx.uint32(4294967295), threadIdx_x // 32, 0, 32, 32)  # noqa: F841,E501
        tx: Tx.let[Tx.int32] = threadIdx_x
        v: Tx.let[Tx.int32] = blockIdx_x
        Tx.evaluate(v)
        if tx == 0:
            Tx.evaluate(tx)
    # fmt: on

    compare(before, after, LowerTIRx)


def test_lower_buffer_offset():
    # fmt: off
    @Tx.prim_func(private=True, tirx=True)
    def before():
        with Tx.kernel():
            Tx.cta_id([1], parent="kernel")
            with Tx.cta():
                Tx.thread_id([128], parent="cta")
                with Tx.thread():
                    A = Tx.alloc_buffer([64, 64], "float16", scope="local")
                    A0 = Tx.decl_buffer([64], "float16", A.data, elem_offset=A.elem_offset_of([32, 32]))  # noqa: E501
                    with Tx.thread():
                        Tx.evaluate(Tx.address_of(A0[32]))
    # fmt: on

    # fmt: off
    @Tx.prim_func(private=True, tirx=True)
    def after():
        blockIdx_x = Tx.launch_thread("blockIdx.x", 1)
        threadIdx_x = Tx.launch_thread("threadIdx.x", 128)
        warp_id_in_cta: Tx.let[Tx.int32] = Tx.tvm_warp_shuffle(Tx.uint32(4294967295), threadIdx_x // 32, 0, 32, 32)  # noqa: F841,E501
        v: Tx.let[Tx.int32] = threadIdx_x
        v_1: Tx.let[Tx.int32] = blockIdx_x
        Tx.evaluate(v_1)
        Tx.evaluate(v)
        A = Tx.alloc_local((4096,), "float16", layout=None)
        A0 = Tx.decl_buffer((64,), "float16", data=A.data, elem_offset=2080, scope="local", layout=None)  # noqa: E501
        Tx.address_of(A0[32])
    # fmt: on

    compare(before, after, LowerTIRx)


def test_lower_alloc_decl_buffer_outside_of_parser():
    # fmt: off
    @Tx.meta_class
    class State:
        def __init__(self, smem):
            self.A = Tx.alloc_local([1], "float16", name="A")
            self.B = Tx.alloc_local([1], "float16", name="B")
            self.C = Tx.decl_buffer([1], "float16", smem, elem_offset=0, scope="shared.dyn", name="C")  # noqa: E501

    def int_var1(val):
        buf = Tx.local_scalar("int32")
        if val is not None:
            Tx.buffer_store(buf.buffer, val, 0)
        return buf

    def int_var2(val):
        buf = Tx.alloc_local([1], "int32")
        if val is not None:
            Tx.buffer_store(buf, val, 0)
        return buf

    # fmt: off
    @Tx.prim_func(private=True, tirx=True)
    def before():
        with Tx.kernel():
            with Tx.thread():
                smem = Tx.alloc_buffer([100], "uint8", scope="shared.dyn")
                state = State(smem.data)
                state.A[0] = Tx.float16(1)
                state.B[0] = Tx.float16(2)
                state.C[0] = Tx.float16(3)
                D = int_var1(1)
                D = D + 1
                E = int_var1(2)
                E = E + 2
                F = int_var2(3)
                F[0] = F[0] + 3
                G = int_var2(4)
                G[0] = G[0] + 4
    # fmt: on

    # fmt: off
    @Tx.prim_func(private=True, tirx=True)
    def after():
        smem = Tx.alloc_buffer([100], "uint8", scope="shared.dyn", layout=None)
        A = Tx.alloc_local((1,), "float16", layout=None)
        B = Tx.alloc_local((1,), "float16", layout=None)
        C = Tx.decl_buffer((1,), "float16", data=smem.data, elem_offset=0, scope="shared.dyn", layout=None)  # noqa: E501
        A[0] = Tx.float16(1)
        B[0] = Tx.float16(2)
        C[0] = Tx.float16(3)

        D = Tx.alloc_local((1,), "int32", layout=None)
        D = 1
        D = D[0] + 1

        E = Tx.alloc_local((1,), "int32", layout=None)
        E = 2
        E = E[0] + 2

        F = Tx.alloc_local((1,), "int32", layout=None)
        F = 3
        F = F[0] + 3

        G = Tx.alloc_local((1,), "int32", layout=None)
        G = 4
        G = G[0] + 4
    # fmt: on

    compare(before, after, LowerTIRx)


def test_alloc_buffer_with_thread_axis_layout():
    """alloc_buffer with thread-axis layout should lower to 1D physical buffer with memory-axis span."""  # noqa: E501

    # fmt: off
    @Tx.prim_func(private=True, tirx=True)
    def before(out: Tx.Buffer((128, 4), "float32")) -> None:
        with Tx.kernel():
            bx, by, bz = Tx.cta_id([1, 1, 1], parent="kernel")
            Tx.warpgroup_id([1], parent="cta")
            warp_id = Tx.warp_id([4], parent="warpgroup")
            lane_id = Tx.thread_id([32], parent="warp")

            with Tx.warpgroup():
                with Tx.thread():
                    # Single-step alloc with thread-axis layout
                    reg_wg = Tx.alloc_buffer((128, 4), "float32", scope="local",
                                              layout=wg_local_layout(4))
                    # Access via .local() to decompose thread and memory axes
                    reg = reg_wg.local(4)
                    for i in Tx.serial(4):
                        reg[i] = out[lane_id + warp_id * 32, i]

    @Tx.prim_func(private=True, tirx=True)
    def after(out_handle: Tx.handle):
        out = Tx.match_buffer(out_handle, (128, 4), layout=None)
        out_1 = Tx.decl_buffer((512,), data=out.data, layout=None)
        blockIdx_x = Tx.launch_thread("blockIdx.x", 1)
        threadIdx_x = Tx.launch_thread("threadIdx.x", 128)
        blockIdx_y = Tx.launch_thread("blockIdx.y", 1)
        blockIdx_z = Tx.launch_thread("blockIdx.z", 1)
        warp_id_in_cta: Tx.let[Tx.int32] = Tx.tvm_warp_shuffle(Tx.uint32(4294967295), threadIdx_x // 32, 0, 32, 32)  # noqa: E501
        bx: Tx.let[Tx.int32] = blockIdx_x  # noqa: F841
        by: Tx.let[Tx.int32] = blockIdx_y  # noqa: F841
        bz: Tx.let[Tx.int32] = blockIdx_z  # noqa: F841
        v: Tx.let[Tx.int32] = warp_id_in_cta // 4
        warp_id: Tx.let[Tx.int32] = warp_id_in_cta % 4  # noqa: F841
        lane_id: Tx.let[Tx.int32] = threadIdx_x % 32  # noqa: F841
        Tx.evaluate(v)
        reg_wg = Tx.alloc_local((4,), layout=None)
        reg = Tx.decl_buffer((4,), data=reg_wg.data, scope="local", layout=None)
        for i in range(4):
            reg[i] = out_1[warp_id_in_cta % 4 * 128 + threadIdx_x % 32 * 4 + i]
    # fmt: on

    compare(before, after, LowerTIRx)


def test_scope_id_compliment_no_div_by_zero():
    """Regression test: Compliment must not divide by zero when kernel extent < cluster extent.

    Before the fix, defining cluster cta_id with extent > kernel cta_id extent would crash
    with a divide-by-zero in the Compliment function during ScopeIdDef verification.
    After the fix, it raises a validation error instead of crashing.
    """
    # The well-formedness verifier (which calls Compliment internally) runs at parse time.
    # Before the fix this would crash with divide-by-zero; now it raises a clean error.
    with pytest.raises(Exception):
        # fmt: off
        @Tx.prim_func(tirx=True)
        def func(A: Tx.Buffer((1,))):
            with Tx.kernel():
                cb_m, cb_n = Tx.cta_id([2, 2], parent="cluster")
                bx = Tx.cta_id([1], parent="kernel")
                tx = Tx.thread_id([128], parent="cta")
                with Tx.thread():
                    Tx.evaluate(bx + cb_m + cb_n + tx)
        # fmt: on


def test_scope_id_compliment_non_divisible():
    """Regression test: Compliment must error on provably non-divisible extents.

    cta->thread=100 and cta->warp=3 would produce warp->thread = floordiv(100, 3) = 33,
    which is semantically wrong. The fix detects this and raises an error.
    """
    with pytest.raises(Exception):
        # fmt: off
        @Tx.prim_func(tirx=True)
        def func():
            with Tx.kernel():
                bx = Tx.cta_id([1], parent="kernel")
                wid = Tx.warp_id([3], parent="cta")
                tx = Tx.thread_id([100], parent="cta")
                with Tx.thread():
                    Tx.evaluate(bx + wid + tx)
        # fmt: on


def test_empty_kernel_no_thread_id():
    """Regression test: kernel with ScopeIdDefs but no thread launch params must error early.

    Before the fix, this would crash late in codegen with poor diagnostics.
    """

    # fmt: off
    @Tx.prim_func(tirx=True)
    def func():
        with Tx.kernel():
            bx = Tx.cta_id([32], parent="kernel")
            with Tx.cta():
                with Tx.thread():
                    Tx.evaluate(bx)
    # fmt: on
    with pytest.raises(Exception, match="kernel has no thread launch parameters"):
        with tvm.target.Target("cuda"):
            LowerTIRx()(tvm.IRModule({"main": func}))


def test_lower_preferred_cluster():
    # fmt: off
    @Tx.prim_func(private=True, tirx=True)
    def before() -> None:
        with Tx.kernel():
            bx = Tx.cta_id([8], parent="kernel")
            cbx, cby = Tx.cta_id([2, 1], parent="cluster", preferred=[2, 2])
            tx = Tx.thread_id([128], parent="cta")
            with Tx.thread():
                Tx.evaluate(bx + cbx + cby + tx)
    # fmt: on

    with tvm.target.Target("cuda"):
        after_mod = LowerTIRx()(tvm.IRModule({"main": before}))
    assert not _contains_exec_scope(after_mod)
    after_str = str(after_mod["main"])

    # Fallback cluster size
    assert 'launch_thread("clusterCtaIdx.x", 2)' in after_str
    assert 'launch_thread("clusterCtaIdx.y", 1)' in after_str
    # Preferred cluster size
    assert 'launch_thread("preferredClusterCtaIdx.x", 2)' in after_str
    assert 'launch_thread("preferredClusterCtaIdx.y", 2)' in after_str
    # Variables resolve to clusterCtaIdx registers
    assert "clusterCtaIdx_x" in after_str
    assert "clusterCtaIdx_y" in after_str


def test_lower_tirx_dedup_tensormap():
    """Two identical TMA copy_async calls from the same global buffer should share one tensormap."""
    from tvm.tirx.operator.scope_op_dispatch.cuda.tma_utils import mma_shared_layout

    shared_layout = mma_shared_layout("float16", 3, (8, 256))

    # fmt: off
    @Tx.prim_func(private=True, tirx=True)
    def before(A_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, (8, 256), "float16")
        with Tx.kernel():
            Tx.cta_id([1], parent="kernel")
            Tx.thread_id([128], parent="cta")
            with Tx.thread():
                dyn = Tx.alloc_buffer([6208], "uint8", scope="shared.dyn")
                A_smem = Tx.decl_buffer((8, 256), "float16", dyn.data, elem_offset=0, layout=shared_layout)  # noqa: E501
                A_smem2 = Tx.decl_buffer((8, 256), "float16", dyn.data, elem_offset=2048, layout=shared_layout)  # noqa: E501
                mbarrier = Tx.decl_buffer([1], "uint64", dyn.data, elem_offset=1024)
                mbar_ptr = Tx.meta_var(mbarrier.ptr_to([0]))
                with Tx.thread()[0:1]:
                    Tx.copy_async(A_smem[:, :], A[:, :], dispatch="tma", mbar=mbar_ptr)
                with Tx.thread()[0:1]:
                    Tx.copy_async(A_smem2[:, :], A[:, :], dispatch="tma", mbar=mbar_ptr)

    @Tx.prim_func(private=True, tirx=True)
    def after(A_ptr: Tx.handle):
        A = Tx.match_buffer(A_ptr, (8, 256), "float16", layout=None)
        A_1 = A.view(2048)  # noqa: F841
        A_ptr_tensormap: Tx.let[Tx.handle("tensormap", "global")] = Tx.tvm_stack_alloca("tensormap", 1)  # noqa: E501
        Tx.call_packed("runtime.cuTensorMapEncodeTiled", A_ptr_tensormap, "float16", 3, A.data, 64, 8, 4, 512, 128, 64, 8, 4, 1, 1, 1, 0, 3, 2, 0)  # noqa: E501
        with Tx.launch_thread("blockIdx.x", 1) as blockIdx_x:
            threadIdx_x = Tx.launch_thread("threadIdx.x", 128)
            warp_id_in_cta: Tx.let[Tx.int32] = Tx.tvm_warp_shuffle(Tx.uint32(4294967295), threadIdx_x // 32, 0, 32, 32)  # noqa: E501, F841
            v: Tx.let[Tx.int32] = blockIdx_x
            v_1: Tx.let[Tx.int32] = threadIdx_x
            Tx.evaluate(v)
            Tx.evaluate(v_1)
            dyn = Tx.alloc_buffer((6208,), "uint8", scope="shared.dyn", layout=None)
            A_smem = Tx.decl_buffer((2048,), "float16", data=dyn.data, scope="shared.dyn", layout=None)  # noqa: E501, F841
            A_smem2 = Tx.decl_buffer((2048,), "float16", data=dyn.data, elem_offset=2048, scope="shared.dyn", layout=None)  # noqa: E501
            mbarrier = Tx.decl_buffer((1,), "uint64", data=dyn.data, elem_offset=1024, scope="shared.dyn", layout=None)  # noqa: E501
            if threadIdx_x >= 0 and threadIdx_x < 1:
                for loop_vars in range(1):
                    s_buf_w_offset = A_smem2.partition[-2048:0]
                    Tx.ptx.cp_async.bulk.tensor.g2c(3, Tx.address_of(s_buf_w_offset[0]), Tx.address_of(mbarrier[0]), A_ptr_tensormap, 0, 1, "", 0, 0, 0)  # noqa: E501
            if threadIdx_x >= 0 and threadIdx_x < 1:
                for loop_vars in range(1):
                    s_buf_w_offset = Tx.decl_buffer((2048,), "float16", data=dyn.data, elem_offset=2048, scope="shared.dyn", layout=None)  # noqa: E501
                    Tx.ptx.cp_async.bulk.tensor.g2c(3, Tx.address_of(s_buf_w_offset[0]), Tx.address_of(mbarrier[0]), A_ptr_tensormap, 0, 1, "", 0, 0, 0)  # noqa: E501
    # fmt: on

    compare(before, after, LowerTIRx)


def test_lower_tirx_keep_different_tensormaps():
    """Same global buffer but different smem shapes should get separate tensormaps."""
    from tvm.tirx.operator.scope_op_dispatch.cuda.tma_utils import mma_shared_layout

    layout_8x64 = mma_shared_layout("float16", 3, (8, 64))
    layout_8x128 = mma_shared_layout("float16", 3, (8, 128))

    # fmt: off
    @Tx.prim_func(private=True, tirx=True)
    def before(A_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, (8, 256), "float16")
        with Tx.kernel():
            Tx.cta_id([1], parent="kernel")
            Tx.thread_id([128], parent="cta")
            with Tx.thread():
                dyn = Tx.alloc_buffer([4160], "uint8", scope="shared.dyn")
                A_smem1 = Tx.decl_buffer((8, 64), "float16", dyn.data, elem_offset=0, layout=layout_8x64)  # noqa: E501
                A_smem2 = Tx.decl_buffer((8, 128), "float16", dyn.data, elem_offset=1024, layout=layout_8x128)  # noqa: E501
                mbarrier = Tx.decl_buffer([1], "uint64", dyn.data, elem_offset=512)
                mbar_ptr = Tx.meta_var(mbarrier.ptr_to([0]))
                with Tx.thread()[0:1]:
                    Tx.copy_async(A_smem1[:, :], A[0:8, 0:64], dispatch="tma", mbar=mbar_ptr)
                with Tx.thread()[0:1]:
                    Tx.copy_async(A_smem2[:, :], A[0:8, 0:128], dispatch="tma", mbar=mbar_ptr)

    @Tx.prim_func(private=True, tirx=True)
    def after(A_ptr: Tx.handle):
        A = Tx.match_buffer(A_ptr, (8, 256), "float16", layout=None)
        A_1 = A.view(2048)  # noqa: F841
        A_ptr_tensormap: Tx.let[Tx.handle("tensormap", "global")] = Tx.tvm_stack_alloca("tensormap", 1)  # noqa: E501
        Tx.call_packed("runtime.cuTensorMapEncodeTiled", A_ptr_tensormap, "float16", 3, A.data, 64, 8, 4, 512, 128, 64, 8, 2, 1, 1, 1, 0, 3, 2, 0)  # noqa: E501
        A_ptr_tensormap_1: Tx.let[Tx.handle("tensormap", "global")] = Tx.tvm_stack_alloca("tensormap", 1)  # noqa: E501
        Tx.call_packed("runtime.cuTensorMapEncodeTiled", A_ptr_tensormap_1, "float16", 2, A.data, 256, 8, 512, 64, 8, 1, 1, 0, 3, 2, 0)  # noqa: E501
        with Tx.launch_thread("blockIdx.x", 1) as blockIdx_x:
            threadIdx_x = Tx.launch_thread("threadIdx.x", 128)
            warp_id_in_cta: Tx.let[Tx.int32] = Tx.tvm_warp_shuffle(Tx.uint32(4294967295), threadIdx_x // 32, 0, 32, 32)  # noqa: E501, F841
            v: Tx.let[Tx.int32] = blockIdx_x
            v_1: Tx.let[Tx.int32] = threadIdx_x
            Tx.evaluate(v)
            Tx.evaluate(v_1)
            dyn = Tx.alloc_buffer((4160,), "uint8", scope="shared.dyn", layout=None)
            A_smem1 = Tx.decl_buffer((512,), "float16", data=dyn.data, scope="shared.dyn", layout=None)  # noqa: E501, F841
            A_smem2 = Tx.decl_buffer((1024,), "float16", data=dyn.data, elem_offset=1024, scope="shared.dyn", layout=None)  # noqa: E501
            mbarrier = Tx.decl_buffer((1,), "uint64", data=dyn.data, elem_offset=512, scope="shared.dyn", layout=None)  # noqa: E501
            if threadIdx_x >= 0 and threadIdx_x < 1:
                for loop_vars in range(1):
                    s_buf_w_offset = A_smem2.partition[-1024:-512]
                    Tx.ptx.cp_async.bulk.tensor.g2c(2, Tx.address_of(s_buf_w_offset[0]), Tx.address_of(mbarrier[0]), A_ptr_tensormap_1, 0, 1, "", 0, 0)  # noqa: E501
            if threadIdx_x >= 0 and threadIdx_x < 1:
                for loop_vars in range(1):
                    s_buf_w_offset = Tx.decl_buffer((1024,), "float16", data=dyn.data, elem_offset=1024, scope="shared.dyn", layout=None)  # noqa: E501
                    Tx.ptx.cp_async.bulk.tensor.g2c(3, Tx.address_of(s_buf_w_offset[0]), Tx.address_of(mbarrier[0]), A_ptr_tensormap, 0, 1, "", 0, 0, 0)  # noqa: E501
    # fmt: on

    compare(before, after, LowerTIRx)


def test_lower_tirx_dedup_smem_descriptor():
    """Two gemm_async calls using the same smem buffers should share descriptors."""
    from tvm.tirx.layout import S, TCol, TileLayout, TLane
    from tvm.tirx.operator.scope_op_dispatch.cuda.tma_utils import mma_shared_layout

    A_layout = mma_shared_layout("float16", 3, (128, 64))
    B_layout = mma_shared_layout("float16", 3, (128, 64))

    # fmt: off
    @Tx.prim_func(private=True, tirx=True)
    def before(A_ptr: Tx.handle, B_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, (128, 64), "float16")  # noqa: F841
        B = Tx.match_buffer(B_ptr, (128, 64), "float16")  # noqa: F841
        with Tx.kernel():
            Tx.cta_id([1], parent="kernel")
            Tx.warpgroup_id([1], parent="cta")
            Tx.thread_id([128], parent="warpgroup")
            A_smem = Tx.alloc_buffer((128, 64), "float16", scope="shared", layout=A_layout)
            B_smem = Tx.alloc_buffer((128, 64), "float16", scope="shared", layout=B_layout)
            tmem_addr = Tx.alloc_shared([1], "uint32")
            with Tx.warp()[0:1]:
                Tx.ptx.tcgen05.alloc(Tx.address_of(tmem_addr), n_cols=128, cta_group=1)
            tmem = Tx.decl_buffer((128, 128), "float32", scope="tmem", allocated_addr=tmem_addr[0], layout=TileLayout(S[(128, 128) : (1 @ TLane, 1 @ TCol)]))  # noqa: E501
            with Tx.thread()[0:1]:
                Tx.gemm_async(tmem[0:128, 0:128], A_smem[0:128, 0:64], B_smem[0:128, 0:64], dispatch="tcgen05")  # noqa: E501
            with Tx.thread()[0:1]:
                Tx.gemm_async(tmem[0:128, 0:128], A_smem[0:128, 0:64], B_smem[0:128, 0:64], dispatch="tcgen05")  # noqa: E501
            with Tx.warp()[0:1]:
                Tx.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=128, cta_group=1)

    _UNIFORM_SRC = "\n        __forceinline__ __device__ void smem_desc_make_lo_uniform_(uint64_t* desc) {\n            SmemDescriptor* d = reinterpret_cast<SmemDescriptor*>(desc);\n            d->lo = __shfl_sync(0xffffffff, d->lo, 0);\n        }\n        "  # noqa: E501
    _OFFSET_SRC = "\n__forceinline__ __device__ uint64_t tvm_builtin_smem_desc_add_16B_offset(uint64_t desc_base, int32_t offset) {\n    SmemDescriptor desc;\n    desc.desc_ = desc_base;\n    desc.lo += static_cast<uint32_t>(offset);\n    return desc.desc_;\n}\n"  # noqa: E501

    @Tx.prim_func(private=True, tirx=True)
    def after(A_ptr: Tx.handle, B_ptr: Tx.handle):
        A = Tx.match_buffer(A_ptr, (128, 64), "float16", layout=None)
        B = Tx.match_buffer(B_ptr, (128, 64), "float16", layout=None)
        B_1 = B.view(8192)  # noqa: F841
        A_1 = A.view(8192)  # noqa: F841
        with Tx.launch_thread("blockIdx.x", 1) as blockIdx_x:
            threadIdx_x = Tx.launch_thread("threadIdx.x", 128)
            warp_id_in_cta: Tx.let[Tx.int32] = Tx.tvm_warp_shuffle(Tx.uint32(4294967295), threadIdx_x // 32, 0, 32, 32)  # noqa: E501
            v: Tx.let[Tx.int32] = blockIdx_x
            v_1: Tx.let[Tx.int32] = warp_id_in_cta // 4
            v_2: Tx.let[Tx.int32] = threadIdx_x % 128
            Tx.evaluate(v)
            Tx.evaluate(v_1)
            Tx.evaluate(v_2)
            A_smem = Tx.alloc_shared((8192,), "float16", layout=None)
            descA = Tx.alloc_local((1,), "uint64", layout=None)
            Tx.ptx.tcgen05.encode_matrix_descriptor(Tx.address_of(descA[0]), Tx.address_of(A_smem[0]), 64, 64, 3)  # noqa: E501
            Tx.cuda.func_call("smem_desc_make_lo_uniform_", Tx.address_of(descA[0]), source_code=_UNIFORM_SRC)  # noqa: E501
            B_smem = Tx.alloc_shared((8192,), "float16", layout=None)
            descB = Tx.alloc_local((1,), "uint64", layout=None)
            Tx.ptx.tcgen05.encode_matrix_descriptor(Tx.address_of(descB[0]), Tx.address_of(B_smem[0]), 64, 64, 3)  # noqa: E501
            Tx.cuda.func_call("smem_desc_make_lo_uniform_", Tx.address_of(descB[0]), source_code=_UNIFORM_SRC)  # noqa: E501
            tmem_addr = Tx.alloc_shared((1,), "uint32", layout=None)
            if warp_id_in_cta >= 0 and warp_id_in_cta < 1:
                Tx.ptx.tcgen05.alloc(Tx.address_of(tmem_addr[0]), 128, 1)
            tmem_addr_1 = Tx.Buffer((1,), "uint32", data=tmem_addr.data, scope="shared")
            tmem = Tx.decl_buffer((16384,), scope="tmem", layout=None, allocated_addr=tmem_addr_1[0])  # noqa: E501, F841
            if threadIdx_x >= 0 and threadIdx_x < 1:
                descI_local = Tx.alloc_local((1,), "uint32", layout=None)
                Tx.ptx.tcgen05.encode_instr_descriptor(Tx.address_of(descI_local[0]), d_dtype="float32", a_dtype="float16", b_dtype="float16", M=128, N=128, K=16, trans_a=Tx.bool(False), trans_b=Tx.bool(False), n_cta_groups=1, neg_a=Tx.bool(False), neg_b=Tx.bool(False), sat_d=Tx.bool(False), is_sparse=Tx.bool(False))  # noqa: E501
                for mi in Tx.unroll(1):
                    for ni in Tx.unroll(1):
                        for ki in Tx.unroll(4):
                            a_val = Tx.alloc_local((1,), "uint64", layout=None)
                            a_val[0] = Tx.cuda.func_call("tvm_builtin_smem_desc_add_16B_offset", descA[0], (mi * 8192 + ki * 16) // 8, source_code=_OFFSET_SRC, return_type="uint64")  # noqa: E501
                            descB_val = Tx.alloc_local((1,), "uint64", layout=None)
                            descB_val[0] = Tx.cuda.func_call("tvm_builtin_smem_desc_add_16B_offset", descB[0], (ni * 8192 + ki * 16) // 8, source_code=_OFFSET_SRC, return_type="uint64")  # noqa: E501
                            should_accum = Tx.alloc_local((1,), "int8", layout=None)
                            should_accum[0] = Tx.Cast("int8", ki != 0)
                            tmem_col = Tx.alloc_local((1,), "int32", layout=None)
                            tmem_col[0] = ni * 128
                            Tx.ptx.tcgen05.mma("float32", "float16", "float16", Tx.cuda.get_tmem_addr(tmem_addr[0], mi * 128, tmem_col[0]), a_val[0], descB_val[0], descI_local[0], Tx.bool(False), 1, Tx.Cast("bool", should_accum[0]), 0, 0, 0, 0, 0)  # noqa: E501
            if threadIdx_x >= 0 and threadIdx_x < 1:
                descI_local = Tx.alloc_local((1,), "uint32", layout=None)
                Tx.ptx.tcgen05.encode_instr_descriptor(Tx.address_of(descI_local[0]), d_dtype="float32", a_dtype="float16", b_dtype="float16", M=128, N=128, K=16, trans_a=Tx.bool(False), trans_b=Tx.bool(False), n_cta_groups=1, neg_a=Tx.bool(False), neg_b=Tx.bool(False), sat_d=Tx.bool(False), is_sparse=Tx.bool(False))  # noqa: E501
                for mi in Tx.unroll(1):
                    for ni in Tx.unroll(1):
                        for ki in Tx.unroll(4):
                            a_val = Tx.alloc_local((1,), "uint64", layout=None)
                            a_val[0] = Tx.cuda.func_call("tvm_builtin_smem_desc_add_16B_offset", descA[0], (mi * 8192 + ki * 16) // 8, source_code=_OFFSET_SRC, return_type="uint64")  # noqa: E501
                            descB_val = Tx.alloc_local((1,), "uint64", layout=None)
                            descB_val[0] = Tx.cuda.func_call("tvm_builtin_smem_desc_add_16B_offset", descB[0], (ni * 8192 + ki * 16) // 8, source_code=_OFFSET_SRC, return_type="uint64")  # noqa: E501
                            should_accum = Tx.alloc_local((1,), "int8", layout=None)
                            should_accum[0] = Tx.Cast("int8", ki != 0)
                            tmem_col = Tx.alloc_local((1,), "int32", layout=None)
                            tmem_col[0] = ni * 128
                            Tx.ptx.tcgen05.mma("float32", "float16", "float16", Tx.cuda.get_tmem_addr(tmem_addr[0], mi * 128, tmem_col[0]), a_val[0], descB_val[0], descI_local[0], Tx.bool(False), 1, Tx.Cast("bool", should_accum[0]), 0, 0, 0, 0, 0)  # noqa: E501
            if warp_id_in_cta >= 0 and warp_id_in_cta < 1:
                Tx.ptx.tcgen05.dealloc(tmem_addr[0], 128, 1)
    # fmt: on

    compare(before, after, LowerTIRx)


if __name__ == "__main__":
    tvm.testing.main()
