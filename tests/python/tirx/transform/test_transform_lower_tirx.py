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
from functools import partial

import pytest

import tvm
import tvm.testing
from tvm.tir.layout import laneid, warpid, tid_in_wg
from tvm.script import tirx as Tx
from tvm.tir.function import PrimFunc
from tvm.tir.transform import LowerTIRx


def compare(before, after, transform):
    if isinstance(before, PrimFunc):
        before = tvm.IRModule({"main": before})
    if isinstance(after, PrimFunc):
        after = tvm.IRModule({"main": after})
    assert isinstance(before, tvm.IRModule)
    assert isinstance(after, tvm.IRModule)

    with tvm.target.Target("cuda"):
        transform()(before).show()
        tvm.ir.assert_structural_equal(transform()(before), after, map_free_vars=False)


L_LANE = Tx.TileLayout(Tx.S[32 : 1 @ laneid])


def test_lower_view_get():
    # fmt: off
    @Tx.prim_func(private=True, tirx=True)
    def before1(in_buf: Tx.Buffer(64, "float32"), out: Tx.Buffer(64, "float32")) -> None:
        with Tx.kernel():
            bx, by, bz = Tx.cta_id([1, 1, 1], parent="kernel")
            warp_id = Tx.warp_id([1], parent="cta")
            lane_id = Tx.thread_id([32], parent="warp")

            with Tx.thread():
                A = Tx.alloc_buffer([2], dtype="float16", scope="local", layout=Tx.TileLayout(Tx.S[2 : 1]))
                B_layout = A.layout.tile(L_LANE, (32,), (2,))
                # load in_buf into A
                with Tx.warp():
                    # warp view of this load
                    B = A.view(64, layout=B_layout)
                    # B[i] = in_buf[i]
                    with Tx.thread():
                        # done by each thread
                        A_local = B.storage(2)
                        for i in Tx.vectorized(2):
                            A_local[i] = Tx.float32(in_buf[lane_id * 2 + i])
                # write A into out
                with Tx.warp():
                    # warp view of this write
                    B = A.view(64, layout=B_layout)
                    # out[i] = B[i]
                    with Tx.thread():
                        # done by each thread
                        A_local = B.storage(2)
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
        warp_id_in_cta: Tx.int32 = Tx.tvm_warp_shuffle(Tx.uint32(4294967295), threadIdx_x // 32, 0, 32, 32)
        bx: Tx.int32 = blockIdx_x
        by: Tx.int32 = blockIdx_y
        bz: Tx.int32 = blockIdx_z
        warp_id: Tx.int32 = warp_id_in_cta
        lane_id: Tx.int32 = threadIdx_x % 32
        with Tx.kernel():
            with Tx.thread():
                A = Tx.alloc_local((2,), "float16", layout=None)
                with Tx.warp():
                    B = Tx.decl_buffer((64,), "float16", data=A.data, scope="local", layout=None)
                    with Tx.thread():
                        A_local = Tx.decl_buffer((2,), "float16", data=A.data, scope="local", layout=None)
                        for i in Tx.vectorized(2):
                            A_local[i] = Tx.Cast("float16", in_buf_1[threadIdx_x * 2 + i])
                with Tx.warp():
                    B = Tx.decl_buffer((64,), "float16", data=A.data, scope="local", layout=None)
                    with Tx.thread():
                        A_local = Tx.decl_buffer((2,), "float16", data=A.data, scope="local", layout=None)
                        for i in Tx.vectorized(2):
                            out_1[threadIdx_x * 2 + i] = Tx.Cast("float32", A_local[i])
    # fmt: on

    compare(before1, after1, LowerTIRx)

    # fmt: off
    @Tx.prim_func(private=True, tirx=True)
    def before2(in_buf: Tx.Buffer((16, 16), "float32"), out: Tx.Buffer((16, 16), "float32")) -> None:
        with Tx.kernel():
            bx, by, bz = Tx.cta_id([1, 1, 1], parent="kernel")
            warp_id = Tx.warp_id([1], parent="cta")
            lane_id = Tx.thread_id([32], parent="warp")

            with Tx.thread():
                atom = Tx.TileLayout(Tx.S[(1, 2) : (2, 1)])
                tile = Tx.TileLayout(Tx.S[(2, 2) : (2, 1)])
                warp_atom = atom.tile(L_LANE, (8, 4), (1, 2))

                A = Tx.alloc_buffer([4, 2], dtype="float32", scope="local", layout=atom.tile(tile, (2, 2), (1, 2)))
                B_layout = warp_atom.tile(tile, (2, 2), (8, 8))

                # load in_buf into A
                with Tx.warp():
                    # warp view of this load
                    B = A.view(16, 16, layout=B_layout)
                    with Tx.thread():
                        # done by each thread
                        A_local = B.storage(2, 2, 2)
                        for i in Tx.unroll(4):
                            for j in Tx.vectorized(2):
                                A_local[i // 2, i % 2, j] = in_buf[i // 2 * 8 + lane_id // 4, i % 2 * 8 + lane_id % 4 + j]
                # write A into out
                with Tx.warp():
                    # warp view of this write
                    B = A.view(16, 16, layout=B_layout)
                    with Tx.thread():
                        # done by each thread
                        A_local = B.storage(8)
                        for i in Tx.vectorized(2):
                            out[lane_id // 4 * 8 + i // 2 * 8 + lane_id % 4, lane_id % 4 * 2 + i % 2] = A_local[i]

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
        warp_id_in_cta: Tx.int32 = Tx.tvm_warp_shuffle(Tx.uint32(4294967295), threadIdx_x // 32, 0, 32, 32)
        bx: Tx.int32 = blockIdx_x
        by: Tx.int32 = blockIdx_y
        bz: Tx.int32 = blockIdx_z
        warp_id: Tx.int32 = warp_id_in_cta
        lane_id: Tx.int32 = threadIdx_x % 32
        with Tx.kernel():
            with Tx.thread():
                A = Tx.alloc_local((8,), layout=None)
                with Tx.warp():
                    B = Tx.decl_buffer((256,), data=A.data, scope="local", layout=None)
                    with Tx.thread():
                        A_local = Tx.decl_buffer((8,), data=A.data, scope="local", layout=None)
                        for i in Tx.unroll(4):
                            for j in Tx.vectorized(2):
                                A_local[i * 2 + j] = in_buf_1[i // 2 * 128 + threadIdx_x // 4 * 16 + i % 2 * 8 + j + threadIdx_x % 4]
                with Tx.warp():
                    B = Tx.decl_buffer((256,), data=A.data, scope="local", layout=None)
                    with Tx.thread():
                        A_local = Tx.decl_buffer((8,), data=A.data, scope="local", layout=None)
                        for i in Tx.vectorized(2):
                            out_1[threadIdx_x // 4 * 128 + threadIdx_x % 4 * 18 + i] = A_local[i]
    # fmt: on

    compare(before2, after2, LowerTIRx)

    # fmt: off
    @Tx.prim_func(private=True, tirx=True)
    def before3_wgmma_layout(in_buf: Tx.Buffer((128, 128), "float32"), out: Tx.Buffer((128, 128), "float32")) -> None:
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
                acc = Tx.alloc_buffer([64,], dtype="float32", scope="local", layout=atom.tile(tile, (2, 128 // 8), (1, 2)))

                # load in_buf into acc
                with Tx.cta():
                    # cta view of this load
                    A = acc.view(128, 128, layout=layout)
                    with Tx.thread():
                        # done by each thread
                        acc_local = A.storage(16, 2, 2, layout=atom.tile(tile, (2, 128 // 8), (1, 2)))
                        for i in Tx.serial(128 // 8):
                            for j in Tx.unroll(2):
                                for vec in Tx.vectorized(2):
                                    acc_local[i, j, vec] = in_buf[wg_id * 64 + warp_id_in_wg * 16 + j * 8 + lane_id // 4, i * 8 + lane_id % 4 * 2 + vec]

                # write acc into out
                with Tx.cta():
                    # cta view of this write
                    A = acc.view(128, 128, layout=layout)
                    with Tx.thread():
                        # done by each thread
                        acc_local = A.storage(64, layout=atom.tile(tile, (2, 128 // 8), (1, 2)))
                        for i in Tx.serial(128 // 8):
                            for j in Tx.unroll(2):
                                for vec in Tx.vectorized(2):
                                    out[wg_id * 64 + warp_id_in_wg * 16 + j * 8 + lane_id // 4, i * 8 + lane_id % 4 * 2 + vec] = acc_local[i * 4 + j * 2 + vec]

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
        warp_id_in_cta: Tx.int32 = Tx.tvm_warp_shuffle(Tx.uint32(4294967295), threadIdx_x // 32, 0, 32, 32)
        bx: Tx.int32 = blockIdx_x
        by: Tx.int32 = blockIdx_y
        bz: Tx.int32 = blockIdx_z
        wg_id: Tx.int32 = warp_id_in_cta // 4
        warp_id_in_wg: Tx.int32 = warp_id_in_cta % 4
        lane_id: Tx.int32 = threadIdx_x % 32
        with Tx.kernel():
            with Tx.thread():
                acc = Tx.alloc_local((64,), layout=None)
                with Tx.cta():
                    A = Tx.decl_buffer((16384,), data=acc.data, scope="local", layout=None)
                    with Tx.thread():
                        acc_local = Tx.decl_buffer((64,), data=acc.data, scope="local", layout=None)
                        for i in range(16):
                            for j in Tx.unroll(2):
                                for vec in Tx.vectorized(2):
                                    acc_local[i % 8 * 8 + j * 4 + i // 8 * 2 + vec] = in_buf_1[warp_id_in_cta * 2048 + j * 1024 + threadIdx_x % 32 // 4 * 128 + i * 8 + threadIdx_x % 4 * 2 + vec]
                with Tx.cta():
                    A = Tx.decl_buffer((16384,), data=acc.data, scope="local", layout=None)
                    with Tx.thread():
                        acc_local = Tx.decl_buffer((64,), data=acc.data, scope="local", layout=None)
                        for i in range(16):
                            for j in Tx.unroll(2):
                                for vec in Tx.vectorized(2):
                                    out_1[warp_id_in_cta * 2048 + j * 1024 + threadIdx_x % 32 // 4 * 128 + i * 8 + threadIdx_x % 4 * 2 + vec] = acc_local[i % 8 * 8 + j * 4 + i // 8 * 2 + vec]
    # fmt: on

    compare(before3_wgmma_layout, after3_wgmma_layout, LowerTIRx)


def test_lower_tirx_dedup_cu_tensormap():
    # fmt: off
    @Tx.prim_func(private=True, tirx=True)
    def before(A_ptr: Tx.handle, cond: Tx.bool) -> None:
        A = Tx.match_buffer(A_ptr, (64, 32), "float16", scope="global")

        # Two tensormap allocations
        map1: Tx.handle("tensormap") = Tx.tvm_stack_alloca("tensormap", 1)
        map2: Tx.handle("tensormap") = Tx.tvm_stack_alloca("tensormap", 1)
        map3: Tx.handle("tensormap") = Tx.tvm_stack_alloca("tensormap", 1)

        if cond:
            # Two identical cuTensorMapEncodeTiled initializations
            Tx.call_packed(
                "runtime.cuTensorMapEncodeTiled",
                map1, "float16", 2, A.data,
                32, 64,    # global_shape (reversed)
                64,        # global stride
                16, 16,    # shared_shape (boxDim)
                1, 1,      # shared_strides
                0, 0, 0, 0 # interleave, swizzle, l2, oob
            )
            Tx.call_packed(
                "runtime.cuTensorMapEncodeTiled",
                map2, "float16", 2, A.data,
                32, 64, 64,
                16, 16,
                1, 1,
                0, 0, 0, 0
            )
            # unused map, should be removed in the pass
            Tx.call_packed(
                "runtime.cuTensorMapEncodeTiled",
                map3, "float16", 2, A.data,
                32, 64, 64,
                16, 16,
                1, 1,
                0, 0, 0, 0
            )
        else:
            Tx.call_packed(
                "runtime.cuTensorMapEncodeTiled",
                map1, "float16", 2, A.data,
                32, 64, 64,
                16, 16,
                1, 1,
                0, 0, 0, 0
            )

        # Use site of map2, should be rewritten to map1
        with Tx.kernel():
            with Tx.cta():
                S = Tx.alloc_buffer((16, 16), "float16", scope="shared")
                Tx.ptx.cp_async.bulk.tensor.s2g(2, S.ptr_to([0, 0]), map2, 0, 0)

    @Tx.prim_func(private=True, tirx=True)
    def after(A_ptr: Tx.handle, cond: Tx.bool) -> None:
        A = Tx.match_buffer(A_ptr, (64, 32), "float16", layout=None)
        A_1 = Tx.decl_buffer((2048,), "float16", data=A.data, layout=None)
        tmap: Tx.handle("tensormap") = Tx.tvm_stack_alloca("tensormap", 1)
        if cond:
            Tx.call_packed(
                "runtime.cuTensorMapEncodeTiled",
                tmap, "float16", 2, A.data,
                32, 64, 64,
                16, 16,
                1, 1,
                0, 0, 0, 0
            )
        else:
            Tx.call_packed(
                "runtime.cuTensorMapEncodeTiled",
                tmap, "float16", 2, A.data,
                32, 64, 64,
                16, 16,
                1, 1,
                0, 0, 0, 0
            )
        with Tx.kernel():
            with Tx.cta():
                S = Tx.alloc_buffer((256,), "float16", scope="shared", layout=None)
                Tx.ptx.cp_async.bulk.tensor.s2g(2, Tx.address_of(S[0]), tmap, 0, 0)
    # fmt: on

    compare(before, after, LowerTIRx)


def test_lower_tirx_keep_different_cu_tensormaps():
    # fmt: off
    @Tx.prim_func(private=True, tirx=True)
    def before(A_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, (64, 32), "float16", scope="global")

        map1: Tx.handle("tensormap") = Tx.tvm_stack_alloca("tensormap", 1)
        map2: Tx.handle("tensormap") = Tx.tvm_stack_alloca("tensormap", 1)

        # First tensormap uses float16
        Tx.call_packed(
            "runtime.cuTensorMapEncodeTiled",
            map1, "float16", 2, A.data,
            32, 64, 64,
            16, 16,
            1, 1,
            0, 0, 0, 0
        )

        # Second tensormap differs in dtype (float32), so must not be deduped
        Tx.call_packed(
            "runtime.cuTensorMapEncodeTiled",
            map2, "float32", 2, A.data,
            32, 64, 64,
            16, 16,
            1, 1,
            0, 0, 0, 0
        )

        # Use both maps in body to confirm no rewrite across different params
        with Tx.kernel():
            with Tx.cta():
                S = Tx.alloc_buffer((16, 16), "float16", scope="shared")
                Tx.ptx.cp_async.bulk.tensor.s2g(2, S.ptr_to([0, 0]), map1, 0, 0)
                Tx.ptx.cp_async.bulk.tensor.s2g(2, S.ptr_to([0, 0]), map2, 0, 0)

    @Tx.prim_func(private=True, tirx=True)
    def after(A_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, (64, 32), "float16", layout=None)
        A_1 = Tx.decl_buffer((2048,), "float16", data=A.data, layout=None)
        map1: Tx.handle("tensormap") = Tx.tvm_stack_alloca("tensormap", 1)
        map2: Tx.handle("tensormap") = Tx.tvm_stack_alloca("tensormap", 1)
        Tx.call_packed(
            "runtime.cuTensorMapEncodeTiled",
            map1, "float16", 2, A.data,
            32, 64, 64,
            16, 16,
            1, 1,
            0, 0, 0, 0
        )
        Tx.call_packed(
            "runtime.cuTensorMapEncodeTiled",
            map2, "float32", 2, A.data,
            32, 64, 64,
            16, 16,
            1, 1,
            0, 0, 0, 0
        )
        with Tx.kernel():
            with Tx.cta():
                S = Tx.alloc_buffer((256,), "float16", scope="shared", layout=None)
                Tx.ptx.cp_async.bulk.tensor.s2g(2, Tx.address_of(S[0]), map1, 0, 0)
                Tx.ptx.cp_async.bulk.tensor.s2g(2, Tx.address_of(S[0]), map2, 0, 0)
    # fmt: on

    compare(before, after, LowerTIRx)


def test_lower_tirx_keep_different_swizzle():
    # fmt: off
    @Tx.prim_func(private=True, tirx=True)
    def before(A_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, (64, 32), "float16", scope="global")

        map1: Tx.handle("tensormap") = Tx.tvm_stack_alloca("tensormap", 1)
        map2: Tx.handle("tensormap") = Tx.tvm_stack_alloca("tensormap", 1)

        # Identical except swizzle_kind (0 vs 1)
        Tx.call_packed(
            "runtime.cuTensorMapEncodeTiled",
            map1, "float16", 2, A.data,
            32, 64, 64,
            16, 16,
            1, 1,
            0, 0, 0, 0
        )
        Tx.call_packed(
            "runtime.cuTensorMapEncodeTiled",
            map2, "float16", 2, A.data,
            32, 64, 64,
            16, 16,
            1, 1,
            0, 1, 0, 0   # swizzle differs
        )

        with Tx.kernel():
            with Tx.cta():
                S = Tx.alloc_buffer((256,), "float16", scope="shared", layout=None)
                Tx.ptx.cp_async.bulk.tensor.s2g(2, Tx.address_of(S[0]), map1, 0, 0)
                Tx.ptx.cp_async.bulk.tensor.s2g(2, Tx.address_of(S[0]), map2, 0, 0)

    @Tx.prim_func(private=True, tirx=True)
    def after(A_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, (64, 32), "float16", layout=None)
        A_1 = Tx.decl_buffer((2048,), "float16", data=A.data, layout=None)
        map1: Tx.handle("tensormap") = Tx.tvm_stack_alloca("tensormap", 1)
        map2: Tx.handle("tensormap") = Tx.tvm_stack_alloca("tensormap", 1)
        Tx.call_packed(
            "runtime.cuTensorMapEncodeTiled",
            map1, "float16", 2, A.data,
            32, 64, 64,
            16, 16,
            1, 1,
            0, 0, 0, 0
        )
        Tx.call_packed(
            "runtime.cuTensorMapEncodeTiled",
            map2, "float16", 2, A.data,
            32, 64, 64,
            16, 16,
            1, 1,
            0, 1, 0, 0
        )
        with Tx.kernel():
            with Tx.cta():
                S = Tx.alloc_buffer((256,), "float16", scope="shared", layout=None)
                Tx.ptx.cp_async.bulk.tensor.s2g(2, Tx.address_of(S[0]), map1, 0, 0)
                Tx.ptx.cp_async.bulk.tensor.s2g(2, Tx.address_of(S[0]), map2, 0, 0)
    # fmt: on

    compare(before, after, LowerTIRx)

    # fmt: off
    @Tx.prim_func(private=True, tirx=True)
    def before4_multi_view_get(in_buf: Tx.Buffer(64, "float32"), out: Tx.Buffer(64, "float32")) -> None:
        with Tx.kernel():
            bx, by, bz = Tx.cta_id([1, 1, 1], parent="kernel")
            warp_id = Tx.warp_id([1], parent="cta")
            lane_id = Tx.thread_id([32], parent="warp")

            with Tx.thread():
                A = Tx.alloc_buffer([2], dtype="float16", scope="local", layout=Tx.TileLayout(Tx.S[2 : 1]))
                B_layout = A.layout.tile(L_LANE, (32,), (2,))
                with Tx.warp():
                    # warp view of this load
                    B = A.view(64, layout=B_layout) # TODO(@bohan): consider making view API directly accepts shard parameters
                    B_1 = A.view(64, layout=B_layout) # TODO(@bohan): consider making view API directly accepts shard parameters
                    # B[i] = in_buf[i]
                    with Tx.thread():
                        # done by each thread
                        A_local = B.storage(2)
                        A_local[0] = Tx.float32(in_buf[lane_id * 2])
                        # done by each thread
                        A_local_1 = B_1.storage(2)
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
                        A_local = B.storage(2)
                        out[lane_id * 2] = Tx.float32(A_local[0])
                        # done by each thread
                        A_local_1 = B_1.storage(2)
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
        warp_id_in_cta: Tx.int32 = Tx.tvm_warp_shuffle(Tx.uint32(4294967295), threadIdx_x // 32, 0, 32, 32)
        bx: Tx.int32 = blockIdx_x
        by: Tx.int32 = blockIdx_y
        bz: Tx.int32 = blockIdx_z
        warp_id: Tx.int32 = warp_id_in_cta
        lane_id: Tx.int32 = threadIdx_x % 32
        with Tx.kernel():
            with Tx.thread():
                A = Tx.alloc_local((2,), "float16", layout=None)
                with Tx.warp():
                    B = Tx.decl_buffer((64,), "float16", data=A.data, scope="local", layout=None)
                    B_1 = Tx.decl_buffer((64,), "float16", data=A.data, scope="local", layout=None)
                    with Tx.thread():
                        A_local = Tx.decl_buffer((2,), "float16", data=A.data, scope="local", layout=None)
                        A_local[0] = Tx.Cast("float16", in_buf_1[threadIdx_x * 2])
                        A_local_1 = Tx.decl_buffer((2,), "float16", data=A.data, scope="local", layout=None)
                        A_local_1[1] = Tx.Cast("float16", in_buf_1[threadIdx_x * 2 + 1])
                with Tx.warp():
                    B = Tx.decl_buffer((64,), "float16", data=A.data, scope="local", layout=None)
                    B_1 = Tx.decl_buffer((64,), "float16", data=A.data, scope="local", layout=None)
                    with Tx.thread():
                        A_local = Tx.decl_buffer((2,), "float16", data=A.data, scope="local", layout=None)
                        out_1[threadIdx_x * 2] = Tx.Cast("float32", A_local[0])
                        A_local_1 = Tx.decl_buffer((2,), "float16", data=A.data, scope="local", layout=None)
                        out_1[threadIdx_x * 2 + 1] = Tx.Cast("float32", A_local_1[1])
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
        bx: Tx.int32 = blockIdx_x
        by: Tx.int32 = blockIdx_y
        bz: Tx.int32 = blockIdx_z
        tx: Tx.int32 = threadIdx_x
        with Tx.kernel():
            with Tx.thread():
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
        warp_id_in_cta: Tx.int32 = Tx.tvm_warp_shuffle(Tx.uint32(4294967295), threadIdx_x // 32, 0, 32, 32)
        cbx: Tx.int32 = clusterCtaIdx_x
        cby: Tx.int32 = clusterCtaIdx_y
        cbz: Tx.int32 = clusterCtaIdx_z
        bx: Tx.int32 = blockIdx_x
        by: Tx.int32 = blockIdx_y
        bz: Tx.int32 = blockIdx_z
        warp_id: Tx.int32 = warp_id_in_cta
        lane_id: Tx.int32 = threadIdx_x % 32
        with Tx.kernel():
            with Tx.thread():
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
        warp_id_in_cta: Tx.int32 = Tx.tvm_warp_shuffle(Tx.uint32(4294967295), threadIdx_x // 32, 0, 32, 32)
        bx: Tx.int32 = blockIdx_x
        by: Tx.int32 = blockIdx_y
        bz: Tx.int32 = blockIdx_z
        cbx: Tx.int32 = clusterCtaIdx_x
        cby: Tx.int32 = clusterCtaIdx_y
        cbz: Tx.int32 = clusterCtaIdx_z
        clx: Tx.int32 = Tx.ptx.fetch_register(32, "clusterid.x")
        cly: Tx.int32 = Tx.ptx.fetch_register(32, "clusterid.y")
        clz: Tx.int32 = Tx.ptx.fetch_register(32, "clusterid.z")
        wg_id: Tx.int32 = warp_id_in_cta // 4
        warp_id: Tx.int32 = warp_id_in_cta % 4
        lane_id: Tx.int32 = threadIdx_x % 32
        tid_in_wg: Tx.int32 = threadIdx_x % 128
        with Tx.kernel():
            with Tx.cta():
                with Tx.warpgroup():
                    with Tx.thread():
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
        warp_id_in_cta: Tx.int32 = Tx.tvm_warp_shuffle(Tx.uint32(4294967295), threadIdx_x // 32, 0, 32, 32)
        wg_id: Tx.int32 = warp_id_in_cta // 4
        bx: Tx.int32 = blockIdx_x
        by: Tx.int32 = blockIdx_y
        bz: Tx.int32 = blockIdx_z
        warp_id: Tx.int32 = warp_id_in_cta
        tx: Tx.int32 = threadIdx_x
        with Tx.kernel():
            with Tx.cta():
                with Tx.thread():
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
            warp_id_in_cta: Tx.int32 = Tx.tvm_warp_shuffle(Tx.uint32(4294967295), threadIdx_x // 32, 0, 32, 32)
            bx: Tx.int32 = blockIdx_x
            by: Tx.int32 = blockIdx_y
            bz: Tx.int32 = blockIdx_z
            warp_id: Tx.int32 = warp_id_in_cta
            tx: Tx.int32 = threadIdx_x
            with Tx.kernel():
                with Tx.cta():
                    with Tx.thread():
                        Tx.evaluate(bx + by + bz + warp_id + tx)
        blockIdx_x = Tx.launch_thread("blockIdx.x", 6)
        threadIdx_x = Tx.launch_thread("threadIdx.x", 256)
        blockIdx_y = Tx.launch_thread("blockIdx.y", 7)
        blockIdx_z = Tx.launch_thread("blockIdx.z", 8)
        warp_id_in_cta: Tx.int32 = Tx.tvm_warp_shuffle(Tx.uint32(4294967295), threadIdx_x // 32, 0, 32, 32)
        bx: Tx.int32 = blockIdx_x
        by: Tx.int32 = blockIdx_y
        bz: Tx.int32 = blockIdx_z
        warp_id: Tx.int32 = warp_id_in_cta
        tx: Tx.int32 = threadIdx_x
        with Tx.kernel():
            with Tx.cta():
                with Tx.thread():
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
        warp_id_in_cta: Tx.int32 = Tx.tvm_warp_shuffle(Tx.uint32(4294967295), threadIdx_x // 32, 0, 32, 32)
        bx: Tx.int32 = blockIdx_x
        by: Tx.int32 = blockIdx_y
        bz: Tx.int32 = blockIdx_z
        warp_id: Tx.int32 = warp_id_in_cta
        tx: Tx.int32 = threadIdx_x
        with Tx.kernel():
            if blockIdx_x >= 0 and blockIdx_x < 1 and blockIdx_y >= 0 and blockIdx_y < 2 and blockIdx_z >= 0 and blockIdx_z < 3:
                with Tx.cta():
                    if threadIdx_x >= 0 and threadIdx_x < 64:
                        with Tx.thread():
                            Tx.evaluate(tx)
                            Tx.evaluate(warp_id)
                    if Tx.ptx.elect_sync():
                        with Tx.thread():
                            Tx.evaluate(tx)
                    if tx == 0:
                        with Tx.thread():
                            Tx.evaluate(tx)
    # fmt: on

    compare(before, after, LowerTIRx)


def test_lower_scope_partition1():
    # fmt: off
    @Tx.prim_func(private=True, tirx=True)
    def before():
        with Tx.kernel():
            bx, by, bz = Tx.cta_id([3, 4, 5], parent="kernel")
            warp_id = Tx.warp_id([4], parent="cta")
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
        warp_id_in_cta: Tx.int32 = Tx.tvm_warp_shuffle(Tx.uint32(4294967295), threadIdx_x // 32, 0, 32, 32)
        bx: Tx.int32 = blockIdx_x
        by: Tx.int32 = blockIdx_y
        bz: Tx.int32 = blockIdx_z
        warp_id: Tx.int32 = warp_id_in_cta
        tx: Tx.int32 = threadIdx_x
        with Tx.kernel():
            with Tx.cta():
                if threadIdx_x >= 0 and threadIdx_x < 32:
                    with Tx.thread():
                        Tx.evaluate(tx)
                else:
                    if threadIdx_x >= 32 and threadIdx_x < 64:
                        with Tx.thread():
                            Tx.evaluate(tx)
                    else:
                        if threadIdx_x >= 64 and threadIdx_x < 96:
                            with Tx.thread():
                                Tx.evaluate(tx)
                        else:
                            if threadIdx_x >= 96 and threadIdx_x < 128:
                                with Tx.thread():
                                    Tx.evaluate(tx)
    # fmt: on

    compare(before, main, LowerTIRx)


def test_lower_scope_partition2():
    # fmt: off
    @Tx.prim_func(private=True, tirx=True)
    def before():
        with Tx.kernel():
            cbx, cby = Tx.cta_id([2, 1], parent="cluster")
            bx = Tx.cta_id([148], parent="kernel")
            wg_id = Tx.warpgroup_id([2], parent="cta")
            warp_id = Tx.warp_id([4], parent="warpgroup")
            lane_id = Tx.thread_id([32], parent="warp")
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
        warp_id_in_cta: Tx.int32 = Tx.tvm_warp_shuffle(Tx.uint32(4294967295), threadIdx_x // 32, 0, 32, 32)
        cbx: Tx.int32 = clusterCtaIdx_x
        cby: Tx.int32 = clusterCtaIdx_y
        bx: Tx.int32 = blockIdx_x
        wg_id: Tx.int32 = warp_id_in_cta // 4
        warp_id: Tx.int32 = warp_id_in_cta % 4
        lane_id: Tx.int32 = threadIdx_x % 32
        with Tx.kernel():
            with Tx.cta():
                if warp_id_in_cta // 4 >= 1 and warp_id_in_cta // 4 < 2:
                    with Tx.warpgroup():
                        if warp_id_in_cta % 4 >= 3 and warp_id_in_cta % 4 < 4:
                            with Tx.warp():
                                Tx.evaluate(warp_id)
                        else:
                            if warp_id_in_cta % 4 >= 2 and warp_id_in_cta % 4 < 3:
                                with Tx.warp():
                                    Tx.evaluate(warp_id)
                            else:
                                if warp_id_in_cta % 4 >= 0 and warp_id_in_cta % 4 < 1:
                                    with Tx.warp():
                                        Tx.evaluate(warp_id)
                else:
                    if warp_id_in_cta // 4 >= 0 and warp_id_in_cta // 4 < 1:
                        with Tx.warpgroup():
                            with Tx.thread():
                                Tx.evaluate(warp_id)
    # fmt: on

    compare(before, after, LowerTIRx)


def test_lower_layout():
    # fmt: off
    @Tx.prim_func(private=True, tirx=True)
    def before(A: Tx.Buffer((128, 32), "float16")) -> None:
        with Tx.kernel():
            bx, by, bz = Tx.cta_id([1, 1, 1], parent="kernel")
            warp_id = Tx.warp_id([4], parent="cta")
            lane_id = Tx.thread_id([32], parent="warp")
            tid = Tx.thread_id([128], parent="cta")

            with Tx.cta():
                A_smem = Tx.alloc_buffer([128, 32], dtype="float16", scope="shared", layout=Tx.SwizzleLayout(3, 3, 3))

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
        warp_id_in_cta: Tx.int32 = Tx.tvm_warp_shuffle(Tx.uint32(4294967295), threadIdx_x // 32, 0, 32, 32)
        bx: Tx.int32 = blockIdx_x
        by: Tx.int32 = blockIdx_y
        bz: Tx.int32 = blockIdx_z
        warp_id: Tx.int32 = warp_id_in_cta
        lane_id: Tx.int32 = threadIdx_x % 32
        tid: Tx.int32 = threadIdx_x
        with Tx.kernel():
            with Tx.cta():
                A_smem = Tx.alloc_shared((4096,), "float16", layout=None)
                with Tx.thread():
                    for tile in range(4):
                        for vec in Tx.vectorized(8):
                            A_smem[Tx.shift_left(Tx.bitwise_xor(tile * 128 + threadIdx_x, Tx.shift_right(Tx.bitwise_and(tile * 128 + threadIdx_x, 56), 3)), 3) + vec] = A_1[tile * 1024 + threadIdx_x * 8 + vec]
    # fmt: on

    compare(before, after, LowerTIRx)


def test_lower_opcall_fail():
    # fmt: off
    @Tx.prim_func(tirx=True)
    def test(A_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, (64,), "float32", scope="global")

        with Tx.kernel():
            bx, by, bz = Tx.cta_id([1, 1, 1], parent="kernel")
            warp_id = Tx.warp_id([1], parent="cta")
            lane_id = Tx.thread_id([32], parent="warp")
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
            bx = Tx.cta_id([1], parent="kernel")
            tx = Tx.thread_id([128], parent="cta")
            with Tx.cta():
                buf = Tx.alloc_buffer([1024], "uint8", scope="shared.dyn")
                A = Tx.decl_buffer([128], "float16", buf.data, elem_offset=32)

                with Tx.thread():
                    Tx.evaluate(A.access_ptr("rw", offset=A.elem_offset_of([64])))

    @Tx.prim_func(private=True, tirx=True)
    def after():
        blockIdx_x = Tx.launch_thread("blockIdx.x", 1)
        threadIdx_x = Tx.launch_thread("threadIdx.x", 128)
        bx: Tx.int32 = blockIdx_x
        tx: Tx.int32 = threadIdx_x
        with Tx.kernel():
            with Tx.cta():
                buf = Tx.alloc_buffer((1024,), "uint8", scope="shared.dyn", layout=None)
                A = Tx.decl_buffer((128,), "float16", data=buf.data, elem_offset=32, scope="shared.dyn", layout=None)
                with Tx.thread():
                    Tx.tvm_access_ptr(Tx.type_annotation("float16"), buf.data, Tx.Add(32, 64), Tx.Sub(128, 64), 3)
    # fmt: on

    compare(before, after, LowerTIRx)


def test_lower_separate_scope_id_def():
    # fmt: off
    @Tx.prim_func(private=True, tirx=True)
    def before():
        with Tx.kernel():
            bx = Tx.cta_id([1], parent="kernel")
            with Tx.cta():
                tx = Tx.thread_id([128], parent="cta")
                with Tx.thread()[tx == 0]:
                    Tx.evaluate(tx)

    @Tx.prim_func(private=True, tirx=True)
    def after():
        blockIdx_x = Tx.launch_thread("blockIdx.x", 1)
        threadIdx_x = Tx.launch_thread("threadIdx.x", 128)
        tx: Tx.int32 = threadIdx_x
        bx: Tx.int32 = blockIdx_x
        with Tx.kernel():
            with Tx.cta():
                if tx == 0:
                    with Tx.thread():
                        Tx.evaluate(tx)
    # fmt: on

    compare(before, after, LowerTIRx)


def test_lower_buffer_offset():
    # fmt: off
    @Tx.prim_func(private=True, tirx=True)
    def before():
        with Tx.kernel():
            bx = Tx.cta_id([1], parent="kernel")
            with Tx.cta():
                tx = Tx.thread_id([128], parent="cta")
                with Tx.thread():
                    A = Tx.alloc_buffer([64, 64], "float16", scope="local")
                    A0 = Tx.decl_buffer([64], "float16", A.data, elem_offset=A.elem_offset_of([32, 32]))
                    with Tx.thread():
                        Tx.evaluate(Tx.address_of(A0[32]))
    # fmt: on

    # fmt: off
    @Tx.prim_func(private=True, tirx=True)
    def after():
        blockIdx_x = Tx.launch_thread("blockIdx.x", 1)
        threadIdx_x = Tx.launch_thread("threadIdx.x", 128)
        tx: Tx.int32 = threadIdx_x
        bx: Tx.int32 = blockIdx_x
        with Tx.kernel():
            with Tx.cta():
                with Tx.thread():
                    A = Tx.alloc_local((4096,), "float16", layout=None)
                    A0 = Tx.decl_buffer((64,), "float16", data=A.data, elem_offset=2080, scope="local", layout=None)
                    with Tx.thread():
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
            self.C = Tx.decl_buffer([1], "float16", smem, elem_offset=0, scope="shared.dyn", name="C")

    def int_var1(val):
        buf = Tx.local_cell("int32")
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
        with Tx.kernel():
            with Tx.thread():
                smem = Tx.alloc_buffer([100], "uint8", scope="shared.dyn", layout=None)
                A = Tx.alloc_local((1,), "float16", layout=None)
                B = Tx.alloc_local((1,), "float16", layout=None)
                C = Tx.decl_buffer((1,), "float16", data=smem.data, elem_offset=0, scope="shared.dyn", layout=None)
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
    """alloc_buffer with thread-axis layout should lower to 1D physical buffer with memory-axis span."""
    # fmt: off
    @Tx.prim_func(private=True, tirx=True)
    def before(out: Tx.Buffer((128, 4), "float32")) -> None:
        with Tx.kernel():
            bx, by, bz = Tx.cta_id([1, 1, 1], parent="kernel")
            wg_id = Tx.warpgroup_id([1], parent="cta")
            warp_id = Tx.warp_id([4], parent="warpgroup")
            lane_id = Tx.thread_id([32], parent="warp")

            with Tx.warpgroup():
                with Tx.thread():
                    # Single-step alloc with thread-axis layout
                    reg_wg = Tx.alloc_buffer((128, 4), "float32", scope="local",
                                              layout=Tx.TileLayout(Tx.S[(128, 4) : (1 @ tid_in_wg, 1)]))
                    # Access via .storage() to decompose thread and memory axes
                    reg = reg_wg.storage(4)
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
        warp_id_in_cta: Tx.int32 = Tx.tvm_warp_shuffle(Tx.uint32(4294967295), threadIdx_x // 32, 0, 32, 32)
        bx: Tx.int32 = blockIdx_x
        by: Tx.int32 = blockIdx_y
        bz: Tx.int32 = blockIdx_z
        wg_id: Tx.int32 = warp_id_in_cta // 4
        warp_id: Tx.int32 = warp_id_in_cta % 4
        lane_id: Tx.int32 = threadIdx_x % 32
        with Tx.kernel():
            with Tx.warpgroup():
                with Tx.thread():
                    reg_wg = Tx.alloc_local((4,), layout=None)
                    reg = Tx.decl_buffer((4,), data=reg_wg.data, scope="local", layout=None)
                    for i in range(4):
                        reg[i] = out_1[warp_id_in_cta % 4 * 128 + threadIdx_x % 32 * 4 + i]
    # fmt: on

    compare(before, after, LowerTIRx)


if __name__ == "__main__":
    tvm.testing.main()
