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
from tvm.script import tir as T
from tvm.script import tirp as Tp
from tvm.script.ir_builder import IRBuilder
from tvm.tir.event import EventImpl
from tvm.tir.function import PrimFunc
from tvm.tir.transform import LowerTIRp


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


L_LANE = T.TileLayout(shard=([32], [(1, "laneid")]))


def test_lower_view_get():
    # fmt: off
    @T.prim_func(private=True, tirp=True)
    def before1(in_buf: T.Buffer(64, "float32"), out: T.Buffer(64, "float32")) -> None:
        with T.kernel():
            bx, by, bz = T.cta_id([1, 1, 1], parent="kernel")
            warp_id = T.warp_id([1], parent="cta")
            lane_id = T.thread_id([32], parent="warp")

            with T.thread():
                A = T.alloc_buffer([2], dtype="float16", scope="local", layout=T.TileLayout(shard=([2], [1])))
                B_layout = A.layout.tile(L_LANE, (32,), (2,))
                # load in_buf into A
                with T.warp():
                    # warp view of this load
                    B = A.view(64, layout=B_layout)
                    # B[i] = in_buf[i]
                    with T.thread():
                        # done by each thread
                        A_local = B.storage(2)
                        for i in T.vectorized(2):
                            A_local[i] = T.float32(in_buf[lane_id * 2 + i])
                # write A into out
                with T.warp():
                    # warp view of this write
                    B = A.view(64, layout=B_layout)
                    # out[i] = B[i]
                    with T.thread():
                        # done by each thread
                        A_local = B.storage(2)
                        for i in T.vectorized(2):
                            out[lane_id * 2 + i] = T.float32(A_local[i])

    @T.prim_func(private=True, tirp=True)
    def after1(in_buf_handle: T.handle, out_handle: T.handle):
        in_buf = T.match_buffer(in_buf_handle, (64,), layout=None)
        out = T.match_buffer(out_handle, (64,), layout=None)
        out_1 = T.decl_buffer((64,), data=out.data, layout=None)
        in_buf_1 = T.decl_buffer((64,), data=in_buf.data, layout=None)
        blockIdx_x = T.launch_thread("blockIdx.x", 1)
        threadIdx_x = T.launch_thread("threadIdx.x", 32)
        blockIdx_y = T.launch_thread("blockIdx.y", 1)
        blockIdx_z = T.launch_thread("blockIdx.z", 1)
        with T.kernel():
            with T.thread():
                A = T.alloc_local((2,), "float16", layout=None)
                with T.warp():
                    B = T.decl_buffer((64,), "float16", data=A.data, scope="local", layout=None)
                    with T.thread():
                        A_local = T.decl_buffer((2,), "float16", data=A.data, scope="local", layout=None)
                        for i in T.vectorized(2):
                            A_local[i] = T.Cast("float16", in_buf_1[threadIdx_x * 2 + i])
                with T.warp():
                    B = T.decl_buffer((64,), "float16", data=A.data, scope="local", layout=None)
                    with T.thread():
                        A_local = T.decl_buffer((2,), "float16", data=A.data, scope="local", layout=None)
                        for i in T.vectorized(2):
                            out_1[threadIdx_x * 2 + i] = T.Cast("float32", A_local[i])
    # fmt: on

    compare(before1, after1, LowerTIRp)

    # fmt: off
    @T.prim_func(private=True, tirp=True)
    def before2(in_buf: T.Buffer((16, 16), "float32"), out: T.Buffer((16, 16), "float32")) -> None:
        with T.kernel():
            bx, by, bz = T.cta_id([1, 1, 1], parent="kernel")
            warp_id = T.warp_id([1], parent="cta")
            lane_id = T.thread_id([32], parent="warp")

            with T.thread():
                atom = T.TileLayout(shard=([1, 2], [2, 1]))
                tile = T.TileLayout(shard=([2, 2], [2, 1]))
                warp_atom = atom.tile(L_LANE, (8, 4), (1, 2))
                
                A = T.alloc_buffer([4, 2], dtype="float32", scope="local", layout=atom.tile(tile, (2, 2), (1, 2)))
                B_layout = warp_atom.tile(tile, (2, 2), (8, 8))
                
                # load in_buf into A
                with T.warp():
                    # warp view of this load
                    B = A.view(16, 16, layout=B_layout)
                    with T.thread():
                        # done by each thread
                        A_local = B.storage(2, 2, 2)
                        for i in T.unroll(4):
                            for j in T.vectorized(2):
                                A_local[i // 2, i % 2, j] = in_buf[i // 2 * 8 + lane_id // 4, i % 2 * 8 + lane_id % 4 + j]
                # write A into out
                with T.warp():
                    # warp view of this write
                    B = A.view(16, 16, layout=B_layout)
                    with T.thread():
                        # done by each thread
                        A_local = B.storage(8)
                        for i in T.vectorized(2):
                            out[lane_id // 4 * 8 + i // 2 * 8 + lane_id % 4, lane_id % 4 * 2 + i % 2] = A_local[i]

    @T.prim_func(private=True, tirp=True)
    def after2(in_buf_handle: T.handle, out_handle: T.handle):
        in_buf = T.match_buffer(in_buf_handle, (16, 16), layout=None)
        out = T.match_buffer(out_handle, (16, 16), layout=None)
        out_1 = T.decl_buffer((256,), data=out.data, layout=None)
        in_buf_1 = T.decl_buffer((256,), data=in_buf.data, layout=None)
        blockIdx_x = T.launch_thread("blockIdx.x", 1)
        threadIdx_x = T.launch_thread("threadIdx.x", 32)
        blockIdx_y = T.launch_thread("blockIdx.y", 1)
        blockIdx_z = T.launch_thread("blockIdx.z", 1)
        with T.kernel():
            with T.thread():
                A = T.alloc_local((8,), layout=None)
                with T.warp():
                    B = T.decl_buffer((256,), data=A.data, scope="local", layout=None)
                    with T.thread():
                        A_local = T.decl_buffer((8,), data=A.data, scope="local", layout=None)
                        for i in T.unroll(4):
                            for j in T.vectorized(2):
                                A_local[i * 2 + j] = in_buf_1[i // 2 * 128 + threadIdx_x // 4 * 16 + i % 2 * 8 + j + threadIdx_x % 4]
                with T.warp():
                    B = T.decl_buffer((256,), data=A.data, scope="local", layout=None)
                    with T.thread():
                        A_local = T.decl_buffer((8,), data=A.data, scope="local", layout=None)
                        for i in T.vectorized(2):
                            out_1[threadIdx_x // 4 * 128 + threadIdx_x % 4 * 18 + i] = A_local[i]
    # fmt: on

    compare(before2, after2, LowerTIRp)

    # fmt: off
    @T.prim_func(private=True, tirp=True)
    def before3_wgmma_layout(in_buf: T.Buffer((128, 128), "float32"), out: T.Buffer((128, 128), "float32")) -> None:
        with T.kernel():
            bx, by, bz = T.cta_id([1, 1, 1], parent="kernel")
            wg_id = T.warpgroup_id([2], parent="cta")
            warp_id_in_wg = T.warp_id([4], parent="warpgroup")
            lane_id = T.thread_id([32], parent="warp")

            with T.thread():
                # shard from thread to warp
                atom = T.TileLayout((1, 2))
                warp_atom = atom.tile(L_LANE, (8, 4), (1, 2))
                # tile
                tile = T.TileLayout(shard=([2, 128 // 8], [1, 2])) # column-major
                warp_layout = warp_atom.tile(tile, (2, 128 // 8), (8, 8))
                # shard from warp to cta
                L_warp = T.TileLayout(shard=([8], [(1, "warpid")]))
                layout = warp_layout.tile(L_warp, (8, 1), (16, 128))
                # alloc
                acc = T.alloc_buffer([64,], dtype="float32", scope="local", layout=atom.tile(tile, (2, 128 // 8), (1, 2)))

                # load in_buf into acc
                with T.cta():
                    # cta view of this load
                    A = acc.view(128, 128, layout=layout)
                    with T.thread():
                        # done by each thread
                        acc_local = A.storage(16, 2, 2, layout=atom.tile(tile, (2, 128 // 8), (1, 2)))
                        for i in T.serial(128 // 8):
                            for j in T.unroll(2):
                                for vec in T.vectorized(2):
                                    acc_local[i, j, vec] = in_buf[wg_id * 64 + warp_id_in_wg * 16 + j * 8 + lane_id // 4, i * 8 + lane_id % 4 * 2 + vec]

                # write acc into out
                with T.cta():
                    # cta view of this write
                    A = acc.view(128, 128, layout=layout)
                    with T.thread():
                        # done by each thread
                        acc_local = A.storage(64, layout=atom.tile(tile, (2, 128 // 8), (1, 2)))
                        for i in T.serial(128 // 8):
                            for j in T.unroll(2):
                                for vec in T.vectorized(2):
                                    out[wg_id * 64 + warp_id_in_wg * 16 + j * 8 + lane_id // 4, i * 8 + lane_id % 4 * 2 + vec] = acc_local[i * 4 + j * 2 + vec]

    @T.prim_func(private=True, tirp=True)
    def after3_wgmma_layout(in_buf_handle: T.handle, out_handle: T.handle):
        in_buf = T.match_buffer(in_buf_handle, (128, 128), layout=None)
        out = T.match_buffer(out_handle, (128, 128), layout=None)
        out_1 = T.decl_buffer((16384,), data=out.data, layout=None)
        in_buf_1 = T.decl_buffer((16384,), data=in_buf.data, layout=None)
        blockIdx_x = T.launch_thread("blockIdx.x", 1)
        threadIdx_x = T.launch_thread("threadIdx.x", 256)
        blockIdx_y = T.launch_thread("blockIdx.y", 1)
        blockIdx_z = T.launch_thread("blockIdx.z", 1)
        with T.kernel():
            with T.thread():
                acc = T.alloc_local((64,), layout=None)
                with T.cta():
                    A = T.decl_buffer((16384,), data=acc.data, scope="local", layout=None)
                    with T.thread():
                        acc_local = T.decl_buffer((64,), data=acc.data, scope="local", layout=None)
                        for i in range(16):
                            for j in T.unroll(2):
                                for vec in T.vectorized(2):
                                    acc_local[i % 8 * 8 + j * 4 + i // 8 * 2 + vec] = in_buf_1[threadIdx_x // 32 * 2048 + j * 1024 + threadIdx_x % 32 // 4 * 128 + i * 8 + threadIdx_x % 4 * 2 + vec]
                with T.cta():
                    A = T.decl_buffer((16384,), data=acc.data, scope="local", layout=None)
                    with T.thread():
                        acc_local = T.decl_buffer((64,), data=acc.data, scope="local", layout=None)
                        for i in range(16):
                            for j in T.unroll(2):
                                for vec in T.vectorized(2):
                                    out_1[threadIdx_x // 32 * 2048 + j * 1024 + threadIdx_x % 32 // 4 * 128 + i * 8 + threadIdx_x % 4 * 2 + vec] = acc_local[i % 8 * 8 + j * 4 + i // 8 * 2 + vec]
    # fmt: on

    compare(before3_wgmma_layout, after3_wgmma_layout, LowerTIRp)


def test_lower_tirp_dedup_cu_tensormap():
    # fmt: off
    @T.prim_func(private=True, tirp=True)
    def before(A_ptr: T.handle, cond: T.bool) -> None:
        A = T.match_buffer(A_ptr, (64, 32), "float16", scope="global")

        # Two tensormap allocations
        map1: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
        map2: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
        map3: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)

        if cond:
            # Two identical cuTensorMapEncodeTiled initializations
            T.call_packed(
                "runtime.cuTensorMapEncodeTiled",
                map1, "float16", 2, A.data,
                32, 64,    # global_shape (reversed)
                64,        # global stride
                16, 16,    # shared_shape (boxDim)
                1, 1,      # shared_strides
                0, 0, 0, 0 # interleave, swizzle, l2, oob
            )
            T.call_packed(
                "runtime.cuTensorMapEncodeTiled",
                map2, "float16", 2, A.data,
                32, 64, 64,
                16, 16,
                1, 1,
                0, 0, 0, 0
            )
            # unused map, should be removed in the pass
            T.call_packed(
                "runtime.cuTensorMapEncodeTiled",
                map3, "float16", 2, A.data,
                32, 64, 64,
                16, 16,
                1, 1,
                0, 0, 0, 0
            )
        else:
            T.call_packed(
                "runtime.cuTensorMapEncodeTiled",
                map1, "float16", 2, A.data,
                32, 64, 64,
                16, 16,
                1, 1,
                0, 0, 0, 0
            )
        
        # Use site of map2, should be rewritten to map1
        with T.kernel():
            with T.cta():
                S = T.alloc_buffer((16, 16), "float16", scope="shared")
                T.ptx.cp_async.bulk.tensor.s2g(2, S.ptr_to([0, 0]), map2, 0, 0)

    @T.prim_func(private=True, tirp=True)
    def after(A_ptr: T.handle, cond: T.bool) -> None:
        A = T.match_buffer(A_ptr, (64, 32), "float16", layout=None)
        A_1 = T.decl_buffer((2048,), "float16", data=A.data, layout=None)
        tmap: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
        if cond:
            T.call_packed(
                "runtime.cuTensorMapEncodeTiled",
                tmap, "float16", 2, A.data,
                32, 64, 64,
                16, 16,
                1, 1,
                0, 0, 0, 0
            )
        else:
            T.call_packed(
                "runtime.cuTensorMapEncodeTiled",
                tmap, "float16", 2, A.data,
                32, 64, 64,
                16, 16,
                1, 1,
                0, 0, 0, 0
            )
        with T.kernel():
            with T.cta():
                S = T.alloc_buffer((256,), "float16", scope="shared", layout=None)
                T.ptx.cp_async.bulk.tensor.s2g(2, T.address_of(S[0]), tmap, 0, 0)
    # fmt: on

    compare(before, after, LowerTIRp)


def test_lower_tirp_keep_different_cu_tensormaps():
    # fmt: off
    @T.prim_func(private=True, tirp=True)
    def before(A_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, (64, 32), "float16", scope="global")

        map1: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
        map2: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)

        # First tensormap uses float16
        T.call_packed(
            "runtime.cuTensorMapEncodeTiled",
            map1, "float16", 2, A.data,
            32, 64, 64,
            16, 16,
            1, 1,
            0, 0, 0, 0
        )

        # Second tensormap differs in dtype (float32), so must not be deduped
        T.call_packed(
            "runtime.cuTensorMapEncodeTiled",
            map2, "float32", 2, A.data,
            32, 64, 64,
            16, 16,
            1, 1,
            0, 0, 0, 0
        )

        # Use both maps in body to confirm no rewrite across different params
        with T.kernel():
            with T.cta():
                S = T.alloc_buffer((16, 16), "float16", scope="shared")
                T.ptx.cp_async.bulk.tensor.s2g(2, S.ptr_to([0, 0]), map1, 0, 0)
                T.ptx.cp_async.bulk.tensor.s2g(2, S.ptr_to([0, 0]), map2, 0, 0)

    @T.prim_func(private=True, tirp=True)
    def after(A_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, (64, 32), "float16", layout=None)
        A_1 = T.decl_buffer((2048,), "float16", data=A.data, layout=None)
        map1: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
        map2: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
        T.call_packed(
            "runtime.cuTensorMapEncodeTiled",
            map1, "float16", 2, A.data,
            32, 64, 64,
            16, 16,
            1, 1,
            0, 0, 0, 0
        )
        T.call_packed(
            "runtime.cuTensorMapEncodeTiled",
            map2, "float32", 2, A.data,
            32, 64, 64,
            16, 16,
            1, 1,
            0, 0, 0, 0
        )
        with T.kernel():
            with T.cta():
                S = T.alloc_buffer((256,), "float16", scope="shared", layout=None)
                T.ptx.cp_async.bulk.tensor.s2g(2, T.address_of(S[0]), map1, 0, 0)
                T.ptx.cp_async.bulk.tensor.s2g(2, T.address_of(S[0]), map2, 0, 0)
    # fmt: on

    compare(before, after, LowerTIRp)


def test_lower_tirp_keep_different_swizzle():
    # fmt: off
    @T.prim_func(private=True, tirp=True)
    def before(A_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, (64, 32), "float16", scope="global")

        map1: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
        map2: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)

        # Identical except swizzle_kind (0 vs 1)
        T.call_packed(
            "runtime.cuTensorMapEncodeTiled",
            map1, "float16", 2, A.data,
            32, 64, 64,
            16, 16,
            1, 1,
            0, 0, 0, 0
        )
        T.call_packed(
            "runtime.cuTensorMapEncodeTiled",
            map2, "float16", 2, A.data,
            32, 64, 64,
            16, 16,
            1, 1,
            0, 1, 0, 0   # swizzle differs
        )

        with T.kernel():
            with T.cta():
                S = T.alloc_buffer((256,), "float16", scope="shared", layout=None)
                T.ptx.cp_async.bulk.tensor.s2g(2, T.address_of(S[0]), map1, 0, 0)
                T.ptx.cp_async.bulk.tensor.s2g(2, T.address_of(S[0]), map2, 0, 0)

    @T.prim_func(private=True, tirp=True)
    def after(A_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, (64, 32), "float16", layout=None)
        A_1 = T.decl_buffer((2048,), "float16", data=A.data, layout=None)
        map1: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
        map2: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
        T.call_packed(
            "runtime.cuTensorMapEncodeTiled",
            map1, "float16", 2, A.data,
            32, 64, 64,
            16, 16,
            1, 1,
            0, 0, 0, 0
        )
        T.call_packed(
            "runtime.cuTensorMapEncodeTiled",
            map2, "float16", 2, A.data,
            32, 64, 64,
            16, 16,
            1, 1,
            0, 1, 0, 0
        )
        with T.kernel():
            with T.cta():
                S = T.alloc_buffer((256,), "float16", scope="shared", layout=None)
                T.ptx.cp_async.bulk.tensor.s2g(2, T.address_of(S[0]), map1, 0, 0)
                T.ptx.cp_async.bulk.tensor.s2g(2, T.address_of(S[0]), map2, 0, 0)
    # fmt: on

    compare(before, after, LowerTIRp)

    # fmt: off
    @T.prim_func(private=True, tirp=True)
    def before4_multi_view_get(in_buf: T.Buffer(64, "float32"), out: T.Buffer(64, "float32")) -> None:
        with T.kernel():
            bx, by, bz = T.cta_id([1, 1, 1], parent="kernel")
            warp_id = T.warp_id([1], parent="cta")
            lane_id = T.thread_id([32], parent="warp")

            with T.thread():
                A = T.alloc_buffer([2], dtype="float16", scope="local", layout=T.TileLayout(shard=([2], [1])))
                B_layout = A.layout.tile(L_LANE, (32,), (2,))
                with T.warp():
                    # warp view of this load
                    B = A.view(64, layout=B_layout) # TODO(@bohan): consider making view API directly accepts shard parameters
                    B_1 = A.view(64, layout=B_layout) # TODO(@bohan): consider making view API directly accepts shard parameters
                    # B[i] = in_buf[i]
                    with T.thread():
                        # done by each thread
                        A_local = B.storage(2)
                        A_local[0] = T.float32(in_buf[lane_id * 2])
                        # done by each thread
                        A_local_1 = B_1.storage(2)
                        A_local_1[1] = T.float32(in_buf[lane_id * 2 + 1])
                """
                write A into out
                """
                with T.warp():
                    # warp view of this write
                    B = A.view(64, layout=B_layout)
                    B_1 = A.view(64, layout=B_layout)
                    # out[i] = B[i]
                    with T.thread():
                        # done by each thread
                        A_local = B.storage(2)
                        out[lane_id * 2] = T.float32(A_local[0])
                        # done by each thread
                        A_local_1 = B_1.storage(2)
                        out[lane_id * 2 + 1] = T.float32(A_local_1[1])

    @T.prim_func(private=True, tirp=True)
    def after4_multi_view_get(in_buf_handle: T.handle, out_handle: T.handle):
        in_buf = T.match_buffer(in_buf_handle, (64,), layout=None)
        out = T.match_buffer(out_handle, (64,), layout=None)
        out_1 = T.decl_buffer((64,), data=out.data, layout=None)
        in_buf_1 = T.decl_buffer((64,), data=in_buf.data, layout=None)
        blockIdx_x = T.launch_thread("blockIdx.x", 1)
        threadIdx_x = T.launch_thread("threadIdx.x", 32)
        blockIdx_y = T.launch_thread("blockIdx.y", 1)
        blockIdx_z = T.launch_thread("blockIdx.z", 1)
        with T.kernel():
            with T.thread():
                A = T.alloc_local((2,), "float16", layout=None)
                with T.warp():
                    B = T.decl_buffer((64,), "float16", data=A.data, scope="local", layout=None)
                    B_1 = T.decl_buffer((64,), "float16", data=A.data, scope="local", layout=None)
                    with T.thread():
                        A_local = T.decl_buffer((2,), "float16", data=A.data, scope="local", layout=None)
                        A_local[0] = T.Cast("float16", in_buf_1[threadIdx_x * 2])
                        A_local_1 = T.decl_buffer((2,), "float16", data=A.data, scope="local", layout=None)
                        A_local_1[1] = T.Cast("float16", in_buf_1[threadIdx_x * 2 + 1])
                with T.warp():
                    B = T.decl_buffer((64,), "float16", data=A.data, scope="local", layout=None)
                    B_1 = T.decl_buffer((64,), "float16", data=A.data, scope="local", layout=None)
                    with T.thread():
                        A_local = T.decl_buffer((2,), "float16", data=A.data, scope="local", layout=None)
                        out_1[threadIdx_x * 2] = T.Cast("float32", A_local[0])
                        A_local_1 = T.decl_buffer((2,), "float16", data=A.data, scope="local", layout=None)
                        out_1[threadIdx_x * 2 + 1] = T.Cast("float32", A_local_1[1])    
    # fmt: on

    compare(before4_multi_view_get, after4_multi_view_get, LowerTIRp)


def test_lower_scope_id():
    # fmt: off
    @T.prim_func(private=True, tirp=True)
    def before1() -> None:
        with T.kernel():
            bx, by, bz = T.cta_id([3, 4, 5], parent="kernel")
            tx = T.thread_id([32], parent="cta")

            with T.thread():
                T.evaluate(bx + by + bz + tx)

    @T.prim_func(private=True, tirp=True)
    def after1() -> None:
        blockIdx_x = T.launch_thread("blockIdx.x", 3)
        threadIdx_x = T.launch_thread("threadIdx.x", 32)
        blockIdx_y = T.launch_thread("blockIdx.y", 4)
        blockIdx_z = T.launch_thread("blockIdx.z", 5)
        with T.kernel():
            with T.thread():
                T.evaluate(blockIdx_x + blockIdx_y + blockIdx_z + threadIdx_x)
    # fmt: on
    compare(before1, after1, LowerTIRp)

    # fmt: off
    @T.prim_func(private=True, tirp=True)
    def before2() -> None:
        with T.kernel():
            cbx, cby, cbz = T.cta_id([2, 2, 2], parent="cluster")
            bx, by, bz = T.cta_id([8, 8, 8], parent="kernel")
            warp_id = T.warp_id([4], parent="cta")
            lane_id = T.thread_id([32], parent="warp")

            with T.thread():
                T.evaluate(bx + by + bz + warp_id + lane_id + cbx + cby + cbz)

    @T.prim_func(private=True, tirp=True)
    def after2() -> None:
        clusterCtaIdx_x = T.launch_thread("clusterCtaIdx.x", 2)
        blockIdx_z = T.launch_thread("blockIdx.z", 8)
        clusterCtaIdx_y = T.launch_thread("clusterCtaIdx.y", 2)
        clusterCtaIdx_z = T.launch_thread("clusterCtaIdx.z", 2)
        blockIdx_x = T.launch_thread("blockIdx.x", 8)
        threadIdx_x = T.launch_thread("threadIdx.x", 128)
        blockIdx_y = T.launch_thread("blockIdx.y", 8)
        with T.kernel():
            with T.thread():
                T.evaluate(blockIdx_x + blockIdx_y + blockIdx_z + threadIdx_x // 32 + threadIdx_x % 32 + clusterCtaIdx_x + clusterCtaIdx_y + clusterCtaIdx_z)

    # fmt: on
    compare(before2, after2, LowerTIRp)

    # fmt: off
    @T.prim_func(private=True, tirp=True)
    def before3() -> None:
        with T.kernel():
            bx, by, bz = T.cta_id([8, 10, 12], parent="kernel")
            cbx, cby, cbz = T.cta_id([2, 2, 1], parent="cluster")
            clx, cly, clz = T.cluster_id([4, 5, 12], parent="kernel")
            wg_id = T.warpgroup_id([3], parent="cta")
            warp_id = T.warp_id([4], parent="warpgroup")
            lane_id = T.thread_id([32], parent="warp")
            tid_in_wg = T.thread_id([128], parent="warpgroup")
            with T.cta():
                with T.warpgroup():
                    with T.thread():
                        T.evaluate(bx + by + bz)
                        T.evaluate(cbx + cby + cbz)
                        T.evaluate(clx + cly + clz)
                        T.evaluate(wg_id + warp_id + lane_id + tid_in_wg)

    @T.prim_func(private=True, tirp=True)   
    def after3() -> None:
        clusterCtaIdx_x = T.launch_thread("clusterCtaIdx.x", 2)
        blockIdx_z = T.launch_thread("blockIdx.z", 12)
        clusterCtaIdx_y = T.launch_thread("clusterCtaIdx.y", 2)
        clusterCtaIdx_z = T.launch_thread("clusterCtaIdx.z", 1)
        blockIdx_x = T.launch_thread("blockIdx.x", 8)
        threadIdx_x = T.launch_thread("threadIdx.x", 384)
        blockIdx_y = T.launch_thread("blockIdx.y", 10)
        with T.kernel():
            with T.cta():
                with T.warpgroup():
                    with T.thread():
                        T.evaluate(blockIdx_x + blockIdx_y + blockIdx_z)
                        T.evaluate(clusterCtaIdx_x + clusterCtaIdx_y + clusterCtaIdx_z)
                        T.evaluate(T.ptx.fetch_register(32, "clusterid.x") + T.ptx.fetch_register(32, "clusterid.y") + T.ptx.fetch_register(32, "clusterid.z"))
                        T.evaluate(threadIdx_x // 128 + threadIdx_x % 128 // 32 + threadIdx_x % 32 + threadIdx_x % 128)
    # fmt: on

    compare(before3, after3, LowerTIRp)


def test_lower_scope_id2():
    # fmt: off
    @T.macro
    def func(warp_id, tx):
        with T.cta():
            wg_id = T.warpgroup_id([2], parent="cta")
            with T.thread():
                T.evaluate(wg_id + warp_id + tx)

    @T.prim_func(private=True, tirp=True)
    def before():
        with T.kernel():
            bx, by, bz = T.cta_id([3, 4, 5], parent="kernel")
            warp_id = T.warp_id([8], parent="cta")
            tx = T.thread_id([256], parent="cta")
            
            func(warp_id, tx)
    # fmt: on

    # fmt: off
    @T.prim_func(private=True, tirp=True)
    def after():
        blockIdx_x = T.launch_thread("blockIdx.x", 3)
        threadIdx_x = T.launch_thread("threadIdx.x", 256)
        blockIdx_y = T.launch_thread("blockIdx.y", 4)
        blockIdx_z = T.launch_thread("blockIdx.z", 5)
        with T.kernel():
            with T.cta():
                with T.thread():
                    T.evaluate(threadIdx_x // 128 + threadIdx_x // 32 + threadIdx_x)
    # fmt: on

    compare(before, after, LowerTIRp)


def test_lower_scope_id3():
    # fmt: off
    @T.prim_func(private=True, tirp=True)
    def before():
        with T.kernel():
            bx, by, bz = T.cta_id([3, 4, 5], parent="kernel")
            warp_id = T.warp_id([4], parent="cta")
            tx = T.thread_id([128], parent="cta")

            with T.cta():
                with T.thread():
                    T.evaluate(bx + by + bz + warp_id + tx)
        with T.kernel():
            bx, by, bz = T.cta_id([6, 7, 8], parent="kernel")
            warp_id = T.warp_id([8], parent="cta")
            tx = T.thread_id([256], parent="cta")

            with T.cta():
                with T.thread():
                    T.evaluate(bx + by + bz + warp_id + tx)
    # fmt: on

    # fmt: off
    @T.prim_func(private=True, tirp=True)
    def after():
        with T.launch_thread("blockIdx.x", 3) as blockIdx_x:
            threadIdx_x = T.launch_thread("threadIdx.x", 128)
            blockIdx_y = T.launch_thread("blockIdx.y", 4)
            blockIdx_z = T.launch_thread("blockIdx.z", 5)
            with T.kernel():
                with T.cta():
                    with T.thread():
                        T.evaluate(blockIdx_x + blockIdx_y + blockIdx_z + threadIdx_x // 32 + threadIdx_x)
        blockIdx_x = T.launch_thread("blockIdx.x", 6)
        threadIdx_x = T.launch_thread("threadIdx.x", 256)
        blockIdx_y = T.launch_thread("blockIdx.y", 7)
        blockIdx_z = T.launch_thread("blockIdx.z", 8)
        with T.kernel():
            with T.cta():
                with T.thread():
                    T.evaluate(blockIdx_x + blockIdx_y + blockIdx_z + threadIdx_x // 32 + threadIdx_x)
    # fmt: on

    compare(before, after, LowerTIRp)


def test_lower_scope_slice():
    # fmt: off
    @T.prim_func(private=True, tirp=True)
    def before():
        with T.kernel():
            bx, by, bz = T.cta_id([3, 4, 5], parent="kernel")
            warp_id = T.warp_id([4], parent="cta")
            tx = T.thread_id([128], parent="cta")

            with T.cta()[0:1, 0:2, 0:3]:
                with T.thread()[0:64]:
                    T.evaluate(tx)
                    T.evaluate(warp_id)
                with T.thread()[T.ptx.elect_sync(0xFFFFFFFF)]:
                    T.evaluate(tx)
                with T.thread()[tx == 0]:
                    T.evaluate(tx)

    @T.prim_func(private=True, tirp=True)
    def after():
        blockIdx_x = T.launch_thread("blockIdx.x", 3)
        threadIdx_x = T.launch_thread("threadIdx.x", 128)
        blockIdx_y = T.launch_thread("blockIdx.y", 4)
        blockIdx_z = T.launch_thread("blockIdx.z", 5)
        with T.kernel():
            if blockIdx_x >= 0 and blockIdx_x < 1 and blockIdx_y >= 0 and blockIdx_y < 2 and blockIdx_z >= 0 and blockIdx_z < 3:
                with T.cta():
                    if threadIdx_x >= 0 and threadIdx_x < 64:
                        with T.thread():
                            T.evaluate(threadIdx_x)
                            T.evaluate(threadIdx_x // 32)
                    if T.ptx.elect_sync(0xFFFFFFFF):
                        with T.thread():
                            T.evaluate(threadIdx_x)
                    if threadIdx_x == 0:
                        with T.thread():
                            T.evaluate(threadIdx_x)
    # fmt: on

    compare(before, after, LowerTIRp)


def test_lower_scope_partition1():
    # fmt: off
    @T.prim_func(private=True, tirp=True)
    def before():
        with T.kernel():
            bx, by, bz = T.cta_id([3, 4, 5], parent="kernel")
            warp_id = T.warp_id([4], parent="cta")
            tx = T.thread_id([128], parent="cta")

            with T.cta():
                T.block_attr({"tirp.scope_partition": True})
                with T.thread()[0:32]:
                    T.evaluate(tx)
                with T.thread()[32:64]:
                    T.evaluate(tx)
                with T.thread()[64:96]:
                    T.evaluate(tx)
                with T.thread()[96:128]:
                    T.evaluate(tx)

    @T.prim_func(private=True, tirp=True)
    def main():
        blockIdx_x = T.launch_thread("blockIdx.x", 3)
        threadIdx_x = T.launch_thread("threadIdx.x", 128)
        blockIdx_y = T.launch_thread("blockIdx.y", 4)
        blockIdx_z = T.launch_thread("blockIdx.z", 5)
        with T.kernel():
            with T.cta():
                if threadIdx_x >= 0 and threadIdx_x < 32:
                    with T.thread():
                        T.evaluate(threadIdx_x)
                elif threadIdx_x >= 32 and threadIdx_x < 64:
                    with T.thread():
                        T.evaluate(threadIdx_x)
                elif threadIdx_x >= 64 and threadIdx_x < 96:
                    with T.thread():
                        T.evaluate(threadIdx_x)
                elif threadIdx_x >= 96 and threadIdx_x < 128:
                    with T.thread():
                        T.evaluate(threadIdx_x)
    # fmt: on

    compare(before, main, LowerTIRp)


def test_lower_scope_partition2():
    # fmt: off
    @T.prim_func(private=True, tirp=True)
    def before():
        with T.kernel():
            cbx, cby = T.cta_id([2, 1], parent="cluster")
            bx = T.cta_id([148], parent="kernel")
            wg_id = T.warpgroup_id([2], parent="cta")
            warp_id = T.warp_id([4], parent="warpgroup")
            lane_id = T.thread_id([32], parent="warp")
            with T.cta():
                T.block_attr({"tirp.scope_partition": True})
                with T.warpgroup()[1:2]:
                    T.block_attr({"tirp.scope_partition": True})
                    with T.warp(parent="warpgroup")[3:4]:
                        T.evaluate(warp_id)
                    with T.warp(parent="warpgroup")[2:3]:
                        T.evaluate(warp_id)
                    with T.warp(parent="warpgroup")[0:1]:
                        T.evaluate(warp_id)
                with T.warpgroup()[0:1]:
                    with T.thread():
                        T.evaluate(warp_id)
    # fmt: on

    # fmt: off
    @T.prim_func(private=True, tirp=True)
    def after():
        clusterCtaIdx_x = T.launch_thread("clusterCtaIdx.x", 2)
        clusterCtaIdx_y = T.launch_thread("clusterCtaIdx.y", 1)
        blockIdx_x = T.launch_thread("blockIdx.x", 148)
        threadIdx_x = T.launch_thread("threadIdx.x", 256)
        with T.kernel():
            with T.cta():
                if threadIdx_x // 128 >= 1 and threadIdx_x // 128 < 2:
                    with T.warpgroup():
                        if threadIdx_x % 128 // 32 >= 3 and threadIdx_x % 128 // 32 < 4:
                            with T.warp():
                                T.evaluate(threadIdx_x % 128 // 32)
                        else:
                            if threadIdx_x % 128 // 32 >= 2 and threadIdx_x % 128 // 32 < 3:
                                with T.warp():
                                    T.evaluate(threadIdx_x % 128 // 32)
                            else:
                                if threadIdx_x % 128 // 32 >= 0 and threadIdx_x % 128 // 32 < 1:
                                    with T.warp():
                                        T.evaluate(threadIdx_x % 128 // 32)
                else:
                    if threadIdx_x // 128 >= 0 and threadIdx_x // 128 < 1:
                        with T.warpgroup():
                            with T.thread():
                                T.evaluate(threadIdx_x % 128 // 32)
    # fmt: on

    compare(before, after, LowerTIRp)


def test_lower_layout():
    # fmt: off
    @T.prim_func(private=True, tirp=True)
    def before(A: T.Buffer((128, 32), "float16")) -> None:
        with T.kernel():
            bx, by, bz = T.cta_id([1, 1, 1], parent="kernel")
            warp_id = T.warp_id([4], parent="cta")
            lane_id = T.thread_id([32], parent="warp")
            tid = T.thread_id([128], parent="cta")

            with T.cta():
                A_smem = T.alloc_buffer([128, 32], dtype="float16", scope="shared", layout=T.SwizzleLayout(3, 3, 3))

                with T.thread():
                    thread_col = T.meta_var(4)
                    thread_row = T.meta_var(32)

                    for tile in T.serial(128 // thread_row):
                        row = T.meta_var(tile * thread_row + tid // thread_col)
                        col = T.meta_var(tid % thread_col * 8)
                        for vec in T.vectorized(8):
                            A_smem[row, col + vec] = A[bx * 128 + row, col + vec]

    @T.prim_func(private=True, tirp=True)
    def after(A: T.Buffer((128, 32), "float16", layout=None)) -> None:
        A_1 = T.decl_buffer((4096,), "float16", data=A.data, layout=None)
        blockIdx_x = T.launch_thread("blockIdx.x", 1)
        threadIdx_x = T.launch_thread("threadIdx.x", 128)
        blockIdx_y = T.launch_thread("blockIdx.y", 1)
        blockIdx_z = T.launch_thread("blockIdx.z", 1)
        with T.kernel():
            with T.cta():
                A_smem = T.alloc_buffer((4096,), "float16", scope="shared", layout=None)
                with T.thread():
                    for tile in range(4):
                        for vec in T.vectorized(8):
                            A_smem[T.shift_left(T.bitwise_xor(tile * 128 + threadIdx_x, T.shift_right(T.bitwise_and(tile * 128 + threadIdx_x, 56), 3)), 3) + vec] = A_1[tile * 1024 + threadIdx_x * 8 + vec]
    # fmt: on

    compare(before, after, LowerTIRp)


def test_lower_opcall_fail():
    # fmt: off
    @T.prim_func(tirp=True)
    def test(A_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, (64,), "float32", scope="global")

        with T.kernel():
            bx, by, bz = T.cta_id([1, 1, 1], parent="kernel")
            warp_id = T.warp_id([1], parent="cta")
            lane_id = T.thread_id([32], parent="warp")
            with T.cta():
                A_smem = T.alloc_buffer([64], dtype="float32", scope="shared")

                Tp.copy(A[0:64], A_smem[0:64])
                for i in range(10):
                    Tp.fill(A_smem[0:64], T.float32(0))
                    Tp.gemm(A_smem, A_smem, A_smem, A_smem)
                Tp.copy(A_smem[0:64], A[0:64])
    # fmt: on

    with pytest.raises(Exception):
        LowerTIRp()(tvm.IRModule({"main": test}))


def test_lower_decl_buffer_access_ptr():
    # fmt: off
    @T.prim_func(private=True, tirp=True)
    def before():
        with T.kernel():
            bx = T.cta_id([1], parent="kernel")
            tx = T.thread_id([128], parent="cta")
            with T.cta():
                buf = T.alloc_buffer([1024], "uint8", scope="shared.dyn")
                A = T.decl_buffer([128], "float16", buf.data, elem_offset=32)

                with T.thread():
                    T.evaluate(A.access_ptr("rw", offset=A.elem_offset_of([64])))

    @T.prim_func(private=True, tirp=True)
    def after():
        blockIdx_x = T.launch_thread("blockIdx.x", 1)
        threadIdx_x = T.launch_thread("threadIdx.x", 128)
        with T.kernel():
            with T.cta():
                buf = T.alloc_buffer((1024,), "uint8", scope="shared.dyn", layout=None)
                A = T.decl_buffer((128,), "float16", data=buf.data, elem_offset=32, scope="shared.dyn", layout=None)
                with T.thread():
                    T.tvm_access_ptr(T.type_annotation("float16"), buf.data, T.Add(32, 64), T.Sub(128, 64), 3)
    # fmt: on

    compare(before, after, LowerTIRp)


def test_lower_separate_scope_id_def():
    # fmt: off
    @T.prim_func(private=True, tirp=True)
    def before():
        with T.kernel():
            bx = T.cta_id([1], parent="kernel")
            with T.cta():
                tx = T.thread_id([128], parent="cta")
                with T.thread()[tx == 0]:
                    T.evaluate(tx)

    @T.prim_func(private=True, tirp=True)
    def after():
        blockIdx_x = T.launch_thread("blockIdx.x", 1)
        threadIdx_x = T.launch_thread("threadIdx.x", 128)
        with T.kernel():
            with T.cta():
                if threadIdx_x == 0:
                    with T.thread():
                        T.evaluate(threadIdx_x)
    # fmt: on

    compare(before, after, LowerTIRp)


def test_lower_buffer_offset():
    # fmt: off
    @T.prim_func(private=True, tirp=True)
    def before():
        with T.kernel():
            bx = T.cta_id([1], parent="kernel")
            with T.cta():
                tx = T.thread_id([128], parent="cta")
                with T.thread():
                    A = T.alloc_buffer([64, 64], "float16", scope="local")
                    A0 = T.decl_buffer([64], "float16", A.data, elem_offset=A.elem_offset_of([32, 32]))
                    with T.thread():
                        T.evaluate(T.address_of(A0[32]))
    # fmt: on

    # fmt: off
    @T.prim_func(private=True, tirp=True)
    def after():
        blockIdx_x = T.launch_thread("blockIdx.x", 1)
        threadIdx_x = T.launch_thread("threadIdx.x", 128)
        with T.kernel():
            with T.cta():
                with T.thread():
                    A = T.alloc_local((4096,), "float16", layout=None)
                    A0 = T.decl_buffer((64,), "float16", data=A.data, elem_offset=2080, scope="local", layout=None)
                    with T.thread():
                        T.address_of(A0[32])
    # fmt: on

    compare(before, after, LowerTIRp)


def test_lower_cell_buffer():
    # fmt: off
    @T.prim_func(private=True, tirp=True)
    def before():
        with T.kernel():
            with T.thread():
                A = T.local_cell("float16")
                event = Tp.alloc_semaphore_event_tensor(EventImpl.kTMALoadOnly, state=[A.buffer], shape=[1])
                T.evaluate(0)
    # fmt: on
    
    # fmt: off
    @T.prim_func(private=True, tirp=True)
    def after():
        with T.kernel():
            with T.thread():
                A = T.alloc_local((1,), "float16", layout=None)
                T.evaluate(0)
    # fmt: on
    compare(before, after, LowerTIRp)


def test_lower_alloc_decl_buffer_outside_of_parser():
    # fmt: off
    class State:
        def __init__(self, smem):
            self.A = T.alloc_local([1], "float16", name="A")
            self.B = T.alloc_local([1], "float16", name="B")
            self.C = T.decl_buffer([1], "float16", smem, elem_offset=0, scope="shared.dyn", name="C")

    def int_var1(val):
        buf = T.local_cell("int32")
        if val is not None:
            T.buffer_store(buf.buffer, val, 0)
        return buf
    
    def int_var2(val):
        frame = T.alloc_local([1], "int32")
        frame.add_callback(partial(frame.__exit__, None, None, None))
        buf = frame.__enter__()
        if val is not None:
            T.buffer_store(buf, val, 0)
        return buf

    # fmt: off
    @T.prim_func(private=True, tirp=True)
    def before():
        with T.kernel():
            with T.thread():
                smem = T.alloc_buffer([100], "uint8", scope="shared.dyn")
                state = T.meta_var(State(smem.data))
                state.A[0] = T.float16(1)
                state.B[0] = T.float16(2)
                state.C[0] = T.float16(3)
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
    @T.prim_func(private=True, tirp=True)
    def after():
        with T.kernel():
            with T.thread():
                smem = T.alloc_buffer([100], "uint8", scope="shared.dyn", layout=None)
                A = T.alloc_local((1,), "float16", layout=None)
                B = T.alloc_local((1,), "float16", layout=None)
                C = T.decl_buffer((1,), "float16", data=smem.data, elem_offset=0, scope="shared.dyn", layout=None)
                A[0] = T.float16(1)
                B[0] = T.float16(2)
                C[0] = T.float16(3)
                
                D = T.alloc_local((1,), "int32", layout=None)
                D = 1
                D = D[0] + 1
                
                E = T.alloc_local((1,), "int32", layout=None)
                E = 2
                E = E[0] + 2

                F = T.alloc_local((1,), "int32", layout=None)
                F = 3
                F = F[0] + 3

                G = T.alloc_local((1,), "int32", layout=None)
                G = 4
                G = G[0] + 4
    # fmt: on

    compare(before, after, LowerTIRp)


if __name__ == "__main__":
    tvm.testing.main()
