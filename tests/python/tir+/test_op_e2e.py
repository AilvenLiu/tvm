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

import tvm
import tvm.testing
from tvm import te
from tvm.script import tir as T
from tvm.tir.transform import LowerTIRp
from tvm.contrib import cublas


def test_gemm_ampere():
    # no pipeline, no write cache, fully manual impl
    M, N, K = 4096, 4096, 4096
    BLK_M, BLK_N, BLK_K = 128, 128, 32
    VEC = 8

    np.random.seed(0)

    A_np = np.random.randn(M, K).astype(np.float16)
    B_np = np.random.randn(N, K).astype(np.float16)
    DEV = tvm.cuda()
    A_tvm = tvm.nd.array(A_np, device=DEV)
    B_tvm = tvm.nd.array(B_np, device=DEV)

    target = tvm.target.Target("nvidia/geforce-rtx-4090")
    # fmt: off
    @T.prim_func(tirp=True)
    def func(A_ptr: T.handle, B_ptr: T.handle, C_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, (M, K), "float16", scope="global")
        B = T.match_buffer(B_ptr, (N, K), "float16", scope="global")
        C = T.match_buffer(C_ptr, (M, N), "float32", scope="global")

        with T.kernel():
            bx, by = T.cta_id([M // BLK_M, N // BLK_N], parent="kernel")
            warp_id = T.warp_id([4], parent="cta")
            lane_id = T.thread_id([32], parent="warp")
            tid = T.thread_id([128], parent="cta")

            with T.cta():
                mma_layout = T.TileLayout.from_nested_tuple(data=(1, 2), strides=(2, 1))
                mma_layout_atom = T.TileLayout.shard(
                    shape=(8, 8),
                    mesh=(8, 4),
                    strategy="S0S1",
                    inner=mma_layout,
                    from_to=("thread", "warp"),
                )
                acc_layout_cta_atom = T.TileLayout.shard(
                    shape=(16, 16),
                    mesh=(2, 2),
                    strategy="S0S1",
                    inner=mma_layout_atom,
                    from_to=("warp", "cta"),
                )
                tiling = T.TileLayout.from_nested_tuple(
                    data=(BLK_M // 16, BLK_N // 16), strides=(BLK_N // 16, 1)
                )
                acc_layout_cta = T.TileLayout.tile(tiling, acc_layout_cta_atom)

                acc_storage = T.alloc_buffer([BLK_N // 16, BLK_M // 16, 2], dtype="float32", scope="local")
                A_smem = T.alloc_buffer([BLK_M, BLK_K], dtype="float16", scope="shared", layout=None) # TODO: swizzle layout
                B_smem = T.alloc_buffer([BLK_N, BLK_K], dtype="float16", scope="shared", layout=None) # TODO: swizzle layout

                acc = T.view(acc_storage, layout=acc_layout_cta)

                # T.fill(acc, 0)
                with T.thread():
                    T.reads()
                    T.writes(acc[:, :])
                    acc_local = T.get(acc)
                    for i, j in T.grid(BLK_N // 16, BLK_N // 16):
                        for vec in T.vectorized(2):
                            acc_local[i, j, vec] = 0

                for k in T.serial(T.ceildiv(K, BLK_K)):
                    T.tvm_storage_sync("shared")
                    T.static_assert(M % BLK_M == 0, f"M={M}, BLK_M={BLK_M}")
                    T.static_assert(N % BLK_N == 0, f"N={N}, BLK_N={BLK_N}")
                    T.static_assert(K % BLK_K == 0, f"K={K}, BLK_K={BLK_K}")
                    # T.load(A_smem, A[bx * BLK_M: bx * BLK_M + BLK_M, k * BLK_K: k * BLK_K + BLK_K])
                    with T.thread():
                        T.reads(A[:, :])
                        T.writes(A_smem[:, :])
                        T.static_assert(BLK_K % VEC == 0, "BLK_K should be multiple of VEC")
                        T.static_assert(128 % (BLK_K // VEC) == 0, "128 should be multiple of BLK_K // VEC")
                        T.static_assert(BLK_M % (128 // (BLK_K // VEC)) == 0, "BLK_M should be multiple of 128 // (BLK_K // VEC)")
                        thread_per_col = T.meta_var(BLK_K // VEC)
                        thread_per_row = T.meta_var(128 // thread_per_col)

                        for tile in T.serial(BLK_M // thread_per_row):
                            row = T.meta_var(tile * thread_per_row + tid // thread_per_col)
                            col = T.meta_var(tid % thread_per_col * VEC)
                            for vec in T.vectorized(VEC):
                                A_smem[row, col + vec] = A[bx * BLK_M + row, k * BLK_K + col + vec]
                    T.tvm_storage_sync("shared")

                    # T.load(B_smem, B[k * BLK_K: k * BLK_K + BLK_K, by * BLK_N: by * BLK_N + BLK_N])
                    with T.thread():
                        T.reads(B[:, :])
                        T.writes(B_smem[:, :])
                        T.static_assert(BLK_K % VEC == 0, "BLK_K should be multiple of VEC")
                        T.static_assert(128 % (BLK_K // VEC) == 0, "128 should be multiple of BLK_K // VEC")
                        T.static_assert(BLK_N % (128 // (BLK_K // VEC)) == 0, "BLK_N should be multiple of 128 // (BLK_K // VEC)")
                        thread_per_col = T.meta_var(BLK_K // VEC)
                        thread_per_row = T.meta_var(128 // thread_per_col)

                        for tile in T.serial(BLK_N // thread_per_row):
                            row = T.meta_var(tile * thread_per_row + tid // thread_per_col)
                            col = T.meta_var(tid % thread_per_col * VEC)
                            for vec in T.vectorized(VEC):
                                B_smem[row, col + vec] = B[by * BLK_N + row, k * BLK_K + col + vec]
                    T.tvm_storage_sync("shared")

                    # acc += A_smem @ B_smem
                    with T.warp():
                        T.reads(A_smem[:, :], B_smem[:, :])
                        T.writes(acc[:, :])
                        A_storage = T.alloc_buffer([BLK_M // 32, 4, 2], dtype="float16", scope="local")
                        B_storage = T.alloc_buffer([BLK_N // 16, 2, 2], dtype="float16", scope="local")
                        
                        tiling_AB = T.TileLayout.from_nested_tuple(data=(BLK_M // 16, 2), strides=(2, 1))
                        AB_layout = T.TileLayout.tile(tiling_AB, mma_layout_atom)
                        A_warp = T.view(A_storage, layout=AB_layout)
                        B_warp = T.view(B_storage, layout=AB_layout)

                        acc_warp_layout = T.TileLayout.tile(tiling, mma_layout_atom)
                        acc_warp = T.view(acc_storage, layout=acc_warp_layout)

                        for k_inner in T.serial(BLK_K // 16):
                            st_m = T.meta_var((warp_id // 2) * (BLK_M // 2))
                            # T.load(A_warp, A_smem[st_m: st_m + BLK_M // 2, k_inner * 16: k_inner * 16 + 16])
                            with T.warp():
                                T.reads(A_smem[:, :])
                                T.writes(A_warp[:, :])
                                T.static_assert(BLK_M % 32 == 0, "BLK_M should be multiple of 32")
                                for i in T.serial(BLK_M // 32):
                                    # 16x16 ptx ldmatrix
                                    # T.load(A_warp[i * 16 : i * 16 + 16, :], 
                                    #        A_smem[st_m + i * 16 : st_m + i * 16 + 16, k_inner * 16 : k_inner * 16 + 16])
                                    with T.thread():
                                        A_local = T.get(A_warp)
                                        T.ptx_ldmatrix(
                                            "float16", False, 4, ".b16",
                                            # TODO: change the signature of ptx_ldmatrix / introduce an op to get data from buffer
                                            A_storage.data, A_local.offset_of([i, 0, 0])[0],
                                            A_smem.data, A_smem.offset_of([st_m + i * 16 + lane_id % 16, 
                                                                           k_inner * 16 + lane_id // 16 * 8])[0]
                                        )

                            st_n = T.meta_var((warp_id % 2) * (BLK_N // 2))
                            # T.load(B_warp, B_smem[st_n: st_n + BLK_N // 2, k_inner * 16: k_inner * 16 + 16]) 
                            with T.warp():
                                T.reads(B_smem[:, :])
                                T.writes(B_warp[:, :])
                                T.static_assert(BLK_N % 32 == 0, "BLK_N should be multiple of 32")
                                for i in T.serial(BLK_N // 32):
                                    # 16x16 ptx ldmatrix
                                    # T.load(B_warp[i * 16 : i * 16 + 16, :],
                                    #        B_smem[st_n + i * 16 : st_n + i * 16 + 16, k_inner * 16 : k_inner * 16 + 16])
                                    with T.thread():
                                        B_local = T.get(B_warp)
                                        T.ptx_ldmatrix(
                                            "float16", False, 4, ".b16",
                                            # TODO: change the signature of ptx_ldmatrix / introduce an op to get data from buffer
                                            B_storage.data, B_local.offset_of([i * 2, 0, 0])[0],
                                            B_smem.data, B_smem.offset_of([st_n + i * 16 + lane_id // 16 * 8 + lane_id % 8,
                                                                           k_inner * 16 + lane_id % 16 // 8 * 8])[0]
                                        )

                            # acc_warp += A_warp @ B_warp
                            with T.warp():
                                T.reads(A_warp[:, :], B_warp[:, :])
                                T.writes(acc_warp[:, :])
                                for tile_m in range(BLK_M // 32):
                                    for tile_n in range(BLK_N // 32):
                                        # 16x16x16 ptx mma
                                        # acc_warp[tile_m * 16 : tile_m * 16 + 16, tile_n * 16 : tile_n * 16 + 16] +=
                                        #     A_warp[tile_m * 16 : tile_m * 16 + 16, :] @ B_warp[tile_n * 16 : tile_n * 16 + 16, :]    
                                        with T.thread():
                                            A_local = T.get(A_warp)
                                            B_local = T.get(B_warp)
                                            acc_local = T.get(acc_warp)
                                            T.ptx_mma(
                                                "m16n8k16", "row", "col",
                                                "fp16", "fp16", "fp32",
                                                # TODO: change the signature of ptx_ldmatrix / introduce an op to get data from buffer
                                                A_storage.data, A_local.offset_of([tile_m, 0, 0])[0],
                                                B_storage.data, B_local.offset_of([tile_n * 2, 0, 0])[0],
                                                acc_storage.data, acc_local.offset_of([tile_n * 2, tile_m * 2, 0])[0],
                                                False,
                                                dtype="int32"
                                            )
                                            T.ptx_mma(
                                                "m16n8k16", "row", "col",
                                                "fp16", "fp16", "fp32",
                                                # TODO: change the signature of ptx_ldmatrix / introduce an op to get data from buffer
                                                A_storage.data, A_local.offset_of([tile_m, 0, 0])[0],
                                                B_storage.data, B_local.offset_of([tile_n * 2 + 1, 0, 0])[0],
                                                acc_storage.data, acc_local.offset_of([tile_n * 2 + 1, tile_m * 2, 0])[0],
                                                False,
                                                dtype="int32"
                                            )

                # T.store(C[bx * BLK_M: bx * BLK_M + BLK_M, by * BLK_N: by * BLK_N + BLK_N], acc)
                with T.thread():
                    T.reads(acc[:, :])
                    T.writes(C[:, :])
                    acc_local = T.get(acc)
                    # TODO: Add write cache (Ampere)
                    for j, i in T.grid(BLK_N // 16, BLK_M // 16):
                        st_m = T.meta_var(bx * BLK_M + warp_id // 2 * BLK_M // 2 + i * 8 + lane_id // 4)
                        st_n = T.meta_var(by * BLK_N + warp_id % 2 * BLK_N // 2 + j * 8 + lane_id % 4 * 2)
                        for vec in T.serial(2):
                            C[st_m, st_n + vec] = acc_local[j, i, vec]

    # fmt: on

    def tvm_gemm():
        with tvm.transform.PassContext(config={"tir.disable_storage_rewrite": True}):
            mod = tvm.IRModule({"main": func})
            mod = LowerTIRp()(mod)
            mod = tvm.build(mod, target=target)
            C_np = np.zeros((M, N), dtype=np.float32)
            C_tvm = tvm.nd.array(C_np, device=DEV)
            timer = mod.time_evaluator(mod.entry_name, DEV, number=10, repeat=3)
            res = timer(A_tvm, B_tvm, C_tvm)
            print(res)
        return C_tvm

    # cublas
    def cublas_gemm():
        A = te.placeholder((M, K), name="A", dtype="float16")
        B = te.placeholder((N, K), name="B", dtype="float16")
        C = cublas.matmul(A, B, transb=True, dtype="float32")
        s = te.create_schedule(C.op)

        C_np = np.zeros((M, N), dtype=np.float32)
        C_tvm = tvm.nd.array(C_np, device=DEV)
        mod_cublaslt = tvm.build(s, [A, B, C], target)
        mod_cublaslt(A_tvm, B_tvm, C_tvm)
        timer = mod_cublaslt.time_evaluator(mod_cublaslt.entry_name, DEV, number=10, repeat=3)
        res = timer(A_tvm, B_tvm, C_tvm)
        print(res)

        return C_tvm

    C_tvm = tvm_gemm()
    C_cublas = cublas_gemm()

    tvm.testing.assert_allclose(C_tvm.asnumpy(), C_cublas.asnumpy(), rtol=1e-3, atol=1e-3)
    print("test passed")


if __name__ == "__main__":
    test_gemm_ampere()
