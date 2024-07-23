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
import tvm
import numpy as np
import tvm.testing
from tvm.script import ir as I
from tvm.script import tir as T
from tvm.script import tirp as Tp


def test_barrier_cta():
    CTA_COUNT = 8

    # fmt: off
    @T.prim_func(tirp=True)
    def test(A_ptr: T.handle, B_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, (CTA_COUNT * 64,), "float32", scope="global")
        B = T.match_buffer(B_ptr, (CTA_COUNT * 64,), "float32", scope="global")

        with T.kernel():
            bx, by, bz = T.cta_id([CTA_COUNT, 1, 1], parent="kernel")
            tid = T.thread_id([128], parent="cta")

            with T.cta():
                A_smem = T.alloc_buffer([64], dtype="float32", scope="shared")
                b = Tp.alloc_barrier(thread_scope="cta", name_hint="b")

                with T.thread():
                    b.init(128)

                    if (tid < 64):
                        # producer
                        A_smem[tid] = A[bx * 64 + tid]
                        b.arrive()
                        b.wait()
                    else:
                        # consumer
                        b.arrive_and_wait()
                        B[bx * 64 + tid - 64] = A_smem[tid - 64] + 1

    @I.ir_module
    class Module:
        @T.prim_func(tirp=True)
        def main(A: T.Buffer((512,), "float32"), B: T.Buffer((512,), "float32")):
            T.func_attr({"global_symbol": "test"})
            with T.kernel():
                for blockIdx in T.thread_binding(8, thread="blockIdx.x"):
                    for threadIdx in T.thread_binding(128, thread="threadIdx.x"):
                        with T.cta():
                            A_smem = T.alloc_buffer((64,), scope="shared", logical_scope="cta")
                            b = Tp.alloc_barrier_array("cta", 1, "b")
                            T.cuda_barrier_create("cta", 0, 1)
                            with T.thread():
                                T.cuda_barrier_init(128, 0, 0)
                                if threadIdx % 128 < 64:
                                    A_1 = T.Buffer((512,), data=A.data, logical_scope="kernel")
                                    A_smem[threadIdx] = A_1[blockIdx * 64 + threadIdx]
                                    T.cuda_barrier_arrive(0, 0)
                                    T.cuda_barrier_wait(0, 0)
                                else:
                                    T.cuda_barrier_arrive_and_wait(0, 0)
                                    B_1 = T.Buffer((512,), data=B.data, logical_scope="kernel")
                                    B_1[blockIdx * 64 + threadIdx + -64] = A_smem[threadIdx + -64] + T.float32(1)
    # fmt: on

    target = tvm.target.Target("cuda")
    DEV = tvm.cuda(0)

    with target:
        mod = tvm.IRModule({"main": test})
        mod = tvm.tir.transform.LowerTIRp()(mod)
        tvm.ir.assert_structural_equal(mod, Module)
        mod = tvm.build(mod, target=target)

        np.random.seed(0)
        A_np = np.random.randn(CTA_COUNT * 64).astype("float32")
        B_np = np.zeros(CTA_COUNT * 64, dtype="float32")

        A = tvm.nd.array(A_np, device=DEV)
        B = tvm.nd.array(B_np, device=DEV)
        mod(A, B)
        tvm.testing.assert_allclose(B.asnumpy(), A.asnumpy() + 1)


def test_pipeline_no_specialize_cta():
    # fmt: off
    N = 32 * 32
    M = 128 * 32
    N_STAGES = 1

    @T.prim_func(tirp=True)
    def test(A_ptr: T.handle, B_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, (N, M), "float32", scope="global", 
                           layout=T.TileLayout.from_nested_tuple((1024, 4096)))
        B = T.match_buffer(B_ptr, (N, 128), "float32", scope="global",
                           layout=T.TileLayout.from_nested_tuple((1024, 128)))

        with T.kernel():
            bx = T.cta_id([N // 32], parent="kernel")
            tx = T.thread_id([128], parent="cta")

            with T.cta():
                A_smem = T.alloc_buffer([N_STAGES, 32, 128], dtype="float32", scope="shared", 
                                        layout=T.TileLayout.from_nested_tuple((N_STAGES, 32, 128)))
                O_smem = T.alloc_buffer([32, 128], dtype="float32", scope="shared", 
                                        layout=T.TileLayout.from_nested_tuple((32, 128)))

                pipe = Tp.alloc_pipeline(thread_scope="cta", depth=0, specialize=False)

                # T.fill(O_smem, 0.0)
                with T.thread():
                    for k in range(32):
                        O_smem[k, tx] = 0.0
                    T.tvm_storage_sync("shared")

                for i in range(N_STAGES - 1):
                    pipe.producer_copy_async(A_smem[i, :, :], A[bx * 32 : (bx + 1) * 32, i * 128 : (i + 1) * 128])
                    pipe.producer_commit_stage()

                for j in range(0, M // 128 - N_STAGES + 1):
                    i = T.meta_var(j + N_STAGES - 1)
                    pipe.producer_copy_async(A_smem[i % N_STAGES, :, :], A[bx * 32 : (bx + 1) * 32, i * 128 : (i + 1) * 128])
                    pipe.producer_commit_stage()

                    pipe.consumer_wait(num_stages=N_STAGES - 1)
                    with T.thread():
                        for k in range(32):
                            O_smem[k, tx] += A_smem[j % N_STAGES, k, tx]
                        T.tvm_storage_sync("shared")

                # pipe.consumer_wait(num_stages=0)
                for j in range(N_STAGES - 1):
                    i = T.meta_var(j + M // 128 - N_STAGES + 1)
                    with T.thread():
                        for k in range(32):
                            O_smem[k, tx] += A_smem[i % N_STAGES, k, tx]
                    T.tvm_storage_sync("shared")

                Tp.copy(B[bx * 32 : (bx + 1) * 32, 0:128], O_smem)

    @I.ir_module
    class Module:
        @T.prim_func(tirp=True)
        def main(A_ptr: T.handle, B_ptr: T.handle):
            T.func_attr({"global_symbol": "test"})
            A = T.match_buffer(A_ptr, (1024, 4096), logical_scope="kernel", layout=T.TileLayout.from_nested_tuple(data=(1024, 4096), strides=(4096, 1)))
            B = T.match_buffer(B_ptr, (1024, 128), logical_scope="kernel", layout=T.TileLayout.from_nested_tuple(data=(1024, 128), strides=(128, 1)))
            with T.kernel():
                for blockIdx in T.thread_binding(32, thread="blockIdx.x"):
                    for threadIdx in T.thread_binding(128, thread="threadIdx.x"):
                        with T.cta():
                            A_smem = T.alloc_buffer((4096,), scope="shared", logical_scope="cta")
                            O_smem = T.alloc_buffer((4096,), scope="shared", logical_scope="cta")
                            pipeline = Tp.alloc_pipeline("cta", 0, False, "")
                            with T.thread():
                                for k in range(32):
                                    O_smem[k * 128 + threadIdx] = T.float32(0)
                                T.tvm_storage_sync("shared")
                            for i in range(0):
                                with T.thread():
                                    for s in range(8):
                                        T.ptx_cp_async("void", A_smem.data, i * 4096 + s * 512 + threadIdx * 4, A.data, (blockIdx * 131072 + s * 16384 + threadIdx // 32 * 4096 + i * 128 + threadIdx % 32 * 4) // 4194304 * 4194304 + blockIdx * 131072 + s * 16384 + threadIdx // 32 * 4096 + i * 128 + threadIdx % 32 * 4, 16)
                                    T.ptx_commit_group()
                            for j in range(32):
                                with T.thread():
                                    for s in range(8):
                                        T.ptx_cp_async("void", A_smem.data, s * 512 + threadIdx * 4, A.data, blockIdx * 131072 + s * 16384 + threadIdx // 32 * 4096 + j * 128 + threadIdx % 32 * 4, 16)
                                    T.ptx_commit_group()
                                    T.ptx_wait_group(0)
                                    T.tvm_storage_sync("shared")
                                    for k in range(32):
                                        O_smem[k * 128 + threadIdx] = O_smem[k * 128 + threadIdx] + A_smem[k * 128 + threadIdx]
                                    T.tvm_storage_sync("shared")
                            for j in range(0):
                                with T.thread():
                                    for k in range(32):
                                        O_smem[k * 128 + threadIdx] = O_smem[k * 128 + threadIdx] + A_smem[k * 128 + threadIdx]
                                T.tvm_storage_sync("shared")
                            with T.thread():
                                for s in range(8):
                                    for vec in T.vectorized(4):
                                        B_1 = T.Buffer((131072,), data=B.data, logical_scope="kernel")
                                        B_1[blockIdx * 4096 + s * 512 + threadIdx * 4 + vec] = O_smem[s * 512 + threadIdx * 4 + vec]
    # fmt: on

    target = tvm.target.Target("nvidia/geforce-rtx-4090")
    DEV = tvm.cuda(0)

    with target:
        mod = tvm.IRModule({"main": test})
        mod = tvm.tir.transform.LowerTIRp()(mod)
        tvm.ir.assert_structural_equal(mod, Module)
        mod = tvm.build(mod, target=target)

        np.random.seed(0)
        A_np = np.ones((N, M)).astype("float32") * 10
        B_np = np.zeros((N, 128), dtype="float32")

        A = tvm.nd.array(A_np, device=DEV)
        B = tvm.nd.array(B_np, device=DEV)
        mod(A, B)

        B_np_ref = np.sum(A_np.reshape(N, 128, M // 128), axis=2)
        tvm.testing.assert_allclose(B.asnumpy(), B_np_ref)


if __name__ == "__main__":
    test_barrier_cta()
    test_pipeline_no_specialize_cta()
