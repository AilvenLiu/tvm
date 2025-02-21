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


def test_pipeline_no_specialize_cta():
    N = 32 * 32
    M = 128 * 32
    N_STAGES = 3

    # fmt: off
    @T.prim_func(tirp=True)
    def test(A_ptr: T.handle, B_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, (N, M), "float32", scope="global", 
                           layout=T.TileLayout.from_tuple((1024, 4096)))
        B = T.match_buffer(B_ptr, (N, 128), "float32", scope="global",
                           layout=T.TileLayout.from_tuple((1024, 128)))

        with T.kernel():
            bx = T.cta_id([N // 32], parent="kernel")
            tx = T.thread_id([128], parent="cta")

            with T.cta():
                A_smem = T.alloc_buffer([N_STAGES, 32, 128], dtype="float32", scope="shared.dyn", 
                                        layout=T.TileLayout.from_tuple((N_STAGES, 32, 128)))
                O_smem = T.alloc_buffer([32, 128], dtype="float32", scope="shared.dyn", 
                                        layout=T.TileLayout.from_tuple((32, 128)))

                pipe = Tp.alloc_copy_pipeline(thread_scope="cta", depth=0, separate_pc=False)

                with T.thread():
                    for k in range(32):
                        O_smem[k, tx] = 0.0
                    T.tvm_storage_sync("shared")

                for i in range(N_STAGES - 1):
                    pipe.copy(A_smem[i, :, :], A[bx * 32 : (bx + 1) * 32, i * 128 : (i + 1) * 128])
                    pipe.producer_commit()

                for j in range(0, M // 128 - N_STAGES + 1):
                    i = T.meta_var(j + N_STAGES - 1)
                    pipe.copy(A_smem[i % N_STAGES, :, :], A[bx * 32 : (bx + 1) * 32, i * 128 : (i + 1) * 128])
                    pipe.producer_commit()

                    pipe.consumer_wait(num_stages=N_STAGES - 1)
                    with T.thread():
                        T.tvm_storage_sync("shared")
                        for k in range(32):
                            O_smem[k, tx] += A_smem[j % N_STAGES, k, tx]
                        T.tvm_storage_sync("shared")

                pipe.consumer_wait(num_stages=0)
                for j in range(N_STAGES - 1):
                    i = T.meta_var(j + M // 128 - N_STAGES + 1)
                    with T.thread():
                        for k in range(32):
                            O_smem[k, tx] += A_smem[i % N_STAGES, k, tx]
                        T.tvm_storage_sync("shared")

                Tp.copy(B[bx * 32 : (bx + 1) * 32, 0:128], O_smem)
    # fmt: on

    DEV = tvm.cuda(0)
    target = tvm.target.Target.from_device(DEV)

    with target:
        mod = tvm.IRModule({"main": test})
        mod = tvm.tir.transform.LowerTIRp()(mod)
        mod.show()
        mod = tvm.build(mod, target=target, pipeline="tirp")

        np.random.seed(0)
        A_np = np.ones((N, M)).astype("float32") * 10
        B_np = np.zeros((N, 128), dtype="float32")

        A = tvm.nd.array(A_np, device=DEV)
        B = tvm.nd.array(B_np, device=DEV)
        mod(A, B)

        B_np_ref = np.sum(A_np.reshape(N, 128, M // 128), axis=2)
        tvm.testing.assert_allclose(B.asnumpy(), B_np_ref)


if __name__ == "__main__":
    # test_barrier_cta()
    test_pipeline_no_specialize_cta()
