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
                b = Tp.alloc_barrier(exec_scope="cta", name_hint="b")

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
    # fmt: on

    target = tvm.target.Target("cuda")
    DEV = tvm.cuda(0)

    with target:
        mod = tvm.IRModule({"main": test})
        mod = tvm.tir.transform.LowerTIRp()(mod)
        mod = tvm.build(mod, target=target)
        print(mod.imported_modules[0].get_source())

        np.random.seed(0)
        A_np = np.random.randn(CTA_COUNT * 64).astype("float32")
        B_np = np.zeros(CTA_COUNT * 64, dtype="float32")

        A = tvm.nd.array(A_np, device=DEV)
        B = tvm.nd.array(B_np, device=DEV)
        mod(A, B)
        tvm.testing.assert_allclose(B.asnumpy(), A.asnumpy() + 1)


if __name__ == "__main__":
    test_barrier_cta()
