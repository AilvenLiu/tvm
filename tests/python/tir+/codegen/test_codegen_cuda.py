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
# pylint: disable=missing-function-docstring
import numpy as np

import tvm
from tvm.script import tir as T
import tvm.testing


DEV = tvm.device("cuda")


def _get_source(func: tvm.tir.PrimFunc) -> str:
    target = tvm.target.Target("cuda")
    mod = tvm.IRModule({"main": func})
    mod = tvm.compile(mod, target=target, tir_pipeline="tirp")
    src = mod.mod.imported_modules[0].get_source()
    return src, mod


def test_cuda_atomic_add():
    @T.prim_func(tirp=True)
    def main(A: T.Buffer((1,), "int32"), B: T.Buffer((1,), "float32")):
        with T.kernel():
            bx = T.cta_id([1], parent="kernel")
            tx = T.thread_id([32], parent="cta")
            with T.thread()[tx == 0]:
                T.cuda.atomic_add(A.data, T.int32(1))
                T.cuda.atomic_add(B.data, T.float32(1.0))

    src, mod = _get_source(main)
    assert "tvm_builtin_cuda_atomic_add" in src
    A_np = np.zeros(1, dtype="int32")
    B_np = np.zeros(1, dtype="float32")
    A_tvm = tvm.nd.array(A_np, device=DEV)
    B_tvm = tvm.nd.array(B_np, device=DEV)
    mod["main"](A_tvm, B_tvm)
    np.testing.assert_allclose(A_tvm.numpy(), 1)
    np.testing.assert_allclose(B_tvm.numpy(), 1.0)


def test_cuda_thread_fence():
    @T.prim_func(tirp=True)
    def main(A: T.Buffer((16, 16), "int32")):
        with T.kernel():
            bx = T.cta_id([1], parent="kernel")
            tx = T.thread_id([32], parent="cta")
            with T.thread()[tx == 0]:
                T.cuda.thread_fence()

    src, mod = _get_source(main)
    assert "tvm_builtin_cuda_thread_fence" in src


def test_cuda_nano_sleep():
    @T.prim_func(tirp=True)
    def main(A: T.Buffer((16, 16), "int32")):
        with T.kernel():
            bx = T.cta_id([1], parent="kernel")
            tx = T.thread_id([32], parent="cta")
            with T.thread()[tx == 0]:
                T.cuda.nano_sleep(1)

    src, mod = _get_source(main)
    assert "tvm_builtin_cuda_nano_sleep" in src


def test_cuda_func_call():
    def test_add_one():
        add_one = """
__device__ int32_t add_one(int32_t a) {
    return a + 1;
}
"""

        @T.prim_func(tirp=True)
        def main(a: T.Buffer((16, 16), "int32"), b: T.Buffer((16, 16), "int32")):
            with T.kernel():
                bx = T.cta_id([1], parent="kernel")
                tx = T.thread_id([32], parent="cta")
                with T.thread()[tx == 0]:
                    for i, j in T.grid(16, 16):
                        b[i, j] = T.cuda.func_call(
                            "add_one",
                            a[i, j],
                            source_code=add_one,
                            return_type="int32",
                        )

        src, mod = _get_source(main)
        A = np.random.randint(0, 10, (16, 16)).astype("int32")
        B = np.zeros((16, 16), dtype="int32")
        A_tvm = tvm.nd.array(A, device=DEV)
        B_tvm = tvm.nd.array(B, device=DEV)
        mod["main"](A_tvm, B_tvm)
        np.testing.assert_allclose(B_tvm.numpy(), A + 1)
        print(src)

    test_add_one()

    def test_print():
        print_func = """
__device__ void print(int32_t a) {
    printf("%d\\n", a);
}
"""

        @T.prim_func(tirp=True)
        def main(a: T.Buffer((16, 16), "int32")):
            with T.kernel():
                bx = T.cta_id([1], parent="kernel")
                tx = T.thread_id([32], parent="cta")
                with T.thread()[tx == 0]:
                    for i, j in T.grid(16, 16):
                        T.cuda.func_call("print", a[i, j], source_code=print_func)

        src, mod = _get_source(main)
        A = np.random.randint(0, 10, (16, 16)).astype("int32")
        A_tvm = tvm.nd.array(A, device=DEV)
        mod["main"](A_tvm)
        print(src)

    test_print()


def test_warp_shuffle_xor_sync():
    # fmt: off
    @T.prim_func(tirp=True)
    def func(A_ptr: T.handle):
        A = T.match_buffer(A_ptr, (32,), dtype="float32", align=16)

        with T.kernel():
            bx = T.cta_id([1], parent="kernel")
            warp_id = T.warp_id([1], parent="cta")
            lane_id = T.thread_id([32], parent="warp")

            with T.thread():
                A_local = T.alloc_buffer([1], "float32", scope="local")
                i = T.alloc_buffer([1], "int32", scope="local")

                A_local[0] = T.float32(31 - lane_id)
                i[0] = 16
                while i[0] >= 1:
                    A_local[0] += T.tvm_warp_shuffle_xor(0xFFFFFFFF, A_local[0], i[0], 32, 32)
                    i[0] = i[0] // 2

                A[lane_id] = A_local[0]
    # fmt: on

    DEV = tvm.cuda(0)
    target = tvm.target.Target("cuda")
    mod = tvm.IRModule({"main": func})
    mod = tvm.compile(mod, target=target, tir_pipeline="tirp")
    A_np = np.zeros(32, dtype="float32")
    A = tvm.nd.array(A_np, device=DEV)
    mod(A)
    assert "__shfl_xor_sync" in mod.mod.imported_modules[0].get_source()
    A_ref = np.ones(32, dtype="float32") * 496
    np.testing.assert_allclose(A.numpy(), A_ref)


if __name__ == "__main__":
    tvm.testing.main()
