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
import pytest
import torch

import tvm
import tvm.testing
from tvm.script import tir as T
from tvm.script import tirp as Tp

DEV = tvm.device("cuda")


def _get_source(func: tvm.tir.PrimFunc) -> str:
    target = tvm.target.Target("cuda")
    mod = tvm.IRModule({"main": func})
    mod = tvm.compile(mod, target=target, tir_pipeline="tirp")
    src = mod.mod.imports[0].inspect_source()
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
    A_tvm = tvm.runtime.tensor(A_np, device=DEV)
    B_tvm = tvm.runtime.tensor(B_np, device=DEV)
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


def test_cuda_atomic_cas():
    @T.prim_func(tirp=True)
    def main(A: T.Buffer((16, 16), "int32")):
        with T.kernel():
            bx = T.cta_id([1], parent="kernel")
            tx = T.thread_id([32], parent="cta")
            with T.thread()[tx == 0]:
                T.cuda.atomic_cas(A.data, T.int32(1), T.int32(2))

    src, mod = _get_source(main)
    assert "tvm_builtin_cuda_atomic_cas" in src


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
        A_tvm = tvm.runtime.tensor(A, device=DEV)
        B_tvm = tvm.runtime.tensor(B, device=DEV)
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
        A_tvm = tvm.runtime.tensor(A, device=DEV)
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
    A = tvm.runtime.tensor(A_np, device=DEV)
    mod(A)
    assert "__shfl_xor_sync" in mod.mod.imports[0].inspect_source()
    A_ref = np.ones(32, dtype="float32") * 496
    np.testing.assert_allclose(A.numpy(), A_ref)


@pytest.mark.parametrize("cp_size", [4, 8, 16])
@pytest.mark.parametrize("cache_hint", ["", "evict_last"])
@pytest.mark.parametrize("prefetch_size", [-1, 64, 128, 256])
@pytest.mark.parametrize("predicate", [-1, T.int32(0), T.int32(1)])
@pytest.mark.parametrize("fill_mode", ["", "zero"])
def test_ptx_cp_async(cp_size, cache_hint, prefetch_size, predicate, fill_mode):
    if fill_mode != "" and predicate == -1:
        return

    N = cp_size // 2
    # fmt: off
    @T.prim_func(tirp=True)
    def main(A: T.Buffer((N), "float16")):
        with T.kernel():
            bx = T.cta_id([1], parent="kernel")
            tx = T.thread_id([1], parent="cta")
            with T.thread():
                A_shared = T.alloc_shared([N], "float16")
                for i in T.vectorized(N):
                    A_shared[i] = 5.0
                T.ptx.fence.proxy("shared")
                T.ptx.cp_async(A_shared.ptr_to([0]), A.ptr_to([0]), cp_size, cache_hint, prefetch_size, predicate, fill_mode)
                T.ptx.cp_async.commit_group()
                T.ptx.cp_async.wait_group(0)
                for i in T.serial(N):
                    A[i] = A_shared[i] + 1.0
    # fmt: on

    src, mod = _get_source(main)
    A_np = np.ones(N, dtype="float16")
    A = tvm.runtime.tensor(A_np, device=DEV)
    mod(A)
    A_ref = np.ones(N, dtype="float16") * 2
    if int(predicate) == 0:
        if fill_mode == "zero":
            A_ref = np.ones(N, dtype="float16")
        else:
            A_ref = np.ones(N, dtype="float16") * 6

    np.testing.assert_allclose(A.numpy(), A_ref)
    print(src)


@pytest.mark.parametrize("trans", [False, True])
@pytest.mark.parametrize("num", [1, 2, 4])
def test_ptx_ldmatrix(trans, num):
    dtype = ".b16"

    # fmt: off
    @T.prim_func(tirp=True)
    def main(A: T.Buffer((16, 16), "float16"), B: T.Buffer((16, 16), "float16")):
        with T.kernel():
            bx = T.cta_id([1], parent="kernel")
            tx = T.thread_id([32], parent="cta")
            A_shared = T.alloc_shared([16, 16], "float16")
            with T.thread()[tx == 0]:
                for i, j in T.grid(16, 16):
                    A_shared[i, j] = A[i, j]
            T.tvm_storage_sync("shared")
            with T.thread():
                A_local = T.alloc_local([8], "float16")
                A_local[0] = -1.0
                T.ptx.ldmatrix(
                    trans, num, dtype, A_local.ptr_to([0]), A_shared.ptr_to([tx % 16, tx // 16 * 8])
                )
                for i in range(8):
                    row = (i // 2) % 2 * 8
                    col = (i // 4) * 8
                    B[row + tx // 4, col + tx % 4 * 2 + i % 2] = A_local[i]
    # fmt: on

    src, mod = _get_source(main)
    A_np = np.arange(16 * 16, dtype="float16").reshape((16, 16))
    A = tvm.runtime.tensor(A_np, device=DEV)
    B_np = np.zeros((16, 16), dtype="float16")
    B_ref = np.zeros((16, 16), dtype="float16")
    B = tvm.runtime.tensor(B_np, device=DEV)

    mod(A, B)
    if num == 1:
        B_ref[0:8, 0:8] = A_np[0:8, 0:8] if not trans else A_np[0:8, 0:8].T
    elif num == 2:
        B_ref[0:8, 0:8] = A_np[0:8, 0:8] if not trans else A_np[0:8, 0:8].T
        B_ref[8:16, 0:8] = A_np[8:16, 0:8] if not trans else A_np[8:16, 0:8].T
    elif num == 4:
        B_ref[0:8, 0:8] = A_np[0:8, 0:8] if not trans else A_np[0:8, 0:8].T
        B_ref[0:8, 8:16] = A_np[0:8, 8:16] if not trans else A_np[0:8, 8:16].T
        B_ref[8:16, 0:8] = A_np[8:16, 0:8] if not trans else A_np[8:16, 0:8].T
        B_ref[8:16, 8:16] = A_np[8:16, 8:16] if not trans else A_np[8:16, 8:16].T

    np.testing.assert_allclose(B.numpy(), B_ref)


@pytest.mark.parametrize("d_type", ["float16", "float32"])
@pytest.mark.parametrize("no_c_ptr", [False, True])
def test_ptx_mma_half_m16n8k16(d_type, no_c_ptr):
    shape = "m16n8k16"
    a_type = "float16"
    b_type = "float16"
    c_type = d_type
    a_layout = "row"
    b_layout = "col"

    # fmt: off
    @T.prim_func(tirp=True)
    def main(
        D: T.Buffer((16, 8), d_type),
        A: T.Buffer((16, 16), a_type),
        B: T.Buffer((16, 8), b_type),
        C: T.Buffer((16, 8), c_type),
    ):
        with T.kernel():
            bx = T.cta_id([1], parent="kernel")
            tx = T.thread_id([32], parent="cta")
            with T.thread():
                D_local = T.alloc_local([4], d_type)
                A_local = T.alloc_local([8], a_type)
                B_local = T.alloc_local([4], b_type)
                C_local = T.alloc_local([4], c_type)
                
                @T.macro
                def G2L(buf_local, buf_global, block_8x8, mode="row"):
                    if mode == "row":
                        for i in range(block_8x8):
                            row = T.meta_var(i % 2 * 8 + tx // 4)
                            col = T.meta_var(i // 2 * 8 + (tx % 4) * 2)
                            for j in range(2):
                                buf_local[i * 2 + j] = buf_global[row, col + j]
                    elif mode == "col":
                        for i in range(block_8x8):
                            row = T.meta_var(i % 2 * 8 + (tx % 4) * 2)
                            col = T.meta_var(i // 2 * 8 + tx // 4)
                            for j in range(2):
                                buf_local[i * 2 + j] = buf_global[row + j, col]

                @T.macro
                def L2G(buf_local, buf_global, block_8x8):
                    for i in range(block_8x8):
                        row = T.meta_var(i % 2 * 8 + tx // 4)
                        col = T.meta_var(i // 2 * 8 + (tx % 4) * 2)
                        for j in range(2):
                            buf_global[row, col + j] = buf_local[i * 2 + j]

                G2L(D_local, D, 2)
                G2L(A_local, A, 4)
                G2L(B_local, B, 2, "col")
                G2L(C_local, C, 2)

                if no_c_ptr:
                    T.ptx.mma(shape, a_layout, b_layout, d_type, a_type, b_type, c_type, 
                              D_local.ptr_to([0]), A_local.ptr_to([0]), B_local.ptr_to([0]))
                else:
                    T.ptx.mma(shape, a_layout, b_layout, d_type, a_type, b_type, c_type, 
                              D_local.ptr_to([0]), A_local.ptr_to([0]), B_local.ptr_to([0]), C_local.ptr_to([0]))

                L2G(D_local, D, 2)    
    # fmt: on

    src, mod = _get_source(main)
    np.random.seed(0)

    D_np = np.zeros((16, 8), dtype=d_type)
    A_np = np.random.randn(16, 16).astype(a_type)
    B_np = np.random.randn(16, 8).astype(b_type)
    C_np = np.random.randn(16, 8).astype(c_type)

    D = tvm.runtime.tensor(D_np, device=DEV)
    A = tvm.runtime.tensor(A_np, device=DEV)
    B = tvm.runtime.tensor(B_np, device=DEV)
    C = tvm.runtime.tensor(C_np, device=DEV)
    mod(D, A, B, C)

    D_torch = torch.zeros((16, 8), dtype=torch.float16)
    A_torch = torch.from_numpy(A_np)
    B_torch = torch.from_numpy(B_np)
    C_torch = torch.from_numpy(C_np)
    if no_c_ptr:
        D_torch = A_torch @ B_torch
    else:
        D_torch = A_torch @ B_torch + C_torch

    np.testing.assert_allclose(D.numpy(), D_torch.numpy(), atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("d_type", ["float16", "float32"])
@pytest.mark.parametrize("no_c_ptr", [False, True])
def test_ptx_mma_half_m16n8k8(d_type, no_c_ptr):
    shape = "m16n8k8"
    a_type = "float16"
    b_type = "float16"
    c_type = d_type
    a_layout = "row"
    b_layout = "col"

    # fmt: off
    @T.prim_func(tirp=True)
    def main(
        D: T.Buffer((16, 8), d_type),
        A: T.Buffer((16, 8), a_type),
        B: T.Buffer((8, 8), b_type),
        C: T.Buffer((16, 8), c_type),
    ):
        with T.kernel():
            bx = T.cta_id([1], parent="kernel")
            tx = T.thread_id([32], parent="cta")
            with T.thread():
                D_local = T.alloc_local([4], d_type)
                A_local = T.alloc_local([4], a_type)
                B_local = T.alloc_local([2], b_type)
                C_local = T.alloc_local([4], c_type)

                @T.macro
                def G2L(buf_local, buf_global, block_8x8, mode="row"):
                    if mode == "row":
                        for i in range(block_8x8):
                            row = T.meta_var(i % 2 * 8 + tx // 4)
                            col = T.meta_var(i // 2 * 8 + (tx % 4) * 2)
                            for j in range(2):
                                buf_local[i * 2 + j] = buf_global[row, col + j]
                    elif mode == "col":
                        for i in range(block_8x8):
                            row = T.meta_var(i % 2 * 8 + (tx % 4) * 2)
                            col = T.meta_var(i // 2 * 8 + tx // 4)
                            for j in range(2):
                                buf_local[i * 2 + j] = buf_global[row + j, col]

                @T.macro
                def L2G(buf_local, buf_global, block_8x8):
                    for i in range(block_8x8):
                        row = T.meta_var(i % 2 * 8 + tx // 4)
                        col = T.meta_var(i // 2 * 8 + (tx % 4) * 2)
                        for j in range(2):
                            buf_global[row, col + j] = buf_local[i * 2 + j]

                G2L(D_local, D, 2)
                G2L(A_local, A, 2)  
                G2L(B_local, B, 1, "col")
                G2L(C_local, C, 2)

                if no_c_ptr:
                    T.ptx.mma(shape, a_layout, b_layout, d_type, a_type, b_type, c_type, 
                              D_local.ptr_to([0]), A_local.ptr_to([0]), B_local.ptr_to([0]))
                else:
                    T.ptx.mma(shape, a_layout, b_layout, d_type, a_type, b_type, c_type, 
                              D_local.ptr_to([0]), A_local.ptr_to([0]), B_local.ptr_to([0]), C_local.ptr_to([0]))

                L2G(D_local, D, 2)    
    # fmt: on

    src, mod = _get_source(main)
    np.random.seed(0)

    D_np = np.zeros((16, 8), dtype=d_type)
    A_np = np.random.randn(16, 8).astype(a_type)
    B_np = np.random.randn(8, 8).astype(b_type)
    C_np = np.random.randn(16, 8).astype(c_type)

    D = tvm.runtime.tensor(D_np, device=DEV)
    A = tvm.runtime.tensor(A_np, device=DEV)
    B = tvm.runtime.tensor(B_np, device=DEV)
    C = tvm.runtime.tensor(C_np, device=DEV)
    mod(D, A, B, C)

    D_torch = torch.zeros((16, 8), dtype=torch.float16)
    A_torch = torch.from_numpy(A_np)
    B_torch = torch.from_numpy(B_np)
    C_torch = torch.from_numpy(C_np)
    if no_c_ptr:
        D_torch = A_torch @ B_torch
    else:
        D_torch = A_torch @ B_torch + C_torch

    np.testing.assert_allclose(D.numpy(), D_torch.numpy(), atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    tvm.testing.main()
