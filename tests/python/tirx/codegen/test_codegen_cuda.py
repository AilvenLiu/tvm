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
from tvm.script import tirx as Tx

DEV = tvm.device("cuda")


def _get_source(func: tvm.tirx.PrimFunc) -> str:
    target = tvm.target.Target("cuda")
    mod = tvm.IRModule({"main": func})
    mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
    src = mod.mod.imports[0].inspect_source()
    return src, mod


def test_cuda_atomic_add():
    @Tx.prim_func(tirx=True)
    def main(A: Tx.Buffer((1,), "int32"), B: Tx.Buffer((1,), "float32")):
        with Tx.kernel():
            Tx.cta_id([1], parent="kernel")
            tx = Tx.thread_id([32], parent="cta")
            with Tx.thread()[tx == 0]:
                Tx.cuda.atomic_add(A.data, Tx.int32(1))
                Tx.cuda.atomic_add(B.data, Tx.float32(1.0))

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
    @Tx.prim_func(tirx=True)
    def main(A: Tx.Buffer((16, 16), "int32")):
        with Tx.kernel():
            Tx.cta_id([1], parent="kernel")
            tx = Tx.thread_id([32], parent="cta")
            with Tx.thread()[tx == 0]:
                Tx.cuda.thread_fence()

    src, mod = _get_source(main)
    assert "tvm_builtin_cuda_thread_fence" in src


def test_cuda_nano_sleep():
    @Tx.prim_func(tirx=True)
    def main(A: Tx.Buffer((16, 16), "int32")):
        with Tx.kernel():
            Tx.cta_id([1], parent="kernel")
            tx = Tx.thread_id([32], parent="cta")
            with Tx.thread()[tx == 0]:
                Tx.cuda.nano_sleep(1)

    src, mod = _get_source(main)
    assert "tvm_builtin_cuda_nano_sleep" in src


def test_cuda_atomic_cas():
    @Tx.prim_func(tirx=True)
    def main(A: Tx.Buffer((16, 16), "int32")):
        with Tx.kernel():
            Tx.cta_id([1], parent="kernel")
            tx = Tx.thread_id([32], parent="cta")
            with Tx.thread()[tx == 0]:
                Tx.cuda.atomic_cas(A.data, Tx.int32(1), Tx.int32(2))

    src, mod = _get_source(main)
    assert "tvm_builtin_cuda_atomic_cas" in src


def test_cuda_func_call():
    def test_add_one():
        add_one = """
__device__ int32_t add_one(int32_t a) {
    return a + 1;
}
"""

        @Tx.prim_func(tirx=True)
        def main(a: Tx.Buffer((16, 16), "int32"), b: Tx.Buffer((16, 16), "int32")):
            with Tx.kernel():
                Tx.cta_id([1], parent="kernel")
                tx = Tx.thread_id([32], parent="cta")
                with Tx.thread()[tx == 0]:
                    for i, j in Tx.grid(16, 16):
                        b[i, j] = Tx.cuda.func_call(
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

        @Tx.prim_func(tirx=True)
        def main(a: Tx.Buffer((16, 16), "int32")):
            with Tx.kernel():
                Tx.cta_id([1], parent="kernel")
                tx = Tx.thread_id([32], parent="cta")
                with Tx.thread()[tx == 0]:
                    for i, j in Tx.grid(16, 16):
                        Tx.cuda.func_call("print", a[i, j], source_code=print_func)

        src, mod = _get_source(main)
        A = np.random.randint(0, 10, (16, 16)).astype("int32")
        A_tvm = tvm.runtime.tensor(A, device=DEV)
        mod["main"](A_tvm)
        print(src)

    test_print()


def test_warp_shuffle_xor_sync():
    # fmt: off
    @Tx.prim_func(tirx=True)
    def func(A_ptr: Tx.handle):
        A = Tx.match_buffer(A_ptr, (32,), dtype="float32", align=16)

        with Tx.kernel():
            Tx.cta_id([1], parent="kernel")
            Tx.warp_id([1], parent="cta")
            lane_id = Tx.thread_id([32], parent="warp")

            with Tx.thread():
                A_local = Tx.alloc_buffer([1], "float32", scope="local")
                i = Tx.alloc_buffer([1], "int32", scope="local")

                A_local[0] = Tx.float32(31 - lane_id)
                i[0] = 16
                while i[0] >= 1:
                    A_local[0] += Tx.tvm_warp_shuffle_xor(0xFFFFFFFF, A_local[0], i[0], 32, 32)
                    i[0] = i[0] // 2

                A[lane_id] = A_local[0]
    # fmt: on

    DEV = tvm.cuda(0)
    target = tvm.target.Target("cuda")
    mod = tvm.IRModule({"main": func})
    mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
    A_np = np.zeros(32, dtype="float32")
    A = tvm.runtime.tensor(A_np, device=DEV)
    mod(A)
    assert "__shfl_xor_sync" in mod.mod.imports[0].inspect_source()
    A_ref = np.ones(32, dtype="float32") * 496
    np.testing.assert_allclose(A.numpy(), A_ref)


@pytest.mark.parametrize("cp_size", [4, 8, 16])
@pytest.mark.parametrize("cache_hint", ["", "evict_last"])
@pytest.mark.parametrize("prefetch_size", [-1, 64, 128, 256])
@pytest.mark.parametrize("predicate", [-1, Tx.int32(0), Tx.int32(1)])
@pytest.mark.parametrize("fill_mode", ["", "zero"])
def test_ptx_cp_async(cp_size, cache_hint, prefetch_size, predicate, fill_mode):
    if fill_mode != "" and predicate == -1:
        return

    N = cp_size // 2

    # fmt: off
    @Tx.prim_func(tirx=True)
    def main(A: Tx.Buffer((N), "float16")):
        with Tx.kernel():
            Tx.cta_id([1], parent="kernel")
            Tx.thread_id([1], parent="cta")
            with Tx.thread():
                A_shared = Tx.alloc_shared([N], "float16")
                for i in Tx.vectorized(N):
                    A_shared[i] = 5.0
                Tx.ptx.fence.proxy_async("shared::cta")
                Tx.ptx.cp_async(A_shared.ptr_to([0]), A.ptr_to([0]), cp_size, cache_hint=cache_hint, prefetch_size=prefetch_size, predicate=predicate, fill_mode=fill_mode)  # noqa: E501
                Tx.ptx.cp_async.commit_group()
                Tx.ptx.cp_async.wait_group(0)
                for i in Tx.serial(N):
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
    @Tx.prim_func(tirx=True)
    def main(A: Tx.Buffer((16, 16), "float16"), B: Tx.Buffer((16, 16), "float16")):
        with Tx.kernel():
            Tx.cta_id([1], parent="kernel")
            tx = Tx.thread_id([32], parent="cta")
            A_shared = Tx.alloc_shared([16, 16], "float16")
            with Tx.thread()[tx == 0]:
                for i, j in Tx.grid(16, 16):
                    A_shared[i, j] = A[i, j]
            Tx.cuda.cta_sync()
            with Tx.thread():
                A_local = Tx.alloc_local([8], "float16")
                A_local[0] = -1.0
                Tx.ptx.ldmatrix(
                    trans, num, dtype, A_local.ptr_to([0]), A_shared.ptr_to([tx % 16, tx // 16 * 8])
                )
                for i in range(8):
                    row: Tx.let = (i // 2) % 2 * 8
                    col: Tx.let = (i // 4) * 8
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
    @Tx.prim_func(tirx=True)
    def main(
        D: Tx.Buffer((16, 8), d_type),
        A: Tx.Buffer((16, 16), a_type),
        B: Tx.Buffer((16, 8), b_type),
        C: Tx.Buffer((16, 8), c_type),
    ):
        with Tx.kernel():
            Tx.cta_id([1], parent="kernel")
            tx = Tx.thread_id([32], parent="cta")
            with Tx.thread():
                D_local = Tx.alloc_local([4], d_type)
                A_local = Tx.alloc_local([8], a_type)
                B_local = Tx.alloc_local([4], b_type)
                C_local = Tx.alloc_local([4], c_type)

                @Tx.inline
                def G2L(buf_local, buf_global, block_8x8, mode="row"):
                    if mode == "row":
                        for i in range(block_8x8):
                            row = Tx.meta_var(i % 2 * 8 + tx // 4)
                            col = Tx.meta_var(i // 2 * 8 + (tx % 4) * 2)
                            for j in range(2):
                                buf_local[i * 2 + j] = buf_global[row, col + j]
                    elif mode == "col":
                        for i in range(block_8x8):
                            row = Tx.meta_var(i % 2 * 8 + (tx % 4) * 2)
                            col = Tx.meta_var(i // 2 * 8 + tx // 4)
                            for j in range(2):
                                buf_local[i * 2 + j] = buf_global[row + j, col]

                @Tx.inline
                def L2G(buf_local, buf_global, block_8x8):
                    for i in range(block_8x8):
                        row = Tx.meta_var(i % 2 * 8 + tx // 4)
                        col = Tx.meta_var(i // 2 * 8 + (tx % 4) * 2)
                        for j in range(2):
                            buf_global[row, col + j] = buf_local[i * 2 + j]

                G2L(D_local, D, 2)
                G2L(A_local, A, 4)
                G2L(B_local, B, 2, "col")
                G2L(C_local, C, 2)

                if no_c_ptr:
                    Tx.ptx.mma(shape, a_layout, b_layout, d_type, a_type, b_type, c_type,
                              D_local.ptr_to([0]), A_local.ptr_to([0]), B_local.ptr_to([0]))
                else:
                    Tx.ptx.mma(shape, a_layout, b_layout, d_type, a_type, b_type, c_type,
                              D_local.ptr_to([0]), A_local.ptr_to([0]), B_local.ptr_to([0]), C_local.ptr_to([0]))  # noqa: E501

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
    @Tx.prim_func(tirx=True)
    def main(
        D: Tx.Buffer((16, 8), d_type),
        A: Tx.Buffer((16, 8), a_type),
        B: Tx.Buffer((8, 8), b_type),
        C: Tx.Buffer((16, 8), c_type),
    ):
        with Tx.kernel():
            Tx.cta_id([1], parent="kernel")
            tx = Tx.thread_id([32], parent="cta")
            with Tx.thread():
                D_local = Tx.alloc_local([4], d_type)
                A_local = Tx.alloc_local([4], a_type)
                B_local = Tx.alloc_local([2], b_type)
                C_local = Tx.alloc_local([4], c_type)

                @Tx.inline
                def G2L(buf_local, buf_global, block_8x8, mode="row"):
                    if mode == "row":
                        for i in range(block_8x8):
                            row = Tx.meta_var(i % 2 * 8 + tx // 4)
                            col = Tx.meta_var(i // 2 * 8 + (tx % 4) * 2)
                            for j in range(2):
                                buf_local[i * 2 + j] = buf_global[row, col + j]
                    elif mode == "col":
                        for i in range(block_8x8):
                            row = Tx.meta_var(i % 2 * 8 + (tx % 4) * 2)
                            col = Tx.meta_var(i // 2 * 8 + tx // 4)
                            for j in range(2):
                                buf_local[i * 2 + j] = buf_global[row + j, col]

                @Tx.inline
                def L2G(buf_local, buf_global, block_8x8):
                    for i in range(block_8x8):
                        row = Tx.meta_var(i % 2 * 8 + tx // 4)
                        col = Tx.meta_var(i // 2 * 8 + (tx % 4) * 2)
                        for j in range(2):
                            buf_global[row, col + j] = buf_local[i * 2 + j]

                G2L(D_local, D, 2)
                G2L(A_local, A, 2)
                G2L(B_local, B, 1, "col")
                G2L(C_local, C, 2)

                if no_c_ptr:
                    Tx.ptx.mma(shape, a_layout, b_layout, d_type, a_type, b_type, c_type,
                              D_local.ptr_to([0]), A_local.ptr_to([0]), B_local.ptr_to([0]))
                else:
                    Tx.ptx.mma(shape, a_layout, b_layout, d_type, a_type, b_type, c_type,
                              D_local.ptr_to([0]), A_local.ptr_to([0]), B_local.ptr_to([0]), C_local.ptr_to([0]))  # noqa: E501

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
