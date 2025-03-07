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
import math
import numpy as np

import tvm
from tvm.tir import Buffer
from tvm.script import tir as T, ir as I
import tvm.testing
from tvm.tir.transform import LowerTIRp


def _get_source(func: tvm.tir.PrimFunc) -> str:
    target = tvm.target.Target("nvidia/nvidia-h100")
    mod = tvm.IRModule({"main": func})
    mod = tvm.build(mod, target=target, pipeline="tirp")
    src = mod.imported_modules[0].get_source()
    return src


@pytest.mark.parametrize("inc", [False, True])
@tvm.testing.requires_cuda_compute_version(9)
def test_setmaxnreg(inc):
    # fmt: off
    @T.prim_func(tirp=True)
    def func(A: T.Buffer((1))):
        with T.kernel():
            bx = T.cta_id([1], parent="kernel")
            tx = T.thread_id([128], parent="cta")
            with T.thread():
                T.setmaxnreg(inc, 32)
    # fmt: on

    src = _get_source(func)
    assert "setmaxnreg" in src
    if inc:
        assert "inc" in src
    else:
        assert "dec" in src


@pytest.mark.parametrize("trans", [False, True])
@tvm.testing.requires_cuda_compute_version(9)
def test_stmatrix_sync_aligned(trans):
    # fmt: off
    @T.prim_func(tirp=True)
    def func(A: T.Buffer((16, 16), "float16")):
        with T.kernel():
            bx = T.cta_id([1], parent="kernel")
            tx = T.thread_id([32], parent="cta")
            with T.cta():
                A_smem = T.alloc_buffer((16, 16), "float16", scope="shared", align=16)
                with T.thread():
                    reg = T.alloc_buffer((8,), "float16", scope="local")
                    for i in range(8):
                        reg[i] = tx * 8 + i
                    T.stmatrix_sync_aligned(4, trans,
                                            A_smem.access_ptr("w", offset=A_smem.offset_of_p([tx % 16, tx // 16 * 8])),
                                            reg[0], reg[1], reg[2], reg[3],
                                            reg[4], reg[5], reg[6], reg[7])
                    if tx == 0:
                        for i, j in T.grid(16, 16):
                            A[i, j] = A_smem[i, j]
    # fmt: on

    DEV = tvm.cuda(0)
    target = tvm.target.Target("cuda")
    mod = tvm.IRModule({"main": func})
    with target:
        mod = tvm.build(mod, target=target, pipeline="tirp")
        src = mod.imported_modules[0].get_source()
        if not trans:
            assert "stmatrix.sync.aligned.m8n8.x4.shared.b16" in src
        else:
            assert "stmatrix.sync.aligned.m8n8.x4.trans.shared.b16" in src
        A_np = np.zeros((16, 16), dtype="float16")
        A = tvm.nd.array(A_np, device=DEV)
        mod(A)
        A_ref = np.zeros((16, 16), dtype="float16")
        for tx in range(32):
            row = tx // 4
            col = tx % 4 * 2
            if not trans:
                A_ref[row, col] = tx * 8
                A_ref[row, col + 1] = tx * 8 + 1
                A_ref[row + 8, col] = tx * 8 + 2
                A_ref[row + 8, col + 1] = tx * 8 + 3
                A_ref[row, col + 8] = tx * 8 + 4
                A_ref[row, col + 9] = tx * 8 + 5
                A_ref[row + 8, col + 8] = tx * 8 + 6
                A_ref[row + 8, col + 9] = tx * 8 + 7
            else:
                A_ref[col, row] = tx * 8
                A_ref[col + 1, row] = tx * 8 + 1
                A_ref[col + 8, row] = tx * 8 + 2
                A_ref[col + 9, row] = tx * 8 + 3
                A_ref[col, row + 8] = tx * 8 + 4
                A_ref[col + 1, row + 8] = tx * 8 + 5
                A_ref[col + 8, row + 8] = tx * 8 + 6
                A_ref[col + 9, row + 8] = tx * 8 + 7
        np.testing.assert_allclose(A.asnumpy(), A_ref)


@tvm.testing.requires_cuda_compute_version(9)
def test_bar_arrive():
    # fmt: off
    @T.prim_func(tirp=True)
    def func(A: T.Buffer((1))):
        with T.kernel():
            bx = T.cta_id([1], parent="kernel")
            tx = T.thread_id([128], parent="cta")
            with T.thread():
                T.named_barrier_arrive(0, 128)
    # fmt: on

    src = _get_source(func)
    assert 'bar.arrive %0, %1;" : : "r"(0), "r"(128)' in src


@tvm.testing.requires_cuda_compute_version(9)
def test_bar_sync():
    # fmt: off
    @T.prim_func(tirp=True)
    def func(A: T.Buffer((1))):
        with T.kernel():
            bx = T.cta_id([1], parent="kernel")
            tx = T.thread_id([128], parent="cta")
            with T.thread():
                T.named_barrier_sync(0, 128)
    # fmt: on

    src = _get_source(func)
    assert 'bar.sync %0, %1;" : : "r"(0), "r"(128)' in src


@tvm.testing.requires_cuda_compute_version(9)
def test_fence_mbarrier_init_release_clsuter():
    # fmt: off
    @T.prim_func(tirp=True)
    def func(A: T.Buffer((1))):
        with T.kernel():
            bx = T.cta_id([1], parent="kernel")
            tx = T.thread_id([128], parent="cta")
            with T.thread():
                T.fence_mbarrier_init_release_cluster()
    # fmt: on

    src = _get_source(func)
    assert "fence.mbarrier_init.release.cluster" in src


@tvm.testing.requires_cuda_compute_version(9)
def test_elect_sync():
    # fmt: off
    @T.prim_func(tirp=True)
    def func(A: T.Buffer((1))):
        with T.kernel():
            bx = T.cta_id([1], parent="kernel")
            tx = T.thread_id([128], parent="cta")
            with T.thread():
                if (T.elect_sync(0xFFFFFFFF)):
                    A[tx] = tx
    # fmt: on

    src = _get_source(func)
    assert "elect.sync %rx|%px, %2;" in src


@tvm.testing.requires_cuda_compute_version(9)
def test_barrier_cluster():
    # fmt: off
    @T.prim_func(tirp=True)
    def func(A: T.Buffer((1))):
        with T.kernel():
            cbx = T.cta_id([2], parent="cluster")
            bx = T.cta_id([2], parent="kernel")
            tx = T.thread_id([128], parent="cta")
            with T.thread():
                T.barrier_cluster_arrive("relaxed")
                T.barrier_cluster_wait()
    # fmt: on

    src = _get_source(func)
    assert "barrier.cluster.arrive.relaxed.aligned" in src
    assert "barrier.cluster.wait.aligned" in src


@tvm.testing.requires_cuda_compute_version(9)
def test_fence_proxy_async():
    # fmt: off
    @T.prim_func(tirp=True)
    def func(A: T.Buffer((1))):
        with T.kernel():
            bx = T.cta_id([1], parent="kernel")
            tx = T.thread_id([128], parent="cta")
            with T.thread():
                T.cuda_fence_proxy_async("global")
                T.cuda_fence_proxy_async("shared")

    # fmt: on

    src = _get_source(func)
    assert "fence.proxy.async.global" in src
    assert "fence.proxy.async.shared::cta" in src


@tvm.testing.requires_cuda_compute_version(9)
@pytest.mark.parametrize("dtype", ["float16", "float32", "e4m3_float8", "e5m2_float8"])
@pytest.mark.parametrize(
    "inputs",
    [
        ((128,), [128, 128, 1, 0, 0, 0, 0]),
        ((16, 16), [16, 16, 16, 16, 16, 1, 1, 0, 0, 0, 0]),
        ((16, 64), [64, 16, 64, 64, 16, 1, 1, 0, 0, 0, 0]),
    ],
)
def test_cp_async_bulk_tensor_global_to_shared_unicast(dtype, inputs):
    import ml_dtypes

    def get_ir(shape, tma_args):
        t_dtype = tvm.DataType(dtype)
        total_bytes = math.prod(shape) * t_dtype.bits // 8
        coord = [0 for _ in shape]
        tma_args_copy = tma_args.copy()
        for i in range(len(shape) - 1):
            tma_args_copy[len(shape) + i] *= t_dtype.bits // 8
        # fmt: off
        @T.prim_func(tirp=True)
        def main(A_ptr: T.handle, B_ptr: T.handle):
            A = T.match_buffer(A_ptr, shape, dtype=dtype, align=16, logical_scope="kernel")
            B = T.match_buffer(B_ptr, shape, dtype=dtype, align=16, logical_scope="kernel")

            A_map: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
            T.call_packed("runtime.cuTensorMapEncodeTiled", A_map, dtype, len(shape), A.data, *tma_args_copy)
            B_map: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
            T.call_packed("runtime.cuTensorMapEncodeTiled", B_map, dtype, len(shape), B.data, *tma_args_copy)

            with T.kernel():
                for blockIdx in T.thread_binding(1, thread="blockIdx.x"):
                    for threadIdx in T.thread_binding(128, thread="threadIdx.x"):
                        with T.thread():
                            bar = T.alloc_buffer((1,), "uint64", scope="shared", logical_scope="cta", align=8)
                            phase = T.alloc_buffer((1,), "int32", scope="local", logical_scope="thread")
                            A_smem = T.alloc_buffer(shape, dtype, scope="shared", logical_scope="cta", align=128)

                            phase[0] = 0
                            if threadIdx == 0:
                                T.mbarrier_init(bar.data, 1)
                                T.cuda_fence_proxy_async("shared")
                                T.cp_async_bulk_tensor_global_to_cluster(len(shape), A_smem.data, bar.data, A_map, *coord)
                                T.mbarrier_arrive_expect_tx(bar.data, total_bytes)
                            T.mbarrier_wait(bar.data, phase[0])
                            phase[0] = phase[0] ^ 1

                            T.tvm_storage_sync("shared")
                            T.cuda_fence_proxy_async("shared")

                            if threadIdx == 0:
                                T.cp_async_bulk_tensor_shared_to_global(len(shape), A_smem.access_ptr("r", offset=0), B_map, *coord)
                                T.cp_async_bulk_tensor_commit_group()
                                T.cp_async_bulk_tensor_wait_group(0)
        # fmt: on

        return main

    DEV = tvm.cuda(0)
    target = tvm.target.Target("cuda")
    shape, tma_args = inputs
    mod = tvm.IRModule({"main": get_ir(shape, tma_args)})
    mod = tvm.build(mod, target=target, pipeline="tirp")
    src = mod.imported_modules[0].get_source()
    assert "const __grid_constant__ CUtensorMap" in src

    A_np = np.random.randn(math.prod(shape))

    def get_np_dtype(dtype):
        if dtype == "e4m3_float8":
            return ml_dtypes.float8_e4m3fn
        if dtype == "e5m2_float8":
            return ml_dtypes.float8_e5m2
        return np.dtype(dtype)

    A_np = np.array(A_np).reshape(shape).astype(get_np_dtype(dtype))
    B_np = np.zeros(shape).astype(get_np_dtype(dtype))
    A = tvm.nd.array(A_np, device=DEV)
    B = tvm.nd.array(B_np, device=DEV)
    mod(A, B)
    assert np.allclose(A.asnumpy().astype("float32"), B.asnumpy().astype("float32"))


@pytest.mark.parametrize("swizzle", [1, 2, 3])
@pytest.mark.parametrize("dtype", ["uint8", "float16", "float32"])
@tvm.testing.requires_cuda_compute_version(9)
def test_cp_async_bulk_tensor_global_to_shared_swizzle(swizzle, dtype):
    def get_ir(swizzle, dtype):
        dtype = tvm.DataType(dtype)
        elem_bytes = dtype.bits // 8

        shape = [16, 64]
        tma_args = [16, 64, 16, 16, 64, 1, 1, 0, 0, 0, 0]  # 8x16B, atom for WGMMA
        shape[0] = shape[0] * (1 << swizzle) // elem_bytes
        tma_args[0] = tma_args[0] * (1 << swizzle) // elem_bytes
        tma_args[2] = tma_args[2] * (1 << swizzle)
        tma_args[3] = tma_args[3] * (1 << swizzle) // elem_bytes

        load_args = tma_args.copy()
        load_args[-3] = swizzle
        store_args = tma_args.copy()

        shape = tuple(shape)
        total_elems = math.prod(shape)
        total_bytes = total_elems * elem_bytes
        coord = [0 for _ in shape]

        # fmt: off
        @T.prim_func(tirp=True)
        def main(A_ptr: T.handle, B_ptr: T.handle):
            A = T.match_buffer(A_ptr, total_elems, dtype=dtype, align=16, logical_scope="kernel")
            B = T.match_buffer(B_ptr, total_elems, dtype=dtype, align=16, logical_scope="kernel")

            A_map: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
            T.call_packed("runtime.cuTensorMapEncodeTiled", A_map, dtype, len(shape), A.data, *load_args)
            B_map: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
            T.call_packed("runtime.cuTensorMapEncodeTiled", B_map, dtype, len(shape), B.data, *store_args)

            with T.kernel():
                for blockIdx in T.thread_binding(1, thread="blockIdx.x"):
                    for threadIdx in T.thread_binding(128, thread="threadIdx.x"):
                        with T.thread():
                            A_smem = T.alloc_buffer((total_elems,), dtype, scope="shared", logical_scope="cta", align=128)
                            bar = T.alloc_buffer((1,), "uint64", scope="shared", logical_scope="cta", align=8)
                            phase = T.alloc_buffer((1,), "int32", scope="local", logical_scope="thread")

                            phase[0] = 0
                            if threadIdx == 0:
                                T.mbarrier_init(bar.data, 1)
                                T.cuda_fence_proxy_async("shared")
                                T.cp_async_bulk_tensor_global_to_cluster(len(shape), A_smem.data, bar.data, A_map, *coord)
                                T.mbarrier_arrive_expect_tx(bar.data, total_bytes)
                            T.mbarrier_wait(bar.data, phase[0])
                            phase[0] = phase[0] ^ 1

                            T.tvm_storage_sync("shared")
                            T.cuda_fence_proxy_async("shared")

                            if threadIdx == 0:
                                T.cp_async_bulk_tensor_shared_to_global(len(shape), A_smem.access_ptr("r", offset=0), B_map, *coord)
                                T.cp_async_bulk_tensor_commit_group()
                                T.cp_async_bulk_tensor_wait_group(0)
        # fmt: on

        return main, shape

    DEV = tvm.cuda(0)
    target = tvm.target.Target("cuda")
    func, shape = get_ir(swizzle, dtype)
    mod = tvm.IRModule({"main": func})
    mod = tvm.build(mod, target=target, pipeline="tirp")
    src = mod.imported_modules[0].get_source()
    assert "const __grid_constant__ CUtensorMap" in src

    total_elems = math.prod(shape)
    A_np = [i for i in range(total_elems)]
    A_np = np.array(A_np).astype(dtype)
    B_np = np.zeros((total_elems,)).astype(dtype)
    A = tvm.nd.array(A_np, device=DEV)
    B = tvm.nd.array(B_np, device=DEV)
    mod(A, B)
    dtype = tvm.DataType(dtype)
    layout = T.SwizzleLayout(
        per_element=int(math.log2(128 // dtype.bits)), swizzle_len=swizzle, atom_len=3
    )
    B_np = B.asnumpy()
    B_swizzle = [B_np[int(layout.apply(i)[0])] for i in range(total_elems)]
    B_swizzle = np.array(B_swizzle).astype(str(dtype))
    assert np.allclose(A.asnumpy(), B_swizzle)


@pytest.mark.parametrize(
    "inputs",
    [
        ((128,), [128, 128, 1, 0, 0, 0, 0]),
        ((16, 16), [16, 16, 64, 16, 16, 1, 1, 0, 0, 0, 0]),
        ((4, 4, 4), [4, 4, 4, 16, 64, 4, 4, 4, 1, 1, 1, 0, 0, 0, 0]),
        ((4, 4, 4, 4), [4, 4, 4, 4, 16, 64, 256, 4, 4, 4, 4, 1, 1, 1, 1, 0, 0, 0, 0]),
        (
            (4, 2, 2, 2, 2),
            [4, 2, 2, 2, 2, 16, 32, 64, 128, 4, 2, 2, 2, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        ),
    ],
)
@tvm.testing.requires_cuda_compute_version(9)
def test_cp_async_bulk_tensor_global_to_shared_multicast1(inputs):
    # 1 CTA does the copy, and then multicast to all CTAs in the cluster
    def get_ir(shape, tma_args):
        total_bytes = 4 * math.prod(shape)
        coord = [0 for _ in shape]
        # fmt: off
        @T.prim_func(tirp=True)
        def main(A_ptr: T.handle, B_ptr: T.handle):
            A = T.match_buffer(A_ptr, shape, dtype="float32", align=16, logical_scope="kernel")
            B = T.match_buffer(B_ptr, shape, dtype="float32", align=16, logical_scope="kernel")

            A_map: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
            T.call_packed("runtime.cuTensorMapEncodeTiled", A_map, "float32", len(shape), A.data, *tma_args)
            B_map: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
            T.call_packed("runtime.cuTensorMapEncodeTiled", B_map, "float32", len(shape), A.data, *tma_args)

            with T.kernel():
                for clusterCtaIdx in T.thread_binding(4, thread="clusterCtaIdx.x"):
                    for bx in T.thread_binding(4, thread="blockIdx.x"):
                        for tx in T.thread_binding(128, thread="threadIdx.x"):
                            with T.thread():
                                bar = T.alloc_buffer((1,), "uint64", scope="shared", logical_scope="cta", align=8)
                                phase = T.alloc_buffer((1,), "int32", scope="local", logical_scope="thread")
                                A_smem = T.alloc_buffer(shape[::-1], "float32", scope="shared", logical_scope="cta", align=128)

                                phase[0] = 0
                                if tx == 0:
                                    # leader thread in each CTA
                                    T.mbarrier_init(bar.data, 1)
                                    T.cuda_fence_proxy_async("shared")
                                    T.mbarrier_arrive_expect_tx(bar.data, total_bytes)
                                    if clusterCtaIdx == 0:
                                        # only the first CTA in the cluster does the copy, and then multicast
                                        T.cp_async_bulk_tensor_global_to_cluster(len(shape), A_smem.data,
                                                                                 bar.data, A_map, *coord, cta_mask=int("1111", 2))
                                # wait for the copy to finish
                                T.mbarrier_wait(bar.data, phase[0])
                                phase[0] = phase[0] ^ 1
                                T.tvm_storage_sync("shared")
                                T.cuda_fence_proxy_async("shared")

                                if bx == 2:
                                    if tx == 0:
                                        T.cp_async_bulk_tensor_shared_to_global(len(shape), A_smem.access_ptr("r", offset=0), B_map, *coord)
                                        T.cp_async_bulk_tensor_commit_group()
                                        T.cp_async_bulk_tensor_wait_group(0)
        # fmt: on

        return main

    DEV = tvm.cuda(0)
    target = tvm.target.Target("cuda")
    shape, tma_args = inputs
    mod = tvm.IRModule({"main": get_ir(shape, tma_args)})
    mod = tvm.build(mod, target=target, pipeline="tirp")
    src = mod.imported_modules[0].get_source()
    assert "const __grid_constant__ CUtensorMap" in src

    A_np = [i for i in range(math.prod(shape))]
    A_np = np.array(A_np, dtype="float32").reshape(shape)
    B_np = np.zeros(shape, dtype="float32")
    A = tvm.nd.array(A_np, device=DEV)
    B = tvm.nd.array(B_np, device=DEV)
    mod(A, B)


@pytest.mark.parametrize(
    "inputs",
    [
        ((128,), [128, 32, 1, 0, 0, 0, 0]),
        ((16, 16), [16, 16, 64, 16, 4, 1, 1, 0, 0, 0, 0]),
        ((16, 16, 4), [16, 16, 4, 64, 64 * 16, 16, 16, 1, 1, 1, 1, 0, 0, 0, 0]),
    ],
)
@tvm.testing.requires_cuda_compute_version(9)
def test_cp_async_bulk_tensor_global_to_shared_multicast2(inputs):
    # 4 CTAs in the cluster do the copy of separate chunks, and then multicast to all CTAs in the cluster
    def get_ir(shape, tma_args):
        assert shape[0] % 4 == 0
        total_bytes = 4 * math.prod(shape)
        coord0 = [0 for _ in shape]
        coord1 = [0 for _ in shape[:-1]] + [shape[-1] // 4]
        coord2 = [0 for _ in shape[:-1]] + [shape[-1] // 2]
        coord3 = [0 for _ in shape[:-1]] + [3 * shape[-1] // 4]

        tma_store_args = tma_args.copy()
        tma_store_args[3 * len(shape) - 2] = shape[-1]

        # fmt: off
        @T.prim_func(tirp=True)
        def main(A_ptr: T.handle, B_ptr: T.handle):
            A = T.match_buffer(A_ptr, shape, dtype="float32", align=16, logical_scope="kernel")
            B = T.match_buffer(B_ptr, shape, dtype="float32", align=16, logical_scope="kernel")

            A_map: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
            T.call_packed("runtime.cuTensorMapEncodeTiled", A_map, "float32", len(shape), A.data, *tma_args)
            B_map: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
            T.call_packed("runtime.cuTensorMapEncodeTiled", B_map, "float32", len(shape), B.data, *tma_store_args)

            with T.kernel():
                for clusterCtaIdx in T.thread_binding(4, thread="clusterCtaIdx.x"):
                    for bx in T.thread_binding(4, thread="blockIdx.x"):
                        for tx in T.thread_binding(128, thread="threadIdx.x"):
                            with T.thread():
                                bar = T.alloc_buffer((1,), "uint64", scope="shared", logical_scope="cta", align=8)
                                phase = T.alloc_buffer((1,), "int32", scope="local", logical_scope="thread")
                                A_smem = T.alloc_buffer(shape[::-1], "float32", scope="shared", logical_scope="cta", align=128)

                                phase[0] = 0
                                if tx == 0:
                                    # leader thread in each CTA
                                    T.mbarrier_init(bar.data, 1)
                                    T.cuda_fence_proxy_async("shared")
                                    T.mbarrier_arrive_expect_tx(bar.data, total_bytes)
                                    if clusterCtaIdx == 0:
                                        T.cp_async_bulk_tensor_global_to_cluster(len(shape), A_smem.access_ptr(Buffer.WRITE, offset=A_smem.offset_of_p(coord0[::-1])),
                                                                                 bar.data, A_map, *coord0, cta_mask=int("1111", 2))
                                    if clusterCtaIdx == 1:
                                        T.cp_async_bulk_tensor_global_to_cluster(len(shape), A_smem.access_ptr(Buffer.WRITE, offset=A_smem.offset_of_p(coord1[::-1])),
                                                                                 bar.data, A_map, *coord1, cta_mask=int("1111", 2))
                                    if clusterCtaIdx == 2:
                                        T.cp_async_bulk_tensor_global_to_cluster(len(shape), A_smem.access_ptr(Buffer.WRITE, offset=A_smem.offset_of_p(coord2[::-1])),
                                                                                 bar.data, A_map, *coord2, cta_mask=int("1111", 2))
                                    if clusterCtaIdx == 3:
                                        T.cp_async_bulk_tensor_global_to_cluster(len(shape), A_smem.access_ptr(Buffer.WRITE, offset=A_smem.offset_of_p(coord3[::-1])),
                                                                                 bar.data, A_map, *coord3, cta_mask=int("1111", 2))
                                # wait for the copy to finish
                                T.mbarrier_wait(bar.data, phase[0])
                                phase[0] = phase[0] ^ 1
                                T.tvm_storage_sync("shared")

                                if bx == 1:
                                    if tx == 0:
                                        T.cp_async_bulk_tensor_shared_to_global(len(shape), A_smem.access_ptr("r", offset=0), B_map, *coord0)
                                        T.cp_async_bulk_tensor_commit_group()
                                        T.cp_async_bulk_tensor_wait_group(0)
        # fmt: on

        return main

    DEV = tvm.cuda(0)
    target = tvm.target.Target("cuda")
    shape, tma_args = inputs
    mod = tvm.IRModule({"main": get_ir(shape, tma_args)})
    mod = tvm.build(mod, target=target, pipeline="tirp")
    src = mod.imported_modules[0].get_source()
    assert "const __grid_constant__ CUtensorMap" in src

    A_np = [i for i in range(math.prod(shape))]
    A_np = np.array(A_np, dtype="float32").reshape(shape)
    B_np = np.zeros(shape, dtype="float32")
    A = tvm.nd.array(A_np, device=DEV)
    B = tvm.nd.array(B_np, device=DEV)
    mod(A, B)
    assert np.allclose(A.asnumpy(), B.asnumpy())


@pytest.mark.parametrize(
    "inputs",
    [
        ((128,), [128, 128, 1, 0, 0, 0, 0]),
        ((16, 16), [16, 16, 64, 16, 16, 1, 1, 0, 0, 0, 0]),
        ((16, 16, 4), [16, 16, 4, 64, 64 * 16, 16, 16, 4, 1, 1, 1, 0, 0, 0, 0]),
    ],
)
@tvm.testing.requires_cuda_compute_version(9)
def test_cp_async_bulk_tensor_shared_to_global(inputs):
    def get_ir(shape, tma_args):
        assert shape[0] % 4 == 0
        elems = math.prod(shape)
        coord = [0 for _ in shape]

        # fmt: off
        @T.prim_func(tirp=True)
        def main(A_ptr: T.handle):
            A = T.match_buffer(A_ptr, shape, dtype="float32", align=16, logical_scope="kernel")
            
            A_map: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
            T.call_packed("runtime.cuTensorMapEncodeTiled", A_map, "float32", len(shape), A.data, *tma_args)

            with T.kernel():
                bx = T.cta_id([1], parent="kernel")
                tx = T.thread_id([128], parent="cta")

                with T.thread():
                    A_smem = T.alloc_buffer(elems, "float32", scope="shared", logical_scope="cta", align=128)

                    if tx == 0:
                        for i in T.serial(0, elems):
                            A_smem[i] = i
                    T.cuda_fence_proxy_async("shared")
                    T.tvm_storage_sync("shared")
                    
                    if tx == 0:
                        T.cp_async_bulk_tensor_shared_to_global(len(shape), A_smem.access_ptr("r", offset=0), A_map, *coord)
                        T.cp_async_bulk_tensor_commit_group()
                        T.cp_async_bulk_tensor_wait_group(0)
        # fmt: on

        return main

    DEV = tvm.cuda(0)
    target = tvm.target.Target("cuda")
    shape, tma_args = inputs
    mod = tvm.IRModule({"main": get_ir(shape, tma_args)})
    mod = tvm.build(mod, target=target, pipeline="tirp")
    src = mod.imported_modules[0].get_source()
    assert "const __grid_constant__ CUtensorMap" in src

    A_np = np.zeros(shape, dtype="float32")
    A = tvm.nd.array(A_np, device=DEV)
    mod(A)

    A_ref = [i for i in range(math.prod(shape))]
    A_ref = np.array(A_ref, dtype="float32").reshape(shape)
    np.testing.assert_allclose(A.asnumpy(), A_ref)


@tvm.testing.requires_cuda_compute_version(9)
def test_wgmma_ss_nt():
    def get_ir(
        shapeA,
        shapeB,
        shapeC,
        A_tma_args,
        B_tma_args,
        in_dtype,
        out_dtype,
        A_encode_args,
        B_encode_args,
    ):
        coordA = [0 for _ in shapeA]
        coordB = [0 for _ in shapeB]
        A_bytes = tvm.DataType(in_dtype).bits // 8 * math.prod(shapeA)
        B_bytes = tvm.DataType(in_dtype).bits // 8 * math.prod(shapeB)

        C_elems = math.prod(shapeC) // 128

        M, K = shapeA if not transA else shapeA[::-1]
        N, _ = shapeB if not transB else shapeB[::-1]

        def get_init_value(dtype):
            if dtype == "float32":
                return T.float32(0.0)
            assert False, f"Unsupported dtype {dtype}"

        def get_accum_list(C, C_elems):
            return [C[i] for i in range(C_elems)]

        # fmt: off
        @T.prim_func(tirp=True)
        def main(A_ptr: T.handle, B_ptr: T.handle, C_ptr: T.handle):
            A = T.match_buffer(A_ptr, shapeA, dtype=in_dtype, align=16)
            B = T.match_buffer(B_ptr, shapeB, dtype=in_dtype, align=16)
            C = T.match_buffer(C_ptr, shapeC, dtype=out_dtype, align=16)

            A_map: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
            T.call_packed("runtime.cuTensorMapEncodeTiled", A_map, in_dtype, len(shapeA), A.data, *A_tma_args)
            B_map: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
            T.call_packed("runtime.cuTensorMapEncodeTiled", B_map, in_dtype, len(shapeB), B.data, *B_tma_args)

            with T.kernel():
                bx = T.cta_id([1], parent="kernel")
                tx = T.thread_id([128], parent="cta") # A warpgroup is 128 threads

                with T.thread():
                    A_smem = T.alloc_buffer(shapeA, in_dtype, scope="shared", align=1024)
                    B_smem = T.alloc_buffer(shapeB, in_dtype, scope="shared", align=1024)
                    bar = T.alloc_buffer((1,), "uint64", scope="shared", align=8)
                    phase = T.alloc_buffer((1,), "int32", scope="local")
                    
                    descA = T.alloc_buffer((1,), "uint64", scope="local")
                    descB = T.alloc_buffer((1,), "uint64", scope="local")
                    C_local = T.alloc_buffer((C_elems,), out_dtype, scope="local")

                    # init phase and bar
                    phase[0] = 0
                    if tx == 0:
                        T.mbarrier_init(bar.data, 1)
                    T.cuda_fence_proxy_async("shared")
                    T.tvm_storage_sync("shared")
                    # load A and B to smem
                    if tx == 0:
                        T.cp_async_bulk_tensor_global_to_cluster(len(shapeA), A_smem.data, bar.data, A_map, *coordA)
                        T.cp_async_bulk_tensor_global_to_cluster(len(shapeB), B_smem.data, bar.data, B_map, *coordB)
                        T.mbarrier_arrive_expect_tx(bar.data, A_bytes + B_bytes)
                    T.mbarrier_wait(bar.data, phase[0])
                    phase[0] = phase[0] ^ 1
                    T.tvm_storage_sync("shared")

                    # init C_local
                    for i in T.serial(0, C_elems):
                        C_local[i] = T.Cast(out_dtype, get_init_value(out_dtype))
                        T.wgmma_fence_operand(C_local[i])
                    
                    # do wgmma
                    T.encode_matrix_descriptor(descA.data, A_smem.data, *A_encode_args)
                    T.encode_matrix_descriptor(descB.data, B_smem.data, *B_encode_args)
                    T.wgmma_arrive()
                    T.wgmma_mma_async_ss(M, N, K, in_dtype, out_dtype, transA, transB, 1.0, 1.0, False, 
                                        descA[0], descB[0], *get_accum_list(C_local, C_elems))
                    T.wgmma_commit_group()
                    T.wgmma_wait_group(0)
                    
                    for i in T.serial(0, C_elems):
                        T.wgmma_fence_operand(C_local[i])
                    
                    # store C_local to C
                    for i in T.serial(0, C_elems // 4):
                        row = T.meta_var((tx % 32) // 4 + (tx // 32) * 16)
                        col = T.meta_var(i * 8 + tx % 4 * 2)
                        C[row, col] = C_local[i * 4]
                        C[row, col + 1] = C_local[i * 4 + 1]
                        C[row + 8, col] = C_local[i * 4 + 2]
                        C[row + 8, col + 1] = C_local[i * 4 + 3]
        # fmt: on

        return main

    in_dtype = "float16"
    out_dtype = "float32"
    transA = transB = True
    swizzleA = swizzleB = 3

    t_in_dtype = tvm.DataType(in_dtype)
    elem_bytes = t_in_dtype.bits // 8

    DEV = tvm.cuda(0)
    target = tvm.target.Target("nvidia/nvidia-h100")
    M = 64
    N = 64
    K = 256 // t_in_dtype.bits
    shapeA = (M, K) if not transA else (K, M)
    shapeB = (N, K) if not transB else (K, N)
    shapeC = (M, N)

    # A tma args
    A_outer, A_inner = shapeA
    A_tma_args = [A_inner, A_outer, A_inner * elem_bytes, A_inner, A_outer, 1, 1, 0, swizzleA, 0, 0]
    # B tma args
    B_outer, B_inner = shapeB
    B_tma_args = [B_inner, B_outer, B_inner * elem_bytes, B_inner, B_outer, 1, 1, 0, swizzleB, 0, 0]
    # A encode args
    A_encode_args = [1, 64, swizzleA]
    B_encode_args = [1, 64, swizzleB]

    func = get_ir(
        shapeA,
        shapeB,
        shapeC,
        A_tma_args,
        B_tma_args,
        in_dtype,
        out_dtype,
        A_encode_args,
        B_encode_args,
    )
    mod = tvm.IRModule({"main": func})
    mod = tvm.build(mod, target=target, pipeline="tirp")

    np.random.seed(0)
    A_np = np.random.randn(*shapeA).astype(in_dtype)
    B_np = np.random.randn(*shapeB).astype(in_dtype)
    C_np = np.zeros(shapeC).astype(out_dtype)

    A_tvm = tvm.nd.array(A_np, device=DEV)
    B_tvm = tvm.nd.array(B_np, device=DEV)
    C_tvm = tvm.nd.array(C_np, device=DEV)
    mod(A_tvm, B_tvm, C_tvm)

    C_ref = np.dot(A_np.T, B_np).astype(out_dtype)
    tvm.testing.assert_allclose(C_tvm.asnumpy(), C_ref, rtol=1e-3, atol=1e-3)


@tvm.testing.requires_cuda_compute_version(9)
def test_wgmma_rs_nt():
    def get_ir(
        shapeA,
        shapeB,
        shapeC,
        B_tma_args,
        in_dtype,
        in_dtype_bits,
        out_dtype,
        B_encode_args,
    ):
        coordB = [0 for _ in shapeB]
        B_bytes = tvm.DataType(in_dtype).bits // 8 * math.prod(shapeB)

        A_elems = math.prod(shapeA) // 128
        C_elems = math.prod(shapeC) // 128

        M, K = shapeA if not transA else shapeA[::-1]
        N, _ = shapeB if not transB else shapeB[::-1]

        def get_init_value(dtype):
            if dtype == "float32":
                return T.float32(0.0)
            assert False, f"Unsupported dtype {dtype}"

        def get_A_list(A_local, A_elems):
            return [A_local[i] for i in range(A_elems)]

        def get_accum_list(C, C_elems):
            return [C[i] for i in range(C_elems)]

        # fmt: off
        @T.prim_func(tirp=True)
        def main(A_ptr: T.handle, B_ptr: T.handle, C_ptr: T.handle):
            A = T.match_buffer(A_ptr, shapeA, dtype=in_dtype, align=16)
            B = T.match_buffer(B_ptr, shapeB, dtype=in_dtype, align=16)
            C = T.match_buffer(C_ptr, shapeC, dtype=out_dtype, align=16)

            B_map: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
            T.call_packed("runtime.cuTensorMapEncodeTiled", B_map, in_dtype, len(shapeB), B.data, *B_tma_args)

            with T.kernel():
                bx = T.cta_id([1], parent="kernel")
                tx = T.thread_id([128], parent="cta") # A warpgroup is 128 threads

                with T.thread():
                    B_smem = T.alloc_buffer(shapeB, in_dtype, scope="shared", align=1024)
                    bar = T.alloc_buffer((1,), "uint64", scope="shared", align=8)

                    descB = T.alloc_buffer((1,), "uint64", scope="local")
                    A_local = T.alloc_buffer((A_elems,), in_dtype, scope="local")
                    C_local = T.alloc_buffer((C_elems,), out_dtype, scope="local")
                    
                    A_elems_b32 = T.meta_var(A_elems // (32 // in_dtype_bits))
                    A_local_b32 = T.decl_buffer((A_elems_b32,), "uint32", data=A_local.data)

                    # load A to regs
                    for i in T.serial(0, A_elems // 4):
                        row = T.meta_var((tx % 32) // 4 + (tx // 32) * 16)
                        col = T.meta_var(i * 8 + tx % 4 * 2)
                        A_local[i * 4] = A[row, col]
                        A_local[i * 4 + 1] = A[row, col + 1]
                        A_local[i * 4 + 2] = A[row + 8, col]
                        A_local[i * 4 + 3] = A[row + 8, col + 1]
                    # init bar, and make sure it's visible to all threads and async proxy
                    if tx == 0:
                        T.mbarrier_init(bar.data, 1)
                    T.cuda_fence_proxy_async("shared")
                    T.tvm_storage_sync("shared")
                    # load B to smem
                    if tx == 0:
                        T.cp_async_bulk_tensor_global_to_cluster(len(shapeB), B_smem.data, bar.data, B_map, *coordB)
                        T.mbarrier_arrive_expect_tx(bar.data, B_bytes)
                    T.mbarrier_wait(bar.data, 0)
                    T.tvm_storage_sync("shared")

                    # init C_local
                    for i in T.serial(0, C_elems):
                        C_local[i] = T.Cast(out_dtype, get_init_value(out_dtype))

                    # fence A_local and C_local
                    for i in T.serial(0, A_elems_b32):
                        T.wgmma_fence_operand(A_local_b32[i])
                    for i in T.serial(0, C_elems):
                        T.wgmma_fence_operand(C_local[i]) 
                    # do wgmma
                    T.encode_matrix_descriptor(descB.data, B_smem.data, *B_encode_args)
                    T.wgmma_arrive()
                    T.wgmma_mma_async_rs(M, N, K, in_dtype, out_dtype, transA, transB, 1.0, 1.0, False, 
                                        descB[0], *(get_A_list(A_local_b32, A_elems_b32) + get_accum_list(C_local, C_elems)))
                    T.wgmma_commit_group()
                    T.wgmma_wait_group(0)

                    # fence A_local
                    for i in T.serial(0, A_elems_b32):
                        T.wgmma_fence_operand(A_local_b32[i])
                    # fence C_local
                    for i in T.serial(0, C_elems):
                        T.wgmma_fence_operand(C_local[i])

                    # store C_local to C
                    for i in T.serial(0, C_elems // 4):
                        row = T.meta_var((tx % 32) // 4 + (tx // 32) * 16)
                        col = T.meta_var(i * 8 + tx % 4 * 2)
                        C[row, col] = C_local[i * 4]
                        C[row, col + 1] = C_local[i * 4 + 1]
                        C[row + 8, col] = C_local[i * 4 + 2]
                        C[row + 8, col + 1] = C_local[i * 4 + 3]
        # fmt: on

        return main

    in_dtype = "float16"
    in_dtype_bits = 16
    out_dtype = "float32"
    transA = False
    transB = True
    swizzleA = swizzleB = 3

    t_in_dtype = tvm.DataType(in_dtype)
    elem_bytes = t_in_dtype.bits // 8

    DEV = tvm.cuda(0)
    target = tvm.target.Target("nvidia/nvidia-h100")
    M = 64
    N = 64
    K = 256 // t_in_dtype.bits
    shapeA = (M, K) if not transA else (K, M)
    shapeB = (N, K) if not transB else (K, N)
    shapeC = (M, N)

    # B tma args
    B_outer, B_inner = shapeB
    B_tma_args = [B_inner, B_outer, B_inner * elem_bytes, B_inner, B_outer, 1, 1, 0, swizzleB, 0, 0]
    # B encode args
    B_encode_args = [1, 64, swizzleB]

    func = get_ir(
        shapeA,
        shapeB,
        shapeC,
        B_tma_args,
        in_dtype,
        in_dtype_bits,
        out_dtype,
        B_encode_args,
    )
    mod = tvm.IRModule({"main": func})
    mod = tvm.build(mod, target=target, pipeline="tirp")

    np.random.seed(0)
    A_np = np.random.randn(*shapeA).astype(in_dtype)
    B_np = np.random.randn(*shapeB).astype(in_dtype)
    C_np = np.zeros(shapeC).astype(out_dtype)

    A_tvm = tvm.nd.array(A_np, device=DEV)
    B_tvm = tvm.nd.array(B_np, device=DEV)
    C_tvm = tvm.nd.array(C_np, device=DEV)
    mod(A_tvm, B_tvm, C_tvm)

    np.printoptions(threshold=np.inf)
    np.printoptions(linewidth=np.inf)
    np.printoptions(precision=2)

    C_ref = np.dot(A_np, B_np).astype(out_dtype)
    tvm.testing.assert_allclose(C_tvm.asnumpy(), C_ref, rtol=1e-3, atol=1e-3)


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
    mod = tvm.build(mod, target=target, pipeline="tirp")
    A_np = np.zeros(32, dtype="float32")
    A = tvm.nd.array(A_np, device=DEV)
    mod(A)
    assert "__shfl_xor_sync" in mod.imported_modules[0].get_source()
    A_ref = np.ones(32, dtype="float32") * 496
    np.testing.assert_allclose(A.asnumpy(), A_ref)


if __name__ == "__main__":
    test_stmatrix_sync_aligned()
    test_bar_sync()
    test_elect_sync()
    test_barrier_cluster()
    test_fence_proxy_async()
    test_cp_async_bulk_tensor_global_to_shared_unicast()
    test_cp_async_bulk_tensor_global_to_shared_swizzle()
    test_cp_async_bulk_tensor_global_to_shared_multicast1()
    test_cp_async_bulk_tensor_global_to_shared_multicast2()
    test_cp_async_bulk_tensor_shared_to_global()
    test_wgmma_ss_nt()
    test_wgmma_rs_nt()
    test_warp_shuffle_xor_sync()
