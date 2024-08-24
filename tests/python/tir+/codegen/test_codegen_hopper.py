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
from tvm.script import tir as T, ir as I
from tvm.tir.transform import LowerTIRp


def _skip_test(target: tvm.target.Target):
    arch = target.arch
    assert arch.startswith("sm_")
    if int(arch.split("sm_")[1]) < 90:
        pytest.xfail("Test requires sm_90 or higher")


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
                pass

    # fmt: on
    target = tvm.target.Target("cuda")
    _skip_test(target)
    with target:
        mod = LowerTIRp()(tvm.IRModule({"main": func}))
    mod = tvm.build(mod, target=target)
    src = mod.imported_modules[0].get_source()
    assert "fence.proxy.async.global" in src
    assert "fence.proxy.async.shared::cta" in src


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
def test_cp_async_bulk_tensor_global_to_shared_unicast(inputs):
    def get_ir(shape, tma_args):
        total_bytes = 4 * math.prod(shape)
        coord = [0 for _ in shape]
        # fmt: off
        @T.prim_func(tirp=True)
        def main(A_ptr: T.handle, B_ptr: T.handle):
            A = T.match_buffer(A_ptr, shape, dtype="float32", align=16, logical_scope="kernel")
            B = T.match_buffer(B_ptr, shape, dtype="float32", align=16, logical_scope="kernel")
            
            A_map: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
            T.call_packed("runtime.cuTensorMapInit", A_map, "float32", len(shape), A.data, *tma_args)
            B_map: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
            T.call_packed("runtime.cuTensorMapInit", B_map, "float32", len(shape), B.data, *tma_args)

            with T.kernel():
                for blockIdx in T.thread_binding(1, thread="blockIdx.x"):
                    for threadIdx in T.thread_binding(128, thread="threadIdx.x"):
                        with T.thread():
                            bar = T.alloc_buffer((1,), "uint64", scope="shared", logical_scope="cta", align=8)
                            phase = T.alloc_buffer((1,), "int32", scope="local", logical_scope="thread")
                            A_smem = T.alloc_buffer(shape[::-1], "float32", scope="shared", logical_scope="cta", align=128)

                            phase[0] = 0
                            if threadIdx == 0:
                                T.mbarrier_init(bar.data, 1)
                                T.cuda_fence_proxy_async("shared")
                                T.cp_async_bulk_tensor_global_to_cluster(len(shape), A_smem.data, 0, bar.data, A_map, *coord)
                                T.mbarrier_arrive_expect_tx(bar.data, total_bytes)
                            T.mbarrier_wait(bar.data, phase[0])
                            
                            T.tvm_storage_sync("shared")
                            T.cuda_fence_proxy_async("shared")
                            
                            if threadIdx == 0:
                                T.cp_async_bulk_tensor_shared_to_global(len(shape), A_smem.data, 0, B_map, *coord)
                                T.cp_async_bulk_tensor_commit_group()
                                T.cp_async_bulk_tensor_wait_group(0)
        # fmt: on

        return main

    DEV = tvm.cuda(0)
    target = tvm.target.Target("cuda")
    _skip_test(target)
    shape, tma_args = inputs
    mod = tvm.IRModule({"main": get_ir(shape, tma_args)})
    mod = tvm.build(mod, target=target)
    src = mod.imported_modules[0].get_source()
    assert "const __grid_constant__ CUtensorMap" in src

    DEV = tvm.cuda(0)
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
        ((4, 4, 4), [4, 4, 4, 16, 64, 4, 4, 4, 1, 1, 1, 0, 0, 0, 0]),
        ((4, 4, 4, 4), [4, 4, 4, 4, 16, 64, 256, 4, 4, 4, 4, 1, 1, 1, 1, 0, 0, 0, 0]),
        (
            (4, 2, 2, 2, 2),
            [4, 2, 2, 2, 2, 16, 32, 64, 128, 4, 2, 2, 2, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        ),
    ],
)
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
            T.call_packed("runtime.cuTensorMapInit", A_map, "float32", len(shape), A.data, *tma_args)
            B_map: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
            T.call_packed("runtime.cuTensorMapInit", B_map, "float32", len(shape), A.data, *tma_args)

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
                                        T.cp_async_bulk_tensor_global_to_cluster(len(shape), A_smem.data, 0, 
                                                                                 bar.data, A_map, *coord, cta_mask=int("1111", 2))
                                # wait for the copy to finish
                                T.mbarrier_wait(bar.data, phase[0])
                                T.tvm_storage_sync("shared")
                                T.cuda_fence_proxy_async("shared")

                                if bx == 2:
                                    if tx == 0:
                                        T.cp_async_bulk_tensor_shared_to_global(len(shape), A_smem.data, 0, B_map, *coord)
                                        T.cp_async_bulk_tensor_commit_group()
                                        T.cp_async_bulk_tensor_wait_group(0)
        # fmt: on

        return main

    DEV = tvm.cuda(0)
    target = tvm.target.Target("cuda")
    _skip_test(target)
    shape, tma_args = inputs
    mod = tvm.IRModule({"main": get_ir(shape, tma_args)})
    mod = tvm.build(mod, target=target)
    src = mod.imported_modules[0].get_source()
    assert "const __grid_constant__ CUtensorMap" in src

    DEV = tvm.cuda(0)
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
            T.call_packed("runtime.cuTensorMapInit", A_map, "float32", len(shape), A.data, *tma_args)
            B_map: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
            T.call_packed("runtime.cuTensorMapInit", B_map, "float32", len(shape), B.data, *tma_store_args)

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
                                        T.cp_async_bulk_tensor_global_to_cluster(len(shape), A_smem.data, A_smem.offset_of_p(coord0[::-1]), 
                                                                                 bar.data, A_map, *coord0, cta_mask=int("1111", 2))
                                    if clusterCtaIdx == 1:
                                        T.cp_async_bulk_tensor_global_to_cluster(len(shape), A_smem.data, A_smem.offset_of_p(coord1[::-1]),
                                                                                 bar.data, A_map, *coord1, cta_mask=int("1111", 2))
                                    if clusterCtaIdx == 2:
                                        T.cp_async_bulk_tensor_global_to_cluster(len(shape), A_smem.data, A_smem.offset_of_p(coord2[::-1]),
                                                                                 bar.data, A_map, *coord2, cta_mask=int("1111", 2))
                                    if clusterCtaIdx == 3:
                                        T.cp_async_bulk_tensor_global_to_cluster(len(shape), A_smem.data, A_smem.offset_of_p(coord3[::-1]),
                                                                                 bar.data, A_map, *coord3, cta_mask=int("1111", 2))
                                # wait for the copy to finish
                                T.mbarrier_wait(bar.data, phase[0])
                                T.tvm_storage_sync("shared")

                                if bx == 1:
                                    if tx == 0:
                                        T.cp_async_bulk_tensor_shared_to_global(len(shape), A_smem.data, 0, B_map, *coord0)
                                        T.cp_async_bulk_tensor_commit_group()
                                        T.cp_async_bulk_tensor_wait_group(0)
        # fmt: on

        return main

    DEV = tvm.cuda(0)
    target = tvm.target.Target("cuda")
    _skip_test(target)
    shape, tma_args = inputs
    mod = tvm.IRModule({"main": get_ir(shape, tma_args)})
    with target:
        mod = LowerTIRp()(mod)
    mod = tvm.build(mod, target=target)
    src = mod.imported_modules[0].get_source()
    assert "const __grid_constant__ CUtensorMap" in src

    DEV = tvm.cuda(0)
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
            T.call_packed("runtime.cuTensorMapInit", A_map, "float32", len(shape), A.data, *tma_args)

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
                        T.cp_async_bulk_tensor_shared_to_global(len(shape), A_smem.data, 0, A_map, *coord)
                        T.cp_async_bulk_tensor_commit_group()
                        T.cp_async_bulk_tensor_wait_group(0)
        # fmt: on

        return main

    DEV = tvm.cuda(0)
    target = tvm.target.Target("cuda")
    _skip_test(target)
    shape, tma_args = inputs
    mod = tvm.IRModule({"main": get_ir(shape, tma_args)})
    with target:
        mod = LowerTIRp()(mod)
    mod = tvm.build(mod, target=target)
    src = mod.imported_modules[0].get_source()
    assert "const __grid_constant__ CUtensorMap" in src

    DEV = tvm.cuda(0)
    A_np = np.zeros(shape, dtype="float32")
    A = tvm.nd.array(A_np, device=DEV)
    mod(A)

    A_ref = [i for i in range(math.prod(shape))]
    A_ref = np.array(A_ref, dtype="float32").reshape(shape)
    np.testing.assert_allclose(A.asnumpy(), A_ref)


if __name__ == "__main__":
    test_fence_proxy_async()
    test_cp_async_bulk_tensor_global_to_shared_unicast()
    test_cp_async_bulk_tensor_global_to_shared_multicast1()
    test_cp_async_bulk_tensor_global_to_shared_multicast2()
    test_cp_async_bulk_tensor_shared_to_global()
