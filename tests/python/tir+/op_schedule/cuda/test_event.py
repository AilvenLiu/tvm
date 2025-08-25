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
# pylint: disable=invalid-name, missing-function-docstring
import functools

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.script import tir as T
from tvm.script import tirp as Tp
from tvm.tir.event import EventImpl
from tvm.tir.layout import TileLayout
from tvm.tirp.op_schedule.cuda.copy_async import (
    tma_atom_layout,
    tma_atom_shape,
    tma_shared_layout,
)


@pytest.mark.parametrize(
    "task",
    [
        ################ A[0:8, 0:8] -> A_smem[0:8, 0:8] -> B[0:8, 0:8] ################
        (
            (16, 16),  # g_shape
            (8, 8),  # s_shape
            (0, 0),  # g_st
            (8, 8),  # g_extent
            8,  # thread_cnt
            TileLayout([16, 16]),  # layoutA
            TileLayout([16, 16]),  # layoutB
            TileLayout([8, 8]),  # layoutS
        ),
        ################ A[0:128, 0:32] -> A_smem[0:128, 0:32] -> B[0:128, 0:32] ################
        (
            (128, 32),  # g_shape
            (128, 32),  # s_shape
            (0, 0),  # g_st
            (128, 32),  # g_extent
            32,  # thread_cnt
            TileLayout([128, 32]),  # layoutA
            TileLayout([128, 32]),  # layoutB
            TileLayout([128, 32]),  # layoutS
        ),
        ################ A[32:64, 32:64] -> A_smem[0:32, 0:32] -> B[32:64, 32:64] ################
        (
            (64, 64),  # g_shape
            (32, 32),  # s_shape
            (32, 0),  # g_st
            (32, 32),  # g_extent
            32,  # thread_cnt
            TileLayout([64, 64]),  # layoutA
            TileLayout([64, 64]),  # layoutB
            TileLayout([32, 32]),  # layoutS
        ),
        ################ A[0:1, 0:32, 0:32] -> A_smem[0:32, 0:32] -> B[0:1, 0:32, 0:32] ################
        (
            (4, 32, 32),  # g_shape
            (32, 32),  # s_shape
            (0, 0, 0),  # g_st
            (1, 32, 32),  # g_extent
            32,  # thread_cnt
            TileLayout([4, 32, 32]),  # layoutA
            TileLayout([4, 32, 32]),  # layoutB
            TileLayout([32, 32]),  # layoutS
        ),
    ],
)
@pytest.mark.parametrize(
    "dtype", ["int8", "float8_e4m3fn", "float8_e5m2", "float16", "bfloat16", "float32"]
)
def test_copy_g2s_s2g_cta_vec_load(task, dtype):
    g_shape, s_shape, g_st, g_extent, thread_cnt, layoutA, layoutB, layoutS = task
    dev = tvm.cuda(0)

    r_smem = list(slice(None) for i in range(len(s_shape)))
    r_gmem = list(slice(g_st[i], g_st[i] + g_extent[i]) for i in range(len(g_shape)))

    # fmt: off
    @T.prim_func(tirp=True)
    def copy_async(A_ptr: T.handle, B_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, g_shape, dtype, layout=layoutA)
        B = T.match_buffer(B_ptr, g_shape, dtype, layout=layoutB)
        
        with T.kernel():
            bx = T.cta_id([1], parent="kernel")
            tx = T.thread_id([thread_cnt], parent="cta")

            with T.cta():
                A_smem = T.alloc_buffer(s_shape, dtype, scope="shared", layout=layoutS)

                event = Tp.alloc_bulk_group_event(EventImpl.kCpAsync)
                event.init()
                Tp.copy_async(A_smem[*r_smem], A[*r_gmem], event)
                event.commit()
                event.wait(0)
                T.tvm_storage_sync("shared")
                Tp.copy(B[*r_gmem], A_smem[*r_smem])
    # fmt: on

    np_dtype = tvm.testing.np_dtype_from_str(dtype)
    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": copy_async})
        mod = tvm.tir.transform.LowerTIRp()(mod)
        mod = tvm.compile(mod, target=target, tir_pipeline="tirp")

        np.random.seed(0)
        A_np = np.random.rand(*g_shape).astype(np_dtype)
        B_np = np.zeros(g_shape, dtype=np_dtype)

        A = tvm.nd.array(A_np, dev)
        B = tvm.nd.array(B_np, dev)
        mod(A, B)

        B_ref = B_np.copy()
        B_ref[*r_gmem] = A_np[*r_gmem]
        np.testing.assert_allclose(B_ref, B.numpy())


@tvm.testing.requires_cuda_compute_version(9)
@pytest.mark.parametrize("swizzle_len", [3, 2, 1, 0])
@pytest.mark.parametrize(
    "dtype", ["int8", "uint8", "float8_e4m3fn", "float8_e5m2", "float16", "bfloat16", "float32"]
)
@pytest.mark.parametrize(
    "task",
    [
        (
            (8, 256),  # global_shape
            ((0, 8), (0, 256)),  # global_region
            (8, 256),  # shared_shape
            ((0, 8), (0, 256)),  # shared_region
            8,  # thread count per CTA
            TileLayout([8, 256]),  # A_layout
            TileLayout([8, 256]),  # B_layout
            lambda dtype, swizzle_len: tma_shared_layout(dtype, swizzle_len, (8, 256)),
        ),
        (
            (64, 256),  # global_shape
            ((0, 64), (0, 256)),  # global_region
            (3, 64, 256),  # shared_shape
            ((1, 2), (0, 64), (0, 256)),  # shared_region
            64,  # thread count per CTA
            TileLayout([64, 256]),  # A_layout
            TileLayout([64, 256]),  # B_layout
            lambda dtype, swizzle_len: tma_shared_layout(dtype, swizzle_len, (3, 64, 256)),
        ),
        (
            (32, 512),  # global_shape
            ((0, 32), (0, 512)),  # global_region
            (32, 512),  # shared_shape
            ((0, 32), (0, 512)),  # shared_region
            64,  # thread count per CTA
            TileLayout([32, 512]),  # A_layout
            TileLayout([32, 512]),  # B_layout
            lambda dtype, swizzle_len: (
                tma_atom_layout(dtype, swizzle_len)
                .tile_to((16, 256), tma_atom_shape(dtype, swizzle_len))
                .tile_to((32, 512), (16, 256))
                if swizzle_len > 0
                else None
            ),
        ),
        (
            (8192, 8192),
            ((0, 128), (0, 64)),
            (128, 64),
            ((0, 128), (0, 64)),
            128,
            TileLayout((8192, 8192)),
            TileLayout((8192, 8192)),
            lambda dtype, swizzle_len: (
                tma_shared_layout(dtype, swizzle_len, (128, 64)) if dtype == "float16" else None
            ),
        ),
    ],
)
@pytest.mark.parametrize("cache_hint", ["evict_last", ""])
def test_copy_g2s_cta_tma_load(task, dtype, swizzle_len, cache_hint):
    g_shape, g_region, s_shape, s_region, thread_cnt, layoutA, layoutB, layoutS_fn = task
    dev = tvm.cuda(0)

    # Compute the shared layout using the provided swizzle length
    shared_layout = layoutS_fn(dtype, swizzle_len)
    if shared_layout is None:
        # skip the test
        # TODO(@bohan): box_dim must have each dim less than or equal to 256
        return

    total_bytes = functools.reduce(lambda acc, region: acc * (region[1] - region[0]), s_region, 1)
    total_bytes = total_bytes * tvm.DataType(dtype).bits // 8

    smem_bytes = functools.reduce(lambda acc, extent: acc * extent, s_shape, 1)
    smem_bytes = smem_bytes * tvm.DataType(dtype).bits // 8

    r_smem = [slice(s_region[i][0], s_region[i][1]) for i in range(len(s_shape))]
    r_gmem = [slice(g_region[i][0], g_region[i][1]) for i in range(len(g_shape))]

    # fmt: off
    @T.prim_func(tirp=True)
    def copy_async(A_ptr: T.handle, B_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, g_shape, dtype, layout=layoutA)
        B = T.match_buffer(B_ptr, g_shape, dtype, layout=layoutB)

        with T.kernel():
            bx = T.cta_id([1], parent="kernel")
            tx = T.thread_id([thread_cnt], parent="cta")

            # Allocate shared memory and mbarrier in a unified buffer
            with T.thread():
                dyn = T.alloc_buffer([smem_bytes + 8], "uint8", scope="shared.dyn")
                A_smem = T.decl_buffer(s_shape, dtype, dyn.data, elem_offset=0, layout=shared_layout)
                mbarrier = T.decl_buffer([1], "uint64", dyn.data, elem_offset=smem_bytes // 8, layout=TileLayout((1,)))
                phase = T.alloc_buffer([1], "int32", scope="local")
                tx_cnt = T.alloc_buffer([1], "int32", scope="local")

                with T.cta():
                    event = Tp.alloc_semaphore_event_tensor(EventImpl.kTMALoad, state=[mbarrier, phase, tx_cnt])
                    event[0].init(1)

                    Tp.copy_async(A_smem[*r_smem], A[*r_gmem], event[0], schedule_config={"cache_hint": cache_hint})
                    event[0].commit()
                    event[0].wait()

                    Tp.copy(B[*r_gmem], A_smem[*r_smem])
    # fmt: on
    np_dtype = tvm.testing.np_dtype_from_str(dtype)
    target = tvm.target.Target("cuda")

    with target:
        mod = tvm.IRModule({"main": copy_async})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirp")
        print(mod.mod.imports[0].inspect_source())

        np.random.seed(0)
        A_np = tvm.testing.generate_random_array(dtype, g_shape)
        B_np = np.zeros(g_shape, dtype=np_dtype)

        A = tvm.nd.array(A_np, dev)
        B = tvm.nd.array(B_np, dev)
        mod(A, B)

        B_ref = np.zeros(g_shape, dtype=np_dtype)
        B_ref[*r_gmem] = A_np[*r_gmem]
        np.testing.assert_allclose(B_ref, B.numpy())


@tvm.testing.requires_cuda_compute_version(9)
@pytest.mark.parametrize("swizzle_len", [3, 2, 1, 0])
@pytest.mark.parametrize(
    "dtype", ["int8", "uint8", "float8_e4m3fn", "float8_e5m2", "float16", "bfloat16", "float32"]
)
@pytest.mark.parametrize(
    "task",
    [
        (
            (3, 8, 256),  # global_shape
            (8, 256),  # shared_shape
            8,  # thread count per CTA
            TileLayout([3, 8, 256]),  # A_layout
            TileLayout([3, 8, 256]),  # B_layout
            lambda dtype, swizzle_len: tma_shared_layout(dtype, swizzle_len, (8, 256)),
        ),
        (
            (5, 64, 256),  # global_shape
            (64, 256),  # shared_shape
            64,  # thread count per CTA
            TileLayout([5, 64, 256]),  # A_layout
            TileLayout([5, 64, 256]),  # B_layout
            lambda dtype, swizzle_len: tma_shared_layout(dtype, swizzle_len, (64, 256)),
        ),
        (
            (7, 32, 512),  # global_shape
            (32, 512),  # shared_shape
            32,  # thread count per CTA
            TileLayout([7, 32, 512]),  # A_layout
            TileLayout([7, 32, 512]),  # B_layout
            lambda dtype, swizzle_len: (
                tma_atom_layout(dtype, swizzle_len)
                .tile_to((16, 256), tma_atom_shape(dtype, swizzle_len))
                .tile_to((32, 512), (16, 256))
                if swizzle_len > 0
                else None
            ),
        ),
    ],
)
def test_copy_g2s_cta_tma_load_multi_phase(task, dtype, swizzle_len):
    g_shape, s_shape, thread_cnt, layoutA, layoutB, layoutS_fn = task
    dev = tvm.cuda(0)
    n = g_shape[0]

    # Compute the shared layout using the provided swizzle length
    shared_layout = layoutS_fn(dtype, swizzle_len)
    if shared_layout is None:
        # skip the test
        # TODO(@bohan): box_dim must have each dim less than or equal to 256
        return

    smem_bytes = functools.reduce(lambda acc, extent: acc * extent, s_shape, 1)
    smem_bytes = smem_bytes * tvm.DataType(dtype).bits // 8

    r_smem = [slice(0, s) for s in s_shape]

    def r_gmem(stage):
        return [
            slice(stage, stage + 1),
            *[slice(0, g_shape[i]) for i in range(1, len(g_shape))],
        ]

    # fmt: off
    @T.prim_func(tirp=True)
    def copy_async(A_ptr: T.handle, B_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, g_shape, dtype, layout=layoutA)
        B = T.match_buffer(B_ptr, g_shape, dtype, layout=layoutB)

        with T.kernel():
            bx = T.cta_id([1], parent="kernel")
            tx = T.thread_id([thread_cnt], parent="cta")

            # Allocate shared memory and mbarrier in a unified buffer
            with T.thread():
                dyn = T.alloc_buffer([smem_bytes + 8], "uint8", scope="shared.dyn")
                A_smem = T.decl_buffer(s_shape, dtype, dyn.data, elem_offset=0, layout=shared_layout)
                mbarrier = T.decl_buffer([1], "uint64", dyn.data, elem_offset=smem_bytes // 8, layout=TileLayout((1,)))
                phase = T.alloc_buffer([1], "int32", scope="local")
                tx_cnt = T.alloc_buffer([1], "int32", scope="local")

                with T.cta():
                    event = Tp.alloc_semaphore_event_tensor(EventImpl.kTMALoad, state=[mbarrier, phase, tx_cnt])
                    event[0].init(1)

                    for stage in range(n):
                        Tp.copy_async(A_smem[*r_smem], A[*r_gmem(stage)], event[0])
                        event[0].commit()
                        event[0].wait()

                        Tp.copy(B[*r_gmem(stage)], A_smem[*r_smem])
    # fmt: on
    np_dtype = tvm.testing.np_dtype_from_str(dtype)
    target = tvm.target.Target("cuda")

    with target:
        mod = tvm.IRModule({"main": copy_async})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirp")

        np.random.seed(0)
        A_np = tvm.testing.generate_random_array(dtype, g_shape)
        B_np = np.zeros(g_shape, dtype=np_dtype)

        A = tvm.nd.array(A_np, dev)
        B = tvm.nd.array(B_np, dev)
        mod(A, B)

        np.testing.assert_allclose(A_np, B.numpy())


@tvm.testing.requires_cuda_compute_version(9)
@pytest.mark.parametrize("swizzle_len", [3, 2, 1, 0])
@pytest.mark.parametrize(
    "dtype", ["int8", "uint8", "float8_e4m3fn", "float8_e5m2", "float16", "bfloat16", "float32"]
)
@pytest.mark.parametrize(
    "task",
    [
        (
            (3, 8, 256),  # global_shape
            (8, 256),  # shared_shape
            8,  # thread count per CTA
            TileLayout([3, 8, 256]),  # A_layout
            TileLayout([3, 8, 256]),  # B_layout
            lambda dtype, swizzle_len: tma_shared_layout(dtype, swizzle_len, (8, 256)),
        ),
        (
            (5, 64, 256),  # global_shape
            (64, 256),  # shared_shape
            64,  # thread count per CTA
            TileLayout([5, 64, 256]),  # A_layout
            TileLayout([5, 64, 256]),  # B_layout
            lambda dtype, swizzle_len: tma_shared_layout(dtype, swizzle_len, (64, 256)),
        ),
        (
            (7, 32, 512),  # global_shape
            (32, 512),  # shared_shape
            32,  # thread count per CTA
            TileLayout([7, 32, 512]),  # A_layout
            TileLayout([7, 32, 512]),  # B_layout
            lambda dtype, swizzle_len: (
                tma_atom_layout(dtype, swizzle_len)
                .tile_to((16, 256), tma_atom_shape(dtype, swizzle_len))
                .tile_to((32, 512), (16, 256))
                if swizzle_len > 0
                else None
            ),
        ),
    ],
)
def test_copy_s2g_tma_store(task, dtype, swizzle_len):
    g_shape, s_shape, thread_cnt, layoutA, layoutB, layoutS_fn = task
    dev = tvm.cuda(0)
    n = g_shape[0]

    # Compute the shared layout using the provided swizzle length
    shared_layout = layoutS_fn(dtype, swizzle_len)
    if shared_layout is None:
        # skip the test
        # TODO(@bohan): box_dim must have each dim less than or equal to 256
        return

    smem_bytes = functools.reduce(lambda acc, extent: acc * extent, s_shape, 1)
    smem_bytes = smem_bytes * tvm.DataType(dtype).bits // 8

    r_smem = [slice(0, s) for s in s_shape]

    def r_gmem(stage):
        return [
            slice(stage, stage + 1),
            *[slice(0, g_shape[i]) for i in range(1, len(g_shape))],
        ]

    # fmt: off
    @T.prim_func(tirp=True)
    def copy_async(A_ptr: T.handle, B_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, g_shape, dtype, layout=layoutA)
        B = T.match_buffer(B_ptr, g_shape, dtype, layout=layoutB)

        with T.kernel():
            bx = T.cta_id([1], parent="kernel")
            tx = T.thread_id([thread_cnt], parent="cta")

            with T.thread():
                dyn = T.alloc_buffer([smem_bytes], "uint8", scope="shared.dyn")
                A_smem = T.decl_buffer(s_shape, dtype, dyn.data, elem_offset=0, layout=shared_layout)

                with T.cta():
                    evt = Tp.alloc_bulk_group_event(EventImpl.kTMAStore)
                    evt.init()
                    for stage in range(n):
                        Tp.copy(A_smem[*r_smem], A[*r_gmem(stage)])
                        T.ptx.fence.proxy("shared")
                        Tp.copy_async(B[*r_gmem(stage)], A_smem[*r_smem], evt)
                        evt.commit()
                        evt.wait(0)
                        T.tvm_storage_sync("shared")
    # fmt: on
    np_dtype = tvm.testing.np_dtype_from_str(dtype)
    target = tvm.target.Target("cuda")

    with target:
        mod = tvm.IRModule({"main": copy_async})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirp")
        print(mod.mod.imports[0].inspect_source())

        np.random.seed(0)
        A_np = tvm.testing.generate_random_array(dtype, g_shape)
        B_np = np.zeros(g_shape, dtype=np_dtype)

        A = tvm.nd.array(A_np, dev)
        B = tvm.nd.array(B_np, dev)
        mod(A, B)

        np.testing.assert_allclose(A_np, B.numpy())


def test_kernel_sempaphore():
    # fmt: off
    @T.prim_func(tirp=True)
    def gpu_semaphore_wait(semaphore: T.handle):
        sem = T.match_buffer(semaphore, (64,), dtype="int32", scope="global")
        with T.kernel():
            bx = T.cta_id([128], parent="kernel")
            tx = T.thread_id([128], parent="cta")
            
            state = T.alloc_local((1,), "int32")
            evt = Tp.alloc_semaphore_event_tensor(EventImpl.kGlobalSemaphore, state=[sem, state], shape=[64])

            T.tvm_global_barrier_kinit()
            evt.init(1)
            T.tvm_storage_sync("global", True, 128)
            
            with T.cta()[0:64]:
                evt[bx].commit()
            with T.cta()[64:128]:
                evt[bx - 64].wait()
    # fmt: on

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": gpu_semaphore_wait})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirp")
        print(mod.mod.imports[0].inspect_source())
        semaphore = tvm.nd.array(np.full((64,), 2, dtype=np.int32), device=tvm.cuda(0))
        mod(semaphore)
        np.testing.assert_equal(semaphore.numpy(), np.full((64,), 0, dtype=np.int32))


if __name__ == "__main__":
    tvm.testing.main()
