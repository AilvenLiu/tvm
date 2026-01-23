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
from tvm.ir.type import PointerType, PrimType
from tvm.script import tir as T
from tvm.script import tirx as Tx
from tvm.tir.layout import TileLayout
from tvm.tirx.op_schedule.cuda.copy_async import (
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
    @T.prim_func(tirx=True)
    def copy_async(A_ptr: T.handle, B_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, g_shape, dtype, layout=layoutA)
        B = T.match_buffer(B_ptr, g_shape, dtype, layout=layoutB)

        with T.kernel():
            bx = T.cta_id([1], parent="kernel")
            tx = T.thread_id([thread_cnt], parent="cta")

            with T.cta():
                A_smem = T.alloc_buffer(s_shape, dtype, scope="shared", layout=layoutS)

                Tx.copy_async(A_smem[*r_smem], A[*r_gmem], dispatch="non-bulk-copy")
                T.ptx.cp_async.commit_group()
                T.ptx.cp_async.wait_group()
                T.cuda.cta_sync()
                Tx.copy(B[*r_gmem], A_smem[*r_smem])
    # fmt: on

    np_dtype = tvm.testing.np_dtype_from_str(dtype)
    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": copy_async})
        mod = tvm.tir.transform.LowerTIRx()(mod)
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")

        np.random.seed(0)
        A_np = np.random.rand(*g_shape).astype(np_dtype)
        B_np = np.zeros(g_shape, dtype=np_dtype)

        A = tvm.runtime.tensor(A_np, dev)
        B = tvm.runtime.tensor(B_np, dev)
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
        pytest.skip(f"dtype {dtype} + swizzle_len {swizzle_len} is not supported")

    total_bytes = functools.reduce(lambda acc, region: acc * (region[1] - region[0]), s_region, 1)
    total_bytes = total_bytes * tvm.DataType(dtype).bits // 8

    smem_bytes = functools.reduce(lambda acc, extent: acc * extent, s_shape, 1)
    smem_bytes = smem_bytes * tvm.DataType(dtype).bits // 8

    r_smem = [slice(s_region[i][0], s_region[i][1]) for i in range(len(s_shape))]
    r_gmem = [slice(g_region[i][0], g_region[i][1]) for i in range(len(g_shape))]

    # fmt: off
    @T.prim_func(tirx=True)
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
                mbarrier = T.decl_buffer([1], "uint64", dyn.data, byte_offset=smem_bytes // 8)

                mbar_ptr = mbarrier.ptr_to([0])
                with T.thread()[0:1]:
                    T.ptx.mbarrier.init(mbar_ptr, 1)
                T.ptx.fence.proxy("shared")
                T.cuda.cta_sync()

                with T.thread()[0:1]:
                    Tx.copy_async(A_smem[*r_smem], A[*r_gmem], dispatch="tma", mbar=mbar_ptr, cache_hint=cache_hint)
                    T.ptx.mbarrier.arrive.expect_tx(mbar_ptr, total_bytes)

                T.ptx.mbarrier.try_wait(mbar_ptr, 0)
                T.cuda.cta_sync()
                with T.cta():
                    Tx.copy(B[*r_gmem], A_smem[*r_smem])
    # fmt: on
    np_dtype = tvm.testing.np_dtype_from_str(dtype)
    target = tvm.target.Target("cuda")

    with target:
        mod = tvm.IRModule({"main": copy_async})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
        print(mod.mod.imports[0].inspect_source())

        np.random.seed(0)
        A_np = tvm.testing.generate_random_array(dtype, g_shape)
        B_np = np.zeros(g_shape, dtype=np_dtype)

        A = tvm.runtime.tensor(A_np, dev)
        B = tvm.runtime.tensor(B_np, dev)
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
        pytest.skip(f"dtype {dtype} + swizzle_len {swizzle_len} is not supported")

    smem_bytes = functools.reduce(lambda acc, extent: acc * extent, s_shape, 1)
    smem_bytes = smem_bytes * tvm.DataType(dtype).bits // 8

    r_smem = [slice(0, s) for s in s_shape]

    def r_gmem(stage):
        return [
            slice(stage, stage + 1),
            *[slice(0, g_shape[i]) for i in range(1, len(g_shape))],
        ]

    # fmt: off
    @T.prim_func(tirx=True)
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
                mbarrier = T.decl_buffer([1], "uint64", dyn.data, elem_offset=smem_bytes // 8)
                phase = T.local_cell("int32")

                phase = 0
                with T.thread()[0:1]:
                    T.ptx.mbarrier.init(mbarrier.ptr_to([0]), 1)
                T.ptx.fence.proxy("shared")
                T.cuda.cta_sync()

                for stage in range(n):
                    with T.thread()[0:1]:
                        Tx.copy_async(A_smem[*r_smem], A[*r_gmem(stage)], dispatch="tma", mbar=mbarrier.ptr_to([0]))
                        T.ptx.mbarrier.arrive.expect_tx(mbarrier.ptr_to([0]), smem_bytes)

                    T.ptx.mbarrier.try_wait(mbarrier.ptr_to([0]), phase)
                    phase = phase ^ 1

                    T.ptx.fence.proxy("shared")
                    T.cuda.cta_sync()
                    with T.cta():
                        Tx.copy(B[*r_gmem(stage)], A_smem[*r_smem])
    # fmt: on
    np_dtype = tvm.testing.np_dtype_from_str(dtype)
    target = tvm.target.Target("cuda")

    with target:
        mod = tvm.IRModule({"main": copy_async})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
        print(mod.mod.imports[0].inspect_source())

        np.random.seed(0)
        A_np = tvm.testing.generate_random_array(dtype, g_shape)
        B_np = np.zeros(g_shape, dtype=np_dtype)

        A = tvm.runtime.tensor(A_np, dev)
        B = tvm.runtime.tensor(B_np, dev)
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
@pytest.mark.parametrize("cache_hint", ["evict_last", ""])
def test_copy_s2g_tma_store(task, dtype, swizzle_len, cache_hint):
    g_shape, s_shape, thread_cnt, layoutA, layoutB, layoutS_fn = task
    dev = tvm.cuda(0)
    n = g_shape[0]

    # Compute the shared layout using the provided swizzle length
    shared_layout = layoutS_fn(dtype, swizzle_len)
    if shared_layout is None:
        pytest.skip(f"dtype {dtype} + swizzle_len {swizzle_len} is not supported")

    smem_bytes = functools.reduce(lambda acc, extent: acc * extent, s_shape, 1)
    smem_bytes = smem_bytes * tvm.DataType(dtype).bits // 8

    r_smem = [slice(0, s) for s in s_shape]

    def r_gmem(stage):
        return [
            slice(stage, stage + 1),
            *[slice(0, g_shape[i]) for i in range(1, len(g_shape))],
        ]

    # fmt: off
    @T.prim_func(tirx=True)
    def copy_async(A_ptr: T.handle, B_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, g_shape, dtype, layout=layoutA)
        B = T.match_buffer(B_ptr, g_shape, dtype, layout=layoutB)

        with T.kernel():
            bx = T.cta_id([1], parent="kernel")
            tx = T.thread_id([thread_cnt], parent="cta")

            with T.thread():
                dyn = T.alloc_buffer([smem_bytes], "uint8", scope="shared.dyn")
                A_smem = T.decl_buffer(s_shape, dtype, dyn.data, elem_offset=0, layout=shared_layout)

                for stage in range(n):
                    Tx.copy(A_smem[*r_smem], A[*r_gmem(stage)])
                    T.ptx.fence.proxy("shared")
                    with T.thread()[0:1]:
                        Tx.copy_async(B[*r_gmem(stage)], A_smem[*r_smem], dispatch="tma", cache_hint=cache_hint)
                        T.ptx.cp_async.bulk.commit_group()
                        T.ptx.cp_async.bulk.wait_group()
                    T.cuda.cta_sync()
    # fmt: on
    np_dtype = tvm.testing.np_dtype_from_str(dtype)
    target = tvm.target.Target("cuda")

    with target:
        mod = tvm.IRModule({"main": copy_async})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
        print(mod.mod.imports[0].inspect_source())

        np.random.seed(0)
        A_np = tvm.testing.generate_random_array(dtype, g_shape)
        B_np = np.zeros(g_shape, dtype=np_dtype)

        A = tvm.runtime.tensor(A_np, dev)
        B = tvm.runtime.tensor(B_np, dev)
        mod(A, B)

        np.testing.assert_allclose(A_np, B.numpy())


@tvm.testing.requires_cuda_compute_version(9)
@pytest.mark.parametrize(
    "task",
    [
        (
            (128, 64),  # global_shape
            ((0, 128), (0, 64)),  # global_region
            (2, 2, 128, 64),  # shared_shape
            ((0, 1), (0, 1), (0, 128), (0, 64)),  # shared_region
            128,  # thread count per CTA
            TileLayout([128, 64]).canonicalize(),  # A_layout
            TileLayout([128, 64]).canonicalize(),  # B_layout
            lambda dtype, swizzle_len: tma_shared_layout(
                dtype, swizzle_len, (2, 2, 128, 64)
            ).canonicalize(),
        ),
        (
            (128, 64),  # global_shape
            ((64, 64 + 24), (0, 64)),  # global_region
            (2, 2, 24, 64),  # shared_shape
            ((0, 1), (0, 1), (0, 24), (0, 64)),  # shared_region
            128,  # thread count per CTA
            TileLayout([128, 64]).canonicalize(),  # A_layout
            TileLayout([128, 64]).canonicalize(),  # B_layout
            lambda dtype, swizzle_len: tma_shared_layout(
                dtype, swizzle_len, (2, 2, 24, 64)
            ).canonicalize(),
        ),
        (
            (256, 64),  # global_shape
            ((128, 256), (0, 64)),  # global_region
            (256, 64),  # shared_shape
            ((0, 128), (0, 64)),  # shared_region
            128,  # thread count per CTA
            TileLayout([256, 64]).canonicalize(),  # A_layout
            TileLayout([256, 64]).canonicalize(),  # B_layout
            lambda dtype, swizzle_len: tma_shared_layout(
                dtype, swizzle_len, (256, 64)
            ).canonicalize(),
        ),
    ],
)
def test_copy_g2s_cta_tma_load_edge_case(task, dtype="float16", swizzle_len=3):
    g_shape, g_region, s_shape, s_region, thread_cnt, layoutA, layoutB, layoutS_fn = task
    dev = tvm.cuda(0)

    # Compute the shared layout using the provided swizzle length
    shared_layout = layoutS_fn(dtype, swizzle_len)

    total_bytes = functools.reduce(lambda acc, region: acc * (region[1] - region[0]), s_region, 1)
    total_bytes = total_bytes * tvm.DataType(dtype).bits // 8

    smem_bytes = functools.reduce(lambda acc, extent: acc * extent, s_shape, 1)
    smem_bytes = smem_bytes * tvm.DataType(dtype).bits // 8

    r_smem = [slice(s_region[i][0], s_region[i][1]) for i in range(len(s_shape))]
    r_gmem = [slice(g_region[i][0], g_region[i][1]) for i in range(len(g_shape))]

    # fmt: off
    @T.prim_func(tirx=True)
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
                mbarrier = T.decl_buffer([1], "uint64", dyn.data, elem_offset=smem_bytes // 8)
                mbar_ptr = T.meta_var(mbarrier.ptr_to([0]))

                with T.thread()[0:1]:
                    T.ptx.mbarrier.init(mbar_ptr, 1)
                T.ptx.fence.proxy("shared")
                T.cuda.cta_sync()

                with T.thread()[0:1]:
                    Tx.copy_async(A_smem[*r_smem], A[*r_gmem], dispatch="tma", mbar=mbar_ptr)
                    T.ptx.mbarrier.arrive.expect_tx(mbar_ptr, total_bytes)
                T.ptx.mbarrier.try_wait(mbar_ptr, 0)

                T.ptx.fence.proxy("shared")
                T.cuda.cta_sync()
                with T.cta():
                    Tx.copy(B[*r_gmem], A_smem[*r_smem])
    # fmt: on
    np_dtype = tvm.testing.np_dtype_from_str(dtype)
    target = tvm.target.Target("cuda")

    with target:
        mod = tvm.IRModule({"main": copy_async})
        mod.show()
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
        print(mod.mod.imports[0].inspect_source())

        np.random.seed(0)
        A_np = tvm.testing.generate_random_array(dtype, g_shape)
        B_np = np.zeros(g_shape, dtype=np_dtype)

        A = tvm.runtime.tensor(A_np, dev)
        B = tvm.runtime.tensor(B_np, dev)
        mod(A, B)

        B_ref = np.zeros(g_shape, dtype=np_dtype)
        B_ref[*r_gmem] = A_np[*r_gmem]
        np.testing.assert_allclose(B_ref, B.numpy())


@tvm.testing.requires_cuda_compute_version(10)
@pytest.mark.parametrize(
    "task",
    [
        # 4D TMA copy mimicking FA4 scenario where g_ext has unit dimension
        # but atom_shape has non-unit in that dimension.
        # Global: (batch, seq, heads, dim) - copy one head at a time
        # Shared: (pipe, blk, seq, dim) - different axis arrangement
        # g_ext[2] = 1 (one head), but atom_global[2] = 8
        # This tests box_dim = [min(a, e) for a, e in zip(atom_shape_global, g_ext)]
        (
            (2, 128, 8, 64),  # global_shape: (batch, seq, heads, dim)
            ((0, 1), (0, 128), (0, 1), (0, 64)),  # global_region: copy 1 batch, 128 seq, 1 head, 64 dim
            (1, 1, 128, 64),  # shared_shape: (pipe_stage, blk_k, blk_m, blk_k)
            ((0, 1), (0, 1), (0, 128), (0, 64)),  # shared_region
            128,  # thread count
            TileLayout([2, 128, 8, 64]).canonicalize(),  # A_layout (global)
            TileLayout([2, 128, 8, 64]).canonicalize(),  # B_layout (global)
            lambda dtype, swizzle_len: tma_shared_layout(
                dtype, swizzle_len, (1, 1, 128, 64)
            ).canonicalize(),
        ),
        # Another variant with different head/batch sizes
        (
            (4, 64, 4, 128),  # global_shape
            ((0, 1), (0, 64), (0, 1), (0, 128)),  # global_region: g_ext = [1, 64, 1, 128]
            (1, 1, 64, 128),  # shared_shape
            ((0, 1), (0, 1), (0, 64), (0, 128)),  # shared_region
            128,  # thread count
            TileLayout([4, 64, 4, 128]).canonicalize(),
            TileLayout([4, 64, 4, 128]).canonicalize(),
            lambda dtype, swizzle_len: tma_shared_layout(
                dtype, swizzle_len, (1, 1, 64, 128)
            ).canonicalize(),
        ),
    ],
)
def test_copy_g2s_tma_4d_axis_reorder(task, dtype="float16", swizzle_len=3):
    """Test 4D TMA copy with axis reordering where g_ext < atom_shape in unit dimensions.

    This mimics the FA4 kernel scenario where:
    - Global tensor is (batch, seq, heads, dim)
    - We copy one head at a time: g_ext = [1, seq_len, 1, head_dim]
    - atom_shape_global = [1, 1, 8, 64] for 4D
    - g_ext[2] = 1 < atom_shape[2] = 8

    Without the fix (box_dim = [min(a, e) for a, e in zip(atom_shape_global, g_ext)]),
    box_dim[2] = 8 while g_ext[2] = 1, causing g_ext[2] // box_dim[2] = 0 iterations,
    which hangs on mbarrier wait.
    """
    g_shape, g_region, s_shape, s_region, thread_cnt, layoutA, layoutB, layoutS_fn = task
    dev = tvm.cuda(0)

    # Compute the shared layout using the provided swizzle length
    shared_layout = layoutS_fn(dtype, swizzle_len)

    total_bytes = functools.reduce(lambda acc, region: acc * (region[1] - region[0]), s_region, 1)
    total_bytes = total_bytes * tvm.DataType(dtype).bits // 8

    smem_bytes = functools.reduce(lambda acc, extent: acc * extent, s_shape, 1)
    smem_bytes = smem_bytes * tvm.DataType(dtype).bits // 8

    r_smem = [slice(s_region[i][0], s_region[i][1]) for i in range(len(s_shape))]
    r_gmem = [slice(g_region[i][0], g_region[i][1]) for i in range(len(g_shape))]

    # fmt: off
    @T.prim_func(tirx=True)
    def copy_async(A_ptr: T.handle, B_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, g_shape, dtype, layout=layoutA)
        B = T.match_buffer(B_ptr, g_shape, dtype, layout=layoutB)

        with T.kernel():
            bx = T.cta_id([1], parent="kernel")
            tx = T.thread_id([thread_cnt], parent="cta")

            with T.thread():
                dyn = T.alloc_buffer([smem_bytes + 8], "uint8", scope="shared.dyn")
                A_smem = T.decl_buffer(s_shape, dtype, dyn.data, elem_offset=0, layout=shared_layout)
                mbarrier = T.decl_buffer([1], "uint64", dyn.data, elem_offset=smem_bytes // 8)
                mbar_ptr = T.meta_var(mbarrier.ptr_to([0]))

                with T.thread()[0:1]:
                    T.ptx.mbarrier.init(mbar_ptr, 1)
                T.ptx.fence.proxy("shared")
                T.cuda.cta_sync()

                with T.thread()[0:1]:
                    Tx.copy_async(A_smem[*r_smem], A[*r_gmem], dispatch="tma", mbar=mbar_ptr)
                    T.ptx.mbarrier.arrive.expect_tx(mbar_ptr, total_bytes)
                T.ptx.mbarrier.try_wait(mbar_ptr, 0)

                T.ptx.fence.proxy("shared")
                T.cuda.cta_sync()
                with T.cta():
                    Tx.copy(B[*r_gmem], A_smem[*r_smem])
    # fmt: on
    np_dtype = tvm.testing.np_dtype_from_str(dtype)
    target = tvm.target.Target("cuda")

    with target:
        mod = tvm.IRModule({"main": copy_async})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")

        np.random.seed(0)
        A_np = tvm.testing.generate_random_array(dtype, g_shape)
        B_np = np.zeros(g_shape, dtype=np_dtype)

        A = tvm.runtime.tensor(A_np, dev)
        B = tvm.runtime.tensor(B_np, dev)
        mod(A, B)

        B_ref = np.zeros(g_shape, dtype=np_dtype)
        B_ref[*r_gmem] = A_np[*r_gmem]
        np.testing.assert_allclose(B_ref, B.numpy())


@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("width_32b", [4, 8, 16, 32])
def test_copy_tmem2reg_async(dtype, width_32b):
    """Test async tmem<->local copy using copy_async instead of copy.

    This tests the new copy_async dispatch for tmem<->local that doesn't
    immediately wait after the operation, allowing for pipelining.
    """

    def next_power_of_2(x):
        """Return the smallest power of 2 greater than or equal to x."""
        if x <= 1:
            return 1
        return 1 << (x - 1).bit_length()

    bits = tvm.runtime.DataType(dtype).bits
    if 128 % bits != 0 or 32 % bits != 0:
        pytest.skip(f"dtype {dtype} is not supported")

    WIDTH = width_32b * (32 // bits)
    VEC_LEN = 128 // bits
    if WIDTH % VEC_LEN != 0:
        pytest.skip(f"dtype {dtype} + width {width_32b} is not supported")

    g_layout = TileLayout(shard=([128, WIDTH // VEC_LEN, VEC_LEN], [WIDTH, VEC_LEN, 1]))
    local_view = TileLayout(shard=([128, WIDTH], [(1, "tid_in_wg"), (1, "m")]))

    # fmt: off
    @T.prim_func(tirx=True)
    def copy_async_test(A_ptr: T.handle, B_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, (128, WIDTH), dtype)
        B = T.match_buffer(B_ptr, (128, WIDTH), dtype)

        A_flat = A.view(-1)
        B_flat = B.view(-1)

        with T.kernel():
            bx = T.cta_id([1], parent="kernel")
            wg_id = T.warpgroup_id([1], parent="cta")
            warp_id = T.warp_id([4], parent="warpgroup")
            lane_id = T.thread_id([32], parent="warp")
            tid_in_wg = T.thread_id([128], parent="cta")

            tmem_addr = T.alloc_shared([1], "uint32")

            with T.warpgroup()[0:1]:
                with T.warp()[0:1]:
                    T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=max(32, next_power_of_2(width_32b)), cta_group=1)

                T.tvm_storage_sync("shared")

                tmem = T.decl_buffer((128, WIDTH), dtype, scope="tmem", allocated_addr=tmem_addr[0],
                                     layout=TileLayout(([128, WIDTH], [(1, "TLane"), (1, "TCol")])))

                A_reg = T.alloc_local((WIDTH), dtype)
                B_reg = T.alloc_local((WIDTH), dtype)
                A_local = A_reg.view(128, WIDTH, layout=local_view)
                B_local = B_reg.view(128, WIDTH, layout=local_view)

                # A -> A_local
                with T.thread():
                    for i in range(WIDTH // VEC_LEN):
                        g_offset = T.meta_var(g_layout.apply(tid_in_wg, i, 0)["m"])
                        Tx.copy(A_reg[i * VEC_LEN: i * VEC_LEN + VEC_LEN], A_flat[g_offset: g_offset + VEC_LEN])
                    for i in range(WIDTH):
                        B_reg[i] = T.cast(0, dtype)
                T.cuda.cta_sync()

                # A_local -> tmem (async)
                Tx.copy_async(tmem[:, :], A_local[:, :])
                T.ptx.tcgen05.wait.st()  # explicit wait
                T.cuda.cta_sync()

                # tmem -> B_local (async)
                Tx.copy_async(B_local[:, :], tmem[:, :])
                T.ptx.tcgen05.wait.ld()  # explicit wait
                T.cuda.cta_sync()

                # B_local -> B
                with T.thread():
                    for i in range(WIDTH // VEC_LEN):
                        g_offset = T.meta_var(g_layout.apply(tid_in_wg, i, 0)["m"])
                        Tx.copy(B_flat[g_offset: g_offset + VEC_LEN], B_reg[i * VEC_LEN: i * VEC_LEN + VEC_LEN])

                with T.warp()[0:1]:
                    T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
                    T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=max(32, next_power_of_2(width_32b)), cta_group=1)
    # fmt: on

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": copy_async_test})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
        A_np = tvm.testing.generate_random_array(dtype, (128, WIDTH))
        B_np = np.zeros((128, WIDTH), dtype=dtype)
        DEV = tvm.cuda(0)
        A = tvm.runtime.tensor(A_np, DEV)
        B = tvm.runtime.tensor(B_np, DEV)
        mod(A, B)
        np.testing.assert_allclose(B.numpy(), A_np)


@tvm.testing.requires_cuda_compute_version(9)
@pytest.mark.parametrize(
    "task",
    [
        # Test case: 2D global -> 3D shared with partial copy region
        # This tests that enlarge_factor is computed based on copy extent, not full buffer extent.
        # Bug scenario: shared buffer is (6, 128, 64), but we only copy (1, 32, 64).
        # Without fix: enlarge_factor = 16 (128/8), new_box_dim = 128 > g_ext[0]=32, no enlargement
        # With fix: enlarge_factor = min(16, 32/8) = 4, new_box_dim = 32, single TMA call
        (
            (128, 256),  # global_shape: 2D
            ((0, 32), (0, 64)),  # global_region: copy 32x64
            (6, 128, 64),  # shared_shape: 3D with extra pipeline dimension
            ((0, 1), (0, 32), (0, 64)),  # shared_region: unit extent in first dim
            128,  # thread count
            TileLayout([128, 256]).canonicalize(),  # A_layout (global)
            TileLayout([128, 256]).canonicalize(),  # B_layout (global)
            lambda dtype, swizzle_len: tma_shared_layout(
                dtype, swizzle_len, (6, 128, 64)
            ).canonicalize(),
        ),
        # Another variant: different copy extent (64 rows instead of 32)
        (
            (256, 512),  # global_shape: 2D
            ((0, 64), (0, 64)),  # global_region: copy 64x64
            (4, 256, 64),  # shared_shape: 3D
            ((1, 2), (0, 64), (0, 64)),  # shared_region: different slice index
            128,  # thread count
            TileLayout([256, 512]).canonicalize(),
            TileLayout([256, 512]).canonicalize(),
            lambda dtype, swizzle_len: tma_shared_layout(
                dtype, swizzle_len, (4, 256, 64)
            ).canonicalize(),
        ),
    ],
)
def test_copy_g2s_tma_partial_region_3d_shared(task, dtype="float16", swizzle_len=3):
    """Test TMA copy from 2D global to 3D shared with partial copy region.

    This tests the fix for a bug where enlarge_factor was computed based on
    the full shared buffer extent rather than the actual copy region extent.
    This caused unnecessary multiple TMA instructions when a single one would suffice.

    For example, with shared buffer shape (6, 128, 64) and copy region (1, 32, 64):
    - atom_shape = [1, 8, 64]
    - Old behavior: enlarge_factor = 128/8 = 16, new_box_dim = 128 > 32, no enlargement
      -> box_dim = [8, 64], needs 32/8 = 4 TMA calls
    - Fixed behavior: enlarge_factor = min(16, 32/8) = 4, new_box_dim = 32
      -> box_dim = [32, 64], needs only 1 TMA call
    """
    g_shape, g_region, s_shape, s_region, thread_cnt, layoutA, layoutB, layoutS_fn = task
    dev = tvm.cuda(0)

    shared_layout = layoutS_fn(dtype, swizzle_len)
    if shared_layout is None:
        pytest.skip(f"dtype {dtype} + swizzle_len {swizzle_len} is not supported")

    total_bytes = functools.reduce(lambda acc, region: acc * (region[1] - region[0]), s_region, 1)
    total_bytes = total_bytes * tvm.DataType(dtype).bits // 8

    smem_bytes = functools.reduce(lambda acc, extent: acc * extent, s_shape, 1)
    smem_bytes = smem_bytes * tvm.DataType(dtype).bits // 8

    r_smem = [slice(s_region[i][0], s_region[i][1]) for i in range(len(s_shape))]
    r_gmem = [slice(g_region[i][0], g_region[i][1]) for i in range(len(g_shape))]

    # fmt: off
    @T.prim_func(tirx=True)
    def copy_async(A_ptr: T.handle, B_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, g_shape, dtype, layout=layoutA)
        B = T.match_buffer(B_ptr, g_shape, dtype, layout=layoutB)

        with T.kernel():
            bx = T.cta_id([1], parent="kernel")
            tx = T.thread_id([thread_cnt], parent="cta")

            with T.thread():
                dyn = T.alloc_buffer([smem_bytes + 64], "uint8", scope="shared.dyn")
                A_smem = T.decl_buffer(s_shape, dtype, dyn.data, elem_offset=0, layout=shared_layout)
                mbarrier = T.decl_buffer([1], "uint64", dyn.data, elem_offset=smem_bytes // 8)
                mbar_ptr = T.meta_var(mbarrier.ptr_to([0]))

                with T.thread()[0:1]:
                    T.ptx.mbarrier.init(mbar_ptr, 1)
                T.ptx.fence.proxy("shared")
                T.cuda.cta_sync()

                with T.thread()[0:1]:
                    Tx.copy_async(A_smem[*r_smem], A[*r_gmem], dispatch="tma", mbar=mbar_ptr)
                    T.ptx.mbarrier.arrive.expect_tx(mbar_ptr, total_bytes)

                T.ptx.mbarrier.try_wait(mbar_ptr, 0)
                T.ptx.fence.proxy("shared")
                T.cuda.cta_sync()

                with T.cta():
                    Tx.copy(B[*r_gmem], A_smem[*r_smem])
    # fmt: on

    np_dtype = tvm.testing.np_dtype_from_str(dtype)
    target = tvm.target.Target("cuda")

    with target:
        mod = tvm.IRModule({"main": copy_async})

        # Verify that LowerTIRx generates exactly 1 TMA instruction
        lowered = tvm.tir.transform.LowerTIRx()(mod)
        lowered_str = str(lowered)

        # Check there's exactly one g2c call in the IR
        assert lowered_str.count("cp_async.bulk.tensor.g2c") == 1, (
            "Expected exactly 1 cp_async.bulk.tensor.g2c call in lowered IR"
        )

        # Verify the loop has only 1 iteration (T.grid(1, 1) means 1x1=1 TMA call)
        # This ensures the fix is working - without fix it would be T.grid(4, 1) or similar
        import re
        grid_match = re.search(r'T\.grid\((\d+),\s*(\d+)\)', lowered_str)
        if grid_match:
            iters_0 = int(grid_match.group(1))
            iters_1 = int(grid_match.group(2))
            total_iters = iters_0 * iters_1
            assert total_iters == 1, (
                f"Expected 1 TMA iteration, got {iters_0}x{iters_1}={total_iters}. "
                "This indicates enlarge_factor was not computed correctly based on copy extent."
            )

        # Now compile and verify correctness
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")

        np.random.seed(0)
        A_np = tvm.testing.generate_random_array(dtype, g_shape)
        B_np = np.zeros(g_shape, dtype=np_dtype)

        A = tvm.runtime.tensor(A_np, dev)
        B = tvm.runtime.tensor(B_np, dev)
        mod(A, B)

        B_ref = np.zeros(g_shape, dtype=np_dtype)
        B_ref[*r_gmem] = A_np[*r_gmem]
        np.testing.assert_allclose(B_ref, B.numpy())


if __name__ == "__main__":
    tvm.testing.main()
