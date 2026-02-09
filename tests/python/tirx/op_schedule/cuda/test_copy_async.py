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
from tvm.script import tirx as Tx
from tvm.tir.layout import TileLayout
from tvm.tir.stmt_functor import StmtExprVisitor
from tvm.tirx.op_schedule.cuda.copy_async import (
    tma_atom_layout,
    tma_atom_shape,
    tma_shared_layout,
)


class TMACounter(StmtExprVisitor):
    """Visitor to count total TMA operations including loop iterations.

    This verifies that TMA copy operations are optimized correctly,
    resulting in minimal TMA instructions instead of multiple iterations.
    """

    def __init__(self):
        super().__init__()
        self.loop_extents = []  # Stack of loop extents
        self.total_tma_ops = 0

    def visit_for_(self, op):
        extent = op.extent
        self.loop_extents.append(extent)
        self.visit_stmt(op.body)
        self.loop_extents.pop()

    def visit_evaluate_(self, op):
        if isinstance(op.value, tvm.tir.Call):
            if op.value.op.name == "tir.ptx_cp_async_bulk_tensor_global_to_cluster":
                # Multiply all enclosing loop extents
                iters = 1
                for ext in self.loop_extents:
                    iters *= ext
                self.total_tma_ops += iters


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
    @Tx.prim_func(tirx=True)
    def copy_async(A_ptr: Tx.handle, B_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, g_shape, dtype, layout=layoutA)
        B = Tx.match_buffer(B_ptr, g_shape, dtype, layout=layoutB)

        with Tx.kernel():
            bx = Tx.cta_id([1], parent="kernel")
            tx = Tx.thread_id([thread_cnt], parent="cta")

            with Tx.cta():
                A_smem = Tx.alloc_buffer(s_shape, dtype, scope="shared", layout=layoutS)

                Tx.copy_async(A_smem[*r_smem], A[*r_gmem], dispatch="non-bulk-copy")
                Tx.ptx.cp_async.commit_group()
                Tx.ptx.cp_async.wait_group()
                Tx.cuda.cta_sync()
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
    @Tx.prim_func(tirx=True)
    def copy_async(A_ptr: Tx.handle, B_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, g_shape, dtype, layout=layoutA)
        B = Tx.match_buffer(B_ptr, g_shape, dtype, layout=layoutB)

        with Tx.kernel():
            bx = Tx.cta_id([1], parent="kernel")
            tx = Tx.thread_id([thread_cnt], parent="cta")

            # Allocate shared memory and mbarrier in a unified buffer
            with Tx.thread():
                dyn = Tx.alloc_buffer([smem_bytes + 8], "uint8", scope="shared.dyn")
                A_smem = Tx.decl_buffer(s_shape, dtype, dyn.data, elem_offset=0, layout=shared_layout)
                mbarrier = Tx.decl_buffer([1], "uint64", dyn.data, byte_offset=smem_bytes // 8)

                mbar_ptr = mbarrier.ptr_to([0])
                with Tx.thread()[0:1]:
                    Tx.ptx.mbarrier.init(mbar_ptr, 1)
                Tx.ptx.fence.proxy("shared")
                Tx.cuda.cta_sync()

                with Tx.thread()[0:1]:
                    Tx.copy_async(A_smem[*r_smem], A[*r_gmem], dispatch="tma", mbar=mbar_ptr, cache_hint=cache_hint)
                    Tx.ptx.mbarrier.arrive.expect_tx(mbar_ptr, total_bytes)

                Tx.ptx.mbarrier.try_wait(mbar_ptr, 0)
                Tx.cuda.cta_sync()
                with Tx.cta():
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
    @Tx.prim_func(tirx=True)
    def copy_async(A_ptr: Tx.handle, B_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, g_shape, dtype, layout=layoutA)
        B = Tx.match_buffer(B_ptr, g_shape, dtype, layout=layoutB)

        with Tx.kernel():
            bx = Tx.cta_id([1], parent="kernel")
            tx = Tx.thread_id([thread_cnt], parent="cta")

            # Allocate shared memory and mbarrier in a unified buffer
            with Tx.thread():
                dyn = Tx.alloc_buffer([smem_bytes + 8], "uint8", scope="shared.dyn")
                A_smem = Tx.decl_buffer(s_shape, dtype, dyn.data, elem_offset=0, layout=shared_layout)
                mbarrier = Tx.decl_buffer([1], "uint64", dyn.data, elem_offset=smem_bytes // 8)
                phase = Tx.local_cell("int32")

                phase = 0
                with Tx.thread()[0:1]:
                    Tx.ptx.mbarrier.init(mbarrier.ptr_to([0]), 1)
                Tx.ptx.fence.proxy("shared")
                Tx.cuda.cta_sync()

                for stage in range(n):
                    with Tx.thread()[0:1]:
                        Tx.copy_async(A_smem[*r_smem], A[*r_gmem(stage)], dispatch="tma", mbar=mbarrier.ptr_to([0]))
                        Tx.ptx.mbarrier.arrive.expect_tx(mbarrier.ptr_to([0]), smem_bytes)

                    Tx.ptx.mbarrier.try_wait(mbarrier.ptr_to([0]), phase)
                    phase = phase ^ 1

                    Tx.ptx.fence.proxy("shared")
                    Tx.cuda.cta_sync()
                    with Tx.cta():
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
    @Tx.prim_func(tirx=True)
    def copy_async(A_ptr: Tx.handle, B_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, g_shape, dtype, layout=layoutA)
        B = Tx.match_buffer(B_ptr, g_shape, dtype, layout=layoutB)

        with Tx.kernel():
            bx = Tx.cta_id([1], parent="kernel")
            tx = Tx.thread_id([thread_cnt], parent="cta")

            with Tx.thread():
                dyn = Tx.alloc_buffer([smem_bytes], "uint8", scope="shared.dyn")
                A_smem = Tx.decl_buffer(s_shape, dtype, dyn.data, elem_offset=0, layout=shared_layout)

                for stage in range(n):
                    Tx.copy(A_smem[*r_smem], A[*r_gmem(stage)])
                    Tx.ptx.fence.proxy("shared")
                    with Tx.thread()[0:1]:
                        Tx.copy_async(B[*r_gmem(stage)], A_smem[*r_smem], dispatch="tma", cache_hint=cache_hint)
                        Tx.ptx.cp_async.bulk.commit_group()
                        Tx.ptx.cp_async.bulk.wait_group()
                    Tx.cuda.cta_sync()
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
    @Tx.prim_func(tirx=True)
    def copy_async(A_ptr: Tx.handle, B_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, g_shape, dtype, layout=layoutA)
        B = Tx.match_buffer(B_ptr, g_shape, dtype, layout=layoutB)

        with Tx.kernel():
            bx = Tx.cta_id([1], parent="kernel")
            tx = Tx.thread_id([thread_cnt], parent="cta")

            # Allocate shared memory and mbarrier in a unified buffer
            with Tx.thread():
                dyn = Tx.alloc_buffer([smem_bytes + 8], "uint8", scope="shared.dyn")
                A_smem = Tx.decl_buffer(s_shape, dtype, dyn.data, elem_offset=0, layout=shared_layout)
                mbarrier = Tx.decl_buffer([1], "uint64", dyn.data, elem_offset=smem_bytes // 8)
                mbar_ptr = Tx.meta_var(mbarrier.ptr_to([0]))

                with Tx.thread()[0:1]:
                    Tx.ptx.mbarrier.init(mbar_ptr, 1)
                Tx.ptx.fence.proxy("shared")
                Tx.cuda.cta_sync()

                with Tx.thread()[0:1]:
                    Tx.copy_async(A_smem[*r_smem], A[*r_gmem], dispatch="tma", mbar=mbar_ptr)
                    Tx.ptx.mbarrier.arrive.expect_tx(mbar_ptr, total_bytes)
                Tx.ptx.mbarrier.try_wait(mbar_ptr, 0)

                Tx.ptx.fence.proxy("shared")
                Tx.cuda.cta_sync()
                with Tx.cta():
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
            (
                (0, 1),
                (0, 128),
                (0, 1),
                (0, 64),
            ),  # global_region: copy 1 batch, 128 seq, 1 head, 64 dim
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
    @Tx.prim_func(tirx=True)
    def copy_async(A_ptr: Tx.handle, B_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, g_shape, dtype, layout=layoutA)
        B = Tx.match_buffer(B_ptr, g_shape, dtype, layout=layoutB)

        with Tx.kernel():
            bx = Tx.cta_id([1], parent="kernel")
            tx = Tx.thread_id([thread_cnt], parent="cta")

            with Tx.thread():
                dyn = Tx.alloc_buffer([smem_bytes + 8], "uint8", scope="shared.dyn")
                A_smem = Tx.decl_buffer(s_shape, dtype, dyn.data, elem_offset=0, layout=shared_layout)
                mbarrier = Tx.decl_buffer([1], "uint64", dyn.data, elem_offset=smem_bytes // 8)
                mbar_ptr = Tx.meta_var(mbarrier.ptr_to([0]))

                with Tx.thread()[0:1]:
                    Tx.ptx.mbarrier.init(mbar_ptr, 1)
                Tx.ptx.fence.proxy("shared")
                Tx.cuda.cta_sync()

                with Tx.thread()[0:1]:
                    Tx.copy_async(A_smem[*r_smem], A[*r_gmem], dispatch="tma", mbar=mbar_ptr)
                    Tx.ptx.mbarrier.arrive.expect_tx(mbar_ptr, total_bytes)
                Tx.ptx.mbarrier.try_wait(mbar_ptr, 0)

                Tx.ptx.fence.proxy("shared")
                Tx.cuda.cta_sync()
                with Tx.cta():
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
    @Tx.prim_func(tirx=True)
    def copy_async_test(A_ptr: Tx.handle, B_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, (128, WIDTH), dtype)
        B = Tx.match_buffer(B_ptr, (128, WIDTH), dtype)

        A_flat = A.view(-1)
        B_flat = B.view(-1)

        with Tx.kernel():
            bx = Tx.cta_id([1], parent="kernel")
            wg_id = Tx.warpgroup_id([1], parent="cta")
            warp_id = Tx.warp_id([4], parent="warpgroup")
            lane_id = Tx.thread_id([32], parent="warp")
            tid_in_wg = Tx.thread_id([128], parent="cta")

            tmem_addr = Tx.alloc_shared([1], "uint32")

            with Tx.warpgroup()[0:1]:
                with Tx.warp()[0:1]:
                    Tx.ptx.tcgen05.alloc(Tx.address_of(tmem_addr), n_cols=max(32, next_power_of_2(width_32b)), cta_group=1)

                Tx.tvm_storage_sync("shared")

                tmem = Tx.decl_buffer((128, WIDTH), dtype, scope="tmem", allocated_addr=tmem_addr[0],
                                     layout=TileLayout(([128, WIDTH], [(1, "TLane"), (1, "TCol")])))

                A_reg = Tx.alloc_local((WIDTH), dtype)
                B_reg = Tx.alloc_local((WIDTH), dtype)
                A_local = A_reg.view(128, WIDTH, layout=local_view)
                B_local = B_reg.view(128, WIDTH, layout=local_view)

                # A -> A_local
                with Tx.thread():
                    for i in range(WIDTH // VEC_LEN):
                        g_offset = Tx.meta_var(g_layout.apply(tid_in_wg, i, 0)["m"])
                        Tx.copy(A_reg[i * VEC_LEN: i * VEC_LEN + VEC_LEN], A_flat[g_offset: g_offset + VEC_LEN])
                    for i in range(WIDTH):
                        B_reg[i] = Tx.cast(0, dtype)
                Tx.cuda.cta_sync()

                # A_local -> tmem (async)
                Tx.copy_async(tmem[:, :], A_local[:, :])
                Tx.ptx.tcgen05.wait.st()  # explicit wait
                Tx.cuda.cta_sync()

                # tmem -> B_local (async)
                Tx.copy_async(B_local[:, :], tmem[:, :])
                Tx.ptx.tcgen05.wait.ld()  # explicit wait
                Tx.cuda.cta_sync()

                # B_local -> B
                with Tx.thread():
                    for i in range(WIDTH // VEC_LEN):
                        g_offset = Tx.meta_var(g_layout.apply(tid_in_wg, i, 0)["m"])
                        Tx.copy(B_flat[g_offset: g_offset + VEC_LEN], B_reg[i * VEC_LEN: i * VEC_LEN + VEC_LEN])

                with Tx.warp()[0:1]:
                    Tx.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
                    Tx.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=max(32, next_power_of_2(width_32b)), cta_group=1)
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
    @Tx.prim_func(tirx=True)
    def copy_async(A_ptr: Tx.handle, B_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, g_shape, dtype, layout=layoutA)
        B = Tx.match_buffer(B_ptr, g_shape, dtype, layout=layoutB)

        with Tx.kernel():
            bx = Tx.cta_id([1], parent="kernel")
            tx = Tx.thread_id([thread_cnt], parent="cta")

            with Tx.thread():
                dyn = Tx.alloc_buffer([smem_bytes + 64], "uint8", scope="shared.dyn")
                A_smem = Tx.decl_buffer(s_shape, dtype, dyn.data, elem_offset=0, layout=shared_layout)
                mbarrier = Tx.decl_buffer([1], "uint64", dyn.data, elem_offset=smem_bytes // 8)
                mbar_ptr = Tx.meta_var(mbarrier.ptr_to([0]))

                with Tx.thread()[0:1]:
                    Tx.ptx.mbarrier.init(mbar_ptr, 1)
                Tx.ptx.fence.proxy("shared")
                Tx.cuda.cta_sync()

                with Tx.thread()[0:1]:
                    Tx.copy_async(A_smem[*r_smem], A[*r_gmem], dispatch="tma", mbar=mbar_ptr)
                    Tx.ptx.mbarrier.arrive.expect_tx(mbar_ptr, total_bytes)

                Tx.ptx.mbarrier.try_wait(mbar_ptr, 0)
                Tx.ptx.fence.proxy("shared")
                Tx.cuda.cta_sync()

                with Tx.cta():
                    Tx.copy(B[*r_gmem], A_smem[*r_smem])
    # fmt: on

    np_dtype = tvm.testing.np_dtype_from_str(dtype)
    target = tvm.target.Target("cuda")

    with target:
        mod = tvm.IRModule({"main": copy_async})

        # Verify that LowerTIRx generates exactly 1 TMA instruction
        lowered = tvm.tir.transform.LowerTIRx()(mod)
        lowered_str = str(lowered)

        # Count total TMA operations including loop iterations using a TIR visitor
        # This verifies that enlarge_factor is correctly computed based on copy extent,
        # resulting in a single TMA operation instead of multiple iterations.
        counter = TMACounter()
        counter.visit_stmt(lowered["main"].body)

        assert counter.total_tma_ops == 1, (
            f"Expected exactly 1 TMA operation, got {counter.total_tma_ops}. "
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


def test_copy_g2s_tma_multicast():
    pass


@tvm.testing.requires_cuda_compute_version(9)
@pytest.mark.parametrize("swizzle_len", [3])
@pytest.mark.parametrize("dtype", ["float16"])
def test_copy_g2s_tma_symbolic_dimension(dtype, swizzle_len):
    """Test TMA copy with symbolic dimension in global buffer (like hgemm pattern).

    This tests the pattern:
        Tx.copy_async(A_smem[ks, :, :], A[m_st : m_st + BLK_M, k_start : k_start + BLK_K], **tma_copy)

    Where M is a symbolic dimension in the global buffer.
    """
    # Fixed dimensions
    K = 256
    BLK_M = 64
    BLK_K = 64
    SMEM_PIPE_DEPTH = 2
    M_CONCRETE = 128  # Concrete value for testing
    thread_cnt = 128

    dev = tvm.cuda(0)

    # Shared memory layout with swizzle
    shared_layout = Tx.ComposeLayout(
        Tx.SwizzleLayout(3, swizzle_len, 3, swizzle_inner=True),
        Tx.TileLayout(shard=((SMEM_PIPE_DEPTH, BLK_M, BLK_K), (BLK_M * BLK_K, BLK_K, 1))),
    )

    # Compute bytes for mbarrier
    smem_bytes = SMEM_PIPE_DEPTH * BLK_M * BLK_K * tvm.DataType(dtype).bits // 8
    copy_bytes = BLK_M * BLK_K * tvm.DataType(dtype).bits // 8

    # fmt: off
    @Tx.prim_func(tirx=True)
    def copy_async(A_ptr: Tx.handle, B_ptr: Tx.handle) -> None:
        M = Tx.int32()
        A = Tx.match_buffer(A_ptr, [M, K], dtype)
        B = Tx.match_buffer(B_ptr, [SMEM_PIPE_DEPTH, BLK_M, BLK_K], dtype)

        with Tx.kernel():
            bx = Tx.cta_id([1], parent="kernel")
            tx = Tx.thread_id([thread_cnt], parent="cta")

            with Tx.thread():
                dyn = Tx.alloc_buffer([smem_bytes + 64], "uint8", scope="shared.dyn")
                A_smem = Tx.decl_buffer(
                    [SMEM_PIPE_DEPTH, BLK_M, BLK_K], dtype, dyn.data, elem_offset=0, layout=shared_layout
                )
                mbarrier = Tx.decl_buffer([1], "uint64", dyn.data, elem_offset=smem_bytes // 8)
                mbar_ptr = Tx.meta_var(mbarrier.ptr_to([0]))

                with Tx.thread()[0:1]:
                    Tx.ptx.mbarrier.init(mbar_ptr, 1)
                Tx.ptx.fence.proxy("shared")
                Tx.cuda.cta_sync()

                # Copy with pipeline index (like hgemm pattern)
                for ks in range(SMEM_PIPE_DEPTH):
                    with Tx.thread()[0:1]:
                        Tx.copy_async(
                            A_smem[ks, :, :],
                            A[0:BLK_M, ks * BLK_K:(ks + 1) * BLK_K],
                            dispatch="tma",
                            mbar=mbar_ptr
                        )
                        Tx.ptx.mbarrier.arrive.expect_tx(mbar_ptr, copy_bytes)

                    Tx.ptx.mbarrier.try_wait(mbar_ptr, ks % 2)

                Tx.ptx.fence.proxy("shared")
                Tx.cuda.cta_sync()

                # Copy back to global for verification
                with Tx.cta():
                    for ks in range(SMEM_PIPE_DEPTH):
                        Tx.copy(
                            B[ks, :, :],
                            A_smem[ks, :, :]
                        )
    # fmt: on

    np_dtype = tvm.testing.np_dtype_from_str(dtype)
    target = tvm.target.Target("cuda")

    with target:
        mod = tvm.IRModule({"main": copy_async})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")

        np.random.seed(0)
        A_np = tvm.testing.generate_random_array(dtype, (M_CONCRETE, K))
        B_np = np.zeros((SMEM_PIPE_DEPTH, BLK_M, BLK_K), dtype=np_dtype)

        A = tvm.runtime.tensor(A_np, dev)
        B = tvm.runtime.tensor(B_np, dev)
        mod(A, B)

        # Verify: B[ks, :, :] should equal A[0:BLK_M, ks*BLK_K:(ks+1)*BLK_K]
        B_ref = np.zeros((SMEM_PIPE_DEPTH, BLK_M, BLK_K), dtype=np_dtype)
        for ks in range(SMEM_PIPE_DEPTH):
            B_ref[ks, :, :] = A_np[0:BLK_M, ks * BLK_K:(ks + 1) * BLK_K]
        np.testing.assert_allclose(B_ref, B.numpy())


@tvm.testing.requires_cuda_compute_version(9)
@pytest.mark.parametrize("swizzle_len", [3])
@pytest.mark.parametrize("dtype", ["float16"])
def test_copy_g2s_tma_3d_with_view(dtype, swizzle_len):
    """Test 3D TMA copy using buffer view and swizzle layout (like flash attention pattern).

    This tests the pattern from FA4:
        Q_smem allocated as 4D: (SMEM_PIPE_DEPTH, NUM_BLK_K, BLK_M, BLK_K)
        Q_smem_3d = Q_smem.view(SMEM_PIPE_DEPTH, NUM_BLK_K, SEQ_TILE, GQA_RATIO, BLK_K)
        Tx.copy_async(Q_smem_3d[pipe_idx, blk_k_idx, :, :, :],
                      Q[batch, seq_start:seq_end, head_start:head_end, k_start:k_end], ...)

    Where we copy 3D regions from a 4D global buffer to shared memory.
    The key insight is that the layout is 3D: (SMEM_PIPE_DEPTH, BLK_M, HEAD_DIM)
    while the allocation is 4D: (SMEM_PIPE_DEPTH, NUM_BLK_K, BLK_M, BLK_K)
    where NUM_BLK_K * BLK_K = HEAD_DIM.
    """
    dev = tvm.cuda(0)
    smem_bytes = 2 * 2 * 128 * 64 * tvm.DataType(dtype).bits // 8
    copy_bytes_per_blk = 32 * 4 * 64 * tvm.DataType(dtype).bits // 8

    # Shared memory layout with swizzle
    # Layout is 3D: (SMEM_PIPE_DEPTH, BLK_M, HEAD_DIM)
    # This matches FA4 pattern where Q_layout has 3D TileLayout
    shared_layout = Tx.ComposeLayout(
        Tx.SwizzleLayout(3, swizzle_len, 3, swizzle_inner=True),
        Tx.TileLayout(shard=((2, 128, 128), (128 * 128, 128, 1))),
    )

    # fmt: off
    @Tx.prim_func(tirx=True)
    def copy_async(Q_ptr: Tx.handle, B_ptr: Tx.handle) -> None:
        Q = Tx.match_buffer(Q_ptr, (2, 128, 8, 128), dtype)
        B = Tx.match_buffer(B_ptr, (32, 4, 64), dtype)

        with Tx.kernel():
            bx = Tx.cta_id([1], parent="kernel")
            tx = Tx.thread_id([128], parent="cta")

            with Tx.thread():
                dyn = Tx.alloc_buffer([smem_bytes + 64], "uint8", scope="shared.dyn")
                # Allocate as 4D like FA4: (SMEM_PIPE_DEPTH, NUM_BLK_K, BLK_M, BLK_K)
                Q_smem = Tx.decl_buffer(
                    (2, 2, 128, 64),
                    dtype, dyn.data, elem_offset=0, layout=shared_layout
                )
                mbarrier = Tx.decl_buffer([1], "uint64", dyn.data, elem_offset=smem_bytes // 8)
                mbar_ptr = Tx.meta_var(mbarrier.ptr_to([0]))

                # Create 5D view for 3D copy pattern
                # View reshapes BLK_M -> (SEQ_Q_PER_TILE, GQA_RATIO) for GQA pattern
                Q_smem_5d = Q_smem.view(2, 2, 32, 4, 64)

                with Tx.thread()[0:1]:
                    Tx.ptx.mbarrier.init(mbar_ptr, 1)
                Tx.ptx.fence.proxy("shared")
                Tx.cuda.cta_sync()

                # Copy with pipeline and block loop (like flash attention pattern)

                with Tx.thread()[0:1]:
                    # 3D copy: [SEQ_Q_PER_TILE, GQA_RATIO, BLK_K]
                    Tx.copy_async(
                        Q_smem_5d[0, 0, :, :, :],
                        Q[0, 0:32, 0:4, 0:64],
                        dispatch="tma",
                        mbar=mbar_ptr
                    )
                    Tx.ptx.mbarrier.arrive.expect_tx(mbar_ptr, copy_bytes_per_blk)

                Tx.ptx.mbarrier.try_wait(mbar_ptr, 0)


                Tx.ptx.fence.proxy("shared")
                Tx.cuda.cta_sync()

                # Copy back to global for verification
                with Tx.cta():
                    Tx.copy(
                        B[:, :, :],
                        Q_smem_5d[0, 0, :, :, :]
                    )
    # fmt: on

    np_dtype = tvm.testing.np_dtype_from_str(dtype)
    target = tvm.target.Target("cuda")

    with target:
        mod = tvm.IRModule({"main": copy_async})

        # Verify that LowerTIRx generates exactly 1 TMA instruction
        lowered = tvm.tir.transform.LowerTIRx()(mod)
        counter = TMACounter()
        counter.visit_stmt(lowered["main"].body)

        assert counter.total_tma_ops == 1, (
            f"Expected exactly 1 TMA operation, got {counter.total_tma_ops}. "
            "This indicates the 3D TMA copy with view is not generating optimal code."
        )

        # Now compile and verify correctness
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")

        np.random.seed(0)
        Q_np = tvm.testing.generate_random_array(dtype, (2, 128, 8, 128))
        B_np = np.zeros((32, 4, 64), dtype=np_dtype)

        Q = tvm.runtime.tensor(Q_np, dev)
        B = tvm.runtime.tensor(B_np, dev)
        mod(Q, B)

        B_ref = np.zeros((32, 4, 64), dtype=np_dtype)
        B_ref[:, :, :] = Q_np[0, 0:32, 0:4, 0:64]
        np.testing.assert_allclose(B_ref, B.numpy())


if __name__ == "__main__":
    tvm.testing.main()
