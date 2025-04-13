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
from tvm.tir.layout import TileLayout

from tvm.tir.async_structs import CopyPipeline
from tvm.tirp.op_schedule.cuda.async_structs import (
    tma_shared_layout,
    tma_atom_layout,
    tma_atom_shape,
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
            TileLayout.from_tuple((16, 16)),  # layoutA
            TileLayout.from_tuple((16, 16)),  # layoutB
            TileLayout.from_tuple((8, 8)),  # layoutS
        ),
        ################ A[0:128, 0:32] -> A_smem[0:128, 0:32] -> B[0:128, 0:32] ################
        (
            (128, 32),  # g_shape
            (128, 32),  # s_shape
            (0, 0),  # g_st
            (128, 32),  # g_extent
            32,  # thread_cnt
            TileLayout.from_tuple((128, 32)),  # layoutA
            TileLayout.from_tuple((128, 32)),  # layoutB
            TileLayout.from_tuple((128, 32)),  # layoutS
        ),
        ################ A[32:64, 32:64] -> A_smem[0:32, 0:32] -> B[32:64, 32:64] ################
        (
            (64, 64),  # g_shape
            (32, 32),  # s_shape
            (32, 0),  # g_st
            (32, 32),  # g_extent
            32,  # thread_cnt
            TileLayout.from_tuple((64, 64)),  # layoutA
            TileLayout.from_tuple((64, 64)),  # layoutB
            TileLayout.from_tuple((32, 32)),  # layoutS
        ),
        ################ A[0:1, 0:32, 0:32] -> A_smem[0:32, 0:32] -> B[0:1, 0:32, 0:32] ################
        (
            (4, 32, 32),  # g_shape
            (32, 32),  # s_shape
            (0, 0, 0),  # g_st
            (1, 32, 32),  # g_extent
            32,  # thread_cnt
            TileLayout.from_tuple((4, 32, 32)),  # layoutA
            TileLayout.from_tuple((4, 32, 32)),  # layoutB
            TileLayout.from_tuple((32, 32)),  # layoutS
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

                pipeline = Tp.alloc_copy_pipeline("cta", depth=0, separate_pc=False,
                                                  schedule_config={CopyPipeline.StrategyKind.IMPL: CopyPipeline.Impl.VEC_LOAD})
                pipeline.copy(A_smem[*r_smem], A[*r_gmem])
                pipeline.producer_commit()
                pipeline.consumer_wait(0)
                Tp.copy(B[*r_gmem], A_smem[*r_smem])
    # fmt: on

    np_dtype = tvm.testing.np_dtype_from_str(dtype)
    target = tvm.target.Target.from_device(dev)
    with target:
        mod = tvm.IRModule({"main": copy_async})
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
            TileLayout.from_tuple((8, 256)),  # A_layout
            TileLayout.from_tuple((8, 256)),  # B_layout
            lambda dtype, swizzle_len: tma_shared_layout(dtype, swizzle_len, (8, 256)),
        ),
        (
            (64, 256),  # global_shape
            ((0, 64), (0, 256)),  # global_region
            (3, 64, 256),  # shared_shape
            ((1, 2), (0, 64), (0, 256)),  # shared_region
            64,  # thread count per CTA
            TileLayout.from_tuple((64, 256)),  # A_layout
            TileLayout.from_tuple((64, 256)),  # B_layout
            lambda dtype, swizzle_len: tma_shared_layout(dtype, swizzle_len, (3, 64, 256)),
        ),
        (
            (32, 512),  # global_shape
            ((0, 32), (0, 512)),  # global_region
            (32, 512),  # shared_shape
            ((0, 32), (0, 512)),  # shared_region
            64,  # thread count per CTA
            TileLayout.from_tuple((32, 512)),  # A_layout
            TileLayout.from_tuple((32, 512)),  # B_layout
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
def test_copy_g2s_cta_tma_load(task, dtype, swizzle_len):
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
                mbarrier = T.decl_buffer([1], "uint64", dyn.data, elem_offset=smem_bytes // 8, layout=TileLayout.from_tuple((1,)))
                phase = T.alloc_buffer([1], "int32", scope="local")

                with T.cta():
                    pipeline = Tp.alloc_copy_pipeline("cta", depth=0, separate_pc=False,
                                                      workspace={"mbarrier": mbarrier, "phase": phase},
                                                      schedule_config={CopyPipeline.StrategyKind.IMPL: CopyPipeline.Impl.TMA})
                    pipeline.init()
                    pipeline.copy(A_smem[*r_smem], A[*r_gmem])
                    pipeline.producer_commit(tma_bytes=total_bytes)
                    pipeline.consumer_wait()

                    Tp.copy(B[*r_gmem], A_smem[*r_smem])
    # fmt: on
    np_dtype = tvm.testing.np_dtype_from_str(dtype)
    target = tvm.target.Target.from_device(dev)

    with target:
        mod = tvm.IRModule({"main": copy_async})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirp")
        print(mod.mod.imported_modules[0].get_source())

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
            TileLayout.from_tuple((3, 8, 256)),  # A_layout
            TileLayout.from_tuple((3, 8, 256)),  # B_layout
            lambda dtype, swizzle_len: tma_shared_layout(dtype, swizzle_len, (8, 256)),
        ),
        (
            (5, 64, 256),  # global_shape
            (64, 256),  # shared_shape
            64,  # thread count per CTA
            TileLayout.from_tuple((5, 64, 256)),  # A_layout
            TileLayout.from_tuple((5, 64, 256)),  # B_layout
            lambda dtype, swizzle_len: tma_shared_layout(dtype, swizzle_len, (64, 256)),
        ),
        (
            (7, 32, 512),  # global_shape
            (32, 512),  # shared_shape
            32,  # thread count per CTA
            TileLayout.from_tuple((7, 32, 512)),  # A_layout
            TileLayout.from_tuple((7, 32, 512)),  # B_layout
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
                mbarrier = T.decl_buffer([1], "uint64", dyn.data, elem_offset=smem_bytes // 8, layout=TileLayout.from_tuple((1,)))
                phase = T.alloc_buffer([1], "int32", scope="local")

                with T.cta():
                    pipeline = Tp.alloc_copy_pipeline("cta", depth=0, separate_pc=False,
                                                      workspace={"mbarrier": mbarrier, "phase": phase},
                                                      schedule_config={CopyPipeline.StrategyKind.IMPL: CopyPipeline.Impl.TMA})
                    pipeline.init()
                    for stage in range(n):
                        pipeline.copy(A_smem[*r_smem], A[*r_gmem(stage)])
                        pipeline.producer_commit(tma_bytes=smem_bytes)
                        pipeline.consumer_wait()
                        Tp.copy(B[*r_gmem(stage)], A_smem[*r_smem])
    # fmt: on
    np_dtype = tvm.testing.np_dtype_from_str(dtype)
    target = tvm.target.Target.from_device(dev)

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


def test_copy_pipeline_no_specialize_cta_vec_load():
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

                pipe = Tp.alloc_copy_pipeline(thread_scope="cta", depth=0, separate_pc=False,
                                              schedule_config={CopyPipeline.StrategyKind.IMPL: CopyPipeline.Impl.VEC_LOAD})

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
        mod = tvm.compile(mod, target=target, tir_pipeline="tirp")

        np.random.seed(0)
        A_np = np.ones((N, M)).astype("float32") * 10
        B_np = np.zeros((N, 128), dtype="float32")

        A = tvm.nd.array(A_np, device=DEV)
        B = tvm.nd.array(B_np, device=DEV)
        mod(A, B)

        B_np_ref = np.sum(A_np.reshape((N, 128, M // 128)), axis=2)
        tvm.testing.assert_allclose(B.numpy(), B_np_ref)


if __name__ == "__main__":
    tvm.testing.main()
