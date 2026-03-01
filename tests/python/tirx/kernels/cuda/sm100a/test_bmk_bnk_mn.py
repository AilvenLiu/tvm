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

# Batch-Reduction GEMM: D[m,n] = sum_s(A[s,m,k] @ B[s,n,k].T)
# Transcribed from DeepGEMM sm100_bmk_bnk_mn.cuh
# Used in DeepSeek MLA (Multi-Latent Attention)
#
# Architecture: Single CTA (no cluster), CTA_GROUP=1
# Producer WG1: warp 3 = TMA load, warp 0 = MMA issue
# Consumer WG0: TMEM -> RF -> SMEM -> TMA store
# K-loop iterates over S * (K / BLK_K) stages total

import torch

import tvm
import tvm.testing
from tvm.script import tirx as Tx
from tvm.tir.layout import S, TCol, TileLayout, TLane, tid_in_wg
from tvm.tirx.bench.utils import ProtonContext, bench
from tvm.tirx.pipeline import MBarrier, TCGen05Bar, TMABar
from tvm.tirx.tile_scheduler import GroupMajor3D

# Architecture
CTA_GROUP = 1
WG_NUMBER = 2
WARP_NUMBER = 4
NUM_THREADS = 32 * WARP_NUMBER * WG_NUMBER
SM_NUMBER = 148

# Problem dimensions
S_DIM = 4
M, N, K = 4096, 128, 7168
BLK_M, BLK_N, BLK_K = 128, 128, 64

# MMA dimensions
MMA_M = 128
MMA_N = 128
MMA_K = 16

# Pipeline
SMEM_PIPE_DEPTH = 4
TMEM_PIPE_DEPTH = 2

# Split-K parameters
TILE_K_BLOCKS = 64
TOTAL_SK_BLOCKS = S_DIM * (K // BLK_K)  # 448
K_TILES = TOTAL_SK_BLOCKS // TILE_K_BLOCKS  # 7
NUM_K_ITERS = TILE_K_BLOCKS  # 64 (each CTA processes 64 K-blocks)
PIPE_CYCLE = NUM_K_ITERS // SMEM_PIPE_DEPTH  # 16
PIPE_REMAIN_NUM = NUM_K_ITERS % SMEM_PIPE_DEPTH  # 0

# Data types
a_type = tvm.DataType("bfloat16")
b_type = tvm.DataType("bfloat16")
d_type = tvm.DataType("float32")

F16_BYTES = 2
F32_BYTES = 4
F128_BYTES = 16

# Epilogue: standard, iterate N chunks
# EPI_TILE = N-chunk size, TMEM_LD_SIZE = cols per tcgen05.ld
EPI_TILE = 32
TMEM_LD_SIZE = 8
N_COLS = TMEM_PIPE_DEPTH * MMA_N
SWIZZLE = 3

assert TMEM_PIPE_DEPTH * MMA_N <= 512

# Tile scheduler
assert M % BLK_M == 0
assert N % BLK_N == 0
TILE_M_NUM = M // BLK_M
TILE_N_NUM = N // BLK_N
TILE_GROUPS_ROW_SIZE = 8


def flops(ms):
    return 2 * S_DIM * M * N * K / (ms * 1e-3)


def calc_diff(x, y):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


def get_source(func):
    target = tvm.target.Target("cuda")
    mod = tvm.IRModule({"main": func})
    mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
    src = mod.mod.imports[0].inspect_source()
    return src, mod


@Tx.inline
def skip():
    pass


def prepare_data():
    A = torch.randn((S_DIM, M, K), dtype=torch.bfloat16, device="cuda")
    B = torch.randn((S_DIM, N, K), dtype=torch.bfloat16, device="cuda")
    D = torch.zeros((M, N), dtype=torch.float32, device="cuda")
    A_flat = A.reshape(S_DIM * M, K).contiguous()
    B_flat = B.reshape(S_DIM * N, K).contiguous()
    D_ref = torch.einsum("smk,snk->mn", A.float(), B.float())
    return A, B, A_flat, B_flat, D, D_ref


A_layout = Tx.ComposeLayout(
    Tx.SwizzleLayout(3, 3, 3, swizzle_inner=True),
    Tx.TileLayout(Tx.S[(SMEM_PIPE_DEPTH, BLK_M, BLK_K) : (BLK_M * BLK_K, BLK_K, 1)]),
)
B_layout = Tx.ComposeLayout(
    Tx.SwizzleLayout(3, 3, 3, swizzle_inner=True),
    Tx.TileLayout(Tx.S[(SMEM_PIPE_DEPTH, BLK_N, BLK_K) : (BLK_N * BLK_K, BLK_K, 1)]),
)
D_layout = Tx.TileLayout(Tx.S[(TMEM_PIPE_DEPTH, BLK_M, EPI_TILE) : (BLK_M * EPI_TILE, EPI_TILE, 1)])


# fmt: off
@Tx.prim_func(tirx=True)
def bmk_bnk_mn_gemm(
    A: Tx.Buffer((S_DIM * M, K), a_type),
    B: Tx.Buffer((S_DIM * N, K), b_type),
    D: Tx.Buffer((M, N), d_type),
):
    with Tx.kernel():
        bx = Tx.cta_id([SM_NUMBER], parent="kernel")
        wg_id = Tx.warpgroup_id([WG_NUMBER], parent="cta")  # noqa: F841
        warp_id = Tx.warp_id([WARP_NUMBER], parent="warpgroup")
        lane_id = Tx.thread_id([32], parent="warp")
        with Tx.cta():
            # Shared memory via PoolAllocator
            pool = Tx.meta_var(Tx.PoolAllocator())
            tmem_addr = Tx.decl_scalar("uint32", pool.ptr, scope="shared.dyn", elem_offset=0)
            pool.move_base_to(8)

            # Barriers
            tma2mma_bar = TMABar(pool, SMEM_PIPE_DEPTH)
            mma2tma_bar = TCGen05Bar(pool, SMEM_PIPE_DEPTH)
            mma2ld_bar = TCGen05Bar(pool, TMEM_PIPE_DEPTH)
            ld2mma_bar = MBarrier(pool, TMEM_PIPE_DEPTH)

            pool.move_base_to(1024)
            A_smem = pool.alloc((SMEM_PIPE_DEPTH, BLK_M, BLK_K), a_type, layout=A_layout)
            B_smem = pool.alloc((SMEM_PIPE_DEPTH, BLK_N, BLK_K), b_type, layout=B_layout)
            D_smem = pool.alloc((TMEM_PIPE_DEPTH, BLK_M, EPI_TILE), d_type, layout=D_layout)
            pool.commit()

            # Local memory
            reg = Tx.alloc_buffer((TMEM_LD_SIZE,), "float32", scope="local")
            reg_wg = reg.view(128, TMEM_LD_SIZE, layout=TileLayout(S[(128, TMEM_LD_SIZE) : (1@tid_in_wg, 1)]))  # noqa: E501
            stage: Tx.int32
            phase = Tx.alloc_buffer((1,), "int32", scope="local")
            descA: Tx.uint64
            descB: Tx.uint64
            descI: Tx.uint32

            # Tile scheduler
            tile_scheduler = GroupMajor3D(
                "tile_scheduler",
                m_tiles=TILE_M_NUM,
                n_tiles=TILE_N_NUM,
                k_tiles=K_TILES,
                group_rows=TILE_GROUPS_ROW_SIZE,
                step=SM_NUMBER,
            )

            tma2mma_bar.init(1)
            mma2ld_bar.init(1)
            mma2tma_bar.init(1)
            ld2mma_bar.init(128)

            tile_scheduler.init(bx)

            # Alloc TMEM
            with Tx.warp()[0:1]:
                Tx.ptx.tcgen05.alloc(Tx.address_of(tmem_addr), n_cols=N_COLS, cta_group=1)
                Tx.cuda.warp_sync()

            Tx.ptx.fence.proxy_async("shared::cta")
            Tx.ptx.fence.mbarrier_init()
            Tx.cuda.cta_sync()
            Tx.cuda.trap_when_assert_failed(tmem_addr == 0)
            tmem = Tx.decl_buffer(
                (128, N_COLS), "float32", scope="tmem", allocated_addr=0,
                layout=TileLayout(S[(128, N_COLS) : (1@TLane, 1@TCol)]),
            )

            @Tx.inline
            def partitioned_loop(main_loop, epilogue1, epilogue2):
                for ko in Tx.serial(PIPE_CYCLE):
                    for ks in Tx.unroll(SMEM_PIPE_DEPTH):
                        stage = ko * SMEM_PIPE_DEPTH + ks
                        main_loop(False, ks)
                    phase[0] = phase[0] ^ 1
                if PIPE_REMAIN_NUM > 0:
                    for ks in Tx.unroll(PIPE_REMAIN_NUM):
                        stage = PIPE_CYCLE * SMEM_PIPE_DEPTH + ks  # noqa: F841
                        main_loop(True, ks)
                    epilogue1()
                    for ks in Tx.unroll(PIPE_REMAIN_NUM, SMEM_PIPE_DEPTH):
                        epilogue2(ks)
                    phase[0] = phase[0] ^ 1
                else:
                    epilogue1()

            with Tx.cta():
                Tx.attr({"tirx.scope_partition": True})

                # === Producer warpgroup (WG1) ===
                with Tx.warpgroup()[1:2]:
                    if warp_id == 3:
                        # TMA warp: load A and B tiles
                        phase[0] = 0
                        while tile_scheduler.valid():
                            m_idx = Tx.meta_var(tile_scheduler.m_idx)
                            n_idx = Tx.meta_var(tile_scheduler.n_idx)
                            k_idx = Tx.meta_var(tile_scheduler.k_idx)
                            m_start = Tx.meta_var(m_idx * BLK_M)
                            n_start = Tx.meta_var(n_idx * BLK_N)
                            with Tx.thread()[Tx.ptx.elect_sync()]:
                                @Tx.inline
                                def tma_load(is_remain, ks):
                                    # Compute batch and K offset from global stage index
                                    global_k_block = Tx.meta_var(k_idx * TILE_K_BLOCKS + stage)
                                    sk_offset = Tx.meta_var(global_k_block * BLK_K)
                                    k_start = Tx.meta_var(sk_offset % K)
                                    s_idx = Tx.meta_var(sk_offset // K)
                                    # A is [S*M, K], B is [S*N, K] (batches concatenated)
                                    m_global = Tx.meta_var(s_idx * M + m_start)
                                    n_global = Tx.meta_var(s_idx * N + n_start)

                                    tma_copy = Tx.meta_var({
                                        "dispatch": "tma",
                                        "mbar": tma2mma_bar.ptr_to([ks]),
                                        "cta_group": CTA_GROUP,
                                    })
                                    mma2tma_bar.wait(ks, phase[0] ^ 1)
                                    Tx.copy_async(
                                        A_smem[ks, :, :],
                                        A[m_global : m_global + BLK_M, k_start : k_start + BLK_K],
                                        **tma_copy,
                                    )
                                    Tx.copy_async(
                                        B_smem[ks, :, :],
                                        B[n_global : n_global + BLK_N, k_start : k_start + BLK_K],
                                        **tma_copy,
                                    )
                                    tma2mma_bar.arrive(ks, BLK_K * (BLK_M + BLK_N) * F16_BYTES)

                                @Tx.inline
                                def tma_load_epilogue(ks):
                                    mma2tma_bar.wait(ks, phase[0] ^ 1)
                                    Tx.ptx.mbarrier.arrive(tma2mma_bar.ptr_to([ks]))

                                partitioned_loop(tma_load, skip, tma_load_epilogue)
                            tile_scheduler.next_tile()

                    elif warp_id == 0:
                        # MMA warp: issue tcgen05 UMMA
                        tmem_idx = Tx.local_scalar("int32", "tmem_idx")
                        tmem_phase = Tx.local_scalar("int32", "tmem_phase")
                        Tx.ptx.tcgen05.encode_instr_descriptor(
                            Tx.address_of(descI), "float32", a_type, b_type,  # noqa: F821
                            MMA_M, MMA_N, MMA_K, False, False, CTA_GROUP,
                        )
                        phase[0] = 0
                        while tile_scheduler.valid():
                            m_idx = Tx.meta_var(tile_scheduler.m_idx)
                            n_idx = Tx.meta_var(tile_scheduler.n_idx)
                            k_idx = Tx.meta_var(tile_scheduler.k_idx)
                            if Tx.ptx.elect_sync():
                                tmem_idx = tile_scheduler.tile_idx % TMEM_PIPE_DEPTH
                                tmem_phase = (tile_scheduler.tile_idx // TMEM_PIPE_DEPTH) & 1

                                ld2mma_bar.wait(tmem_idx, tmem_phase ^ 1)
                                Tx.ptx.tcgen05.fence.after_thread_sync()

                                @Tx.inline
                                def mma(is_remain, ks):
                                    tma2mma_bar.wait(ks, phase[0])
                                    Tx.ptx.tcgen05.fence.after_thread_sync()
                                    for ki in Tx.unroll(BLK_K // MMA_K):
                                        Tx.ptx.tcgen05.encode_matrix_descriptor(
                                            Tx.address_of(descA),  # noqa: F821
                                            A_smem.ptr_to([ks, 0, ki * MMA_K]),
                                            ldo=1,
                                            sdo=8 * BLK_K * F16_BYTES // F128_BYTES,
                                            swizzle=SWIZZLE,
                                        )
                                        Tx.ptx.tcgen05.encode_matrix_descriptor(
                                            Tx.address_of(descB),  # noqa: F821
                                            B_smem.ptr_to([ks, 0, ki * MMA_K]),
                                            ldo=1,
                                            sdo=8 * BLK_K * F16_BYTES // F128_BYTES,
                                            swizzle=SWIZZLE,
                                        )
                                        # Standard A,B order: TMEM = A@B^T -> [M_lanes, N_cols]
                                        if (stage == 0 and ki == 0) and (
                                            (not is_remain) or (is_remain and PIPE_CYCLE == 0)
                                        ):
                                            Tx.ptx.tcgen05.mma(
                                                "float32", a_type, b_type,
                                                tmem_idx * MMA_N, descA, descB, descI,  # noqa: F821
                                                False, CTA_GROUP, False,
                                            )
                                        else:
                                            Tx.ptx.tcgen05.mma(
                                                "float32", a_type, b_type,
                                                tmem_idx * MMA_N, descA, descB, descI,  # noqa: F821
                                                False, CTA_GROUP, True,
                                            )
                                    mma2tma_bar.arrive(ks, CTA_GROUP, 1)

                                @Tx.inline
                                def mma_epilogue1():
                                    mma2ld_bar.arrive(tmem_idx, CTA_GROUP, 1)

                                @Tx.inline
                                def mma_epilogue2(ks):
                                    tma2mma_bar.wait(ks, phase[0])
                                    mma2tma_bar.arrive(ks, CTA_GROUP, 1)

                                partitioned_loop(mma, mma_epilogue1, mma_epilogue2)

                            tile_scheduler.next_tile()

                # === Consumer warpgroup (WG0) ===
                # Standard epilogue: TMEM[M_lanes, N_cols] -> iterate N chunks
                with Tx.warpgroup()[0:1]:
                    Tx.cuda.trap_when_assert_failed(tmem_addr == 0)
                    tmem_idx = Tx.local_scalar("int32", "tmem_idx")
                    tmem_phase = Tx.local_scalar("int32", "tmem_phase")
                    phase[0] = 0
                    while tile_scheduler.valid():
                        m_idx = Tx.meta_var(tile_scheduler.m_idx)
                        n_idx = Tx.meta_var(tile_scheduler.n_idx)
                        k_idx = Tx.meta_var(tile_scheduler.k_idx)
                        tmem_idx = tile_scheduler.tile_idx % TMEM_PIPE_DEPTH
                        tmem_phase = (tile_scheduler.tile_idx // TMEM_PIPE_DEPTH) & 1

                        # Wait previous TMA store
                        if lane_id == 0 and warp_id == 0:
                            Tx.ptx.cp_async.bulk.wait_group(0)
                        Tx.cuda.warpgroup_sync(10)

                        # Wait MMA completion
                        mma2ld_bar.wait(tmem_idx, tmem_phase)
                        Tx.ptx.tcgen05.fence.after_thread_sync()

                        for ko in Tx.unroll(MMA_N // EPI_TILE):
                            stage = (tile_scheduler.tile_idx * MMA_N // EPI_TILE + ko) % TMEM_PIPE_DEPTH  # noqa: E501

                            if ko >= TMEM_PIPE_DEPTH:
                                if lane_id == 0 and warp_id == 0:
                                    Tx.ptx.cp_async.bulk.wait_group(TMEM_PIPE_DEPTH - 1)
                                Tx.cuda.warpgroup_sync(10)

                            # TMEM -> RF -> SMEM (f32)
                            for ki in Tx.unroll(EPI_TILE // TMEM_LD_SIZE):
                                col_st = Tx.meta_var(
                                    tmem_idx * MMA_N + ko * EPI_TILE + ki * TMEM_LD_SIZE
                                )
                                Tx.copy(reg_wg[:, :], tmem[:, col_st : col_st + TMEM_LD_SIZE])
                                with Tx.thread():
                                    st = Tx.meta_var(ki * TMEM_LD_SIZE)
                                    Tx.copy(
                                        D_smem[stage, warp_id * 32 + lane_id,
                                               st : st + TMEM_LD_SIZE],
                                        reg[:],
                                    )

                            # Signal TMEM is free after last N chunk
                            if ko == MMA_N // EPI_TILE - 1:
                                Tx.ptx.tcgen05.fence.before_thread_sync()
                                ld2mma_bar.arrive(tmem_idx)

                            Tx.ptx.fence.proxy_async("shared::cta")
                            Tx.cuda.warpgroup_sync(10)

                            # SMEM -> GMEM via TMA store with reduce-add for split-K
                            with Tx.thread()[lane_id == 0 and warp_id == 0]:
                                m_st = Tx.meta_var(m_idx * BLK_M)
                                n_st = Tx.meta_var(n_idx * BLK_N + ko * EPI_TILE)
                                Tx.copy_async(
                                    D[m_st : m_st + BLK_M, n_st : n_st + EPI_TILE],
                                    D_smem[stage, :, :],
                                    dispatch="tma",
                                    use_tma_reduce="add",
                                )
                                Tx.ptx.cp_async.bulk.commit_group()

                        tile_scheduler.next_tile()

                    with Tx.thread()[lane_id == 0 and warp_id == 0]:
                        Tx.ptx.cp_async.bulk.wait_group(0)
                    Tx.cuda.warpgroup_sync(10)

            # Dealloc TMEM
            with Tx.warp()[0:1]:
                Tx.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
                Tx.ptx.tcgen05.dealloc(tmem_addr, n_cols=N_COLS, cta_group=1)

            Tx.cuda.cta_sync()
# fmt: on


@tvm.testing.requires_cuda_compute_version(10, exact=True)
def test_bmk_bnk_mn():
    A, B, A_flat, B_flat, D_out, D_ref = prepare_data()

    target = tvm.target.Target("cuda")
    with target:
        src, mod = get_source(bmk_bnk_mn_gemm)

        def run():
            D_out.zero_()
            mod(A_flat, B_flat, D_out)

        ms = bench(run, warmup=10, repeat=30, proton_name="tir")
        tflops = flops(ms) / 1e12
        print(f"TIR: {tflops:.2f} TFLOPS, time: {ms:.3f} ms")

    diff = calc_diff(D_out, D_ref.to("cuda"))
    print(f"calc_diff: {diff:.6f}")
    assert diff < 2e-3, f"Correctness check failed: calc_diff={diff}"
    print("Test passed!")


def bench_bmk_bnk_mn():
    import deep_gemm

    A = torch.randn((S_DIM, M, K), dtype=torch.bfloat16, device="cuda")
    B = torch.randn((S_DIM, N, K), dtype=torch.bfloat16, device="cuda")
    A_flat = A.reshape(S_DIM * M, K).contiguous()
    B_flat = B.reshape(S_DIM * N, K).contiguous()

    target = tvm.target.Target("cuda")
    with target:
        src, mod = get_source(bmk_bnk_mn_gemm)

    def std():
        # DeepGEMM einsum 'bmk,bnk->mn': c=out enables f32 accumulation
        out = torch.zeros((M, N), dtype=torch.float32, device="cuda")

        def run():
            out.zero_()
            deep_gemm.einsum("bmk,bnk->mn", A, B, out, c=out)

        ms = bench(run, warmup=50, repeat=50, proton_name="std")
        out.zero_()
        deep_gemm.einsum("bmk,bnk->mn", A, B, out, c=out)
        return ms, out

    def tir():
        D_out = torch.zeros((M, N), dtype=torch.float32, device="cuda")
        ms = bench(
            lambda: (D_out.zero_(), mod(A_flat, B_flat, D_out))[-1],
            warmup=50,
            repeat=50,
            proton_name="tir",
        )
        D_out.zero_()
        mod(A_flat, B_flat, D_out)
        return ms, D_out

    with ProtonContext():
        tir_ms, tir_out = tir()
        print(f"TIR flops: {flops(tir_ms) / 1e12:.2f} TFLOPS, time: {tir_ms:.3f} ms")
        std_ms, std_out = std()
        print(f"Std flops: {flops(std_ms) / 1e12:.2f} TFLOPS, time: {std_ms:.3f} ms")
        diff = calc_diff(tir_out, std_out)
        print(f"calc_diff(tir, std): {diff:.6f}")
        assert diff < 2e-3, f"Correctness check failed: calc_diff={diff}"
        print("Benchmark passed!")


if __name__ == "__main__":
    bench_bmk_bnk_mn()
