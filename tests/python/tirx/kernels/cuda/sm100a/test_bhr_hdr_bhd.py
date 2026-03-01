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

# Batched per-head GEMM: D[b,h,d] = A[b,h,r] @ B[h,d,r].T
# Transcribed from DeepGEMM sm100_bf16_gemm.cuh with GemmType::Batched
# Used in DeepSeek MLA inference path
#
# Architecture: Single CTA (no cluster), CTA_GROUP=1
# Producer WG1: warp 3 = TMA load, warp 0 = MMA issue
# Consumer WG0: TMEM -> RF -> cast(f32->bf16) -> SMEM -> TMA store
# Scheduler: GroupMajor3D over (M_tiles, N_tiles, H_groups)

import tvm
import tvm.testing
from tvm.script import tirx as Tx
from tvm.tir.layout import TileLayout, tid_in_wg, TLane, TCol, S
from tvm.tirx.bench.utils import bench, ProtonContext
from tvm.tirx.pipeline import MBarrier, TMABar, TCGen05Bar
from tvm.tirx.tile_scheduler import GroupMajor3D

import torch

# Architecture
CTA_GROUP = 1
WG_NUMBER = 2
WARP_NUMBER = 4
NUM_THREADS = 32 * WARP_NUMBER * WG_NUMBER
SM_NUMBER = 148

# Problem dimensions: D[B,H,D] = A[B,H,R] @ B_mat[H,D,R].T
# B_dim = batch, H = heads, D_dim = output dim, R = reduction dim
B_DIM = 128
H_DIM = 8
D_DIM = 128
R_DIM = 128

# GEMM mapping: M=B_dim, N=D_dim, K=R_dim, Groups=H_dim
M, N, K = B_DIM, D_DIM, R_DIM
BLK_M, BLK_N, BLK_K = 128, 128, 64

# MMA dimensions
MMA_M = 128
MMA_N = 128
MMA_K = 16

# Pipeline
SMEM_PIPE_DEPTH = 6
TMEM_PIPE_DEPTH = 2

NUM_K_ITERS = K // BLK_K
PIPE_CYCLE = NUM_K_ITERS // SMEM_PIPE_DEPTH
PIPE_REMAIN_NUM = NUM_K_ITERS % SMEM_PIPE_DEPTH

# Data types
a_type = tvm.DataType("bfloat16")
b_type = tvm.DataType("bfloat16")
d_type = tvm.DataType("bfloat16")

F16_BYTES = 2
F32_BYTES = 4
F128_BYTES = 16

# Epilogue
EPI_TILE = 32
TMEM_LD_SIZE = 8
N_COLS = TMEM_PIPE_DEPTH * MMA_N
SWIZZLE = 3

assert TMEM_PIPE_DEPTH * MMA_N <= 512

# Tile scheduler: (M_tiles, N_tiles, H_groups)
assert M % BLK_M == 0
assert N % BLK_N == 0
TILE_M_NUM = M // BLK_M
TILE_N_NUM = N // BLK_N
TILE_GROUPS_ROW_SIZE = 8


def flops(ms):
    return 2 * B_DIM * H_DIM * D_DIM * R_DIM / (ms * 1e-3)


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
    A = torch.randn((B_DIM, H_DIM, R_DIM), dtype=torch.bfloat16, device="cuda")
    B_mat = torch.randn((H_DIM, D_DIM, R_DIM), dtype=torch.bfloat16, device="cuda")
    D_out = torch.zeros((B_DIM, H_DIM, D_DIM), dtype=torch.bfloat16, device="cuda")
    # A[b,h,:] @ B[h,:,:].T -> D[b,h,:]
    D_ref = torch.einsum("bhr,hdr->bhd", A.float(), B_mat.float()).bfloat16()
    # Flatten for kernel: A -> [H*B, R] (head-major), B -> [H*D, R], D -> [H*B, D] (head-major)
    # Kernel indexes A as h_idx * B_DIM + m_start, so head's batch rows must be contiguous
    A_flat = A.permute(1, 0, 2).reshape(H_DIM * B_DIM, R_DIM).contiguous()
    B_flat = B_mat.reshape(H_DIM * D_DIM, R_DIM).contiguous()
    D_flat = D_out.permute(1, 0, 2).reshape(H_DIM * B_DIM, D_DIM).contiguous()
    D_ref_flat = D_ref.permute(1, 0, 2).reshape(H_DIM * B_DIM, D_DIM).contiguous()
    return A_flat, B_flat, D_flat, D_ref_flat


A_layout = Tx.ComposeLayout(
    Tx.SwizzleLayout(3, 3, 3, swizzle_inner=True),
    Tx.TileLayout(Tx.S[(SMEM_PIPE_DEPTH, BLK_M, BLK_K) : (BLK_M * BLK_K, BLK_K, 1)]),
)
B_layout = Tx.ComposeLayout(
    Tx.SwizzleLayout(3, 3, 3, swizzle_inner=True),
    Tx.TileLayout(Tx.S[(SMEM_PIPE_DEPTH, BLK_N, BLK_K) : (BLK_N * BLK_K, BLK_K, 1)]),
)
# D_smem: bf16 output, (EPI_TILE, MMA_N) per stage, no swizzle
D_layout = Tx.TileLayout(
    Tx.S[(TMEM_PIPE_DEPTH, EPI_TILE, MMA_N) : (EPI_TILE * MMA_N, MMA_N, 1)]
)


# fmt: off
@Tx.prim_func(tirx=True)
def bhr_hdr_bhd_gemm(
    A: Tx.Buffer((B_DIM * H_DIM, R_DIM), a_type),   # [B*H, R] (batches×heads concatenated along M)
    B_mat: Tx.Buffer((H_DIM * D_DIM, R_DIM), b_type),  # [H*D, R] (heads concatenated along N)
    D: Tx.Buffer((B_DIM * H_DIM, D_DIM), d_type),    # [B*H, D]
):
    with Tx.kernel():
        bx = Tx.cta_id([SM_NUMBER], parent="kernel")
        wg_id = Tx.warpgroup_id([WG_NUMBER], parent="cta")
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
            D_smem = pool.alloc((TMEM_PIPE_DEPTH, EPI_TILE, MMA_N), d_type, layout=D_layout)
            pool.commit()

            # Local memory
            reg = Tx.alloc_buffer((TMEM_LD_SIZE,), "float32", scope="local")
            reg_bf16 = Tx.alloc_buffer((TMEM_LD_SIZE,), d_type, scope="local")
            reg_wg = reg.view(128, TMEM_LD_SIZE, layout=TileLayout(S[(128, TMEM_LD_SIZE) : (1@tid_in_wg, 1)]))
            stage: Tx.int32
            phase = Tx.alloc_buffer((1,), "int32", scope="local")
            descA: Tx.uint64
            descB: Tx.uint64
            descI: Tx.uint32

            # Tile scheduler: k_tiles = H_DIM (head groups)
            tile_scheduler = GroupMajor3D(
                "tile_scheduler",
                m_tiles=TILE_M_NUM,
                n_tiles=TILE_N_NUM,
                k_tiles=H_DIM,
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
                        stage = PIPE_CYCLE * SMEM_PIPE_DEPTH + ks
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
                            h_idx = Tx.meta_var(tile_scheduler.k_idx)
                            m_start = Tx.meta_var(m_idx * BLK_M)
                            n_start = Tx.meta_var(n_idx * BLK_N)
                            # A is [B*H, R]: head h starts at row h*B
                            a_m_global = Tx.meta_var(h_idx * B_DIM + m_start)
                            # B is [H*D, R]: head h starts at row h*D
                            b_n_global = Tx.meta_var(h_idx * D_DIM + n_start)
                            with Tx.thread()[Tx.ptx.elect_sync()]:
                                @Tx.inline
                                def tma_load(is_remain, ks):
                                    k_start = Tx.meta_var(stage * BLK_K)
                                    tma_copy = Tx.meta_var({
                                        "dispatch": "tma",
                                        "mbar": tma2mma_bar.ptr_to([ks]),
                                        "cta_group": CTA_GROUP,
                                    })
                                    mma2tma_bar.wait(ks, phase[0] ^ 1)
                                    Tx.copy_async(
                                        A_smem[ks, :, :],
                                        A[a_m_global : a_m_global + BLK_M, k_start : k_start + BLK_K],
                                        **tma_copy,
                                    )
                                    Tx.copy_async(
                                        B_smem[ks, :, :],
                                        B_mat[b_n_global : b_n_global + BLK_N, k_start : k_start + BLK_K],
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
                            Tx.address_of(descI), "float32", a_type, b_type,
                            MMA_N, MMA_M, MMA_K, False, False, CTA_GROUP,
                        )
                        phase[0] = 0
                        while tile_scheduler.valid():
                            m_idx = Tx.meta_var(tile_scheduler.m_idx)
                            n_idx = Tx.meta_var(tile_scheduler.n_idx)
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
                                            Tx.address_of(descA),
                                            A_smem.ptr_to([ks, 0, ki * MMA_K]),
                                            ldo=1,
                                            sdo=8 * BLK_K * F16_BYTES // F128_BYTES,
                                            swizzle=SWIZZLE,
                                        )
                                        Tx.ptx.tcgen05.encode_matrix_descriptor(
                                            Tx.address_of(descB),
                                            B_smem.ptr_to([ks, 0, ki * MMA_K]),
                                            ldo=1,
                                            sdo=8 * BLK_K * F16_BYTES // F128_BYTES,
                                            swizzle=SWIZZLE,
                                        )
                                        # Swap B,A: TMEM = B@A^T -> [N_lanes, M_cols]
                                        if (stage == 0 and ki == 0) and (
                                            (not is_remain) or (is_remain and PIPE_CYCLE == 0)
                                        ):
                                            Tx.ptx.tcgen05.mma(
                                                "float32", a_type, b_type,
                                                tmem_idx * MMA_M, descB, descA, descI,
                                                False, CTA_GROUP, False,
                                            )
                                        else:
                                            Tx.ptx.tcgen05.mma(
                                                "float32", a_type, b_type,
                                                tmem_idx * MMA_M, descB, descA, descI,
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
                with Tx.warpgroup()[0:1]:
                    Tx.cuda.trap_when_assert_failed(tmem_addr == 0)
                    tmem_idx = Tx.local_scalar("int32", "tmem_idx")
                    tmem_phase = Tx.local_scalar("int32", "tmem_phase")
                    phase[0] = 0
                    while tile_scheduler.valid():
                        m_idx = Tx.meta_var(tile_scheduler.m_idx)
                        n_idx = Tx.meta_var(tile_scheduler.n_idx)
                        h_idx = Tx.meta_var(tile_scheduler.k_idx)
                        tmem_idx = tile_scheduler.tile_idx % TMEM_PIPE_DEPTH
                        tmem_phase = (tile_scheduler.tile_idx // TMEM_PIPE_DEPTH) & 1

                        # Wait previous TMA store
                        if lane_id == 0 and warp_id == 0:
                            Tx.ptx.cp_async.bulk.wait_group(0)
                        Tx.cuda.warpgroup_sync(10)

                        # Wait MMA completion
                        mma2ld_bar.wait(tmem_idx, tmem_phase)
                        Tx.ptx.tcgen05.fence.after_thread_sync()

                        for ko in Tx.unroll(MMA_M // EPI_TILE):
                            stage = (tile_scheduler.tile_idx * MMA_M // EPI_TILE + ko) % TMEM_PIPE_DEPTH

                            if ko >= TMEM_PIPE_DEPTH:
                                if lane_id == 0 and warp_id == 0:
                                    Tx.ptx.cp_async.bulk.wait_group(TMEM_PIPE_DEPTH - 1)
                                Tx.cuda.warpgroup_sync(10)

                            # TMEM -> RF (f32) -> cast to bf16 -> SMEM
                            for ki in Tx.unroll(EPI_TILE // TMEM_LD_SIZE):
                                col_st = Tx.meta_var(
                                    tmem_idx * MMA_M + ko * EPI_TILE + ki * TMEM_LD_SIZE
                                )
                                Tx.copy(reg_wg[:, :], tmem[:, col_st : col_st + TMEM_LD_SIZE])
                                with Tx.thread():
                                    Tx.cast(reg_bf16[:], reg[:])
                                    Tx.copy(
                                        D_smem[stage, ki * TMEM_LD_SIZE : (ki + 1) * TMEM_LD_SIZE,
                                               warp_id * 32 + lane_id],
                                        reg_bf16[:],
                                    )

                            # Signal TMEM is free after last chunk
                            if ko == MMA_M // EPI_TILE - 1:
                                Tx.ptx.tcgen05.fence.before_thread_sync()
                                ld2mma_bar.arrive(tmem_idx)

                            Tx.ptx.fence.proxy_async("shared::cta")
                            Tx.cuda.warpgroup_sync(10)

                            # SMEM -> GMEM via TMA store
                            # D is [B*H, D]: head h row = h*B + m_start
                            with Tx.thread()[lane_id == 0 and warp_id == 0]:
                                m_st = Tx.meta_var(h_idx * B_DIM + m_idx * BLK_M + ko * EPI_TILE)
                                n_st = Tx.meta_var(n_idx * BLK_N)
                                Tx.copy_async(
                                    D[m_st : m_st + EPI_TILE, n_st : n_st + BLK_N],
                                    D_smem[stage, :, :],
                                    dispatch="tma",
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
def test_bhr_hdr_bhd():
    A_flat, B_flat, D_flat, D_ref_flat = prepare_data()

    target = tvm.target.Target("cuda")
    with target:
        src, mod = get_source(bhr_hdr_bhd_gemm)
        func = lambda: mod(A_flat, B_flat, D_flat)
        ms = bench(func, warmup=10, repeat=30, proton_name="tir")
        tflops = flops(ms) / 1e12
        print(f"TIR: {tflops:.2f} TFLOPS, time: {ms:.3f} ms")

    diff = calc_diff(D_flat, D_ref_flat)
    print(f"calc_diff: {diff:.6f}")
    assert diff < 2e-3, f"Correctness check failed: calc_diff={diff}"
    print("Test passed!")


def bench_bhr_hdr_bhd():
    import deep_gemm

    A = torch.randn((B_DIM, H_DIM, R_DIM), dtype=torch.bfloat16, device="cuda")
    B_mat = torch.randn((H_DIM, D_DIM, R_DIM), dtype=torch.bfloat16, device="cuda")
    A_flat = A.permute(1, 0, 2).reshape(H_DIM * B_DIM, R_DIM).contiguous()
    B_flat = B_mat.reshape(H_DIM * D_DIM, R_DIM).contiguous()

    target = tvm.target.Target("cuda")
    with target:
        src, mod = get_source(bhr_hdr_bhd_gemm)

    def std():
        out = torch.empty((B_DIM, H_DIM, D_DIM), dtype=torch.bfloat16, device="cuda")
        ms = bench(
            lambda: deep_gemm.einsum("bhr,hdr->bhd", A, B_mat, out),
            warmup=50,
            repeat=50,
            proton_name="std",
        )
        deep_gemm.einsum("bhr,hdr->bhd", A, B_mat, out)
        out_flat = out.permute(1, 0, 2).reshape(H_DIM * B_DIM, D_DIM).contiguous()
        return ms, out_flat

    def tir():
        D_out = torch.zeros((H_DIM * B_DIM, D_DIM), dtype=torch.bfloat16, device="cuda")
        ms = bench(
            lambda: mod(A_flat, B_flat, D_out),
            warmup=50,
            repeat=50,
            proton_name="tir",
        )
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
    bench_bhr_hdr_bhd()
