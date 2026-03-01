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

# FP8 Block-Scaled Batched Per-Head GEMM: D[b,h,d] = FP8(A[b,h,r]) @ FP8(B[h,d,r]).T
# Combines:
#   - Batched per-head scheduling from test_bhr_hdr_bhd.py (CTA_GROUP=1)
#   - FP8 block-scaled MMA + scale factor pipeline from test_deepgemm.py
# Used in DeepSeek MLA inference path (FP8 variant)
#
# Architecture: Single CTA (no cluster), CTA_GROUP=1
# Producer WG1: warp 3 = TMA load, warp 2 = transpose SF, warp 0 = MMA issue
# Consumer WG0: TMEM -> RF -> cast(f32->bf16) -> SMEM -> TMA store
# Scheduler: GroupMajor3D over (M_tiles, N_tiles, H_groups)

import ml_dtypes
import numpy as np

import tvm
import tvm.testing
from tvm.script import tirx as Tx
from tvm.tir.layout import TileLayout, TLane, TCol, S
from tvm.tir.layout import tid_in_wg as axis_tid_in_wg
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
B_DIM = 128
H_DIM = 8
D_DIM = 128
R_DIM = 512

# GEMM mapping: M=B_DIM, N=D_DIM, K=R_DIM, Groups=H_DIM
M, N, K = B_DIM, D_DIM, R_DIM
BLK_M, BLK_N, BLK_K = 128, 128, 128

# MMA dimensions
MMA_M = 128
MMA_N = 128
MMA_K = 32

# Pipeline
SMEM_PIPE_DEPTH = 4
TMEM_PIPE_DEPTH = 2

NUM_K_ITERS = K // BLK_K  # 512/128 = 4
PIPE_CYCLE = NUM_K_ITERS // SMEM_PIPE_DEPTH  # 4/4 = 1
PIPE_REMAIN_NUM = NUM_K_ITERS % SMEM_PIPE_DEPTH  # 0

# Data types
a_type = tvm.DataType("float8_e4m3fn")
b_type = tvm.DataType("float8_e4m3fn")
d_type = tvm.DataType("bfloat16")
sfa_type = tvm.DataType("float8_e8m0fnu")
sfb_type = tvm.DataType("float8_e8m0fnu")

F8_BYTES = 1
F16_BYTES = 2
F32_BYTES = 4
F128_BYTES = 16

# FP8 quantization
QUANT_SIZE = BLK_K  # 128
BLK_SFA = 128
BLK_SFB = 128

# Epilogue
EPI_TILE = 32
TMEM_LD_SIZE = 8
SWIZZLE = 3

# TMEM columns: 2*MMA_N (accum) + 2*BLK_SFA//32 (SFA) + 2*BLK_SFB//32 (SFB)
# Logical: 2*128 + 2*(4+4) = 272, but TMEM alloc requires power of 2 -> 512
N_COLS = 512
SFA_TMEM_START_COL = TMEM_PIPE_DEPTH * MMA_N  # 256
SFB_TMEM_START_COL = TMEM_PIPE_DEPTH * MMA_N + TMEM_PIPE_DEPTH * BLK_SFA // 32  # 264

assert TMEM_PIPE_DEPTH * (MMA_N + BLK_SFA // 32 + BLK_SFB // 32) <= 512

# Tile scheduler: (M_tiles, N_tiles, H_groups)
assert M % BLK_M == 0
assert N % BLK_N == 0
TILE_M_NUM = M // BLK_M
TILE_N_NUM = N // BLK_N
TILE_GROUPS_ROW_SIZE = 8


def get_source(func):
    target = tvm.target.Target("cuda")
    mod = tvm.IRModule({"main": func})
    mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
    src = mod.mod.imports[0].inspect_source()
    return src, mod


def flops(ms):
    return 2 * B_DIM * H_DIM * D_DIM * R_DIM / (ms * 1e-3)


def ceildiv(a, b):
    return (a + b - 1) // b


def calc_diff(x, y):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


@Tx.inline
def skip():
    pass


A_layout = Tx.ComposeLayout(
    Tx.SwizzleLayout(4, 3, 3, swizzle_inner=True),
    Tx.TileLayout(Tx.S[(SMEM_PIPE_DEPTH, BLK_M, BLK_K) : (BLK_M * BLK_K, BLK_K, 1)]),
)
B_layout = Tx.ComposeLayout(
    Tx.SwizzleLayout(4, 3, 3, swizzle_inner=True),
    Tx.TileLayout(Tx.S[(SMEM_PIPE_DEPTH, BLK_N, BLK_K) : (BLK_N * BLK_K, BLK_K, 1)]),
)
D_layout = Tx.ComposeLayout(
    Tx.SwizzleLayout(3, 2, 3, swizzle_inner=True),
    Tx.TileLayout(Tx.S[(TMEM_PIPE_DEPTH, BLK_M, EPI_TILE) : (BLK_M * EPI_TILE, EPI_TILE, 1)]),
)
SFA_layout = Tx.TileLayout(Tx.S[(SMEM_PIPE_DEPTH, BLK_SFA // 32, 32) : (BLK_SFA, 32, 1)])
SFB_layout = Tx.TileLayout(Tx.S[(SMEM_PIPE_DEPTH, BLK_SFB // 32, 32) : (BLK_SFB, 32, 1)])


def prepare_data():
    A_origin = torch.randn((B_DIM, H_DIM, R_DIM), dtype=torch.float32)
    B_origin = torch.randn((H_DIM, D_DIM, R_DIM), dtype=torch.float32)

    def ceil_to_ue8m0(x):
        assert x.view(-1).amax().item() > 0
        return torch.pow(2.0, torch.ceil(torch.log2(x.abs())))

    # Reshape to GEMM layout: A[B,H,R] -> [H,B,R] -> [H*B, R], B[H,D,R] -> [H*D, R]
    # Kernel accesses A[h*B+m, :] for head h, batch m
    A_2d = A_origin.permute(1, 0, 2).contiguous().reshape(H_DIM * B_DIM, R_DIM)
    B_2d = B_origin.reshape(H_DIM * D_DIM, R_DIM)

    # Quantize A (per-token / per-row with block size = QUANT_SIZE)
    padded_k = ceildiv(R_DIM, QUANT_SIZE) * QUANT_SIZE
    A_padded = torch.zeros((B_DIM * H_DIM, padded_k), dtype=torch.float32)
    A_padded[:, :R_DIM] = A_2d
    A_view = A_padded.view(B_DIM * H_DIM, -1, QUANT_SIZE)
    A_amax = A_view.abs().float().amax(dim=2).view(B_DIM * H_DIM, -1).clamp(1e-4)
    sfa = ceil_to_ue8m0(A_amax / 448.0)
    A_fp8 = (
        (A_view * (1.0 / sfa.unsqueeze(2)))
        .to(torch.float8_e4m3fn)
        .view_as(A_padded)[:, :R_DIM]
        .contiguous()
    )
    sfa = sfa.to(torch.float8_e8m0fnu).contiguous()

    # Quantize B (per-block with block size = QUANT_SIZE along both N and K)
    padded_n = ceildiv(H_DIM * D_DIM, 128) * 128
    B_padded = torch.zeros((padded_n, padded_k), dtype=torch.float32)
    B_padded[: H_DIM * D_DIM, :R_DIM] = B_2d
    B_view = B_padded.view(-1, 128, padded_k // 128, 128)
    B_amax = B_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    sfb = ceil_to_ue8m0(B_amax / 448.0)
    B_fp8 = (
        (B_view * (1.0 / sfb))
        .to(torch.float8_e4m3fn)
        .view_as(B_padded)[: H_DIM * D_DIM, :R_DIM]
        .contiguous()
    )
    sfb = (
        sfb.to(torch.float8_e8m0fnu)
        .view(B_view.shape[0], B_view.shape[2])
        .repeat(QUANT_SIZE, 1)[: H_DIM * D_DIM, :]
        .contiguous()
    )

    # Pack scales to uint32
    sfa_pack = sfa.view(torch.uint32).T
    sfb_pack = sfb.view(torch.uint32).T

    # Dequantize for reference
    A_fp8_de = A_fp8.to(torch.float32)
    B_fp8_de = B_fp8.to(torch.float32)
    sfa_de = sfa.to(torch.float32)
    sfb_de = sfb.to(torch.float32)

    A_de = (A_fp8_de.reshape(B_DIM * H_DIM, R_DIM // QUANT_SIZE, QUANT_SIZE) * sfa_de[:, :, None]).reshape(
        B_DIM * H_DIM, R_DIM
    )
    B_de = (B_fp8_de.reshape(H_DIM * D_DIM, R_DIM // QUANT_SIZE, QUANT_SIZE) * sfb_de[:, :, None]).reshape(
        H_DIM * D_DIM, R_DIM
    )

    # Reference: per-head batched matmul D[b,h,d] = A[b,h,r] @ B[h,d,r].T
    # A_de is [H*B, R] (transposed), B_de is [H*D, R]
    # Kernel output is D[H*B, D] with layout [h*B+b, d]
    A_3d = A_de.reshape(H_DIM, B_DIM, R_DIM)  # [H, B, R]
    B_3d = B_de.reshape(H_DIM, D_DIM, R_DIM)  # [H, D, R]
    D_ref = torch.einsum("hbr,hdr->hbd", A_3d.float(), B_3d.float()).to(torch.bfloat16)
    D_ref = D_ref.reshape(H_DIM * B_DIM, D_DIM).contiguous()

    return A_fp8, B_fp8, sfa_pack, sfb_pack, D_ref


# fmt: off
@Tx.prim_func(tirx=True)
def fp8_bhr_hdr_bhd_gemm(
    A: Tx.Buffer((B_DIM * H_DIM, R_DIM), a_type),
    B_mat: Tx.Buffer((H_DIM * D_DIM, R_DIM), b_type),
    D: Tx.Buffer((B_DIM * H_DIM, D_DIM), d_type),
    SFA: Tx.Buffer((ceildiv(R_DIM, QUANT_SIZE) // 4, B_DIM * H_DIM), "uint32"),
    SFB: Tx.Buffer((ceildiv(R_DIM, QUANT_SIZE) // 4, H_DIM * D_DIM), "uint32"),
):
    with Tx.kernel():
        bx = Tx.cta_id([SM_NUMBER], parent="kernel")
        wg_id = Tx.warpgroup_id([WG_NUMBER], parent="cta")
        warp_id = Tx.warp_id([WARP_NUMBER], parent="warpgroup")
        tid_in_wg = Tx.thread_id([128], parent="warpgroup")
        lane_id = Tx.thread_id([32], parent="warp")
        with Tx.cta():
            # Shared memory via PoolAllocator
            pool = Tx.meta_var(Tx.PoolAllocator())
            tmem_addr = Tx.decl_scalar("uint32", pool.ptr, scope="shared.dyn", elem_offset=0)
            pool.move_base_to(8)

            # Barriers: 5 barrier groups
            tma2trans_bar = TMABar(pool, SMEM_PIPE_DEPTH)
            trans2mma_bar = MBarrier(pool, SMEM_PIPE_DEPTH)
            mma2tma_bar = TCGen05Bar(pool, SMEM_PIPE_DEPTH)
            mma2ld_bar = TCGen05Bar(pool, TMEM_PIPE_DEPTH)
            ld2mma_bar = MBarrier(pool, TMEM_PIPE_DEPTH)

            pool.move_base_to(1024)
            A_smem = pool.alloc((SMEM_PIPE_DEPTH, BLK_M, BLK_K), a_type, layout=A_layout)
            B_smem = pool.alloc((SMEM_PIPE_DEPTH, BLK_N, BLK_K), b_type, layout=B_layout)
            D_smem = pool.alloc((TMEM_PIPE_DEPTH, BLK_M, EPI_TILE), d_type, layout=D_layout)
            SFA_smem = pool.alloc((SMEM_PIPE_DEPTH, BLK_SFA // 32, 32), "uint32", layout=SFA_layout)
            SFB_smem = pool.alloc((SMEM_PIPE_DEPTH, BLK_SFB // 32, 32), "uint32", layout=SFB_layout)
            SFA_smem_2d = SFA_smem.view(SMEM_PIPE_DEPTH, BLK_SFA)
            SFB_smem_2d = SFB_smem.view(SMEM_PIPE_DEPTH, BLK_SFB)
            pool.commit()

            # Local memory
            reg = Tx.alloc_buffer((TMEM_LD_SIZE,), "float32", scope="local")
            reg_wg = reg.view(128, TMEM_LD_SIZE, layout=TileLayout(S[(128, TMEM_LD_SIZE) : (1@axis_tid_in_wg, 1)]))
            reg_fp16 = Tx.alloc_buffer((BLK_N,), d_type, scope="local")
            stage: Tx.int32
            descA: Tx.uint64
            descB: Tx.uint64
            descSFA: Tx.uint64
            descSFB: Tx.uint64
            descI: Tx.uint32

            phase: Tx.int32

            # Tile scheduler: k_tiles = H_DIM (head groups)
            tile_scheduler = GroupMajor3D(
                "tile_scheduler",
                m_tiles=TILE_M_NUM,
                n_tiles=TILE_N_NUM,
                k_tiles=H_DIM,
                group_rows=TILE_GROUPS_ROW_SIZE,
                step=SM_NUMBER,
            )

            tma2trans_bar.init(1)
            trans2mma_bar.init(CTA_GROUP * 32)  # 32
            mma2ld_bar.init(1)
            mma2tma_bar.init(1)
            ld2mma_bar.init(CTA_GROUP * 128)  # 128

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
                        main_loop(ks)
                    phase = phase ^ 1
                if PIPE_REMAIN_NUM > 0:
                    for ks in Tx.unroll(PIPE_REMAIN_NUM):
                        stage = PIPE_CYCLE * SMEM_PIPE_DEPTH + ks
                        main_loop(ks)
                    epilogue1()
                    for ks in Tx.unroll(PIPE_REMAIN_NUM, SMEM_PIPE_DEPTH):
                        epilogue2(ks)
                    phase = phase ^ 1
                else:
                    epilogue1()

            with Tx.cta():
                Tx.attr({"tirx.scope_partition": True})

                # === Producer warpgroup (WG1) ===
                with Tx.warpgroup()[wg_id == 1]:
                    Tx.attr({"tirx.scope_partition": True})

                    # --- TMA warp (warp 3): load A, B, SFA, SFB ---
                    with Tx.warp(parent="warpgroup")[warp_id == 3]:
                        phase = 0
                        while tile_scheduler.valid():
                            m_idx = Tx.meta_var(tile_scheduler.m_idx)
                            n_idx = Tx.meta_var(tile_scheduler.n_idx)
                            h_idx = Tx.meta_var(tile_scheduler.k_idx)
                            m_start = Tx.meta_var(m_idx * BLK_M)
                            n_start = Tx.meta_var(n_idx * BLK_N)
                            # A is [B*H, R]: head h starts at row h*B_DIM
                            a_m_global = Tx.meta_var(h_idx * B_DIM + m_start)
                            # B is [H*D, R]: head h starts at row h*D_DIM
                            b_n_global = Tx.meta_var(h_idx * D_DIM + n_start)

                            @Tx.inline
                            def tma_load(ks):
                                k_start = Tx.meta_var(stage * BLK_K)
                                mma2tma_bar.wait(ks, phase ^ 1)
                                tma_copy = Tx.meta_var({
                                    "dispatch": "tma",
                                    "mbar": tma2trans_bar.ptr_to([ks]),
                                    "cta_group": CTA_GROUP,
                                })
                                with Tx.thread()[Tx.ptx.elect_sync()]:
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
                                    if stage % 4 == 0:
                                        Tx.copy_async(
                                            SFA_smem_2d[ks, :],
                                            SFA[stage // 4, a_m_global : a_m_global + BLK_SFA],
                                            **tma_copy,
                                        )
                                        Tx.copy_async(
                                            SFB_smem_2d[ks, :],
                                            SFB[stage // 4, b_n_global : b_n_global + BLK_SFB],
                                            **tma_copy,
                                        )
                                    AB_bytes = Tx.meta_var(BLK_M * BLK_K * F8_BYTES + BLK_N * BLK_K * F8_BYTES)
                                    SFAB_bytes = Tx.meta_var((BLK_SFA + BLK_SFB) * F32_BYTES)
                                    tma2trans_bar.arrive(ks, Tx.if_then_else(stage % 4 == 0, AB_bytes + SFAB_bytes, AB_bytes))

                            @Tx.inline
                            def tma_load_epilogue(ks):
                                mma2tma_bar.wait(ks, phase ^ 1)
                                with Tx.thread()[Tx.ptx.elect_sync()]:
                                    tma2trans_bar.arrive(ks, 0)

                            partitioned_loop(tma_load, skip, tma_load_epilogue)
                            tile_scheduler.next_tile()

                    # --- Transposer warp (warp 2): transpose SF in SMEM ---
                    with Tx.warp(parent="warpgroup")[warp_id == 2]:
                        phase = 0
                        while tile_scheduler.valid():
                            m_idx = Tx.meta_var(tile_scheduler.m_idx)
                            n_idx = Tx.meta_var(tile_scheduler.n_idx)

                            @Tx.inline
                            def transpose(ks):
                                # Wait for TMA to finish loading data
                                tma2trans_bar.wait(ks, phase)
                                if stage % 4 == 0:
                                    Tx.permute_dims(SFA_smem[ks], [0, 2, 1])
                                    Tx.permute_dims(SFB_smem[ks], [0, 2, 1])
                                    Tx.ptx.fence.proxy_async("shared::cta")
                                # Signal transpose completion
                                trans2mma_bar.arrive(ks)

                            @Tx.inline
                            def transpose_epilogue(ks):
                                tma2trans_bar.wait(ks, phase)
                                trans2mma_bar.arrive(ks)

                            partitioned_loop(transpose, skip, transpose_epilogue)
                            tile_scheduler.next_tile()

                    # --- MMA warp (warp 0): issue block-scaled MMA ---
                    with Tx.warp(parent="warpgroup")[warp_id == 0]:
                        tmem_idx = Tx.local_scalar("int32", "tmem_idx")
                        tmem_phase = Tx.local_scalar("int32", "tmem_phase")
                        Tx.ptx.tcgen05.encode_instr_descriptor_block_scaled(
                            Tx.address_of(descI), "float32", a_type, b_type, sfa_type, sfb_type,
                            0, 0, MMA_M, MMA_N, MMA_K, False, False, CTA_GROUP,
                        )
                        phase = 0
                        while tile_scheduler.valid():
                            m_idx = Tx.meta_var(tile_scheduler.m_idx)
                            n_idx = Tx.meta_var(tile_scheduler.n_idx)
                            with Tx.thread()[Tx.ptx.elect_sync()]:
                                tmem_idx = tile_scheduler.tile_idx % TMEM_PIPE_DEPTH
                                tmem_phase = (tile_scheduler.tile_idx // TMEM_PIPE_DEPTH) & 1

                                # Wait for TMEM to be free from consumer
                                ld2mma_bar.wait(tmem_idx, tmem_phase ^ 1)
                                Tx.ptx.tcgen05.fence.after_thread_sync()

                                @Tx.inline
                                def mma(ks):
                                    # Wait for TMA + transpose completion
                                    trans2mma_bar.wait(ks, phase)
                                    Tx.ptx.tcgen05.fence.after_thread_sync()

                                    # Copy SF to TMEM at first stage of each 4-stage group
                                    if stage % 4 == 0:
                                        for ki in Tx.unroll(0, BLK_SFA // 128):
                                            Tx.ptx.tcgen05.encode_matrix_descriptor(
                                                Tx.address_of(descSFA), SFA_smem.ptr_to([ks, ki * 4, 0]),
                                                ldo=16, sdo=8 * 4 * F32_BYTES // F128_BYTES, swizzle=0,
                                            )
                                            Tx.ptx.tcgen05.cp(
                                                0, 0, SFA_TMEM_START_COL + tmem_idx * BLK_SFA // 32 + ki * 4,
                                                descSFA, "32x128b", "uint32", "uint32", CTA_GROUP, "warpx4",
                                            )
                                        for ki in Tx.unroll(0, BLK_SFB // 128):
                                            Tx.ptx.tcgen05.encode_matrix_descriptor(
                                                Tx.address_of(descSFB), SFB_smem.ptr_to([ks, ki * 4, 0]),
                                                ldo=16, sdo=8 * 4 * F32_BYTES // F128_BYTES, swizzle=0,
                                            )
                                            Tx.ptx.tcgen05.cp(
                                                0, 0, SFB_TMEM_START_COL + tmem_idx * BLK_SFB // 32 + ki * 4,
                                                descSFB, "32x128b", "uint32", "uint32", CTA_GROUP, "warpx4",
                                            )

                                    # Issue block-scaled MMA
                                    Tx.cuda.runtime_instr_desc(Tx.address_of(descI), stage % 4)
                                    for ki in Tx.unroll(BLK_K // MMA_K):
                                        Tx.ptx.tcgen05.encode_matrix_descriptor(
                                            Tx.address_of(descA), A_smem.ptr_to([ks, 0, ki * MMA_K]),
                                            ldo=1, sdo=8 * BLK_K * F8_BYTES // F128_BYTES, swizzle=SWIZZLE,
                                        )
                                        Tx.ptx.tcgen05.encode_matrix_descriptor(
                                            Tx.address_of(descB), B_smem.ptr_to([ks, 0, ki * MMA_K]),
                                            ldo=1, sdo=8 * BLK_K * F8_BYTES // F128_BYTES, swizzle=SWIZZLE,
                                        )
                                        if stage == 0 and ki == 0:
                                            Tx.ptx.tcgen05.mma.block_scale(
                                                "float32", a_type, b_type, sfa_type, sfb_type,
                                                tmem_idx * MMA_N, descA, descB,
                                                SFA_TMEM_START_COL + tmem_idx * BLK_SFA // 32,
                                                SFB_TMEM_START_COL + tmem_idx * BLK_SFB // 32,
                                                descI, False, CTA_GROUP, False,
                                            )
                                        else:
                                            Tx.ptx.tcgen05.mma.block_scale(
                                                "float32", a_type, b_type, sfa_type, sfb_type,
                                                tmem_idx * MMA_N, descA, descB,
                                                SFA_TMEM_START_COL + tmem_idx * BLK_SFA // 32,
                                                SFB_TMEM_START_COL + tmem_idx * BLK_SFB // 32,
                                                descI, False, CTA_GROUP, True,
                                            )
                                    mma2tma_bar.arrive(ks, CTA_GROUP, 1)

                                @Tx.inline
                                def mma_epilogue1():
                                    mma2ld_bar.arrive(tmem_idx, CTA_GROUP, 1)

                                @Tx.inline
                                def mma_epilogue2(ks):
                                    trans2mma_bar.wait(ks, phase)
                                    mma2tma_bar.arrive(ks, CTA_GROUP, 1)

                                partitioned_loop(mma, mma_epilogue1, mma_epilogue2)

                            tile_scheduler.next_tile()

                # === Consumer warpgroup (WG0): TMEM -> SMEM -> GMEM ===
                with Tx.warpgroup()[wg_id == 0]:
                    Tx.cuda.trap_when_assert_failed(tmem_addr == 0)
                    tmem_idx = Tx.local_scalar("int32", "tmem_idx")
                    tmem_phase = Tx.local_scalar("int32", "tmem_phase")
                    phase = 0
                    while tile_scheduler.valid():
                        m_idx = Tx.meta_var(tile_scheduler.m_idx)
                        n_idx = Tx.meta_var(tile_scheduler.n_idx)
                        h_idx = Tx.meta_var(tile_scheduler.k_idx)
                        tmem_idx = tile_scheduler.tile_idx % TMEM_PIPE_DEPTH
                        tmem_phase = (tile_scheduler.tile_idx // TMEM_PIPE_DEPTH) & 1

                        # Wait previous TMA store
                        if tid_in_wg == 0:
                            Tx.ptx.cp_async.bulk.wait_group(0)
                        Tx.cuda.warpgroup_sync(10)

                        # Wait MMA completion
                        mma2ld_bar.wait(tmem_idx, tmem_phase)
                        Tx.ptx.tcgen05.fence.after_thread_sync()

                        for ko in Tx.unroll(MMA_N // EPI_TILE):
                            stage = (tile_scheduler.tile_idx * MMA_N // EPI_TILE + ko) % TMEM_PIPE_DEPTH

                            if ko >= TMEM_PIPE_DEPTH:
                                if tid_in_wg == 0:
                                    Tx.ptx.cp_async.bulk.wait_group(TMEM_PIPE_DEPTH - 1)
                                Tx.cuda.warpgroup_sync(10)

                            # TMEM -> RF (f32) -> cast to bf16 -> SMEM
                            for ki in Tx.unroll(EPI_TILE // TMEM_LD_SIZE):
                                col_st = Tx.meta_var(tmem_idx * MMA_N + ko * EPI_TILE + ki * TMEM_LD_SIZE)
                                Tx.copy(reg_wg[:, :], tmem[:, col_st : col_st + TMEM_LD_SIZE])
                                with Tx.thread():
                                    st = Tx.meta_var(ki * TMEM_LD_SIZE)
                                    Tx.cast(reg_fp16[st : st + TMEM_LD_SIZE], reg[:])
                                    Tx.copy(
                                        D_smem[stage, warp_id * 32 + lane_id, st : st + TMEM_LD_SIZE],
                                        reg_fp16[st : st + TMEM_LD_SIZE],
                                    )

                            # Signal TMEM is free after last chunk
                            if ko == MMA_N // EPI_TILE - 1:
                                Tx.ptx.tcgen05.fence.before_thread_sync()
                                ld2mma_bar.arrive(tmem_idx)

                            Tx.ptx.fence.proxy_async("shared::cta")
                            Tx.cuda.warpgroup_sync(10)

                            # SMEM -> GMEM via TMA store
                            # D is [B*H, D]: head h row = h*B_DIM + m_start
                            m_st: Tx.let = h_idx * B_DIM + m_idx * BLK_M
                            n_st: Tx.let = n_idx * BLK_N + ko * EPI_TILE
                            with Tx.thread(parent="warpgroup")[tid_in_wg == 0]:
                                Tx.copy_async(
                                    D[m_st : m_st + BLK_M, n_st : n_st + EPI_TILE],
                                    D_smem[stage, :, :],
                                    dispatch="tma",
                                )
                                Tx.ptx.cp_async.bulk.commit_group()

                        tile_scheduler.next_tile()

                    if tid_in_wg == 0:
                        Tx.ptx.cp_async.bulk.wait_group(0)
                    Tx.cuda.warpgroup_sync(10)

            # Dealloc TMEM
            with Tx.warp()[0:1]:
                Tx.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
                Tx.ptx.tcgen05.dealloc(tmem_addr, n_cols=N_COLS, cta_group=1)

            Tx.cuda.cta_sync()
# fmt: on


@tvm.testing.requires_cuda_compute_version(10, exact=True)
def test_fp8_bhr_hdr_bhd():
    DEV = tvm.cuda(0)
    A_fp8, B_fp8, sfa_pack, sfb_pack, D_ref = prepare_data()

    A_tvm = tvm.runtime.tensor(
        A_fp8.view(torch.int8).numpy().view(ml_dtypes.float8_e4m3fn), device=DEV
    )
    B_tvm = tvm.runtime.tensor(
        B_fp8.view(torch.int8).numpy().view(ml_dtypes.float8_e4m3fn), device=DEV
    )
    sfa_tvm = tvm.runtime.tensor(sfa_pack.numpy(), device=DEV)
    sfb_tvm = tvm.runtime.tensor(sfb_pack.numpy(), device=DEV)
    D_out = torch.empty((B_DIM * H_DIM, D_DIM), dtype=torch.bfloat16, device="cuda")

    target = tvm.target.Target("cuda")
    with target:
        src, mod = get_source(fp8_bhr_hdr_bhd_gemm)
        func = lambda: mod(A_tvm, B_tvm, D_out, sfa_tvm, sfb_tvm)
        ms = bench(func, warmup=10, repeat=30, proton_name="tir")
        tflops = flops(ms) / 1e12
        print(f"TIR: {tflops:.2f} TFLOPS, time: {ms:.3f} ms")

    D_ref_cuda = D_ref.to("cuda")
    diff = calc_diff(D_out, D_ref_cuda)
    print(f"calc_diff: {diff:.6f}")
    assert diff < 2e-3, f"Correctness check failed: calc_diff={diff}"
    print("Test passed!")


def bench_fp8_bhr_hdr_bhd():
    import deep_gemm
    from deep_gemm.utils.math import per_block_cast_to_fp8, per_token_cast_to_fp8

    DEV = tvm.cuda(0)
    A_fp8, B_fp8, sfa_pack, sfb_pack, D_ref = prepare_data()

    A_tvm = tvm.runtime.tensor(
        A_fp8.view(torch.int8).numpy().view(ml_dtypes.float8_e4m3fn), device=DEV
    )
    B_tvm = tvm.runtime.tensor(
        B_fp8.view(torch.int8).numpy().view(ml_dtypes.float8_e4m3fn), device=DEV
    )
    sfa_tvm = tvm.runtime.tensor(sfa_pack.numpy(), device=DEV)
    sfb_tvm = tvm.runtime.tensor(sfb_pack.numpy(), device=DEV)

    target = tvm.target.Target("cuda")
    with target:
        src, mod = get_source(fp8_bhr_hdr_bhd_gemm)

    def std():
        # DeepGEMM fp8_einsum with same quantization as test_einsum.py
        A_bf16 = torch.randn((B_DIM, H_DIM, R_DIM), dtype=torch.bfloat16, device="cuda")
        B_bf16 = torch.randn((H_DIM, D_DIM, R_DIM), dtype=torch.bfloat16, device="cuda")
        x_fp8 = per_token_cast_to_fp8(A_bf16.view(-1, R_DIM), use_ue8m0=True)
        x_fp8 = x_fp8[0].view(B_DIM, H_DIM, R_DIM), x_fp8[1].view(B_DIM, H_DIM, ceildiv(R_DIM, 128))
        y_fp8 = (
            torch.empty_like(B_bf16, dtype=torch.float8_e4m3fn),
            torch.empty((H_DIM, ceildiv(D_DIM, 128), ceildiv(R_DIM, 128)), device="cuda", dtype=torch.float),
        )
        for i in range(H_DIM):
            y_fp8[0][i], y_fp8[1][i] = per_block_cast_to_fp8(B_bf16[i], use_ue8m0=True)
        z = torch.empty((B_DIM, H_DIM, D_DIM), dtype=torch.bfloat16, device="cuda")
        ms = bench(
            lambda: deep_gemm.fp8_einsum("bhr,hdr->bhd", x_fp8, y_fp8, z),
            warmup=50,
            repeat=50,
            proton_name="std",
        )
        return ms

    def tir():
        D_out = torch.empty((B_DIM * H_DIM, D_DIM), dtype=torch.bfloat16, device="cuda")
        ms = bench(
            lambda: mod(A_tvm, B_tvm, D_out, sfa_tvm, sfb_tvm),
            warmup=50,
            repeat=50,
            proton_name="tir",
        )
        D_ref_cuda = D_ref.to("cuda")
        diff = calc_diff(D_out, D_ref_cuda)
        return ms, diff

    with ProtonContext():
        tir_ms, tir_diff = tir()
        print(f"TIR flops: {flops(tir_ms) / 1e12:.2f} TFLOPS, time: {tir_ms:.3f} ms")
        print(f"calc_diff(tir, ref): {tir_diff:.6f}")
        std_ms = std()
        print(f"Std flops: {flops(std_ms) / 1e12:.2f} TFLOPS, time: {std_ms:.3f} ms")
        assert tir_diff < 2e-3, f"Correctness check failed: calc_diff={tir_diff}"
        print("Benchmark passed!")


if __name__ == "__main__":
    bench_fp8_bhr_hdr_bhd()
