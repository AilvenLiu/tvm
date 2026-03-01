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

"""Causal linear attention kernel with SM100 warp specialization.

Warp-specialized rewrite using tcgen05 MMA, TMEM, and producer/consumer
warp separation. 9 MMA phases per chunk with all matrix multiplications
offloaded to tcgen05 MMA (Q@K^T, att@V_lo, att@V_hi, Q_dec@kvs×2,
K_dec^T@V×4).

Architecture:
  cta_group=1 (single CTA, no clustering)
  MMA_M=64, MMA_N=64, MMA_K=16 (f16)
  WG1 (Producer+MMA): warp 3 = TMA, warps 0-1 = MMA
  WG0 (Consumer): TMEM read, causal mask + decay, staging prep

Per-chunk MMA phases:
  Phase 1:  Q@K^T → att (descI, transA=F, transB=F)
  Phase 2a: att@V_lo → o_lo (descI_tb, transA=F, transB=T)
  Phase 2b: att@V_hi → o_hi (descI_tb, transA=F, transB=T)
  Phase 3:  Q_dec@kvs_lo → o_lo inter (descI_tb, transA=F, transB=T, 8 K-iters)
  Phase 4:  Q_dec@kvs_hi → o_hi inter (descI_tb, 8 K-iters)
  Phase 5:  K_lo_dec^T@V_lo → kvs[0:64,0:64] (descI_state, transA=T, transB=T)
  Phase 6:  K_lo_dec^T@V_hi → kvs[0:64,64:128] (descI_state)
  Phase 7:  K_hi_dec^T@V_lo → kvs[64:128,0:64] (descI_state)
  Phase 8:  K_hi_dec^T@V_hi → kvs[64:128,64:128] (descI_state)

Key challenge: F=128 and D=128 exceed TMEM N_COLS=64 limit.
  - TMA SWIZZLE_128B requires inner dim <= 128B (64 f16 elements).
    So all 128-wide buffers (Q, K, V, D_out) are split into _lo/_hi
    halves of [64,64] each, loaded/stored as separate TMA ops.
  - Q@K^T -> [64,64]: fits in TMEM. K-dim=128 -> 8 MMA_K iterations.
  - att@V -> split into V_lo/V_hi, two MMA phases.
  - Q@kvs -> staging_a + staging_b via SMEM MMA (Phases 3-4).
  - K_dec^T@V -> offloaded to MMA via staging buffers (Phases 5-8).
"""

import math
import pytest
import numpy as np
import torch

import tvm
import tvm.testing
from tvm.ir import PointerType, PrimType
from tvm.script import tirx as Tx
from tvm.tir.layout import TileLayout, tid_in_wg, TLane, TCol, S
from tvm.tirx.bench.utils import ProtonContext, bench
from tvm.tirx.pipeline import MBarrier, TMABar, TCGen05Bar

# Constants
CHUNK = 64
F = 128       # Q/K feature dimension
D = 128       # V/O output dimension
HALF = 64     # half of F and D for TMA/MMA splitting
SM_COUNT = 148

# Warp-specialized layout
WG_NUMBER = 2        # WG0=consumer, WG1=producer+MMA
WARP_NUMBER = 4      # per warpgroup
NUM_THREADS = (32 * WARP_NUMBER) * WG_NUMBER  # 256

# MMA dimensions
MMA_M = 64
TMEM_ROWS = 128
MMA_N = 64
MMA_K = 16
CTA_GROUP = 1
SWIZZLE = 3  # 128B swizzle (64 f16 * 2B = 128B per row)

# Pipeline
PIPE_DEPTH = 2

# Memory sizes
F16_BYTES = 2
F32_BYTES = 4
F128_BYTES = 16
N_COLS = 64          # MMA output width (f32 columns for results)
TMEM_LD_SIZE = 64

# SMEM budget (bytes):
#   barriers=1024
#   Q_lo[2,64,64]*f16=16K + Q_hi[2,64,64]*f16=16K
#   K_lo[2,64,64]*f16=16K + K_hi[2,64,64]*f16=16K
#   V_lo[2,64,64]*f16=16K + V_hi[2,64,64]*f16=16K
#   work[64,64]*f16=8K
#   D_out_lo[64,64]*f16=8K + D_out_hi[64,64]*f16=8K
#   kvs[128,128]*f32=64K + slopes_s[1]*f32=4
#   staging_a_lo[64,64]*f16=8K + staging_a_hi[64,64]*f16=8K
#   staging_b_lo[64,64]*f16=8K + staging_b_hi[64,64]*f16=8K
#   + 1024B alignment padding for staging buffers (~1K)
#   Total ~218K


def ceildiv(a, b):
    return (a + b - 1) // b


def prepare_data(batch, heads, seq_len):
    q = torch.randn(batch * heads, seq_len, F, dtype=torch.float16, device="cuda") / (F ** 0.5)
    k = torch.randn(batch * heads, seq_len, F, dtype=torch.float16, device="cuda") / (F ** 0.5)
    v = torch.randn(batch * heads, seq_len, D, dtype=torch.float16, device="cuda")
    slopes = torch.rand(batch * heads, dtype=torch.float32, device="cuda")
    return q, k, v, slopes


def naive_linear_attention(q, k, v, slopes):
    """Reference: causal linear attention with exponential decay, chunked."""
    BH, L, _ = q.shape
    NC = L // CHUNK
    qf = q.reshape(BH, NC, CHUNK, F).float()
    kf = k.reshape(BH, NC, CHUNK, F).float()
    vf = v.reshape(BH, NC, CHUNK, D).float()

    o = torch.zeros(BH, NC, CHUNK, D, dtype=torch.float32, device=q.device)
    kv_state = torch.zeros(BH, F, D, dtype=torch.float32, device=q.device)

    # Causal mask with decay for intra-chunk
    idx = torch.arange(CHUNK, device=q.device).float()
    diff = idx.unsqueeze(1) - idx.unsqueeze(0)  # (CHUNK, CHUNK)
    causal = (diff >= 0).float()

    for c in range(NC):
        qc, kc, vc = qf[:, c], kf[:, c], vf[:, c]
        sl = slopes.unsqueeze(-1).unsqueeze(-1)  # (BH, 1, 1)

        # Intra-chunk: att = Q @ K^T with causal decay
        decay = torch.exp(-sl * diff.unsqueeze(0)) * causal.unsqueeze(0)  # (BH, CHUNK, CHUNK)
        att = torch.matmul(qc, kc.transpose(-2, -1)) * decay
        o_intra = torch.matmul(att, vc)

        # Inter-chunk: Q @ kv_state with q_decay
        q_decay = torch.exp(-slopes.unsqueeze(-1) * idx.unsqueeze(0))  # (BH, CHUNK)
        o_inter = torch.matmul(qc, kv_state) * q_decay.unsqueeze(-1)

        o[:, c] = o_intra + o_inter

        # Update KV state: decay + K_dec^T @ V
        block_decay = torch.exp(-slopes * CHUNK)  # (BH,)
        k_decay = torch.exp(-slopes.unsqueeze(-1) * (CHUNK - idx.unsqueeze(0)))  # (BH, CHUNK)
        k_dec = kc * k_decay.unsqueeze(-1)  # (BH, CHUNK, F)
        kv_state = kv_state * block_decay.unsqueeze(-1).unsqueeze(-1) + torch.matmul(k_dec.transpose(-2, -1), vc)

    return o.reshape(BH, L, D).half()


# ---- Layouts ----
# All buffers use SWIZZLE_128B (3) with 64 f16 = 128B inner dimension.
# This satisfies TMA requirement of inner dim <= 128B for SWIZZLE_128B.

half_pipe_layout = Tx.ComposeLayout(
    Tx.SwizzleLayout(SWIZZLE, SWIZZLE, SWIZZLE, swizzle_inner=True),
    Tx.TileLayout(S[(PIPE_DEPTH, CHUNK, HALF) : (CHUNK * HALF, HALF, 1)]),
)
half_layout = Tx.ComposeLayout(
    Tx.SwizzleLayout(SWIZZLE, SWIZZLE, SWIZZLE, swizzle_inner=True),
    Tx.TileLayout(S[(CHUNK, HALF) : (HALF, 1)]),
)


def get_linear_attention_kernel():
    a_type = tvm.DataType("float16")

    # fmt: off
    @Tx.prim_func(tirx=True)
    def linear_attention(q_ptr: Tx.handle, k_ptr: Tx.handle, v_ptr: Tx.handle,
                         slopes_ptr: Tx.handle, o_ptr: Tx.handle):
        total_bh = Tx.int32()
        seq_len = Tx.int32()
        q_g = Tx.match_buffer(q_ptr, [total_bh, seq_len, F], "float16", scope="global")
        k_g = Tx.match_buffer(k_ptr, [total_bh, seq_len, F], "float16", scope="global")
        v_g = Tx.match_buffer(v_ptr, [total_bh, seq_len, D], "float16", scope="global")
        slopes_g = Tx.match_buffer(slopes_ptr, [total_bh], "float32", scope="global")
        o_g = Tx.match_buffer(o_ptr, [total_bh, seq_len, D], "float16", scope="global")

        with Tx.kernel():
            bx = Tx.cta_id([SM_COUNT], parent="kernel")
            wg_id = Tx.warpgroup_id([WG_NUMBER], parent="cta")
            warp_id = Tx.warp_id([WARP_NUMBER], parent="warpgroup")
            lane_id = Tx.thread_id([32], parent="warp")

            with Tx.cta():
                # ---- Shared memory via PoolAllocator ----
                pool = Tx.meta_var(Tx.PoolAllocator())
                tmem_addr = Tx.decl_scalar("uint32", pool.ptr, scope="shared.dyn", elem_offset=0)
                pool.move_base_to(8)

                # Barriers
                tma2mma_bar = TMABar(pool, PIPE_DEPTH)
                mma2tma_bar = TCGen05Bar(pool, PIPE_DEPTH)
                mma2consumer_bar = TCGen05Bar(pool, 1)
                consumer2mma_bar = MBarrier(pool, 1)
                workitem_sync_bar = TMABar(pool, 1)

                pool.move_base_to(1024)
                Q_lo = pool.alloc((PIPE_DEPTH, CHUNK, HALF), "float16", layout=half_pipe_layout)
                Q_hi = pool.alloc((PIPE_DEPTH, CHUNK, HALF), "float16", layout=half_pipe_layout)
                K_lo = pool.alloc((PIPE_DEPTH, CHUNK, HALF), "float16", layout=half_pipe_layout)
                K_hi = pool.alloc((PIPE_DEPTH, CHUNK, HALF), "float16", layout=half_pipe_layout)
                V_lo = pool.alloc((PIPE_DEPTH, CHUNK, HALF), "float16", layout=half_pipe_layout)
                V_hi = pool.alloc((PIPE_DEPTH, CHUNK, HALF), "float16", layout=half_pipe_layout)
                work_smem = pool.alloc((CHUNK, HALF), "float16", layout=half_layout)
                D_out_lo = pool.alloc((CHUNK, HALF), "float16", layout=half_layout)
                D_out_hi = pool.alloc((CHUNK, HALF), "float16", layout=half_layout)

                # Non-swizzled buffers
                kvs = pool.alloc((F, D), "float32")
                slopes_s = pool.alloc((1,), "float32")

                # Staging buffers (1024B-aligned for MMA swizzle)
                staging_a_lo = pool.alloc((HALF, HALF), "float16", layout=half_layout, align=1024)
                staging_a_hi = pool.alloc((HALF, HALF), "float16", layout=half_layout)
                staging_b_lo = pool.alloc((HALF, HALF), "float16", layout=half_layout)
                staging_b_hi = pool.alloc((HALF, HALF), "float16", layout=half_layout)
                pool.commit()

                # ---- Local variables ----
                descI: Tx.uint32
                descA: Tx.uint64
                descB: Tx.uint64
                phase = Tx.alloc_buffer((1,), "int32", scope="local")

                tma2mma_bar.init(1)
                mma2tma_bar.init(1)
                mma2consumer_bar.init(1)
                consumer2mma_bar.init(128)  # full consumer WG
                workitem_sync_bar.init(1)

                ptr: Tx.let[Tx.Var(name="ptr", dtype=PointerType(PrimType("uint64")))] = Tx.reinterpret("handle", Tx.ptx.map_shared_rank(tma2mma_bar.ptr_to([0]), 0))
                tma_finished = Tx.decl_buffer([PIPE_DEPTH], "uint64", data=ptr, scope="shared")

                # ---- TMEM allocation ----
                with Tx.warp()[0:1]:
                    Tx.ptx.tcgen05.alloc(Tx.address_of(tmem_addr), n_cols=N_COLS, cta_group=CTA_GROUP)
                    Tx.cuda.warp_sync()

                Tx.ptx.fence.proxy_async("shared::cta")
                Tx.ptx.fence.mbarrier_init()
                Tx.cuda.cta_sync()
                Tx.cuda.trap_when_assert_failed(tmem_addr == 0)
                tmem = Tx.decl_buffer((TMEM_ROWS, N_COLS), "float32", scope="tmem",
                                      allocated_addr=0,
                                      layout=TileLayout(S[(TMEM_ROWS, N_COLS) : (1@TLane, 1@TCol)]))
                # ---- Consumer TMEM read setup ----
                reg = Tx.alloc_buffer((TMEM_LD_SIZE,), "float32", scope="local")
                reg_wg = reg.view(128, TMEM_LD_SIZE,
                                  layout=TileLayout(S[(128, TMEM_LD_SIZE) : (1@tid_in_wg, 1)]))

                # MMA descriptor params (all buffers 64-col with SWIZZLE_128B)
                MMA_LDO = 1
                MMA_SDO = 8 * HALF * F16_BYTES // F128_BYTES  # 8*64*2/16 = 64

                # ---- Warp-specialized main body ----
                with Tx.cta():
                    Tx.attr({"tirx.scope_partition": True})

                    # === Producer warpgroup (WG1): TMA + MMA warps ===
                    with Tx.warpgroup()[1:2]:
                        if warp_id == 3:
                            # ---- TMA producer warp ----
                            phase[0] = 0
                            wi_phase = Tx.alloc_buffer((1,), "int32", scope="local")
                            wi_phase[0] = 0
                            wid_tma = Tx.local_scalar("int32", "wid_tma")
                            cid_tma = Tx.local_scalar("int32", "cid_tma")
                            nc_tma = Tx.local_scalar("int32", "nc_tma")
                            wid_tma = bx
                            while wid_tma < total_bh:
                                if wid_tma != bx:
                                    with Tx.thread()[Tx.ptx.elect_sync()]:
                                        workitem_sync_bar.wait(0, wi_phase[0])
                                    wi_phase[0] = wi_phase[0] ^ 1
                                nc_tma = seq_len // CHUNK
                                cid_tma = 0
                                while cid_tma < nc_tma:
                                    with Tx.thread()[Tx.ptx.elect_sync()]:
                                        tic = Tx.meta_var(cid_tma % PIPE_DEPTH)
                                        q_row = Tx.meta_var(cid_tma * CHUNK)
                                        tma_copy = Tx.meta_var({"dispatch": "tma", "mbar": tma_finished.ptr_to([tic]), "cta_group": CTA_GROUP})
                                        # Wait for MMA to release this SMEM slot
                                        if cid_tma >= PIPE_DEPTH:
                                            mma2tma_bar.wait(tic, phase[0] ^ 1)
                                        # Load Q, K, V halves (6 TMA loads per chunk)
                                        Tx.copy_async(Q_lo[tic, :, :], q_g[wid_tma, q_row : q_row + CHUNK, 0 : HALF], **tma_copy)
                                        Tx.copy_async(Q_hi[tic, :, :], q_g[wid_tma, q_row : q_row + CHUNK, HALF : F], **tma_copy)
                                        Tx.copy_async(K_lo[tic, :, :], k_g[wid_tma, q_row : q_row + CHUNK, 0 : HALF], **tma_copy)
                                        Tx.copy_async(K_hi[tic, :, :], k_g[wid_tma, q_row : q_row + CHUNK, HALF : F], **tma_copy)
                                        Tx.copy_async(V_lo[tic, :, :], v_g[wid_tma, q_row : q_row + CHUNK, 0 : HALF], **tma_copy)
                                        Tx.copy_async(V_hi[tic, :, :], v_g[wid_tma, q_row : q_row + CHUNK, HALF : D], **tma_copy)
                                        # Signal that load is done (total bytes for 6 halves)
                                        tma2mma_bar.arrive(tic, 6 * CHUNK * HALF * F16_BYTES)
                                    if cid_tma % PIPE_DEPTH == PIPE_DEPTH - 1:
                                        phase[0] = phase[0] ^ 1
                                    cid_tma = cid_tma + 1
                                wid_tma = wid_tma + SM_COUNT

                        elif warp_id == 0:
                            # ---- MMA warp ----
                            # Phase 1: Q@K^T — transA=False, transB=False
                            Tx.ptx.tcgen05.encode_instr_descriptor(
                                Tx.address_of(descI), "float32", "float16", "float16",
                                MMA_M * CTA_GROUP, MMA_N, MMA_K, False, False, CTA_GROUP)

                            # Phases 2a/2b/3/4: transA=False, transB=True
                            descI_tb: Tx.uint32
                            Tx.ptx.tcgen05.encode_instr_descriptor(
                                Tx.address_of(descI_tb), "float32", "float16", "float16",
                                MMA_M * CTA_GROUP, MMA_N, MMA_K, False, True, CTA_GROUP)

                            # Phases 5-8: K_dec^T@V — transA=True, transB=True
                            descI_state: Tx.uint32
                            Tx.ptx.tcgen05.encode_instr_descriptor(
                                Tx.address_of(descI_state), "float32", "float16", "float16",
                                MMA_M * CTA_GROUP, MMA_N, MMA_K, True, True, CTA_GROUP)

                            # staging_b_lo/hi are [64,64] with same SDO as other half buffers

                            phase[0] = 0
                            wid_mma = Tx.local_scalar("int32", "wid_mma")
                            cid_mma = Tx.local_scalar("int32", "cid_mma")
                            nc_mma = Tx.local_scalar("int32", "nc_mma")
                            phase_c2m: Tx.int32
                            phase_c2m = 0
                            wid_mma = bx
                            while wid_mma < total_bh:
                                nc_mma = seq_len // CHUNK
                                cid_mma = 0
                                while cid_mma < nc_mma:
                                    with Tx.thread()[Tx.ptx.elect_sync()]:
                                        tic = Tx.meta_var(cid_mma % PIPE_DEPTH)

                                        # == Phase 1: Q @ K^T -> att (in TMEM) ==
                                        tma2mma_bar.wait(tic, phase[0])
                                        Tx.ptx.tcgen05.fence.after_thread_sync()
                                        for ki in Tx.unroll(HALF // MMA_K):
                                            Tx.ptx.tcgen05.encode_matrix_descriptor(
                                                Tx.address_of(descA), Q_lo.ptr_to([tic, 0, ki * MMA_K]),
                                                MMA_LDO, MMA_SDO, SWIZZLE)
                                            Tx.ptx.tcgen05.encode_matrix_descriptor(
                                                Tx.address_of(descB), K_lo.ptr_to([tic, 0, ki * MMA_K]),
                                                MMA_LDO, MMA_SDO, SWIZZLE)
                                            Tx.ptx.tcgen05.mma("float32", "float16", "float16",
                                                Tx.cuda.get_tmem_addr(tmem_addr, 0, 0),
                                                descA, descB, descI, False, CTA_GROUP, ki > 0)
                                        for ki in Tx.unroll(HALF // MMA_K):
                                            Tx.ptx.tcgen05.encode_matrix_descriptor(
                                                Tx.address_of(descA), Q_hi.ptr_to([tic, 0, ki * MMA_K]),
                                                MMA_LDO, MMA_SDO, SWIZZLE)
                                            Tx.ptx.tcgen05.encode_matrix_descriptor(
                                                Tx.address_of(descB), K_hi.ptr_to([tic, 0, ki * MMA_K]),
                                                MMA_LDO, MMA_SDO, SWIZZLE)
                                            Tx.ptx.tcgen05.mma("float32", "float16", "float16",
                                                Tx.cuda.get_tmem_addr(tmem_addr, 0, 0),
                                                descA, descB, descI, False, CTA_GROUP, True)
                                        mma2consumer_bar.arrive(0, CTA_GROUP, 1)

                                        # Wait consumer done with Phase 1
                                        consumer2mma_bar.wait(0, phase_c2m)
                                        Tx.ptx.tcgen05.fence.after_thread_sync()

                                        # == Phase 2a: work @ V_lo -> o_intra_lo ==
                                        for ki in Tx.unroll(CHUNK // MMA_K):
                                            Tx.ptx.tcgen05.encode_matrix_descriptor(
                                                Tx.address_of(descA), work_smem.ptr_to([0, ki * MMA_K]),
                                                MMA_LDO, MMA_SDO, SWIZZLE)
                                            Tx.ptx.tcgen05.encode_matrix_descriptor(
                                                Tx.address_of(descB), V_lo.ptr_to([tic, ki * MMA_K, 0]),
                                                MMA_LDO, MMA_SDO, SWIZZLE)
                                            Tx.ptx.tcgen05.mma("float32", "float16", "float16",
                                                Tx.cuda.get_tmem_addr(tmem_addr, 0, 0),
                                                descA, descB, descI_tb, False, CTA_GROUP, ki > 0)
                                        mma2consumer_bar.arrive(0, CTA_GROUP, 1)

                                        # Wait consumer done with Phase 2a
                                        consumer2mma_bar.wait(0, phase_c2m ^ 1)
                                        Tx.ptx.tcgen05.fence.after_thread_sync()

                                        # == Phase 2b: work @ V_hi -> o_intra_hi ==
                                        for ki in Tx.unroll(CHUNK // MMA_K):
                                            Tx.ptx.tcgen05.encode_matrix_descriptor(
                                                Tx.address_of(descA), work_smem.ptr_to([0, ki * MMA_K]),
                                                MMA_LDO, MMA_SDO, SWIZZLE)
                                            Tx.ptx.tcgen05.encode_matrix_descriptor(
                                                Tx.address_of(descB), V_hi.ptr_to([tic, ki * MMA_K, 0]),
                                                MMA_LDO, MMA_SDO, SWIZZLE)
                                            Tx.ptx.tcgen05.mma("float32", "float16", "float16",
                                                Tx.cuda.get_tmem_addr(tmem_addr, 0, 0),
                                                descA, descB, descI_tb, False, CTA_GROUP, ki > 0)
                                        mma2consumer_bar.arrive(0, CTA_GROUP, 1)

                                        # Wait consumer done with Phase 2b
                                        consumer2mma_bar.wait(0, phase_c2m)
                                        Tx.ptx.tcgen05.fence.after_thread_sync()

                                        # == Phases 3-4: Q_dec @ kvs_f16 -> o_inter (SMEM-based) ==
                                        # A: staging_a_lo/hi (Q_dec) [M=64, K=64] transA=False
                                        # B: staging_b_lo/hi (kvs f16) [K=64, N=64] transB=True
                                        for phase_i in Tx.unroll(2):
                                            # First half: Q_lo_dec[64,64] @ staging_b (K=64)
                                            for ki in Tx.unroll(HALF // MMA_K):
                                                Tx.ptx.tcgen05.encode_matrix_descriptor(
                                                    Tx.address_of(descA), staging_a_lo.ptr_to([0, ki * MMA_K]),
                                                    MMA_LDO, MMA_SDO, SWIZZLE)
                                                Tx.ptx.tcgen05.encode_matrix_descriptor(
                                                    Tx.address_of(descB), staging_b_lo.ptr_to([ki * MMA_K, 0]),
                                                    MMA_LDO, MMA_SDO, SWIZZLE)
                                                Tx.ptx.tcgen05.mma("float32", "float16", "float16",
                                                    Tx.cuda.get_tmem_addr(tmem_addr, 0, 0),
                                                    descA, descB, descI_tb, False, CTA_GROUP, ki > 0)
                                            # Second half: Q_hi_dec[64,64] @ staging_b (K=64, accumulate)
                                            for ki in Tx.unroll(HALF // MMA_K):
                                                Tx.ptx.tcgen05.encode_matrix_descriptor(
                                                    Tx.address_of(descA), staging_a_hi.ptr_to([0, ki * MMA_K]),
                                                    MMA_LDO, MMA_SDO, SWIZZLE)
                                                Tx.ptx.tcgen05.encode_matrix_descriptor(
                                                    Tx.address_of(descB), staging_b_hi.ptr_to([ki * MMA_K, 0]),
                                                    MMA_LDO, MMA_SDO, SWIZZLE)
                                                Tx.ptx.tcgen05.mma("float32", "float16", "float16",
                                                    Tx.cuda.get_tmem_addr(tmem_addr, 0, 0),
                                                    descA, descB, descI_tb, False, CTA_GROUP, True)
                                            mma2consumer_bar.arrive(0, CTA_GROUP, 1)

                                            consumer2mma_bar.wait(0, phase_c2m ^ (1 - phase_i % 2))
                                            Tx.ptx.tcgen05.fence.after_thread_sync()

                                        # == Phases 5-8: K_dec^T @ V -> kvs update ==
                                        # staging_a_lo holds K_lo_dec[64,64], staging_a_hi holds K_hi_dec[64,64]
                                        # staging_a is swizzled (half_layout), use SWIZZLE in descriptor
                                        for state_i in Tx.unroll(4):
                                            for ki in Tx.unroll(CHUNK // MMA_K):
                                                # A: staging_a_lo (state_i<2) or staging_a_hi (state_i>=2)
                                                # transA=True: A is [K=pos, M=f], iterate K dim rows
                                                if state_i < 2:
                                                    Tx.ptx.tcgen05.encode_matrix_descriptor(
                                                        Tx.address_of(descA), staging_a_lo.ptr_to([ki * MMA_K, 0]),
                                                        MMA_LDO, MMA_SDO, SWIZZLE)
                                                else:
                                                    Tx.ptx.tcgen05.encode_matrix_descriptor(
                                                        Tx.address_of(descA), staging_a_hi.ptr_to([ki * MMA_K, 0]),
                                                        MMA_LDO, MMA_SDO, SWIZZLE)
                                                # B: V_lo (state_i even) or V_hi (state_i odd)
                                                if state_i % 2 == 0:
                                                    Tx.ptx.tcgen05.encode_matrix_descriptor(
                                                        Tx.address_of(descB), V_lo.ptr_to([tic, ki * MMA_K, 0]),
                                                        MMA_LDO, MMA_SDO, SWIZZLE)
                                                else:
                                                    Tx.ptx.tcgen05.encode_matrix_descriptor(
                                                        Tx.address_of(descB), V_hi.ptr_to([tic, ki * MMA_K, 0]),
                                                        MMA_LDO, MMA_SDO, SWIZZLE)
                                                Tx.ptx.tcgen05.mma("float32", "float16", "float16",
                                                    Tx.cuda.get_tmem_addr(tmem_addr, 0, 0),
                                                    descA, descB, descI_state, False, CTA_GROUP, ki > 0)
                                            mma2consumer_bar.arrive(0, CTA_GROUP, 1)

                                            consumer2mma_bar.wait(0, phase_c2m ^ (1 - state_i % 2))
                                            Tx.ptx.tcgen05.fence.after_thread_sync()

                                        # Release SMEM slot AFTER Phase 8 (V needed through Phase 8)
                                        Tx.ptx.mbarrier.arrive(mma2tma_bar.ptr_to([tic]))
                                        # 9 phases = odd transitions -> flip phase_c2m each chunk
                                        phase_c2m = phase_c2m ^ 1

                                    if cid_mma % PIPE_DEPTH == PIPE_DEPTH - 1:
                                        phase[0] = phase[0] ^ 1
                                    cid_mma = cid_mma + 1
                                with Tx.thread()[Tx.ptx.elect_sync()]:
                                    Tx.ptx.mbarrier.arrive(workitem_sync_bar.ptr_to([0]))
                                wid_mma = wid_mma + SM_COUNT

                    # === Consumer warpgroup (WG0) ===
                    with Tx.warpgroup()[0:1]:
                        Tx.cuda.trap_when_assert_failed(tmem_addr == 0)

                        # Per-thread local storage
                        o_lo_reg = Tx.alloc_buffer((HALF,), "float32", scope="local")
                        o_hi_reg = Tx.alloc_buffer((HALF,), "float32", scope="local")
                        phase_m2c: Tx.int32
                        phase_m2c = 0
                        phase_c2m_c: Tx.int32
                        phase_c2m_c = 0

                        wid_con = Tx.local_scalar("int32", "wid_con")
                        cid_con = Tx.local_scalar("int32", "cid_con")
                        nc_con = Tx.local_scalar("int32", "nc_con")
                        wid_con = bx
                        while wid_con < total_bh:
                            nc_con = seq_len // CHUNK

                            # Load slope into SMEM
                            with Tx.thread():
                                tid_flat = Tx.meta_var(warp_id * 32 + lane_id)
                                if tid_flat == 0:
                                    slopes_s[0] = slopes_g[wid_con]

                            # Zero KV state (128*128=16384 f32 / 128 threads = 128 per thread)
                            with Tx.thread():
                                tid_flat = Tx.meta_var(warp_id * 32 + lane_id)
                                for ki in Tx.serial(128):
                                    idx = Tx.meta_var(tid_flat * 128 + ki)
                                    row = Tx.meta_var(idx // D)
                                    col = Tx.meta_var(idx % D)
                                    kvs[row, col] = 0.0
                            Tx.cuda.warpgroup_sync(10)

                            cid_con = 0
                            while cid_con < nc_con:
                                tic_c = Tx.meta_var(cid_con % PIPE_DEPTH)

                                # ======== Phase 1: Read att from TMEM ========
                                mma2consumer_bar.wait(0, phase_m2c)
                                phase_m2c = phase_m2c ^ 1
                                Tx.ptx.tcgen05.fence.after_thread_sync()

                                Tx.copy(reg_wg[:, :], tmem[:, 0:TMEM_LD_SIZE])

                                with Tx.thread():
                                    out_row = Tx.meta_var(warp_id * 16 + lane_id)
                                    if lane_id < 16:
                                        slope_val = Tx.meta_var(slopes_s[0])
                                        for j in Tx.serial(CHUNK):
                                            if out_row >= j:
                                                diff_val = Tx.meta_var(Tx.cast(out_row - j, "float32"))
                                                reg[j] = reg[j] * Tx.exp(0.0 - slope_val * diff_val)
                                            else:
                                                reg[j] = 0.0
                                        for j in Tx.serial(HALF):
                                            work_smem[out_row, j] = Tx.cast(reg[j], "float16")

                                Tx.ptx.tcgen05.fence.before_thread_sync()
                                consumer2mma_bar.arrive(0)
                                phase_c2m_c = phase_c2m_c ^ 1

                                # ======== Phase 2a: Read o_intra_lo, prep Q_dec staging + kvs_lo cast ========
                                mma2consumer_bar.wait(0, phase_m2c)
                                phase_m2c = phase_m2c ^ 1
                                Tx.ptx.tcgen05.fence.after_thread_sync()

                                Tx.copy(reg_wg[:, :], tmem[:, 0:TMEM_LD_SIZE])

                                with Tx.thread():
                                    out_row = Tx.meta_var(warp_id * 16 + lane_id)
                                    if lane_id < 16:
                                        for j in Tx.serial(HALF):
                                            o_lo_reg[j] = reg[j]

                                # Prepare staging_a SMEM with Q_dec (swizzled layout)
                                with Tx.thread():
                                    tid_flat = Tx.meta_var(warp_id * 32 + lane_id)
                                    slope_val = Tx.meta_var(slopes_s[0])
                                    for i in Tx.serial(32):
                                        flat = Tx.meta_var(tid_flat * 32 + i)
                                        srow = Tx.meta_var(flat // HALF)
                                        scol = Tx.meta_var(flat % HALF)
                                        q_decay_v = Tx.meta_var(Tx.exp(0.0 - slope_val * Tx.cast(srow, "float32")))
                                        staging_a_lo[srow, scol] = Tx.cast(
                                            Tx.cast(Q_lo[tic_c, srow, scol], "float32") * q_decay_v, "float16")
                                        staging_a_hi[srow, scol] = Tx.cast(
                                            Tx.cast(Q_hi[tic_c, srow, scol], "float32") * q_decay_v, "float16")

                                # staging_b_lo[f,d] = f16(kvs[f,d]) for f in [0,64), d in [0,64)
                                # staging_b_hi[f,d] = f16(kvs[HALF+f,d]) for f in [0,64), d in [0,64)
                                with Tx.thread():
                                    tid_flat = Tx.meta_var(warp_id * 32 + lane_id)
                                    for i in Tx.serial(32):
                                        flat = Tx.meta_var(tid_flat * 32 + i)
                                        brow = Tx.meta_var(flat // HALF)
                                        bcol = Tx.meta_var(flat % HALF)
                                        staging_b_lo[brow, bcol] = Tx.cast(kvs[brow, bcol], "float16")
                                        staging_b_hi[brow, bcol] = Tx.cast(kvs[HALF + brow, bcol], "float16")

                                Tx.ptx.tcgen05.fence.before_thread_sync()
                                consumer2mma_bar.arrive(0)
                                phase_c2m_c = phase_c2m_c ^ 1

                                # ======== Phase 2b: Read o_intra_hi ========
                                mma2consumer_bar.wait(0, phase_m2c)
                                phase_m2c = phase_m2c ^ 1
                                Tx.ptx.tcgen05.fence.after_thread_sync()

                                Tx.copy(reg_wg[:, :], tmem[:, 0:TMEM_LD_SIZE])

                                with Tx.thread():
                                    out_row = Tx.meta_var(warp_id * 16 + lane_id)
                                    if lane_id < 16:
                                        for j in Tx.serial(HALF):
                                            o_hi_reg[j] = reg[j]

                                Tx.ptx.tcgen05.fence.before_thread_sync()
                                consumer2mma_bar.arrive(0)
                                phase_c2m_c = phase_c2m_c ^ 1

                                # ======== Phase 3: Read Q_dec@kvs_lo from TMEM ========
                                mma2consumer_bar.wait(0, phase_m2c)
                                phase_m2c = phase_m2c ^ 1
                                Tx.ptx.tcgen05.fence.after_thread_sync()

                                Tx.copy(reg_wg[:, :], tmem[:, 0:TMEM_LD_SIZE])

                                # Add inter-chunk o_lo contribution from MMA
                                with Tx.thread():
                                    out_row = Tx.meta_var(warp_id * 16 + lane_id)
                                    if lane_id < 16:
                                        for j in Tx.serial(HALF):
                                            o_lo_reg[j] = o_lo_reg[j] + reg[j]

                                # Prep kvs_hi in staging_b for Phase 4
                                with Tx.thread():
                                    tid_flat = Tx.meta_var(warp_id * 32 + lane_id)
                                    for i in Tx.serial(32):
                                        flat = Tx.meta_var(tid_flat * 32 + i)
                                        brow = Tx.meta_var(flat // HALF)
                                        bcol = Tx.meta_var(flat % HALF)
                                        staging_b_lo[brow, bcol] = Tx.cast(kvs[brow, HALF + bcol], "float16")
                                        staging_b_hi[brow, bcol] = Tx.cast(kvs[HALF + brow, HALF + bcol], "float16")

                                Tx.ptx.tcgen05.fence.before_thread_sync()
                                consumer2mma_bar.arrive(0)
                                phase_c2m_c = phase_c2m_c ^ 1

                                # ======== Phase 4: Read Q_dec@kvs_hi from TMEM ========
                                mma2consumer_bar.wait(0, phase_m2c)
                                phase_m2c = phase_m2c ^ 1
                                Tx.ptx.tcgen05.fence.after_thread_sync()

                                Tx.copy(reg_wg[:, :], tmem[:, 0:TMEM_LD_SIZE])

                                # Add inter-chunk o_hi contribution from MMA
                                with Tx.thread():
                                    out_row = Tx.meta_var(warp_id * 16 + lane_id)
                                    if lane_id < 16:
                                        for j in Tx.serial(HALF):
                                            o_hi_reg[j] = o_hi_reg[j] + reg[j]

                                        # Write output to D_out
                                        for j in Tx.serial(HALF):
                                            D_out_lo[out_row, j] = Tx.cast(o_lo_reg[j], "float16")
                                        for j in Tx.serial(HALF):
                                            D_out_hi[out_row, j] = Tx.cast(o_hi_reg[j], "float16")

                                # TMA store output (two halves)
                                Tx.ptx.fence.proxy_async("shared::cta")
                                Tx.cuda.warpgroup_sync(10)
                                with Tx.thread()[lane_id == 0 and warp_id == 0]:
                                    q_row_out = Tx.meta_var(cid_con * CHUNK)
                                    Tx.copy_async(o_g[wid_con, q_row_out : q_row_out + CHUNK, 0 : HALF],
                                                  D_out_lo[:, :], dispatch="tma", cache_hint="evict_last")
                                    Tx.copy_async(o_g[wid_con, q_row_out : q_row_out + CHUNK, HALF : D],
                                                  D_out_hi[:, :], dispatch="tma", cache_hint="evict_last")
                                    Tx.ptx.cp_async.bulk.commit_group()
                                    Tx.ptx.cp_async.bulk.wait_group(0)
                                Tx.cuda.warpgroup_sync(10)

                                # Prep K_dec staging for Phases 5-8
                                # (block_decay moved into Phases 5-8 per-quadrant)
                                with Tx.thread():
                                    tid_flat = Tx.meta_var(warp_id * 32 + lane_id)
                                    slope_val = Tx.meta_var(slopes_s[0])

                                    # staging_a_lo[pos,f] = f16(K_lo[tic,pos,f] * k_decay(pos))
                                    # staging_a_hi[pos,f] = f16(K_hi[tic,pos,f] * k_decay(pos))
                                    # 64*64=4096 / 128 = 32 per thread
                                    for i in Tx.serial(32):
                                        flat = Tx.meta_var(tid_flat * 32 + i)
                                        srow = Tx.meta_var(flat // HALF)  # pos
                                        scol = Tx.meta_var(flat % HALF)   # f
                                        k_decay_v = Tx.meta_var(Tx.exp(0.0 - slope_val * Tx.cast(CHUNK - srow, "float32")))
                                        staging_a_lo[srow, scol] = Tx.cast(
                                            Tx.cast(K_lo[tic_c, srow, scol], "float32") * k_decay_v, "float16")
                                        staging_a_hi[srow, scol] = Tx.cast(
                                            Tx.cast(K_hi[tic_c, srow, scol], "float32") * k_decay_v, "float16")

                                Tx.ptx.tcgen05.fence.before_thread_sync()
                                consumer2mma_bar.arrive(0)
                                phase_c2m_c = phase_c2m_c ^ 1

                                # ======== Phases 5-8: Read K_dec^T@V from TMEM → update kvs ========
                                for state_i in Tx.serial(4):
                                    mma2consumer_bar.wait(0, phase_m2c)
                                    phase_m2c = phase_m2c ^ 1
                                    Tx.ptx.tcgen05.fence.after_thread_sync()

                                    Tx.copy(reg_wg[:, :], tmem[:, 0:TMEM_LD_SIZE])

                                    # Apply block_decay to this kvs quadrant, then add TMEM result
                                    # Decay uses all 128 threads: 64*64=4096 / 128 = 32 per thread
                                    # state_i=0: kvs[0:64, 0:64], state_i=1: kvs[0:64, 64:128]
                                    # state_i=2: kvs[64:128, 0:64], state_i=3: kvs[64:128, 64:128]
                                    with Tx.thread():
                                        tid_flat = Tx.meta_var(warp_id * 32 + lane_id)
                                        slope_val = Tx.meta_var(slopes_s[0])
                                        block_decay = Tx.meta_var(Tx.exp(0.0 - slope_val * Tx.cast(CHUNK, "float32")))
                                        for i in Tx.serial(32):
                                            flat = Tx.meta_var(tid_flat * 32 + i)
                                            drow = Tx.meta_var(flat // HALF)
                                            dcol = Tx.meta_var(flat % HALF)
                                            if state_i == 0:
                                                kvs[drow, dcol] = kvs[drow, dcol] * block_decay
                                            elif state_i == 1:
                                                kvs[drow, HALF + dcol] = kvs[drow, HALF + dcol] * block_decay
                                            elif state_i == 2:
                                                kvs[HALF + drow, dcol] = kvs[HALF + drow, dcol] * block_decay
                                            else:
                                                kvs[HALF + drow, HALF + dcol] = kvs[HALF + drow, HALF + dcol] * block_decay

                                    # Add TMEM result to correct kvs quadrant (64 active threads)
                                    with Tx.thread():
                                        out_row = Tx.meta_var(warp_id * 16 + lane_id)
                                        if lane_id < 16:
                                            for j in Tx.serial(HALF):
                                                if state_i == 0:
                                                    kvs[out_row, j] = kvs[out_row, j] + reg[j]
                                                elif state_i == 1:
                                                    kvs[out_row, HALF + j] = kvs[out_row, HALF + j] + reg[j]
                                                elif state_i == 2:
                                                    kvs[HALF + out_row, j] = kvs[HALF + out_row, j] + reg[j]
                                                else:
                                                    kvs[HALF + out_row, HALF + j] = kvs[HALF + out_row, HALF + j] + reg[j]

                                    Tx.ptx.tcgen05.fence.before_thread_sync()
                                    consumer2mma_bar.arrive(0)
                                    phase_c2m_c = phase_c2m_c ^ 1

                                cid_con = cid_con + 1
                            wid_con = wid_con + SM_COUNT

                # ---- TMEM deallocation ----
                with Tx.warp()[0:1]:
                    Tx.ptx.tcgen05.relinquish_alloc_permit(cta_group=CTA_GROUP)
                    Tx.ptx.tcgen05.dealloc(tmem_addr, n_cols=N_COLS, cta_group=CTA_GROUP)

                Tx.cuda.cta_sync()
    # fmt: on
    return linear_attention


@pytest.mark.parametrize("batch,heads,seq_len", [(1, 8, 256), (1, 8, 1024), (2, 4, 512), (10, 16, 128)])
def test_linear_attention(batch, heads, seq_len):
    q, k, v, slopes = prepare_data(batch, heads, seq_len)
    o_ref = naive_linear_attention(q, k, v, slopes)

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": get_linear_attention_kernel()})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")

    DEV = tvm.cuda(0)
    BH = batch * heads
    L = seq_len
    o_np = np.zeros((BH, L, D), dtype=np.float16)
    q_tvm = tvm.runtime.tensor(q.cpu().numpy(), DEV)
    k_tvm = tvm.runtime.tensor(k.cpu().numpy(), DEV)
    v_tvm = tvm.runtime.tensor(v.cpu().numpy(), DEV)
    slopes_tvm = tvm.runtime.tensor(slopes.cpu().numpy(), DEV)
    o_tvm = tvm.runtime.tensor(o_np, DEV)
    mod(q_tvm, k_tvm, v_tvm, slopes_tvm, o_tvm)

    o_tir = o_tvm.numpy()
    o_ref_np = o_ref.cpu().numpy()
    np.testing.assert_allclose(o_tir, o_ref_np, rtol=5e-2, atol=5e-2)
    print(f"PASSED: batch={batch}, heads={heads}, seq_len={seq_len}")


def bench_linear_attention():
    """Benchmark linear attention kernel at TK-equivalent dimensions."""
    batch, heads = 8, 16
    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": get_linear_attention_kernel()})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")

    DEV = tvm.cuda(0)
    BH = batch * heads

    # FLOPS: Q@K^T (2*BH*N*CHUNK*F) + att@V (2*BH*N*D*CHUNK) + Q@kvs (2*BH*N*F*D)
    #        + state_update (2*BH*N*F*D) per chunk pair
    # Simplified: count all matmul ops
    def flops(ms, seq_len):
        nc = seq_len // CHUNK
        # Phase 1: Q@K^T  [64,F] @ [F,64] = 2*64*64*F per chunk
        # Phase 2: att@V_lo [64,64] @ [64,64] = 2*64*64*64 per chunk
        # Phase 3: att@V_hi [64,64] @ [64,64] = 2*64*64*64 per chunk
        # Consumer: Q@kvs [1,F] @ [F,D] = 2*F*D per row, 64 rows = 2*64*F*D per chunk
        # Consumer: state_update K^T@V = 2*64*F*D per chunk (but outer product)
        mma_flops = nc * BH * (2 * 64 * 64 * F + 2 * 2 * 64 * 64 * 64)
        consumer_flops = nc * BH * (2 * 64 * F * D + 2 * 64 * F * D)
        return (mma_flops + consumer_flops) / (ms * 1e-3)

    print(f"\n{'='*60}")
    print(f"Linear Attention Benchmark (B={batch}, H={heads})")
    print(f"{'='*60}")

    with ProtonContext("linear_attention"):
        for seq_len in [1024, 2048, 4096, 8192]:
            q, k, v, slopes = prepare_data(batch, heads, seq_len)
            L = seq_len
            o_np = np.zeros((BH, L, D), dtype=np.float16)
            q_tvm = tvm.runtime.tensor(q.cpu().numpy(), DEV)
            k_tvm = tvm.runtime.tensor(k.cpu().numpy(), DEV)
            v_tvm = tvm.runtime.tensor(v.cpu().numpy(), DEV)
            slopes_tvm = tvm.runtime.tensor(slopes.cpu().numpy(), DEV)
            o_tvm = tvm.runtime.tensor(o_np, DEV)

            func = lambda: mod(q_tvm, k_tvm, v_tvm, slopes_tvm, o_tvm)
            ms = bench(func, warmup=100, repeat=300, proton_name=f"linear_attn_N{seq_len}")
            tflops = flops(ms, seq_len) / 1e12
            print(f"  N={seq_len:>5d}: {tflops:.2f} TFLOPS, {ms:.3f} ms")

    print()


if __name__ == "__main__":
    test_linear_attention(1, 8, 256)
    bench_linear_attention()
