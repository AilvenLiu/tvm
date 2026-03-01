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


"""Hedgehog attention kernel with SM100 warp specialization.

Warp-specialized rewrite using tcgen05 MMA, TMEM, and producer/consumer
warp separation. 12 MMA phases per chunk combining sliding window attention
(Q@K^T with causal mask + softmax) and linear attention (featurized Q @ kv_state),
with per-chunk KV state updates. Linear attention and state updates are offloaded
to MMA phases rather than scalar consumer loops.

Architecture:
  cta_group=1 (single CTA, no clustering)
  MMA_M=64, MMA_N=64, MMA_K=16 (f16)
  WG1 (Producer+MMA): warp 3 = TMA, warps 0-1 = MMA
  WG0 (Consumer): TMEM read, softmax, featurize, operand staging

12 MMA Phases per chunk:
  Phase 1: Q@K_prev^T (transB=False, cross-buffer lo+hi)
  Phase 2: Q@K_curr^T (transB=False, cross-buffer lo+hi)
  Phase 3: att@V_prev_lo + att@V_curr_lo (transB=True, cross-buffer work_a/work_b)
  Phase 4: att@V_prev_hi + att@V_curr_hi (transB=True, cross-buffer)
  Phase 5: Q@qmap (transB=True, cross-buffer lo+hi)
  Phase 6: feat_q_pos@kv[0:64,0:64] + feat_q_neg@kv[64:128,0:64] -> o_lo_add
  Phase 7: feat_q_pos@kv[0:64,64:128] + feat_q_neg@kv[64:128,64:128] -> o_hi_add
  Phase 8: K_prev@kmap (transB=True, cross-buffer lo+hi)
  Phase 9: feat_k_pos^T@V_prev_lo (transA=True, transB=True) -> kv_state[0:64,0:64] update
  Phase 10: feat_k_pos^T@V_prev_hi (transA=True, transB=True) -> kv_state[0:64,64:128] update
  Phase 11: feat_k_neg^T@V_prev_lo (transA=True, transB=True) -> kv_state[64:128,0:64] update
  Phase 12: feat_k_neg^T@V_prev_hi (transA=True, transB=True) -> kv_state[64:128,64:128] update

12 phases = even -> phase_c2m does NOT flip between chunks.
"""

import numpy as np
import pytest
import torch

import tvm
import tvm.testing
from tvm.ir import PointerType, PrimType
from tvm.script import tirx as Tx
from tvm.tir.layout import S, TCol, TileLayout, TLane, tid_in_wg
from tvm.tirx.bench.utils import ProtonContext, bench
from tvm.tirx.pipeline import MBarrier, TCGen05Bar, TMABar

# Constants
CHUNK = 64
ATTN_D = 128  # Q/K/V/O dimension
HALF = 64  # half of ATTN_D for TMA/MMA splitting
HALF_F = 64  # projection output dim (pos or neg half)
ATTN_F = 128  # featurized dim = 2 * HALF_F
SM_COUNT = 148

# Warp-specialized layout
WG_NUMBER = 2  # WG0=consumer, WG1=producer+MMA
WARP_NUMBER = 4  # per warpgroup
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
N_COLS = 64
TMEM_LD_SIZE = 64

TEMP = 0.08838834764  # 1/sqrt(128)

# Memory element counts (used in TMA byte calculations and aliases)
HALF_SINGLE = CHUNK * HALF  # 4096 f16 elems = 8KB


def prepare_data(B, H, N):
    q = torch.randn(B, H, N, ATTN_D, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(B, H, N, ATTN_D, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(B, H, N, ATTN_D, dtype=torch.bfloat16, device="cuda") / 5
    qmap = torch.randn(H, ATTN_D, HALF_F, dtype=torch.bfloat16, device="cuda")
    kmap = torch.randn(H, ATTN_D, HALF_F, dtype=torch.bfloat16, device="cuda")
    alphas = torch.rand(H, dtype=torch.float32, device="cuda") * 3
    betas = torch.rand(H, dtype=torch.float32, device="cuda")
    return q, k, v, qmap, kmap, alphas, betas


def softmax_featuremap(x, map_mat):
    """x: (B,H,N,D) @ map_mat: (H,D,HALF_F) -> (B,H,N,ATTN_F)"""
    proj = torch.einsum("bhmd,hdn->bhmn", x, map_mat)
    x_pos = proj - proj.amax(dim=-1, keepdim=True)
    x_neg = -proj - (-proj).amax(dim=-1, keepdim=True)
    x_pos = torch.exp(x_pos) / torch.exp(x_pos).sum(dim=-1, keepdim=True)
    x_neg = torch.exp(x_neg) / torch.exp(x_neg).sum(dim=-1, keepdim=True)
    return torch.cat([x_pos, x_neg], dim=-1).clamp(min=1e-6)


def naive_hedgehog(q, k, v, qmap, kmap, alphas, betas):
    """Reference matching TK hedgehog kernel."""
    B, H, N, D = q.shape
    NC = N // CHUNK
    qf = q.float()
    kf = k.float()
    vf = v.float()

    Qs = softmax_featuremap(qf, qmap.float())
    Ks = softmax_featuremap(kf, kmap.float())

    # Build masks (from TK gentests.py)
    generator_mat = torch.block_diag(*[torch.ones(CHUNK, CHUNK, device=q.device)] * NC)
    gen_rolled = torch.roll(generator_mat, -CHUNK, -1)
    generator_mat = (generator_mat + gen_rolled).clamp(max=1.0)[:N, :N]

    lin_mask = torch.tril(1 - generator_mat).unsqueeze(0).unsqueeze(0)
    exp_mask = torch.tril(generator_mat).unsqueeze(0).unsqueeze(0)
    exp_mask = 10000 * exp_mask - 10000

    a_lin = torch.einsum("bhmd,bhnd->bhmn", Qs, Ks).float()
    a_exp = torch.einsum("bhmd,bhnd->bhmn", qf, kf).float()

    a_lin = a_lin * lin_mask * alphas.reshape(1, -1, 1, 1)
    a_exp = a_exp + exp_mask
    a_exp = a_exp - a_exp.amax(dim=-1, keepdim=True)
    a_exp = torch.exp(a_exp / (ATTN_D**0.5)) * betas.reshape(1, -1, 1, 1)

    a = a_exp + a_lin
    a = a / (a.sum(dim=-1, keepdim=True) + 1e-6)
    out = torch.einsum("bhmn,bhnd->bhmd", a.float(), vf).to(torch.bfloat16)

    # State: accumulate chunks 0..NC-2
    kv_state = torch.einsum("bhlf,bhld->bhfd", Ks[:, :, :-CHUNK, :], vf[:, :, :-CHUNK, :]).float()
    k_state = Ks[:, :, :-CHUNK, :].float().sum(dim=-2)

    return out, kv_state, k_state


# ---- Layouts ----
# All buffers use SWIZZLE_128B (3) with 64 f16 = 128B inner dimension.

half_pipe_layout = Tx.ComposeLayout(
    Tx.SwizzleLayout(SWIZZLE, SWIZZLE, SWIZZLE, swizzle_inner=True),
    Tx.TileLayout(S[(PIPE_DEPTH, CHUNK, HALF) : (CHUNK * HALF, HALF, 1)]),
)
half_layout = Tx.ComposeLayout(
    Tx.SwizzleLayout(SWIZZLE, SWIZZLE, SWIZZLE, swizzle_inner=True),
    Tx.TileLayout(S[(CHUNK, HALF) : (HALF, 1)]),
)
# kv_state_f16: [128, 64] f16 — MMA operand for feat_q @ kv_state
kvs_f16_layout = Tx.ComposeLayout(
    Tx.SwizzleLayout(SWIZZLE, SWIZZLE, SWIZZLE, swizzle_inner=True),
    Tx.TileLayout(S[(ATTN_F, HALF) : (HALF, 1)]),
)


def get_hedgehog_kernel():
    a_type = tvm.DataType("float16")  # noqa: F841

    # fmt: off
    @Tx.prim_func(tirx=True)
    def hedgehog(
        q_ptr: Tx.handle, k_ptr: Tx.handle, v_ptr: Tx.handle,
        qmap_ptr: Tx.handle, kmap_ptr: Tx.handle,
        alphas_ptr: Tx.handle, betas_ptr: Tx.handle,
        o_ptr: Tx.handle, kvs_ptr: Tx.handle, ks_ptr: Tx.handle,
    ):
        total_bh = Tx.int32()
        H_param = Tx.int32()
        N_param = Tx.int32()
        q_g = Tx.match_buffer(q_ptr, [total_bh, N_param, ATTN_D], "float16", scope="global")
        k_g = Tx.match_buffer(k_ptr, [total_bh, N_param, ATTN_D], "float16", scope="global")
        v_g = Tx.match_buffer(v_ptr, [total_bh, N_param, ATTN_D], "float16", scope="global")
        qmap_g = Tx.match_buffer(qmap_ptr, [H_param, ATTN_D, HALF_F], "float16", scope="global")
        kmap_g = Tx.match_buffer(kmap_ptr, [H_param, ATTN_D, HALF_F], "float16", scope="global")
        alphas_g = Tx.match_buffer(alphas_ptr, [H_param], "float32", scope="global")
        betas_g = Tx.match_buffer(betas_ptr, [H_param], "float32", scope="global")
        o_g = Tx.match_buffer(o_ptr, [total_bh, N_param, ATTN_D], "float16", scope="global")
        kvs_g = Tx.match_buffer(kvs_ptr, [total_bh, ATTN_F, ATTN_D], "float32", scope="global")
        ks_g = Tx.match_buffer(ks_ptr, [total_bh, ATTN_F], "float32", scope="global")

        with Tx.kernel():
            bx = Tx.cta_id([SM_COUNT], parent="kernel")
            wg_id = Tx.warpgroup_id([WG_NUMBER], parent="cta")  # noqa: F841
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

                # Swizzled data buffers
                Q_lo = pool.alloc((PIPE_DEPTH, CHUNK, HALF), "float16", layout=half_pipe_layout)
                Q_hi = pool.alloc((PIPE_DEPTH, CHUNK, HALF), "float16", layout=half_pipe_layout)
                K_lo = pool.alloc((PIPE_DEPTH, CHUNK, HALF), "float16", layout=half_pipe_layout)
                K_hi = pool.alloc((PIPE_DEPTH, CHUNK, HALF), "float16", layout=half_pipe_layout)
                V_lo = pool.alloc((PIPE_DEPTH, CHUNK, HALF), "float16", layout=half_pipe_layout)
                V_hi = pool.alloc((PIPE_DEPTH, CHUNK, HALF), "float16", layout=half_pipe_layout)
                work_a = pool.alloc((CHUNK, HALF), "float16", layout=half_layout)
                work_b = pool.alloc((CHUNK, HALF), "float16", layout=half_layout)
                qmap_lo = pool.alloc((CHUNK, HALF), "float16", layout=half_layout)
                qmap_hi = pool.alloc((CHUNK, HALF), "float16", layout=half_layout)
                kmap_lo = pool.alloc((CHUNK, HALF), "float16", layout=half_layout)
                kmap_hi = pool.alloc((CHUNK, HALF), "float16", layout=half_layout)

                # kv_state_f16: swizzled staging buffer for MMA
                # Also aliases D_out_lo [0:64,:] and D_out_hi [64:128,:] for TMA store
                kvs_f16_byte_off = pool.offset
                kv_state_f16 = pool.alloc((ATTN_F, HALF), "float16", layout=kvs_f16_layout)
                D_out_lo = Tx.decl_buffer((CHUNK, HALF), "float16", pool.ptr,
                                          layout=half_layout, byte_offset=kvs_f16_byte_off)
                D_out_hi = Tx.decl_buffer((CHUNK, HALF), "float16", pool.ptr,
                                          layout=half_layout, byte_offset=kvs_f16_byte_off + HALF_SINGLE * F16_BYTES)  # noqa: E501

                # Non-swizzled buffers
                kv_state = pool.alloc((ATTN_F, ATTN_D), "float32")
                k_cumsum = pool.alloc((ATTN_F,), "float32")
                alphas_s = pool.alloc((1,), "float32")
                betas_s = pool.alloc((1,), "float32")
                # feat_k reuses work_a (pos half) and work_b (neg half) — swizzled [64,64] f16

                pool.commit()

                # ---- Local variables ----
                descI_nn: Tx.uint32  # transB=False
                descI_nt: Tx.uint32  # transB=True
                descA: Tx.uint64
                descB: Tx.uint64
                phase = Tx.alloc_buffer((1,), "int32", scope="local")

                # ---- Barrier init ----
                tma2mma_bar.init(1)
                mma2tma_bar.init(1)
                mma2consumer_bar.init(1)
                consumer2mma_bar.init(128)  # full consumer WG
                workitem_sync_bar.init(1)

                ptr: Tx.let[Tx.Var(name="ptr", dtype=PointerType(PrimType("uint64")))] = Tx.reinterpret("handle", Tx.ptx.map_shared_rank(tma2mma_bar.ptr_to([0]), 0))  # noqa: E501
                tma_finished = Tx.decl_buffer([PIPE_DEPTH], "uint64", data=ptr, scope="shared")

                # ---- TMEM allocation ----
                with Tx.warp()[0:1]:
                    Tx.ptx.tcgen05.alloc(Tx.address_of(tmem_addr), n_cols=N_COLS, cta_group=CTA_GROUP)  # noqa: E501
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

                # NOTE: qmap/kmap are per-head constants loaded per work item in consumer.

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
                                nc_tma = N_param // CHUNK
                                cid_tma = 0
                                while cid_tma < nc_tma:
                                    with Tx.thread()[Tx.ptx.elect_sync()]:
                                        tic = Tx.meta_var(cid_tma % PIPE_DEPTH)
                                        q_row = Tx.meta_var(cid_tma * CHUNK)
                                        tma_copy = Tx.meta_var({"dispatch": "tma", "mbar": tma_finished.ptr_to([tic]), "cta_group": CTA_GROUP})  # noqa: E501
                                        # Wait for MMA to release this SMEM slot
                                        if cid_tma >= PIPE_DEPTH:
                                            mma2tma_bar.wait(tic, phase[0] ^ 1)
                                        # Load Q, K, V halves (6 TMA loads per chunk)
                                        Tx.copy_async(Q_lo[tic, :, :], q_g[wid_tma, q_row : q_row + CHUNK, 0 : HALF], **tma_copy)  # noqa: E501
                                        Tx.copy_async(Q_hi[tic, :, :], q_g[wid_tma, q_row : q_row + CHUNK, HALF : ATTN_D], **tma_copy)  # noqa: E501
                                        Tx.copy_async(K_lo[tic, :, :], k_g[wid_tma, q_row : q_row + CHUNK, 0 : HALF], **tma_copy)  # noqa: E501
                                        Tx.copy_async(K_hi[tic, :, :], k_g[wid_tma, q_row : q_row + CHUNK, HALF : ATTN_D], **tma_copy)  # noqa: E501
                                        Tx.copy_async(V_lo[tic, :, :], v_g[wid_tma, q_row : q_row + CHUNK, 0 : HALF], **tma_copy)  # noqa: E501
                                        Tx.copy_async(V_hi[tic, :, :], v_g[wid_tma, q_row : q_row + CHUNK, HALF : ATTN_D], **tma_copy)  # noqa: E501
                                        # Signal that load is done
                                        tma2mma_bar.arrive(tic, 6 * CHUNK * HALF * F16_BYTES)
                                    if cid_tma % PIPE_DEPTH == PIPE_DEPTH - 1:
                                        phase[0] = phase[0] ^ 1
                                    cid_tma = cid_tma + 1
                                wid_tma = wid_tma + SM_COUNT

                        elif warp_id == 0:
                            # ---- MMA warp ----
                            # Instruction descriptor: transB=False for phases 1-2
                            Tx.ptx.tcgen05.encode_instr_descriptor(
                                Tx.address_of(descI_nn), "float32", "float16", "float16",  # noqa: F821
                                MMA_M * CTA_GROUP, MMA_N, MMA_K, False, False, CTA_GROUP)
                            # Instruction descriptor: transB=True for phases 3-8, 6-7
                            Tx.ptx.tcgen05.encode_instr_descriptor(
                                Tx.address_of(descI_nt), "float32", "float16", "float16",  # noqa: F821
                                MMA_M * CTA_GROUP, MMA_N, MMA_K, False, True, CTA_GROUP)
                            # Instruction descriptor: transA=True, transB=True for phases 9-12
                            descI_tt: Tx.uint32
                            Tx.ptx.tcgen05.encode_instr_descriptor(
                                Tx.address_of(descI_tt), "float32", "float16", "float16",  # noqa: F821
                                MMA_M * CTA_GROUP, MMA_N, MMA_K, True, True, CTA_GROUP)

                            phase[0] = 0
                            wid_mma = Tx.local_scalar("int32", "wid_mma")
                            cid_mma = Tx.local_scalar("int32", "cid_mma")
                            nc_mma = Tx.local_scalar("int32", "nc_mma")
                            phase_c2m: Tx.int32
                            phase_c2m = 0
                            wid_mma = bx
                            while wid_mma < total_bh:
                                nc_mma = N_param // CHUNK
                                cid_mma = 0
                                while cid_mma < nc_mma:
                                    with Tx.thread()[Tx.ptx.elect_sync()]:
                                        tic = Tx.meta_var(cid_mma % PIPE_DEPTH)
                                        prev_tic = Tx.meta_var(1 - tic)

                                        # == Phase 1: Q@K_prev^T (transB=False) ==
                                        # Q_lo[tic]@K_lo[prev_tic] + Q_hi[tic]@K_hi[prev_tic]
                                        tma2mma_bar.wait(tic, phase[0])
                                        Tx.ptx.tcgen05.fence.after_thread_sync()
                                        for ki in Tx.unroll(HALF // MMA_K):  # 4 iterations
                                            Tx.ptx.tcgen05.encode_matrix_descriptor(
                                                Tx.address_of(descA), Q_lo.ptr_to([tic, 0, ki * MMA_K]),  # noqa: E501, F821
                                                MMA_LDO, MMA_SDO, SWIZZLE)
                                            Tx.ptx.tcgen05.encode_matrix_descriptor(
                                                Tx.address_of(descB), K_lo.ptr_to([prev_tic, 0, ki * MMA_K]),  # noqa: E501, F821
                                                MMA_LDO, MMA_SDO, SWIZZLE)
                                            Tx.ptx.tcgen05.mma("float32", "float16", "float16",
                                                Tx.cuda.get_tmem_addr(tmem_addr, 0, 0),
                                                descA, descB, descI_nn, False, CTA_GROUP, ki > 0)  # noqa: F821
                                        for ki in Tx.unroll(HALF // MMA_K):  # 4 iterations
                                            Tx.ptx.tcgen05.encode_matrix_descriptor(
                                                Tx.address_of(descA), Q_hi.ptr_to([tic, 0, ki * MMA_K]),  # noqa: E501, F821
                                                MMA_LDO, MMA_SDO, SWIZZLE)
                                            Tx.ptx.tcgen05.encode_matrix_descriptor(
                                                Tx.address_of(descB), K_hi.ptr_to([prev_tic, 0, ki * MMA_K]),  # noqa: E501, F821
                                                MMA_LDO, MMA_SDO, SWIZZLE)
                                            Tx.ptx.tcgen05.mma("float32", "float16", "float16",
                                                Tx.cuda.get_tmem_addr(tmem_addr, 0, 0),
                                                descA, descB, descI_nn, False, CTA_GROUP, True)  # noqa: F821
                                        mma2consumer_bar.arrive(0, CTA_GROUP, 1)

                                        # Wait consumer done with Phase 1
                                        consumer2mma_bar.wait(0, phase_c2m)
                                        Tx.ptx.tcgen05.fence.after_thread_sync()

                                        # == Phase 2: Q@K_curr^T (transB=False) ==
                                        # Q_lo[tic]@K_lo[tic] + Q_hi[tic]@K_hi[tic]
                                        for ki in Tx.unroll(HALF // MMA_K):
                                            Tx.ptx.tcgen05.encode_matrix_descriptor(
                                                Tx.address_of(descA), Q_lo.ptr_to([tic, 0, ki * MMA_K]),  # noqa: E501, F821
                                                MMA_LDO, MMA_SDO, SWIZZLE)
                                            Tx.ptx.tcgen05.encode_matrix_descriptor(
                                                Tx.address_of(descB), K_lo.ptr_to([tic, 0, ki * MMA_K]),  # noqa: E501, F821
                                                MMA_LDO, MMA_SDO, SWIZZLE)
                                            Tx.ptx.tcgen05.mma("float32", "float16", "float16",
                                                Tx.cuda.get_tmem_addr(tmem_addr, 0, 0),
                                                descA, descB, descI_nn, False, CTA_GROUP, ki > 0)  # noqa: F821
                                        for ki in Tx.unroll(HALF // MMA_K):
                                            Tx.ptx.tcgen05.encode_matrix_descriptor(
                                                Tx.address_of(descA), Q_hi.ptr_to([tic, 0, ki * MMA_K]),  # noqa: E501, F821
                                                MMA_LDO, MMA_SDO, SWIZZLE)
                                            Tx.ptx.tcgen05.encode_matrix_descriptor(
                                                Tx.address_of(descB), K_hi.ptr_to([tic, 0, ki * MMA_K]),  # noqa: E501, F821
                                                MMA_LDO, MMA_SDO, SWIZZLE)
                                            Tx.ptx.tcgen05.mma("float32", "float16", "float16",
                                                Tx.cuda.get_tmem_addr(tmem_addr, 0, 0),
                                                descA, descB, descI_nn, False, CTA_GROUP, True)  # noqa: F821
                                        mma2consumer_bar.arrive(0, CTA_GROUP, 1)

                                        # Wait consumer done with Phase 2
                                        consumer2mma_bar.wait(0, phase_c2m ^ 1)
                                        Tx.ptx.tcgen05.fence.after_thread_sync()

                                        # == Phase 3: work_a@V_prev_lo + work_b@V_curr_lo (transB=True) ==  # noqa: E501
                                        for ki in Tx.unroll(CHUNK // MMA_K):
                                            Tx.ptx.tcgen05.encode_matrix_descriptor(
                                                Tx.address_of(descA), work_a.ptr_to([0, ki * MMA_K]),  # noqa: E501, F821
                                                MMA_LDO, MMA_SDO, SWIZZLE)
                                            Tx.ptx.tcgen05.encode_matrix_descriptor(
                                                Tx.address_of(descB), V_lo.ptr_to([prev_tic, ki * MMA_K, 0]),  # noqa: E501, F821
                                                MMA_LDO, MMA_SDO, SWIZZLE)
                                            Tx.ptx.tcgen05.mma("float32", "float16", "float16",
                                                Tx.cuda.get_tmem_addr(tmem_addr, 0, 0),
                                                descA, descB, descI_nt, False, CTA_GROUP, ki > 0)  # noqa: F821
                                        for ki in Tx.unroll(CHUNK // MMA_K):
                                            Tx.ptx.tcgen05.encode_matrix_descriptor(
                                                Tx.address_of(descA), work_b.ptr_to([0, ki * MMA_K]),  # noqa: E501, F821
                                                MMA_LDO, MMA_SDO, SWIZZLE)
                                            Tx.ptx.tcgen05.encode_matrix_descriptor(
                                                Tx.address_of(descB), V_lo.ptr_to([tic, ki * MMA_K, 0]),  # noqa: E501, F821
                                                MMA_LDO, MMA_SDO, SWIZZLE)
                                            Tx.ptx.tcgen05.mma("float32", "float16", "float16",
                                                Tx.cuda.get_tmem_addr(tmem_addr, 0, 0),
                                                descA, descB, descI_nt, False, CTA_GROUP, True)  # noqa: F821
                                        mma2consumer_bar.arrive(0, CTA_GROUP, 1)

                                        # Wait consumer done with Phase 3
                                        consumer2mma_bar.wait(0, phase_c2m)
                                        Tx.ptx.tcgen05.fence.after_thread_sync()

                                        # == Phase 4: work_a@V_prev_hi + work_b@V_curr_hi (transB=True) ==  # noqa: E501
                                        for ki in Tx.unroll(CHUNK // MMA_K):
                                            Tx.ptx.tcgen05.encode_matrix_descriptor(
                                                Tx.address_of(descA), work_a.ptr_to([0, ki * MMA_K]),  # noqa: E501, F821
                                                MMA_LDO, MMA_SDO, SWIZZLE)
                                            Tx.ptx.tcgen05.encode_matrix_descriptor(
                                                Tx.address_of(descB), V_hi.ptr_to([prev_tic, ki * MMA_K, 0]),  # noqa: E501, F821
                                                MMA_LDO, MMA_SDO, SWIZZLE)
                                            Tx.ptx.tcgen05.mma("float32", "float16", "float16",
                                                Tx.cuda.get_tmem_addr(tmem_addr, 0, 0),
                                                descA, descB, descI_nt, False, CTA_GROUP, ki > 0)  # noqa: F821
                                        for ki in Tx.unroll(CHUNK // MMA_K):
                                            Tx.ptx.tcgen05.encode_matrix_descriptor(
                                                Tx.address_of(descA), work_b.ptr_to([0, ki * MMA_K]),  # noqa: E501, F821
                                                MMA_LDO, MMA_SDO, SWIZZLE)
                                            Tx.ptx.tcgen05.encode_matrix_descriptor(
                                                Tx.address_of(descB), V_hi.ptr_to([tic, ki * MMA_K, 0]),  # noqa: E501, F821
                                                MMA_LDO, MMA_SDO, SWIZZLE)
                                            Tx.ptx.tcgen05.mma("float32", "float16", "float16",
                                                Tx.cuda.get_tmem_addr(tmem_addr, 0, 0),
                                                descA, descB, descI_nt, False, CTA_GROUP, True)  # noqa: F821
                                        mma2consumer_bar.arrive(0, CTA_GROUP, 1)

                                        # Wait consumer done with Phase 4
                                        consumer2mma_bar.wait(0, phase_c2m ^ 1)
                                        Tx.ptx.tcgen05.fence.after_thread_sync()

                                        # == Phase 5: Q_lo@qmap_lo + Q_hi@qmap_hi (transB=True) ==
                                        for ki in Tx.unroll(HALF // MMA_K):
                                            Tx.ptx.tcgen05.encode_matrix_descriptor(
                                                Tx.address_of(descA), Q_lo.ptr_to([tic, 0, ki * MMA_K]),  # noqa: E501, F821
                                                MMA_LDO, MMA_SDO, SWIZZLE)
                                            Tx.ptx.tcgen05.encode_matrix_descriptor(
                                                Tx.address_of(descB), qmap_lo.ptr_to([ki * MMA_K, 0]),  # noqa: E501, F821
                                                MMA_LDO, MMA_SDO, SWIZZLE)
                                            Tx.ptx.tcgen05.mma("float32", "float16", "float16",
                                                Tx.cuda.get_tmem_addr(tmem_addr, 0, 0),
                                                descA, descB, descI_nt, False, CTA_GROUP, ki > 0)  # noqa: F821
                                        for ki in Tx.unroll(HALF // MMA_K):
                                            Tx.ptx.tcgen05.encode_matrix_descriptor(
                                                Tx.address_of(descA), Q_hi.ptr_to([tic, 0, ki * MMA_K]),  # noqa: E501, F821
                                                MMA_LDO, MMA_SDO, SWIZZLE)
                                            Tx.ptx.tcgen05.encode_matrix_descriptor(
                                                Tx.address_of(descB), qmap_hi.ptr_to([ki * MMA_K, 0]),  # noqa: E501, F821
                                                MMA_LDO, MMA_SDO, SWIZZLE)
                                            Tx.ptx.tcgen05.mma("float32", "float16", "float16",
                                                Tx.cuda.get_tmem_addr(tmem_addr, 0, 0),
                                                descA, descB, descI_nt, False, CTA_GROUP, True)  # noqa: F821
                                        mma2consumer_bar.arrive(0, CTA_GROUP, 1)

                                        # Wait consumer done with Phase 5 (featurize, stage operands)  # noqa: E501
                                        consumer2mma_bar.wait(0, phase_c2m)
                                        Tx.ptx.tcgen05.fence.after_thread_sync()

                                        # == Phase 6: feat_q_pos@kv[0:64,0:64] + feat_q_neg@kv[64:128,0:64] -> o_lo_add ==  # noqa: E501
                                        # A=work_a[64,64] (feat_q_pos), B=kv_state_f16[0:64,0:64]
                                        for ki in Tx.unroll(HALF // MMA_K):
                                            Tx.ptx.tcgen05.encode_matrix_descriptor(
                                                Tx.address_of(descA), work_a.ptr_to([0, ki * MMA_K]),  # noqa: E501, F821
                                                MMA_LDO, MMA_SDO, SWIZZLE)
                                            Tx.ptx.tcgen05.encode_matrix_descriptor(
                                                Tx.address_of(descB), kv_state_f16.ptr_to([ki * MMA_K, 0]),  # noqa: E501, F821
                                                MMA_LDO, MMA_SDO, SWIZZLE)
                                            Tx.ptx.tcgen05.mma("float32", "float16", "float16",
                                                Tx.cuda.get_tmem_addr(tmem_addr, 0, 0),
                                                descA, descB, descI_nt, False, CTA_GROUP, ki > 0)  # noqa: F821
                                        # A=work_b[64,64] (feat_q_neg), B=kv_state_f16[64:128,0:64]
                                        for ki in Tx.unroll(HALF // MMA_K):
                                            Tx.ptx.tcgen05.encode_matrix_descriptor(
                                                Tx.address_of(descA), work_b.ptr_to([0, ki * MMA_K]),  # noqa: E501, F821
                                                MMA_LDO, MMA_SDO, SWIZZLE)
                                            Tx.ptx.tcgen05.encode_matrix_descriptor(
                                                Tx.address_of(descB), kv_state_f16.ptr_to([HALF + ki * MMA_K, 0]),  # noqa: E501, F821
                                                MMA_LDO, MMA_SDO, SWIZZLE)
                                            Tx.ptx.tcgen05.mma("float32", "float16", "float16",
                                                Tx.cuda.get_tmem_addr(tmem_addr, 0, 0),
                                                descA, descB, descI_nt, False, CTA_GROUP, True)  # noqa: F821
                                        mma2consumer_bar.arrive(0, CTA_GROUP, 1)

                                        # Wait consumer done with Phase 6 (add to o_lo, stage kv hi cols)  # noqa: E501
                                        consumer2mma_bar.wait(0, phase_c2m ^ 1)
                                        Tx.ptx.tcgen05.fence.after_thread_sync()

                                        # == Phase 7: feat_q_pos@kv[0:64,64:128] + feat_q_neg@kv[64:128,64:128] -> o_hi_add ==  # noqa: E501
                                        for ki in Tx.unroll(HALF // MMA_K):
                                            Tx.ptx.tcgen05.encode_matrix_descriptor(
                                                Tx.address_of(descA), work_a.ptr_to([0, ki * MMA_K]),  # noqa: E501, F821
                                                MMA_LDO, MMA_SDO, SWIZZLE)
                                            Tx.ptx.tcgen05.encode_matrix_descriptor(
                                                Tx.address_of(descB), kv_state_f16.ptr_to([ki * MMA_K, 0]),  # noqa: E501, F821
                                                MMA_LDO, MMA_SDO, SWIZZLE)
                                            Tx.ptx.tcgen05.mma("float32", "float16", "float16",
                                                Tx.cuda.get_tmem_addr(tmem_addr, 0, 0),
                                                descA, descB, descI_nt, False, CTA_GROUP, ki > 0)  # noqa: F821
                                        for ki in Tx.unroll(HALF // MMA_K):
                                            Tx.ptx.tcgen05.encode_matrix_descriptor(
                                                Tx.address_of(descA), work_b.ptr_to([0, ki * MMA_K]),  # noqa: E501, F821
                                                MMA_LDO, MMA_SDO, SWIZZLE)
                                            Tx.ptx.tcgen05.encode_matrix_descriptor(
                                                Tx.address_of(descB), kv_state_f16.ptr_to([HALF + ki * MMA_K, 0]),  # noqa: E501, F821
                                                MMA_LDO, MMA_SDO, SWIZZLE)
                                            Tx.ptx.tcgen05.mma("float32", "float16", "float16",
                                                Tx.cuda.get_tmem_addr(tmem_addr, 0, 0),
                                                descA, descB, descI_nt, False, CTA_GROUP, True)  # noqa: F821
                                        mma2consumer_bar.arrive(0, CTA_GROUP, 1)

                                        # Wait consumer done with Phase 7 (add to o_hi, output, prepare)  # noqa: E501
                                        consumer2mma_bar.wait(0, phase_c2m)
                                        Tx.ptx.tcgen05.fence.after_thread_sync()

                                        # == Phase 8: K_prev_lo@kmap_lo + K_prev_hi@kmap_hi (transB=True) ==  # noqa: E501
                                        for ki in Tx.unroll(HALF // MMA_K):
                                            Tx.ptx.tcgen05.encode_matrix_descriptor(
                                                Tx.address_of(descA), K_lo.ptr_to([prev_tic, 0, ki * MMA_K]),  # noqa: E501, F821
                                                MMA_LDO, MMA_SDO, SWIZZLE)
                                            Tx.ptx.tcgen05.encode_matrix_descriptor(
                                                Tx.address_of(descB), kmap_lo.ptr_to([ki * MMA_K, 0]),  # noqa: E501, F821
                                                MMA_LDO, MMA_SDO, SWIZZLE)
                                            Tx.ptx.tcgen05.mma("float32", "float16", "float16",
                                                Tx.cuda.get_tmem_addr(tmem_addr, 0, 0),
                                                descA, descB, descI_nt, False, CTA_GROUP, ki > 0)  # noqa: F821
                                        for ki in Tx.unroll(HALF // MMA_K):
                                            Tx.ptx.tcgen05.encode_matrix_descriptor(
                                                Tx.address_of(descA), K_hi.ptr_to([prev_tic, 0, ki * MMA_K]),  # noqa: E501, F821
                                                MMA_LDO, MMA_SDO, SWIZZLE)
                                            Tx.ptx.tcgen05.encode_matrix_descriptor(
                                                Tx.address_of(descB), kmap_hi.ptr_to([ki * MMA_K, 0]),  # noqa: E501, F821
                                                MMA_LDO, MMA_SDO, SWIZZLE)
                                            Tx.ptx.tcgen05.mma("float32", "float16", "float16",
                                                Tx.cuda.get_tmem_addr(tmem_addr, 0, 0),
                                                descA, descB, descI_nt, False, CTA_GROUP, True)  # noqa: F821
                                        mma2consumer_bar.arrive(0, CTA_GROUP, 1)

                                        # Wait consumer done with Phase 8 (featurize K, write feat_k to work_a/b)  # noqa: E501
                                        consumer2mma_bar.wait(0, phase_c2m ^ 1)
                                        Tx.ptx.tcgen05.fence.after_thread_sync()

                                        # == Phases 9-12: feat_k^T @ V -> kv_state update ==
                                        # work_a=feat_k_pos[64,64], work_b=feat_k_neg[64,64]
                                        # V_lo[prev_tic,64,64], V_hi[prev_tic,64,64]
                                        # Phase 9:  work_a^T @ V_lo -> update kv_state[0:64, 0:64]
                                        # Phase 10: work_a^T @ V_hi -> update kv_state[0:64, 64:128]
                                        # Phase 11: work_b^T @ V_lo -> update kv_state[64:128, 0:64]
                                        # Phase 12: work_b^T @ V_hi -> update kv_state[64:128, 64:128]  # noqa: E501
                                        for state_phase in Tx.unroll(4):
                                            for ki in Tx.unroll(CHUNK // MMA_K):  # 4 K-iterations
                                                # Select A operand: work_a (phases 0,1) or work_b (phases 2,3)  # noqa: E501
                                                if state_phase < 2:
                                                    Tx.ptx.tcgen05.encode_matrix_descriptor(
                                                        Tx.address_of(descA), work_a.ptr_to([ki * MMA_K, 0]),  # noqa: E501, F821
                                                        MMA_LDO, MMA_SDO, SWIZZLE)
                                                else:
                                                    Tx.ptx.tcgen05.encode_matrix_descriptor(
                                                        Tx.address_of(descA), work_b.ptr_to([ki * MMA_K, 0]),  # noqa: E501, F821
                                                        MMA_LDO, MMA_SDO, SWIZZLE)
                                                # Select B operand: V_lo (phases 0,2) or V_hi (phases 1,3)  # noqa: E501
                                                if state_phase % 2 == 0:
                                                    Tx.ptx.tcgen05.encode_matrix_descriptor(
                                                        Tx.address_of(descB), V_lo.ptr_to([prev_tic, ki * MMA_K, 0]),  # noqa: E501, F821
                                                        MMA_LDO, MMA_SDO, SWIZZLE)
                                                else:
                                                    Tx.ptx.tcgen05.encode_matrix_descriptor(
                                                        Tx.address_of(descB), V_hi.ptr_to([prev_tic, ki * MMA_K, 0]),  # noqa: E501, F821
                                                        MMA_LDO, MMA_SDO, SWIZZLE)
                                                Tx.ptx.tcgen05.mma("float32", "float16", "float16",
                                                    Tx.cuda.get_tmem_addr(tmem_addr, 0, 0),
                                                    descA, descB, descI_tt, False, CTA_GROUP, ki > 0)  # noqa: E501, F821
                                            mma2consumer_bar.arrive(0, CTA_GROUP, 1)
                                            # Wait consumer done with this state update phase
                                            consumer2mma_bar.wait(0, phase_c2m ^ (state_phase % 2))
                                            Tx.ptx.tcgen05.fence.after_thread_sync()

                                        # Release prev_tic SMEM slot after ALL phases that read it.
                                        # Skip chunk 0 since prev_tic=1 holds chunk 1's fresh TMA data.  # noqa: E501
                                        if cid_mma > 0:
                                            Tx.ptx.mbarrier.arrive(mma2tma_bar.ptr_to([prev_tic]))
                                        # 12 phases = even -> do NOT flip phase_c2m

                                    if cid_mma % PIPE_DEPTH == PIPE_DEPTH - 1:
                                        phase[0] = phase[0] ^ 1
                                    cid_mma = cid_mma + 1
                                # Release the last chunk's tic slot (the inner loop only
                                # releases prev_tic, so the final tic is never released).
                                # This is needed for persistent grid: TMA's mma2tma phase
                                # must be balanced across work items.
                                with Tx.thread()[Tx.ptx.elect_sync()]:
                                    last_tic = Tx.meta_var((nc_mma - 1) % PIPE_DEPTH)
                                    Tx.ptx.mbarrier.arrive(mma2tma_bar.ptr_to([last_tic]))
                                    Tx.ptx.mbarrier.arrive(workitem_sync_bar.ptr_to([0]))
                                wid_mma = wid_mma + SM_COUNT

                    # === Consumer warpgroup (WG0) ===
                    with Tx.warpgroup()[0:1]:
                        Tx.cuda.trap_when_assert_failed(tmem_addr == 0)

                        # Per-thread local storage
                        att_prev_reg = Tx.alloc_buffer((HALF,), "float32", scope="local")
                        o_lo_reg = Tx.alloc_buffer((HALF,), "float32", scope="local")
                        o_hi_reg = Tx.alloc_buffer((HALF,), "float32", scope="local")
                        proj_reg = Tx.alloc_buffer((HALF,), "float32", scope="local")
                        feat_q_pos = Tx.alloc_buffer((HALF,), "float32", scope="local")
                        feat_q_neg = Tx.alloc_buffer((HALF,), "float32", scope="local")
                        # Scalar accumulators (cannot use Tx.meta_var for accumulation)
                        sc = Tx.alloc_buffer((8,), "float32", scope="local")
                        # sc[0]=row_max, sc[1]=exp_sum/s_norm, sc[2]=l_norm
                        # sc[3]=pos_max, sc[4]=pos_sum, sc[5]=neg_max, sc[6]=neg_sum, sc[7]=denom
                        phase_m2c: Tx.int32
                        phase_m2c = 0
                        phase_c2m_c: Tx.int32
                        phase_c2m_c = 0

                        wid_con = Tx.local_scalar("int32", "wid_con")
                        cid_con = Tx.local_scalar("int32", "cid_con")
                        nc_con = Tx.local_scalar("int32", "nc_con")
                        wid_con = bx
                        while wid_con < total_bh:
                            nc_con = N_param // CHUNK
                            bx_head_con = Tx.meta_var(wid_con % H_param)

                            # Load alpha, beta into SMEM
                            with Tx.thread():
                                tid_flat = Tx.meta_var(warp_id * 32 + lane_id)
                                if tid_flat == 0:
                                    alphas_s[0] = alphas_g[bx_head_con]
                                    betas_s[0] = betas_g[bx_head_con]

                            # Load qmap, kmap into swizzled SMEM
                            # qmap_g[head, 0:64, 0:64] -> qmap_lo, qmap_g[head, 64:128, 0:64] -> qmap_hi  # noqa: E501
                            # kmap_g[head, 0:64, 0:64] -> kmap_lo, kmap_g[head, 64:128, 0:64] -> kmap_hi  # noqa: E501
                            with Tx.thread():
                                tid_flat = Tx.meta_var(warp_id * 32 + lane_id)
                                # 4096 elements per buffer, 128 threads -> 32 elements per thread
                                for ki in Tx.serial(32):
                                    idx = Tx.meta_var(tid_flat * 32 + ki)
                                    row = Tx.meta_var(idx // HALF)
                                    col = Tx.meta_var(idx % HALF)
                                    qmap_lo[row, col] = qmap_g[bx_head_con, row, col]
                                    qmap_hi[row, col] = qmap_g[bx_head_con, HALF + row, col]
                                    kmap_lo[row, col] = kmap_g[bx_head_con, row, col]
                                    kmap_hi[row, col] = kmap_g[bx_head_con, HALF + row, col]

                            # Zero KV state (128*128=16384 f32 / 128 threads = 128 per thread)
                            with Tx.thread():
                                tid_flat = Tx.meta_var(warp_id * 32 + lane_id)
                                for ki in Tx.serial(128):
                                    idx = Tx.meta_var(tid_flat * 128 + ki)
                                    row = Tx.meta_var(idx // ATTN_D)
                                    col = Tx.meta_var(idx % ATTN_D)
                                    kv_state[row, col] = 0.0
                                # Zero k_cumsum
                                if tid_flat < ATTN_F:
                                    k_cumsum[tid_flat] = 0.0
                            Tx.cuda.warpgroup_sync(10)

                            cid_con = 0
                            while cid_con < nc_con:
                                tic_c = Tx.meta_var(cid_con % PIPE_DEPTH)
                                prev_tic_c = Tx.meta_var(1 - tic_c)  # noqa: F841

                                # ======== Phase 1: Read att_prev from TMEM ========
                                mma2consumer_bar.wait(0, phase_m2c)
                                phase_m2c = phase_m2c ^ 1
                                Tx.ptx.tcgen05.fence.after_thread_sync()

                                Tx.copy(reg_wg[:, :], tmem[:, 0:TMEM_LD_SIZE])

                                with Tx.thread():
                                    out_row = Tx.meta_var(warp_id * 16 + lane_id)
                                    if lane_id < 16:
                                        if cid_con == 0:
                                            # Chunk 0: no previous chunk, mask all to -10000
                                            for j in Tx.serial(HALF):
                                                att_prev_reg[j] = -10000.0
                                        else:
                                            for j in Tx.serial(HALF):
                                                att_prev_reg[j] = reg[j]

                                Tx.ptx.tcgen05.fence.before_thread_sync()
                                consumer2mma_bar.arrive(0)
                                phase_c2m_c = phase_c2m_c ^ 1

                                # ======== Phase 2: Read att_curr from TMEM, softmax, write work ========  # noqa: E501
                                mma2consumer_bar.wait(0, phase_m2c)
                                phase_m2c = phase_m2c ^ 1
                                Tx.ptx.tcgen05.fence.after_thread_sync()

                                Tx.copy(reg_wg[:, :], tmem[:, 0:TMEM_LD_SIZE])

                                with Tx.thread():
                                    out_row = Tx.meta_var(warp_id * 16 + lane_id)
                                    if lane_id < 16:
                                        # Apply causal mask to att_curr
                                        for j in Tx.serial(HALF):
                                            if out_row < j:
                                                reg[j] = -10000.0

                                        # Scale by temperature
                                        for j in Tx.serial(HALF):
                                            att_prev_reg[j] = att_prev_reg[j] * TEMP
                                            reg[j] = reg[j] * TEMP

                                        # Row max across 128 values
                                        sc[0] = -10000.0  # row_max
                                        for j in Tx.serial(HALF):
                                            if att_prev_reg[j] > sc[0]:
                                                sc[0] = att_prev_reg[j]
                                            if reg[j] > sc[0]:
                                                sc[0] = reg[j]

                                        # Exp and sum
                                        sc[1] = 0.0  # exp_sum
                                        for j in Tx.serial(HALF):
                                            att_prev_reg[j] = Tx.exp(att_prev_reg[j] - sc[0]) * betas_s[0]  # noqa: E501
                                            sc[1] = sc[1] + att_prev_reg[j]
                                            reg[j] = Tx.exp(reg[j] - sc[0]) * betas_s[0]
                                            sc[1] = sc[1] + reg[j]

                                        # Write att_prev to work_a, att_curr to work_b as f16
                                        for j in Tx.serial(HALF):
                                            work_a[out_row, j] = Tx.cast(att_prev_reg[j], "float16")
                                            work_b[out_row, j] = Tx.cast(reg[j], "float16")

                                        # Save s_norm for later normalization
                                        att_prev_reg[0] = sc[1]

                                Tx.ptx.tcgen05.fence.before_thread_sync()
                                consumer2mma_bar.arrive(0)
                                phase_c2m_c = phase_c2m_c ^ 1

                                # ======== Phase 3: Read o_lo from TMEM ========
                                mma2consumer_bar.wait(0, phase_m2c)
                                phase_m2c = phase_m2c ^ 1
                                Tx.ptx.tcgen05.fence.after_thread_sync()

                                Tx.copy(reg_wg[:, :], tmem[:, 0:TMEM_LD_SIZE])

                                with Tx.thread():
                                    out_row = Tx.meta_var(warp_id * 16 + lane_id)
                                    if lane_id < 16:
                                        for j in Tx.serial(HALF):
                                            o_lo_reg[j] = reg[j]

                                Tx.ptx.tcgen05.fence.before_thread_sync()
                                consumer2mma_bar.arrive(0)
                                phase_c2m_c = phase_c2m_c ^ 1

                                # ======== Phase 4: Read o_hi from TMEM ========
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

                                # ======== Phase 5: Read Q@qmap from TMEM -> featurize -> stage operands ========  # noqa: E501
                                mma2consumer_bar.wait(0, phase_m2c)
                                phase_m2c = phase_m2c ^ 1
                                Tx.ptx.tcgen05.fence.after_thread_sync()

                                Tx.copy(reg_wg[:, :], tmem[:, 0:TMEM_LD_SIZE])

                                with Tx.thread():
                                    out_row = Tx.meta_var(warp_id * 16 + lane_id)
                                    if lane_id < 16:
                                        sc[2] = 0.0  # l_norm

                                        if cid_con > 0:
                                            # Save projection for featurize
                                            for j in Tx.serial(HALF):
                                                proj_reg[j] = reg[j]

                                            # Positive softmax featurize
                                            sc[3] = -10000.0  # pos_max
                                            for j in Tx.serial(HALF):
                                                if proj_reg[j] > sc[3]:
                                                    sc[3] = proj_reg[j]
                                            sc[4] = 0.0  # pos_sum
                                            for j in Tx.serial(HALF):
                                                feat_q_pos[j] = Tx.exp(proj_reg[j] - sc[3])
                                                sc[4] = sc[4] + feat_q_pos[j]
                                            for j in Tx.serial(HALF):
                                                feat_q_pos[j] = feat_q_pos[j] / sc[4] * alphas_s[0]

                                            # Negative softmax featurize
                                            sc[5] = -10000.0  # neg_max
                                            for j in Tx.serial(HALF):
                                                sc[7] = 0.0 - proj_reg[j]
                                                if sc[7] > sc[5]:
                                                    sc[5] = sc[7]
                                            sc[6] = 0.0  # neg_sum
                                            for j in Tx.serial(HALF):
                                                feat_q_neg[j] = Tx.exp(0.0 - proj_reg[j] - sc[5])
                                                sc[6] = sc[6] + feat_q_neg[j]
                                            for j in Tx.serial(HALF):
                                                feat_q_neg[j] = feat_q_neg[j] / sc[6] * alphas_s[0]

                                            # l_norm = feat_q . k_cumsum
                                            for f_idx in Tx.serial(HALF):
                                                sc[2] = sc[2] + feat_q_pos[f_idx] * k_cumsum[f_idx]
                                                sc[2] = sc[2] + feat_q_neg[f_idx] * k_cumsum[HALF + f_idx]  # noqa: E501

                                            # Write feat_q to work_a (pos) / work_b (neg) for MMA phases 6-7  # noqa: E501
                                            for j in Tx.serial(HALF):
                                                work_a[out_row, j] = Tx.cast(feat_q_pos[j], "float16")  # noqa: E501
                                                work_b[out_row, j] = Tx.cast(feat_q_neg[j], "float16")  # noqa: E501

                                # Stage kv_state[0:128, 0:64] as f16 for Phase 6 MMA (all threads cooperate)  # noqa: E501
                                # 128*64=8192 elements / 128 threads = 64 per thread
                                with Tx.thread():
                                    tid_flat = Tx.meta_var(warp_id * 32 + lane_id)
                                    if cid_con > 0:
                                        for ki in Tx.serial(64):
                                            idx = Tx.meta_var(tid_flat * 64 + ki)
                                            row = Tx.meta_var(idx // HALF)
                                            col = Tx.meta_var(idx % HALF)
                                            kv_state_f16[row, col] = Tx.cast(kv_state[row, col], "float16")  # noqa: E501

                                Tx.ptx.tcgen05.fence.before_thread_sync()
                                consumer2mma_bar.arrive(0)
                                phase_c2m_c = phase_c2m_c ^ 1

                                # ======== Phase 6: Read feat_q@kv_lo from TMEM -> add to o_lo ========  # noqa: E501
                                mma2consumer_bar.wait(0, phase_m2c)
                                phase_m2c = phase_m2c ^ 1
                                Tx.ptx.tcgen05.fence.after_thread_sync()

                                Tx.copy(reg_wg[:, :], tmem[:, 0:TMEM_LD_SIZE])

                                with Tx.thread():
                                    out_row = Tx.meta_var(warp_id * 16 + lane_id)
                                    if lane_id < 16:
                                        if cid_con > 0:
                                            for j in Tx.serial(HALF):
                                                o_lo_reg[j] = o_lo_reg[j] + reg[j]

                                # Stage kv_state[0:128, 64:128] as f16 for Phase 7 MMA
                                with Tx.thread():
                                    tid_flat = Tx.meta_var(warp_id * 32 + lane_id)
                                    if cid_con > 0:
                                        for ki in Tx.serial(64):
                                            idx = Tx.meta_var(tid_flat * 64 + ki)
                                            row = Tx.meta_var(idx // HALF)
                                            col = Tx.meta_var(idx % HALF)
                                            kv_state_f16[row, col] = Tx.cast(kv_state[row, HALF + col], "float16")  # noqa: E501

                                Tx.ptx.tcgen05.fence.before_thread_sync()
                                consumer2mma_bar.arrive(0)
                                phase_c2m_c = phase_c2m_c ^ 1

                                # ======== Phase 7: Read feat_q@kv_hi from TMEM -> add to o_hi -> normalize -> output ========  # noqa: E501
                                mma2consumer_bar.wait(0, phase_m2c)
                                phase_m2c = phase_m2c ^ 1
                                Tx.ptx.tcgen05.fence.after_thread_sync()

                                Tx.copy(reg_wg[:, :], tmem[:, 0:TMEM_LD_SIZE])

                                with Tx.thread():
                                    out_row = Tx.meta_var(warp_id * 16 + lane_id)
                                    if lane_id < 16:
                                        if cid_con > 0:
                                            for j in Tx.serial(HALF):
                                                o_hi_reg[j] = o_hi_reg[j] + reg[j]

                                        # Normalize: o /= max(s_norm + l_norm, 1e-6)
                                        sc[7] = att_prev_reg[0] + sc[2]  # denom = s_norm + l_norm
                                        if sc[7] < 0.000001:
                                            sc[7] = 0.000001
                                        for j in Tx.serial(HALF):
                                            D_out_lo[out_row, j] = Tx.cast(o_lo_reg[j] / sc[7], "float16")  # noqa: E501
                                        for j in Tx.serial(HALF):
                                            D_out_hi[out_row, j] = Tx.cast(o_hi_reg[j] / sc[7], "float16")  # noqa: E501

                                # TMA store output (two halves)
                                Tx.ptx.fence.proxy_async("shared::cta")
                                Tx.cuda.warpgroup_sync(10)
                                with Tx.thread()[lane_id == 0 and warp_id == 0]:
                                    q_row_out = Tx.meta_var(cid_con * CHUNK)
                                    Tx.copy_async(o_g[wid_con, q_row_out : q_row_out + CHUNK, 0 : HALF],  # noqa: E501
                                                  D_out_lo[:, :], dispatch="tma", cache_hint="evict_last")  # noqa: E501
                                    Tx.copy_async(o_g[wid_con, q_row_out : q_row_out + CHUNK, HALF : ATTN_D],  # noqa: E501
                                                  D_out_hi[:, :], dispatch="tma", cache_hint="evict_last")  # noqa: E501
                                    Tx.ptx.cp_async.bulk.commit_group()
                                    Tx.ptx.cp_async.bulk.wait_group(0)
                                Tx.cuda.warpgroup_sync(10)

                                Tx.ptx.tcgen05.fence.before_thread_sync()
                                consumer2mma_bar.arrive(0)
                                phase_c2m_c = phase_c2m_c ^ 1

                                # ======== Phase 8: Read K_prev@kmap from TMEM -> featurize -> write feat_k ========  # noqa: E501
                                mma2consumer_bar.wait(0, phase_m2c)
                                phase_m2c = phase_m2c ^ 1
                                Tx.ptx.tcgen05.fence.after_thread_sync()

                                Tx.copy(reg_wg[:, :], tmem[:, 0:TMEM_LD_SIZE])

                                with Tx.thread():
                                    out_row = Tx.meta_var(warp_id * 16 + lane_id)
                                    if lane_id < 16:
                                        if cid_con > 0:
                                            # Save projection
                                            for j in Tx.serial(HALF):
                                                proj_reg[j] = reg[j]

                                            # Positive softmax featurize (no alpha scaling for K)
                                            sc[3] = -10000.0  # pos_max
                                            for j in Tx.serial(HALF):
                                                if proj_reg[j] > sc[3]:
                                                    sc[3] = proj_reg[j]
                                            sc[4] = 0.0  # pos_sum
                                            for j in Tx.serial(HALF):
                                                feat_q_pos[j] = Tx.exp(proj_reg[j] - sc[3])
                                                sc[4] = sc[4] + feat_q_pos[j]
                                            for j in Tx.serial(HALF):
                                                feat_q_pos[j] = feat_q_pos[j] / sc[4]

                                            # Negative softmax featurize
                                            sc[5] = -10000.0  # neg_max
                                            for j in Tx.serial(HALF):
                                                sc[7] = 0.0 - proj_reg[j]
                                                if sc[7] > sc[5]:
                                                    sc[5] = sc[7]
                                            sc[6] = 0.0  # neg_sum
                                            for j in Tx.serial(HALF):
                                                feat_q_neg[j] = Tx.exp(0.0 - proj_reg[j] - sc[5])
                                                sc[6] = sc[6] + feat_q_neg[j]
                                            for j in Tx.serial(HALF):
                                                feat_q_neg[j] = feat_q_neg[j] / sc[6]

                                            # Write feat_k to work_a (pos) / work_b (neg) — swizzled
                                            for j in Tx.serial(HALF):
                                                work_a[out_row, j] = Tx.cast(feat_q_pos[j], "float16")  # noqa: E501
                                                work_b[out_row, j] = Tx.cast(feat_q_neg[j], "float16")  # noqa: E501

                                Tx.cuda.warpgroup_sync(10)

                                # k_cumsum update (light: column sum of feat_k)
                                with Tx.thread():
                                    out_row = Tx.meta_var(warp_id * 16 + lane_id)
                                    if lane_id < 16:
                                        if cid_con > 0:
                                            for col_off in Tx.serial(2):
                                                f_col = Tx.meta_var(out_row * 2 + col_off)
                                                sc[0] = 0.0
                                                if f_col < HALF:
                                                    for pos in Tx.serial(CHUNK):
                                                        sc[0] = sc[0] + Tx.cast(work_a[pos, f_col], "float32")  # noqa: E501
                                                else:
                                                    for pos in Tx.serial(CHUNK):
                                                        sc[0] = sc[0] + Tx.cast(work_b[pos, f_col - HALF], "float32")  # noqa: E501
                                                k_cumsum[f_col] = k_cumsum[f_col] + sc[0]

                                Tx.ptx.tcgen05.fence.before_thread_sync()
                                consumer2mma_bar.arrive(0)
                                phase_c2m_c = phase_c2m_c ^ 1

                                # ======== Phases 9-12: Read feat_k^T@V from TMEM -> update kv_state ========  # noqa: E501
                                # Phase 9:  kv_state[0:64, 0:64] += TMEM
                                # Phase 10: kv_state[0:64, 64:128] += TMEM
                                # Phase 11: kv_state[64:128, 0:64] += TMEM
                                # Phase 12: kv_state[64:128, 64:128] += TMEM
                                for state_phase in Tx.serial(4):
                                    mma2consumer_bar.wait(0, phase_m2c)
                                    phase_m2c = phase_m2c ^ 1
                                    Tx.ptx.tcgen05.fence.after_thread_sync()

                                    Tx.copy(reg_wg[:, :], tmem[:, 0:TMEM_LD_SIZE])

                                    with Tx.thread():
                                        out_row = Tx.meta_var(warp_id * 16 + lane_id)
                                        if lane_id < 16:
                                            if cid_con > 0:
                                                # Compute row/col offsets based on state_phase
                                                # phase 0: row_base=0, col_base=0
                                                # phase 1: row_base=0, col_base=64
                                                # phase 2: row_base=64, col_base=0
                                                # phase 3: row_base=64, col_base=64
                                                row_base = Tx.meta_var((state_phase // 2) * HALF)
                                                col_base = Tx.meta_var((state_phase % 2) * HALF)
                                                for j in Tx.serial(HALF):
                                                    kv_state[row_base + out_row, col_base + j] = kv_state[row_base + out_row, col_base + j] + reg[j]  # noqa: E501

                                    Tx.ptx.tcgen05.fence.before_thread_sync()
                                    consumer2mma_bar.arrive(0)
                                    phase_c2m_c = phase_c2m_c ^ 1

                                cid_con = cid_con + 1

                            # ---- Write KV state and k_cumsum to global ----
                            with Tx.thread():
                                out_row = Tx.meta_var(warp_id * 16 + lane_id)
                                if lane_id < 16:
                                    # Each thread writes 2 rows of kv_state
                                    for row_off in Tx.unroll(2):
                                        my_row = Tx.meta_var(out_row * 2 + row_off)
                                        for d in Tx.serial(ATTN_D):
                                            kvs_g[wid_con, my_row, d] = kv_state[my_row, d]
                                    # k_cumsum: 128 values / 64 threads = 2 per thread
                                    for col_off in Tx.serial(2):
                                        f_col = Tx.meta_var(out_row * 2 + col_off)
                                        ks_g[wid_con, f_col] = k_cumsum[f_col]
                            Tx.cuda.warpgroup_sync(10)

                            wid_con = wid_con + SM_COUNT

                # ---- TMEM deallocation ----
                with Tx.warp()[0:1]:
                    Tx.ptx.tcgen05.relinquish_alloc_permit(cta_group=CTA_GROUP)
                    Tx.ptx.tcgen05.dealloc(tmem_addr, n_cols=N_COLS, cta_group=CTA_GROUP)

                Tx.cuda.cta_sync()
    # fmt: on
    return hedgehog


@pytest.mark.parametrize(
    "B,H,N", [(1, 2, 128), (2, 2, 256), (2, 2, 512), (1, 4, 1024), (10, 16, 128)]
)
def test_hedgehog(B, H, N):
    q, k, v, qmap, kmap, alphas, betas = prepare_data(B, H, N)
    o_ref, kv_ref, k_ref = naive_hedgehog(q, k, v, qmap, kmap, alphas, betas)

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": get_hedgehog_kernel()})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
        src = mod.mod.imports[0].inspect_source()
        print(src[:3000])

    DEV = tvm.cuda(0)
    BH = B * H
    o_np = np.zeros((BH, N, ATTN_D), dtype=np.float16)
    kv_np = np.zeros((BH, ATTN_F, ATTN_D), dtype=np.float32)
    k_np = np.zeros((BH, ATTN_F), dtype=np.float32)

    q_tvm = tvm.runtime.tensor(
        q.to(torch.float16).reshape(BH, N, ATTN_D).contiguous().cpu().numpy(), DEV
    )
    k_tvm = tvm.runtime.tensor(
        k.to(torch.float16).reshape(BH, N, ATTN_D).contiguous().cpu().numpy(), DEV
    )
    v_tvm = tvm.runtime.tensor(
        v.to(torch.float16).reshape(BH, N, ATTN_D).contiguous().cpu().numpy(), DEV
    )
    qmap_tvm = tvm.runtime.tensor(qmap.to(torch.float16).cpu().numpy(), DEV)
    kmap_tvm = tvm.runtime.tensor(kmap.to(torch.float16).cpu().numpy(), DEV)
    alphas_tvm = tvm.runtime.tensor(alphas.cpu().numpy(), DEV)
    betas_tvm = tvm.runtime.tensor(betas.cpu().numpy(), DEV)
    o_tvm = tvm.runtime.tensor(o_np, DEV)
    kv_tvm = tvm.runtime.tensor(kv_np, DEV)
    k_tvm_out = tvm.runtime.tensor(k_np, DEV)

    mod(q_tvm, k_tvm, v_tvm, qmap_tvm, kmap_tvm, alphas_tvm, betas_tvm, o_tvm, kv_tvm, k_tvm_out)

    o_tir = o_tvm.numpy().reshape(B, H, N, ATTN_D)
    o_ref_np = o_ref.to(torch.float16).cpu().numpy()
    np.testing.assert_allclose(o_tir, o_ref_np, rtol=1e-1, atol=1e-1)
    print(f"PASSED output: B={B}, H={H}, N={N}")

    kv_tir = kv_tvm.numpy().reshape(B, H, ATTN_F, ATTN_D)
    kv_ref_np = kv_ref.cpu().numpy()
    np.testing.assert_allclose(kv_tir, kv_ref_np, rtol=1e-1, atol=1e-1)
    print(f"PASSED kv_state: B={B}, H={H}, N={N}")

    k_tir = k_tvm_out.numpy().reshape(B, H, ATTN_F)
    k_ref_np = k_ref.cpu().numpy()
    np.testing.assert_allclose(k_tir, k_ref_np, rtol=1e-1, atol=1e-1)
    print(f"PASSED k_state: B={B}, H={H}, N={N}")


def bench_hedgehog():
    """Benchmark hedgehog attention kernel at TK-equivalent dimensions."""
    batch, heads = 16, 16
    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": get_hedgehog_kernel()})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")

    DEV = tvm.cuda(0)
    BH = batch * heads

    # FLOPS from TK harness.impl:
    # map_flops = 2 * 2*128*64 * BH * N  (Q,K maps: BH*N*128 @ 128x64)
    # sliding_flops = 2*(96*128 + 96*4 + 96*128) * BH * N  (avg window len 96)
    # linear_flops = (128*128*2 + 128*128*2 + 128*4*2 + 128*4*2) * BH * N
    def flops(ms, N):
        map_flops = 2 * 2 * 128 * 64 * BH * N
        sliding_flops = 2 * (96 * 128 + 96 * 4 + 96 * 128) * BH * N
        linear_flops = (128 * 128 * 2 + 128 * 128 * 2 + 128 * 4 * 2 + 128 * 4 * 2) * BH * N
        return (map_flops + sliding_flops + linear_flops) / (ms * 1e-3)

    print(f"\n{'=' * 60}")
    print(f"Hedgehog Attention Benchmark (B={batch}, H={heads})")
    print(f"{'=' * 60}")

    with ProtonContext("hedgehog"):
        for N in [1024, 2048, 4096, 8192]:
            q = torch.randn(batch, heads, N, ATTN_D, dtype=torch.bfloat16, device="cuda")
            k = torch.randn(batch, heads, N, ATTN_D, dtype=torch.bfloat16, device="cuda")
            v = torch.randn(batch, heads, N, ATTN_D, dtype=torch.bfloat16, device="cuda")
            qmap = torch.randn(heads, ATTN_D, HALF_F, dtype=torch.bfloat16, device="cuda")
            kmap = torch.randn(heads, ATTN_D, HALF_F, dtype=torch.bfloat16, device="cuda")
            alphas = torch.ones(heads, dtype=torch.float32, device="cuda")
            betas = torch.ones(heads, dtype=torch.float32, device="cuda")

            o_np = np.zeros((BH, N, ATTN_D), dtype=np.float16)
            kv_np = np.zeros((BH, ATTN_F, ATTN_D), dtype=np.float32)
            k_np = np.zeros((BH, ATTN_F), dtype=np.float32)

            q_tvm = tvm.runtime.tensor(
                q.to(torch.float16).reshape(BH, N, ATTN_D).contiguous().cpu().numpy(), DEV
            )
            k_tvm = tvm.runtime.tensor(
                k.to(torch.float16).reshape(BH, N, ATTN_D).contiguous().cpu().numpy(), DEV
            )
            v_tvm = tvm.runtime.tensor(
                v.to(torch.float16).reshape(BH, N, ATTN_D).contiguous().cpu().numpy(), DEV
            )
            qmap_tvm = tvm.runtime.tensor(qmap.to(torch.float16).cpu().numpy(), DEV)
            kmap_tvm = tvm.runtime.tensor(kmap.to(torch.float16).cpu().numpy(), DEV)
            alphas_tvm = tvm.runtime.tensor(alphas.cpu().numpy(), DEV)
            betas_tvm = tvm.runtime.tensor(betas.cpu().numpy(), DEV)
            o_tvm = tvm.runtime.tensor(o_np, DEV)
            kv_tvm = tvm.runtime.tensor(kv_np, DEV)
            k_tvm_out = tvm.runtime.tensor(k_np, DEV)

            func = lambda: mod(  # noqa: E731
                q_tvm,
                k_tvm,
                v_tvm,
                qmap_tvm,
                kmap_tvm,
                alphas_tvm,
                betas_tvm,
                o_tvm,
                kv_tvm,
                k_tvm_out,
            )
            ms = bench(func, warmup=100, repeat=300, proton_name=f"hedgehog_N{N}")
            tflops = flops(ms, N) / 1e12
            print(f"  N={N:>5d}: {tflops:.2f} TFLOPS, {ms:.3f} ms")

    print()


if __name__ == "__main__":
    test_hedgehog(1, 2, 128)
    bench_hedgehog()
