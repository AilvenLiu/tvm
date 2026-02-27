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
warp separation. 3 MMA phases per chunk (Q@K^T, att@V_lo, att@V_hi) with
thread-level Q@kvs, KV state update, and exponential decay in consumer WG.

Architecture:
  cta_group=1 (single CTA, no clustering)
  MMA_M=64, MMA_N=64, MMA_K=16 (f16)
  WG1 (Producer+MMA): warp 3 = TMA, warps 0-1 = MMA
  WG0 (Consumer): TMEM read, causal mask + decay, Q@kvs, state update

Key challenge: F=128 and D=128 exceed TMEM N_COLS=64 limit.
  - TMA SWIZZLE_128B requires inner dim <= 128B (64 f16 elements).
    So all 128-wide buffers (Q, K, V, D_out) are split into _lo/_hi
    halves of [64,64] each, loaded/stored as separate TMA ops.
  - Q@K^T -> [64,64]: fits in TMEM. K-dim=128 -> 8 MMA_K iterations
    (4 from Q_lo@K_lo, 4 from Q_hi@K_hi with accumulation).
  - att@V -> [64,128]: split V into V_lo[64,64] and V_hi[64,64], two MMA phases.
  - Q@kvs -> [64,128]: done thread-level (kvs is f32 in SMEM).
  - K_dec^T@V -> [128,128]: done thread-level (state update).
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
N_COLS = 64
TMEM_LD_SIZE = 64

# SMEM budget (bytes):
#   barriers=1024
#   Q_lo[2,64,64]*f16=16K + Q_hi[2,64,64]*f16=16K
#   K_lo[2,64,64]*f16=16K + K_hi[2,64,64]*f16=16K
#   V_lo[2,64,64]*f16=16K + V_hi[2,64,64]*f16=16K
#   work[64,64]*f16=8K
#   D_out_lo[64,64]*f16=8K + D_out_hi[64,64]*f16=8K
#   kvs[128,128]*f32=64K + slopes_s[1]*f32=4
#   Total ~185K
SMEM_SIZE = (
    1024  # barriers + alignment
    + 2 * PIPE_DEPTH * CHUNK * HALF * F16_BYTES  # Q_lo + Q_hi
    + 2 * PIPE_DEPTH * CHUNK * HALF * F16_BYTES  # K_lo + K_hi
    + 2 * PIPE_DEPTH * CHUNK * HALF * F16_BYTES  # V_lo + V_hi
    + CHUNK * HALF * F16_BYTES                   # work_smem
    + 2 * CHUNK * HALF * F16_BYTES               # D_out_lo + D_out_hi
    + F * D * F32_BYTES                          # kvs (f32, not swizzled)
    + F32_BYTES                                  # slopes_s
)
assert SMEM_SIZE <= 232448


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


# ---- Barrier classes (following mamba2/BASED pattern) ----

@Tx.meta_class
class Barriers:
    """Base barrier class for mbarrier-based synchronization."""
    def __init__(self, shared_buffer_base, shared_buffer_offs, pipe_depth, is_p2c):
        self.mbar: tvm.tir.Buffer = Tx.decl_buffer(
            (pipe_depth,), "uint64", shared_buffer_base, elem_offset=shared_buffer_offs
        )
        self.init_phase = 0 if is_p2c else 1
        self.pipe_depth = pipe_depth

    @Tx.inline
    def init(self, threads_num_wait):
        with Tx.thread()[0:1]:
            for i in Tx.serial(self.pipe_depth):
                Tx.ptx.mbarrier.init(self.mbar.ptr_to([i]), threads_num_wait)

    @Tx.inline
    def wait(self, idx, phase):
        Tx.ptx.mbarrier.try_wait(self.mbar.ptr_to([idx]), self.init_phase ^ phase)


class BarTMA2MMA(Barriers):
    """TMA load done -> MMA can start reading SMEM."""
    @Tx.inline
    def arrive(self, idx, expected_bytes):
        Tx.ptx.mbarrier.arrive.expect_tx(self.mbar.ptr_to([idx]), expected_bytes)

    @Tx.inline
    def arrive_only(self, idx):
        Tx.ptx.mbarrier.arrive(self.mbar.ptr_to([idx]))


class BarMMA2TMA(Barriers):
    """MMA done with SMEM -> TMA can overwrite."""
    @Tx.inline
    def arrive(self, idx):
        Tx.ptx.tcgen05.commit(self.mbar.ptr_to([idx]), cta_group=CTA_GROUP)


class BarMMA2Consumer(Barriers):
    """MMA done -> consumer can read TMEM."""
    @Tx.inline
    def arrive(self, idx):
        Tx.ptx.tcgen05.commit(self.mbar.ptr_to([idx]), cta_group=CTA_GROUP)


class BarConsumer2MMA(Barriers):
    """Consumer done -> MMA can proceed."""
    @Tx.inline
    def arrive(self, idx):
        Tx.ptx.mbarrier.arrive(self.mbar.ptr_to([idx]), cta_id=0, pred=True)


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
                # ---- Shared memory allocation ----
                buf = Tx.alloc_buffer([SMEM_SIZE], "uint8", scope="shared.dyn")
                tmem_addr = Tx.decl_scalar("uint32", buf.data, scope="shared.dyn", elem_offset=0)

                # SMEM layout: all swizzled [64,64] f16 halves first, then non-swizzled.
                # Each [pipe,64,64] f16 = pipe*8192 bytes, each [64,64] f16 = 8192 bytes.
                # Each half-buffer is PIPE_DEPTH*CHUNK*HALF = 8192 f16 elems = 16KB
                HALF_BUF = PIPE_DEPTH * CHUNK * HALF  # 8192 f16 elems
                HALF_SINGLE = CHUNK * HALF             # 4096 f16 elems

                q_lo_off = 1024 // F16_BYTES  # 512
                Q_lo = Tx.decl_buffer((PIPE_DEPTH, CHUNK, HALF), "float16", buf.data,
                                      layout=half_pipe_layout, elem_offset=q_lo_off)
                q_hi_off = q_lo_off + HALF_BUF  # 8704
                Q_hi = Tx.decl_buffer((PIPE_DEPTH, CHUNK, HALF), "float16", buf.data,
                                      layout=half_pipe_layout, elem_offset=q_hi_off)
                k_lo_off = q_hi_off + HALF_BUF  # 16896
                K_lo = Tx.decl_buffer((PIPE_DEPTH, CHUNK, HALF), "float16", buf.data,
                                      layout=half_pipe_layout, elem_offset=k_lo_off)
                k_hi_off = k_lo_off + HALF_BUF  # 25088
                K_hi = Tx.decl_buffer((PIPE_DEPTH, CHUNK, HALF), "float16", buf.data,
                                      layout=half_pipe_layout, elem_offset=k_hi_off)
                v_lo_off = k_hi_off + HALF_BUF  # 33280
                V_lo = Tx.decl_buffer((PIPE_DEPTH, CHUNK, HALF), "float16", buf.data,
                                      layout=half_pipe_layout, elem_offset=v_lo_off)
                v_hi_off = v_lo_off + HALF_BUF  # 41472
                V_hi = Tx.decl_buffer((PIPE_DEPTH, CHUNK, HALF), "float16", buf.data,
                                      layout=half_pipe_layout, elem_offset=v_hi_off)
                work_off = v_hi_off + HALF_BUF  # 49664
                work_smem = Tx.decl_buffer((CHUNK, HALF), "float16", buf.data,
                                           layout=half_layout, elem_offset=work_off)
                d_out_lo_off = work_off + HALF_SINGLE  # 53760
                D_out_lo = Tx.decl_buffer((CHUNK, HALF), "float16", buf.data,
                                          layout=half_layout, elem_offset=d_out_lo_off)
                d_out_hi_off = d_out_lo_off + HALF_SINGLE  # 57856
                D_out_hi = Tx.decl_buffer((CHUNK, HALF), "float16", buf.data,
                                          layout=half_layout, elem_offset=d_out_hi_off)

                # --- Non-swizzled buffers (f32) ---
                nonswizzled_byte = (d_out_hi_off + HALF_SINGLE) * F16_BYTES  # 123904
                kvs_off = nonswizzled_byte // F32_BYTES  # 30976
                kvs = Tx.decl_buffer((F, D), "float32", buf.data, elem_offset=kvs_off)
                slopes_off = kvs_off + F * D  # 47360
                slopes_s = Tx.decl_buffer((1,), "float32", buf.data, elem_offset=slopes_off)

                # ---- Local variables ----
                descI: Tx.uint32
                descA: Tx.uint64
                descB: Tx.uint64
                phase = Tx.alloc_buffer((1,), "int32", scope="local")

                # ---- Barrier setup ----
                tma2mma_bar = BarTMA2MMA(buf.data, 4, PIPE_DEPTH, True)
                mma2tma_bar = BarMMA2TMA(buf.data, 4 + PIPE_DEPTH, PIPE_DEPTH, False)
                mma2consumer_bar = BarMMA2Consumer(buf.data, 4 + 2 * PIPE_DEPTH, 1, True)
                consumer2mma_bar = BarConsumer2MMA(buf.data, 4 + 2 * PIPE_DEPTH + 1, 1, True)
                tma2mma_bar.init(1)
                mma2tma_bar.init(1)
                mma2consumer_bar.init(1)
                consumer2mma_bar.init(128)  # full consumer WG
                # MMA→TMA direction: MMA arrives after finishing a work item,
                # TMA waits before starting the next. Uses BarTMA2MMA (is_p2c=True)
                # so that init_phase=0 and the first wait blocks until an arrive.
                workitem_sync_bar = BarTMA2MMA(buf.data, 4 + 2 * PIPE_DEPTH + 2, 1, True)
                workitem_sync_bar.init(1)

                ptr: Tx.let[Tx.Var(name="ptr", dtype=PointerType(PrimType("uint64")))] = Tx.reinterpret("handle", Tx.ptx.map_shared_rank(tma2mma_bar.mbar.ptr_to([0]), 0))
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
                                            mma2tma_bar.wait(tic, phase[0])
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
                            # Phase 1: Q@K^T — transB=False
                            Tx.ptx.tcgen05.encode_instr_descriptor(
                                Tx.address_of(descI), "float32", "float16", "float16",
                                MMA_M * CTA_GROUP, MMA_N, MMA_K, False, False, CTA_GROUP)

                            # Phase 2a/2b: att@V_half — transB=True
                            descI_tb: Tx.uint32
                            Tx.ptx.tcgen05.encode_instr_descriptor(
                                Tx.address_of(descI_tb), "float32", "float16", "float16",
                                MMA_M * CTA_GROUP, MMA_N, MMA_K, False, True, CTA_GROUP)

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
                                        # Q[64,128], K[64,128] -> att[64,64]
                                        # Split: Q_lo@K_lo (4 iters) + Q_hi@K_hi (4 iters) = 8 MMA_K iters
                                        tma2mma_bar.wait(tic, phase[0])
                                        Tx.ptx.tcgen05.fence.after_thread_sync()
                                        # First half: Q_lo[64,64] @ K_lo[64,64]^T
                                        for ki in Tx.unroll(HALF // MMA_K):  # 4 iterations
                                            Tx.ptx.tcgen05.encode_matrix_descriptor(
                                                Tx.address_of(descA), Q_lo.ptr_to([tic, 0, ki * MMA_K]),
                                                MMA_LDO, MMA_SDO, SWIZZLE)
                                            Tx.ptx.tcgen05.encode_matrix_descriptor(
                                                Tx.address_of(descB), K_lo.ptr_to([tic, 0, ki * MMA_K]),
                                                MMA_LDO, MMA_SDO, SWIZZLE)
                                            Tx.ptx.tcgen05.mma("float32", "float16", "float16",
                                                Tx.cuda.get_tmem_addr(tmem_addr, 0, 0),
                                                descA, descB, descI, False, CTA_GROUP, ki > 0)
                                        # Second half: Q_hi[64,64] @ K_hi[64,64]^T (accumulate)
                                        for ki in Tx.unroll(HALF // MMA_K):  # 4 iterations
                                            Tx.ptx.tcgen05.encode_matrix_descriptor(
                                                Tx.address_of(descA), Q_hi.ptr_to([tic, 0, ki * MMA_K]),
                                                MMA_LDO, MMA_SDO, SWIZZLE)
                                            Tx.ptx.tcgen05.encode_matrix_descriptor(
                                                Tx.address_of(descB), K_hi.ptr_to([tic, 0, ki * MMA_K]),
                                                MMA_LDO, MMA_SDO, SWIZZLE)
                                            Tx.ptx.tcgen05.mma("float32", "float16", "float16",
                                                Tx.cuda.get_tmem_addr(tmem_addr, 0, 0),
                                                descA, descB, descI, False, CTA_GROUP, True)
                                        mma2consumer_bar.arrive(0)

                                        # Wait consumer done with Phase 1
                                        consumer2mma_bar.wait(0, phase_c2m)
                                        Tx.ptx.tcgen05.fence.after_thread_sync()

                                        # == Phase 2a: work @ V_lo -> o_intra_lo (in TMEM) ==
                                        for ki in Tx.unroll(CHUNK // MMA_K):  # 4 iterations
                                            Tx.ptx.tcgen05.encode_matrix_descriptor(
                                                Tx.address_of(descA), work_smem.ptr_to([0, ki * MMA_K]),
                                                MMA_LDO, MMA_SDO, SWIZZLE)
                                            Tx.ptx.tcgen05.encode_matrix_descriptor(
                                                Tx.address_of(descB), V_lo.ptr_to([tic, ki * MMA_K, 0]),
                                                MMA_LDO, MMA_SDO, SWIZZLE)
                                            Tx.ptx.tcgen05.mma("float32", "float16", "float16",
                                                Tx.cuda.get_tmem_addr(tmem_addr, 0, 0),
                                                descA, descB, descI_tb, False, CTA_GROUP, ki > 0)
                                        mma2consumer_bar.arrive(0)

                                        # Wait consumer done with Phase 2a
                                        consumer2mma_bar.wait(0, phase_c2m ^ 1)
                                        Tx.ptx.tcgen05.fence.after_thread_sync()

                                        # == Phase 2b: work @ V_hi -> o_intra_hi (in TMEM) ==
                                        for ki in Tx.unroll(CHUNK // MMA_K):  # 4 iterations
                                            Tx.ptx.tcgen05.encode_matrix_descriptor(
                                                Tx.address_of(descA), work_smem.ptr_to([0, ki * MMA_K]),
                                                MMA_LDO, MMA_SDO, SWIZZLE)
                                            Tx.ptx.tcgen05.encode_matrix_descriptor(
                                                Tx.address_of(descB), V_hi.ptr_to([tic, ki * MMA_K, 0]),
                                                MMA_LDO, MMA_SDO, SWIZZLE)
                                            Tx.ptx.tcgen05.mma("float32", "float16", "float16",
                                                Tx.cuda.get_tmem_addr(tmem_addr, 0, 0),
                                                descA, descB, descI_tb, False, CTA_GROUP, ki > 0)
                                        mma2consumer_bar.arrive(0)

                                        # Wait consumer done with Phase 2b
                                        # (consumer finishes KV state update, reading K/V SMEM)
                                        consumer2mma_bar.wait(0, phase_c2m)
                                        Tx.ptx.tcgen05.fence.after_thread_sync()
                                        # Release SMEM slot AFTER consumer finishes reading K/V
                                        Tx.ptx.mbarrier.arrive(mma2tma_bar.mbar.ptr_to([tic]))
                                        # 3 phases = odd transitions -> flip phase_c2m each chunk
                                        phase_c2m = phase_c2m ^ 1

                                    if cid_mma % PIPE_DEPTH == PIPE_DEPTH - 1:
                                        phase[0] = phase[0] ^ 1
                                    cid_mma = cid_mma + 1
                                with Tx.thread()[Tx.ptx.elect_sync()]:
                                    workitem_sync_bar.arrive_only(0)
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

                                # ======== Phase 2a: Read o_intra_lo from TMEM ========
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

                                # ======== Phase 2b: Read o_intra_hi from TMEM ========
                                mma2consumer_bar.wait(0, phase_m2c)
                                phase_m2c = phase_m2c ^ 1
                                Tx.ptx.tcgen05.fence.after_thread_sync()

                                Tx.copy(reg_wg[:, :], tmem[:, 0:TMEM_LD_SIZE])

                                with Tx.thread():
                                    out_row = Tx.meta_var(warp_id * 16 + lane_id)
                                    if lane_id < 16:
                                        for j in Tx.serial(HALF):
                                            o_hi_reg[j] = reg[j]

                                        # ---- Thread-level: Q @ kvs -> o_inter ----
                                        # Accumulate directly into o_lo/hi_reg (like BASED's o_reg pattern)
                                        slope_val = Tx.meta_var(slopes_s[0])
                                        q_decay_val = Tx.meta_var(Tx.exp(0.0 - slope_val * Tx.cast(out_row, "float32")))

                                        # Q@kvs lo half: o_lo_reg[d] += sum_f Q[row,f] * kvs[f,d] * q_decay
                                        for f_idx in Tx.serial(HALF):
                                            qf_lo = Tx.meta_var(Tx.cast(Q_lo[tic_c, out_row, f_idx], "float32") * q_decay_val)
                                            for d in Tx.serial(HALF):
                                                o_lo_reg[d] = o_lo_reg[d] + qf_lo * kvs[f_idx, d]
                                        for f_idx in Tx.serial(HALF):
                                            qf_hi = Tx.meta_var(Tx.cast(Q_hi[tic_c, out_row, f_idx], "float32") * q_decay_val)
                                            for d in Tx.serial(HALF):
                                                o_lo_reg[d] = o_lo_reg[d] + qf_hi * kvs[HALF + f_idx, d]

                                        # Q@kvs hi half: o_hi_reg[d] += sum_f Q[row,f] * kvs[f, HALF+d] * q_decay
                                        for f_idx in Tx.serial(HALF):
                                            qf_lo = Tx.meta_var(Tx.cast(Q_lo[tic_c, out_row, f_idx], "float32") * q_decay_val)
                                            for d in Tx.serial(HALF):
                                                o_hi_reg[d] = o_hi_reg[d] + qf_lo * kvs[f_idx, HALF + d]
                                        for f_idx in Tx.serial(HALF):
                                            qf_hi = Tx.meta_var(Tx.cast(Q_hi[tic_c, out_row, f_idx], "float32") * q_decay_val)
                                            for d in Tx.serial(HALF):
                                                o_hi_reg[d] = o_hi_reg[d] + qf_hi * kvs[HALF + f_idx, HALF + d]

                                        # Write output
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

                                # ---- Thread-level: KV state update ----
                                with Tx.thread():
                                    out_row = Tx.meta_var(warp_id * 16 + lane_id)
                                    if lane_id < 16:
                                        slope_val = Tx.meta_var(slopes_s[0])
                                        block_decay = Tx.meta_var(Tx.exp(0.0 - slope_val * Tx.cast(CHUNK, "float32")))

                                        # Row 0: my_row = out_row*2 (0..126 even) -> in lo half (0..63) or hi half (64..127)
                                        # Row 1: my_row = out_row*2+1
                                        # out_row in [0..31] -> my_row in [0..63] -> K_lo
                                        # out_row in [32..63] -> my_row in [64..127] -> K_hi
                                        # Since out_row = warp_id*16+lane_id and lane_id<16,
                                        # warp 0: out_row 0..15 -> my_row 0..31 (K_lo)
                                        # warp 1: out_row 16..31 -> my_row 32..63 (K_lo)
                                        # warp 2: out_row 32..47 -> my_row 64..95 (K_hi)
                                        # warp 3: out_row 48..63 -> my_row 96..127 (K_hi)
                                        # We can split based on warp_id < 2 vs >= 2
                                        for row_off in Tx.unroll(2):
                                            my_row = Tx.meta_var(out_row * 2 + row_off)
                                            for d in Tx.serial(D):
                                                kvs[my_row, d] = kvs[my_row, d] * block_decay

                                        if out_row < 32:
                                            # my_row in [0..63] -> K_lo
                                            for row_off in Tx.unroll(2):
                                                my_row = Tx.meta_var(out_row * 2 + row_off)
                                                for pos in Tx.serial(CHUNK):
                                                    k_dec_val = Tx.meta_var(
                                                        Tx.cast(K_lo[tic_c, pos, my_row], "float32") *
                                                        Tx.exp(0.0 - slope_val * Tx.cast(CHUNK - pos, "float32"))
                                                    )
                                                    for d in Tx.serial(HALF):
                                                        kvs[my_row, d] = kvs[my_row, d] + k_dec_val * Tx.cast(V_lo[tic_c, pos, d], "float32")
                                                    for d in Tx.serial(HALF):
                                                        kvs[my_row, HALF + d] = kvs[my_row, HALF + d] + k_dec_val * Tx.cast(V_hi[tic_c, pos, d], "float32")
                                        else:
                                            # my_row in [64..127] -> K_hi, index = my_row - HALF
                                            for row_off in Tx.unroll(2):
                                                my_row = Tx.meta_var(out_row * 2 + row_off)
                                                k_idx = Tx.meta_var(my_row - HALF)
                                                for pos in Tx.serial(CHUNK):
                                                    k_dec_val = Tx.meta_var(
                                                        Tx.cast(K_hi[tic_c, pos, k_idx], "float32") *
                                                        Tx.exp(0.0 - slope_val * Tx.cast(CHUNK - pos, "float32"))
                                                    )
                                                    for d in Tx.serial(HALF):
                                                        kvs[my_row, d] = kvs[my_row, d] + k_dec_val * Tx.cast(V_lo[tic_c, pos, d], "float32")
                                                    for d in Tx.serial(HALF):
                                                        kvs[my_row, HALF + d] = kvs[my_row, HALF + d] + k_dec_val * Tx.cast(V_hi[tic_c, pos, d], "float32")

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
