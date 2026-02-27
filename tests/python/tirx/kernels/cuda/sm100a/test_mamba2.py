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

"""Mamba2 SSD forward kernel with SM100 warp specialization.

Warp-specialized rewrite using tcgen05 MMA, TMEM, and producer/consumer
warp separation. 4 sequential 64x64 matmuls per chunk with elementwise
ops in consumer warpgroup.

Architecture:
  cta_group=1 (single CTA, no clustering)
  MMA_M=64, MMA_N=64, MMA_K=16 (f16)
  WG1 (Producer+MMA): warp 3 = TMA, warps 0-1 = MMA
  WG0 (Consumer): TMEM read, elementwise, SMEM writeback
"""

import math
import pytest
import numpy as np
import torch

import tvm
import tvm.testing
from tvm.ir import PointerType, PrimType
from tvm.script import tirx as Tx
from tvm.tirx.bench.utils import ProtonContext, bench
from tvm.tir.layout import TileLayout, tid_in_wg, TLane, TCol, S

# Kernel constants
CHUNK = 64
D_MODEL = 64
SM_COUNT = 148

# Warp-specialized layout
WG_NUMBER = 2        # WG0=consumer, WG1=producer+MMA
WARP_NUMBER = 4      # per warpgroup
NUM_THREADS = (32 * WARP_NUMBER) * WG_NUMBER  # 256

# MMA dimensions
MMA_M = 64     # actual MMA M dimension (64 valid rows)
TMEM_ROWS = 128  # TMEM always has 128 TLanes; copy dispatch requires 128
MMA_N = 64
MMA_K = 16
CTA_GROUP = 1
SWIZZLE = 3  # 128B swizzle

# Pipeline
PIPE_DEPTH = 2  # double-buffer for TMA loads

# Memory sizes
F16_BYTES = 2
F32_BYTES = 4
F128_BYTES = 16
N_COLS = 64  # TMEM columns
TMEM_LD_SIZE = 64  # read full row at once

# SMEM budget (bytes):
#   barriers~1K + Q[2,64,64]*f16=16K + K[2,64,64]*f16=16K + V[2,64,64]*f16=16K
#   + work[64,64]*f16=8K + D_out[64,64]*f16=8K
#   + A[2,64]*f32=512 + kv_f32[64,64]*f32=16K + acs[64]*f32=256
#   Total ~82K
# NOTE: Swizzled buffers (Q,K,V,work,D_out) must be 1024-byte aligned for
# SwizzleLayout(3,3,3) to work correctly with TMA. Non-swizzled buffers
# (A, kv_f32, acs) are placed AFTER swizzled buffers to preserve alignment.
SMEM_SIZE = (
    1024  # barriers + alignment
    + PIPE_DEPTH * CHUNK * D_MODEL * F16_BYTES  # Q (swizzled)
    + PIPE_DEPTH * CHUNK * D_MODEL * F16_BYTES  # K (swizzled)
    + PIPE_DEPTH * CHUNK * D_MODEL * F16_BYTES  # V (swizzled)
    + CHUNK * D_MODEL * F16_BYTES  # work_smem (swizzled)
    + CHUNK * D_MODEL * F16_BYTES  # D_out (swizzled)
    + PIPE_DEPTH * CHUNK * F32_BYTES  # A
    + CHUNK * D_MODEL * F32_BYTES  # kv_f32
    + CHUNK * F32_BYTES  # acs
)
assert SMEM_SIZE <= 232448


def ceildiv(a, b):
    return (a + b - 1) // b


def prepare_data(batch, heads, seq_len):
    q = torch.randn(batch * heads, seq_len, D_MODEL, dtype=torch.float16, device="cuda")
    k = torch.randn(batch * heads, seq_len, D_MODEL, dtype=torch.float16, device="cuda")
    v = torch.randn(batch * heads, seq_len, D_MODEL, dtype=torch.float16, device="cuda")
    a = -torch.rand(batch * heads, seq_len, dtype=torch.float32, device="cuda") * 0.1
    return q, k, v, a


def naive_mamba2(q, k, v, a):
    """Reference implementation of mamba2 SSD forward."""
    BH, L, Dim = q.shape
    NC = L // CHUNK
    qf = q.reshape(BH, NC, CHUNK, Dim).float()
    kf = k.reshape(BH, NC, CHUNK, Dim).float()
    vf = v.reshape(BH, NC, CHUNK, Dim).float()
    af = a.reshape(BH, NC, CHUNK)

    o = torch.zeros_like(qf)
    kv = torch.zeros(BH, Dim, Dim, dtype=torch.float32, device=q.device)
    causal = torch.tril(torch.ones(CHUNK, CHUNK, device=q.device))

    for c in range(NC):
        qc, kc, vc, ac = qf[:, c], kf[:, c], vf[:, c], af[:, c]
        acs = torch.cumsum(ac, dim=-1)
        decay = torch.exp(acs.unsqueeze(-1) - acs.unsqueeze(-2)) * causal
        att = torch.matmul(qc, kc.transpose(-2, -1)) * decay
        o_intra = torch.matmul(att, vc)
        q_dec = qc * torch.exp(acs).unsqueeze(-1)
        o_inter = torch.matmul(q_dec, kv)
        o[:, c] = o_intra + o_inter
        last = acs[:, -1]
        kv = kv * torch.exp(last).unsqueeze(-1).unsqueeze(-1)
        k_dec = kc * torch.exp(last.unsqueeze(-1) - acs).unsqueeze(-1)
        kv = kv + torch.matmul(k_dec.transpose(-2, -1), vc)

    return o.reshape(BH, L, Dim).half()


# ---- Barrier classes (following test_hgemm_1consumer pattern) ----

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

# Swizzled f16 SMEM layouts for MMA operands
Q_layout = Tx.ComposeLayout(
    Tx.SwizzleLayout(SWIZZLE, SWIZZLE, SWIZZLE, swizzle_inner=True),
    Tx.TileLayout(Tx.S[(PIPE_DEPTH, CHUNK, D_MODEL) : (CHUNK * D_MODEL, D_MODEL, 1)]),
)
K_layout = Tx.ComposeLayout(
    Tx.SwizzleLayout(SWIZZLE, SWIZZLE, SWIZZLE, swizzle_inner=True),
    Tx.TileLayout(Tx.S[(PIPE_DEPTH, CHUNK, D_MODEL) : (CHUNK * D_MODEL, D_MODEL, 1)]),
)
V_layout = Tx.ComposeLayout(
    Tx.SwizzleLayout(SWIZZLE, SWIZZLE, SWIZZLE, swizzle_inner=True),
    Tx.TileLayout(Tx.S[(PIPE_DEPTH, CHUNK, D_MODEL) : (CHUNK * D_MODEL, D_MODEL, 1)]),
)
work_layout = Tx.ComposeLayout(
    Tx.SwizzleLayout(SWIZZLE, SWIZZLE, SWIZZLE, swizzle_inner=True),
    Tx.TileLayout(Tx.S[(CHUNK, D_MODEL) : (D_MODEL, 1)]),
)
D_out_layout = Tx.ComposeLayout(
    Tx.SwizzleLayout(SWIZZLE, SWIZZLE, SWIZZLE, swizzle_inner=True),
    Tx.TileLayout(Tx.S[(CHUNK, D_MODEL) : (D_MODEL, 1)]),
)


def get_mamba2_kernel():
    a_type = tvm.DataType("float16")

    # fmt: off
    @Tx.prim_func(tirx=True)
    def mamba2(q_ptr: Tx.handle, k_ptr: Tx.handle, v_ptr: Tx.handle,
               a_ptr: Tx.handle, o_ptr: Tx.handle):
        total_bh = Tx.int32()
        seq_len = Tx.int32()
        q_g = Tx.match_buffer(q_ptr, [total_bh, seq_len, D_MODEL], "float16", scope="global")
        k_g = Tx.match_buffer(k_ptr, [total_bh, seq_len, D_MODEL], "float16", scope="global")
        v_g = Tx.match_buffer(v_ptr, [total_bh, seq_len, D_MODEL], "float16", scope="global")
        a_g = Tx.match_buffer(a_ptr, [total_bh, seq_len], "float32", scope="global")
        o_g = Tx.match_buffer(o_ptr, [total_bh, seq_len, D_MODEL], "float16", scope="global")

        with Tx.kernel():
            bx = Tx.cta_id([SM_COUNT], parent="kernel")
            wg_id = Tx.warpgroup_id([WG_NUMBER], parent="cta")
            warp_id = Tx.warp_id([WARP_NUMBER], parent="warpgroup")
            lane_id = Tx.thread_id([32], parent="warp")

            with Tx.cta():
                # ---- Shared memory allocation ----
                buf = Tx.alloc_buffer([SMEM_SIZE], "uint8", scope="shared.dyn")
                tmem_addr = Tx.decl_scalar("uint32", buf.data, scope="shared.dyn", elem_offset=0)

                # SMEM layout: swizzled buffers first (1024B aligned), then non-swizzled.
                # All swizzled buffers with SwizzleLayout(3,3,3) require 1024-byte
                # alignment for correct TMA swizzle interpretation.
                base_off = 1024 // F16_BYTES  # skip barrier region (1024 bytes)
                Q_smem = Tx.decl_buffer((PIPE_DEPTH, CHUNK, D_MODEL), "float16", buf.data,
                                        layout=Q_layout, elem_offset=base_off)
                K_smem = Tx.decl_buffer((PIPE_DEPTH, CHUNK, D_MODEL), "float16", buf.data,
                                        layout=K_layout,
                                        elem_offset=base_off + PIPE_DEPTH * CHUNK * D_MODEL)
                V_smem = Tx.decl_buffer((PIPE_DEPTH, CHUNK, D_MODEL), "float16", buf.data,
                                        layout=V_layout,
                                        elem_offset=base_off + 2 * PIPE_DEPTH * CHUNK * D_MODEL)
                # Work buffer: f16 swizzled (MMA A operand, reused across phases)
                swizzled_end = 1024 + 3 * PIPE_DEPTH * CHUNK * D_MODEL * F16_BYTES  # 50176
                work_off = swizzled_end // F16_BYTES  # 25088
                work_smem = Tx.decl_buffer((CHUNK, D_MODEL), "float16", buf.data,
                                           layout=work_layout, elem_offset=work_off)
                # Output staging buffer: f16 swizzled
                d_out_off = work_off + CHUNK * D_MODEL  # 29184
                D_out_smem = Tx.decl_buffer((CHUNK, D_MODEL), "float16", buf.data,
                                            layout=D_out_layout, elem_offset=d_out_off)
                # --- Non-swizzled buffers (after all swizzled buffers) ---
                # A is f32, not swizzled (vector, not MMA operand)
                nonswizzled_byte = (d_out_off + CHUNK * D_MODEL) * F16_BYTES  # 66560
                A_smem = Tx.decl_buffer((PIPE_DEPTH, CHUNK), "float32", buf.data,
                                        elem_offset=nonswizzled_byte // F32_BYTES)
                # KV state: f32, persists across chunks
                kv_f32_off = nonswizzled_byte // F32_BYTES + PIPE_DEPTH * CHUNK
                kv_f32 = Tx.decl_buffer((D_MODEL, D_MODEL), "float32", buf.data,
                                        elem_offset=kv_f32_off)
                # Cumsum buffer
                acs_off = kv_f32_off + D_MODEL * D_MODEL
                acs_smem = Tx.decl_buffer((CHUNK,), "float32", buf.data,
                                          elem_offset=acs_off)

                # ---- Local variables ----
                descI: Tx.uint32
                descA: Tx.uint64
                descB: Tx.uint64
                phase = Tx.alloc_buffer((1,), "int32", scope="local")

                # ---- Barrier setup ----
                # 4 barrier types: tma2mma, mma2tma, mma2consumer, consumer2mma
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
                                        # Skip wait for first PIPE_DEPTH loads (pipeline priming)
                                        if cid_tma >= PIPE_DEPTH:
                                            mma2tma_bar.wait(tic, phase[0])
                                        # Load Q, K, V, A for this chunk
                                        Tx.copy_async(Q_smem[tic, :, :], q_g[wid_tma, q_row : q_row + CHUNK, 0 : D_MODEL], **tma_copy)
                                        Tx.copy_async(K_smem[tic, :, :], k_g[wid_tma, q_row : q_row + CHUNK, 0 : D_MODEL], **tma_copy)
                                        Tx.copy_async(V_smem[tic, :, :], v_g[wid_tma, q_row : q_row + CHUNK, 0 : D_MODEL], **tma_copy)
                                        # A is f32 scalar vector — use TMA for it too
                                        Tx.copy_async(A_smem[tic, :], a_g[wid_tma, q_row : q_row + CHUNK], **tma_copy)
                                        # Signal that load is done
                                        tma2mma_bar.arrive(tic, (3 * CHUNK * D_MODEL * F16_BYTES + CHUNK * F32_BYTES))
                                    if cid_tma % PIPE_DEPTH == PIPE_DEPTH - 1:
                                        phase[0] = phase[0] ^ 1
                                    cid_tma = cid_tma + 1
                                wid_tma = wid_tma + SM_COUNT

                        elif warp_id == 0:
                            # ---- MMA warp ----
                            # Manual descriptor encoding (gemm_async computes wrong ldo for 64x64)
                            # Instruction descriptor: transB=False for Phase 1 (B=[N,K] format)
                            Tx.ptx.tcgen05.encode_instr_descriptor(
                                Tx.address_of(descI), "float32", "float16", "float16",
                                MMA_M * CTA_GROUP, MMA_N, MMA_K, False, False, CTA_GROUP)
                            # Instruction descriptor: transB=True for Phases 2-4 (B=[K,N] format)
                            descI_tb: Tx.uint32
                            Tx.ptx.tcgen05.encode_instr_descriptor(
                                Tx.address_of(descI_tb), "float32", "float16", "float16",
                                MMA_M * CTA_GROUP, MMA_N, MMA_K, False, True, CTA_GROUP)

                            # SMEM descriptor params for 64x64 f16 tiles with 128B swizzle
                            MMA_LDO = 1  # leading byte offset in 16B units
                            MMA_SDO = 8 * D_MODEL * F16_BYTES // F128_BYTES  # stride byte offset (=64)

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
                                        # Q[M,K=D_MODEL], K[N=CHUNK,K=D_MODEL] -> att[M,N]
                                        # Both Q,K stored [rows, D_MODEL] with D_MODEL contiguous
                                        # K is [N,K] format -> transB=False
                                        tma2mma_bar.wait(tic, phase[0])
                                        Tx.ptx.tcgen05.fence.after_thread_sync()
                                        for ki in Tx.unroll(D_MODEL // MMA_K):
                                            Tx.ptx.tcgen05.encode_matrix_descriptor(
                                                Tx.address_of(descA), Q_smem.ptr_to([tic, 0, ki * MMA_K]),
                                                MMA_LDO, MMA_SDO, SWIZZLE)
                                            Tx.ptx.tcgen05.encode_matrix_descriptor(
                                                Tx.address_of(descB), K_smem.ptr_to([tic, 0, ki * MMA_K]),
                                                MMA_LDO, MMA_SDO, SWIZZLE)
                                            Tx.ptx.tcgen05.mma("float32", "float16", "float16",
                                                Tx.cuda.get_tmem_addr(tmem_addr, 0, 0),
                                                descA, descB, descI, False, CTA_GROUP, ki > 0)
                                        mma2consumer_bar.arrive(0)

                                        # Wait consumer done with Phase 1
                                        consumer2mma_bar.wait(0, phase_c2m)
                                        Tx.ptx.tcgen05.fence.after_thread_sync()

                                        # == Phase 2: att @ V -> o_intra (in TMEM) ==
                                        # work[M,K=CHUNK], V[K=CHUNK,N=D_MODEL] -> o[M,N]
                                        # V is [K,N] format (K contiguous if reading columns) -> transB=True
                                        for ki in Tx.unroll(CHUNK // MMA_K):
                                            Tx.ptx.tcgen05.encode_matrix_descriptor(
                                                Tx.address_of(descA), work_smem.ptr_to([0, ki * MMA_K]),
                                                MMA_LDO, MMA_SDO, SWIZZLE)
                                            Tx.ptx.tcgen05.encode_matrix_descriptor(
                                                Tx.address_of(descB), V_smem.ptr_to([tic, ki * MMA_K, 0]),
                                                MMA_LDO, MMA_SDO, SWIZZLE)
                                            Tx.ptx.tcgen05.mma("float32", "float16", "float16",
                                                Tx.cuda.get_tmem_addr(tmem_addr, 0, 0),
                                                descA, descB, descI_tb, False, CTA_GROUP, ki > 0)
                                        mma2consumer_bar.arrive(0)

                                        # Wait consumer done with Phase 2
                                        consumer2mma_bar.wait(0, phase_c2m ^ 1)
                                        Tx.ptx.tcgen05.fence.after_thread_sync()

                                        # == Phase 3: q_dec @ kv -> o_inter (in TMEM) ==
                                        # work[M,K=D_MODEL], Q_smem[tic] reused for kv[K=D_MODEL,N=D_MODEL]
                                        # kv stored as [K,N] format -> transB=True
                                        for ki in Tx.unroll(D_MODEL // MMA_K):
                                            Tx.ptx.tcgen05.encode_matrix_descriptor(
                                                Tx.address_of(descA), work_smem.ptr_to([0, ki * MMA_K]),
                                                MMA_LDO, MMA_SDO, SWIZZLE)
                                            Tx.ptx.tcgen05.encode_matrix_descriptor(
                                                Tx.address_of(descB), Q_smem.ptr_to([tic, ki * MMA_K, 0]),
                                                MMA_LDO, MMA_SDO, SWIZZLE)
                                            Tx.ptx.tcgen05.mma("float32", "float16", "float16",
                                                Tx.cuda.get_tmem_addr(tmem_addr, 0, 0),
                                                descA, descB, descI_tb, False, CTA_GROUP, ki > 0)
                                        mma2consumer_bar.arrive(0)

                                        # Wait consumer done with Phase 3
                                        consumer2mma_bar.wait(0, phase_c2m)
                                        Tx.ptx.tcgen05.fence.after_thread_sync()

                                        # == Phase 4: k_dec_t @ V -> kv_update (in TMEM) ==
                                        # work[M=D_MODEL,K=CHUNK] (k_dec transposed), V[K=CHUNK,N=D_MODEL]
                                        # V is [K,N] format -> transB=True
                                        for ki in Tx.unroll(CHUNK // MMA_K):
                                            Tx.ptx.tcgen05.encode_matrix_descriptor(
                                                Tx.address_of(descA), work_smem.ptr_to([0, ki * MMA_K]),
                                                MMA_LDO, MMA_SDO, SWIZZLE)
                                            Tx.ptx.tcgen05.encode_matrix_descriptor(
                                                Tx.address_of(descB), V_smem.ptr_to([tic, ki * MMA_K, 0]),
                                                MMA_LDO, MMA_SDO, SWIZZLE)
                                            Tx.ptx.tcgen05.mma("float32", "float16", "float16",
                                                Tx.cuda.get_tmem_addr(tmem_addr, 0, 0),
                                                descA, descB, descI_tb, False, CTA_GROUP, ki > 0)
                                        mma2tma_bar.arrive(tic)  # release SMEM slot
                                        mma2consumer_bar.arrive(0)

                                        # Wait consumer done with Phase 4
                                        consumer2mma_bar.wait(0, phase_c2m ^ 1)
                                        Tx.ptx.tcgen05.fence.after_thread_sync()
                                        # NOTE: Do NOT flip phase_c2m here. 4 phases use
                                        # {c2m, c2m^1, c2m, c2m^1} which is 4 barrier
                                        # transitions, returning to the original phase.

                                    if cid_mma % PIPE_DEPTH == PIPE_DEPTH - 1:
                                        phase[0] = phase[0] ^ 1
                                    cid_mma = cid_mma + 1
                                with Tx.thread()[Tx.ptx.elect_sync()]:
                                    workitem_sync_bar.arrive_only(0)
                                wid_mma = wid_mma + SM_COUNT

                    # === Consumer warpgroup (WG0) ===
                    with Tx.warpgroup()[0:1]:
                        Tx.cuda.trap_when_assert_failed(tmem_addr == 0)

                        # Per-thread local storage for consumer
                        o_intra_reg = Tx.alloc_buffer((D_MODEL,), "float32", scope="local")
                        kv_row = Tx.alloc_buffer((D_MODEL,), "float32", scope="local")
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

                            # Zero KV state in SMEM (all 128 threads help)
                            # 64*64=4096 f32 values / 128 threads = 32 per thread
                            with Tx.thread():
                                tid_flat = Tx.meta_var(warp_id * 32 + lane_id)
                                for ki in Tx.serial(32):
                                    idx = Tx.meta_var(tid_flat * 32 + ki)
                                    row = Tx.meta_var(idx // D_MODEL)
                                    col = Tx.meta_var(idx % D_MODEL)
                                    kv_f32[row, col] = 0.0
                            Tx.cuda.warpgroup_sync(10)

                            cid_con = 0
                            while cid_con < nc_con:
                                tic_c = Tx.meta_var(cid_con % PIPE_DEPTH)
                                tid_in_wg_val = Tx.meta_var(warp_id * 32 + lane_id)

                                # ======== Phase 1: Read att from TMEM ========
                                mma2consumer_bar.wait(0, phase_m2c)
                                phase_m2c = phase_m2c ^ 1
                                Tx.ptx.tcgen05.fence.after_thread_sync()

                                # Read TMEM -> reg (thread tid_in_wg gets TMEM lane tid_in_wg)
                                # For M=64 cta_group=1: valid data in first 16 lanes per warp
                                # TMEM lane mapping: warp w, lane l -> output row w*16+l (l<16)
                                Tx.copy(reg_wg[:, :], tmem[:, 0:TMEM_LD_SIZE])

                                with Tx.thread():
                                    out_row = Tx.meta_var(warp_id * 16 + lane_id)
                                    if lane_id < 16:
                                        # Load A into acs_smem using remapped row
                                        acs_smem[out_row] = A_smem[tic_c, out_row]
                                Tx.cuda.warpgroup_sync(10)

                                # Serial cumsum by thread 0
                                with Tx.thread():
                                    if tid_in_wg_val == 0:
                                        for ci in Tx.serial(CHUNK - 1):
                                            acs_smem[ci + 1] = acs_smem[ci + 1] + acs_smem[ci]
                                Tx.cuda.warpgroup_sync(10)

                                with Tx.thread():
                                    out_row = Tx.meta_var(warp_id * 16 + lane_id)
                                    if lane_id < 16:
                                        # Apply causal mask + exp decay to att row
                                        my_acs = Tx.meta_var(acs_smem[out_row])
                                        for j in Tx.serial(D_MODEL):
                                            if out_row >= j:
                                                reg[j] = reg[j] * Tx.exp(my_acs - acs_smem[j])
                                            else:
                                                reg[j] = 0.0
                                        # Write att row to work_smem as f16
                                        for j in Tx.serial(D_MODEL):
                                            work_smem[out_row, j] = Tx.cast(reg[j], "float16")

                                # Signal consumer done with Phase 1
                                Tx.ptx.tcgen05.fence.before_thread_sync()
                                consumer2mma_bar.arrive(0)
                                phase_c2m_c = phase_c2m_c ^ 1

                                # ======== Phase 2: Read o_intra from TMEM ========
                                mma2consumer_bar.wait(0, phase_m2c)
                                phase_m2c = phase_m2c ^ 1
                                Tx.ptx.tcgen05.fence.after_thread_sync()

                                Tx.copy(reg_wg[:, :], tmem[:, 0:TMEM_LD_SIZE])

                                with Tx.thread():
                                    out_row = Tx.meta_var(warp_id * 16 + lane_id)
                                    if lane_id < 16:
                                        # Save o_intra in registers
                                        for j in Tx.serial(D_MODEL):
                                            o_intra_reg[j] = reg[j]

                                        # Decay Q: q_dec[i,:] = Q[i,:] * exp(acs[i])
                                        my_acs = Tx.meta_var(acs_smem[out_row])
                                        decay_val = Tx.meta_var(Tx.exp(my_acs))
                                        for j in Tx.serial(D_MODEL):
                                            work_smem[out_row, j] = Tx.cast(
                                                Tx.cast(Q_smem[tic_c, out_row, j], "float32") * decay_val, "float16")

                                        # Also prepare kv_f16 into Q_smem[tic] (reuse buffer)
                                        # kv_f32 is (D,D), thread tid handles row tid
                                        for j in Tx.serial(D_MODEL):
                                            Q_smem[tic_c, out_row, j] = Tx.cast(kv_f32[out_row, j], "float16")

                                Tx.ptx.tcgen05.fence.before_thread_sync()
                                consumer2mma_bar.arrive(0)
                                phase_c2m_c = phase_c2m_c ^ 1

                                # ======== Phase 3: Read o_inter from TMEM ========
                                mma2consumer_bar.wait(0, phase_m2c)
                                phase_m2c = phase_m2c ^ 1
                                Tx.ptx.tcgen05.fence.after_thread_sync()

                                Tx.copy(reg_wg[:, :], tmem[:, 0:TMEM_LD_SIZE])

                                with Tx.thread():
                                    out_row = Tx.meta_var(warp_id * 16 + lane_id)
                                    if lane_id < 16:
                                        # o_total = o_intra + o_inter, store to D_out_smem
                                        for j in Tx.serial(D_MODEL):
                                            D_out_smem[out_row, j] = Tx.cast(
                                                o_intra_reg[j] + reg[j], "float16")

                                # TMA store output
                                Tx.ptx.fence.proxy_async("shared::cta")
                                Tx.cuda.warpgroup_sync(10)
                                with Tx.thread()[lane_id == 0 and warp_id == 0]:
                                    q_row_out = Tx.meta_var(cid_con * CHUNK)
                                    Tx.copy_async(o_g[wid_con, q_row_out : q_row_out + CHUNK, 0 : D_MODEL],
                                                  D_out_smem[:, :], dispatch="tma", cache_hint="evict_last")
                                    Tx.ptx.cp_async.bulk.commit_group()
                                    Tx.ptx.cp_async.bulk.wait_group(0)
                                Tx.cuda.warpgroup_sync(10)

                                with Tx.thread():
                                    out_row = Tx.meta_var(warp_id * 16 + lane_id)
                                    if lane_id < 16:
                                        # Decay K: k_dec[i,:] = K[i,:] * exp(acs[-1] - acs[i])
                                        last_acs = Tx.meta_var(acs_smem[CHUNK - 1])
                                        k_decay = Tx.meta_var(Tx.exp(last_acs - acs_smem[out_row]))
                                        for j in Tx.serial(D_MODEL):
                                            kv_row[j] = Tx.cast(K_smem[tic_c, out_row, j], "float32") * k_decay

                                        # Transpose K_dec: write row tid to column tid of work_smem
                                        # k_dec_t[j, out_row] = kv_row[j]
                                        for j in Tx.serial(D_MODEL):
                                            work_smem[j, out_row] = Tx.cast(kv_row[j], "float16")

                                        # Decay kv_state: kv_f32 *= exp(acs[-1])
                                        total_decay = Tx.meta_var(Tx.exp(last_acs))
                                        for j in Tx.serial(D_MODEL):
                                            kv_f32[out_row, j] = kv_f32[out_row, j] * total_decay

                                Tx.ptx.tcgen05.fence.before_thread_sync()
                                consumer2mma_bar.arrive(0)
                                phase_c2m_c = phase_c2m_c ^ 1

                                # ======== Phase 4: Read kv_update from TMEM ========
                                mma2consumer_bar.wait(0, phase_m2c)
                                phase_m2c = phase_m2c ^ 1
                                Tx.ptx.tcgen05.fence.after_thread_sync()

                                Tx.copy(reg_wg[:, :], tmem[:, 0:TMEM_LD_SIZE])

                                with Tx.thread():
                                    out_row = Tx.meta_var(warp_id * 16 + lane_id)
                                    if lane_id < 16:
                                        # kv_f32 += kv_update
                                        for j in Tx.serial(D_MODEL):
                                            kv_f32[out_row, j] = kv_f32[out_row, j] + reg[j]

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
    return mamba2


@pytest.mark.parametrize("batch,heads,seq_len", [(1, 16, 256), (2, 8, 512), (10, 16, 128)])
def test_mamba2(batch, heads, seq_len):
    q, k, v, a = prepare_data(batch, heads, seq_len)
    o_ref = naive_mamba2(q, k, v, a)

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": get_mamba2_kernel()})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
        src = mod.mod.imports[0].inspect_source()
        print(src[:2000])

    DEV = tvm.cuda(0)
    BH = batch * heads
    L = seq_len
    o_np = np.zeros((BH, L, D_MODEL), dtype=np.float16)
    q_tvm = tvm.runtime.tensor(q.cpu().numpy(), DEV)
    k_tvm = tvm.runtime.tensor(k.cpu().numpy(), DEV)
    v_tvm = tvm.runtime.tensor(v.cpu().numpy(), DEV)
    a_tvm = tvm.runtime.tensor(a.cpu().numpy(), DEV)
    o_tvm = tvm.runtime.tensor(o_np, DEV)
    mod(q_tvm, k_tvm, v_tvm, a_tvm, o_tvm)

    o_tir = o_tvm.numpy()
    o_ref_np = o_ref.cpu().numpy()
    np.testing.assert_allclose(o_tir, o_ref_np, rtol=5e-2, atol=5e-2)
    print(f"PASSED: batch={batch}, heads={heads}, seq_len={seq_len}")


def bench_mamba2():
    """Benchmark Mamba2 SSD kernel at TK-equivalent dimensions."""
    batch, heads = 16, 16
    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": get_mamba2_kernel()})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")

    DEV = tvm.cuda(0)
    BH = batch * heads

    # FLOPS from TK Mamba2 benchmark (SSD-specific):
    # chunk=64, state=64, ngroups=1
    # mask/decay + center blocks + low-rank factors + inter-chunk + output
    def flops(ms, seq_len):
        chunk = CHUNK
        state = D_MODEL
        ngroups = 1
        num_chunks = seq_len // chunk
        f = 0
        # Mask: cumsum + segsum + exp
        f += seq_len * heads + seq_len * heads * chunk
        # Center blocks: QK^T and QKV
        f += 2 * num_chunks * ngroups * chunk * chunk * state
        f += num_chunks * heads * chunk * chunk
        f += 2 * num_chunks * heads * chunk * chunk * D_MODEL
        # Low-rank: decay states + state computation
        f += num_chunks * heads * chunk
        f += 2 * num_chunks * heads * chunk * D_MODEL * state
        # Inter-chunk: state update
        f += 2 * num_chunks * heads * chunk * D_MODEL * state
        # Output
        f += num_chunks * heads * chunk
        f += 2 * num_chunks * heads * chunk * D_MODEL * state
        f += num_chunks * heads * state
        f += num_chunks * heads * chunk * D_MODEL
        return batch * f / (ms * 1e-3)

    print(f"\n{'='*60}")
    print(f"Mamba2 SSD Benchmark (B={batch}, H={heads})")
    print(f"{'='*60}")

    with ProtonContext("mamba2"):
        for seq_len in [1024, 2048, 4096, 8192]:
            q, k, v, a = prepare_data(batch, heads, seq_len)
            L = seq_len
            o_np = np.zeros((BH, L, D_MODEL), dtype=np.float16)
            q_tvm = tvm.runtime.tensor(q.cpu().numpy(), DEV)
            k_tvm = tvm.runtime.tensor(k.cpu().numpy(), DEV)
            v_tvm = tvm.runtime.tensor(v.cpu().numpy(), DEV)
            a_tvm = tvm.runtime.tensor(a.cpu().numpy(), DEV)
            o_tvm = tvm.runtime.tensor(o_np, DEV)

            func = lambda: mod(q_tvm, k_tvm, v_tvm, a_tvm, o_tvm)
            ms = bench(func, warmup=100, repeat=300, proton_name=f"mamba2_N{seq_len}")
            tflops = flops(ms, seq_len) / 1e12
            print(f"  N={seq_len:>5d}: {tflops:.2f} TFLOPS, {ms:.3f} ms")

    print()


if __name__ == "__main__":
    test_mamba2(1, 16, 256)
    bench_mamba2()
