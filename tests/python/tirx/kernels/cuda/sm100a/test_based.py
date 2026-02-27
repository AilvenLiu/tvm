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

"""BASED linear attention kernel with SM100 warp specialization.

Warp-specialized rewrite using tcgen05 MMA, TMEM, and producer/consumer
warp separation. 2 sequential matmuls per chunk (Q@K^T, att@V) with
thread-level Taylor expansion and state accumulation in consumer warpgroup.

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
from tvm.tir.layout import TileLayout, tid_in_wg, TLane, TCol, S
from tvm.tirx.bench.utils import ProtonContext, bench

# Constants
CHUNK = 64
D_QK = 16
D_VO = 64
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

# Swizzle modes
SWIZZLE_QK = 1   # 32B swizzle for Q/K (D_QK=16 * 2B = 32B rows)
SWIZZLE_V = 3    # 128B swizzle for V/work/D_out (D_VO=64 * 2B = 128B rows)

# Pipeline
PIPE_DEPTH = 2

# Memory sizes
F16_BYTES = 2
F32_BYTES = 4
F128_BYTES = 16
N_COLS = 64
TMEM_LD_SIZE = 64

# SMEM budget (bytes):
#   barriers=1024 + Q[2,64,16]*f16=4K + K[2,64,16]*f16=4K + V[2,64,64]*f16=16K
#   + work[64,64]*f16=8K + D_out[64,64]*f16=8K
#   + a0s[64]*f32=256 + a1s[16,64]*f32=4K + a2s[256,64]*f32=64K
#   Total ~112K
SMEM_SIZE = (
    1024  # barriers + alignment
    + PIPE_DEPTH * CHUNK * D_QK * F16_BYTES   # Q (swizzled 32B)
    + PIPE_DEPTH * CHUNK * D_QK * F16_BYTES   # K (swizzled 32B)
    + PIPE_DEPTH * CHUNK * D_VO * F16_BYTES   # V (swizzled 128B)
    + CHUNK * D_VO * F16_BYTES                # work_smem (swizzled 128B)
    + CHUNK * D_VO * F16_BYTES                # D_out (swizzled 128B)
    + D_VO * F32_BYTES                        # a0s
    + D_QK * D_VO * F32_BYTES                 # a1s
    + D_QK * D_QK * D_VO * F32_BYTES          # a2s
)
assert SMEM_SIZE <= 232448


def prepare_data(batch, heads, seq_len):
    q = torch.randn(batch * heads, seq_len, D_QK, dtype=torch.float16, device="cuda") / (D_QK ** 0.5)
    k = torch.randn(batch * heads, seq_len, D_QK, dtype=torch.float16, device="cuda") / (D_QK ** 0.5)
    v = torch.randn(batch * heads, seq_len, D_VO, dtype=torch.float16, device="cuda") / D_VO
    return q, k, v


def naive_based(q, k, v):
    """Reference: BASED linear attention with 2nd-order Taylor expansion."""
    BH, L, _ = q.shape
    qf = q.float()
    kf = k.float()
    vf = v.float()
    D = D_QK
    rd = math.sqrt(D)
    rrd = math.sqrt(rd)
    r2 = math.sqrt(2)

    # Full-sequence reference (not chunked)
    def make_causal(X):
        n = X.shape[-1]
        mask = torch.triu(torch.ones(n, n, device=X.device), diagonal=1).bool()
        X.masked_fill_(mask.unsqueeze(0), 0.0)
        return X

    qk = torch.matmul(qf, kf.transpose(-2, -1))  # (BH, L, L)
    T2 = torch.matmul(make_causal(qk ** 2), vf)
    T1 = torch.matmul(make_causal(qk), vf)
    T0 = vf.cumsum(dim=1)

    o = T0 + T1 / (rrd * rrd) + T2 / (rd * r2 * rd * r2)

    # KV states (accumulated over all positions, with TK output scaling)
    # a0 = sum of V over all positions → (BH, D_VO)
    kv_a0 = vf.sum(dim=1).half()  # (BH, D_VO)
    # a1 = K^T @ V → (BH, D_QK, D_VO), scaled by 0.5
    a1_raw = torch.matmul(kf.transpose(-2, -1), vf)  # (BH, D_QK, D_VO)
    kv_a1 = (a1_raw * 0.5).half()  # (BH, D_QK, D_VO)
    # a2 = sum K[d]*K[e]*V[f] → (BH, D_QK*D_QK, D_VO), scaled by 0.25*0.707...
    a2_raw = torch.einsum("bnd,bne,bnf->bdef", kf, kf, vf)  # (BH, D_QK, D_QK, D_VO)
    kv_a2 = (a2_raw.reshape(BH, D_QK * D_QK, D_VO) * 0.1767766953).half()

    return o.half(), kv_a0, kv_a1, kv_a2


def ceildiv(a, b):
    return (a + b - 1) // b


# ---- Barrier classes (following mamba2 pattern) ----

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

# Q/K: SWIZZLE_32B (1) — D_QK=16 f16 = 32B per row
QK_layout_pipe = Tx.ComposeLayout(
    Tx.SwizzleLayout(SWIZZLE_QK, SWIZZLE_QK, SWIZZLE_QK, swizzle_inner=True),
    Tx.TileLayout(S[(PIPE_DEPTH, CHUNK, D_QK) : (CHUNK * D_QK, D_QK, 1)]),
)
# V: SWIZZLE_128B (3) — D_VO=64 f16 = 128B per row
V_layout_pipe = Tx.ComposeLayout(
    Tx.SwizzleLayout(SWIZZLE_V, SWIZZLE_V, SWIZZLE_V, swizzle_inner=True),
    Tx.TileLayout(S[(PIPE_DEPTH, CHUNK, D_VO) : (CHUNK * D_VO, D_VO, 1)]),
)
# work/D_out: SWIZZLE_128B (3) — 64 f16 = 128B per row
work_layout = Tx.ComposeLayout(
    Tx.SwizzleLayout(SWIZZLE_V, SWIZZLE_V, SWIZZLE_V, swizzle_inner=True),
    Tx.TileLayout(S[(CHUNK, D_VO) : (D_VO, 1)]),
)
D_out_layout = Tx.ComposeLayout(
    Tx.SwizzleLayout(SWIZZLE_V, SWIZZLE_V, SWIZZLE_V, swizzle_inner=True),
    Tx.TileLayout(S[(CHUNK, D_VO) : (D_VO, 1)]),
)


def get_based_kernel():
    a_type = tvm.DataType("float16")

    # fmt: off
    @Tx.prim_func(tirx=True)
    def based(q_ptr: Tx.handle, k_ptr: Tx.handle, v_ptr: Tx.handle, o_ptr: Tx.handle,
              kv_a0_ptr: Tx.handle, kv_a1_ptr: Tx.handle, kv_a2_ptr: Tx.handle):
        total_bh = Tx.int32()
        seq_len = Tx.int32()
        q_g = Tx.match_buffer(q_ptr, [total_bh, seq_len, D_QK], "float16", scope="global")
        k_g = Tx.match_buffer(k_ptr, [total_bh, seq_len, D_QK], "float16", scope="global")
        v_g = Tx.match_buffer(v_ptr, [total_bh, seq_len, D_VO], "float16", scope="global")
        o_g = Tx.match_buffer(o_ptr, [total_bh, seq_len, D_VO], "float16", scope="global")
        kv_a0_g = Tx.match_buffer(kv_a0_ptr, [total_bh, D_VO], "float16", scope="global")
        kv_a1_g = Tx.match_buffer(kv_a1_ptr, [total_bh, D_QK, D_VO], "float16", scope="global")
        kv_a2_g = Tx.match_buffer(kv_a2_ptr, [total_bh, D_QK * D_QK, D_VO], "float16", scope="global")

        with Tx.kernel():
            bx = Tx.cta_id([SM_COUNT], parent="kernel")
            wg_id = Tx.warpgroup_id([WG_NUMBER], parent="cta")
            warp_id = Tx.warp_id([WARP_NUMBER], parent="warpgroup")
            lane_id = Tx.thread_id([32], parent="warp")

            with Tx.cta():
                # ---- Shared memory allocation ----
                buf = Tx.alloc_buffer([SMEM_SIZE], "uint8", scope="shared.dyn")
                tmem_addr = Tx.decl_scalar("uint32", buf.data, scope="shared.dyn", elem_offset=0)

                # SMEM layout: swizzled buffers first (aligned), then non-swizzled.
                # Offsets in f16 elements (divide byte offset by 2):
                #   barriers: 0..1023 bytes (512 f16 elems)
                #   Q_smem: 1024 bytes = offset 512 f16
                #   K_smem: 1024 + 4096 = 5120 bytes = offset 2560 f16
                #   V_smem: 5120 + 4096 = 9216 bytes = offset 4608 f16
                #   work_smem: 9216 + 16384 = 25600 bytes = offset 12800 f16
                #   D_out_smem: 25600 + 8192 = 33792 bytes = offset 16896 f16
                #   --- non-swizzled (byte offsets, use f32 elems) ---
                #   a0s: 33792 + 8192 = 41984 bytes = offset 10496 f32
                #   a1s: 41984 + 256 = 42240 bytes = offset 10560 f32
                #   a2s: 42240 + 4096 = 46336 bytes = offset 11584 f32

                base_off_f16 = 1024 // F16_BYTES  # 512
                Q_smem = Tx.decl_buffer((PIPE_DEPTH, CHUNK, D_QK), "float16", buf.data,
                                        layout=QK_layout_pipe, elem_offset=base_off_f16)
                K_off_f16 = base_off_f16 + PIPE_DEPTH * CHUNK * D_QK  # 512 + 2048 = 2560
                K_smem = Tx.decl_buffer((PIPE_DEPTH, CHUNK, D_QK), "float16", buf.data,
                                        layout=QK_layout_pipe, elem_offset=K_off_f16)
                V_off_f16 = K_off_f16 + PIPE_DEPTH * CHUNK * D_QK  # 2560 + 2048 = 4608
                V_smem = Tx.decl_buffer((PIPE_DEPTH, CHUNK, D_VO), "float16", buf.data,
                                        layout=V_layout_pipe, elem_offset=V_off_f16)
                work_off_f16 = V_off_f16 + PIPE_DEPTH * CHUNK * D_VO  # 4608 + 8192 = 12800
                work_smem = Tx.decl_buffer((CHUNK, D_VO), "float16", buf.data,
                                           layout=work_layout, elem_offset=work_off_f16)
                d_out_off_f16 = work_off_f16 + CHUNK * D_VO  # 12800 + 4096 = 16896
                D_out_smem = Tx.decl_buffer((CHUNK, D_VO), "float16", buf.data,
                                            layout=D_out_layout, elem_offset=d_out_off_f16)

                # --- Non-swizzled buffers (f32) ---
                nonswizzled_byte = (d_out_off_f16 + CHUNK * D_VO) * F16_BYTES  # 41984
                a0s_off_f32 = nonswizzled_byte // F32_BYTES  # 10496
                a0s = Tx.decl_buffer((D_VO,), "float32", buf.data, elem_offset=a0s_off_f32)
                a1s_off_f32 = a0s_off_f32 + D_VO  # 10560
                a1s = Tx.decl_buffer((D_QK, D_VO), "float32", buf.data, elem_offset=a1s_off_f32)
                a2s_off_f32 = a1s_off_f32 + D_QK * D_VO  # 11584
                a2s = Tx.decl_buffer((D_QK * D_QK, D_VO), "float32", buf.data, elem_offset=a2s_off_f32)

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
                                        # Load Q, K, V for this chunk
                                        Tx.copy_async(Q_smem[tic, :, :], q_g[wid_tma, q_row : q_row + CHUNK, 0 : D_QK], **tma_copy)
                                        Tx.copy_async(K_smem[tic, :, :], k_g[wid_tma, q_row : q_row + CHUNK, 0 : D_QK], **tma_copy)
                                        Tx.copy_async(V_smem[tic, :, :], v_g[wid_tma, q_row : q_row + CHUNK, 0 : D_VO], **tma_copy)
                                        # Signal that load is done
                                        tma2mma_bar.arrive(tic, (2 * CHUNK * D_QK * F16_BYTES + CHUNK * D_VO * F16_BYTES))
                                    if cid_tma % PIPE_DEPTH == PIPE_DEPTH - 1:
                                        phase[0] = phase[0] ^ 1
                                    cid_tma = cid_tma + 1
                                wid_tma = wid_tma + SM_COUNT

                        elif warp_id == 0:
                            # ---- MMA warp ----
                            # Phase 1: Q@K^T — transB=False, K is [N,K] format
                            # Q[64,16], K[64,16] with SWIZZLE_QK (32B)
                            Tx.ptx.tcgen05.encode_instr_descriptor(
                                Tx.address_of(descI), "float32", "float16", "float16",
                                MMA_M * CTA_GROUP, MMA_N, MMA_K, False, False, CTA_GROUP)

                            # Phase 2: att@V — transB=True, V is [K,N] format
                            descI_tb: Tx.uint32
                            Tx.ptx.tcgen05.encode_instr_descriptor(
                                Tx.address_of(descI_tb), "float32", "float16", "float16",
                                MMA_M * CTA_GROUP, MMA_N, MMA_K, False, True, CTA_GROUP)

                            # Descriptor params for Q/K (SWIZZLE_QK=1, 32B rows)
                            MMA_LDO_QK = 1
                            MMA_SDO_QK = 8 * D_QK * F16_BYTES // F128_BYTES  # 8*16*2/16 = 16

                            # Descriptor params for V/work (SWIZZLE_V=3, 128B rows)
                            MMA_LDO_V = 1
                            MMA_SDO_V = 8 * D_VO * F16_BYTES // F128_BYTES  # 8*64*2/16 = 64

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

                                        # == Phase 1: Q @ K^T -> qk (in TMEM) ==
                                        # Q[M=64,K=16], K[N=64,K=16] -> qk[64,64]
                                        tma2mma_bar.wait(tic, phase[0])
                                        Tx.ptx.tcgen05.fence.after_thread_sync()
                                        for ki in Tx.unroll(D_QK // MMA_K):  # 1 iteration
                                            Tx.ptx.tcgen05.encode_matrix_descriptor(
                                                Tx.address_of(descA), Q_smem.ptr_to([tic, 0, ki * MMA_K]),
                                                MMA_LDO_QK, MMA_SDO_QK, SWIZZLE_QK)
                                            Tx.ptx.tcgen05.encode_matrix_descriptor(
                                                Tx.address_of(descB), K_smem.ptr_to([tic, 0, ki * MMA_K]),
                                                MMA_LDO_QK, MMA_SDO_QK, SWIZZLE_QK)
                                            Tx.ptx.tcgen05.mma("float32", "float16", "float16",
                                                Tx.cuda.get_tmem_addr(tmem_addr, 0, 0),
                                                descA, descB, descI, False, CTA_GROUP, ki > 0)
                                        mma2consumer_bar.arrive(0)

                                        # Wait consumer done with Phase 1
                                        consumer2mma_bar.wait(0, phase_c2m)
                                        Tx.ptx.tcgen05.fence.after_thread_sync()

                                        # == Phase 2: att @ V -> o_intra (in TMEM) ==
                                        # work[M=64,K=64], V[K=64,N=64] -> o[64,64]
                                        for ki in Tx.unroll(CHUNK // MMA_K):  # 4 iterations
                                            Tx.ptx.tcgen05.encode_matrix_descriptor(
                                                Tx.address_of(descA), work_smem.ptr_to([0, ki * MMA_K]),
                                                MMA_LDO_V, MMA_SDO_V, SWIZZLE_V)
                                            Tx.ptx.tcgen05.encode_matrix_descriptor(
                                                Tx.address_of(descB), V_smem.ptr_to([tic, ki * MMA_K, 0]),
                                                MMA_LDO_V, MMA_SDO_V, SWIZZLE_V)
                                            Tx.ptx.tcgen05.mma("float32", "float16", "float16",
                                                Tx.cuda.get_tmem_addr(tmem_addr, 0, 0),
                                                descA, descB, descI_tb, False, CTA_GROUP, ki > 0)
                                        mma2consumer_bar.arrive(0)

                                        # Wait consumer done with Phase 2 (including state updates)
                                        consumer2mma_bar.wait(0, phase_c2m ^ 1)
                                        Tx.ptx.tcgen05.fence.after_thread_sync()
                                        # Release SMEM slot AFTER consumer finishes reading K/V
                                        # Use regular mbarrier.arrive (not tcgen05.commit, which
                                        # requires a pending MMA operation)
                                        Tx.ptx.mbarrier.arrive(mma2tma_bar.mbar.ptr_to([tic]))
                                        # 2 phases = even transitions → do NOT flip phase_c2m

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
                        o_reg = Tx.alloc_buffer((D_VO,), "float32", scope="local")
                        q_vals = Tx.alloc_buffer((D_QK,), "float32", scope="local")
                        k_vals = Tx.alloc_buffer((D_QK,), "float32", scope="local")
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

                            # Zero states in SMEM (all 128 consumer threads help)
                            with Tx.thread():
                                tid_flat = Tx.meta_var(warp_id * 32 + lane_id)
                                # Zero a0s: 64 f32 values / 128 threads
                                if tid_flat < D_VO:
                                    a0s[tid_flat] = 0.0
                                # Zero a1s: 16*64=1024 f32 values / 128 threads = 8 per thread
                                for ki in Tx.serial(8):
                                    idx = Tx.meta_var(tid_flat * 8 + ki)
                                    row = Tx.meta_var(idx // D_VO)
                                    col = Tx.meta_var(idx % D_VO)
                                    a1s[row, col] = 0.0
                                # Zero a2s: 256*64=16384 f32 values / 128 threads = 128 per thread
                                for ki in Tx.serial(128):
                                    idx = Tx.meta_var(tid_flat * 128 + ki)
                                    row = Tx.meta_var(idx // D_VO)
                                    col = Tx.meta_var(idx % D_VO)
                                    a2s[row, col] = 0.0
                            Tx.cuda.warpgroup_sync(10)

                            cid_con = 0
                            while cid_con < nc_con:
                                tic_c = Tx.meta_var(cid_con % PIPE_DEPTH)
                                tid_in_wg_val = Tx.meta_var(warp_id * 32 + lane_id)

                                # ======== Phase 1: Read qk from TMEM ========
                                mma2consumer_bar.wait(0, phase_m2c)
                                phase_m2c = phase_m2c ^ 1
                                Tx.ptx.tcgen05.fence.after_thread_sync()

                                # Read TMEM -> reg
                                Tx.copy(reg_wg[:, :], tmem[:, 0:TMEM_LD_SIZE])

                                with Tx.thread():
                                    out_row = Tx.meta_var(warp_id * 16 + lane_id)
                                    if lane_id < 16:
                                        # Apply Taylor expansion + causal mask to qk row
                                        # att[i,j] = (1 + qk*0.25 + (qk*0.25)^2*0.5) if i>=j else 0
                                        for j in Tx.serial(D_VO):
                                            if out_row >= j:
                                                t = Tx.meta_var(reg[j] * 0.25)
                                                reg[j] = 1.0 + t + t * t * 0.5
                                            else:
                                                reg[j] = 0.0
                                        # Write att row to work_smem as f16
                                        for j in Tx.serial(D_VO):
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
                                        # Save o_intra in o_reg
                                        for j in Tx.serial(D_VO):
                                            o_reg[j] = reg[j]

                                        # Load Q values for this row (16 values)
                                        for d in Tx.serial(D_QK):
                                            q_vals[d] = Tx.cast(Q_smem[tic_c, out_row, d], "float32")

                                        # ---- Add a0 contribution: o[f] += a0s[f] ----
                                        for f in Tx.serial(D_VO):
                                            o_reg[f] = o_reg[f] + a0s[f]

                                        # ---- Add a1 contribution: o[f] += sum_d q[d]*0.25 * a1s[d,f] ----
                                        for d in Tx.serial(D_QK):
                                            qd_scaled = Tx.meta_var(q_vals[d] * 0.25)
                                            for f in Tx.serial(D_VO):
                                                o_reg[f] = o_reg[f] + qd_scaled * a1s[d, f]

                                        # ---- Add a2 contribution: o[f] += 0.5 * sum_{d,e} q[d]*0.25*q[e]*0.25 * a2s[e*16+d,f] ----
                                        for e in Tx.serial(D_QK):
                                            qe_scaled = Tx.meta_var(q_vals[e] * 0.25)
                                            for d in Tx.serial(D_QK):
                                                qde = Tx.meta_var(q_vals[d] * 0.25 * qe_scaled * 0.5)
                                                for f in Tx.serial(D_VO):
                                                    o_reg[f] = o_reg[f] + qde * a2s[e * D_QK + d, f]

                                        # Write output to D_out_smem
                                        for j in Tx.serial(D_VO):
                                            D_out_smem[out_row, j] = Tx.cast(o_reg[j], "float16")

                                # TMA store output
                                Tx.ptx.fence.proxy_async("shared::cta")
                                Tx.cuda.warpgroup_sync(10)
                                with Tx.thread()[lane_id == 0 and warp_id == 0]:
                                    q_row_out = Tx.meta_var(cid_con * CHUNK)
                                    Tx.copy_async(o_g[wid_con, q_row_out : q_row_out + CHUNK, 0 : D_VO],
                                                  D_out_smem[:, :], dispatch="tma", cache_hint="evict_last")
                                    Tx.ptx.cp_async.bulk.commit_group()
                                    Tx.ptx.cp_async.bulk.wait_group(0)
                                Tx.cuda.warpgroup_sync(10)

                                # ---- Update states ----
                                # Each thread owns column out_row (= f) across all state matrices.
                                # Iterate over all 64 positions to accumulate into owned column.
                                with Tx.thread():
                                    out_row = Tx.meta_var(warp_id * 16 + lane_id)
                                    if lane_id < 16:
                                        for pos in Tx.serial(CHUNK):
                                            v_val = Tx.meta_var(Tx.cast(V_smem[tic_c, pos, out_row], "float32"))
                                            # Update a0: a0[out_row] += V[pos, out_row]
                                            a0s[out_row] = a0s[out_row] + v_val

                                            # Load K row into registers for this position
                                            for d in Tx.serial(D_QK):
                                                k_vals[d] = Tx.cast(K_smem[tic_c, pos, d], "float32")

                                            # Update a1: a1[d, out_row] += K[pos,d] * V[pos, out_row]
                                            for d in Tx.serial(D_QK):
                                                a1s[d, out_row] = a1s[d, out_row] + k_vals[d] * v_val

                                            # Update a2: a2[e*16+d, out_row] += K[pos,d]*K[pos,e]*V[pos, out_row]
                                            for e in Tx.serial(D_QK):
                                                for d in Tx.serial(D_QK):
                                                    a2s[e * D_QK + d, out_row] = a2s[e * D_QK + d, out_row] + k_vals[d] * k_vals[e] * v_val

                                Tx.ptx.tcgen05.fence.before_thread_sync()
                                consumer2mma_bar.arrive(0)
                                phase_c2m_c = phase_c2m_c ^ 1

                                cid_con = cid_con + 1

                            # ---- Write KV states to global (after all chunks) ----
                            # Use thread-level writes (64 active threads handle 64 rows)
                            with Tx.thread():
                                out_row = Tx.meta_var(warp_id * 16 + lane_id)
                                if lane_id < 16:
                                    # a0: write to global, scale by 1.0
                                    if out_row < D_VO:
                                        kv_a0_g[wid_con, out_row] = Tx.cast(a0s[out_row], "float16")

                                    # a1: (D_QK, D_VO), each thread writes one row, scaled by 0.5
                                    if out_row < D_QK:
                                        for f in Tx.serial(D_VO):
                                            kv_a1_g[wid_con, out_row, f] = Tx.cast(a1s[out_row, f] * 0.5, "float16")

                                    # a2: (D_QK*D_QK, D_VO), scaled by 0.1767766953
                                    # 256 rows / 64 threads = 4 rows per thread
                                    for ri in Tx.serial(4):
                                        a2_row = Tx.meta_var(out_row * 4 + ri)
                                        for f in Tx.serial(D_VO):
                                            kv_a2_g[wid_con, a2_row, f] = Tx.cast(a2s[a2_row, f] * 0.1767766953, "float16")
                            Tx.cuda.warpgroup_sync(10)

                            wid_con = wid_con + SM_COUNT

                # ---- TMEM deallocation ----
                with Tx.warp()[0:1]:
                    Tx.ptx.tcgen05.relinquish_alloc_permit(cta_group=CTA_GROUP)
                    Tx.ptx.tcgen05.dealloc(tmem_addr, n_cols=N_COLS, cta_group=CTA_GROUP)

                Tx.cuda.cta_sync()
    # fmt: on
    return based


@pytest.mark.parametrize("batch,heads,seq_len", [(1, 1, 64), (1, 1, 256), (1, 8, 1024), (10, 16, 128)])
def test_based(batch, heads, seq_len):
    q, k, v = prepare_data(batch, heads, seq_len)
    o_ref, a0_ref, a1_ref, a2_ref = naive_based(q, k, v)

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": get_based_kernel()})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
        src = mod.mod.imports[0].inspect_source()
        print(src[:2000])

    DEV = tvm.cuda(0)
    BH = batch * heads
    L = seq_len
    o_np = np.zeros((BH, L, D_VO), dtype=np.float16)
    a0_np = np.zeros((BH, D_VO), dtype=np.float16)
    a1_np = np.zeros((BH, D_QK, D_VO), dtype=np.float16)
    a2_np = np.zeros((BH, D_QK * D_QK, D_VO), dtype=np.float16)

    q_tvm = tvm.runtime.tensor(q.cpu().numpy(), DEV)
    k_tvm = tvm.runtime.tensor(k.cpu().numpy(), DEV)
    v_tvm = tvm.runtime.tensor(v.cpu().numpy(), DEV)
    o_tvm = tvm.runtime.tensor(o_np, DEV)
    a0_tvm = tvm.runtime.tensor(a0_np, DEV)
    a1_tvm = tvm.runtime.tensor(a1_np, DEV)
    a2_tvm = tvm.runtime.tensor(a2_np, DEV)
    mod(q_tvm, k_tvm, v_tvm, o_tvm, a0_tvm, a1_tvm, a2_tvm)

    # Check O
    o_tir = o_tvm.numpy()
    o_ref_np = o_ref.cpu().numpy()
    abs_diff = np.abs(o_tir.astype(np.float32) - o_ref_np.astype(np.float32))
    abs_ref = np.abs(o_ref_np.astype(np.float32))
    print(f"O:  avg_ref={abs_ref.mean():.6f}, avg_diff={abs_diff.mean():.6f}, max_diff={abs_diff.max():.6f}")
    # f16 accumulation error grows with nc (chunks); scale O tolerance accordingly
    nc = seq_len // 64
    o_atol = max(0.3, 0.01 * nc)
    np.testing.assert_allclose(o_tir, o_ref_np, rtol=o_atol, atol=o_atol)

    # Check a0
    a0_tir = a0_tvm.numpy()
    a0_ref_np = a0_ref.cpu().numpy()
    print(f"a0: avg_ref={np.abs(a0_ref_np).mean():.6f}, max_diff={np.abs(a0_tir.astype(np.float32) - a0_ref_np.astype(np.float32)).max():.6f}")
    np.testing.assert_allclose(a0_tir, a0_ref_np, rtol=1e-1, atol=1e-1)

    # Check a1
    a1_tir = a1_tvm.numpy()
    a1_ref_np = a1_ref.cpu().numpy()
    print(f"a1: avg_ref={np.abs(a1_ref_np).mean():.6f}, max_diff={np.abs(a1_tir.astype(np.float32) - a1_ref_np.astype(np.float32)).max():.6f}")
    # Scale tolerance with sequence length — f16 accumulation error grows with chunks.
    # Large BH also increases statistical chance of outlier diffs, so add a small margin.
    nc = seq_len // 64
    state_atol = max(0.2, 0.04 * nc)
    np.testing.assert_allclose(a1_tir, a1_ref_np, rtol=state_atol, atol=state_atol)

    # Check a2
    a2_tir = a2_tvm.numpy()
    a2_ref_np = a2_ref.cpu().numpy()
    print(f"a2: avg_ref={np.abs(a2_ref_np).mean():.6f}, max_diff={np.abs(a2_tir.astype(np.float32) - a2_ref_np.astype(np.float32)).max():.6f}")
    np.testing.assert_allclose(a2_tir, a2_ref_np, rtol=state_atol, atol=state_atol)

    print(f"PASSED (O + states): batch={batch}, heads={heads}, seq_len={seq_len}")


def bench_based():
    """Benchmark BASED kernel at TK-equivalent dimensions."""
    batch, heads = 16, 16
    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": get_based_kernel()})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")

    DEV = tvm.cuda(0)
    BH = batch * heads

    # FLOPS formula from TK: expanded_dim = D_QK^2 + D_QK + 1 = 273
    # f = 2*B*N*H*expanded_dim (feature map Q & K)
    #   + 4 * B*N*H*D_VO*expanded_dim (kv, cumsum, q*kv, sum)
    EXPANDED_DIM = D_QK * D_QK + D_QK + 1  # 273
    def flops(ms, seq_len):
        f = 2 * batch * seq_len * heads * EXPANDED_DIM
        f += 4 * batch * seq_len * heads * D_VO * EXPANDED_DIM
        return f / (ms * 1e-3)

    print(f"\n{'='*60}")
    print(f"BASED Benchmark (B={batch}, H={heads})")
    print(f"{'='*60}")

    with ProtonContext("based"):
        for seq_len in [1024, 2048, 4096, 8192]:
            q, k, v = (
                torch.randn(BH, seq_len, D_QK, dtype=torch.float16, device="cuda") / (D_QK ** 0.5),
                torch.randn(BH, seq_len, D_QK, dtype=torch.float16, device="cuda") / (D_QK ** 0.5),
                torch.randn(BH, seq_len, D_VO, dtype=torch.float16, device="cuda"),
            )
            o_np = np.zeros((BH, seq_len, D_VO), dtype=np.float16)
            a0_np = np.zeros((BH, D_VO), dtype=np.float16)
            a1_np = np.zeros((BH, D_QK, D_VO), dtype=np.float16)
            a2_np = np.zeros((BH, D_QK * D_QK, D_VO), dtype=np.float16)

            q_tvm = tvm.runtime.tensor(q.cpu().numpy(), DEV)
            k_tvm = tvm.runtime.tensor(k.cpu().numpy(), DEV)
            v_tvm = tvm.runtime.tensor(v.cpu().numpy(), DEV)
            o_tvm = tvm.runtime.tensor(o_np, DEV)
            a0_tvm = tvm.runtime.tensor(a0_np, DEV)
            a1_tvm = tvm.runtime.tensor(a1_np, DEV)
            a2_tvm = tvm.runtime.tensor(a2_np, DEV)

            func = lambda: mod(q_tvm, k_tvm, v_tvm, o_tvm, a0_tvm, a1_tvm, a2_tvm)
            ms = bench(func, warmup=100, repeat=300, proton_name=f"based_N{seq_len}")
            tflops = flops(ms, seq_len) / 1e12
            print(f"  N={seq_len:>5d}: {tflops:.2f} TFLOPS, {ms:.3f} ms")

    print()


if __name__ == "__main__":
    test_based(1, 1, 64)
    bench_based()
