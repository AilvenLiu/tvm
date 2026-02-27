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

"""Monarch FFT convolution kernel with SM100 warp specialization.

Warp-specialized rewrite using tcgen05 MMA, TMEM, and producer/consumer
warp separation. 7 MMA phases for complex Monarch FFT convolution on
64x64 matrices with 3 interleaved complex elementwise multiplications.

Architecture:
  cta_group=1 (single CTA, no clustering)
  MMA_M=64, MMA_N=64, MMA_K=16 (f16)
  WG1 (Producer+MMA): warp 3 = TMA, warp 0 = MMA
  WG0 (Consumer): TMEM read, elementwise, SMEM writeback

Algorithm (Monarch FFT convolution):
  Phase 1: FR @ X        (4 K-iters)
  Phase 2: FI @ X        (4 K-iters)
  -> tw multiply, write work_A/work_B
  Phase 3: WR@FR + (-WI)@FI  (8 K-iters, cross-buffer accumulation)
  Phase 4: WR@FI + WI@FR     (8 K-iters)
  -> kf multiply, write work_A/work_B
  Phase 5: WR@FiR + (-WI)@FiI (8 K-iters)
  Phase 6: WR@FiI + WI@FiR    (8 K-iters)
  -> twinv multiply, write work_A/work_B
  Phase 7: FiR@WR + FiI@(-WI) (8 K-iters, only real part needed)
  -> TMA store
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
N1 = 64
N = N1 * N1  # 4096
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
PIPE_DEPTH = 1

# Memory sizes
F16_BYTES = 2
F32_BYTES = 4
F128_BYTES = 16
N_COLS = 64
TMEM_LD_SIZE = 64

# SMEM budget (bytes):
#   barriers=1024
#   F_real[64,64]*f16=8K + F_imag[64,64]*f16=8K (swizzled, MMA operand)
#   Finv_real[64,64]*f16=8K + Finv_imag[64,64]*f16=8K (swizzled, MMA operand)
#   X_smem[64,64]*f16=8K (swizzled, MMA operand, TMA loaded)
#   work_A[64,64]*f16=8K + work_B[64,64]*f16=8K (swizzled, MMA operand)
#   D_out[64,64]*f16=8K (swizzled, TMA store)
#   tw_real[64,64]*f16=8K + tw_imag[64,64]*f16=8K (not swizzled, elementwise)
#   twinv_real[64,64]*f16=8K + twinv_imag[64,64]*f16=8K (not swizzled)
#   kf_real[64,64]*f16=8K + kf_imag[64,64]*f16=8K (not swizzled, per-head)
#   w_real_s[64,64]*f32=16K + w_imag_s[64,64]*f32=16K (not swizzled, consumer temp)
#   Total = 1024 + 8*8K + 6*8K + 2*16K = 1024 + 64K + 48K + 32K = ~145K
SMEM_SIZE = (
    1024  # barriers + alignment
    + 8 * N1 * N1 * F16_BYTES   # 8 swizzled buffers (F_r/i, Finv_r/i, X, work_A/B, D_out)
    + 6 * N1 * N1 * F16_BYTES   # 6 non-swizzled f16 (tw_r/i, twinv_r/i, kf_r/i)
    + 2 * N1 * N1 * F32_BYTES   # 2 non-swizzled f32 (w_real_s, w_imag_s)
)
assert SMEM_SIZE <= 232448


def ceildiv(a, b):
    return (a + b - 1) // b


def prepare_data(batch, heads):
    """Prepare input, filter, and FFT constant matrices."""
    torch.manual_seed(42)
    u = torch.randn(batch, heads, N, dtype=torch.float64, device="cuda")
    k = torch.randn(heads, N, dtype=torch.float64, device="cuda")

    # DFT matrix (symmetric: F = F^T)
    n = torch.arange(N1, dtype=torch.float64, device="cuda")
    F_mat = torch.exp(-2j * math.pi * n.unsqueeze(1) * n.unsqueeze(0) / N1)
    Finv_mat = torch.exp(2j * math.pi * n.unsqueeze(1) * n.unsqueeze(0) / N1)

    # Twiddle factors
    tw = torch.exp(-2j * math.pi * n.unsqueeze(1) * n.unsqueeze(0) / N)
    twinv = torch.exp(2j * math.pi * n.unsqueeze(1) * n.unsqueeze(0) / N) / N
    twinv_t = twinv.T.contiguous()

    # Filter: FFT -> Monarch permutation (transpose within each head)
    k_f = torch.fft.fft(k, n=N)
    k_f_perm = k_f.reshape(heads, N1, N1).transpose(-1, -2).contiguous()

    # Input reshape to 2D
    u_2d = u.reshape(batch, heads, N1, N1)

    # Convert to f16 split real/imag
    def split_complex_f16(c):
        return c.real.half(), c.imag.half()

    f_real, f_imag = split_complex_f16(F_mat)
    finv_real, finv_imag = split_complex_f16(Finv_mat)
    tw_real, tw_imag = split_complex_f16(tw)
    twinv_real, twinv_imag = split_complex_f16(twinv_t)
    kf_real, kf_imag = split_complex_f16(k_f_perm)

    u_f16 = u_2d.half()

    return (u_f16, kf_real, kf_imag,
            f_real, f_imag, finv_real, finv_imag,
            tw_real, tw_imag, twinv_real, twinv_imag,
            u, k)


def ref_fftconv(u, k):
    """Standard FFT convolution reference."""
    u_f = torch.fft.fft(u.double(), n=N)
    k_f = torch.fft.fft(k.double(), n=N)
    y = torch.fft.ifft(u_f * k_f, n=N).real
    return y


# ---- Barrier classes (following linear_attention/BASED/mamba2 pattern) ----

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
# All MMA operand buffers use SWIZZLE_128B (3) with 64 f16 = 128B inner dimension.

# Swizzled [64,64] f16 layout (MMA operands, non-pipelined)
swizzled_layout = Tx.ComposeLayout(
    Tx.SwizzleLayout(SWIZZLE, SWIZZLE, SWIZZLE, swizzle_inner=True),
    Tx.TileLayout(S[(N1, N1) : (N1, 1)]),
)

# Swizzled [pipe,64,64] f16 layout (pipelined TMA buffers)
swizzled_pipe_layout = Tx.ComposeLayout(
    Tx.SwizzleLayout(SWIZZLE, SWIZZLE, SWIZZLE, swizzle_inner=True),
    Tx.TileLayout(S[(PIPE_DEPTH, N1, N1) : (N1 * N1, N1, 1)]),
)


def get_fftconv_kernel():
    # fmt: off
    @Tx.prim_func(tirx=True)
    def fftconv(
        x_ptr: Tx.handle,
        kf_real_ptr: Tx.handle, kf_imag_ptr: Tx.handle,
        f_real_ptr: Tx.handle, f_imag_ptr: Tx.handle,
        finv_real_ptr: Tx.handle, finv_imag_ptr: Tx.handle,
        tw_real_ptr: Tx.handle, tw_imag_ptr: Tx.handle,
        twinv_real_ptr: Tx.handle, twinv_imag_ptr: Tx.handle,
        o_ptr: Tx.handle,
    ):
        B = Tx.int32()
        H = Tx.int32()
        x_g = Tx.match_buffer(x_ptr, [B, H, N1, N1], "float16", scope="global")
        kf_real_g = Tx.match_buffer(kf_real_ptr, [H, N1, N1], "float16", scope="global")
        kf_imag_g = Tx.match_buffer(kf_imag_ptr, [H, N1, N1], "float16", scope="global")
        f_real_g = Tx.match_buffer(f_real_ptr, [N1, N1], "float16", scope="global")
        f_imag_g = Tx.match_buffer(f_imag_ptr, [N1, N1], "float16", scope="global")
        finv_real_g = Tx.match_buffer(finv_real_ptr, [N1, N1], "float16", scope="global")
        finv_imag_g = Tx.match_buffer(finv_imag_ptr, [N1, N1], "float16", scope="global")
        tw_real_g = Tx.match_buffer(tw_real_ptr, [N1, N1], "float16", scope="global")
        tw_imag_g = Tx.match_buffer(tw_imag_ptr, [N1, N1], "float16", scope="global")
        twinv_real_g = Tx.match_buffer(twinv_real_ptr, [N1, N1], "float16", scope="global")
        twinv_imag_g = Tx.match_buffer(twinv_imag_ptr, [N1, N1], "float16", scope="global")
        o_g = Tx.match_buffer(o_ptr, [B, H, N1, N1], "float16", scope="global")

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
                # Offsets in f16 elements (divide byte offset by 2):
                BUF_64x64 = N1 * N1  # 4096 f16 elems = 8192 bytes

                base_off_f16 = 1024 // F16_BYTES  # 512

                # Swizzled MMA operand buffers
                f_real_off = base_off_f16  # 512
                F_real_s = Tx.decl_buffer((N1, N1), "float16", buf.data,
                                          layout=swizzled_layout, elem_offset=f_real_off)
                f_imag_off = f_real_off + BUF_64x64  # 4608
                F_imag_s = Tx.decl_buffer((N1, N1), "float16", buf.data,
                                          layout=swizzled_layout, elem_offset=f_imag_off)
                finv_real_off = f_imag_off + BUF_64x64  # 8704
                Finv_real_s = Tx.decl_buffer((N1, N1), "float16", buf.data,
                                             layout=swizzled_layout, elem_offset=finv_real_off)
                finv_imag_off = finv_real_off + BUF_64x64  # 12800
                Finv_imag_s = Tx.decl_buffer((N1, N1), "float16", buf.data,
                                             layout=swizzled_layout, elem_offset=finv_imag_off)
                x_smem_off = finv_imag_off + BUF_64x64  # 16896
                X_smem = Tx.decl_buffer((PIPE_DEPTH, N1, N1), "float16", buf.data,
                                        layout=swizzled_pipe_layout, elem_offset=x_smem_off)
                work_a_off = x_smem_off + PIPE_DEPTH * BUF_64x64  # 20992
                work_A = Tx.decl_buffer((N1, N1), "float16", buf.data,
                                        layout=swizzled_layout, elem_offset=work_a_off)
                work_b_off = work_a_off + BUF_64x64  # 25088
                work_B = Tx.decl_buffer((N1, N1), "float16", buf.data,
                                        layout=swizzled_layout, elem_offset=work_b_off)
                d_out_off = work_b_off + BUF_64x64  # 29184
                D_out = Tx.decl_buffer((N1, N1), "float16", buf.data,
                                       layout=swizzled_layout, elem_offset=d_out_off)

                # Non-swizzled buffers (f16) — elementwise only
                nonswiz_f16_off = d_out_off + BUF_64x64  # 33280
                tw_real_s = Tx.decl_buffer((N1, N1), "float16", buf.data,
                                           elem_offset=nonswiz_f16_off)
                tw_imag_off = nonswiz_f16_off + BUF_64x64  # 37376
                tw_imag_s = Tx.decl_buffer((N1, N1), "float16", buf.data,
                                           elem_offset=tw_imag_off)
                twinv_real_off = tw_imag_off + BUF_64x64  # 41472
                twinv_real_s = Tx.decl_buffer((N1, N1), "float16", buf.data,
                                              elem_offset=twinv_real_off)
                twinv_imag_off = twinv_real_off + BUF_64x64  # 45568
                twinv_imag_s = Tx.decl_buffer((N1, N1), "float16", buf.data,
                                              elem_offset=twinv_imag_off)
                kf_real_off = twinv_imag_off + BUF_64x64  # 49664
                kf_real_s = Tx.decl_buffer((N1, N1), "float16", buf.data,
                                           elem_offset=kf_real_off)
                kf_imag_off = kf_real_off + BUF_64x64  # 53760
                kf_imag_s = Tx.decl_buffer((N1, N1), "float16", buf.data,
                                           elem_offset=kf_imag_off)

                # Non-swizzled buffers (f32) — consumer temporaries
                nonswiz_f32_byte = (kf_imag_off + BUF_64x64) * F16_BYTES  # 115712 bytes
                w_real_off_f32 = nonswiz_f32_byte // F32_BYTES  # 28928
                w_real_s = Tx.decl_buffer((N1, N1), "float32", buf.data,
                                          elem_offset=w_real_off_f32)
                w_imag_off_f32 = w_real_off_f32 + N1 * N1  # 33024
                w_imag_s = Tx.decl_buffer((N1, N1), "float32", buf.data,
                                          elem_offset=w_imag_off_f32)

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

                # ---- Load 8 constant matrices from global -> SMEM ----
                # All 256 threads cooperatively load before scope_partition.
                # Swizzled buffers: use element-by-element writes (layout handles swizzle).
                # Non-swizzled buffers: direct element writes.
                with Tx.thread():
                    tid_flat = Tx.meta_var(wg_id * (WARP_NUMBER * 32) + warp_id * 32 + lane_id)
                    # 256 threads, 4096 elements per matrix = 16 elements per thread
                    for ei in Tx.serial(16):
                        idx = Tx.meta_var(tid_flat * 16 + ei)
                        row = Tx.meta_var(idx // N1)
                        col = Tx.meta_var(idx % N1)
                        F_real_s[row, col] = f_real_g[row, col]
                        F_imag_s[row, col] = f_imag_g[row, col]
                        Finv_real_s[row, col] = finv_real_g[row, col]
                        Finv_imag_s[row, col] = finv_imag_g[row, col]
                        tw_real_s[row, col] = tw_real_g[row, col]
                        tw_imag_s[row, col] = tw_imag_g[row, col]
                        twinv_real_s[row, col] = twinv_real_g[row, col]
                        twinv_imag_s[row, col] = twinv_imag_g[row, col]

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
                MMA_SDO = 8 * N1 * F16_BYTES // F128_BYTES  # 8*64*2/16 = 64

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
                            total_bh_tma = Tx.local_scalar("int32", "total_bh_tma")
                            head_tma = Tx.local_scalar("int32", "head_tma")
                            batch_tma = Tx.local_scalar("int32", "batch_tma")
                            total_bh_tma = B * H
                            wid_tma = bx
                            while wid_tma < total_bh_tma:
                                if wid_tma != bx:
                                    with Tx.thread()[Tx.ptx.elect_sync()]:
                                        workitem_sync_bar.wait(0, wi_phase[0])
                                    wi_phase[0] = wi_phase[0] ^ 1
                                with Tx.thread()[Tx.ptx.elect_sync()]:
                                    tic = Tx.meta_var(0)
                                    head_tma = wid_tma % H
                                    batch_tma = wid_tma // H
                                    tma_copy = Tx.meta_var({"dispatch": "tma", "mbar": tma_finished.ptr_to([tic]), "cta_group": CTA_GROUP})
                                    # Load X for this work item
                                    Tx.copy_async(X_smem[tic, :, :], x_g[batch_tma, head_tma, 0 : N1, 0 : N1], **tma_copy)
                                    tma2mma_bar.arrive(tic, N1 * N1 * F16_BYTES)
                                phase[0] = phase[0] ^ 1
                                wid_tma = wid_tma + SM_COUNT

                        elif warp_id == 0:
                            # ---- MMA warp ----
                            # All phases use transB=True (B operand is [K,N])
                            descI_tb: Tx.uint32
                            Tx.ptx.tcgen05.encode_instr_descriptor(
                                Tx.address_of(descI_tb), "float32", "float16", "float16",
                                MMA_M * CTA_GROUP, MMA_N, MMA_K, False, True, CTA_GROUP)

                            phase[0] = 0
                            wid_mma = Tx.local_scalar("int32", "wid_mma")
                            total_bh_mma = Tx.local_scalar("int32", "total_bh_mma")
                            phase_c2m: Tx.int32
                            phase_c2m = 0
                            total_bh_mma = B * H
                            wid_mma = bx
                            while wid_mma < total_bh_mma:
                                with Tx.thread()[Tx.ptx.elect_sync()]:
                                    tic = Tx.meta_var(0)

                                    # == Phase 1: FR @ X -> W.real (in TMEM) ==
                                    # F_real[64,64] @ X[64,64]^T -> [64,64]
                                    tma2mma_bar.wait(tic, phase[0])
                                    Tx.ptx.tcgen05.fence.after_thread_sync()
                                    for ki in Tx.unroll(N1 // MMA_K):  # 4 iterations
                                        Tx.ptx.tcgen05.encode_matrix_descriptor(
                                            Tx.address_of(descA), F_real_s.ptr_to([0, ki * MMA_K]),
                                            MMA_LDO, MMA_SDO, SWIZZLE)
                                        Tx.ptx.tcgen05.encode_matrix_descriptor(
                                            Tx.address_of(descB), X_smem.ptr_to([tic, ki * MMA_K, 0]),
                                            MMA_LDO, MMA_SDO, SWIZZLE)
                                        Tx.ptx.tcgen05.mma("float32", "float16", "float16",
                                            Tx.cuda.get_tmem_addr(tmem_addr, 0, 0),
                                            descA, descB, descI_tb, False, CTA_GROUP, ki > 0)
                                    mma2consumer_bar.arrive(0)

                                    # Wait consumer done with Phase 1
                                    consumer2mma_bar.wait(0, phase_c2m)
                                    Tx.ptx.tcgen05.fence.after_thread_sync()

                                    # == Phase 2: FI @ X -> W.imag (in TMEM) ==
                                    for ki in Tx.unroll(N1 // MMA_K):  # 4 iterations
                                        Tx.ptx.tcgen05.encode_matrix_descriptor(
                                            Tx.address_of(descA), F_imag_s.ptr_to([0, ki * MMA_K]),
                                            MMA_LDO, MMA_SDO, SWIZZLE)
                                        Tx.ptx.tcgen05.encode_matrix_descriptor(
                                            Tx.address_of(descB), X_smem.ptr_to([tic, ki * MMA_K, 0]),
                                            MMA_LDO, MMA_SDO, SWIZZLE)
                                        Tx.ptx.tcgen05.mma("float32", "float16", "float16",
                                            Tx.cuda.get_tmem_addr(tmem_addr, 0, 0),
                                            descA, descB, descI_tb, False, CTA_GROUP, ki > 0)
                                    mma2consumer_bar.arrive(0)

                                    # Wait consumer done with Phase 2 (tw multiply, write work_A/B)
                                    consumer2mma_bar.wait(0, phase_c2m ^ 1)
                                    Tx.ptx.tcgen05.fence.after_thread_sync()
                                    # Release X_smem for TMA (no longer needed after phase 2)
                                    Tx.ptx.mbarrier.arrive(mma2tma_bar.mbar.ptr_to([tic]))

                                    # == Phase 3: WR@FR + (-WI)@FI -> new W.real (cross-buffer accum) ==
                                    # work_A has W'.real, work_B has -W'.imag
                                    # First half: work_A @ F_real (4 iters, init)
                                    for ki in Tx.unroll(N1 // MMA_K):
                                        Tx.ptx.tcgen05.encode_matrix_descriptor(
                                            Tx.address_of(descA), work_A.ptr_to([0, ki * MMA_K]),
                                            MMA_LDO, MMA_SDO, SWIZZLE)
                                        Tx.ptx.tcgen05.encode_matrix_descriptor(
                                            Tx.address_of(descB), F_real_s.ptr_to([ki * MMA_K, 0]),
                                            MMA_LDO, MMA_SDO, SWIZZLE)
                                        Tx.ptx.tcgen05.mma("float32", "float16", "float16",
                                            Tx.cuda.get_tmem_addr(tmem_addr, 0, 0),
                                            descA, descB, descI_tb, False, CTA_GROUP, ki > 0)
                                    # Second half: work_B @ F_imag (4 iters, accumulate)
                                    # work_B = -W'.imag, F_imag = FI
                                    # (-WI) @ FI accumulated onto WR@FR gives WR@FR - WI@FI = new_W.real
                                    for ki in Tx.unroll(N1 // MMA_K):
                                        Tx.ptx.tcgen05.encode_matrix_descriptor(
                                            Tx.address_of(descA), work_B.ptr_to([0, ki * MMA_K]),
                                            MMA_LDO, MMA_SDO, SWIZZLE)
                                        Tx.ptx.tcgen05.encode_matrix_descriptor(
                                            Tx.address_of(descB), F_imag_s.ptr_to([ki * MMA_K, 0]),
                                            MMA_LDO, MMA_SDO, SWIZZLE)
                                        Tx.ptx.tcgen05.mma("float32", "float16", "float16",
                                            Tx.cuda.get_tmem_addr(tmem_addr, 0, 0),
                                            descA, descB, descI_tb, False, CTA_GROUP, True)
                                    mma2consumer_bar.arrive(0)

                                    # Wait consumer done with Phase 3 (save W.real, flip work_B)
                                    consumer2mma_bar.wait(0, phase_c2m)
                                    Tx.ptx.tcgen05.fence.after_thread_sync()

                                    # == Phase 4: WR@FI + WI@FR -> new W.imag (cross-buffer accum) ==
                                    # work_A still has W'.real, work_B now has +W'.imag
                                    # First half: work_A @ F_imag (4 iters, init)
                                    for ki in Tx.unroll(N1 // MMA_K):
                                        Tx.ptx.tcgen05.encode_matrix_descriptor(
                                            Tx.address_of(descA), work_A.ptr_to([0, ki * MMA_K]),
                                            MMA_LDO, MMA_SDO, SWIZZLE)
                                        Tx.ptx.tcgen05.encode_matrix_descriptor(
                                            Tx.address_of(descB), F_imag_s.ptr_to([ki * MMA_K, 0]),
                                            MMA_LDO, MMA_SDO, SWIZZLE)
                                        Tx.ptx.tcgen05.mma("float32", "float16", "float16",
                                            Tx.cuda.get_tmem_addr(tmem_addr, 0, 0),
                                            descA, descB, descI_tb, False, CTA_GROUP, ki > 0)
                                    # Second half: work_B @ F_real (4 iters, accumulate)
                                    # work_B = +WI, FR -> WR@FI + WI@FR = new_W.imag
                                    for ki in Tx.unroll(N1 // MMA_K):
                                        Tx.ptx.tcgen05.encode_matrix_descriptor(
                                            Tx.address_of(descA), work_B.ptr_to([0, ki * MMA_K]),
                                            MMA_LDO, MMA_SDO, SWIZZLE)
                                        Tx.ptx.tcgen05.encode_matrix_descriptor(
                                            Tx.address_of(descB), F_real_s.ptr_to([ki * MMA_K, 0]),
                                            MMA_LDO, MMA_SDO, SWIZZLE)
                                        Tx.ptx.tcgen05.mma("float32", "float16", "float16",
                                            Tx.cuda.get_tmem_addr(tmem_addr, 0, 0),
                                            descA, descB, descI_tb, False, CTA_GROUP, True)
                                    mma2consumer_bar.arrive(0)

                                    # Wait consumer done with Phase 4 (kf multiply, write work_A/B)
                                    consumer2mma_bar.wait(0, phase_c2m ^ 1)
                                    Tx.ptx.tcgen05.fence.after_thread_sync()

                                    # == Phase 5: WR@FiR + (-WI)@FiI -> new W.real ==
                                    for ki in Tx.unroll(N1 // MMA_K):
                                        Tx.ptx.tcgen05.encode_matrix_descriptor(
                                            Tx.address_of(descA), work_A.ptr_to([0, ki * MMA_K]),
                                            MMA_LDO, MMA_SDO, SWIZZLE)
                                        Tx.ptx.tcgen05.encode_matrix_descriptor(
                                            Tx.address_of(descB), Finv_real_s.ptr_to([ki * MMA_K, 0]),
                                            MMA_LDO, MMA_SDO, SWIZZLE)
                                        Tx.ptx.tcgen05.mma("float32", "float16", "float16",
                                            Tx.cuda.get_tmem_addr(tmem_addr, 0, 0),
                                            descA, descB, descI_tb, False, CTA_GROUP, ki > 0)
                                    for ki in Tx.unroll(N1 // MMA_K):
                                        Tx.ptx.tcgen05.encode_matrix_descriptor(
                                            Tx.address_of(descA), work_B.ptr_to([0, ki * MMA_K]),
                                            MMA_LDO, MMA_SDO, SWIZZLE)
                                        Tx.ptx.tcgen05.encode_matrix_descriptor(
                                            Tx.address_of(descB), Finv_imag_s.ptr_to([ki * MMA_K, 0]),
                                            MMA_LDO, MMA_SDO, SWIZZLE)
                                        Tx.ptx.tcgen05.mma("float32", "float16", "float16",
                                            Tx.cuda.get_tmem_addr(tmem_addr, 0, 0),
                                            descA, descB, descI_tb, False, CTA_GROUP, True)
                                    mma2consumer_bar.arrive(0)

                                    # Wait consumer done with Phase 5
                                    consumer2mma_bar.wait(0, phase_c2m)
                                    Tx.ptx.tcgen05.fence.after_thread_sync()

                                    # == Phase 6: WR@FiI + WI@FiR -> new W.imag ==
                                    for ki in Tx.unroll(N1 // MMA_K):
                                        Tx.ptx.tcgen05.encode_matrix_descriptor(
                                            Tx.address_of(descA), work_A.ptr_to([0, ki * MMA_K]),
                                            MMA_LDO, MMA_SDO, SWIZZLE)
                                        Tx.ptx.tcgen05.encode_matrix_descriptor(
                                            Tx.address_of(descB), Finv_imag_s.ptr_to([ki * MMA_K, 0]),
                                            MMA_LDO, MMA_SDO, SWIZZLE)
                                        Tx.ptx.tcgen05.mma("float32", "float16", "float16",
                                            Tx.cuda.get_tmem_addr(tmem_addr, 0, 0),
                                            descA, descB, descI_tb, False, CTA_GROUP, ki > 0)
                                    for ki in Tx.unroll(N1 // MMA_K):
                                        Tx.ptx.tcgen05.encode_matrix_descriptor(
                                            Tx.address_of(descA), work_B.ptr_to([0, ki * MMA_K]),
                                            MMA_LDO, MMA_SDO, SWIZZLE)
                                        Tx.ptx.tcgen05.encode_matrix_descriptor(
                                            Tx.address_of(descB), Finv_real_s.ptr_to([ki * MMA_K, 0]),
                                            MMA_LDO, MMA_SDO, SWIZZLE)
                                        Tx.ptx.tcgen05.mma("float32", "float16", "float16",
                                            Tx.cuda.get_tmem_addr(tmem_addr, 0, 0),
                                            descA, descB, descI_tb, False, CTA_GROUP, True)
                                    mma2consumer_bar.arrive(0)

                                    # Wait consumer done with Phase 6 (twinv multiply, write work_A/B)
                                    consumer2mma_bar.wait(0, phase_c2m ^ 1)
                                    Tx.ptx.tcgen05.fence.after_thread_sync()

                                    # == Phase 7: FiR@WR + FiI@(-WI) -> final W.real ==
                                    # Finv on LEFT: A=Finv, B=work
                                    # work_A has W'.real, work_B has -W'.imag
                                    for ki in Tx.unroll(N1 // MMA_K):
                                        Tx.ptx.tcgen05.encode_matrix_descriptor(
                                            Tx.address_of(descA), Finv_real_s.ptr_to([0, ki * MMA_K]),
                                            MMA_LDO, MMA_SDO, SWIZZLE)
                                        Tx.ptx.tcgen05.encode_matrix_descriptor(
                                            Tx.address_of(descB), work_A.ptr_to([ki * MMA_K, 0]),
                                            MMA_LDO, MMA_SDO, SWIZZLE)
                                        Tx.ptx.tcgen05.mma("float32", "float16", "float16",
                                            Tx.cuda.get_tmem_addr(tmem_addr, 0, 0),
                                            descA, descB, descI_tb, False, CTA_GROUP, ki > 0)
                                    # FiI @ (-WI): Finv_imag @ work_B (work_B = -W'.imag)
                                    # FiI @ (-WI) accumulated = FiR@WR - FiI@WI = real part
                                    for ki in Tx.unroll(N1 // MMA_K):
                                        Tx.ptx.tcgen05.encode_matrix_descriptor(
                                            Tx.address_of(descA), Finv_imag_s.ptr_to([0, ki * MMA_K]),
                                            MMA_LDO, MMA_SDO, SWIZZLE)
                                        Tx.ptx.tcgen05.encode_matrix_descriptor(
                                            Tx.address_of(descB), work_B.ptr_to([ki * MMA_K, 0]),
                                            MMA_LDO, MMA_SDO, SWIZZLE)
                                        Tx.ptx.tcgen05.mma("float32", "float16", "float16",
                                            Tx.cuda.get_tmem_addr(tmem_addr, 0, 0),
                                            descA, descB, descI_tb, False, CTA_GROUP, True)
                                    mma2consumer_bar.arrive(0)

                                    # Wait consumer done with Phase 7 (TMA store)
                                    consumer2mma_bar.wait(0, phase_c2m)
                                    Tx.ptx.tcgen05.fence.after_thread_sync()
                                    # 7 phases = odd -> flip phase_c2m each work item
                                    phase_c2m = phase_c2m ^ 1
                                phase[0] = phase[0] ^ 1
                                with Tx.thread()[Tx.ptx.elect_sync()]:
                                    workitem_sync_bar.arrive_only(0)
                                wid_mma = wid_mma + SM_COUNT

                    # === Consumer warpgroup (WG0) ===
                    with Tx.warpgroup()[0:1]:
                        Tx.cuda.trap_when_assert_failed(tmem_addr == 0)

                        phase_m2c: Tx.int32
                        phase_m2c = 0
                        phase_c2m_c: Tx.int32
                        phase_c2m_c = 0

                        wid_con = Tx.local_scalar("int32", "wid_con")
                        total_bh_con = Tx.local_scalar("int32", "total_bh_con")
                        head_con = Tx.local_scalar("int32", "head_con")
                        batch_con = Tx.local_scalar("int32", "batch_con")
                        total_bh_con = B * H
                        wid_con = bx
                        while wid_con < total_bh_con:
                            head_con = wid_con % H
                            batch_con = wid_con // H

                            # Load kf for this head (consumer threads, overlaps with MMA phases 1-2)
                            with Tx.thread():
                                tid_flat = Tx.meta_var(warp_id * 32 + lane_id)
                                for ei in Tx.serial(32):
                                    idx = Tx.meta_var(tid_flat * 32 + ei)
                                    row = Tx.meta_var(idx // N1)
                                    col = Tx.meta_var(idx % N1)
                                    kf_real_s[row, col] = kf_real_g[head_con, row, col]
                                    kf_imag_s[row, col] = kf_imag_g[head_con, row, col]
                            Tx.cuda.warpgroup_sync(10)

                            # ======== Phase 1: Read W.real from TMEM ========
                            mma2consumer_bar.wait(0, phase_m2c)
                            phase_m2c = phase_m2c ^ 1
                            Tx.ptx.tcgen05.fence.after_thread_sync()

                            Tx.copy(reg_wg[:, :], tmem[:, 0:TMEM_LD_SIZE])

                            with Tx.thread():
                                out_row = Tx.meta_var(warp_id * 16 + lane_id)
                                if lane_id < 16:
                                    # Save W.real to w_real_s (f32)
                                    for j in Tx.serial(N1):
                                        w_real_s[out_row, j] = reg[j]

                            Tx.ptx.tcgen05.fence.before_thread_sync()
                            consumer2mma_bar.arrive(0)
                            phase_c2m_c = phase_c2m_c ^ 1

                            # ======== Phase 2: Read W.imag from TMEM ========
                            mma2consumer_bar.wait(0, phase_m2c)
                            phase_m2c = phase_m2c ^ 1
                            Tx.ptx.tcgen05.fence.after_thread_sync()

                            Tx.copy(reg_wg[:, :], tmem[:, 0:TMEM_LD_SIZE])

                            with Tx.thread():
                                out_row = Tx.meta_var(warp_id * 16 + lane_id)
                                if lane_id < 16:
                                    # Save W.imag to w_imag_s (f32)
                                    for j in Tx.serial(N1):
                                        w_imag_s[out_row, j] = reg[j]

                            Tx.cuda.warpgroup_sync(10)

                            # Apply tw multiply: W' = W * tw (complex)
                            # Then write work_A = W'.real, work_B = -W'.imag
                            with Tx.thread():
                                out_row = Tx.meta_var(warp_id * 16 + lane_id)
                                if lane_id < 16:
                                    for j in Tx.serial(N1):
                                        wr = Tx.meta_var(w_real_s[out_row, j])
                                        wi = Tx.meta_var(w_imag_s[out_row, j])
                                        tr = Tx.meta_var(Tx.cast(tw_real_s[out_row, j], "float32"))
                                        ti = Tx.meta_var(Tx.cast(tw_imag_s[out_row, j], "float32"))
                                        new_real = Tx.meta_var(wr * tr - wi * ti)
                                        new_imag = Tx.meta_var(wr * ti + wi * tr)
                                        work_A[out_row, j] = Tx.cast(new_real, "float16")
                                        work_B[out_row, j] = Tx.cast(0.0 - new_imag, "float16")

                            Tx.ptx.tcgen05.fence.before_thread_sync()
                            consumer2mma_bar.arrive(0)
                            phase_c2m_c = phase_c2m_c ^ 1

                            # ======== Phase 3: Read new W.real from TMEM ========
                            mma2consumer_bar.wait(0, phase_m2c)
                            phase_m2c = phase_m2c ^ 1
                            Tx.ptx.tcgen05.fence.after_thread_sync()

                            Tx.copy(reg_wg[:, :], tmem[:, 0:TMEM_LD_SIZE])

                            with Tx.thread():
                                out_row = Tx.meta_var(warp_id * 16 + lane_id)
                                if lane_id < 16:
                                    # Save new W.real to w_real_s
                                    for j in Tx.serial(N1):
                                        w_real_s[out_row, j] = reg[j]
                                    # Flip work_B from -W'.imag to +W'.imag for Phase 4
                                    for j in Tx.serial(N1):
                                        work_B[out_row, j] = Tx.cast(0.0 - Tx.cast(work_B[out_row, j], "float32"), "float16")

                            Tx.ptx.tcgen05.fence.before_thread_sync()
                            consumer2mma_bar.arrive(0)
                            phase_c2m_c = phase_c2m_c ^ 1

                            # ======== Phase 4: Read new W.imag from TMEM ========
                            mma2consumer_bar.wait(0, phase_m2c)
                            phase_m2c = phase_m2c ^ 1
                            Tx.ptx.tcgen05.fence.after_thread_sync()

                            Tx.copy(reg_wg[:, :], tmem[:, 0:TMEM_LD_SIZE])

                            with Tx.thread():
                                out_row = Tx.meta_var(warp_id * 16 + lane_id)
                                if lane_id < 16:
                                    # Save new W.imag to w_imag_s
                                    for j in Tx.serial(N1):
                                        w_imag_s[out_row, j] = reg[j]

                            Tx.cuda.warpgroup_sync(10)

                            # Apply kf multiply: W' = W * kf (complex)
                            # Then write work_A = W'.real, work_B = -W'.imag
                            with Tx.thread():
                                out_row = Tx.meta_var(warp_id * 16 + lane_id)
                                if lane_id < 16:
                                    for j in Tx.serial(N1):
                                        wr = Tx.meta_var(w_real_s[out_row, j])
                                        wi = Tx.meta_var(w_imag_s[out_row, j])
                                        kr = Tx.meta_var(Tx.cast(kf_real_s[out_row, j], "float32"))
                                        ki_val = Tx.meta_var(Tx.cast(kf_imag_s[out_row, j], "float32"))
                                        new_real = Tx.meta_var(wr * kr - wi * ki_val)
                                        new_imag = Tx.meta_var(wr * ki_val + wi * kr)
                                        work_A[out_row, j] = Tx.cast(new_real, "float16")
                                        work_B[out_row, j] = Tx.cast(0.0 - new_imag, "float16")

                            Tx.ptx.tcgen05.fence.before_thread_sync()
                            consumer2mma_bar.arrive(0)
                            phase_c2m_c = phase_c2m_c ^ 1

                            # ======== Phase 5: Read new W.real from TMEM ========
                            mma2consumer_bar.wait(0, phase_m2c)
                            phase_m2c = phase_m2c ^ 1
                            Tx.ptx.tcgen05.fence.after_thread_sync()

                            Tx.copy(reg_wg[:, :], tmem[:, 0:TMEM_LD_SIZE])

                            with Tx.thread():
                                out_row = Tx.meta_var(warp_id * 16 + lane_id)
                                if lane_id < 16:
                                    # Save new W.real to w_real_s
                                    for j in Tx.serial(N1):
                                        w_real_s[out_row, j] = reg[j]
                                    # Flip work_B from -W'.imag to +W'.imag for Phase 6
                                    for j in Tx.serial(N1):
                                        work_B[out_row, j] = Tx.cast(0.0 - Tx.cast(work_B[out_row, j], "float32"), "float16")

                            Tx.ptx.tcgen05.fence.before_thread_sync()
                            consumer2mma_bar.arrive(0)
                            phase_c2m_c = phase_c2m_c ^ 1

                            # ======== Phase 6: Read new W.imag from TMEM ========
                            mma2consumer_bar.wait(0, phase_m2c)
                            phase_m2c = phase_m2c ^ 1
                            Tx.ptx.tcgen05.fence.after_thread_sync()

                            Tx.copy(reg_wg[:, :], tmem[:, 0:TMEM_LD_SIZE])

                            with Tx.thread():
                                out_row = Tx.meta_var(warp_id * 16 + lane_id)
                                if lane_id < 16:
                                    # Save new W.imag (for twinv multiply)
                                    for j in Tx.serial(N1):
                                        w_imag_s[out_row, j] = reg[j]

                            Tx.cuda.warpgroup_sync(10)

                            # Apply twinv multiply: W' = W * twinv (complex)
                            # Then write work_A = W'.real, work_B = -W'.imag
                            with Tx.thread():
                                out_row = Tx.meta_var(warp_id * 16 + lane_id)
                                if lane_id < 16:
                                    for j in Tx.serial(N1):
                                        wr = Tx.meta_var(w_real_s[out_row, j])
                                        wi = Tx.meta_var(w_imag_s[out_row, j])
                                        tr = Tx.meta_var(Tx.cast(twinv_real_s[out_row, j], "float32"))
                                        ti = Tx.meta_var(Tx.cast(twinv_imag_s[out_row, j], "float32"))
                                        new_real = Tx.meta_var(wr * tr - wi * ti)
                                        new_imag = Tx.meta_var(wr * ti + wi * tr)
                                        work_A[out_row, j] = Tx.cast(new_real, "float16")
                                        work_B[out_row, j] = Tx.cast(0.0 - new_imag, "float16")

                            Tx.ptx.tcgen05.fence.before_thread_sync()
                            consumer2mma_bar.arrive(0)
                            phase_c2m_c = phase_c2m_c ^ 1

                            # ======== Phase 7: Read final W.real from TMEM ========
                            mma2consumer_bar.wait(0, phase_m2c)
                            phase_m2c = phase_m2c ^ 1
                            Tx.ptx.tcgen05.fence.after_thread_sync()

                            Tx.copy(reg_wg[:, :], tmem[:, 0:TMEM_LD_SIZE])

                            # Write to D_out and TMA store
                            with Tx.thread():
                                out_row = Tx.meta_var(warp_id * 16 + lane_id)
                                if lane_id < 16:
                                    for j in Tx.serial(N1):
                                        D_out[out_row, j] = Tx.cast(reg[j], "float16")

                            Tx.ptx.fence.proxy_async("shared::cta")
                            Tx.cuda.warpgroup_sync(10)
                            with Tx.thread()[lane_id == 0 and warp_id == 0]:
                                Tx.copy_async(o_g[batch_con, head_con, 0 : N1, 0 : N1],
                                              D_out[:, :], dispatch="tma", cache_hint="evict_last")
                                Tx.ptx.cp_async.bulk.commit_group()
                                Tx.ptx.cp_async.bulk.wait_group(0)
                            Tx.cuda.warpgroup_sync(10)

                            Tx.ptx.tcgen05.fence.before_thread_sync()
                            consumer2mma_bar.arrive(0)
                            phase_c2m_c = phase_c2m_c ^ 1

                            wid_con = wid_con + SM_COUNT

                # ---- TMEM deallocation ----
                with Tx.warp()[0:1]:
                    Tx.ptx.tcgen05.relinquish_alloc_permit(cta_group=CTA_GROUP)
                    Tx.ptx.tcgen05.dealloc(tmem_addr, n_cols=N_COLS, cta_group=CTA_GROUP)

                Tx.cuda.cta_sync()
    # fmt: on
    return fftconv


@pytest.mark.parametrize("batch,heads", [(2, 4), (1, 8), (10, 16)])
def test_fftconv(batch, heads):
    (u_f16, kf_real, kf_imag,
     f_real, f_imag, finv_real, finv_imag,
     tw_real, tw_imag, twinv_real, twinv_imag,
     u_orig, k_orig) = prepare_data(batch, heads)

    # Reference
    o_ref = ref_fftconv(u_orig, k_orig)
    o_ref_np = o_ref.float().cpu().numpy()

    # Compile
    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": get_fftconv_kernel()})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
        src = mod.mod.imports[0].inspect_source()
        print(src[:2000])

    # Run
    DEV = tvm.cuda(0)
    o_np = np.zeros((batch, heads, N1, N1), dtype=np.float16)

    x_tvm = tvm.runtime.tensor(u_f16.cpu().numpy(), DEV)
    kf_real_tvm = tvm.runtime.tensor(kf_real.cpu().numpy(), DEV)
    kf_imag_tvm = tvm.runtime.tensor(kf_imag.cpu().numpy(), DEV)
    f_real_tvm = tvm.runtime.tensor(f_real.cpu().numpy(), DEV)
    f_imag_tvm = tvm.runtime.tensor(f_imag.cpu().numpy(), DEV)
    finv_real_tvm = tvm.runtime.tensor(finv_real.cpu().numpy(), DEV)
    finv_imag_tvm = tvm.runtime.tensor(finv_imag.cpu().numpy(), DEV)
    tw_real_tvm = tvm.runtime.tensor(tw_real.cpu().numpy(), DEV)
    tw_imag_tvm = tvm.runtime.tensor(tw_imag.cpu().numpy(), DEV)
    twinv_real_tvm = tvm.runtime.tensor(twinv_real.cpu().numpy(), DEV)
    twinv_imag_tvm = tvm.runtime.tensor(twinv_imag.cpu().numpy(), DEV)
    o_tvm = tvm.runtime.tensor(o_np, DEV)

    mod(x_tvm, kf_real_tvm, kf_imag_tvm,
        f_real_tvm, f_imag_tvm, finv_real_tvm, finv_imag_tvm,
        tw_real_tvm, tw_imag_tvm, twinv_real_tvm, twinv_imag_tvm,
        o_tvm)

    o_tir = o_tvm.numpy().reshape(batch, heads, N).astype(np.float32)
    o_ref_flat = o_ref_np.reshape(batch, heads, N)

    np.testing.assert_allclose(o_tir, o_ref_flat, rtol=0.05, atol=2.0)
    print(f"PASSED: batch={batch}, heads={heads}")


def bench_fftconv():
    """Benchmark FFT convolution kernel at TK-equivalent dimensions."""
    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": get_fftconv_kernel()})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")

    DEV = tvm.cuda(0)

    # FLOPS from TK: 2 * (10 * N * log2(N) * d_model * B)
    # N=4096, d_model = heads * 1 (each head processes one channel)
    def flops(ms, batch, heads):
        return 2 * (10 * N * math.log2(N) * heads * batch) / (ms * 1e-3)

    print(f"\n{'='*60}")
    print(f"FFTConv Benchmark (N={N}={N1}x{N1})")
    print(f"{'='*60}")

    with ProtonContext("fftconv"):
        for batch, heads in [(1, 128), (2, 64), (4, 32)]:
            (u_f16, kf_real, kf_imag,
             f_real, f_imag, finv_real, finv_imag,
             tw_real, tw_imag, twinv_real, twinv_imag,
             _, _) = prepare_data(batch, heads)

            o_np = np.zeros((batch, heads, N1, N1), dtype=np.float16)
            x_tvm = tvm.runtime.tensor(u_f16.cpu().numpy(), DEV)
            kf_real_tvm = tvm.runtime.tensor(kf_real.cpu().numpy(), DEV)
            kf_imag_tvm = tvm.runtime.tensor(kf_imag.cpu().numpy(), DEV)
            f_real_tvm = tvm.runtime.tensor(f_real.cpu().numpy(), DEV)
            f_imag_tvm = tvm.runtime.tensor(f_imag.cpu().numpy(), DEV)
            finv_real_tvm = tvm.runtime.tensor(finv_real.cpu().numpy(), DEV)
            finv_imag_tvm = tvm.runtime.tensor(finv_imag.cpu().numpy(), DEV)
            tw_real_tvm = tvm.runtime.tensor(tw_real.cpu().numpy(), DEV)
            tw_imag_tvm = tvm.runtime.tensor(tw_imag.cpu().numpy(), DEV)
            twinv_real_tvm = tvm.runtime.tensor(twinv_real.cpu().numpy(), DEV)
            twinv_imag_tvm = tvm.runtime.tensor(twinv_imag.cpu().numpy(), DEV)
            o_tvm = tvm.runtime.tensor(o_np, DEV)

            func = lambda: mod(
                x_tvm, kf_real_tvm, kf_imag_tvm,
                f_real_tvm, f_imag_tvm, finv_real_tvm, finv_imag_tvm,
                tw_real_tvm, tw_imag_tvm, twinv_real_tvm, twinv_imag_tvm,
                o_tvm,
            )
            ms = bench(func, warmup=100, repeat=300, proton_name=f"fftconv_B{batch}_H{heads}")
            tflops = flops(ms, batch, heads) / 1e12
            print(f"  B={batch:>2d}, H={heads:>4d}: {tflops:.2f} TFLOPS, {ms:.3f} ms")

    print()


if __name__ == "__main__":
    test_fftconv(2, 4)
    bench_fftconv()
