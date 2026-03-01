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
  Phase 4: WR@FI + (-WI)@(-FR) (8 K-iters, uses neg_F_real, no sign flip)
  -> kf multiply, write work_A/work_B
  Phase 5: WR@FiR + (-WI)@FiI (8 K-iters)
  Phase 6: WR@FiI + (-WI)@(-FiR) (8 K-iters, uses neg_Finv_real, no sign flip)
  -> twinv multiply, write work_A/work_B
  Phase 7: FiR@WR + FiI@(-WI) (8 K-iters, only real part needed)
  -> TMA store
"""

import math

import numpy as np
import pytest
import torch

import tvm
import tvm.testing
from tvm.script import tirx as Tx
from tvm.tir.layout import S, TCol, TileLayout, TLane, tid_in_wg
from tvm.tirx.bench.utils import ProtonContext, bench
from tvm.tirx.pipeline import MBarrier, TCGen05Bar, TMABar

# Constants
N1 = 64
N = N1 * N1  # 4096
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
#   neg_F_real[64,64]*f16=8K (swizzled, MMA operand, negated F_real)
#   neg_Finv_real[64,64]*f16=8K (swizzled, MMA operand, negated Finv_real)
#   D_out[64,64]*f16=8K (swizzled, TMA store)
#   tw_real[64,64]*f16=8K + tw_imag[64,64]*f16=8K (swizzled, elementwise)
#   twinv_real[64,64]*f16=8K + twinv_imag[64,64]*f16=8K (swizzled, elementwise)
#   kf_real[64,64]*f16=8K + kf_imag[64,64]*f16=8K (swizzled, per-head)
#   Total = 1024 + 16*8K = ~129K
#   w_real/w_imag kept in registers (64 f32 per thread), no SMEM f32 buffers needed
SMEM_SIZE = (
    1024  # barriers + alignment
    + 16 * N1 * N1 * F16_BYTES  # 16 swizzled f16 buffers (all 64x64)
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

    return (
        u_f16,
        kf_real,
        kf_imag,
        f_real,
        f_imag,
        finv_real,
        finv_imag,
        tw_real,
        tw_imag,
        twinv_real,
        twinv_imag,
        u,
        k,
    )


def ref_fftconv(u, k):
    """Standard FFT convolution reference."""
    u_f = torch.fft.fft(u.double(), n=N)
    k_f = torch.fft.fft(k.double(), n=N)
    y = torch.fft.ifft(u_f * k_f, n=N).real
    return y


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
                F_real_s = pool.alloc((N1, N1), "float16", layout=swizzled_layout)
                F_imag_s = pool.alloc((N1, N1), "float16", layout=swizzled_layout)
                Finv_real_s = pool.alloc((N1, N1), "float16", layout=swizzled_layout)
                Finv_imag_s = pool.alloc((N1, N1), "float16", layout=swizzled_layout)
                X_smem = pool.alloc((PIPE_DEPTH, N1, N1), "float16", layout=swizzled_pipe_layout)
                work_A = pool.alloc((N1, N1), "float16", layout=swizzled_layout)
                work_B = pool.alloc((N1, N1), "float16", layout=swizzled_layout)
                neg_F_real_s = pool.alloc((N1, N1), "float16", layout=swizzled_layout)
                neg_Finv_real_s = pool.alloc((N1, N1), "float16", layout=swizzled_layout)
                D_out = pool.alloc((N1, N1), "float16", layout=swizzled_layout)
                tw_real_s = pool.alloc((N1, N1), "float16", layout=swizzled_layout)
                tw_imag_s = pool.alloc((N1, N1), "float16", layout=swizzled_layout)
                twinv_real_s = pool.alloc((N1, N1), "float16", layout=swizzled_layout)
                twinv_imag_s = pool.alloc((N1, N1), "float16", layout=swizzled_layout)
                kf_real_s = pool.alloc((N1, N1), "float16", layout=swizzled_layout)
                kf_imag_s = pool.alloc((N1, N1), "float16", layout=swizzled_layout)
                pool.commit()

                # ---- Local variables ----
                descI: Tx.uint32  # noqa: F842
                descA: Tx.uint64
                descB: Tx.uint64
                phase = Tx.alloc_buffer((1,), "int32", scope="local")

                tma2mma_bar.init(1)
                mma2tma_bar.init(1)
                mma2consumer_bar.init(1)
                consumer2mma_bar.init(128)  # full consumer WG
                workitem_sync_bar.init(1)

                # ---- TMEM allocation ----
                with Tx.warp()[0:1]:
                    Tx.ptx.tcgen05.alloc(Tx.address_of(tmem_addr), n_cols=N_COLS, cta_group=CTA_GROUP)  # noqa: E501
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
                        neg_F_real_s[row, col] = Tx.cast(0.0 - Tx.cast(f_real_g[row, col], "float32"), "float16")  # noqa: E501
                        neg_Finv_real_s[row, col] = Tx.cast(0.0 - Tx.cast(finv_real_g[row, col], "float32"), "float16")  # noqa: E501
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
                                    tma_copy = Tx.meta_var({"dispatch": "tma", "mbar": tma2mma_bar.ptr_to([tic]), "cta_group": CTA_GROUP})  # noqa: E501
                                    # Load X for this work item
                                    Tx.copy_async(X_smem[tic, :, :], x_g[batch_tma, head_tma, 0 : N1, 0 : N1], **tma_copy)  # noqa: E501
                                    tma2mma_bar.arrive(tic, N1 * N1 * F16_BYTES)
                                phase[0] = phase[0] ^ 1
                                wid_tma = wid_tma + SM_COUNT

                        elif warp_id == 0:
                            # ---- MMA warp ----
                            # All phases use transB=True (B operand is [K,N])
                            descI_tb: Tx.uint32
                            Tx.ptx.tcgen05.encode_instr_descriptor(
                                Tx.address_of(descI_tb), "float32", "float16", "float16",  # noqa: F821
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
                                            Tx.address_of(descA), F_real_s.ptr_to([0, ki * MMA_K]),  # noqa: F821
                                            MMA_LDO, MMA_SDO, SWIZZLE)
                                        Tx.ptx.tcgen05.encode_matrix_descriptor(
                                            Tx.address_of(descB), X_smem.ptr_to([tic, ki * MMA_K, 0]),  # noqa: E501, F821
                                            MMA_LDO, MMA_SDO, SWIZZLE)
                                        Tx.ptx.tcgen05.mma("float32", "float16", "float16",
                                            Tx.cuda.get_tmem_addr(tmem_addr, 0, 0),
                                            descA, descB, descI_tb, False, CTA_GROUP, ki > 0)  # noqa: F821
                                    mma2consumer_bar.arrive(0, CTA_GROUP, 1)

                                    # Wait consumer done with Phase 1
                                    consumer2mma_bar.wait(0, phase_c2m)
                                    Tx.ptx.tcgen05.fence.after_thread_sync()

                                    # == Phase 2: FI @ X -> W.imag (in TMEM) ==
                                    for ki in Tx.unroll(N1 // MMA_K):  # 4 iterations
                                        Tx.ptx.tcgen05.encode_matrix_descriptor(
                                            Tx.address_of(descA), F_imag_s.ptr_to([0, ki * MMA_K]),  # noqa: F821
                                            MMA_LDO, MMA_SDO, SWIZZLE)
                                        Tx.ptx.tcgen05.encode_matrix_descriptor(
                                            Tx.address_of(descB), X_smem.ptr_to([tic, ki * MMA_K, 0]),  # noqa: E501, F821
                                            MMA_LDO, MMA_SDO, SWIZZLE)
                                        Tx.ptx.tcgen05.mma("float32", "float16", "float16",
                                            Tx.cuda.get_tmem_addr(tmem_addr, 0, 0),
                                            descA, descB, descI_tb, False, CTA_GROUP, ki > 0)  # noqa: F821
                                    mma2consumer_bar.arrive(0, CTA_GROUP, 1)

                                    # Wait consumer done with Phase 2 (tw multiply, write work_A/B)
                                    consumer2mma_bar.wait(0, phase_c2m ^ 1)
                                    Tx.ptx.tcgen05.fence.after_thread_sync()
                                    # Release X_smem for TMA (no longer needed after phase 2)
                                    Tx.ptx.mbarrier.arrive(mma2tma_bar.ptr_to([tic]))

                                    # == Phase 3: WR@FR + (-WI)@FI -> new W.real (cross-buffer accum) ==  # noqa: E501
                                    # work_A has W'.real, work_B has -W'.imag
                                    # First half: work_A @ F_real (4 iters, init)
                                    for ki in Tx.unroll(N1 // MMA_K):
                                        Tx.ptx.tcgen05.encode_matrix_descriptor(
                                            Tx.address_of(descA), work_A.ptr_to([0, ki * MMA_K]),  # noqa: F821
                                            MMA_LDO, MMA_SDO, SWIZZLE)
                                        Tx.ptx.tcgen05.encode_matrix_descriptor(
                                            Tx.address_of(descB), F_real_s.ptr_to([ki * MMA_K, 0]),  # noqa: F821
                                            MMA_LDO, MMA_SDO, SWIZZLE)
                                        Tx.ptx.tcgen05.mma("float32", "float16", "float16",
                                            Tx.cuda.get_tmem_addr(tmem_addr, 0, 0),
                                            descA, descB, descI_tb, False, CTA_GROUP, ki > 0)  # noqa: F821
                                    # Second half: work_B @ F_imag (4 iters, accumulate)
                                    # work_B = -W'.imag, F_imag = FI
                                    # (-WI) @ FI accumulated onto WR@FR gives WR@FR - WI@FI = new_W.real  # noqa: E501
                                    for ki in Tx.unroll(N1 // MMA_K):
                                        Tx.ptx.tcgen05.encode_matrix_descriptor(
                                            Tx.address_of(descA), work_B.ptr_to([0, ki * MMA_K]),  # noqa: F821
                                            MMA_LDO, MMA_SDO, SWIZZLE)
                                        Tx.ptx.tcgen05.encode_matrix_descriptor(
                                            Tx.address_of(descB), F_imag_s.ptr_to([ki * MMA_K, 0]),  # noqa: F821
                                            MMA_LDO, MMA_SDO, SWIZZLE)
                                        Tx.ptx.tcgen05.mma("float32", "float16", "float16",
                                            Tx.cuda.get_tmem_addr(tmem_addr, 0, 0),
                                            descA, descB, descI_tb, False, CTA_GROUP, True)  # noqa: F821
                                    mma2consumer_bar.arrive(0, CTA_GROUP, 1)

                                    # Wait consumer done with Phase 3 (save W.real, flip work_B)
                                    consumer2mma_bar.wait(0, phase_c2m)
                                    Tx.ptx.tcgen05.fence.after_thread_sync()

                                    # == Phase 4: WR@FI + WI@FR -> new W.imag (cross-buffer accum) ==  # noqa: E501
                                    # work_A still has W'.real, work_B now has +W'.imag
                                    # First half: work_A @ F_imag (4 iters, init)
                                    for ki in Tx.unroll(N1 // MMA_K):
                                        Tx.ptx.tcgen05.encode_matrix_descriptor(
                                            Tx.address_of(descA), work_A.ptr_to([0, ki * MMA_K]),  # noqa: F821
                                            MMA_LDO, MMA_SDO, SWIZZLE)
                                        Tx.ptx.tcgen05.encode_matrix_descriptor(
                                            Tx.address_of(descB), F_imag_s.ptr_to([ki * MMA_K, 0]),  # noqa: F821
                                            MMA_LDO, MMA_SDO, SWIZZLE)
                                        Tx.ptx.tcgen05.mma("float32", "float16", "float16",
                                            Tx.cuda.get_tmem_addr(tmem_addr, 0, 0),
                                            descA, descB, descI_tb, False, CTA_GROUP, ki > 0)  # noqa: F821
                                    # Second half: work_B @ neg_F_real (4 iters, accumulate)
                                    # work_B = -WI, neg_FR = -FR -> (-WI)@(-FR) = WI@FR
                                    # WR@FI + WI@FR = new_W.imag (no sign flip needed!)
                                    for ki in Tx.unroll(N1 // MMA_K):
                                        Tx.ptx.tcgen05.encode_matrix_descriptor(
                                            Tx.address_of(descA), work_B.ptr_to([0, ki * MMA_K]),  # noqa: F821
                                            MMA_LDO, MMA_SDO, SWIZZLE)
                                        Tx.ptx.tcgen05.encode_matrix_descriptor(
                                            Tx.address_of(descB), neg_F_real_s.ptr_to([ki * MMA_K, 0]),  # noqa: E501, F821
                                            MMA_LDO, MMA_SDO, SWIZZLE)
                                        Tx.ptx.tcgen05.mma("float32", "float16", "float16",
                                            Tx.cuda.get_tmem_addr(tmem_addr, 0, 0),
                                            descA, descB, descI_tb, False, CTA_GROUP, True)  # noqa: F821
                                    mma2consumer_bar.arrive(0, CTA_GROUP, 1)

                                    # Wait consumer done with Phase 4 (kf multiply, write work_A/B)
                                    consumer2mma_bar.wait(0, phase_c2m ^ 1)
                                    Tx.ptx.tcgen05.fence.after_thread_sync()

                                    # == Phase 5: WR@FiR + (-WI)@FiI -> new W.real ==
                                    for ki in Tx.unroll(N1 // MMA_K):
                                        Tx.ptx.tcgen05.encode_matrix_descriptor(
                                            Tx.address_of(descA), work_A.ptr_to([0, ki * MMA_K]),  # noqa: F821
                                            MMA_LDO, MMA_SDO, SWIZZLE)
                                        Tx.ptx.tcgen05.encode_matrix_descriptor(
                                            Tx.address_of(descB), Finv_real_s.ptr_to([ki * MMA_K, 0]),  # noqa: E501, F821
                                            MMA_LDO, MMA_SDO, SWIZZLE)
                                        Tx.ptx.tcgen05.mma("float32", "float16", "float16",
                                            Tx.cuda.get_tmem_addr(tmem_addr, 0, 0),
                                            descA, descB, descI_tb, False, CTA_GROUP, ki > 0)  # noqa: F821
                                    for ki in Tx.unroll(N1 // MMA_K):
                                        Tx.ptx.tcgen05.encode_matrix_descriptor(
                                            Tx.address_of(descA), work_B.ptr_to([0, ki * MMA_K]),  # noqa: F821
                                            MMA_LDO, MMA_SDO, SWIZZLE)
                                        Tx.ptx.tcgen05.encode_matrix_descriptor(
                                            Tx.address_of(descB), Finv_imag_s.ptr_to([ki * MMA_K, 0]),  # noqa: E501, F821
                                            MMA_LDO, MMA_SDO, SWIZZLE)
                                        Tx.ptx.tcgen05.mma("float32", "float16", "float16",
                                            Tx.cuda.get_tmem_addr(tmem_addr, 0, 0),
                                            descA, descB, descI_tb, False, CTA_GROUP, True)  # noqa: F821
                                    mma2consumer_bar.arrive(0, CTA_GROUP, 1)

                                    # Wait consumer done with Phase 5
                                    consumer2mma_bar.wait(0, phase_c2m)
                                    Tx.ptx.tcgen05.fence.after_thread_sync()

                                    # == Phase 6: WR@FiI + WI@FiR -> new W.imag ==
                                    for ki in Tx.unroll(N1 // MMA_K):
                                        Tx.ptx.tcgen05.encode_matrix_descriptor(
                                            Tx.address_of(descA), work_A.ptr_to([0, ki * MMA_K]),  # noqa: F821
                                            MMA_LDO, MMA_SDO, SWIZZLE)
                                        Tx.ptx.tcgen05.encode_matrix_descriptor(
                                            Tx.address_of(descB), Finv_imag_s.ptr_to([ki * MMA_K, 0]),  # noqa: E501, F821
                                            MMA_LDO, MMA_SDO, SWIZZLE)
                                        Tx.ptx.tcgen05.mma("float32", "float16", "float16",
                                            Tx.cuda.get_tmem_addr(tmem_addr, 0, 0),
                                            descA, descB, descI_tb, False, CTA_GROUP, ki > 0)  # noqa: F821
                                    # Second half: work_B @ neg_Finv_real (4 iters, accumulate)
                                    # work_B = -WI, neg_FiR = -FiR -> (-WI)@(-FiR) = WI@FiR
                                    # WR@FiI + WI@FiR = new_W.imag (no sign flip needed!)
                                    for ki in Tx.unroll(N1 // MMA_K):
                                        Tx.ptx.tcgen05.encode_matrix_descriptor(
                                            Tx.address_of(descA), work_B.ptr_to([0, ki * MMA_K]),  # noqa: F821
                                            MMA_LDO, MMA_SDO, SWIZZLE)
                                        Tx.ptx.tcgen05.encode_matrix_descriptor(
                                            Tx.address_of(descB), neg_Finv_real_s.ptr_to([ki * MMA_K, 0]),  # noqa: E501, F821
                                            MMA_LDO, MMA_SDO, SWIZZLE)
                                        Tx.ptx.tcgen05.mma("float32", "float16", "float16",
                                            Tx.cuda.get_tmem_addr(tmem_addr, 0, 0),
                                            descA, descB, descI_tb, False, CTA_GROUP, True)  # noqa: F821
                                    mma2consumer_bar.arrive(0, CTA_GROUP, 1)

                                    # Wait consumer done with Phase 6 (twinv multiply, write work_A/B)  # noqa: E501
                                    consumer2mma_bar.wait(0, phase_c2m ^ 1)
                                    Tx.ptx.tcgen05.fence.after_thread_sync()

                                    # == Phase 7: FiR@WR + FiI@(-WI) -> final W.real ==
                                    # Finv on LEFT: A=Finv, B=work
                                    # work_A has W'.real, work_B has -W'.imag
                                    for ki in Tx.unroll(N1 // MMA_K):
                                        Tx.ptx.tcgen05.encode_matrix_descriptor(
                                            Tx.address_of(descA), Finv_real_s.ptr_to([0, ki * MMA_K]),  # noqa: E501, F821
                                            MMA_LDO, MMA_SDO, SWIZZLE)
                                        Tx.ptx.tcgen05.encode_matrix_descriptor(
                                            Tx.address_of(descB), work_A.ptr_to([ki * MMA_K, 0]),  # noqa: F821
                                            MMA_LDO, MMA_SDO, SWIZZLE)
                                        Tx.ptx.tcgen05.mma("float32", "float16", "float16",
                                            Tx.cuda.get_tmem_addr(tmem_addr, 0, 0),
                                            descA, descB, descI_tb, False, CTA_GROUP, ki > 0)  # noqa: F821
                                    # FiI @ (-WI): Finv_imag @ work_B (work_B = -W'.imag)
                                    # FiI @ (-WI) accumulated = FiR@WR - FiI@WI = real part
                                    for ki in Tx.unroll(N1 // MMA_K):
                                        Tx.ptx.tcgen05.encode_matrix_descriptor(
                                            Tx.address_of(descA), Finv_imag_s.ptr_to([0, ki * MMA_K]),  # noqa: E501, F821
                                            MMA_LDO, MMA_SDO, SWIZZLE)
                                        Tx.ptx.tcgen05.encode_matrix_descriptor(
                                            Tx.address_of(descB), work_B.ptr_to([ki * MMA_K, 0]),  # noqa: F821
                                            MMA_LDO, MMA_SDO, SWIZZLE)
                                        Tx.ptx.tcgen05.mma("float32", "float16", "float16",
                                            Tx.cuda.get_tmem_addr(tmem_addr, 0, 0),
                                            descA, descB, descI_tb, False, CTA_GROUP, True)  # noqa: F821
                                    mma2consumer_bar.arrive(0, CTA_GROUP, 1)

                                    # Wait consumer done with Phase 7 (TMA store)
                                    consumer2mma_bar.wait(0, phase_c2m)
                                    Tx.ptx.tcgen05.fence.after_thread_sync()
                                    # 7 phases = odd -> flip phase_c2m each work item
                                    phase_c2m = phase_c2m ^ 1
                                phase[0] = phase[0] ^ 1
                                with Tx.thread()[Tx.ptx.elect_sync()]:
                                    Tx.ptx.mbarrier.arrive(workitem_sync_bar.ptr_to([0]))
                                wid_mma = wid_mma + SM_COUNT

                    # === Consumer warpgroup (WG0) ===
                    with Tx.warpgroup()[0:1]:
                        Tx.cuda.trap_when_assert_failed(tmem_addr == 0)

                        phase_m2c: Tx.int32
                        phase_m2c = 0
                        phase_c2m_c: Tx.int32
                        phase_c2m_c = 0

                        # Register buffers for w_real/w_imag (replaces SMEM f32 buffers)
                        # Each active thread (lane_id < 16) owns one row of 64 f32 values
                        w_real_reg = Tx.alloc_buffer((N1,), "float32", scope="local")
                        w_imag_reg = Tx.alloc_buffer((N1,), "float32", scope="local")

                        # Reusable f16 register buffers for preloaded twiddle constants
                        # Preloaded during light phases (1/3/5), consumed in heavy phases (2/4/6)
                        const_real_reg = Tx.alloc_buffer((N1,), "float16", scope="local")
                        const_imag_reg = Tx.alloc_buffer((N1,), "float16", scope="local")

                        # Shuffle buffers for write splitting (must be real buffers, not meta_var)
                        shuf_real_buf = Tx.alloc_buffer((32,), "float32", scope="local")
                        shuf_imag_buf = Tx.alloc_buffer((32,), "float32", scope="local")

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

                            # ======== Phase 1: Read W.real from TMEM -> registers ========
                            # Also preload tw constants into registers for Phase 2
                            mma2consumer_bar.wait(0, phase_m2c)
                            phase_m2c = phase_m2c ^ 1
                            Tx.ptx.tcgen05.fence.after_thread_sync()

                            Tx.copy(reg_wg[:, :], tmem[:, 0:TMEM_LD_SIZE])

                            with Tx.thread():
                                out_row = Tx.meta_var(warp_id * 16 + lane_id)
                                if lane_id < 16:
                                    for j in Tx.serial(N1):
                                        w_real_reg[j] = reg[j]
                                    # Preload tw constants for Phase 2
                                    for j in Tx.serial(N1):
                                        const_real_reg[j] = tw_real_s[out_row, j]
                                        const_imag_reg[j] = tw_imag_s[out_row, j]

                            Tx.ptx.tcgen05.fence.before_thread_sync()
                            consumer2mma_bar.arrive(0)
                            phase_c2m_c = phase_c2m_c ^ 1

                            # ======== Phase 2: Read W.imag from TMEM -> registers ========
                            # Then apply tw multiply and write work_A/work_B (all in registers, no sync needed)  # noqa: E501
                            mma2consumer_bar.wait(0, phase_m2c)
                            phase_m2c = phase_m2c ^ 1
                            Tx.ptx.tcgen05.fence.after_thread_sync()

                            Tx.copy(reg_wg[:, :], tmem[:, 0:TMEM_LD_SIZE])

                            with Tx.thread():
                                out_row = Tx.meta_var(warp_id * 16 + (lane_id % 16))
                                if lane_id < 16:
                                    # Save W.imag to registers, then immediately apply tw multiply
                                    for j in Tx.serial(N1):
                                        w_imag_reg[j] = reg[j]
                                    # tw multiply: compute new_imag first (uses old w_real), then new_real  # noqa: E501
                                    for j in Tx.serial(N1):
                                        tr = Tx.meta_var(Tx.cast(const_real_reg[j], "float32"))
                                        ti = Tx.meta_var(Tx.cast(const_imag_reg[j], "float32"))
                                        # new_imag = wr*ti + wi*tr (written first, reads old w_real)
                                        shuf_real_buf[j % 32] = w_real_reg[j] * ti + w_imag_reg[j] * tr  # noqa: E501
                                        # new_real = wr*tr - wi*ti (reads old w_real before overwrite)  # noqa: E501
                                        w_real_reg[j] = w_real_reg[j] * tr - w_imag_reg[j] * ti
                                        w_imag_reg[j] = shuf_real_buf[j % 32]
                                    # Lanes 0-15 write columns 0-31
                                    for j in Tx.serial(32):
                                        work_A[out_row, j] = Tx.cast(w_real_reg[j], "float16")
                                        work_B[out_row, j] = Tx.cast(0.0 - w_imag_reg[j], "float16")
                                # Shuffle columns 32-63 from lanes 0-15 to lanes 16-31
                                for j in Tx.serial(32):
                                    shuf_real_buf[j] = Tx.tvm_warp_shuffle_xor(0xFFFFFFFF, w_real_reg[32 + j], 16, 32, 32)  # noqa: E501
                                    shuf_imag_buf[j] = Tx.tvm_warp_shuffle_xor(0xFFFFFFFF, w_imag_reg[32 + j], 16, 32, 32)  # noqa: E501
                                if lane_id >= 16:
                                    for j in Tx.serial(32):
                                        work_A[out_row, 32 + j] = Tx.cast(shuf_real_buf[j], "float16")  # noqa: E501
                                        work_B[out_row, 32 + j] = Tx.cast(0.0 - shuf_imag_buf[j], "float16")  # noqa: E501

                            Tx.ptx.tcgen05.fence.before_thread_sync()
                            consumer2mma_bar.arrive(0)
                            phase_c2m_c = phase_c2m_c ^ 1

                            # ======== Phase 3: Read new W.real from TMEM -> registers ========
                            # Also preload kf constants into registers for Phase 4
                            mma2consumer_bar.wait(0, phase_m2c)
                            phase_m2c = phase_m2c ^ 1
                            Tx.ptx.tcgen05.fence.after_thread_sync()

                            Tx.copy(reg_wg[:, :], tmem[:, 0:TMEM_LD_SIZE])

                            with Tx.thread():
                                out_row = Tx.meta_var(warp_id * 16 + lane_id)
                                if lane_id < 16:
                                    for j in Tx.serial(N1):
                                        w_real_reg[j] = reg[j]
                                    # Preload kf constants for Phase 4
                                    for j in Tx.serial(N1):
                                        const_real_reg[j] = kf_real_s[out_row, j]
                                        const_imag_reg[j] = kf_imag_s[out_row, j]

                            Tx.ptx.tcgen05.fence.before_thread_sync()
                            consumer2mma_bar.arrive(0)
                            phase_c2m_c = phase_c2m_c ^ 1

                            # ======== Phase 4: Read new W.imag from TMEM -> registers ========
                            # Then apply kf multiply and write work_A/work_B (all in registers)
                            mma2consumer_bar.wait(0, phase_m2c)
                            phase_m2c = phase_m2c ^ 1
                            Tx.ptx.tcgen05.fence.after_thread_sync()

                            Tx.copy(reg_wg[:, :], tmem[:, 0:TMEM_LD_SIZE])

                            with Tx.thread():
                                out_row = Tx.meta_var(warp_id * 16 + (lane_id % 16))
                                if lane_id < 16:
                                    # Save W.imag to registers, then immediately apply kf multiply
                                    for j in Tx.serial(N1):
                                        w_imag_reg[j] = reg[j]
                                    # kf multiply: compute new_imag first (uses old w_real), then new_real  # noqa: E501
                                    for j in Tx.serial(N1):
                                        kr = Tx.meta_var(Tx.cast(const_real_reg[j], "float32"))
                                        ki_val = Tx.meta_var(Tx.cast(const_imag_reg[j], "float32"))
                                        shuf_real_buf[j % 32] = w_real_reg[j] * ki_val + w_imag_reg[j] * kr  # noqa: E501
                                        w_real_reg[j] = w_real_reg[j] * kr - w_imag_reg[j] * ki_val
                                        w_imag_reg[j] = shuf_real_buf[j % 32]
                                    # Lanes 0-15 write columns 0-31
                                    for j in Tx.serial(32):
                                        work_A[out_row, j] = Tx.cast(w_real_reg[j], "float16")
                                        work_B[out_row, j] = Tx.cast(0.0 - w_imag_reg[j], "float16")
                                # Shuffle columns 32-63 from lanes 0-15 to lanes 16-31
                                for j in Tx.serial(32):
                                    shuf_real_buf[j] = Tx.tvm_warp_shuffle_xor(0xFFFFFFFF, w_real_reg[32 + j], 16, 32, 32)  # noqa: E501
                                    shuf_imag_buf[j] = Tx.tvm_warp_shuffle_xor(0xFFFFFFFF, w_imag_reg[32 + j], 16, 32, 32)  # noqa: E501
                                if lane_id >= 16:
                                    for j in Tx.serial(32):
                                        work_A[out_row, 32 + j] = Tx.cast(shuf_real_buf[j], "float16")  # noqa: E501
                                        work_B[out_row, 32 + j] = Tx.cast(0.0 - shuf_imag_buf[j], "float16")  # noqa: E501

                            Tx.ptx.tcgen05.fence.before_thread_sync()
                            consumer2mma_bar.arrive(0)
                            phase_c2m_c = phase_c2m_c ^ 1

                            # ======== Phase 5: Read new W.real from TMEM -> registers ========
                            # Also preload twinv constants into registers for Phase 6
                            mma2consumer_bar.wait(0, phase_m2c)
                            phase_m2c = phase_m2c ^ 1
                            Tx.ptx.tcgen05.fence.after_thread_sync()

                            Tx.copy(reg_wg[:, :], tmem[:, 0:TMEM_LD_SIZE])

                            with Tx.thread():
                                out_row = Tx.meta_var(warp_id * 16 + lane_id)
                                if lane_id < 16:
                                    for j in Tx.serial(N1):
                                        w_real_reg[j] = reg[j]
                                    # Preload twinv constants for Phase 6
                                    for j in Tx.serial(N1):
                                        const_real_reg[j] = twinv_real_s[out_row, j]
                                        const_imag_reg[j] = twinv_imag_s[out_row, j]

                            Tx.ptx.tcgen05.fence.before_thread_sync()
                            consumer2mma_bar.arrive(0)
                            phase_c2m_c = phase_c2m_c ^ 1

                            # ======== Phase 6: Read new W.imag from TMEM -> registers ========
                            # Then apply twinv multiply and write work_A/work_B (all in registers)
                            mma2consumer_bar.wait(0, phase_m2c)
                            phase_m2c = phase_m2c ^ 1
                            Tx.ptx.tcgen05.fence.after_thread_sync()

                            Tx.copy(reg_wg[:, :], tmem[:, 0:TMEM_LD_SIZE])

                            with Tx.thread():
                                out_row = Tx.meta_var(warp_id * 16 + (lane_id % 16))
                                if lane_id < 16:
                                    # Save W.imag to registers, then immediately apply twinv multiply  # noqa: E501
                                    for j in Tx.serial(N1):
                                        w_imag_reg[j] = reg[j]
                                    # twinv multiply: compute new_imag first (uses old w_real), then new_real  # noqa: E501
                                    for j in Tx.serial(N1):
                                        tr = Tx.meta_var(Tx.cast(const_real_reg[j], "float32"))
                                        ti = Tx.meta_var(Tx.cast(const_imag_reg[j], "float32"))
                                        shuf_real_buf[j % 32] = w_real_reg[j] * ti + w_imag_reg[j] * tr  # noqa: E501
                                        w_real_reg[j] = w_real_reg[j] * tr - w_imag_reg[j] * ti
                                        w_imag_reg[j] = shuf_real_buf[j % 32]
                                    # Lanes 0-15 write columns 0-31
                                    for j in Tx.serial(32):
                                        work_A[out_row, j] = Tx.cast(w_real_reg[j], "float16")
                                        work_B[out_row, j] = Tx.cast(0.0 - w_imag_reg[j], "float16")
                                # Shuffle columns 32-63 from lanes 0-15 to lanes 16-31
                                for j in Tx.serial(32):
                                    shuf_real_buf[j] = Tx.tvm_warp_shuffle_xor(0xFFFFFFFF, w_real_reg[32 + j], 16, 32, 32)  # noqa: E501
                                    shuf_imag_buf[j] = Tx.tvm_warp_shuffle_xor(0xFFFFFFFF, w_imag_reg[32 + j], 16, 32, 32)  # noqa: E501
                                if lane_id >= 16:
                                    for j in Tx.serial(32):
                                        work_A[out_row, 32 + j] = Tx.cast(shuf_real_buf[j], "float16")  # noqa: E501
                                        work_B[out_row, 32 + j] = Tx.cast(0.0 - shuf_imag_buf[j], "float16")  # noqa: E501

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
                                out_row = Tx.meta_var(warp_id * 16 + (lane_id % 16))
                                if lane_id < 16:
                                    for j in Tx.serial(32):
                                        D_out[out_row, j] = Tx.cast(reg[j], "float16")
                                # Shuffle columns 32-63 from lanes 0-15 to lanes 16-31
                                for j in Tx.serial(32):
                                    shuf_real_buf[j] = Tx.tvm_warp_shuffle_xor(0xFFFFFFFF, reg[32 + j], 16, 32, 32)  # noqa: E501
                                if lane_id >= 16:
                                    for j in Tx.serial(32):
                                        D_out[out_row, 32 + j] = Tx.cast(shuf_real_buf[j], "float16")  # noqa: E501

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
    (
        u_f16,
        kf_real,
        kf_imag,
        f_real,
        f_imag,
        finv_real,
        finv_imag,
        tw_real,
        tw_imag,
        twinv_real,
        twinv_imag,
        u_orig,
        k_orig,
    ) = prepare_data(batch, heads)

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

    mod(
        x_tvm,
        kf_real_tvm,
        kf_imag_tvm,
        f_real_tvm,
        f_imag_tvm,
        finv_real_tvm,
        finv_imag_tvm,
        tw_real_tvm,
        tw_imag_tvm,
        twinv_real_tvm,
        twinv_imag_tvm,
        o_tvm,
    )

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

    print(f"\n{'=' * 60}")
    print(f"FFTConv Benchmark (N={N}={N1}x{N1})")
    print(f"{'=' * 60}")

    with ProtonContext("fftconv"):
        for batch, heads in [(1, 128), (2, 64), (4, 32)]:
            (
                u_f16,
                kf_real,
                kf_imag,
                f_real,
                f_imag,
                finv_real,
                finv_imag,
                tw_real,
                tw_imag,
                twinv_real,
                twinv_imag,
                _,
                _,
            ) = prepare_data(batch, heads)

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

            func = lambda: mod(  # noqa: E731
                x_tvm,
                kf_real_tvm,
                kf_imag_tvm,
                f_real_tvm,
                f_imag_tvm,
                finv_real_tvm,
                finv_imag_tvm,
                tw_real_tvm,
                tw_imag_tvm,
                twinv_real_tvm,
                twinv_imag_tvm,
                o_tvm,
            )
            ms = bench(func, warmup=100, repeat=300, proton_name=f"fftconv_B{batch}_H{heads}")
            tflops = flops(ms, batch, heads) / 1e12
            print(f"  B={batch:>2d}, H={heads:>4d}: {tflops:.2f} TFLOPS, {ms:.3f} ms")

    print()


if __name__ == "__main__":
    test_fftconv(2, 4)
    bench_fftconv()
