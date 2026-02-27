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

"""Flux fused GEMM kernels on Blackwell (SM100a).

flux_gelu: out[i,j] = GELU(X @ W^T + bias[j])
flux_gate: out[i,j] = (X @ W^T + bias[j]) * gate[j] + Y[i,j]

Transcribed from ThunderKittens flux kernels to TIRX.
"""

import numpy as np

import tvm
import tvm.testing
from tvm.ir.type import PointerType, PrimType
from tvm.script import tirx as Tx
from tvm.tir.layout import TileLayout, TLane, TCol, tid_in_wg, S
from tvm.tirx.bench.utils import ProtonContext, bench
from tvm.tirx.tile_scheduler import ClusterPersistentScheduler2D

# cluster: [2, 1], cta_num = 2
# warpgroup:
#   wg_id = 0,1: ld & to GMEM
#   wg_id = 2:
#       warp_id = 0,1: mma
#       warp_id = 3: tma

M_CLUSTER = 2
N_CLUSTER = 1
WG_NUMBER = 3
WARP_NUMBER = 4
NUM_CONSUMER = 2
NUM_THREADS = (32 * WARP_NUMBER) * WG_NUMBER
SM_NUMBER = 148

PIPELINE_DEPTH = 4

F16_BYTES = 2
F32_BYTES = 4
F128_BYTES = 16

a_type = tvm.DataType("float16")
b_type = tvm.DataType("float16")
d_type = tvm.DataType("float16")
M, N, K = 3072, 3072, 3072
BLK_M, BLK_N, BLK_K = 128, 128, 64
MMA_M, MMA_N, MMA_K = 256, 256, 16
EPI_TILE = 64
QUANT_SIZE = BLK_K
SWIZZLE = 3
SMEM_SIZE = (
    PIPELINE_DEPTH * NUM_CONSUMER * BLK_M * BLK_K * F16_BYTES
    + PIPELINE_DEPTH * BLK_N * BLK_K * F16_BYTES
    + NUM_CONSUMER * BLK_M * EPI_TILE * F16_BYTES
    + 1024
)
assert SMEM_SIZE <= 232448

TMEM_LD_SIZE = 64
N_COLS = 512
CTA_GROUP = 2

PIPE_CYCLE = (K // BLK_K) // PIPELINE_DEPTH
PIPE_REMAIN_NUM = (K // BLK_K) % PIPELINE_DEPTH
assert PIPELINE_DEPTH == 4

DEBUG = False

TILE_GROUPS_ROW_SIZE = 8
assert M % (NUM_CONSUMER * BLK_M * CTA_GROUP) == 0
assert N % (BLK_N * CTA_GROUP) == 0
TILE_M_NUM = M // (NUM_CONSUMER * BLK_M * CTA_GROUP)
TILE_N_NUM = N // (BLK_N * CTA_GROUP)


@Tx.inline
def skip():
    pass


def get_source(func):
    target = tvm.target.Target("cuda")
    mod = tvm.IRModule({"main": func})
    mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
    src = mod.mod.imports[0].inspect_source()
    return src, mod


def ceildiv(a, b):
    return (a + b - 1) // b


def flops(ms):
    return M * N * K * 2 / (ms * 1e-3)


@Tx.meta_class
class Barriers:
    def __init__(self, shared_buffer_base, shared_buffer_offs, pipe_depth, pipe_width, is_p2c):
        self.mbar = Tx.decl_buffer(
            (pipe_depth, pipe_width), "uint64", shared_buffer_base, elem_offset=shared_buffer_offs
        )
        self.init_phase = 0 if is_p2c else 1
        self.pipe_depth = pipe_depth
        self.pipe_width = pipe_width

    @Tx.inline
    def init(self, threads_num_wait):
        with Tx.thread()[0:1]:
            for i in Tx.serial(self.pipe_depth):
                for j in Tx.serial(self.pipe_width):
                    Tx.ptx.mbarrier.init(self.mbar.ptr_to([i, j]), threads_num_wait)

    @Tx.inline
    def wait(self, idx_d, idx_w, phase):
        Tx.ptx.mbarrier.try_wait(self.mbar.ptr_to([idx_d, idx_w]), self.init_phase ^ phase)


class BarTMA2MMA(Barriers):
    @Tx.inline
    def arrive(self, idx, expected_bytes):
        Tx.ptx.mbarrier.arrive.expect_tx(self.mbar.ptr_to([idx, 0]), expected_bytes)

    @Tx.inline
    def arrive_only(self, idx):
        Tx.ptx.mbarrier.arrive(self.mbar.ptr_to([idx, 0]))


class BarMMA2LD(Barriers):
    @Tx.inline
    def arrive(self, idx):
        Tx.ptx.tcgen05.commit(self.mbar.ptr_to([0, idx]), cta_group=CTA_GROUP, cta_mask=3)


class BarMMA2TMA(Barriers):
    @Tx.inline
    def arrive(self, idx):
        Tx.ptx.tcgen05.commit(self.mbar.ptr_to([idx, 0]), cta_group=CTA_GROUP, cta_mask=3)


class BarLD2MMA(Barriers):
    @Tx.inline
    def arrive(self, idx):
        Tx.ptx.mbarrier.arrive(self.mbar.ptr_to([0, idx]), cta_id=0, pred=True)


# ============================================================
# flux_gelu: out = GELU(X @ W^T + bias)
# ============================================================

@tvm.testing.requires_cuda_compute_version(10, exact=True)
def test_flux_gelu():
    A_layout = Tx.ComposeLayout(
        Tx.SwizzleLayout(3, 3, 3, swizzle_inner=True),
        Tx.TileLayout(
            Tx.S[
                (PIPELINE_DEPTH, NUM_CONSUMER, BLK_M, BLK_K) : (
                    NUM_CONSUMER * BLK_M * BLK_K,
                    BLK_M * BLK_K,
                    BLK_K,
                    1,
                )
            ]
        ),
    )
    B_layout = Tx.ComposeLayout(
        Tx.SwizzleLayout(3, 3, 3, swizzle_inner=True),
        Tx.TileLayout(Tx.S[(PIPELINE_DEPTH, BLK_N, BLK_K) : (BLK_N * BLK_K, BLK_K, 1)]),
    )
    D_layout = Tx.ComposeLayout(
        Tx.SwizzleLayout(3, 3, 3, swizzle_inner=True),
        Tx.TileLayout(Tx.S[(NUM_CONSUMER, BLK_M, EPI_TILE) : (BLK_M * EPI_TILE, EPI_TILE, 1)]),
    )

    @Tx.prim_func(tirx=True)
    def flux_gelu_kernel(
        A: Tx.Buffer((M, K), a_type),
        B: Tx.Buffer((N, K), b_type),
        bias: Tx.Buffer((N,), a_type),
        D: Tx.Buffer((M, N), d_type),
    ):
        # fmt: off
        with Tx.kernel():
            cbx, cby = Tx.cta_id([M_CLUSTER, N_CLUSTER], parent="cluster")
            bx = Tx.cta_id([SM_NUMBER], parent="kernel")
            wg_id = Tx.warpgroup_id([WG_NUMBER], parent="cta")
            warp_id = Tx.warp_id([WARP_NUMBER], parent="warpgroup")
            lane_id = Tx.thread_id([32], parent="warp")
            with Tx.cta():
                buf = Tx.alloc_buffer([SMEM_SIZE], "uint8", scope="shared.dyn")
                tmem_addr = Tx.decl_scalar("uint32", buf.data, scope="shared.dyn", elem_offset=0)
                A_smem = Tx.decl_buffer((PIPELINE_DEPTH, NUM_CONSUMER, BLK_M, BLK_K), a_type, buf.data, layout=A_layout,
                                        elem_offset=1024 // F16_BYTES)
                B_smem = Tx.decl_buffer((PIPELINE_DEPTH, BLK_N, BLK_K), b_type, buf.data, layout=B_layout,
                                        elem_offset=1024 // F16_BYTES + PIPELINE_DEPTH * NUM_CONSUMER * BLK_M * BLK_K)
                D_smem = Tx.decl_buffer((NUM_CONSUMER, BLK_M, EPI_TILE), d_type, buf.data, layout=D_layout,
                                        elem_offset=1024 // F16_BYTES + PIPELINE_DEPTH * (NUM_CONSUMER * BLK_M + BLK_N) * BLK_K)

                descI: Tx.uint32
                descA: Tx.uint64
                descB: Tx.uint64
                phase = Tx.alloc_buffer((1,), "int32", scope="local")
                stage: Tx.int32

                tma2mma = BarTMA2MMA(buf.data, 4, PIPELINE_DEPTH, 1, is_p2c=True)
                mma2tma = BarMMA2TMA(buf.data, 4 + PIPELINE_DEPTH, PIPELINE_DEPTH, 1, is_p2c=False)
                mma2ld = BarMMA2LD(buf.data, 4 + 2 * PIPELINE_DEPTH, 1, NUM_CONSUMER, is_p2c=True)
                ld2mma = BarLD2MMA(buf.data, 4 + 2 * PIPELINE_DEPTH + NUM_CONSUMER, 1, NUM_CONSUMER, is_p2c=False)

                tma2mma.init(1)
                mma2tma.init(NUM_CONSUMER)
                mma2ld.init(1)
                ld2mma.init(128 * NUM_CONSUMER)

                ptr: Tx.let[Tx.Var(name="ptr", dtype=PointerType(PrimType("uint64")))] = Tx.reinterpret("handle", Tx.ptx.map_shared_rank(tma2mma.mbar.ptr_to([0, 0]), 0))
                tma_finished = Tx.decl_buffer([PIPELINE_DEPTH], "uint64", data=ptr, scope="shared")

                tile_scheduler = ClusterPersistentScheduler2D("tile_scheduler", num_m_tiles=TILE_M_NUM, num_n_tiles=TILE_N_NUM, num_clusters=SM_NUMBER // 2, l2_group_size=TILE_GROUPS_ROW_SIZE)
                tile_scheduler.init(bx // 2)
                m_idx = Tx.meta_var(tile_scheduler.m_idx)
                n_idx = Tx.meta_var(tile_scheduler.n_idx)

                with Tx.warp()[0:1]:
                    Tx.ptx.tcgen05.alloc(Tx.address_of(tmem_addr), n_cols=N_COLS, cta_group=CTA_GROUP)

                Tx.cuda.cluster_sync()
                Tx.cuda.cta_sync()
                Tx.ptx.fence.proxy("shared")
                Tx.ptx.fence.mbarrier_init()
                Tx.cuda.trap_when_assert_failed(tmem_addr == 0)
                tmem = Tx.decl_buffer((128, N_COLS), "float32", scope="tmem", allocated_addr=0, layout=TileLayout(S[(128, N_COLS) : (1@TLane, 1@TCol)]))

                @Tx.inline
                def paritioned_loop(main_loop, epilogue1, epilogue2):
                    for ko in Tx.serial(PIPE_CYCLE):
                        for ks in Tx.unroll(PIPELINE_DEPTH):
                            stage = ko * PIPELINE_DEPTH + ks
                            main_loop(False, ks)
                        phase[0] = phase[0] ^ 1
                    if PIPE_REMAIN_NUM > 0:
                        for ks in Tx.unroll(PIPE_REMAIN_NUM):
                            stage = PIPE_CYCLE * PIPELINE_DEPTH + ks
                            main_loop(True, ks)
                        epilogue1()
                        for ks in Tx.unroll(PIPE_REMAIN_NUM, PIPELINE_DEPTH):
                            epilogue2(ks)
                        phase[0] = phase[0] ^ 1
                    else:
                        epilogue1()

                with Tx.cta():
                    Tx.attr({"tirx.scope_partition": True})
                    # === Producer warpgroup (WG2) ===
                    with Tx.warpgroup()[NUM_CONSUMER:NUM_CONSUMER + 1]:
                        Tx.ptx.setmaxnreg(False, 56)
                        if warp_id == 3:
                            phase[0] = 0
                            while tile_scheduler.valid():
                                with Tx.thread(parent="warp")[Tx.ptx.elect_sync()]:
                                    m_start0 = Tx.meta_var((m_idx * NUM_CONSUMER * CTA_GROUP + cbx) * BLK_M)
                                    m_start1 = Tx.meta_var((m_idx * NUM_CONSUMER * CTA_GROUP + CTA_GROUP + cbx) * BLK_M)
                                    n_start = Tx.meta_var((n_idx * CTA_GROUP + cbx) * BLK_N)
                                    k_start = Tx.meta_var(stage * BLK_K)

                                    @Tx.inline
                                    def tma_load(is_remain, ks):
                                        tma_copy = Tx.meta_var({"dispatch": "tma", "mbar": tma_finished.ptr_to([ks]), "cta_group": CTA_GROUP})
                                        mma2tma.wait(ks, 0, phase[0])
                                        Tx.copy_async(A_smem[ks, 0, :, :], A[m_start0 : m_start0 + BLK_M, k_start : k_start + BLK_K], **tma_copy)
                                        Tx.copy_async(A_smem[ks, 1, :, :], A[m_start1 : m_start1 + BLK_M, k_start : k_start + BLK_K], **tma_copy)
                                        Tx.copy_async(B_smem[ks, :, :], B[n_start : n_start + BLK_N, k_start : k_start + BLK_K], **tma_copy)
                                        if cbx == 0:
                                            tma2mma.arrive(ks, NUM_CONSUMER * BLK_K * (BLK_M * NUM_CONSUMER + BLK_N) * F16_BYTES)

                                    @Tx.inline
                                    def tma_load_epilogue(ks):
                                        mma2tma.wait(ks, 0, phase[0])
                                        if cbx == 0:
                                            tma2mma.arrive_only(ks)

                                    paritioned_loop(tma_load, skip, tma_load_epilogue)
                                tile_scheduler.next_tile()

                        elif warp_id < 2 and cbx == 0:
                            phase_tmem: Tx.int32
                            phase_tmem = 0
                            phase[0] = 0

                            Tx.ptx.tcgen05.encode_instr_descriptor(Tx.address_of(descI), "float32", a_type, b_type, MMA_M, MMA_N, MMA_K, False, False, CTA_GROUP)

                            while tile_scheduler.valid():
                                m_idx = Tx.meta_var(tile_scheduler.m_idx)
                                n_idx = Tx.meta_var(tile_scheduler.n_idx)
                                with Tx.thread()[Tx.ptx.elect_sync()]:
                                    ld2mma.wait(0, warp_id, phase_tmem)
                                    Tx.ptx.tcgen05.fence.after_thread_sync()

                                    @Tx.inline
                                    def mma(is_remain, ks):
                                        tma2mma.wait(ks, 0, phase[0])
                                        Tx.gemm_async(tmem[:, warp_id * MMA_N: warp_id * MMA_N + MMA_N], A_smem[ks, warp_id, :, :], B_smem[ks, :, :], dispatch="tcgen05", cta_group=CTA_GROUP, descI=descI,
                                                        accum=tvm.tir.Not(stage == 0 and ((not is_remain) or (is_remain and PIPE_CYCLE == 0))))
                                        mma2tma.arrive(ks)

                                    @Tx.inline
                                    def mma_epilogue1():
                                        mma2ld.arrive(warp_id)

                                    @Tx.inline
                                    def mma_epilogue2(ks):
                                        tma2mma.wait(ks, 0, phase[0])
                                        mma2tma.arrive(ks)

                                    paritioned_loop(mma, mma_epilogue1, mma_epilogue2)
                                    phase_tmem = phase_tmem ^ 1
                                tile_scheduler.next_tile()

                    # === Consumer warpgroups (WG0, WG1) — with GELU epilogue ===
                    with Tx.warpgroup()[0:NUM_CONSUMER]:
                        Tx.ptx.setmaxnreg(True, 224)

                        reg = Tx.alloc_buffer((TMEM_LD_SIZE,), "float32", scope="local")
                        reg_wg = reg.view(128, TMEM_LD_SIZE, layout=TileLayout(S[(128, TMEM_LD_SIZE) : (1@tid_in_wg, 1)]))
                        reg_fp16 = Tx.alloc_buffer((BLK_N * CTA_GROUP,), d_type, scope="local")
                        phase_tmem: Tx.int32

                        phase_tmem = 0
                        while tile_scheduler.valid():
                            mma2ld.wait(0, wg_id, phase_tmem)
                            phase_tmem = phase_tmem ^ 1
                            Tx.ptx.tcgen05.fence.after_thread_sync()
                            # TMEM -> RF with bias + GELU
                            for i in Tx.unroll(MMA_N // TMEM_LD_SIZE):
                                col_st = Tx.meta_var(wg_id * MMA_N + i * TMEM_LD_SIZE)
                                Tx.copy(reg_wg[:, :], tmem[:, col_st : col_st + TMEM_LD_SIZE])
                                with Tx.thread():
                                    n_col = Tx.meta_var(n_idx * BLK_N * CTA_GROUP + i * TMEM_LD_SIZE)
                                    # Add bias
                                    for j in Tx.serial(TMEM_LD_SIZE):
                                        reg[j] = reg[j] + Tx.cast(bias[n_col + j], "float32")
                                    # GELU: x * sigmoid(x * (1.5957691 + 0.07106856 * x * x))
                                    for j in Tx.serial(TMEM_LD_SIZE):
                                        reg[j] = reg[j] * Tx.sigmoid(reg[j] * (1.5957691 + 0.07106856 * reg[j] * reg[j]))
                                    Tx.cast(reg_fp16[i * TMEM_LD_SIZE : (i + 1) * TMEM_LD_SIZE], reg[:])

                            ld2mma.arrive(wg_id)
                            # RF -> GMEM via SMEM + TMA
                            for i in Tx.unroll(NUM_CONSUMER * BLK_N // EPI_TILE):
                                with Tx.thread():
                                    Tx.copy(D_smem[wg_id, warp_id * 32 + lane_id, :], reg_fp16[i * EPI_TILE : (i + 1) * EPI_TILE])
                                Tx.cuda.warpgroup_sync(wg_id)
                                Tx.ptx.fence.proxy(scope="shared")
                                with Tx.thread()[lane_id == 0 and warp_id == 0]:
                                    m_start: Tx.let = (m_idx * NUM_CONSUMER * CTA_GROUP + wg_id * CTA_GROUP + cbx) * BLK_M
                                    n_start: Tx.let = n_idx * BLK_N * CTA_GROUP + i * EPI_TILE
                                    Tx.copy_async(D[m_start : m_start + BLK_M, n_start : n_start + EPI_TILE], D_smem[wg_id, :, :], dispatch="tma")
                                    Tx.ptx.cp_async.bulk.commit_group()
                                    Tx.ptx.cp_async.bulk.wait_group(0)
                                Tx.cuda.warpgroup_sync(wg_id)
                            tile_scheduler.next_tile()

                with Tx.warp()[0:1]:
                    Tx.ptx.tcgen05.relinquish_alloc_permit(cta_group=CTA_GROUP)
                    Tx.ptx.tcgen05.dealloc(tmem_addr, n_cols=N_COLS, cta_group=CTA_GROUP)

                Tx.cuda.cluster_sync()
        # fmt: on

    # --- Prepare data ---
    import torch

    A_data = torch.randn((M, K), dtype=torch.float16)
    B_data = torch.randn((N, K), dtype=torch.float16)
    bias_data = torch.randn((N,), dtype=torch.float16)

    def ref_flux_gelu(A_data, B_data, bias_data):
        A_f32 = A_data.float()
        B_f32 = B_data.float()
        bias_f32 = bias_data.float()
        C = A_f32 @ B_f32.T + bias_f32.unsqueeze(0)
        # GELU via sigmoid approximation: x * sigmoid(1.5957691*x + 0.07106856*x^3)
        C = C * torch.sigmoid(C * (1.5957691 + 0.07106856 * C * C))
        return C.half().numpy()

    def tir_flux_gelu(A_data, B_data, bias_data):
        DEV = tvm.cuda(0)
        A_tvm = tvm.runtime.tensor(A_data.numpy(), device=DEV)
        B_tvm = tvm.runtime.tensor(B_data.numpy(), device=DEV)
        bias_tvm = tvm.runtime.tensor(bias_data.numpy(), device=DEV)
        D_tvm = tvm.runtime.tensor(np.zeros((M, N), dtype="float16"), device=DEV)
        target = tvm.target.Target("cuda")
        with target:
            src, mod = get_source(flux_gelu_kernel)
            print(src)
            func = lambda: mod(A_tvm, B_tvm, bias_tvm, D_tvm)
            ms = bench(func, warmup=0, repeat=30, proton_name="tir_flux_gelu", debug=DEBUG)
            print(f"TIR flux_gelu flops: {flops(ms) / 1e12} TFLOPS, time: {ms:.3f} ms")
        return D_tvm.numpy()

    C_ref = ref_flux_gelu(A_data, B_data, bias_data)
    C_tir = tir_flux_gelu(A_data, B_data, bias_data)
    np.testing.assert_allclose(C_tir, C_ref, rtol=1e-2, atol=1e-1)
    print("flux_gelu: PASSED")


# ============================================================
# flux_gate: out = (X @ W^T + bias) * gate + Y
# ============================================================

@tvm.testing.requires_cuda_compute_version(10, exact=True)
def test_flux_gate():
    A_layout = Tx.ComposeLayout(
        Tx.SwizzleLayout(3, 3, 3, swizzle_inner=True),
        Tx.TileLayout(
            Tx.S[
                (PIPELINE_DEPTH, NUM_CONSUMER, BLK_M, BLK_K) : (
                    NUM_CONSUMER * BLK_M * BLK_K,
                    BLK_M * BLK_K,
                    BLK_K,
                    1,
                )
            ]
        ),
    )
    B_layout = Tx.ComposeLayout(
        Tx.SwizzleLayout(3, 3, 3, swizzle_inner=True),
        Tx.TileLayout(Tx.S[(PIPELINE_DEPTH, BLK_N, BLK_K) : (BLK_N * BLK_K, BLK_K, 1)]),
    )
    D_layout = Tx.ComposeLayout(
        Tx.SwizzleLayout(3, 3, 3, swizzle_inner=True),
        Tx.TileLayout(Tx.S[(NUM_CONSUMER, BLK_M, EPI_TILE) : (BLK_M * EPI_TILE, EPI_TILE, 1)]),
    )

    @Tx.prim_func(tirx=True)
    def flux_gate_kernel(
        A: Tx.Buffer((M, K), a_type),
        B: Tx.Buffer((N, K), b_type),
        bias: Tx.Buffer((N,), a_type),
        gate: Tx.Buffer((N,), a_type),
        Y: Tx.Buffer((M, N), a_type),
        D: Tx.Buffer((M, N), d_type),
    ):
        # fmt: off
        with Tx.kernel():
            cbx, cby = Tx.cta_id([M_CLUSTER, N_CLUSTER], parent="cluster")
            bx = Tx.cta_id([SM_NUMBER], parent="kernel")
            wg_id = Tx.warpgroup_id([WG_NUMBER], parent="cta")
            warp_id = Tx.warp_id([WARP_NUMBER], parent="warpgroup")
            lane_id = Tx.thread_id([32], parent="warp")
            with Tx.cta():
                buf = Tx.alloc_buffer([SMEM_SIZE], "uint8", scope="shared.dyn")
                tmem_addr = Tx.decl_scalar("uint32", buf.data, scope="shared.dyn", elem_offset=0)
                A_smem = Tx.decl_buffer((PIPELINE_DEPTH, NUM_CONSUMER, BLK_M, BLK_K), a_type, buf.data, layout=A_layout,
                                        elem_offset=1024 // F16_BYTES)
                B_smem = Tx.decl_buffer((PIPELINE_DEPTH, BLK_N, BLK_K), b_type, buf.data, layout=B_layout,
                                        elem_offset=1024 // F16_BYTES + PIPELINE_DEPTH * NUM_CONSUMER * BLK_M * BLK_K)
                D_smem = Tx.decl_buffer((NUM_CONSUMER, BLK_M, EPI_TILE), d_type, buf.data, layout=D_layout,
                                        elem_offset=1024 // F16_BYTES + PIPELINE_DEPTH * (NUM_CONSUMER * BLK_M + BLK_N) * BLK_K)

                descI: Tx.uint32
                descA: Tx.uint64
                descB: Tx.uint64
                phase = Tx.alloc_buffer((1,), "int32", scope="local")
                stage: Tx.int32

                tma2mma = BarTMA2MMA(buf.data, 4, PIPELINE_DEPTH, 1, is_p2c=True)
                mma2tma = BarMMA2TMA(buf.data, 4 + PIPELINE_DEPTH, PIPELINE_DEPTH, 1, is_p2c=False)
                mma2ld = BarMMA2LD(buf.data, 4 + 2 * PIPELINE_DEPTH, 1, NUM_CONSUMER, is_p2c=True)
                ld2mma = BarLD2MMA(buf.data, 4 + 2 * PIPELINE_DEPTH + NUM_CONSUMER, 1, NUM_CONSUMER, is_p2c=False)

                tma2mma.init(1)
                mma2tma.init(NUM_CONSUMER)
                mma2ld.init(1)
                ld2mma.init(128 * NUM_CONSUMER)

                ptr: Tx.let[Tx.Var(name="ptr", dtype=PointerType(PrimType("uint64")))] = Tx.reinterpret("handle", Tx.ptx.map_shared_rank(tma2mma.mbar.ptr_to([0, 0]), 0))
                tma_finished = Tx.decl_buffer([PIPELINE_DEPTH], "uint64", data=ptr, scope="shared")

                tile_scheduler = ClusterPersistentScheduler2D("tile_scheduler", num_m_tiles=TILE_M_NUM, num_n_tiles=TILE_N_NUM, num_clusters=SM_NUMBER // 2, l2_group_size=TILE_GROUPS_ROW_SIZE)
                tile_scheduler.init(bx // 2)
                m_idx = Tx.meta_var(tile_scheduler.m_idx)
                n_idx = Tx.meta_var(tile_scheduler.n_idx)

                with Tx.warp()[0:1]:
                    Tx.ptx.tcgen05.alloc(Tx.address_of(tmem_addr), n_cols=N_COLS, cta_group=CTA_GROUP)

                Tx.cuda.cluster_sync()
                Tx.cuda.cta_sync()
                Tx.ptx.fence.proxy("shared")
                Tx.ptx.fence.mbarrier_init()
                Tx.cuda.trap_when_assert_failed(tmem_addr == 0)
                tmem = Tx.decl_buffer((128, N_COLS), "float32", scope="tmem", allocated_addr=0, layout=TileLayout(S[(128, N_COLS) : (1@TLane, 1@TCol)]))

                @Tx.inline
                def paritioned_loop(main_loop, epilogue1, epilogue2):
                    for ko in Tx.serial(PIPE_CYCLE):
                        for ks in Tx.unroll(PIPELINE_DEPTH):
                            stage = ko * PIPELINE_DEPTH + ks
                            main_loop(False, ks)
                        phase[0] = phase[0] ^ 1
                    if PIPE_REMAIN_NUM > 0:
                        for ks in Tx.unroll(PIPE_REMAIN_NUM):
                            stage = PIPE_CYCLE * PIPELINE_DEPTH + ks
                            main_loop(True, ks)
                        epilogue1()
                        for ks in Tx.unroll(PIPE_REMAIN_NUM, PIPELINE_DEPTH):
                            epilogue2(ks)
                        phase[0] = phase[0] ^ 1
                    else:
                        epilogue1()

                with Tx.cta():
                    Tx.attr({"tirx.scope_partition": True})
                    # === Producer warpgroup (WG2) ===
                    with Tx.warpgroup()[NUM_CONSUMER:NUM_CONSUMER + 1]:
                        Tx.ptx.setmaxnreg(False, 56)
                        if warp_id == 3:
                            phase[0] = 0
                            while tile_scheduler.valid():
                                with Tx.thread(parent="warp")[Tx.ptx.elect_sync()]:
                                    m_start0 = Tx.meta_var((m_idx * NUM_CONSUMER * CTA_GROUP + cbx) * BLK_M)
                                    m_start1 = Tx.meta_var((m_idx * NUM_CONSUMER * CTA_GROUP + CTA_GROUP + cbx) * BLK_M)
                                    n_start = Tx.meta_var((n_idx * CTA_GROUP + cbx) * BLK_N)
                                    k_start = Tx.meta_var(stage * BLK_K)

                                    @Tx.inline
                                    def tma_load(is_remain, ks):
                                        tma_copy = Tx.meta_var({"dispatch": "tma", "mbar": tma_finished.ptr_to([ks]), "cta_group": CTA_GROUP})
                                        mma2tma.wait(ks, 0, phase[0])
                                        Tx.copy_async(A_smem[ks, 0, :, :], A[m_start0 : m_start0 + BLK_M, k_start : k_start + BLK_K], **tma_copy)
                                        Tx.copy_async(A_smem[ks, 1, :, :], A[m_start1 : m_start1 + BLK_M, k_start : k_start + BLK_K], **tma_copy)
                                        Tx.copy_async(B_smem[ks, :, :], B[n_start : n_start + BLK_N, k_start : k_start + BLK_K], **tma_copy)
                                        if cbx == 0:
                                            tma2mma.arrive(ks, NUM_CONSUMER * BLK_K * (BLK_M * NUM_CONSUMER + BLK_N) * F16_BYTES)

                                    @Tx.inline
                                    def tma_load_epilogue(ks):
                                        mma2tma.wait(ks, 0, phase[0])
                                        if cbx == 0:
                                            tma2mma.arrive_only(ks)

                                    paritioned_loop(tma_load, skip, tma_load_epilogue)
                                tile_scheduler.next_tile()

                        elif warp_id < 2 and cbx == 0:
                            phase_tmem: Tx.int32
                            phase_tmem = 0
                            phase[0] = 0

                            Tx.ptx.tcgen05.encode_instr_descriptor(Tx.address_of(descI), "float32", a_type, b_type, MMA_M, MMA_N, MMA_K, False, False, CTA_GROUP)

                            while tile_scheduler.valid():
                                m_idx = Tx.meta_var(tile_scheduler.m_idx)
                                n_idx = Tx.meta_var(tile_scheduler.n_idx)
                                with Tx.thread()[Tx.ptx.elect_sync()]:
                                    ld2mma.wait(0, warp_id, phase_tmem)
                                    Tx.ptx.tcgen05.fence.after_thread_sync()

                                    @Tx.inline
                                    def mma(is_remain, ks):
                                        tma2mma.wait(ks, 0, phase[0])
                                        Tx.gemm_async(tmem[:, warp_id * MMA_N: warp_id * MMA_N + MMA_N], A_smem[ks, warp_id, :, :], B_smem[ks, :, :], dispatch="tcgen05", cta_group=CTA_GROUP, descI=descI,
                                                        accum=tvm.tir.Not(stage == 0 and ((not is_remain) or (is_remain and PIPE_CYCLE == 0))))
                                        mma2tma.arrive(ks)

                                    @Tx.inline
                                    def mma_epilogue1():
                                        mma2ld.arrive(warp_id)

                                    @Tx.inline
                                    def mma_epilogue2(ks):
                                        tma2mma.wait(ks, 0, phase[0])
                                        mma2tma.arrive(ks)

                                    paritioned_loop(mma, mma_epilogue1, mma_epilogue2)
                                    phase_tmem = phase_tmem ^ 1
                                tile_scheduler.next_tile()

                    # === Consumer warpgroups (WG0, WG1) — with gate epilogue ===
                    with Tx.warpgroup()[0:NUM_CONSUMER]:
                        Tx.ptx.setmaxnreg(True, 224)

                        reg = Tx.alloc_buffer((TMEM_LD_SIZE,), "float32", scope="local")
                        reg_wg = reg.view(128, TMEM_LD_SIZE, layout=TileLayout(S[(128, TMEM_LD_SIZE) : (1@tid_in_wg, 1)]))
                        reg_fp16 = Tx.alloc_buffer((BLK_N * CTA_GROUP,), d_type, scope="local")
                        phase_tmem: Tx.int32

                        phase_tmem = 0
                        while tile_scheduler.valid():
                            mma2ld.wait(0, wg_id, phase_tmem)
                            phase_tmem = phase_tmem ^ 1
                            Tx.ptx.tcgen05.fence.after_thread_sync()
                            # TMEM -> RF with bias + gate + Y
                            for i in Tx.unroll(MMA_N // TMEM_LD_SIZE):
                                col_st = Tx.meta_var(wg_id * MMA_N + i * TMEM_LD_SIZE)
                                Tx.copy(reg_wg[:, :], tmem[:, col_st : col_st + TMEM_LD_SIZE])
                                with Tx.thread():
                                    n_col = Tx.meta_var(n_idx * BLK_N * CTA_GROUP + i * TMEM_LD_SIZE)
                                    m_row = Tx.meta_var((m_idx * NUM_CONSUMER * CTA_GROUP + wg_id * CTA_GROUP + cbx) * BLK_M + warp_id * 32 + lane_id)
                                    # Add bias
                                    for j in Tx.serial(TMEM_LD_SIZE):
                                        reg[j] = reg[j] + Tx.cast(bias[n_col + j], "float32")
                                    # Multiply gate
                                    for j in Tx.serial(TMEM_LD_SIZE):
                                        reg[j] = reg[j] * Tx.cast(gate[n_col + j], "float32")
                                    # Add Y
                                    for j in Tx.serial(TMEM_LD_SIZE):
                                        reg[j] = reg[j] + Tx.cast(Y[m_row, n_col + j], "float32")
                                    Tx.cast(reg_fp16[i * TMEM_LD_SIZE : (i + 1) * TMEM_LD_SIZE], reg[:])

                            ld2mma.arrive(wg_id)
                            # RF -> GMEM via SMEM + TMA
                            for i in Tx.unroll(NUM_CONSUMER * BLK_N // EPI_TILE):
                                with Tx.thread():
                                    Tx.copy(D_smem[wg_id, warp_id * 32 + lane_id, :], reg_fp16[i * EPI_TILE : (i + 1) * EPI_TILE])
                                Tx.cuda.warpgroup_sync(wg_id)
                                Tx.ptx.fence.proxy(scope="shared")
                                with Tx.thread()[lane_id == 0 and warp_id == 0]:
                                    m_start: Tx.let = (m_idx * NUM_CONSUMER * CTA_GROUP + wg_id * CTA_GROUP + cbx) * BLK_M
                                    n_start: Tx.let = n_idx * BLK_N * CTA_GROUP + i * EPI_TILE
                                    Tx.copy_async(D[m_start : m_start + BLK_M, n_start : n_start + EPI_TILE], D_smem[wg_id, :, :], dispatch="tma")
                                    Tx.ptx.cp_async.bulk.commit_group()
                                    Tx.ptx.cp_async.bulk.wait_group(0)
                                Tx.cuda.warpgroup_sync(wg_id)
                            tile_scheduler.next_tile()

                with Tx.warp()[0:1]:
                    Tx.ptx.tcgen05.relinquish_alloc_permit(cta_group=CTA_GROUP)
                    Tx.ptx.tcgen05.dealloc(tmem_addr, n_cols=N_COLS, cta_group=CTA_GROUP)

                Tx.cuda.cluster_sync()
        # fmt: on

    # --- Prepare data ---
    import torch

    A_data = torch.randn((M, K), dtype=torch.float16)
    B_data = torch.randn((N, K), dtype=torch.float16)
    bias_data = torch.randn((N,), dtype=torch.float16)
    gate_data = torch.randn((N,), dtype=torch.float16)
    Y_data = torch.randn((M, N), dtype=torch.float16)

    def ref_flux_gate(A_data, B_data, bias_data, gate_data, Y_data):
        A_f32 = A_data.float()
        B_f32 = B_data.float()
        bias_f32 = bias_data.float()
        gate_f32 = gate_data.float()
        Y_f32 = Y_data.float()
        C = (A_f32 @ B_f32.T + bias_f32.unsqueeze(0)) * gate_f32.unsqueeze(0) + Y_f32
        return C.half().numpy()

    def tir_flux_gate(A_data, B_data, bias_data, gate_data, Y_data):
        DEV = tvm.cuda(0)
        A_tvm = tvm.runtime.tensor(A_data.numpy(), device=DEV)
        B_tvm = tvm.runtime.tensor(B_data.numpy(), device=DEV)
        bias_tvm = tvm.runtime.tensor(bias_data.numpy(), device=DEV)
        gate_tvm = tvm.runtime.tensor(gate_data.numpy(), device=DEV)
        Y_tvm = tvm.runtime.tensor(Y_data.numpy(), device=DEV)
        D_tvm = tvm.runtime.tensor(np.zeros((M, N), dtype="float16"), device=DEV)
        target = tvm.target.Target("cuda")
        with target:
            src, mod = get_source(flux_gate_kernel)
            print(src)
            func = lambda: mod(A_tvm, B_tvm, bias_tvm, gate_tvm, Y_tvm, D_tvm)
            ms = bench(func, warmup=0, repeat=30, proton_name="tir_flux_gate", debug=DEBUG)
            print(f"TIR flux_gate flops: {flops(ms) / 1e12} TFLOPS, time: {ms:.3f} ms")
        return D_tvm.numpy()

    C_ref = ref_flux_gate(A_data, B_data, bias_data, gate_data, Y_data)
    C_tir = tir_flux_gate(A_data, B_data, bias_data, gate_data, Y_data)
    np.testing.assert_allclose(C_tir, C_ref, rtol=1e-2, atol=1e-1)
    print("flux_gate: PASSED")


if __name__ == "__main__":
    test_flux_gelu()
    test_flux_gate()
