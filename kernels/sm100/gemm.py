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
import argparse
import math

import torch
import tvm
import deep_gemm
from deep_gemm.utils.math import per_block_cast_to_fp8, per_token_cast_to_fp8, ceil_to_ue8m0
from deep_gemm.testing import calc_diff

from tvm.script import tir as T
from tvm.script import tirp as Tp
from tvm.tirp.bench.utils import ProtonContext, bench

from tvm.tirp.op_schedule.cuda.copy_async import tma_shared_layout, SwizzleMode
from tvm.tir.layout import TileLayout, TLane, TCol, tid_in_wg as axis_tid_in_wg
from tvm.tirp.tile_scheduler import GroupMajor2D
from tvm.ir import PointerType, PrimType


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument(
        "--dtype",
        type=str,
        required=False,
        choices=["fp16", "bf16", "fp8-blockwise"],
        default="fp16",
    )
    parser.add_argument("--profile-warmup", type=int, required=False, default=10)
    parser.add_argument("--profile-repeat", type=int, required=False, default=30)
    return parser.parse_args()


########################################################################
# FP16(BF16) GEMM
########################################################################


def fp16_bf16_prepare_data(dtype, M, N, K):
    torch_dev = torch.device("cuda")

    if dtype == "fp16":
        dtype = torch.float16
    elif dtype == "bf16":
        dtype = torch.bfloat16
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    A = torch.randn(M, K).to(dtype).to(torch_dev)
    B = torch.randn(N, K).to(dtype).to(torch_dev)
    C = torch.zeros((M, N), dtype=dtype).to(torch_dev)
    return A, B, C


def fp16_bf16_gemm_2cta_2consumer(dtype: str, M: int, N: int, K: int):
    if dtype == "fp16":
        a_type = tvm.DataType("float16")
        b_type = tvm.DataType("float16")
        d_type = tvm.DataType("float16")
    elif dtype == "bf16":
        a_type = tvm.DataType("bfloat16")
        b_type = tvm.DataType("bfloat16")
        d_type = tvm.DataType("bfloat16")
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    acc_type = tvm.DataType("float32")

    CTA_GROUP = 2
    NUM_CONSUMER = 2
    BLK_M, BLK_N, BLK_K = 128, 128, 64
    MMA_M, MMA_N, MMA_K = 256, 256, 16

    PIPE_DEPTH = 4
    PIPE_CYCLE = (K // BLK_K) // PIPE_DEPTH
    PIPE_REMAIN_NUM = (K // BLK_K) % PIPE_DEPTH

    EPI_N = 64
    TMEM_LD_N = 8

    A_layout = tma_shared_layout(
        a_type, SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, NUM_CONSUMER, BLK_M, BLK_K)
    )
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, BLK_N, BLK_K))
    D_layout = tma_shared_layout(
        d_type, SwizzleMode.SWIZZLE_128B_ATOM, (NUM_CONSUMER, BLK_M, EPI_N)
    )

    F16_SIZE = 2
    SMEM_SIZE = (
        1024
        + PIPE_DEPTH * NUM_CONSUMER * BLK_M * BLK_K * F16_SIZE
        + PIPE_DEPTH * BLK_N * BLK_K * F16_SIZE
        + NUM_CONSUMER * BLK_M * EPI_N * F16_SIZE
    )

    SM_COUNT = 148  # number of Stream Multiprocessors for B200
    WG_NUMBER = 3  # WG2 (warp0: mma, warp1: mma, warp3: load), WG0 (LD_TMEM + writeback), WG1 (LD_TMEM + writeback)

    @T.prim_func(tirp=True)
    def fp16_bf16_gemm_2cta_2consumer_kernel(
        A: T.Buffer((M, K), a_type),
        B: T.Buffer((N, K), b_type),
        D: T.Buffer((M, N), d_type),
    ):
        # fmt: off
        with T.kernel():
            cbx, cby = T.cta_id([CTA_GROUP, 1], parent="cluster")
            bx = T.cta_id([SM_COUNT], parent="kernel")
            wg_id = T.warpgroup_id([WG_NUMBER], parent="cta")
            warp_id = T.warp_id([4], parent="warpgroup")
            lane_id = T.thread_id([32], parent="warp")

            # smem allocation
            buf = T.alloc_buffer([SMEM_SIZE], "uint8", scope="shared.dyn")
            pool = T.meta_var(Tp.PoolAllocator(buf.data))
            tmem_addr = pool.alloc((1,), "uint32")

            tma2mma = pool.alloc((PIPE_DEPTH,), "uint64", align=8) # align is required for mbarrier
            mma2tma = pool.alloc((PIPE_DEPTH,), "uint64", align=8) # align is required for mbarrier
            mma2ld = pool.alloc((NUM_CONSUMER,), "uint64", align=8) # align is required for mbarrier
            ld2mma = pool.alloc((NUM_CONSUMER,), "uint64", align=8) # align is required for mbarrier
            pool.move_base_to(1024)
            Asmem = pool.alloc((PIPE_DEPTH, NUM_CONSUMER, BLK_M, BLK_K), a_type, layout=A_layout)
            Bsmem = pool.alloc((PIPE_DEPTH, BLK_N, BLK_K), b_type, layout=B_layout)
            Dsmem = pool.alloc((NUM_CONSUMER, BLK_M, EPI_N), d_type, layout=D_layout)

            # mbarrier initialization
            if warp_id == 0 and lane_id == 0:
                for i in T.unroll(PIPE_DEPTH):
                    T.ptx.mbarrier.init(tma2mma.ptr_to([i]), 1) # singaled by warp3 of CTA-0
                    T.ptx.mbarrier.init(mma2tma.ptr_to([i]), NUM_CONSUMER) # signaled by warp0 & 1 of CTA-0
                for i in T.unroll(NUM_CONSUMER):
                    T.ptx.mbarrier.init(mma2ld.ptr_to([i]), 1) # signaled by warp0(1) of CTA-0
                    T.ptx.mbarrier.init(ld2mma.ptr_to([i]), 128 * CTA_GROUP) # signaled by warpgroup1(2) of both CTAs

            # CTA-0 is responsible for receiving signals of the finish of TMA of both 2 CTAs
            ptr: T.Var(name="ptr", dtype=PointerType(PrimType("uint64"))) = T.reinterpret("handle", T.ptx.map_shared_rank(tma2mma.ptr_to([0]), 0))
            tma2mma_cta0 = T.decl_buffer([PIPE_DEPTH], "uint64", data=ptr, scope="shared")

            # tmem allocation
            if wg_id == 0 and warp_id == 0:
                T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=512, cta_group=CTA_GROUP)

            T.ptx.fence.proxy("shared")
            T.ptx.fence.mbarrier_init()
            T.cuda.cluster_sync()

            tmem_addr_local = T.local_cell("uint32")
            tmem_addr_local = tmem_addr[0]

            tmem = T.decl_buffer((128, 512), "float32", scope="tmem", allocated_addr=0, layout=TileLayout(([128, 512], [1@TLane, 1@TCol])))

            tile_scheduler = T.meta_var(GroupMajor2D("tile_scheduler", m_tiles=M // MMA_M // NUM_CONSUMER, n_tiles=N // MMA_N, group_rows=8, step=SM_COUNT // CTA_GROUP))
            tile_scheduler.init(bx // CTA_GROUP)
            m_idx = T.meta_var(tile_scheduler.m_idx)
            n_idx = T.meta_var(tile_scheduler.n_idx)
            m_st = T.meta_var((m_idx * NUM_CONSUMER * CTA_GROUP + cbx) * BLK_M)
            n_st = T.meta_var((n_idx * CTA_GROUP + cbx) * BLK_N)

            if wg_id == 2:
                if warp_id == 3:
                    # load warp
                    phase_mma2tma = T.local_cell("int32")
                    phase_mma2tma = 1 # start from 1 to avoid waiting for the first tile

                    @T.macro
                    def tma_load_stage(stage, k_st):
                        T.ptx.mbarrier.try_wait(mma2tma.ptr_to([stage]), phase_mma2tma) # both CTAs wait for the mma (issued by warp0/1 of CTA-0) to finish]
                        tma_config = T.meta_var({"dispatch": "tma", "cta_group": CTA_GROUP, "mbar": tma2mma_cta0.ptr_to([stage])})
                        Tp.copy_async(Asmem[stage, 0, :, :], A[m_st : m_st + BLK_M, k_st : k_st + BLK_K], **tma_config)
                        Tp.copy_async(Asmem[stage, 1, :, :], A[m_st + CTA_GROUP * BLK_M : m_st + (CTA_GROUP + 1) * BLK_M, k_st : k_st + BLK_K], **tma_config)
                        Tp.copy_async(Bsmem[stage, :, :], B[n_st : n_st + BLK_N, k_st : k_st + BLK_K], **tma_config)
                        if cbx == 0:
                            T.ptx.mbarrier.arrive.expect_tx(tma2mma_cta0.ptr_to([stage]), CTA_GROUP * (NUM_CONSUMER * BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE) # signal CTA-0 the issue of tma

                    @T.macro
                    def tma_load():
                        for ko in T.serial(PIPE_CYCLE):
                            for ks in T.unroll(PIPE_DEPTH):
                                tma_load_stage(ks, (ko * PIPE_DEPTH + ks) * BLK_K)
                            phase_mma2tma = phase_mma2tma ^ 1
                        if PIPE_REMAIN_NUM > 0:
                            for ks in T.unroll(PIPE_REMAIN_NUM):
                                tma_load_stage(ks, (PIPE_CYCLE * PIPE_DEPTH + ks) * BLK_K)
                            for ks in T.unroll(PIPE_REMAIN_NUM, PIPE_DEPTH):
                                T.ptx.mbarrier.try_wait(mma2tma.ptr_to([ks]), phase_mma2tma)
                                if cbx == 0:
                                    T.ptx.mbarrier.arrive(tma2mma.ptr_to([ks]))
                            phase_mma2tma = phase_mma2tma ^ 1

                    with T.thread(parent="warp")[T.ptx.elect_sync()]:
                        while tile_scheduler.valid():
                            tma_load()
                            tile_scheduler.next_tile()

                elif warp_id < 2 and cbx == 0:
                    # mma warp
                    phase_tma2mma = T.local_cell("int32")
                    phase_tma2mma = 0
                    phase_ld2mma = T.local_cell("int32")
                    phase_ld2mma = 1 # start from 1 to avoid waiting for the first tile

                    descI = T.local_cell("uint32")
                    T.ptx.tcgen05.encode_instr_descriptor(T.address_of(descI), acc_type, a_type, b_type, MMA_M, MMA_N, MMA_K, False, False, CTA_GROUP)
                    T.ptx.tcgen05.fence.after_thread_sync()

                    @T.macro
                    def mma_stage(stage, accum):
                        T.ptx.mbarrier.try_wait(tma2mma.ptr_to([stage]), phase_tma2mma) # wait for the tma to finish
                        Tp.gemm_async(tmem[:, warp_id * MMA_N: warp_id * MMA_N + BLK_N], Asmem[stage, warp_id, :, :], Bsmem[stage, :, :], accum=accum, dispatch="tcgen05", cta_group=CTA_GROUP, descI=descI)
                        T.ptx.tcgen05.commit(mma2tma.ptr_to([stage]), cta_group=CTA_GROUP, cta_mask=3) # signal (both CTAs) the issue of mma

                    @T.macro
                    def mma():
                        T.ptx.mbarrier.try_wait(ld2mma.ptr_to([warp_id]), phase_ld2mma) # wait for the ld of corresponding consumer (of both CTAs) to finish
                        phase_ld2mma = phase_ld2mma ^ 1
                        for ko in T.serial(PIPE_CYCLE):
                            for ks in T.unroll(PIPE_DEPTH):
                                mma_stage(ks, not (PIPE_CYCLE > 0 and ko == 0 and ks == 0))
                            phase_tma2mma = phase_tma2mma ^ 1
                        if PIPE_REMAIN_NUM > 0:
                            for ks in T.unroll(PIPE_REMAIN_NUM):
                                mma_stage(ks, not (PIPE_CYCLE == 0 and ks == 0))
                        T.ptx.tcgen05.commit(mma2ld.ptr_to([warp_id]), cta_group=CTA_GROUP, cta_mask=3) # signal the corresponding consumer (of both CTAs) the issue of all mmas of current tile
                        if PIPE_REMAIN_NUM > 0:
                            for ks in T.unroll(PIPE_REMAIN_NUM, PIPE_DEPTH):
                                T.ptx.mbarrier.try_wait(tma2mma.ptr_to([ks]), phase_tma2mma)
                                T.ptx.tcgen05.commit(mma2tma.ptr_to([ks]), cta_group=CTA_GROUP, cta_mask=3)
                            phase_tma2mma = phase_tma2mma ^ 1

                    with T.thread(parent="warp")[T.ptx.elect_sync()]:
                        while tile_scheduler.valid():
                            mma()
                            tile_scheduler.next_tile()

            elif wg_id < 2:
                phase_mma2ld = T.local_cell("int32")
                phase_mma2ld = 0

                @T.macro
                def writeback():
                    T.ptx.mbarrier.try_wait(mma2ld.ptr_to([wg_id]), phase_mma2ld) # wait for the issues of all mmas issued by CTA-0 (warp0(1)) to finish
                    phase_mma2ld = phase_mma2ld ^ 1
                    T.ptx.tcgen05.fence.after_thread_sync()

                    Dreg_16b = T.alloc_local((MMA_N,), a_type)
                    for no in T.unroll(MMA_N // TMEM_LD_N):
                        Dreg = T.alloc_local((TMEM_LD_N,), acc_type)
                        with T.warpgroup():
                            Dreg_wg = Dreg.view(128, TMEM_LD_N, layout=TileLayout(([128, TMEM_LD_N], [1@axis_tid_in_wg, 1])))
                            n_tmem_ld_st = T.meta_var(wg_id * MMA_N + no * TMEM_LD_N)
                            Tp.copy(Dreg_wg[:, :], tmem[:, n_tmem_ld_st : n_tmem_ld_st + TMEM_LD_N])
                        with T.thread():
                            Tp.cast(Dreg_16b[no * TMEM_LD_N : no * TMEM_LD_N + TMEM_LD_N], Dreg[:])

                    # signal the finish of LD_TMEM from TMEM
                    T.ptx.mbarrier.arrive(ld2mma.ptr_to([wg_id]), cta_id=0, pred=True) # signal CTA-0 (warp0(1)) the finish of LD_TMEM from TMEM, such that CTA-0 (warp0(1)) can proceed to issue next mma

                    for no in T.unroll(MMA_N // EPI_N):
                        with T.thread():
                            Tp.copy(Dsmem[wg_id, warp_id * 32 + lane_id, :], Dreg_16b[no * EPI_N : (no + 1) * EPI_N])
                            T.ptx.fence.proxy(scope="shared")
                        T.cuda.warpgroup_sync(wg_id + 10)

                        # smem -> gmem
                        with T.thread(parent="warpgroup")[warp_id == 0 and lane_id == 0]:
                            m_st_epi = T.meta_var((m_idx * NUM_CONSUMER * CTA_GROUP + wg_id * CTA_GROUP + cbx) * BLK_M)
                            n_st_epi = T.meta_var(n_idx * MMA_N + no * EPI_N)
                            Tp.copy_async(D[m_st_epi: m_st_epi + BLK_M, n_st_epi: n_st_epi + EPI_N], Dsmem[wg_id, :, :], dispatch="tma")
                            T.ptx.cp_async.bulk.commit_group()
                            T.ptx.cp_async.bulk.wait_group(0)
                        T.cuda.warpgroup_sync(wg_id + 10)

                # LD_TMEM warpgroup
                while tile_scheduler.valid():
                    writeback()
                    tile_scheduler.next_tile()

            T.cuda.cluster_sync()
            # dealloc TMEM
            if wg_id == 0 and warp_id == 0:
                T.ptx.tcgen05.relinquish_alloc_permit(cta_group=CTA_GROUP)
                T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=CTA_GROUP)

    return fp16_bf16_gemm_2cta_2consumer_kernel


def fp16_bf16_tir_gemm(
    A,
    B,
    C,
    kernel,
    warmup,
    repeat,
):
    C_tvm = torch.zeros_like(C, device="cuda")
    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": kernel})
        ex = tvm.compile(mod, target=target, tir_pipeline="tirp")
        func = lambda: ex(A, B, C_tvm)
        bench(func, warmup=warmup, repeat=repeat, proton_name="tir")
    return C_tvm


def fp16_bf16_torch_cublas_gemm(A, B, C, warmup, repeat):
    import torch

    C_torch = torch.zeros_like(C).to("cuda")
    func = lambda: torch.matmul(A, B.T, out=C_torch)
    bench(func, warmup=warmup, repeat=repeat, proton_name="torch-cublas")
    return C_torch


def fp16_bf16_deepgemm_cublaslt(A, B, C, warmup, repeat):
    import deep_gemm

    C_torch = torch.zeros_like(C).to("cuda")
    func = lambda: deep_gemm.cublaslt_gemm_nt(A, B, C_torch)
    bench(func, warmup=warmup, repeat=repeat, proton_name="deepgemm-cublaslt")
    return C_torch


def bf16_deepgemm_bf16_gemm(A, B, C, warmup, repeat):
    import deep_gemm

    C_torch = torch.zeros_like(C).to("cuda")
    func = lambda: deep_gemm.bf16_gemm_nt(A, B, C_torch)
    bench(func, warmup=warmup, repeat=repeat, proton_name="deepgemm-bf16_gemm")
    return C_torch


def profile_fp16_bf16_gemm(dtype: str, M: int, N: int, K: int, kernel, warmup: int, repeat: int):
    A, B, C = fp16_bf16_prepare_data(dtype, M, N, K)

    with ProtonContext("gemm"):
        C_cublas = fp16_bf16_torch_cublas_gemm(A, B, C, warmup, repeat)
        C_deepgemm_cublaslt = fp16_bf16_deepgemm_cublaslt(A, B, C, warmup, repeat)
        if dtype == "bf16":
            C_deepgemm_bf16_gemm = bf16_deepgemm_bf16_gemm(A, B, C, warmup, repeat)
        C_tir = fp16_bf16_tir_gemm(A, B, C, kernel, warmup, repeat)

    torch.testing.assert_close(C_cublas, C_tir, rtol=1e-3, atol=1e-2)
    torch.testing.assert_close(C_cublas, C_deepgemm_cublaslt, rtol=1e-3, atol=1e-2)
    if dtype == "bf16":
        torch.testing.assert_close(C_cublas, C_deepgemm_bf16_gemm, rtol=1e-3, atol=1e-2)


########################################################################
# FP8-blockwise GEMM
########################################################################


def fp8_blockwise_prepare_data(M: int, N: int, K: int):
    A_origin = torch.randn((M, K), dtype=torch.float32)
    B_origin = torch.randn((N, K), dtype=torch.float32)

    A_fp8, sfa = per_token_cast_to_fp8(A_origin, use_ue8m0=True)
    B_fp8, sfb = per_block_cast_to_fp8(B_origin, use_ue8m0=True)

    sfa_uint8 = (sfa.view(torch.int32) >> 23).to(torch.uint8).contiguous()
    sfb_uint8 = (sfb.view(torch.int32) >> 23).to(torch.uint8).contiguous().repeat(128, 1)[:N, :]
    sfa_pack = sfa_uint8.view(torch.uint32).T.contiguous()
    sfb_pack = sfb_uint8.view(torch.uint32).T.contiguous()

    # Dequantization
    A_fp8_de = A_fp8.to(torch.float32)
    B_fp8_de = B_fp8.to(torch.float32)

    # Vectorized dequantization for A
    A_de = (
        A_fp8_de.reshape(M, K // 128, 128)
        * (2.0 ** (sfa_uint8[:, :, None].to(torch.float32) - 127))
    ).reshape(M, K)
    # Vectorized dequantization for B
    B_de = (
        B_fp8_de.reshape(N, K // 128, 128)
        * (2.0 ** (sfb_uint8[:, :, None].to(torch.float32) - 127))
    ).reshape(N, K)

    C_ref = torch.matmul(A_de, B_de.T).to(torch.bfloat16)

    return (
        A_fp8.to("cuda"),
        B_fp8.to("cuda"),
        sfa.to("cuda"),
        sfb.to("cuda"),
        sfa_pack.to("cuda"),
        sfb_pack.to("cuda"),
        C_ref.to("cuda"),
        A_origin.to("cuda"),
        B_origin.to("cuda"),
    )


def fp8_blockwise_tir_kernel(M: int, N: int, K: int):
    M_CLUSTER = 2
    N_CLUSTER = 1
    WG_NUMBER = 2
    WARP_NUMBER = 4
    NUM_THREADS = (32 * WARP_NUMBER) * WG_NUMBER
    SM_NUMBER = 148

    SMEM_PIPE_DEPTH = 6
    TMEM_PIPE_DEPTH = 2

    F8_BYTES = 1
    F16_BYTES = 2
    F32_BYTES = 4
    F128_BYTES = 16

    a_type = tvm.DataType("float8_e4m3fn")
    b_type = tvm.DataType("float8_e4m3fn")
    d_type = tvm.DataType("bfloat16")
    sfa_type = tvm.DataType("float8_e8m0fnu")
    sfb_type = tvm.DataType("float8_e8m0fnu")
    BLK_M, BLK_N, BLK_K = 128, 112, 128
    MMA_M, MMA_N, MMA_K = 256, 224, 32
    BLK_SFA, BLK_SFB = 128, 256
    PIPE_CIRCLE_NUM = (K // BLK_K) // SMEM_PIPE_DEPTH
    PIPE_REMAIN_NUM = (K // BLK_K) % SMEM_PIPE_DEPTH
    EPI_TILE = 32
    TMEM_LD_SIZE = 8
    N_COLS = 512
    SFA_TMEM_START_COL = TMEM_PIPE_DEPTH * MMA_N
    SFB_TMEM_START_COL = TMEM_PIPE_DEPTH * MMA_N + TMEM_PIPE_DEPTH * BLK_SFA // 32
    CTA_GROUP = 2
    QUANT_SIZE = BLK_K
    SWIZZLE = 3
    SMEM_SIZE = (
        SMEM_PIPE_DEPTH * BLK_M * BLK_K * F8_BYTES
        + SMEM_PIPE_DEPTH * BLK_N * BLK_K * F8_BYTES
        + TMEM_PIPE_DEPTH * BLK_M * EPI_TILE * F16_BYTES
        + SMEM_PIPE_DEPTH * BLK_SFA * F32_BYTES
        + SMEM_PIPE_DEPTH * BLK_SFB * F32_BYTES
        + 1024
    )

    assert SMEM_SIZE <= 232448
    assert TMEM_PIPE_DEPTH * (MMA_N + BLK_SFA // 32 + BLK_SFB // 32) <= 512

    TILE_GROUPS_ROW_SIZE = 16
    assert M % (BLK_M * CTA_GROUP) == 0
    # assert N % (BLK_N * CTA_GROUP) == 0
    TILE_M_NUM = M // (BLK_M * CTA_GROUP)
    TILE_N_NUM = math.ceil(N / (BLK_N * CTA_GROUP))

    class Barriers:

        def __init__(self, shared_buffer_base, shared_buffer_offs, pipe_depth, is_p2c):
            self.mbar: tvm.tir.Buffer = T.decl_buffer(
                (pipe_depth,), "uint64", shared_buffer_base, elem_offset=shared_buffer_offs
            ).buffer
            self.init_phase = 0 if is_p2c else 1
            self.pipe_depth = pipe_depth

        @T.macro
        def init(self, threads_num_wait):
            with T.thread()[0:1]:
                for i in T.serial(self.pipe_depth):
                    T.ptx.mbarrier.init(self.mbar.ptr_to([i]), threads_num_wait)

        @T.macro
        def wait(self, idx, phase):
            T.ptx.mbarrier.try_wait(self.mbar.ptr_to([idx]), self.init_phase ^ phase)

    class BarTRANS2MMA(Barriers):

        @T.macro
        def arrive(self, idx):
            T.ptx.mbarrier.arrive(self.mbar.ptr_to([idx]), cta_id=0, pred=True)

    class BarMMA2LD(Barriers):

        @T.macro
        def arrive(self, idx):
            T.ptx.tcgen05.commit(self.mbar.ptr_to([idx]), cta_group=CTA_GROUP, cta_mask=3)

    class BarMMA2TMA(Barriers):

        @T.macro
        def arrive(self, idx):
            T.ptx.tcgen05.commit(self.mbar.ptr_to([idx]), cta_group=CTA_GROUP, cta_mask=3)

    class BarLD2MMA(Barriers):

        @T.macro
        def arrive(self, idx):
            T.ptx.mbarrier.arrive(self.mbar.ptr_to([idx]), cta_id=0, pred=True)

    @T.macro
    def skip():
        pass

    A_layout = T.ComposeLayout(
        T.SwizzleLayout(4, 3, 3, swizzle_inner=True),
        T.TileLayout(shard=((SMEM_PIPE_DEPTH, BLK_M, BLK_K), (BLK_M * BLK_K, BLK_K, 1))),
    )
    B_layout = T.ComposeLayout(
        T.SwizzleLayout(4, 3, 3, swizzle_inner=True),
        T.TileLayout(shard=((SMEM_PIPE_DEPTH, BLK_N, BLK_K), (BLK_N * BLK_K, BLK_K, 1))),
    )
    D_layout = T.ComposeLayout(
        T.SwizzleLayout(3, 2, 3, swizzle_inner=True),
        T.TileLayout(shard=((TMEM_PIPE_DEPTH, BLK_M, EPI_TILE), (BLK_M * EPI_TILE, EPI_TILE, 1))),
    )

    SFA_layout = T.TileLayout(shard=((SMEM_PIPE_DEPTH, BLK_SFA // 32, 32), (BLK_SFA, 32, 1)))
    SFB_layout = T.TileLayout(shard=((SMEM_PIPE_DEPTH, BLK_SFB // 32, 32), (BLK_SFB, 32, 1)))

    @T.prim_func(tirp=True)
    def fp8_blockwise_deepgemm_kernel(
        A: T.Buffer((M, K), a_type),
        B: T.Buffer((N, K), b_type),
        D: T.Buffer((M, N), d_type),
        SFA: T.Buffer((math.ceil(K / QUANT_SIZE) // 4, M), "uint32"),
        SFB: T.Buffer((math.ceil(K / QUANT_SIZE) // 4, N), "uint32"),
    ):
        # fmt: off
        with T.kernel():
            cbx, cby = T.cta_id([M_CLUSTER, N_CLUSTER], parent="cluster")
            bx = T.cta_id([SM_NUMBER], parent="kernel")
            wg_id = T.warpgroup_id([WG_NUMBER], parent="cta")
            warp_id = T.warp_id([WARP_NUMBER], parent="warpgroup")
            tid_in_wg = T.thread_id([128], parent="warpgroup")
            lane_id = T.thread_id([32], parent="warp")
            with T.cta():
                # alloc shared memory
                buf = T.alloc_buffer([SMEM_SIZE], "uint8", scope="shared.dyn")
                pool = T.meta_var(Tp.PoolAllocator(buf.data))
                tmem_addr = T.decl_cell("uint32", buf.data, scope="shared.dyn", elem_offset=0)
                pool.move_base_to(1024)
                A_smem = pool.alloc((SMEM_PIPE_DEPTH, BLK_M, BLK_K), a_type, layout=A_layout)
                B_smem = pool.alloc((SMEM_PIPE_DEPTH, BLK_N, BLK_K), b_type, layout=B_layout)
                D_smem = pool.alloc((TMEM_PIPE_DEPTH, BLK_M, EPI_TILE), d_type, layout=D_layout)
                SFA_smem = pool.alloc((SMEM_PIPE_DEPTH, BLK_SFA // 32, 32), "uint32", layout=SFA_layout)
                SFB_smem = pool.alloc((SMEM_PIPE_DEPTH, BLK_SFB // 32, 32), "uint32", layout=SFB_layout)
                SFA_smem_2d = SFA_smem.view(SMEM_PIPE_DEPTH, BLK_SFA)
                SFB_smem_2d = SFB_smem.view(SMEM_PIPE_DEPTH, BLK_SFB)

                # alloc local memory
                reg = T.alloc_buffer((TMEM_LD_SIZE,), "float32", scope="local")
                reg_wg = reg.view(128, TMEM_LD_SIZE, layout=TileLayout(([128, TMEM_LD_SIZE], [1@axis_tid_in_wg, 1])))
                reg_fp16 = T.alloc_buffer((BLK_N * CTA_GROUP,), d_type, scope="local")
                stage = T.local_cell("int32")
                descA = T.local_cell("uint64")
                descB = T.local_cell("uint64")
                descSFA = T.local_cell("uint64")
                descSFB = T.local_cell("uint64")
                descI = T.local_cell("uint32")

                phase = T.local_cell("int32")

                # initialize
                tma2trans_bar = T.meta_var(Barriers(buf.data, 6, SMEM_PIPE_DEPTH, True))
                trans2mma_bar = T.meta_var(BarTRANS2MMA(buf.data, 6 + SMEM_PIPE_DEPTH, SMEM_PIPE_DEPTH, True))
                mma2tma_bar = T.meta_var(BarMMA2TMA(buf.data, 6 + 2 * SMEM_PIPE_DEPTH, SMEM_PIPE_DEPTH, False))
                mma2ld_bar = T.meta_var(BarMMA2LD(buf.data, 6 + 3 * SMEM_PIPE_DEPTH, TMEM_PIPE_DEPTH, True))
                ld2mma_bar = T.meta_var(BarLD2MMA(buf.data, 6 + 3 * SMEM_PIPE_DEPTH + TMEM_PIPE_DEPTH, TMEM_PIPE_DEPTH, False))
                tile_scheduler = T.meta_var(GroupMajor2D("tile_scheduler", m_tiles=TILE_M_NUM, n_tiles=TILE_N_NUM, group_rows=TILE_GROUPS_ROW_SIZE, step=SM_NUMBER // 2))

                tma2trans_bar.init(1)
                trans2mma_bar.init(CTA_GROUP * 32)
                mma2ld_bar.init(1)
                mma2tma_bar.init(1)
                ld2mma_bar.init(CTA_GROUP * 128)
                tile_scheduler.init(bx // 2)

                # alloc TMEM
                with T.warp()[0:1]:
                    T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=N_COLS, cta_group=2)
                    T.cuda.warp_sync()

                # sync
                T.ptx.fence.proxy("shared")
                T.ptx.fence.mbarrier_init()
                T.cuda.cluster_sync()
                T.cuda.trap_when_assert_failed(tmem_addr == 0)
                tmem = T.decl_buffer((128, N_COLS), "float32", scope="tmem", allocated_addr=0, layout=TileLayout(([128, N_COLS], [1@TLane, 1@TCol])))

                @T.macro
                def paritioned_loop(main_loop, epilogue1, epilogue2):
                    for ko in T.serial(PIPE_CIRCLE_NUM):
                        for ks in T.unroll(SMEM_PIPE_DEPTH):
                            stage = ko * SMEM_PIPE_DEPTH + ks
                            main_loop(ks)
                        phase = phase ^ 1
                    if PIPE_REMAIN_NUM > 0:
                        # last remained loop
                        for ks in T.unroll(PIPE_REMAIN_NUM):
                            stage = PIPE_CIRCLE_NUM * SMEM_PIPE_DEPTH + ks
                            main_loop(ks)
                        epilogue1()
                        # for unaligned cases
                        for ks in T.unroll(PIPE_REMAIN_NUM, SMEM_PIPE_DEPTH):
                            epilogue2(ks)
                        phase = phase ^ 1
                    else:
                        epilogue1()

                with T.cta():
                    T.block_attr({"tirp.scope_partition": True})
                    with T.warpgroup()[wg_id == 1]:
                        T.block_attr({"tirp.scope_partition": True})
                        with T.warp(parent="warpgroup")[warp_id == 3]:
                            phase = 0
                            while tile_scheduler.valid():
                                m_idx = T.meta_var(tile_scheduler.m_idx)
                                n_idx = T.meta_var(tile_scheduler.n_idx)
                                m_start = T.meta_var((m_idx * CTA_GROUP + cbx) * BLK_M)
                                n_start = T.meta_var((n_idx * CTA_GROUP + cbx) * BLK_N)
                                n_start_sf = T.meta_var(n_idx * CTA_GROUP * BLK_N)
                                k_start = T.meta_var(stage * BLK_K)

                                @T.macro
                                def tma_load(ks):
                                    mma2tma_bar.wait(ks, phase)
                                    tma_copy = T.meta_var({"dispatch": "tma", "mbar": tma2trans_bar.mbar.ptr_to([ks]), "cta_group": CTA_GROUP})
                                    with T.thread()[T.ptx.elect_sync()]:
                                        Tp.copy_async(A_smem[ks, :, :], A[m_start: m_start + BLK_M, k_start: k_start + BLK_K], **tma_copy)
                                        Tp.copy_async(B_smem[ks, :, :], B[n_start: n_start + BLK_N, k_start: k_start + BLK_K], **tma_copy)
                                        if stage % 4 == 0:
                                            Tp.copy_async(SFA_smem_2d[ks, :], SFA[stage // 4, m_start: m_start + BLK_M], **tma_copy)
                                            Tp.copy_async(SFB_smem_2d[ks, 0:BLK_N * CTA_GROUP], SFB[stage // 4, n_start_sf: n_start_sf + BLK_N * CTA_GROUP], **tma_copy)
                                        AB_bytes = T.meta_var(BLK_M * BLK_K * F8_BYTES + BLK_N * BLK_K * F8_BYTES)
                                        SFAB_bytes = T.meta_var((BLK_N * CTA_GROUP + BLK_M) * F32_BYTES)
                                        T.ptx.mbarrier.arrive.expect_tx(tma2trans_bar.mbar.ptr_to([ks]), T.if_then_else(stage % 4 == 0, AB_bytes + SFAB_bytes, AB_bytes))

                                @T.macro
                                def tma_load_epilogue(ks):
                                    mma2tma_bar.wait(ks, phase)
                                    with T.thread()[T.ptx.elect_sync()]:
                                        T.ptx.mbarrier.arrive.expect_tx(tma2trans_bar.mbar.ptr_to([ks]), 0)

                                paritioned_loop(tma_load, skip, tma_load_epilogue)
                                tile_scheduler.next_tile()

                        with T.warp(parent="warpgroup")[warp_id == 2]:
                            # transpose
                            phase = 0
                            # reg_trans = T.alloc_buffer((4,), "uint32", scope="local")
                            while tile_scheduler.valid():
                                m_idx = T.meta_var(tile_scheduler.m_idx)
                                n_idx = T.meta_var(tile_scheduler.n_idx)

                                @T.macro
                                def transpose(ks):
                                    # wait for sf has been prepared
                                    T.ptx.mbarrier.try_wait(tma2trans_bar.mbar.ptr_to([ks]), phase)
                                    if stage % 4 == 0:
                                        Tp.permute_dims(SFA_smem[ks], [0, 2, 1])
                                        Tp.permute_dims(SFB_smem[ks, :4], [0, 2, 1])
                                        Tp.permute_dims(SFB_smem[ks, 4:], [0, 2, 1])
                                        T.ptx.fence.proxy("shared")
                                    # mark that transpose is completed
                                    trans2mma_bar.arrive(ks)

                                @T.macro
                                def transpose_epilogue(ks):
                                    T.ptx.mbarrier.try_wait(tma2trans_bar.mbar.ptr_to([ks]), phase)
                                    trans2mma_bar.arrive(ks)

                                paritioned_loop(transpose, skip, transpose_epilogue)
                                tile_scheduler.next_tile()

                        with T.warp(parent="warpgroup")[warp_id == 0]:
                            if cbx == 0:
                                tmem_idx = T.local_cell("int32", "tmem_idx")
                                tmem_phase = T.local_cell("int32", "tmem_phase")
                                T.ptx.tcgen05.encode_instr_descriptor_block_scaled(T.address_of(descI), "float32", a_type, b_type, sfa_type, sfb_type,
                                                                                    0, 0, MMA_M, MMA_N, MMA_K, False, False, CTA_GROUP)
                                phase = 0
                                while tile_scheduler.valid():
                                    m_idx = T.meta_var(tile_scheduler.m_idx)
                                    n_idx = T.meta_var(tile_scheduler.n_idx)
                                    with T.thread()[T.ptx.elect_sync()]:
                                        tmem_idx = tile_scheduler.tile_idx % TMEM_PIPE_DEPTH
                                        tmem_phase = (tile_scheduler.tile_idx // TMEM_PIPE_DEPTH) & 1

                                        # wait for the tmem result to be consumed
                                        ld2mma_bar.wait(tmem_idx, tmem_phase)
                                        T.ptx.tcgen05.fence.after_thread_sync()

                                        @T.macro
                                        def mma(ks):
                                            # wait tma and sf-transpose arrival
                                            trans2mma_bar.wait(ks, phase)
                                            T.ptx.tcgen05.fence.after_thread_sync()

                                            # copy sf to tmem
                                            if stage % 4 == 0:
                                                for ki in T.unroll(0, BLK_SFA // 128):
                                                    T.ptx.tcgen05.encode_matrix_descriptor(
                                                        T.address_of(descSFA), SFA_smem.ptr_to([ks, ki * 4, 0]),
                                                        ldo=16, sdo=8 * 4 * F32_BYTES // F128_BYTES, swizzle=0
                                                    )
                                                    T.ptx.tcgen05.cp(0, 0, SFA_TMEM_START_COL + tmem_idx * BLK_SFA // 32 + ki * 4,
                                                                        descSFA, "32x128b", "uint32", "uint32", CTA_GROUP, "warpx4")
                                                for ki in T.unroll(0, BLK_SFB // 128):
                                                    T.ptx.tcgen05.encode_matrix_descriptor(
                                                        T.address_of(descSFB), SFB_smem.ptr_to([ks, ki * 4, 0]),
                                                        ldo=16, sdo=8 * 4 * F32_BYTES // F128_BYTES, swizzle=0
                                                    )
                                                    T.ptx.tcgen05.cp(0, 0, SFB_TMEM_START_COL + tmem_idx * BLK_SFB // 32 + ki * 4,
                                                                        descSFB, "32x128b", "uint32", "uint32", CTA_GROUP, "warpx4")

                                            # issue mma
                                            T.cuda.runtime_instr_desc(T.address_of(descI), stage % 4)
                                            for ki in T.unroll(BLK_K // MMA_K):
                                                T.ptx.tcgen05.encode_matrix_descriptor(T.address_of(descA), A_smem.ptr_to([ks, 0, ki * MMA_K]),
                                                                                    ldo=1, sdo=8 * BLK_K * F8_BYTES // F128_BYTES, swizzle=SWIZZLE)
                                                T.ptx.tcgen05.encode_matrix_descriptor(T.address_of(descB), B_smem.ptr_to([ks, 0, ki * MMA_K]),
                                                                                    ldo=1, sdo=8 * BLK_K * F8_BYTES // F128_BYTES, swizzle=SWIZZLE)

                                                if stage == 0 and ki == 0:
                                                    T.ptx.tcgen05.mma.block_scale("float32", a_type, b_type, sfa_type, sfb_type, tmem_idx * MMA_N, descA, descB,
                                                                                    SFA_TMEM_START_COL + tmem_idx * BLK_SFA // 32,
                                                                                    SFB_TMEM_START_COL + tmem_idx * BLK_SFB // 32,
                                                                                    descI, False, CTA_GROUP, False)
                                                else:
                                                    T.ptx.tcgen05.mma.block_scale("float32", a_type, b_type, sfa_type, sfb_type, tmem_idx * MMA_N, descA, descB,
                                                                                    SFA_TMEM_START_COL + tmem_idx * BLK_SFA // 32,
                                                                                    SFB_TMEM_START_COL + tmem_idx * BLK_SFB // 32,
                                                                                    descI, False, CTA_GROUP, True)
                                            mma2tma_bar.arrive(ks)

                                        @T.macro
                                        def mma_epilogue1():
                                            mma2ld_bar.arrive(tmem_idx)

                                        @T.macro
                                        def mma_epilogue2(ks):
                                            trans2mma_bar.wait(ks, phase)
                                            mma2tma_bar.arrive(ks)

                                        paritioned_loop(mma, mma_epilogue1, mma_epilogue2)

                                    tile_scheduler.next_tile()

                    with T.warpgroup()[wg_id == 0]:
                        T.cuda.trap_when_assert_failed(tmem_addr == 0)
                        tmem_idx = T.local_cell("int32", "tmem_idx")
                        tmem_phase = T.local_cell("int32", "tmem_phase")
                        phase = 0
                        while tile_scheduler.valid():
                            m_idx = T.meta_var(tile_scheduler.m_idx)
                            n_idx = T.meta_var(tile_scheduler.n_idx)
                            tmem_idx = tile_scheduler.tile_idx % TMEM_PIPE_DEPTH
                            tmem_phase = (tile_scheduler.tile_idx // TMEM_PIPE_DEPTH) & 1

                            # flush previous tma
                            if tid_in_wg == 0:
                                T.ptx.cp_async.bulk.wait_group(0)
                            T.cuda.warpgroup_sync(10)
                            # wait for the completion of all the mma of the same tile
                            mma2ld_bar.wait(tmem_idx, tmem_phase)
                            T.ptx.tcgen05.fence.after_thread_sync()

                            for ko in T.unroll(MMA_N // EPI_TILE):
                                stage = (tile_scheduler.tile_idx * MMA_N // EPI_TILE + ko) % TMEM_PIPE_DEPTH

                                # wait the smem to be free
                                if ko >= TMEM_PIPE_DEPTH:
                                    if tid_in_wg == 0:
                                        T.ptx.cp_async.bulk.wait_group(TMEM_PIPE_DEPTH - 1)
                                    T.cuda.warpgroup_sync(10)

                                # tmem -> rf (ld) -> smem
                                for ki in T.unroll(EPI_TILE // TMEM_LD_SIZE):
                                    col_st = T.meta_var(tmem_idx * MMA_N + ko * EPI_TILE + ki * TMEM_LD_SIZE)
                                    Tp.copy(reg_wg[:, :], tmem[:, col_st : col_st + TMEM_LD_SIZE])
                                    with T.thread():
                                        st = T.meta_var(ki * TMEM_LD_SIZE)
                                        Tp.cast(reg_fp16[st : st + TMEM_LD_SIZE], reg[:])
                                        Tp.copy(D_smem[stage, warp_id * 32 + lane_id, st : st + TMEM_LD_SIZE], reg_fp16[st : st + TMEM_LD_SIZE])

                                # the tmem can be overwritten
                                if ko == MMA_N // EPI_TILE - 1:
                                    T.ptx.tcgen05.fence.before_thread_sync()
                                    ld2mma_bar.arrive(tmem_idx)

                                T.ptx.fence.proxy(scope="shared")
                                T.cuda.warpgroup_sync(10)

                                # smem -> gmem
                                m_start = (m_idx * CTA_GROUP + cbx) * BLK_M
                                n_start = n_idx * CTA_GROUP * BLK_N + ko * EPI_TILE
                                with T.thread(parent="warpgroup")[tid_in_wg == 0]:
                                    Tp.copy_async(D[m_start: m_start + BLK_M, n_start: n_start + EPI_TILE], D_smem[stage, :, :], dispatch="tma")
                                    T.ptx.cp_async.bulk.commit_group()

                            tile_scheduler.next_tile()

                        if tid_in_wg == 0:
                            T.ptx.cp_async.bulk.wait_group(0)
                        T.cuda.warpgroup_sync(10)

                # dealloc TMEM
                with T.warp()[0:1]:
                    T.ptx.tcgen05.relinquish_alloc_permit(cta_group=2)
                    T.ptx.tcgen05.dealloc(tmem_addr, n_cols=N_COLS, cta_group=2)

                T.cuda.cluster_sync()
        # fmt: off

    return fp8_blockwise_deepgemm_kernel


def fp8_blockwise_tir_gemm(A_fp8, B_fp8, sfa_pack, sfb_pack, C, kernel, warmup, repeat):
    A_fp8, B_fp8, sfa_pack, sfb_pack = (
        A_fp8.clone(),
        B_fp8.clone(),
        sfa_pack.clone(),
        sfb_pack.clone(),
    )
    C_tvm = torch.zeros_like(C).to(torch.bfloat16).to("cuda")
    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": kernel})
        ex = tvm.compile(mod, target=target, tir_pipeline="tirp")
        func = lambda: ex(A_fp8, B_fp8, C_tvm, sfa_pack, sfb_pack)
        bench(func, warmup=warmup, repeat=repeat, proton_name="tir")
    return C_tvm


def fp8_blockwise_deep_gemm(A_fp8, B_fp8, sfa, sfb, C_ref, warmup, repeat, A_origin, B_origin):
    # A_fp8, B_fp8, sfa, sfb = A_fp8.clone(), B_fp8.clone(), sfa.clone(), sfb.clone()
    a = per_token_cast_to_fp8(A_origin, use_ue8m0=True)
    b = per_block_cast_to_fp8(B_origin, use_ue8m0=True)
    out = torch.zeros_like(C_ref).to(torch.bfloat16).to("cuda")
    func = lambda: deep_gemm.fp8_gemm_nt(
        a,
        b,
        out,
        disable_ue8m0_cast=False,
        recipe=None,
    )
    bench(func, warmup=warmup, repeat=repeat, proton_name="deepgemm")
    return out


def profile_fp8_blockwise_gemm(
    dtype: str, M: int, N: int, K: int, kernel, warmup: int, repeat: int
):
    A_fp8, B_fp8, sfa, sfb, sfa_pack, sfb_pack, C_ref, A_origin, B_origin = (
        fp8_blockwise_prepare_data(M, N, K)
    )

    with ProtonContext("fp8-blockwise-gemm"):
        C_tir = fp8_blockwise_tir_gemm(
            A_fp8, B_fp8, sfa_pack, sfb_pack, C_ref, kernel, warmup, repeat
        )
        C_deepgemm = fp8_blockwise_deep_gemm(
            A_fp8, B_fp8, sfa, sfb, C_ref, warmup, repeat, A_origin, B_origin
        )

    assert calc_diff(C_tir, C_ref.to("cuda")) < 1e-3
    assert calc_diff(C_deepgemm, C_ref.to("cuda")) < 1e-3


if __name__ == "__main__":
    args = parse_args()
    if args.dtype in ["fp16", "bf16"]:
        profile_fp16_bf16_gemm(
            args.dtype,
            args.m,
            args.n,
            args.k,
            fp16_bf16_gemm_2cta_2consumer(args.dtype, args.m, args.n, args.k),
            args.profile_warmup,
            args.profile_repeat,
        )
    elif args.dtype == "fp8-blockwise":
        profile_fp8_blockwise_gemm(
            args.dtype,
            args.m,
            args.n,
            args.k,
            fp8_blockwise_tir_kernel(args.m, args.n, args.k),
            args.profile_warmup,
            args.profile_repeat,
        )
    else:
        raise ValueError(f"Unsupported dtype: {args.dtype}")
