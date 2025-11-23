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

import torch
import tvm

from tvm.script import tir as T
from tvm.script import tirp as Tp
from tvm.tirp.bench.utils import ProtonContext, bench

from tvm.tirp.op_schedule.cuda.copy_async import tma_shared_layout, SwizzleMode
from tvm.tir.layout import TileLayout, TLane, TCol, tid_in_wg
from tvm.tirp.tile_scheduler import GroupMajor2D
from tvm.ir import PointerType, PrimType


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument(
        "--dtype", type=str, required=False, choices=["fp16", "bf16"], default="fp16"
    )
    parser.add_argument("--profile-warmup", type=int, required=False, default=10)
    parser.add_argument("--profile-repeat", type=int, required=False, default=30)
    return parser.parse_args()


def prepare_data(dtype, M, N, K):
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


def tir_gemm(
    A,
    B,
    C,
    kernel,
    warmup,
    repeat,
):
    def torch_to_tvm(tensor: torch.Tensor):
        return tvm.runtime.from_dlpack(torch.to_dlpack(tensor))

    def tvm_to_torch(tensor: tvm.runtime.Tensor):
        return torch.from_dlpack(tensor._to_dlpack())

    DEV = tvm.cuda(0)
    A_tvm = torch_to_tvm(A)
    B_tvm = torch_to_tvm(B)
    C_tvm = torch_to_tvm(torch.zeros_like(C))
    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": kernel})
        ex = tvm.compile(mod, target=target, tir_pipeline="tirp")
        func = lambda: ex(A_tvm, B_tvm, C_tvm)
        bench(func, warmup=warmup, repeat=repeat, proton_name="tir")
    return tvm_to_torch(C_tvm)


########################################################################
# FP16(BF16) GEMM
########################################################################


def fp16_bf16_cublas_gemm(A, B, C, warmup, repeat):
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
                            Dreg_wg = Dreg.view(128, TMEM_LD_N, layout=TileLayout(([128, TMEM_LD_N], [1@tid_in_wg, 1])))
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


def profile_fp16_bf16_gemm(dtype: str, M: int, N: int, K: int, kernel, warmup: int, repeat: int):
    A, B, C = prepare_data(dtype, M, N, K)

    with ProtonContext("gemm"):
        C_cublas = fp16_bf16_cublas_gemm(A, B, C, warmup, repeat)
        C_deepgemm_cublaslt = fp16_bf16_deepgemm_cublaslt(A, B, C, warmup, repeat)
        if dtype == "bf16":
            C_deepgemm_bf16_gemm = bf16_deepgemm_bf16_gemm(A, B, C, warmup, repeat)
        C_tir = tir_gemm(A, B, C, kernel, warmup, repeat)

        torch.testing.assert_close(C_cublas, C_tir, rtol=1e-3, atol=1e-2)
        torch.testing.assert_close(C_cublas, C_deepgemm_cublaslt, rtol=1e-3, atol=1e-2)
        if dtype == "bf16":
            torch.testing.assert_close(C_cublas, C_deepgemm_bf16_gemm, rtol=1e-3, atol=1e-2)


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
    else:
        raise ValueError(f"Unsupported dtype: {args.dtype}")
