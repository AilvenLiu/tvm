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

import deep_gemm
import torch

import tvm
from tvm.script import tirx as Tx
from tvm.tir.layout import S, TCol, TileLayout, TLane
from tvm.tir.layout import tid_in_wg as axis_tid_in_wg
from tvm.tirx.bench.utils import ProtonContext, bench
from tvm.tirx.op_schedule.cuda.common import SwizzleMode, tma_shared_layout
from tvm.tirx.pipeline import MBarrier, PipelineState, TCGen05Bar, TMABar
from tvm.tirx.tile_scheduler import ClusterPersistentScheduler2D


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--dtype", type=str, choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--profile-warmup", type=int, default=10)
    parser.add_argument("--profile-repeat", type=int, default=30)
    return parser.parse_args()


########################################################################
# FP16(BF16) GEMM
########################################################################


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


def tir_kernel(dtype: str, M: int, N: int, K: int):
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
    MMA_M, MMA_N, MMA_K = 256, 256, 16  # noqa: F841

    PIPE_DEPTH = 4
    K_TILES = K // BLK_K

    EPI_N = 64
    TMEM_LD_N = 8
    DTYPE_SIZE = a_type.bits // 8  # 2 for fp16/bf16

    A_layout = tma_shared_layout(
        a_type, SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, NUM_CONSUMER, BLK_M, BLK_K)
    )
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, BLK_N, BLK_K))
    D_layout = tma_shared_layout(
        d_type, SwizzleMode.SWIZZLE_128B_ATOM, (NUM_CONSUMER, BLK_M, EPI_N)
    )

    SM_COUNT = 148  # number of Stream Multiprocessors for B200
    WG_NUMBER = 3  # WG2 (warp0: mma, warp1: mma, warp3: load), WG0 (LD_TMEM + writeback), WG1 (LD_TMEM + writeback)  # noqa: E501

    @Tx.prim_func(tirx=True)
    def kernel(
        A: Tx.Buffer((M, K), a_type),
        B: Tx.Buffer((N, K), b_type),
        D: Tx.Buffer((M, N), d_type),
    ):
        # fmt: off
        with Tx.kernel():
            cbx, cby = Tx.cta_id([CTA_GROUP, 1], parent="cluster")
            bx = Tx.cta_id([SM_COUNT], parent="kernel")
            wg_id = Tx.warpgroup_id([WG_NUMBER], parent="cta")
            warp_id = Tx.warp_id([4], parent="warpgroup")
            lane_id = Tx.thread_id([32], parent="warp")

            # smem allocation
            pool = Tx.PoolAllocator()
            tmem_addr = pool.alloc((1,), "uint32")

            tma2mma = TMABar(pool, PIPE_DEPTH, "tma2mma")
            mma2tma = TCGen05Bar(pool, PIPE_DEPTH, "mma2tma")
            mma2ld = TCGen05Bar(pool, NUM_CONSUMER, "mma2ld")
            ld2mma = MBarrier(pool, NUM_CONSUMER, "ld2mma")
            pool.move_base_to(1024)
            Asmem = pool.alloc((PIPE_DEPTH, NUM_CONSUMER, BLK_M, BLK_K), a_type, layout=A_layout)
            Bsmem = pool.alloc((PIPE_DEPTH, BLK_N, BLK_K), b_type, layout=B_layout)
            Dsmem = pool.alloc((NUM_CONSUMER, BLK_M, EPI_N), d_type, layout=D_layout)
            pool.commit()

            # mbarrier initialization
            tma2mma.init(1)   # signaled by warp3 of CTA-0
            mma2tma.init(NUM_CONSUMER)  # signaled by warp0 & 1 of CTA-0
            mma2ld.init(1)    # signaled by warp0(1) of CTA-0
            ld2mma.init(128 * CTA_GROUP)  # signaled by warpgroup1(2) of both CTAs

            # CTA-0 is responsible for receiving signals of the finish of TMA of both 2 CTAs
            tma2mma_cta0 = tma2mma.remote_view(0)

            # tmem allocation
            if wg_id == 0 and warp_id == 0:
                Tx.ptx.tcgen05.alloc(Tx.address_of(tmem_addr), n_cols=512, cta_group=CTA_GROUP)

            Tx.ptx.fence.proxy_async("shared::cta")
            Tx.ptx.fence.mbarrier_init()
            Tx.cuda.cluster_sync()

            tmem_addr_local: Tx.uint32
            tmem_addr_local = tmem_addr[0]  # noqa: F841

            tmem = Tx.decl_buffer((128, 512), "float32", scope="tmem", allocated_addr=0, layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)]))  # noqa: E501

            tile_scheduler = ClusterPersistentScheduler2D("tile_scheduler", num_m_tiles=M // MMA_M // NUM_CONSUMER, num_n_tiles=N // MMA_N, l2_group_size=8, num_clusters=SM_COUNT // CTA_GROUP)  # noqa: E501
            tile_scheduler.init(bx // CTA_GROUP)
            m_idx = Tx.meta_var(tile_scheduler.m_idx)
            n_idx = Tx.meta_var(tile_scheduler.n_idx)

            if wg_id == 2:
                if warp_id == 3:
                    # load warp
                    tma_phase = PipelineState("tma", PIPE_DEPTH)
                    tma_phase.init(is_producer=True)

                    @Tx.inline
                    def tma_load_stage(stage, k_tile):
                        mma2tma.wait(stage, tma_phase.phase) # both CTAs wait for the mma (issued by warp0/1 of CTA-0) to finish]  # noqa: E501
                        tma_config = Tx.meta_var({"dispatch": "tma", "cta_group": CTA_GROUP, "mbar": tma2mma_cta0.ptr_to([stage])})  # noqa: E501
                        A_tile = A.partition(tile_shape=(NUM_CONSUMER * CTA_GROUP * BLK_M, BLK_K), select=(m_idx, k_tile))  # noqa: E501
                        Tx.copy_async(Asmem[stage, 0, :, :], A_tile.partition(tile_shape=(BLK_M, BLK_K), select=(cbx, 0)), **tma_config)  # noqa: E501
                        Tx.copy_async(Asmem[stage, 1, :, :], A_tile.partition(tile_shape=(BLK_M, BLK_K), select=(cbx + CTA_GROUP, 0)), **tma_config)  # noqa: E501
                        Tx.copy_async(Bsmem[stage, :, :],
                                      B.partition(tile_shape=(CTA_GROUP * BLK_N, BLK_K), select=(n_idx, k_tile))  # noqa: E501
                                       .partition(tile_shape=(BLK_N, BLK_K), select=(cbx, 0)),
                                      **tma_config)
                        if cbx == 0:
                            tma2mma_cta0.arrive(stage, CTA_GROUP * (NUM_CONSUMER * BLK_M * BLK_K + BLK_N * BLK_K) * DTYPE_SIZE) # signal CTA-0 the issue of tma  # noqa: E501

                    @Tx.inline
                    def tma_load():
                        for k_tile in Tx.serial(K_TILES):
                            tma_load_stage(tma_phase.stage, k_tile)
                            tma_phase.move_to_next_stage()

                    with Tx.thread(parent="warp")[Tx.ptx.elect_sync()]:
                        while tile_scheduler.valid():
                            tma_load()
                            tile_scheduler.next_tile()

                elif warp_id < 2 and cbx == 0:
                    # mma warp
                    mma_phase = PipelineState("mma", PIPE_DEPTH)
                    mma_phase.init(is_producer=False)
                    ld_phase = PipelineState("ld", 1)
                    ld_phase.init(is_producer=True)

                    accum: Tx.int32

                    @Tx.inline
                    def mma_stage(stage):
                        tma2mma.wait(stage, mma_phase.phase) # wait for the tma to finish
                        Tx.gemm_async(tmem[:, warp_id * MMA_N: warp_id * MMA_N + MMA_N], Asmem[stage, warp_id, :, :], Bsmem[stage, :, :], accum=accum, dispatch="tcgen05", cta_group=CTA_GROUP)  # noqa: E501, F821, F823
                        accum = 1  # noqa: F841
                        mma2tma.arrive(stage, cta_group=CTA_GROUP, cta_mask=3) # signal (both CTAs) the issue of mma  # noqa: E501

                    @Tx.inline
                    def mma():
                        ld2mma.wait(warp_id, ld_phase.phase)
                        ld_phase.move_to_next_stage()
                        accum = 0  # noqa: F841
                        for k_tile in Tx.serial(K_TILES):
                            mma_stage(mma_phase.stage)
                            mma_phase.move_to_next_stage()
                        mma2ld.arrive(warp_id, cta_group=CTA_GROUP, cta_mask=3)

                    with Tx.thread(parent="warp")[Tx.ptx.elect_sync()]:
                        while tile_scheduler.valid():
                            mma()
                            tile_scheduler.next_tile()

            elif wg_id < 2:
                wb_phase = PipelineState("wb", 1)
                wb_phase.init(is_producer=False)

                @Tx.inline
                def writeback():
                    mma2ld.wait(wg_id, wb_phase.phase) # wait for the issues of all mmas issued by CTA-0 (warp0(1)) to finish  # noqa: E501
                    wb_phase.move_to_next_stage()
                    Tx.ptx.tcgen05.fence.after_thread_sync()

                    Dreg_16b = Tx.alloc_local((MMA_N,), a_type)
                    for no in Tx.unroll(MMA_N // TMEM_LD_N):
                        Dreg = Tx.alloc_local((TMEM_LD_N,), acc_type)
                        with Tx.warpgroup():
                            Dreg_wg = Dreg.view(128, TMEM_LD_N, layout=TileLayout(S[(128, TMEM_LD_N) : (1@axis_tid_in_wg, 1)]))  # noqa: E501
                            n_tmem_ld_st = Tx.meta_var(wg_id * MMA_N + no * TMEM_LD_N)
                            Tx.copy(Dreg_wg[:, :], tmem[:, n_tmem_ld_st : n_tmem_ld_st + TMEM_LD_N])
                        with Tx.thread():
                            Tx.cast(Dreg_16b[no * TMEM_LD_N : no * TMEM_LD_N + TMEM_LD_N], Dreg[:])

                    # signal the finish of LD_TMEM from TMEM
                    ld2mma.arrive(wg_id, cta_id=0, pred=True) # signal CTA-0 (warp0(1)) the finish of LD_TMEM from TMEM, such that CTA-0 (warp0(1)) can proceed to issue next mma  # noqa: E501

                    for no in Tx.unroll(MMA_N // EPI_N):
                        with Tx.thread():
                            Tx.copy(Dsmem[wg_id, warp_id * 32 + lane_id, :], Dreg_16b[no * EPI_N : (no + 1) * EPI_N])  # noqa: E501
                            Tx.ptx.fence.proxy_async("shared::cta")
                        Tx.cuda.warpgroup_sync(wg_id + 10)

                        # smem -> gmem
                        with Tx.thread(parent="warpgroup")[warp_id == 0 and lane_id == 0]:
                            D_tile = D.partition(tile_shape=(NUM_CONSUMER * CTA_GROUP * BLK_M, MMA_N), select=(m_idx, n_idx)) \
                                      .partition(tile_shape=(CTA_GROUP * BLK_M, EPI_N), select=(wg_id, no)) \
                                      .partition(tile_shape=(BLK_M, EPI_N), select=(cbx, 0))  # noqa: E501
                            Tx.copy_async(D_tile, Dsmem[wg_id, :, :], dispatch="tma")
                            Tx.ptx.cp_async.bulk.commit_group()
                            Tx.ptx.cp_async.bulk.wait_group(0)
                        Tx.cuda.warpgroup_sync(wg_id + 10)

                # LD_TMEM warpgroup
                while tile_scheduler.valid():
                    writeback()
                    tile_scheduler.next_tile()

            Tx.cuda.cluster_sync()
            # dealloc TMEM
            if wg_id == 0 and warp_id == 0:
                Tx.ptx.tcgen05.relinquish_alloc_permit(cta_group=CTA_GROUP)
                Tx.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=CTA_GROUP)

    return kernel


def tir_gemm(A, B, C, kernel, warmup, repeat):
    C_tvm = torch.zeros_like(C, device="cuda")
    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": kernel})
        ex = tvm.compile(mod, target=target, tir_pipeline="tirx")
        bench(lambda: ex(A, B, C_tvm), warmup=warmup, repeat=repeat, proton_name="tir")
    return C_tvm


def torch_cublas_gemm(A, B, C, warmup, repeat):
    C_torch = torch.zeros_like(C, device="cuda")
    bench(
        lambda: torch.matmul(A, B.T, out=C_torch),
        warmup=warmup,
        repeat=repeat,
        proton_name="torch-cublas",
    )
    return C_torch


def deepgemm_cublaslt(A, B, C, warmup, repeat):
    C_out = torch.zeros_like(C, device="cuda")
    bench(
        lambda: deep_gemm.cublaslt_gemm_nt(A, B, C_out, None),
        warmup=warmup,
        repeat=repeat,
        proton_name="deepgemm-cublaslt",
    )
    return C_out


def deepgemm_bf16(A, B, C, warmup, repeat):
    C_out = torch.zeros_like(C, device="cuda")
    bench(
        lambda: deep_gemm.bf16_gemm_nt(A, B, C_out, None),
        warmup=warmup,
        repeat=repeat,
        proton_name="deepgemm-bf16",
    )
    return C_out


def profile_gemm(dtype: str, M: int, N: int, K: int, kernel, warmup: int, repeat: int):
    A, B, C = prepare_data(dtype, M, N, K)

    with ProtonContext("gemm"):
        C_cublas = torch_cublas_gemm(A, B, C, warmup, repeat)
        C_cublaslt = deepgemm_cublaslt(A, B, C, warmup, repeat)
        C_bf16 = deepgemm_bf16(A, B, C, warmup, repeat) if dtype == "bf16" else None
        C_tir = tir_gemm(A, B, C, kernel, warmup, repeat)

    torch.testing.assert_close(C_cublas, C_tir, rtol=1e-3, atol=1e-2)
    torch.testing.assert_close(C_cublas, C_cublaslt, rtol=1e-3, atol=1e-2)
    if C_bf16 is not None:
        torch.testing.assert_close(C_cublas, C_bf16, rtol=1e-3, atol=1e-2)


if __name__ == "__main__":
    args = parse_args()
    profile_gemm(
        args.dtype,
        args.m,
        args.n,
        args.k,
        tir_kernel(args.dtype, args.m, args.n, args.k),
        args.profile_warmup,
        args.profile_repeat,
    )
