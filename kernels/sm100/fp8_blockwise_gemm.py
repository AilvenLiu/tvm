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
from deep_gemm.utils.math import per_block_cast_to_fp8, per_token_cast_to_fp8
from deep_gemm.testing import calc_diff

from tvm.script import tirx as Tx
from tvm.tirx.bench.utils import ProtonContext, bench
from tvm.tirx.tile_scheduler import ClusterPersistentScheduler2D
from tvm.tirx.pipeline import PipelineState, MBarrier, TMABar, TCGen05Bar

from tvm.tirx.op_schedule.cuda.copy_async import tma_shared_layout, SwizzleMode
from tvm.tir.layout import TileLayout, TLane, TCol, tid_in_wg as axis_tid_in_wg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--profile-warmup", type=int, default=10)
    parser.add_argument("--profile-repeat", type=int, default=30)
    return parser.parse_args()


########################################################################
# FP8-blockwise GEMM
########################################################################


def prepare_data(M: int, N: int, K: int):
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


def tir_kernel(M: int, N: int, K: int):
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
    K_TILES = K // BLK_K
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

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (SMEM_PIPE_DEPTH, BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (SMEM_PIPE_DEPTH, BLK_N, BLK_K))
    D_layout = tma_shared_layout(d_type, SwizzleMode.SWIZZLE_64B_ATOM, (TMEM_PIPE_DEPTH, BLK_M, EPI_TILE))

    SFA_layout = Tx.TileLayout(shard=((SMEM_PIPE_DEPTH, BLK_SFA // 32, 32), (BLK_SFA, 32, 1)))
    SFB_layout = Tx.TileLayout(shard=((SMEM_PIPE_DEPTH, BLK_SFB // 32, 32), (BLK_SFB, 32, 1)))

    # TMEM SF layouts: stride-0 K_MMA (all MMA iters share same SF since
    # quantization group = BLK_K = 128 = 4 MMAs × 32).
    # Shard: [pipe, M, 32, K_MMA] — pipe as leading dim for natural slicing.
    sf_mma_k = 1  # fp8 + e8m0fnu
    M_sf = 128 // 32  # 4 row chunks
    sfa_epc = 32 // sfa_type.bits  # 4
    sfb_epc = 32 // sfb_type.bits  # 4
    K_ITERS = BLK_K // MMA_K  # 4 MMA iterations per pipe stage
    K_PER_PIPE = sf_mma_k * K_ITERS  # 4 SF elements per pipe stage
    M_sf_sfb = BLK_SFB // 32  # 8 row chunks for N > 128
    SFA_tmem_layout = TileLayout(shard=([TMEM_PIPE_DEPTH, M_sf, 32, K_ITERS], [M_sf * sfa_epc @ TCol, sfa_epc @ TCol, 1 @ TLane, 0 @ TCol]), replica=([4], [32 @ TLane]))
    SFB_tmem_layout = TileLayout(shard=([TMEM_PIPE_DEPTH, M_sf_sfb, 32, K_ITERS], [M_sf_sfb * sfb_epc @ TCol, sfb_epc @ TCol, 1 @ TLane, 0 @ TCol]), replica=([4], [32 @ TLane]))

    @Tx.prim_func(tirx=True)
    def kernel(A: Tx.Buffer((M, K), a_type), B: Tx.Buffer((N, K), b_type), D: Tx.Buffer((M, N), d_type), SFA: Tx.Buffer((math.ceil(K / QUANT_SIZE) // 4, M), "uint32"), SFB: Tx.Buffer((math.ceil(K / QUANT_SIZE) // 4, N), "uint32")):
        # fmt: off
        with Tx.kernel():
            cbx, cby = Tx.cta_id([M_CLUSTER, N_CLUSTER], parent="cluster")
            bx = Tx.cta_id([SM_NUMBER], parent="kernel")
            wg_id = Tx.warpgroup_id([WG_NUMBER], parent="cta")
            warp_id = Tx.warp_id([WARP_NUMBER], parent="warpgroup")
            tid_in_wg = Tx.thread_id([128], parent="warpgroup")
            lane_id = Tx.thread_id([32], parent="warp")
            with Tx.cta():
                # alloc shared memory
                buf = Tx.alloc_buffer([SMEM_SIZE], "uint8", scope="shared.dyn")
                pool = Tx.meta_var(Tx.PoolAllocator(buf.data))
                tmem_addr = pool.alloc([1], "uint32", align=4)
                # Pipeline mbarriers
                tma2trans = Tx.meta_var(TMABar(pool, SMEM_PIPE_DEPTH, "tma2trans"))
                trans2mma = Tx.meta_var(MBarrier(pool, SMEM_PIPE_DEPTH, "trans2mma"))
                mma2tma = Tx.meta_var(TCGen05Bar(pool, SMEM_PIPE_DEPTH, "mma2tma"))
                mma2ld = Tx.meta_var(TCGen05Bar(pool, TMEM_PIPE_DEPTH, "mma2ld"))
                ld2mma = Tx.meta_var(MBarrier(pool, TMEM_PIPE_DEPTH, "ld2mma"))
                pool.move_base_to(1024)
                A_smem = pool.alloc((SMEM_PIPE_DEPTH, BLK_M, BLK_K), a_type, layout=A_layout)
                B_smem = pool.alloc((SMEM_PIPE_DEPTH, BLK_N, BLK_K), b_type, layout=B_layout)
                D_smem = pool.alloc((TMEM_PIPE_DEPTH, BLK_M, EPI_TILE), d_type, layout=D_layout)
                SFA_smem = pool.alloc((SMEM_PIPE_DEPTH, BLK_SFA // 32, 32), "uint32", layout=SFA_layout)
                SFB_smem = pool.alloc((SMEM_PIPE_DEPTH, BLK_SFB // 32, 32), "uint32", layout=SFB_layout)
                SFA_smem_2d = SFA_smem.view(SMEM_PIPE_DEPTH, BLK_SFA)
                SFB_smem_2d = SFB_smem.view(SMEM_PIPE_DEPTH, BLK_SFB)

                # alloc local memory
                reg = Tx.alloc_buffer((TMEM_LD_SIZE,), "float32", scope="local")
                cast_layout = TileLayout(([128, TMEM_LD_SIZE], [1@axis_tid_in_wg, 1]))
                reg_wg = reg.view(128, TMEM_LD_SIZE, layout=cast_layout)
                reg_16b = Tx.alloc_buffer((TMEM_LD_SIZE,), d_type, scope="local")
                reg_16b_wg = reg_16b.view(128, TMEM_LD_SIZE, layout=cast_layout)
                stage = Tx.local_cell("int32")
                descSFA = Tx.local_cell("uint64")
                descSFB = Tx.local_cell("uint64")
                descI = Tx.local_cell("uint32")

                tile_scheduler = Tx.meta_var(ClusterPersistentScheduler2D("tile_scheduler", num_m_tiles=TILE_M_NUM, num_n_tiles=TILE_N_NUM, l2_group_size=TILE_GROUPS_ROW_SIZE, num_clusters=SM_NUMBER // 2))

                # initialize barriers
                tma2trans.init(1)
                trans2mma.init(CTA_GROUP * 32)
                mma2ld.init(1)
                mma2tma.init(1)
                ld2mma.init(CTA_GROUP * 128)
                tile_scheduler.init(bx // CTA_GROUP)

                # alloc TMEM
                with Tx.warp()[0:1]:
                    Tx.ptx.tcgen05.alloc(Tx.address_of(tmem_addr[0]), n_cols=N_COLS, cta_group=2)
                    Tx.cuda.warp_sync()

                # sync
                Tx.ptx.fence.proxy("shared")
                Tx.ptx.fence.mbarrier_init()
                Tx.cuda.cluster_sync()
                Tx.cuda.trap_when_assert_failed(tmem_addr[0] == 0)
                tmem = Tx.decl_buffer((128, N_COLS), "float32", scope="tmem", allocated_addr=0, layout=TileLayout(([128, N_COLS], [1@TLane, 1@TCol])))
                SFA_tmem = Tx.decl_buffer((TMEM_PIPE_DEPTH, 128, K_PER_PIPE), sfa_type, scope="tmem", allocated_addr=SFA_TMEM_START_COL, layout=SFA_tmem_layout)
                SFB_tmem = Tx.decl_buffer((TMEM_PIPE_DEPTH, BLK_SFB, K_PER_PIPE), sfb_type, scope="tmem", allocated_addr=SFB_TMEM_START_COL, layout=SFB_tmem_layout)

                with Tx.cta():
                    Tx.attr({"tirx.scope_partition": True})
                    with Tx.warpgroup()[wg_id == 1]:
                        Tx.attr({"tirx.scope_partition": True})
                        with Tx.warp(parent="warpgroup")[warp_id == 3]:
                            tma_state = Tx.meta_var(PipelineState("tma", SMEM_PIPE_DEPTH))
                            tma_state.init(is_producer=True)
                            while tile_scheduler.valid():
                                m_idx = Tx.meta_var(tile_scheduler.m_idx)
                                n_idx = Tx.meta_var(tile_scheduler.n_idx)
                                m_start = Tx.meta_var((m_idx * CTA_GROUP + cbx) * BLK_M)
                                n_start = Tx.meta_var((n_idx * CTA_GROUP + cbx) * BLK_N)
                                n_start_sf = Tx.meta_var(n_idx * CTA_GROUP * BLK_N)

                                @Tx.macro
                                def tma_load(ks, k_tile):
                                    k_start = Tx.meta_var(k_tile * BLK_K)
                                    mma2tma.wait(ks, tma_state.phase)
                                    tma_copy = Tx.meta_var({"dispatch": "tma", "mbar": tma2trans.ptr_to([ks]), "cta_group": CTA_GROUP})
                                    with Tx.thread()[Tx.ptx.elect_sync()]:
                                        Tx.copy_async(A_smem[ks, :, :], A[m_start: m_start + BLK_M, k_start: k_start + BLK_K], **tma_copy)
                                        Tx.copy_async(B_smem[ks, :, :], B[n_start: n_start + BLK_N, k_start: k_start + BLK_K], **tma_copy)
                                        if k_tile % 4 == 0:
                                            Tx.copy_async(SFA_smem_2d[ks, :], SFA[k_tile // 4, m_start: m_start + BLK_M], **tma_copy)
                                            Tx.copy_async(SFB_smem_2d[ks, 0:BLK_N * CTA_GROUP], SFB[k_tile // 4, n_start_sf: n_start_sf + BLK_N * CTA_GROUP], **tma_copy)
                                        AB_bytes = Tx.meta_var(BLK_M * BLK_K * F8_BYTES + BLK_N * BLK_K * F8_BYTES)
                                        SFAB_bytes = Tx.meta_var((BLK_N * CTA_GROUP + BLK_M) * F32_BYTES)
                                        tma2trans.arrive(ks, Tx.if_then_else(k_tile % 4 == 0, AB_bytes + SFAB_bytes, AB_bytes))

                                for k_tile in Tx.serial(K_TILES):
                                    tma_load(tma_state.stage, k_tile)
                                    tma_state.move_to_next_stage()
                                tile_scheduler.next_tile()

                        with Tx.warp(parent="warpgroup")[warp_id == 2]:
                            # transpose
                            trans_state = Tx.meta_var(PipelineState("trans", SMEM_PIPE_DEPTH))
                            trans_state.init(is_producer=False)
                            while tile_scheduler.valid():
                                m_idx = Tx.meta_var(tile_scheduler.m_idx)
                                n_idx = Tx.meta_var(tile_scheduler.n_idx)

                                @Tx.macro
                                def transpose(ks, k_tile):
                                    # wait for sf has been prepared
                                    tma2trans.wait(ks, trans_state.phase)
                                    if k_tile % 4 == 0:
                                        Tx.permute_dims(SFA_smem[ks], [0, 2, 1])
                                        Tx.permute_dims(SFB_smem[ks, :4], [0, 2, 1])
                                        Tx.permute_dims(SFB_smem[ks, 4:], [0, 2, 1])
                                        Tx.ptx.fence.proxy("shared")
                                    # mark that transpose is completed
                                    trans2mma.arrive(ks)

                                for k_tile in Tx.serial(K_TILES):
                                    transpose(trans_state.stage, k_tile)
                                    trans_state.move_to_next_stage()
                                tile_scheduler.next_tile()

                        with Tx.warp(parent="warpgroup")[warp_id == 0]:
                            if cbx == 0:
                                tmem_idx = Tx.local_cell("int32", "tmem_idx")
                                tmem_phase = Tx.local_cell("int32", "tmem_phase")
                                Tx.ptx.tcgen05.encode_instr_descriptor_block_scaled(Tx.address_of(descI), "float32", a_type, b_type, sfa_type, sfb_type, 0, 0, MMA_M, MMA_N, MMA_K, False, False, CTA_GROUP)
                                mma_state = Tx.meta_var(PipelineState("mma", SMEM_PIPE_DEPTH))
                                mma_state.init(is_producer=False)
                                accum = Tx.local_cell("int32")
                                while tile_scheduler.valid():
                                    m_idx = Tx.meta_var(tile_scheduler.m_idx)
                                    n_idx = Tx.meta_var(tile_scheduler.n_idx)
                                    with Tx.thread()[Tx.ptx.elect_sync()]:
                                        tmem_idx = tile_scheduler.tile_idx % TMEM_PIPE_DEPTH
                                        tmem_phase = (tile_scheduler.tile_idx // TMEM_PIPE_DEPTH) & 1

                                        # wait for the tmem result to be consumed
                                        ld2mma.wait(tmem_idx, tmem_phase ^ 1)
                                        Tx.ptx.tcgen05.fence.after_thread_sync()

                                        @Tx.macro
                                        def mma(ks, k_tile):
                                            # wait tma and sf-transpose arrival
                                            trans2mma.wait(ks, mma_state.phase)
                                            Tx.ptx.tcgen05.fence.after_thread_sync()

                                            # copy sf to tmem
                                            if k_tile % 4 == 0:
                                                for ki in Tx.unroll(0, BLK_SFA // 128):
                                                    Tx.ptx.tcgen05.encode_matrix_descriptor(Tx.address_of(descSFA), SFA_smem.ptr_to([ks, ki * 4, 0]), ldo=16, sdo=8 * 4 * F32_BYTES // F128_BYTES, swizzle=0)
                                                    Tx.ptx.tcgen05.cp(0, 0, SFA_TMEM_START_COL + tmem_idx * BLK_SFA // 32 + ki * 4, descSFA, "32x128b", "uint32", "uint32", CTA_GROUP, "warpx4")
                                                for ki in Tx.unroll(0, BLK_SFB // 128):
                                                    Tx.ptx.tcgen05.encode_matrix_descriptor(Tx.address_of(descSFB), SFB_smem.ptr_to([ks, ki * 4, 0]), ldo=16, sdo=8 * 4 * F32_BYTES // F128_BYTES, swizzle=0)
                                                    Tx.ptx.tcgen05.cp(0, 0, SFB_TMEM_START_COL + tmem_idx * BLK_SFB // 32 + ki * 4, descSFB, "32x128b", "uint32", "uint32", CTA_GROUP, "warpx4")

                                            # issue mma
                                            Tx.cuda.runtime_instr_desc(Tx.address_of(descI), k_tile % 4)
                                            Tx.gemm_async(tmem[:, tmem_idx * MMA_N: tmem_idx * MMA_N + MMA_N], A_smem[ks, :, :], B_smem[ks, :, :], SFA=SFA_tmem[tmem_idx, :, :], SFB=SFB_tmem[tmem_idx, :, :], accum=accum, dispatch="tcgen05", cta_group=CTA_GROUP, descI=descI)
                                            accum = 1
                                            mma2tma.arrive(ks, cta_group=CTA_GROUP, cta_mask=3)

                                        accum = 0
                                        for k_tile in Tx.serial(K_TILES):
                                            mma(mma_state.stage, k_tile)
                                            mma_state.move_to_next_stage()
                                        mma2ld.arrive(tmem_idx, cta_group=CTA_GROUP, cta_mask=3)

                                    tile_scheduler.next_tile()

                    with Tx.warpgroup()[wg_id == 0]:
                        Tx.cuda.trap_when_assert_failed(tmem_addr[0] == 0)
                        tmem_idx = Tx.local_cell("int32", "tmem_idx")
                        tmem_phase = Tx.local_cell("int32", "tmem_phase")
                        while tile_scheduler.valid():
                            m_idx = Tx.meta_var(tile_scheduler.m_idx)
                            n_idx = Tx.meta_var(tile_scheduler.n_idx)
                            tmem_idx = tile_scheduler.tile_idx % TMEM_PIPE_DEPTH
                            tmem_phase = (tile_scheduler.tile_idx // TMEM_PIPE_DEPTH) & 1

                            # flush previous tma
                            if tid_in_wg == 0:
                                Tx.ptx.cp_async.bulk.wait_group(0)
                            Tx.cuda.warpgroup_sync(10)
                            # wait for the completion of all the mma of the same tile
                            mma2ld.wait(tmem_idx, tmem_phase)
                            Tx.ptx.tcgen05.fence.after_thread_sync()

                            for ko in Tx.unroll(MMA_N // EPI_TILE):
                                stage = (tile_scheduler.tile_idx * MMA_N // EPI_TILE + ko) % TMEM_PIPE_DEPTH

                                # wait the smem to be free
                                if ko >= TMEM_PIPE_DEPTH:
                                    if tid_in_wg == 0:
                                        Tx.ptx.cp_async.bulk.wait_group(TMEM_PIPE_DEPTH - 1)
                                    Tx.cuda.warpgroup_sync(10)

                                # tmem -> rf (ld) -> smem
                                for ki in Tx.unroll(EPI_TILE // TMEM_LD_SIZE):
                                    col_st = Tx.meta_var(tmem_idx * MMA_N + ko * EPI_TILE + ki * TMEM_LD_SIZE)
                                    Tx.copy(reg_wg[:, :], tmem[:, col_st : col_st + TMEM_LD_SIZE])
                                    Tx.cast(reg_16b_wg[:, :], reg_wg[:, :])
                                    with Tx.thread():
                                        st = Tx.meta_var(ki * TMEM_LD_SIZE)
                                        Tx.copy(D_smem[stage, warp_id * 32 + lane_id, st : st + TMEM_LD_SIZE], reg_16b[:])

                                # the tmem can be overwritten
                                if ko == MMA_N // EPI_TILE - 1:
                                    Tx.ptx.tcgen05.fence.before_thread_sync()
                                    ld2mma.arrive(tmem_idx)

                                Tx.ptx.fence.proxy(scope="shared")
                                Tx.cuda.warpgroup_sync(10)

                                # smem -> gmem
                                m_start = (m_idx * CTA_GROUP + cbx) * BLK_M
                                n_start = n_idx * CTA_GROUP * BLK_N + ko * EPI_TILE
                                with Tx.thread(parent="warpgroup")[tid_in_wg == 0]:
                                    Tx.copy_async(D[m_start: m_start + BLK_M, n_start: n_start + EPI_TILE], D_smem[stage, :, :], dispatch="tma")
                                    Tx.ptx.cp_async.bulk.commit_group()

                            tile_scheduler.next_tile()

                        if tid_in_wg == 0:
                            Tx.ptx.cp_async.bulk.wait_group(0)
                        Tx.cuda.warpgroup_sync(10)

                # dealloc TMEM
                with Tx.warp()[0:1]:
                    Tx.ptx.tcgen05.relinquish_alloc_permit(cta_group=2)
                    Tx.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=N_COLS, cta_group=2)

                Tx.cuda.cluster_sync()
        # fmt: off

    return kernel


def tir_gemm(A_fp8, B_fp8, sfa_pack, sfb_pack, C, kernel, warmup, repeat):
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
        ex = tvm.compile(mod, target=target, tir_pipeline="tirx")
        bench(
            lambda: ex(A_fp8, B_fp8, C_tvm, sfa_pack, sfb_pack),
            warmup=warmup,
            repeat=repeat,
            proton_name="tir",
        )
    return C_tvm


def deepgemm(A_fp8, B_fp8, sfa, sfb, C_ref, warmup, repeat, A_origin, B_origin):
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


def profile_gemm(M: int, N: int, K: int, kernel, warmup: int, repeat: int):
    A_fp8, B_fp8, sfa, sfb, sfa_pack, sfb_pack, C_ref, A_origin, B_origin = prepare_data(M, N, K)

    with ProtonContext("fp8-blockwise-gemm"):
        C_tir = tir_gemm(A_fp8, B_fp8, sfa_pack, sfb_pack, C_ref, kernel, warmup, repeat)
        C_deepgemm = deepgemm(A_fp8, B_fp8, sfa, sfb, C_ref, warmup, repeat, A_origin, B_origin)

    assert calc_diff(C_tir, C_ref.to("cuda")) < 1e-3
    assert calc_diff(C_deepgemm, C_ref.to("cuda")) < 1e-3


if __name__ == "__main__":
    args = parse_args()
    profile_gemm(
        args.m,
        args.n,
        args.k,
        tir_kernel(args.m, args.n, args.k),
        args.profile_warmup,
        args.profile_repeat,
    )
