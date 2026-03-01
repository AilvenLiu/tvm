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

import ml_dtypes
import numpy as np

import tvm
import tvm.testing
from tvm.script import tirx as Tx
from tvm.tirx.tile_scheduler import ClusterPersistentScheduler2D
from tvm.tirx.bench.utils import bench, ProtonContext
from tvm.tirx.pipeline import MBarrier, TMABar, TCGen05Bar
from tvm.tir.layout import TileLayout, TLane, TCol, S
from tvm.tir.layout import tid_in_wg as axis_tid_in_wg

import torch
import deep_gemm
from deep_gemm.utils.math import per_block_cast_to_fp8, per_token_cast_to_fp8

# can get 3250 TFLOPs on a single B200 with (m, n, k) = (8192, 8064, 8192), which is aligned to DeepSeek's

# cluster: [2, 1], cta_num = 2
# warpgroup:
#   wg_id = 0: ld & to GMEM
#   wg_id = 1:
#       warp_id = 0: cp & mma
#       warp_id = 2: transpose
#       warp_id = 3: tma

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
M, N, K = 8192, 8064, 8192
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
assert TMEM_PIPE_DEPTH * (MMA_N + BLK_SFA // 32 + BLK_SFB // 32) <= 512


def get_source(func: tvm.tir.PrimFunc) -> str:
    target = tvm.target.Target("cuda")
    mod = tvm.IRModule({"main": func})
    mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
    src = mod.mod.imports[0].inspect_source()
    return src, mod


def ceildiv(a, b):
    return (a + b - 1) // b


def flops(ms):
    return M * N * K * 2 / (ms * 1e-3)


def calc_diff(x: torch.Tensor, y: torch.Tensor):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


TILE_GROUPS_ROW_SIZE = 16
assert M % (BLK_M * CTA_GROUP) == 0
assert N % (BLK_N * CTA_GROUP) == 0
TILE_M_NUM = M // (BLK_M * CTA_GROUP)
TILE_N_NUM = N // (BLK_N * CTA_GROUP)


@Tx.inline
def skip():
    pass


A_layout = Tx.ComposeLayout(
    Tx.SwizzleLayout(4, 3, 3, swizzle_inner=True),
    Tx.TileLayout(Tx.S[(SMEM_PIPE_DEPTH, BLK_M, BLK_K) : (BLK_M * BLK_K, BLK_K, 1)]),
)
B_layout = Tx.ComposeLayout(
    Tx.SwizzleLayout(4, 3, 3, swizzle_inner=True),
    Tx.TileLayout(Tx.S[(SMEM_PIPE_DEPTH, BLK_N, BLK_K) : (BLK_N * BLK_K, BLK_K, 1)]),
)
D_layout = Tx.ComposeLayout(
    Tx.SwizzleLayout(3, 2, 3, swizzle_inner=True),
    Tx.TileLayout(Tx.S[(TMEM_PIPE_DEPTH, BLK_M, EPI_TILE) : (BLK_M * EPI_TILE, EPI_TILE, 1)]),
)

SFA_layout = Tx.TileLayout(Tx.S[(SMEM_PIPE_DEPTH, BLK_SFA // 32, 32) : (BLK_SFA, 32, 1)])
SFB_layout = Tx.TileLayout(Tx.S[(SMEM_PIPE_DEPTH, BLK_SFB // 32, 32) : (BLK_SFB, 32, 1)])


@Tx.prim_func(tirx=True)
def deepgemm(
    A: Tx.Buffer((M, K), a_type),
    B: Tx.Buffer((N, K), b_type),
    D: Tx.Buffer((M, N), d_type),
    SFA: Tx.Buffer((ceildiv(K, QUANT_SIZE) // 4, M), "uint32"),
    SFB: Tx.Buffer((ceildiv(K, QUANT_SIZE) // 4, N), "uint32"),
):
    # fmt: off
    Tx.func_attr({"global_symbol": "main"})
    D_tensor_map: Tx.let[Tx.handle("tensormap")] = Tx.tvm_stack_alloca("tensormap", 1)
    Tx.call_packed("runtime.cuTensorMapEncodeTiled", D_tensor_map, d_type, 2, D.data, N, M, N * F16_BYTES, EPI_TILE, BLK_M, 1, 1, 0, 2, 0, 0)

    with Tx.kernel():
        cbx, cby = Tx.cta_id([M_CLUSTER, N_CLUSTER], parent="cluster")
        bx = Tx.cta_id([SM_NUMBER], parent="kernel")
        wg_id = Tx.warpgroup_id([WG_NUMBER], parent="cta")
        warp_id = Tx.warp_id([WARP_NUMBER], parent="warpgroup")
        tid_in_wg = Tx.thread_id([128], parent="warpgroup")
        lane_id = Tx.thread_id([32], parent="warp")
        with Tx.cta():
            # alloc shared memory
            pool = Tx.meta_var(Tx.PoolAllocator())
            tmem_addr = Tx.decl_scalar("uint32", pool.ptr, scope="shared.dyn", elem_offset=0)
            pool.move_base_to(8)

            # Barriers
            tma2trans_bar = TMABar(pool, SMEM_PIPE_DEPTH)
            trans2mma_bar = MBarrier(pool, SMEM_PIPE_DEPTH)
            mma2tma_bar = TCGen05Bar(pool, SMEM_PIPE_DEPTH)
            mma2ld_bar = TCGen05Bar(pool, TMEM_PIPE_DEPTH)
            ld2mma_bar = MBarrier(pool, TMEM_PIPE_DEPTH)

            pool.move_base_to(1024)
            A_smem = pool.alloc((SMEM_PIPE_DEPTH, BLK_M, BLK_K), a_type, layout=A_layout)
            B_smem = pool.alloc((SMEM_PIPE_DEPTH, BLK_N, BLK_K), b_type, layout=B_layout)
            D_smem = pool.alloc((TMEM_PIPE_DEPTH, BLK_M, EPI_TILE), d_type, layout=D_layout)
            SFA_smem = pool.alloc((SMEM_PIPE_DEPTH, BLK_SFA // 32, 32), "uint32", layout=SFA_layout)
            SFB_smem = pool.alloc((SMEM_PIPE_DEPTH, BLK_SFB // 32, 32), "uint32", layout=SFB_layout)
            SFA_smem_2d = SFA_smem.view(SMEM_PIPE_DEPTH, BLK_SFA)
            SFB_smem_2d = SFB_smem.view(SMEM_PIPE_DEPTH, BLK_SFB)
            pool.commit()

            # alloc local memory
            reg = Tx.alloc_buffer((TMEM_LD_SIZE,), "float32", scope="local")
            reg_wg = reg.view(128, TMEM_LD_SIZE, layout=TileLayout(S[(128, TMEM_LD_SIZE) : (1@axis_tid_in_wg, 1)]))
            reg_fp16 = Tx.alloc_buffer((BLK_N * CTA_GROUP,), d_type, scope="local")
            stage: Tx.int32
            descA: Tx.uint64
            descB: Tx.uint64
            descSFA: Tx.uint64
            descSFB: Tx.uint64
            descI: Tx.uint32

            phase: Tx.int32
            tile_scheduler = ClusterPersistentScheduler2D("tile_scheduler", num_m_tiles=TILE_M_NUM, num_n_tiles=TILE_N_NUM, num_clusters=SM_NUMBER // 2, l2_group_size=TILE_GROUPS_ROW_SIZE)

            tma2trans_bar.init(1)
            trans2mma_bar.init(CTA_GROUP * 32)
            mma2ld_bar.init(1)
            mma2tma_bar.init(1)
            ld2mma_bar.init(CTA_GROUP * 128)
            tile_scheduler.init(bx // 2)

            # alloc TMEM
            with Tx.warp()[0:1]:
                Tx.ptx.tcgen05.alloc(Tx.address_of(tmem_addr), n_cols=N_COLS, cta_group=2)
                Tx.cuda.warp_sync()

            # sync
            Tx.ptx.fence.proxy_async("shared::cta")
            Tx.ptx.fence.mbarrier_init()
            Tx.cuda.cluster_sync()
            Tx.cuda.trap_when_assert_failed(tmem_addr == 0)
            tmem = Tx.decl_buffer((128, N_COLS), "float32", scope="tmem", allocated_addr=0, layout=TileLayout(S[(128, N_COLS) : (1@TLane, 1@TCol)]))

            @Tx.inline
            def paritioned_loop(main_loop, epilogue1, epilogue2):
                for ko in Tx.serial(PIPE_CIRCLE_NUM):
                    for ks in Tx.unroll(SMEM_PIPE_DEPTH):
                        stage = ko * SMEM_PIPE_DEPTH + ks
                        main_loop(ks)
                    phase = phase ^ 1
                if PIPE_REMAIN_NUM > 0:
                    # last remained loop
                    for ks in Tx.unroll(PIPE_REMAIN_NUM):
                        stage = PIPE_CIRCLE_NUM * SMEM_PIPE_DEPTH + ks
                        main_loop(ks)
                    epilogue1()
                    # for unaligned cases
                    for ks in Tx.unroll(PIPE_REMAIN_NUM, SMEM_PIPE_DEPTH):
                        epilogue2(ks)
                    phase = phase ^ 1
                else:
                    epilogue1()

            with Tx.cta():
                Tx.attr({"tirx.scope_partition": True})
                with Tx.warpgroup()[wg_id == 1]:
                    Tx.attr({"tirx.scope_partition": True})
                    with Tx.warp(parent="warpgroup")[warp_id == 3]:
                        phase = 0
                        while tile_scheduler.valid():
                            m_idx = Tx.meta_var(tile_scheduler.m_idx)
                            n_idx = Tx.meta_var(tile_scheduler.n_idx)
                            m_start = Tx.meta_var((m_idx * CTA_GROUP + cbx) * BLK_M)
                            n_start = Tx.meta_var((n_idx * CTA_GROUP + cbx) * BLK_N)
                            n_start_sf = Tx.meta_var(n_idx * CTA_GROUP * BLK_N)
                            k_start = Tx.meta_var(stage * BLK_K)

                            @Tx.inline
                            def tma_load(ks):
                                mma2tma_bar.wait(ks, phase ^ 1)
                                tma_copy = Tx.meta_var({"dispatch": "tma", "mbar": tma2trans_bar.ptr_to([ks]), "cta_group": CTA_GROUP})
                                with Tx.thread()[Tx.ptx.elect_sync()]:
                                    Tx.copy_async(A_smem[ks, :, :], A[m_start: m_start + BLK_M, k_start: k_start + BLK_K], **tma_copy)
                                    Tx.copy_async(B_smem[ks, :, :], B[n_start: n_start + BLK_N, k_start: k_start + BLK_K], **tma_copy)
                                    if stage % 4 == 0:
                                        Tx.copy_async(SFA_smem_2d[ks, :], SFA[stage // 4, m_start: m_start + BLK_M], **tma_copy)
                                        Tx.copy_async(SFB_smem_2d[ks, 0:BLK_N * CTA_GROUP], SFB[stage // 4, n_start_sf: n_start_sf + BLK_N * CTA_GROUP], **tma_copy)
                                    AB_bytes = Tx.meta_var(BLK_M * BLK_K * F8_BYTES + BLK_N * BLK_K * F8_BYTES)
                                    SFAB_bytes = Tx.meta_var((BLK_N * CTA_GROUP + BLK_M) * F32_BYTES)
                                    tma2trans_bar.arrive(ks, Tx.if_then_else(stage % 4 == 0, AB_bytes + SFAB_bytes, AB_bytes))

                            @Tx.inline
                            def tma_load_epilogue(ks):
                                mma2tma_bar.wait(ks, phase ^ 1)
                                with Tx.thread()[Tx.ptx.elect_sync()]:
                                    tma2trans_bar.arrive(ks, 0)

                            paritioned_loop(tma_load, skip, tma_load_epilogue)
                            tile_scheduler.next_tile()

                    with Tx.warp(parent="warpgroup")[warp_id == 2]:
                        # transpose
                        phase = 0
                        # reg_trans = Tx.alloc_buffer((4,), "uint32", scope="local")
                        while tile_scheduler.valid():
                            m_idx = Tx.meta_var(tile_scheduler.m_idx)
                            n_idx = Tx.meta_var(tile_scheduler.n_idx)

                            @Tx.inline
                            def transpose(ks):
                                # wait for sf has been prepared
                                tma2trans_bar.wait(ks, phase)
                                if stage % 4 == 0:
                                    Tx.permute_dims(SFA_smem[ks], [0, 2, 1])
                                    Tx.permute_dims(SFB_smem[ks, :4], [0, 2, 1])
                                    Tx.permute_dims(SFB_smem[ks, 4:], [0, 2, 1])
                                    Tx.ptx.fence.proxy_async("shared::cta")
                                # mark that transpose is completed
                                trans2mma_bar.arrive(ks)

                            @Tx.inline
                            def transpose_epilogue(ks):
                                tma2trans_bar.wait(ks, phase)
                                trans2mma_bar.arrive(ks)

                            paritioned_loop(transpose, skip, transpose_epilogue)
                            tile_scheduler.next_tile()

                    with Tx.warp(parent="warpgroup")[warp_id == 0]:
                        if cbx == 0:
                            tmem_idx = Tx.local_scalar("int32", "tmem_idx")
                            tmem_phase = Tx.local_scalar("int32", "tmem_phase")
                            Tx.ptx.tcgen05.encode_instr_descriptor_block_scaled(Tx.address_of(descI), "float32", a_type, b_type, sfa_type, sfb_type,
                                                                                0, 0, MMA_M, MMA_N, MMA_K, False, False, CTA_GROUP)
                            phase = 0
                            while tile_scheduler.valid():
                                m_idx = Tx.meta_var(tile_scheduler.m_idx)
                                n_idx = Tx.meta_var(tile_scheduler.n_idx)
                                with Tx.thread()[Tx.ptx.elect_sync()]:
                                    tmem_idx = tile_scheduler.tile_idx % TMEM_PIPE_DEPTH
                                    tmem_phase = (tile_scheduler.tile_idx // TMEM_PIPE_DEPTH) & 1

                                    # wait for the tmem result to be consumed
                                    ld2mma_bar.wait(tmem_idx, tmem_phase ^ 1)
                                    Tx.ptx.tcgen05.fence.after_thread_sync()

                                    @Tx.inline
                                    def mma(ks):
                                        # wait tma and sf-transpose arrival
                                        trans2mma_bar.wait(ks, phase)
                                        Tx.ptx.tcgen05.fence.after_thread_sync()

                                        # copy sf to tmem
                                        if stage % 4 == 0:
                                            for ki in Tx.unroll(0, BLK_SFA // 128):
                                                Tx.ptx.tcgen05.encode_matrix_descriptor(
                                                    Tx.address_of(descSFA), SFA_smem.ptr_to([ks, ki * 4, 0]),
                                                    ldo=16, sdo=8 * 4 * F32_BYTES // F128_BYTES, swizzle=0
                                                )
                                                Tx.ptx.tcgen05.cp(0, 0, SFA_TMEM_START_COL + tmem_idx * BLK_SFA // 32 + ki * 4,
                                                                    descSFA, "32x128b", "uint32", "uint32", CTA_GROUP, "warpx4")
                                            for ki in Tx.unroll(0, BLK_SFB // 128):
                                                Tx.ptx.tcgen05.encode_matrix_descriptor(
                                                    Tx.address_of(descSFB), SFB_smem.ptr_to([ks, ki * 4, 0]),
                                                    ldo=16, sdo=8 * 4 * F32_BYTES // F128_BYTES, swizzle=0
                                                )
                                                Tx.ptx.tcgen05.cp(0, 0, SFB_TMEM_START_COL + tmem_idx * BLK_SFB // 32 + ki * 4,
                                                                    descSFB, "32x128b", "uint32", "uint32", CTA_GROUP, "warpx4")

                                        # issue mma
                                        Tx.cuda.runtime_instr_desc(Tx.address_of(descI), stage % 4)
                                        for ki in Tx.unroll(BLK_K // MMA_K):
                                            Tx.ptx.tcgen05.encode_matrix_descriptor(Tx.address_of(descA), A_smem.ptr_to([ks, 0, ki * MMA_K]),
                                                                                ldo=1, sdo=8 * BLK_K * F8_BYTES // F128_BYTES, swizzle=SWIZZLE)
                                            Tx.ptx.tcgen05.encode_matrix_descriptor(Tx.address_of(descB), B_smem.ptr_to([ks, 0, ki * MMA_K]),
                                                                                ldo=1, sdo=8 * BLK_K * F8_BYTES // F128_BYTES, swizzle=SWIZZLE)

                                            if stage == 0 and ki == 0:
                                                Tx.ptx.tcgen05.mma.block_scale("float32", a_type, b_type, sfa_type, sfb_type, tmem_idx * MMA_N, descA, descB,
                                                                                SFA_TMEM_START_COL + tmem_idx * BLK_SFA // 32,
                                                                                SFB_TMEM_START_COL + tmem_idx * BLK_SFB // 32,
                                                                                descI, False, CTA_GROUP, False)
                                            else:
                                                Tx.ptx.tcgen05.mma.block_scale("float32", a_type, b_type, sfa_type, sfb_type, tmem_idx * MMA_N, descA, descB,
                                                                                SFA_TMEM_START_COL + tmem_idx * BLK_SFA // 32,
                                                                                SFB_TMEM_START_COL + tmem_idx * BLK_SFB // 32,
                                                                                descI, False, CTA_GROUP, True)
                                        mma2tma_bar.arrive(ks, CTA_GROUP, 3)

                                    @Tx.inline
                                    def mma_epilogue1():
                                        mma2ld_bar.arrive(tmem_idx, CTA_GROUP, 3)

                                    @Tx.inline
                                    def mma_epilogue2(ks):
                                        trans2mma_bar.wait(ks, phase)
                                        mma2tma_bar.arrive(ks, CTA_GROUP, 3)

                                    paritioned_loop(mma, mma_epilogue1, mma_epilogue2)

                                tile_scheduler.next_tile()

                with Tx.warpgroup()[wg_id == 0]:
                    Tx.cuda.trap_when_assert_failed(tmem_addr == 0)
                    tmem_idx = Tx.local_scalar("int32", "tmem_idx")
                    tmem_phase = Tx.local_scalar("int32", "tmem_phase")
                    phase = 0
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
                        mma2ld_bar.wait(tmem_idx, tmem_phase)
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
                                with Tx.thread():
                                    st = Tx.meta_var(ki * TMEM_LD_SIZE)
                                    Tx.cast(reg_fp16[st : st + TMEM_LD_SIZE], reg[:])
                                    Tx.copy(D_smem[stage, warp_id * 32 + lane_id, st : st + TMEM_LD_SIZE], reg_fp16[st : st + TMEM_LD_SIZE])

                            # the tmem can be overwritten
                            if ko == MMA_N // EPI_TILE - 1:
                                Tx.ptx.tcgen05.fence.before_thread_sync()
                                ld2mma_bar.arrive(tmem_idx)

                            Tx.ptx.fence.proxy_async("shared::cta")
                            Tx.cuda.warpgroup_sync(10)

                            # smem -> gmem
                            m_start: Tx.let = (m_idx * CTA_GROUP + cbx) * BLK_M
                            n_start: Tx.let = n_idx * CTA_GROUP * BLK_N + ko * EPI_TILE
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
                Tx.ptx.tcgen05.dealloc(tmem_addr, n_cols=N_COLS, cta_group=2)

            Tx.cuda.cluster_sync()


# fmt: on


def prepare_data():
    A_origin = torch.randn((M, K), dtype=torch.float32)
    B_origin = torch.randn((N, K), dtype=torch.float32)

    def ceil_to_ue8m0(x: torch.Tensor):
        assert x.view(-1).amax().item() > 0
        return torch.pow(2.0, torch.ceil(torch.log2(x.abs())))

    # Vectorized Quantization
    # For A
    padded_k = ceildiv(K, 128) * 128
    A_padded = torch.empty((M, padded_k), dtype=A_origin.dtype, device=A_origin.device).fill_(0)
    A_padded[:, :K] = A_origin
    A_view = A_padded.view(M, -1, 128)
    A_amax = A_view.abs().float().amax(dim=2).view(M, -1).clamp(1e-4)
    sfa = ceil_to_ue8m0(A_amax / 448.0)
    A_fp8 = (
        (A_view * (1.0 / sfa.unsqueeze(2)))
        .to(torch.float8_e4m3fn)
        .view_as(A_padded)[:, :K]
        .contiguous()
    )
    sfa = sfa.to(torch.float8_e8m0fnu).contiguous()

    # For B
    padded_n = ceildiv(N, 128) * 128
    B_padded = torch.empty(
        (padded_n, padded_k), dtype=B_origin.dtype, device=B_origin.device
    ).fill_(0)
    B_padded[:N, :K] = B_origin
    B_view = B_padded.view(-1, 128, padded_k // 128, 128)
    B_amax = B_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    sfb = ceil_to_ue8m0(B_amax / 448.0)
    B_fp8 = (B_view * (1.0 / sfb)).to(torch.float8_e4m3fn).view_as(B_padded)[:N, :K].contiguous()
    sfb = (
        sfb.to(torch.float8_e8m0fnu)
        .view(B_view.shape[0], B_view.shape[2])
        .repeat(QUANT_SIZE, 1)[:N, :]
        .contiguous()
    )

    # pack the scales
    sfa_pack = sfa.view(torch.uint32).T
    sfb_pack = sfb.view(torch.uint32).T

    # Dequantization
    A_fp8_de = A_fp8.to(torch.float32)
    B_fp8_de = B_fp8.to(torch.float32)
    sfa_de = sfa.to(torch.float32)
    sfb_de = sfb.to(torch.float32)

    # Vectorized dequantization for A
    A_de = (A_fp8_de.reshape(M, K // QUANT_SIZE, QUANT_SIZE) * sfa_de[:, :, None]).reshape(M, K)
    # Vectorized dequantization for B
    B_de = (B_fp8_de.reshape(N, K // QUANT_SIZE, QUANT_SIZE) * sfb_de[:, :, None]).reshape(N, K)

    C_ref = torch.matmul(A_de, B_de.T).to(torch.bfloat16)

    return A_fp8, B_fp8, sfa_pack, sfb_pack, C_ref, A_origin, B_origin


@tvm.testing.requires_cuda_compute_version(10, exact=True)
def test_deepgemm():

    DEV = tvm.cuda(0)
    A_fp8, B_fp8, sfa_pack, sfb_pack, C_ref, A_origin, B_origin = prepare_data()
    A_tvm = tvm.runtime.tensor(
        A_fp8.view(torch.int8).numpy().view(ml_dtypes.float8_e4m3fn), device=DEV
    )
    B_tvm = tvm.runtime.tensor(
        B_fp8.view(torch.int8).numpy().view(ml_dtypes.float8_e4m3fn), device=DEV
    )
    sfa_tvm = tvm.runtime.tensor(sfa_pack.numpy(), device=DEV)
    sfb_tvm = tvm.runtime.tensor(sfb_pack.numpy(), device=DEV)
    C_tvm = torch.empty((M, N), dtype=torch.bfloat16, device="cuda")
    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": deepgemm})
        src, mod = get_source(deepgemm)

    def std():
        a = per_token_cast_to_fp8(A_origin.to(torch.bfloat16).to("cuda"), use_ue8m0=True)
        b = per_block_cast_to_fp8(B_origin.to(torch.bfloat16).to("cuda"), use_ue8m0=True)
        out = torch.empty((M, N), dtype=torch.bfloat16, device="cuda")
        ms = bench(
            lambda: deep_gemm.fp8_gemm_nt(a, b, out, c=None, disable_ue8m0_cast=False, recipe=None),
            warmup=50,
            repeat=50,
            proton_name="std",
        )
        return ms, out

    def tir():
        ms = bench(
            lambda: mod(A_tvm, B_tvm, C_tvm, sfa_tvm, sfb_tvm),
            warmup=50,
            repeat=50,
            proton_name="tir",
        )
        return ms, C_tvm

    # It seems that the tir and std profiling will interfere with each other
    # And also the value of warmup and repeat affect the profiling result abnormally
    # May need to find a better way to do the profiling
    with ProtonContext():
        tir_ms, tir_out = tir()
        print(f"TIR flops: {flops(tir_ms) / 1e12} TFLOPS, time: {tir_ms:.3f} ms")
        std_ms, std_out = std()
        print(f"Std flops: {flops(std_ms) / 1e12} TFLOPS, time: {std_ms:.3f} ms")
        # np.testing.assert_allclose(C_tvm.numpy(), C_ref, rtol=1e-3, atol=1e-2)
        assert calc_diff(std_out, tir_out) < 2e-3
        assert calc_diff(std_out, C_ref.to("cuda")) < 2e-3
        print("Test passed!")


if __name__ == "__main__":
    test_deepgemm()
