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
from enum import IntEnum

import torch
import tvm
import flashinfer
from flashinfer import SfLayout, nvfp4_quantize
import torch.nn.functional as F

from tvm.script import tirx as Tx
from tvm.tirx.bench.utils import ProtonContext, bench

from tvm.tirx.op_schedule.cuda.copy_async import tma_shared_layout, SwizzleMode
from tvm.tirx.op_schedule.cuda.gemm_async import sf_tmem_layout
from tvm.tir.layout import TileLayout, S, TLane, TCol, tid_in_wg as axis_tid_in_wg
from tvm.tirx.tile_scheduler import ClusterPersistentScheduler2D
from tvm.tirx.pipeline import PipelineState, MBarrier, TMABar, TCGen05Bar


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--profile-warmup", type=int, default=100)
    parser.add_argument("--profile-repeat", type=int, default=300)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--nsight", action="store_true")
    return parser.parse_args()


########################################################################
# NVFP4 GEMM
########################################################################


def prepare_data(M: int, N: int, K: int):
    torch.manual_seed(0)
    A_origin = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    B_origin = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    A_global_sf = (448 * 6) / A_origin.float().abs().nan_to_num().max()
    B_global_sf = (448 * 6) / B_origin.float().abs().nan_to_num().max()

    # CUTLASS path expects 128x4 scale layout and no shuffling.
    A_fp4, A_sf = nvfp4_quantize(
        A_origin, A_global_sf, sfLayout=SfLayout.layout_128x4, do_shuffle=False
    )
    B_fp4, B_sf = nvfp4_quantize(
        B_origin, B_global_sf, sfLayout=SfLayout.layout_128x4, do_shuffle=False
    )
    alpha = 1.0 / (A_global_sf * B_global_sf)
    C_ref = torch.mm(A_origin, B_origin.T)

    return A_fp4, B_fp4, A_sf, B_sf, alpha, C_ref


def tir_ws_kernel(M: int, N: int, K: int):
    SM100_TMEM_CAPACITY_COLUMNS = 512
    SM_COUNT = 148  # B200 SM count
    CTA_GROUP = 2  # 2SM cooperative mode for tcgen05 instructions
    CLUSTER_M = 2  # cluster dimension for 2SM pairs
    CLUSTER_N = 1  # cluster dimension along N
    CLUSTER_SIZE = CLUSTER_M * CLUSTER_N
    assert CLUSTER_M % CTA_GROUP == 0

    CTA_M, CTA_N, CTA_K = 128, 128, 256  # single CTA shape
    MMA_M, MMA_N, MMA_K = (
        CTA_M * CTA_GROUP,
        CTA_N * CTA_GROUP,
        64,
    )  # 2SM cooperative MMA shape (256x128)
    SFB_N = math.ceil(MMA_N / 128) * 128
    assert SFB_N in [128, 256]
    MMA_K_BLOCKS = CTA_K // MMA_K

    EPI_TILE = 64
    TMEM_LD_SIZE = 64
    F16_BYTES = 2
    WB_PIPE_DEPTH = 2

    if SFB_N == 128:
        PIPE_DEPTH = 7  # pipeline depth
        TMEM_SFA = MMA_N * 2
        TMEM_SFB = TMEM_SFA + 4
        TMEM_PIPE_DEPTH = 2
    elif SFB_N == 256:
        PIPE_DEPTH = 5  # pipeline depth
        TMEM_SFA = MMA_N * 2 - EPI_TILE  # 2 stages overlap
        TMEM_SFB = TMEM_SFA + 4 * 4
        TMEM_PIPE_DEPTH = 1
    else:
        raise ValueError(f"Unsupported SFB_N: {SFB_N}")

    M_TILES = M // CTA_M
    N_TILES = N // MMA_N
    K_TILES = K // CTA_K
    assert M % CTA_M == 0 and N % MMA_N == 0 and K % CTA_K == 0
    assert M_TILES % CLUSTER_M == 0 and N_TILES % CLUSTER_N == 0

    CLUSTER_M_TILES = M_TILES // CLUSTER_M
    CLUSTER_N_TILES = N_TILES // CLUSTER_N
    NUM_CLUSTERS = SM_COUNT // CLUSTER_SIZE
    GRID_SIZE = NUM_CLUSTERS * CLUSTER_SIZE

    SF_VEC_SIZE = 16
    SF_CTA_K = CTA_K // SF_VEC_SIZE
    SF_K = K // SF_VEC_SIZE
    assert MMA_K == 64 and SF_VEC_SIZE == 16

    assert M % 128 == 0 and N % 128 == 0 and SF_CTA_K % 4 == 0
    assert CTA_M % 128 == 0 and SFB_N % 128 == 0 and SF_CTA_K % 4 == 0

    NUM_WARPS = 8

    # layouts
    A_layout_pipe = tma_shared_layout(
        "uint8", SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, CTA_M, CTA_K // 2)
    )
    B_layout_pipe = tma_shared_layout(
        "uint8", SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, CTA_N, CTA_K // 2)
    )
    # float4 view layouts for gemm_async dispatch (same physical layout as uint8)
    A_layout_fp4 = tma_shared_layout(
        "float4_e2m1fn", SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, CTA_M, CTA_K)
    )
    B_layout_fp4 = tma_shared_layout(
        "float4_e2m1fn", SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, CTA_N, CTA_K)
    )
    SFA_layout_pipe = TileLayout(S[PIPE_DEPTH, CTA_M // 128, SF_CTA_K // 4, 256])
    SFB_layout_pipe = TileLayout(S[PIPE_DEPTH, SFB_N // 128, SF_CTA_K // 4, 256])
    D_layout_wb = tma_shared_layout(
        "bfloat16", SwizzleMode.SWIZZLE_128B_ATOM, (WB_PIPE_DEPTH, CTA_M, EPI_TILE)
    )
    # layouts for descriptor generation (used by tcgen05.cp for SF→TMEM copy)
    SFA_layout_desc = TileLayout(S[PIPE_DEPTH, CTA_M // 128, SF_CTA_K // 4, 32, 16]).pack(16)
    SFB_layout_desc = TileLayout(S[PIPE_DEPTH, SFB_N // 128, SF_CTA_K // 4, 32, 16]).pack(16)

    A_BYTES = (CTA_M * CTA_K // 2) * CTA_GROUP
    B_BYTES = (CTA_N * CTA_K // 2) * CTA_GROUP
    SFA_BYTES = (CTA_M * SF_CTA_K) * CTA_GROUP
    SFB_BYTES = (SFB_N * SF_CTA_K) * CTA_GROUP

    class WarpRole(IntEnum):
        MMA = 0
        TMA = 2
        EPILOGUE = 4

    print(
        f"using ws kernel, cluster shape: {CLUSTER_M}x{CLUSTER_N} num clusters: {NUM_CLUSTERS} grid size: {GRID_SIZE} pipeline depth: {PIPE_DEPTH} Tile shape: {CTA_M}x{CTA_N}x{CTA_K}"
    )

    @Tx.meta_class
    class SmemDescriptor:
        def __init__(self, prefix: str):
            self.desc = Tx.local_cell("uint64", name=prefix + "sdesc")

        @Tx.macro
        def init(self, smem_ptr, ldo, sdo, swizzle):
            Tx.ptx.tcgen05.encode_matrix_descriptor(
                Tx.address_of(self.desc), smem_ptr, ldo, sdo, swizzle
            )

        def add_16B_offset(self, offset):
            func_name = "tvm_builtin_smem_desc_add_16B_offset"
            source_code = f"""
__forceinline__ __device__ uint64_t {func_name}(uint64_t desc_base, int32_t offset) {{
    SmemDescriptor desc;
    desc.desc_ = desc_base;
    desc.lo += static_cast<uint32_t>(offset);
    return desc.desc_;
}}
"""
            return Tx.cuda.func_call(
                func_name, self.desc, offset, source_code=source_code, return_type="uint64"
            )

    warp_id_func_source = R"""
__forceinline__ __device__ int canonical_warp_idx_sync() {
    return __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
}
"""

    copy_128b_source_code = R"""
__forceinline__ __device__ void tvm_builtin_copy_128b(void* dst_ptr, void* src_ptr) {
    uint4* src_ = reinterpret_cast<uint4*>(src_ptr);
    uint4* dst_ = reinterpret_cast<uint4*>(dst_ptr);
    *dst_ = *src_;
}
"""

    pointer_offset_source_code = R"""
template <typename T>
__forceinline__ __device__ T* tvm_builtin_pointer_offset(T* ptr, int offset) {
    return ptr + offset;
}
"""

    # fmt: off
    @Tx.macro
    def copy_128b(src, dst):
        Tx.cuda.func_call(
            "tvm_builtin_copy_128b", src, dst, source_code=copy_128b_source_code, return_type="void"
        )
    # fmt: off

    def pointer_offset(ptr, offset):
        return Tx.cuda.func_call(
            "tvm_builtin_pointer_offset", ptr, offset, source_code=pointer_offset_source_code, return_type="handle"
        )

    @Tx.meta_class
    class RowiseSwizzleOffset:
        def __init__(self, swizzle_len, atom_len, per_element, row_base, prefix: str = "row_sw_offset"):
            self.swizzle_len = swizzle_len
            self.atom_len = atom_len
            self.per_element = per_element
            self.row_base = row_base
            self.signed_strides = Tx.alloc_buffer(
                [self.atom_len], "int32", scope="local", name=prefix + "_sign"
            )
            self.n_dim = self.swizzle_len + 1
            self.shape = [2] * self.n_dim
            self.shape[-1] = 1 << self.per_element

        @Tx.macro
        def init(self):
            for i in Tx.unroll(self.swizzle_len):
                y_i = Tx.meta_var(self.row_base & (1 << i))
                stride_i = Tx.meta_var(1 << (i + self.per_element))
                self.signed_strides[i] = -stride_i if y_i > 0 else stride_i
            for i in Tx.unroll(self.swizzle_len, self.atom_len):
                stride_i = Tx.meta_var(1 << (i + self.per_element))
                self.signed_strides[i] = stride_i

        def apply(self, offset):
            offset_layout = TileLayout(
                S[self.shape : [self.signed_strides[self.swizzle_len - 1 - i] for i in range(self.atom_len)] + [0]]
            )
            return offset_layout.apply(offset)["m"]

    # fmt: off
    @Tx.prim_func(tirx=True)
    def kernel(A_packed: Tx.Buffer((M, K // 2), "uint8"), B_packed: Tx.Buffer((N, K // 2), "uint8"), SFA_in: Tx.Buffer((M, SF_K), "uint8"), SFB_in: Tx.Buffer((N, SF_K), "uint8"), alpha: Tx.Buffer((1,), "float32"), D: Tx.Buffer((M, N), "bfloat16")):
        SFA_gmem = Tx.decl_buffer((M // 128, SF_K // 4, 256), "uint16", data=SFA_in.data, scope="global")
        SFB_gmem = Tx.decl_buffer((N // 128, SF_K // 4, 256), "uint16", data=SFB_in.data, scope="global")

        with Tx.kernel():
            cluster_rank_ = Tx.cta_id([CLUSTER_SIZE], parent="cluster")
            cta_idx = Tx.cta_id([GRID_SIZE], parent="kernel")
            warp_id_ = Tx.warp_id([NUM_WARPS], parent="cta")
            lane_id = Tx.thread_id([32], parent="warp")
            tid_in_wg = Tx.thread_id([128], parent="warpgroup")
            
            warp_id = Tx.local_cell("int32")
            warp_id = Tx.cuda.func_call("canonical_warp_idx_sync", return_type="int32", source_code=warp_id_func_source)

            cluster_rank = Tx.local_cell("int32")
            cluster_rank = cluster_rank_
            # General pair calculations - works for any cluster shape
            cb_m = cluster_rank % CLUSTER_M
            cb_n = cluster_rank // CLUSTER_M
            pair_id = cluster_rank // CTA_GROUP  # which 2SM pair
            id_in_pair = cluster_rank % CTA_GROUP  # 0=leader, 1=follower
            pair_leader_rank = pair_id * CTA_GROUP  # rank of pair's leader

            # tile scheduler
            tile_scheduler = ClusterPersistentScheduler2D("tile_scheduler", num_m_tiles=CLUSTER_M_TILES, num_n_tiles=CLUSTER_N_TILES, num_clusters=NUM_CLUSTERS)
            tile_scheduler.init(cta_idx // CLUSTER_SIZE)
            m_idx = Tx.meta_var(tile_scheduler.m_idx)
            n_idx = Tx.meta_var(tile_scheduler.n_idx)

            ############################ SMEM allocation #################################
            pool = Tx.PoolAllocator()
            tmem_addr = pool.alloc([1], "uint32", align=4)
            # Pipeline mbarriers: one per stage
            tma_full = TMABar(pool, PIPE_DEPTH, "tma_full")
            tma_empty = TCGen05Bar(pool, PIPE_DEPTH, "tma_empty")
            acc_full = TCGen05Bar(pool, TMEM_PIPE_DEPTH, "acc_full")
            acc_empty = MBarrier(pool, TMEM_PIPE_DEPTH, "acc_empty")
            # Pipelined SMEM buffers
            A_smem_packed = pool.alloc((PIPE_DEPTH, CTA_M, CTA_K // 2), "uint8", layout=A_layout_pipe, align=1024)
            B_smem_packed = pool.alloc((PIPE_DEPTH, CTA_N, CTA_K // 2), "uint8", layout=B_layout_pipe, align=1024)
            SFA_smem = pool.alloc((PIPE_DEPTH, CTA_M // 128, SF_CTA_K // 4, 256), "uint16", layout=SFA_layout_pipe, align=1024)
            SFB_smem = pool.alloc((PIPE_DEPTH, SFB_N // 128, SF_CTA_K // 4, 256), "uint16", layout=SFB_layout_pipe, align=1024)
            output_smem = pool.alloc((WB_PIPE_DEPTH, CTA_M, EPI_TILE), "bfloat16", layout=D_layout_wb, align=1024)
            pool.commit()
            tma_full.init(1)
            tma_empty.init(1)
            acc_full.init(1)
            acc_empty.init(CTA_GROUP * 128)

            ############################ TMEM allocation #################################
            with Tx.warp()[warp_id == 0]:
                Tx.ptx.tcgen05.alloc(Tx.address_of(tmem_addr[0]), n_cols=SM100_TMEM_CAPACITY_COLUMNS, cta_group=CTA_GROUP)
                Tx.cuda.warp_sync()
            tmem = Tx.decl_buffer((CTA_M, SM100_TMEM_CAPACITY_COLUMNS), "float32", scope="tmem", allocated_addr=0, layout=TileLayout(S[(CTA_M, SM100_TMEM_CAPACITY_COLUMNS) : (1@TLane, 1@TCol)]))
            # float4 views for gemm_async dispatch (share data with uint8 packed buffers)
            # Must pass elem_offset to preserve pool allocation offset (uint8 elem_offset = byte offset,
            # fp4 has 2 elements per byte, so fp4 elem_offset = uint8 elem_offset * 2)
            A_smem = Tx.decl_buffer((PIPE_DEPTH, CTA_M, CTA_K), "float4_e2m1fn", data=A_smem_packed.data, elem_offset=A_smem_packed.elem_offset * 2, scope="shared.dyn", layout=A_layout_fp4)
            B_smem = Tx.decl_buffer((PIPE_DEPTH, CTA_N, CTA_K), "float4_e2m1fn", data=B_smem_packed.data, elem_offset=B_smem_packed.elem_offset * 2, scope="shared.dyn", layout=B_layout_fp4)
            # SFA/SFB TMEM buffers for gemm_async dispatch (atom ⊕ outer layout)
            SFB_CHUNK = Tx.meta_var(4 if SFB_N == 128 else 8)
            sf_mma_k = Tx.meta_var(4)  # nvfp4: MMA_K=64, SF_VEC_SIZE=16, sf_mma_k=4
            SFB_n_chunks = Tx.meta_var(SFB_N // 128)  # 1 for SFB_N=128, 2 for SFB_N=256
            SFA_tmem = Tx.decl_buffer((128, sf_mma_k * MMA_K_BLOCKS), "float8_e4m3fn", scope="tmem", allocated_addr=TMEM_SFA, layout=sf_tmem_layout(128, sf_mma_k, MMA_K_BLOCKS, dtype="float8_e4m3fn"))
            SFB_tmem = Tx.decl_buffer((128 * SFB_n_chunks, sf_mma_k * MMA_K_BLOCKS), "float8_e4m3fn", scope="tmem", allocated_addr=TMEM_SFB, layout=sf_tmem_layout(128 * SFB_n_chunks, sf_mma_k, MMA_K_BLOCKS, dtype="float8_e4m3fn"))

            Tx.ptx.fence.proxy("shared")
            Tx.ptx.fence.mbarrier_init()
            Tx.cuda.cluster_sync()

            # Get pointer to pair leader's mbar_tma array for TMA arrive
            tma_full_cta0 = tma_full.remote_view(pair_leader_rank)

            # descriptor cells
            descSFA = SmemDescriptor("SFA")
            descSFB = SmemDescriptor("SFB")

            # The mask containing the pair leader and follower in the cluster
            pair_mask = Tx.local_cell("int32")
            pair_mask = 0
            pair_mask = pair_mask | (1 << pair_leader_rank)
            pair_mask = pair_mask | (1 << (pair_leader_rank + 1))

            tma_state = PipelineState("state", PIPE_DEPTH)
            tma_state.init(is_producer=True)
            mma_state = PipelineState("state", PIPE_DEPTH)
            mma_state.init(is_producer=False)
            acc_state = PipelineState("acc", TMEM_PIPE_DEPTH)
            acc_state.init(is_producer=True)
            accum = Tx.local_cell("int32")
            accum = 0
            epi_state = PipelineState("epi", TMEM_PIPE_DEPTH)
            epi_state.init(is_producer=False)
            epi_wb_state = PipelineState("epi_wb", WB_PIPE_DEPTH)
            epi_wb_state.init(is_producer=True)

            alpha_local = Tx.local_cell("float32")
            alpha_local = alpha[0]

            # TMA warp
            if warp_id == int(WarpRole.TMA):
                while tile_scheduler.valid():
                    cta_m = m_idx * CLUSTER_M + cb_m
                    cta_n = n_idx * CLUSTER_N + cb_n
                    m_st = cta_m * CTA_M
                    n_st = cta_n * MMA_N + id_in_pair * CTA_N
                    out_n_st = cta_n * MMA_N

                    @Tx.macro
                    def issue_tma_load(stage, phase, k_tile: Tx.int32):
                        k_st = Tx.meta_var(k_tile * CTA_K // 2)
                        sfk_st = Tx.meta_var(k_tile * SF_CTA_K)
                        sfk_st_tile = Tx.meta_var(sfk_st // 4)
                        m_st_tile = Tx.meta_var(m_st // 128)
                        n_st_tile = Tx.meta_var(out_n_st // 128)

                        # wait for current stage to be empty
                        tma_empty.wait(stage, phase)
                        # issue tma load
                        tma_config = Tx.meta_var({"dispatch": "tma", "cta_group": CTA_GROUP, "mbar": tma_full_cta0.ptr_to([stage]), "cache_hint": "evict_normal"})
                        with Tx.thread()[Tx.ptx.elect_sync()]:
                            Tx.copy_async(A_smem_packed[stage, :, :], A_packed[m_st : m_st + CTA_M, k_st : k_st + CTA_K // 2], **tma_config)
                            Tx.copy_async(B_smem_packed[stage, :, :], B_packed[n_st : n_st + CTA_N, k_st : k_st + CTA_K // 2], **tma_config)
                            Tx.copy_async(SFA_smem[stage, :, :, :], SFA_gmem[m_st_tile : m_st_tile + CTA_M // 128, sfk_st_tile : sfk_st_tile + SF_CTA_K // 4, :], **tma_config)
                            if SFB_N == 128:
                                Tx.copy_async(SFB_smem[stage, :, :, :], SFB_gmem[n_st_tile : n_st_tile + SFB_N // 128, sfk_st_tile : sfk_st_tile + SF_CTA_K // 4, :], **tma_config)
                            else:
                                tma_multicast_config = Tx.meta_var({"dispatch": "tma", "cta_group": CTA_GROUP, "mbar": tma_full_cta0.ptr_to([stage]), "cta_mask": pair_mask, "cache_hint": "evict_normal"})
                                Tx.copy_async(SFB_smem[stage, cb_m: cb_m + 1 :, :], SFB_gmem[n_st_tile + cb_m : n_st_tile + cb_m + 1, sfk_st_tile : sfk_st_tile + SF_CTA_K // 4, :], **tma_multicast_config)
                            # signal tma is issued to pair leader
                            if id_in_pair == 0:
                                total_bytes = Tx.meta_var(A_BYTES + B_BYTES + SFA_BYTES + SFB_BYTES)
                                tma_full_cta0.arrive(stage, total_bytes)

                    for k_tile in Tx.serial(K_TILES):
                        issue_tma_load(tma_state.stage, tma_state.phase, k_tile)
                        tma_state.move_to_next_stage()

                    tile_scheduler.next_tile()

            elif warp_id == int(WarpRole.MMA) and id_in_pair == 0:
                descSFA.init(smem_ptr=SFA_smem.ptr_to([0, 0, 0, 0]), ldo=1, sdo=8, swizzle=0)
                descSFB.init(smem_ptr=SFB_smem.ptr_to([0, 0, 0, 0]), ldo=1, sdo=8, swizzle=0)

                while tile_scheduler.valid():
                    @Tx.macro
                    def execute_mma(stage, phase):
                        # wait for tma to finish
                        tma_full.wait(stage, phase)
                        Tx.ptx.tcgen05.fence.after_thread_sync()

                        with Tx.thread()[Tx.ptx.elect_sync()]:
                            # move sf to tmem
                            for k_block in Tx.unroll(MMA_K_BLOCKS):
                                k_offset = Tx.meta_var(k_block * MMA_K // 2)
                                Tx.ptx.tcgen05.cp(0, 0, TMEM_SFA + k_block * 4, descSFA.add_16B_offset(SFA_layout_desc.apply(stage, 0, k_block, 0, 0 )["m"]), "32x128b", "uint8", "uint8", CTA_GROUP, "warpx4")
                                Tx.ptx.tcgen05.cp(0, 0, TMEM_SFB + k_block * SFB_CHUNK, descSFB.add_16B_offset(SFB_layout_desc.apply(stage, 0, k_block, 0, 0)["m"]), "32x128b", "uint8", "uint8", CTA_GROUP, "warpx4")
                                if SFB_N == 256:
                                    Tx.ptx.tcgen05.cp(0, 0, TMEM_SFB + k_block * SFB_CHUNK + 4, descSFB.add_16B_offset(SFB_layout_desc.apply(stage, 1, k_block, 0, 0)["m"]), "32x128b", "uint8", "uint8", CTA_GROUP, "warpx4")

                            # issue mma
                            tmem_acc = Tx.meta_var(acc_state.stage * MMA_N if MMA_N == 128 else (acc_state.phase ^ 1) * (MMA_N - EPI_TILE))
                            Tx.gemm_async(tmem[:, tmem_acc: tmem_acc + MMA_N], A_smem[stage, :, :], B_smem[stage, :, :], SFA=SFA_tmem[:, :], SFB=SFB_tmem[:, :], accum=accum, dispatch="tcgen05", cta_group=CTA_GROUP)
                            accum = 1

                            # signal mma is issued to both CTAs in the pair
                            tma_empty.arrive(stage, cta_group=CTA_GROUP, cta_mask=pair_mask)

                    acc_empty.wait(acc_state.stage, acc_state.phase)
                    accum = 0
                    for k_tile in Tx.serial(K_TILES):
                        execute_mma(mma_state.stage, mma_state.phase)
                        mma_state.move_to_next_stage()
                    
                    with Tx.thread()[Tx.ptx.elect_sync()]:
                        acc_full.arrive(acc_state.stage, cta_group=CTA_GROUP, cta_mask=pair_mask)
                    acc_state.move_to_next_stage()

                    tile_scheduler.next_tile()

            elif warp_id >= int(WarpRole.EPILOGUE):
                row_sw_offset = RowiseSwizzleOffset(3, 3, 3, tid_in_wg) # D's swizzle settings
                row_sw_offset.init()

                while tile_scheduler.valid():
                    cta_m = m_idx * CLUSTER_M + cb_m
                    cta_n = n_idx * CLUSTER_N + cb_n
                    m_st = cta_m * CTA_M
                    out_n_st = cta_n * MMA_N
                    
                    @Tx.macro
                    def epilogue(stage, phase):
                        # wait for accumulator to finish
                        acc_full.wait(stage, phase)
                        reg = Tx.alloc_buffer((TMEM_LD_SIZE,), "float32", scope="local")
                        reg_16b = Tx.alloc_buffer((EPI_TILE,), "bfloat16", scope="local")

                        for no in Tx.unroll(MMA_N // EPI_TILE):
                            col_st = Tx.local_cell("int32")
                            if SFB_N == 128:
                                col_st = epi_state.stage * MMA_N + no * EPI_TILE
                            elif SFB_N == 256:
                                if epi_state.phase == 0:
                                    col_st = MMA_N - (no + 1) * EPI_TILE
                                else:
                                    col_st = MMA_N - EPI_TILE + no * EPI_TILE
                                    
                            cast_layout = TileLayout(S[(CTA_M, TMEM_LD_SIZE) : (1@axis_tid_in_wg, 1)])
                            for ni in Tx.unroll(EPI_TILE // TMEM_LD_SIZE):
                                with Tx.warpgroup():
                                    # TMEM -> RF
                                    reg_wg = reg.view(CTA_M, TMEM_LD_SIZE, layout=cast_layout)
                                    Tx.copy(reg_wg[:, :], tmem[:, col_st + ni * TMEM_LD_SIZE : col_st + (ni + 1) * TMEM_LD_SIZE])
                                    # multiply by alpha
                                    for v in range(TMEM_LD_SIZE):
                                        reg[v] = reg[v] * alpha_local
                                    # cast fp32 -> bf16 via dispatch (vec2)
                                    reg_16b_wg = reg_16b.view(CTA_M, TMEM_LD_SIZE, layout=cast_layout)
                                    Tx.cast(reg_16b_wg[:, :], reg_wg[:, :])

                            singal_iter = Tx.meta_var(no == MMA_N // EPI_TILE - 1 if SFB_N == 128 else no == 0)
                            if singal_iter:
                                # both CTAs in the pair singal mma warp (of leader) to start mma
                                acc_empty.arrive(stage, cta_id=pair_leader_rank, pred=True)

                            # RF -> SMEM
                            with Tx.thread()[tid_in_wg == 0]:
                                Tx.ptx.cp_async.bulk.wait_group(WB_PIPE_DEPTH - 1)
                            Tx.cuda.warpgroup_sync(10)
                            # with Tx.thread():
                            #     Tx.copy(output_smem[epi_wb_state.stage, tid_in_wg, :], reg_16b[:])
                            row_st = Tx.local_cell("int32")
                            row_st = output_smem.elem_offset_of([epi_wb_state.stage, tid_in_wg, 0])
                            for ni in Tx.unroll(EPI_TILE // 8):
                                copy_128b(pointer_offset(output_smem.ptr_to([0, 0, 0]), row_st + row_sw_offset.apply(ni * 8)), reg_16b.ptr_to([ni * 8]))
                            Tx.ptx.fence.proxy(scope="shared")
                            Tx.cuda.warpgroup_sync(10)
                            
                            # launch tma copy from smem to gmem
                            out_n = Tx.local_cell("int32")
                            if SFB_N == 128:
                                out_n = out_n_st + no * EPI_TILE
                            elif SFB_N == 256:
                                if epi_state.phase == 0:
                                    out_n = out_n_st + MMA_N - (no + 1) * EPI_TILE
                                else:
                                    out_n = out_n_st + no * EPI_TILE
                            with Tx.thread()[tid_in_wg == 0]:
                                Tx.copy_async(D[m_st : m_st + CTA_M, out_n : out_n + EPI_TILE], output_smem[epi_wb_state.stage, :, :], dispatch="tma", cache_hint="evict_normal")
                                Tx.ptx.cp_async.bulk.commit_group()
                            epi_wb_state.move_to_next_stage()

                    epilogue(epi_state.stage, epi_state.phase)
                    epi_state.move_to_next_stage()
                    
                    tile_scheduler.next_tile()
            
                with Tx.thread()[tid_in_wg == 0]:
                    Tx.ptx.cp_async.bulk.wait_group(0)
                Tx.cuda.warpgroup_sync(10)

            # Cleanup after all tiles processed
            with Tx.warp()[warp_id == 0]:
                Tx.ptx.tcgen05.relinquish_alloc_permit(cta_group=CTA_GROUP)
                Tx.ptx.tcgen05.dealloc(0, n_cols=SM100_TMEM_CAPACITY_COLUMNS, cta_group=CTA_GROUP)
    # fmt: on

    return kernel


def tir_gemm(
    A_fp4, B_fp4, A_sf, B_sf, alpha, C_ref, kernel, warmup, repeat, debug=False, nsight=False
):
    A_fp4, B_fp4, A_sf, B_sf, alpha, C_ref = (
        A_fp4.clone(),
        B_fp4.clone(),
        A_sf.clone(),
        B_sf.clone(),
        alpha.clone(),
        C_ref.clone(),
    )
    alpha = torch.tensor([alpha.item()], device="cuda", dtype=torch.float)
    out = torch.empty_like(C_ref).to("cuda").to(torch.bfloat16)
    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": kernel})
        ex = tvm.compile(mod, target=target, tir_pipeline="tirx")
        bench(
            lambda: ex.mod(A_fp4, B_fp4, A_sf, B_sf, alpha, out),
            warmup=warmup,
            repeat=repeat,
            proton_name="tir",
            debug=debug,
            nsight=nsight,
            flush_l2_size=int(8e9 // 4),
        )
    return out


def flashinfer_gemm(
    A_fp4, B_fp4, A_sf, B_sf, alpha, C_ref, warmup, repeat, debug=False, nsight=False
):
    A_fp4, B_fp4, A_sf, B_sf, alpha, C_ref = (
        A_fp4.clone(),
        B_fp4.clone().T,
        A_sf.clone(),
        B_sf.clone().T,
        alpha.clone(),
        C_ref.clone(),
    )
    out = torch.empty_like(C_ref).to("cuda").to(torch.bfloat16)
    with flashinfer.autotune(False):
        func = lambda: flashinfer.gemm.mm_fp4(
            A_fp4,
            B_fp4,
            A_sf,
            B_sf,
            alpha,
            torch.bfloat16,
            out,
            block_size=16,
            backend="cutlass",
            use_nvfp4=True,
        )
        bench(
            func,
            warmup=warmup,
            repeat=repeat,
            proton_name="flashinfer",
            debug=debug,
            nsight=nsight,
            flush_l2_size=int(8e9 // 4),
        )
    return out


def profile_gemm(
    M: int,
    N: int,
    K: int,
    kernel,
    warmup: int,
    repeat: int,
    debug: bool = False,
    nsight: bool = False,
):
    A_fp4, B_fp4, A_sf, B_sf, alpha, C_ref = prepare_data(M, N, K)

    with ProtonContext("gemm", debug=debug, nsight=nsight):
        C_tir = tir_gemm(
            A_fp4, B_fp4, A_sf, B_sf, alpha, C_ref, kernel, warmup, repeat, debug, nsight
        )
        C_flashinfer = flashinfer_gemm(
            A_fp4, B_fp4, A_sf, B_sf, alpha, C_ref, warmup, repeat, debug, nsight
        )

    cosine_sim = F.cosine_similarity(C_tir.reshape(-1), C_ref.to("cuda").reshape(-1), dim=0)
    print(f"TIR vs Reference cosine similarity: {cosine_sim.item():.6f}")
    assert cosine_sim > 0.97

    cosine_sim = F.cosine_similarity(C_flashinfer.reshape(-1), C_tir.reshape(-1), dim=0)
    print(f"Flashinfer vs TIR cosine similarity: {cosine_sim.item():.6f}")
    assert cosine_sim > 0.97


if __name__ == "__main__":
    args = parse_args()
    kernel = tir_ws_kernel(args.m, args.n, args.k)
    profile_gemm(
        args.m,
        args.n,
        args.k,
        kernel,
        args.profile_warmup,
        args.profile_repeat,
        args.debug,
        args.nsight,
    )
