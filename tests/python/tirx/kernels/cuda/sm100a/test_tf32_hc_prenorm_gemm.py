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

"""
TF32 HC Prenorm GEMM kernel (DeepGEMM sm100_tf32_hc_prenorm_gemm transcription).

D[M,N] = A[M,K] (bf16) @ B[N,K]^T (f32) -> f32
sqr_sum[M] = sum(A^2, dim=1) (for RMSNorm)

Architecture: 256 threads = 2 warpgroups
  WG0 (128 threads): Cast warps - load bf16 A from SMEM, cast to f32, store to TMEM, accumulate sqr_sum
  WG1 (128 threads): TMA + MMA + Epilogue
    warp 3: TMA load (A bf16, B f32 into SMEM)
    warp 1: MMA issue (TS mode - A from TMEM, B from SMEM)
    all warps: Epilogue after K-loop (TMEM -> SMEM -> TMA store)
"""

import numpy as np
import ml_dtypes

import tvm
import tvm.testing
from tvm.script import tirx as Tx
from tvm.tirx.bench.utils import bench, ProtonContext
from tvm.tirx.pipeline import MBarrier, TMABar, TCGen05Bar
from tvm.tir.layout import TileLayout, TLane, TCol, S
from tvm.tir.layout import tid_in_wg as axis_tid_in_wg

import torch
import deep_gemm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WG_NUMBER = 2
WARP_NUMBER = 4
NUM_THREADS = (32 * WARP_NUMBER) * WG_NUMBER  # 256
SM_NUMBER = 148

a_type = tvm.DataType("bfloat16")
b_type = tvm.DataType("float32")
d_type = tvm.DataType("float32")

M, N, K = 4096, 32, 7168
BLOCK_M, BLOCK_N, BLOCK_K = 64, 32, 64
UMMA_K = 8  # 32B / 4B per tf32 element
kNumStages = 12
kNumCastStages = 2
kNumSplits = 2
CTA_GROUP = 1

F16_BYTES = 2
F32_BYTES = 4
F128_BYTES = 16

# TMEM layout: cast A double-buffer + D output
# [0, 64):    Cast A stage 0 (f32, BLOCK_K=64 cols)
# [64, 128):  Cast A stage 1 (f32, BLOCK_K=64 cols)
# [128, 160): D MMA output (f32, BLOCK_N=32 cols)
D_TMEM_START_COL = BLOCK_K * kNumCastStages  # 128
N_COLS = 256  # next power of 2 >= 128+32=160

NUM_K_ITERS = K // BLOCK_K  # total K blocks
NUM_K_ITERS_PER_SPLIT = NUM_K_ITERS // kNumSplits
NUM_MMA_K_ITERS = BLOCK_K // UMMA_K  # 8
assert NUM_K_ITERS % kNumSplits == 0, f"K blocks ({NUM_K_ITERS}) must be evenly divisible by kNumSplits ({kNumSplits})"

# Swizzle modes (per_element = log2(bank_group_elements), swizzle_len=3, atom_len=3)
# A (bf16): 128B swizzle, 16B/2B=8 elements/bg -> SwizzleLayout(3,3,3)
# B (f32):  128B swizzle, 16B/4B=4 elements/bg -> SwizzleLayout(2,3,3)
# D (f32):  128B swizzle, 16B/4B=4 elements/bg -> SwizzleLayout(2,3,3)
SWIZZLE_A = 3  # 128B swizzle mode for UMMA descriptor
SWIZZLE_B = 3  # 128B swizzle mode for UMMA descriptor

BLOCK_SWIZZLED_BK_B = 128 // F32_BYTES  # 32 f32 elements in 128B
BLOCK_K_ATOM_B = 128 // F32_BYTES  # 32 f32 = 128B, TMA atom size for B
NUM_B_TMA_ATOMS = BLOCK_K // BLOCK_K_ATOM_B  # 2 TMA loads per B tile

# Epilogue params
EPI_TILE = BLOCK_N  # 32 — single pass, no tiling needed
TMEM_LD_SIZE = 8

assert BLOCK_K * F16_BYTES == 128, "A swizzle requires BLOCK_K*sizeof(bf16)==128"
assert BLOCK_N * F32_BYTES == 128, "D swizzle requires BLOCK_N*sizeof(f32)==128"

# SMEM sizes
SMEM_A_SIZE_PER_STAGE = BLOCK_M * BLOCK_K * F16_BYTES   # 8192
SMEM_B_SIZE_PER_STAGE = BLOCK_N * BLOCK_K * F32_BYTES   # 8192
SMEM_D_SIZE = BLOCK_M * BLOCK_N * F32_BYTES             # 8192

BLOCK_M_PER_WARP = BLOCK_M // 4  # 16 rows per sub-warp in cast warp

# Raw byte offset from SMEM base to A_smem stage 0:
# Layout: [tmem_addr(1024B)][D_smem(8192B)][A_smem stages...]
A_SMEM_BYTE_BASE = 1024 + SMEM_D_SIZE  # 9216

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_source(func):
    target = tvm.target.Target("cuda")
    mod = tvm.IRModule({"main": func})
    mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
    src = mod.mod.imports[0].inspect_source()
    return src, mod



def calc_diff(x, y):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


# ---------------------------------------------------------------------------
# Barrier classes (CTA_GROUP=1, no cluster)
# ---------------------------------------------------------------------------


@Tx.inline
def skip():
    pass


# ---------------------------------------------------------------------------
# SmemDescriptor helper (from flash_attention4)
# ---------------------------------------------------------------------------

@Tx.meta_class
class SmemDescriptor:
    def __init__(self, prefix):
        self.desc = Tx.local_scalar("uint64", name=prefix + "_sdesc")

    @Tx.inline
    def init(self, smem_ptr, ldo, sdo, swizzle):
        Tx.ptx.tcgen05.encode_matrix_descriptor(
            Tx.address_of(self.desc), smem_ptr, ldo, sdo, swizzle
        )
        self._make_lo_uniform()

    def _make_lo_uniform(self):
        func_name = "smem_desc_make_lo_uniform"
        source_code = f"""
__forceinline__ __device__ void {func_name}(uint64_t* desc) {{
    SmemDescriptor* d = reinterpret_cast<SmemDescriptor*>(desc);
    d->lo = __shfl_sync(0xffffffff, d->lo, 0);
}}
"""
        return Tx.cuda.func_call(
            func_name, Tx.address_of(self.desc),
            source_code=source_code, return_type="void"
        )

    def add_16B_offset(self, offset):
        func_name = "smem_desc_add_16B_offset"
        source_code = f"""
__forceinline__ __device__ uint64_t {func_name}(uint64_t desc_base, int32_t offset) {{
    SmemDescriptor desc;
    desc.desc_ = desc_base;
    desc.lo += static_cast<uint32_t>(offset);
    return desc.desc_;
}}
"""
        return Tx.cuda.func_call(
            func_name, self.desc, offset,
            source_code=source_code, return_type="uint64"
        )


# ---------------------------------------------------------------------------
# SMEM layouts
# ---------------------------------------------------------------------------

A_layout = Tx.ComposeLayout(
    Tx.SwizzleLayout(3, 3, 3, swizzle_inner=True),
    Tx.TileLayout(S[(kNumStages, BLOCK_M, BLOCK_K) : (BLOCK_M * BLOCK_K, BLOCK_K, 1)]),
)
B_layout = Tx.ComposeLayout(
    Tx.SwizzleLayout(2, 3, 3, swizzle_inner=True),
    Tx.TileLayout(S[(kNumStages, NUM_B_TMA_ATOMS, BLOCK_N, BLOCK_K_ATOM_B) :
                    (NUM_B_TMA_ATOMS * BLOCK_N * BLOCK_K_ATOM_B, BLOCK_N * BLOCK_K_ATOM_B, BLOCK_K_ATOM_B, 1)]),
)
D_layout = Tx.ComposeLayout(
    Tx.SwizzleLayout(2, 3, 3, swizzle_inner=True),
    Tx.TileLayout(S[(BLOCK_M, BLOCK_N) : (BLOCK_N, 1)]),
)


# ---------------------------------------------------------------------------
# Custom bf16→f32 cast + SMEM→TMEM store helper (via inline PTX)
# ---------------------------------------------------------------------------

cast_and_store_src = """
__forceinline__ __device__ void cast_bf16_store_tmem(
    void* smem_base_ptr, uint32_t smem_byte_offset,
    uint32_t tmem_addr, uint32_t row_offset, uint32_t tmem_col,
    uint32_t sub_warp_idx, uint32_t lane_idx,
    float* sum0_x, float* sum0_y, float* sum1_x, float* sum1_y,
    void* empty_cast_mbar_ptr, uint32_t empty_cast_phase
) {
    // smem_base = smem_base_ptr + smem_byte_offset (un-swizzled base for this stage)
    void* smem_base = reinterpret_cast<uint8_t*>(smem_base_ptr) + smem_byte_offset;

    constexpr uint32_t kSwizzleAMode = 128;  // bytes
    constexpr uint32_t BLOCK_K_LOCAL = 64;
    constexpr uint32_t kNumBankGroupBytes = 16;
    constexpr uint32_t kNumElemsPerBankGroup = kNumBankGroupBytes / sizeof(nv_bfloat16);  // 8
    constexpr uint32_t kNumLoads = BLOCK_K_LOCAL / kNumElemsPerBankGroup;  // 8

    uint8_t* smem_stage_ptr = reinterpret_cast<uint8_t*>(smem_base) +
                              sub_warp_idx * 16 * kSwizzleAMode;

    // Phase 1: Load from SMEM using LDSM (no TMEM dependency — overlaps with TMEM busy)
    uint32_t uint32_values[2][kNumLoads];
    #pragma unroll
    for (uint32_t i = 0; i < kNumLoads; i += 2) {
        uint32_t bank_group_idx_base = i + lane_idx / 16;
        uint32_t lane_in_group = lane_idx % 16;
        auto row = (bank_group_idx_base / (kSwizzleAMode / kNumBankGroupBytes)) + lane_in_group;
        auto col = bank_group_idx_base % (kSwizzleAMode / kNumBankGroupBytes);
        col ^= row % (kSwizzleAMode / kNumBankGroupBytes);
        auto smem_offset = row * 128 + col * kNumBankGroupBytes;

        auto smem_ptr = smem_stage_ptr + smem_offset;
        uint32_t smem_addr = __cvta_generic_to_shared(smem_ptr);
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
            : "=r"(uint32_values[0][i+0]), "=r"(uint32_values[1][i+0]),
              "=r"(uint32_values[0][i+1]), "=r"(uint32_values[1][i+1])
            : "r"(smem_addr)
        );
    }

    // Wait for TMEM to become empty (between LDSM and TMEM store for overlap)
    {
        uint32_t mbar_addr = static_cast<uint32_t>(
            __cvta_generic_to_shared(empty_cast_mbar_ptr));
        uint32_t done = 0;
        while (!done) {
            asm volatile(
                "{\\n"
                ".reg .pred p;\\n"
                "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\\n"
                "selp.u32 %0, 1, 0, p;\\n"
                "}\\n"
                : "=r"(done)
                : "r"(mbar_addr), "r"(empty_cast_phase)
                : "memory"
            );
        }
    }

    // Phase 2: Cast bf16->f32, accumulate sqr_sum, store to TMEM
    float2 fp32x2_values[2][kNumLoads];
    float s0x = *sum0_x, s0y = *sum0_y, s1x = *sum1_x, s1y = *sum1_y;

    #pragma unroll
    for (uint32_t i = 0; i < kNumLoads; ++i) {
        #pragma unroll
        for (uint32_t u = 0; u < 2; ++u) {
            fp32x2_values[u][i] = __bfloat1622float2(*reinterpret_cast<nv_bfloat162*>(&uint32_values[u][i]));
            if (u == 0) {
                s0x = __fmaf_rn(fp32x2_values[u][i].x, fp32x2_values[u][i].x, s0x);
                s0y = __fmaf_rn(fp32x2_values[u][i].y, fp32x2_values[u][i].y, s0y);
            } else {
                s1x = __fmaf_rn(fp32x2_values[u][i].x, fp32x2_values[u][i].x, s1x);
                s1y = __fmaf_rn(fp32x2_values[u][i].y, fp32x2_values[u][i].y, s1y);
            }
        }

        // Store upper and lower part to TMEM using SM100_TMEM_STORE_16dp256b1x
        uint32_t* upper = reinterpret_cast<uint32_t*>(&fp32x2_values[0][i]);
        uint32_t* lower = reinterpret_cast<uint32_t*>(&fp32x2_values[1][i]);
        uint32_t col_addr = get_tmem_addr(tmem_addr, row_offset, tmem_col + i * 8);
        asm volatile(
            "tcgen05.st.sync.aligned.16x256b.x1.b32 [%0], {%1, %2, %3, %4};"
            :
            : "r"(col_addr), "r"(upper[0]), "r"(upper[1]), "r"(lower[0]), "r"(lower[1])
        );
    }

    // Wait for all TMEM stores to complete
    asm volatile("tcgen05.wait::st.sync.aligned;\\n" ::: "memory");

    *sum0_x = s0x;
    *sum0_y = s0y;
    *sum1_x = s1x;
    *sum1_y = s1y;
}
"""

warp_reduce_sum4_src = """
__forceinline__ __device__ float warp_reduce_sum4(float value) {
    value += __shfl_xor_sync(0xffffffff, value, 2);
    value += __shfl_xor_sync(0xffffffff, value, 1);
    return value;
}
"""

# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------

num_m_blocks = M // BLOCK_M
num_k_blocks = K // BLOCK_K
num_blocks_total = num_m_blocks * kNumSplits


@Tx.prim_func(tirx=True)
def tf32_hc_prenorm_gemm(
    A: Tx.Buffer((M, K), a_type),
    B: Tx.Buffer((N, K), b_type),
    D: Tx.Buffer((kNumSplits, M, N), d_type),
    sqr_sum_out: Tx.Buffer((kNumSplits * M,), "float32"),
):
    # fmt: off
    Tx.func_attr({"global_symbol": "main"})

    # TMA tensor map for D (3D for split-K: dims = (N, M, kNumSplits))
    D_tensor_map: Tx.let[Tx.handle("tensormap")] = Tx.tvm_stack_alloca("tensormap", 1)
    Tx.call_packed("runtime.cuTensorMapEncodeTiled",
                   D_tensor_map, d_type, 3, D.data,
                   N, M, kNumSplits,             # globalDim: (inner=N, mid=M, outer=kNumSplits)
                   N * F32_BYTES,                 # globalStride[0] (M dim stride in bytes)
                   M * N * F32_BYTES,             # globalStride[1] (kNumSplits dim stride in bytes)
                   BLOCK_N, BLOCK_M, 1,           # boxDim: (inner=BLOCK_N, mid=BLOCK_M, outer=1)
                   1, 1, 1,                       # boxStride: all 1
                   0, 3, 0, 0)                    # interleave=0, swizzle=3 (128B), l2=0, oob=0

    # TMA tensor map for B (explicit, with atom-sized inner dim for 128B swizzle)
    # B is (N, K) row-major, K is contiguous (inner dim)
    # TMA box: (BLOCK_K_ATOM_B=32, BLOCK_N=32) with 128B swizzle
    B_tensor_map: Tx.let[Tx.handle("tensormap")] = Tx.tvm_stack_alloca("tensormap", 1)
    Tx.call_packed("runtime.cuTensorMapEncodeTiled",
                   B_tensor_map, b_type, 2, B.data,
                   K, N,                     # globalDim: (inner=K, outer=N)
                   K * F32_BYTES,            # globalStride[0] (outer dim stride in bytes)
                   BLOCK_K_ATOM_B, BLOCK_N,  # boxDim: (inner=32 f32=128B, outer=32)
                   1, 1, 0, 3, 0, 0)        # swizzle=3 (128B)
    # A TMA tensor map is auto-generated by Tx.copy_async

    with Tx.kernel():
        bx = Tx.cta_id([num_blocks_total], parent="kernel")
        wg_id = Tx.warpgroup_id([WG_NUMBER], parent="cta")
        warp_id = Tx.warp_id([WARP_NUMBER], parent="warpgroup")
        tid_in_wg = Tx.thread_id([128], parent="warpgroup")
        lane_id = Tx.thread_id([32], parent="warp")

        with Tx.cta():
            # ---------------------------------------------------------------
            # Shared memory allocation
            # ---------------------------------------------------------------
            pool = Tx.meta_var(Tx.PoolAllocator())
            tmem_addr = Tx.decl_scalar("uint32", pool.ptr, scope="shared.dyn", elem_offset=0)
            pool.move_base_to(8)

            # Barriers (5 groups)
            full_barriers = TMABar(pool, kNumStages)
            full_cast_barriers = MBarrier(pool, kNumCastStages)
            empty_barriers = TCGen05Bar(pool, kNumStages)
            empty_cast_barriers = TCGen05Bar(pool, kNumCastStages)
            tmem_full_barriers = TCGen05Bar(pool, 1)

            pool.move_base_to(1024)
            D_smem = pool.alloc((BLOCK_M, BLOCK_N), d_type, layout=D_layout)
            A_smem = pool.alloc((kNumStages, BLOCK_M, BLOCK_K), a_type, layout=A_layout)
            B_smem = pool.alloc((kNumStages, NUM_B_TMA_ATOMS, BLOCK_N, BLOCK_K_ATOM_B), b_type, layout=B_layout)
            pool.commit()

            # ---------------------------------------------------------------
            # Local memory
            # ---------------------------------------------------------------
            reg = Tx.alloc_buffer((TMEM_LD_SIZE,), "float32", scope="local")
            reg_wg = reg.view(128, TMEM_LD_SIZE,
                              layout=TileLayout(S[(128, TMEM_LD_SIZE) : (1@axis_tid_in_wg, 1)]))

            descB: Tx.uint64
            descI: Tx.uint32
            phase: Tx.int32

            # Initialize barriers
            full_barriers.init(1)
            full_cast_barriers.init(128)  # cast warp has 128 threads
            empty_barriers.init(1)
            empty_cast_barriers.init(1)
            tmem_full_barriers.init(1)

            # ---------------------------------------------------------------
            # Allocate TMEM (warp 0 of any WG)
            # ---------------------------------------------------------------
            with Tx.warp()[0:1]:
                Tx.ptx.tcgen05.alloc(Tx.address_of(tmem_addr), n_cols=N_COLS, cta_group=1)
                Tx.cuda.warp_sync()

            Tx.ptx.fence.proxy_async("shared::cta")
            Tx.ptx.fence.mbarrier_init()
            Tx.cuda.cta_sync()
            Tx.cuda.trap_when_assert_failed(tmem_addr == 0)

            tmem = Tx.decl_buffer((128, N_COLS), "float32", scope="tmem",
                                  allocated_addr=0,
                                  layout=TileLayout(S[(128, N_COLS) : (1@TLane, 1@TCol)]))

            # ---------------------------------------------------------------
            # Grid scheduling: split-K
            # ---------------------------------------------------------------
            m_block_idx: Tx.let = bx // kNumSplits
            k_split_idx: Tx.let = bx % kNumSplits
            k_offset: Tx.let = k_split_idx * NUM_K_ITERS_PER_SPLIT * BLOCK_K
            m_offset: Tx.let = M * k_split_idx

            with Tx.cta():
                Tx.attr({"tirx.scope_partition": True})

                # ===========================================================
                # WG1: TMA + MMA + Epilogue (warps 0-3)
                # ===========================================================
                with Tx.warpgroup()[wg_id == 1]:
                    # TMA warp (warp 0 of WG1, single elected thread)
                    if warp_id == 0:
                        with Tx.thread()[Tx.ptx.elect_sync()]:
                            for s in Tx.serial(NUM_K_ITERS_PER_SPLIT):
                                stage_idx: Tx.let = s % kNumStages
                                empty_barriers.wait(stage_idx, ((s // kNumStages) & 1) ^ 1)

                                m_idx: Tx.let = m_block_idx * BLOCK_M
                                k_idx: Tx.let = k_offset + s * BLOCK_K

                                tma_copy_cfg = Tx.meta_var({"dispatch": "tma", "mbar": full_barriers.ptr_to([stage_idx]), "cta_group": CTA_GROUP})
                                # A: bf16, BLOCK_K*bf16=128B fits in one TMA atom
                                Tx.copy_async(A_smem[stage_idx, :, :], A[m_idx: m_idx + BLOCK_M, k_idx: k_idx + BLOCK_K], **tma_copy_cfg)
                                # B: f32, BLOCK_K*f32=256B > 128B, split into 2 explicit TMA atoms
                                for bi in Tx.unroll(NUM_B_TMA_ATOMS):
                                    Tx.ptx.cp_async.bulk.tensor.g2c(
                                        2,
                                        B_smem.ptr_to([stage_idx, bi, 0, 0]),
                                        full_barriers.ptr_to([stage_idx]),
                                        B_tensor_map,
                                        k_idx + bi * BLOCK_K_ATOM_B,  # inner coord (K)
                                        0,  # outer coord (N)
                                        cta_group=CTA_GROUP,
                                    )

                                AB_bytes = Tx.meta_var(SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE)
                                full_barriers.arrive(stage_idx, AB_bytes)

                    # MMA warp (warp 1 of WG1) — TS mode TF32
                    # Descriptor init must be warp-wide (shfl_sync needs all threads)
                    if warp_id == 1:
                        Tx.ptx.tcgen05.encode_instr_descriptor(
                            Tx.address_of(descI),
                            "float32", "tf32", "tf32",
                            BLOCK_M, BLOCK_N, UMMA_K,
                            False, False, CTA_GROUP,
                        )

                        descB_obj = SmemDescriptor("B")
                        descB_obj.init(
                            B_smem.ptr_to([0, 0, 0, 0]),
                            ldo=0,
                            sdo=8 * BLOCK_SWIZZLED_BK_B * F32_BYTES // F128_BYTES,
                            swizzle=SWIZZLE_B,
                        )

                        with Tx.thread()[Tx.ptx.elect_sync()]:
                            first_mma_done = Tx.local_scalar("int32", "first_mma_done")
                            first_mma_done = 0

                            for s in Tx.serial(NUM_K_ITERS_PER_SPLIT):
                                cast_stage_idx: Tx.let = s % kNumCastStages
                                stage_idx: Tx.let = s % kNumStages

                                full_cast_barriers.wait(cast_stage_idx, (s // kNumCastStages) & 1)
                                Tx.ptx.tcgen05.fence.after_thread_sync()

                                for k in Tx.unroll(NUM_MMA_K_ITERS):
                                    a_tmem_col = Tx.meta_var(cast_stage_idx * BLOCK_K + k * UMMA_K)

                                    b_stage_16B = Tx.meta_var(stage_idx * SMEM_B_SIZE_PER_STAGE // F128_BYTES)
                                    atom_idx = Tx.meta_var((k * UMMA_K) // BLOCK_SWIZZLED_BK_B)
                                    in_atom_idx = Tx.meta_var((k * UMMA_K) % BLOCK_SWIZZLED_BK_B)
                                    b_k_offset = Tx.meta_var(
                                        b_stage_16B
                                        + atom_idx * BLOCK_N * BLOCK_SWIZZLED_BK_B * F32_BYTES // F128_BYTES
                                        + in_atom_idx * F32_BYTES // F128_BYTES
                                    )

                                    Tx.ptx.tcgen05.mma(
                                        "float32", "tf32", "tf32",
                                        Tx.cuda.get_tmem_addr(0, 0, D_TMEM_START_COL),
                                        Tx.cuda.get_tmem_addr(0, 0, a_tmem_col),
                                        descB_obj.add_16B_offset(b_k_offset),
                                        descI,
                                        use_a_tmem=True,
                                        cta_group=CTA_GROUP,
                                        enable_input_d=first_mma_done,
                                    )
                                    first_mma_done = 1

                                empty_cast_barriers.arrive(cast_stage_idx, CTA_GROUP, 0)
                                empty_barriers.arrive(stage_idx, CTA_GROUP, 0)

                            tmem_full_barriers.arrive(0, CTA_GROUP, 0)

                    # -------------------------------------------------------
                    # Epilogue (all 128 threads of WG1): TMEM -> SMEM -> TMA store
                    # -------------------------------------------------------
                    tmem_full_barriers.wait(0, 0)
                    Tx.ptx.tcgen05.fence.after_thread_sync()

                    # BLOCK_M=64 uses TMEM Layout F: only first 16 lanes per warp
                    # have valid MMA output. Each warp covers 16 rows.
                    for ki in Tx.unroll(BLOCK_N // TMEM_LD_SIZE):
                        col_st = Tx.meta_var(D_TMEM_START_COL + ki * TMEM_LD_SIZE)
                        Tx.copy(reg_wg[:, :], tmem[:, col_st : col_st + TMEM_LD_SIZE])
                        with Tx.thread():
                            st = Tx.meta_var(ki * TMEM_LD_SIZE)
                            if lane_id < BLOCK_M // WARP_NUMBER:
                                Tx.copy(D_smem[warp_id * (BLOCK_M // WARP_NUMBER) + lane_id, st : st + TMEM_LD_SIZE], reg[:])
                        Tx.cuda.warp_sync()

                    Tx.ptx.fence.proxy_async("shared::cta")
                    Tx.cuda.warpgroup_sync(10)

                    with Tx.thread(parent="warpgroup")[tid_in_wg == 0]:
                        m_st: Tx.let = m_block_idx * BLOCK_M
                        Tx.ptx.cp_async.bulk.tensor.s2g(
                            3,
                            D_smem.ptr_to([0, 0]),
                            D_tensor_map,
                            0,            # inner coord (N)
                            m_st,         # mid coord (M)
                            k_split_idx,  # outer coord (split)
                        )
                        Tx.ptx.cp_async.bulk.commit_group()

                    if tid_in_wg == 0:
                        Tx.ptx.cp_async.bulk.wait_group(0)
                    Tx.cuda.warpgroup_sync(10)

                # ===========================================================
                # WG0: Cast warps (all 128 threads)
                # bf16 A from SMEM -> cast to f32 -> store to TMEM + sqr_sum
                # ===========================================================
                with Tx.warpgroup()[wg_id == 0]:
                    Tx.cuda.trap_when_assert_failed(tmem_addr == 0)

                    sum0_x = Tx.alloc_local([1], "float32")
                    sum0_y = Tx.alloc_local([1], "float32")
                    sum1_x = Tx.alloc_local([1], "float32")
                    sum1_y = Tx.alloc_local([1], "float32")
                    with Tx.thread():
                        sum0_x[0] = Tx.float32(0)
                        sum0_y[0] = Tx.float32(0)
                        sum1_x[0] = Tx.float32(0)
                        sum1_y[0] = Tx.float32(0)

                    for s in Tx.serial(NUM_K_ITERS_PER_SPLIT):
                        stage_idx: Tx.let = s % kNumStages
                        cast_stage_idx: Tx.let = s % kNumCastStages

                        full_barriers.wait(stage_idx, (s // kNumStages) & 1)
                        # empty_cast_barriers.wait moved inside inline fn (after LDSM, before TMEM store)

                        a_smem_byte_off = Tx.meta_var(A_SMEM_BYTE_BASE + stage_idx * SMEM_A_SIZE_PER_STAGE)
                        Tx.cuda.func_call(
                            "cast_bf16_store_tmem",
                            pool.ptr,
                            Tx.uint32(a_smem_byte_off),
                            tmem_addr,
                            Tx.int32(0),
                            Tx.int32(cast_stage_idx * BLOCK_K),
                            warp_id,
                            lane_id,
                            Tx.address_of(sum0_x[0]),
                            Tx.address_of(sum0_y[0]),
                            Tx.address_of(sum1_x[0]),
                            Tx.address_of(sum1_y[0]),
                            empty_cast_barriers.ptr_to([cast_stage_idx]),
                            Tx.uint32(((s // kNumCastStages) & 1) ^ 1),
                            source_code=cast_and_store_src,
                            return_type="void",
                        )

                        Tx.ptx.tcgen05.fence.before_thread_sync()
                        full_cast_barriers.arrive(cast_stage_idx)

                    # Warp-reduce sqr_sum: must call shfl from ALL threads,
                    # then only lane%4==0 writes the result.
                    # Use alloc_local to force evaluation before the if-guard.
                    reduced0_buf = Tx.alloc_local([1], "float32")
                    reduced1_buf = Tx.alloc_local([1], "float32")
                    with Tx.thread():
                        local_sum0: Tx.let = sum0_x[0] + sum0_y[0]
                        reduced0_buf[0] = Tx.cuda.func_call(
                            "warp_reduce_sum4",
                            local_sum0,
                            source_code=warp_reduce_sum4_src,
                            return_type="float32",
                        )
                        local_sum1: Tx.let = sum1_x[0] + sum1_y[0]
                        reduced1_buf[0] = Tx.cuda.func_call(
                            "warp_reduce_sum4",
                            local_sum1,
                            source_code=warp_reduce_sum4_src,
                            return_type="float32",
                        )

                        m_idx0: Tx.let = m_block_idx * BLOCK_M + warp_id * BLOCK_M_PER_WARP + lane_id // 4
                        if lane_id % 4 == 0:
                            if m_idx0 < M:
                                sqr_sum_out[m_offset + m_idx0] = reduced0_buf[0]

                        m_idx1: Tx.let = m_block_idx * BLOCK_M + warp_id * BLOCK_M_PER_WARP + lane_id // 4 + 8
                        if lane_id % 4 == 0:
                            if m_idx1 < M:
                                sqr_sum_out[m_offset + m_idx1] = reduced1_buf[0]

            # ---------------------------------------------------------------
            # Deallocate TMEM
            # ---------------------------------------------------------------
            with Tx.warp()[1:2]:
                Tx.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
                Tx.ptx.tcgen05.dealloc(tmem_addr, n_cols=N_COLS, cta_group=1)

            Tx.cuda.cta_sync()


# fmt: on


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def bf16_to_tvm(tensor, dev):
    """Convert a bf16 CUDA tensor to TVM ndarray via ml_dtypes."""
    return tvm.runtime.tensor(
        tensor.cpu().view(torch.int16).numpy().view(np.int16).view(ml_dtypes.bfloat16),
        device=dev,
    )


def flops(ms):
    return M * N * K * 2 / (ms * 1e-3)


# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------


@tvm.testing.requires_cuda_compute_version(10, exact=True)
def test_tf32_hc_prenorm_gemm():
    DEV = tvm.cuda(0)

    # Generate test data
    A_torch = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B_torch = torch.randn(N, K, dtype=torch.float32, device="cuda")

    # Reference computation
    D_ref = (A_torch.float() @ B_torch.T)  # (M, N) f32
    sqr_sum_ref = (A_torch.float() ** 2).sum(dim=1)  # (M,) f32

    # Convert to TVM tensors
    A_tvm = bf16_to_tvm(A_torch, DEV)
    B_tvm = tvm.runtime.tensor(B_torch.cpu().numpy(), device=DEV)

    # Output buffers (split-K: D is (kNumSplits, M, N), sqr_sum is (kNumSplits * M,))
    D_out_splits = torch.zeros(kNumSplits, M, N, dtype=torch.float32, device="cuda")
    sqr_sum_out_splits = torch.zeros(kNumSplits * M, dtype=torch.float32, device="cuda")

    # Compile
    target = tvm.target.Target("cuda")
    with target:
        src, mod = get_source(tf32_hc_prenorm_gemm)

    # Run
    mod(A_tvm, B_tvm, D_out_splits, sqr_sum_out_splits)

    # Reduce splits
    D_out = D_out_splits.sum(dim=0)  # (M, N)
    sqr_sum_out = sqr_sum_out_splits.view(kNumSplits, M).sum(dim=0)  # (M,)

    # Verify D
    d_diff = calc_diff(D_out.cpu(), D_ref.cpu())
    print(f"D calc_diff: {d_diff:.6e}")
    assert d_diff < 1e-3, f"D diff too large: {d_diff}"

    # Verify sqr_sum
    sqr_diff = calc_diff(sqr_sum_out.cpu(), sqr_sum_ref.cpu())
    print(f"sqr_sum calc_diff: {sqr_diff:.6e}")
    assert sqr_diff < 1e-3, f"sqr_sum diff too large: {sqr_diff}"

    # Cross-check with DeepGEMM
    D_dg = torch.empty(M, N, dtype=torch.float32, device="cuda")
    sqr_sum_dg = torch.empty(M, dtype=torch.float32, device="cuda")
    deep_gemm.tf32_hc_prenorm_gemm(A_torch, B_torch, D_dg, sqr_sum_dg, num_splits=None)

    dg_d_diff = calc_diff(D_out.cpu(), D_dg.cpu())
    dg_sqr_diff = calc_diff(sqr_sum_out.cpu(), sqr_sum_dg.cpu())
    print(f"vs DeepGEMM: D diff={dg_d_diff:.6e}, sqr_sum diff={dg_sqr_diff:.6e}")
    assert dg_d_diff < 1e-3, f"D diff vs DeepGEMM too large: {dg_d_diff}"
    assert dg_sqr_diff < 1e-3, f"sqr_sum diff vs DeepGEMM too large: {dg_sqr_diff}"

    print("Test passed!")


# ---------------------------------------------------------------------------
# Benchmark: TIRX vs DeepGEMM
# ---------------------------------------------------------------------------


def bench_tf32_hc_prenorm_gemm():
    DEV = tvm.cuda(0)

    A_torch = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B_torch = torch.randn(N, K, dtype=torch.float32, device="cuda")

    # Reference
    D_ref = (A_torch.float() @ B_torch.T)
    sqr_sum_ref = (A_torch.float() ** 2).sum(dim=1)

    # TVM tensors
    A_tvm = bf16_to_tvm(A_torch, DEV)
    B_tvm = tvm.runtime.tensor(B_torch.cpu().numpy(), device=DEV)

    # Compile
    target = tvm.target.Target("cuda")
    with target:
        src, mod = get_source(tf32_hc_prenorm_gemm)

    def tir():
        D_out_splits = torch.zeros(kNumSplits, M, N, dtype=torch.float32, device="cuda")
        sqr_sum_out_splits = torch.zeros(kNumSplits * M, dtype=torch.float32, device="cuda")
        ms = bench(
            lambda: mod(A_tvm, B_tvm, D_out_splits, sqr_sum_out_splits),
            warmup=50,
            repeat=50,
            proton_name="tir",
        )
        D_out = D_out_splits.sum(dim=0)
        sqr_sum_out = sqr_sum_out_splits.view(kNumSplits, M).sum(dim=0)
        return ms, D_out, sqr_sum_out

    def std():
        D_dg = torch.empty(M, N, dtype=torch.float32, device="cuda")
        sqr_sum_dg = torch.empty(M, dtype=torch.float32, device="cuda")
        ms = bench(
            lambda: deep_gemm.tf32_hc_prenorm_gemm(
                A_torch, B_torch, D_dg, sqr_sum_dg, num_splits=None
            ),
            warmup=50,
            repeat=50,
            proton_name="std",
        )
        return ms, D_dg, sqr_sum_dg

    with ProtonContext():
        tir_ms, tir_D, tir_sqr = tir()
        print(f"TIR flops: {flops(tir_ms) / 1e12:.2f} TFLOPS, time: {tir_ms:.3f} ms")

        std_ms, std_D, std_sqr = std()
        print(f"Std flops: {flops(std_ms) / 1e12:.2f} TFLOPS, time: {std_ms:.3f} ms")

        d_diff = calc_diff(tir_D.cpu(), std_D.cpu())
        sqr_diff = calc_diff(tir_sqr.cpu(), std_sqr.cpu())
        print(f"TIR vs DeepGEMM: D diff={d_diff:.6e}, sqr_sum diff={sqr_diff:.6e}")
        assert d_diff < 2e-3, f"D diff too large: {d_diff}"
        assert sqr_diff < 2e-3, f"sqr_sum diff too large: {sqr_diff}"
        print("Benchmark passed!")


if __name__ == "__main__":
    bench_tf32_hc_prenorm_gemm()
