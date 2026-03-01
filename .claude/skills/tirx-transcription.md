<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# General DSL → TIRX Transcription Guide

## Overview

TIRX is a Python-embedded DSL for writing GPU kernels targeting NVIDIA SM80+ (Ampere through Blackwell). It compiles through TVM's TIR pipeline to PTX/CUBIN. This guide covers patterns, primitives, and best practices for transcribing kernels from other DSLs (CUDA, Triton, ThunderKittens, etc.) into TIRX.

**Important:** When transcribing to SM100a (Blackwell/B200) patterns, the source kernel must either be B200-native or architecture-agnostic. Hopper (SM90a) WGMMA kernels cannot be mechanically transcribed — they require a full algorithm redesign using tcgen05 instructions. See "Step 0: Verify Source Kernel Hardware Target" below.

---

## TIRX Primitive Catalog

### Kernel Structure

```python
@Tx.prim_func(tirx=True)
def kernel(ptr: Tx.handle, ...):
    dim = Tx.int32()                          # dynamic dimension (inferred from runtime tensor)
    buf = Tx.match_buffer(ptr, [dim, N], "float16", scope="global")

    with Tx.kernel():
        bx = Tx.cta_id([GRID_SIZE], parent="kernel")
        tx, ty = Tx.thread_id([BDX, BDY], parent="cta")

        with Tx.cta():
            pool = Tx.PoolAllocator()
            smem = pool.alloc([M, N], "float16")
            pool.commit()                     # finalizes shared memory layout

            with Tx.thread():
                reg = Tx.alloc_local([K], "float32")  # per-thread registers
                # ... kernel body ...
```

### Memory Hierarchy

| Scope | API | Notes |
|-------|-----|-------|
| Global | `Tx.match_buffer(..., scope="global")` | Kernel arguments, dynamic shapes |
| Shared | `pool.alloc([shape], dtype)` | Via PoolAllocator, inside `Tx.cta()` |
| Register | `Tx.alloc_local([shape], dtype)` | Per-thread, inside `Tx.thread()` |
| TMEM | `Tx.ptx.tcgen05.alloc(...)` + `Tx.decl_buffer(..., scope="tmem")` | SM100a only, for tcgen05 MMA accumulators |

### Data Movement

```python
# Global ↔ Register (vectorized)
vec = Tx.alloc_local([VEC], "float16")
Tx.copy(vec[:], global_buf[row, col : col + VEC], vec_len=VEC)

# Register ↔ Shared
Tx.copy(smem[row, col : col + VEC], vec[:])
Tx.copy(vec[:], smem[row, col : col + VEC])

# TMA async (SM90+)
Tx.copy_async(smem_buf[stage, ...], global_buf[...], dispatch="tma", mbar_ptr=mbar.ptr_to([stage]))

# Type casting
Tx.cast(val, "float32")           # scalar cast
Tx.cast(dst_vec[:], src_vec[:])   # vector cast
```

### Loop Constructs

```python
Tx.unroll(N)    # Compile-time unroll. N should be small (<32). Loop var is compile-time constant.
Tx.serial(N)    # Runtime loop. For larger iteration counts or dynamic bounds.
Tx.grid(N)      # Parallel grid iteration (used in megakernel tiles).
```

### Synchronization

```python
Tx.cuda.cta_sync()                  # __syncthreads()
Tx.cuda.cta_sync(barrier_id)        # Named barrier sync
Tx.ptx.mbarrier.init(ptr, count)    # Initialize mbarrier
Tx.ptx.mbarrier.arrive(ptr)         # Signal mbarrier
Tx.ptx.mbarrier.try_wait(ptr, phase) # Wait on mbarrier
Tx.ptx.fence.proxy_async("shared::cta")  # Fence for TMA proxy
```

### Math Operations

```python
Tx.exp(x)       # expf
Tx.log(x)       # logf
Tx.rsqrt(x)     # rsqrtf
Tx.max(a, b)    # fmaxf
Tx.min(a, b)    # fminf
Tx.abs(x)       # fabsf
Tx.cos(x)       # cosf
Tx.sin(x)       # sinf
Tx.tanh(x)      # tanhf
Tx.sqrt(x)      # sqrtf
Tx.if_then_else(cond, true_val, false_val)  # ternary
```

### Warp-Level

```python
Tx.tvm_warp_shuffle_xor(mask, val, offset, width, warp_size)  # __shfl_xor_sync
Tx.tvm_warp_shuffle_down(mask, val, offset, width, warp_size)  # __shfl_down_sync
```

### tcgen05 MMA (SM100a only)

```python
Tx.gemm_async(tmem, A_smem, B_smem, dispatch="tcgen05",
              cta_group=CTA_GROUP, descI=descI, accum=True/False)
Tx.ptx.tcgen05.commit(mbar_ptr, cta_group=CTA_GROUP, cta_mask=mask)
```

---

## Kernel Pattern Taxonomy

### Pattern 1: 2-CTA Cluster GEMM (SM100a)

**When to use:** Kernel dominated by large matmuls (M, N >= 128), pure GEMM with epilogue.

**Architecture:**
- 2-CTA cluster (`M_CLUSTER=2, CTA_GROUP=2`)
- 3 warpgroups: WG0/WG1 = LD consumer (epilogue), WG2 = MMA + TMA producer
- TMEM accumulator (N_COLS=512), pipelined K-reduction
- PIPELINE_DEPTH=4, MMA tile 256×256
- Persistent tile scheduler (ClusterPersistentScheduler2D)

**Key components:**
```python
from tvm.tirx.tile_scheduler import ClusterPersistentScheduler2D
from tvm.tirx.pipeline import PipelineState, TMABar, TCGen05Bar

# Cluster setup
cbx, cby = Tx.cta_id([M_CLUSTER, N_CLUSTER], parent="cluster")
bx = Tx.cta_id([SM_NUMBER], parent="kernel")
wg_id = Tx.warpgroup_id([WG_NUMBER], parent="cta")

# Pipeline: producer prefetches while consumers compute
# Consumer: Tx.gemm_async(tmem, ..., dispatch="tcgen05", accum=...)
# Producer: Tx.copy_async(smem, global, dispatch="tma", ...)
```

**Variants:**

| Variant | dtype | Special feature | Reference |
|---------|-------|-----------------|-----------|
| Standard f16 GEMM | f16×f16→f32 | Basic epilogue | `test_hgemm.py` |
| Fused GEMM + GELU/gate | f16×f16→f32→f16 | Epilogue activation | `test_flux.py` |
| FP8 block-scaled GEMM | fp8×fp8→f32 | Scale factor pipeline (extra warp) | `test_fp8_bhr_hdr_bhd.py` |
| TF32 GEMM + RMSNorm | bf16→f32, f32×f32→f32 | Cast WG (bf16→f32→TMEM) + sqr_sum | `test_tf32_hc_prenorm_gemm.py` |
| Batched per-head GEMM | f16×f16→f32 | 3D batch indexing | `test_bhr_hdr_bhd.py` |
| Batch-reduction GEMM | f16×f16→f32 | Split-K via GroupMajor3D + TMA reduce-add, 409 TFLOPS (92% DG) | `test_bmk_bnk_mn.py` |

**FP8 block-scaled variant details:**
- Producer WG has 3 active warps: warp 3 = TMA, warp 2 = scale factor transpose, warp 0 = MMA
- Scale factors loaded alongside A/B tiles, transposed in SMEM by warp 2
- MMA uses FP8 input types with separate scale factor application in accumulator

**TF32 + RMSNorm variant details:**
- WG0 = "cast warp group": loads bf16 A from SMEM, casts to f32, stores to TMEM, accumulates sqr_sum
- WG1 = TMA + MMA (TS mode: A from TMEM via cast WG, B from SMEM) + epilogue
- MMA reads A from TMEM (use_a_tmem-like: written by WG0, consumed by WG1 MMA)

**Reference:** `test_hgemm.py`, `test_flux.py`, `test_fp8_bhr_hdr_bhd.py`, `test_tf32_hc_prenorm_gemm.py`

### Pattern 2: 1-Consumer Warp-Specialization (SM100a)

**When to use:** Attention/SSM kernels with multiple sequential MMA phases interleaved with consumer-side compute (softmax, state accumulation, featurization). The dominant pattern for TK-transcribed kernels.

**Architecture:**
- Single CTA, no clustering (`CTA_GROUP=1`)
- 2 warpgroups: WG0 = consumer, WG1 = MMA + TMA producer
- `WG_NUMBER=2, WARP_NUMBER=4, NUM_THREADS=256`
- Within WG1: warp 0 = MMA, warp 3 = TMA producer (both gated by `Tx.ptx.elect_sync()`; warps 1-2 idle)
- TMEM accumulator (N_COLS=64), MMA tile 64×64
- PIPE_DEPTH=1 or 2 (double-buffered for kernels that read both current and previous chunk)
- Persistent grid: `cta_id([SM_COUNT])` with nested while loops (outer over batch×head, inner over chunks)
- 128-wide buffers split into `_lo`/`_hi` halves (TMEM N_COLS=64 limit)

**Multi-phase MMA flow:**
Each chunk processes N MMA phases sequentially. Between phases, the consumer reads TMEM results, does compute (softmax, normalize, featurize), writes transformed data back to SMEM, and signals MMA to proceed.

```
Phase k: MMA computes A@B → TMEM
         MMA signals mma2consumer
         Consumer: wait mma2consumer, read TMEM, compute, write SMEM
         Consumer signals consumer2mma
         MMA: wait consumer2mma → next phase
```

**Barrier classes (from `tvm.tirx.pipeline`):**
```python
from tvm.tirx.pipeline import MBarrier, TMABar, TCGen05Bar

tma2mma_bar = TMABar(pool, PIPE_DEPTH)         # TMA arrival → MMA wait
mma2tma_bar = TCGen05Bar(pool, PIPE_DEPTH)      # MMA release → TMA can reuse slot
mma2consumer_bar = TCGen05Bar(pool, 1)           # MMA done → consumer reads TMEM
consumer2mma_bar = MBarrier(pool, 1)             # Consumer done → MMA next phase
workitem_sync_bar = TMABar(pool, 1)              # Work item sync (MMA → TMA)
```

**Phase parity rule:** If a chunk uses N MMA phases with the same mma2consumer/consumer2mma barrier pair:
- N even → `phase_c2m` does NOT flip between chunks
- N odd → `phase_c2m` flips once (`phase_c2m = phase_c2m ^ 1`)

**Instruction descriptors:**
Some kernels need two descriptors when both transB=False and transB=True MMA operations occur. Convention: `descI` (transB=False) and `descI_tb` (transB=True). Hedgehog uses `descI_nn`/`descI_nt` instead.
```python
descI: Tx.uint32
Tx.ptx.tcgen05.encode_instr_descriptor(
    Tx.address_of(descI), "float32", "float16", "float16",
    MMA_M * CTA_GROUP, MMA_N, MMA_K, False, False, CTA_GROUP)
descI_tb: Tx.uint32
Tx.ptx.tcgen05.encode_instr_descriptor(
    Tx.address_of(descI_tb), "float32", "float16", "float16",
    MMA_M * CTA_GROUP, MMA_N, MMA_K, False, True, CTA_GROUP)
```

**Cross-buffer accumulation (128-wide inputs):**
When input dimension > 64, split into lo/hi and accumulate across both:
```python
# Phase: Q[64,128] @ K^T[64,128] → [64,64]
# Split into Q_lo@K_lo (4 iters, accum from ki>0) + Q_hi@K_hi (4 iters, accum=True)
for ki in Tx.unroll(HALF // MMA_K):  # 4 iterations
    Tx.ptx.tcgen05.encode_matrix_descriptor(
        Tx.address_of(descA), Q_lo.ptr_to([tic, 0, ki * MMA_K]), MMA_LDO, MMA_SDO, SWIZZLE)
    Tx.ptx.tcgen05.encode_matrix_descriptor(
        Tx.address_of(descB), K_lo.ptr_to([tic, 0, ki * MMA_K]), MMA_LDO, MMA_SDO, SWIZZLE)
    Tx.ptx.tcgen05.mma("float32", "float16", "float16",
        Tx.cuda.get_tmem_addr(tmem_addr, 0, 0), descA, descB, descI, False, CTA_GROUP, ki > 0)
for ki in Tx.unroll(HALF // MMA_K):  # 4 iterations (accumulate into same TMEM)
    Tx.ptx.tcgen05.encode_matrix_descriptor(
        Tx.address_of(descA), Q_hi.ptr_to([tic, 0, ki * MMA_K]), MMA_LDO, MMA_SDO, SWIZZLE)
    Tx.ptx.tcgen05.encode_matrix_descriptor(
        Tx.address_of(descB), K_hi.ptr_to([tic, 0, ki * MMA_K]), MMA_LDO, MMA_SDO, SWIZZLE)
    Tx.ptx.tcgen05.mma("float32", "float16", "float16",
        Tx.cuda.get_tmem_addr(tmem_addr, 0, 0), descA, descB, descI, False, CTA_GROUP, True)
```

**Consumer TMEM read:**
```python
# Allocate once at consumer scope entry:
reg = Tx.alloc_buffer((TMEM_LD_SIZE,), "float32", scope="local")
reg_wg = reg.view(128, TMEM_LD_SIZE,
                  layout=TileLayout(S[(128, TMEM_LD_SIZE) : (1@tid_in_wg, 1)]))

# In each phase handler:
out_row = Tx.meta_var(warp_id * 16 + lane_id)  # 64 active rows, lane_id < 16
if lane_id < 16:
    Tx.copy(reg_wg[:, :], tmem[:, 0:TMEM_LD_SIZE])  # N_COLS == TMEM_LD_SIZE for TK kernels
    for ri in Tx.unroll(TMEM_LD_SIZE):
        val = reg[ri]
        # process val for column ri
```

**SMEM layout rules:**
1. Swizzled buffers (TMA targets, MMA operands) go first — `SwizzleLayout(3,3,3)` for [64,64] f16
2. Non-swizzled buffers (f32 state, scalars) go last
3. Explicit byte offsets via `Tx.decl_buffer(..., buf.data, elem_offset=off)`
4. Total must fit ≤ 232KB

**SMEM release strategies:**
After the last MMA phase's consumer signals done, MMA releases the SMEM slot so TMA can refill:

*Standard (read current tic only):* Release current tic unconditionally. Used by linear_attention, based, mamba2, fftconv.
```python
mma2tma_bar.arrive(tic, CTA_GROUP, cta_mask)  # TCGen05Bar
```

*Delayed (read both current AND prev tic):* Release `prev_tic` with chunk-0 guard. Used by hedgehog where the consumer reads V_prev from the previous tic's slot for state updates.
```python
if cid_mma > 0:  # Skip chunk 0 — prev_tic holds initial fill, not stale data
    mma2tma_bar.arrive(prev_tic, CTA_GROUP, cta_mask)
```

**Work item synchronization (`workitem_sync`) for persistent grid (BH > SM_COUNT):**

When `total_bh > SM_COUNT`, each CTA processes multiple work items via `while wid < total_bh: ... wid += SM_COUNT`. TMA races ahead of MMA/consumer at work item boundaries because TMA's inner loop finishes much faster (just DMA commands). When TMA enters the next work item, it skips `mma2tma_bar.wait` for the first `PIPE_DEPTH` chunks (the `if cid_tma >= PIPE_DEPTH:` guard), loading new data into SMEM slots that MMA/consumer from the previous work item may still be reading.

**Fix:** Add a single `workitem_sync` mbarrier that synchronizes MMA and TMA at work item boundaries:

```python
# Barrier setup (allocated from PoolAllocator):
workitem_sync_bar = TMABar(pool, 1)
workitem_sync_bar.init(1)

# TMA warp: wait before non-first work items
wi_phase = Tx.alloc_buffer((1,), "int32", scope="local")
wi_phase[0] = 0
wid_tma = bx
while wid_tma < total_bh:
    if wid_tma != bx:  # skip first work item
        with Tx.thread()[Tx.ptx.elect_sync()]:
            workitem_sync_bar.wait(0, wi_phase[0])
        wi_phase[0] = wi_phase[0] ^ 1
    # ... inner chunk loop (TMA loads) ...
    wid_tma = wid_tma + SM_COUNT

# MMA warp: arrive after inner loop completes
while wid_mma < total_bh:
    # ... inner chunk loop (MMA phases) ...
    with Tx.thread()[Tx.ptx.elect_sync()]:
        Tx.ptx.mbarrier.arrive(workitem_sync_bar.ptr_to([0]))
    wid_mma = wid_mma + SM_COUNT
```

MMA's arrive implicitly guarantees consumer is done because it happens after the last `consumer2mma` wait. The `wi_phase` variable toggles independently of operational barrier phases.

**Important:** For kernels that release `prev_tic` instead of `tic` (e.g., hedgehog), the MMA must also release the final chunk's `tic` slot before arriving on `workitem_sync`, to keep `mma2tma` barrier phases balanced across work items. See pitfall #14.

**fftconv special case (PIPE_DEPTH=1, no inner chunk loop):** In addition to `workitem_sync`, add `phase[0] = phase[0] ^ 1` after each work item in both TMA and MMA warps, because the `tma2mma` barrier toggles once per WI but `phase[0]` was never flipped (no `cid % PIPE_DEPTH == PIPE_DEPTH - 1` check without an inner chunk loop).

**Constraint:** N must be a multiple of `2*CHUNK` (ensuring even `nc`) for operational barrier phases to naturally reset between work items. All existing tests satisfy this.

**Reference files:**

**Pattern 2 (warp-specialized, multi-phase):**

| Kernel | Phases | Phase parity | Descriptors | PIPE_DEPTH | Unique feature |
|--------|--------|-------------|-------------|------------|----------------|
| `test_linear_attention.py` | 9 | odd (flip) | 2: descI + descI_tb | 2 | Cross-buffer Q@K, KV state accum, lo/hi split |
| `test_based.py` | 8 | even | 4: descI + descI_tb + 2 a2 descriptors (incl. transA) | 2 | Dual swizzle (QK=1, V=3), Taylor, a2 state via MMA |
| `test_mamba2.py` | 4 | even | 2: descI + descI_tb | 2 | SSD with f32 KV state + A cumsum |
| `test_fftconv.py` | 7 | odd (flip) | 1: descI_tb only | 1 | Complex Monarch FFT, negated constants, reg preload |
| `test_hedgehog.py` | 12 | even | 2: descI_nn + descI_nt | 2 | Hybrid sliding+linear attn, prev_tic release (needs final tic balance) |

**Pattern 1 (cluster GEMM):**

| Kernel | CTA_GROUP | Cluster | Special | Reference |
|--------|-----------|---------|---------|-----------|
| Flux GELU/gate | 2 | 2×1 | Fused epilogue (GELU/gate+residual) | `test_flux.py` |
| Batched per-head | 1 | 1×1 | 3D batch indexing [B,H,R]@[H,D,R]^T | `test_bhr_hdr_bhd.py` |
| Batch-reduction | 1 | 1×1 | Split-K (GroupMajor3D k_tiles=7), TMA reduce-add | `test_bmk_bnk_mn.py` |
| FP8 block-scaled | 1 | 1×1 | Scale factor pipeline, extra transpose warp | `test_fp8_bhr_hdr_bhd.py` |
| TF32 + RMSNorm | 2 | ? | Cast WG (bf16→f32→TMEM), TS MMA mode | `test_tf32_hc_prenorm_gemm.py` |

#### Pattern 1 Detailed Reference: 2-CTA Cluster GEMM (Flux)

**Architecture details from `test_flux.py`:**
- 2-CTA cluster (`M_CLUSTER=2, N_CLUSTER=1`), `CTA_GROUP=2`
- 3 warpgroups: WG0/WG1 = LD consumers (epilogue), WG2 = producer (TMA warp 3 + MMA warps 0-1)
- MMA tile: 256×256 (each warp handles 256×256 output block)
- TMEM: N_COLS=512, each consumer warp reads 64 columns at a time via `Tx.copy(reg_wg[:,:], tmem[:, col_st:col_st+64])`
- Pipeline depth 4, K-reduction across tiles
- `ClusterPersistentScheduler2D` for tile scheduling across SMs

**Barrier protocol (different from Pattern 2):**
- `TMABar(pool, depth=4)` — TMA→MMA per pipeline slot
- `TCGen05Bar(pool, depth=4)` — MMA→TMA release per pipeline slot
- `TCGen05Bar(pool, depth=1)` — MMA→consumer, one per consumer WG
- `MBarrier(pool, depth=1)` — consumer→MMA, one per consumer WG
- `mma2tma.init(NUM_CONSUMER)` — NUM_CONSUMER arrivals expected (not 1)
- `ld2mma.init(128 * NUM_CONSUMER)` — full WG thread count

**Partitioned K-loop (`paritioned_loop`):**
```python
@Tx.inline
def paritioned_loop(main_loop, epilogue1, epilogue2):
    for ko in Tx.serial(PIPE_CYCLE):           # full pipeline cycles
        for ks in Tx.unroll(PIPELINE_DEPTH):    # stages within cycle
            stage = ko * PIPELINE_DEPTH + ks
            main_loop(False, ks)
        phase[0] = phase[0] ^ 1
    if PIPE_REMAIN_NUM > 0:                     # remaining stages
        for ks in Tx.unroll(PIPE_REMAIN_NUM):
            stage = PIPE_CYCLE * PIPELINE_DEPTH + ks
            main_loop(True, ks)
        epilogue1()                              # signal consumer
        for ks in Tx.unroll(PIPE_REMAIN_NUM, PIPELINE_DEPTH):
            epilogue2(ks)                        # drain unused slots
        phase[0] = phase[0] ^ 1
    else:
        epilogue1()
```

**MMA warp (`warp_id < 2`, only `cbx == 0` CTA):**
```python
Tx.gemm_async(tmem[:, warp_id*MMA_N : warp_id*MMA_N+MMA_N],
              A_smem[ks, warp_id, :, :], B_smem[ks, :, :],
              dispatch="tcgen05", cta_group=CTA_GROUP, descI=descI,
              accum=tvm.tir.Not(stage == 0 and ...))  # init on first stage
mma2tma.arrive(ks)  # release SMEM slot immediately after MMA reads it
```

**Consumer epilogue pattern (TMEM → registers → GELU/gate → SMEM → TMA store):**
```python
for i in Tx.unroll(MMA_N // TMEM_LD_SIZE):  # read 64 cols at a time
    col_st = Tx.meta_var(wg_id * MMA_N + i * TMEM_LD_SIZE)
    Tx.copy(reg_wg[:, :], tmem[:, col_st : col_st + TMEM_LD_SIZE])
    with Tx.thread():
        for j in Tx.serial(TMEM_LD_SIZE):
            reg[j] = reg[j] + Tx.cast(bias[n_col + j], "float32")  # bias
            # ... GELU or gate*Y epilogue ...
        Tx.cast(reg_fp16[i*64:(i+1)*64], reg[:])  # f32→f16 vector cast
ld2mma.arrive(wg_id)  # signal MMA can overwrite TMEM

# SMEM→GMEM via TMA store (tiled by EPI_TILE=64 columns)
for i in Tx.unroll(NUM_CONSUMER * BLK_N // EPI_TILE):
    with Tx.thread():
        Tx.copy(D_smem[wg_id, warp_id*32+lane_id, :], reg_fp16[i*EPI_TILE:(i+1)*EPI_TILE])
    Tx.cuda.warpgroup_sync(wg_id)
    Tx.ptx.fence.proxy_async("shared::cta")
    with Tx.thread()[lane_id == 0 and warp_id == 0]:
        Tx.copy_async(D[m_start:m_start+BLK_M, n_start:n_start+EPI_TILE],
                       D_smem[wg_id, :, :], dispatch="tma")
        Tx.ptx.cp_async.bulk.commit_group()
        Tx.ptx.cp_async.bulk.wait_group(0)
    Tx.cuda.warpgroup_sync(wg_id)
```

**Key differences from Pattern 2:**
| Aspect | Pattern 1 (Flux) | Pattern 2 (Based/Mamba2) |
|--------|------------------|--------------------------|
| CTA_GROUP | 2 | 1 |
| Cluster | 2-CTA | Single CTA |
| WGs | 3 (2 consumer + 1 producer) | 2 (1 consumer + 1 producer) |
| MMA API | `Tx.gemm_async(..., dispatch="tcgen05")` | Manual `Tx.ptx.tcgen05.mma(...)` |
| TMEM N_COLS | 512 | 64 |
| MMA tile | 256×256 | 64×64 |
| Multi-phase | No (single GEMM + epilogue) | Yes (2-12 sequential phases) |
| Consumer sync | `mma2ld`/`ld2mma` per WG | `mma2consumer`/`consumer2mma` single pair |
| Register pressure | `Tx.ptx.setmaxnreg(True, 224)` consumer, `setmaxnreg(False, 56)` producer | Not used |
| Tile scheduling | `ClusterPersistentScheduler2D` | Manual persistent CTA loop |

#### Pattern 1 Variant: Single-CTA GEMM (DeepGEMM-style)

Some GEMM kernels use `CTA_GROUP=1` without clustering but still follow the producer/consumer pattern. These are simpler than Flux but follow the same MMA→epilogue flow:

- `test_bhr_hdr_bhd.py`: Batched per-head GEMM, `D[b,h,d] = A[b,h,r] @ B[h,d,r]^T`. Persistent grid over batch×head, no clustering. **Caveat:** A and D must be flattened head-major (`A.permute(1,0,2).reshape(H*B, R)`) to match kernel's `h*B+m` indexing (see pitfall #16). Test dim `B_DIM` must be >= `BLK_M`.
- `test_bmk_bnk_mn.py`: Batch-reduction, `D[m,n] = sum_s A[s,m,k] @ B[s,n,k]^T`. Uses `GroupMajor3D` scheduler with split-K (`TILE_K_BLOCKS=64`, `K_TILES=7`). Each CTA handles 64 K-blocks; TMA store uses `use_tma_reduce="add"` for atomic accumulation. `D_out.zero_()` required before each call. 409 TFLOPS (92% of DeepGEMM).

**Key transcription differences from Flux:**
- No `cbx`/`cby` cluster coordinates — single CTA
- Persistent grid indexed by batch/head (like Pattern 2) instead of tile scheduler
- MMA tile is smaller (64×64 or 128×64 instead of 256×256)
- May use manual `Tx.ptx.tcgen05.mma(...)` instead of `Tx.gemm_async`

#### Pattern 1 Variant: FP8 Block-Scaled GEMM

FP8 GEMM requires per-block scale factors applied to the MMA output. This adds complexity to the producer WG:

- **3 active warps in producer WG:** warp 0 = MMA, warp 2 = scale factor transpose, warp 3 = TMA
- **Scale factor pipeline:** TMA loads scale vectors alongside A/B tiles. Warp 2 transposes scales in SMEM so they align with MMA output layout.
- **Accumulator scaling:** After MMA, scale factors are applied to the f32 accumulator before epilogue.

#### Pattern 1 Variant: TF32 + Fused RMSNorm (TS Mode)

When the A operand needs type conversion (bf16→f32) before MMA, use TS mode where one WG casts data into TMEM and MMA reads it via use_a_tmem:

- **WG0 = cast warps:** Load bf16 A from SMEM, cast to f32, write to TMEM. Also accumulate sqr_sum for RMSNorm.
- **WG1 = MMA (TS mode):** A comes from TMEM (written by WG0), B comes from SMEM. This is a same-SM cross-WG TMEM transfer — works because TMEM is physically shared across WGs on the same SM.
- **Sync:** WG0→WG1 via mbarrier after cast is complete.

### Pattern 3: Thread-Level Compute

**When to use:** Small matmuls (< 128×128), element-wise ops, reductions, or mixed compute where tcgen05 MMA is not applicable.

**Architecture:**
- Single CTA, no clustering
- BDX=8, BDY=32 (256 threads) — avoids division/modulo in indexing
- Shared memory via PoolAllocator
- Persistent CTA loop

**Thread-level matmul (64×64):**
```python
# C[64,64] = A[64,K] @ B^T[64,K], each thread computes [ty*2+r, tx*8+c]
for r in Tx.unroll(2):
    for c in Tx.unroll(8):
        acc[r * 8 + c] = 0.0
for kidx in Tx.serial(K):
    for r in Tx.unroll(2):
        t1[0] = Tx.cast(A_smem[ty * 2 + r, kidx], "float32")
        for c in Tx.unroll(8):
            acc[r * 8 + c] += t1[0] * Tx.cast(B_smem[tx * 8 + c, kidx], "float32")
```

**Reference:** `test_fused_add_rms_norm.py`, `test_rmsnorm.py`

### Pattern 4: Hybrid TMA + Thread Compute (SM100a)

**When to use:** Attention-style kernels with large data movement + moderate compute, where the attention mechanism uses a non-standard warpgroup layout.

**Architecture:**
- TMA for bulk global→shared loads
- Thread-level compute on shared memory tiles
- Pipeline barriers for overlap

**Reference:** `test_flash_attention4.py`

### Pattern 5: Vectorized Element-Wise

**When to use:** Pure element-wise ops (RoPE, activation functions, normalization).

**Architecture:**
- Vectorized loads/stores with `vec_len=VEC`
- Warp shuffle reductions for cross-thread communication
- Minimal shared memory

**Reference:** `test_rope.py`, `test_fused_split_silu_multiply.py`

### Pattern 6: Megakernel Tiles

**When to use:** Fusing multiple operations into a single kernel launch.

**Architecture:**
- `Tx.grid(N)` for tile iteration
- Tile-based scheduling across SM
- Shared memory reuse across tiles

**Reference:** `tests/python/tirx/kernels/cuda/megakernel/`

---

## Transcription Methodology

### Step 0: Verify Source Kernel Hardware Target

**CRITICAL: Only B200/SM100a-native kernels can be transcribed to TIRX SM100a patterns.**

TIRX SM100a patterns (Patterns 1–2) use Blackwell-specific instructions: `tcgen05` MMA, TMEM, TMA with cluster support. These have **no equivalent** on Hopper (SM90a), which uses WGMMA instead. The instruction sets are fundamentally different and not interchangeable.

**Before starting any transcription, confirm:**
1. The source kernel targets **SM100a / Blackwell / B200** (or is architecture-agnostic thread-level code)
2. If the source uses WGMMA (`warpgroup::mma_async`, `wgmma.mma_async`), it is an **SM90a/Hopper kernel** and **cannot be directly transcribed** to TIRX SM100a patterns

**What CAN be transcribed:**
- B200-native kernels (using tcgen05 MMA, TMEM) → Pattern 1 or 2
- Architecture-agnostic CUDA/Triton kernels (thread-level math, no hardware MMA) → Pattern 3 or 5
- Algorithm descriptions / pseudocode → any pattern (design from scratch)

**What CANNOT be transcribed:**
- SM90a/Hopper WGMMA kernels → the WGMMA calls have no tcgen05 equivalent; the entire warp-specialization structure, barrier protocol, and data flow must be **redesigned from scratch** for SM100a
- SM80/Ampere `mma.sync` kernels → same issue, different MMA instruction set

**Example — ThunderKittens (TK):**
- TK has B200-native GEMM kernels (`bf16_b200`, `fp8_b200`, `mxfp8_b200`, `nvfp4_b200`) → these CAN be transcribed
- TK's attention/SSM kernels (`based`, `fftconv`, `hedgehog`, `linear_attention`, `mamba2`) use **Hopper WGMMA only** → these CANNOT be transcribed directly. The TIRX versions of these kernels were **redesigned from scratch** using SM100a tcgen05 instructions, not mechanically transcribed from TK source

**Example — DeepGEMM:**
- DeepGEMM GEMM kernels (fp8, bf16, tf32 variants) target SM90a (Hopper) but the algorithm is architecture-agnostic. The TIRX versions were redesigned for SM100a tcgen05 instructions, preserving the same data flow (persistent scheduling, pipelined K-reduction, block-scaled FP8) but using completely different MMA and barrier primitives.
- Key DeepGEMM patterns transcribed: FP8 block-scaled GEMM (`test_fp8_bhr_hdr_bhd.py`), TF32 with fused RMSNorm (`test_tf32_hc_prenorm_gemm.py`), batch-reduction GEMM (`test_bmk_bnk_mn.py`)

**How to check:** Look for `warpgroup::mma_async`, `wgmma`, `#ifdef KITTENS_HOPPER` in the source. If present, it's Hopper-only. Look for `tcgen05`, `#ifdef KITTENS_BLACKWELL`, `GPU=B200` for B200-native code.

### Step 1: Analyze the Source Kernel

1. **Identify compute pattern**: Is it GEMM-dominated? Element-wise? Mixed?
2. **Measure matmul sizes**: M, N, K dimensions determine tcgen05 vs thread-level
3. **Map data flow**: What goes global→shared→register? What's reused?
4. **Identify synchronization points**: Where are barriers needed?

### Step 2: Choose Pattern

| Source kernel characteristic | TIRX pattern |
|------------------------------|--------------|
| Large single GEMM (M,N >= 128) with epilogue | Pattern 1: 2-CTA Cluster GEMM |
| Multi-phase MMA with interleaved consumer compute (attention, SSM, FFT) | Pattern 2: 1-Consumer Warp-Spec |
| Small matmuls (< 128) or no matmuls | Pattern 3: Thread-level |
| Large data loads + moderate compute | Pattern 4: Hybrid TMA |
| Pure element-wise / reductions | Pattern 5: Vectorized |
| Multi-op fusion | Pattern 6: Megakernel |

### Step 3: Map Constructs

**From CUDA:**

| CUDA | TIRX |
|------|------|
| `__shared__ float smem[M][N]` | `pool.alloc([M, N], "float32")` |
| `threadIdx.x` | `tx` from `Tx.thread_id(...)` |
| `blockIdx.x` | `bx` from `Tx.cta_id(...)` |
| `__syncthreads()` | `Tx.cuda.cta_sync()` |
| `atomicAdd(&smem[i], val)` | `smem[i] = smem[i] + val` (with proper sync) |
| `__shfl_xor_sync(mask, v, d)` | `Tx.tvm_warp_shuffle_xor(mask, v, d, 32, 32)` |
| `expf(x)` | `Tx.exp(x)` |
| `__float2half(x)` | `Tx.cast(x, "float16")` |
| `if (threadIdx.x == 0)` | `if thread_id == 0:` |
| `for (int i=0; i<N; i++)` | `for i in Tx.serial(N):` |
| `#pragma unroll` | `for i in Tx.unroll(N):` |

**From Triton:**

| Triton | TIRX |
|--------|------|
| `tl.load(ptr + offsets)` | `Tx.copy(vec[:], buf[...], vec_len=VEC)` |
| `tl.store(ptr + offsets, val)` | `Tx.copy(buf[...], vec[:], vec_len=VEC)` |
| `tl.dot(a, b)` | Thread-level matmul loop or `Tx.gemm_async` |
| `tl.program_id(0)` | `bx = Tx.cta_id([N], parent="kernel")` |
| `tl.arange(0, N)` | Explicit thread-id-based indexing |
| `tl.sum(x, axis=0)` | Warp shuffle reduction |
| `tl.where(cond, a, b)` | `Tx.if_then_else(cond, a, b)` |

### Step 4: Handle Dynamic Shapes

```python
# Declare dynamic dimension
total_batch = Tx.int32()
seq_len = Tx.int32()
buf = Tx.match_buffer(ptr, [total_batch, seq_len, D], "float16", scope="global")

# Use in runtime expressions
nc[0] = seq_len // CHUNK  # computed at runtime
```

### Step 5: Persistent Kernel Pattern

```python
wid[0] = bx  # CTA index
while wid[0] < total_work:
    # ... process work item ...
    wid[0] = wid[0] + SM_COUNT
```

For sequential dependencies (recurrent state across chunks):
```python
wid[0] = bx
while wid[0] < total_batch:
    # init state
    cid[0] = 0
    while cid[0] < num_chunks:
        # process chunk, update state
        cid[0] = cid[0] + 1
    wid[0] = wid[0] + SM_COUNT
```

---

## Common Pitfalls

### 1. `@Tx.meta_class` required for helper classes
Any class instantiated inside `@Tx.prim_func` needs `@Tx.meta_class`. However, **barrier classes should NOT be custom-defined** — use `MBarrier`, `TMABar`, `TCGen05Bar` from `tvm.tirx.pipeline` instead (they already have `@Tx.meta_class`). Only use `@Tx.meta_class` for non-barrier helpers (e.g., `CastStageHelper` in TF32 kernels).
```python
@Tx.meta_class
class MyHelper:
    ...
```
**Error without it:** `Cannot automatically inference the type. value=<...object at ...>`

### 2. No `//` or `%` on meta_var for buffer indices
```python
# WRONG — produces OpaquePyObject error
idx = Tx.meta_var(flat // D)
buf[idx] = val

# CORRECT — design thread grid to avoid division
# BDX=8, BDY=32: row = ty*2+r, col = tx*8+c
buf[ty * 2 + r, tx * 8 + c] = val
```

### 3. `Tx.serial` vs `Tx.unroll`
- `Tx.unroll(N)`: N must be small compile-time constant. Loop var is compile-time.
- `Tx.serial(N)`: Runtime loop. Loop var can index shared memory.
- Use `Tx.serial` for matmul K-loops (K=64), `Tx.unroll` for small tile iterations.

### 4. No `+=` on shared memory
```python
# May not work:
smem[i, j] += val
# Correct:
smem[i, j] = smem[i, j] + val
```

### 5. `pool.commit()` placement
Must be called after ALL `pool.alloc()` calls, before any use of allocated buffers. Commit finalizes the shared memory layout.

### 6. `vec_len` parameter
Required for global memory vector loads/stores to generate efficient code:
```python
Tx.copy(vec[:], global_buf[row, col : col + VEC], vec_len=VEC)
```
Not needed for shared memory ↔ register copies.

### 7. Scalar local variables
Use 1-element arrays for mutable scalars (loop counters, accumulators):
```python
wid = Tx.alloc_local([1], "int32")
wid[0] = bx
```

### 8. `Tx.meta_var` for compile-time expressions
Use for expressions that should be evaluated at compile time:
```python
thread_id = Tx.meta_var(ty * BDX + tx)
is_leader = Tx.meta_var(bx == 0)
```

### 9. SwizzleLayout SMEM alignment for TMA (SM100a)
Any SMEM buffer with `SwizzleLayout` that participates in a TMA copy (`Tx.copy_async` with `dispatch="tma"`) must have its byte offset aligned to the **swizzle period**.

| SwizzleLayout | Swizzle Atom | Period (bytes) |
|---------------|-------------|----------------|
| `SwizzleLayout(1,_,_)` | 32B | 32 × 2^S rows |
| `SwizzleLayout(2,_,_)` | 64B | 64 × 2^S rows |
| `SwizzleLayout(3,3,3)` | 128B | 1024 bytes (= 8 rows × 128B) |

**Failure mode:** No CUDA error. Output values are all individually correct (min/max matches reference) but appear at wrong column positions — a deterministic column permutation caused by the swizzle XOR being applied with a row offset.

**Rule:** Always lay out SMEM with swizzled buffers first, non-swizzled buffers (scalar arrays, f32 accumulators, small scratch) last. Consecutive swizzled tiles of size `TILE_M × TILE_N × sizeof(dtype)` are naturally aligned when each tile is a multiple of the swizzle period (e.g., 64×64×2B = 8192B, divisible by 1024). Inserting a non-swizzled buffer (even a small one like 256 bytes) between swizzled buffers breaks alignment.

**Debugging:** If output has correct values in wrong positions, use identity-matrix inputs to expose the permutation pattern, then verify `byte_offset % period == 0` for every swizzled buffer.

### 10. Barrier phase tracking in persistent/multi-iteration kernels
`mbarrier.try_wait(addr, phase)` waits for the barrier to flip FROM the given phase. Each arrive-wait cycle flips the phase once. When a single iteration (chunk/tile) uses the same barrier N times:
- **N even** → barrier returns to original phase → do NOT flip phase tracking var between iterations
- **N odd** → barrier ends flipped → flip phase tracking var once between iterations

**Common mistake:** Unconditionally flipping the phase variable at the end of each iteration. This causes the second iteration to skip the wait (passes immediately with wrong phase), producing correct output for iteration 1 but wrong output for all subsequent iterations. **Symptom:** single-iteration test passes, multi-iteration test fails with ~50% correct elements (first half correct, rest wrong).

### 11. TMA requires compact global layout — flatten batch dimensions
TMA `copy_async` requires the accessed global buffer dimensions to be contiguous in memory. With `[B, H, N, D]` buffers and symbolic indexing `buf[wid // H, wid % H, ...]`, TVM cannot prove compactness and fails with:
```
ValueError: Global layout is not compact: stride=D * N * H * extent=B
```
**Fix:** Flatten batch+head into a single dimension: `buf[total_bh, N, D]` where `total_bh = B * H`. Index directly with `wid`. Reshape in the test function:
```python
q_tvm = tvm.runtime.tensor(q.reshape(B*H, N, D).numpy(), DEV)
result = out_tvm.numpy().reshape(B, H, N, D)
```

### 12. PIPE_DEPTH=2 chunk-0 release hazard
With `PIPE_DEPTH=2`, the TMA pre-fills both slots (tic=0 for chunk 0, tic=1 for chunk 1) before MMA starts. If the MMA unconditionally releases `mma2tma[prev_tic]` after every chunk, chunk 0 releases slot `prev_tic=1` — which still holds chunk 1's fresh TMA data. TMA then overwrites it with chunk 2's data before MMA reads it as "previous" in chunk 2.

**Fix:** Guard the release with `if cid_mma > 0:`. Phase tracking remains correct because TMA's `try_wait` for slot 1 waits for the first completion, which comes from chunk 1's release (not chunk 0's skipped release).

**Symptom:** Chunks 0 and 1 produce correct output, chunk 2+ diverges. Narrowing with `alphas=0` (disable linear attention) still shows divergence, confirming it's a data-movement bug not a compute bug.

### 13. TMA races ahead at persistent grid work item boundaries (BH > SM_COUNT)
When `total_bh > SM_COUNT=148`, each CTA handles multiple work items (`while wid < total_bh: wid += SM_COUNT`). TMA's inner loop finishes much faster than MMA/consumer (it just issues DMA commands). When TMA enters the next work item, it skips `mma2tma_bar.wait` for the first `PIPE_DEPTH` chunks (`if cid_tma >= PIPE_DEPTH:` guard), overwriting SMEM that MMA/consumer from the previous work item is still reading.

**Symptom:** CUDA error 719 (illegal instruction / unspecified launch failure) when BH > 148. Works fine for BH <= 148 (each CTA handles exactly one work item).

**Fix:** Add a `workitem_sync` mbarrier — MMA arrives after completing each work item's inner loop, TMA waits before starting each non-first work item. See Pattern 2 documentation for full code pattern.

### 14. Unbalanced mma2tma release with prev_tic pattern (hedgehog)
Kernels that release `mma2tma[prev_tic]` (instead of `mma2tma[tic]`) skip one slot per work item — the final chunk's `tic` is never released. In single-work-item mode this is harmless, but with persistent grid (BH > SM_COUNT) the unreleased slot causes `mma2tma` barrier phase drift: TMA's `try_wait` in the next work item sees a stale banked completion and passes immediately, reading SMEM before MMA has finished.

**Symptom:** Correctness errors (small % of output elements diverge) only when BH > SM_COUNT. BH <= SM_COUNT passes. Not a crash — the kernel runs but produces wrong results.

**Fix:** After the MMA inner chunk loop, release the final chunk's slot before `workitem_sync`:
```python
# After inner loop, before workitem_sync arrive:
with Tx.thread()[Tx.ptx.elect_sync()]:
    last_tic = Tx.meta_var((nc_mma - 1) % PIPE_DEPTH)
    Tx.ptx.mbarrier.arrive(mma2tma_bar.ptr_to([last_tic]))
    Tx.ptx.mbarrier.arrive(workitem_sync_bar.ptr_to([0]))
```

### 15. transA/transB descriptor semantics
The `transA` and `transB` flags in `encode_instr_descriptor` control how the MMA reads matrix data from SMEM:
- `transA=False`: A is stored as `[M, K]` — rows are M, columns are K
- `transA=True`: A is stored as `[K, M]` — rows are K, columns are M. MMA internally transposes to get `[M, K]`
- `transB=False`: B is stored as `[N, K]` — rows are N, columns are K
- `transB=True`: B is stored as `[K, N]` — rows are K, columns are N

**Critical: when transA=True, the A descriptor pointer iterates K along rows.**
For ki-th K-iteration: `ptr = smem.ptr_to([ki * MMA_K, col_offset])`, not `[0, ki * MMA_K]`.
The SDO must correspond to the full buffer row stride, not the chunk width.

**Example: KK^T @ V using transA (from Based kernel Phases 4-7):**
```python
# KK_smem[64, 256] stores KK. Want KK_chunk_i^T[64,64] @ V[64,64].
# KK_chunk_i = KK[:, i*64:(i+1)*64], shape [64, 64] in SMEM.
# With transA=True: MMA reads A as [K=64, M=64], transposes to [M=64, K=64].
# This gives us KK_chunk_i^T — exactly what we need.
for ki in Tx.unroll(CHUNK // MMA_K):  # 4 iterations
    Tx.ptx.tcgen05.encode_matrix_descriptor(
        Tx.address_of(descA), qqt_kk_smem.ptr_to([ki * MMA_K, chunk_i * CHUNK]),
        MMA_LDO_WIDE, MMA_SDO_WIDE, SWIZZLE_V)  # SDO for full [64,256] buffer = 256
```

### 16. Batched tensor flatten ordering must match kernel indexing

When flattening multi-dimensional tensors (e.g., `[B, H, R]`) for TMA-compatible 2D buffers, the flatten order determines which dimension's rows are contiguous. **The kernel's indexing formula must match.**

**Example (BHR·HDR→BHD kernel):**
- Kernel indexes A as `row = h_idx * B_DIM + m_start` (head-major: each head's batch rows are contiguous)
- `prepare_data()` must flatten A as head-major: `A.permute(1, 0, 2).reshape(H*B, R)` → `[H*B, R]`
- NOT `A.reshape(B*H, R)` which is batch-major: row `(b,h)` = `b*H + h` (batch rows contiguous per batch, NOT per head)

**Why it matters:** TMA loads `BLK_M` contiguous rows. With head-major indexing but batch-major flatten, TMA loads rows from different heads — producing garbage.

**General rule:**
| Kernel indexes as | Required flatten |
|-------------------|-----------------|
| `h * B + b` (head-major) | `tensor.permute(1, 0, ...).reshape(H*B, ...)` |
| `b * H + h` (batch-major) | `tensor.reshape(B*H, ...)` |

**Symptom:** Large rel_diff (8+), all values wrong. Not a crash, not a position error — just completely wrong numbers because TMA loads wrong data.

**Both A (input) and D (output) must use the same ordering.** B (weight, indexed by `h * D + n`) uses its natural `[H, D, R]` → `[H*D, R]` flatten.

### 17. SMEM release timing with many MMA phases
When adding more MMA phases that read SMEM (e.g., V_smem), the `mma2tma_bar.arrive(tic)` release must be deferred until the **last phase** that reads from that SMEM slot. In Based's 7-phase design, V_smem is read in Phases 2 and 4-7, so release happens after Phase 7 (not Phase 2).

**Rule:** Trace all SMEM reads across all phases. Release each slot only after its last reader completes.

### 18. Debugging SM100a warp-specialized kernel correctness
Systematic checklist when output is wrong:
1. **Values correct but wrong positions** → swizzle alignment (pitfall #9). Use identity inputs to confirm.
2. **First chunk/tile correct, rest wrong** → barrier phase tracking (pitfall #10). Count total transitions per iteration.
3. **All values wrong/garbage** → likely a race condition, missing sync, or wrong SMEM indexing.
4. **Incorrect matmul results** → verify transA/transB descriptor matches data layout (pitfall #15). Wrong transA/transB produces incorrect results, not crashes.
5. **SASS comparison red herring**: Control word differences between working and broken kernels are often a symptom of different code structure, not the root cause. Don't waste time patching SASS bytes.
6. **Intercept compilation** for debugging: `tvm_ffi.register_global_func("tvm_callback_cuda_compile", override=True)` to save intermediates, add `-Xptxas -v`, or inspect generated CUDA.

### 19. SwizzleLayout parameter formula for TMA 128B swizzle (CRITICAL)
The correct SwizzleLayout for TMA 128B swizzle depends on the element size:
```
SwizzleLayout(per_element=log2(16/sizeof(dtype)), swizzle_len=3, atom_len=3, swizzle_inner=True)
```

| dtype | sizeof | per_element | SwizzleLayout |
|-------|--------|-------------|---------------|
| fp8   | 1B     | log2(16) = 4 | (4, 3, 3) |
| bf16  | 2B     | log2(8) = 3  | (3, 3, 3) |
| f32   | 4B     | log2(4) = 2  | (2, 3, 3) |

**Common mistake:** Using `(3, 2, 3)` for f32 — this has `swizzle_len=2` (XORs only 2 bits instead of 3), producing 50% incorrect element positions. The `(2, 3, 3)` variant XORs all 3 bank-group bits correctly.

**Verification method:**
```python
from tvm.tir.layout import SwizzleLayout, ComposeLayout, TileLayout, S
layout = ComposeLayout(SwizzleLayout(2, 3, 3, swizzle_inner=True), TileLayout(S[(M, N):(N, 1)]))
# Compare layout.apply(row*N+col) against TMA formula:
# physical = row*N + ((col/bg_size) ^ (row%8)) * bg_size + (col%bg_size)
# where bg_size = 16/sizeof(dtype)
```

**Symptom:** D output has ~50% calc_diff (0.5-0.9). Values are individually correct but in wrong column positions — a deterministic column permutation within each 128B row.

### 20. `__shfl_sync` deadlock inside `elect_sync()` or conditional branches
`__shfl_sync(0xffffffff, ...)` requires ALL 32 lanes to participate. Calling it inside:
- `if elect_sync():` (single-lane guard) → 31 lanes blocked, deadlock
- `if lane_id < N:` (partial-lane guard) → (32-N) lanes blocked, deadlock

**Fix:** Move `__shfl_sync` operations outside the guard. For SmemDescriptor, call `init()` (which uses `__shfl_sync`) at the warp level before entering `elect_sync()`.

**Also applies to:** `__shfl_xor_sync` in warp reductions. If the reduction must run inside a guarded block, use `alloc_local` buffers to force evaluation of the shuffle results before the if-guard:
```python
reduced_buf = Tx.alloc_local([1], "float32")
with Tx.thread():
    reduced_buf[0] = warp_reduce_sum4(value)  # shuffle runs for all 32 lanes
    if lane_id % 4 == 0:
        output[idx] = reduced_buf[0]  # only write from lane 0 of each group
```

### 21. B descriptor SBO uses BLOCK_SWIZZLED_BK, not full BLOCK_K
For the UMMA B descriptor's stride byte offset (SBO):
```python
sdo = 8 * BLOCK_SWIZZLED_BK * sizeof(dtype) // 16  # in 16B units
```
Where `BLOCK_SWIZZLED_BK = 128 // sizeof(dtype)` (elements per 128B swizzle atom).

**Common mistake:** Using `BLOCK_K` instead of `BLOCK_SWIZZLED_BK`, which doubles the SBO when `BLOCK_K > BLOCK_SWIZZLED_BK`. This causes the MMA to read B data from wrong SMEM addresses as it strides across the N dimension.

**Symptom:** calc_diff ~0.5-0.9 with the B matrix contribution scrambled.

### 22. Cast warp barrier ordering: LDSM before empty_cast_barriers.wait

In TF32 TS-mode kernels (Pattern 1 variant), the cast warp loads bf16 A from SMEM (LDSM), then stores f32 to TMEM. The naive transcription places `empty_cast_barriers.wait` (wait for TMEM to become empty) BEFORE the LDSM loads:

```
full_barriers.wait  →  empty_cast_barriers.wait  →  LDSM(SMEM)  →  TMEM store
```

**Correct order (matching DeepGEMM):**
```
full_barriers.wait  →  LDSM(SMEM)  →  empty_cast_barriers.wait  →  TMEM store
```

LDSM reads from SMEM, which has no dependency on TMEM being empty. By doing LDSM first, the SMEM reads overlap with MMA still consuming TMEM data. The TMEM wait only blocks right before the TMEM store that actually needs it.

**Impact:** ~10% performance improvement (e.g., 39.91 → ~45 TFLOPS in TF32 HC prenorm GEMM).

**Implementation in inline PTX:** Move `empty_cast_barriers.wait` inside the `cast_bf16_store_tmem` inline function, between the LDSM load block and the TMEM store block. Pass the barrier pointer and phase as additional function arguments.

### 23. K-splitting: 3D TMA tensor map for split-K output

When implementing split-K (multiple CTAs per M-block, each handling K/kNumSplits iterations), the D output tensor becomes 3D `(kNumSplits, M, N)`:

**TMA tensor map (3D):**
```python
Tx.call_packed("runtime.cuTensorMapEncodeTiled",
               D_tensor_map, d_type, 3, D.data,        # rank=3
               N, M, kNumSplits,                         # globalDim: (inner=N, mid=M, outer=kNumSplits)
               N * F32_BYTES,                             # globalStride[0] (M dim, in bytes)
               M * N * F32_BYTES,                         # globalStride[1] (split dim, in bytes)
               BLOCK_N, BLOCK_M, 1,                       # boxDim
               1, 1, 1,                                   # boxStride (all 1)
               0, 3, 0, 0)                               # interleave, swizzle, l2, oob
```

**TMA store (3D):**
```python
Tx.ptx.cp_async.bulk.tensor.s2g(
    3,                  # 3D
    D_smem.ptr_to([0, 0]),
    D_tensor_map,
    0,                  # inner coord (N)
    m_st,               # mid coord (M)
    k_split_idx,        # outer coord (split)
)
```

**Key differences from 2D:**
- `rank=3` in `cuTensorMapEncodeTiled` → requires 3 globalDim, 2 globalStrides, 3 boxDim, 3 boxStrides, 4 trailing params = 19 args total
- `s2g(3, ...)` with 3 coordinates instead of 2
- sqr_sum uses flat `(kNumSplits * M,)` buffer with `m_offset = M * k_split_idx`
- Host-side reduction after kernel: `D_out = D_splits.sum(dim=0)`, `sqr_sum = sqr_sum_splits.view(kNumSplits, M).sum(dim=0)`

**Grid mapping:** `bx // kNumSplits = m_block_idx`, `bx % kNumSplits = k_split_idx`, total CTAs = `num_m_blocks * kNumSplits`.

### 24. TMEM Layout F epilogue (BLOCK_M=64): only 16 lanes valid per warp
For BLOCK_M=64, the MMA output occupies physical TMEM rows {0-15, 32-47, 64-79, 96-111} (Layout F). Each 32-lane warp block has only 16 valid rows. The epilogue must guard SMEM stores:
```python
if lane_id < BLOCK_M // WARP_NUMBER:  # lane_id < 16 for BLOCK_M=64, 4 warps
    Tx.copy(D_smem[warp_id * 16 + lane_id, col:col+N], reg[:])
Tx.cuda.warp_sync()  # required since conditional store
```

**Common mistake:** Using `warp_id * 32 + lane_id` (mapping 2 warps to 64 rows, ignoring Layout F) or omitting the `lane_id < 16` guard.

### 25. SMEM pipeline depth must match DeepGEMM's heuristic

DeepGEMM auto-selects `kNumStages` by searching from 32 downward until SMEM fits. When transcribing, don't hardcode a small `SMEM_PIPE_DEPTH` (e.g., 4) if the SMEM budget allows more stages.

**Formula:**
```python
smem_per_stage = BLK_M * BLK_K * sizeof(A) + BLK_N * BLK_K * sizeof(B)
smem_fixed = 1024 + TMEM_PIPE_DEPTH * EPI_TILE * MMA_N * sizeof(D) + barriers
max_depth = (SMEM_CAPACITY - smem_fixed) // smem_per_stage
SMEM_PIPE_DEPTH = min(max_depth, NUM_K_ITERS)  # no point exceeding K iterations
```

**Example (BHR-HDR-BHD):** BLK_M=128, BLK_N=128, BLK_K=64, bf16 → 32KB/stage. With SMEM=228KB, fixed~17KB → max 6 stages. Changing from 4→6: **60% → 95% of DeepGEMM** (B=1024,R=512), **60% → 115%** (B=128,R=128).

**Rule of thumb:** Always compute max SMEM_PIPE_DEPTH and use it. Under-staging leaves TMA latency unhidden.

---

## Infrastructure Utilities

### Pipeline State and Barriers (`tvm.tirx.pipeline`) — **REQUIRED for all kernels**

**Do NOT define custom barrier classes.** Use the reusable classes from `tvm.tirx.pipeline` instead. They handle init, wait, arrive, and phase tracking correctly.

```python
from tvm.tirx.pipeline import PipelineState, MBarrier, TMABar, TCGen05Bar

# ---- Pipeline state: tracks stage index + phase for producer/consumer ----
ps = PipelineState("name", pipe_depth=N)
ps.init(is_producer=True)   # producer starts phase=1, consumer starts phase=0
ps.move_to_next_stage()      # increments stage, flips phase at wrap

# ---- Barrier classes (allocated from PoolAllocator) ----
pool = Tx.PoolAllocator()    # bump allocator over shared.dyn

tma2mma = TMABar(pool, depth=SMEM_PIPE_DEPTH)    # TMA signals via expect_tx
mma2tma = TCGen05Bar(pool, depth=SMEM_PIPE_DEPTH) # MMA signals via tcgen05.commit
mma2ld  = TCGen05Bar(pool, depth=TMEM_PIPE_DEPTH) # MMA → consumer TMEM load
ld2mma  = MBarrier(pool, depth=TMEM_PIPE_DEPTH)   # consumer → MMA (regular arrive)

# Init barriers
tma2mma.init(count=1)        # 1 TMA thread arrives
mma2tma.init(count=1)        # tcgen05.commit — thread count doesn't matter
mma2ld.init(count=1)
ld2mma.init(count=128)       # all 128 consumer threads must arrive

# Usage
tma2mma.arrive(ks, tx_count=BLK_K * (BLK_M + BLK_N) * F16_BYTES)  # TMA expected bytes
tma2mma.wait(ks, phase)
mma2tma.arrive(ks, cta_group=CTA_GROUP, cta_mask=1)  # tcgen05 commit
mma2tma.wait(ks, phase)
ld2mma.arrive(idx)           # regular mbarrier arrive
ld2mma.wait(idx, phase)
```

**Barrier type selection:**

| Source → Destination | Signal mechanism | Class |
|---------------------|-----------------|-------|
| TMA → MMA | `mbarrier.arrive.expect_tx` (byte count) | `TMABar` |
| MMA → TMA/consumer | `tcgen05.commit` (cta_group, cta_mask) | `TCGen05Bar` |
| Consumer → MMA | `mbarrier.arrive` (regular) | `MBarrier` |
| CTA_GROUP=2 cross-CTA | Use `mbar.remote_view(rank=0)` for CTA1 to see CTA0's barriers | `MBarrier.remote_view()` |

**All transcribed kernels** (Based, Hedgehog, FFTConv, Mamba2, Linear Attention, BMK-BNK-MN, BHR-HDR-BHD, FP8-BHR-HDR-BHD, DeepGEMM, TF32-HC-Prenorm-GEMM) have been migrated to `pipeline.py` classes + `PoolAllocator`. No kernel should use custom `class Barriers` anymore.

### Tile Schedulers (`tvm.tirx.tile_scheduler`)
```python
from tvm.tirx.tile_scheduler import ClusterPersistentScheduler2D

scheduler = ClusterPersistentScheduler2D(
    M_tiles, N_tiles,
    cta_m_in_cluster=M_CLUSTER,
    cta_n_in_cluster=1,
    pool=pool
)
# scheduler.init(...) → scheduler.next(...) loop
```

Available schedulers: `ClusterPersistentScheduler2D`, `FlashAttentionLinearScheduler`, `FlashAttentionLPTScheduler`, `ParallelPersistentScheduler2D`, and more.

### Benchmarking (`tvm.tirx.bench.utils`)

**Rule: Always benchmark against the source library.** When a kernel is transcribed from a specific library (DeepGEMM, ThunderKittens, etc.), the benchmark's `std()` baseline MUST use that library's API, not `torch.einsum`. This gives an apples-to-apples performance comparison.

```python
from tvm.tirx.bench.utils import ProtonContext, bench

ms = bench(func, warmup=10, repeat=30, proton_name="name")
with ProtonContext("label"):
    run_tests()
```

#### DeepGEMM Library Benchmarks

For kernels transcribed from DeepGEMM, use `deep_gemm.einsum` / `deep_gemm.fp8_einsum` as the baseline:

```python
import deep_gemm
from deep_gemm.utils.math import per_block_cast_to_fp8, per_token_cast_to_fp8

# BF16 einsum (bhr,hdr->bhd / bmk,bnk->mn / etc.)
out = torch.empty((...), dtype=torch.bfloat16, device="cuda")
deep_gemm.einsum("bhr,hdr->bhd", A, B, out)

# BF16 bmk,bnk->mn with f32 accumulation: must pass c=out and zero before each call
out = torch.zeros((M, N), dtype=torch.float32, device="cuda")
def run():
    out.zero_()
    deep_gemm.einsum("bmk,bnk->mn", A, B, out, c=out)

# FP8 einsum: inputs are (tensor, scales) tuples
# A: per-token quantization, B: per-block quantization (per-head for batched)
x_fp8 = per_token_cast_to_fp8(A.view(-1, R), use_ue8m0=True)
x_fp8 = x_fp8[0].view(B, H, R), x_fp8[1].view(B, H, ceildiv(R, 128))
y_fp8 = (torch.empty_like(B_mat, dtype=torch.float8_e4m3fn),
         torch.empty((H, ceildiv(D, 128), ceildiv(R, 128)), device="cuda", dtype=torch.float))
for i in range(H):
    y_fp8[0][i], y_fp8[1][i] = per_block_cast_to_fp8(B_mat[i], use_ue8m0=True)
deep_gemm.fp8_einsum("bhr,hdr->bhd", x_fp8, y_fp8, z)
```

**Supported einsum expressions** (check `DeepGEMM/tests/test_einsum.py` for the full list):
- `bhr,hdr->bhd` (batched per-head)
- `bhd,hdr->bhr` (batched per-head, reversed)
- `bmk,bnk->mn` (batch-reduction)
- Same patterns with `fp8_einsum` for FP8 variants

**Note**: `bmk,bnk->mn` with f32 output requires `c=out` parameter for accumulation. Without zeroing `out` before each call, results accumulate across iterations → segfault or wrong results.

---

## Compilation & Testing

```python
import tvm

target = tvm.target.Target("cuda")
with target:
    mod = tvm.IRModule({"main": get_kernel()})
    mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
    src = mod.mod.imports[0].inspect_source()  # view generated CUDA
    print(src[:2000])

DEV = tvm.cuda(0)
buf_tvm = tvm.runtime.tensor(np_array, DEV)
mod(buf_tvm, ...)
result = buf_tvm.numpy()

np.testing.assert_allclose(result, reference, rtol=5e-2, atol=5e-2)
```

---

## File Templates

### Thread-Level Template (Pattern 3)
```python
import math, pytest, numpy as np, torch
import tvm
from tvm.script import tirx as Tx

SM_COUNT = 148; BDX = 8; BDY = 32; VEC = 8

def get_kernel():
    @Tx.prim_func(tirx=True)
    def kernel(ptr1: Tx.handle, ...):
        dim1 = Tx.int32()
        buf1 = Tx.match_buffer(ptr1, [dim1, ...], "float16", scope="global")
        with Tx.kernel():
            bx = Tx.cta_id([SM_COUNT], parent="kernel")
            tx, ty = Tx.thread_id([BDX, BDY], parent="cta")
            with Tx.cta():
                pool = Tx.PoolAllocator()
                smem = pool.alloc([...], "float16")
                pool.commit()
                with Tx.thread():
                    # kernel body
    return kernel
```

### 1-Consumer Warp-Spec Template (Pattern 2)
```python
import math, pytest, numpy as np, torch
import tvm
from tvm.script import tirx as Tx
from tvm.tir.layout import TileLayout, tid_in_wg, TLane, TCol, S

SM_COUNT = 148
WG_NUMBER = 2; WARP_NUMBER = 4; NUM_THREADS = WG_NUMBER * WARP_NUMBER * 32
CTA_GROUP = 1; PIPE_DEPTH = 2
MMA_M = 64; MMA_N = 64; MMA_K = 16; SWIZZLE = 3
TMEM_ROWS = 128; N_COLS = 64; TMEM_LD_SIZE = 64

half_pipe_layout = Tx.ComposeLayout(
    Tx.SwizzleLayout(SWIZZLE, SWIZZLE, SWIZZLE, swizzle_inner=True),
    Tx.TileLayout(S[(PIPE_DEPTH, CHUNK, HALF) : (CHUNK * HALF, HALF, 1)]),
)
half_layout = Tx.ComposeLayout(
    Tx.SwizzleLayout(SWIZZLE, SWIZZLE, SWIZZLE, swizzle_inner=True),
    Tx.TileLayout(S[(CHUNK, HALF) : (HALF, 1)]),
)

def get_kernel():
    @Tx.prim_func(tirx=True)
    def kernel(ptr1: Tx.handle, ...):
        total_bh = Tx.int32()
        buf = Tx.match_buffer(ptr1, [total_bh, N, D], "float16", scope="global")
        with Tx.kernel():
            bx = Tx.cta_id([SM_COUNT], parent="kernel")
            wg_id = Tx.warpgroup_id([WG_NUMBER], parent="cta")
            warp_id = Tx.warp_id([WARP_NUMBER], parent="warpgroup")
            lane_id = Tx.thread_id([32], parent="warp")

            with Tx.cta():
                # ---- Shared memory allocation via PoolAllocator ----
                pool = Tx.meta_var(Tx.PoolAllocator())
                tmem_addr = Tx.decl_scalar("uint32", pool.ptr, scope="shared.dyn", elem_offset=0)
                pool.move_base_to(8)

                # Barriers (from tvm.tirx.pipeline)
                tma2mma_bar = TMABar(pool, PIPE_DEPTH)
                mma2tma_bar = TCGen05Bar(pool, PIPE_DEPTH)
                mma2consumer_bar = TCGen05Bar(pool, 1)
                consumer2mma_bar = MBarrier(pool, 1)
                workitem_sync_bar = TMABar(pool, 1)

                # Swizzled buffers require 1024-byte alignment
                pool.move_base_to(1024)
                # Swizzled buffers via pool.alloc(..., layout=swizzled_layout)
                # Non-swizzled buffers (f32 state, scalars) last
                pool.commit()

                with Tx.warp()[0:1]:
                    Tx.ptx.tcgen05.alloc(Tx.address_of(tmem_addr), n_cols=N_COLS, cta_group=CTA_GROUP)
                    Tx.cuda.warp_sync()
                # ... barrier init, constant loading ...
                Tx.cuda.cta_sync()
                tmem = Tx.decl_buffer((TMEM_ROWS, N_COLS), "float32", scope="tmem",
                                      allocated_addr=0,
                                      layout=TileLayout(S[(TMEM_ROWS, N_COLS) : (1@TLane, 1@TCol)]))

                with Tx.cta():
                    Tx.attr({"tirx.scope_partition": True})

                    # === Producer warpgroup (WG1): TMA + MMA warps ===
                    with Tx.warpgroup()[1:2]:
                        if warp_id == 3:
                            # TMA producer warp
                            with Tx.thread()[Tx.ptx.elect_sync()]:
                                # TMA loads, signal tma2mma
                        elif warp_id == 0:
                            # MMA warp
                            with Tx.thread()[Tx.ptx.elect_sync()]:
                                # encode descI, descI_tb
                                # multi-phase tcgen05.mma sequence

                    # === Consumer warpgroup (WG0) ===
                    with Tx.warpgroup()[0:1]:
                        Tx.cuda.trap_when_assert_failed(tmem_addr == 0)
                        # read TMEM, softmax, state update, TMA store
    return kernel
```

### Test Function Template
```python
@pytest.mark.parametrize("B,H,N", [(1,2,128), (2,2,256)])
def test_kernel(B, H, N):
    data = prepare_data(B, H, N)
    ref = naive_reference(*data)
    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.compile(tvm.IRModule({"main": get_kernel()}), target=target, tir_pipeline="tirx")
    DEV = tvm.cuda(0)
    BH = B * H  # flatten batch*head for TMA-compatible buffers
    inp_tvm = tvm.runtime.tensor(inp.reshape(BH, N, D).numpy(), DEV)
    mod(inp_tvm, ...)
    result = inp_tvm.numpy().reshape(B, H, N, D)
    np.testing.assert_allclose(result, ref, rtol=5e-2, atol=5e-2)
```
