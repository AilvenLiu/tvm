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
bx, by = Tx.cta_id([M_CLUSTER, 1], parent="kernel")
tx, ty, wg = Tx.thread_id([32, 4, WG_NUMBER], parent="cta")
is_leader_cta = Tx.meta_var(bx == 0)
wg_id = Tx.meta_var(wg)

# Pipeline: producer prefetches while consumers compute
# Consumer: Tx.gemm_async(tmem, ..., dispatch="tcgen05", accum=...)
# Producer: Tx.copy_async(smem, global, dispatch="tma", ...)
```

**Reference:** `test_hgemm.py`, `test_flux.py`

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

**Barrier classes (manual mbarriers, NOT PipelineState):**
```python
@Tx.meta_class
class BarTMA2MMA:       # TMA arrival → MMA wait (PIPE_DEPTH slots)
class BarMMA2TMA:       # MMA release → TMA can reuse slot (PIPE_DEPTH slots)
class BarMMA2Consumer:  # MMA done → consumer reads TMEM (1 slot)
class BarConsumer2MMA:  # Consumer done → MMA next phase (1 slot)
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
Tx.ptx.mbarrier.arrive(mma2tma_bar.mbar.ptr_to([tic]))
```

*Delayed (read both current AND prev tic):* Release `prev_tic` with chunk-0 guard. Used by hedgehog where the consumer reads V_prev from the previous tic's slot for state updates.
```python
if cid_mma > 0:  # Skip chunk 0 — prev_tic holds initial fill, not stale data
    Tx.ptx.mbarrier.arrive(mma2tma_bar.mbar.ptr_to([prev_tic]))
```

**Work item synchronization (`workitem_sync`) for persistent grid (BH > SM_COUNT):**

When `total_bh > SM_COUNT`, each CTA processes multiple work items via `while wid < total_bh: ... wid += SM_COUNT`. TMA races ahead of MMA/consumer at work item boundaries because TMA's inner loop finishes much faster (just DMA commands). When TMA enters the next work item, it skips `mma2tma_bar.wait` for the first `PIPE_DEPTH` chunks (the `if cid_tma >= PIPE_DEPTH:` guard), loading new data into SMEM slots that MMA/consumer from the previous work item may still be reading.

**Fix:** Add a single `workitem_sync` mbarrier that synchronizes MMA and TMA at work item boundaries:

```python
# Barrier setup (after consumer2mma_bar):
# MMA→TMA direction: MMA arrives after finishing a work item,
# TMA waits before starting the next. Uses BarTMA2MMA (is_p2c=True)
# so that init_phase=0 and the first wait blocks until an arrive.
workitem_sync_bar = BarTMA2MMA(buf.data, 4 + 2 * PIPE_DEPTH + 2, 1, True)
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
        workitem_sync_bar.arrive_only(0)
    wid_mma = wid_mma + SM_COUNT
```

MMA's arrive implicitly guarantees consumer is done because it happens after the last `consumer2mma` wait. The `wi_phase` variable toggles independently of operational barrier phases.

**Important:** For kernels that release `prev_tic` instead of `tic` (e.g., hedgehog), the MMA must also release the final chunk's `tic` slot before arriving on `workitem_sync`, to keep `mma2tma` barrier phases balanced across work items. See pitfall #14.

**fftconv special case (PIPE_DEPTH=1, no inner chunk loop):** In addition to `workitem_sync`, add `phase[0] = phase[0] ^ 1` after each work item in both TMA and MMA warps, because the `tma2mma` barrier toggles once per WI but `phase[0]` was never flipped (no `cid % PIPE_DEPTH == PIPE_DEPTH - 1` check without an inner chunk loop).

**Constraint:** N must be a multiple of `2*CHUNK` (ensuring even `nc`) for operational barrier phases to naturally reset between work items. All existing tests satisfy this.

**Reference files (all TK-transcribed, all warp-specialized):**

| Kernel | Phases | Phase parity | Descriptors | PIPE_DEPTH | Unique feature |
|--------|--------|-------------|-------------|------------|----------------|
| `test_linear_attention.py` | 3 | odd (flip) | 2: descI + descI_tb | 2 | Cross-buffer Q@K, KV state accum |
| `test_based.py` | 2 | even | 2: descI + descI_tb | 2 | Dual swizzle (QK=1, V=3), Taylor coeffs |
| `test_mamba2.py` | 4 | even | 2: descI + descI_tb | 2 | SSD with f32 KV state + A cumsum |
| `test_fftconv.py` | 7 | odd (flip) | 1: descI_tb only | 1 | Complex Monarch FFT, most phases |
| `test_hedgehog.py` | 6 | even | 2: descI_nn + descI_nt | 2 | Hybrid sliding+linear attn, prev_tic release (needs final tic balance) |
| `test_flux.py` | — | — | — | 4 | Uses Pattern 1 (2-CTA cluster), NOT this pattern |

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
Any class instantiated inside `@Tx.prim_func` needs `@Tx.meta_class`:
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
    Tx.ptx.mbarrier.arrive(mma2tma_bar.mbar.ptr_to([last_tic]))
    workitem_sync_bar.arrive_only(0)
```

### 15. Debugging SM100a warp-specialized kernel correctness
Systematic checklist when output is wrong:
1. **Values correct but wrong positions** → swizzle alignment (pitfall #9). Use identity inputs to confirm.
2. **First chunk/tile correct, rest wrong** → barrier phase tracking (pitfall #10). Count total transitions per iteration.
3. **All values wrong/garbage** → likely a race condition, missing sync, or wrong SMEM indexing.
4. **SASS comparison red herring**: Control word differences between working and broken kernels are often a symptom of different code structure, not the root cause. Don't waste time patching SASS bytes.
5. **Intercept compilation** for debugging: `tvm_ffi.register_global_func("tvm_callback_cuda_compile", override=True)` to save intermediates, add `-Xptxas -v`, or inspect generated CUDA.

---

## Infrastructure Utilities

### Pipeline State (`tvm.tirx.pipeline`)
```python
from tvm.tirx.pipeline import PipelineState, MBarrier, TMABar, TCGen05Bar

ps = PipelineState("name", pipe_depth=N)
ps.init(is_producer=True/False)
ps.move_to_next_stage()

mbar = MBarrier(pool, depth=N)      # Regular arrive
tma_bar = TMABar(pool, depth=N)     # TMA arrive (expect_tx)
tc_bar = TCGen05Bar(pool, depth=N)  # tcgen05 commit
```

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
```python
from tvm.tirx.bench.utils import ProtonContext, bench

ms = bench(func, warmup=10, repeat=30, proton_name="name")
with ProtonContext("label"):
    run_tests()
```

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

# Barrier classes: see test_linear_attention.py for full implementations
# BarTMA2MMA, BarMMA2TMA, BarMMA2Consumer, BarConsumer2MMA
# All inherit from a Barriers base class with @Tx.meta_class

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
                # ---- Shared memory allocation (manual, NOT PoolAllocator) ----
                buf = Tx.alloc_buffer([SMEM_SIZE], "uint8", scope="shared.dyn")
                tmem_addr = Tx.decl_scalar("uint32", buf.data, scope="shared.dyn", elem_offset=0)
                # Swizzled buffers first via Tx.decl_buffer(..., buf.data, elem_offset=off)
                # Non-swizzled buffers (f32 state, scalars) last
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
