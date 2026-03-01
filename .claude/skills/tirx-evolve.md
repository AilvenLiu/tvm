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

# TIRX Kernel Evolution / Optimization Guide

## Overview

This skill covers techniques for optimizing already-working TIRX kernels — improving throughput without changing correctness. Applies to both Pattern 1 (cluster GEMM) and Pattern 2 (warp-specialized multi-phase) kernels on SM100a.

**Workflow:** Profile → identify bottleneck → choose technique → implement → verify correctness → benchmark.

---

## Bottleneck Identification

### Pattern 2: Consumer-bound vs MMA-bound

Pattern 2 kernels alternate between MMA and consumer work. The critical path is whichever takes longer:

- **Consumer-bound (common):** MMA finishes quickly, waits on `consumer2mma_bar`. Consumer has heavy scalar loops, nested accumulations, or wide data preparation. Symptoms: TFLOPS well below theoretical peak, MMA utilization low in profiler.
- **MMA-bound:** Consumer finishes quickly, waits on `mma2consumer_bar`. Happens when K-reduction is large or many K-iterations per phase.
- **SMEM-bandwidth-bound:** Consumer work is dominated by SMEM reads/writes (not compute). Eliminating compute has minimal effect. Symptoms: moderate TFLOPS, profiler shows high SMEM utilization.

**Quick check:** Comment out consumer compute (replace with dummy values), re-benchmark. If speedup is large → consumer-bound. If minimal speedup → MMA-bound or SMEM-bound.

### Pattern 1: Epilogue-bound vs MMA-bound vs Memory-bound

Pattern 1 (cluster GEMM) bottlenecks:

- **MMA-bound (ideal):** Large K dimension, MMA is the critical path. Already near peak. Check by comparing ms to `2*M*N*K / peak_flops`.
- **Epilogue-bound:** Consumer epilogue (GELU, gate, normalization, TMA store) is on the critical path. Symptoms: K is small relative to M/N, consumer WGs stall on `mma2ld.wait`.
- **Memory-bound:** Problem is bandwidth-limited (small K, large M×N). TMA loads dominate. Check by computing arithmetic intensity: `2*M*N*K / (M*K + N*K + M*N) / bytes_per_elem`. If < ~100, likely memory-bound.

### Profiling

```bash
# Select idle GPU
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t',' -k2 -n | head -1 | cut -d',' -f1 | tr -d ' ')

# Run benchmark
cd /home/tlopexh/tvm/build
/home/tlopexh/miniconda3/envs/te/bin/python ../tests/python/tirx/kernels/cuda/sm100a/test_<kernel>.py
```

### Decision Tree

```
Is it Pattern 1 (GEMM) or Pattern 2 (multi-phase)?
├── Pattern 1:
│   ├── CTAs << SM_COUNT (< 80% utilization)? → K-splitting (#12)
│   ├── SMEM_PIPE_DEPTH < max allowed by SMEM budget? → Increase pipeline depth (#13) — FIRST CHECK
│   ├── TS-mode with cast warps? → Cast barrier reorder (#14): LDSM before TMEM wait
│   ├── Far from peak TFLOPS? → Check epilogue cost, try pipelined TMA store, larger EPI_TILE
│   ├── Small K? → Memory-bound, try larger BLK_K or split-K
│   └── Near peak? → Done
└── Pattern 2:
    ├── Consumer has >10K FMAs/thread? → Technique #1 (offload to MMA)
    ├── Consumer has nested loops = matmul? → Technique #1 + #2 (transA)
    ├── Consumer SMEM reads dominate? → Technique #8 (swizzle) + #10 (reg preload)
    ├── Consumer SMEM writes dominate? → Technique #9 (negated buffers), #11 (wider thread shuffle)
    ├── Consumer has sign flip / redundant ops? → Technique #9 (negated buffers)
    ├── Multiple phases with uneven load? → Technique #5 (redistribute)
    ├── f32 intermediates in SMEM? → Technique #7 (register buffers)
    ├── SMEM budget tight? → Technique #3 (buffer reuse/aliasing)
    ├── Need A^T @ B without explicit transpose? → Technique #2 (transA)
    └── Already balanced / MMA-bound? → Likely at limit
```

---

## Technique 1: Offload Scalar Loops to MMA Phases

**When to apply:**
- Consumer has deeply nested loops (O(N³) or O(N⁴)) doing scalar FMAs
- The computation is mathematically equivalent to a matrix multiplication `C += A @ B`
- SMEM budget allows staging buffers for A and B in f16

**Impact:** Based kernel: 1.45 → 10.7 TFLOPS (7.4×). Hedgehog kernel: 1.6 → 41 TFLOPS (25×).

### Method

1. **Identify the matmul** in the scalar loops. Common forms:
   - `o[f] += sum_{d,e} Q[d]*Q[e] * state[e*D+d, f]` → outer product matrix @ state
   - `state[d*E+e, f] += K[d]*K[e]*V[f]` → (needs transpose) outer product^T @ V

2. **Add SMEM staging buffers** for the MMA operands:
   - New swizzled f16 buffers after existing SMEM allocations
   - Budget: each [64, 256] f16 buffer = 32KB, [256, 64] f16 = 32KB
   - Verify total SMEM ≤ 232KB

3. **Consumer prepares operands** in SMEM (all 128 threads cooperate):
   ```python
   # 16384 elements / 128 threads = 128 per thread
   with Tx.thread():
       tid_flat = Tx.meta_var(warp_id * 32 + lane_id)
       for i in Tx.serial(128):
           flat = Tx.meta_var(tid_flat * 128 + i)
           row = Tx.meta_var(flat // COLS)
           col = Tx.meta_var(flat % COLS)
           smem[row, col] = Tx.cast(compute_value(...), "float16")
   ```

4. **Add MMA phases** and corresponding consumer read-back phases.

5. **Update phase parity:** Count total phases per chunk. Odd → add `phase_c2m ^= 1`.

### Case Study: Based a2 Operations

**Before (2 phases):** Two 4-nested loops (32K scalar FMAs/thread):
```
Phase 1: Q @ K^T → TMEM         (MMA)
Phase 2: att @ V → TMEM          (MMA) + a0/a1/a2 contrib + all state updates (consumer)
```

**After (7 phases):** Scalar loops replaced by 5 new MMA phases:
```
Phase 1: Q @ K^T → TMEM                              (unchanged)
Phase 2: att @ V → TMEM + a0/a1 state + prepare QQT/a2s_f16 (consumer restructured)
Phase 3: QQT[64,256] @ a2s_f16[256,64] → TMEM        (NEW — a2 contribution)
Phase 4: KK_chunk0^T[64,64] @ V[64,64] → TMEM        (NEW — a2 state update)
Phase 5: KK_chunk1^T[64,64] @ V[64,64] → TMEM        (NEW)
Phase 6: KK_chunk2^T[64,64] @ V[64,64] → TMEM        (NEW)
Phase 7: KK_chunk3^T[64,64] @ V[64,64] → TMEM        (NEW)
```

**New SMEM buffers:**
| Buffer | Shape | dtype | Size | Purpose |
|--------|-------|-------|------|---------|
| `qqt_kk_smem` | [64, 256] | f16 | 32KB | Shared: QQT for Phase 3, then KK for Phases 4-7 |
| `a2s_f16_smem` | [256, 64] | f16 | 32KB | Phase 3 only: f16 cast of f32 a2s state |

**Per-thread work comparison:**
| Operation | Before | After |
|-----------|--------|-------|
| a2 contribution | 16,384 FMAs (nested loop) | ~256 muls (form QQT) + MMA |
| a2 state update | 16,384 FMAs (nested loop) | ~256 muls (form KK) + 4 × 64 adds (TMEM→a2s) |

### Case Study: Hedgehog feat_q @ kv_state and feat_k^T @ V

**Before (6 phases):** Two massive scalar loops in consumer (32K FMAs/thread total):
```
Phase 5 consumer: feat_q[128] @ kv_state[128,128] → o[128]  (16K FMAs)
Phase 6 consumer: feat_k^T[128] @ V[64,128] → kv_state update (16K FMAs)
```

**After (12 phases):** Scalar loops replaced by 6 new MMA phases:
```
Phase 5:  Q @ qmap → TMEM           (unchanged, consumer now featurizes Q, stages kv_state→f16)
Phase 6:  feat_q_pos @ kv[0:64,0:64] + feat_q_neg @ kv[64:128,0:64] → o_lo  (NEW)
Phase 7:  feat_q_pos @ kv[0:64,64:128] + feat_q_neg @ kv[64:128,64:128] → o_hi (NEW)
Phase 8:  K @ kmap → TMEM           (was Phase 6, consumer featurizes K)
Phase 9:  feat_k_chunk0^T @ V → TMEM (NEW — kv_state update, transA=True)
Phase 10: feat_k_chunk1^T @ V → TMEM (NEW)
Phase 11: feat_k_chunk2^T @ V → TMEM (NEW)
Phase 12: feat_k_chunk3^T @ V → TMEM (NEW)
```

**Key techniques combined:**
- Technique #1: Offload both contribution and state update loops to MMA
- Technique #2: transA=True for feat_k^T @ V (avoids explicit transpose)
- Technique #3: Alias D_out_lo/D_out_hi into kv_state_f16 buffer (saves 16KB SMEM)
- Technique #5: Redistribute work — Phase 5 consumer prepares operands, Phase 7 consumer does normalize+store

**SMEM staging buffer:**
| Buffer | Shape | dtype | Size | Purpose |
|--------|-------|-------|------|---------|
| `kv_state_f16` | [128, 64] | f16 | 16KB | f32→f16 cast of kv_state columns; aliased with D_out |

**Phase parity:** 12 phases = even → `phase_c2m` does NOT flip between chunks.

### Splitting Large Matmuls into MMA-Sized Chunks

MMA M is 64 (or 128 with cta_group=2). When the result matrix exceeds 64 rows, split into chunks:

```
# KK^T[256,64] @ V[64,64] → a2s_update[256,64]
# Split into 4 chunks: KK_i^T[64,64] @ V[64,64] → update[64,64], i=0..3
# Each chunk = 1 MMA phase with 4 K-iterations
```

In MMA warp, use `Tx.unroll` for chunk loop (compile-time):
```python
for chunk_i in Tx.unroll(N_CHUNKS):
    for ki in Tx.unroll(CHUNK // MMA_K):
        # encode descriptors, issue MMA
    mma2consumer_bar.arrive(0)
    consumer2mma_bar.wait(0, phase_c2m ^ ...)
```

In consumer, use `Tx.serial` for chunk loop (runtime):
```python
for chunk_i in Tx.serial(N_CHUNKS):
    mma2consumer_bar.wait(0, phase_m2c)
    phase_m2c = phase_m2c ^ 1
    # read TMEM, accumulate into state
    consumer2mma_bar.arrive(0)
    phase_c2m_c = phase_c2m_c ^ 1
```

---

## Technique 2: transA for In-Place Transpose

**When to apply:** Need `A^T @ B` but A is stored as [K, M] in a wider SMEM buffer and explicit transpose would waste SMEM or consumer cycles.

### Method

Set `transA=True` in the instruction descriptor:
```python
descI_transA: Tx.uint32
Tx.ptx.tcgen05.encode_instr_descriptor(
    Tx.address_of(descI_transA), "float32", "float16", "float16",
    MMA_M * CTA_GROUP, MMA_N, MMA_K, True, True, CTA_GROUP)
```

**Descriptor pointer iteration with transA:** K iterates along rows (not columns):
```python
# A is [K=64, M=64] sub-block of wide_smem[64, 256]
# SDO must be for the FULL buffer width, not the chunk
MMA_SDO_WIDE = 8 * 256 * F16_BYTES // F128_BYTES  # = 256

for ki in Tx.unroll(CHUNK // MMA_K):  # 4 iterations over K dimension
    Tx.ptx.tcgen05.encode_matrix_descriptor(
        Tx.address_of(descA),
        wide_smem.ptr_to([ki * MMA_K, chunk_i * CHUNK]),  # advance ROWS by MMA_K
        MMA_LDO_WIDE, MMA_SDO_WIDE, SWIZZLE_V)
```

**transA/transB semantics summary:**
| Flag | A memory layout | B memory layout | MMA computes |
|------|----------------|-----------------|--------------|
| transA=False, transB=False | [M, K] | [N, K] | A @ B^T conceptually |
| transA=False, transB=True | [M, K] | [K, N] | A @ B |
| transA=True, transB=True | [K, M] | [K, N] | A^T @ B |
| transA=True, transB=False | [K, M] | [N, K] | A^T @ B^T conceptually |

---

## Technique 3: SMEM Buffer Reuse

**When to apply:** Multiple MMA phases need staging buffers but not simultaneously.

### Method

Alias the same SMEM region for different phases:
```python
# Phase 3 uses qqt_kk_smem as QQT[64,256]
# Phase 3 consumer overwrites it with KK[64,256]
# Phases 4-7 use qqt_kk_smem as KK[64,256]
# Same buffer, same layout, different semantic contents per phase
```

**Rules:**
- Writer must complete and signal barrier before reader starts
- All 128 consumer threads must finish writing before `consumer2mma_bar.arrive`
- `Tx.ptx.tcgen05.fence.before_thread_sync()` before the barrier arrive

---

## Technique 4: use_a_tmem (Eliminate SMEM Staging)

**When to apply:** An MMA operand A is already computed and available in TMEM from a previous phase. Instead of reading TMEM → registers → SMEM → MMA, read directly from TMEM.

**Status:** Used in FA4 (`test_flash_attention4.py`). **Blocked** for consumer-computed operands that must cross warpgroup boundaries (see constraints below).

### Pattern (from FA4 P@V phase)

Consumer writes P (attention weights) to TMEM as f16, then MMA reads TMEM as A-operand:

```python
# MMA: use_a_tmem=True — A comes from TMEM, B from SMEM
Tx.ptx.tcgen05.mma(
    "float32", "float16", "float16",
    Tx.cuda.get_tmem_addr(0, 0, tmem_col_output),     # output TMEM region
    Tx.cuda.get_tmem_addr(0, 0, tmem_col_a_operand),   # A-operand from TMEM
    descV.add_16B_offset(v_offset), descI_pv,
    use_a_tmem=True, cta_group=CTA_GROUP,
    enable_input_d=should_accumulate)
```

**Requirements:**
- TMEM N_COLS must be large enough to hold both MMA output and A-operand data
- A-operand stored as f16 in TMEM (consumer writes via `Tx.copy_async(tmem_as_f16, ...)`)
- Output region and A-operand region must not overlap

**Trade-off:** Saves SMEM and consumer→SMEM write cycles, but requires larger TMEM allocation (more N_COLS).

### Critical Constraints (Learned from Linear Attention attempt)

1. **Cross-WG tcgen05.st is unreliable:** WG0 (consumer) writing to TMEM via tcgen05.st, WG1 (MMA) reading via use_a_tmem → intermittent 0-5% mismatch. TMEM visibility between warpgroups not guaranteed even with barriers + tcgen05.wait::st. **Only same-WG TMEM writes work reliably.**

2. **tcgen05.cp + swizzled SMEM column offsets break:** Advancing descriptor start_address by column offsets (`ptr_to([0, ki*8])`) breaks SWIZZLE_128B because XOR swizzle doesn't commute with address addition. Row 0 correct, other rows garbled (~3% mismatch).

3. **tcgen05.cp + SWIZZLE_NONE:** Descriptor LDO is NOT ignored for layout_type=0. Must set `LDO = row_pitch_bytes / 16`. Even with correct LDO, data integrity issues persist (~15% mismatch).

4. **FA4 works because:** A-operand (attention P) is written to TMEM by WG1's own threads (same-WG). MMA output → softmax → write back to TMEM as f16 → use_a_tmem. No cross-WG transfer.

**When use_a_tmem IS feasible:**
- A-operand is a transformation of data already in the MMA warpgroup's TMEM
- A-operand is loaded via TMA directly to TMEM

**When use_a_tmem is NOT feasible:**
- A-operand computed by a different warpgroup (consumer WG0) and must cross WG boundary
- A-operand requires tcgen05.cp from SMEM with column-strided descriptors on swizzled buffers

---

## Technique 5: Move Work Between Phases

**When to apply:** One consumer phase is much heavier than others, causing the MMA warp to stall.

### Method

Redistribute consumer work across phases to balance load:

**Before (Based original):**
- Phase 2 consumer: read TMEM + a0/a1/a2 contributions + D_out store + ALL state updates (a0 + a1 + a2) → **very heavy**

**After:**
- Phase 2 consumer: read TMEM + a0/a1 contributions + a0/a1 state updates + prepare QQT/a2s_f16 → **moderate**
- Phase 3 consumer: read TMEM + add a2 contrib + D_out store + prepare KK → **moderate**
- Phases 4-7 consumer: read TMEM + 64 f32 adds each → **light**

**Principle:** MMA phases are "free" compute if the consumer has enough work to overlap. Move expensive consumer work earlier (before MMA needs the signal) or spread across multiple phases.

---

## Technique 6: Wider SMEM Buffers with Non-Standard Swizzle

**When to apply:** MMA operand width exceeds 64 (e.g., [64, 256] for outer-product-style computations).

### Layout

```python
# [64, 256] f16 with 128B swizzle — row width = 256 * 2 = 512B
wide_layout = Tx.ComposeLayout(
    Tx.SwizzleLayout(SWIZZLE_V, SWIZZLE_V, SWIZZLE_V, swizzle_inner=True),
    Tx.TileLayout(S[(ROWS, COLS) : (COLS, 1)]),
)
# SDO = 8 * COLS * dtype_bytes / 16
# For [64, 256] f16: SDO = 8 * 256 * 2 / 16 = 256
MMA_SDO_WIDE = 8 * COLS * F16_BYTES // F128_BYTES
```

The swizzle mode (SWIZZLE_V=3 = 128B) is still valid because swizzle operates on the inner 128B of each row. The wider row just means more 128B groups per row, each independently swizzled.

---

## Technique 7: Replace SMEM f32 Intermediates with Register Buffers

**When to apply:** Consumer saves MMA results (f32) to SMEM, then reads them back in a later phase for elementwise computation. Only active threads (lane_id < 16) participate, so each thread owns exactly one row.

**Impact:** FFTConv kernel: 1.6 → 2.27 TFLOPS (1.4× speedup). Modest because the real bottleneck is SMEM reads for elementwise constants (tw/kf/twinv), not the f32 roundtrip.

### Method

Replace SMEM f32 buffers with per-thread local register buffers:
```python
# Before: SMEM f32 buffer shared across threads
w_real_s = Tx.decl_buffer((N1, N1), "float32", buf.data, ...)  # 16KB SMEM

# After: per-thread register buffer (only active threads use it)
w_real_reg = Tx.alloc_buffer((N1,), "float32", scope="local")  # 64 regs/thread
```

**Benefits:**
- Eliminates SMEM write + read latency for the f32 intermediate
- Removes `warpgroup_sync` barriers between save and multiply steps (no SMEM coherence needed)
- Frees SMEM budget (e.g., 2 × 16KB = 32KB freed in FFTConv)

**Requirements:**
- Each active thread must own its entire row — no cross-thread data sharing for the intermediate
- Register pressure must be acceptable (64 f32 regs = 256B per buffer per thread)

### Case Study: FFTConv w_real/w_imag

**Before:** Phase 1 saves TMEM → w_real_s (SMEM f32), Phase 2 saves TMEM → w_imag_s (SMEM f32), then `warpgroup_sync`, then tw multiply reads both from SMEM.

**After:** Phase 1 saves TMEM → w_real_reg (registers), Phase 2 saves TMEM → w_imag_reg (registers) and immediately does tw multiply in the same `Tx.thread()` block — no sync needed.

**Limitation:** This technique gives modest gains when the elementwise multiply itself is the bottleneck (many SMEM reads for constants like tw/kf/twinv). For larger gains, the elementwise work itself needs to be reduced or offloaded.

---

## Technique 8: Swizzle Consumer-Only SMEM Buffers to Reduce Bank Conflicts

**When to apply:** Consumer threads read elementwise constant buffers (twiddle factors, filter coefficients, etc.) that are stored in SMEM without swizzle. Multiple threads reading different rows of the same column cause bank conflicts.

**Impact:** FFTConv kernel: 2.27 → 3.42 TFLOPS (+51% speedup). Pure layout change, zero algorithmic modification.

### Method

Add `SwizzleLayout` to consumer-only buffers, same as MMA operand buffers:
```python
# Before: non-swizzled (bank conflicts when 16 threads read same column)
tw_real_s = Tx.decl_buffer((N1, N1), "float16", buf.data, elem_offset=off)

# After: swizzled (different rows map to different banks)
tw_real_s = Tx.decl_buffer((N1, N1), "float16", buf.data,
                           layout=swizzled_layout, elem_offset=off)
```

**Requirements:**
- Buffer byte offset must be 1024B-aligned (same as MMA operand buffers)
- Each buffer is [64,64] f16 = 8KB = 8×1024B, so consecutive buffers stay aligned
- All reads/writes go through the layout (element-by-element `buf[row, col]`), so swizzle is transparent

**Why it works:** Without swizzle, 16 threads reading column `j` of different rows all hit the same SMEM bank (rows differ by 128B = 4 banks, causing 4-way conflicts). With 128B swizzle, row addresses are XOR-remapped so different rows land in different banks.

---

## SMEM Release Timing with Many Phases

When extending a kernel with additional MMA phases, the SMEM slot release (`mma2tma_bar.arrive(tic)`) must be deferred to **after the last phase that reads from the TMA-loaded buffers** (Q, K, V).

**Example:** Based kernel reads V_smem in Phase 2 and Phases 4-7. Release must happen after Phase 7, not Phase 2.

```python
# MMA warp: after ALL phases complete
Tx.ptx.mbarrier.arrive(mma2tma_bar.mbar.ptr_to([tic]))  # release after Phase 7
phase_c2m = phase_c2m ^ 1  # 7 = odd → flip
```

---

## Checklist for Adding MMA Phases

1. **SMEM budget:** Calculate new buffer sizes, verify total ≤ 232KB
2. **Layouts:** Define swizzled layouts for new buffers, compute SDO
3. **Instruction descriptors:** Add new `descI_*` for each unique (transA, transB) combination
4. **MMA warp:** Add phases inside `Tx.thread()[Tx.ptx.elect_sync()]`, use `Tx.unroll` for K-iterations
5. **Consumer:** Add corresponding wait/read/signal blocks
6. **Phase parity:** Count total phases. Odd → `phase_c2m ^= 1` in MMA, consumer's `phase_m2c` and `phase_c2m_c` naturally flip odd times
7. **SMEM release:** Move `mma2tma_bar.arrive(tic)` to after the last SMEM-reading phase
8. **Barrier wait pattern:** MMA alternates `consumer2mma_bar.wait(0, phase_c2m)` and `consumer2mma_bar.wait(0, phase_c2m ^ 1)` for successive phases
9. **Test:** Run all parametrized test cases, then benchmark

---

## Lessons Learned: When Optimization Doesn't Help

### Redistribution (Technique #5)

Not all consumer work redistribution yields speedup:

- **Serial bottleneck shift:** Moving work from Phase N to Phase N+1 doesn't help if they execute serially — it just moves the bottleneck. (Mamba2: 22.0 → 22.0 TFLOPS)
- **Small loop overhead:** Adding a new MMA phase + barrier round-trip for <1024 FMAs cancels the savings. (Based a1 offload: 10.7 → 10.8 TFLOPS)
- **Already MMA-bound:** When MMA phases have many K-iters, consumer work is already hidden. (Linear Attention: 23.3 → 23.3 TFLOPS)

**When redistribution DOES help:** Consumer work >10K FMAs/thread AND clearly consumer-bound (MMA stalls on consumer2mma_bar). Examples: Based 32K FMAs → 7.4×, Hedgehog 32K FMAs → 25×.

### Micro-optimizations on SMEM-bound kernels

Eliminating SMEM reads (register preloading #10) and redundant writes (negated buffers #9) gives 10-15% gains when SMEM writes remain the dominant cost. (FFTConv: 3.42 → 3.84 TFLOPS, +12%)

However, wider thread participation (#11) via warp shuffle can break through the SMEM-write wall: doubling the writing threads halved per-thread write count, yielding +33% (FFTConv: 3.84 → 5.10 TFLOPS). This is the most impactful technique for SMEM-write-bound phases.

**Heuristic:** If consumer work per phase is <1000 FMAs or <128 SMEM ops/thread, micro-optimization is unlikely to help. For SMEM-write-bound kernels, try wider thread participation (#11) first — it addresses the root cause (too few writers) rather than symptoms.

### Aliasing hazard

Do NOT alias a precomputed constant buffer with a per-work-item buffer (e.g., D_out). The constant is loaded once at init and must survive all work items. Aliasing causes corruption from the second work item onward. (FFTConv neg_Finv_real aliasing attempt: 7.48% mismatch on large batches)

---

## Technique 5 Addendum: Staging Buffer Data Races

When redistributing work, beware of data races between concurrent MMA and consumer:

**Problem:** Consumer writes to `staging_a` during Phase N, but MMA reads `staging_a` during Phase N (they run concurrently). This corrupts MMA results.

**Rule:** Consumer can only write to a staging buffer AFTER the last MMA phase that reads it. The buffer becomes "free" when MMA signals `mma2consumer_bar.arrive` for the phase that consumed it.

**Example (Linear Attention):**
- `staging_a_lo/hi` is read by Phase 3 and Phase 4 MMA (Q_dec @ kvs)
- Consumer can only overwrite `staging_a` with K_dec staging in Phase 4 consumer (after Phase 4 MMA is done)
- Attempting to write K_dec staging in Phase 3 consumer causes a data race

---

## Technique 9: Negated Constant Buffers to Eliminate Sign Flips

**When to apply:** Consumer must negate an entire SMEM buffer between MMA phases because the MMA uses `-X` in one phase and `+X` in the next, but the buffer stores only one sign. The sign flip loop costs ~128 SMEM reads + 128 SMEM writes per active thread.

**Impact:** FFTConv kernel: eliminates 2 sign-flip loops (Phases 3 and 5), saving ~256 SMEM reads + 256 SMEM writes per work item.

### Method

Pre-compute the negated version of a constant matrix at init time and store it in a separate SMEM buffer:

```python
# Init: load both positive and negated copies
F_real_s[row, col] = f_real_g[row, col]
neg_F_real_s[row, col] = Tx.cast(0.0 - Tx.cast(f_real_g[row, col], "float32"), "float16")
```

Then use `neg_F_real_s` as the MMA B-operand where the negated version is needed:

```python
# Phase 4 second half: work_B = -WI, neg_FR = -FR
# (-WI) @ (-FR) = WI @ FR — correct imaginary part, no sign flip needed
Tx.ptx.tcgen05.encode_matrix_descriptor(
    Tx.address_of(descB), neg_F_real_s.ptr_to([ki * MMA_K, 0]), ...)
```

**Trade-off:** +8KB SMEM per negated buffer. Only worthwhile when:
- The constant is read-only (loaded once, reused across work items)
- The sign flip was on the critical path (consumer-bound phases)
- SMEM budget has headroom

**Aliasing caution:** Do NOT alias a negated buffer with a buffer that gets overwritten during the kernel loop (e.g., D_out). The negated buffer is loaded once at init and must survive across all work items. Aliasing with D_out causes corruption from the second work item onward (~7% mismatch on large batch sizes, first work item passes).

### Case Study: FFTConv

**Before:** Phase 3 consumer flips `work_B = -work_B` (128 reads + 128 writes) to convert `-W'.imag` to `+W'.imag` for Phase 4. Same at Phase 5 for Phase 6.

**After:** Added `neg_F_real_s` and `neg_Finv_real_s`. Phase 4 MMA uses `work_B @ neg_F_real_s` instead of `work_B @ F_real_s`. Phase 6 uses `work_B @ neg_Finv_real_s`. Sign flip loops deleted entirely.

---

## Technique 10: Preload Constants to Registers During Light Phases

**When to apply:** Consumer multiply phases read 2+ constant buffers from SMEM (128+ SMEM reads per active thread). A preceding phase is light (just TMEM→register copy, ~200 cycles). The SMEM reads can be shifted to the light phase as a preload, converting SMEM reads to register reads in the heavy phase.

**Impact:** FFTConv kernel: reduces 3 multiply phases from ~128 SMEM reads each to 0 SMEM reads (use registers instead). Moderate speedup since SMEM writes still dominate.

### Method

Declare reusable f16 register buffers (one per constant dimension):

```python
const_real_reg = Tx.alloc_buffer((N1,), "float16", scope="local")
const_imag_reg = Tx.alloc_buffer((N1,), "float16", scope="local")
```

In the light phase consumer, load constants into registers:

```python
# Phase 1 consumer: light (just TMEM → w_real_reg)
# Also preload tw constants for Phase 2
with Tx.thread():
    out_row = Tx.meta_var(warp_id * 16 + lane_id)
    if lane_id < 16:
        for j in Tx.serial(N1):
            w_real_reg[j] = reg[j]
        # Preload — SMEM reads happen here, overlapping with MMA
        for j in Tx.serial(N1):
            const_real_reg[j] = tw_real_s[out_row, j]
            const_imag_reg[j] = tw_imag_s[out_row, j]
```

In the heavy phase consumer, read from registers:

```python
# Phase 2 consumer: multiply using registers (no SMEM reads)
tr = Tx.meta_var(Tx.cast(const_real_reg[j], "float32"))
ti = Tx.meta_var(Tx.cast(const_imag_reg[j], "float32"))
```

**Register budget:** f16 registers are half the size of f32. 64 f16 values = 32 f32-equivalent registers per buffer. Budget: w_real(64) + w_imag(64) + const_real(32) + const_imag(32) = 192 regs/thread (limit 256).

**Pattern:** Preload in Phase N (light) → use in Phase N+1 (heavy):
- Phase 1 → Phase 2: tw_real/tw_imag
- Phase 3 → Phase 4: kf_real/kf_imag
- Phase 5 → Phase 6: twinv_real/twinv_imag

**When NOT effective:**
- When the light phase is actually heavy (preloading adds work to an already-loaded phase)
- When register pressure is already high (>200 regs/thread)
- When SMEM reads are not the bottleneck (e.g., SMEM writes dominate)

---

## Technique 11: Wider Thread Participation via Warp Shuffle

**When to apply:** Consumer-bound multiply phases where only lanes 0-15 (per warp) participate due to TMEM physical layout constraints (CTA_GROUP=1, MMA_M=64 → only lanes 0-15 have valid TMEM data). Lanes 16-31 are idle during SMEM writes.

**Impact:** FFTConv kernel: 3.84 → 5.10 TFLOPS (+33%). Halves SMEM writes per active thread from 128 to 64 by doubling the number of writing threads.

### Background: TMEM Lane Constraint

With CTA_GROUP=1, MMA_M=64, warp `w`, lane `l`:
- TMEM physical row = `w*32 + l`
- Valid MMA output at logical row `w*16 + l` only for `l < 16`
- Lanes 16-31 read garbage from TMEM → only lanes 0-15 can compute and write

### Method

After lanes 0-15 compute all N columns, use `__shfl_xor_sync` with mask 16 to send the second half (columns 32-63) to lanes 16-31:

```python
# Row mapping: both lane halves write the same row
out_row = Tx.meta_var(warp_id * 16 + (lane_id % 16))

if lane_id < 16:
    # Compute all 64 columns into w_real_reg, w_imag_reg
    # Write columns 0-31 immediately (32 writes to work_A + 32 to work_B)
    for j in Tx.serial(32):
        work_A[out_row, j] = Tx.cast(w_real_reg[j], "float16")
        work_B[out_row, j] = Tx.cast(0.0 - w_imag_reg[j], "float16")

# Shuffle columns 32-63 to partner lanes (ALL 32 lanes must participate)
for j in Tx.serial(32):
    shuf_real_buf[j] = Tx.tvm_warp_shuffle_xor(0xFFFFFFFF, w_real_reg[32 + j], 16, 32, 32)
    shuf_imag_buf[j] = Tx.tvm_warp_shuffle_xor(0xFFFFFFFF, w_imag_reg[32 + j], 16, 32, 32)

# Lanes 16-31 write columns 32-63
if lane_id >= 16:
    for j in Tx.serial(32):
        work_A[out_row, 32 + j] = Tx.cast(shuf_real_buf[j], "float16")
        work_B[out_row, 32 + j] = Tx.cast(0.0 - shuf_imag_buf[j], "float16")
```

### Critical Implementation Details

1. **Shuffle results MUST use real buffer writes, NOT `Tx.meta_var`:**
   `Tx.meta_var` is expression substitution — if the shuffle result is only used inside `if lane_id >= 16`, the compiler inlines the shuffle into the conditional, meaning only lanes 16-31 execute the warp-collective `__shfl_xor_sync`. This causes a **deadlock** (all 32 lanes must participate). Fix: write shuffle results to an `Tx.alloc_buffer` so the shuffle executes unconditionally.

   ```python
   # WRONG — deadlocks:
   shuf = Tx.meta_var(Tx.tvm_warp_shuffle_xor(...))
   if lane_id >= 16:
       buf[i] = shuf  # shuffle inlined here, only lanes 16-31 execute

   # CORRECT — shuffle executes for all lanes:
   shuf_buf[j] = Tx.tvm_warp_shuffle_xor(...)  # all lanes execute
   if lane_id >= 16:
       buf[i] = shuf_buf[j]  # only reads the result
   ```

2. **In-place complex multiply read-after-write hazard:**
   When storing results back into `w_real_reg`/`w_imag_reg` (needed for the shuffle), the computation `new_real = wr*tr - wi*ti; new_imag = wr*ti + wi*tr` has a dependency: both need the original `wr` and `wi`. With `Tx.meta_var`, the variables get inlined — so writing `w_real_reg[j] = ...` first corrupts the `wr` value used in the `w_imag_reg[j] = ...` line.

   Fix: use a temp buffer to break the dependency:
   ```python
   # Store new_imag in temp first (reads old w_real and w_imag)
   shuf_real_buf[j % 32] = w_real_reg[j] * ti + w_imag_reg[j] * tr
   # Overwrite w_real (reads old w_real and w_imag — both still original)
   w_real_reg[j] = w_real_reg[j] * tr - w_imag_reg[j] * ti
   # Copy temp to w_imag
   w_imag_reg[j] = shuf_real_buf[j % 32]
   ```

3. **Shuffle XOR semantics:** Lane L and lane L^16 exchange data bidirectionally. Lane 0 sends valid data to lane 16; lane 16 sends garbage (uninitialized regs) to lane 0. The garbage on lane 0 is harmless — it's written to `shuf_real_buf` but never read (only `if lane_id >= 16` reads it).

4. **out_row formula:** Change from `warp_id * 16 + lane_id` to `warp_id * 16 + (lane_id % 16)` so lanes 0-15 and 16-31 map to the same rows. In light phases (where only `lane_id < 16` is active), `lane_id % 16 == lane_id`, so no change needed.

### Applicability

Works for any consumer phase where:
- Only lanes 0-15 have valid data (TMEM constraint)
- The dominant cost is SMEM writes (not compute)
- Column count is even (can split 50/50)
- Register pressure allows a small shuffle buffer (32 × f32 = 128B)

Also applies to D_out write phases (Phase 7 in FFTConv) — same pattern with a single buffer instead of two.

---

## Technique 12: K-Splitting for SM Utilization

**When to apply:** The number of M-tiles (CTAs) is significantly less than SM_COUNT (148 on B200). For example, M=4096 / BLOCK_M=64 = 64 CTAs → only 43% SM utilization.

**Impact:** TF32 HC Prenorm GEMM: 51.38 → 76.37 TFLOPS (49% speedup, 116% of DeepGEMM). SM utilization 43% → 86%.

### Method

Split the K dimension across multiple CTAs. Each CTA handles `K / kNumSplits` iterations. Grid size = `num_m_blocks * kNumSplits`.

**Constants:**
```python
kNumSplits = 2  # or higher, targeting ~SM_COUNT total CTAs
NUM_K_ITERS_PER_SPLIT = NUM_K_ITERS // kNumSplits
assert NUM_K_ITERS % kNumSplits == 0
```

**Grid indexing:**
```python
m_block_idx = bx // kNumSplits
k_split_idx = bx % kNumSplits
k_offset = k_split_idx * NUM_K_ITERS_PER_SPLIT * BLOCK_K
m_offset = M * k_split_idx  # for sqr_sum flat indexing
```

**Output tensors (3D for D, flat for sqr_sum):**
```python
D: Tx.Buffer((kNumSplits, M, N), d_type)           # 3D output
sqr_sum_out: Tx.Buffer((kNumSplits * M,), "float32")  # flat, indexed by m_offset + m_idx
```

**TMA tensor map:** Must be 3D (rank=3). See transcription pitfall #23 for the exact `cuTensorMapEncodeTiled` argument layout.

**TMA store:** Use `s2g(3, ...)` with `k_split_idx` as the third coordinate.

**K-loop:** All three loops (TMA, MMA, cast warp) iterate `NUM_K_ITERS_PER_SPLIT` instead of `NUM_K_ITERS`. TMA loads start from `k_offset + s * BLOCK_K`.

**Host-side reduction after kernel:**
```python
D_out = D_splits.sum(dim=0)           # (M, N)
sqr_sum = sqr_sum_splits.view(kNumSplits, M).sum(dim=0)  # (M,)
```

### When NOT to apply

- K is small relative to M → few K blocks, splitting provides no benefit
- Already at SM_COUNT CTAs → no utilization headroom
- The reduce overhead is significant relative to kernel time (very small M×N)

### Choosing kNumSplits

Target `num_m_blocks * kNumSplits ≈ SM_COUNT` (148 on B200). Overshooting wastes compute on reduction; undershooting leaves SMs idle.

| num_m_blocks | kNumSplits | Total CTAs | SM utilization |
|---|---|---|---|
| 64 | 1 | 64 | 43% |
| 64 | 2 | 128 | 86% |
| 64 | 3 | 192 | 130% (some CTAs queue) |
| 148+ | 1 | 148+ | 100% (no split needed) |

---

## Technique 13: Pipeline Depth (kNumStages) Tuning

**When to apply:** The kernel uses fewer pipeline stages than SMEM budget allows. More stages improve TMA/MMA overlap.

**Impact:**
- TF32 HC Prenorm GEMM: 39.91 → ~45 TFLOPS (kNumStages 4→12, ~13% improvement)
- BHR-HDR-BHD: 2.5 → 3.1 TFLOPS / 38.5 → 87.4 TFLOPS (depth 4→6, 60% → 95-115% of DeepGEMM)

### Method

DeepGEMM's heuristic: start from max stages, decrease until SMEM fits:
```python
smem_per_stage = BLOCK_M * BLOCK_K * sizeof(A_dtype) + BLOCK_N * BLOCK_K * sizeof(B_dtype)
smem_fixed = SMEM_CD_SIZE + barriers + tmem_ptr + epilogue_smem
max_depth = (228 * 1024 - smem_fixed) // smem_per_stage
SMEM_PIPE_DEPTH = min(max_depth, NUM_K_ITERS)
```

**Examples:**
- TF32 HC prenorm (BLOCK_M=64, BLOCK_N=32, BLOCK_K=64): 16KB/stage → 12 stages fits
- BHR-HDR-BHD (BLOCK_M=128, BLOCK_N=128, BLOCK_K=64): 32KB/stage → 6 stages max

### Trade-offs

- More stages → more SMEM → less per-CTA SMEM available (rarely a problem on SM100a)
- Diminishing returns beyond the point where TMA latency is fully hidden
- Must ensure barrier count stays manageable (`full_barriers[kNumStages]` etc.)
- **Always compute max depth and use it** — under-staging is a common transcription mistake (see transcription pitfall #25)

---

## Technique 14: Cast Barrier Reorder (TS Mode)

**When to apply:** TF32 TS-mode kernels where cast warps load from SMEM (LDSM) then store to TMEM.

**Impact:** ~10% performance improvement by overlapping SMEM reads with TMEM busy.

### Method

Move `empty_cast_barriers.wait` (wait for TMEM empty) from before LDSM to between LDSM and TMEM store:

**Before:**
```
full_barriers.wait → empty_cast_barriers.wait → LDSM(SMEM) → TMEM store
```

**After:**
```
full_barriers.wait → LDSM(SMEM) → empty_cast_barriers.wait → TMEM store
```

LDSM reads SMEM data that has no dependency on TMEM state. Moving the wait between LDSM and TMEM store allows SMEM reads to overlap with MMA still consuming the previous TMEM data.

**Implementation:** Move the barrier wait inside the inline PTX function, passing the barrier pointer and phase as additional arguments. See transcription pitfall #22 for details.

---

## Pattern 1 (Cluster GEMM) Optimization

Pattern 1 kernels (Flux, DeepGEMM) have different optimization axes than Pattern 2. The main GEMM K-loop is typically well-pipelined; bottlenecks are in the epilogue and tile scheduling.

### CTA_GROUP=2 (2-CTA Cluster) Implementation Guide

**When to use:** N >= 256 and the B tile benefits from multicast TMA (both CTAs share B). Authoritative reference: `test_hgemm.py`.

**When NOT to use:** Small N (128 or less), batch-reduction kernels where each CTA processes all K stages independently. The cluster sync overhead exceeds multicast benefit. (bmk_bnk_mn: 2-CTA was 2.5× slower than 1-CTA for N=128.)

**Kernel Structure Differences from CTA_GROUP=1:**

1. **Scope IDs:**
   ```python
   cbx, cby = Tx.cta_id([M_CLUSTER, N_CLUSTER], parent="cluster")  # within-cluster CTA index
   bx = Tx.cta_id([SM_NUMBER], parent="kernel")
   ```
   Final sync: `Tx.cuda.cluster_sync()` (not `cta_sync()`)

2. **TMA Barrier (cross-CTA visibility):**
   ```python
   # Must use map_shared_rank to get CTA0's barrier address from CTA1
   ptr = Tx.reinterpret("handle", Tx.ptx.map_shared_rank(tma2mma_bar.mbar.ptr_to([0]), 0))
   tma_finished = Tx.decl_buffer([SMEM_PIPE_DEPTH], "uint64", data=ptr, scope="shared")
   # TMA copy uses tma_finished (CTA0's barrier) as mbar
   tma_copy = {"dispatch": "tma", "mbar": tma_finished.ptr_to([ks]), "cta_group": CTA_GROUP}
   # Only leader CTA signals expected bytes (both CTAs' TMA writes arrive at same barrier)
   if cbx == 0:
       tma2mma_bar.arrive(ks, CTA_GROUP * expected_bytes)
   ```

3. **MMA (leader CTA only):**
   ```python
   if cbx == 0:  # only CTA0 issues MMA — instruction handles both CTAs via cta_group=2
       Tx.ptx.tcgen05.mma(..., cta_group=CTA_GROUP, ...)
   ```

4. **Barriers:**
   - `tcgen05.commit(..., cta_group=2, cta_mask=3)` — cta_mask=3 for 2-CTA
   - `ld2mma_bar.init(CTA_GROUP * 128)` — consumer WG from BOTH CTAs must arrive

5. **Tile Scheduler:**
   ```python
   num_clusters = SM_NUMBER // 2  # 74 clusters of 2 SMs
   tile_scheduler.init(bx // 2)   # cluster index
   m_start = (m_idx * CTA_GROUP + cbx) * BLK_M  # each CTA handles different M rows
   ```

6. **MMA Descriptor (symmetric 256×256):**
   ```python
   Tx.ptx.tcgen05.encode_instr_descriptor(..., MMA_M=256, MMA_N=256, MMA_K=16, ..., cta_group=2)
   N_COLS = TMEM_PIPE_DEPTH * MMA_M  # 512
   col_offset = tmem_idx * MMA_M     # 0 or 256
   ```

7. **TMEM alloc/dealloc:** `cta_group=2`, `n_cols=N_COLS`

**CRITICAL: Swap epilogue is INCOMPATIBLE with cta_group=2.** The second MMA descriptor is multicast across CTAs. With swap `mma(descB, descA)`, the second desc (A = M-dimension) would be identical for both CTAs → CTA1 produces all zeros. Always use standard `mma(descA, descB)` with cta_group=2. Standard epilogue iterates N chunks with D_smem = `(BLK_M, EPI_TILE)`.

### Epilogue Pipelining

**Problem:** Consumer epilogue does N sequential TMA stores, each with `commit_group()` + `wait_group(0)` (fully blocking). For large N (e.g., BLK_N=128 with EPI_TILE=64 → 4 stores per consumer), the serial store chain becomes the bottleneck.

**Fix:** Pipeline the stores — use `commit_group()` without immediate `wait_group(0)`, then `wait_group(N-1)` at the end:
```python
for i in Tx.unroll(NUM_STORES):
    # ... write to D_smem, fence ...
    Tx.copy_async(D[...], D_smem[...], dispatch="tma")
    Tx.ptx.cp_async.bulk.commit_group()
Tx.ptx.cp_async.bulk.wait_group(0)  # wait for all at once
```

**Caution:** Need to ensure D_smem is not overwritten before the previous TMA store completes. May require double-buffered D_smem or warpgroup_sync between tiles.

### Larger EPI_TILE

Increasing `EPI_TILE` (e.g., 64 → 128) reduces the number of TMA store iterations. Trade-off: larger SMEM for epilogue tiles.

### Tile Shape Tuning

BLK_M, BLK_N, BLK_K affect the balance between compute (MMA) and memory (TMA loads). Key relationships:
- Larger BLK_K → more K-iters per pipeline stage, better MMA utilization
- Larger BLK_M × BLK_N → more output elements per tile, better amortization of epilogue
- Larger tiles → fewer tiles total → less scheduling overhead, but fewer SMs utilized for small problems

### Register Pressure (setmaxnreg)

Pattern 1 uses `Tx.ptx.setmaxnreg(False, 56)` for producer and `Tx.ptx.setmaxnreg(True, 224)` for consumer. If consumer epilogue is simple, consider reducing consumer regs and giving more to producer to enable deeper software pipelining.

---

## Technique Applicability Summary

| Technique | Pattern 1 | Pattern 2 | When to use |
|-----------|-----------|-----------|-------------|
| #1 Offload to MMA | N/A | High impact | Consumer has >10K FMAs in scalar loops |
| #2 transA | N/A | High impact | Need A^T @ B, A stored as [K,M] |
| #3 Buffer reuse | Both | Both | SMEM budget tight, non-overlapping lifetimes |
| #4 use_a_tmem | N/A | Blocked | Same-WG TMEM only (FA4 pattern) |
| #5 Redistribute | N/A | Conditional | Only when consumer >> MMA, see lessons |
| #6 Wide SMEM buffers | N/A | Medium | MMA operand width > 64 |
| #7 Register buffers | N/A | Medium | f32 SMEM intermediates owned by single thread |
| #8 Swizzle consumer bufs | N/A | High impact | Consumer reads constant SMEM with bank conflicts |
| #9 Negated constants | N/A | Medium | Eliminate sign-flip loops on critical path |
| #10 Register preload | N/A | Medium | Light phase precedes heavy SMEM-reading phase |
| #11 Wider thread shuffle | N/A | High impact | Consumer SMEM writes dominate, lanes 16-31 idle |
| #12 K-splitting | High impact | N/A | CTAs << SM_COUNT, split K for more CTAs |
| #13 Pipeline depth | High impact | N/A | kNumStages < SMEM budget allows |
| #14 Cast barrier reorder | High impact | N/A | TS-mode: LDSM before TMEM wait |
| Epilogue pipeline | High impact | N/A | Serial TMA stores in consumer epilogue |
| Tile shape tuning | High impact | N/A | Far from peak, need to balance compute/memory |

---

## Reference: Kernel Evolution History

| Kernel | Pattern | Original | Optimized | Technique | Speedup |
|--------|---------|----------|-----------|-----------|---------|
| Based | 2 | 2 phases, 1.45 TFLOPS | 8 phases, 10.8 TFLOPS | Offload a2 scalar loops to MMA (#1), transA (#2), buffer reuse (#3), work redistribution (#5) | 7.4× |
| Hedgehog | 2 | 6 phases, 1.6 TFLOPS | 12 phases, 41 TFLOPS | Offload feat_q@kv_state + feat_k^T@V to MMA (#1), transA (#2), D_out/kv_state_f16 aliasing (#3), work redistribution (#5) | 25× |
| FFTConv | 2 | 7 phases, 1.6 TFLOPS | 7 phases, 5.10 TFLOPS | Registers for intermediates (#7), swizzle elementwise buffers (#8), negated constant buffers (#9), register preloading (#10), wider thread shuffle (#11) | 3.2× |
| Mamba2 | 2 | 4 phases, 22.0 TFLOPS | 4 phases, 22.0 TFLOPS | Redistribution attempted (#5) — no gain, already balanced | 1× |
| Linear Attn | 2 | 9 phases, 23.3 TFLOPS | 9 phases, 23.3 TFLOPS | Redistribution (#5) — no gain, MMA-bound. use_a_tmem (#4) — blocked by cross-WG TMEM + swizzle/cp constraints | 1× |
| TF32 HC Prenorm | 1 (TS) | 39.91 TFLOPS (60.6% DG) | 76.37 TFLOPS (116% DG) | Pipeline depth 4→12 (#13), cast barrier reorder (#14), K-splitting kNumSplits=2 (#12) | 1.91× |
| bmk_bnk_mn | 1 | 178 TFLOPS (40% DG) | 409 TFLOPS (92% DG) | Split-K via GroupMajor3D + TMA reduce-add (#12). 32 tiles / 148 SM → 224 tiles / 148 SM | 2.3× |
| BHR-HDR-BHD | 1 | 2.5 TFLOPS (60% DG) | 3.1 TFLOPS (115% DG) | Pipeline depth 4→6 (#13) | 1.24× (small shape); 2.3× at B=1024,R=512 |
| Flux | 1 | — | — | Transcribed from TK. Epilogue pipeline + tile tuning are untested | — |
