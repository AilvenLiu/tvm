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

# TIR Project

A compiler for high-performance AI kernels, built on top of Apache TVM. Adds **TIRX** — an extended tensor IR with hardware-aware layout abstractions, hierarchical execution scopes, async primitives, and an operator scheduling framework. Targets NVIDIA GPUs (Ampere/Hopper/Blackwell) and AWS Trainium.

## Project Structure

The codebase is an Apache TVM fork. Custom additions are concentrated in these areas:

### Core IR Extensions (`python/tvm/tir/`, `src/tir/ir/`, `include/tvm/tir/`)
- **Layout system** — `layout.py`, `src/tir/ir/layout/`, `layout.h` — Axe Layout abstraction (TileLayout, SwizzleLayout, ComposeLayout, Iter, Axis)
- **Execution scopes** — `exec_scope.py`, `exec_scope.h`, `exec_scope.cc` — Hierarchical scope model (world > kernel > cluster > cta > warpgroup > warp > thread)
- **TIRX statements** — `tirx_stmt.h/.cc`, `tirx_op.h` — OpCall node for operator invocation
- **Device op codegen** — `python/tvm/tir/device_op_codegen/cuda/` — PTX/CUDA code generation for barriers, MMA, wgmma, TMA, nvshmem

### TIRX DSL & Compiler (`python/tvm/tirx/`)
- **Operator framework** — `operator/op.py` — Base operator classes (Unary, Binary, Reduction, etc.)
- **Op schedule** — `op_schedule/cuda/`, `op_schedule/trn/` — Target-specific kernel scheduling (GEMM, copy_async, reduction, etc.)
- **Transforms** — `transform/` — Buffer allocation, event tensor legalization
- **Pipeline & async** — `pipeline.py`, `tile_scheduler.py` — Async pipeline management, tile scheduling
- **Megakernel** — `megakernel/kernels/` (23 fused LLM kernels), `megakernel/model/` (Llama, Qwen), `megakernel/utils/`

### Script Frontend (`python/tvm/script/`, `src/script/`)
- **TIRX DSL API** — `python/tvm/script/tirx.py`, `ir_builder/tir/tirx.py` — 40+ builtin ops (Tx.gemm_async, Tx.copy_async, etc.)
- **Printer/Parser** — `src/script/printer/tir/`, `src/script/ir_builder/tir/` — TIRX-aware printing and parsing

### Lowering Passes (`src/tir/transform/`)
- `lower_tirx.cc` — Entry point
- `lower_tirx_scope_ids.cc` — Scope ID resolution
- `lower_tirx_scope_slices.cc` — Execution scope slicing
- `lower_tirx_schedule_ops.cc` — Op schedule lowering
- `lower_tirx_dedup_tensormap.cc` — TensorMap deduplication
- `lower_tirx_cleanup.cc` — Post-lowering optimization
- `lower_tirx_opaque.cc` — Opaque block handling

### Tests (`tests/python/tirx/`)
- Core: `test_layout.py`, `test_parser_printer.py`, `test_verifier.py`, `test_exec_scope.py`
- Codegen: `codegen/test_codegen_cuda.py`, `test_codegen_hopper.py`, `test_codegen_blackwell.py`, `test_codegen_nki.py`
- Transforms: `transform/test_transform_lower_tirx.py`, `test_stmt_functor.py`, `test_expr_functor.py`
- Kernels: `kernels/cuda/sm80/`, `sm90a/`, `sm100a/`, `megakernel/`; `kernels/trn/`

### Other
- `kernels/sm100/` — Standalone SM100 GEMM kernels (fp16, fp8, nvfp4)
- `tirx_doc/` — Tutorials (e.g., `gemm_tutorial.ipynb`)
- `python/tvm/tir/pipeline.py` — Compilation pipelines: `default_tir_pipeline()`, `tirx_pipeline()`, `trn_pipeline()`

## Axe Layout

Axe Layout is the hardware-aware layout abstraction in TIR/TIRX that maps logical tensor coordinates to a multi-axis physical space via named axes. When working with `TLayout`, `TileLayout`, `SwizzleLayout`, `ComposeLayout`, `Iter`, `Axis`, or layout operations (tile, slice, canonicalize, group, direct_sum, apply, is_tile_inner/outer), use the **axe-layout** skill for detailed reference.

Key files:
- Python API: `python/tvm/tir/layout.py`
- C++ implementation: `src/tir/ir/layout/`
- Tests: `tests/python/tirx/test_layout.py`

## Building

Build TVM (from project root):
```bash
mkdir -p build && cd build && cmake .. && make -j$(nproc)
```

## Testing

**Build first (mandatory)**: Before running any tests, ensure TVM is built and `build/` is up to date (you do not need to recreate the directory each time):
```bash
cmake --build build -j$(nproc)
```
If `build/` does not exist yet, run the commands in `## Building` first.

**GPU selection**: Before running GPU tests, check if this is a multi-GPU machine (`nvidia-smi --query-gpu=index --format=csv,noheader | wc -l`). If so, select the least busy GPU to avoid conflicts:
```bash
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t',' -k2 -n | head -1 | cut -d',' -f1 | tr -d ' ')
```

Run TIRX tests:
```bash
pytest tests/python/tirx/ -n 16
```

**Kernel performance**: When modifying anything that affects code generation (kernels, op schedules, lowering passes, codegen, device ops), verify performance by running square GEMM benchmarks at M=N=K in {1024, 2048, 4096, 8192, 16384} for the three GEMM variants (fp16, fp8, nvfp4). The kernel scripts have built-in benchmarking — just run them and record the output.

## Code Style

All formatting and linting is managed via `pre-commit` (see `.pre-commit-config.yaml`):

- **Python**: `ruff check --fix` (lint) + `ruff format` (format)
- **C++**: `clang-format`
- **Cython**: `cython-lint`

Run on staged files:
```bash
pre-commit run
```

Run on all files:
```bash
pre-commit run --all-files
```

## Commits and Branches

Follow [Conventional Commits](https://www.conventionalcommits.org/):
```
<type>(<scope>): <short imperative description>

Types: feat, fix, refactor, perf, test, docs, chore, ci, build
Scopes: layout, kernel, op, op-schedule, lower-tirx, tvmscript, megakernel, infra
```

Default branch: **`tirx`** (not `main`). Always use `--base tirx` when creating PRs.

**NEVER force push to `tirx`** — no exceptions, no `--force`, no `--force-with-lease`. Always verify the current branch before any push.

Branch naming: `<type>/<kebab-case-description>`, e.g. `feat/direct-sum`, `fix/slice-swizzle`.

## Common Gotchas

- **C++ ↔ Python sync**: Changing IR nodes (layout, stmt, op) requires updating both the C++ side (`include/` + `src/`) and Python bindings (`python/tvm/tir/`). Don't forget FFI registration.
- **Printer ↔ Parser sync**: Modifications to IR printing (`src/script/printer/tir/`) must have corresponding parser changes (`python/tvm/script/parser/tir/`) and vice versa. Run `test_parser_printer.py` to verify round-trip.
- **New op checklist**: Adding a TIRX operator requires registration in `operator/op.py`, a schedule in `op_schedule/cuda/` (and/or `trn/`), a DSL entry in `ir_builder/tir/tirx.py`, and tests.
- Keep documentation in sync with code changes: when modifying code that is referenced in this document or in `.claude/skills/`, update the corresponding documentation immediately.
- **`Tx.meta_var` is not an accumulator**: `Tx.meta_var` creates a TIR expression, not a mutable variable. Writing `acc = Tx.meta_var(0); for i: acc = acc + x` does NOT accumulate — each reassignment creates a new expression and the compiler may optimize the loop away. For accumulation, write directly into a buffer: `buf[d] = buf[d] + x` (using `Tx.alloc_buffer` or SMEM buffers). Read-only scalars (`val = Tx.meta_var(expr)`) are fine.
- **Flaky tests**: If tests fail intermittently, first check `nvidia-smi` to see if the GPU is occupied by other workloads. Switch to an idle GPU before re-running.
