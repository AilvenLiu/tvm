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

---
name: axe-layout
description: >
  Expert knowledge on Axe Layout — the hardware-aware layout abstraction in TIR/TIRX.
  Use this skill when working with TLayout, TileLayout, SwizzleLayout, ComposeLayout, Iter, Axis,
  or layout operations (tile, slice, canonicalize, group, direct_sum, apply, is_tile_inner/outer).
---

# Axe Layout — Reference Index

Do NOT read all references at once. Use the commands below to find what you need.

## Source files

- Python API: `python/tvm/tir/layout.py`
- C++ impl: `src/tir/ir/layout/*.cc`
- C++ header: `include/tvm/tir/layout.h`
- Tests: `tests/python/tirx/test_layout.py`
- Usage: `python/tvm/tirx/op_dispatch/cuda/copy_async.py`

## Lookup commands

```bash
# Core concepts (Iter, Axis, Shard/Replica/Offset, Shape admission, Span)
grep -n "##" .claude/skills/axe-layout/references/formal-definitions.md

# Layout types (TileLayout, SwizzleLayout, ComposeLayout, construction)
grep -n "class TileLayout\|class SwizzleLayout\|class ComposeLayout\|def __init__" python/tvm/tir/layout.py

# Python API for a specific operation (e.g. tile, slice, apply, group, direct_sum)
grep -n "def tile\|def slice\|def apply\|def group\|def direct_sum\|def canonicalize\|def pack\|def unpack" python/tvm/tir/layout.py

# Axis names (thread axes vs memory axes)
grep -n "AxisRegEntry\|Register" src/tir/ir/layout/axis_registry.cc

# Algorithm details (canonicalize, group-by-shape, tile, slice, direct-sum)
grep -n "##" .claude/skills/axe-layout/references/algorithms.md

# Codegen patterns (TMA, Trainium, thread-local loops, CTA collective)
grep -n "##" .claude/skills/axe-layout/references/codegen.md

# Test examples for a specific operation (e.g. tile, slice)
grep -n "def test_.*tile\|def test_.*slice\|def test_.*canon\|def test_.*group\|def test_.*direct_sum" tests/python/tirx/test_layout.py

# C++ implementation of a specific operation
ls src/tir/ir/layout/  # tile_core.cc, tile_tile_ops.cc, tile_slice.cc, tile_canonicalize.cc, tile_direct_sum_ops.cc, swizzle_layout.cc, compose_layout.cc
```
