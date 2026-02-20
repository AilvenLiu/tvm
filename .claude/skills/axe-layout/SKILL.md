---
name: axe-layout
description: >
  Expert knowledge on Axe Layout — the hardware-aware layout abstraction in TIR/TIRX that maps
  logical tensor coordinates to a multi-axis physical space via named axes. Use this skill whenever
  working with TLayout, TileLayout, SwizzleLayout, ComposeLayout, Iter, Axis, or layout operations
  (tile, slice, canonicalize, group, direct_sum, apply, is_tile_inner/outer). Also use when writing
  or debugging TIRX kernels that involve data layout, memory tiling, thread binding, distributed
  sharding, TMA copy, or AI accelerator (Trainium) memory mappings.
---

# Axe Layout Skill

## 1. Core Concept

Axe Layout extends the classical shape-stride model by allowing strides to be **named** and bound
to different **axes** representing hardware resources (memory, threads, devices). An Axe layout maps
a logical index to a **set** of coordinates on named axes, decomposed into three components:

```
L(x) = { D(x) + r + O | r in R }
```

- **D (Shard)**: Ordered list of iters `(extent, stride, axis)` that partition logical indices
  across axes. Generalizes shape-stride to multiple axes.
- **R (Replica)**: Set of replication iters producing offsets independent of the logical index.
  Adds broadcasting/replication. Written in `[brackets]`.
- **O (Offset)**: Fixed coordinate offset per axis. Places data at specific base positions.

### Iter

The fundamental building block. A triple `(extent, stride, axis)`:
- `extent`: number of values (>0)
- `stride`: memory/axis step per increment
- `axis`: named hardware resource

### Shape Admission

A shape `S = (S_0, ..., S_{r-1})` is **admitted** by layout `L` iff `prod(S_i) = prod(extent_i for i in D)`.
The layout maps multi-dimensional coordinates by first flattening to a linear index (row-major),
then unflattening through the iter list.

### Axis-wise Span

```
span_a(L) = 1 + sum(|s_I| * (e_I - 1) for I in D+R where axis_I == a)
```

The span is the max-min range + 1 on each axis. Used in tiling to avoid overlap.

## 2. Layout Types

### TileLayout

Primary layout type. Contains:
- `shard`: `Array[Iter]` — the D part
- `replica`: `Array[Iter]` — the R part
- `offset`: `Map[Axis, PrimExpr]` — the O part

### SwizzleLayout

Memory access pattern optimization (XOR-based swizzling for shared memory bank conflicts).
Parameters: `per_element`, `swizzle_len`, `atom_len`, `swizzle_inner`.

### ComposeLayout

Composition of `SwizzleLayout` (applied first) and `TileLayout`. Used for shared memory with
swizzle + tiling.

## 3. Axes

Axes are singleton objects obtained via `Axis.get(name)` or `Axis.<name>`. Two categories:

**Thread axes** (have scope/subscope, represent parallel execution units):
`pid`, `bx`, `by`, `bz`, `cbx`, `cby`, `cbz`, `tx`, `warpid`, `laneid`, `wgid`, `tid_in_wg`, `wid_in_wg`

**Memory axes** (represent address dimensions):
`m` (default memory), `P` (partition), `F` (free), `Bank`, `TCol`, `TLane`

The `@` operator binds a stride to an axis: `4 @ Axis.laneid` means stride=4 on the laneid axis.

## 4. Python API

### Construction

```python
from tvm.tir.layout import TileLayout, SwizzleLayout, ComposeLayout, Iter, Axis

# Auto-infer row-major strides (strides = [8, 1])
layout = TileLayout([8, 8])

# Explicit extents and strides (all on default 'm' axis)
layout = TileLayout(shard=([8, 8], [8, 1]))

# With named axes: 4@laneid means stride=4 on laneid axis
layout = TileLayout(shard=([8, 4, 2], [4 @ Axis.laneid, 1 @ Axis.laneid, 1]))

# With replica
layout = TileLayout(
    shard=([8, 8], [8, 1]),
    replica=([2], [4 @ Axis.warpid])
)

# With offset
layout = TileLayout(shard=([8, 8], [8, 1]), offset=5 @ Axis.warpid)

# From Iter objects directly
layout = TileLayout(
    shard=[Iter(8, 4, Axis.laneid), Iter(4, 1, Axis.laneid), Iter(2, 1, "m")],
    replica=[Iter(2, 4, Axis.warpid)]
)

# Trainium layout
layout = TileLayout.trainium("PF", (128, 128))  # P=partition, F=free dims
```

### Core Operations

```python
# Size: total number of logical elements
layout.size()            # total size
layout.size("laneid")    # size on specific axis

# Span: max-min range + 1 per axis
layout.span()            # total span
layout.span("m")         # span on memory axis

# Apply: map logical coordinates to physical axis values
result = layout.apply(i)                        # 1D linear index
result = layout.apply(i, j, shape=[8, 8])       # multi-dim with shape
# Returns Dict[str, PrimExpr]: {"m": ..., "laneid": ..., ...}

# Canonicalize: simplify to canonical form (fuse adjacent same-axis iters, remove unit extents)
canon = layout.canonicalize()

# Verify well-formedness
layout.verify_well_formed()

# Check properties
layout.is_trivial()    # identity mapping (pure memory, row-major)
layout.is_trainium()   # Trainium hardware layout
layout.storage()       # filter out thread-axis components, keep only memory part
```

### Tile (Kronecker Product)

Composes an inner layout with an outer layout. The outer layout's strides are scaled by the
inner layout's span to avoid overlap.

```python
# T = inner.tile(outer, outer_shape, inner_shape)
# f_T(x || y) = f_outer(x) * span(f_inner) + f_inner(y)
tiled = inner_layout.tile(outer_layout, outer_shape=[M, N], inner_shape=[m, n])

# Convenience: tile_to target shape
tiled = inner_layout.tile_to(to_shape=[64, 64], current_shape=[8, 8])
```

### Tile Recognition (Inverse of Tile)

```python
# Check if inner_layout is the inner part of tiled_layout, recover outer
outer = inner_layout.is_tile_inner(tiled_layout, tiled_shape=[64, 64], inner_shape=[8, 8])

# Check if outer_layout is the outer part of tiled_layout, recover inner
inner = outer_layout.is_tile_outer(tiled_layout, tiled_shape=[64, 64], outer_shape=[8, 8])
```

### Direct Sum (Unscaled Composition)

Like tile but **without** span scaling. Useful when instructions accept strided atoms.

```python
# result = right.direct_sum(left, left_shape, right_shape)
# f_{A+B}(x || y) = f_A(x) + f_B(y)   (no span scaling!)
result = B.direct_sum(A, left_shape=[2, 2], right_shape=[2, 2])
```

### Direct Sum Recognition

```python
left = right.is_direct_sum_right(sum_layout, interleaved_shape, right_shape)
right = left.is_direct_sum_left(sum_layout, interleaved_shape, left_shape)
```

### Slice

Extract a sub-layout for a rectangular region.

```python
# slice(shape, region) where region is list of (begin, end) tuples
sliced = layout.slice(shape=[64], region=[(5, 8)])
sliced = layout.slice(shape=[16, 24], region=[(0, 8), (8, 24)])
# Returns None if slicing is not possible for the given region
```

### Group

Split/fuse iters to match a target shape (precondition for tile/direct_sum).

```python
grouped_layout, separators = tile_layout.group(shape=[4, 8])
```

### Pack / Unpack

```python
# Unpack: one element -> num contiguous elements (multiply strides by num, add inner iter)
unpacked = layout.unpack(num=2)

# Pack: num contiguous elements -> one element (divide strides by num)
packed = layout.pack(num=2)
```

## 5. Key Algorithms (see references/ for details)

Read `references/algorithms.md` for detailed algorithm descriptions:

- **Canonicalize**: Rewrite rules D0 (remove unit extent), D1 (merge adjacent same-axis iters),
  C0-C2 (normalize replica). Produces unique canonical form under gap condition.
- **Group-By-Shape**: GCD-driven algorithm to split/fuse iters into blocks matching target shape.
- **Tile**: Group both layouts, compute span scaling vector, interleave scaled-outer and inner iters.
- **Tile-Of Check**: Recover C from A = C x B by greedy subsequence matching + span descaling.
- **Slice**: Greedy peeling from fastest digit, with no-wrap and symmetric one-wrap sufficient forms.
- **Direct Sum**: Interleave iter blocks without span scaling.

## 6. Code Generation Patterns (see references/ for details)

Read `references/codegen.md` for code generation patterns:

- **TMA async copy**: Determine shared-memory swizzle atom, derive CuTensorMap, loop over atoms.
- **Trainium TensorEngine GEMM**: Group layouts by M/N/K, intersect K/M/N dimensions, find
  largest instruction shape.
- **Thread-local loop transformation** (CuTe style): Partition work per thread via layout algebra.
- **CTA collective semantics** (Triton style): Organize registers as CTA-collective tensor.

## 7. File Locations

**Python API**: `python/tvm/tir/layout.py`
- `TLayout` (base), `TileLayout`, `SwizzleLayout`, `ComposeLayout`, `Iter`, `Axis`

**C++ Header**: `include/tvm/tir/layout.h`
- `TLayoutNode`, `TileLayoutNode`, `SwizzleLayoutNode`, `ComposeLayoutNode`
- `IterNode`, `AxisNode`, `AxisRegEntry`

**C++ Implementation**: `src/tir/ir/layout/`
- `layout.cc` — base FFI bindings
- `tile_core.cc` — TileLayout construction, apply, size, span, verify
- `tile_tile_ops.cc` — tile, is_tile_inner, is_tile_outer
- `tile_direct_sum_ops.cc` — direct_sum, is_direct_sum_left/right
- `tile_canonicalize.cc` — canonicalization algorithm
- `tile_slice.cc` — slice algorithm
- `swizzle_layout.cc` — SwizzleLayout implementation
- `compose_layout.cc` — ComposeLayout (SwizzleLayout + TileLayout)
- `axis_registry.cc` — axis registration
- `utils.h` — helper utilities

**Tests**: `tests/python/tirx/test_layout.py` (comprehensive, ~1600 lines)

**Usage in compiler**: `python/tvm/tirx/op_schedule/cuda/copy_async.py`
- TMA layout derivation, swizzle atom computation, box dimension extraction

## 8. Common Patterns

### NVIDIA Tensor Core Fragment

```python
# 8x16 tile mapped to 2 warps x 32 lanes x 2 registers
frag = TileLayout(
    shard=([8, 2, 4, 2],
           [4 @ Axis.laneid, 1 @ Axis.warpid, 1 @ Axis.laneid, 1]),  # 1 = 1@m (register)
    replica=([2], [4 @ Axis.warpid]),
    offset=5 @ Axis.warpid
)
```

### Distributed Sharding (4 GPUs, 2x2 mesh)

```python
# Fully sharded 64x128 across 4 GPUs
dist = TileLayout(shard=([2, 32, 2, 64], [1 @ gpuid, 128, 2 @ gpuid, 1]))

# Shard rows, replicate columns
dist = TileLayout(
    shard=([2, 32, 128], [1 @ gpuid, 128, 1]),
    replica=([2], [2 @ gpuid])
)
```

### AI Accelerator (Trainium) SBUF Layout

```python
# 128 partitions, 256x512 2D tiling
sbuf = TileLayout(shard=([2, 128, 512], [(512, "F"), (1, "P"), (1, "F")]))
# Or use the convenience constructor:
sbuf = TileLayout.trainium("PF", (128, 512))
```

### Shared Memory with Swizzle

```python
swizzle = SwizzleLayout(per_element=4, swizzle_len=3, atom_len=4, swizzle_inner=True)
tile = TileLayout(shard=([rows, cols], [cols, 1]))
smem_layout = ComposeLayout(swizzle, tile)
```
