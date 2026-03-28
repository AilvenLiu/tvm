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

"""Implementation of copy (sync) operator dispatchs for CUDA targets.

Registered ops: copy (4 variants).
See the @register_dispatch blocks below for detailed documentation with
before/after IR examples.

Also provides impl functions (copy_tmem_local_impl, copy_smem_tmem_impl) used
by copy_async.py for the async variants of tmem/smem copy.
"""

from collections.abc import Iterable

import tvm
from tvm.arith import Analyzer
from tvm.runtime import DataType
from tvm.script import tirx as Tx
from tvm.tirx import Buffer, PrimFunc
from tvm.tirx.layout import S, TCol, TileLayout, TLane, tid_in_wg
from tvm.tirx.op_dispatch.dispatcher import fail, predicate, register_dispatch
from tvm.tirx.op_dispatch.registry import DispatchContext
from tvm.tirx.stmt import ScopeOpCall

from .common import (
    CopyInstType,
    copy_vec_load_impl,
    get_st_extent,
    get_vec_len,
    smem_desc_add_16B_offset,
    validate_copy_op,
)
from .exec_scope_utils import exec_scope_ok
from .tma_utils import SwizzleMode, get_swizzle_mode_from_layout


def copy_default_impl(
    op_call: ScopeOpCall,
    sctx: DispatchContext,
) -> PrimFunc | None:
    """Schedule copy operation
    The implementation serves as a fallback for copy operations that uses a single thread
    to move data element by element.
    """
    op_call = ScopeOpCall.downcast(op_call)
    dst_buffer_region, src_buffer_region = op_call.dst, op_call.src

    src: Buffer = src_buffer_region.buffer
    dst: Buffer = dst_buffer_region.buffer

    # Extract regions and validate dimensions
    src_st, src_extent = get_st_extent(src_buffer_region)
    dst_st, dst_extent = get_st_extent(dst_buffer_region)

    def copy(dst, src):
        dst_indices = [i for i in range(len(dst.shape)) if dst_extent[i] != 1]
        src_indices = [i for i in range(len(src.shape)) if src_extent[i] != 1]
        assert len(dst_indices) == len(src_indices)
        copy_extents = [dst_extent[i] for i in dst_indices]

        def get_dst_coord(lvs):
            if isinstance(lvs, tvm.tirx.Var):
                lvs = [lvs]
            coord = [dst_st[i] for i in range(len(dst.shape))]
            for i, lv in enumerate(lvs):
                coord[dst_indices[i]] += lv
            return coord

        def get_src_coord(lvs):
            if isinstance(lvs, tvm.tirx.Var):
                lvs = [lvs]
            coord = [src_st[i] for i in range(len(src.shape))]
            for i, lv in enumerate(lvs):
                coord[src_indices[i]] += lv
            return coord

        with Tx.grid(*copy_extents) as lvs:
            Tx.buffer_store(dst, src[tuple(get_src_coord(lvs))], get_dst_coord(lvs))

    if sctx.exec_scope.name == "cta":
        tx = sctx.launch_params["threadIdx.x"].dom.extent
        assert "threadIdx.y" not in sctx.launch_params and "threadIdx.z" not in sctx.launch_params

        # fmt: off
        @Tx.prim_func(tirx=True, check_well_formed=False)
        def impl():
            for tid_x in Tx.thread_binding(tx, "threadIdx.x"):
                with Tx.thread()[tid_x == 0]:
                    copy(dst, src)

            if dst.scope().startswith("shared"):
                Tx.tvm_storage_sync("shared")
        # fmt: on
    elif sctx.exec_scope.name == "thread":
        # fmt: off
        @Tx.prim_func(tirx=True, check_well_formed=False)
        def impl():
            copy(dst, src)
        # fmt: on
    else:
        fail(f"unsupported exec_scope {sctx.exec_scope.name}")

    return impl


# ---------------------------------------------------------------------------
# Rich dispatcher variants with predicates and failure reasons
# ---------------------------------------------------------------------------


DEFAULT_ALLOWED_PAIRS: tuple[tuple[str, str], ...] = (
    ("global", "shared*"),
    ("shared*", "global"),
    ("global", "local"),
    ("local", "global"),
    ("shared*", "local"),
    ("local", "shared*"),
)


def _match_scope(scope: str, pattern: str) -> bool:
    """Glob-lite: 'shared*' => prefix match; otherwise exact."""
    if pattern.endswith("*"):
        return scope.startswith(pattern[:-1])
    return scope == pattern


def _scope_allowed(
    op_call: ScopeOpCall,
    sctx: DispatchContext,
    allowed_pairs: Iterable[tuple[str, str]] = DEFAULT_ALLOWED_PAIRS,
):
    op_call = ScopeOpCall.downcast(op_call)
    dst_buffer_region, src_buffer_region = op_call.dst, op_call.src
    src_scope = src_buffer_region.buffer.scope()
    dst_scope = dst_buffer_region.buffer.scope()

    ok = any(
        _match_scope(src_scope, src_pat) and _match_scope(dst_scope, dst_pat)
        for src_pat, dst_pat in allowed_pairs
    )
    if not ok:
        allowed_str = ", ".join(f"{a}->{b}" for a, b in allowed_pairs)
        return False, (
            f"unsupported memory scopes src={src_scope} dst={dst_scope}; allowed: {allowed_str}"
        )
    return True, None


def _is_valid_copy(op_call: ScopeOpCall, sctx: DispatchContext):
    return (validate_copy_op(op_call, sctx), "validate_copy_op failed")


def _vec_len_possible(op_call: ScopeOpCall, sctx: DispatchContext):
    # mirror get_vec_len inputs
    op_call = ScopeOpCall.downcast(op_call)
    dst_buffer_region, src_buffer_region = op_call.dst, op_call.src
    if sctx.exec_scope.name == "cta":
        tx = sctx.launch_params["threadIdx.x"].dom.extent
    elif sctx.exec_scope.name == "thread":
        tx = 1
    else:
        return False, f"unsupported exec_scope {sctx.exec_scope.name} for vec_len"
    vec_len = op_call.config.get("vec_len", None)
    if vec_len is None:
        vec_len = get_vec_len(
            dst_buffer_region,
            src_buffer_region,
            [
                128 // tvm.runtime.DataType(src_buffer_region.buffer.dtype).bits,
                64 // tvm.runtime.DataType(src_buffer_region.buffer.dtype).bits,
                32 // tvm.runtime.DataType(src_buffer_region.buffer.dtype).bits,
                1,
            ],
            thread_cnt=tx,
        )
    if vec_len is None:
        return False, "no valid vector length; check alignment/extents/thread-count"
    return True, None


# === Variant: copy/vec_load (priority=10) ===
#
# When: copy between global↔shared, global↔local, or shared↔local, and the
# layout allows vectorized access (vec_len > 1 for the element type).
#
# Before (ScopeOpCall):
#     with Tx.cta():
#         Tx.copy(A_smem[0:64, 0:64], A[0:64, 0:64])
#         # A: global float16, A_smem: shared float16
#
# After (thread_cnt=128, vec_len=8):
#     for s in Tx.serial(ceildiv(4096, 8 * 128)):
#         for vec in Tx.vectorized(8):
#             fused = s * 1024 + threadIdx.x * 8 + vec
#             if fused < 4096:
#                 A_smem[fused // 64, fused % 64] = A[fused // 64, fused % 64]
@register_dispatch(
    "copy",
    "cuda",
    variant="vec_load",
    priority=10,
    when=[
        predicate("validate_copy_op", _is_valid_copy),
        predicate("storage_scope", _scope_allowed),
        predicate("exec_scope", exec_scope_ok, expected_scopes=["cta", "thread"]),
        predicate("vec_len", _vec_len_possible),
    ],
)
def copy_schedule_vec_load(op_call: ScopeOpCall, sctx: DispatchContext) -> PrimFunc:
    # Delegate to the fast vectorized path
    return copy_vec_load_impl(op_call, sctx, CopyInstType.NORMAL)


# === Variant: copy/default (priority=0) ===
#
# When: any valid copy op where vec_load predicates fail (e.g. non-power-of-2
# extent, or unsupported scope pair for vectorization). Scalar element loop.
#
# After: nested for-loops over each dimension, one element at a time:
#     for i in Tx.serial(ext0):
#         for j in Tx.serial(ext1):
#             dst[dst_st0+i, dst_st1+j] = src[src_st0+i, src_st1+j]
@register_dispatch(
    "copy",
    "cuda",
    variant="default",
    priority=0,
    when=[
        predicate("validate_copy_op", _is_valid_copy),
        predicate("exec_scope", exec_scope_ok, expected_scopes=["cta", "thread"]),
    ],
)
def copy_schedule_default(op_call: ScopeOpCall, sctx: DispatchContext) -> PrimFunc:
    # Conservative scalar fallback
    return copy_default_impl(op_call, sctx)


def copy_tmem_local_impl(
    op_call: ScopeOpCall, sctx: DispatchContext, async_op=False
) -> PrimFunc | None:
    op_call = ScopeOpCall.downcast(op_call)
    dst_buffer_region, src_buffer_region = op_call.dst, op_call.src
    dst: Buffer = dst_buffer_region.buffer
    src: Buffer = src_buffer_region.buffer

    if src.scope() == "tmem" and dst.scope() == "local":
        direction = "tmem2local"
        tmem_region, local_region = src_buffer_region, dst_buffer_region
    elif src.scope() == "local" and dst.scope() == "tmem":
        direction = "local2tmem"
        local_region, tmem_region = src_buffer_region, dst_buffer_region
    else:
        raise ValueError(f"Unsupported src scope {src.scope()} and dst scope {dst.scope()}")

    tmem_buf, local_buf = tmem_region.buffer, local_region.buffer

    assert tmem_buf.layout is not None
    assert local_buf.layout is not None
    assert tmem_buf.dtype == local_buf.dtype

    analyzer = Analyzer()
    elem_size = DataType(local_buf.dtype).bits
    elem_per_32b = 32 // elem_size
    assert len(local_buf.shape) == len(tmem_buf.shape) == 2
    # local: 128xWIDTH <-> tmem: 128xSHAPE[1]
    assert analyzer.can_prove_equal(local_buf.shape[0], 128)
    assert analyzer.can_prove_equal(tmem_buf.shape[0], 128)

    # Check width is valid for 32x32b, and determine num
    width = local_region.region[1].extent
    candidates = [1, 2, 4, 8, 16, 32, 64, 128]

    if not analyzer.can_prove_equal(tvm.tirx.floormod(width, elem_per_32b), 0):
        raise ValueError(f"Width {width} is not valid for tcgen05.ld/st with shape 32x32b")

    num = None
    for n in candidates:
        if analyzer.can_prove_equal(tvm.tirx.floordiv(width, elem_per_32b), n):
            num = n
            break
    else:
        raise ValueError(f"Width {width} is not valid for tcgen05.ld/st with shape 32x32b")

    tmem_st, tmem_extent = get_st_extent(tmem_region)
    local_st, local_extent = get_st_extent(local_region)
    # tmem layout (128, WIDTH):(1@TLane, 1@TCol)
    tmem_layout = TileLayout(S[(128, tmem_buf.shape[1]) : (1 @ TLane, 1 @ TCol)]).canonicalize()
    # local layout
    TileLayout(S[(128, width) : (1 @ tid_in_wg, 1)]).canonicalize()

    # tmem allocated addr is not None
    assert tmem_buf.allocated_addr is not None
    tvm.ir.assert_structural_equal(tmem_buf.layout.canonicalize(), tmem_layout)
    # tvm.ir.assert_structural_equal(local_buf.layout.canonicalize(), local_layout)
    # local: [0:128, 0:WIDTH] <-> tmem: [0:128, st:st+WIDTH]
    assert analyzer.can_prove_equal(tmem_st[0], 0)
    assert analyzer.can_prove_equal(tmem_extent[0], 128)

    assert analyzer.can_prove_equal(local_st[0], 0)
    assert analyzer.can_prove_equal(local_extent[0], 128)

    offset = tmem_st[1]
    assert analyzer.can_prove_equal(tvm.tirx.floormod(offset, elem_per_32b), 0)
    offset_32b = tvm.tirx.floordiv(offset, elem_per_32b)
    assert analyzer.can_prove_equal(tmem_extent[1], width), (
        f"tmem_extent[1]: {tmem_extent[1]}, width: {width}"
    )

    # assert analyzer.can_prove_equal(local_st[1], 0)
    assert analyzer.can_prove_equal(local_extent[1], width)

    op = Tx.ptx.tcgen05.ld if direction == "tmem2local" else Tx.ptx.tcgen05.st
    wait_op = Tx.ptx.tcgen05.wait.ld if direction == "tmem2local" else Tx.ptx.tcgen05.wait.st

    # fmt: off
    @Tx.prim_func(tirx=True, check_well_formed=False)
    def impl():
        with Tx.warp():
            local_storage = local_buf.view(local_buf.shape[1] * elem_per_32b, layout=TileLayout(S[num * elem_per_32b]))  # noqa: E501
            local_32b = local_storage.view("uint32")
            op(tmem_buf.allocated_addr[0], 0, offset_32b, "32x32b", num, False, *[local_32b[local_st[1] // elem_per_32b+i] for i in range(num)])  # noqa: E501
            if not async_op:
                wait_op()
    # fmt: on
    return impl


# === Variant: copy/tmem<->local (priority=10) ===
#
# When: one buffer is in tmem (tensor memory, Blackwell SM100+) and the other
# is in local scope, at warpgroup exec scope.
#
# Before (ScopeOpCall):
#     with Tx.warpgroup():
#         Tx.copy(acc_local[0:16, 0:128], acc_tmem[0:16, 0:128])
#
# After (tmem→local uses tcgen05.ld):
#     Tx.ptx.tcgen05.ld(num_regs, tmem_addr, ...)
@register_dispatch(
    "copy",
    "cuda",
    variant="tmem<->local",
    priority=10,
    when=[
        predicate("validate_copy_op", _is_valid_copy),
        predicate("exec_scope", exec_scope_ok, expected_scopes=["warpgroup"]),
        predicate(
            "storage_scope", _scope_allowed, allowed_pairs=[("tmem", "local"), ("local", "tmem")]
        ),
    ],
)
def copy_schedule_tmem_local(op_call: ScopeOpCall, sctx: DispatchContext) -> PrimFunc | None:
    return copy_tmem_local_impl(op_call, sctx)


def copy_smem_tmem_impl(
    op_call: ScopeOpCall, sctx: DispatchContext, async_op=False
) -> PrimFunc | None:
    """Schedule SMEM -> TMEM copy using tcgen05.cp.

    This implements the copy from shared memory to tensor memory using the tcgen05.cp
    instruction. The copy is issued by a single thread, and data is multicast to all
    warps in the warpgroup (for 32x128b shape) or copied directly (for 128x256b/128x128b).

    Supported copy shapes:
        - "32x128b": 32 rows x 128 bits, multicast="warpx4"
        - "128x256b": 128 rows x 256 bits, multicast=""
        - "128x128b": 128 rows x 128 bits, multicast=""

    For sync (copy): waits on mbarrier after commit.
    For async (copy_async): defers synchronization to caller (e.g., pipelined MMA).
    """
    dst_buffer_region, src_buffer_region = op_call.args[:2]
    dst: Buffer = dst_buffer_region.buffer
    src: Buffer = src_buffer_region.buffer

    if not (src.scope().startswith("shared") and dst.scope() == "tmem"):
        raise ValueError(f"Expected shared->tmem, got {src.scope()}->{dst.scope()}")

    smem_buf, tmem_buf = src, dst
    smem_region, tmem_region = src_buffer_region, dst_buffer_region

    analyzer = Analyzer()

    # Extract region bounds
    smem_st, smem_ext = get_st_extent(smem_region)
    tmem_st, tmem_ext = get_st_extent(tmem_region)

    # Validate 2D buffers
    if len(smem_buf.shape) != 2 or len(tmem_buf.shape) != 2:
        raise ValueError("smem and tmem buffers must be 2D")

    # Validate tmem constraints
    if not analyzer.can_prove_equal(tmem_buf.shape[0], 128):
        raise ValueError("tmem buffer must have 128 rows")
    if not analyzer.can_prove_equal(tmem_st[0], 0):
        raise ValueError("tmem row start must be 0")
    if not analyzer.can_prove_equal(tmem_ext[0], 128):
        raise ValueError("tmem row extent must be 128")
    if tmem_buf.allocated_addr is None:
        raise ValueError("tmem buffer must have allocated_addr")

    # Determine copy shape based on smem rows
    smem_rows = smem_ext[0]
    smem_dtype_bits = DataType(smem_buf.dtype).bits
    tmem_dtype_bits = DataType(tmem_buf.dtype).bits

    if analyzer.can_prove_equal(smem_rows, 32):
        copy_shape = "32x128b"
        multicast = "warpx4"
        bits_per_copy = 128
        copy_rows = 32
    elif analyzer.can_prove_equal(smem_rows, 128):
        # Choose 128x256b or 128x128b based on alignment
        col_bits = smem_ext[1] * smem_dtype_bits
        if analyzer.can_prove_equal(tvm.tirx.floormod(col_bits, 256), 0):
            copy_shape = "128x256b"
            bits_per_copy = 256
        else:
            copy_shape = "128x128b"
            bits_per_copy = 128
        multicast = ""
        copy_rows = 128
    else:
        raise ValueError(f"smem rows must be 32 or 128, got {smem_rows}")

    # Validate row alignment
    if not analyzer.can_prove_equal(tvm.tirx.floormod(smem_st[0], copy_rows), 0):
        raise ValueError(f"smem row start must be aligned to {copy_rows}")

    # Validate column alignment (128b boundary)
    elem_per_128b = 128 // smem_dtype_bits
    if not analyzer.can_prove_equal(tvm.tirx.floormod(smem_st[1], elem_per_128b), 0):
        raise ValueError(f"smem col start must be aligned to {elem_per_128b} elements (128b)")
    if not analyzer.can_prove_equal(tvm.tirx.floormod(tmem_st[1], elem_per_128b), 0):
        raise ValueError(f"tmem col start must be aligned to {elem_per_128b} elements (128b)")
    if not analyzer.can_prove_equal(tvm.tirx.floormod(smem_ext[1], elem_per_128b), 0):
        raise ValueError(f"smem col extent must be aligned to {elem_per_128b} elements (128b)")

    # Validate bit-width match
    smem_col_bits = smem_ext[1] * smem_dtype_bits
    tmem_col_bits = tmem_ext[1] * tmem_dtype_bits
    if not analyzer.can_prove_equal(smem_col_bits, tmem_col_bits):
        raise ValueError("smem and tmem column bit-widths must match")

    # Get swizzle mode from layout
    swizzle_mode = get_swizzle_mode_from_layout(smem_buf.layout)
    if swizzle_mode is None:
        raise ValueError(f"Cannot determine swizzle mode from smem layout: {smem_buf.layout}")

    # Validate tmem layout: must be TileLayout(([128, WIDTH], [1@TLane, 1@TCol]))
    expected_tmem_layout = TileLayout(
        S[(128, tmem_buf.shape[1]) : (1 @ TLane, 1 @ TCol)]
    ).canonicalize()
    if not tvm.ir.structural_equal(tmem_buf.layout.canonicalize(), expected_tmem_layout):
        raise ValueError("tmem layout must be (128, WIDTH):(1@TLane, 1@TCol)")

    # Compute LDO/SDO using unified formula
    atom_row_bytes = {
        SwizzleMode.SWIZZLE_NONE: 16,
        SwizzleMode.SWIZZLE_32B_ATOM: 32,
        SwizzleMode.SWIZZLE_64B_ATOM: 64,
        SwizzleMode.SWIZZLE_128B_ATOM: 128,
    }[swizzle_mode]
    sdo = 8 * atom_row_bytes // 16
    ldo = (copy_rows // 8) * sdo

    # Compute iteration parameters
    vec_len = bits_per_copy // smem_dtype_bits
    num_col_iters_expr = smem_ext[1] // vec_len
    num_col_iters = int(analyzer.simplify(num_col_iters_expr))
    if num_col_iters < 1:
        raise ValueError("smem column extent must cover at least one vector")

    # desc_offset_16B computation: ci * SMEM_ROWS * BYTES_PER_COPY // 16
    bytes_per_copy = bits_per_copy // 8

    def desc_offset_16B(ci):
        if swizzle_mode == SwizzleMode.SWIZZLE_NONE:
            return ci * copy_rows * bytes_per_copy // 16
        else:
            atom_cols = atom_row_bytes * 8 // smem_dtype_bits
            frags_per_atom = atom_cols // vec_len
            return (ci // frags_per_atom) * copy_rows * bytes_per_copy // 16 * frags_per_atom + (
                ci % frags_per_atom
            ) * (bytes_per_copy // 16)

    # Get config
    cta_group = op_call.config.get("cta_group", 1)
    mbar = op_call.config.get("mbar", None)
    cta_mask = op_call.config.get("cta_mask", None)

    # mbar is required for tcgen05.commit
    if mbar is None:
        raise ValueError("mbar must be provided in config for smem->tmem copy")

    tmem_addr = tmem_buf.allocated_addr
    smem_ptr_base = smem_buf.ptr_to([smem_st[0], smem_st[1]])
    tmem_col_start = tmem_st[1]

    # fmt: off
    @Tx.prim_func(tirx=True, check_well_formed=False)
    def impl():
        cp_desc = Tx.alloc_local([1], "uint64", name="cp_desc")
        Tx.ptx.tcgen05.encode_matrix_descriptor(
            cp_desc.data, smem_ptr_base, ldo, sdo, swizzle_mode.value
        )
        for ci in Tx.unroll(num_col_iters):
            offset_16B = Tx.meta_var(desc_offset_16B(ci))
            tmem_col = Tx.meta_var((tmem_col_start + ci * vec_len) * smem_dtype_bits // 32)
            desc_val = smem_desc_add_16B_offset(cp_desc[0], offset_16B)
            Tx.ptx.tcgen05.cp(
                tmem_addr[0],
                0,
                tmem_col,
                desc_val,
                copy_shape,
                smem_buf.dtype,
                tmem_buf.dtype,
                cta_group,
                multicast
            )
        if cta_mask is not None:
            Tx.ptx.tcgen05.commit(mbar, cta_group=cta_group, cta_mask=cta_mask)
        else:
            Tx.ptx.tcgen05.commit(mbar, cta_group=cta_group)
    # fmt: on

    return impl


def _is_valid_smem_tmem_copy(op_call: ScopeOpCall, sctx: DispatchContext):
    """Validate smem->tmem copy operation.

    Unlike generic copy validation, this allows different dtypes as long as
    column bit-widths match.
    """
    dst_region, src_region = op_call.args[:2]
    src: Buffer = src_region.buffer
    dst: Buffer = dst_region.buffer

    # Check storage scopes
    if not (src.scope().startswith("shared") and dst.scope() == "tmem"):
        return False, f"expected shared->tmem, got {src.scope()}->{dst.scope()}"

    # Check layouts exist
    if not (src.layout and dst.layout):
        return False, "both buffers must have layouts"

    # Check 2D buffers
    if len(src.shape) != 2 or len(dst.shape) != 2:
        return False, "both buffers must be 2D"

    # Check tmem has allocated_addr
    if dst.allocated_addr is None:
        return False, "tmem buffer must have allocated_addr"

    # Check bit-width of columns match (allowing different dtypes)
    analyzer = Analyzer()
    src_ext = [r.extent for r in src_region.region]
    dst_ext = [r.extent for r in dst_region.region]
    src_dtype_bits = DataType(src.dtype).bits
    dst_dtype_bits = DataType(dst.dtype).bits
    src_col_bits = src_ext[1] * src_dtype_bits
    dst_col_bits = dst_ext[1] * dst_dtype_bits
    if not analyzer.can_prove_equal(src_col_bits, dst_col_bits):
        return False, "column bit-widths must match"

    return True, None


def _single_thread_exec(op_call: ScopeOpCall, sctx: DispatchContext):
    """Check if execution scope is single-thread."""
    exec_scope = sctx.exec_scope.name
    ok = exec_scope == "thread"
    return ok, None if ok else f"expected thread exec_scope, got {exec_scope}"


# === Variant: copy/smem->tmem (priority=10) ===
#
# When: src is shared memory, dst is tensor memory (Blackwell SM100+),
# at single-thread exec scope.
#
# Before (ScopeOpCall):
#     Tx.copy(acc_tmem[0:16, 0:128], A_smem[0:16, 0:128])
#
# After (tcgen05.cp with descriptor encoding):
#     Tx.ptx.tcgen05.cp(num_regs, tmem_addr, smem_desc, ...)
@register_dispatch(
    "copy",
    "cuda",
    variant="smem->tmem",
    priority=10,
    when=[
        predicate("validate_smem_tmem_copy", _is_valid_smem_tmem_copy),
        predicate("exec_scope", _single_thread_exec),
    ],
)
def copy_schedule_smem_tmem(op_call: ScopeOpCall, sctx: DispatchContext) -> PrimFunc | None:
    return copy_smem_tmem_impl(op_call, sctx, async_op=False)
