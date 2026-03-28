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

"""CUDA copy operator dispatch: smem->tmem variant.

Registered ops: copy (smem->tmem, priority=10).

Also provides copy_smem_tmem_impl used by copy_async.py.
"""

import tvm
from tvm.arith import Analyzer
from tvm.runtime import DataType
from tvm.script import tirx as Tx
from tvm.tirx import Buffer, PrimFunc
from tvm.tirx.layout import S, TCol, TileLayout, TLane
from tvm.tirx.operator.scope_op_dispatch.dispatcher import predicate, register_dispatch
from tvm.tirx.operator.scope_op_dispatch.registry import DispatchContext
from tvm.tirx.stmt import ScopeOpCall

from ..common import (
    get_st_extent,
    smem_desc_add_16B_offset,
)
from ..tma_utils import SwizzleMode, get_swizzle_mode_from_layout
from .utils import _is_valid_smem_tmem_copy, _single_thread_exec


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
