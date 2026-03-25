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

"""Implementation of gemm_async operator dispatch for CUDA targets.

Registered op: gemm_async (1 variant: "tcgen05").
See the @register_dispatch block below for detailed documentation with
before/after IR examples.
"""

import functools
import operator

import tvm
from tvm.arith.analyzer import Analyzer
from tvm.runtime import DataType
from tvm.script import tirx as Tx
from tvm.tir import PrimFunc
from tvm.tir.layout import ComposeLayout, R, S, TCol, TileLayout, TLane
from tvm.tir.stmt import AllocBuffer, Evaluate, OpCall, SeqStmt
from tvm.tirx.op_dispatch import DispatchContext, predicate, register_dispatch
from tvm.tirx.operator.op import KernelReplacePoint

from .common import get_st_extent, smem_desc_add_16B_offset
from .exec_scope_utils import single_thread
from .tma_utils import SwizzleMode, tma_atom_layout, tma_atom_shape


def sf_tmem_layout(rows, sf_mma_k, K, dtype="float8_e8m0fnu"):
    """Create a TileLayout for SFA/SFB TMEM via atom direct_sum outer.

    Args:
        rows: total rows (multiple of 32)
        sf_mma_k: scale factors per MMA in K direction (1, 2, or 4)
        K: total outer K iterations
        dtype: scale factor dtype (for computing elem_per_col)

    Buffer shape should be (rows, sf_mma_k * K).
    """
    M = rows // 32
    epc = 32 // DataType(dtype).bits  # elem_per_col

    # Atom: one 32-row chunk, one MMA's worth of SF
    atom = TileLayout(S[(32, sf_mma_k) : (1 @ TLane, 1 @ TCol)] + R[4 : 32 @ TLane])

    if K == 1:
        outer = TileLayout(S[M : epc @ TCol])
    else:
        # Pack consecutive ki's within one uint32 TMEM column when possible
        pack_factor = epc // sf_mma_k
        while pack_factor > 1 and K % pack_factor != 0:
            pack_factor //= 2
        if pack_factor > 1:
            K_outer = K // pack_factor
            if K_outer == 1:
                outer = TileLayout(S[(M, pack_factor) : (epc @ TCol, sf_mma_k @ TCol)])
            else:
                outer = TileLayout(
                    S[(M, K_outer, pack_factor) : (epc @ TCol, M * epc @ TCol, sf_mma_k @ TCol)]
                )
        else:
            outer = TileLayout(S[(M, K) : (epc @ TCol, M * epc @ TCol)])

    return atom.direct_sum(outer, left_shape=[M, K], right_shape=[32, sf_mma_k])


def _compute_sf_mma_k(data_dtype, sf_dtype):
    """Compute sf_mma_k (scale factor elements per MMA iteration) from dtypes.

    This is determined by hardware constraints:
    - fp8 data + e8m0fnu SF: MMA_K=32, one SF per MMA → sf_mma_k=1
    - fp4 data + e8m0fnu SF: MMA_K=64, SF_VEC=32 → sf_mma_k=2
    - fp4 data + e4m3fn SF (nvfp4): MMA_K=64, SF_VEC=16 → sf_mma_k=4
    """
    data_dtype = str(data_dtype)
    sf_dtype = str(sf_dtype)
    if data_dtype in ("float8_e4m3fn", "float8_e5m2"):
        return 1  # MMA_K=32, one SF per MMA
    elif data_dtype == "float4_e2m1fn":
        if sf_dtype == "float8_e8m0fnu":
            return 2  # MMA_K=64, SF_VEC=32
        elif sf_dtype == "float8_e4m3fn":
            return 4  # MMA_K=64, SF_VEC=16 (nvfp4)
    raise ValueError(f"Unsupported data_dtype={data_dtype}, sf_dtype={sf_dtype} for sf_mma_k")


def _validate_sf_tmem_layout(slice_layout, rows, sf_K_total, sf_mma_k, name):
    """Validate SFA/SFB TMEM sliced layout matches atom direct_sum outer pattern.

    Validates that slice_layout (already sliced to last 2D: rows x sf_K_total)
    matches the atom:
      shard = ([32, sf_mma_k], [1@TLane, 1@TCol])
      replica = ([4], [32@TLane])
    """
    assert isinstance(slice_layout, TileLayout), (
        f"{name}: sliced layout must be TileLayout, got {type(slice_layout)}"
    )
    M = rows // 32

    assert sf_K_total % sf_mma_k == 0, (
        f"{name}: sf_K_total={sf_K_total} must be divisible by sf_mma_k={sf_mma_k}"
    )
    K = sf_K_total // sf_mma_k

    atom = TileLayout(S[(32, sf_mma_k) : (1 @ TLane, 1 @ TCol)] + R[4 : 32 @ TLane])
    # interleaved_shape is the interleaved domain [M, 32, K, sf_mma_k]
    outer = atom.is_direct_sum_right(slice_layout, [M, 32, K, sf_mma_k], [32, sf_mma_k])
    assert outer is not None, f"{name}: layout does not match atom direct_sum outer pattern"


def _choose_mma_tile(M, N, cta_group, MMA_N_MIN):
    """Select per-instruction (M_mma, N_mma) for tcgen05 tile decomposition.

    M_mma: largest valid M that divides M.
      Valid values: {128, 64} for cta_group=1, {256, 128} for cta_group=2.
    N_mma: if N <= 256 and N % MMA_N_MIN == 0, use N directly.
      Otherwise, largest valid N_mma <= 256 that divides N and is divisible by MMA_N_MIN.
    """
    valid_M = [128, 64] if cta_group == 1 else [256, 128]
    M_mma = next((m for m in valid_M if M % m == 0), None)
    assert M_mma is not None, (
        f"tcgen05: M={M} not divisible by any valid MMA M for cta_group={cta_group} "
        f"(valid: {valid_M})"
    )

    if N <= 256 and N % MMA_N_MIN == 0:
        N_mma = N
    else:
        N_mma = next((n for n in range(256, MMA_N_MIN - 1, -MMA_N_MIN) if N % n == 0), None)
        assert N_mma is not None, (
            f"tcgen05: No valid N_mma <= 256 that divides N={N} (MMA_N_MIN={MMA_N_MIN})"
        )

    return M_mma, N_mma


def gemm_async_tcgen05_impl(op_call: OpCall, sctx: DispatchContext) -> PrimFunc:
    """Schedule an asynchronous GEMM operation using tcgen05.mma (Blackwell Tensor Core).

    Computes C = A @ B (with optional transpose on A/B and accumulation).
    Supports both regular MMA and block-scaled MMA for low-precision dtypes.

    When called from warp scope, automatically wraps tcgen05.mma with elect_sync
    so that only one thread in the warp issues the MMA instruction.

    Args:
        op_call: The OpCall containing:
            Regular (6 args):
            - args[0:3]: C, A, B buffer regions
            - args[3:6]: transA, transB, accum flags
            Block-scaled (8 args):
            - args[0:3]: C, A, B buffer regions
            - args[3:5]: SFA, SFB buffer regions (scale factors in tmem)
            - args[5:8]: transA, transB, accum flags
            Config:
            - config["cta_group"]: CTA group in tcgen05 instructions (default 1)
            - config["descI"]: Optional pre-encoded instruction descriptor
        sctx: Schedule context (single-thread or warp execution scope)

    Returns:
        A PrimFunc implementing the tcgen05 MMA schedule.

    Raises:
        ValueError: If buffer scopes are invalid (C must be tmem, A must be shared or tmem,
            B must be shared).
        AssertionError: If shape/layout constraints are not satisfied.
    """
    warp_scope = sctx.exec_scope.name == "warp"
    op_call = OpCall.downcast(op_call)
    is_block_scaled = op_call.is_block_scaled

    C_buffer_region: tvm.tir.BufferRegion = op_call.output
    A_buffer_region: tvm.tir.BufferRegion = op_call.lhs
    B_buffer_region: tvm.tir.BufferRegion = op_call.rhs
    C_buffer, A_buffer, B_buffer = (
        C_buffer_region.buffer,
        A_buffer_region.buffer,
        B_buffer_region.buffer,
    )

    C_scope, A_scope, B_scope = C_buffer.scope(), A_buffer.scope(), B_buffer.scope()
    a_is_tmem = A_scope == "tmem"
    if a_is_tmem:
        if not (C_scope == "tmem" and B_scope.startswith("shared")):
            raise ValueError(
                f"tcgen05 schedule expected C_scope=tmem, B_scope=shared when A is tmem, "
                f"got C_scope={C_scope}, B_scope={B_scope}"
            )
    elif not (C_scope == "tmem" and A_scope.startswith("shared") and B_scope.startswith("shared")):
        raise ValueError(
            f"tcgen05 schedule expected C_scope=tmem, A_scope=shared, B_scope=shared, got C_scope={C_scope}, A_scope={A_scope}, B_scope={B_scope}"  # noqa: E501
        )

    analyzer = Analyzer()

    C_type, A_type, B_type = C_buffer.dtype, A_buffer.dtype, B_buffer.dtype
    assert C_type == "float32", f"tcgen05 schedule expected C_type=float32, got {C_type}"

    # Valid A/B dtypes for block-scaled MMA (low-precision with per-block scale factors)
    _BLOCK_SCALED_DTYPES = [
        "float4_e2m1fn",
        "float8_e4m3fn",
    ]

    _SCALE_FACTOR_DTYPES = [
        "float8_e8m0fnu",
        "float8_e4m3fn",
    ]

    if is_block_scaled:
        assert A_type in _BLOCK_SCALED_DTYPES, (
            f"tcgen05 block-scaled schedule expected A_type in {_BLOCK_SCALED_DTYPES}, got {A_type}"
        )
        assert B_type in _BLOCK_SCALED_DTYPES, (
            f"tcgen05 block-scaled schedule expected B_type in {_BLOCK_SCALED_DTYPES}, got {B_type}"
        )
    else:
        assert A_type in [
            "float16",
            "bfloat16",
        ], f"tcgen05 schedule expected A_type=float16 or bfloat16, got {A_type}"
        assert B_type in [
            "float16",
            "bfloat16",
        ], f"tcgen05 schedule expected B_type=float16 or bfloat16, got {B_type}"
    assert A_type == B_type, (
        f"tcgen05 schedule expect A_type and B_type to be the same, got A_type={A_type}, B_type={B_type}"  # noqa: E501
    )

    # Parse SFA/SFB and transA/transB/accum based on arg layout
    if is_block_scaled:
        SFA_buffer_region, SFB_buffer_region = op_call.sfa, op_call.sfb
        transA, transB, accum = op_call.transA, op_call.transB, op_call.accum
        SFA_buffer: tvm.tir.Buffer = SFA_buffer_region.buffer
        SFB_buffer: tvm.tir.Buffer = SFB_buffer_region.buffer
        SFA_scope, SFB_scope = SFA_buffer.scope(), SFB_buffer.scope()
        if not (SFA_scope == "tmem" and SFB_scope == "tmem"):
            raise ValueError(
                f"tcgen05 block-scaled schedule expected SFA_scope=tmem, SFB_scope=tmem, "
                f"got SFA_scope={SFA_scope}, SFB_scope={SFB_scope}"
            )
        SFA_type, SFB_type = SFA_buffer.dtype, SFB_buffer.dtype
        SFA_slice_layout = SFA_buffer.layout.slice(SFA_buffer.shape, SFA_buffer_region.region)
        SFB_slice_layout = SFB_buffer.layout.slice(SFB_buffer.shape, SFB_buffer_region.region)
        SFA_elem_per_col = 32 // DataType(SFA_type).bits
        SFB_elem_per_col = 32 // DataType(SFB_type).bits
        assert SFA_type in _SCALE_FACTOR_DTYPES, (
            f"tcgen05 block-scaled schedule expected SFA_type in {_SCALE_FACTOR_DTYPES}, got {SFA_type}"  # noqa: E501
        )
        assert SFB_type in _SCALE_FACTOR_DTYPES, (
            f"tcgen05 block-scaled schedule expected SFB_type in {_SCALE_FACTOR_DTYPES}, got {SFB_type}"  # noqa: E501
        )
        # Compute sf_mma_k from data/SF dtypes and validate layouts
        sfa_sf_mma_k = _compute_sf_mma_k(A_type, SFA_type)
        sfb_sf_mma_k = _compute_sf_mma_k(B_type, SFB_type)
        assert sfa_sf_mma_k == sfb_sf_mma_k, (
            f"SFA and SFB must have same sf_mma_k, got sfa={sfa_sf_mma_k}, sfb={sfb_sf_mma_k}"
        )
        SFA_rows = int(SFA_buffer_region.region[-2].extent)
        SFA_K_total = int(SFA_buffer_region.region[-1].extent)
        SFB_rows = int(SFB_buffer_region.region[-2].extent)
        SFB_K_total = int(SFB_buffer_region.region[-1].extent)
        _validate_sf_tmem_layout(SFA_slice_layout, SFA_rows, SFA_K_total, sfa_sf_mma_k, "SFA")
        _validate_sf_tmem_layout(SFB_slice_layout, SFB_rows, SFB_K_total, sfb_sf_mma_k, "SFB")
    else:
        transA, transB, accum = op_call.transA, op_call.transB, op_call.accum

    cta_group = op_call.config.get("cta_group", 1)
    assert cta_group in [1, 2], f"tcgen05 schedule expected cta_group=1 or 2, got {cta_group}"
    # descI: pre-encoded instruction descriptor (uint32), if None we encode it locally
    descI = op_call.config.get("descI", None)

    C_elem_size = DataType(C_type).bits
    C_elem_per_32b = 32 // C_elem_size
    C_st, C_extent = get_st_extent(C_buffer_region)
    _, A_extent = get_st_extent(A_buffer_region)
    _, B_extent = get_st_extent(B_buffer_region)
    A_slice_layout = A_buffer.layout.slice(A_buffer.shape, A_buffer_region.region)
    B_slice_layout = B_buffer.layout.slice(B_buffer.shape, B_buffer_region.region)
    C_slice_layout = C_buffer.layout.slice(C_buffer.shape, C_buffer_region.region)
    # Extract pre-swizzle tile layout for descriptor offset computation
    if not a_is_tmem:
        A_slice_tile = (
            A_slice_layout.tile_layout
            if isinstance(A_slice_layout, ComposeLayout)
            else A_slice_layout
        )
    B_slice_tile = (
        B_slice_layout.tile_layout if isinstance(B_slice_layout, ComposeLayout) else B_slice_layout
    )

    assert len(C_extent) == 2 and len(A_extent) >= 2 and len(B_extent) >= 2, (
        "Only 2D C, A, B are supported for gemm"
    )

    def _mat_dim_vals(extent, name):
        """Extract the two non-unit dimension values from a GEMM operand extent."""
        vals = [int(e) for e in extent if not analyzer.can_prove_equal(e, 1)]
        assert len(vals) == 2, (
            f"Expected exactly 2 non-unit dims in {name}_extent {[int(e) for e in extent]}"
        )
        return vals[0], vals[1]

    M = int(C_extent[-2])
    N = int(C_extent[-1])

    # Majorness (a_mn_major / b_mn_major) is determined later by
    # compute_canonical_params via dual-atom matching on the physical
    # SMEM layout.  Extract dim extents here for cross-validation.
    # Use non-unit dims (not last-2) to handle unit dims in the middle
    # (e.g. region shape [M, 1, K]).
    A_dim2, A_dim1 = _mat_dim_vals(A_extent, "A")
    B_dim2, B_dim1 = _mat_dim_vals(B_extent, "B")

    # Compute SMEM descriptor parameters (swizzle mode, ldo, sdo) and infer
    # majorness by matching the sliced layout against both K-major atom
    # [8, T*s] and MN-major atom [T*s, 8] via is_tile_inner.
    #
    # Priority: MN-major atom match → definitively MN-major (column-major SMEM).
    # K-major atom match → use extent matching to determine semantic majorness,
    # since tma_shared_layout creates K-major layouts for both [M,K] and [K,M].
    def compute_canonical_params(buf, buf_region, dtype, is_transposed):
        """Compute descriptor parameters from buffer layout.

        Uses is_transposed (from op's transA/transB) to determine which
        atom orientation corresponds to K-major for this buffer:
        - transposed=False: buffer is [MN, K], K-major atom = [8, T*s]
        - transposed=True:  buffer is [K, MN], K-major atom = [T*s, 8]

        Then tries both atom orientations with is_tile_inner.  Whichever
        matches determines the physical majorness.

        Strips unit dims and passes 2D shapes to is_tile_inner on the
        sliced layout — handles >2D regions like [1, M, K] or [1, 1, M, K].

        Returns:
            Tuple of (swizzle_mode, ldo, sdo, is_mn_major).
        """
        region = list(buf_region.region)
        slice_layout = buf.layout.slice(buf.shape, region)
        # Strip unit dims to get the 2D matrix shape.
        shape_2d = [int(r.extent) for r in region if int(r.extent) != 1]
        assert len(shape_2d) == 2, (
            f"Expected exactly 2 non-unit dims in region {[int(r.extent) for r in region]}"
        )

        def _try_atom(atom, atom_shape):
            if any(s % a != 0 for s, a in zip(shape_2d, atom_shape)):
                return None
            atom_size = functools.reduce(operator.mul, atom_shape, 1)
            tiler = atom.is_tile_inner(slice_layout, shape_2d, atom_shape)
            if tiler is None:
                return None
            tiler_shape = [s // a for s, a in zip(shape_2d, atom_shape)]
            tiler_grouped, seps = tiler.canonicalize().group(tiler_shape)
            elem_per_128b = 128 // tvm.DataType(dtype).bits
            ldo = (tiler_grouped.shard[-1].stride * atom_size) // elem_per_128b
            sdo = (tiler_grouped.shard[-2].stride * atom_size) // elem_per_128b
            return mode, ldo, sdo

        for mode in (
            SwizzleMode.SWIZZLE_128B_ATOM,
            SwizzleMode.SWIZZLE_64B_ATOM,
            SwizzleMode.SWIZZLE_32B_ATOM,
        ):
            swizzle_atom = tma_atom_layout(dtype, mode)
            base_shape = tma_atom_shape(dtype, mode)  # [8, T*s]
            swapped_shape = [base_shape[1], base_shape[0]]  # [T*s, 8]

            # MN-major atom: compose SwizzleLayout with stride-reversed TileLayout
            # so the first dim (T*s) is contiguous instead of the second.
            # Needed when the penultimate dim is physically contiguous.
            mn_tile = TileLayout(S[tuple(swapped_shape) : (1, swapped_shape[0])])
            mn_atom = ComposeLayout(swizzle_atom, mn_tile)

            # Determine K-major vs MN-major based on which dim is contiguous.
            # K-major: K dim contiguous (last dim for [MN,K], first dim for [K,MN])
            # MN-major: MN dim contiguous
            #
            # The plain swizzle_atom has last dim contiguous.
            # The mn_atom has first dim contiguous.
            #
            # For non-transposed [MN, K]: K is last dim
            #   - K-major = swizzle_atom with [8, T*s] (K contiguous in last dim)
            #   - MN-major = mn_atom with [T*s, 8] (MN contiguous in first dim)
            # For transposed [K, MN]: MN is last dim
            #   - K-major = mn_atom with [T*s, 8] (K contiguous in first dim)
            #   - MN-major = swizzle_atom with [8, T*s] (MN contiguous in last dim)
            if is_transposed:
                candidates = [
                    (False, mn_atom, swapped_shape),  # K-major: K in first dim
                    (True, swizzle_atom, base_shape),  # MN-major: MN in last dim
                ]
            else:
                candidates = [
                    (False, swizzle_atom, base_shape),  # K-major: K in last dim
                    (True, mn_atom, swapped_shape),  # MN-major: MN in first dim
                ]

            for is_mn_major, atom, atom_shape in candidates:
                result = _try_atom(atom, atom_shape)
                if result is not None:
                    sw, ldo_val, sdo_val = result
                    # shard[-1] = last-dim groups, shard[-2] = first-dim groups.
                    # LBO strides MN-groups for MN-major, K-groups for K-major.
                    # Non-transposed [MN,K]: last=K, first=MN → swap for MN-major
                    # Transposed [K,MN]: last=MN, first=K → swap for K-major
                    if is_mn_major != is_transposed:
                        ldo_val, sdo_val = sdo_val, ldo_val
                    return sw, ldo_val, sdo_val, is_mn_major

        raise ValueError(
            f"No compatible swizzle mode found for dtype {dtype} with region shape {shape_2d}"
        )

    if a_is_tmem:
        # TMEM A: hardware requires transA=False (no transpose from TMEM)
        assert not transA, "tcgen05 schedule: transA must be False when A is in tmem"
        a_mn_major = False
    else:
        A_swizzle_mode, A_ldo, A_sdo, a_mn_major = compute_canonical_params(
            A_buffer, A_buffer_region, A_type, transA
        )
    B_swizzle_mode, B_ldo, B_sdo, b_mn_major = compute_canonical_params(
        B_buffer, B_buffer_region, B_type, transB
    )

    # Extract K from A dims using transA (shape order).
    # transA tells us which dim is K; a_mn_major tells us the layout orientation.
    # transA=False [M, K]: K = dim[-1]; transA=True [K, M]: K = dim[-2]
    K = A_dim2 if transA else A_dim1

    # tcgen05 MMA hardware constraints
    # K dimension per MMA iteration depends on A/B dtype
    if A_type == "float4_e2m1fn":
        MMA_K = 64
    elif A_type in ["float8_e4m3fn", "float8_e5m2"]:
        MMA_K = 32
    else:  # float16, bfloat16
        MMA_K = 16
    MMA_N_MIN = 8 if cta_group == 1 else 16  # Minimum N dimension

    M_mma, N_mma = _choose_mma_tile(M, N, cta_group, MMA_N_MIN)
    M_tiles = M // M_mma
    N_tiles = N // N_mma
    K_iters = K // MMA_K
    N_mma_per_cta = N_mma // cta_group
    assert K % MMA_K == 0, f"tcgen05 schedule expected K % {MMA_K} == 0, got {K}"

    # Cross-validate A dimensions (shape order from transA)
    A_M = A_dim1 if transA else A_dim2
    assert A_M == M, f"tcgen05: A_M={A_M} doesn't match M={M} from C region"

    # Cross-validate K between A and B
    B_K = B_dim1 if not transB else B_dim2
    assert K == B_K, f"tcgen05: A_K={K} doesn't match B_K={B_K}"

    # Cross-validate B's N with C's N and cta_group
    B_N = B_dim2 if not transB else B_dim1
    assert B_N * cta_group == N, (
        f"tcgen05: B_N={B_N} * cta_group={cta_group}={B_N * cta_group} doesn't match N={N}"
    )

    # Validate SFA/SFB region shapes
    if is_block_scaled:
        assert SFA_rows == M, f"tcgen05: SFA rows={SFA_rows} must equal M={M}"
        assert SFB_rows >= N, f"tcgen05: SFB rows={SFB_rows} must be >= N={N}"
        valid_sfa_K = {sfa_sf_mma_k, sfa_sf_mma_k * K_iters}
        valid_sfb_K = {sfb_sf_mma_k, sfb_sf_mma_k * K_iters}
        assert SFA_K_total in valid_sfa_K, (
            f"tcgen05: SFA K extent={SFA_K_total} must be in {valid_sfa_K}"
        )
        assert SFB_K_total in valid_sfb_K, (
            f"tcgen05: SFB K extent={SFB_K_total} must be in {valid_sfb_K}"
        )

    # Check C's sliced layout: (M, N):(1@TLane, 1@TCol), allow offset
    base = TileLayout(S[(M, N) : (1 @ TLane, 1 @ TCol)])
    expected_c_layout = TileLayout.from_iters(
        base.shard, base.replica, C_slice_layout.offset
    ).canonicalize()
    tvm.ir.assert_structural_equal(C_slice_layout.canonicalize(), expected_c_layout)
    assert C_buffer.allocated_addr is not None
    tmem_addr = C_buffer.allocated_addr[0]
    tmem_offset_32b = C_slice_layout.offset.get(TCol, 0)

    # Validate TMEM A layout: (A_dim2, A_dim1):(1@TLane, 1@TCol)
    if a_is_tmem:
        A_tmem_base = TileLayout(S[(A_dim2, A_dim1) : (1 @ TLane, 1 @ TCol)])
        expected_a_layout = TileLayout.from_iters(
            A_tmem_base.shard, A_tmem_base.replica, A_slice_layout.offset
        ).canonicalize()
        tvm.ir.assert_structural_equal(A_slice_layout.canonicalize(), expected_a_layout)
        assert A_buffer.allocated_addr is not None, "TMEM A buffer must have allocated_addr"
        A_tmem_addr = A_buffer.allocated_addr[0]
        A_elem_per_32b = 32 // DataType(A_type).bits
        # TCol offset is in element units (not 32-bit columns) for sub-32-bit dtypes.
        # Convert to 32-bit column units for get_tmem_addr.
        A_tmem_offset_32b = A_slice_layout.offset.get(TCol, 0) // A_elem_per_32b

    # Convert accum to TIR bool outside the macro (TIR AST evaluator doesn't
    # support short-circuit evaluation, so accum.dtype inside macro would fail
    # when accum is a Python bool).
    if isinstance(accum, bool):
        accum_expr = tvm.tir.const(int(accum), "bool")
    elif isinstance(accum, tvm.tir.PrimExpr) and accum.dtype != "bool":
        accum_expr = tvm.tir.Cast("bool", accum)
    else:
        accum_expr = accum

    # 16B element count for descriptor offset computation
    B_elem_per_16B = 128 // DataType(B_type).bits
    if not a_is_tmem:
        A_elem_per_16B = 128 // DataType(A_type).bits

    # Allocate descriptor cells and encode once, right after A/B buffer defs.
    # The callback is inserted as a flat SeqStmt after the target buffer def.
    # Descriptors with identical construction parameters are cached and reused
    # across dispatch calls via sctx.shared_state.
    B_base = [0] * len(B_buffer.shape)
    krp = KernelReplacePoint(workspace={}, config={})

    def _make_lo_uniform(desc):
        """Shuffle the lower 32 bits of the descriptor to ensure warp-uniformity."""
        func_name = "smem_desc_make_lo_uniform_"
        source_code = f"""
        __forceinline__ __device__ void {func_name}(uint64_t* desc) {{
            SmemDescriptor* d = reinterpret_cast<SmemDescriptor*>(desc);
            d->lo = __shfl_sync(0xffffffff, d->lo, 0);
        }}
        """
        return Tx.cuda.func_call(
            func_name,
            Tx.address_of(desc),
            source_code=source_code,
            return_type="void",
        )

    def _make_desc_wrap(desc_buf, smem_buf, base, ldo, sdo, swizzle_val):
        """Build: { AllocBuffer(desc); encode(desc, smem); krp }"""
        encode_call = tvm.tir.call_intrin(
            "",
            "tir.ptx_tcgen05_encode_matrix_descriptor",
            tvm.tir.address_of(desc_buf[0]),
            smem_buf.ptr_to(base),
            ldo,
            sdo,
            swizzle_val,
        )
        return SeqStmt(
            [
                AllocBuffer(desc_buf),
                Evaluate(encode_call),
                Evaluate(_make_lo_uniform(desc_buf[0])),
                krp,
            ]
        )

    def _get_or_create_desc(smem_buf, base, ldo, sdo, swizzle_val, name):
        """Return a cached desc_buf or create and register a new one."""
        cache_key = f"smem_desc:{hash(smem_buf)}:{int(ldo)}:{int(sdo)}:{int(swizzle_val)}"
        cached = sctx.cache_get(cache_key)
        if cached is not None:
            return cached
        desc_buf = tvm.tir.decl_buffer((1,), "uint64", name=name, scope="local")
        wrap = _make_desc_wrap(desc_buf, smem_buf, base, ldo, sdo, swizzle_val)
        sctx.add_post_buffer_def_stmt(smem_buf, wrap)
        sctx.cache_set(cache_key, desc_buf)
        return desc_buf

    descB_buf = _get_or_create_desc(B_buffer, B_base, B_ldo, B_sdo, B_swizzle_mode.value, "descB")
    if not a_is_tmem:
        A_base = [0] * len(A_buffer.shape)
        descA_buf = _get_or_create_desc(
            A_buffer, A_base, A_ldo, A_sdo, A_swizzle_mode.value, "descA"
        )
    elect_pred = Tx.ptx.elect_sync() if warp_scope else True

    # Helper: compute B descriptor value for a given (ni, ki) tile
    def _b_desc_val(descB_in, ni, ki):
        B_linear = (
            ki * MMA_K * B_extent[-1] + ni * N_mma_per_cta
            if transB
            else ni * N_mma_per_cta * B_extent[-1] + ki * MMA_K
        )
        B_offset = tvm.tir.floordiv(B_slice_tile.apply(B_linear)["m"], B_elem_per_16B)
        return smem_desc_add_16B_offset(descB_in, B_offset)

    # Helper: compute A operand (TMEM address or SMEM descriptor) for a given (mi, ki) tile
    def _a_operand(mi, ki, descA_in=None):
        if a_is_tmem:
            # A is [M, K] non-transposed: M→TLane (rows), K→TCol (cols)
            a_row = mi * M_mma
            a_col = A_tmem_offset_32b + ki * (MMA_K // A_elem_per_32b)
            return Tx.cuda.get_tmem_addr(A_tmem_addr, a_row, a_col)
        else:
            A_linear = (
                ki * MMA_K * A_extent[-1] + mi * M_mma
                if transA
                else mi * M_mma * A_extent[-1] + ki * MMA_K
            )
            A_offset = tvm.tir.floordiv(A_slice_tile.apply(A_linear)["m"], A_elem_per_16B)
            return smem_desc_add_16B_offset(descA_in, A_offset)

    if is_block_scaled:
        # Compute per-ki SF element steps from region extents
        sfa_elems_per_ki = SFA_K_total // K_iters if K_iters > 0 else 0
        sfb_elems_per_ki = SFB_K_total // K_iters if K_iters > 0 else 0

        sfa_base = SFA_buffer.allocated_addr[0]
        sfb_base = SFB_buffer.allocated_addr[0]

        # Compute initial SFA/SFB addresses (for ki=0)
        # apply(0)["TCol"] at row 0 gives physical TCol offset
        sfa_tcol_0 = SFA_slice_layout.apply(0).get("TCol", 0)
        sfb_tcol_0 = SFB_slice_layout.apply(0).get("TCol", 0)
        SFA_init_addr = analyzer.simplify(sfa_base + tvm.tir.floordiv(sfa_tcol_0, SFA_elem_per_col))
        SFB_init_addr = analyzer.simplify(sfb_base + tvm.tir.floordiv(sfb_tcol_0, SFB_elem_per_col))

        # Determine if sf_id rotation is needed:
        # sf_mma_k < epc means multiple ki's pack in one column, AND we need per-ki
        # distinct SF (i.e. sfa_elems_per_ki > 0 so each ki advances to a new element)
        needs_sf_id = sfa_sf_mma_k < SFA_elem_per_col and sfa_elems_per_ki > 0 and descI is None

    # Build main_impl: descA_in is None when A is in TMEM (ignored by _a_operand).
    # fmt: off
    if is_block_scaled:
        @Tx.inline
        def main_impl(descA_in, descB_in, descI_in):
            for mi in Tx.unroll(M_tiles):
              for ni in Tx.unroll(N_tiles):
                for ki in Tx.unroll(K_iters):
                    a_val = _a_operand(mi, ki, descA_in)
                    descB_val = _b_desc_val(descB_in, ni, ki)
                    should_accum = tvm.tir.any(ki != 0, accum_expr)
                    sfa_linear = mi * M_mma * SFA_K_total + ki * sfa_elems_per_ki
                    sfb_linear = ni * N_mma_per_cta * SFB_K_total + ki * sfb_elems_per_ki
                    sfa_tcol = SFA_slice_layout.apply(sfa_linear).get("TCol", 0)
                    sfb_tcol = SFB_slice_layout.apply(sfb_linear).get("TCol", 0)
                    sfa_addr = sfa_base + tvm.tir.floordiv(sfa_tcol, SFA_elem_per_col)
                    sfb_addr = sfb_base + tvm.tir.floordiv(sfb_tcol, SFB_elem_per_col)
                    if needs_sf_id:
                        sf_id = Tx.meta_var(analyzer.simplify(tvm.tir.floormod(sfa_tcol, SFA_elem_per_col)))  # noqa: E501
                        Tx.cuda.runtime_instr_desc(Tx.address_of(descI_in), sf_id)
                    tmem_col = tmem_offset_32b + ni * (N_mma // C_elem_per_32b)
                    if elect_pred:
                        Tx.ptx.tcgen05.mma.block_scale(
                            C_type, A_type, B_type, SFA_type, SFB_type,
                            Tx.cuda.get_tmem_addr(tmem_addr, mi * M_mma, tmem_col),
                            a_val, descB_val,
                            sfa_addr, sfb_addr,
                            descI_in, a_is_tmem, cta_group, should_accum,
                        )
    else:
        @Tx.inline
        def main_impl(descA_in, descB_in, descI_in):
            for mi in Tx.unroll(M_tiles):
              for ni in Tx.unroll(N_tiles):
                for ki in Tx.unroll(K_iters):
                    a_val = _a_operand(mi, ki, descA_in)
                    descB_val = _b_desc_val(descB_in, ni, ki)
                    should_accum = tvm.tir.any(ki != 0, accum_expr)
                    tmem_col = tmem_offset_32b + ni * (N_mma // C_elem_per_32b)
                    if elect_pred:
                        Tx.ptx.tcgen05.mma(
                            "float32", A_type, B_type,
                            Tx.cuda.get_tmem_addr(tmem_addr, mi * M_mma, tmem_col),
                            a_val, descB_val, descI_in, a_is_tmem, cta_group, should_accum,
                        )

    descA_val = None if a_is_tmem else descA_buf[0]

    if descI is not None:
        @Tx.prim_func(tirx=True, check_well_formed=False)
        def impl():
            main_impl(descA_val, descB_buf[0], descI)
    elif is_block_scaled:
        @Tx.prim_func(tirx=True, check_well_formed=False)
        def impl():
            descI_local: Tx.uint32
            Tx.ptx.tcgen05.encode_instr_descriptor_block_scaled(Tx.address_of(descI_local), C_type, A_type, B_type, SFA_type, SFB_type,  # noqa: E501, F821
                                                               SFA_init_addr, SFB_init_addr,
                                                               M_mma * cta_group, N_mma, MMA_K, a_mn_major, b_mn_major, cta_group)  # noqa: E501
            main_impl(descA_val, descB_buf[0], descI_local)  # noqa: F821
    else:
        @Tx.prim_func(tirx=True, check_well_formed=False)
        def impl():
            descI_local: Tx.uint32
            Tx.ptx.tcgen05.encode_instr_descriptor(Tx.address_of(descI_local), C_type, A_type, B_type,  # noqa: E501, F821
                                                  M_mma * cta_group, N_mma, MMA_K, a_mn_major, b_mn_major, cta_group)  # noqa: E501
            main_impl(descA_val, descB_buf[0], descI_local)  # noqa: F821
    # fmt: on

    return impl


# === Variant: gemm_async/tcgen05 (priority=10) ===
#
# When: gemm_async op at single-thread exec scope on Blackwell (SM100+).
# Requires A in smem (with TMA-compatible swizzle layout) or tmem, B in smem, accum in tmem.
#
# Before (OpCall — regular MMA):
#     Tx.gemm_async(C_tmem[0:64, 0:256], A_smem[0:64, 0:64], B_smem[0:256, 0:64])
#     # A: shared float16, B: shared float16, C: tmem float32
#
# After (encodes instruction descriptor + calls tcgen05.mma):
#     descI_local: uint32
#     Tx.ptx.tcgen05.encode_instr_descriptor(
#         &descI_local, C_type="f32", A_type="f16", B_type="f16",
#         M=64, N=256, MMA_K=64, transA=False, transB=True, cta_group=1)
#     Tx.ptx.tcgen05.mma(descA_buf[0], descB_buf[0], descI_local)
#
# Before (OpCall — block-scaled fp8 MMA):
#     Tx.gemm_async(C_tmem, A_smem, B_smem,
#                   scale_A=SFA_tmem, scale_B=SFB_tmem)
#     # A/B: shared float8_e4m3, SFA/SFB: tmem float8_e8m0fnu
#
# After (adds scale factor descriptors):
#     Tx.ptx.tcgen05.mma(descA, descB, descI,
#                        scale_A=sfA_desc, scale_B=sfB_desc)
#
# Scale factor layout (sf_tmem_layout) must match tcgen05 hardware requirements:
# rows = M or N, sf_mma_k = ceil(MMA_K / sf_block_size), specific TileLayout
# structure with direct_sum atom tiling.
@register_dispatch(
    "gemm_async",
    "cuda",
    variant="tcgen05",
    priority=10,
    when=[
        predicate(
            "single_thread_or_warp",
            lambda op, sctx: (
                single_thread(op, sctx) or sctx.exec_scope.name == "warp",
                f"unsupported exec_scope {sctx.exec_scope}, expected single thread or warp scope",
            ),
        ),
    ],
)
def gemm_async_dispatch_tcgen05(op_call: OpCall, sctx: DispatchContext) -> PrimFunc:
    return gemm_async_tcgen05_impl(op_call, sctx)
