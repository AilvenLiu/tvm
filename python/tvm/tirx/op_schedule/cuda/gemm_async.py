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

import functools
import operator
from typing import List, Optional, Tuple

import tvm
from tvm.script import tirx as Tx
from tvm.tir import PrimFunc
from tvm.runtime import DataType
from tvm.arith.analyzer import Analyzer
from tvm.tir.stmt import OpCall, AllocBuffer, SeqStmt, Evaluate
from tvm.tir.layout import TileLayout, ComposeLayout, TLane, TCol
from tvm.tirx.op_schedule import ScheduleContext, register_dispatch, predicate
from tvm.tirx.operator.op import KernelReplacePoint
from .common import single_thread, validate_gemm_op, get_st_extent
from .copy_async import SwizzleMode, tma_atom_layout, tma_atom_shape


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
    atom = TileLayout(shard=([32, sf_mma_k], [1 @ TLane, 1 @ TCol]), replica=([4], [32 @ TLane]))

    if K == 1:
        outer = TileLayout(shard=([M], [epc @ TCol]))
    else:
        # Pack consecutive ki's within one uint32 TMEM column when possible
        pack_factor = epc // sf_mma_k
        while pack_factor > 1 and K % pack_factor != 0:
            pack_factor //= 2
        if pack_factor > 1:
            K_outer = K // pack_factor
            if K_outer == 1:
                outer = TileLayout(shard=([M, pack_factor], [epc @ TCol, sf_mma_k @ TCol]))
            else:
                outer = TileLayout(
                    shard=([M, K_outer, pack_factor], [epc @ TCol, M * epc @ TCol, sf_mma_k @ TCol])
                )
        else:
            outer = TileLayout(shard=([M, K], [epc @ TCol, M * epc @ TCol]))

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
    assert isinstance(
        slice_layout, TileLayout
    ), f"{name}: sliced layout must be TileLayout, got {type(slice_layout)}"
    M = rows // 32

    assert (
        sf_K_total % sf_mma_k == 0
    ), f"{name}: sf_K_total={sf_K_total} must be divisible by sf_mma_k={sf_mma_k}"
    K = sf_K_total // sf_mma_k

    atom = TileLayout(
        shard=([32, sf_mma_k], [1 @ TLane, 1 @ TCol]),
        replica=([4], [32 @ TLane]),
    )
    # interleaved_shape is the interleaved domain [M, 32, K, sf_mma_k]
    outer = atom.is_direct_sum_right(slice_layout, [M, 32, K, sf_mma_k], [32, sf_mma_k])
    assert outer is not None, f"{name}: layout does not match atom direct_sum outer pattern"


def gemm_async_tcgen05_impl(op_call: OpCall, sctx: ScheduleContext) -> PrimFunc:
    """Schedule an asynchronous GEMM operation using tcgen05.mma (Blackwell Tensor Core).

    Computes C = A @ B (with optional transpose on A/B and accumulation).
    Supports both regular MMA and block-scaled MMA for low-precision dtypes.

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
        sctx: Schedule context (must be single-thread execution scope)

    Returns:
        A PrimFunc implementing the tcgen05 MMA schedule.

    Raises:
        ValueError: If buffer scopes are invalid (C must be tmem, A/B must be shared).
        AssertionError: If shape/layout constraints are not satisfied.
    """
    # Detect block-scaled vs regular by arg count
    is_block_scaled = len(op_call.args) == 8

    C_buffer_region: tvm.tir.BufferRegion = op_call.args[0]
    A_buffer_region: tvm.tir.BufferRegion = op_call.args[1]
    B_buffer_region: tvm.tir.BufferRegion = op_call.args[2]
    C_buffer, A_buffer, B_buffer = (
        C_buffer_region.buffer,
        A_buffer_region.buffer,
        B_buffer_region.buffer,
    )

    C_scope, A_scope, B_scope = C_buffer.scope(), A_buffer.scope(), B_buffer.scope()
    if not (C_scope == "tmem" and A_scope.startswith("shared") and B_scope.startswith("shared")):
        raise ValueError(
            f"tcgen05 schedule expected C_scope=tmem, A_scope=shared, B_scope=shared, got C_scope={C_scope}, A_scope={A_scope}, B_scope={B_scope}"
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
        assert (
            A_type in _BLOCK_SCALED_DTYPES
        ), f"tcgen05 block-scaled schedule expected A_type in {_BLOCK_SCALED_DTYPES}, got {A_type}"
        assert (
            B_type in _BLOCK_SCALED_DTYPES
        ), f"tcgen05 block-scaled schedule expected B_type in {_BLOCK_SCALED_DTYPES}, got {B_type}"
    else:
        assert A_type in [
            "float16",
            "bfloat16",
        ], f"tcgen05 schedule expected A_type=float16 or bfloat16, got {A_type}"
        assert B_type in [
            "float16",
            "bfloat16",
        ], f"tcgen05 schedule expected B_type=float16 or bfloat16, got {B_type}"
    assert (
        A_type == B_type
    ), f"tcgen05 schedule expect A_type and B_type to be the same, got A_type={A_type}, B_type={B_type}"

    # Parse SFA/SFB and transA/transB/accum based on arg layout
    if is_block_scaled:
        SFA_buffer_region, SFB_buffer_region = op_call.args[3:5]
        transA, transB, accum = op_call.args[5:8]
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
        assert (
            SFA_type in _SCALE_FACTOR_DTYPES
        ), f"tcgen05 block-scaled schedule expected SFA_type in {_SCALE_FACTOR_DTYPES}, got {SFA_type}"
        assert (
            SFB_type in _SCALE_FACTOR_DTYPES
        ), f"tcgen05 block-scaled schedule expected SFB_type in {_SCALE_FACTOR_DTYPES}, got {SFB_type}"
        # Compute sf_mma_k from data/SF dtypes and validate layouts
        sfa_sf_mma_k = _compute_sf_mma_k(A_type, SFA_type)
        sfb_sf_mma_k = _compute_sf_mma_k(B_type, SFB_type)
        assert (
            sfa_sf_mma_k == sfb_sf_mma_k
        ), f"SFA and SFB must have same sf_mma_k, got sfa={sfa_sf_mma_k}, sfb={sfb_sf_mma_k}"
        SFA_rows = int(SFA_buffer_region.region[-2].extent)
        SFA_K_total = int(SFA_buffer_region.region[-1].extent)
        SFB_rows = int(SFB_buffer_region.region[-2].extent)
        SFB_K_total = int(SFB_buffer_region.region[-1].extent)
        _validate_sf_tmem_layout(SFA_slice_layout, SFA_rows, SFA_K_total, sfa_sf_mma_k, "SFA")
        _validate_sf_tmem_layout(SFB_slice_layout, SFB_rows, SFB_K_total, sfb_sf_mma_k, "SFB")
    else:
        transA, transB, accum = op_call.args[3:6]

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
    A_slice_tile = (
        A_slice_layout.tile_layout if isinstance(A_slice_layout, ComposeLayout) else A_slice_layout
    )
    B_slice_tile = (
        B_slice_layout.tile_layout if isinstance(B_slice_layout, ComposeLayout) else B_slice_layout
    )

    assert (
        len(C_extent) == 2 and len(A_extent) >= 2 and len(B_extent) >= 2
    ), "Only 2D C, A, B are supported for gemm"
    for buf_name, extent in [("C", C_extent), ("A", A_extent), ("B", B_extent)]:
        assert all(
            analyzer.can_prove_equal(ext, 1) for ext in extent[:-2]
        ), f"tcgen05 schedule expected {buf_name}_extent to be 1 before the last two dimensions"

    M = int(C_extent[-2])
    N = int(C_extent[-1])
    K = int(A_extent[-2] if transA else A_extent[-1])

    # tcgen05 MMA hardware constraints
    MMA_M = 128  # Fixed M dimension for tcgen05 MMA
    # K dimension per MMA iteration depends on A/B dtype
    if A_type == "float4_e2m1fn":
        MMA_K = 64
    elif A_type in ["float8_e4m3fn", "float8_e5m2"]:
        MMA_K = 32
    else:  # float16, bfloat16
        MMA_K = 16
    MMA_N_MIN = 8 if cta_group == 1 else 16  # Minimum N dimension
    MMA_N_MAX = 256  # Maximum N dimension

    assert M == MMA_M, f"tcgen05 schedule expected M={MMA_M}, got {M}"
    assert N >= MMA_N_MIN, f"tcgen05 schedule expected N >= {MMA_N_MIN}, got {N}"
    assert N <= MMA_N_MAX, f"tcgen05 schedule expected N <= {MMA_N_MAX}, got {N}"
    assert N % MMA_N_MIN == 0, f"tcgen05 schedule expected N % {MMA_N_MIN} == 0, got {N}"
    assert K % MMA_K == 0, f"tcgen05 schedule expected K % {MMA_K} == 0, got {K}"

    # Cross-validate A dimensions
    A_M = int(A_extent[-1] if transA else A_extent[-2])
    assert A_M == M, f"tcgen05: A_M={A_M} doesn't match M={M} from C region"

    # Cross-validate K between A and B
    B_K = int(B_extent[-1] if not transB else B_extent[-2])
    assert K == B_K, f"tcgen05: A_K={K} doesn't match B_K={B_K}"

    # Cross-validate B's N with C's N and cta_group
    B_N = int(B_extent[-2] if not transB else B_extent[-1])
    assert (
        B_N * cta_group == N
    ), f"tcgen05: B_N={B_N} * cta_group={cta_group}={B_N * cta_group} doesn't match N={N}"

    # Validate SFA/SFB region shapes
    if is_block_scaled:
        num_ki = K // MMA_K

        assert SFA_rows == M, f"tcgen05: SFA rows={SFA_rows} must equal M={M}"
        assert SFB_rows >= N, f"tcgen05: SFB rows={SFB_rows} must be >= N={N}"
        valid_sfa_K = {sfa_sf_mma_k, sfa_sf_mma_k * num_ki}
        valid_sfb_K = {sfb_sf_mma_k, sfb_sf_mma_k * num_ki}
        assert (
            SFA_K_total in valid_sfa_K
        ), f"tcgen05: SFA K extent={SFA_K_total} must be in {valid_sfa_K}"
        assert (
            SFB_K_total in valid_sfb_K
        ), f"tcgen05: SFB K extent={SFB_K_total} must be in {valid_sfb_K}"

    # Check C's sliced layout: (M, N):(1@TLane, 1@TCol), allow offset
    base = TileLayout(([M, N], [1 @ TLane, 1 @ TCol]))
    expected_c_layout = TileLayout.from_iters(
        base.shard, base.replica, C_slice_layout.offset
    ).canonicalize()
    tvm.ir.assert_structural_equal(C_slice_layout.canonicalize(), expected_c_layout)
    assert C_buffer.allocated_addr is not None
    tmem_addr = C_buffer.allocated_addr[0]
    tmem_offset_32b = C_slice_layout.offset.get(TCol, 0)

    # Check A and B's subregion layout and compute descriptor offsets
    def swizzle_check(slice_layout, dtype) -> Tuple[SwizzleMode, int, int]:
        """Check subregion layout compatibility and compute descriptor offsets.

        Uses the sliced (subregion) layout to find a compatible swizzle mode.
        The sliced layout preserves the buffer's swizzle/stride structure while
        capturing the subregion's extent and offset.

        Returns:
            Tuple of (swizzle_mode, ldo, sdo) where:
                - ldo: leading dimension offset (stride along last dimension)
                - sdo: striding dimension offset (stride along second-to-last dimension)

        Raises:
            ValueError: If no compatible swizzle mode is found.
        """
        slice_tile = (
            slice_layout.tile_layout if isinstance(slice_layout, ComposeLayout) else slice_layout
        )
        sub_shape = [int(s.extent) for s in slice_tile.shard]
        for mode in (
            SwizzleMode.SWIZZLE_128B_ATOM,
            SwizzleMode.SWIZZLE_64B_ATOM,
            SwizzleMode.SWIZZLE_32B_ATOM,
        ):
            swizzle_atom = tma_atom_layout(dtype, mode)
            atom_shape = tma_atom_shape(dtype, mode, sub_shape)
            atom_size = functools.reduce(operator.mul, atom_shape, 1)
            tiler = swizzle_atom.is_tile_inner(slice_layout, sub_shape, atom_shape)
            if tiler is not None:
                tiler_shape = [s // a for s, a in zip(sub_shape, atom_shape)]
                tiler_grouped, seps = tiler.canonicalize().group(tiler_shape)
                assert seps[-3] == seps[-1] - 2
                assert seps[-2] == seps[-1] - 1
                # ldo: leading dimension offset, sdo: striding dimension offset
                # These are used in matrix descriptor encoding for tcgen05 MMA
                elem_per_128b = 128 // tvm.DataType(dtype).bits
                ldo = (tiler_grouped.shard[-1].stride * atom_size) // elem_per_128b
                sdo = (tiler_grouped.shard[-2].stride * atom_size) // elem_per_128b
                return mode, ldo, sdo
        raise ValueError(
            f"No compatible swizzle mode found for dtype {dtype} "
            f"with subregion shape {sub_shape}"
        )

    A_swizzle_mode, A_ldo, A_sdo = swizzle_check(A_slice_layout, A_type)
    B_swizzle_mode, B_ldo, B_sdo = swizzle_check(B_slice_layout, B_type)

    # Convert accum to TIR bool outside the macro (TIR AST evaluator doesn't
    # support short-circuit evaluation, so accum.dtype inside macro would fail
    # when accum is a Python bool).
    if isinstance(accum, bool):
        accum_expr = tvm.tir.const(int(accum), "bool")
    elif isinstance(accum, tvm.tir.PrimExpr) and accum.dtype != "bool":
        accum_expr = tvm.tir.Cast("bool", accum)
    else:
        accum_expr = accum

    # SmemDescriptor optimization: encode once at buffer def, add_16B_offset per ki
    def smem_desc_add_16B_offset(desc_cell, offset):
        func_name = "tvm_builtin_smem_desc_add_16B_offset"
        source_code = f"""
__forceinline__ __device__ uint64_t {func_name}(uint64_t desc_base, int32_t offset) {{
    SmemDescriptor desc;
    desc.desc_ = desc_base;
    desc.lo += static_cast<uint32_t>(offset);
    return desc.desc_;
}}
"""
        return Tx.cuda.func_call(
            func_name, desc_cell, offset, source_code=source_code, return_type="uint64"
        )

    # 16B element count for descriptor offset computation
    A_elem_per_16B = 128 // DataType(A_type).bits
    B_elem_per_16B = 128 // DataType(B_type).bits

    # Allocate descriptor cells and encode once, right after A/B buffer defs.
    # Uses add_post_buffer_def_stmt with kernel_replace_point so the DeclBuffer
    # for descA/descB wraps the continuation (everything after A/B's DeclBuffer).
    A_base = [0] * len(A_buffer.shape)
    B_base = [0] * len(B_buffer.shape)
    descA_buf = tvm.tir.decl_buffer((1,), "uint64", name="descA", scope="local")
    descB_buf = tvm.tir.decl_buffer((1,), "uint64", name="descB", scope="local")
    krp = KernelReplacePoint(workspace={}, config={})

    def _make_desc_wrap(desc_buf, smem_buf, base, ldo, sdo, swizzle_val):
        """Build: AllocBuffer(desc, { encode(desc, smem); krp })"""
        encode_call = tvm.tir.call_intrin(
            "",
            "tir.ptx_tcgen05_encode_matrix_descriptor",
            tvm.tir.address_of(desc_buf[0]),
            smem_buf.ptr_to(base),
            ldo,
            sdo,
            swizzle_val,
        )
        return AllocBuffer(desc_buf, SeqStmt([Evaluate(encode_call), krp]))

    wrap_A = _make_desc_wrap(descA_buf, A_buffer, A_base, A_ldo, A_sdo, A_swizzle_mode.value)
    wrap_B = _make_desc_wrap(descB_buf, B_buffer, B_base, B_ldo, B_sdo, B_swizzle_mode.value)
    sctx.add_post_buffer_def_stmt(A_buffer, wrap_A)
    sctx.add_post_buffer_def_stmt(B_buffer, wrap_B)

    if is_block_scaled:
        # Compute per-ki SF element steps from region extents
        sfa_elems_per_ki = SFA_K_total // num_ki if num_ki > 0 else 0
        sfb_elems_per_ki = SFB_K_total // num_ki if num_ki > 0 else 0

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

        # fmt: off
        @Tx.macro
        def main_impl(descA_in, descB_in, descI_in):
            for ki in Tx.serial(tvm.tir.floordiv(K, MMA_K)):
                A_ki_linear = Tx.meta_var(ki * MMA_K * A_extent[-1] if transA else ki * MMA_K)
                B_ki_linear = Tx.meta_var(ki * MMA_K * B_extent[-1] if transB else ki * MMA_K)
                A_offset = Tx.meta_var(tvm.tir.floordiv(A_slice_tile.apply(A_ki_linear)["m"], A_elem_per_16B))
                B_offset = Tx.meta_var(tvm.tir.floordiv(B_slice_tile.apply(B_ki_linear)["m"], B_elem_per_16B))
                descA_val = Tx.meta_var(smem_desc_add_16B_offset(descA_in, A_offset))
                descB_val = Tx.meta_var(smem_desc_add_16B_offset(descB_in, B_offset))
                should_accum = Tx.meta_var(tvm.tir.any(ki != 0, accum_expr))
                sfa_k_pos = Tx.meta_var(ki * sfa_elems_per_ki)
                sfb_k_pos = Tx.meta_var(ki * sfb_elems_per_ki)
                # apply(k_pos)["TCol"] at row 0 gives physical TCol offset
                sfa_tcol = Tx.meta_var(SFA_slice_layout.apply(sfa_k_pos).get("TCol", 0))
                sfb_tcol = Tx.meta_var(SFB_slice_layout.apply(sfb_k_pos).get("TCol", 0))
                sfa_addr = Tx.meta_var(sfa_base + tvm.tir.floordiv(sfa_tcol, SFA_elem_per_col))
                sfb_addr = Tx.meta_var(sfb_base + tvm.tir.floordiv(sfb_tcol, SFB_elem_per_col))
                if needs_sf_id:
                    sf_id = Tx.meta_var(analyzer.simplify(tvm.tir.floormod(sfa_tcol, SFA_elem_per_col)))
                    Tx.cuda.runtime_instr_desc(Tx.address_of(descI_in), sf_id)
                Tx.ptx.tcgen05.mma.block_scale(C_type, A_type, B_type, SFA_type, SFB_type,
                                              Tx.cuda.get_tmem_addr(tmem_addr, 0, tmem_offset_32b),
                                              descA_val, descB_val,
                                              sfa_addr, sfb_addr,
                                              descI_in, False, cta_group, should_accum)
        # fmt: on
    else:
        # fmt: off
        @Tx.macro
        def main_impl(descA_in, descB_in, descI_in):
            for ki in Tx.serial(tvm.tir.floordiv(K, MMA_K)):
                A_ki_linear = Tx.meta_var(ki * MMA_K * A_extent[-1] if transA else ki * MMA_K)
                B_ki_linear = Tx.meta_var(ki * MMA_K * B_extent[-1] if transB else ki * MMA_K)
                A_offset = Tx.meta_var(tvm.tir.floordiv(A_slice_tile.apply(A_ki_linear)["m"], A_elem_per_16B))
                B_offset = Tx.meta_var(tvm.tir.floordiv(B_slice_tile.apply(B_ki_linear)["m"], B_elem_per_16B))
                descA_val = Tx.meta_var(smem_desc_add_16B_offset(descA_in, A_offset))
                descB_val = Tx.meta_var(smem_desc_add_16B_offset(descB_in, B_offset))
                should_accum = Tx.meta_var(tvm.tir.any(ki != 0, accum_expr))
                Tx.ptx.tcgen05.mma("float32", A_type, B_type, Tx.cuda.get_tmem_addr(tmem_addr, 0, tmem_offset_32b),
                                  descA_val, descB_val, descI_in, False, cta_group, should_accum)
        # fmt: on

    if descI is not None:
        # fmt: off
        @Tx.prim_func(tirx=True, check_well_formed=False)
        def impl():
            main_impl(descA_buf[0], descB_buf[0], descI)
        # fmt: on
    elif is_block_scaled:
        # fmt: off
        @Tx.prim_func(tirx=True, check_well_formed=False)
        def impl():
            descI_local = Tx.local_cell("uint32")
            Tx.ptx.tcgen05.encode_instr_descriptor_block_scaled(Tx.address_of(descI_local), C_type, A_type, B_type, SFA_type, SFB_type,
                                                               SFA_init_addr, SFB_init_addr,
                                                               M * cta_group, N, MMA_K, transA, transB, cta_group)
            main_impl(descA_buf[0], descB_buf[0], descI_local)
        # fmt: on
    else:
        # fmt: off
        @Tx.prim_func(tirx=True, check_well_formed=False)
        def impl():
            descI_local = Tx.local_cell("uint32")
            Tx.ptx.tcgen05.encode_instr_descriptor(Tx.address_of(descI_local), C_type, A_type, B_type,
                                                  M * cta_group, N, MMA_K, transA, transB, cta_group)
            main_impl(descA_buf[0], descB_buf[0], descI_local)
        # fmt: on

    return impl


@register_dispatch(
    "gemm_async",
    "cuda",
    variant="tcgen05",
    priority=10,
    when=[
        predicate(
            "single_thread",
            lambda op, sctx: (
                single_thread(op, sctx),
                f"unsupported exec_scope {sctx.exec_scope}, expected single thread",
            ),
        ),
    ],
)
def gemm_async_dispatch_tcgen05(op_call: OpCall, sctx: ScheduleContext) -> PrimFunc:
    return gemm_async_tcgen05_impl(op_call, sctx)
