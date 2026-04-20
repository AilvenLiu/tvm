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
# pylint: disable=redefined-builtin, invalid-name, too-many-arguments, too-many-locals, line-too-long
"""PTX tcgen05 operations (Blackwell tensor memory, MMA).

Simple fence/wait/shift/alloc/dealloc ops are declared via ``ptx_intrinsic(...)``
at the top. Ops that combine dim/shape/dtype-parameterized register arrays,
descriptor bitfield assembly, or optional multicast/cta_group/scale-factor
operands (ld/st/cp/commit/mma family, descriptor encoders) are hand-written
below and registered via ``@register_codegen(...)``.
"""

import tvm
from tvm.tirx.op import cuda_func_call

from .._schema import IntAttr, Operand, ptx_intrinsic
from .registry import register_codegen
from .types import PTXDataType
from .utils import parse_str, validate_cta_group, validate_power_of_two_range

# =============================================================================
# Schema-declared ops.
# =============================================================================

# -----------------------------------------------------------------------------
# tcgen05.fence — thread-sync ordering.
# Python API:
#     Tx.ptx.tcgen05.fence.before_thread_sync()
#     Tx.ptx.tcgen05.fence.after_thread_sync()
# -----------------------------------------------------------------------------
ptx_intrinsic(
    op_name="ptx_tcgen05_fence_before_thread_sync",
    ptx_template="tcgen05.fence::before_thread_sync;",
    helper_name="tvm_builtin_ptx_tcgen05_fence_before_thread_sync",
)
ptx_intrinsic(
    op_name="ptx_tcgen05_fence_after_thread_sync",
    ptx_template="tcgen05.fence::after_thread_sync;",
    helper_name="tvm_builtin_ptx_tcgen05_fence_after_thread_sync",
)


# -----------------------------------------------------------------------------
# tcgen05.wait — wait for prior ld/st completion.
# Python API:
#     Tx.ptx.tcgen05.wait.ld()
#     Tx.ptx.tcgen05.wait.st()
# -----------------------------------------------------------------------------
ptx_intrinsic(
    op_name="ptx_tcgen05_wait_ld",
    ptx_template="tcgen05.wait::ld.sync.aligned;",
    helper_name="tvm_builtin_ptx_tcgen05_wait_ld",
)
ptx_intrinsic(
    op_name="ptx_tcgen05_wait_st",
    ptx_template="tcgen05.wait::st.sync.aligned;",
    helper_name="tvm_builtin_ptx_tcgen05_wait_st",
)


# -----------------------------------------------------------------------------
# tcgen05.shift — shift TMEM rows down.
# Python API:
#     Tx.ptx.tcgen05.shift(taddr, cta_group)
# taddr: TMEM address of matrix; cta_group ∈ {1, 2}.
# -----------------------------------------------------------------------------
ptx_intrinsic(
    op_name="ptx_tcgen05_shift",
    operands=[Operand("taddr", c_type="uint32_t")],
    attrs=[IntAttr("cta_group", choices=(1, 2))],
    ptx_template="tcgen05.shift.cta_group::{cta_group}.down [%0];",
    helper_name_template="ptx_tcgen05_shift_cta_group_{cta_group}",
)


# -----------------------------------------------------------------------------
# tcgen05.relinquish_alloc_permit — release TMEM allocation permit.
# Python API:
#     Tx.ptx.tcgen05.relinquish_alloc_permit(cta_group)
# cta_group ∈ {1, 2}.
# -----------------------------------------------------------------------------
ptx_intrinsic(
    op_name="ptx_tcgen05_relinquish_alloc_permit",
    attrs=[IntAttr("n_cta_group", choices=(1, 2))],
    ptx_template="tcgen05.relinquish_alloc_permit.cta_group::{n_cta_group}.sync.aligned;",
    helper_name_template="tvm_builtin_ptx_tcgen05_relinquish_alloc_permit_cta_group_{n_cta_group}",
)


# -----------------------------------------------------------------------------
# tcgen05.alloc / dealloc — request/release TMEM columns.
# Python API:
#     Tx.ptx.tcgen05.alloc(dst_shared_ptr, n_cols, cta_group)
#     Tx.ptx.tcgen05.dealloc(taddr, n_cols, cta_group)
# n_cols: number of 32-bit TMEM columns to allocate/free (runtime int).
# -----------------------------------------------------------------------------
ptx_intrinsic(
    op_name="ptx_tcgen05_alloc",
    operands=[
        Operand("dst", c_type="void*", cvta_to_shared=True),
        Operand("nCols", c_type="int"),
    ],
    attrs=[IntAttr("n_cta_group", choices=(1, 2))],
    ptx_template="tcgen05.alloc.cta_group::{n_cta_group}.sync.aligned.shared::cta.b32 [%0], %1;",
    helper_name_template="tvm_builtin_ptx_tcgen05_alloc_cta_group_{n_cta_group}",
)
ptx_intrinsic(
    op_name="ptx_tcgen05_dealloc",
    operands=[
        Operand("taddr", c_type="uint32_t"),
        Operand("nCols", c_type="int"),
    ],
    attrs=[IntAttr("n_cta_group", choices=(1, 2))],
    ptx_template="tcgen05.dealloc.cta_group::{n_cta_group}.sync.aligned.b32 %0, %1;",
    helper_name_template="tvm_builtin_ptx_tcgen05_dealloc_cta_group_{n_cta_group}",
)


# =============================================================================
# Hand-written ops.
# =============================================================================


@register_codegen("ptx_tcgen05_ld")
def codegen_ptx_tcgen05_ld(src_addr, row_offset, col_offset, shape, num, pack, *regs):
    shape = parse_str(shape)
    num = validate_power_of_two_range(num, 1, 128, "repeat factor of ptx_tcgen05_ld")
    pack = bool(pack)

    if shape in ["16x32bx2", "16x64b", "32x32b"]:
        expected_n_regs = num
    elif shape == "16x128b":
        if num > 64:
            raise ValueError(
                "The repeat factor of ptx_tcgen05_ld for shape 16x128b is invalid, "
                f"expect a value within range [1, 64], got {num}"
            )
        expected_n_regs = 2 * num
    elif shape == "16x256b":
        if num > 32:
            raise ValueError(
                "The repeat factor of ptx_tcgen05_ld for shape 16x256b is invalid, "
                f"expect a value within range [1, 32], got {num}"
            )
        expected_n_regs = 4 * num
    else:
        raise ValueError(
            "The input shape of ptx_tcgen05_ld is invalid, expect one of [16x32bx2, 16x64b, "
            f"32x32b, 16x128b, 16x256b], got {shape}"
        )

    if len(regs) != expected_n_regs:
        raise ValueError(
            "The number of arguments for ptx_tcgen05_ld is incorrect, expected "
            f"{6 + expected_n_regs} total args (meaning {expected_n_regs} register args), "
            f"but got {len(regs)} register args."
        )

    reg_args = ", ".join([f"void* reg{i}" for i in range(len(regs))])
    regs_placeholder = ", ".join([f"%{i}" for i in range(len(regs))])
    src_placeholder = str(len(regs))

    imm_arg = ""
    if shape == "16x32bx2":
        imm = 2 * num if pack else num
        imm_arg = f", {imm}"

    reg_operands = ", ".join([f'"=r"(*(uint32_t*)reg{i})' for i in range(len(regs))])
    pack_str = ".pack::16b" if pack else ""

    func_name = "tvm_builtin_ptx_tcgen05_ld_" + shape + "_x" + str(num) + ("_pack" if pack else "")
    source_code = f"""
__forceinline__ __device__ void {func_name}(uint32_t src_addr, uint32_t row_offset, uint32_t col_offset, {reg_args}) {{
    asm volatile(
        "tcgen05.ld.sync.aligned.{shape}.x{num}{pack_str}.b32 "
        "{{{regs_placeholder}}}, "
        "[%{src_placeholder}]{imm_arg};\\n"
        :  {reg_operands}
        :  "r"(get_tmem_addr(src_addr, row_offset, col_offset))
        :
    );
}}
"""  # noqa: E501
    regs = [tvm.tirx.address_of(reg) for reg in regs]

    return cuda_func_call(
        func_name,
        src_addr,
        row_offset,
        col_offset,
        *regs,
        source_code=source_code,
    ), ["get_tmem_addr"]


@register_codegen("ptx_tcgen05_st")
def codegen_ptx_tcgen05_st(dst_addr, row_offset, col_offset, shape, num, unpack, *regs):
    shape = parse_str(shape)
    num = validate_power_of_two_range(num, 1, 128, "repeat factor of ptx_tcgen05_st")
    unpack = bool(unpack)

    if shape in ["16x32bx2", "16x64b", "32x32b"]:
        expected_n_regs = num
    elif shape == "16x128b":
        if num > 64:
            raise ValueError(
                "The repeat factor of ptx_tcgen05_st for shape 16x128b is invalid, "
                f"expect a value within range [1, 64], got {num}"
            )
        expected_n_regs = 2 * num
    elif shape == "16x256b":
        if num > 32:
            raise ValueError(
                "The repeat factor of ptx_tcgen05_st for shape 16x256b is invalid, "
                f"expect a value within range [1, 32], got {num}"
            )
        expected_n_regs = 4 * num
    else:
        raise ValueError(
            "The input shape of ptx_tcgen05_st is invalid, expect one of [16x32bx2, 16x64b, "
            f"32x32b, 16x128b, 16x256b], got {shape}"
        )

    if len(regs) != expected_n_regs:
        raise ValueError(
            "The number of arguments for ptx_tcgen05_st is incorrect, expected "
            f"{6 + expected_n_regs} total args (meaning {expected_n_regs} register args), "
            f"but got {len(regs)} register args."
        )

    reg_args = ", ".join([f"void* reg{i}" for i in range(len(regs))])
    regs_placeholder = ", ".join([f"%{i + 1}" for i in range(len(regs))])

    imm_arg = ""
    if shape == "16x32bx2":
        imm = 2 * num if unpack else num
        imm_arg = f", {imm}"

    reg_operands = ", ".join([f'"r"(*(uint32_t*)reg{i})' for i in range(len(regs))])
    unpack_str = ".unpack::16b" if unpack else ""

    func_name = (
        "tvm_builtin_ptx_tcgen05_st_" + shape + "_x" + str(num) + ("_unpack" if unpack else "")
    )
    source_code = f"""
__forceinline__ __device__ void {func_name}(uint32_t dst_addr, uint32_t row_offset, uint32_t col_offset, {reg_args}) {{
    asm volatile(
        "tcgen05.st.sync.aligned.{shape}.x{num}{unpack_str}.b32 "
        "[%0]{imm_arg}, "
        "{{{regs_placeholder}}};\\n"
        :
        :  "r"(get_tmem_addr(dst_addr, row_offset, col_offset)), {reg_operands}
    );
}}
"""  # noqa: E501
    regs = [tvm.tirx.address_of(reg) for reg in regs]
    return cuda_func_call(
        func_name,
        dst_addr,
        row_offset,
        col_offset,
        *regs,
        source_code=source_code,
    ), ["get_tmem_addr"]


# -----------------------------------------------------------------------------
# tcgen05.wait — schema-driven.
@register_codegen("ptx_tcgen05_encode_matrix_descriptor")
def codegen_ptx_tcgen05_encode_matrix_descriptor(desc, addr, ldo, sdo, swizzle):
    valid_swizzle_modes = [0, 1, 2, 3, 4]
    swizzle = int(swizzle)
    if swizzle not in valid_swizzle_modes:
        raise ValueError(
            f"Invalid swizzle mode. Expected a value in {valid_swizzle_modes}, but got {swizzle}"
        )

    func_name = "tvm_builtin_ptx_tcgen05_encode_matrix_descriptor"
    source_code = f"""
__forceinline__ __device__ void {func_name}(uint64_t* desc, void* addr, int ldo, int sdo, int swizzle) {{
  SmemDescriptor _desc;

  _desc.version_ = 1;
  _desc.lbo_mode_ = 0;

  switch (swizzle) {{
    case 0: _desc.layout_type_ = uint8_t(0); break; // No swizzle
    case 1: _desc.layout_type_ = uint8_t(6); break; // 32B swizzle
    case 2: _desc.layout_type_ = uint8_t(4); break; // 64B swizzle
    case 3: _desc.layout_type_ = uint8_t(2); break; // 128B swizzle
    case 4: _desc.layout_type_ = uint8_t(1); break; // 128B_base32B swizzle
  }}

  uint32_t start_address = __cvta_generic_to_shared(addr);
  _desc.start_address_ = static_cast<uint16_t>(start_address >> 4);

  constexpr uint8_t base_offset = 0;
  _desc.base_offset_ = base_offset;

  _desc.stride_byte_offset_  = static_cast<uint32_t>(sdo);
  _desc.leading_byte_offset_ = static_cast<uint32_t>(ldo);

  *desc = (uint64_t)_desc;
}}
"""  # noqa: E501
    return cuda_func_call(func_name, desc, addr, ldo, sdo, swizzle, source_code=source_code), [
        "smem_descriptor"
    ]


# Dtype sets used to classify tcgen05 MMA variants.
_FP8_FAMILY = frozenset(
    {
        PTXDataType.FLOAT8_E4M3FN,
        PTXDataType.FLOAT8_E4M3FNUZ,
        PTXDataType.FLOAT8_E5M2,
        PTXDataType.FLOAT6_E2M3FN,
        PTXDataType.FLOAT6_E3M2FN,
        PTXDataType.FLOAT4_E2M1FN,
    }
)
_E8M0 = frozenset({PTXDataType.FLOAT8_E8M0FNU})
_E4M3 = frozenset({PTXDataType.FLOAT8_E4M3FN, PTXDataType.FLOAT8_E4M3FNUZ})


# Rules for classifying (d_dtype, a_dtype, b_dtype [, sfa_dtype, sfb_dtype]) into
# one of tcgen05.mma's kind tags. First matching rule wins; rules are disjoint.
# sf_required=True means a non-empty sfa_dtype AND sfb_dtype must be provided.
_TCGEN05_MMA_RULES: tuple[tuple, ...] = (
    # (kind, d_in, a_in, b_in, sf_required, sfa_in, sfb_in)
    (
        "f16",
        frozenset({PTXDataType.FLOAT16}),
        frozenset({PTXDataType.FLOAT16}),
        frozenset({PTXDataType.FLOAT16}),
        False,
        None,
        None,
    ),
    (
        "f16",
        frozenset({PTXDataType.FLOAT32}),
        frozenset({PTXDataType.FLOAT16, PTXDataType.BFLOAT16}),
        frozenset({PTXDataType.FLOAT16, PTXDataType.BFLOAT16}),
        False,
        None,
        None,
    ),
    (
        "tf32",
        frozenset({PTXDataType.FLOAT32}),
        frozenset({PTXDataType.TENSOR_FLOAT32}),
        frozenset({PTXDataType.TENSOR_FLOAT32}),
        False,
        None,
        None,
    ),
    (
        "i8",
        frozenset({PTXDataType.INT32}),
        frozenset({PTXDataType.INT8, PTXDataType.UINT8}),
        frozenset({PTXDataType.INT8, PTXDataType.UINT8}),
        False,
        None,
        None,
    ),
    (
        "f8f6f4",
        frozenset({PTXDataType.FLOAT32, PTXDataType.FLOAT16}),
        _FP8_FAMILY,
        _FP8_FAMILY,
        False,
        None,
        None,
    ),
    (
        "mxf4",
        frozenset({PTXDataType.FLOAT32}),
        frozenset({PTXDataType.FLOAT4_E2M1FN}),
        frozenset({PTXDataType.FLOAT4_E2M1FN}),
        True,
        _E8M0,
        _E8M0,
    ),
    (
        "mxf4nvf4",
        frozenset({PTXDataType.FLOAT32}),
        frozenset({PTXDataType.FLOAT4_E2M1FN}),
        frozenset({PTXDataType.FLOAT4_E2M1FN}),
        True,
        _E4M3,
        _E4M3,
    ),
    ("mxf8f6f4", frozenset({PTXDataType.FLOAT32}), _FP8_FAMILY, _FP8_FAMILY, True, _E8M0, _E8M0),
)


def _get_tcgen05_mma_kind(
    d_dtype: str, a_dtype: str, b_dtype: str, sfa_dtype: str = "", sfb_dtype: str = ""
) -> str:
    """Classify an MMA dtype tuple into one of tcgen05.mma's kind tags.

    Reads from ``_TCGEN05_MMA_RULES``; first match wins. Raises ``ValueError`` if
    no rule matches.
    """
    d = PTXDataType.from_string(d_dtype)
    a = PTXDataType.from_string(a_dtype)
    b = PTXDataType.from_string(b_dtype)
    has_sf = bool(sfa_dtype) and bool(sfb_dtype)
    sfa = PTXDataType.from_string(sfa_dtype) if sfa_dtype else None
    sfb = PTXDataType.from_string(sfb_dtype) if sfb_dtype else None

    # mxf4/mxf4nvf4 take precedence over mxf8f6f4 for fp4xfp4; rule order handles that.
    for kind, d_in, a_in, b_in, sf_required, sfa_in, sfb_in in _TCGEN05_MMA_RULES:
        if d not in d_in or a not in a_in or b not in b_in:
            continue
        if sf_required != has_sf:
            continue
        if sf_required and (sfa not in sfa_in or sfb not in sfb_in):
            continue
        return kind

    raise ValueError(
        f"Invalid multiplicand data types for Tcgen05 MMA, check failed for d: {d_dtype}, "
        f"a: {a_dtype}, b: {b_dtype}, scale_a: {sfa_dtype}, scale_b: {sfb_dtype}"
    )


# Shape constraints for tcgen05 MMA.
# Each entry: (kinds, cta_group, m_to_n_step, extra_ms_for_n, k_dense, k_sparse).
#   - kinds: which MMA kinds this entry applies to
#   - cta_group: 1 or 2
#   - m_to_n_step: {valid_m: n_step} - N must be in range(n_step, 257, n_step)
#   - extra_ns_for_any_m: extra N values allowed (e.g. {8, 24} for i8 cta_group=1)
#   - k_dense / k_sparse: required K value
_TCGEN05_MMA_SHAPE_RULES: tuple[tuple, ...] = (
    # (kinds_set, cta_group, m_to_n_step, extra_ns, k_dense, k_sparse)
    (frozenset({"f16", "tf32", "f8f6f4"}), 1, {64: 8, 128: 16}, frozenset(), None, None),
    (frozenset({"f16", "tf32", "f8f6f4"}), 2, {128: 32, 256: 32}, frozenset(), None, None),
    (frozenset({"i8"}), 1, {64: 16, 128: 16}, frozenset({8, 24}), 32, 64),
    (frozenset({"i8"}), 2, {128: 32, 256: 32}, frozenset(), 32, 64),
    (frozenset({"mxf8f6f4", "mxf4", "mxf4nvf4"}), 1, {128: 8}, frozenset(), None, None),
    (frozenset({"mxf8f6f4", "mxf4", "mxf4nvf4"}), 2, {128: 16, 256: 16}, frozenset(), None, None),
)

# K values are kind-dependent (dense/sparse pair).
_TCGEN05_MMA_K: dict[str, tuple[int, int]] = {
    "f16": (16, 32),
    "tf32": (8, 16),
    "f8f6f4": (32, 64),
    "i8": (32, 64),
    "mxf8f6f4": (32, 64),
    "mxf4": (64, 128),
    "mxf4nvf4": (64, 128),
}


def _check_tcgen05_mma_matrix_shape(
    kind: str, cta_group: int, m: int, n: int, k: int, is_sparse: bool
) -> bool:
    err = (
        f"Invalid matrix shape for Tcgen05 MMA, check failed for kind: {kind}, "
        f"is_sparse: {is_sparse}, cta_group: {cta_group}, M: {m}, N: {n}, K: {k}"
    )

    # Find the matching shape rule.
    for kinds, cg, m_to_n_step, extra_ns, _, _ in _TCGEN05_MMA_SHAPE_RULES:
        if kind not in kinds or cg != cta_group:
            continue
        # Extra constraint: mxf* with cta_group=2 and is_sparse requires m=256.
        if kind in {"mxf8f6f4", "mxf4", "mxf4nvf4"} and cta_group == 2 and is_sparse and m != 256:
            raise ValueError(err)
        if m not in m_to_n_step:
            raise ValueError(err)
        n_step = m_to_n_step[m]
        if n not in extra_ns and not (n_step <= n <= 256 and n % n_step == 0):
            raise ValueError(err)
        break
    else:
        raise ValueError(err)

    # K must match the kind-specific (dense, sparse) pair.
    k_pair = _TCGEN05_MMA_K.get(kind)
    if k_pair is None:
        raise ValueError(err)
    k_dense, k_sparse = k_pair
    expected_k = k_sparse if is_sparse else k_dense
    if k != expected_k:
        raise ValueError(err)

    return True


@register_codegen("ptx_tcgen05_encode_instr_descriptor")
def codegen_ptx_tcgen05_encode_instr_descriptor(
    desc,
    d_dtype,
    a_dtype,
    b_dtype,
    M,
    N,
    K,
    trans_a,
    trans_b,
    n_cta_group,
    neg_a,
    neg_b,
    sat_d,
    is_sparse,
):
    a_dtype = parse_str(a_dtype)
    b_dtype = parse_str(b_dtype)
    d_dtype = parse_str(d_dtype)
    M = int(M)
    N = int(N)
    K = int(K)
    n_cta_group = validate_cta_group(n_cta_group)
    trans_a = bool(trans_a)
    trans_b = bool(trans_b)
    neg_a = bool(neg_a)
    neg_b = bool(neg_b)
    sat_d = bool(sat_d)
    is_sparse = bool(is_sparse)

    kind = _get_tcgen05_mma_kind(d_dtype, a_dtype, b_dtype)
    if kind not in ["f16", "tf32", "f8f6f4", "i8"]:
        raise ValueError(
            f"Check failed for Data Type Kind. d_dtype: {d_dtype}, a_dtype: {a_dtype}, b_dtype: {b_dtype}"  # noqa: E501
        )

    if not _check_tcgen05_mma_matrix_shape(kind, n_cta_group, M, N, K, is_sparse):
        raise ValueError(f"Invalid matrix shape ({M}, {N}, {K}) for kind '{kind}'")

    format_map = {
        PTXDataType.FLOAT16: 0,
        PTXDataType.BFLOAT16: 1,
        PTXDataType.TENSOR_FLOAT32: 2,
        PTXDataType.FLOAT8_E4M3FN: 0,
        PTXDataType.FLOAT8_E4M3FNUZ: 0,
        PTXDataType.FLOAT8_E5M2: 1,
        PTXDataType.FLOAT6_E2M3FN: 3,
        PTXDataType.FLOAT6_E3M2FN: 4,
        PTXDataType.FLOAT4_E2M1FN: 5,
        PTXDataType.UINT8: 0,
        PTXDataType.INT8: 1,
        PTXDataType.FLOAT32: 1,
        PTXDataType.INT32: 2,
    }
    dtype = PTXDataType.from_string(d_dtype)
    atype = PTXDataType.from_string(a_dtype)
    btype = PTXDataType.from_string(b_dtype)

    d_format = format_map[dtype]
    a_format = format_map[atype]
    b_format = format_map[btype]

    valid_dtypes_for_trans = {
        PTXDataType.FLOAT8_E4M3FN,
        PTXDataType.FLOAT8_E4M3FNUZ,
        PTXDataType.FLOAT8_E5M2,
        PTXDataType.INT8,
        PTXDataType.UINT8,
        PTXDataType.FLOAT16,
        PTXDataType.BFLOAT16,
        PTXDataType.TENSOR_FLOAT32,
    }
    if trans_a and atype not in valid_dtypes_for_trans:
        raise ValueError(f"Invalid a_dtype for transpose: {a_dtype}")
    if trans_b and btype not in valid_dtypes_for_trans:
        raise ValueError(f"Invalid b_dtype for transpose: {b_dtype}")
    if (neg_a or neg_b) and kind not in ["f16", "tf32", "f8f6f4"]:
        raise ValueError(f"Invalid kind for negate: {kind}")
    if sat_d and kind != "i8":
        raise ValueError(f"Invalid kind for saturate: {kind}")

    func_name = "ptx_tcgen05_encode_instr_descriptor"
    source_code = f"""
__forceinline__ __device__ void {func_name}(uint32_t* desc, int M, int N, int d_format,
                                            int a_format, int b_format, bool trans_a, bool trans_b,
                                            bool neg_a, bool neg_b, bool sat_d, bool is_sparse) {{
  InstrDescriptor _desc;

  _desc.a_format_ = uint8_t(a_format);
  _desc.b_format_ = uint8_t(b_format);
  _desc.c_format_ = uint8_t(d_format);

  _desc.m_dim_ = (M >> 4);
  _desc.n_dim_ = (N >> 3);

  _desc.a_major_ = static_cast<uint8_t>(trans_a);
  _desc.b_major_ = static_cast<uint8_t>(trans_b);

  _desc.a_negate_ = static_cast<uint8_t>(neg_a);
  _desc.b_negate_ = static_cast<uint8_t>(neg_b);
  _desc.saturate_ = static_cast<uint8_t>(sat_d);

  _desc.sparse_flag_ = is_sparse;
  _desc.sparse_id2_  = 0;                          // should modify in sparse case

  _desc.max_shift_ = uint8_t(0);                   // WS not used

  *desc = (uint32_t)_desc;
}}
"""
    return cuda_func_call(
        func_name,
        desc,
        M,
        N,
        d_format,
        a_format,
        b_format,
        trans_a,
        trans_b,
        neg_a,
        neg_b,
        sat_d,
        is_sparse,
        source_code=source_code,
    ), ["instr_descriptor"]


@register_codegen("ptx_tcgen05_encode_instr_descriptor_block_scaled")
def codegen_ptx_tcgen05_encode_instr_descriptor_block_scaled(
    desc,
    d_dtype,
    a_dtype,
    b_dtype,
    sfa_dtype,
    sfb_dtype,
    sfa_tmem_addr,
    sfb_tmem_addr,
    M,
    N,
    K,
    trans_a,
    trans_b,
    n_cta_group,
    neg_a,
    neg_b,
    is_sparse,
):
    a_dtype = parse_str(a_dtype)
    b_dtype = parse_str(b_dtype)
    d_dtype = parse_str(d_dtype)
    sfa_dtype = parse_str(sfa_dtype)
    sfb_dtype = parse_str(sfb_dtype)
    M = int(M)
    N = int(N)
    K = int(K)
    n_cta_group = validate_cta_group(n_cta_group)
    trans_a = bool(trans_a)
    trans_b = bool(trans_b)
    neg_a = bool(neg_a)
    neg_b = bool(neg_b)
    is_sparse = bool(is_sparse)

    kind = _get_tcgen05_mma_kind(d_dtype, a_dtype, b_dtype, sfa_dtype, sfb_dtype)
    valid_kinds = {"mxf8f6f4", "mxf4", "mxf4nvf4"}
    if kind not in valid_kinds:
        raise ValueError(
            f"Check failed for Data Type Kind. Expected one of {valid_kinds}, but got '{kind}' "
            f"for d:{d_dtype}, a:{a_dtype}, b:{b_dtype}, sfa:{sfa_dtype}, sfb:{sfb_dtype}"
        )

    _check_tcgen05_mma_matrix_shape(kind, n_cta_group, M, N, K, is_sparse)

    # Phase 2: Map data types to integer format codes
    format_map = {
        PTXDataType.FLOAT8_E4M3FN: 0,
        PTXDataType.FLOAT8_E4M3FNUZ: 0,
        PTXDataType.FLOAT8_E5M2: 1,
        PTXDataType.FLOAT6_E2M3FN: 3,
        PTXDataType.FLOAT6_E3M2FN: 4,
        PTXDataType.FLOAT4_E2M1FN: 5,
    }
    format_map_sf = {
        PTXDataType.FLOAT8_E4M3FN: 0,
        PTXDataType.FLOAT8_E4M3FNUZ: 0,
        PTXDataType.FLOAT8_E8M0FNU: 1,
    }

    atype_enum = PTXDataType.from_string(a_dtype)
    btype_enum = PTXDataType.from_string(b_dtype)
    stype_enum = PTXDataType.from_string(sfa_dtype)

    if kind == "mxf8f6f4":
        a_format = format_map[atype_enum]
        b_format = format_map[btype_enum]
    else:  # mxf4 and mxf4nvf4
        a_format = 1  # Corresponds to E5M2 in the map, a specific hardware encoding choice
        b_format = 1

    s_format = format_map_sf[stype_enum]

    # Phase 3: Detailed conditional validation for transpose
    valid_dtypes_for_trans = {
        PTXDataType.FLOAT8_E4M3FN,
        PTXDataType.FLOAT8_E4M3FNUZ,
        PTXDataType.FLOAT8_E5M2,
    }
    if trans_a and atype_enum not in valid_dtypes_for_trans:
        raise ValueError(f"Invalid a_dtype for transpose: {a_dtype}")
    if trans_b and btype_enum not in valid_dtypes_for_trans:
        raise ValueError(f"Invalid b_dtype for transpose: {b_dtype}")

    func_name = "ptx_tcgen05_encode_instr_descriptor_block_scaled"
    source_code = f"""
__forceinline__ __device__ void {func_name}(uint32_t* desc, int M, int N, int a_format,
                                            int b_format, int s_format, bool trans_a, bool trans_b,
                                            bool neg_a, bool neg_b, bool is_sparse) {{
  InstrDescriptorBlockScaled _desc;

  _desc.a_format_ = uint8_t(a_format);
  _desc.b_format_ = uint8_t(b_format);
  _desc.scale_format_ = uint8_t(s_format);

  _desc.a_sf_id_ = 0;
  _desc.b_sf_id_ = 0;

  _desc.m_dim_ = (M >> 4);
  _desc.n_dim_ = (N >> 3);

  _desc.a_major_ = static_cast<uint8_t>(trans_a);
  _desc.b_major_ = static_cast<uint8_t>(trans_b);

  _desc.a_negate_ = static_cast<uint8_t>(neg_a);
  _desc.b_negate_ = static_cast<uint8_t>(neg_b);

  _desc.sparse_flag_ = is_sparse;
  _desc.sparse_id2_  = 0;                          // should modify in sparse case

  *desc = (uint32_t)_desc;
}}
"""
    return cuda_func_call(
        func_name,
        desc,
        M,
        N,
        a_format,
        b_format,
        s_format,
        trans_a,
        trans_b,
        neg_a,
        neg_b,
        is_sparse,
        source_code=source_code,
    ), ["instr_descriptor_block_scaled"]


def _tcgen05_mma_common(
    d_dtype,
    a_dtype,
    b_dtype,
    d_tmem_addr,
    a_operand,
    b_desc,
    i_desc,
    use_a_tmem,
    cta_group,
    enable_input_d,
    scale_input_d,
    *disable_output_lane,
    sparse=False,
    sp_tmem_addr=None,
):
    d_dtype = parse_str(d_dtype)
    a_dtype = parse_str(a_dtype)
    b_dtype = parse_str(b_dtype)
    use_a_tmem = bool(use_a_tmem)
    cta_group = validate_cta_group(cta_group)
    scale_input_d = int(scale_input_d)
    if not (0 <= scale_input_d <= 15):
        raise ValueError(
            f"scale_input_d is incorrect, expected a value within [0, 15], got {scale_input_d}"
        )

    expected_vec_size = 8 if cta_group == 2 else 4
    if len(disable_output_lane) != expected_vec_size:
        raise ValueError(
            "The number of arguments for ptx_tcgen05_mma is incorrect, expected "
            f"{11 + expected_vec_size} total args (meaning {expected_vec_size} lane mask args), "
            f"but got {len(disable_output_lane)}."
        )

    kind = _get_tcgen05_mma_kind(d_dtype, a_dtype, b_dtype)
    valid_kinds = {"f16", "tf32", "f8f6f4", "i8"}
    if kind not in valid_kinds:
        raise ValueError(
            f"Check failed for Data Type Kind. Got '{kind}', expected one of {valid_kinds}"
        )

    if scale_input_d > 0 and kind not in {"f16", "tf32"}:
        raise ValueError(f"scale_input_d is only valid for kind 'f16' or 'tf32', not '{kind}'")

    if sparse:
        p_operand_idx = 5
        sparse_instr_suffix = ".sp"
        i_sp_operand_str = "[%3], %4,"
        sp_tmem_addr_str = "uint32_t sp_tmem_addr, "
        mask_start_idx = 6
    else:
        p_operand_idx = 4
        sparse_instr_suffix = ""
        i_sp_operand_str = "%3,"
        sp_tmem_addr_str = ""
        mask_start_idx = 5

    mask_signature = ", ".join([f"uint32_t mask{i}" for i in range(len(disable_output_lane))])

    a_operand_str = "[%1]" if use_a_tmem else "%1"
    a_operand_type = "uint32_t" if use_a_tmem else "uint64_t"
    mask_placeholders = ", ".join(
        [f"%{mask_start_idx + i}" for i in range(len(disable_output_lane))]
    )

    scale_placeholder = ""
    if scale_input_d > 0:
        scale_operand_idx = mask_start_idx + len(disable_output_lane)
        scale_placeholder = f", %{scale_operand_idx}"

    # Build the list of input operands for the asm block
    input_operands_list = [
        '"r"(d_tmem_addr)',  # %0
        f'"{("r" if use_a_tmem else "l")}"(a_operand)',  # %1
        '"l"(b_desc)',  # %2
    ]
    if sparse:
        input_operands_list.append('"r"(sp_tmem_addr)')  # %3 (sparse only)
    input_operands_list.extend(
        [
            '"r"(i_desc)',  # %3 or %4
            '"r"(enable_input_d)',  # %4 or %5
        ]
    )
    for i in range(len(disable_output_lane)):
        input_operands_list.append(f'"r"(mask{i})')
    if scale_input_d > 0:
        input_operands_list.append(f'"n"({scale_input_d})')

    input_operands_list = ", ".join(input_operands_list)

    func_name = (
        f"ptx_tcgen05_mma_cta_{cta_group}_kind_{kind}"
        + ("_sp" if sparse else "")
        + ("TS" if use_a_tmem else "SS")
        + (f"_{scale_input_d}" if scale_input_d > 0 else "")
    )
    source_code = f"""
__forceinline__ __device__ void {func_name}(uint32_t d_tmem_addr, {a_operand_type} a_operand, uint64_t b_desc, {sp_tmem_addr_str}uint32_t i_desc, int enable_input_d, {mask_signature}) {{
    asm volatile(
        "{{\\n"
        ".reg .pred p;\\n"
        "setp.ne.b32 p, %{p_operand_idx}, 0;\\n"
        "tcgen05.mma{sparse_instr_suffix}.cta_group::{cta_group}.kind::{kind} [%0], {a_operand_str}, %2, {i_sp_operand_str} "
        "{{{mask_placeholders}}}, p{scale_placeholder};\\n"
        "}}\\n"
        :
        :  {input_operands_list}
    );
}}
"""  # noqa: E501

    args = [func_name, d_tmem_addr, a_operand, b_desc]
    if sparse:
        args.append(sp_tmem_addr)
    args.append(i_desc)
    args.append(enable_input_d)
    args.extend(disable_output_lane)

    return cuda_func_call(*args, source_code=source_code)


@register_codegen("ptx_tcgen05_mma")
def codegen_ptx_tcgen05_mma(
    d_dtype,
    a_dtype,
    b_dtype,
    d_tmem_addr,
    a_operand,
    b_desc,
    i_desc,
    use_a_tmem,
    cta_group,
    enable_input_d,
    scale_input_d,
    *disable_output_lane,
):
    return _tcgen05_mma_common(
        d_dtype,
        a_dtype,
        b_dtype,
        d_tmem_addr,
        a_operand,
        b_desc,
        i_desc,
        use_a_tmem,
        cta_group,
        enable_input_d,
        scale_input_d,
        *disable_output_lane,
    )


@register_codegen("ptx_tcgen05_mma_sp")
def codegen_ptx_tcgen05_mma_sp(
    d_dtype,
    a_dtype,
    b_dtype,
    d_tmem_addr,
    a_operand,
    b_desc,
    sp_tmem_addr,
    i_desc,
    use_a_tmem,
    cta_group,
    enable_input_d,
    scale_input_d,
    *disable_output_lane,
):
    return _tcgen05_mma_common(
        d_dtype,
        a_dtype,
        b_dtype,
        d_tmem_addr,
        a_operand,
        b_desc,
        sp_tmem_addr,
        i_desc,
        use_a_tmem,
        cta_group,
        enable_input_d,
        scale_input_d,
        *disable_output_lane,
        sparse=True,
        sp_tmem_addr=sp_tmem_addr,
    )


def _get_tcgen05_mma_scale_vec_size(kind: str, scale_dtype: str) -> int:
    """
    Determines the scale vector size for a tcgen05 MMA instruction.
    This is a direct translation of the C++ GetTcgen05MMAScaleVecSize function.
    """
    scale_vec_size = 0
    stype = PTXDataType.from_string(scale_dtype)

    if kind == "mxf8f6f4" and stype == PTXDataType.FLOAT8_E8M0FNU:
        scale_vec_size = 1
    elif kind == "mxf4" and stype == PTXDataType.FLOAT8_E8M0FNU:
        scale_vec_size = 2
    elif kind == "mxf4nvf4" and stype == PTXDataType.FLOAT8_E8M0FNU:
        scale_vec_size = 2
    elif kind == "mxf4nvf4" and stype in {PTXDataType.FLOAT8_E4M3FN, PTXDataType.FLOAT8_E4M3FNUZ}:
        scale_vec_size = 4

    if scale_vec_size <= 0:
        raise ValueError(
            f"Invalid scale vector size for Tcgen05 MMA, check failed for kind::{kind}, "
            f"scale_dtype: {scale_dtype}"
        )
    return scale_vec_size


def _tcgen05_mma_block_scaled_common(
    d_dtype,
    a_dtype,
    b_dtype,
    sfa_dtype,
    sfb_dtype,
    d_tmem_addr,
    a_operand,
    b_desc,
    sfa_tmem_addr,
    sfb_tmem_addr,
    i_desc,
    use_a_tmem,
    cta_group,
    enable_input_d,
    sparse=False,
    sp_tmem_addr=None,
):
    d_dtype = parse_str(d_dtype)
    a_dtype = parse_str(a_dtype)
    b_dtype = parse_str(b_dtype)
    sfa_dtype = parse_str(sfa_dtype)
    sfb_dtype = parse_str(sfb_dtype)
    use_a_tmem = bool(use_a_tmem)
    cta_group = validate_cta_group(cta_group)

    kind = _get_tcgen05_mma_kind(d_dtype, a_dtype, b_dtype, sfa_dtype, sfb_dtype)
    valid_kinds = {"mxf8f6f4", "mxf4", "mxf4nvf4"}
    if kind not in valid_kinds:
        raise ValueError(
            f"Check failed for Data Type Kind. Expected one of {valid_kinds}, but got '{kind}' "
            f"for d:{d_dtype}, a:{a_dtype}, b:{b_dtype}, sfa:{sfa_dtype}, sfb:{sfb_dtype}"
        )

    scale_vec_size = _get_tcgen05_mma_scale_vec_size(kind, sfa_dtype)

    sparse_instr_suffix = ".sp" if sparse else ""
    sparse_placeholder = "[%7], " if sparse else ""
    a_constraint = '"r"' if use_a_tmem else '"l"'
    a_operand_type = "uint32_t" if use_a_tmem else "uint64_t"
    a_operand_placeholder = "[%1]" if use_a_tmem else "%1"
    sp_tmem_addr_str = "uint32_t sp_tmem_addr, " if sparse else ""
    sp_tmem_addr_operand = f', "r"({sp_tmem_addr})' if sparse else ""

    func_name = (
        f"ptx_tcgen05_mma_block_scaled_cta_{cta_group}_kind_{kind}_scale_vec_{scale_vec_size}"
        + ("_sp" if sparse else "")
        + ("TS" if use_a_tmem else "SS")
    )
    source_code = f"""
__forceinline__ __device__ void {func_name}(uint32_t d_tmem_addr, {a_operand_type} a_operand, uint64_t b_desc, {sp_tmem_addr_str}uint32_t i_desc, uint32_t sfa_tmem_addr, uint32_t sfb_tmem_addr, int accum) {{
    asm volatile(
        "{{\\n"
        ".reg .pred p;\\n"
        "setp.ne.b32 p, %4, 0;\\n"
        "tcgen05.mma{sparse_instr_suffix}.cta_group::{cta_group}.kind::{kind}.block_scale.scale_vec::{scale_vec_size}X "
        "[%0], {a_operand_placeholder}, %2, {sparse_placeholder}%3, [%5], [%6], p;\\n"
        "}}\\n"
        :
        : "r"(d_tmem_addr), {a_constraint}(a_operand), "l"(b_desc), "r"(i_desc), "r"(accum), "r"(sfa_tmem_addr), "r"(sfb_tmem_addr){sp_tmem_addr_operand}
    );
}}
"""  # noqa: E501
    args = [func_name, d_tmem_addr, a_operand, b_desc]
    if sparse:
        args.append(sp_tmem_addr)
    args.append(i_desc)
    args.append(sfa_tmem_addr)
    args.append(sfb_tmem_addr)
    args.append(enable_input_d)

    return cuda_func_call(*args, source_code=source_code)


@register_codegen("ptx_tcgen05_mma_block_scale")
def codegen_ptx_tcgen05_mma_block_scale(
    d_dtype,
    a_dtype,
    b_dtype,
    sfa_dtype,
    sfb_dtype,
    d_tmem_addr,
    a_operand,
    b_desc,
    sfa_tmem_addr,
    sfb_tmem_addr,
    i_desc,
    use_a_tmem,
    cta_group,
    enable_input_d=True,
):
    return _tcgen05_mma_block_scaled_common(
        d_dtype,
        a_dtype,
        b_dtype,
        sfa_dtype,
        sfb_dtype,
        d_tmem_addr,
        a_operand,
        b_desc,
        sfa_tmem_addr,
        sfb_tmem_addr,
        i_desc,
        use_a_tmem,
        cta_group,
        enable_input_d,
    )


@register_codegen("ptx_tcgen05_mma_sp_block_scale")
def codegen_ptx_tcgen05_mma_sp_block_scale(
    d_dtype,
    a_dtype,
    b_dtype,
    sfa_dtype,
    sfb_dtype,
    d_tmem_addr,
    a_operand,
    b_desc,
    sfa_tmem_addr,
    sfb_tmem_addr,
    sp_tmem_addr,
    i_desc,
    use_a_tmem,
    cta_group,
    enable_input_d=True,
):
    return _tcgen05_mma_block_scaled_common(
        d_dtype,
        a_dtype,
        b_dtype,
        sfa_dtype,
        sfb_dtype,
        d_tmem_addr,
        a_operand,
        b_desc,
        sfa_tmem_addr,
        sfb_tmem_addr,
        i_desc,
        use_a_tmem,
        cta_group,
        enable_input_d,
        sparse=True,
        sp_tmem_addr=sp_tmem_addr,
    )


@register_codegen("ptx_tcgen05_commit")
def codegen_ptx_tcgen05_commit(bar, cta_group, cta_mask):
    cta_group = int(cta_group)

    if cta_group not in [1, 2]:
        raise ValueError(f"The number of cta_group is incorrect, expected 1 or 2, got {cta_group}")

    is_multicast = not (
        isinstance(cta_mask, tvm.tirx.IntImm) and bin(int(cta_mask)).count("1") <= 1
    )

    if is_multicast:
        multicast_str = ".multicast::cluster"
        mask_operand_str = ", %1"
        cta_mask_arg_str = ', "h"(cta_mask)'
    else:
        multicast_str = ""
        mask_operand_str = ""
        cta_mask_arg_str = ""

    func_name = "ptx_tcgen05_commit_cta_group_" + str(cta_group)
    if is_multicast:
        func_name += "_multicast"

    source_code = f"""
__forceinline__ __device__ void {func_name}(void* bar, int cta_mask_) {{
  unsigned int bar_addr = __cvta_generic_to_shared(bar);
  uint16_t cta_mask = static_cast<uint16_t>(cta_mask_);
    __asm__ __volatile__(
        "tcgen05.commit.cta_group::{cta_group}.mbarrier::arrive::one.shared::cluster{multicast_str}.b64 [%0]{mask_operand_str};"
        :
        :"r"(bar_addr){cta_mask_arg_str}
    );
}}
"""  # noqa: E501
    return cuda_func_call(func_name, bar, cta_mask, source_code=source_code)


@register_codegen("ptx_tcgen05_cp")
def codegen_ptx_tcgen05_cp(
    taddr,
    src_desc,
    shape,
    cta_group,
    multicast,
    decompress,
    row,
    col,
):
    shape = parse_str(shape)
    multicast = parse_str(multicast)
    decompress = parse_str(decompress)
    cta_group = validate_cta_group(cta_group)

    multicast_str = f".{multicast}" if multicast else ""
    decompress_str = f".{decompress}" if decompress else ""

    multicast_safe = multicast.replace("::", "_")
    decompress_safe = decompress.replace(".", "_")
    func_name = (
        f"ptx_tcgen05_cp_cta_group_{cta_group}_shape_{shape}"
        f"_multicast_{multicast_safe}_decompress_{decompress_safe}"
    )
    source_code = f"""
__forceinline__ __device__ void {func_name}(uint32_t taddr, int row_offset, int col_offset, uint64_t src_desc) {{
    asm volatile(
        "tcgen05.cp.cta_group::{cta_group}.{shape}{multicast_str}{decompress_str} [%0], %1;"
        :
        : "r"(get_tmem_addr(taddr, row_offset, col_offset)), "l"(src_desc)
    );
}}
"""  # noqa: E501
    return cuda_func_call(
        func_name,
        taddr,
        row,
        col,
        src_desc,
        source_code=source_code,
    ), ["get_tmem_addr"]
