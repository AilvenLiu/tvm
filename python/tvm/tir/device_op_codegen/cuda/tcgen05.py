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
"""PTX tcgen05 operations (Blackwell tensor memory, MMA)."""
import tvm

from tvm.tir.op import cuda_func_call

from .registry import register_codegen
from .types import PTXDataType
from .utils import parse_str, is_power_of_two, validate_cta_group, validate_power_of_two_range


@register_codegen("ptx_tcgen05_alloc")
def codegen_ptx_tcgen05_alloc(dst_shared_ptr, n_cols, n_cta_group):
    n_cols = int(n_cols)
    if not (32 <= n_cols <= 512 and n_cols % 32 == 0 and is_power_of_two(n_cols)):
        raise ValueError(
            "The number of columns to allocate in Tensor Memory is invalid, "
            f"expect a value within range [32, 512] and be a multiple of 32 "
            f"and a power of 2, got {n_cols}"
        )
    n_cta_group = validate_cta_group(n_cta_group, "allocating Tensor Memory")

    func_name = f"tvm_builtin_ptx_tcgen05_alloc_cta_group_{n_cta_group}"
    source_code = f"""
__forceinline__ __device__ void {func_name}(void* dst, int nCols) {{
  unsigned int smem_addr = __cvta_generic_to_shared(dst);
    __asm__ __volatile__(
      "tcgen05.alloc.cta_group::{n_cta_group}.sync.aligned.shared::cta.b32 [%0], %1;"
      :: "r"(smem_addr), "r"(nCols)
      : "memory"
    );
}}
"""
    return cuda_func_call(func_name, dst_shared_ptr, n_cols, source_code=source_code)


@register_codegen("ptx_tcgen05_dealloc")
def codegen_ptx_tcgen05_dealloc(taddr, n_cols, n_cta_group):
    n_cols = int(n_cols)
    if not (32 <= n_cols <= 512 and n_cols % 32 == 0 and is_power_of_two(n_cols)):
        raise ValueError(
            "The number of columns to deallocate in Tensor Memory is invalid, expect a value within"
            f"range [32, 512] and be a multiple of 32 and a power of 2, got {n_cols}"
        )
    n_cta_group = validate_cta_group(n_cta_group, "deallocating Tensor Memory")

    func_name = f"tvm_builtin_ptx_tcgen05_dealloc_cta_group_{n_cta_group}"
    source_code = f"""
__forceinline__ __device__ void {func_name}(uint32_t taddr, int nCols) {{
    __asm__ __volatile__(
      "tcgen05.dealloc.cta_group::{n_cta_group}.sync.aligned.b32 %0, %1;"
      :: "r"(taddr), "r"(nCols)
      : "memory"
    );
}}
"""
    return cuda_func_call(func_name, taddr, n_cols, source_code=source_code)


@register_codegen("ptx_tcgen05_relinquish_alloc_permit")
def codegen_ptx_tcgen05_relinquish_alloc_permit(n_cta_group):
    n_cta_group = validate_cta_group(n_cta_group, "relinquishing alloc permit")

    func_name = f"tvm_builtin_ptx_tcgen05_relinquish_alloc_permit_cta_group_{n_cta_group}"
    source_code = f"""
__forceinline__ __device__ void {func_name}() {{
    __asm__ __volatile__(
        "tcgen05.relinquish_alloc_permit.cta_group::{n_cta_group}.sync.aligned;"
        ::: "memory"
    );
}}
"""
    return cuda_func_call(func_name, source_code=source_code)


@register_codegen("ptx_tcgen05_fence_before_thread_sync")
def codegen_ptx_tcgen05_fence_before_thread_sync():
    func_name = "tvm_builtin_ptx_tcgen05_fence_before_thread_sync"
    source_code = f"""
__forceinline__ __device__ void {func_name}() {{
  asm volatile("tcgen05.fence::before_thread_sync;\\n" ::: "memory");
}}
"""
    return cuda_func_call(func_name, source_code=source_code)


@register_codegen("ptx_tcgen05_fence_after_thread_sync")
def codegen_ptx_tcgen05_fence_after_thread_sync():
    func_name = "tvm_builtin_ptx_tcgen05_fence_after_thread_sync"
    source_code = f"""
__forceinline__ __device__ void {func_name}() {{
  asm volatile("tcgen05.fence::after_thread_sync;\\n" ::: "memory");
}}
"""
    return cuda_func_call(func_name, source_code=source_code)


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
    );
}}
"""
    regs = [tvm.tir.address_of(reg) for reg in regs]

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
"""
    regs = [tvm.tir.address_of(reg) for reg in regs]
    return cuda_func_call(
        func_name,
        dst_addr,
        row_offset,
        col_offset,
        *regs,
        source_code=source_code,
    ), ["get_tmem_addr"]


@register_codegen("ptx_tcgen05_wait_ld")
def codegen_ptx_tcgen05_wait_ld():
    func_name = "tvm_builtin_ptx_tcgen05_wait_ld"
    source_code = f"""
__forceinline__ __device__ void {func_name}() {{
    asm volatile("tcgen05.wait::ld.sync.aligned;\\n" ::: "memory");
}}
"""
    return cuda_func_call(func_name, source_code=source_code)


@register_codegen("ptx_tcgen05_wait_st")
def codegen_ptx_tcgen05_wait_st():
    func_name = "tvm_builtin_ptx_tcgen05_wait_st"
    source_code = f"""
__forceinline__ __device__ void {func_name}() {{
    asm volatile("tcgen05.wait::st.sync.aligned;\\n" ::: "memory");
}}
"""
    return cuda_func_call(func_name, source_code=source_code)


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
"""
    return cuda_func_call(func_name, desc, addr, ldo, sdo, swizzle, source_code=source_code), [
        "smem_descriptor"
    ]


def _get_tcgen05_mma_kind(
    d_dtype: str, a_dtype: str, b_dtype: str, sfa_dtype: str = "", sfb_dtype: str = ""
) -> str:
    kind = ""

    dtype = PTXDataType.from_string(d_dtype)
    atype = PTXDataType.from_string(a_dtype)
    btype = PTXDataType.from_string(b_dtype)
    kind = ""

    if (
        atype == PTXDataType.FLOAT16
        and btype == PTXDataType.FLOAT16
        and dtype == PTXDataType.FLOAT16
    ):
        kind = "f16"
    elif (
        (atype in {PTXDataType.BFLOAT16, PTXDataType.FLOAT16})
        and (btype in {PTXDataType.BFLOAT16, PTXDataType.FLOAT16})
        and dtype == PTXDataType.FLOAT32
    ):
        kind = "f16"
    elif (
        atype == PTXDataType.TENSOR_FLOAT32
        and btype == PTXDataType.TENSOR_FLOAT32
        and dtype == PTXDataType.FLOAT32
    ):
        kind = "tf32"
    elif (
        atype == PTXDataType.FLOAT4_E2M1FN
        and btype == PTXDataType.FLOAT4_E2M1FN
        and sfa_dtype
        and sfb_dtype
        and dtype == PTXDataType.FLOAT32
    ):
        sfa_dtype_enum = PTXDataType.from_string(sfa_dtype)
        sfb_dtype_enum = PTXDataType.from_string(sfb_dtype)
        if (
            sfa_dtype_enum == PTXDataType.FLOAT8_E8M0FNU
            and sfb_dtype_enum == PTXDataType.FLOAT8_E8M0FNU
        ):
            kind = "mxf4"
        elif (sfa_dtype_enum in {PTXDataType.FLOAT8_E4M3FN, PTXDataType.FLOAT8_E4M3FNUZ}) and (
            sfb_dtype_enum in {PTXDataType.FLOAT8_E4M3FN, PTXDataType.FLOAT8_E4M3FNUZ}
        ):
            kind = "mxf4nvf4"
    elif (
        atype
        in {
            PTXDataType.FLOAT8_E4M3FN,
            PTXDataType.FLOAT8_E4M3FNUZ,
            PTXDataType.FLOAT8_E5M2,
            PTXDataType.FLOAT6_E2M3FN,
            PTXDataType.FLOAT6_E3M2FN,
            PTXDataType.FLOAT4_E2M1FN,
        }
    ) and (
        btype
        in {
            PTXDataType.FLOAT8_E4M3FN,
            PTXDataType.FLOAT8_E4M3FNUZ,
            PTXDataType.FLOAT8_E5M2,
            PTXDataType.FLOAT6_E2M3FN,
            PTXDataType.FLOAT6_E3M2FN,
            PTXDataType.FLOAT4_E2M1FN,
        }
    ):
        if not sfa_dtype and not sfb_dtype:
            if dtype in {PTXDataType.FLOAT32, PTXDataType.FLOAT16}:
                kind = "f8f6f4"
        elif sfa_dtype and sfb_dtype:
            sfa_dtype_enum = PTXDataType.from_string(sfa_dtype)
            sfb_dtype_enum = PTXDataType.from_string(sfb_dtype)
            if (
                sfa_dtype_enum == PTXDataType.FLOAT8_E8M0FNU
                and sfb_dtype_enum == PTXDataType.FLOAT8_E8M0FNU
                and dtype == PTXDataType.FLOAT32
            ):
                kind = "mxf8f6f4"
    elif (
        (atype in {PTXDataType.INT8, PTXDataType.UINT8})
        and (btype in {PTXDataType.INT8, PTXDataType.UINT8})
        and dtype == PTXDataType.INT32
    ):
        kind = "i8"

    if not kind:
        raise ValueError(
            f"Invalid multiplicand data types for Tcgen05 MMA, check failed for d: {d_dtype}, "
            f"a: {a_dtype}, b: {b_dtype}, scale_a: {sfa_dtype}, scale_b: {sfb_dtype}"
        )
    return kind


def _check_tcgen05_mma_matrix_shape(
    kind: str, cta_group: int, m: int, n: int, k: int, is_sparse: bool
) -> bool:
    err = (
        f"Invalid matrix shape for Tcgen05 MMA, check failed for kind: {kind}, "
        f"is_sparse: {is_sparse}, cta_group: {cta_group}, M: {m}, N: {n}, K: {k}"
    )

    if kind in ["f16", "tf32", "f8f6f4"]:
        if cta_group == 1:
            if m == 64:
                if not (8 <= n <= 256 and n % 8 == 0):
                    raise ValueError(err)
            elif m == 128:
                if not (16 <= n <= 256 and n % 16 == 0):
                    raise ValueError(err)
            else:
                raise ValueError(err)
        elif cta_group == 2:
            if not (m in [128, 256]):
                raise ValueError(err)
            if not (32 <= n <= 256 and n % 32 == 0):
                raise ValueError(err)
    elif kind == "i8":
        if cta_group == 1:
            if not (m in [64, 128]):
                raise ValueError(err)
            is_n_valid = (n == 8) or (n == 24) or (16 <= n <= 256 and n % 16 == 0)
            if not is_n_valid:
                raise ValueError(err)
        elif cta_group == 2:
            if not (m in [128, 256]):
                raise ValueError(err)
            if not (32 <= n <= 256 and n % 32 == 0):
                raise ValueError(err)
    elif kind in ["mxf8f6f4", "mxf4", "mxf4nvf4"]:
        if cta_group == 1:
            if not (m == 128):
                raise ValueError(err)
            if not (8 <= n <= 256 and n % 8 == 0):
                raise ValueError(err)
        elif cta_group == 2:
            if is_sparse:
                if not (m == 256):
                    raise ValueError(err)
            else:  # dense
                if not (m in [128, 256]):
                    raise ValueError(err)
            if not (16 <= n <= 256 and n % 16 == 0):
                raise ValueError(err)
    else:
        raise ValueError(err)

    if kind == "f16":
        if not (k == (32 if is_sparse else 16)):
            raise ValueError(err)
    elif kind == "tf32":
        if not (k == (16 if is_sparse else 8)):
            raise ValueError(err)
    elif kind in ["f8f6f4", "i8", "mxf8f6f4"]:
        if not (k == (64 if is_sparse else 32)):
            raise ValueError(err)
    elif kind in ["mxf4", "mxf4nvf4"]:
        if not (k == (128 if is_sparse else 64)):
            raise ValueError(err)
    else:
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
            f"Check failed for Data Type Kind. d_dtype: {d_dtype}, a_dtype: {a_dtype}, b_dtype: {b_dtype}"
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
                                            bool neg_a, bool neg_b, bool is_sparse,
                                            uint32_t sfa_tmem_addr, uint32_t sfb_tmem_addr) {{
  InstrDescriptorBlockScaled _desc;

  _desc.a_format_ = uint8_t(a_format);
  _desc.b_format_ = uint8_t(b_format);
  _desc.scale_format_ = uint8_t(s_format);

  _desc.a_sf_id_ = (sfa_tmem_addr & 0xC0000000) >> 30;
  _desc.b_sf_id_ = (sfb_tmem_addr & 0xC0000000) >> 30;

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
        sfa_tmem_addr,
        sfb_tmem_addr,
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
"""

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
"""
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

    is_multicast = not (isinstance(cta_mask, tvm.tir.IntImm) and int(cta_mask) == 0)

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
"""
    return cuda_func_call(func_name, bar, cta_mask, source_code=source_code)


@register_codegen("ptx_tcgen05_cp")
def codegen_ptx_tcgen05_cp(
    dst_addr,
    row_offset,
    col_offset,
    src_desc,
    shape,
    dst_dtype,
    src_dtype,
    cta_group=1,
    multicast="",
):
    shape = parse_str(shape)
    dst_dtype = parse_str(dst_dtype)
    src_dtype = parse_str(src_dtype)
    cta_group = validate_cta_group(cta_group)
    multicast = parse_str(multicast)

    valid_shapes = {"128x256b", "4x256b", "128x128b", "64x128b", "32x128b"}
    if shape not in valid_shapes:
        raise ValueError(f"Invalid shape for tcgen05 copy, check failed for shape: {shape}")

    err_msg = f"Invalid multicast for tcgen05 copy, check failed for shape: {shape}, multicast: {multicast}"
    if shape == "64x128b":
        if multicast not in {"warpx2::02_13", "warpx2::01_23"}:
            raise ValueError(err_msg)
    elif shape == "32x128b":
        if multicast != "warpx4":
            raise ValueError(err_msg)
    else:
        if multicast != "":
            raise ValueError(err_msg)

    # 2. Decompression Format Logic
    # ===============================
    dst_src_fmt = ""
    dtype_enum = PTXDataType.from_string(dst_dtype)
    stype_enum = PTXDataType.from_string(src_dtype)

    fp8_types = {
        PTXDataType.FLOAT8_E4M3FN,
        PTXDataType.FLOAT8_E4M3FNUZ,
        PTXDataType.FLOAT8_E5M2,
        PTXDataType.FLOAT8_E8M0FNU,
    }
    fp6_types = {PTXDataType.FLOAT6_E2M3FN, PTXDataType.FLOAT6_E3M2FN}

    if dtype_enum in fp8_types:
        if stype_enum == PTXDataType.FLOAT4_E2M1FN:
            dst_src_fmt = ".b8x16.b4x16_p64"
        elif stype_enum in fp6_types:
            dst_src_fmt = ".b8x16.b6x16_p32"

    multicast_str = f".{multicast}" if multicast else ""

    func_name = f"ptx_tcgen05_cp_cta_group_{cta_group}_shape_{shape}_multicast_{multicast}"
    source_code = f"""
__forceinline__ __device__ void {func_name}(uint32_t dst_addr, int row_offset, int col_offset, uint64_t src_desc) {{
    asm volatile(
        "tcgen05.cp.cta_group::{cta_group}.{shape}{multicast_str}{dst_src_fmt} [%0], %1;"
        :
        : "r"(get_tmem_addr(dst_addr, row_offset, col_offset)), "l"(src_desc)
    );
}}
"""
    return cuda_func_call(
        func_name,
        dst_addr,
        row_offset,
        col_offset,
        src_desc,
        source_code=source_code,
    ), ["get_tmem_addr"]


@register_codegen("ptx_tcgen05_shift")
def codegen_ptx_tcgen05_shift(taddr, cta_group):
    cta_group = validate_cta_group(cta_group)

    func_name = f"ptx_tcgen05_shift_cta_group_{cta_group}"
    source_code = f"""
__forceinline__ __device__ void {func_name}(uint32_t taddr) {{
  __asm__ __volatile__(
    "tcgen05.shift.cta_group::{cta_group}.down [%0];"
    :: "r"(taddr)
  );
}}
"""
    return cuda_func_call(func_name, taddr, source_code=source_code)
