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
# pylint: disable=redefined-builtin, invalid-name, too-many-arguments, too-many-locals
"""PTX WGMMA operations (Hopper warpgroup MMA)."""
import tvm

from tvm.tir.op import cuda_func_call

from .registry import register_codegen
from .types import PTXDataType
from .utils import parse_str


@register_codegen("ptx_wgmma_encode_matrix_descriptor")
def codegen_ptx_wgmma_encode_matrix_descriptor(desc, addr, ldo, sdo, swizzle):
    valid_swizzle_modes = {0, 1, 2, 3}
    if swizzle not in valid_swizzle_modes:
        raise ValueError(
            f"Invalid swizzle mode. Expected a value in {valid_swizzle_modes}, but got {swizzle}"
        )

    func_name = "ptx_wgmma_encode_matrix_descriptor"
    source_code = f"""
__forceinline__ __device__ void {func_name}(uint64_t* desc, void* addr, int ldo, int sdo, int swizzle) {{
  GmmaDescriptor _desc;

  switch (swizzle) {{
    case 0: _desc.bitfield.layout_type_ = uint8_t(0); break; // No swizzle
    case 1: _desc.bitfield.layout_type_ = uint8_t(3); break; // 32B swizzle
    case 2: _desc.bitfield.layout_type_ = uint8_t(2); break; // 64B swizzle
    case 3: _desc.bitfield.layout_type_ = uint8_t(1); break; // 128B swizzle
  }}

  uint32_t start_address = __cvta_generic_to_shared(addr);
  _desc.bitfield.start_address_ = static_cast<uint16_t>(start_address >> 4);

  constexpr uint8_t base_offset = 0;
  _desc.bitfield.base_offset_ = base_offset;

  _desc.bitfield.stride_byte_offset_  = static_cast<uint32_t>(sdo);
  _desc.bitfield.leading_byte_offset_ = static_cast<uint32_t>(ldo);

  *desc = (uint64_t)_desc;
}}"""
    return cuda_func_call(func_name, desc, addr, ldo, sdo, swizzle, source_code=source_code), [
        "gmma_descriptor"
    ]


@register_codegen("ptx_wgmma_noop_barrier")
def codegen_ptx_wgmma_noop_barrier(reg):
    dtype = str(reg.dtype)
    dtype_enum = PTXDataType.from_string(dtype)
    if dtype_enum == PTXDataType.UINT32:
        format_str, dtype_str = "r", "uint32_t"
    elif dtype_enum == PTXDataType.FLOAT32:
        format_str, dtype_str = "f", "float"
    else:
        raise ValueError(f"Only support uint32/float32 for wgmma_fence, but got {dtype}.")

    # 2. The function name is generated dynamically based on the data type.
    func_name = f"ptx_wgmma_fence_{dtype_str}"

    # 3. Populate the C++ template. The empty asm string "" with memory clobber
    #    and an input/output operand acts as a compiler optimization barrier.
    source_code = f"""
__forceinline__ __device__ void {func_name}({dtype_str} reg) {{
  asm volatile("" : "+{format_str}"(reg)::"memory");
}}"""
    return cuda_func_call(func_name, reg, source_code=source_code)


@register_codegen("ptx_wgmma_fence")
def codegen_ptx_wgmma_fence():
    func_name = "ptx_wgmma_fence"
    source_code = f"""
__forceinline__ __device__ void {func_name}() {{
  asm volatile("wgmma.fence.sync.aligned;\\n" ::: "memory");
}}"""
    return cuda_func_call(func_name, source_code=source_code)


@register_codegen("ptx_wgmma_commit_group")
def codegen_ptx_wgmma_commit_group():
    func_name = "ptx_wgmma_commit_group"
    source_code = f"""
__forceinline__ __device__ void {func_name}() {{
  asm volatile("wgmma.commit_group.sync.aligned;\\n" ::: "memory");
}}"""
    return cuda_func_call(func_name, source_code=source_code)


@register_codegen("ptx_wgmma_wait_group")
def codegen_ptx_wgmma_wait_group(n):
    n = int(n)
    func_name = f"ptx_wgmma_wait_group_{n}"
    source_code = f"""
__forceinline__ __device__ void {func_name}() {{
  asm volatile("wgmma.wait_group.sync.aligned %0;\\n" :: "n"({n}) : "memory");
}}"""
    return cuda_func_call(func_name, source_code=source_code)


@register_codegen("ptx_wgmma_mma_async_ss")
def codegen_ptx_wgmma_mma_async_ss(
    M, N, K, in_dtype, out_dtype, transA, transB, scaleA, scaleB, scaleD, descA, descB, *accums
):
    M = int(M)
    N = int(N)
    K = int(K)
    in_dtype = parse_str(in_dtype)
    out_dtype = parse_str(out_dtype)
    transA = bool(transA)
    transB = bool(transB)
    scaleA = float(scaleA)
    scaleB = float(scaleB)

    expected_accm_cnt = M * N // 128
    if len(accums) != expected_accm_cnt:
        raise ValueError(
            "The number of arguments is incorrect. Expected "
            f"{12 + expected_accm_cnt} total args (meaning {expected_accm_cnt} accumulator args), "
            f"but got {len(accums)}."
        )

    if out_dtype != "float32":
        raise ValueError("Codegen only supports float32 as output dtype for WGMMA.")

    allow_transpose = in_dtype in {"float16", "bfloat16"}
    if not allow_transpose and (transA or transB):
        raise ValueError("Transpose is only supported for .f16/.bf16 types in WGMMA.")

    itype = PTXDataType.from_string(in_dtype)
    otype = PTXDataType.from_string(out_dtype)

    num_accums = len(accums)
    descA_idx = num_accums
    descB_idx = num_accums + 1
    scaleD_idx = num_accums + 2
    scaleA_idx = num_accums + 3
    scaleB_idx = num_accums + 4
    transA_idx = num_accums + 5
    transB_idx = num_accums + 6

    if allow_transpose:
        transpose_r_code = f", %{transA_idx}, %{transB_idx}"
        transpose_constraints = f', "n"({1 if transA else 0}), "n"({1 if transB else 0})'
    else:
        transpose_r_code = ""
        transpose_constraints = ""

    descA_idx, descB_idx, scaleD_idx, scaleA_idx, scaleB_idx = (num_accums + i for i in range(5))
    transA_idx, transB_idx = (num_accums + 5, num_accums + 6)

    accum_params = ", ".join([f"float& p_acc{i}" for i in range(num_accums)])
    accum_r_list = ", ".join([f"%{i}" for i in range(num_accums)])
    accum_constraints = ", ".join([f'"+f"(p_acc{i})' for i in range(num_accums)])

    func_name = f"ptx_wgmma_mma_async_ss_{M}x{N}x{K}_{in_dtype.replace('.', '')}_{out_dtype.replace('.', '')}_{int(scaleA)}_{int(scaleB)}_{int(transA)}_{int(transB)}"
    source_code = f"""
__forceinline__ __device__ void {func_name}({accum_params}, uint64_t p_descA, uint64_t p_descB, int p_scaleD) {{
    /* T.ptx_wgmma_mma_async_ss() */
    asm volatile(
      "{{    \\n"
      ".reg .pred p;\\n"
      "setp.ne.b32 p, %{scaleD_idx}, 0;\\n"
      "wgmma.mma_async.sync.aligned.m{M}n{N}k{K}{otype.to_string()}{itype.to_string()}{itype.to_string()} "
      "{{{accum_r_list}}},"
      "%{descA_idx}, %{descB_idx},"
      "p, %{scaleA_idx}, %{scaleB_idx}{transpose_r_code};\\n"
      "}}\\n"
      : {accum_constraints}
      : "l"(p_descA), "l"(p_descB), "r"(p_scaleD), "n"({int(scaleA)}), "n"({int(scaleB)}){transpose_constraints}
    );
}}"""
    return cuda_func_call(func_name, *accums, descA, descB, scaleD, source_code=source_code)


@register_codegen("ptx_wgmma_mma_async_rs")
def codegen_ptx_wgmma_mma_async_rs(
    M, N, K, in_dtype, out_dtype, transA, transB, scaleA, scaleB, scaleD, descB, *reg_list
):
    M = int(M)
    N = int(N)
    K = int(K)
    transA = bool(transA)
    transB = bool(transB)
    scaleA = float(scaleA)
    scaleB = float(scaleB)
    in_dtype = parse_str(in_dtype)
    out_dtype = parse_str(out_dtype)

    if out_dtype != "float32":
        raise ValueError("This generator only supports float32 as the output dtype for WGMMA.")

    in_dtype_bits = tvm.runtime.DataType(in_dtype).bits
    if in_dtype_bits is None:
        raise ValueError(f"Bit width not defined for input dtype: {in_dtype}")

    expected_A_cnt = M * K // 128 // (32 // in_dtype_bits)
    expected_accm_cnt = M * N // 128

    if len(reg_list) != expected_A_cnt + expected_accm_cnt:
        raise ValueError(
            f"Incorrect number of A registers. Expected {expected_A_cnt}, got {len(reg_list)}"
        )
    A_regs = reg_list[:expected_A_cnt]
    accums = reg_list[expected_A_cnt:]

    allow_transpose = in_dtype in {"float16", "bfloat16"}
    if not allow_transpose and (transA or transB):
        raise ValueError("Transpose is only supported for .f16/.bf16 types in WGMMA.")

    itype = PTXDataType.from_string(in_dtype)
    otype = PTXDataType.from_string(out_dtype)
    allow_transpose = in_dtype in {"float16", "bfloat16"}

    accum_params = ", ".join([f"float& p_acc{i}" for i in range(expected_accm_cnt)])
    A_reg_params = ", ".join([f"uint32_t& p_A{i}" for i in range(expected_A_cnt)])

    accum_r_list = ", ".join([f"%{i}" for i in range(expected_accm_cnt)])
    A_reg_r_list = ", ".join([f"%{expected_accm_cnt + i}" for i in range(expected_A_cnt)])

    base_idx = expected_accm_cnt + expected_A_cnt
    descB_idx, scaleD_idx, scaleA_idx, scaleB_idx, transB_idx = (base_idx + i for i in range(5))

    accum_constraints = ", ".join([f'"+f"(p_acc{i})' for i in range(expected_accm_cnt)])
    A_reg_constraints = ", ".join([f'"r"(p_A{i})' for i in range(expected_A_cnt)])

    if allow_transpose:
        transpose_r_code = f", %{transB_idx}"
        transpose_constraints = f', "n"({1 if transB else 0})'
    else:
        transpose_r_code, transpose_constraints = "", ""

    func_name = f"ptx_wgmma_mma_async_rs_{M}x{N}x{K}_{in_dtype.replace('.', '')}_{out_dtype.replace('.', '')}_{int(scaleA)}_{int(scaleB)}_{int(transA)}_{int(transB)}"
    source_code = f"""
__forceinline__ __device__ void {func_name}({accum_params}, {A_reg_params}, uint64_t p_descB, int p_scaleD) {{
    /* T.ptx_wgmma_mma_async_rs() */
    asm volatile(
      "{{    \\n"
      ".reg .pred p;\\n"
      "setp.ne.b32 p, %{scaleD_idx}, 0;\\n"
      "wgmma.mma_async.sync.aligned.m{M}n{N}k{K}{otype.to_string()}{itype.to_string()}{itype.to_string()} "
      "{{{accum_r_list}}},"
      "{{{A_reg_r_list}}}, %{descB_idx},"
      "p, %{scaleA_idx}, %{scaleB_idx}{transpose_r_code};\\n"
      "}}\\n"
      : {accum_constraints}
      : {A_reg_constraints}, "l"(p_descB), "r"(p_scaleD), "n"({int(scaleA)}), "n"({int(scaleB)}){transpose_constraints}
    );
}}"""
    return cuda_func_call(func_name, *accums, *A_regs, descB, scaleD, source_code=source_code)
