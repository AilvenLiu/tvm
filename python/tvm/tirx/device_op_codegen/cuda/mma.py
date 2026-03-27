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
"""PTX MMA operations (Volta/Turing/Ampere matrix ops)."""

import re
from dataclasses import dataclass

import tvm
from tvm import DataType
from tvm.tirx.op import cuda_func_call

from .registry import register_codegen
from .types import PTXDataType
from .utils import parse_str


@dataclass
class FragAttrs:
    """Fragment attributes."""

    reg_type: str
    size: int
    ptr_type: str


_FRAG_ATTRS_MAP = {
    PTXDataType.BIT1: FragAttrs("r", 32, "uint32_t"),
    PTXDataType.INT4: FragAttrs("r", 32, "uint32_t"),
    PTXDataType.UINT4: FragAttrs("r", 32, "uint32_t"),
    PTXDataType.INT8: FragAttrs("r", 32, "uint32_t"),
    PTXDataType.UINT8: FragAttrs("r", 32, "uint32_t"),
    PTXDataType.FLOAT8_E4M3FN: FragAttrs("r", 32, "uint32_t"),
    PTXDataType.FLOAT8_E5M2: FragAttrs("r", 32, "uint32_t"),
    PTXDataType.BIT16: FragAttrs("r", 32, "uint32_t"),
    PTXDataType.FLOAT16: FragAttrs("r", 32, "uint32_t"),
    PTXDataType.BFLOAT16: FragAttrs("r", 32, "uint32_t"),
    PTXDataType.TENSOR_FLOAT32: FragAttrs("r", 32, "uint32_t"),
    PTXDataType.INT32: FragAttrs("r", 32, "int32_t"),
    PTXDataType.FLOAT32: FragAttrs("f", 32, "float"),
    PTXDataType.FLOAT64: FragAttrs("d", 64, "double"),
}


@register_codegen("ptx_mma")
def codegen_ptx_mma(
    shape,
    a_layout,
    b_layout,
    d_type,
    a_type,
    b_type,
    c_type,
    d_ptr,
    a_ptr,
    b_ptr,
    c_ptr=0,
    saturate=False,
    bit_op=None,
):
    shape = parse_str(shape)
    a_layout = parse_str(a_layout)
    b_layout = parse_str(b_layout)
    d_type = parse_str(d_type)
    a_type = parse_str(a_type)
    b_type = parse_str(b_type)
    c_type = parse_str(c_type)
    saturate = bool(saturate)
    if bit_op is not None:
        bit_op = parse_str(bit_op)

    func_name = f"ptx_mma_{shape}_{a_layout}_{b_layout}_{d_type}_{a_type}_{b_type}_{c_type}"

    def parse_mma_shape(shape_str: str) -> tuple[int, int, int]:
        # Extracts the values m, n, and k from a string like "m16n8k16".
        match = re.search(r"m(\d+)n(\d+)k(\d+)", shape_str)
        if not match:
            raise ValueError(f"Cannot parse MMA shape from string: '{shape_str}'")
        return tuple(map(int, match.groups()))

    def get_mma_threads(m, n, k):
        if m == 8 and n == 8 and k == 4 and a_type == "float16":
            return 32 // 4
        else:
            return 32 // 1

    m, n, k = parse_mma_shape(shape)

    if isinstance(c_ptr, tvm.tirx.IntImm) and int(c_ptr) == 0:
        no_c_ptr = True
        func_name += "_no_c_ptr"
    else:
        no_c_ptr = False

    if saturate:
        func_name += "_saturate"
        saturate_inst = ".satfinite"
    else:
        saturate_inst = ""

    if bit_op:
        bit_op_inst = f".{bit_op}"
    else:
        bit_op_inst = ""

    d_ptx_type = PTXDataType.from_string(d_type)
    c_ptx_type = PTXDataType.from_string(c_type)
    a_ptx_type = PTXDataType.from_string(a_type)
    b_ptx_type = PTXDataType.from_string(b_type)

    d_type_inst = d_ptx_type.to_string()
    c_type_inst = c_ptx_type.to_string()
    a_type_inst = a_ptx_type.to_string()
    b_type_inst = b_ptx_type.to_string()

    d_frag_attrs = _FRAG_ATTRS_MAP[d_ptx_type]
    c_frag_attrs = _FRAG_ATTRS_MAP[c_ptx_type]
    a_frag_attrs = _FRAG_ATTRS_MAP[a_ptx_type]
    b_frag_attrs = _FRAG_ATTRS_MAP[b_ptx_type]
    mma_threads = get_mma_threads(m, n, k)

    d_args_cnt = m * n * DataType(d_type).bits // mma_threads // d_frag_attrs.size
    a_args_cnt = m * k * DataType(a_type).bits // mma_threads // a_frag_attrs.size
    b_args_cnt = k * n * DataType(b_type).bits // mma_threads // b_frag_attrs.size
    c_args_cnt = m * n * DataType(c_type).bits // mma_threads // c_frag_attrs.size

    def get_arg_array(start_idx, cnt):
        return "{" + ", ".join(f"%{start_idx + i}" for i in range(cnt)) + "}"

    # arg templates
    d_args = get_arg_array(0, d_args_cnt)
    a_args = get_arg_array(d_args_cnt, a_args_cnt)
    b_args = get_arg_array(d_args_cnt + a_args_cnt, b_args_cnt)
    c_args = get_arg_array(d_args_cnt + a_args_cnt + b_args_cnt, c_args_cnt)
    args_template = f"{d_args}, {a_args}, {b_args}, {c_args}"

    # operands
    d_inputs = ", ".join([f'"={d_frag_attrs.reg_type}"(d_ptr[{i}])' for i in range(d_args_cnt)])
    a_inputs = ", ".join([f'"{a_frag_attrs.reg_type}"(a_ptr[{i}])' for i in range(a_args_cnt)])
    b_inputs = ", ".join([f'"{b_frag_attrs.reg_type}"(b_ptr[{i}])' for i in range(b_args_cnt)])
    if no_c_ptr:
        if c_frag_attrs.reg_type == "r":
            c_value = "0"
        elif c_frag_attrs.reg_type == "f":
            c_value = "0.f"
        else:
            raise ValueError(f"Invalid register type: {c_frag_attrs.reg_type}")
        c_inputs = ", ".join([f'"{c_frag_attrs.reg_type}"({c_value})' for i in range(c_args_cnt)])
        c_signature = ""
        c_ptr_cast = ""
    else:
        c_inputs = ", ".join([f'"{c_frag_attrs.reg_type}"(c_ptr[{i}])' for i in range(c_args_cnt)])
        c_signature = ", void* c_ptr_in"
        c_ptr_cast = f"{c_frag_attrs.ptr_type}* c_ptr = ({c_frag_attrs.ptr_type}*)(c_ptr_in);"

    source_code = f"""
__forceinline__ __device__ void {func_name}(void* d_ptr_in, void* a_ptr_in, void* b_ptr_in{c_signature}) {{
    {d_frag_attrs.ptr_type}* d_ptr = ({d_frag_attrs.ptr_type}*)(d_ptr_in);
    {a_frag_attrs.ptr_type}* a_ptr = ({a_frag_attrs.ptr_type}*)(a_ptr_in);
    {b_frag_attrs.ptr_type}* b_ptr = ({b_frag_attrs.ptr_type}*)(b_ptr_in);
    {c_ptr_cast}

    __asm__ __volatile__(
        "mma.sync.aligned.{shape}.{a_layout}.{b_layout}{saturate_inst}{d_type_inst}{a_type_inst}{b_type_inst}{c_type_inst}{bit_op_inst}"
        "{args_template};\\n"
        : {d_inputs}
        : {a_inputs}, {b_inputs}, {c_inputs}
    );
}}
"""  # noqa: E501
    if no_c_ptr:
        return cuda_func_call(func_name, d_ptr, a_ptr, b_ptr, source_code=source_code)
    else:
        return cuda_func_call(func_name, d_ptr, a_ptr, b_ptr, c_ptr, source_code=source_code)


@register_codegen("ptx_ldmatrix")
def codegen_ptx_ldmatrix(trans, num, dtype, local_ptr, smem_ptr):
    trans = bool(trans)
    num = int(num)
    dtype = parse_str(dtype)

    assert num == 1 or num == 2 or num == 4
    func_name = f"ptx_ldmatrix_{num}_{dtype.replace('.', '')}_{int(trans)}"

    shape = ".m8n8"  # sm100 supports m16n16, m8n16 for subbyte types
    trans_instr = ".trans" if trans else ""

    if dtype == ".b16":
        num_b32 = num
    elif dtype == ".b8":
        num_b32 = num // 2
    else:
        raise ValueError(f"Invalid dtype: {dtype}")

    arg_templates = ", ".join([f"%{i}" for i in range(num_b32)])
    operands = ", ".join([f'"=r"(reg[{i}])' for i in range(num_b32)])

    source_code = f"""
__forceinline__ __device__ void {func_name}(void* local_ptr, void* smem_ptr) {{
    uint32_t* reg = (uint32_t*)local_ptr;
    unsigned int addr = __cvta_generic_to_shared(smem_ptr);
    __asm__ __volatile__(
      "ldmatrix.sync.aligned{shape}.x{num}{trans_instr}.shared{dtype} {{{arg_templates}}}, [%{num_b32}];\\n"
      : {operands}
      : "r"(addr)
    );
}}
"""  # noqa: E501
    return cuda_func_call(func_name, local_ptr, smem_ptr, source_code=source_code)


@register_codegen("ptx_stmatrix")
def codegen_ptx_stmatrix(num, trans, smem_ptr, local_ptr):
    num = int(num)
    trans = bool(trans)
    dtype = ".b16"

    assert num == 1 or num == 2 or num == 4
    func_name = f"ptx_stmatrix_{num}_{int(trans)}"

    shape = ".m8n8"  # sm100 supports m16n16, m8n16 for subbyte types
    trans_instr = ".trans" if trans else ""

    num_b32 = num

    arg_templates = ", ".join([f"%{i}" for i in range(num_b32)])
    operands = ", ".join([f'"r"(reg[{i}])' for i in range(num_b32)])

    source_code = f"""
__forceinline__ __device__ void {func_name}(void* smem_ptr, void* local_ptr) {{
    uint32_t* reg = (uint32_t*)local_ptr;
    unsigned int addr = __cvta_generic_to_shared(smem_ptr);
    __asm__ __volatile__(
      "stmatrix.sync.aligned{shape}.x{num}{trans_instr}.shared{dtype} [%{num_b32}], {{{arg_templates}}};\\n"
      :
      : {operands}, "r"(addr)
    );
}}
"""  # noqa: E501
    return cuda_func_call(func_name, smem_ptr, local_ptr, source_code=source_code)
