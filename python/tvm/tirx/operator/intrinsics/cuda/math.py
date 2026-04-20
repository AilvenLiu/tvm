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
# pylint: disable=redefined-builtin, invalid-name, too-many-arguments
"""PTX Math operations using inline assembly."""

from tvm.tirx.op import cuda_func_call

from .registry import register_codegen
from .utils import parse_str

# ptx_{exp2,rcp,reduce3_{max,min}_f32} are schema-declared in misc_ptx.py.


def _codegen_packed_f32x2(op, rounding_mode, has_c, a1, a2, b1, b2, *rest):
    """Shared factory for {add,sub,mul,fma}.{rz}.ftz.f32x2 (sm_100a+).

    ``has_c`` selects the ternary (fma) variant. Last arg in ``rest`` is always
    the output address; for ``has_c=True`` the two preceding args are c1, c2.
    """
    rounding_mode = parse_str(rounding_mode)
    func_name = f"tvm_builtin_ptx_{op}_packed_{rounding_mode}_f32x2"
    if has_c:
        c1, c2, d_addr = rest
        c_param = ", float c1, float c2"
        c_body = "    float2 c = make_float2(c1, c2);\n"
        asm_operands = '"=l"(reinterpret_cast<uint64_t&>(d_p[0])) : "l"(reinterpret_cast<uint64_t&>(a)), "l"(reinterpret_cast<uint64_t&>(b)), "l"(reinterpret_cast<uint64_t&>(c))'  # noqa: E501
        asm_template = f"{op}.{rounding_mode}.ftz.f32x2 %0, %1, %2, %3;"
        call_args = (a1, a2, b1, b2, c1, c2, d_addr)
    else:
        (d_addr,) = rest
        c_param, c_body = "", ""
        asm_operands = '"=l"(reinterpret_cast<uint64_t&>(d_p[0])) : "l"(reinterpret_cast<uint64_t&>(a)), "l"(reinterpret_cast<uint64_t&>(b))'  # noqa: E501
        asm_template = f"{op}.{rounding_mode}.ftz.f32x2 %0, %1, %2;"
        call_args = (a1, a2, b1, b2, d_addr)

    source_code = f"""
__forceinline__ __device__ void {func_name}(float a1, float a2, float b1, float b2{c_param}, float* d) {{
    float2* d_p = (float2*) d;
    float2 a = make_float2(a1, a2);
    float2 b = make_float2(b1, b2);
{c_body}    asm volatile("{asm_template}" : {asm_operands});
}}
"""  # noqa: E501
    return cuda_func_call(func_name, *call_args, source_code=source_code)


@register_codegen("ptx_add_packed_f32x2")
def codegen_ptx_add_packed_f32x2(a1, a2, b1, b2, d_addr, rounding_mode="rz"):
    return _codegen_packed_f32x2("add", rounding_mode, False, a1, a2, b1, b2, d_addr)


@register_codegen("ptx_sub_packed_f32x2")
def codegen_ptx_sub_packed_f32x2(a1, a2, b1, b2, d_addr, rounding_mode="rz"):
    return _codegen_packed_f32x2("sub", rounding_mode, False, a1, a2, b1, b2, d_addr)


@register_codegen("ptx_mul_packed_f32x2")
def codegen_ptx_mul_packed_f32x2(a1, a2, b1, b2, d_addr, rounding_mode="rz"):
    return _codegen_packed_f32x2("mul", rounding_mode, False, a1, a2, b1, b2, d_addr)


@register_codegen("ptx_fma_packed_f32x2")
def codegen_ptx_fma_packed_f32x2(a1, a2, b1, b2, c1, c2, d_addr, rounding_mode="rz"):
    return _codegen_packed_f32x2("fma", rounding_mode, True, a1, a2, b1, b2, c1, c2, d_addr)
