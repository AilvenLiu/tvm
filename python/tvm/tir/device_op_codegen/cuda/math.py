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
from tvm.tir.op import cuda_func_call

from .registry import register_codegen
from .utils import parse_str


@register_codegen("ptx_exp2")
def codegen_ptx_exp2(x):
    """Fast exp2 approximation using PTX ex2.approx.ftz.f32 instruction."""
    func_name = "tvm_builtin_ptx_exp2"
    source_code = f"""
__forceinline__ __device__ float {func_name}(float x) {{
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}}
"""
    return cuda_func_call(func_name, x, source_code=source_code, return_type="float32")


@register_codegen("ptx_rcp")
def codegen_ptx_rcp(x):
    """Fast reciprocal approximation using PTX rcp.approx.ftz.f32 instruction."""
    func_name = "tvm_builtin_ptx_rcp"
    source_code = f"""
__forceinline__ __device__ float {func_name}(float x) {{
    float y;
    asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}}
"""
    return cuda_func_call(func_name, x, source_code=source_code, return_type="float32")


@register_codegen("ptx_reduce3_max_f32")
def codegen_ptx_reduce3_max_f32(a, b, c):
    """3-input max using PTX max.f32 instruction (sm_100a+)."""
    func_name = "tvm_builtin_ptx_reduce3_max_f32"
    source_code = f"""
__forceinline__ __device__ float {func_name}(float a, float b, float c) {{
    float d;
    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(d) : "f"(a), "f"(b), "f"(c));
    return d;
}}
"""
    return cuda_func_call(func_name, a, b, c, source_code=source_code, return_type="float32")


@register_codegen("ptx_reduce3_min_f32")
def codegen_ptx_reduce3_min_f32(a, b, c):
    """3-input min using PTX min.f32 instruction (sm_100a+)."""
    func_name = "tvm_builtin_ptx_reduce3_min_f32"
    source_code = f"""
__forceinline__ __device__ float {func_name}(float a, float b, float c) {{
    float d;
    asm volatile("min.f32 %0, %1, %2, %3;" : "=f"(d) : "f"(a), "f"(b), "f"(c));
    return d;
}}
"""
    return cuda_func_call(func_name, a, b, c, source_code=source_code, return_type="float32")


@register_codegen("ptx_add_packed_f32x2")
def codegen_ptx_add_packed_f32x2(a1, a2, b1, b2, d_addr, rounding_mode="rz"):
    """Packed f32x2 add using PTX add.{rounding_mode}.ftz.f32x2 instruction (sm_100a+)."""
    rounding_mode = parse_str(rounding_mode)
    func_name = f"tvm_builtin_ptx_add_packed_{rounding_mode}_f32x2"
    source_code = f"""
__forceinline__ __device__ void {func_name}(float a1, float a2, float b1, float b2, float* d) {{
    float2* d_p = (float2*) d;
    float2 a = make_float2(a1, a2);
    float2 b = make_float2(b1, b2);
    asm volatile("add.{rounding_mode}.ftz.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(d_p[0])) : "l"(reinterpret_cast<uint64_t&>(a)), "l"(reinterpret_cast<uint64_t&>(b)));
}}
"""
    return cuda_func_call(func_name, a1, a2, b1, b2, d_addr, source_code=source_code)


@register_codegen("ptx_sub_packed_f32x2")
def codegen_ptx_sub_packed_f32x2(a1, a2, b1, b2, d_addr, rounding_mode="rz"):
    """Packed f32x2 subtract using PTX sub.{rounding_mode}.ftz.f32x2 instruction (sm_100a+)."""
    rounding_mode = parse_str(rounding_mode)
    func_name = f"tvm_builtin_ptx_sub_packed_{rounding_mode}_f32x2"
    source_code = f"""
__forceinline__ __device__ void {func_name}(float a1, float a2, float b1, float b2, float* d) {{
    float2* d_p = (float2*) d;
    float2 a = make_float2(a1, a2);
    float2 b = make_float2(b1, b2);
    asm volatile("sub.{rounding_mode}.ftz.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(d_p[0])) : "l"(reinterpret_cast<uint64_t&>(a)), "l"(reinterpret_cast<uint64_t&>(b)));
}}
"""
    return cuda_func_call(func_name, a1, a2, b1, b2, d_addr, source_code=source_code)


@register_codegen("ptx_mul_packed_f32x2")
def codegen_ptx_mul_packed_f32x2(a1, a2, b1, b2, d_addr, rounding_mode="rz"):
    """Packed f32x2 multiply using PTX mul.{rounding_mode}.ftz.f32x2 instruction (sm_100a+)."""
    rounding_mode = parse_str(rounding_mode)
    func_name = f"tvm_builtin_ptx_mul_packed_{rounding_mode}_f32x2"
    source_code = f"""
__forceinline__ __device__ void {func_name}(float a1, float a2, float b1, float b2, float* d) {{
    float2* d_p = (float2*) d;
    float2 a = make_float2(a1, a2);
    float2 b = make_float2(b1, b2);
    asm volatile("mul.{rounding_mode}.ftz.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(d_p[0])) : "l"(reinterpret_cast<uint64_t&>(a)), "l"(reinterpret_cast<uint64_t&>(b)));
}}
"""
    return cuda_func_call(func_name, a1, a2, b1, b2, d_addr, source_code=source_code)


@register_codegen("ptx_fma_packed_f32x2")
def codegen_ptx_fma_packed_f32x2(a1, a2, b1, b2, c1, c2, d_addr, rounding_mode="rz"):
    """Packed f32x2 FMA using PTX fma.{rounding_mode}.ftz.f32x2 instruction (sm_100a+)."""
    rounding_mode = parse_str(rounding_mode)
    func_name = f"tvm_builtin_ptx_fma_packed_{rounding_mode}_f32x2"
    source_code = f"""
__forceinline__ __device__ void {func_name}(float a1, float a2, float b1, float b2, float c1, float c2, float* d) {{
    float2* d_p = (float2*) d;
    float2 a = make_float2(a1, a2);
    float2 b = make_float2(b1, b2);
    float2 c = make_float2(c1, c2);
    asm volatile("fma.{rounding_mode}.ftz.f32x2 %0, %1, %2, %3;" : "=l"(reinterpret_cast<uint64_t&>(d_p[0])) : "l"(reinterpret_cast<uint64_t&>(a)), "l"(reinterpret_cast<uint64_t&>(b)), "l"(reinterpret_cast<uint64_t&>(c)));
}}
"""
    return cuda_func_call(func_name, a1, a2, b1, b2, c1, c2, d_addr, source_code=source_code)
