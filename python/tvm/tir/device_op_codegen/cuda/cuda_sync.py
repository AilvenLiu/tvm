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
"""CUDA C++ miscellaneous operations."""
import tvm
from tvm import DataType
from tvm.tir.op import cuda_func_call

from .registry import register_codegen
from .utils import parse_str


@register_codegen("cuda_atomic_add")
def codegen_cuda_atomic_add(res_addr, value):
    func_name = "tvm_builtin_cuda_atomic_add"
    source_code = f"""
template <typename T>
__forceinline__ __device__ T {func_name}(T* addr, T value) {{
    return atomicAdd(addr, value);
}}
"""
    assert isinstance(value, tvm.tir.PrimExpr)
    return cuda_func_call(
        func_name, res_addr, value, source_code=source_code, return_type=value.dtype
    )


@register_codegen("cuda_thread_fence")
def codegen_cuda_thread_fence():
    func_name = "tvm_builtin_cuda_thread_fence"
    source_code = f"""
__forceinline__ __device__ void {func_name}() {{
    __threadfence();
}}
"""
    return cuda_func_call(func_name, source_code=source_code)


@register_codegen("cuda_warp_sync")
def codegen_cuda_warp_sync():
    func_name = "tvm_builtin_cuda_warp_sync"
    source_code = f"""
__forceinline__ __device__ void {func_name}() {{
    __syncwarp();
}}
"""
    return cuda_func_call(func_name, source_code=source_code)


@register_codegen("cuda_warpgroup_sync")
def codegen_cuda_warpgroup_sync(name_bar_id):
    func_name = "tvm_builtin_cuda_warpgroup_sync"
    source_code = f"""
__forceinline__ __device__ void {func_name}(int name_bar_id) {{
    asm volatile("bar.sync %0, 128;" : : "r"(name_bar_id));
}}
"""
    return cuda_func_call(func_name, name_bar_id, source_code=source_code)


@register_codegen("cuda_cta_sync")
def codegen_cuda_cta_sync():
    func_name = "tvm_builtin_cuda_cta_sync"
    source_code = f"""
__forceinline__ __device__ void {func_name}() {{
    __syncthreads();
}}
"""
    return cuda_func_call(func_name, source_code=source_code)


@register_codegen("cuda_grid_sync")
def codegen_cuda_grid_sync():
    func_name = "tvm_builtin_cuda_grid_sync"
    source_code = f"""
__forceinline__ __device__ void {func_name}() {{
    namespace cg = cooperative_groups;
    cg::this_grid().sync();
}}
"""
    return cuda_func_call(func_name, source_code=source_code), ["cooperative_groups"]


@register_codegen("cuda_cluster_sync")
def codegen_cuda_cluster_sync():
    func_name = "tvm_builtin_cuda_cluster_sync"
    source_code = f"""
__forceinline__ __device__ void {func_name}() {{
    asm("barrier.cluster.arrive.aligned;");
    asm("barrier.cluster.wait.aligned;");
}}
"""
    return cuda_func_call(func_name, source_code=source_code)


@register_codegen("cuda_half2float")
def codegen_cuda_half2float(src):
    func_name = "tvm_builtin_cuda_half2float"
    source_code = f"""
__forceinline__ __device__ float {func_name}(half src) {{
    return __half2float(src);
}}
"""
    return cuda_func_call(func_name, src, source_code=source_code, return_type="float32")


@register_codegen("cuda_bfloat162float")
def codegen_cuda_bfloat162float(src):
    func_name = "tvm_builtin_cuda_bfloat162float"
    source_code = f"""
__forceinline__ __device__ float {func_name}(nv_bfloat16 src) {{
    return __bfloat162float(src);
}}
"""
    return cuda_func_call(func_name, src, source_code=source_code, return_type="float32")


@register_codegen("cuda_float22half2")
def codegen_cuda_float22half2(dst, src):
    func_name = "tvm_builtin_cuda_float22half2"
    source_code = f"""
__forceinline__ __device__ void {func_name}(void* dst, void* src) {{
    half2* dst_p = (half2*) dst;
    float2* src_p = (float2*) src;
    *dst_p = __float22half2_rn(*src_p);
}}
"""
    return cuda_func_call(func_name, dst, src, source_code=source_code)


@register_codegen("cuda_trap_when_assert_failed")
def codegen_cuda_trap_when_assert_failed(cond):
    func_name = "tvm_builtin_cuda_trap_when_assert_failed"
    source_code = f"""
__forceinline__ __device__ void {func_name}(bool cond) {{
    do {{
        if (not (cond))
            asm("trap;");
    }} while (0);
}}
"""
    return cuda_func_call(func_name, cond, source_code=source_code)


@register_codegen("cuda_runtime_instr_desc")
def codegen_cuda_runtime_instr_desc(desc, sf_id):
    func_name = "tvm_builtin_cuda_runtime_instr_desc"
    source_code = f"""
__forceinline__ __device__ void {func_name}(uint32_t* desc, const uint32_t& sf_id) {{
    *desc = (*desc & ~0x60000030) | ((sf_id << 29) | (sf_id << 4));
}}
"""
    return cuda_func_call(func_name, desc, sf_id, source_code=source_code)


@register_codegen("cuda_half8tofloat8")
def codegen_cuda_half8tofloat8(src_addr, dst_addr):
    func_name = "tvm_builtin_cuda_half8tofloat8"
    source_code = f"""
__forceinline__ __device__ void {func_name}(void* src_addr, void* dst_addr) {{
    half2* source = (half2*) src_addr;
    float2* dest = (float2*) dst_addr;
    for (int i = 0; i < 4; i++) {{
        dest[i] = __half22float2(source[i]);
    }}
}}
"""
    return cuda_func_call(func_name, src_addr, dst_addr, source_code=source_code)


@register_codegen("cuda_float8tohalf8")
def codegen_cuda_float8tohalf8(src_addr, dst_addr):
    func_name = "tvm_builtin_cuda_float8tohalf8"
    source_code = f"""
__forceinline__ __device__ void {func_name}(void* src_addr, void* dst_addr) {{
    float2* source = (float2*) src_addr;
    half2* dest = (half2*) dst_addr;
    for (int i = 0; i < 4; i++) {{
        dest[i] = __float22half2_rn(source[i]);
    }}
}}
"""
    return cuda_func_call(func_name, src_addr, dst_addr, source_code=source_code)


@register_codegen("cuda_syncthreads_and")
def codegen_cuda_syncthreads_and(predicate):
    func_name = "tvm_builtin_cuda_syncthreads_and"
    source_code = f"""
__forceinline__ __device__ int {func_name}(int predicate) {{
    return __syncthreads_and(predicate);
}}
"""
    return cuda_func_call(func_name, predicate, source_code=source_code, return_type="int32")


@register_codegen("cuda_syncthreads_or")
def codegen_cuda_syncthreads_or(predicate):
    func_name = "tvm_builtin_cuda_syncthreads_or"
    source_code = f"""
__forceinline__ __device__ int {func_name}(int predicate) {{
    return __syncthreads_or(predicate);
}}
"""
    return cuda_func_call(func_name, predicate, source_code=source_code, return_type="int32")


@register_codegen("cuda_nano_sleep")
def codegen_cuda_nano_sleep(time):
    func_name = "tvm_builtin_cuda_nano_sleep"
    source_code = f"""
__forceinline__ __device__ void {func_name}(uint64_t time) {{
    __nanosleep(time);
}}
"""
    return cuda_func_call(func_name, time, source_code=source_code)


@register_codegen("cuda_atomic_cas")
def codegen_cuda_atomic_cas(ptr, old_val, new_val):
    func_name = "tvm_builtin_cuda_atomic_cas"
    source_code = f"""
template <typename T>
__forceinline__ __device__ T {func_name}(T* address, T compare, T val) {{
    return atomicCAS(address, compare, val);
}}
"""
    return cuda_func_call(
        func_name, ptr, old_val, new_val, source_code=source_code, return_type=old_val.dtype
    )


@register_codegen("cuda_printf")
def codegen_cuda_printf(fmt, *args):
    func_name = "tvm_builtin_cuda_printf"
    if isinstance(fmt, tvm.tir.StringImm):
        fmt = fmt.value
    fmt = repr(fmt)[1:-1]
    source_code = f"""
template<typename... Args>
__forceinline__ __device__ void {func_name}(const char* fmt, Args... args) {{
    printf(fmt, args...);
}}
"""
    return cuda_func_call(func_name, fmt, *args, source_code=source_code)


@register_codegen("cuda_ldg")
def codegen_cuda_ldg(addr, dtype):
    dtype = DataType(parse_str(dtype))
    func_name = "tvm_builtin_cuda_ldg"
    source_code = f"""
template <typename T>
__forceinline__ __device__ T {func_name}(T* src) {{
    return __ldg(src);
}}
"""
    return cuda_func_call(func_name, addr, source_code=source_code, return_type=dtype)


@register_codegen("cuda_get_tmem_addr")
def codegen_cuda_get_tmem_addr(addr, row_offset, col_offset):
    func_name = "tvm_builtin_cuda_get_tmem_addr"
    source_code = f"""
__forceinline__ __device__ uint32_t {func_name}(uint32_t addr, int row_offset, int col_offset) {{
    return get_tmem_addr(addr, row_offset, col_offset);
}}
"""
    return cuda_func_call(
        func_name, addr, row_offset, col_offset, source_code=source_code, return_type="uint32"
    ), ["get_tmem_addr"]


@register_codegen("cuda_reduce3_max_f32")
def codegen_cuda_reduce3_max_f32(a, b, c):
    """3-input max using PTX max.f32 instruction (sm_100a+)."""
    func_name = "tvm_builtin_cuda_reduce3_max_f32"
    source_code = f"""
__forceinline__ __device__ float {func_name}(float a, float b, float c) {{
    float d;
    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(d) : "f"(a), "f"(b), "f"(c));
    return d;
}}
"""
    return cuda_func_call(func_name, a, b, c, source_code=source_code, return_type="float32")


@register_codegen("cuda_reduce3_min_f32")
def codegen_cuda_reduce3_min_f32(a, b, c):
    """3-input min using PTX min.f32 instruction (sm_100a+)."""
    func_name = "tvm_builtin_cuda_reduce3_min_f32"
    source_code = f"""
__forceinline__ __device__ float {func_name}(float a, float b, float c) {{
    float d;
    asm volatile("min.f32 %0, %1, %2, %3;" : "=f"(d) : "f"(a), "f"(b), "f"(c));
    return d;
}}
"""
    return cuda_func_call(func_name, a, b, c, source_code=source_code, return_type="float32")


@register_codegen("cuda_add_packed_f32x2")
def codegen_cuda_add_packed_f32x2(a1, a2, b1, b2, d_addr, rounding_mode="rz"):
    """Packed f32x2 add using PTX add.{rounding_mode}.ftz.f32x2 instruction (sm_100a+)."""
    rounding_mode = parse_str(rounding_mode)
    func_name = f"tvm_builtin_cuda_add_packed_{rounding_mode}_f32x2"
    source_code = f"""
__forceinline__ __device__ void {func_name}(float a1, float a2, float b1, float b2, float* d) {{
    float2* d_p = (float2*) d;
    float2 a = make_float2(a1, a2);
    float2 b = make_float2(b1, b2);
    asm volatile("add.{rounding_mode}.ftz.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(d_p[0])) : "l"(reinterpret_cast<uint64_t&>(a)), "l"(reinterpret_cast<uint64_t&>(b)));
}}
"""
    return cuda_func_call(func_name, a1, a2, b1, b2, d_addr, source_code=source_code)
