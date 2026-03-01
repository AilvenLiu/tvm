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
"""PTX Miscellaneous operations."""

from tvm.tir.op import cuda_func_call

from .registry import register_codegen
from .utils import parse_str


@register_codegen("ptx_setmaxnreg")
def codegen_ptx_setmaxnreg(inc, nreg):
    inc = bool(inc)
    nreg = int(nreg)
    action = "inc" if inc else "dec"
    func_name = f"tvm_builtin_ptx_setmaxnreg_{action}_{nreg}"
    source_code = f"""
__forceinline__ __device__ void {func_name}() {{
    asm volatile( "setmaxnreg.{action}.sync.aligned.u32 %0;\\n" : : "n"({nreg}) );
}}
"""
    return cuda_func_call(func_name, source_code=source_code)


@register_codegen("ptx_ld_global_acquire")
def codegen_ptx_ld_global_acquire(res, addr):
    dtype = str(res.dtype)
    if dtype == "uint32":
        dtype_str, type_str, specifier = "uint32_t", "b32", "r"
    elif dtype == "int32":
        dtype_str, type_str, specifier = "int32_t", "b32", "r"
    elif dtype == "uint64":
        dtype_str, type_str, specifier = "uint64_t", "b64", "l"
    elif dtype == "int64":
        dtype_str, type_str, specifier = "int64_t", "b64", "l"
    else:
        raise ValueError(f"Unsupported data type for ld.global.acquire: {dtype}")

    func_name = f"tvm_builtin_ptx_ld_global_acquire_{type_str}"
    source_code = f"""
__forceinline__ __device__ void {func_name}({dtype_str}& res,{dtype_str}* addr) {{
  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  asm volatile ("ld.global.acquire.gpu.{type_str} %0, [%1];\\n" : "={specifier}"(res) : "l"(addr));
  #else
  asm volatile ("ld.global.cg.{type_str} %0, [%1];\\n" : "={specifier}"(res) : "l"(addr));
  #endif
}}
"""
    return cuda_func_call(func_name, res, addr, source_code=source_code)


@register_codegen("ptx_map_shared_rank")
def codegen_ptx_map_shared_rank(ptr, rank):
    func_name = "tvm_builtin_ptx_map_shared_rank"
    source_code = f"""
__forceinline__ __device__ uint64_t {func_name}(void* addr, uint32_t rank) {{
    uint64_t result;
    asm volatile("mapa.u64  %0, %1, %2;\\n"
                : "=l"(result)
                : "l"(reinterpret_cast<uint64_t>(addr)), "r"(rank));
    return result;
}}
"""
    return cuda_func_call(func_name, ptr, rank, source_code=source_code, return_type="uint64")


@register_codegen("ptx_fetch_register")
def codegen_ptx_fetch_register(bits, reg):
    bits = int(bits)
    reg = parse_str(reg)

    if bits not in [32, 64]:
        raise ValueError(f"Only support 32/64 bits for ptx_fetch_register, but got {bits}.")

    func_name_safe_reg = reg.replace(".", "_")

    func_name = f"tvm_builtin_ptx_fetch_register_{func_name_safe_reg}"
    source_code = f"""
__forceinline__ __device__ int{bits}_t {func_name}() {{
  uint{bits}_t x;
  asm volatile("mov.u{bits} %0, %{reg};\\n" : "=r"(x) : "r"(reg));
  return (int{bits}_t)x;
}}
"""
    return cuda_func_call(func_name, source_code=source_code, return_type=f"int{bits}")


@register_codegen("ptx_any_sync")
def codegen_ptx_any_sync(mask, pred):
    """Warp-wide any predicate using __any_sync intrinsic."""
    func_name = "tvm_builtin_ptx_any_sync"
    source_code = f"""
__forceinline__ __device__ int {func_name}(unsigned mask, int pred) {{
    return __any_sync(mask, pred);
}}
"""
    return cuda_func_call(func_name, mask, pred, source_code=source_code, return_type="int32")
