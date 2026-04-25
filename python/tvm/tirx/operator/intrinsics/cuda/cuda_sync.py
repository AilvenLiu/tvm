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
"""CUDA C++ helper intrinsics (no inline PTX asm).

Most ops are declared via ``cuda_helper_intrinsic(...)`` at the top. Ops with
dtype-string → return-type dispatch (``cuda_ldg``) or variadic templated
bodies (``cuda_printf``) are hand-written below.
"""

import tvm
from tvm import DataType
from tvm.tirx.op import cuda_func_call

from .._schema import cuda_helper_intrinsic
from .registry import register_codegen
from .utils import parse_str

# =============================================================================
# Schema-declared helpers.
# =============================================================================

# -----------------------------------------------------------------------------
# Zero-arg void helpers — direct wrappers around CUDA builtins.
# -----------------------------------------------------------------------------
cuda_helper_intrinsic(
    op_name="cuda_thread_fence",
    c_body="__threadfence();",
)
cuda_helper_intrinsic(
    op_name="cuda_warp_sync",
    c_body="__syncwarp();",
)
cuda_helper_intrinsic(
    op_name="cuda_cta_sync",
    c_body="__syncthreads();",
)
cuda_helper_intrinsic(
    op_name="cuda_grid_sync",
    c_body="namespace cg = cooperative_groups;\n    cg::this_grid().sync();",
    extra_deps=("cooperative_groups",),
)
cuda_helper_intrinsic(
    op_name="cuda_cluster_sync",
    c_body='asm("barrier.cluster.arrive.aligned;");\n    asm("barrier.cluster.wait.aligned;");',
)
cuda_helper_intrinsic(
    op_name="cuda_thread_rank",
    c_body="namespace cg = cooperative_groups;\n    return cg::this_thread_block().thread_rank();",
    return_type="int",
    tvm_return_type="int32",
    extra_deps=("cooperative_groups",),
)

# -----------------------------------------------------------------------------
# Single-operand helpers.
# -----------------------------------------------------------------------------
cuda_helper_intrinsic(
    op_name="cuda_warpgroup_sync",
    c_signature="(int name_bar_id)",
    c_body='asm volatile("bar.sync %0, 128;" : : "r"(name_bar_id));',
)
cuda_helper_intrinsic(
    op_name="cuda_nano_sleep",
    c_signature="(uint64_t time)",
    c_body="__nanosleep(time);",
)
cuda_helper_intrinsic(
    op_name="cuda_half2float",
    c_signature="(half src)",
    c_body="return __half2float(src);",
    return_type="float",
    tvm_return_type="float32",
)
cuda_helper_intrinsic(
    op_name="cuda_bfloat162float",
    c_signature="(nv_bfloat16 src)",
    c_body="return __bfloat162float(src);",
    return_type="float",
    tvm_return_type="float32",
)
cuda_helper_intrinsic(
    op_name="cuda_syncthreads_and",
    c_signature="(int predicate)",
    c_body="return __syncthreads_and(predicate);",
    return_type="int",
    tvm_return_type="int32",
)
cuda_helper_intrinsic(
    op_name="cuda_syncthreads_or",
    c_signature="(int predicate)",
    c_body="return __syncthreads_or(predicate);",
    return_type="int",
    tvm_return_type="int32",
)
cuda_helper_intrinsic(
    op_name="cuda_trap_when_assert_failed",
    c_signature="(bool cond)",
    c_body='do {\n        if (not (cond))\n            asm("trap;");\n    } while (0);',
)

# -----------------------------------------------------------------------------
# Two-operand void helpers (pointer conversions).
# -----------------------------------------------------------------------------
cuda_helper_intrinsic(
    op_name="cuda_float22half2",
    c_signature="(void* dst, void* src)",
    c_body=(
        "half2* dst_p = (half2*) dst;\n"
        "    float2* src_p = (float2*) src;\n"
        "    *dst_p = __float22half2_rn(*src_p);"
    ),
)
cuda_helper_intrinsic(
    op_name="cuda_half8tofloat8",
    c_signature="(void* src_addr, void* dst_addr)",
    c_body=(
        "half2* source = (half2*) src_addr;\n"
        "    float2* dest = (float2*) dst_addr;\n"
        "    for (int i = 0; i < 4; i++) {\n"
        "        dest[i] = __half22float2(source[i]);\n"
        "    }"
    ),
)
cuda_helper_intrinsic(
    op_name="cuda_float8tohalf8",
    c_signature="(void* src_addr, void* dst_addr)",
    c_body=(
        "float2* source = (float2*) src_addr;\n"
        "    half2* dest = (half2*) dst_addr;\n"
        "    for (int i = 0; i < 4; i++) {\n"
        "        dest[i] = __float22half2_rn(source[i]);\n"
        "    }"
    ),
)
cuda_helper_intrinsic(
    op_name="cuda_runtime_instr_desc",
    c_signature="(uint32_t* desc, const uint32_t& sf_id)",
    c_body="*desc = (*desc & ~0x60000030) | ((sf_id << 29) | (sf_id << 4));",
)

# -----------------------------------------------------------------------------
# Templated helpers — `T` deduced from call site.
# -----------------------------------------------------------------------------
cuda_helper_intrinsic(
    op_name="cuda_atomic_add",
    helper_name="tvm_builtin_cuda_atomic_add",
    c_signature="(T* addr, T value)",
    c_body="return atomicAdd(addr, value);",
    return_type="T",
    templated=True,
    tvm_return_type=lambda _addr, value: value.dtype,
)
cuda_helper_intrinsic(
    op_name="cuda_atomic_cas",
    helper_name="tvm_builtin_cuda_atomic_cas",
    c_signature="(T* address, T compare, T val)",
    c_body="return atomicCAS(address, compare, val);",
    return_type="T",
    templated=True,
    tvm_return_type=lambda _p, old, _n: old.dtype,
)

# -----------------------------------------------------------------------------
# Three-operand — tmem address helper (has an extra_deps tag).
# -----------------------------------------------------------------------------
cuda_helper_intrinsic(
    op_name="cuda_get_tmem_addr",
    c_signature="(uint32_t addr, int row_offset, int col_offset)",
    c_body="return get_tmem_addr(addr, row_offset, col_offset);",
    return_type="uint32_t",
    tvm_return_type="uint32",
    extra_deps=("get_tmem_addr",),
)


# =============================================================================
# Hand-written helpers.
# =============================================================================


@register_codegen("cuda_printf")
def codegen_cuda_printf(fmt, *args):
    func_name = "tvm_builtin_cuda_printf"
    if isinstance(fmt, tvm.tirx.StringImm):
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
