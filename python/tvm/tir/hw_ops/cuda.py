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
"""HW level ops for CUDA, along with its codegen ruls"""
# pylint: disable=missing-function-docstring
import functools

import tvm.ffi
from tvm.tir.op import cuda_func_call


CODEGEN_REGISTRY = {}


@tvm.ffi.register_func("tir.hw_ops.cuda.get_codegen")
def get_codegen(op):
    """get the codegen function for a given op"""
    return CODEGEN_REGISTRY.get(op, None)


def register_codegen(op, backend="cuda"):
    """register a codegen function for a given op
    The codegen function should return a cuda_func_call statement
    """

    def decorator(func):

        @functools.wraps(func)
        def wrapper(arg_list):
            return func(*arg_list)  # pylint: disable=not-callable

        CODEGEN_REGISTRY["tir." + op] = wrapper
        return wrapper

    return decorator


@register_codegen("timer_init_cuda")
def codegen_timer_init_cuda(profiler_buffer, profiler_tag, profiler_write_offset):
    func_name = "tvm_builtin_timer_init_cuda"
    source_code = R"""
__forceinline__ __device__ void {func_name}(void* profiler_buffer, void* profiler_tag, void* profiler_write_offset) {{
    // timer init
    const uint32_t NBLOCKS = (uint32_t)(gridDim.x * gridDim.y * gridDim.z);
    const uint32_t BLOCK_IDX = (uint32_t)((blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x);
    const uint32_t NGROUPS = (uint32_t)(blockDim.x >> 7);
    const uint32_t WG_IDX = (uint32_t)(threadIdx.x >> 7);
    const uint32_t BLOCK_GROUP_IDX = BLOCK_IDX * NGROUPS + WG_IDX;
    if ((blockIdx.x == 0) && (blockIdx.y == 0) && (blockIdx.z == 0) && (threadIdx.x == 0)) {
        profiler_buffer[0] = ((uint64_t)NGROUPS << 32) | NBLOCKS;
    }
    profiler_write_offset[0] = 1 + BLOCK_GROUP_IDX;
    profiler_tag[0] = (uint64_t)BLOCK_GROUP_IDX << 12;
}}
"""
    source_code = source_code.format(func_name=func_name)
    return cuda_func_call(
        func_name, profiler_buffer, profiler_tag, profiler_write_offset, source_code=source_code
    )


@register_codegen("timer_start_cuda")
def codegen_timer_start_cuda(
    event_type, profiler_buffer, profiler_tag, profiler_write_offset, profiler_write_stride
):
    func_name = "tvm_builtin_timer_start_cuda"
    source_code = R"""
#ifndef TVM_BUILTIN_GET_TIMESTAMP_ASSEMBLY
#define TVM_BUILTIN_GET_TIMESTAMP_ASSEMBLY
__forceinline__ __device__ uint32_t tvm_builtin_get_timestamp() {
  volatile uint32_t ret;
  asm volatile("mov.u32 %0, %globaltimer_lo;" : "=r"(ret));
  return ret;
}
#endif // TVM_BUILTIN_GET_TIMESTAMP_ASSEMBLY

__forceinline__ __device__ void {func_name}(int event_type, void* profiler_buffer, void* profiler_tag, void* profiler_write_offset, int profiler_write_stride) {{
    // timer start
    if (threadIdx.x % 128 == 0) {
        profiler_tag[profiler_write_offset[0]] = ((uint64_t)tvm_builtin_get_timestamp() << 32) | (profiler_write_offset[0] | (uint32_t)event_type << 2 | 0x0);
        profiler_write_offset[0] += profiler_write_stride;
    }
    __threadfence_block();
}}
"""
    source_code = source_code.format(func_name=func_name)
    return cuda_func_call(
        func_name,
        event_type,
        profiler_buffer,
        profiler_tag,
        profiler_write_offset,
        profiler_write_stride,
        source_code=source_code,
    )


@register_codegen("timer_end_cuda")
def codegen_timer_end_cuda(
    event_type, profiler_buffer, profiler_tag, profiler_write_offset, profiler_write_stride
):
    func_name = "tvm_builtin_timer_end_cuda"
    source_code = R"""
#ifndef TVM_BUILTIN_GET_TIMESTAMP_ASSEMBLY
#define TVM_BUILTIN_GET_TIMESTAMP_ASSEMBLY
__forceinline__ __device__ uint32_t tvm_builtin_get_timestamp() {
  volatile uint32_t ret;
  asm volatile("mov.u32 %0, %globaltimer_lo;" : "=r"(ret));
  return ret;
}
#endif // TVM_BUILTIN_GET_TIMESTAMP_ASSEMBLY

__forceinline__ __device__ void {func_name}(int event_type, void* profiler_buffer, void* profiler_tag, void* profiler_write_offset, int profiler_write_stride) {{
    // timer end
    __threadfence_block();
    if (threadIdx.x % 128 == 0) {
        profiler_buffer[profiler_write_offset[0]] = ((uint64_t)tvm_builtin_get_timestamp() << 32) | (profiler_write_offset[0] | (uint32_t)event_type << 2 | 0x1);
        profiler_write_offset[0] += profiler_write_stride;
    }
}}
"""
    source_code = source_code.format(func_name=func_name)
    return cuda_func_call(
        func_name,
        event_type,
        profiler_buffer,
        profiler_tag,
        profiler_write_offset,
        profiler_write_stride,
        source_code=source_code,
    )


@register_codegen("cuda_atomic_add")
def codegen_cuda_atomic_add(res_addr, value):
    func_name = "tvm_builtin_cuda_atomic_add"
    source_code = R"""
template <typename T>
__forceinline__ __device__ T {func_name}(T* addr, T value) {{
    return atomicAdd(addr, value);
}}
"""
    assert isinstance(value, tvm.tir.PrimExpr)
    source_code = source_code.format(func_name=func_name)
    return cuda_func_call(
        func_name, res_addr, value, source_code=source_code, return_type=value.dtype
    )


@register_codegen("cuda_thread_fence")
def codegen_cuda_thread_fence():
    func_name = "tvm_builtin_cuda_thread_fence"
    source_code = R"""
__forceinline__ __device__ void {func_name}() {{
    __threadfence();
}}
"""
    source_code = source_code.format(func_name=func_name)
    return cuda_func_call(func_name, source_code=source_code)


@register_codegen("cuda_syncthreads_and")
def codegen_cuda_syncthreads_and(predicate):
    func_name = "tvm_builtin_cuda_syncthreads_and"
    source_code = R"""
__forceinline__ __device__ int {func_name}(int predicate) {{
    return __syncthreads_and(predicate);
}}
"""
    source_code = source_code.format(func_name=func_name)
    return cuda_func_call(func_name, predicate, source_code=source_code, return_type="int32")


@register_codegen("cuda_nano_sleep")
def codegen_cuda_nano_sleep(time):
    func_name = "tvm_builtin_cuda_nano_sleep"
    source_code = R"""
__forceinline__ __device__ void {func_name}(uint64_t time) {{
    __nanosleep(time);
}}
"""
    source_code = source_code.format(func_name=func_name)
    return cuda_func_call(func_name, time, source_code=source_code)


@register_codegen("ptx_map_shared_rank")
def codegen_ptx_map_shared_rank(ptr, rank):
    func_name = "tvm_builtin_ptx_map_shared_rank"
    source_code = R"""
__forceinline__ __device__ uint64_t {func_name}(void* addr, uint32_t rank) {{
    uint64_t result;
    asm volatile("mapa.u64  %0, %1, %2;\n"
                : "=l"(result)
                : "l"(reinterpret_cast<uint64_t>(addr)), "r"(rank));
    return result;
}}
"""
    source_code = source_code.format(func_name=func_name)
    return cuda_func_call(func_name, ptr, rank, source_code=source_code, return_type="uint64")
