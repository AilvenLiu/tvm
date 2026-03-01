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
"""CUDA Timer operations."""

from tvm.tir.op import cuda_func_call

from .registry import register_codegen


@register_codegen("timer_init_cuda")
def codegen_timer_init_cuda(
    profiler_buffer, profiler_tag, profiler_write_offset, num_groups, group_id
):
    func_name = "tvm_builtin_timer_init_cuda"
    source_code = f"""
__forceinline__ __device__ void {func_name}(uint64_t* profiler_buffer, uint64_t* profiler_tag, uint32_t* profiler_write_offset, int num_groups, int group_id) {{
    // timer init
    const uint32_t NBLOCKS = (uint32_t)(gridDim.x * gridDim.y * gridDim.z);
    const uint32_t BLOCK_IDX = (uint32_t)((blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x);
    const uint32_t NGROUPS = num_groups;
    const uint32_t GROUP_ID = group_id;
    const uint32_t BLOCK_GROUP_IDX = BLOCK_IDX * NGROUPS + GROUP_ID;
    if ((blockIdx.x == 0) && (blockIdx.y == 0) && (blockIdx.z == 0) && (threadIdx.x == 0)) {{
        profiler_buffer[0] = ((uint64_t)NGROUPS << 32) | NBLOCKS;
    }}
    profiler_write_offset[0] = 1 + BLOCK_GROUP_IDX;
    profiler_tag[0] = (uint64_t)BLOCK_GROUP_IDX << 12;
}}
"""  # noqa: E501
    return cuda_func_call(
        func_name,
        profiler_buffer,
        profiler_tag,
        profiler_write_offset,
        num_groups,
        group_id,
        source_code=source_code,
    )


@register_codegen("timer_start_cuda")
def codegen_timer_start_cuda(
    event_type,
    profiler_buffer,
    profiler_tag,
    profiler_write_offset,
    profiler_write_stride,
    leader_cond,
):
    func_name = "tvm_builtin_timer_start_cuda"
    source_code = f"""
__forceinline__ __device__ void {func_name}(int event_type, uint64_t* profiler_buffer, uint64_t* profiler_tag, uint32_t* profiler_write_offset, int profiler_write_stride, bool leader_cond) {{
    // timer start
    if (leader_cond) {{
        profiler_buffer[profiler_write_offset[0]] = ((uint64_t)tvm_builtin_get_timestamp() << 32) | (profiler_tag[0] | (uint32_t)event_type << 2 | 0x0);
        profiler_write_offset[0] += profiler_write_stride;
    }}
    __threadfence_block();
}}
"""  # noqa: E501
    return cuda_func_call(
        func_name,
        event_type,
        profiler_buffer,
        profiler_tag,
        profiler_write_offset,
        profiler_write_stride,
        leader_cond,
        source_code=source_code,
    ), ["get_time_stamp"]


@register_codegen("timer_end_cuda")
def codegen_timer_end_cuda(
    event_type,
    profiler_buffer,
    profiler_tag,
    profiler_write_offset,
    profiler_write_stride,
    leader_cond,
):
    func_name = "tvm_builtin_timer_end_cuda"
    source_code = f"""
__forceinline__ __device__ void {func_name}(int event_type, uint64_t* profiler_buffer, uint64_t* profiler_tag, uint32_t* profiler_write_offset, int profiler_write_stride, bool leader_cond) {{
    // timer end
    __threadfence_block();
    if (leader_cond) {{
        profiler_buffer[profiler_write_offset[0]] = ((uint64_t)tvm_builtin_get_timestamp() << 32) | (profiler_tag[0] | (uint32_t)event_type << 2 | 0x1);
        profiler_write_offset[0] += profiler_write_stride;
    }}
}}
"""  # noqa: E501
    return cuda_func_call(
        func_name,
        event_type,
        profiler_buffer,
        profiler_tag,
        profiler_write_offset,
        profiler_write_stride,
        leader_cond,
        source_code=source_code,
    ), ["get_time_stamp"]


@register_codegen("timer_finalize_cuda")
def codegen_timer_finalize_cuda(
    profiler_buffer, profiler_tag, profiler_write_offset, profiler_write_stride, leader_cond
):
    func_name = "tvm_builtin_timer_finalize_cuda"
    source_code = f"""
__forceinline__ __device__ void {func_name}(uint64_t* profiler_buffer, uint64_t* profiler_tag, uint32_t* profiler_write_offset, int profiler_write_stride, bool leader_cond) {{
    // timer finalize
    __threadfence_block();
    if (leader_cond) {{
        profiler_buffer[profiler_write_offset[0]] = ((uint64_t)tvm_builtin_get_timestamp() << 32) | (profiler_tag[0] | 0x3);
        profiler_write_offset[0] += profiler_write_stride;
    }}
}}
"""  # noqa: E501
    return cuda_func_call(
        func_name,
        profiler_buffer,
        profiler_tag,
        profiler_write_offset,
        profiler_write_stride,
        leader_cond,
        source_code=source_code,
    ), ["get_time_stamp"]
