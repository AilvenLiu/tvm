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
"""CUDA profiler timer intrinsics (all schema-declared)."""

from .._schema import cuda_helper_intrinsic

_COMMON_PARAMS = (
    "uint64_t* profiler_buffer, uint64_t* profiler_tag, "
    "uint32_t* profiler_write_offset, int profiler_write_stride, bool leader_cond"
)
_EVENT_PARAMS = f"int event_type, {_COMMON_PARAMS}"


def _write_event(event_bits: str) -> str:
    return (
        "profiler_buffer[profiler_write_offset[0]] = "
        "((uint64_t)tvm_builtin_get_timestamp() << 32) | "
        f"(profiler_tag[0] | {event_bits});\n"
        "        profiler_write_offset[0] += profiler_write_stride;"
    )


cuda_helper_intrinsic(
    op_name="timer_init_cuda",
    c_signature=(
        "(uint64_t* profiler_buffer, uint64_t* profiler_tag, "
        "uint32_t* profiler_write_offset, int num_groups, int group_id)"
    ),
    c_body=(
        "const uint32_t NBLOCKS = (uint32_t)(gridDim.x * gridDim.y * gridDim.z);\n"
        "    const uint32_t BLOCK_IDX = (uint32_t)("
        "(blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x);\n"
        "    const uint32_t NGROUPS = num_groups;\n"
        "    const uint32_t GROUP_ID = group_id;\n"
        "    const uint32_t BLOCK_GROUP_IDX = BLOCK_IDX * NGROUPS + GROUP_ID;\n"
        "    if ((blockIdx.x == 0) && (blockIdx.y == 0) && "
        "(blockIdx.z == 0) && (threadIdx.x == 0)) {\n"
        "        profiler_buffer[0] = ((uint64_t)NGROUPS << 32) | NBLOCKS;\n"
        "    }\n"
        "    profiler_write_offset[0] = 1 + BLOCK_GROUP_IDX;\n"
        "    profiler_tag[0] = (uint64_t)BLOCK_GROUP_IDX << 12;"
    ),
)

cuda_helper_intrinsic(
    op_name="timer_start_cuda",
    c_signature=f"({_EVENT_PARAMS})",
    c_body=(
        f"if (leader_cond) {{\n        {_write_event('(uint32_t)event_type << 2 | 0x0')}\n    }}\n"
        "    __threadfence_block();"
    ),
    extra_deps=("get_time_stamp",),
)

cuda_helper_intrinsic(
    op_name="timer_end_cuda",
    c_signature=f"({_EVENT_PARAMS})",
    c_body=(
        "__threadfence_block();\n"
        f"    if (leader_cond) {{\n        {_write_event('(uint32_t)event_type << 2 | 0x1')}\n    }}"  # noqa: E501
    ),
    extra_deps=("get_time_stamp",),
)

cuda_helper_intrinsic(
    op_name="timer_finalize_cuda",
    c_signature=f"({_COMMON_PARAMS})",
    c_body=(
        f"__threadfence_block();\n    if (leader_cond) {{\n        {_write_event('0x3')}\n    }}"
    ),
    extra_deps=("get_time_stamp",),
)
