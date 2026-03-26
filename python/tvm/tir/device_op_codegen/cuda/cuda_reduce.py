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
# pylint: disable=redefined-builtin, invalid-name
"""CUDA reduction codegen: warp-level and CTA-wide reductions."""

from tvm.tir.op import cuda_func_call

from .registry import register_codegen
from .utils import parse_str, validate_power_of_two_range

# (accumulation expression, identity value for cross-warp padding)
_OP_TABLE = {
    "sum": ("val += shuffled;", "T(0)"),
    "max": ("val = max(val, shuffled);", "-INFINITY"),
    "min": ("val = min(val, shuffled);", "INFINITY"),
}


def _validate_op(op_str, context):
    if op_str not in _OP_TABLE:
        raise ValueError(f"Unsupported {context} op '{op_str}', expected one of {list(_OP_TABLE)}")
    return _OP_TABLE[op_str]


def _warp_reduce_source(func_name, width_int, step_expr):
    """Generate the butterfly shuffle-XOR reduction template."""
    return f"""
template <typename T>
__forceinline__ __device__ T {func_name}(T val) {{
    #pragma unroll
    for (int mask = {width_int} >> 1; mask > 0; mask >>= 1) {{
        T shuffled = __shfl_xor_sync(0xFFFFFFFF, val, mask);
        {step_expr}
    }}
    return val;
}}
"""


@register_codegen("cuda_warp_reduce")
def codegen_cuda_warp_reduce(value, op, width):
    op_str = parse_str(op)
    width_int = validate_power_of_two_range(width, 2, 32, "warp_reduce width")
    step_expr, _ = _validate_op(op_str, "warp_reduce")

    func_name = f"tvm_builtin_cuda_warp_reduce_{op_str}_{width_int}"
    source_code = _warp_reduce_source(func_name, width_int, step_expr)
    return cuda_func_call(func_name, value, source_code=source_code, return_type=value.dtype)


@register_codegen("cuda_cta_reduce")
def codegen_cuda_cta_reduce(value, op, num_warps, scratch):
    op_str = parse_str(op)
    nw = validate_power_of_two_range(num_warps, 1, 32, "cta_reduce num_warps")
    step_expr, identity = _validate_op(op_str, "cta_reduce")

    warp_reduce_name = f"tvm_builtin_cuda_warp_reduce_{op_str}_32"
    func_name = f"tvm_builtin_cuda_cta_reduce_{op_str}_{nw}"

    source_code = f"""{_warp_reduce_source(warp_reduce_name, 32, step_expr)}
template <typename T>
__forceinline__ __device__ T {func_name}(T val, void* scratch_raw) {{
    T* scratch = reinterpret_cast<T*>(scratch_raw);
    val = {warp_reduce_name}(val);
    int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    if (lane_id == 0) scratch[warp_id] = val;
    __syncthreads();
    if (warp_id == 0) {{
        T partial = (lane_id < {nw}) ? scratch[lane_id] : {identity};
        partial = {warp_reduce_name}(partial);
        if (lane_id == 0) scratch[0] = partial;
    }}
    __syncthreads();
    return scratch[0];
}}
"""
    return cuda_func_call(
        func_name, value, scratch, source_code=source_code, return_type=value.dtype
    )
