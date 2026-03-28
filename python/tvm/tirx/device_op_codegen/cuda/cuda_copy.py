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
"""CUDA copy codegen: typed load/store copy of N bytes."""

from tvm.tirx.op import cuda_func_call

from .registry import register_codegen

_TYPE_MAP = {
    16: "uint4",
    8: "uint2",
    4: "unsigned int",
    2: "unsigned short",
    1: "unsigned char",
}

_VALID_SIZES = frozenset(_TYPE_MAP)


@register_codegen("cuda_copy_bytes")
def codegen_cuda_copy_bytes(dst, src, num_bytes):
    num_bytes_int = int(num_bytes)
    if num_bytes_int not in _VALID_SIZES:
        raise ValueError(
            f"Unsupported cuda_copy_bytes num_bytes {num_bytes_int}, "
            f"expected one of {sorted(_VALID_SIZES)}"
        )
    cpp_type = _TYPE_MAP[num_bytes_int]
    bits = num_bytes_int * 8
    func_name = f"tvm_builtin_copy_{bits}b"
    source_code = f"""
__forceinline__ __device__ void {func_name}(void* dst_ptr, void* src_ptr) {{
    {cpp_type}* src_ = reinterpret_cast<{cpp_type}*>(src_ptr);
    {cpp_type}* dst_ = reinterpret_cast<{cpp_type}*>(dst_ptr);
    *dst_ = *src_;
}}
"""
    return cuda_func_call(func_name, dst, src, source_code=source_code)
