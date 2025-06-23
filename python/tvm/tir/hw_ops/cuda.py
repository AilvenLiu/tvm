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

import functools

import tvm.ffi
from tvm.tir.op import Op, cuda_func_call


def get_op(name):
    return Op.get(f"tir.name")


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
