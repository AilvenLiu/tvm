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
"""Codegen registry for CUDA HW ops."""
import functools

import tvm_ffi


CODEGEN_REGISTRY = {}


@tvm_ffi.register_global_func("tir.device_op_codegen.cuda.get_codegen")
def get_codegen(op):
    """get the codegen function for a given op"""
    return CODEGEN_REGISTRY.get(op, None)


def register_codegen(op, backend="cuda"):
    """register a codegen function for a given op
    The codegen function should return a cuda_func_call statement,
    and a list of tags that the codegen function needs.
    """

    def decorator(func):

        @functools.wraps(func)
        def wrapper(arg_list):
            res = func(*arg_list)  # pylint: disable=not-callable
            if isinstance(res, tuple):
                return res[0], res[1]
            else:
                return res, list()

        CODEGEN_REGISTRY["tir." + op] = wrapper
        return wrapper

    return decorator
