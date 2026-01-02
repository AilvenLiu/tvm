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

"""Implementation of mapping schedules on CUDA."""

from ..common import MapOpType, register_unary_binary_schedule
from .binary import binary_cuda_impl
from .common import target_cuda
from .unary import unary_cuda_impl


for op_name_, op_type_ in {
    "zero": MapOpType.ZERO,
    "fill": MapOpType.FILL,
    "reciprocal": MapOpType.RECIPROCAL,
    "exp": MapOpType.EXP,
    "sqrt": MapOpType.SQRT,
}.items():
    register_unary_binary_schedule(
        op_name_,
        op_type_,
        "cuda",
        target_cuda,
        [unary_cuda_impl],
    )

for op_name_, op_type_ in {
    "add": MapOpType.ADD,
    "sub": MapOpType.SUB,
    "mul": MapOpType.MUL,
    "fdiv": MapOpType.FDIV,
}.items():
    register_unary_binary_schedule(
        op_name_,
        op_type_,
        "cuda",
        target_cuda,
        [binary_cuda_impl],
    )
