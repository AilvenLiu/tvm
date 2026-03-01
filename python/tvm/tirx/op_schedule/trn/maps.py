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

"""Implementation of mapping schedules."""

from ..common import MapOpType, register_unary_binary_schedule
from .binary import binary_trn
from .common import target_trn
from .unary import unary_trn, unary_with_bias_scale_trn

for op_name_, op_type_ in {
    "reciprocal": MapOpType.RECIPROCAL,
    "memset": MapOpType.MEMSET,
}.items():
    register_unary_binary_schedule(op_name_, op_type_, "trn", target_trn, [unary_trn])


for op_name_, op_type_ in {
    "sqrt": MapOpType.SQRT,
    "exp": MapOpType.EXP,
}.items():
    register_unary_binary_schedule(
        op_name_, op_type_, "trn", target_trn, [unary_with_bias_scale_trn]
    )

for op_name_, op_type_ in {
    "add": MapOpType.ADD,
    "sub": MapOpType.SUB,
    "mul": MapOpType.MUL,
    "maximum": MapOpType.MAX,
    "minimum": MapOpType.MIN,
}.items():
    register_unary_binary_schedule(op_name_, op_type_, "trn", target_trn, [binary_trn])
