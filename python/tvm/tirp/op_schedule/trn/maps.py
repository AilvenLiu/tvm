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


from ..registry import register_schedule
from ..common import _make_schedule, MapOpType
from .binary import binary_trn
from .unary import unary_trn, unary_with_bias_scale_trn

# Register unary mapping schedules.
for op_name, op_type in [
    ("reciprocal", MapOpType.RECIPROCAL),
    ("memset", MapOpType.MEMSET),
]:
    custom_name = f"unary_{op_name}_trn_impl"
    func = _make_schedule(op_type, 1, [unary_trn])
    func.__name__ = custom_name
    func.__doc__ = f"Schedule unary {op_name}."
    register_schedule(op_name)(func)

for op_name, op_type in [
    ("sqrt", MapOpType.SQRT),
]:
    custom_name = f"unary_{op_name}_trn_with_bias_scale_impl"
    func = _make_schedule(op_type, 3, [unary_with_bias_scale_trn])
    func.__name__ = custom_name
    func.__doc__ = f"Schedule unary {op_name} with bias and scale."
    register_schedule(op_name)(func)


# Register binary mapping schedules.
for op_name, op_type in [
    ("add", MapOpType.ADD),
    ("sub", MapOpType.SUB),
    ("mul", MapOpType.MUL),
]:
    custom_name = f"binary_{op_name}_trn_impl"
    func = _make_schedule(op_type, 2, [binary_trn])
    func.__name__ = custom_name
    func.__doc__ = f"Schedule binary {op_name}."
    register_schedule(op_name)(func)
