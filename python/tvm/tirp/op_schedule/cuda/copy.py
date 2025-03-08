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

"""Implementation of copy operator schedules."""

from typing import Optional

from tvm.tir import PrimFunc
from tvm.tir.stmt import OpCall
from tvm.tirp.op_schedule import ScheduleContext, register_schedule

from ..registry import register_schedule
from .common import InstType, copy_g2s_s2g_cta_vec_load_impl, target_cuda


@register_schedule("copy", "cuda")
@target_cuda
def copy_schedule(op_call: OpCall, sctx: ScheduleContext) -> Optional[PrimFunc]:
    """Schedule copy operation between global and shared memory on CUDA."""
    dst_buffer_region, src_buffer_region = op_call.args
    return copy_g2s_s2g_cta_vec_load_impl(
        dst_buffer_region, src_buffer_region, sctx, InstType.NORMAL
    )
