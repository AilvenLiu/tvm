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

from typing import Optional

from tvm.tir import PrimFunc
from tvm.tir.stmt import OpCall
from tvm.tirp.op_schedule import ScheduleContext, register_schedule
from .common import target_cuda
from tvm.tir import event
from tvm.tirp.op_schedule.cuda.async_structs import CopyInstType
from tvm.tirp.op_schedule.cuda.common import copy_g2s_s2g_cta_vec_load_impl
from tvm.tirp.op_schedule.cuda.async_structs import copy_tma_impl


@register_schedule("copy_async", "cuda")
@target_cuda
def copy_async(op: OpCall, sctx: ScheduleContext) -> Optional[PrimFunc]:
    """Schedule copy_async operation."""
    if sctx.exec_scope.name != "cta":
        return None

    evt = op.args[2]
    dst, src = op.args[0], op.args[1]
    if isinstance(evt, event.BulkGroupEvent):
        if evt.impl == event.EventImpl.kCpAsync:
            for schedule in [copy_g2s_s2g_cta_vec_load_impl]:
                res = schedule(dst, src, sctx, CopyInstType.CP_ASYNC)
                if res is not None:
                    return res
            raise ValueError(
                f"No valid implementation found for copy_async op + BulkGroupEvent + impl = {evt.impl}"
            )
        if evt.impl == event.EventImpl.kTMAStore:
            for schedule in [copy_tma_impl]:
                res = schedule(evt, dst, src, sctx)
                if res is not None:
                    return res
            raise ValueError(
                f"No valid implementation found for copy_async op + BulkGroupEvent + impl = {evt.impl}"
            )
    elif isinstance(evt, event.SemaphoreEvent):
        if evt.impl == event.EventImpl.kTMALoad:
            for schedule in [copy_tma_impl]:
                res = schedule(evt, dst, src, sctx)
                if res is not None:
                    return res
            raise ValueError(
                f"No valid implementation found for copy_async op + SemaphoreEvent + impl = {evt.impl}"
            )
    return None
