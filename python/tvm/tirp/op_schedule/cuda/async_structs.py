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

"""Implementation of async structs schedules."""

from typing import Optional

from tvm.script import tir as T
from tvm.tirp.op_schedule import ScheduleContext, register_schedule
from tvm.tir import BufferRegion, PrimFunc
from tvm.tir.async_structs import CopyPipeline, Pipeline
from tvm.tir.stmt import OpCall
from ..registry import register_schedule
from .common import InstType, copy_cuda_g2s_s2g_2d_cta_vec_load_impl


@register_schedule("pipeline_producer_commit")
def pipeline_producer_commit(op: OpCall, sctx: ScheduleContext) -> Optional[PrimFunc]:
    """Schedule pipeline producer commit."""
    if not (sctx.is_cuda() and sctx.exec_scope.name == "cta"):
        return None
    pipeline = op.args[0]
    if isinstance(pipeline, CopyPipeline):
        if pipeline.depth == 0 and not pipeline.separate_pc:
            # copy pipeline without depth and separate pc: async-group mechanism
            @T.prim_func
            def _():
                T.evaluate(T.ptx_commit_group())

            return _

    return None



@register_schedule("pipeline_consumer_wait")
def pipeline_consumer_wait(op: OpCall, sctx: ScheduleContext) -> Optional[PrimFunc]:
    """Schedule pipeline consumer wait."""
    if not (sctx.is_cuda() and sctx.exec_scope.name == "cta"):
        return None
    pipeline, num_stages = op.args
    if isinstance(pipeline, CopyPipeline):
        if pipeline.depth == 0 and not pipeline.separate_pc:
            # copy pipeline without depth and separate pc: async-group mechanism
            @T.prim_func
            def _():
                T.evaluate(T.ptx_wait_group(num_stages))

            return _


@register_schedule("pipeline_copy")
def pipeline_copy(op: OpCall, sctx: ScheduleContext) -> Optional[PrimFunc]:
    """Schedule pipeline copy."""
    if not (sctx.is_cuda() and sctx.exec_scope.name == "cta"):
        return None
    pipeline, dst, src = op.args
    if pipeline.depth == 0 and not pipeline.separate_pc:
        # copy pipeline without depth and separate pc: async-group mechanism
        for schedule in [copy_cuda_g2s_s2g_2d_cta_vec_load_impl]:
            res = schedule(dst, src, sctx, InstType.CP_ASYNC)
            if res is not None:
                return res

    return None
