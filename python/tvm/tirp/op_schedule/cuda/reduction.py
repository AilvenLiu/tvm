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

"""Implementation of reduce operator schedules."""

from typing import Optional, Tuple

from tvm.tir import BufferRegion, PrimFunc
from tvm.tirp.op_schedule import ScheduleContext, register_schedule

from ..registry import register_schedule
from .common import reduction_cuda_shared_nd_sync_cta_impl
from ..common import _make_schedule, ReduceOpType


def reduction_cuda_shared_nd_sync_cta(
    dst_buffer_region: BufferRegion,
    src_buffer_region: BufferRegion,
    axes: Tuple[int],
    accum: bool,
    reduce_op: ReduceOpType,
    sctx: ScheduleContext,
) -> Optional[PrimFunc]:
    """Schedule warp-level tree-reduction operation in shared memory on CUDA.

    Support reduction along the last D dimensions.
    Warp partition follows the rule below:
        For src tensor [s1, s2, ..., r1, r2, ...], where si are spatial axes and ri are reduction axes.
        Use one warp (32 threads) for each si for reduction.
    """

    # FIXME: correctly handle the axes field
    return reduction_cuda_shared_nd_sync_cta_impl(
        dst_buffer_region, src_buffer_region, accum, reduce_op, sctx
    )


# Register unary mapping schedules.
for op_name, op_type in [("sum", ReduceOpType.SUM)]:
    custom_name = f"reduction_{op_name}_cuda_shared_nd_sync_cta_impl"
    func = _make_schedule(op_type, 3, [reduction_cuda_shared_nd_sync_cta])
    func.__name__ = custom_name
    func.__doc__ = f"Schedule reduction {op_name}."
    register_schedule(op_name)(func)
