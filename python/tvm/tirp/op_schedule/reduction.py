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

from typing import Optional

from tvm.tir import BufferRegion, PrimFunc
from tvm.tirp.op_schedule import ScheduleContext, register_schedule

from .registry import register_schedule
from .common import reduction_cuda_shared_nd_sync_cta_impl


def register_reduce_schedule(func):
    """Decorator function to register reduce operator implementations.

    Parameters
    ----------
    func : Callable[..., Union[bool, PrimFunc]]
    The implementation function.

    Returns
    -------
    Callable
        The decorated function.
    """
    return register_schedule("reduce")(func)

@register_reduce_schedule
def reduction_cuda_shared_nd_sync_cta(
    dst_buffer_region: BufferRegion,
    src_buffer_region: BufferRegion,
    accum: bool,
    reduce_op: str,
    sctx: ScheduleContext,
    _,
) -> Optional[PrimFunc]:
    """Schedule warp-level tree-reduction operation in shared memory on CUDA.
    
    Support reduction along the last D dimensions.
    Warp partition follows the rule below:
        For src tensor [s1, s2, ..., r1, r2, ...], where si are spatial axes and ri are reduction axes.
        Use one warp (32 threads) for each si for reduction.
    """
    return reduction_cuda_shared_nd_sync_cta_impl(
        dst_buffer_region, src_buffer_region, accum, reduce_op, sctx
   )
