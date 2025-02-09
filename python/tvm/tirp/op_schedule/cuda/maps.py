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

from typing import Optional, Union, Callable, List

from tvm.script import tir as T
from tvm.tirp.op_schedule import ScheduleContext
from tvm.tir import BufferRegion, PrimFunc
from tvm.tir.expr import FloatImm

from ..registry import register_schedule
from .common import MapOpType, unary_map_cuda_shared_nd_sync_cta_impl, binary_map_cuda_shared_nd_sync_cta_impl


def _make_schedule(
    op_type: MapOpType,
    num_src: int,
    schedule_candidates: List[Callable[..., Optional[PrimFunc]]]
) -> Callable[..., Optional[PrimFunc]]:
    """Return a schedule function that works for both unary and binary cases.

    Parameters
    ----------
    op_type : MapOpType
        The mapping operation type (e.g. ZERO, SQRT, ADD, ...).
    num_src : int
        The number of source arguments (1 for unary, 2 for binary).
    schedule_candidates : List[Callable[..., Optional[PrimFunc]]]
        List of candidate schedule functions to try.

    Returns
    -------
    Callable[..., Optional[PrimFunc]]
        A schedule function that unpacks its source arguments and calls the candidates.
    """
    def impl(dst: BufferRegion, *src: Union[BufferRegion, FloatImm],
             sctx: ScheduleContext) -> Optional[PrimFunc]:
        if not (sctx.is_cuda() and sctx.exec_scope.name == "cta"):
            return None
        if len(src) != num_src:
            raise ValueError(f"Expected {num_src} source arguments, got {len(src)}")
        for schedule in schedule_candidates:
            res = schedule(dst, *src, op_type, sctx)
            if res is not None:
                return res
        return None
    return impl


# Register unary mapping schedules.
for op_name, op_type in [("zero", MapOpType.ZERO), ("sqrt", MapOpType.SQRT)]:
    custom_name = f"unary_{op_name}_cuda_shared_nd_sync_cta_impl"
    func = _make_schedule(op_type, 1, [unary_map_cuda_shared_nd_sync_cta_impl])
    func.__name__ = custom_name
    func.__doc__ = f"Schedule unary {op_name}."
    register_schedule(op_name)(func)


# Register binary mapping schedules.
for op_name, op_type in [
    ("add", MapOpType.ADD),
    ("sub", MapOpType.SUB),
    ("mul", MapOpType.MUL),
    ("fdiv", MapOpType.FDIV)
]:
    custom_name = f"binary_{op_name}_cuda_shared_nd_sync_cta_impl"
    func = _make_schedule(op_type, 2, [binary_map_cuda_shared_nd_sync_cta_impl])
    func.__name__ = custom_name
    func.__doc__ = f"Schedule binary {op_name}."
    register_schedule(op_name)(func)
