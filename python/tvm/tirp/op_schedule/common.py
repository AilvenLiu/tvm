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

from enum import Enum
from typing import Optional, Union, List, Callable

from tvm.tirp.op_schedule import ScheduleContext
from tvm.tir import BufferRegion, PrimFunc
from tvm.tir.expr import FloatImm


class MapOpType(Enum):
    """Enumeration of common unary and binary operator types."""

    ADD = 0
    SUB = 1
    MUL = 2
    FDIV = 3
    ZERO = 4
    SQRT = 5
    RECIPROCAL = 6
    MEMSET = 7


def _make_schedule(
    op_type: MapOpType, num_src: int, schedule_candidates: List[Callable[..., Optional[PrimFunc]]]
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

    def impl(
        dst: BufferRegion, *src: Union[BufferRegion, FloatImm], sctx: ScheduleContext
    ) -> Optional[PrimFunc]:
        if len(src) != num_src:
            raise ValueError(f"Expected {num_src} source arguments, got {len(src)}")
        for schedule in schedule_candidates:
            res = schedule(dst, *src, op_type, sctx)
            if res is not None:
                return res
        return None

    return impl
