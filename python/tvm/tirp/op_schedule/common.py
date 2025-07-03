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
"""TIRp operator schedule common utilities."""

from enum import Enum
from typing import Optional, List, Callable, Union

from tvm.tirp.op_schedule import ScheduleContext
from tvm.tir import PrimFunc
from tvm.tir.stmt import OpCall

from .registry import register_schedule


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
    MAX = 8
    MIN = 9
    EXP = 10
    FILL = 11 # FIXME: FILL and MEMSET are the same. merge them.


class ReduceOpType(Enum):
    """Enumeration of common reduce operator types."""

    SUM = 0
    MAX = 1
    MIN = 2


def register_unary_binary_schedule(
    op_name: str,
    op_type: Union[MapOpType, ReduceOpType],
    target_kind: str,
    target_check: Callable[[ScheduleContext], bool],
    schedule_candidates: List[Callable[[OpCall, Enum, ScheduleContext], Optional[PrimFunc]]],
) -> None:
    """Register a schedule function for a given operation type."""

    @register_schedule(op_name, target_kind)
    @target_check
    def schedule(op: OpCall, sctx: ScheduleContext):
        for schedule in schedule_candidates:
            res = schedule(op, op_type, sctx)
            if res is not None:
                return res
        return None

    return None
