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
"""TIRx operator schedule registry.

All operator dispatch is handled by the rich dispatcher. This module exposes
the global entry `tirx.f_op_scheduler` used by the C++ lowering pass to query a
dispatch result.
"""
from typing import Optional
import os

from tvm_ffi import register_global_func
from tvm.ir import Op
from tvm.tir import PrimFunc
from tvm.tir.stmt import OpCall
from tvm.tirx.op_schedule.schedule_context import ScheduleContext
from tvm.tirx.operator import get_tirx_op


# Note: legacy `register_schedule` is intentionally removed.


@register_global_func("tirx.f_op_scheduler")
def f_op_scheduler(op_call: OpCall, sctx: ScheduleContext):
    """Find and return a schedule for the operator.

    Parameters
    ----------
    op_call : OpCall
        The operator to be scheduled
    sctx : ScheduleContext
        The scheduling context

    Returns
    -------
    Optional[PrimFunc]
        The result of the operator implementation
    """
    assert sctx.target is not None, "Target not found"
    key = (op_call.op, str(sctx.target.kind))

    # Use rich dispatcher for all dispatching
    try:
        from .dispatcher import run_dispatch  # local import to avoid cycles
    except Exception:  # pragma: no cover - fallback if import fails
        run_dispatch = None  # type: ignore

    if run_dispatch is not None:
        try:
            res = run_dispatch(op_call, sctx)
        except Exception:
            # propagate exceptions from dispatcher
            raise
        if res is not None:
            return res
    # Dispatcher reports errors on failure; unreachable on success
    return None
