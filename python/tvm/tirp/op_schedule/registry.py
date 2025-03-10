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
"""TIRp operator schedule registry."""
from typing import Dict, Optional, Callable, Tuple

from tvm._ffi import register_func
from tvm.ir import Op
from tvm.tir import PrimFunc
from tvm.tir.stmt import OpCall
from tvm.tirp.op_schedule.schedule_context import ScheduleContext
from tvm.tirp.operator import get_tirp_op

# Global registry to store operator implementations
_OP_REGISTRY: Dict[Tuple[Op, str], Callable[[OpCall, ScheduleContext], Optional[PrimFunc]]] = {}


def register_schedule(op_name: str, target_kind: str):
    """Decorator function to register operator implementations

    Parameters
    ----------
    op_name : str
        The operator to be registered.

    target_kind : str
        The target kind to be registered.

    impl_func : Callable[[OpCall, ScheduleContext], Optional[PrimFunc]]
        The implementation function.

    Returns
    -------
    Callable
        A decorator function that registers the implementation.
    """

    def decorator(impl_func: Callable[[OpCall, ScheduleContext], Optional[PrimFunc]]):
        # Register the implementation
        op = get_tirp_op(op_name)
        if (op, target_kind) not in _OP_REGISTRY:
            _OP_REGISTRY[(op, target_kind)] = impl_func
        else:
            raise ValueError(f"Operator {op_name} already registered")
        return impl_func

    return decorator


@register_func("tirp.f_op_scheduler")
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
    assert (
        key in _OP_REGISTRY
    ), f"No schedule registered for operator {op_call.op} on target {sctx.target.kind}"
    return _OP_REGISTRY[key](op_call, sctx=sctx)
