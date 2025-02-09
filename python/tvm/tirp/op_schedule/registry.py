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
from typing import Dict

from tvm._ffi import register_object, register_func
from tvm.ir import Op, Range
from tvm.tir import _ffi_api, PrimExpr, Var
from tvm.runtime import Object, Scriptable
from tvm.target import Target
from tvm.tir.expr import PrimExpr
from tvm.tir.exec_scope import ExecScope


def _get_tirp_op(op_name: str):
    """Get TIRp operator by name.

    Parameters
    ----------
    op_name : str
        Name of the TIRp operator
    """
    assert isinstance(op_name, str)
    return Op.get("tirp." + op_name)


@register_object("tir.ScheduleContext")
class ScheduleContext(Object, Scriptable):
    """ScheduleContext node.

    Parameters
    ----------
    target : Target
        The target of the schedule context.

    exec_scope : ExecScope
        The execution scope of the schedule context.

    launch_params : Dict[str, PrimExpr]
        The launch parameters of the schedule context.

    var_range_map : Dict[Var, Range]
        A map from loop variables to their ranges.
    """

    target: Target
    exec_scope: ExecScope
    launch_params: Dict[str, PrimExpr]
    var_range_map: Dict[Var, Range]

    def __init__(
        self, target: Target, exec_scope: ExecScope, launch_params: Dict[str, PrimExpr], var_range_map: Dict[Var, Range]
    ) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.ScheduleContext, target, exec_scope, launch_params, var_range_map  # pylint: disable=no-member
        )

    def is_cuda(self) -> bool:
        """Check if the target is CUDA."""
        return self.target.kind.name == "cuda"
    
    def is_trn(self) -> bool:
        """Check if the target is Trainium."""
        return self.target.kind.name == "trn"


# Global registry to store operator implementations and their selectors
_OP_REGISTRY = {}


def register_schedule(op_name: str):
    """Decorator function to register operator implementations with their selectors.

    Parameters
    ----------
    op_name : str
        The operator to be registered.

    impl_func : Callable[..., Optional[PrimFunc]]
        The implementation function.

    Returns
    -------
    Callable
        A decorator function that registers the implementation.
    """

    def decorator(impl_func):
        # Register the implementation
        op = _get_tirp_op(op_name)
        if op not in _OP_REGISTRY:
            _OP_REGISTRY[op] = []
        _OP_REGISTRY[op].append(impl_func)
        return impl_func

    return decorator


@register_func("tirp.f_op_scheduler")
def f_op_scheduler(op: Op, args, sctx: ScheduleContext):
    """Find and return a schedule for the operator.

    Parameters
    ----------
    op : Op
        The operator to be scheduled
    args : list
        The operator arguments
    sctx : ScheduleContext
        The scheduling context

    Returns
    -------
    Optional[PrimFunc]
        The result of the operator implementation
    """
    assert op in _OP_REGISTRY, f"No implementation found for operator {op}"
    for impl in _OP_REGISTRY[op]:
        res = impl(*args, sctx=sctx)
        if res is not None:
            return res
    assert False, (
        "Internal Error: no implementation found for operator "
        + str(op)
        + " with args "
        + str(args)
        + " and context "
        + str(sctx)
    )
