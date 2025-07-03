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

"""Implementation of event operator schedules."""
from typing import Optional, Dict, Any, Tuple, Callable
from functools import wraps

from tvm.script import tir as T, tirp as Tp
from tvm.tir import PrimFunc, Buffer, Stmt
from tvm.tir.stmt import OpCall
from tvm.tirp.op_schedule import ScheduleContext, register_schedule
from .common import target_cuda
from tvm.tirp.operator import EventTensorInit, EventCommit, EventWait


def target_event_impl(event_impl: str):
    def decorator(fn: Callable[[OpCall, ScheduleContext], Optional[PrimFunc]]):
        """Decorator that ensures the function is only executed for specific event implementations."""

        @wraps(fn)
        def wrapper(*args, **kwargs):
            op_call = OpCall.downcast(args[0])
            if isinstance(op_call, EventTensorInit):
                event_tensor = op_call.event_tensor
            elif isinstance(op_call, (EventCommit, EventWait)):
                event_tensor = op_call.event.buffer
            else:
                raise ValueError(f"Unsupported event operator: {op_call}")

            impl = event_tensor.event_impl
            if impl == "":
                raise NotImplementedError(f"Event implementation deduction is not supported")
            if impl != event_impl:
                return None
            return fn(*args, **kwargs)

        return wrapper

    return decorator


# global semaphore


@register_schedule("event_tensor_init", "cuda")
@target_cuda
@target_event_impl("semaphore")
def event_tensor_init_schedule(op_call: OpCall, sctx: ScheduleContext) -> Optional[PrimFunc]:
    """Schedule event tensor initialization."""
    op_call = OpCall.downcast(op_call)
    assert isinstance(op_call, EventTensorInit), f"{op_call} is not a EventTensorInit"

    event_tensor = op_call.event_tensor
    init_value = op_call.init_value if op_call.init_value is not None else 1

    @T.prim_func(tirp=True)
    def func():
        Tp.fill(event_tensor, T.convert(init_value))
        T.tvm_storage_sync("global")

    return func


@register_schedule("event_commit", "cuda")
@target_cuda
@target_event_impl("semaphore")
def event_commit_schedule(op_call: OpCall, sctx: ScheduleContext) -> Optional[PrimFunc]:
    """Schedule event commit."""
    op_call = OpCall.downcast(op_call)
    assert isinstance(op_call, EventCommit), f"{op_call} is not a EventCommit"

    event_tensor = op_call.event.buffer

    # FIXME: currently ignored the scope slice in the middle
    #        need to formally support synchronizing an exec scope and pick one thread from an exec scope
    if sctx.exec_scope.name == "cta":

        @T.prim_func(tirp=True, check_well_formed=False)
        def func():
            T.tvm_storage_sync("shared")
            with T.thread()[0:1]:
                T.cuda.atomic_add(
                    event_tensor.access_ptr(
                        "rw", offset=event_tensor.offset_of_p(op_call.event.indices)
                    ),
                    -1,
                )
            T.cuda.thread_fence()

    else:
        raise NotImplementedError(f"Event commit is now only supported in cta scope.")

    return func


@register_schedule("event_wait", "cuda")
@target_cuda
@target_event_impl("semaphore")
def event_wait_schedule(op_call: OpCall, sctx: ScheduleContext) -> Optional[PrimFunc]:
    """Schedule event wait."""
    op_call = OpCall.downcast(op_call)
    assert isinstance(op_call, EventWait), f"{op_call} is not a EventWait"

    event_tensor = op_call.event.buffer

    state = op_call.workspace.get("semaphore_state", None)
    assert state is not None, "semaphore state is not allocated"

    # FIXME: currently ignored the scope slice in the middle
    #        need to formally support synchronizing an exec scope and pick one thread from an exec scope
    if sctx.exec_scope.name == "cta":

        @T.prim_func(tirp=True, check_well_formed=False)
        def func():
            with T.thread()[0:1]:
                state[0] = -1
                T.ptx.ld_global_acquire(
                    state[0],
                    event_tensor.access_ptr(
                        "r", offset=event_tensor.offset_of_p(op_call.event.indices)
                    ),
                )
            while T.cuda.syncthreads_and(state[0] != 0):
                T.cuda.nano_sleep(40)
                with T.thread()[0:1]:
                    T.ptx.ld_global_acquire(
                        state[0],
                        event_tensor.access_ptr(
                            "r", offset=event_tensor.offset_of_p(op_call.event.indices)
                        ),
                    )

    else:
        raise NotImplementedError(f"Event wait is now only supported in cta scope.")

    return func


def alloc_state(
    op: OpCall, buffer_dict: Dict[Any, Tuple[Buffer, Optional[Stmt]]], _: ScheduleContext
) -> Dict[str, Buffer]:
    if buffer_dict.get("semaphore_state", None) is None:
        buffer = T.buffer((1,), "int32", scope="local", buffer_name="semaphore_state")
        buffer_dict["semaphore_state"] = (buffer, None)
    return {"semaphore_state": "semaphore_state"}


EventWait.get_private_buffers_cuda = alloc_state
