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
from typing import Optional, Callable, Union
from functools import wraps

from tvm.script import tir as T
from tvm.tir import PrimFunc, Buffer
from tvm.tir.stmt import OpCall
from tvm.tirp.op_schedule import (
    ScheduleContext,
    register_dispatch,
    predicate,
)
from .common import target_cuda, thread_selector
from tvm.tirp.operator import EventInit, EventCommit, EventWait
from tvm.tir.event import (
    EventImpl,
    SemaphoreEventTensor,
    SemaphoreEventTensorItem,
)


EVENT_INIT_IMPL_DICT = dict()
EVENT_COMMIT_IMPL_DICT = dict()
EVENT_WAIT_IMPL_DICT = dict()


def target_event_impl(op_name: str, exp_impl: EventImpl):
    def decorator(fn: Callable[[OpCall, ScheduleContext], Optional[PrimFunc]]):
        """Decorator that ensures the function is only executed for specific event implementations."""

        @wraps(fn)
        def wrapper(*args, **kwargs):
            op_call = OpCall.downcast(args[0])
            assert isinstance(
                op_call, (EventInit, EventCommit, EventWait)
            ), f"{op_call} is not a EventInit, EventCommit, or EventWait"

            event = op_call.event
            impl = event.get_impl()

            if impl != exp_impl:
                return None

            return fn(*args, **kwargs)

        if op_name == "event_init":
            EVENT_INIT_IMPL_DICT[exp_impl] = wrapper
        elif op_name == "event_commit":
            EVENT_COMMIT_IMPL_DICT[exp_impl] = wrapper
        elif op_name == "event_wait":
            EVENT_WAIT_IMPL_DICT[exp_impl] = wrapper
        else:
            raise ValueError(f"Invalid op_name: {op_name}")

        return wrapper

    return decorator


######################## kTMALoad ########################
@target_event_impl("event_init", EventImpl.kTMALoad)
def event_init(op: OpCall, sctx: ScheduleContext) -> Optional[PrimFunc]:
    """Schedule event initialization."""
    evt, expected_count = op.args
    mbar, phase, tx_cnt = evt.get_state()

    if isinstance(evt, SemaphoreEventTensorItem):

        @T.prim_func(tirp=True, check_well_formed=False)
        def func():
            if phase is not None:
                phase[*evt.indices] = 0
            if tx_cnt is not None:
                tx_cnt[*evt.indices] = 0

            @T.macro
            def mbarrier_init():
                T.ptx.mbarrier.init(mbar.ptr_to([*evt.indices]), expected_count)

            thread_selector(sctx, mbarrier_init, macro=True)()

        return func
    elif isinstance(evt, SemaphoreEventTensor):

        @T.prim_func(tirp=True, check_well_formed=False)
        def func():
            for i in T.grid(*evt.shape):
                if phase is not None:
                    phase[i] = 0
                if tx_cnt is not None:
                    tx_cnt[i] = 0

            @T.macro
            def mbarrier_init():
                for i in T.grid(*evt.shape):
                    T.ptx.mbarrier.init(mbar.ptr_to([i]), expected_count)

            thread_selector(sctx, mbarrier_init, macro=True)()

        return func
    else:
        raise ValueError(f"Invalid event type: {type(evt)}")


@target_event_impl("event_commit", EventImpl.kTMALoad)
def event_commit(op: OpCall, sctx: ScheduleContext) -> Optional[PrimFunc]:
    """Schedule event commit."""
    evt = op.args[0]
    assert isinstance(evt, SemaphoreEventTensorItem), "event must be a SemaphoreEventTensor"
    mbar, phase, tx_cnt = evt.get_state()

    if tx_cnt is not None:

        @T.macro
        def func():
            T.ptx.mbarrier.arrive.expect_tx(mbar.ptr_to([*evt.indices]), tx_cnt[*evt.indices])

        return thread_selector(sctx, func)

    else:
        # get tx_cnt from config
        tx_cnt = op.config.get("tx_cnt", None)
        if tx_cnt is None:
            raise ValueError("tx_cnt is neither in event state nor set in config")

        @T.macro
        def func():
            T.ptx.mbarrier.arrive.expect_tx(mbar.ptr_to([*evt.indices]), tx_cnt)

        return thread_selector(sctx, func)


@target_event_impl("event_wait", EventImpl.kTMALoad)
def event_wait(op: OpCall, sctx: ScheduleContext) -> Optional[PrimFunc]:
    """Schedule event wait."""
    evt = op.args[0]
    assert isinstance(evt, SemaphoreEventTensorItem), "event must be a SemaphoreEventTensor"
    mbar, phase, tx_cnt = evt.get_state()

    if phase is not None:
        # phase is managed by the event
        @T.prim_func(tirp=True, check_well_formed=False)
        def func():
            T.ptx.mbarrier.try_wait(mbar.ptr_to([*evt.indices]), phase[*evt.indices])
            if tx_cnt is not None:
                tx_cnt[*evt.indices] = 0
            phase[*evt.indices] = phase[*evt.indices] ^ 1

        return func
    else:
        # phase is not managed by the event, get it from config
        phase = op.config.get("phase", None)
        if phase is None:
            raise ValueError("phase is neither in event state nor set in config")

        @T.prim_func(tirp=True, check_well_formed=False)
        def func():
            T.ptx.mbarrier.try_wait(mbar.ptr_to([*evt.indices]), phase)
            if tx_cnt is not None:
                tx_cnt[*evt.indices] = 0

        return func


######################## kTMAStore ########################
@target_event_impl("event_init", EventImpl.kTMAStore)
def event_init(op: OpCall, sctx: ScheduleContext) -> Optional[PrimFunc]:
    """Schedule event initialization."""

    @T.prim_func(tirp=True, check_well_formed=False)
    def func():
        pass

    return func


@target_event_impl("event_commit", EventImpl.kTMAStore)
def event_commit(op: OpCall, sctx: ScheduleContext) -> Optional[PrimFunc]:
    """Schedule event commit."""

    @T.macro
    def func():
        T.ptx.cp_async.bulk.commit_group()

    return thread_selector(sctx, func)


@target_event_impl("event_wait", EventImpl.kTMAStore)
def event_wait(op: OpCall, sctx: ScheduleContext) -> Optional[PrimFunc]:
    """Schedule event wait."""
    n = op.args[1]

    @T.macro
    def func():
        T.ptx.cp_async.bulk.wait_group(n)

    return thread_selector(sctx, func)


######################## kCpAsync ########################
@target_event_impl("event_init", EventImpl.kCpAsync)
def event_init(op: OpCall, sctx: ScheduleContext) -> Optional[PrimFunc]:
    """Schedule event initialization."""
    if sctx.exec_scope.name not in ["cta", "thread"]:
        return None

    @T.prim_func(tirp=True, check_well_formed=False)
    def func():
        T.evaluate(0)

    return func


@target_event_impl("event_commit", EventImpl.kCpAsync)
def event_commit(op: OpCall, sctx: ScheduleContext) -> Optional[PrimFunc]:
    """Schedule event commit."""
    if sctx.exec_scope.name not in ["cta", "thread"]:
        return None

    @T.prim_func(tirp=True, check_well_formed=False)
    def func():
        T.evaluate(T.ptx.cp_async.commit_group())

    return func


@target_event_impl("event_wait", EventImpl.kCpAsync)
def event_wait(op: OpCall, sctx: ScheduleContext) -> Optional[PrimFunc]:
    """Schedule event wait."""
    if sctx.exec_scope.name not in ["cta", "thread"]:
        return None

    n_groups = op.args[1]

    @T.prim_func(tirp=True, check_well_formed=False)
    def func():
        T.evaluate(T.ptx.cp_async.wait_group(n_groups))

    return func


# ######################## kGlobalSemaphore ########################
@target_event_impl("event_init", EventImpl.kGlobalSemaphore)
def event_init(op: OpCall, sctx: ScheduleContext) -> Optional[PrimFunc]:
    """Schedule event initialization."""
    if sctx.exec_scope.name != "kernel":
        return None

    evt, expected_count = op.args
    sem, state = evt.get_state()

    if isinstance(evt, SemaphoreEventTensor):

        @T.prim_func(tirp=True, check_well_formed=False)
        def func():
            with T.thread()[0:1]:
                for i in T.grid(*sem.shape):
                    sem[i] = expected_count
            T.cuda.thread_fence()

        return func
    elif isinstance(evt, SemaphoreEventTensorItem):

        @T.prim_func(tirp=True, check_well_formed=False)
        def func():
            with T.thread()[0:1]:
                sem[*evt.indices] = expected_count
            T.cuda.thread_fence()

        return func


@target_event_impl("event_commit", EventImpl.kGlobalSemaphore)
def event_commit(op: OpCall, sctx: ScheduleContext) -> Optional[PrimFunc]:
    """Schedule event commit."""
    if sctx.exec_scope.name != "cta":
        return None

    evt = op.args[0]
    assert isinstance(evt, SemaphoreEventTensorItem)
    sem, state = evt.get_state()

    @T.prim_func(tirp=True, check_well_formed=False)
    def func():
        with T.thread()[0:1]:
            T.cuda.atomic_add(sem.ptr_to(evt.indices), -1)
        T.cuda.thread_fence()

    return func


@target_event_impl("event_wait", EventImpl.kGlobalSemaphore)
def event_wait(op: OpCall, sctx: ScheduleContext) -> Optional[PrimFunc]:
    """Schedule event wait."""
    if sctx.exec_scope.name != "cta":
        return None

    evt = op.args[0]
    assert isinstance(evt, SemaphoreEventTensorItem)
    sem, state = evt.get_state()

    @T.prim_func(tirp=True, check_well_formed=False)
    def func():
        with T.thread()[0:1]:
            state[0] = -1
            T.ptx.ld_global_acquire(state[0], sem.ptr_to(evt.indices))
        while T.cuda.syncthreads_and(state[0] != 0):
            T.cuda.nano_sleep(40)
            with T.thread()[0:1]:
                T.ptx.ld_global_acquire(state[0], sem.ptr_to(evt.indices))

    return func


# -----------------------------------------------------------------------------
# Register rich dispatcher variants per event implementation
# -----------------------------------------------------------------------------

_IMPL_NAME = {
    EventImpl.kTMALoad: "kTMALoad",
    EventImpl.kTMAStore: "kTMAStore",
    EventImpl.kCpAsync: "kCpAsync",
    EventImpl.kGlobalSemaphore: "kGlobalSemaphore",
}


def _event_impl_predicate(expected_impl):
    def _pred(op: OpCall, sctx: ScheduleContext):
        evt = op.args[0]
        return (evt.get_impl() == expected_impl), f"event impl is {evt.get_impl()}"

    return _pred


def _register_event_dispatch_table(op_name: str, table):
    for impl, fn in table.items():
        name = _IMPL_NAME.get(impl, str(impl))

        @register_dispatch(
            op_name,
            "cuda",
            variant=name,
            priority=10,
            when=[predicate("event_impl", _event_impl_predicate(impl))],
        )
        def _dispatch(op: OpCall, sctx: ScheduleContext, _fn=fn, _impl=impl):
            # predicate handled by wrapper; explicit predicate gives clearer reason
            return _fn(op, sctx)


_register_event_dispatch_table("event_init", EVENT_INIT_IMPL_DICT)
_register_event_dispatch_table("event_commit", EVENT_COMMIT_IMPL_DICT)
_register_event_dispatch_table("event_wait", EVENT_WAIT_IMPL_DICT)
