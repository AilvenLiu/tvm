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
from tvm.tirp.op_schedule import ScheduleContext, register_schedule
from .common import target_cuda
from tvm.tirp.operator import EventInit, EventCommit, EventWait
from tvm.tir.event import (
    EventImpl,
    SemaphoreEventTensor,
    SemaphoreEventTensorItem,
)


EVENT_INIT_IMPL_DICT = dict()
EVENT_COMMIT_IMPL_DICT = dict()
EVENT_WAIT_IMPL_DICT = dict()


@register_schedule("event_init", "cuda")
@target_cuda
def event_init(op: OpCall, sctx: ScheduleContext) -> Optional[PrimFunc]:
    """Schedule event init."""
    event = op.args[0]
    impl = event.get_impl()
    if impl not in EVENT_INIT_IMPL_DICT:
        return None

    return EVENT_INIT_IMPL_DICT[impl](op, sctx)


@register_schedule("event_commit", "cuda")
@target_cuda
def event_commit(op: OpCall, sctx: ScheduleContext) -> Optional[PrimFunc]:
    """Schedule event commit."""
    event = op.args[0]
    impl = event.get_impl()
    if impl not in EVENT_COMMIT_IMPL_DICT:
        return None

    return EVENT_COMMIT_IMPL_DICT[impl](op, sctx)


@register_schedule("event_wait", "cuda")
@target_cuda
def event_wait(op: OpCall, sctx: ScheduleContext) -> Optional[PrimFunc]:
    """Schedule event wait."""
    event = op.args[0]
    impl = event.get_impl()
    if impl not in EVENT_WAIT_IMPL_DICT:
        return None

    return EVENT_WAIT_IMPL_DICT[impl](op, sctx)


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
    if sctx.exec_scope.name != "cta":
        return None

    evt, expected_count = op.args
    mbar, phase, tx_cnt = evt.get_state()

    if isinstance(evt, SemaphoreEventTensorItem):

        @T.prim_func(tirp=True, check_well_formed=False)
        def func():
            phase[*evt.indices] = 0
            tx_cnt[*evt.indices] = 0
            with T.thread()[0:1]:
                T.evaluate(T.ptx.mbarrier.init(mbar.ptr_to([*evt.indices]), expected_count))
            T.tvm_storage_sync("shared")

        return func
    elif isinstance(evt, SemaphoreEventTensor):

        @T.prim_func(tirp=True, check_well_formed=False)
        def func():
            for i in T.grid(*evt.shape):
                phase[i] = 0
                tx_cnt[i] = 0
            with T.thread()[0:1]:
                for i in T.grid(*evt.shape):
                    T.evaluate(T.ptx.mbarrier.init(mbar.ptr_to([i]), expected_count))
            T.tvm_storage_sync("shared")

        return func
    else:
        raise ValueError(f"Invalid event type: {type(evt)}")


@target_event_impl("event_commit", EventImpl.kTMALoad)
def event_commit(op: OpCall, sctx: ScheduleContext) -> Optional[PrimFunc]:
    """Schedule event commit."""
    if sctx.exec_scope.name != "cta":
        return None

    evt = op.args[0]
    assert isinstance(evt, SemaphoreEventTensorItem), "event must be a SemaphoreEventTensor"
    mbar, phase, tx_cnt = evt.get_state()

    @T.prim_func(tirp=True, check_well_formed=False)
    def func():
        with T.thread()[0:1]:
            T.evaluate(
                T.ptx.mbarrier.arrive.expect_tx(mbar.ptr_to([*evt.indices]), tx_cnt[*evt.indices])
            )

    return func


@target_event_impl("event_wait", EventImpl.kTMALoad)
def event_wait(op: OpCall, sctx: ScheduleContext) -> Optional[PrimFunc]:
    """Schedule event wait."""
    if sctx.exec_scope.name != "cta":
        return None

    evt = op.args[0]
    assert isinstance(evt, SemaphoreEventTensorItem), "event must be a SemaphoreEventTensor"
    mbar, phase, tx_cnt = evt.get_state()

    @T.prim_func(tirp=True, check_well_formed=False)
    def func():
        T.evaluate(T.ptx.mbarrier.try_wait(mbar.ptr_to([*evt.indices]), phase[*evt.indices]))
        tx_cnt[*evt.indices] = 0
        phase[*evt.indices] = phase[*evt.indices] ^ 1
        T.tvm_storage_sync("shared")

    return func


######################## kTMALoadOnly ########################
@target_event_impl("event_init", EventImpl.kTMALoadOnly)
def event_init(op: OpCall, sctx: ScheduleContext) -> Optional[PrimFunc]:
    """Schedule event initialization."""
    if sctx.exec_scope.name != "cta":
        return None

    evt, expected_count = op.args
    mbar, phase, tx_cnt = evt.get_state()

    if isinstance(evt, SemaphoreEventTensor):
        # init all mbars in the tensor
        @T.prim_func(tirp=True, check_well_formed=False)
        def func():
            tx_cnt[0] = 0
            with T.thread()[0:1]:
                for i in T.grid(*evt.shape):
                    T.ptx.mbarrier.init(mbar.ptr_to([i]), expected_count)
            T.tvm_storage_sync("shared")

        return func
    elif isinstance(evt, SemaphoreEventTensorItem):
        # init the mbar
        @T.prim_func(tirp=True, check_well_formed=False)
        def func():
            tx_cnt[0] = 0
            with T.thread()[0:1]:
                T.ptx.mbarrier.init(mbar.ptr_to(evt.indices), expected_count)
            T.tvm_storage_sync("shared")

        return func
    else:
        raise ValueError(f"Invalid event type: {type(evt)}")


@target_event_impl("event_commit", EventImpl.kTMALoadOnly)
def event_commit(op: OpCall, sctx: ScheduleContext) -> Optional[PrimFunc]:
    """Schedule event commit."""
    if sctx.exec_scope.name != "warp":
        return None

    evt = op.args[0]
    assert isinstance(evt, SemaphoreEventTensorItem), "evt must be a SemaphoreEventTensorItem"
    mbar, phase, tx_cnt = evt.get_state()

    @T.prim_func(tirp=True, check_well_formed=False)
    def func():
        with T.thread()[T.ptx.elect_sync()]:
            T.ptx.mbarrier.arrive.expect_tx(mbar.ptr_to(evt.indices), tx_cnt[0])
            tx_cnt[0] = 0

    return func


@target_event_impl("event_wait", EventImpl.kTMALoadOnly)
def event_wait(op: OpCall, sctx: ScheduleContext) -> Optional[PrimFunc]:
    """Schedule event wait."""
    if sctx.exec_scope.name != "warp":
        return None

    evt = op.args[0]
    assert isinstance(evt, SemaphoreEventTensorItem), "evt must be a SemaphoreEventTensorItem"
    mbar, phase, tx_cnt = evt.get_state()

    @T.prim_func(tirp=True, check_well_formed=False)
    def func():
        T.ptx.mbarrier.try_wait(mbar.ptr_to(evt.indices), phase[0])

    return func


######################## kTMAStore ########################
@target_event_impl("event_init", EventImpl.kTMAStore)
def event_init(op: OpCall, sctx: ScheduleContext) -> Optional[PrimFunc]:
    """Schedule event initialization."""
    if sctx.exec_scope.name != "cta":
        return None

    @T.prim_func(tirp=True, check_well_formed=False)
    def func():
        pass

    return func


@target_event_impl("event_commit", EventImpl.kTMAStore)
def event_commit(op: OpCall, sctx: ScheduleContext) -> Optional[PrimFunc]:
    """Schedule event commit."""
    if sctx.exec_scope.name != "cta":
        return None

    @T.prim_func(tirp=True, check_well_formed=False)
    def func():
        T.ptx.cp_async.bulk.commit_group()

    return func


@target_event_impl("event_wait", EventImpl.kTMAStore)
def event_wait(op: OpCall, sctx: ScheduleContext) -> Optional[PrimFunc]:
    """Schedule event wait."""
    if sctx.exec_scope.name != "cta":
        return None

    n = op.args[1]

    @T.prim_func(tirp=True, check_well_formed=False)
    def func():
        T.ptx.cp_async.bulk.wait_group(n)

    return func


######################## kCpAsync ########################
@target_event_impl("event_init", EventImpl.kCpAsync)
def event_init(op: OpCall, sctx: ScheduleContext) -> Optional[PrimFunc]:
    """Schedule event initialization."""
    if sctx.exec_scope.name != "cta":
        return None

    @T.prim_func(tirp=True, check_well_formed=False)
    def func():
        T.evaluate(0)

    return func


@target_event_impl("event_commit", EventImpl.kCpAsync)
def event_commit(op: OpCall, sctx: ScheduleContext) -> Optional[PrimFunc]:
    """Schedule event commit."""
    if sctx.exec_scope.name != "cta":
        return None

    @T.prim_func(tirp=True, check_well_formed=False)
    def func():
        T.evaluate(T.ptx.cp_async.commit_group())

    return func


@target_event_impl("event_wait", EventImpl.kCpAsync)
def event_wait(op: OpCall, sctx: ScheduleContext) -> Optional[PrimFunc]:
    """Schedule event wait."""
    if sctx.exec_scope.name != "cta":
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
