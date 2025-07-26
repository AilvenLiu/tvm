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
"""TIRp event"""

from enum import IntEnum
from typing import List, Any

import tvm.ffi

from tvm.runtime import Object
from ..tirp import _ffi_api
from tvm.tir.stmt import OpCall
from tvm.ir import Op
from tvm.tir import PrimExpr


def make_op_call(op_name: str, *args):
    assert isinstance(op_name, str)
    f = tvm.get_global_func("script.ir_builder.tir.OpCall")
    return f(OpCall(*args, op=Op.get("tirp." + op_name)))


class EventImpl(IntEnum):
    kCpAsync = 0
    kTMALoad = 1
    kTMAStore = 2
    kGlobalSemaphore = 3


@tvm.ffi.register_object("tirp.BaseEvent")
class BaseEvent(Object):
    """Base event"""

    def __init__(self, name: str):
        raise NotImplementedError("BaseEvent is not instantiable")


@tvm.ffi.register_object("tirp.SemaphoreEvent")
class SemaphoreEvent(BaseEvent):
    """Semaphore event"""

    expected_count: int
    impl: EventImpl
    state: List[Any]
    name: str

    def __init__(self, expected_count: int, impl: EventImpl, state: List[Any] = [], name: str = ""):
        self.__init_handle_by_constructor__(
            _ffi_api.SemaphoreEvent, expected_count, impl, state, name
        )

    def init(self):
        return make_op_call("event_init", self)

    def commit(self):
        return make_op_call("event_commit", self)

    def wait(self):
        return make_op_call("event_wait", self)


@tvm.ffi.register_object("tirp.BulkGroupEvent")
class BulkGroupEvent(BaseEvent):
    """Bulk group event"""

    impl: EventImpl
    state: List[Any]
    name: str

    def __init__(self, impl: EventImpl, state: List[Any] = [], name: str = ""):
        self.__init_handle_by_constructor__(_ffi_api.BulkGroupEvent, impl, state, name)

    def init(self):
        return make_op_call("event_init", self)

    def commit(self):
        return make_op_call("event_commit", self)

    def wait(self, n_groups: int = 0):
        return make_op_call("event_wait", self, n_groups)


@tvm.ffi.register_object("tirp.EventTensorItem")
class EventTensorItem(BaseEvent):
    """Event tensor item"""

    tensor: "EventTensor"
    indices: List[PrimExpr]

    def __init__(self, tensor: "EventTensor", indices: List[PrimExpr]):
        self.__init_handle_by_constructor__(_ffi_api.EventTensorItem, tensor, indices)

    def init(self):
        return make_op_call("event_init", self)

    def commit(self):
        return make_op_call("event_commit", self)

    def wait(self):
        return make_op_call("event_wait", self)


@tvm.ffi.register_object("tirp.EventTensor")
class EventTensor(Object):
    """Event tensor"""

    event: SemaphoreEvent
    shape: List[PrimExpr]

    def __init__(self, event: SemaphoreEvent, shape: List[PrimExpr]):
        assert isinstance(
            event, SemaphoreEvent
        ), "EventTensor can only be a tensor of SemaphoreEvents"
        self.__init_handle_by_constructor__(_ffi_api.EventTensor, event, shape)

    def __getitem__(self, *indices: List[PrimExpr]):
        assert len(indices) == len(self.shape), "indices must be the same length as shape"
        return EventTensorItem(self, indices)

    def init(self):
        return make_op_call("event_init", self)
