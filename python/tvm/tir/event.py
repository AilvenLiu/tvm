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
from typing import List, Any, Dict, Optional

import tvm_ffi

from tvm.runtime import Object
from ..tirp import _ffi_api
from tvm.tir.stmt import OpCall
from tvm.ir import Op
from tvm.tir import PrimExpr


def make_op_call(
    op_name: str,
    *args,
    dispatch: Optional[str] = None,
    config: Dict[str, Any] = None,
):
    assert isinstance(op_name, str)
    f = tvm_ffi.get_global_func("script.ir_builder.tir.OpCall")
    return f(OpCall(*args, op=Op.get("tirp." + op_name), config=config, dispatch=dispatch))


class EventImpl(IntEnum):
    # see also include/tvm/tir/event.h kEventImpl
    kCpAsync = 0
    kTMALoad = 1  # state: mbarrier, phase, tx_cnt
    kTMAStore = 2
    kGlobalSemaphore = 3  # state sem, state


@tvm_ffi.register_object("tirp.BaseEvent")
class BaseEvent(Object):
    """Base event"""

    def __init__(self, name: str):
        raise NotImplementedError("BaseEvent is not instantiable")

    def get_impl(self) -> EventImpl:
        return _ffi_api.BaseEventImplGet(self)

    def get_state(self) -> List[Any]:
        return _ffi_api.BaseEventStateGet(self)


@tvm_ffi.register_object("tirp.BulkGroupEvent")
class BulkGroupEvent(BaseEvent):
    """Bulk group event"""

    impl: EventImpl
    state: List[Any]
    name: str

    def __init__(self, impl: EventImpl, state: List[Any] = [], name: str = ""):
        self.__init_handle_by_constructor__(_ffi_api.BulkGroupEvent, impl, state, name)

    def init(self):
        return make_op_call("event_init", self)

    def commit(self, dispatch: Optional[str] = None, **kwargs):
        return make_op_call("event_commit", self, dispatch=dispatch, config=kwargs)

    def wait(self, n_groups: int = 0, dispatch: Optional[str] = None, **kwargs):
        return make_op_call("event_wait", self, n_groups, dispatch=dispatch, config=kwargs)


@tvm_ffi.register_object("tirp.SemaphoreEventTensorItem")
class SemaphoreEventTensorItem(BaseEvent):
    """Event tensor item"""

    tensor: "SemaphoreEventTensor"
    indices: List[PrimExpr]

    def __init__(self, tensor: "SemaphoreEventTensor", indices: List[PrimExpr]):
        self.__init_handle_by_constructor__(_ffi_api.SemaphoreEventTensorItem, tensor, indices)

    def init(self, expected_count: PrimExpr, dispatch: Optional[str] = None, **kwargs):
        return make_op_call("event_init", self, expected_count, dispatch=dispatch, config=kwargs)

    def commit(self, dispatch: Optional[str] = None, **kwargs):
        return make_op_call("event_commit", self, dispatch=dispatch, config=kwargs)

    def wait(self, dispatch: Optional[str] = None, **kwargs):
        return make_op_call("event_wait", self, dispatch=dispatch, config=kwargs)


@tvm_ffi.register_object("tirp.SemaphoreEventTensor")
class SemaphoreEventTensor(Object):
    """Event tensor"""

    impl: EventImpl
    state: List[Any]
    name: str
    shape: List[PrimExpr]

    def __init__(
        self, impl: EventImpl, state: List[Any] = [], shape: List[PrimExpr] = [], name: str = ""
    ):
        self.__init_handle_by_constructor__(_ffi_api.SemaphoreEventTensor, impl, state, shape, name)

    def __getitem__(self, *indices: List[PrimExpr]):
        assert len(indices) == len(
            self.shape
        ), "indices must be the same length as shape, but got {} and {}".format(
            len(indices), len(self.shape)
        )
        return SemaphoreEventTensorItem(self, indices)

    def init(self, expected_count: PrimExpr, dispatch: Optional[str] = None, **kwargs):
        return make_op_call("event_init", self, expected_count, dispatch=dispatch, config=kwargs)

    def get_impl(self) -> EventImpl:
        return self.impl

    def get_state(self) -> List[Any]:
        return self.state
