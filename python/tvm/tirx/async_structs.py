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
"""Async structures for TIR+"""
from . import _ffi_api
import tvm
from typing import Union
from tvm._ffi import register_object
from tvm.runtime import Object
from tvm.ir import Op
from tvm.tirx import BufferRegion, Buffer, PrimExpr
from tvm.tirx.exec_scope import ExecScope


def _get_tirp_op(op_name: str):
    assert isinstance(op_name, str)
    return Op.get("tirp." + op_name)


def make_op_call(op_name: str, args):
    f = tvm.get_global_func("script.ir_builder.tir.OpCall")
    return f(_get_tirp_op(op_name), args)


@register_object("tirx.Barrier")
class Barrier(Object):
    thread_scope: ExecScope
    name_hint: str

    def __init__(self, thread_scope: ExecScope, name_hint: str = ""):
        self.__init_handle_by_constructor__(_ffi_api.Barrier, thread_scope, name_hint)

    def init(self, count: int):
        """Initialize the barrier with the given count.

        Parameters
        ----------
        count : int
            The expected count of the barrier.
        """
        return make_op_call("barrier_init", [self, count])

    def arrive(self):
        """Arrive the barrier."""
        return make_op_call("barrier_arrive", [self])

    def wait(self):
        """Wait for the barrier."""
        return make_op_call("barrier_wait", [self])

    def arrive_and_wait(self):
        """Arrive and wait for the barrier."""
        return make_op_call("barrier_arrive_and_wait", [self])


@register_object("tirx.BarrierArrayElem")
class BarrierArrayElem(Barrier):
    arr: "BarrierArray"
    index: PrimExpr

    def __init__(self, barrier_array: "BarrierArray", index: PrimExpr):
        self.__init_handle_by_constructor__(_ffi_api.BarrierArrayElem, barrier_array, index)


@register_object("tirx.BarrierArray")
class BarrierArray(Object):
    thread_scope: ExecScope
    size: int
    name_hint: str

    def __init__(self, thread_scope: ExecScope, size: int, name_hint: str = ""):
        self.__init_handle_by_constructor__(_ffi_api.BarrierArray, thread_scope, size, name_hint)

    def __getitem__(self, index: PrimExpr):
        return BarrierArrayElem(self, index)


@register_object("tirx.Pipeline")
class Pipeline(Object):
    thread_scope: ExecScope
    name_hint: str
    depth: int
    specialize: bool

    def __init__(
        self, thread_scope: ExecScope, depth: int = 0, specialize: bool = False, name_hint: str = ""
    ):
        self.__init_handle_by_constructor__(
            _ffi_api.Pipeline, thread_scope, name_hint, depth, specialize
        )

    def producer_acquire(self):
        return make_op_call("pipeline_producer_acquire", [self])

    def producer_copy_async(
        self, dst: Union[BufferRegion, Buffer], src: Union[BufferRegion, Buffer]
    ):
        return make_op_call("pipeline_producer_copy_async", [self, dst, src])

    def producer_commit_stage(self):
        return make_op_call("pipeline_producer_commit_stage", [self])

    def consumer_wait(self, num_stages: int = -1):
        return make_op_call("pipeline_consumer_wait", [self, num_stages])

    def consumer_release(self):
        return make_op_call("pipeline_consumer_release", [self])
