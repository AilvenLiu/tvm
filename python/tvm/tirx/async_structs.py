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
from tvm._ffi import register_object
from tvm.runtime import Object
from tvm.ir import Op
from tvm.tirx import PrimExpr
from tvm.script.ir_builder.tir._ffi_api import OpCall


def _get_tirp_op(op_name: str):
    assert isinstance(op_name, str)
    return Op.get("tirp." + op_name)


@register_object("tirx.Barrier")
class Barrier(Object):
    name_hint: str

    def __init__(self, name_hint: str = ""):
        self.__init_handle_by_constructor__(_ffi_api.Barrier, name_hint)

    def init(self, count: int):
        """Initialize the barrier with the given count.

        Parameters
        ----------
        count : int
            The expected count of the barrier.
        """
        return OpCall(_get_tirp_op("barrier_init"), [self, count])

    def arrive(self):
        """Arrive the barrier."""
        return OpCall(_get_tirp_op("barrier_arrive"), [self])

    def wait(self):
        """Wait for the barrier."""
        return OpCall(_get_tirp_op("barrier_wait"), [self])


@register_object("tirx.BarrierArrayElem")
class BarrierArrayElem(Barrier):
    def __init__(self, barrier_array: "BarrierArray", index: PrimExpr):
        self.__init_handle_by_constructor__(_ffi_api.BarrierArrayElem, barrier_array, index)


@register_object("tirx.BarrierArray")
class BarrierArray(Object):
    size: int

    def __init__(self, size: int):
        self.__init_handle_by_constructor__(_ffi_api.BarrierArray, size)

    def __getitem__(self, index: PrimExpr):
        return BarrierArrayElem(self, index)
