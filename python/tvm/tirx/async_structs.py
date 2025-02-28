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
# pylint: disable=no-member
"""Async structures for TIR+"""
from . import _ffi_api
import tvm
from typing import Union, Dict
from tvm._ffi import register_object
from tvm.runtime import Object
from tvm.ir import Op
from tvm.tirx import BufferRegion, Buffer
from tvm.tirx.exec_scope import ExecScope


def _get_tirp_op(op_name: str):
    """Get the TIR+ operator by name.

    Parameters
    ----------
    op_name : str
        Name of the operator

    Returns
    -------
    Op
        The requested TIR+ operator
    """
    assert isinstance(op_name, str)
    return Op.get("tirp." + op_name)


def make_op_call(op_name: str, args, workspace: Dict[str, Buffer] = {}):
    """Create a call to a TIR+ operator.

    Parameters
    ----------
    op_name : str
        Name of the operator
    args : list
        Arguments to the operator

    Returns
    -------
    Call
        The constructed operator call
    """
    f = tvm.get_global_func("script.ir_builder.tir.OpCall")
    return f(_get_tirp_op(op_name), args, workspace)


@register_object("tirx.Pipeline")
class Pipeline(Object):
    """A pipeline object for managing asynchronous operations."""

    thread_scope: ExecScope
    name_hint: str
    depth: int
    separate_pc: bool
    workspace: Dict[str, Buffer]
    def __init__(
        self,
        thread_scope: ExecScope,
        depth: int = 0,
        separate_pc: bool = False,
        name_hint: str = "",
    ):
        self.__init_handle_by_constructor__(
            _ffi_api.Pipeline, thread_scope, name_hint, depth, separate_pc
        )

    def producer_acquire(self):
        """Acquire the producer stage."""
        return make_op_call("pipeline_producer_acquire", [self], workspace=self.workspace)

    def producer_commit(self):
        """Commit the producer stage."""
        return make_op_call("pipeline_producer_commit", [self], workspace=self.workspace)

    def consumer_wait(self, num_stages: int = -1):
        """Wait for the consumer stage."""
        return make_op_call("pipeline_consumer_wait", [self, num_stages], workspace=self.workspace)

    def consumer_release(self):
        """Release the consumer stage."""
        return make_op_call("pipeline_consumer_release", [self], workspace=self.workspace)


@register_object("tirx.CopyPipeline")
class CopyPipeline(Pipeline):
    """A pipeline for copying data asynchronously."""

    def copy(self, dst: Union[BufferRegion, Buffer], src: Union[BufferRegion, Buffer]):
        """Copy data asynchronously from the source to the destination."""
        return make_op_call("pipeline_copy", [self, dst, src], workspace=self.workspace)
