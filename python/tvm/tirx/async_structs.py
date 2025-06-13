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
from typing import Union, Dict, Any

import tvm
from tvm.ffi import register_object
from tvm.runtime import Object
from tvm.tirx import BufferRegion, Buffer, OpCall
from tvm.tirx.exec_scope import ExecScope
from . import _ffi_api


def make_op_call(
    op_name: str, args, workspace: Dict[str, Buffer] = None, schedule_config: Dict[str, Any] = None
):
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
    if workspace is None:
        workspace = {}
    if schedule_config is None:
        schedule_config = {}

    from tvm.tirp.operator import get_tirp_op

    f = tvm.get_global_func("script.ir_builder.tir.OpCall")
    return f(
        OpCall(*args, op=get_tirp_op(op_name), workspace=workspace, schedule_config=schedule_config)
    )


@register_object("tirx.Pipeline")
class Pipeline(Object):
    """A pipeline object for managing asynchronous operations."""

    thread_scope: ExecScope
    name_hint: str
    depth: int
    separate_pc: bool
    workspace: Dict[str, Buffer]
    schedule_config: Dict[str, Any]

    def __init__(
        self,
        thread_scope: ExecScope,
        depth: int = 0,
        separate_pc: bool = False,
        name_hint: str = "",
        workspace: Dict[str, Buffer] = None,
        schedule_config: Dict[str, Any] = None,
    ):
        if workspace is None:
            workspace = {}
        if schedule_config is None:
            schedule_config = {}
        self.__init_handle_by_constructor__(
            _ffi_api.Pipeline,
            thread_scope,
            name_hint,
            depth,
            separate_pc,
            workspace,
            schedule_config,
        )

    def init(self):
        """Initialize the pipeline."""
        return make_op_call("pipeline_init", [self])

    def producer_acquire(self):
        """Acquire the producer stage."""
        return make_op_call("pipeline_producer_acquire", [self])

    def producer_commit(self, tma_bytes: int = -1):
        """Commit the producer stage."""
        return make_op_call("pipeline_producer_commit", [self, tma_bytes])

    def consumer_wait(self, num_stages: int = -1):
        """Wait for the consumer stage."""
        return make_op_call("pipeline_consumer_wait", [self, num_stages])

    def consumer_release(self):
        """Release the consumer stage."""
        return make_op_call("pipeline_consumer_release", [self])


def _to_region(buffer: Union[BufferRegion, Buffer]):
    if isinstance(buffer, Buffer):
        return buffer[[slice(None, None, None) for _ in range(len(buffer.shape))]]
    assert isinstance(buffer, BufferRegion)
    return buffer

@register_object("tirx.CopyPipeline")
class CopyPipeline(Pipeline):
    """A pipeline for copying data asynchronously."""

    class StrategyKind:
        """The strategy of the pipeline."""

        IMPL = "impl"

    class Impl:
        """Strategy: The implementation of the pipeline."""

        VEC_LOAD = "vec_load"
        TMA_LOAD = "tma_load"
        TMA_STORE = "tma_store"

    def copy(self, dst: Union[BufferRegion, Buffer], src: Union[BufferRegion, Buffer]):
        """Copy data asynchronously from the source to the destination."""
        dst = _to_region(dst)
        src = _to_region(src)
        return make_op_call("pipeline_copy", [self, dst, src])
