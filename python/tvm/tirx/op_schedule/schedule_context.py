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
"""TIRx operator schedule context."""
from typing import Dict, List

from tvm_ffi import register_object
from tvm.ir import Range
from tvm.tir import PrimExpr, Var, Buffer, Stmt, IterVar
from tvm.runtime import Object, Scriptable
from tvm.target import Target
from tvm.tir.exec_scope import ExecScope
from tvm.tirx import _ffi_api


@register_object("tirx.ScheduleContext")
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

    callbacks : Dict[str, Object]
        The callbacks of the schedule context.
    """

    target: Target
    exec_scope: ExecScope
    launch_params: Dict[str, IterVar]
    var_range_map: Dict[Var, Range]
    alloc_only: bool
    callbacks: Dict[str, Object]

    kPrivateAlloc = "private_alloc"
    kDeviceInitStmt = "device_init_stmt"
    kHostInitStmt = "host_init_stmt"
    kPostBufferDefStmt = "post_buffer_def_stmt"

    def __init__(
        self,
        target: Target,
        exec_scope: ExecScope,
        launch_params: Dict[str, IterVar],
        var_range_map: Dict[Var, Range],
        alloc_only: bool = False,
        callbacks: Dict[str, Object] = {},
    ) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.ScheduleContext,  # pylint: disable=no-member
            target,
            exec_scope,
            launch_params,
            var_range_map,
            alloc_only,
            callbacks,
        )

    def add_alloc_buffer(self, buffer: Buffer) -> None:
        """Add an allocated buffer to the schedule context.
           Can be called only if alloc_only is True.
           The buffer will be added to the workspace of operator (the key in the workspace is the buffer name).

        Parameters
        ----------
        buffer : Buffer
            The buffer to be added.
        """
        _ffi_api.ScheduleContextAddAllocBuffer(self, buffer)  # pylint: disable=no-member

    def add_init_stmt(self, stmt: Stmt, host: bool = False) -> None:
        """Add an initialization statement to the schedule context.
           Device initialization statements is only allowed if alloc_only is True.
           Host initialization statements will be ignored if alloc_only is True.
           The statements will be added to the beginning of the kernel.

        Parameters
        ----------
        stmt : Stmt
            The initialization statement to be added.
        host : bool
            Whether the statement is a host statement.
            If True, the statement will be added to the host code (before the kernel).
            If False, the statement will be added to the kernel body (at the beginning of the kernel).
        """
        _ffi_api.ScheduleContextAddInitStmt(self, stmt, host)  # pylint: disable=no-member

    def add_post_buffer_def_stmt(self, buffer: Buffer, stmt: Stmt) -> None:
        """Add a statement to be inserted after a buffer's definition (DeclBuffer/AllocBuffer).

        Parameters
        ----------
        buffer : Buffer
            The buffer whose definition scope the statement should appear in.
        stmt : Stmt
            The statement to be inserted.
        """
        _ffi_api.ScheduleContextAddPostBufferDefStmt(self, buffer, stmt)  # pylint: disable=no-member

    def is_cuda(self) -> bool:
        """Check if the target is CUDA."""
        return self.target.kind.name == "cuda"

    def is_trn(self) -> bool:
        """Check if the target is Trainium."""
        return self.target.kind.name == "trn"
