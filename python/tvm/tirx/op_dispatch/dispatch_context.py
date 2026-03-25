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
"""TIRx operator dispatch context."""

from tvm_ffi import register_object

from tvm.ir import Range
from tvm.runtime import Object, Scriptable
from tvm.target import Target
from tvm.tir import Buffer, IterVar, Stmt, Var
from tvm.tir.exec_scope import ExecScope
from tvm.tirx import _ffi_api


@register_object("tirx.DispatchContext")
class DispatchContext(Object, Scriptable):
    """DispatchContext node.

    Parameters
    ----------
    target : Target
        The target of the dispatch context.

    exec_scope : ExecScope
        The execution scope of the dispatch context.

    launch_params : Dict[str, PrimExpr]
        The launch parameters of the dispatch context.

    var_range_map : Dict[Var, Range]
        A map from loop variables to their ranges.

    callbacks : Dict[str, Object]
        The callbacks of the dispatch context.

    shared_state : Dict[str, Object]
        Shared state persisting across dispatch calls within a single lowering pass.
    """

    target: Target
    exec_scope: ExecScope
    launch_params: dict[str, IterVar]
    var_range_map: dict[Var, Range]
    alloc_only: bool
    callbacks: dict[str, Object]
    shared_state: dict[str, Object]

    kPrivateAlloc = "private_alloc"
    kDeviceInitStmt = "device_init_stmt"
    kHostInitStmt = "host_init_stmt"
    kPostBufferDefStmt = "post_buffer_def_stmt"

    def __init__(
        self,
        target: Target,
        exec_scope: ExecScope,
        launch_params: dict[str, IterVar],
        var_range_map: dict[Var, Range],
        alloc_only: bool = False,
        callbacks: dict[str, Object] = {},
        shared_state: dict[str, Object] = {},
    ) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.DispatchContext,  # pylint: disable=no-member
            target,
            exec_scope,
            launch_params,
            var_range_map,
            alloc_only,
            callbacks,
            shared_state,
        )

    def add_alloc_buffer(self, buffer: Buffer) -> None:
        """Add an allocated buffer to the dispatch context.
           Can be called only if alloc_only is True.
           The buffer will be added to the workspace of operator (the key in the workspace is the buffer name).

        Parameters
        ----------
        buffer : Buffer
            The buffer to be added.
        """  # noqa: E501
        _ffi_api.DispatchContextAddAllocBuffer(self, buffer)  # pylint: disable=no-member

    def add_init_stmt(self, stmt: Stmt, host: bool = False) -> None:
        """Add an initialization statement to the dispatch context.
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
        """  # noqa: E501
        _ffi_api.DispatchContextAddInitStmt(self, stmt, host)  # pylint: disable=no-member

    def add_post_buffer_def_stmt(self, buffer: Buffer, stmt: Stmt) -> None:
        """Add a statement to be inserted after a buffer's definition (DeclBuffer/AllocBuffer).

        Parameters
        ----------
        buffer : Buffer
            The buffer whose definition scope the statement should appear in.
        stmt : Stmt
            The statement to be inserted.
        """
        _ffi_api.DispatchContextAddPostBufferDefStmt(self, buffer, stmt)  # pylint: disable=no-member

    def cache_get(self, key: str) -> Object | None:
        """Look up a cached value by key.

        Parameters
        ----------
        key : str
            Cache key (built by the caller from construction parameters).

        Returns
        -------
        Optional[Object]
            The cached value, or None on miss.
        """
        return _ffi_api.DispatchContextSharedStateGet(self, key)

    def cache_set(self, key: str, value: Object) -> None:
        """Store a value in the cross-dispatch cache.

        Parameters
        ----------
        key : str
            Cache key (built by the caller from construction parameters).
        value : Object
            The object to cache (e.g. a Buffer or Var).
        """
        _ffi_api.DispatchContextSharedStateSet(self, key, value)

    def is_cuda(self) -> bool:
        """Check if the target is CUDA."""
        return self.target.kind.name == "cuda"

    def is_trn(self) -> bool:
        """Check if the target is Trainium."""
        return self.target.kind.name == "trn"
