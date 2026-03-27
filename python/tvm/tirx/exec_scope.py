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
# pylint: disable=no-member, super-init-not-called

"""Definition of execution scope."""

from tvm_ffi import get_global_func, register_object

from tvm.ir import Range
from tvm.runtime import Object

from . import _ffi_api
from .expr import PrimExpr, Var


@register_object("tirx.ScopeIdDef")
class ScopeIdDef(Object):
    """Definition of scope identifiers with their extents and parent-child relationships."""

    def_ids: list[Var]
    extents: list[PrimExpr]
    parent: str
    cur: str

    def __init__(self, def_ids: list[Var], extents: list[PrimExpr], parent: str, cur: str):
        self.__init_handle_by_constructor__(_ffi_api.ScopeIdDef, def_ids, extents, parent, cur)


@register_object("tirx.ExecScope")
class ExecScope(Object):
    """Base class for execution scopes."""

    name: str
    scope_id_def: list[ScopeIdDef]

    def __init__(self, name: str):
        self.__init_handle_by_constructor__(_ffi_api.ExecScope, name)

    @staticmethod
    def create(name: str):
        """Create a new execution scope with the given name.

        Parameters
        ----------
        name : str
            The name of the execution scope.

        Returns
        -------
        ExecScope
            The created execution scope.
        """
        return get_global_func("tirx.ExecScopeCreate")(name)


@register_object("tirx.ExecScopeSlice")
class ExecScopeSlice(ExecScope):
    """A slice of an execution scope with their slices and parent scope name."""

    slices: list[Range] | PrimExpr
    parent: str
    cur: str

    def __init__(self, slices: list[Range] | PrimExpr, parent: str, cur: str):
        self.__init_handle_by_constructor__(_ffi_api.ExecScopeSlice, slices, parent, cur)
