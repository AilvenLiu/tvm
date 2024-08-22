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
"""Definition of execution scope."""
from typing import List

from . import _ffi_api
from tvm._ffi import register_object, get_global_func
from tvm.runtime import Object

from .expr import PrimExpr, Var
from ..ir import Range


@register_object("tirx.ScopeIdDef")
class ScopeIdDef(Object):
    def_ids: List[Var]
    extents: List[PrimExpr]
    parent: str
    cur: str

    def __init__(self, def_ids: List[Var], extents: List[PrimExpr], parent: str, cur: str):
        self.__init_handle_by_constructor__(_ffi_api.ScopeIdDef, def_ids, extents, parent, cur)


@register_object("tirx.ExecScope")
class ExecScope(Object):
    name: str

    def __init__(self, name: str):
        self.__init_handle_by_constructor__(_ffi_api.ExecScope, name)

    @staticmethod
    def create(name: str):
        return get_global_func("tir.ExecScopeCreate")(name)


@register_object("tirx.WorldScope")
class WorldScope(ExecScope):
    scope_id_def: ScopeIdDef

    def __init__(self, scope_id_def: ScopeIdDef):
        self.__init_handle_by_constructor__(_ffi_api.WorldScope, scope_id_def)


@register_object("tirx.KernelScope")
class KernelScope(ExecScope):
    scope_id_def: List[ScopeIdDef]

    def __init__(self, scope_id_def: List[ScopeIdDef]):
        self.__init_handle_by_constructor__(_ffi_api.KernelScope, scope_id_def)


@register_object("tirx.ExecScopeSlice")
class ExecScopeSlice(ExecScope):
    def_ids: List[Var]
    ranges: List[Range]

    def __init__(self, ids: List[PrimExpr], ranges: List[Range], name: str):
        self.__init_handle_by_constructor__(_ffi_api.ExecScopeSlice, ids, ranges, name)
