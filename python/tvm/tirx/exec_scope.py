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
from typing import List, Optional

from . import _ffi_api
from tvm._ffi import register_object
from tvm.runtime import Object

from .expr import PrimExpr, Var
from ..ir import Range


@register_object("tirx.ExecScope")
class ExecScope(Object):
    dims: List[PrimExpr]
    name: str

    def __init__(self, dims: List[PrimExpr], name: str):
        self.__init_handle_by_constructor__(_ffi_api.ExecScope, dims, name)


@register_object("tirx.ThreadingVar")
def ThreadingVar(Var):
    scope: ExecScope

    def __init__(self, scope: ExecScope, name: str):
        self.__init_handle_by_constructor__(_ffi_api.ThreadingVar, scope, name)


@register_object("tirx.SubExecScope")
class SubExecScope(ExecScope):
    def_vars: List[ThreadingVar]
    ranges: Optional[List[Range]]

    def __init__(self, vars: List[PrimExpr], ranges: Optional[List[Range]], name: str):
        self.__init_handle_by_constructor__(_ffi_api.SubExecScope, vars, ranges, name)
