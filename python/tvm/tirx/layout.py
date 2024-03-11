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
"""Definition of layout."""
from typing import List, Optional

from . import _ffi_api
from tvm._ffi import register_object
from tvm.runtime import Object

from .expr import Var, PrimExpr


@register_object("tirx.Layout")
class Layout(Object):
    def __init__(self):
        self.__init_handle_by_constructor__(_ffi_api.Layout)


@register_object("tirx.IterTreeSplit")
class IterTreeSplit(Object):
    parent: Var
    children: List[Var]
    extents: List[PrimExpr]

    def __init__(self, parent: Var, children: List[Var], extents: List[PrimExpr]):
        self.__init_handle_by_constructor__(_ffi_api.IterTreeSplit, parent, children, extents)


@register_object("tirx.IterTree")
class IterTree(Object):
    root: Var
    splits: List[IterTreeSplit]

    def __init__(self, root: Var, splits: List[IterTreeSplit]):
        self.__init_handle_by_constructor__(_ffi_api.IterTree, root, splits)


@register_object("tirx.CoordIterTree")
class CoordIterTree(IterTree):
    coeff: List[PrimExpr]

    def __init__(self, root: Var, splits: List[IterTreeSplit], coeff: List[PrimExpr]):
        self.__init_handle_by_constructor__(_ffi_api.CoordIterTree, root, splits, coeff)


@register_object("tirx.ScopeIdAttr")
class ScopeIdAttr(Object):
    Split = 0
    Replicate = 1
    Exclusive = 2

    type: int
    bound: Optional[Var]
    owner: Optional[PrimExpr]

    def __init__(self, type: int, bound: Optional[Var], owner: Optional[PrimExpr]):
        self.__init_handle_by_constructor__(_ffi_api.ScopeIdAttr, type, bound, owner)


@register_object("tirx.ScopeIdIterTree")
class ScopeIdIterTree(IterTree):
    attrs: List[ScopeIdAttr]

    def __init__(self, root: Var, splits: List[IterTreeSplit], attrs: List[ScopeIdAttr]):
        self.__init_handle_by_constructor__(_ffi_api.ScopeIdIterTree, root, splits, attrs)


@register_object("tirx.TileLayout")
class TileLayout(Layout):
    coord_iter_trees: List[CoordIterTree]
    scope_id_iter_trees: List[ScopeIdIterTree]

    def __init__(
        self, coord_iter_trees: List[CoordIterTree], scope_id_iter_trees: List[ScopeIdIterTree]
    ):
        self.__init_handle_by_constructor__(
            _ffi_api.TileLayout, coord_iter_trees, scope_id_iter_trees
        )
