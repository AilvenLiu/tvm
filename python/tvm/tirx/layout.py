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
from dataclasses import dataclass
import functools
import operator
from typing import List, Optional, Tuple, Union

from . import _ffi_api
from tvm._ffi import register_object, get_global_func
from tvm.runtime import Object

from .expr import Var, PrimExpr
from .exec_scope import ExecScope


@register_object("tirx.TLayout")
class TLayout(Object):
    def __init__(self):
        self.__init_handle_by_constructor__(_ffi_api.TLayout)


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


@register_object("tirx.DataIterTree")
class DataIterTree(IterTree):
    coeff: List[PrimExpr]

    def __init__(self, root: Var, splits: List[IterTreeSplit], coeff: List[PrimExpr]):
        self.__init_handle_by_constructor__(_ffi_api.DataIterTree, root, splits, coeff)


@register_object("tirx.DeviceIterAttr")
class DeviceIterAttr(Object):
    Split = 0
    Replicate = 1
    Exclusive = 2

    type: int
    bound: Optional[Var]
    owner: Optional[PrimExpr]

    def __init__(self, type: int, bound: Optional[Var] = None, owner: Optional[PrimExpr] = None):
        self.__init_handle_by_constructor__(_ffi_api.DeviceIterAttr, type, bound, owner)


@register_object("tirx.DeviceIterTree")
class DeviceIterTree(IterTree):
    attrs: List[DeviceIterAttr]

    def __init__(self, root: Var, splits: List[IterTreeSplit], attrs: List[DeviceIterAttr]):
        self.__init_handle_by_constructor__(_ffi_api.DeviceIterTree, root, splits, attrs)


@dataclass
class S:
    device_index: int


@register_object("tirx.TileLayout")
class TileLayout(TLayout):
    data_trees: List[DataIterTree]
    device_trees: List[DeviceIterTree]
    from_scope: Optional[ExecScope]
    to_scope: Optional[ExecScope]

    def __init__(
        self,
        data_trees: List[DataIterTree],
        device_trees: List[DeviceIterTree],
        from_scope: Optional[ExecScope] = None,
        to_scope: Optional[ExecScope] = None,
    ):
        self.__init_handle_by_constructor__(
            _ffi_api.TileLayout, data_trees, device_trees, from_scope, to_scope
        )

    @staticmethod
    def _construct_device_iter_tree(
        device: Union[Tuple, int, PrimExpr]
    ) -> Tuple[DeviceIterTree, int, List[int]]:
        return get_global_func("tir.DeviceIterTreeFromTuple")(device)

    @staticmethod
    def _construct_data_iter_tree(
        data: Union[Tuple, int, PrimExpr, S],
        strides: Union[Tuple, int, PrimExpr],
        scope_id_attrs: List[DeviceIterAttr],
        device_extents: List[int],
    ):
        root = Var("", "int32")
        if isinstance(data, (int, PrimExpr)):
            assert isinstance(strides, (int, PrimExpr))
            return (
                DataIterTree(root=root, splits=[], coeff=[strides]),
                data,
            )
        if isinstance(data, S):
            assert (
                scope_id_attrs[data.device_index].type == DeviceIterAttr.Replicate
            ), f"Scope ID on axis {data.device_index} has been bound to var {scope_id_attrs[data.device_index].bound}."
            scope_id_attrs[data.device_index] = DeviceIterAttr(
                type=DeviceIterAttr.Split, bound=root, owner=None
            )
            return (
                DataIterTree(root=root, splits=[], coeff=[strides]),
                device_extents[data.device_index],
            )
        splits = []
        children = []
        extents = []
        coeff = []
        if isinstance(data, tuple):
            assert len(data) == len(strides)
            root = Var("", dtype="int32")
            splits = []
            for d, s in zip(reversed(data), reversed(strides)):
                child, sub_extent = TileLayout._construct_data_iter_tree(
                    d, s, scope_id_attrs, device_extents
                )
                splits.extend(child.splits)
                children.append(child.root)
                extents.append(sub_extent)
                coeff.extend(child.coeff)
        splits.append(IterTreeSplit(parent=root, children=children, extents=extents))
        return (
            DataIterTree(root=root, splits=splits, coeff=coeff),
            functools.reduce(operator.mul, extents, 1),
        )

    @staticmethod
    def from_nested_tuple(
        data: Tuple,
        strides: Tuple,
        device: Tuple,
        exclusive: Optional[Tuple] = None,
        from_to: Optional[Tuple[str]] = None,
    ):
        device_iter_tree, _, device_extents = TileLayout._construct_device_iter_tree(device)
        scope_id_attrs = list(device_iter_tree.attrs)
        data_iter_tree, _ = TileLayout._construct_data_iter_tree(
            data, strides, scope_id_attrs, device_extents
        )
        if exclusive:
            for e in exclusive:
                axis, owner = e
                scope_id_attrs[axis] = DeviceIterAttr(
                    type=DeviceIterAttr.Exclusive, bound=scope_id_attrs[axis].bound, owner=owner
                )
        return TileLayout(
            data_trees=[data_iter_tree],
            device_trees=[
                DeviceIterTree(
                    root=device_iter_tree.root, splits=device_iter_tree.splits, attrs=scope_id_attrs
                )
            ],
            from_scope=ExecScope.create(from_to[0]) if from_to else None,
            to_scope=ExecScope.create(from_to[1]) if from_to else None,
        )

    @staticmethod
    def from_tile(
        shape: Tuple,
        inner: TLayout,
        device: Optional[Tuple] = None,
        from_to: Optional[Tuple[str]] = None,
    ):
        return get_global_func("tir.TileLayoutFromTile")(shape, inner, device, from_to)
