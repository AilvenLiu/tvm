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
from typing import List, Optional, Tuple, Union, Dict

from . import _ffi_api
from tvm._ffi import register_object, get_global_func
from tvm.runtime import Object, convert_to_object

from .expr import Var, PrimExpr
from .exec_scope import ExecScope


@register_object("tirx.TLayout")
class TLayout(Object):
    def __init__(self):
        self.__init_handle_by_constructor__(_ffi_api.TLayout)


@register_object("tirx.IterTreeBase")
class IterTreeBase(Object):
    def __init__(self):
        self.__init_handle_by_constructor__(_ffi_api.IterTreeBase)


@register_object("tirx.IterTreeSplit")
class IterTreeSplit(Object):
    extent: PrimExpr
    children: List[IterTreeBase]

    def __init__(self, extent: PrimExpr, children: List[IterTreeBase]):
        self.__init_handle_by_constructor__(_ffi_api.IterTreeSplit, extent, children)


@register_object("tirx.IterTree")
class IterTree(Object):
    root: IterTreeSplit

    def __init__(self, root: IterTreeSplit):
        self.__init_handle_by_constructor__(_ffi_api.IterTree, root)


@register_object("tirx.DataIterTree")
class DataIterTree(IterTree):
    coeff: List[PrimExpr]

    def __init__(self, root: IterTreeSplit, coeff: List[PrimExpr]):
        self.__init_handle_by_constructor__(_ffi_api.DataIterTree, root, coeff)


@register_object("tirx.DeviceIterAttr")
class DeviceIterAttr(Object):
    Split = 0
    Replicate = 1
    Exclusive = 2

    type: int
    bound: Optional[PrimExpr]
    owner: Optional[PrimExpr]

    def __init__(self, type: int, bound: Optional[Var] = None, owner: Optional[PrimExpr] = None):
        self.__init_handle_by_constructor__(_ffi_api.DeviceIterAttr, type, bound, owner)

    @staticmethod
    def replicate():
        return DeviceIterAttr(type=DeviceIterAttr.Replicate)

    @staticmethod
    def split(bound: PrimExpr):
        return DeviceIterAttr(type=DeviceIterAttr.Split, bound=bound)

    @staticmethod
    def exclusive(owner: PrimExpr):
        return DeviceIterAttr(type=DeviceIterAttr.Exclusive, owner=owner)


@register_object("tirx.DeviceIterTree")
class DeviceIterTree(IterTree):
    attrs: List[DeviceIterAttr]

    def __init__(self, root: IterTreeSplit, attrs: List[DeviceIterAttr]):
        self.__init_handle_by_constructor__(_ffi_api.DeviceIterTree, root, attrs)


@dataclass
class S:
    device_index: int


@register_object("tirx.TileLayout")
class TileLayout(TLayout):
    data_tree: DataIterTree
    device_tree: Optional[DeviceIterTree]
    from_scope: Optional[ExecScope]
    to_scope: Optional[ExecScope]

    def __init__(
        self,
        data_tree: DataIterTree,
        device_tree: Optional[DeviceIterTree] = None,
        from_scope: Optional[ExecScope] = None,
        to_scope: Optional[ExecScope] = None,
    ):
        self.__init_handle_by_constructor__(
            _ffi_api.TileLayout, data_tree, device_tree, from_scope, to_scope
        )

    @staticmethod
    def _construct_device_iter_tree(
        device: Union[Tuple, int, PrimExpr]
    ) -> Tuple[DeviceIterTree, List[int]]:
        f = get_global_func("tir.IterTreeFromTuple")
        iter_tree, leaves = f(convert_to_object(device))
        attrs = [DeviceIterAttr(type=DeviceIterAttr.Replicate) for _ in leaves]
        device_iter_tree = DeviceIterTree(root=iter_tree.root, attrs=attrs)
        return device_iter_tree, leaves

    @staticmethod
    def _construct_data_iter_tree(
        data: Union[Tuple, int, PrimExpr, S],
        strides: Union[Tuple, int, PrimExpr],
        device_attrs: Optional[List[DeviceIterAttr]],
        device_leaves: Optional[List[IterTreeBase]],
        inc_leaf_cnt: callable,
    ) -> Tuple[DataIterTree, List[IterTreeBase]]:
        if isinstance(data, (int, PrimExpr)):
            inc_leaf_cnt()
            assert isinstance(
                strides, (int, PrimExpr)
            ), "the leaf of strides must be int or PrimExpr"
            node = IterTreeSplit(extent=data, children=[])
            return DataIterTree(root=node, coeff=[strides]), [node]
        if isinstance(data, S):
            assert data.device_index < len(device_leaves), "device index out of bound"
            device_axis = device_leaves[data.device_index]
            node = IterTreeSplit(extent=device_axis.extent, children=[])
            assert (
                device_attrs[data.device_index].type == DeviceIterAttr.Replicate
            ), "device axis {} can only be bound once".format(data.device_index)
            device_attrs[data.device_index] = DeviceIterAttr.split(bound=inc_leaf_cnt())
            return DataIterTree(root=node, coeff=[strides]), [node]

        children = []
        leaves = []
        coeff = []
        if isinstance(data, tuple):
            assert len(data) == len(strides), "data and strides do not match"
            for d, s in zip(data, strides):
                child, sub_leaves = TileLayout._construct_data_iter_tree(
                    d, s, device_attrs, device_leaves, inc_leaf_cnt
                )
                children.append(child.root)
                leaves.extend(sub_leaves)
                coeff.extend(child.coeff)

        extent = functools.reduce(operator.mul, [c.extent for c in children], 1)
        node = IterTreeSplit(extent=extent, children=children)
        return DataIterTree(root=node, coeff=coeff), leaves

    @staticmethod
    def from_nested_tuple(
        data: Union[Tuple, int],
        strides: Union[Tuple, int],
        device: Optional[Union[Tuple, int]] = None,
        exclusive: Optional[Tuple] = None,
        from_to: Optional[Tuple[str]] = None,
    ) -> "TileLayout":
        leaf_cnt = 0

        def inc_leaf_cnt():
            nonlocal leaf_cnt
            leaf_cnt += 1
            return leaf_cnt - 1

        if not isinstance(data, tuple):
            data = (data,)
        if not isinstance(strides, tuple):
            strides = (strides,)

        if device is None:
            assert exclusive is None, "exclusive must be None if device is None"
            assert from_to is None, "from_to must be None if device is None"

            data_tree, _ = TileLayout._construct_data_iter_tree(
                data, strides, None, None, lambda: 0
            )
            return TileLayout(data_tree=data_tree)

        else:
            if not isinstance(device, tuple):
                device = (device,)

            assert from_to is not None, "from_to must be provided if device is provided"
            assert isinstance(from_to, tuple) and len(from_to) == 2, "from_to must be a tuple of 2"

            device_tree, device_leaves = TileLayout._construct_device_iter_tree(device)
            device_attrs = list(device_tree.attrs)

            data_tree, _ = TileLayout._construct_data_iter_tree(
                data, strides, device_attrs, device_leaves, inc_leaf_cnt
            )
            if exclusive:
                for e in exclusive:
                    axis, owner = e
                    assert axis < len(device_attrs), "device index out of bound"
                    assert (
                        device_attrs[axis].type == DeviceIterAttr.Replicate
                    ), "device axis {} can only either be S or E".format(axis)
                    device_attrs[axis] = DeviceIterAttr.exclusive(owner)
            return TileLayout(
                data_tree=data_tree,
                device_tree=DeviceIterTree(device_tree.root, device_attrs),
                from_scope=ExecScope.create(from_to[0]) if from_to else None,
                to_scope=ExecScope.create(from_to[1]) if from_to else None,
            )

    @staticmethod
    def tile(outer: "TileLayout", inner: "TileLayout") -> "TileLayout":
        return get_global_func("tir.TileLayoutTile")(outer, inner)

    @staticmethod
    def shard(
        shape: Tuple[PrimExpr, int],
        mesh: Tuple,
        strategy: str,
        inner: "TileLayout",
        from_to: Optional[Tuple[str]] = None,
    ) -> "TileLayout":
        assert from_to is not None, "from_to must be provided if device is provided"
        assert isinstance(from_to, tuple) and len(from_to) == 2, "from_to must be a tuple of 2"

        f = get_global_func("tir.IterTreeFromTuple")
        iter_tree, _ = f(convert_to_object(mesh))
        return get_global_func("tir.TileLayoutShard")(
            shape,
            iter_tree,
            strategy,
            inner,
            ExecScope.create(from_to[0]) if from_to else None,
            ExecScope.create(from_to[1]) if from_to else None,
        )

    @staticmethod
    def normalize(layout: "TileLayout") -> "TileLayout":
        return get_global_func("tir.NormalizeTileLayout")(layout)
