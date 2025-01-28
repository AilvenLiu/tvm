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

    @property
    def size(self):
        return get_global_func("tir.TLayoutGetSize")(self)

    @property
    def cosize(self):
        return get_global_func("tir.TLayoutGetCosize")(self)

    def apply(self, *coord: List[PrimExpr], shape: List[PrimExpr] = None) -> PrimExpr:
        if shape is None:
            assert len(coord) == 1, "shape must be provided if coord is not a single element"
            shape = [1]
        return get_global_func("tir.TLayoutApply")(self, coord, shape)

    def is_trivial(self) -> bool:
        """Check if the layout is trivial."""
        return get_global_func("tir.IsTrivialLayout")(self)

    def is_swizzle(self) -> bool:
        """Check if the layout is swizzle."""
        return isinstance(self, SwizzleLayout)
    
@register_object("tirx.DeviceIterAttr")
class DeviceIterAttr(Object):
    Split = 0
    Replicate = 1
    Exclusive = 2

    extent: PrimExpr
    type: int
    bound: Optional[PrimExpr]
    owner: Optional[PrimExpr]

    def __init__(
        self,
        extent: PrimExpr,
        type: int,
        bound: Optional[Var] = None,
        owner: Optional[PrimExpr] = None,
    ):
        self.__init_handle_by_constructor__(_ffi_api.DeviceIterAttr, extent, type, bound, owner)

    @staticmethod
    def replicate(extent: PrimExpr):
        return DeviceIterAttr(extent=extent, type=DeviceIterAttr.Replicate)

    @staticmethod
    def split(extent: PrimExpr, bound: PrimExpr):
        return DeviceIterAttr(extent=extent, type=DeviceIterAttr.Split, bound=bound)

    @staticmethod
    def exclusive(extent: PrimExpr, owner: PrimExpr):
        return DeviceIterAttr(extent=extent, type=DeviceIterAttr.Exclusive, owner=owner)


@register_object("tirx.DataIterAttr")
class DataIterAttr(Object):
    extent: PrimExpr
    stride: PrimExpr

    def __init__(self, extent: PrimExpr, stride: PrimExpr):
        self.__init_handle_by_constructor__(_ffi_api.DataIterAttr, extent, stride)


@dataclass
class S:
    device_index: int


@register_object("tirx.TileLayout")
class TileLayout(TLayout):
    data_iter_array: List[DataIterAttr]
    device_iter_array: List[DeviceIterAttr]
    from_scope: Optional[ExecScope]
    to_scope: Optional[ExecScope]

    def __init__(
        self,
        data_iter_array: List[DataIterAttr],
        device_iter_array: Optional[List[DeviceIterAttr]] = None,
        from_scope: Optional[ExecScope] = None,
        to_scope: Optional[ExecScope] = None,
    ):
        device_iter_array = device_iter_array or []
        self.__init_handle_by_constructor__(
            _ffi_api.TileLayout, data_iter_array, device_iter_array, from_scope, to_scope
        )

    @staticmethod
    def _get_default_strides(data: Tuple[int], stride: int) -> Tuple:
        assert isinstance(data, tuple), "data must be a tuple"
        res = list()
        for t in reversed(data):
            assert isinstance(t, (int, PrimExpr)), "data must be int or PrimExpr"
            res.append(stride)
            stride *= t
        return tuple(reversed(res))

    @staticmethod
    def _construct_device_iter_tree(device: Union[int, PrimExpr]) -> List[DeviceIterAttr]:
        return [DeviceIterAttr.replicate(e) for e in device]

    @staticmethod
    def _construct_data_device_iter_array(
        data: Tuple[int, PrimExpr, S], strides: Tuple[int, PrimExpr], devices: Tuple[int, PrimExpr]
    ) -> List[DataIterAttr]:
        device_iter_array = [DeviceIterAttr.replicate(e) for e in devices]
        data_iter_array = []
        for i, (d, s) in enumerate(zip(data, strides)):
            extent = d
            if isinstance(d, S):
                assert d.device_index < len(device_iter_array), "device index out of bound"
                assert (
                    device_iter_array[d.device_index].type == DeviceIterAttr.Replicate
                ), "device axis {} can be bound for only one time".format(d.device_index)
                extent = device_iter_array[d.device_index].extent
                device_iter_array[d.device_index] = DeviceIterAttr.split(extent, i)
            data_iter_array.append(DataIterAttr(extent=extent, stride=s))

        return data_iter_array, device_iter_array

    @staticmethod
    def from_nested_tuple(
        data: Union[Tuple, int, PrimExpr, S],
        strides: Optional[Union[Tuple, int, PrimExpr]] = None,
        device: Optional[Union[Tuple, int, PrimExpr]] = None,
        exclusive: Optional[Tuple] = None,
        from_to: Optional[Tuple[str]] = None,
    ) -> "TileLayout":
        leaf_cnt = 0

        def inc_leaf_cnt():
            nonlocal leaf_cnt
            leaf_cnt += 1
            return leaf_cnt - 1

        if isinstance(data, list):
            data = tuple(data)
        elif not isinstance(data, tuple):
            data = (data,)

        if strides is None:
            # get the default strides from the data
            strides = TileLayout._get_default_strides(data, 1)

        if not isinstance(strides, tuple):
            strides = (strides,)
        assert len(data) == len(strides), "data and strides must have the same length"

        if device is None:
            device = tuple()
            assert exclusive is None, "exclusive must be None if device is None"
            assert from_to is None, "from_to must be None if device is None"
        else:
            assert from_to is not None, "from_to must be provided if device is provided"
            assert isinstance(from_to, tuple) and len(from_to) == 2, "from_to must be a tuple of 2"

        if not isinstance(device, tuple):
            device = (device,)
        data_iter_array, device_iter_array = TileLayout._construct_data_device_iter_array(
            data, strides, device
        )
        if exclusive:
            for e in exclusive:
                axis, owner = e
                assert axis < len(device_iter_array), "device index out of bound"
                assert (
                    device_iter_array[axis].type == DeviceIterAttr.Replicate
                ), "device axis {} can only either be S or E".format(axis)
                device_iter_array[axis] = DeviceIterAttr.exclusive(
                    device_iter_array[axis].extent, owner
                )
        return TileLayout(
            data_iter_array=data_iter_array,
            device_iter_array=device_iter_array,
            from_scope=ExecScope.create(from_to[0]) if from_to else None,
            to_scope=ExecScope.create(from_to[1]) if from_to else None,
        )

    @staticmethod
    def tile(
        outer: "TileLayout",
        inner: "TileLayout",
        outer_shape: List[PrimExpr],
        inner_shape: List[PrimExpr],
    ) -> "TileLayout":
        return get_global_func("tir.TileLayoutTile")(outer, inner, outer_shape, inner_shape)

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
        return get_global_func("tir.TileLayoutShard")(
            shape,
            mesh,
            strategy,
            inner,
            ExecScope.create(from_to[0]) if from_to else None,
            ExecScope.create(from_to[1]) if from_to else None,
        )

    @staticmethod
    def normalize(layout: "TileLayout") -> "TileLayout":
        return get_global_func("tir.NormalizeTileLayout")(layout)

    @staticmethod
    def is_tile_inner(
        tile_layout: "TileLayout",
        inner: "TileLayout",
        tiled_shape: List[PrimExpr],
        inner_shape: List[PrimExpr],
    ) -> bool:
        # assume outer must be continuous with exactly one layer
        return get_global_func("tir.IsTileLayout_Inner")(
            tile_layout, inner, tiled_shape, inner_shape
        )

    @staticmethod
    def is_tile_outer(
        tile_layout: "TileLayout",
        outer: "TileLayout",
        tiled_shape: List[PrimExpr],
        outer_shape: List[PrimExpr],
    ) -> bool:
        return get_global_func("tir.IsTileLayout_Outer")(
            tile_layout, outer, tiled_shape, outer_shape
        )

    @staticmethod
    def find_optimal_vec_len(layout_A: "TileLayout", layout_B: "TileLayout") -> int:
        return get_global_func("tir.Vec_Len")(layout_A, layout_B)


@register_object("tirx.SwizzleLayout")
class SwizzleLayout(TLayout):
    """A memory layout that swizzles elements to improve memory access patterns."""

    per_element: int
    swizzle_len: int
    atom_len: int
    swizzle_inner: bool

    def __init__(
        self, per_element: int, swizzle_len: int, atom_len: int, swizzle_inner: bool = True
    ):
        self.__init_handle_by_constructor__(
            _ffi_api.SwizzleLayout,  # pylint: disable=no-member
            per_element,
            swizzle_len,
            atom_len,
            swizzle_inner,
        )
