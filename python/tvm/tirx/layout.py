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
# pylint: disable=super-init-not-called
"""Definition of layout."""
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import re
import functools
import operator

from . import _ffi_api
from tvm._ffi import register_object
from tvm.runtime import Object, ShapeTuple
from .expr import Var, PrimExpr
from .exec_scope import ExecScope


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
        self.__init_handle_by_constructor__(
            _ffi_api.DeviceIterAttr,  # pylint: disable=no-member
            extent,
            type,
            bound,
            owner,
        )

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
        self.__init_handle_by_constructor__(
            _ffi_api.DataIterAttr,  # pylint: disable=no-member
            extent,
            stride,
        )


@dataclass
class S:
    device_index: int


@register_object("tirx.TLayout")
class TLayout(Object):
    def __init__(self):
        self.__init_handle_by_constructor__(_ffi_api.TLayout)  # pylint: disable=no-member

    @property
    def size(self):
        return _ffi_api.TLayoutGetSize(self)  # pylint: disable=no-member

    @property
    def cosize(self):
        return _ffi_api.TLayoutGetCosize(self)  # pylint: disable=no-member

    def apply(self, *coord: List[PrimExpr], shape: List[PrimExpr] = None) -> List[PrimExpr]:
        if shape is None:
            assert len(coord) == 1, "shape must be provided if coord is not a single element"
            shape = [1]
        return _ffi_api.TLayoutApply(self, coord, shape)  # pylint: disable=no-member

    def is_trivial(self) -> bool:
        """Check if the layout is trivial."""
        return _ffi_api.IsTrivialLayout(self)  # pylint: disable=no-member

    def is_swizzle(self) -> bool:
        """Check if the layout is swizzle."""
        return isinstance(self, SwizzleLayout)

    def normalize(self) -> "TLayout":
        """Normalize the layout by simplifying and fusing iterators where possible.

        Returns
        -------
        TLayout
            The normalized layout
        """
        raise NotImplementedError("Normalize is not implemented for this layout")

    def tile(
        self,
        outer: "TileLayout",
        outer_shape: List[PrimExpr],
        inner_shape: List[PrimExpr],
    ) -> Union["TileLayout", "ComposeLayout"]:
        """Tile the current layout with an outer layout.

        Parameters
        ----------
        outer : TileLayout
            The outer layout to tile with
        outer_shape : List[PrimExpr]
            The shape of the outer layout
        inner_shape : List[PrimExpr]
            The shape of the inner layout

        Returns
        -------
        Union[TileLayout, ComposeLayout]
            The resulting tiled layout
        """
        raise NotImplementedError("Tile is not implemented for this layout")

    def is_tile_inner(
        self,
        tile_layout: Union["TileLayout", "ComposeLayout"],
        tiled_shape: List[PrimExpr],
        inner_shape: List[PrimExpr],
    ) -> Optional["TileLayout"]:
        """Check if a layout is the inner layout of a tiled layout.

        Parameters
        ----------
        tile_layout : Union[TileLayout, ComposeLayout]
            The tiled layout to check
        tiled_shape : List[PrimExpr]
            The shape of the tiled layout
        inner_shape : List[PrimExpr]
            The shape of the inner layout

        Returns
        -------
        Optional[TileLayout]
            The outer layout if it is the inner layout of the tiled layout, None otherwise
        """
        raise NotImplementedError("is_tile_inner is not implemented for this layout")

    def is_tile_outer(
        self,
        tile_layout: Union["TileLayout", "ComposeLayout"],
        tiled_shape: List[PrimExpr],
        outer_shape: List[PrimExpr],
    ) -> Optional["TLayout"]:
        """Check if a layout is the outer layout of a tiled layout.

        Parameters
        ----------
        tile_layout : Union[TileLayout, ComposeLayout]
            The tiled layout to check
        tiled_shape : List[PrimExpr]
            The shape of the tiled layout
        outer_shape : List[PrimExpr]
            The shape of the outer layout

        Returns
        -------
        Optional[TLayout]
            The inner layout if it is the outer layout of the tiled layout, None otherwise
        """
        raise NotImplementedError("is_tile_outer is not implemented for this layout")


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
            _ffi_api.TileLayout,  # pylint: disable=no-member
            data_iter_array,
            device_iter_array,
            from_scope,
            to_scope,
        )

    def normalize(self) -> "TileLayout":
        return _ffi_api.TileLayoutNormalize(self)  # pylint: disable=no-member

    def tile(
        self,
        outer: "TileLayout",
        outer_shape: List[PrimExpr],
        inner_shape: List[PrimExpr],
    ) -> Union["TileLayout", "ComposeLayout"]:
        return _ffi_api.TileLayoutTile(  # pylint: disable=no-member
            outer, self, outer_shape, inner_shape
        )

    def is_tile_inner(
        self,
        tile_layout: Union["TileLayout", "ComposeLayout"],
        tiled_shape: List[PrimExpr],
        inner_shape: List[PrimExpr],
    ) -> Optional["TileLayout"]:
        return _ffi_api.TileLayoutIsTileInner(  # pylint: disable=no-member
            tile_layout, self, tiled_shape, inner_shape
        )

    def is_tile_outer(
        self,
        tile_layout: Union["TileLayout", "ComposeLayout"],
        tiled_shape: List[PrimExpr],
        outer_shape: List[PrimExpr],
    ) -> Optional["TLayout"]:
        return _ffi_api.TileLayoutIsTileOuter(  # pylint: disable=no-member
            tile_layout, self, tiled_shape, outer_shape
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
    def from_tuple(
        data: Union[Tuple, int, PrimExpr, S],
        strides: Optional[Union[Tuple, int, PrimExpr]] = None,
        device: Optional[Union[Tuple, int, PrimExpr]] = None,
        exclusive: Optional[Tuple] = None,
        from_to: Optional[Tuple[str]] = None,
    ) -> "TileLayout":
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
    def shard(
        shape: Tuple[PrimExpr, int],
        mesh: Tuple,
        strategy: str,
        inner: "TileLayout",
        from_to: Optional[Tuple[str]] = None,
    ) -> "TileLayout":
        assert from_to is not None, "from_to must be provided if device is provided"
        assert isinstance(from_to, tuple) and len(from_to) == 2, "from_to must be a tuple of 2"
        return _ffi_api.TileLayoutShard(  # pylint: disable=no-member
            shape,
            mesh,
            strategy,
            inner,
            ExecScope.create(from_to[0]) if from_to else None,
            ExecScope.create(from_to[1]) if from_to else None,
        )

    @staticmethod
    def find_optimal_vec_len(layout_A: "TileLayout", layout_B: "TileLayout") -> int:
        return _ffi_api.Vec_Len(layout_A, layout_B)  # pylint: disable=no-member


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

    def normalize(self) -> "SwizzleLayout":
        return _ffi_api.SwizzleLayoutNormalize(self)  # pylint: disable=no-member

    def tile(
        self,
        outer: "TileLayout",
        outer_shape: List[PrimExpr],
        inner_shape: List[PrimExpr],
    ) -> Union["TileLayout", "ComposeLayout"]:
        return _ffi_api.SwizzleLayoutTile(  # pylint: disable=no-member
            outer, self, outer_shape, inner_shape
        )

    def is_tile_inner(
        self,
        tile_layout: Union["TileLayout", "ComposeLayout"],
        tiled_shape: List[PrimExpr],
        inner_shape: List[PrimExpr],
    ) -> Optional["TileLayout"]:
        return _ffi_api.SwizzleLayoutIsTileInner(  # pylint: disable=no-member
            tile_layout, self, tiled_shape, inner_shape
        )


@register_object("tirx.ComposeLayout")
class ComposeLayout(TLayout):
    """A memory layout that composes 2 layouts."""

    def __init__(self, layout_A: TLayout, layout_B: TLayout):
        self.__init_handle_by_constructor__(
            _ffi_api.ComposeLayout,  # pylint: disable=no-member
            layout_A,
            layout_B,
        )

    def normalize(self) -> "ComposeLayout":
        return _ffi_api.ComposeLayoutNormalize(self)  # pylint: disable=no-member

    def tile(
        self,
        outer: "TileLayout",
        outer_shape: List[PrimExpr],
        inner_shape: List[PrimExpr],
    ) -> Union["TileLayout", "ComposeLayout"]:
        return _ffi_api.ComposeLayoutTile(  # pylint: disable=no-member
            outer, self, outer_shape, inner_shape
        )

    def is_tile_inner(
        self,
        tile_layout: Union["TileLayout", "ComposeLayout"],
        tiled_shape: List[PrimExpr],
        inner_shape: List[PrimExpr],
    ) -> Optional["TileLayout"]:
        return _ffi_api.ComposeLayoutIsTileInner(  # pylint: disable=no-member
            tile_layout, self, tiled_shape, inner_shape
        )


def parse_dimension_types(dimension_types: Union[Tuple[int], str]) -> Tuple[int]:
    if isinstance(dimension_types, str):
        for c in dimension_types:
            assert c in ["P", "F"], "dimension_types must be a string of 'P' and 'F'"
        dimension_types = [
            TrainiumLayout.Partition if c == "P" else TrainiumLayout.Free for c in dimension_types
        ]
    return ShapeTuple(dimension_types)


@register_object("tirx.TrainiumLayout")
class TrainiumLayout(TLayout):

    Partition = 0
    Free = 1

    dimension_types: Tuple[int]
    combined_1d_layout: TileLayout

    def __init__(
        self,
        dimension_types: Union[Tuple[int], str],
        combined_1d_layout: TileLayout,
    ):
        dimension_types = parse_dimension_types(dimension_types)
        self.__init_handle_by_constructor__(
            _ffi_api.TrainiumLayout,  # pylint: disable=no-member
            dimension_types,
            combined_1d_layout,
        )

    def normalize(self) -> "TrainiumLayout":
        return _ffi_api.TrainiumLayoutNormalize(self)  # pylint: disable=no-member

    @property
    def partition_size(self):
        return _ffi_api.TrainiumLayoutGetPartitionSize(self)  # pylint: disable=no-member

    @classmethod
    def from_annotation(cls, annotation: str, shape: Tuple[PrimExpr]) -> "TrainiumLayout":
        from tvm.arith import Analyzer

        analyzer = Analyzer()
        assert re.fullmatch(
            r"[PF]*", annotation
        ), f"annotation {annotation} must be a string of 'P' and 'F'"
        assert len(annotation) == len(
            shape
        ), f"annotation {annotation} and shape {shape} must have the same length"
        num_p_dim = annotation.count("P")
        if num_p_dim == 1:
            p_idx = annotation.index("P")
            p_dim = shape[p_idx]
            assert analyzer.can_prove(
                p_dim <= 128 or p_dim % 128 == 0
            ), f"There is only 1 P in the annotation. Partition size {p_dim} must be less than or equal to 128 or a multiple of 128"
            if analyzer.can_prove(p_dim > 128):
                # split out the P dimension and put the higher part on the free dimension with largest stride
                annotation = "F" + annotation
                shape = (p_dim // 128,) + shape[:p_idx] + (128,) + shape[p_idx + 1 :]
        elif num_p_dim > 1:
            p_dim_prod = functools.reduce(
                operator.mul, [s for s, c in zip(shape, annotation) if c == "P"]
            )
            assert analyzer.can_prove(
                p_dim_prod <= 128
            ), f"There are {num_p_dim} Ps in the annotation. Partition size {p_dim_prod} must be less than or equal to 128"

        f_shape = [shape[i] for i, c in enumerate(annotation) if c == "F"]
        p_shape = [shape[i] for i, c in enumerate(annotation) if c == "P"]
        f_tile_layout = TileLayout.from_tuple(f_shape)
        p_tile_layout = TileLayout.from_tuple(p_shape)
        result = []
        f_index = p_index = 0

        for char in annotation:
            if char == "F":
                result.append(f_tile_layout.data_iter_array[f_index])
                f_index += 1
            else:  # char == 'P'
                result.append(p_tile_layout.data_iter_array[p_index])
                p_index += 1
        if num_p_dim == 1 and analyzer.can_prove(p_dim > 128):
            # put higher part of P to where it belongs
            higher_P = result[0]
            result = result[1:]
            result = result[:p_idx] + [higher_P] + result[p_idx:]
        combined_1d_layout = TileLayout(result)
        return cls(annotation, combined_1d_layout)


@register_object("tirx.TrainiumPSUMLayout")
class TrainiumPSUMLayout(TrainiumLayout):
    def __init__(
        self,
        dimension_types: Union[Tuple[int], str],
        combined_1d_layout: TileLayout,
    ):
        dimension_types = parse_dimension_types(dimension_types)
        self.__init_handle_by_constructor__(
            _ffi_api.TrainiumPSUMLayout,  # pylint: disable=no-member
            dimension_types,
            combined_1d_layout,
        )
