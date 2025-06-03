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
import functools

from typing import List, Optional, Tuple, Union, TypeAlias, ClassVar, TYPE_CHECKING, Dict
import re
import operator

import tvm
import tvm.ffi
from tvm.runtime import Object, ShapeTuple
from tvm.tirx.expr import PrimExpr
from . import _ffi_api
from .exec_scope import ExecScope


@tvm.ffi.register_object("tir.TLayout")
class TLayout(Object):
    def __init__(self):
        self.__init_handle_by_constructor__(_ffi_api.TLayout)  # pylint: disable=no-member

    def verify_well_formed(self) -> bool:
        """Verify if the layout is well-formed.

        Returns
        -------
        bool
            True if the layout is well-formed, False otherwise
        """
        return _ffi_api.TLayoutVerifyWellFormed(self)  # pylint: disable=no-member

    def size(self, axis_name: Optional[str] = None):
        """Get the size of the layout.

        Parameters
        ----------
        axis_name : Optional[str]
            The name of the axis to get the size of. If not provided, the default input size will be returned.
        """
        return _ffi_api.TLayoutGetSize(self, axis_name)  # pylint: disable=no-member

    def cosize(self, axis_name: Optional[str] = None):
        """Get the cosize of the layout.

        Parameters
        ----------
        axis_name : Optional[str]
            The name of the axis to get the cosize of. If not provided, the default cosize will be returned.
        """
        return _ffi_api.TLayoutGetCosize(self, axis_name)  # pylint: disable=no-member

    def apply(self, *coord: List[PrimExpr], shape: List[PrimExpr] = None) -> Dict[str, PrimExpr]:
        """Apply the layout on the input coordinate and get the mapped output.

        Input cases:
        - coord is a single element -> will be treated as a 1D coordinate
        - coord is a list of elements -> will be treated as a multi-dimensional coordinate
        - shape is provided -> turn the coord with shape into a 1D coordinate
        - shape is not provided -> use the default shape

        Returns
        -------
        Dict[str, PrimExpr]
            The mapped output (axis name -> value on the axis)
        """
        if len(coord) == 1:
            # assert shape is None, "shape must be None if coord is not a list or tuple"
            return _ffi_api.TLayoutApplyLinear(self, coord[0])  # pylint: disable=no-member
        if shape is None:
            return _ffi_api.TLayoutApply(self, coord)  # pylint: disable=no-member
        return _ffi_api.TLayoutApplyWithShape(self, coord, shape)  # pylint: disable=no-member

    def normalize(self) -> "TLayout":
        """Normalize the layout by simplifying and fusing iterators where possible.

        Returns
        -------
        TLayout
            The normalized layout
        """
        return _ffi_api.TLayoutNormalize(self)  # pylint: disable=no-member

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
        return _ffi_api.TLayoutTile(  # pylint: disable=no-member
            self, outer, outer_shape, inner_shape
        )

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
        return _ffi_api.TLayoutIsTileInner(  # pylint: disable=no-member
            self, tile_layout, tiled_shape, inner_shape
        )

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
        return _ffi_api.TLayoutIsTileOuter(  # pylint: disable=no-member
            self, tile_layout, tiled_shape, outer_shape
        )

    def tile_to(self, to_shape: List[PrimExpr], current_shape: List[PrimExpr]) -> "TLayout":
        """Tile the current layout to the given shape.

        Parameters
        ----------
        to_shape : List[PrimExpr]
            The shape to tile to
        current_shape : List[PrimExpr]
            The current shape of the layout
        """

        tile_shape = [to_shape[i] // current_shape[i] for i in range(len(to_shape))]
        return self.tile(
            TileLayout(shard=(tile_shape, TLayout._get_default_strides(tile_shape))),
            tile_shape,
            current_shape,
        )

    @staticmethod
    def _get_default_strides(data: List[Union[int, PrimExpr]], stride: int = 1) -> Tuple:
        assert isinstance(data, (list, tuple)), "data must be a tuple"
        res = list()
        for t in reversed(data):
            assert isinstance(t, (int, PrimExpr)), "data must be int or PrimExpr"
            res.append(stride)
            stride *= t
        return list(reversed(res))

    def is_swizzle(self) -> bool:
        """Check if the layout is swizzle."""
        return isinstance(self, SwizzleLayout)

    def is_trivial(self) -> bool:
        """Check if the layout is trivial."""
        return False

    def is_trainium(self) -> bool:
        """Check if the layout is trainium layout."""
        if not isinstance(self, TileLayout):
            return False
        return _ffi_api.TileLayoutIsTrainium(self)  # pylint: disable=no-member


@tvm.ffi.register_object("tir.Axis")
class Axis(Object):
    """Layout axis wrapper."""

    # ------------------------------------------------------------------
    # Hints for static analysers / editors
    # ------------------------------------------------------------------
    if TYPE_CHECKING:
        # Thread axes
        pid: ClassVar["Axis"]
        bx: ClassVar["Axis"]
        by: ClassVar["Axis"]
        bz: ClassVar["Axis"]
        cbx: ClassVar["Axis"]
        cby: ClassVar["Axis"]
        cbz: ClassVar["Axis"]
        tx: ClassVar["Axis"]
        warpid: ClassVar["Axis"]
        laneid: ClassVar["Axis"]
        wgid: ClassVar["Axis"]
        tid_in_wg: ClassVar["Axis"]
        wid_in_wg: ClassVar["Axis"]
        # Memory axes
        m: ClassVar["Axis"]
        P: ClassVar["Axis"]
        F: ClassVar["Axis"]
        Bank: ClassVar["Axis"]
        TCol: ClassVar["Axis"]
        TLane: ClassVar["Axis"]

    # ---- forbid direct construction ----
    def __init__(self, *args, **kwargs):  # noqa: D401
        raise RuntimeError("Cannot create Axis directly; use Axis.get()")

    # ---- implementation helpers ----
    _NAMES = [
        # Thread axes
        "pid",
        "bx",
        "by",
        "bz",
        "cbx",
        "cby",
        "cbz",
        "tx",
        "warpid",
        "laneid",
        "wgid",
        "tid_in_wg",
        "wid_in_wg",
        # Memory axes
        "m",
        "P",
        "F",
        "Bank",
        "TCol",
        "TLane",
    ]

    @staticmethod
    def _register_axis(name: str) -> "Axis":
        return _ffi_api.AxisGet(name)  # pylint: disable=no-member

    reg_dict: dict[str, "Axis"]  # filled below

    @staticmethod
    def get(name: str) -> "Axis":
        """Get the axis object by name."""
        return Axis.reg_dict[name]

    def is_thread(self) -> bool:
        """Check if the axis is a thread axis."""
        return _ffi_api.AxisIsThreadAxis(self)  # pylint: disable=no-member

    def is_memory(self) -> bool:
        """Check if the axis is a memory axis."""
        return _ffi_api.AxisIsMemoryAxis(self)  # pylint: disable=no-member

    def get_scope(self) -> Optional[ExecScope]:
        """Get the scope of the axis."""
        return _ffi_api.AxisGetScope(self)  # pylint: disable=no-member

    def get_subscope(self) -> Optional[ExecScope]:
        """Get the subscope of the axis."""
        return _ffi_api.AxisGetSubscope(self)  # pylint: disable=no-member


# ------------------------------------------------------------------
# 2)  Runtime: actually attach the attributes
# ------------------------------------------------------------------
Axis.reg_dict = {}
for _n in Axis._NAMES:  # pylint: disable=protected-access
    _axis_obj = Axis._register_axis(_n)  # pylint: disable=protected-access
    setattr(Axis, _n, _axis_obj)
    Axis.reg_dict[_n] = _axis_obj


@tvm.ffi.register_object("tir.Iter")
class Iter(Object):
    """A memory layout that tiles data across devices."""

    extent: PrimExpr
    stride: PrimExpr
    axis: Axis

    def __init__(self, extent: PrimExpr, stride: PrimExpr, axis: Union[Axis, str]):
        if isinstance(axis, str):
            axis = Axis.get(axis)
        self.__init_handle_by_constructor__(
            _ffi_api.Iter, extent, stride, axis  # pylint: disable=no-member
        )


@tvm.ffi.register_object("tir.TileLayout")
class TileLayout(TLayout):
    """A memory layout that tiles data across devices."""

    shard: List[Iter]
    replicate: List[Iter]
    exclude: List[Tuple[Iter, PrimExpr]]

    IterList: TypeAlias = Tuple[
        List[PrimExpr], List[Union[Tuple[PrimExpr, Union[str, Axis]], PrimExpr]]
    ]

    def __init__(
        self,
        shard: IterList = None,
        replicate: IterList = None,
        exclude: Tuple[IterList, List[PrimExpr]] = None,
    ):
        # Handle None values
        if shard is None:
            shard = ([], [])
        if replicate is None:
            replicate = ([], [])
        if exclude is None:
            exclude = (([], []), [])

        if isinstance(shard, (tuple, list)) and not isinstance(shard[0], (tuple, list)):
            # shard can be just a tuple of extents, infer the default strides
            shard = (shard, TLayout._get_default_strides(shard, 1))

        # Convert to Iter objects
        assert len(shard[0]) == len(shard[1]), "shard's extent and stride must have the same length"
        assert len(replicate[0]) == len(
            replicate[1]
        ), "replicate's extent and stride must have the same length"
        assert len(exclude[0][0]) == len(
            exclude[0][1]
        ), "exclude's extent and stride must have the same length"
        assert len(exclude[1]) == len(
            exclude[0][0]
        ), "exclude's iter and selector must have the same length"

        def process_iter(e, s):
            return Iter(e, s[0], s[1]) if isinstance(s, tuple) else Iter(e, s, "m")

        shard = [process_iter(e, s) for e, s in zip(shard[0], shard[1])]
        replicate = [process_iter(e, s) for e, s in zip(replicate[0], replicate[1])]
        exclude = ([process_iter(e, s) for e, s in zip(exclude[0][0], exclude[0][1])], exclude[1])
        exclude = list(zip(exclude[0], exclude[1]))

        self.__init_handle_by_constructor__(
            _ffi_api.TileLayout,  # pylint: disable=no-member
            shard,
            replicate,
            exclude,
        )

    def is_trivial(self) -> bool:
        """Check if the layout is trivial."""
        return _ffi_api.TileLayoutIsTrivial(self)  # pylint: disable=no-member

    def group_by_shape(self, shape: List[PrimExpr]) -> Tuple["TLayout", List[int]]:
        """Group the current layout by the given shape.

        Parameters
        ----------
        shape : List[PrimExpr]
            The shape to group by

        Returns
        -------
        Tuple[TLayout, List[int]]
            The grouped layout and the separators
        """
        return _ffi_api.TileLayoutGroupByShape(self, shape)  # pylint: disable=no-member

    def get_scope(self) -> Optional[Tuple[ExecScope, ExecScope]]:
        """Get the scope pair of the layout."""
        return _ffi_api.TileLayoutGetScope(self)  # pylint: disable=no-member

    @classmethod
    def trainium(
        cls, annotation: str, shape: Tuple[PrimExpr], is_psum: bool = False
    ) -> "TileLayout":
        """Create a TileLayout from an annotation string and a shape."""
        analyzer = tvm.arith.Analyzer()
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
                shape = [p_dim // 128] + shape[:p_idx] + [128] + shape[p_idx + 1 :]
        elif num_p_dim > 1:
            p_dim_prod = functools.reduce(
                operator.mul, [s for s, c in zip(shape, annotation) if c == "P"]
            )
            assert analyzer.can_prove(
                p_dim_prod <= 128
            ), f"There are {num_p_dim} Ps in the annotation. Partition size {p_dim_prod} must be less than or equal to 128"

        f_shape = [s for i, (s, c) in enumerate(zip(shape, annotation)) if c == "F"]
        p_shape = [s for i, (s, c) in enumerate(zip(shape, annotation)) if c == "P"]
        f_tile_layout = TileLayout(
            shard=(f_shape, [(s, "F") for s in TLayout._get_default_strides(f_shape, 1)])
        )
        p_tile_layout = TileLayout(
            shard=(p_shape, [(s, "P") for s in TLayout._get_default_strides(p_shape, 1)])
        )
        result = []
        f_index = p_index = 0

        for char in annotation:
            if char == "F":
                result.append(f_tile_layout.shard[f_index])
                f_index += 1
            else:  # char == 'P'
                result.append(p_tile_layout.shard[p_index])
                p_index += 1
        if num_p_dim == 1 and analyzer.can_prove(p_dim > 128):
            # put higher part of P to where it belongs
            higher_P = result[0]
            result = result[1:]
            result = result[:p_idx] + [higher_P] + result[p_idx:]

        res = _ffi_api.TileLayout(result, [], [])  # pylint: disable=no-member
        if is_psum:
            res = res.to_psum()
        return res

    kPSUMMaxElemPerBank = 512
    kPSUMBankNum = 8

    def to_psum(self) -> "TileLayout":
        """Convert the layout to a psum layout."""
        analyzer = tvm.arith.Analyzer()
        shard = []
        for i in self.shard:
            if i.axis.name == "F":
                if analyzer.can_prove(i.stride % self.kPSUMMaxElemPerBank == 0):
                    stride = analyzer.simplify(i.stride // self.kPSUMMaxElemPerBank)
                    shard.append(Iter(i.extent, stride, Axis.get("Bank")))
                elif analyzer.can_prove(self.kPSUMMaxElemPerBank % i.stride == 0):
                    c = analyzer.simplify(self.kPSUMMaxElemPerBank // i.stride)
                    if analyzer.can_prove(i.extent < c):
                        shard.append(i)
                    elif analyzer.can_prove(i.extent % c == 0):
                        shard.append(Iter(analyzer.simplify(i.extent // c), 1, Axis.get("Bank")))
                        shard.append(Iter(c, i.stride, Axis.get("F")))
                    else:
                        assert False, f"layout {self} can not be converted to psum layout"
                else:
                    assert False, f"layout {self} can not be converted to psum layout"
            else:
                shard.append(i)
        return _ffi_api.TileLayout(shard, [], [])  # pylint: disable=no-member


@tvm.ffi.register_object("tir.SwizzleLayout")
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


@tvm.ffi.register_object("tir.ComposeLayout")
class ComposeLayout(TLayout):
    """A memory layout that composes 2 layouts."""

    def __init__(self, layout_A: "SwizzleLayout", layout_B: "TileLayout"):
        self.__init_handle_by_constructor__(
            _ffi_api.ComposeLayout,  # pylint: disable=no-member
            layout_A,
            layout_B,
        )
