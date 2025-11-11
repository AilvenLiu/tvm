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
import operator
import re
from typing import (
    TYPE_CHECKING,
    ClassVar,
    Dict,
    List,
    Optional,
    Tuple,
    TypeAlias,
    Union,
)

import tvm
import tvm_ffi
from tvm.runtime import Object, ShapeTuple
from tvm.tirx.expr import PrimExpr

from . import _ffi_api
from .exec_scope import ExecScope


@tvm_ffi.register_object("tir.TLayout")
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
            assert isinstance(t, (int, PrimExpr)), f"data must be int or PrimExpr, but got {t}"
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

    def storage(self) -> "TLayout":
        if isinstance(self, TileLayout):
            # Filter out shard with thread axis
            shard = [iter for iter in self.shard if not iter.axis.is_thread()]
            replicate = [iter for iter in self.replica if not iter.axis.is_thread()]
            exclude = {axis: offset for axis, offset in self.offset.items() if not axis.is_thread()}
            return TileLayout.from_iters(shard, replicate, exclude)  # pylint: disable=no-member

        elif isinstance(self, SwizzleLayout):
            return self
        elif isinstance(self, ComposeLayout):
            return ComposeLayout(self.layout_A.storage(), self.layout_B.storage())
        else:
            raise ValueError(f"Unsupported layout type: {type(self)}")

    def unpack(self, num: int) -> "TLayout":
        """Unpack the layout, where a single element in the layout is unpacked into num contiguous elements.

        Parameters
        ----------
        num : int
            The number of elements to unpack into

        Returns
        -------
        TLayout
            The unpacked layout
        """
        if isinstance(self, TileLayout):
            shard = [Iter(iter.extent, iter.stride * num, iter.axis) for iter in self.shard]
            shard.append(Iter(num, 1, Axis.get("m")))
            return TileLayout.from_iters(shard, self.replica, self.offset)
        elif isinstance(self, SwizzleLayout):
            assert num & (num - 1) == 0, "num must be a power of 2"
            return SwizzleLayout(
                self.per_element + (num.bit_length() - 1),
                self.swizzle_len,
                self.atom_len,
                self.swizzle_inner,
            )
        elif isinstance(self, ComposeLayout):
            return ComposeLayout(self.layout_A.unpack(num), self.layout_B.unpack(num))
        else:
            raise ValueError(f"Unsupported layout type: {type(self)}")

    def pack(self, num: int) -> "TLayout":
        """Pack the layout, where num contiguous elements in the layout are packed into a single element.

        Parameters
        ----------
        num : int
            The number of elements to pack into

        Returns
        -------
        TLayout
            The packed layout
        """
        if isinstance(self, TileLayout):
            inner_iter = self.shard[-1]
            assert (
                inner_iter.stride == 1
                and inner_iter.extent % num == 0
                and inner_iter.axis.is_memory()
            ), f"Layout {self} can not be packed into {num} elements"
            shard = [Iter(iter.extent, iter.stride // num, iter.axis) for iter in self.shard[:-1]]
            shard.append(Iter(inner_iter.extent // num, 1, inner_iter.axis))
            return TileLayout.from_iters(shard, self.replica, self.offset)
        elif isinstance(self, SwizzleLayout):
            assert num & (num - 1) == 0, "num must be a power of 2"
            assert (
                self.per_element >= num.bit_length() - 1
            ), "per_element must be greater than or equal to num.bit_length() - 1"
            return SwizzleLayout(
                self.per_element - (num.bit_length() - 1),
                self.swizzle_len,
                self.atom_len,
                self.swizzle_inner,
            )
        elif isinstance(self, ComposeLayout):
            return ComposeLayout(self.layout_A.pack(num), self.layout_B.pack(num))
        else:
            raise ValueError(f"Unsupported layout type: {type(self)}")


@tvm_ffi.register_object("tir.Axis")
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

    # Enable syntax like `4 @ Axis.laneid` to attach an axis to a stride/term.
    # This mirrors libraries that overload the matrix multiply operator for DSLs.
    def __rmatmul__(self, other: PrimExpr):  # type: ignore[override]
        # Represent a single value bound to an axis.
        return _OnAxis(other, self)


# ------------------------------------------------------------------
# 2)  Runtime: actually attach the attributes
# ------------------------------------------------------------------
Axis.reg_dict = {}
for _n in Axis._NAMES:  # pylint: disable=protected-access
    _axis_obj = Axis._register_axis(_n)  # pylint: disable=protected-access
    setattr(Axis, _n, _axis_obj)
    Axis.reg_dict[_n] = _axis_obj
    # Export axis names at module level for `from tvm.tir.layout import laneid` style.
    globals()[_n] = _axis_obj

# Populate __all__ for axis symbols in addition to classes
try:
    __all__  # type: ignore[name-defined]
except NameError:  # pragma: no cover
    __all__ = []  # type: ignore[var-annotated]
__all__ += Axis._NAMES  # pylint: disable=protected-access


# ------------------------------------------------------------------
# Helper types to support `PrimExpr @ Axis` and `sum` for offsets
# ------------------------------------------------------------------
class _OnAxis:
    """Represents a single value attached to an axis, created via `value @ Axis.X`.

    Used in two places:
    - As stride spec in `TileLayout(..., shard=(extents, [value @ Axis.X]))`
    - As terms to build an offset expression like `1 @ Axis.laneid + 512`
    """

    def __init__(self, value: PrimExpr, axis: Axis):
        self.value = value
        self.axis = axis

    # Arithmetic to build offset sums
    def __add__(self, other: "_OffsetExprLike") -> "_OffsetExpr":
        base = _OffsetExpr({self.axis: self.value})
        return base + other

    def __radd__(self, other: "_OffsetExprLike") -> "_OffsetExpr":
        return self.__add__(other)


class _OffsetExpr:
    """Sum of axis-bound terms forming an offset specification.

    Internally stored as a dict {Axis: PrimExpr}. When a plain PrimExpr is
    provided (without axis), it is treated as `Axis.m` by convention.
    """

    def __init__(self, terms: Optional[Dict[Axis, PrimExpr]] = None):
        self.terms: Dict[Axis, PrimExpr] = dict(terms or {})

    def _add_term(self, axis: Axis, value: PrimExpr):
        if axis in self.terms:
            # Merge if both exist; rely on tvm arith for symbolic add
            self.terms[axis] = self.terms[axis] + value  # type: ignore[operator]
        else:
            self.terms[axis] = value

    def __add__(self, other: "_OffsetExprLike") -> "_OffsetExpr":
        res = _OffsetExpr(dict(self.terms))
        if isinstance(other, _OffsetExpr):
            for ax, v in other.terms.items():
                res._add_term(ax, v)
        elif isinstance(other, _OnAxis):
            res._add_term(other.axis, other.value)
        else:  # PrimExpr-like -> default to Axis.m
            res._add_term(Axis.get("m"), other)  # type: ignore[arg-type]
        return res

    def __radd__(self, other: "_OffsetExprLike") -> "_OffsetExpr":
        return self.__add__(other)


_OffsetExprLike = Union[_OffsetExpr, _OnAxis, PrimExpr, int]


def _to_offset_expr(x: _OffsetExprLike) -> _OffsetExpr:
    if isinstance(x, _OffsetExpr):
        return x
    if isinstance(x, _OnAxis):
        return _OffsetExpr({x.axis: x.value})
    # Fallback: treat plain PrimExpr/int as Axis.m
    return _OffsetExpr({Axis.get("m"): x})  # type: ignore[arg-type]


@tvm_ffi.register_object("tir.Iter")
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


@tvm_ffi.register_object("tir.TileLayout")
class TileLayout(TLayout):
    """A memory layout that tiles data across devices."""

    shard: List[Iter]
    replicate: List[Iter]
    exclude: List[Tuple[Axis, PrimExpr]]

    IterList: TypeAlias = Union[
        Tuple[List[PrimExpr], List[Union[Tuple[PrimExpr, Union[str, Axis]], PrimExpr]]], List[Iter]
    ]

    def __init__(
        self,
        shard: IterList = None,
        replica: IterList = None,
        offset: "_OffsetExprLike" = None,
    ):
        # New API: accept only replica and offset
        offset_dict = dict()
        if offset is not None:
            off_expr = _to_offset_expr(offset)
            for ax, val in off_expr.terms.items():
                offset_dict[ax] = val
        if (
            shard is not None
            and all(isinstance(iter, Iter) for iter in shard)
            and replica is not None
            and all(isinstance(iter, Iter) for iter in replica)
        ):
            self.__init_handle_by_constructor__(
                _ffi_api.TileLayout,  # pylint: disable=no-member
                shard,
                replica,
                offset_dict,
            )
            return
        # Handle None values
        if shard is None or shard == ():
            shard = ([], [])
        if replica is None:
            replica = ([], [])

        if isinstance(shard, (tuple, list)) and not isinstance(shard[0], (tuple, list)):
            # shard can be just a tuple of extents, infer the default strides
            shard = (shard, TLayout._get_default_strides(shard, 1))
        # Convert to Iter objects
        assert len(shard[0]) == len(shard[1]), "shard's extent and stride must have the same length"
        assert len(replica[0]) == len(
            replica[1]
        ), "replicate's extent and stride must have the same length"

        def process_iter(e, s):
            if isinstance(s, _OnAxis):
                return Iter(e, s.value, s.axis)
            return Iter(e, s[0], s[1]) if isinstance(s, tuple) else Iter(e, s, "m")

        shard = [process_iter(e, s) for e, s in zip(shard[0], shard[1])]
        replica = [process_iter(e, s) for e, s in zip(replica[0], replica[1])]

        self.__init_handle_by_constructor__(
            _ffi_api.TileLayout,  # pylint: disable=no-member
            shard,
            replica,
            offset_dict,
        )

    @staticmethod
    def from_iters(
        shard: IterList = None, replica: IterList = None, offset: Dict[Axis, PrimExpr] = None
    ) -> "TileLayout":
        if offset is None:
            offset = dict()
        return _ffi_api.TileLayout(shard, replica, offset)  # pylint: disable=no-member

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

        res = TileLayout.from_iters(result, [], dict())  # pylint: disable=no-member
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
        return TileLayout.from_iters(shard, [], dict())  # pylint: disable=no-member


@tvm_ffi.register_object("tir.SwizzleLayout")
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


@tvm_ffi.register_object("tir.ComposeLayout")
class ComposeLayout(TLayout):
    """A memory layout that composes 2 layouts."""

    def __init__(self, layout_A: "SwizzleLayout", layout_B: "TileLayout"):
        self.__init_handle_by_constructor__(
            _ffi_api.ComposeLayout,  # pylint: disable=no-member
            layout_A,
            layout_B,
        )
