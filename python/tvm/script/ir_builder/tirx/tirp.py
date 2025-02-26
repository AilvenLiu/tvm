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
"""Builtin ops in TIR+"""
from typing import Union, Optional
from tvm.tirx import BufferRegion, Buffer, PrimExpr
from tvm.ir import Op
from tvm.tirx.async_structs import CopyPipeline
from tvm.tirx.exec_scope import ExecScope
from tvm.tirx.expr import FloatImm

from . import _ffi_api, frame


def _get_tirp_op(op_name: str):
    assert isinstance(op_name, str)
    return Op.get("tirp." + op_name)


def _to_region(buffer: Union[BufferRegion, Buffer]):
    if isinstance(buffer, Buffer):
        return buffer.__getitem__([slice(None, None, None) for _ in range(len(buffer.shape))])
    assert isinstance(buffer, BufferRegion)
    return buffer


def wrap_elem_in_tuple(e):
    if isinstance(e, (tuple, list)):
        return e
    return (e,)


def zero(dst: Union[BufferRegion, Buffer], src: Union[BufferRegion, Buffer]):
    """Zero out all elements in src and store to dst.

    Parameters
    ----------
    dst : Union[BufferRegion, Buffer]
        The destination buffer region for zero result.

    src : Union[BufferRegion, Buffer]
        The source buffer region.
    """
    dst = _to_region(dst)
    src = _to_region(src)
    return _ffi_api.OpCall(_get_tirp_op("zero"), [dst, src])  # pylint: disable=no-member


def sqrt(dst: Union[BufferRegion, Buffer], src: Union[BufferRegion, Buffer], bias: Optional[Union[BufferRegion, Buffer, FloatImm]] = None, scale: Optional[FloatImm] = None):
    """Sqrt all elements in src and store to dst.

    dst = sqrt(src * scale + bias)  (if scale or bias are provided)

    Parameters
    ----------
    dst : Union[BufferRegion, Buffer]
        The destination buffer region for sqrt result.

    src : Union[BufferRegion, Buffer]
        The source buffer region.

    bias : Optional[Union[BufferRegion, Buffer, FloatImm]]
        The bias of the sqrt src. Only supported on Trn.

    scale : Optional[FloatImm]
        The scale of the sqrt src. Only supported on Trn.
    """
    dst = _to_region(dst)
    src = _to_region(src)
    if bias is not None and isinstance(bias, Buffer):
        bias = _to_region(bias)
    return _ffi_api.OpCall(_get_tirp_op("sqrt"), [dst, src, bias, scale])  # pylint: disable=no-member


def add(
    dst: Union[BufferRegion, Buffer],
    src1: Union[BufferRegion, Buffer, FloatImm],
    src2: Union[BufferRegion, Buffer, FloatImm],
):
    """Add data from src1 and src2, store to dst.

    Parameters
    ----------
    dst : Union[BufferRegion, Buffer]
        The destination buffer region for add result.

    src1 : Union[BufferRegion, Buffer, FloatImm]
        The source buffer region 1, or float.

    src2 : Union[BufferRegion, Buffer, FloatImm]
        The source buffer region 2, or float.
    """
    dst = _to_region(dst)
    if isinstance(src1, Buffer):
        src1 = _to_region(src1)
    if isinstance(src2, Buffer):
        src2 = _to_region(src2)
    return _ffi_api.OpCall(_get_tirp_op("add"), [dst, src1, src2])  # pylint: disable=no-member


def sub(
    dst: Union[BufferRegion, Buffer],
    src1: Union[BufferRegion, Buffer],
    src2: Union[BufferRegion, Buffer, FloatImm],
):
    """Sub data from src2 to src1, store to dst.

    Parameters
    ----------
    dst : Union[BufferRegion, Buffer]
        The destination buffer region for sub result.

    src1 : Union[BufferRegion, Buffer]
        The source buffer region 1.

    src2 : Union[BufferRegion, Buffer, FloatImm]
        The source buffer region 2, or float.
    """
    dst = _to_region(dst)
    if isinstance(src1, Buffer):
        src1 = _to_region(src1)
    if isinstance(src2, Buffer):
        src2 = _to_region(src2)
    return _ffi_api.OpCall(_get_tirp_op("sub"), [dst, src1, src2])  # pylint: disable=no-member


def mul(
    dst: Union[BufferRegion, Buffer],
    src1: Union[BufferRegion, Buffer, FloatImm],
    src2: Union[BufferRegion, Buffer, FloatImm],
):
    """Multiply data from src1 and src2, store to dst.

    Parameters
    ----------
    dst : Union[BufferRegion, Buffer]
        The destination buffer region for mul result.

    src1 : Union[BufferRegion, Buffer, FloatImm]
        The source buffer region 1, or float.

    src2 : Union[BufferRegion, Buffer, FloatImm]
        The source buffer region 2, or float.
    """
    dst = _to_region(dst)
    if isinstance(src1, Buffer):
        src1 = _to_region(src1)
    if isinstance(src2, Buffer):
        src2 = _to_region(src2)
    return _ffi_api.OpCall(_get_tirp_op("mul"), [dst, src1, src2])  # pylint: disable=no-member


def fdiv(
    dst: Union[BufferRegion, Buffer],
    src1: Union[BufferRegion, Buffer],
    src2: Union[BufferRegion, Buffer, FloatImm],
):
    """(Float) Div data from src2 to src1, store to dst.

    Parameters
    ----------
    dst : Union[BufferRegion, Buffer]
        The destination buffer region for div result.

    src1 : Union[BufferRegion, Buffer]
        The source buffer region 1.

    src2 : Union[BufferRegion, Buffer, FloatImm]
        The source buffer region 2, or float.
    """
    dst = _to_region(dst)
    src1 = _to_region(src1)
    if isinstance(src2, Buffer):
        src2 = _to_region(src2)
    return _ffi_api.OpCall(_get_tirp_op("fdiv"), [dst, src1, src2])  # pylint: disable=no-member


def copy(dst: Union[BufferRegion, Buffer], src: Union[BufferRegion, Buffer]):
    """Copy data from src to dst.

    Parameters
    ----------
    dst : Union[BufferRegion, Buffer]
        The destination buffer region.

    src : Union[BufferRegion, Buffer]
        The source buffer region.
    """
    dst = _to_region(dst)
    src = _to_region(src)
    return _ffi_api.OpCall(_get_tirp_op("copy"), [dst, src])  # pylint: disable=no-member


def fill(dst: Union[BufferRegion, Buffer], value: PrimExpr):
    """Fill the buffer region with the value.

    Parameters
    ----------
    dst : Union[BufferRegion, Buffer]
        The destination buffer region.

    value : PrimExpr
        The value to be filled.
    """
    dst = _to_region(dst)
    return _ffi_api.OpCall(_get_tirp_op("fill"), [dst, value])  # pylint: disable=no-member


def gemm(
    D: Union[BufferRegion, Buffer],
    A: Union[BufferRegion, Buffer],
    B: Union[BufferRegion, Buffer],
    C: Union[BufferRegion, Buffer],
    alpha: PrimExpr = 1.0,
    beta: PrimExpr = 0.0,
):
    """General matrix multiplication.

    D = A * B * alpha + C * beta

    Parameters
    ----------
    D : Union[BufferRegion, Buffer]
        The buffer of matrix D.

    A : Union[BufferRegion, Buffer]
        The buffer of matrix A.

    B : Union[BufferRegion, Buffer]
        The buffer of matrix B.

    C : Union[BufferRegion, Buffer]
        The buffer of matrix C.

    alpha : PrimExpr
        The scalar alpha.

    beta : PrimExpr
        The scalar beta.
    """
    D = _to_region(D)
    A = _to_region(A)
    B = _to_region(B)
    C = _to_region(C)
    return _ffi_api.OpCall(  # pylint: disable=no-member
        _get_tirp_op("gemm"), [D, A, B, C, alpha, beta]
    )


def sum(
    dst: Union[BufferRegion, Buffer],
    src: Union[BufferRegion, Buffer],
    axes: int = -1,
    accum: bool = False,
):
    """
    Sum all elements in src and store to dst.

    Parameters
    ----------
    dst : Union[BufferRegion, Buffer]
        The destination buffer region for sum result.

    src : Union[BufferRegion, Buffer]
        The source buffer region.

    axes : Tuple[int]
        The axis to sum over.

    accum : bool
        Whether dst is accumulated.
    """
    dst = _to_region(dst)
    src = _to_region(src)
    axes = wrap_elem_in_tuple(axes)
    return _ffi_api.OpCall(  # pylint: disable=no-member
        _get_tirp_op("sum"), [dst, src, axes, accum]
    )


def max(
    dst: Union[BufferRegion, Buffer],
    src: Union[BufferRegion, Buffer],
    axes: int = -1,
    accum: bool = False,
):
    """
    Max all elements in src and store to dst.

    Parameters
    ----------
    dst : Union[BufferRegion, Buffer]
        The destination buffer region for sum result.

    src : Union[BufferRegion, Buffer]
        The source buffer region.

    axes : Tuple[int]
        The axis to sum over.

    accum : bool
        Whether dst is accumulated.
    """
    dst = _to_region(dst)
    src = _to_region(src)
    axes = wrap_elem_in_tuple(axes)
    return _ffi_api.OpCall(  # pylint: disable=no-member
        _get_tirp_op("max"), [dst, src, axes, accum]
    )


def min(
    dst: Union[BufferRegion, Buffer],
    src: Union[BufferRegion, Buffer],
    axes: int = -1,
    accum: bool = False,
):
    """
    Min all elements in src and store to dst.

    Parameters
    ----------
    dst : Union[BufferRegion, Buffer]
        The destination buffer region for min result.

    src : Union[BufferRegion, Buffer]
        The source buffer region.

    axes : Tuple[int]
        The axis to sum over.

    accum : bool
        Whether dst is accumulated.
    """
    dst = _to_region(dst)
    src = _to_region(src)
    axes = wrap_elem_in_tuple(axes)
    return _ffi_api.OpCall(  # pylint: disable=no-member
        _get_tirp_op("min"), [dst, src, axes, accum]
    )


def reciprocal(dst: Union[BufferRegion, Buffer], src: Union[BufferRegion, Buffer]):
    """Reciprocal all elements in src and store to dst.

    Parameters
    ----------
    dst : Union[BufferRegion, Buffer]
        The destination buffer region for reciprocal result.

    src : Union[BufferRegion, Buffer]
        The source buffer region.
    """
    dst = _to_region(dst)
    src = _to_region(src)
    return _ffi_api.OpCall(_get_tirp_op("reciprocal"), [dst, src])  # pylint: disable=no-member


def memset(dst: Union[BufferRegion, Buffer], value: PrimExpr):
    """Set all elements in dst to value.

    Parameters
    ----------
    dst : Union[BufferRegion, Buffer]
        The destination buffer region for memset.

    value : PrimExpr
        The value to be set.
    """
    dst = _to_region(dst)
    return _ffi_api.OpCall(_get_tirp_op("memset"), [dst, value])  # pylint: disable=no-member


def alloc_copy_pipeline(
    thread_scope: ExecScope, depth: int, separate_pc: bool, name_hint: str = ""
) -> CopyPipeline:
    """The copy pipeline allocation function.

    Parameters
    ----------
    thread_scope : ExecScope
        The thread scope of the pipeline.

    depth : int
        The depth of the pipeline.

    separate_pc : bool
        The flag whether the pipeline is separate pc.

    name_hint : str
        The name hint of the pipeline.

    Returns
    -------
    res : CopyPipeline
        The allocated copy pipeline.
    """
    if isinstance(thread_scope, str):
        thread_scope = ExecScope(thread_scope)
    else:
        assert isinstance(thread_scope, ExecScope)
    return _ffi_api.AllocCopyPipeline(thread_scope, depth, separate_pc, name_hint)  # type: ignore[attr-defined] # pylint: disable=no-member


def compose_op() -> frame.ComposeOpFrame:
    """Compose a TIRp op.

    Returns
    -------
    res : frame.ComposeOpFrame
        The result ComposeOpFrame.
    """
    return _ffi_api.ComposeOp()  # type: ignore[attr-defined] # pylint: disable=no-member


__all__ = [
    "zero",
    "sqrt",
    "add",
    "sub",
    "mul",
    "fdiv",
    "copy",
    "fill",
    "gemm",
    "reciprocal",
    "memset",
    "sum",
    "max",
    "min",
    "alloc_copy_pipeline",
    "compose_op",
]
