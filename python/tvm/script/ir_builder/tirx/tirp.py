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
from typing import Union, Optional, Dict, Any, Tuple, Callable, List
from tvm.tirx import BufferRegion, Buffer, PrimExpr
from tvm.ir import Op
from tvm.tirx.event import BaseEvent
from tvm.tirx.exec_scope import ExecScope
from tvm.tirx.expr import FloatImm
from tvm.tirx import event
import tvm.tirp.operator as tirp_op
from tvm.tirx.predicate import Predicate
from . import _ffi_api, frame


def _to_region(buffer: Union[BufferRegion, Buffer]):
    if isinstance(buffer, Buffer):
        return buffer[[slice(None, None, None) for _ in range(len(buffer.shape))]]
    assert isinstance(buffer, BufferRegion)
    return buffer


def _wrap_elem_in_tuple(e):
    if isinstance(e, (tuple, list)):
        return e
    return (e,)


f_insert = _ffi_api.OpCall  # pylint: disable=no-member


def zero(
    dst: Union[BufferRegion, Buffer],
    src: Union[BufferRegion, Buffer],
    workspace: Dict[str, Buffer] = None,
    schedule_config: Dict[str, Any] = None,
):
    """Zero out all elements in src and store to dst.

    Parameters
    ----------
    dst : Union[BufferRegion, Buffer]
        The destination buffer region for zero result.

    src : Union[BufferRegion, Buffer]
        The source buffer region.

    workspace : Optional[Dict[str, Buffer]]
        The workspace of the operator.
    """
    if workspace is None:
        workspace = {}
    if schedule_config is None:
        schedule_config = {}
    dst = _to_region(dst)
    src = _to_region(src)
    return f_insert(tirp_op.Zero(dst, src, workspace=workspace, schedule_config=schedule_config))


def sqrt(
    dst: Union[BufferRegion, Buffer],
    src: Union[BufferRegion, Buffer],
    bias: Optional[Union[BufferRegion, Buffer, FloatImm]] = None,
    scale: Optional[FloatImm] = None,
    workspace: Dict[str, Buffer] = None,
    schedule_config: Dict[str, Any] = None,
):
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

    workspace : Optional[Dict[str, Buffer]]
        The workspace of the operator.
    """
    if workspace is None:
        workspace = {}
    if schedule_config is None:
        schedule_config = {}
    dst = _to_region(dst)
    src = _to_region(src)
    if workspace is None:
        workspace = {}
    if bias is not None and isinstance(bias, Buffer):
        bias = _to_region(bias)
    return f_insert(
        tirp_op.Sqrt(dst, src, bias, scale, workspace=workspace, schedule_config=schedule_config)
    )


def add(
    dst: Union[BufferRegion, Buffer],
    src1: Union[BufferRegion, Buffer, FloatImm],
    src2: Union[BufferRegion, Buffer, FloatImm],
    workspace: Dict[str, Buffer] = None,
    schedule_config: Dict[str, Any] = None,
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

    workspace : Optional[Dict[str, Buffer]]
        The workspace of the operator.
    """
    if workspace is None:
        workspace = {}
    if schedule_config is None:
        schedule_config = {}
    dst = _to_region(dst)
    if isinstance(src1, Buffer):
        src1 = _to_region(src1)
    if isinstance(src2, Buffer):
        src2 = _to_region(src2)
    return f_insert(
        tirp_op.Add(dst, src1, src2, workspace=workspace, schedule_config=schedule_config)
    )


def sub(
    dst: Union[BufferRegion, Buffer],
    src1: Union[BufferRegion, Buffer],
    src2: Union[BufferRegion, Buffer, FloatImm],
    workspace: Dict[str, Buffer] = None,
    schedule_config: Dict[str, Any] = None,
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

    workspace : Dict[str, Buffer]
        The workspace of the operator.
    """
    if workspace is None:
        workspace = {}
    if schedule_config is None:
        schedule_config = {}
    dst = _to_region(dst)
    if isinstance(src1, Buffer):
        src1 = _to_region(src1)
    if isinstance(src2, Buffer):
        src2 = _to_region(src2)
    return f_insert(
        tirp_op.Sub(dst, src1, src2, workspace=workspace, schedule_config=schedule_config)
    )


def mul(
    dst: Union[BufferRegion, Buffer],
    src1: Union[BufferRegion, Buffer, FloatImm],
    src2: Union[BufferRegion, Buffer, FloatImm],
    workspace: Dict[str, Buffer] = None,
    schedule_config: Dict[str, Any] = None,
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

    workspace : Dict[str, Buffer]
        The workspace of the operator.
    """
    if workspace is None:
        workspace = {}
    if schedule_config is None:
        schedule_config = {}
    dst = _to_region(dst)
    if isinstance(src1, Buffer):
        src1 = _to_region(src1)
    if isinstance(src2, Buffer):
        src2 = _to_region(src2)
    return f_insert(
        tirp_op.Mul(dst, src1, src2, workspace=workspace, schedule_config=schedule_config)
    )


def fdiv(
    dst: Union[BufferRegion, Buffer],
    src1: Union[BufferRegion, Buffer],
    src2: Union[BufferRegion, Buffer, FloatImm],
    workspace: Dict[str, Buffer] = None,
    schedule_config: Dict[str, Any] = None,
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

    workspace : Optional[Dict[str, Buffer]]
        The workspace of the operator.
    """
    if workspace is None:
        workspace = {}
    if schedule_config is None:
        schedule_config = {}
    dst = _to_region(dst)
    src1 = _to_region(src1)
    if isinstance(src2, Buffer):
        src2 = _to_region(src2)
    return f_insert(
        tirp_op.FDiv(dst, src1, src2, workspace=workspace, schedule_config=schedule_config)
    )


def copy(
    dst: Union[BufferRegion, Buffer],
    src: Union[BufferRegion, Buffer],
    workspace: Dict[str, Buffer] = None,
    schedule_config: Dict[str, Any] = None,
):
    """Copy data from src to dst.

    Parameters
    ----------
    dst : Union[BufferRegion, Buffer]
        The destination buffer region.

    src : Union[BufferRegion, Buffer]
        The source buffer region.

    workspace : Optional[Dict[str, Buffer]]
        The workspace of the operator.

    schedule_config : Optional[Dict[str, Any]]
        The schedule config of the operator.
    """
    if workspace is None:
        workspace = {}
    if schedule_config is None:
        schedule_config = {}
    dst = _to_region(dst)
    src = _to_region(src)
    return f_insert(tirp_op.Copy(dst, src, workspace=workspace, schedule_config=schedule_config))


def copy_async(
    dst: Union[BufferRegion, Buffer],
    src: Union[BufferRegion, Buffer],
    evt: BaseEvent,
    workspace: Dict[str, Buffer] = None,
    schedule_config: Dict[str, Any] = None,
):
    return f_insert(
        tirp_op.CopyAsync(dst, src, evt, workspace=workspace, schedule_config=schedule_config)
    )


def fill(
    dst: Union[BufferRegion, Buffer],
    value: PrimExpr,
    workspace: Dict[str, Buffer] = None,
    schedule_config: Dict[str, Any] = None,
):
    """Fill the buffer region with the value.

    Parameters
    ----------
    dst : Union[BufferRegion, Buffer]
        The destination buffer region.

    value : PrimExpr
        The value to be filled.

    workspace : Optional[Dict[str, Buffer]]
        The workspace of the operator.
    """
    if workspace is None:
        workspace = {}
    if schedule_config is None:
        schedule_config = {}
    dst = _to_region(dst)
    return f_insert(tirp_op.Fill(dst, value, workspace=workspace, schedule_config=schedule_config))


def gemm(
    D: Union[BufferRegion, Buffer],
    A: Union[BufferRegion, Buffer],
    B: Union[BufferRegion, Buffer],
    C: Union[BufferRegion, Buffer],
    transpose_A: bool = False,
    transpose_B: bool = False,
    alpha: PrimExpr = 1.0,
    beta: PrimExpr = 0.0,
    workspace: Dict[str, Buffer] = None,
    schedule_config: Dict[str, Any] = None,
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

    transpose_A : bool
        Whether to transpose A.

    transpose_B : bool
        Whether to transpose B.

    alpha : PrimExpr
        The scalar alpha.

    beta : PrimExpr
        The scalar beta.

    workspace : Optional[Dict[str, Buffer]]
        The workspace of the operator.
    """
    if workspace is None:
        workspace = {}
    if schedule_config is None:
        schedule_config = {}
    D = _to_region(D)
    A = _to_region(A)
    B = _to_region(B)
    C = _to_region(C)
    return f_insert(
        tirp_op.Gemm(
            D,
            A,
            B,
            C,
            transpose_A,
            transpose_B,
            alpha,
            beta,
            workspace=workspace,
            schedule_config=schedule_config,
        )
    )


def sum(
    dst: Union[BufferRegion, Buffer],
    src: Union[BufferRegion, Buffer],
    axes: Union[int, Tuple[int]] = -1,
    accum: bool = False,
    workspace: Dict[str, Buffer] = None,
    schedule_config: Dict[str, Any] = None,
):
    """
    Sum all elements in src and store to dst.

    Parameters
    ----------
    dst : Union[BufferRegion, Buffer]
        The destination buffer region for sum result.

    src : Union[BufferRegion, Buffer]
        The source buffer region.

    axes : Union[int, Tuple[int]]
        The axis to sum over.

    accum : bool
        Whether dst is accumulated.

    workspace : Optional[Dict[str, Buffer]]
        The workspace of the operator.
    """
    if workspace is None:
        workspace = {}
    if schedule_config is None:
        schedule_config = {}
    dst = _to_region(dst)
    src = _to_region(src)
    axes = _wrap_elem_in_tuple(axes)
    return f_insert(
        tirp_op.Sum(dst, src, axes, accum, workspace=workspace, schedule_config=schedule_config)
    )


def max(
    dst: Union[BufferRegion, Buffer],
    src: Union[BufferRegion, Buffer],
    axes: Union[int, Tuple[int]] = -1,
    accum: bool = False,
    workspace: Dict[str, Buffer] = None,
    schedule_config: Dict[str, Any] = None,
):
    """
    Max all elements in src and store to dst.

    Parameters
    ----------
    dst : Union[BufferRegion, Buffer]
        The destination buffer region for sum result.

    src : Union[BufferRegion, Buffer]
        The source buffer region.

    axes : Union[int, Tuple[int]]
        The axis to sum over.

    accum : bool
        Whether dst is accumulated.

    workspace : Optional[Dict[str, Buffer]]
        The workspace of the operator.
    """
    if workspace is None:
        workspace = {}
    if schedule_config is None:
        schedule_config = {}
    dst = _to_region(dst)
    src = _to_region(src)
    axes = _wrap_elem_in_tuple(axes)
    return f_insert(
        tirp_op.Max(dst, src, axes, accum, workspace=workspace, schedule_config=schedule_config)
    )


def min(
    dst: Union[BufferRegion, Buffer],
    src: Union[BufferRegion, Buffer],
    axes: Union[int, Tuple[int]] = -1,
    accum: bool = False,
    workspace: Dict[str, Buffer] = None,
    schedule_config: Dict[str, Any] = None,
):
    """
    Min all elements in src and store to dst.

    Parameters
    ----------
    dst : Union[BufferRegion, Buffer]
        The destination buffer region for min result.

    src : Union[BufferRegion, Buffer]
        The source buffer region.

    axes : Union[int, Tuple[int]]
        The axis to sum over.

    accum : bool
        Whether dst is accumulated.

    workspace : Optional[Dict[str, Buffer]]
        The workspace of the operator.
    """
    if workspace is None:
        workspace = {}
    if schedule_config is None:
        schedule_config = {}
    dst = _to_region(dst)
    src = _to_region(src)
    axes = _wrap_elem_in_tuple(axes)
    return f_insert(
        tirp_op.Min(dst, src, axes, accum, workspace=workspace, schedule_config=schedule_config)
    )


def reciprocal(
    dst: Union[BufferRegion, Buffer],
    src: Union[BufferRegion, Buffer],
    workspace: Dict[str, Buffer] = None,
    schedule_config: Dict[str, Any] = None,
):
    """Reciprocal all elements in src and store to dst.

    Parameters
    ----------
    dst : Union[BufferRegion, Buffer]
        The destination buffer region for reciprocal result.

    src : Union[BufferRegion, Buffer]
        The source buffer region.

    workspace : Optional[Dict[str, Buffer]]
        The workspace of the operator.
    """
    if workspace is None:
        workspace = {}
    if schedule_config is None:
        schedule_config = {}
    dst = _to_region(dst)
    src = _to_region(src)
    return f_insert(
        tirp_op.Reciprocal(dst, src, workspace=workspace, schedule_config=schedule_config)
    )


def memset(
    dst: Union[BufferRegion, Buffer],
    value: PrimExpr,
    workspace: Dict[str, Buffer] = None,
    schedule_config: Dict[str, Any] = None,
):
    """Set all elements in dst to value.

    Parameters
    ----------
    dst : Union[BufferRegion, Buffer]
        The destination buffer region for memset.

    value : PrimExpr
        The value to be set.

    workspace : Optional[Dict[str, Buffer]]
        The workspace of the operator.
    """
    if workspace is None:
        workspace = {}
    if schedule_config is None:
        schedule_config = {}
    dst = _to_region(dst)
    return f_insert(
        tirp_op.Memset(dst, value, workspace=workspace, schedule_config=schedule_config)
    )


def maximum(
    dst: Union[BufferRegion, Buffer],
    src1: Union[BufferRegion, Buffer, FloatImm],
    src2: Union[BufferRegion, Buffer, FloatImm],
    workspace: Dict[str, Buffer] = None,
    schedule_config: Dict[str, Any] = None,
):
    """Maximum all elements in src1 and src2 and store to dst.

    Parameters
    ----------
    dst : Union[BufferRegion, Buffer]
        The destination buffer region for maximum result.

    src1 : Union[BufferRegion, Buffer, FloatImm]
        The source buffer region 1, or float.

    src2 : Union[BufferRegion, Buffer, FloatImm]
        The source buffer region 2, or float.

    workspace : Dict[str, Buffer]
        The workspace of the operator.
    """
    if workspace is None:
        workspace = {}
    if schedule_config is None:
        schedule_config = {}
    dst = _to_region(dst)
    if isinstance(src1, Buffer):
        src1 = _to_region(src1)
    if isinstance(src2, Buffer):
        src2 = _to_region(src2)
    return f_insert(
        tirp_op.Maximum(dst, src1, src2, workspace=workspace, schedule_config=schedule_config)
    )


def minimum(
    dst: Union[BufferRegion, Buffer],
    src1: Union[BufferRegion, Buffer, FloatImm],
    src2: Union[BufferRegion, Buffer, FloatImm],
    workspace: Dict[str, Buffer] = None,
    schedule_config: Dict[str, Any] = None,
):
    """Minimum all elements in src1 and src2 and store to dst.

    Parameters
    ----------
    dst : Union[BufferRegion, Buffer]
        The destination buffer region for minimum result.

    src1 : Union[BufferRegion, Buffer, FloatImm]
        The source buffer region 1, or float.

    src2 : Union[BufferRegion, Buffer, FloatImm]
        The source buffer region 2, or float.

    workspace : Dict[str, Buffer]
        The workspace of the operator.
    """
    if workspace is None:
        workspace = {}
    if schedule_config is None:
        schedule_config = {}
    dst = _to_region(dst)
    if isinstance(src1, Buffer):
        src1 = _to_region(src1)
    if isinstance(src2, Buffer):
        src2 = _to_region(src2)
    return f_insert(
        tirp_op.Minimum(dst, src1, src2, workspace=workspace, schedule_config=schedule_config)
    )


def exp(
    dst: Union[BufferRegion, Buffer],
    src: Union[BufferRegion, Buffer],
    bias: Optional[Union[BufferRegion, Buffer, FloatImm]] = None,
    scale: Optional[FloatImm] = None,
    workspace: Dict[str, Buffer] = None,
    schedule_config: Dict[str, Any] = None,
):
    """Exponentiate all elements in src and store to dst.

    Parameters
    ----------
    dst : Union[BufferRegion, Buffer]
        The destination buffer region for exp result.

    src : Union[BufferRegion, Buffer]
        The source buffer region.

    bias : Optional[Union[BufferRegion, Buffer, FloatImm]]
        The bias of the exp src. Only supported on Trn.

    scale : Optional[FloatImm]
        The scale of the exp src. Only supported on Trn.

    workspace : Dict[str, Buffer]
        The workspace of the operator.
    """
    if workspace is None:
        workspace = {}
    if schedule_config is None:
        schedule_config = {}
    dst = _to_region(dst)
    src = _to_region(src)
    if bias is not None and isinstance(bias, Buffer):
        bias = _to_region(bias)
    return f_insert(
        tirp_op.Exp(dst, src, bias, scale, workspace=workspace, schedule_config=schedule_config)
    )


def compose_op(
    workspace: Dict[str, Buffer] = None, schedule_config: Dict[str, Any] = None
) -> frame.ComposeOpFrame:
    """Compose a TIRp op.

    Parameters
    ----------
    workspace : Optional[Dict[str, Buffer]]
        The workspace of the operator

    Returns
    -------
    res : frame.ComposeOpFrame
        The result ComposeOpFrame.
    """
    if workspace is None:
        workspace = {}
    if schedule_config is None:
        schedule_config = {}
    return _ffi_api.ComposeOp(workspace, schedule_config)  # pylint: disable=no-member


def tvm_kernel_replace_point():
    """A placeholder for the kernel replace point, used in TIRp op scheduling."""
    return f_insert(tirp_op.KernelReplacePoint(workspace={}, schedule_config={}))


def binary_reduce(
    binary_output: Union[BufferRegion, Buffer],
    reduce_output: Union[BufferRegion, Buffer],
    binary_input1: Union[BufferRegion, Buffer, FloatImm],
    binary_input2: Union[BufferRegion, Buffer, FloatImm],
    binary_op: Union[str, Op],
    reduce_op: Union[str, Op],
    reduce_axes: Union[int, Tuple[int]] = -1,
    workspace: Dict[str, Buffer] = {},
    schedule_config: Dict[str, Any] = {},
):
    """Combine a binary operation with a reduction operation.

    Parameters
    ----------
    binary_output : Union[BufferRegion, Buffer]
        The destination buffer region for binary operation result.

    reduce_output : Union[BufferRegion, Buffer]
        The destination buffer region for reduction result.

    binary_input1 : Union[BufferRegion, Buffer, FloatImm]
        The first source input for binary operation.

    binary_input2 : Union[BufferRegion, Buffer, FloatImm]
        The second source input for binary operation.

    binary_op : Union[str, Op]
        The binary operation to perform.

    reduce_op : Union[str, Op]
        The reduction operation to perform.

    reduce_axes : Union[int, Tuple[int]]
        The axes to reduce over.

    workspace : Dict[str, Buffer]
        The workspace of the operator.

    schedule_config : Dict[str, Any]
        The schedule configuration.
    """
    binary_output = _to_region(binary_output)
    reduce_output = _to_region(reduce_output)
    if isinstance(binary_input1, Buffer):
        binary_input1 = _to_region(binary_input1)
    if isinstance(binary_input2, Buffer):
        binary_input2 = _to_region(binary_input2)
    reduce_axes = _wrap_elem_in_tuple(reduce_axes)

    if isinstance(binary_op, str):
        binary_op = tirp_op.get_tirp_op(binary_op)
    if isinstance(reduce_op, str):
        reduce_op = tirp_op.get_tirp_op(reduce_op)

    return f_insert(
        tirp_op.BinaryReduce(
            binary_output,
            reduce_output,
            binary_input1,
            binary_input2,
            binary_op,
            reduce_op,
            reduce_axes,
            workspace=workspace,
            schedule_config=schedule_config,
        )
    )


def unary_reduce(
    unary_output: Union[BufferRegion, Buffer],
    reduce_output: Union[BufferRegion, Buffer],
    unary_input: Union[BufferRegion, Buffer],
    unary_op: Union[str, Op],
    reduce_op: Union[str, Op],
    bias: Optional[Union[BufferRegion, Buffer, FloatImm]] = None,
    scale: Optional[FloatImm] = None,
    reduce_axes: Union[int, Tuple[int]] = -1,
    workspace: Dict[str, Buffer] = {},
    schedule_config: Dict[str, Any] = {},
):
    """Combine a unary operation with a reduction operation.

    Parameters
    ----------
    unary_output : Union[BufferRegion, Buffer]
        The destination buffer region for unary operation result.

    reduce_output : Union[BufferRegion, Buffer]
        The destination buffer region for reduction result.

    unary_input : Union[BufferRegion, Buffer]
        The source input for unary operation.

    unary_op : Union[str, Op]
        The unary operation to perform.

    reduce_op : Union[str, Op]
        The reduction operation to perform.

    bias : Optional[Union[BufferRegion, Buffer, FloatImm]]
        The bias to apply before unary operation.

    scale : Optional[FloatImm]
        The scale to apply before unary operation.

    reduce_axes : Union[int, Tuple[int]]
        The axes to reduce over.

    workspace : Dict[str, Buffer]
        The workspace of the operator.

    schedule_config : Dict[str, Any]
        The schedule configuration.
    """
    unary_output = _to_region(unary_output)
    reduce_output = _to_region(reduce_output)
    unary_input = _to_region(unary_input)

    if bias is not None and isinstance(bias, Buffer):
        bias = _to_region(bias)

    reduce_axes = _wrap_elem_in_tuple(reduce_axes)

    if isinstance(unary_op, str):
        unary_op = tirp_op.get_tirp_op(unary_op)
    if isinstance(reduce_op, str):
        reduce_op = tirp_op.get_tirp_op(reduce_op)

    return f_insert(
        tirp_op.UnaryReduce(
            unary_output,
            reduce_output,
            unary_input,
            unary_op,
            reduce_op,
            bias,
            scale,
            reduce_axes,
            workspace=workspace,
            schedule_config=schedule_config,
        )
    )


def binary_chain(
    output: Union[BufferRegion, Buffer],
    data: Union[BufferRegion, Buffer],
    operand0: Union[BufferRegion, Buffer, FloatImm],
    operand1: Union[BufferRegion, Buffer, FloatImm],
    op0: Union[str, Op],
    op1: Union[str, Op],
    reverse1: bool = False,
    workspace: Dict[str, Buffer] = {},
    schedule_config: Dict[str, Any] = {},
):
    """Chain multiple binary operations together.

    if not reverse1:
        output = (operand0 op0 data) op1 operand1
    else:
        output = operand1 op1 (operand0 op0 data)

    Parameters
    ----------
    output : Union[BufferRegion, Buffer]
        The destination buffer region for the result.

    data : Union[BufferRegion, Buffer]
        The input data to operate on.

    operand0 : Union[BufferRegion, Buffer, FloatImm]
        The first operand to combine with data.

    operand1 : Union[BufferRegion, Buffer, FloatImm]
        The second operand to use in chained operation.

    op0 : Union[str, Op]
        The first binary operation to perform.

    op1 : Union[str, Op]
        The second binary operation to perform.

    reverse1 : bool
        Whether to reverse the order of the second binary operation.

    workspace : Dict[str, Buffer]
        The workspace of the operator.

    schedule_config : Dict[str, Any]
        The schedule configuration.
    """
    output = _to_region(output)
    data = _to_region(data)

    if isinstance(operand0, Buffer):
        operand0 = _to_region(operand0)
    if isinstance(operand1, Buffer):
        operand1 = _to_region(operand1)

    if isinstance(op0, str):
        op0 = tirp_op.get_tirp_op(op0)
    if isinstance(op1, str):
        op1 = tirp_op.get_tirp_op(op1)

    return f_insert(
        tirp_op.BinaryChain(
            output,
            data,
            operand0,
            operand1,
            op0,
            op1,
            reverse1,
            workspace=workspace,
            schedule_config=schedule_config,
        )
    )


def reduce_negate(
    output: Union[BufferRegion, Buffer],
    input: Union[BufferRegion, Buffer],
    reduce_op: Union[str, Op],
    reduce_axes: Union[int, Tuple[int]] = -1,
    accum: bool = False,
    workspace: Dict[str, Buffer] = {},
    schedule_config: Dict[str, Any] = {},
):
    """Negate the result of a reduction operation.

    Parameters
    ----------
    output : Union[BufferRegion, Buffer]
        The destination buffer region for the negated reduction result.

    input : Union[BufferRegion, Buffer]
        The input buffer region to reduce.

    reduce_axes : Union[int, Tuple[int]]
        The axes to reduce over.

    accum : bool
        Whether to accumulate the result into the output.

    reduce_op : Union[str, Op]
        The reduction operation to perform before negation.

    workspace : Dict[str, Buffer]
        The workspace of the operator.

    schedule_config : Dict[str, Any]
        The schedule configuration.
    """
    output = _to_region(output)
    input = _to_region(input)
    reduce_axes = _wrap_elem_in_tuple(reduce_axes)

    if isinstance(reduce_op, str):
        reduce_op = tirp_op.get_tirp_op(reduce_op)

    return f_insert(
        tirp_op.ReduceNegate(
            output,
            input,
            reduce_axes,
            accum,
            reduce_op,
            workspace=workspace,
            schedule_config=schedule_config,
        )
    )


def select(
    dst: Union[BufferRegion, Buffer],
    true_value: Union[BufferRegion, Buffer, FloatImm],
    false_value: Union[BufferRegion, Buffer, FloatImm],
    pred: Union[Predicate, Callable[..., PrimExpr]],
):
    """Select between two values based on a predicate.

    Parameters
    ----------
    dst : Union[BufferRegion, Buffer]
        The destination buffer region for the result.

    true_value : Union[BufferRegion, Buffer, FloatImm]
        The value to select if the predicate is true.

    false_value : Union[BufferRegion, Buffer, FloatImm]
        The value to select if the predicate is false.

    pred : Union[Predicate, Callable[..., PrimExpr]]
        The predicate to evaluate. The callable should take the same number of arguments as the dimensions of the destination buffer.
    """
    dst = _to_region(dst)
    if isinstance(true_value, Buffer):
        true_value = _to_region(true_value)
    if isinstance(false_value, Buffer):
        false_value = _to_region(false_value)
    if not isinstance(pred, Predicate):
        pred = Predicate(pred)
    return f_insert(tirp_op.Select(dst, true_value, false_value, pred))


def alloc_bulk_group_event(impl: event.EventImpl, state: List[Any] = []) -> event.BulkGroupEvent:
    return _ffi_api.AllocBulkGroupEvent(int(impl), state, "")


def alloc_semaphore_event_tensor(
    impl: event.EventImpl, state: List[Any] = [], shape: List[int] = (1,)
) -> event.SemaphoreEventTensor:
    return _ffi_api.AllocSemaphoreEventTensor(int(impl), state, shape, "")


__all__ = [
    "zero",
    "sqrt",
    "add",
    "sub",
    "mul",
    "fdiv",
    "copy",
    "copy_async",
    "fill",
    "gemm",
    "reciprocal",
    "memset",
    "sum",
    "max",
    "min",
    "compose_op",
    "maximum",
    "minimum",
    "exp",
    "tvm_kernel_replace_point",
    "binary_reduce",
    "unary_reduce",
    "binary_chain",
    "reduce_negate",
    "select",
    "alloc_bulk_group_event",
    "alloc_semaphore_event_tensor",
]
