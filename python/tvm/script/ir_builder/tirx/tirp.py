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
from typing import Union
from tvm.tirx import BufferRegion, Buffer, PrimExpr
from tvm.ir import Op
from tvm.tir.async_structs import Barrier, BarrierArray, Pipeline
from tvm.tir.exec_scope import ExecScope

from . import _ffi_api


def _get_tirp_op(op_name: str):
    assert isinstance(op_name, str)
    return Op.get("tirp." + op_name)


def _to_region(buffer: Union[BufferRegion, Buffer]):
    if isinstance(buffer, Buffer):
        return buffer.__getitem__([slice(None, None, None) for _ in range(len(buffer.shape))])
    assert isinstance(buffer, BufferRegion)
    return buffer


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
    return _ffi_api.OpCall(_get_tirp_op("copy"), [dst, src])


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
    return _ffi_api.OpCall(_get_tirp_op("fill"), [dst, value])


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
    return _ffi_api.OpCall(_get_tirp_op("gemm"), [D, A, B, C, alpha, beta])


def alloc_barrier(thread_scope: ExecScope, name_hint: str = "") -> Barrier:
    """The barrier allocation function.

    Parameters
    ----------
    thread_scope : ExecScope
        The execution scope of the barrier.

    name_hint : str
        The name hint of the barrier.

    Returns
    -------
    res : Barrier
        The allocated barrier.
    """
    if isinstance(thread_scope, str):
        thread_scope = ExecScope.create(thread_scope)
    else:
        assert isinstance(thread_scope, ExecScope)
    return _ffi_api.AllocBarrier(thread_scope, name_hint)  # type: ignore[attr-defined] # pylint: disable=no-member


def alloc_barrier_array(thread_scope: ExecScope, size: int, name_hint="") -> BarrierArray:
    """The barrier array allocation function.

    Parameters
    ----------
    thread_scope : ExecScope
        The execution scope of the barrier array.

    size : int
        The size of the barrier array.

    name_hint : str
        The name hint of the barrier array.

    Returns
    -------
    res : BarrierArray
        The allocated barrier array.
    """
    if isinstance(thread_scope, str):
        thread_scope = ExecScope(thread_scope)
    else:
        assert isinstance(thread_scope, ExecScope)
    return _ffi_api.AllocBarrierArray(thread_scope, size, name_hint)  # type: ignore[attr-defined] # pylint: disable=no-member


def alloc_pipeline(
    thread_scope: ExecScope, depth: int, specialize: bool, name_hint: str = ""
) -> Pipeline:
    """The pipeline allocation function.

    Parameters
    ----------
    thread_scope : ExecScope
        The thread scope of the pipeline.

    depth : int
        The depth of the pipeline.

    specialize : bool
        The flag whether the pipeline is specialized.

    name_hint : str
        The name hint of the pipeline.

    Returns
    -------
    res : Pipeline
        The allocated pipeline.
    """
    if isinstance(thread_scope, str):
        thread_scope = ExecScope(thread_scope)
    else:
        assert isinstance(thread_scope, ExecScope)
    return _ffi_api.AllocPipeline(thread_scope, depth, specialize, name_hint)  # type: ignore[attr-defined] # pylint: disable=no-member


__all__ = ["copy", "fill", "gemm", "alloc_barrier", "alloc_barrier_array", "alloc_pipeline"]
