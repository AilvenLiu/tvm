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
from tvm.tirx import BufferRegion, Buffer, PrimExpr
from tvm.ir import Op

from . import _ffi_api


def _get_tirp_op(op_name: str):
    assert isinstance(op_name, str)
    return Op.get("tirp." + op_name)


def copy(src: BufferRegion, dst: BufferRegion):
    """Copy data from src to dst.

    Parameters
    ----------
    src : BufferRegion
        The source buffer region.

    dst : BufferRegion
        The destination buffer region.
    """
    return _ffi_api.OpCall(_get_tirp_op("copy"), [src, dst])


def fill(dst: BufferRegion, value: PrimExpr):
    """Fill the buffer region with the value.

    Parameters
    ----------
    dst : BufferRegion
        The destination buffer region.

    value : PrimExpr
        The value to be filled.
    """
    return _ffi_api.OpCall(_get_tirp_op("fill"), [dst, value])


def gemm(A: Buffer, B: Buffer, C: Buffer, D: Buffer, alpha: PrimExpr = 1.0, beta: PrimExpr = 0.0):
    """General matrix multiplication.

    D = A * B * alpha + C * beta

    Parameters
    ----------
    A : Buffer
        The buffer of matrix A.

    B : Buffer
        The buffer of matrix B.

    C : Buffer
        The buffer of matrix C.

    D : Buffer
        The buffer of matrix D.

    alpha : PrimExpr
        The scalar alpha.

    beta : PrimExpr
        The scalar beta.
    """
    return _ffi_api.OpCall(_get_tirp_op("gemm"), [A, B, C, D, alpha, beta])


__all__ = [
    "copy",
    "fill",
    "gemm",
]
