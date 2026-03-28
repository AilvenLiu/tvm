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

"""copy_async dispatch variant: tma."""

from tvm.tirx import PrimFunc
from tvm.tirx.operator.scope_op_dispatch import (
    DispatchContext,
    predicate,
    register_dispatch,
)
from tvm.tirx.stmt import ScopeOpCall

from ..common import validate_copy_op
from ..exec_scope_utils import single_thread
from .utils import copy_tma_impl


# === Variant: copy_async/tma (priority=10) ===
#
# When: valid async copy at single-thread exec scope. Applies to global↔shared
# copies on Hopper+ (SM90+) where TMA hardware is available.
# Falls back (DispatchFail) for scope pairs TMA can't handle.
#
# Before (ScopeOpCall):
#     Tx.copy_async(A_smem[0:128, 0:64], A[row:row+128, 0:64])
#     # A: global float16, A_smem: shared float16 with 128B swizzle layout
#
# After (generates host-side tensormap + device-side TMA instruction):
#   Host init:
#     tensor_map = Tx.cuda.tma_create_tensormap(
#         tensor_rank=2, gmem_ptr=A.data,
#         global_shape=[M, 64], global_strides=[64, 1],
#         box_dim=[128, 64], element_strides=[1, 1],
#         swizzle_mode="SWIZZLE_128B", ...)
#   Device:
#     Tx.ptx.cp_async.bulk.tensor.g2c(
#         rank=2, smem_ptr=A_smem.ptr_to(0),
#         mbar=barrier, tensor_map=tensor_map,
#         coord0=0, coord1=row)
#
# Key internal algorithm (copy_tma_impl):
#   1. Infer swizzle mode from smem layout (none/32B/64B/128B)
#   2. Group layout axes, find contiguous regions
#   3. Compute box_dim — the tile size per TMA instruction
#   4. Sort global strides to determine TMA addressing order
#   5. Build iteration space if tensor > one TMA box
#   6. Create tensormap descriptor on host
#   7. Emit cp_async.bulk.tensor.g2c (load) or s2g (store)
@register_dispatch(
    "copy_async",
    "cuda",
    variant="tma",
    priority=10,
    when=[
        predicate(
            "validate_copy_op",
            lambda op, sctx: (validate_copy_op(op, sctx), "not a valid copy op"),
        ),
        predicate(
            "single_thread",
            lambda op, sctx: (
                single_thread(op, sctx),
                f"unsupported exec_scope {sctx.exec_scope}, expected single thread",
            ),
        ),
    ],
)
def copy_async_dispatch_tma(op: ScopeOpCall, sctx: DispatchContext) -> PrimFunc:
    return copy_tma_impl(op, sctx)
