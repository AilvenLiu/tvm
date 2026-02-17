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
import math
from typing import Optional

from tvm.script import tirx as Tx
from tvm.tir import PrimFunc, Buffer, BufferRegion
from tvm.tir.stmt import OpCall
from tvm.tirx.op_schedule import (
    ScheduleContext,
    register_dispatch,
    predicate,
)
from .common import get_indices, get_st_extent


def validate_deepgemm_permute_dims(
    op_call: OpCall,
    sctx: ScheduleContext,
) -> bool:
    if isinstance(op_call.args[0], Buffer):
        buffer: Buffer = op_call.args[0]
        extent = buffer.shape
    elif isinstance(op_call.args[0], BufferRegion):
        buffer: Buffer = op_call.args[0].buffer
        st, extent = get_st_extent(op_call.args[0])

    order = op_call.args[1]
    if sctx.exec_scope.name == "warp":
        assert "threadIdx.y" not in sctx.launch_params and "threadIdx.z" not in sctx.launch_params
        ndim = len(order)
        if not list(order) == list(range(ndim - 2)) + [ndim - 1, ndim - 2]:
            return False
        if not math.prod(extent[:-2]) == 1:
            return False
        strides = list(buffer.strides)
        if not (strides == [] or (strides[-1] == 1 and strides[-2] == extent[-1])):
            return False
        return True
    return False


def vectorized_permute_dims_last_2d_impl(
    op_call: OpCall,
    sctx: ScheduleContext,
) -> Optional[PrimFunc]:

    if isinstance(op_call.args[0], Buffer):
        buffer: Buffer = op_call.args[0]
        extent = shape = buffer.shape
        st = [0] * len(extent)
    elif isinstance(op_call.args[0], BufferRegion):
        buffer: Buffer = op_call.args[0].buffer
        shape = buffer.shape
        st, extent = get_st_extent(op_call.args[0])

    M, N = extent[-2:]
    vec_len = op_call.config.get("vec_len")

    if vec_len is None:
        for vec_len in range(4, 0, -1):
            if M % vec_len == 0:
                break

    if not shape[-1] % vec_len == 0:
        vec_len = 1
    if not (st[-2] * shape[-1] + st[-1]) % vec_len == 0:
        vec_len = 1

    # Thread and vectorization setup
    if sctx.exec_scope.name == "warp":
        tid_x = sctx.launch_params["threadIdx.x"]
        assert "threadIdx.y" not in sctx.launch_params and "threadIdx.z" not in sctx.launch_params
        # fmt: off
        @Tx.prim_func(tirx=True)
        def impl():
            warp_size = Tx.meta_var(32)
            lane_id = Tx.meta_var(tid_x % warp_size)
            reg_trans = Tx.alloc_buffer((N // warp_size, M // vec_len, vec_len), buffer.dtype, scope="local")
            for wi in Tx.unroll(0, N // warp_size):
                for vi in Tx.unroll(0, M // vec_len):
                    for vec in Tx.unroll(vec_len):
                        old_index = Tx.meta_var(get_indices((vi * vec_len + vec) * N + wi * warp_size + lane_id, st, extent))
                        reg_trans[wi, vi, vec] = buffer[*old_index]
            Tx.cuda.warp_sync()
            for wi in Tx.unroll(0, N // warp_size):
                for vi in Tx.unroll(0, M // vec_len):
                    for vec in Tx.vectorized(vec_len):
                        new_index = Tx.meta_var(get_indices((wi * warp_size + lane_id) * M + vi * vec_len + vec, st, extent))
                        buffer[*new_index] = reg_trans[wi, vi, vec]
            Tx.cuda.warp_sync()
        # fmt: on
    else:
        raise NotImplementedError
    return impl


@register_dispatch(
    "permute_dims",
    "cuda",
    variant="vectorized_permute_dims_last_2d",
    priority=20,
    when=[
        predicate(
            "validate_deepgemm_permute_dims",
            lambda op, sctx: (
                validate_deepgemm_permute_dims(op, sctx),
                "validate_deepgemm_permute_dims failed",
            ),
        )
    ],
)
def permute_dims_dispatch(op: OpCall, sctx: ScheduleContext) -> Optional[PrimFunc]:
    return vectorized_permute_dims_last_2d_impl(op, sctx)
