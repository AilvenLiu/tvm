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

"""Implementation of cast operator on CUDA."""

import functools
import operator
from typing import Optional

from tvm.arith import Analyzer
from tvm.script import tir as T
from tvm.tir import Buffer, BufferRegion, PrimFunc
from tvm.tir.stmt import OpCall
from tvm.tirp.op_schedule import (
    ScheduleContext,
    register_dispatch,
    predicate,
)

from .common import get_indices, get_st_extent, get_vec_len, target_cuda


def validate_cast_op(
    dst_buffer_region: BufferRegion,
    src_buffer_region: BufferRegion,
    sctx: ScheduleContext,  # pylint: disable=unused-argument
) -> bool:
    """Sanity check for cast op"""
    src: Buffer = src_buffer_region.buffer
    dst: Buffer = dst_buffer_region.buffer
    if not (src.layout and dst.layout):
        return False
    # Extract regions and validate dimensions
    analyzer = Analyzer()
    src_region, dst_region = src_buffer_region.region, dst_buffer_region.region
    # Extract extents and validate non-unit dimensions match
    src_extent_ = [r.extent for r in src_region if r.extent != 1]
    dst_extent_ = [r.extent for r in dst_region if r.extent != 1]
    if len(src_extent_) != len(dst_extent_) or not all(
        analyzer.can_prove_equal(s, d) for s, d in zip(src_extent_, dst_extent_)
    ):
        return False
    return True


cuda_intrinsic_dict = {
    ("float32", "float16"): "__float22half2_rn",
    ("float16", "float32"): "__half22float2",
}

dtypex2_dict = {
    "float32": "float2",
    "float16": "half2",
}


def cast_thread_wise_impl(
    dst_buffer_region: BufferRegion,
    src_buffer_region: BufferRegion,
    sctx: ScheduleContext,
) -> Optional[PrimFunc]:

    if sctx.exec_scope.name != "thread":
        return None

    dst: Buffer = dst_buffer_region.buffer
    src: Buffer = src_buffer_region.buffer

    # Extract regions and validate dimensions
    src_st, src_extent = get_st_extent(src_buffer_region)
    dst_st, dst_extent = get_st_extent(dst_buffer_region)

    # Thread and vectorization setup
    n_elements = functools.reduce(operator.mul, src_extent, 1)
    vec_len = get_vec_len(dst_buffer_region, src_buffer_region, [2, 1])

    intrinsic_name = cuda_intrinsic_dict.get((src.dtype, dst.dtype), None)
    if intrinsic_name is None:
        return None

    src_dtypex2 = dtypex2_dict[src.dtype]
    dst_dtypex2 = dtypex2_dict[dst.dtype]

    func_name = f"tvm_builtin_cast_{src.dtype}x2_{dst.dtype}x2"
    source_code = f"""
__forceinline__ __device__ void {func_name}(void* dst, void* src) {{
    (({dst_dtypex2}*)dst)[0] = {intrinsic_name}((({src_dtypex2}*)src)[0]);
}}
"""

    # fmt: off
    @T.prim_func(tirp=True, check_well_formed=False)
    def impl():
        for s in T.serial(0, n_elements // (vec_len)):
            fused = T.meta_var(s * vec_len)
            dst_indices = T.meta_var(get_indices(fused, dst_st, dst_extent))
            src_indices = T.meta_var(get_indices(fused, src_st, src_extent))
            if vec_len == 2:
                T.cuda.func_call(
                    func_name,
                    dst.ptr_to(dst_indices),
                    src.ptr_to(src_indices),
                    source_code=source_code,
                )
            else:
                dst[*dst_indices] = T.cast(src[*src_indices], dst.dtype)
    # fmt: on
    return impl


# Rich dispatcher variant for cast
@register_dispatch(
    "cast",
    "cuda",
    variant="thread_wise",
    priority=10,
    when=[
        predicate(
            "validate_cast_op",
            lambda op, sctx: (
                validate_cast_op(op.args[0], op.args[1], sctx),
                "validate_cast_op failed",
            ),
        ),
        predicate(
            "exec_scope",
            lambda op, sctx: (
                sctx.exec_scope.name == "thread",
                f"unsupported exec_scope {sctx.exec_scope.name}",
            ),
        ),
    ],
)
def cast_dispatch_thread_wise(op_call: OpCall, sctx: ScheduleContext) -> Optional[PrimFunc]:
    return cast_thread_wise_impl(op_call.args[0], op_call.args[1], sctx)
