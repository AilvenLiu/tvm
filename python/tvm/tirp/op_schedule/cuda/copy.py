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

"""Implementation of copy operator schedules."""
import functools
import operator
from typing import Optional

import tvm
from tvm import DataType
from tvm.script import tir as T
from tvm.tir import PrimFunc, BufferRegion, Buffer
from tvm.tir.stmt import OpCall
from tvm.arith.analyzer import Analyzer
from tvm.tirp.op_schedule import ScheduleContext, register_schedule

from .common import (
    CopyInstType,
    copy_g2s_s2g_cta_vec_load_impl,
    target_cuda,
    validate_copy_op,
    get_indices,
    get_vec_len,
    get_st_extent,
)


def copy_g2s_s2g_cta_default_impl(
    dst_buffer_region: BufferRegion,
    src_buffer_region: BufferRegion,
    sctx: ScheduleContext,
    inst_type: CopyInstType,
) -> Optional[PrimFunc]:
    """Schedule copy operation between global and shared memory on CUDA across a CTA.
    The implementation serves as a fallback for copy operations that uses a single thread
    to move data element by element.
    """

    # Sanity checks
    if sctx.exec_scope.name != "cta":
        return None

    src: Buffer = src_buffer_region.buffer
    dst: Buffer = dst_buffer_region.buffer
    if not (src.scope() == "global" and dst.scope().startswith("shared")) and not (
        src.scope().startswith("shared") and dst.scope() == "global"
    ):
        return None

    assert inst_type == CopyInstType.NORMAL

    # Extract regions and validate dimensions
    src_st, src_extent = get_st_extent(src_buffer_region)
    dst_st, dst_extent = get_st_extent(dst_buffer_region)

    # Thread and vectorization setup
    tx = sctx.launch_params["threadIdx.x"].dom.extent
    assert "threadIdx.y" not in sctx.launch_params and "threadIdx.z" not in sctx.launch_params

    def copy(dst, src):
        dst_indices = [i for i in range(len(dst.shape)) if dst_extent[i] != 1]
        src_indices = [i for i in range(len(src.shape)) if src_extent[i] != 1]
        assert len(dst_indices) == len(src_indices)
        copy_extents = [dst_extent[i] for i in dst_indices]

        def get_dst_coord(lvs):
            if isinstance(lvs, tvm.tir.Var):
                lvs = [lvs]
            coord = [dst_st[i] for i in range(len(dst.shape))]
            for i, lv in enumerate(lvs):
                coord[dst_indices[i]] += lv
            return coord

        def get_src_coord(lvs):
            if isinstance(lvs, tvm.tir.Var):
                lvs = [lvs]
            coord = [src_st[i] for i in range(len(src.shape))]
            for i, lv in enumerate(lvs):
                coord[src_indices[i]] += lv
            return coord

        with T.grid(*copy_extents) as lvs:
            T.buffer_store(dst, src[*get_src_coord(lvs)], get_dst_coord(lvs))

    # fmt: off
    @T.prim_func(tirp=True, check_well_formed=False)
    def impl():
        for tid_x in T.thread_binding(tx, "threadIdx.x"):
            with T.thread()[tid_x == 0]:
                copy(dst, src)

        if dst.scope().startswith("shared"):
            T.tvm_storage_sync("shared")
    # fmt: on

    return impl


def copy_g2l_l2g_vec_load_impl(
    dst_buffer_region: BufferRegion,
    src_buffer_region: BufferRegion,
    sctx: ScheduleContext,
) -> Optional[PrimFunc]:
    """Schedule copy operation between global and local memory on CUDA of a single thread."""

    if sctx.exec_scope.name != "thread":
        return None

    dst: Buffer = dst_buffer_region.buffer
    src: Buffer = src_buffer_region.buffer
    if not (src.scope() == "global" and dst.scope() == "local") and not (
        src.scope() == "local" and dst.scope() == "global"
    ):
        return None

    # Extract regions and validate dimensions
    src_st, src_extent = get_st_extent(src_buffer_region)
    dst_st, dst_extent = get_st_extent(dst_buffer_region)

    # Thread and vectorization setup
    elem_size = DataType(src.dtype).bits  # in bits
    n_elements = functools.reduce(operator.mul, src_extent, 1)

    # Find valid vector length
    vec_len = get_vec_len(
        dst_buffer_region,
        src_buffer_region,
        [128 // elem_size, 64 // elem_size, 32 // elem_size, 1],
    )
    # fmt: off
    @T.prim_func(tirp=True, check_well_formed=False)
    def impl():
        for s in T.serial(0, n_elements // (vec_len)):
            for vec in T.vectorized(vec_len):
                fused = T.meta_var(s * vec_len + vec)
                dst_indices = T.meta_var(get_indices(fused, dst_st, dst_extent))
                src_indices = T.meta_var(get_indices(fused, src_st, src_extent))
                dst[*dst_indices] = src[*src_indices]
    # fmt: on

    return impl


@register_schedule("copy", "cuda")
@target_cuda
def copy_schedule(op_call: OpCall, sctx: ScheduleContext) -> Optional[PrimFunc]:
    """Schedule copy operation between global and shared memory on CUDA."""
    dst_buffer_region, src_buffer_region = op_call.args

    if not validate_copy_op(dst_buffer_region, src_buffer_region, sctx):
        return None

    for schedule_fn, args in [
        (
            copy_g2s_s2g_cta_vec_load_impl,
            (dst_buffer_region, src_buffer_region, sctx, CopyInstType.NORMAL),
        ),
        (
            copy_g2s_s2g_cta_default_impl,
            (dst_buffer_region, src_buffer_region, sctx, CopyInstType.NORMAL),
        ),
        (copy_g2l_l2g_vec_load_impl, (dst_buffer_region, src_buffer_region, sctx)),
    ]:
        res = schedule_fn(*args)
        if res is not None:
            return res
    return None
