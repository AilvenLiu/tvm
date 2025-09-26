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
from typing import Optional

import tvm
from tvm.script import tir as T
from tvm.tir import Buffer, BufferRegion, PrimFunc
from tvm.tir.stmt import OpCall
from tvm.tirp.op_schedule import (
    ScheduleContext,
    register_dispatch,
    predicate,
    fail,
)

from .common import (
    CopyInstType,
    copy_vec_load_impl,
    get_st_extent,
    target_cuda,
    validate_copy_op,
)


def copy_default_impl(
    op_call: OpCall,
    sctx: ScheduleContext,
) -> Optional[PrimFunc]:
    """Schedule copy operation
    The implementation serves as a fallback for copy operations that uses a single thread
    to move data element by element.
    """
    dst_buffer_region, src_buffer_region = op_call.args

    src: Buffer = src_buffer_region.buffer
    dst: Buffer = dst_buffer_region.buffer

    # Extract regions and validate dimensions
    src_st, src_extent = get_st_extent(src_buffer_region)
    dst_st, dst_extent = get_st_extent(dst_buffer_region)

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

    if sctx.exec_scope.name == "cta":
        tx = sctx.launch_params["threadIdx.x"].dom.extent
        assert "threadIdx.y" not in sctx.launch_params and "threadIdx.z" not in sctx.launch_params

        # fmt: off
        @T.prim_func(tirp=True, check_well_formed=False)
        def impl():
            for tid_x in T.thread_binding(tx, "threadIdx.x"):
                with T.thread()[tid_x == 0]:
                    copy(dst, src)

            if dst.scope().startswith("shared"):
                T.tvm_storage_sync("shared")
        # fmt: on
    elif sctx.exec_scope.name == "thread":
        # fmt: off
        @T.prim_func(tirp=True, check_well_formed=False)
        def impl():
            copy(dst, src)
        # fmt: on
    else:
        return None

    return impl


# ---------------------------------------------------------------------------
# Rich dispatcher variants with predicates and failure reasons
# ---------------------------------------------------------------------------


def _scope_allowed(op_call: OpCall, sctx: ScheduleContext):
    dst_buffer_region, src_buffer_region = op_call.args[:2]
    src: Buffer = src_buffer_region.buffer
    dst: Buffer = dst_buffer_region.buffer
    cond = (
        (src.scope() == "global" and dst.scope().startswith("shared"))
        or (src.scope().startswith("shared") and dst.scope() == "global")
        or (src.scope() == "global" and dst.scope() == "local")
        or (src.scope() == "local" and dst.scope() == "global")
        or (src.scope().startswith("shared") and dst.scope() == "local")
        or (dst.scope().startswith("shared") and src.scope() == "local")
    )
    if not cond:
        return False, f"unsupported memory scopes src={src.scope()} dst={dst.scope()}"
    return True, None


def _exec_scope_ok(op_call: OpCall, sctx: ScheduleContext):
    ok = sctx.exec_scope.name in ("cta", "thread")
    return (ok, None if ok else f"unsupported exec_scope {sctx.exec_scope.name}")


def _is_valid_copy(op_call: OpCall, sctx: ScheduleContext):
    return (validate_copy_op(op_call, sctx), "validate_copy_op failed")


def _vec_len_possible(op_call: OpCall, sctx: ScheduleContext):
    # mirror get_vec_len inputs
    dst_buffer_region, src_buffer_region = op_call.args[:2]
    if sctx.exec_scope.name == "cta":
        tx = sctx.launch_params["threadIdx.x"].dom.extent
    elif sctx.exec_scope.name == "thread":
        tx = 1
    else:
        return False, f"unsupported exec_scope {sctx.exec_scope.name} for vec_len"
    vec_len = get_vec_len(
        dst_buffer_region,
        src_buffer_region,
        [
            128 // tvm.runtime.DataType(src_buffer_region.buffer.dtype).bits,
            64 // tvm.runtime.DataType(src_buffer_region.buffer.dtype).bits,
            32 // tvm.runtime.DataType(src_buffer_region.buffer.dtype).bits,
            1,
        ],
        thread_cnt=tx,
    )
    if vec_len is None:
        return False, "no valid vector length; check alignment/extents/thread-count"
    return True, None


@register_dispatch(
    "copy",
    "cuda",
    variant="vec_load",
    priority=10,
    when=[
        predicate("validate_copy_op", _is_valid_copy),
        predicate("scope", _scope_allowed),
        predicate("exec_scope", _exec_scope_ok),
        predicate("vec_len", _vec_len_possible),
    ],
)
def copy_schedule_vec_load(op_call: OpCall, sctx: ScheduleContext) -> Optional[PrimFunc]:
    # Delegate to the fast vectorized path
    return copy_vec_load_impl(op_call, sctx, CopyInstType.NORMAL)


@register_dispatch(
    "copy",
    "cuda",
    variant="default",
    priority=0,
    when=[predicate("validate_copy_op", _is_valid_copy), predicate("exec_scope", _exec_scope_ok)],
)
def copy_schedule_default(op_call: OpCall, sctx: ScheduleContext) -> Optional[PrimFunc]:
    # Conservative scalar fallback
    return copy_default_impl(op_call, sctx)
