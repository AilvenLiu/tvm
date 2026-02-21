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
from typing import Optional, Tuple, Iterable

from tvm.arith import Analyzer
import tvm
from tvm.script import tirx as Tx
from tvm.tir import Buffer, BufferRegion, PrimFunc
from tvm.tir.layout import S, TileLayout, TLane, TCol, tid_in_wg
from tvm.tir.stmt import OpCall
from tvm.runtime import DataType
from tvm.tirx.op_schedule import (
    ScheduleContext,
    register_dispatch,
    predicate,
    fail,
)

from .common import (
    CopyInstType,
    copy_vec_load_impl,
    get_st_extent,
    get_vec_len,
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

        with Tx.grid(*copy_extents) as lvs:
            Tx.buffer_store(dst, src[*get_src_coord(lvs)], get_dst_coord(lvs))

    if sctx.exec_scope.name == "cta":
        tx = sctx.launch_params["threadIdx.x"].dom.extent
        assert "threadIdx.y" not in sctx.launch_params and "threadIdx.z" not in sctx.launch_params

        # fmt: off
        @Tx.prim_func(tirx=True, check_well_formed=False)
        def impl():
            for tid_x in Tx.thread_binding(tx, "threadIdx.x"):
                with Tx.thread()[tid_x == 0]:
                    copy(dst, src)

            if dst.scope().startswith("shared"):
                Tx.tvm_storage_sync("shared")
        # fmt: on
    elif sctx.exec_scope.name == "thread":
        # fmt: off
        @Tx.prim_func(tirx=True, check_well_formed=False)
        def impl():
            copy(dst, src)
        # fmt: on
    else:
        fail(f"unsupported exec_scope {sctx.exec_scope.name}")

    return impl


# ---------------------------------------------------------------------------
# Rich dispatcher variants with predicates and failure reasons
# ---------------------------------------------------------------------------


DEFAULT_ALLOWED_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("global", "shared*"),
    ("shared*", "global"),
    ("global", "local"),
    ("local", "global"),
    ("shared*", "local"),
    ("local", "shared*"),
)


def _match_scope(scope: str, pattern: str) -> bool:
    """Glob-lite: 'shared*' => prefix match; otherwise exact."""
    if pattern.endswith("*"):
        return scope.startswith(pattern[:-1])
    return scope == pattern


def _scope_allowed(
    op_call: OpCall,
    sctx: ScheduleContext,
    allowed_pairs: Iterable[Tuple[str, str]] = DEFAULT_ALLOWED_PAIRS,
):
    dst_buffer_region, src_buffer_region = op_call.args[:2]
    src_scope = src_buffer_region.buffer.scope()
    dst_scope = dst_buffer_region.buffer.scope()

    ok = any(
        _match_scope(src_scope, src_pat) and _match_scope(dst_scope, dst_pat)
        for src_pat, dst_pat in allowed_pairs
    )
    if not ok:
        allowed_str = ", ".join(f"{a}->{b}" for a, b in allowed_pairs)
        return False, (
            f"unsupported memory scopes src={src_scope} dst={dst_scope}; " f"allowed: {allowed_str}"
        )
    return True, None


def _exec_scope_ok(op_call: OpCall, sctx: ScheduleContext, expected_scopes: list[str]):
    ok = sctx.exec_scope.name in expected_scopes
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
    vec_len = op_call.config.get("vec_len", None)
    if vec_len is None:
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
        predicate("storage_scope", _scope_allowed),
        predicate("exec_scope", _exec_scope_ok, expected_scopes=["cta", "thread"]),
        predicate("vec_len", _vec_len_possible),
    ],
)
def copy_schedule_vec_load(op_call: OpCall, sctx: ScheduleContext) -> PrimFunc:
    # Delegate to the fast vectorized path
    return copy_vec_load_impl(op_call, sctx, CopyInstType.NORMAL)


@register_dispatch(
    "copy",
    "cuda",
    variant="default",
    priority=0,
    when=[
        predicate("validate_copy_op", _is_valid_copy),
        predicate("exec_scope", _exec_scope_ok, expected_scopes=["cta", "thread"]),
    ],
)
def copy_schedule_default(op_call: OpCall, sctx: ScheduleContext) -> PrimFunc:
    # Conservative scalar fallback
    return copy_default_impl(op_call, sctx)


def copy_tmem_local_impl(op_call: OpCall, sctx: ScheduleContext, async_op=False) -> Optional[PrimFunc]:
    dst_buffer_region, src_buffer_region = op_call.args[:2]
    dst: Buffer = dst_buffer_region.buffer
    src: Buffer = src_buffer_region.buffer

    if src.scope() == "tmem" and dst.scope() == "local":
        direction = "tmem2local"
        tmem_region, local_region = src_buffer_region, dst_buffer_region
    elif src.scope() == "local" and dst.scope() == "tmem":
        direction = "local2tmem"
        local_region, tmem_region = src_buffer_region, dst_buffer_region
    else:
        raise ValueError(f"Unsupported src scope {src.scope()} and dst scope {dst.scope()}")

    tmem_buf, local_buf = tmem_region.buffer, local_region.buffer

    assert tmem_buf.layout is not None
    assert local_buf.layout is not None
    assert tmem_buf.dtype == local_buf.dtype

    analyzer = Analyzer()
    elem_size = DataType(local_buf.dtype).bits
    elem_per_32b = 32 // elem_size
    assert len(local_buf.shape) == len(tmem_buf.shape) == 2
    # local: 128xWIDTH <-> tmem: 128xSHAPE[1]
    assert analyzer.can_prove_equal(local_buf.shape[0], 128)
    assert analyzer.can_prove_equal(tmem_buf.shape[0], 128)

    # Check width is valid for 32x32b, and determine num
    width = local_region.region[1].extent
    candidates = [1, 2, 4, 8, 16, 32, 64, 128]

    if not analyzer.can_prove_equal(tvm.tir.floormod(width, elem_per_32b), 0):
        raise ValueError(f"Width {width} is not valid for tcgen05.ld/st with shape 32x32b")

    num = None
    for n in candidates:
        if analyzer.can_prove_equal(tvm.tir.floordiv(width, elem_per_32b), n):
            num = n
            break
    else:
        raise ValueError(f"Width {width} is not valid for tcgen05.ld/st with shape 32x32b")

    tmem_st, tmem_extent = get_st_extent(tmem_region)
    local_st, local_extent = get_st_extent(local_region)
    # tmem layout (128, WIDTH):(1@TLane, 1@TCol)
    tmem_layout = TileLayout(S[(128, tmem_buf.shape[1]) : (1 @ TLane, 1 @ TCol)]).canonicalize()
    # local layout
    local_layout = TileLayout(S[(128, width) : (1 @ tid_in_wg, 1)]).canonicalize()

    # tmem allocated addr is not None
    assert tmem_buf.allocated_addr is not None
    tvm.ir.assert_structural_equal(tmem_buf.layout.canonicalize(), tmem_layout)
    # tvm.ir.assert_structural_equal(local_buf.layout.canonicalize(), local_layout)
    # local: [0:128, 0:WIDTH] <-> tmem: [0:128, st:st+WIDTH]
    assert analyzer.can_prove_equal(tmem_st[0], 0)
    assert analyzer.can_prove_equal(tmem_extent[0], 128)

    assert analyzer.can_prove_equal(local_st[0], 0)
    assert analyzer.can_prove_equal(local_extent[0], 128)

    offset = tmem_st[1]
    assert analyzer.can_prove_equal(tvm.tir.floormod(offset, elem_per_32b), 0)
    offset_32b = tvm.tir.floordiv(offset, elem_per_32b)
    assert analyzer.can_prove_equal(tmem_extent[1], width), f"tmem_extent[1]: {tmem_extent[1]}, width: {width}"

    # assert analyzer.can_prove_equal(local_st[1], 0)
    assert analyzer.can_prove_equal(local_extent[1], width)

    op = Tx.ptx.tcgen05.ld if direction == "tmem2local" else Tx.ptx.tcgen05.st
    wait_op = Tx.ptx.tcgen05.wait.ld if direction == "tmem2local" else Tx.ptx.tcgen05.wait.st

    # fmt: off
    @Tx.prim_func(tirx=True, check_well_formed=False)
    def impl():
        with Tx.warp():
            local_storage = local_buf.view(local_buf.shape[1] * elem_per_32b, layout=TileLayout(S[num * elem_per_32b]))
            local_32b = local_storage.view("uint32")
            op(tmem_buf.allocated_addr[0], 0, offset_32b, "32x32b", num, False, *[local_32b[local_st[1] // elem_per_32b+i] for i in range(num)])
            if not async_op:
                wait_op()
    # fmt: on
    return impl


@register_dispatch(
    "copy",
    "cuda",
    variant="tmem<->local",
    priority=10,
    when=[
        predicate("validate_copy_op", _is_valid_copy),
        predicate("exec_scope", _exec_scope_ok, expected_scopes=["warpgroup"]),
        predicate(
            "storage_scope", _scope_allowed, allowed_pairs=[("tmem", "local"), ("local", "tmem")]
        ),
    ],
)
def copy_schedule_tmem_local(op_call: OpCall, sctx: ScheduleContext) -> Optional[PrimFunc]:
    return copy_tmem_local_impl(op_call, sctx)


@register_dispatch(
    "copy_async",
    "cuda",
    variant="tmem<->local",
    priority=10,
    when=[
        predicate("validate_copy_op", _is_valid_copy),
        predicate("exec_scope", _exec_scope_ok, expected_scopes=["warpgroup"]),
        predicate(
            "storage_scope", _scope_allowed, allowed_pairs=[("tmem", "local"), ("local", "tmem")]
        ),
    ],
)
def copy_async_schedule_tmem_local_async(op_call: OpCall, sctx: ScheduleContext) -> Optional[PrimFunc]:
    return copy_tmem_local_impl(op_call, sctx, async_op=True)
