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

"""Common utilities for operator scheduling."""

from enum import Enum
from typing import Optional, Callable
import functools
import operator
from functools import wraps
from tvm.script import tir as T
from tvm.tirp.op_schedule import ScheduleContext
from tvm.runtime import DataType
from tvm.arith.analyzer import Analyzer
from tvm.tir import BufferRegion, PrimFunc, Buffer
from tvm.tir.stmt import OpCall


def get_indices(nth, start, extent):
    """Convert a fused index into multi-dimensional indices."""
    relative = []
    for e in reversed(extent):
        relative.append(nth % e)
        nth //= e
    return [r + s for r, s in zip(reversed(relative), start)]


def target_cuda(fn: Callable[[OpCall, ScheduleContext], Optional[PrimFunc]]):
    """Decorator that ensures the function is only executed for CUDA targets."""

    @wraps(fn)
    def wrapper(*args, **kwargs):
        sctx = kwargs.get("sctx", None)
        if sctx is None:
            assert len(args) == 2 and isinstance(
                args[1], ScheduleContext
            ), "The target_cuda() needs to annotate a function with signature (op_call, sctx)"
            sctx = args[1]
        if not sctx.is_cuda():
            return None
        return fn(*args, **kwargs)

    return wrapper


class InstType(Enum):
    """Enumeration of instruction types for memory operations."""

    NORMAL = 0
    CP_ASYNC = 1


def copy_g2s_s2g_cta_vec_load_impl(
    dst_buffer_region: BufferRegion,
    src_buffer_region: BufferRegion,
    sctx: ScheduleContext,
    inst_type: InstType,
) -> Optional[PrimFunc]:
    """Schedule copy operation between global and shared memory on CUDA."""
    # Basic validation checks
    if sctx.exec_scope.name != "cta":
        return None

    src: Buffer = src_buffer_region.buffer
    dst: Buffer = dst_buffer_region.buffer
    if not all(
        [
            src.layout and dst.layout,
            src.dtype == dst.dtype,
            src.layout.is_trivial() and dst.layout.is_trivial(),
            (src.scope() == "global" and dst.scope().startswith("shared"))
            or (src.scope().startswith("shared") and dst.scope() == "global"),
        ]
    ):
        return None

    # Extract regions and validate dimensions
    analyzer = Analyzer()
    src_region, dst_region = src_buffer_region.region, dst_buffer_region.region
    src_st = [r.min for r in src_region]
    dst_st = [r.min for r in dst_region]
    src_extent = [r.extent for r in src_region]
    dst_extent = [r.extent for r in dst_region]

    # Validate non-unit dimensions match
    src_extent_ = [e for e in src_extent if e != 1]
    dst_extent_ = [e for e in dst_extent if e != 1]
    if not (
        len(src_extent_) == len(dst_extent_)
        and all(analyzer.can_prove_equal(s, d) for s, d in zip(src_extent_, dst_extent_))
    ):
        return None

    # Thread and vectorization setup
    tx = sctx.launch_params["threadIdx.x"]
    assert "threadIdx.y" not in sctx.launch_params and "threadIdx.z" not in sctx.launch_params

    elem_size = DataType(src.dtype).bits  # in bits
    n_elements = functools.reduce(operator.mul, src_extent, 1)

    # Find valid vector length
    if n_elements % tx != 0:
        return None
    for vec_len in [128 // elem_size, 64 // elem_size, 32 // elem_size, 1]:
        if vec_len > 0 and all(
            analyzer.can_prove_equal(x % vec_len, 0)
            for x in [
                src_st[-1],
                dst_st[-1],
                src.shape[-1],
                dst.shape[-1],
                src_extent[-1],
                dst_extent[-1],
                n_elements // tx,
            ]
        ):
            break
    else:
        return None
    # cp-size (the size of data in bytes) can only be 4, 8 and 16 for cp.async
    if inst_type == InstType.CP_ASYNC:
        cp_size = vec_len * elem_size // 8  # in bytes
        if cp_size not in [4, 8, 16]:
            return None

    def get_indices(st, extent, fused):
        indices = []
        product = 1
        for i in reversed(range(len(extent))):
            indices.append(st[i] + (fused // product) % extent[i])
            product *= extent[i]
        return reversed(indices)

    # fmt: off
    @T.prim_func(tirp=True)
    def impl():
        """Implement copy operation with vectorized loads/stores."""
        for s in T.serial(0, n_elements // (tx * vec_len)):
            for tid_x in T.thread_binding(tx, "threadIdx.x"):
                if inst_type == InstType.NORMAL:
                    for vec in T.vectorized(vec_len):
                        fused = T.meta_var((s * tx + tid_x) * vec_len + vec)
                        dst_indices = T.meta_var(get_indices(dst_st, dst_extent, fused))
                        src_indices = T.meta_var(get_indices(src_st, src_extent, fused))
                        dst[*dst_indices] = src[*src_indices]
                elif inst_type == InstType.CP_ASYNC:
                    fused = T.meta_var((s * tx + tid_x) * vec_len)
                    dst_indices = T.meta_var(get_indices(dst_st, dst_extent, fused))
                    src_indices = T.meta_var(get_indices(src_st, src_extent, fused))
                    T.evaluate(T.ptx_cp_async(dst.dtype, dst.data, dst.offset_of_p([*dst_indices]),
                                              src.data, src.offset_of_p([*src_indices]), cp_size))
        if dst.scope().startswith("shared") and inst_type == InstType.NORMAL:
            T.tvm_storage_sync("shared")
    # fmt: on

    return impl
