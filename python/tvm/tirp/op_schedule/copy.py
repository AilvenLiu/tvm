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
import functools
import operator

from tvm.runtime import DataType
from tvm.arith.analyzer import Analyzer
from tvm.script import tir as T
from tvm.tir import BufferRegion, PrimFunc
from tvm.tirp.op_schedule import ScheduleContext, register_schedule

from .common import register_schedule


def register_copy_schedule(func):
    """Decorator function to register copy operator implementations.

    Parameters
    ----------
    func : Callable[..., Union[bool, PrimFunc]]
    The implementation function.

    Returns
    -------
    Callable
        The decorated function.
    """
    return register_schedule("copy")(func)


@register_copy_schedule
def copy_cuda_g2s_s2g_2d_sync_cta_vec_load(
    dst_buffer_region: BufferRegion, src_buffer_region: BufferRegion, sctx: ScheduleContext
) -> Optional[PrimFunc]:
    """Schedule copy operation between global and shared memory on CUDA."""
    if not sctx.is_cuda():
        return None
    if not sctx.exec_scope.name == "cta":
        return None
    src = src_buffer_region.buffer
    dst = dst_buffer_region.buffer
    if (src.layout is None or dst.layout is None) or (src.dtype != dst.dtype):
        return None

    src_region = src_buffer_region.region
    dst_region = dst_buffer_region.region
    dim = len(src_region)
    # TODO(@bohan): support arbitrary dimension
    if dim != len(dst_region) or dim != 2:
        return None

    analyzer = Analyzer()
    src_st = [src_region[i].min for i in range(dim)]
    dst_st = [dst_region[i].min for i in range(dim)]
    src_extent = [src_region[i].extent for i in range(dim)]
    dst_extent = [dst_region[i].extent for i in range(dim)]

    # The copy region must be the same
    for i in range(dim):
        if not analyzer.can_prove_equal(src_extent[i], dst_extent[i]):
            return None

    # Only support global to shared and shared to global
    if not (src.scope() == "global" and dst.scope() == "shared") and not (
        src.scope() == "shared" and dst.scope() == "global"
    ):
        return None
    # only support trivial layout
    if not (src.layout.is_trivial() or dst.layout.is_trivial()):
        return None
    # TODO(@bohan): support 3D threadIdx
    tx = sctx.launch_params["threadIdx.x"]
    assert "threadIdx.y" not in sctx.launch_params and "threadIdx.z" not in sctx.launch_params

    elem_size = DataType(src.dtype).bits // 8
    n_elements = functools.reduce(operator.mul, src_extent, 1)
    # Convert byte sizes to element counts and filter for alignment constraints
    vec_len_candidates = [16, 8, 4, 1]  # bytes
    vec_len_candidates = list(map(lambda size: size // elem_size, vec_len_candidates))  # elements

    def filter_vec_len(vec_len):
        # Check alignment of source/dest addresses and total elements
        return (
            vec_len > 0
            and analyzer.can_prove_equal(src_st[-1] % vec_len, 0)
            and analyzer.can_prove_equal(dst_st[-1] % vec_len, 0)
            and analyzer.can_prove_equal(src.shape[-1] % vec_len, 0)
            and analyzer.can_prove_equal(dst.shape[-1] % vec_len, 0)
            and analyzer.can_prove_equal(src_extent[-1] % vec_len, 0)
            and analyzer.can_prove_equal(dst_extent[-1] % vec_len, 0)
            and analyzer.can_prove_equal(n_elements % (tx * vec_len), 0)
        )

    vec_len_candidates = list(filter(filter_vec_len, vec_len_candidates))
    if not vec_len_candidates:
        return None
    vec_len = vec_len_candidates[0]

    # fmt: off
    @T.prim_func(tirp=True)
    def impl():
        for s in T.serial(0, n_elements // (tx * vec_len)):
            for tid_x in T.thread_binding(tx, "threadIdx.x"):
                for vec in T.vectorized(vec_len):
                    fused = T.meta_var((s * tx + tid_x) * vec_len + vec)
                    dst[dst_st[0] + fused // dst_extent[1], dst_st[1] + fused % dst_extent[1]] = src[src_st[0] + fused // src_extent[1], src_st[1] + fused % src_extent[1]]
        if dst.scope() == "shared":
            T.tvm_storage_sync("shared")
    # fmt: on

    return impl
