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
# pylint: disable=redefined-builtin, invalid-name, too-many-arguments
"""PTX barrier / mbarrier / fence intrinsics.

Schema-expressible ops are declared via ``ptx_intrinsic(...)`` at the top of
this module. Ops whose codegen can't fit the schema (label loops, local/remote
split, predicate register inlining) are hand-written below and registered via
``@register_codegen(...)``.
"""

from tvm.tirx.op import cuda_func_call

from .._schema import Bool, Choice, Derived, IntAttr, Operand, ptx_intrinsic  # noqa: F401
from .registry import register_codegen

# =============================================================================
# Schema-declared ops.
# =============================================================================

# -----------------------------------------------------------------------------
# bar.arrive / bar.sync.
# Python API:
#     Tx.ptx.bar.arrive(name_bar_id, thread_count)
#     Tx.ptx.bar.sync(name_bar_id, thread_count)
# -----------------------------------------------------------------------------
ptx_intrinsic(
    op_name="ptx_bar_arrive",
    operands=[Operand("name_bar_id", c_type="int"), Operand("thread_count", c_type="int")],
    ptx_template="bar.arrive %0, %1;",
    helper_name="tvm_builtin_ptx_bar_arrive",
)
ptx_intrinsic(
    op_name="ptx_bar_sync",
    operands=[Operand("name_bar_id", c_type="int"), Operand("thread_count", c_type="int")],
    ptx_template="bar.sync %0, %1;",
    helper_name="tvm_builtin_ptx_bar_sync",
)


# -----------------------------------------------------------------------------
# fence — memory fence.
# Python API:
#     Tx.ptx.fence(sem, scope)
#         sem ∈ {"sc", "acq_rel"}; scope ∈ {"cta", "cluster", "gpu", "sys"}
# -----------------------------------------------------------------------------
ptx_intrinsic(
    op_name="ptx_fence",
    attrs=[
        Choice("sem", choices=("sc", "acq_rel")),
        Choice("scope", choices=("cta", "cluster", "gpu", "sys")),
    ],
    ptx_template="fence.{sem}.{scope};",
    helper_name_template="tvm_builtin_ptx_fence_{sem}_{scope}",
)


# -----------------------------------------------------------------------------
# fence.proxy.async — cross-proxy ordering.
# Python API:
#     Tx.ptx.fence.proxy_async(space="")
#         space ∈ {"", "global", "shared::cta", "shared::cluster"}
# -----------------------------------------------------------------------------
ptx_intrinsic(
    op_name="ptx_fence_proxy_async",
    attrs=[
        Choice(
            "space",
            default="",
            choices=("", "global", "shared::cta", "shared::cluster"),
            ptx_suffix=".{value}",
        ),
    ],
    derived=[
        Derived(
            "space_sfx",
            from_=lambda a: ("_" + a.space.replace("::", "_")) if a.space else "",
        ),
    ],
    ptx_template="fence.proxy.async{space};",
    helper_name_template="tvm_builtin_ptx_fence_proxy_async{space_sfx}",
)


# -----------------------------------------------------------------------------
# fence.mbarrier_init — mbarrier init visibility.
# Python API:
#     Tx.ptx.fence.mbarrier_init()
# -----------------------------------------------------------------------------
ptx_intrinsic(
    op_name="ptx_fence_mbarrier_init",
    ptx_template="fence.mbarrier_init.release.cluster;",
    helper_name="tvm_builtin_ptx_fence_mbarrier_init",
)


# -----------------------------------------------------------------------------
# barrier.cluster.arrive / wait.
# Python API:
#     Tx.ptx.barrier.cluster.arrive(sem="", aligned=True)
#         sem ∈ {"", "release", "relaxed"}
#     Tx.ptx.barrier.cluster.wait(acquire=False, aligned=True)
# -----------------------------------------------------------------------------
ptx_intrinsic(
    op_name="ptx_barrier_cluster_arrive",
    attrs=[
        Choice("sem", default="", choices=("", "release", "relaxed"), ptx_suffix=".{value}"),
        Bool("aligned", default=True, ptx_suffix=".aligned"),
    ],
    ptx_template="barrier.cluster.arrive{sem}{.aligned?};",
    helper_name_template="tvm_builtin_ptx_barrier_cluster_arrive{_sem?}{_aligned?aligned}",
)
ptx_intrinsic(
    op_name="ptx_barrier_cluster_wait",
    attrs=[
        Bool("acquire", default=False, ptx_suffix=".acquire"),
        Bool("aligned", default=True, ptx_suffix=".aligned"),
    ],
    ptx_template="barrier.cluster.wait{.acquire?}{.aligned?};",
    helper_name_template="tvm_builtin_ptx_barrier_cluster_wait{_acquire?acquire}{_aligned?aligned}",
)


# -----------------------------------------------------------------------------
# mbarrier.init — initialize a shared-memory mbarrier.
# Python API:
#     Tx.ptx.mbarrier.init(bar, thread_count)
# bar is a generic pointer (converted to SMEM via __cvta_generic_to_shared).
# -----------------------------------------------------------------------------
ptx_intrinsic(
    op_name="ptx_mbarrier_init",
    operands=[
        Operand("barrier", c_type="void*", cvta_to_shared=True),
        Operand("thread_count", c_type="int"),
    ],
    ptx_template="mbarrier.init.shared.b64 [%0], %1;",
    helper_name="tvm_builtin_ptx_mbarrier_init",
)


# =============================================================================
# Hand-written ops — schema can't express local/remote dispatch or inline
# PTX label loops with predicate registers.
# =============================================================================


@register_codegen("ptx_elect_sync")
def codegen_ptx_elect_sync():
    func_name = "tvm_builtin_elect_one_sync_op"
    source_code = f"""
__forceinline__ __device__ uint32_t {func_name}() {{
  return tvm_builtin_elect_one_sync();
}}
"""
    return cuda_func_call(func_name, source_code=source_code, return_type="uint32"), [
        "elect_one_sync"
    ]


_MBARRIER_ARRIVE_LOCAL = """
__forceinline__ __device__ void {func_name}(void* barrier) {{
  unsigned int barrier_addr_int = __cvta_generic_to_shared(barrier);
  __asm__ __volatile__(
    "mbarrier.arrive.shared.b64 _, [%0];"
    :: "r"(barrier_addr_int) : "memory"
  );
}}
"""

_MBARRIER_ARRIVE_REMOTE = """
__forceinline__ __device__ void {func_name}(void* barrier, int cta_id, int pred) {{
  unsigned int barrier_addr_int = __cvta_generic_to_shared(barrier);
  asm volatile(
      "{{\\n"
      ".reg .pred p;\\n"
      ".reg .b32 remAddr32;\\n"
      "setp.eq.u32 p, %2, 1;\\n"
      "@p mapa.shared::cluster.u32  remAddr32, %0, %1;\\n"
      "@p mbarrier.arrive.shared::cluster.b64  _, [remAddr32];\\n"
      "}}\\n"
      :
      : "r"(barrier_addr_int), "r"(cta_id), "r"(pred) : "memory");
}}
"""

_MBARRIER_ARRIVE_EXPECT_TX_LOCAL = """
__forceinline__ __device__ void {func_name}(void* barrier, int byte_count) {{
  unsigned int barrier_addr_int = __cvta_generic_to_shared(barrier);
  __asm__ __volatile__(
    "mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"
    :: "r"(barrier_addr_int), "r"(byte_count) : "memory"
  );
}}
"""

_MBARRIER_ARRIVE_EXPECT_TX_REMOTE = """
__forceinline__ __device__ void {func_name}(void* barrier, int byte_count, int cta_id, int pred) {{
  unsigned int barrier_addr_int = __cvta_generic_to_shared(barrier);
  asm volatile(
      "{{\\n"
      ".reg .pred p;\\n"
      ".reg .b32 remAddr32;\\n"
      "setp.eq.u32 p, %2, 1;\\n"
      "@p mapa.shared::cluster.u32  remAddr32, %0, %1;\\n"
      "@p mbarrier.arrive.expect_tx.shared::cluster.b64  _, [remAddr32], %3;\\n"
      "}}\\n"
      :
      : "r"(barrier_addr_int), "r"(cta_id), "r"(pred), "r"(byte_count) : "memory");
}}
"""


@register_codegen("ptx_mbarrier_arrive")
def codegen_ptx_mbarrier_arrive(bar, cta_id=None, pred=None):
    if cta_id is None and pred is None:
        func_name = "tvm_builtin_ptx_mbarrier_arrive"
        return cuda_func_call(
            func_name, bar, source_code=_MBARRIER_ARRIVE_LOCAL.format(func_name=func_name)
        )
    func_name = "tvm_builtin_ptx_mbarrier_arrive_remote"
    return cuda_func_call(
        func_name,
        bar,
        cta_id,
        pred,
        source_code=_MBARRIER_ARRIVE_REMOTE.format(func_name=func_name),
    )


@register_codegen("ptx_mbarrier_arrive_expect_tx")
def codegen_ptx_mbarrier_arrive_expect_tx(bar, byte_count, cta_id=None, pred=None):
    if cta_id is None and pred is None:
        func_name = "tvm_builtin_ptx_mbarrier_arrive_expect_tx"
        return cuda_func_call(
            func_name,
            bar,
            byte_count,
            source_code=_MBARRIER_ARRIVE_EXPECT_TX_LOCAL.format(func_name=func_name),
        )
    func_name = "tvm_builtin_ptx_mbarrier_arrive_expect_tx_remote"
    return cuda_func_call(
        func_name,
        bar,
        byte_count,
        cta_id,
        pred,
        source_code=_MBARRIER_ARRIVE_EXPECT_TX_REMOTE.format(func_name=func_name),
    )


@register_codegen("ptx_mbarrier_try_wait")
def codegen_ptx_mbarrier_try_wait(bar, phase):
    # 0x989680 = 10,000,000 ns = 10ms timeout. The timeout causes the compiler
    # to generate NANOSLEEP.SYNCS instead of YIELD (matches CUTLASS).
    func_name = "tvm_builtin_ptx_mbarrier_wait"
    source_code = f"""
__forceinline__ __device__ void {func_name}(void* barrier, int phase) {{
   unsigned int barrier_addr_int = __cvta_generic_to_shared(barrier);
   unsigned int ticks = 0x989680;
  asm volatile (
      "{{\\n"
      ".reg .pred                P1;\\n"
      "LAB_WAIT:\\n"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2;\\n"
      "@P1                       bra.uni DONE;\\n"
      "bra.uni                   LAB_WAIT;\\n"
      "DONE:\\n"
      "}}\\n"
      ::
      "r"(barrier_addr_int),
      "r"(phase),
      "r"(ticks) : "memory"
  );
}}
"""
    return cuda_func_call(func_name, bar, phase, source_code=source_code)
