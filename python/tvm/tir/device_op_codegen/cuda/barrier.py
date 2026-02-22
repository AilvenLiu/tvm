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
"""PTX barrier and mbarrier operations."""
from tvm.tir.op import cuda_func_call

from .registry import register_codegen
from .utils import parse_str


@register_codegen("ptx_bar_arrive")
def codegen_ptx_bar_arrive(name_bar_id, thread_count):
    func_name = "tvm_builtin_ptx_bar_arrive"
    source_code = f"""
__forceinline__ __device__ void {func_name}(int name_bar_id, int thread_count) {{
    asm volatile("bar.arrive %0, %1;" : : "r"(name_bar_id), "r"(thread_count) : "memory");
}}
"""
    return cuda_func_call(func_name, name_bar_id, thread_count, source_code=source_code)


@register_codegen("ptx_bar_sync")
def codegen_ptx_bar_sync(name_bar_id, thread_count):
    func_name = "tvm_builtin_ptx_bar_sync"
    source_code = f"""
__forceinline__ __device__ void {func_name}(int name_bar_id, int thread_count) {{
    asm volatile("bar.sync %0, %1;" : : "r"(name_bar_id), "r"(thread_count) : "memory");
}}
"""
    return cuda_func_call(func_name, name_bar_id, thread_count, source_code=source_code)


@register_codegen("ptx_fence_proxy")
def codegen_ptx_fence_proxy(scope):
    scope = parse_str(scope)
    func_name = f"tvm_builtin_ptx_fence_proxy_{scope}"

    if scope == "shared":
        ptx_scope = ".async.shared::cta"
    elif scope == "global":
        ptx_scope = ".async.global"
    else:
        raise ValueError(f"Invalid scope for ptx_fence_proxy: {scope}")

    source_code = f"""
__forceinline__ __device__ void {func_name}() {{
  __asm__ __volatile__("fence.proxy{ptx_scope};" : : : "memory");
}}
"""
    return cuda_func_call(func_name, source_code=source_code)


@register_codegen("ptx_fence_mbarrier_init_release_cluster")
def codegen_ptx_fence_mbarrier_init_release_cluster():
    func_name = "tvm_builtin_ptx_fence_mbarrier_init_release_cluster"
    source_code = f"""
__forceinline__ __device__ void {func_name}() {{
    asm volatile("fence.mbarrier_init.release.cluster;" : : : "memory");
}}
"""
    return cuda_func_call(func_name, source_code=source_code)


@register_codegen("ptx_barrier_cluster_arrive")
def codegen_ptx_barrier_cluster_arrive(sem, aligned):
    sem = parse_str(sem)
    aligned = bool(aligned)

    sem_name = "" if len(sem) == 0 else f"_{sem}"
    aligned_name = "_aligned" if aligned else ""

    sem_inst = "" if len(sem) == 0 else f".{sem}"
    aligned_inst = ".aligned" if aligned else ""

    func_name = f"tvm_builtin_ptx_barrier_cluster_arrive{sem_name}{aligned_name}"
    source_code = f"""
__forceinline__ __device__ void {func_name}() {{
    asm volatile("barrier.cluster.arrive{sem_inst}{aligned_inst};\\n" : :);
}}
"""
    return cuda_func_call(func_name, source_code=source_code)


@register_codegen("ptx_barrier_cluster_wait")
def codegen_ptx_barrier_cluster_wait(acquire, aligned):
    acquire_name = "_acquire" if bool(acquire) else ""
    aligned_name = "_aligned" if bool(aligned) else ""

    acquire_inst = ".acquire" if bool(acquire) else ""
    aligned_inst = ".aligned" if aligned else ""

    func_name = f"tvm_builtin_ptx_barrier_cluster_wait{acquire_name}{aligned_name}"
    source_code = f"""
__forceinline__ __device__ void {func_name}() {{
    asm volatile("barrier.cluster.wait{acquire_inst}{aligned_inst};\\n" : :);
}}
"""
    return cuda_func_call(func_name, source_code=source_code)


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


#################### mbarrier


@register_codegen("ptx_mbarrier_init")
def codegen_ptx_mbarrier_init(bar, thread_count):
    func_name = "tvm_builtin_ptx_mbarrier_init"
    source_code = f"""
__forceinline__ __device__ void {func_name}(void* barrier, int thread_count) {{
  unsigned int barrier_addr_int = __cvta_generic_to_shared(barrier);
  __asm__ __volatile__(
    "mbarrier.init.shared.b64 [%0], %1;"
    :: "r"(barrier_addr_int), "r"(thread_count) : "memory"
  );
}}
"""
    return cuda_func_call(func_name, bar, thread_count, source_code=source_code)


@register_codegen("ptx_mbarrier_arrive")
def codegen_ptx_mbarrier_arrive(bar, cta_id=None, pred=None):
    remote = cta_id is not None and pred is not None
    if not remote:
        func_name = "tvm_builtin_ptx_mbarrier_arrive"
        source_code = f"""
__forceinline__ __device__ void {func_name}(void* barrier) {{
  unsigned int barrier_addr_int = __cvta_generic_to_shared(barrier);
  __asm__ __volatile__(
    "mbarrier.arrive.shared.b64 _, [%0];"
    :: "r"(barrier_addr_int) : "memory"
  );
}}
"""
        return cuda_func_call(func_name, bar, source_code=source_code)
    else:
        func_name = "tvm_builtin_ptx_mbarrier_arrive_remote"
        source_code = f"""
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
        return cuda_func_call(func_name, bar, cta_id, pred, source_code=source_code)


@register_codegen("ptx_mbarrier_arrive_expect_tx")
def codegen_ptx_mbarrier_arrive_expect_tx(bar, byte_count, cta_id=None, pred=None):
    remote = cta_id is not None and pred is not None
    if not remote:
        func_name = "tvm_builtin_ptx_mbarrier_arrive_expect_tx"
        source_code = f"""
__forceinline__ __device__ void {func_name}(void* barrier, int byte_count) {{
  unsigned int barrier_addr_int = __cvta_generic_to_shared(barrier);
  __asm__ __volatile__(
    "mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"
    :: "r"(barrier_addr_int), "r"(byte_count) : "memory"
  );
}}
"""
        return cuda_func_call(func_name, bar, byte_count, source_code=source_code)
    else:
        func_name = "tvm_builtin_ptx_mbarrier_arrive_expect_tx_remote"
        source_code = f"""
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
        return cuda_func_call(func_name, bar, byte_count, cta_id, pred, source_code=source_code)


@register_codegen("ptx_mbarrier_try_wait")
def codegen_ptx_mbarrier_try_wait(bar, phase):
    func_name = "tvm_builtin_ptx_mbarrier_wait"
    # Add timeout parameter to mbarrier.try_wait (same as CUTLASS)
    # The timeout causes the compiler to generate NANOSLEEP.SYNCS instead of YIELD
    # 0x989680 = 10,000,000 ns = 10ms timeout
    source_code = f"""
__forceinline__ __device__ void {func_name}(void* barrier, int phase) {{
   unsigned int barrier_addr_int = __cvta_generic_to_shared(barrier);
   unsigned int ticks = 0x989680;  // 10ms timeout
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
