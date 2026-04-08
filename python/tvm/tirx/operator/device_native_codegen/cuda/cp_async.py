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
# pylint: disable=redefined-builtin, invalid-name, too-many-arguments, too-many-locals
"""PTX Data Movement and Conversion Instructions: Asynchronous Copy."""

import tvm
from tvm.tirx.op import cuda_func_call

from .registry import register_codegen
from .utils import parse_str

# see https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/copy_sm90_desc.hpp#L186
CacheHint = {
    "evict_normal": 0x1000000000000000,
    "evict_first": 0x12F0000000000000,
    "evict_last": 0x14F0000000000000,
}


#################### Non-bulk copy


@register_codegen("ptx_cp_async")
def codegen_ptx_cp_async(
    dst_ptr, src_ptr, cp_size, cache_hint, prefetch_size, predicate, fill_mode
):
    cp_size = int(cp_size)
    func_name = f"tvm_builtin_ptx_cp_async_{cp_size}"

    if cache_hint != "":
        cache_hint = parse_str(cache_hint)
        func_name += f"_{cache_hint}"
        cache_hint_inst = ".L2::cache_hint"
        cache_hint_arg = f', "n"({CacheHint[cache_hint]})'
    else:
        cache_hint_inst = ""
        cache_hint_arg = ""

    if prefetch_size != -1:
        prefetch_size = int(prefetch_size)
        func_name += f"_prefetch_{prefetch_size}"
        assert prefetch_size in [64, 128, 256]
        prefetch_inst = f".L2::{prefetch_size}B"
    else:
        prefetch_inst = ""

    if fill_mode != "":
        fill_mode = parse_str(fill_mode)
        func_name += f"_{fill_mode}"

    if predicate != -1:
        func_name += "_predicate"

    assert cp_size in [4, 8, 16]
    if cp_size == 16:
        ca_or_cg = ".cg"
    else:
        ca_or_cg = ".ca"

    if fill_mode != "":
        # fill the dst with zero if predicate is false
        assert predicate != -1
        assert fill_mode == "zero"
        cache_hint_operand = ", %4" if cache_hint != "" else ""
        source_code = f"""
__forceinline__ __device__ void {func_name}(void* dst, void* src, int predicate) {{
  unsigned int dst_addr = __cvta_generic_to_shared(dst);
  int src_size = predicate ? {cp_size} : 0;
  __asm__ __volatile__(
    "cp.async{ca_or_cg}.shared.global{cache_hint_inst}{prefetch_inst} [%0], [%1], %2, %3{cache_hint_operand};\\n"
    :: "r"(dst_addr), "l"(src), "n"({cp_size}), "r"(src_size){cache_hint_arg}
  );
}}
"""  # noqa: E501
        return cuda_func_call(func_name, dst_ptr, src_ptr, predicate, source_code=source_code)
    else:
        if predicate == -1:
            # no predicate, just copy the src to dst
            cache_hint_operand = ", %3" if cache_hint != "" else ""
            source_code = f"""
__forceinline__ __device__ void {func_name}(void* dst, void* src) {{
  unsigned int dst_addr = __cvta_generic_to_shared(dst);
  __asm__ __volatile__(
    "cp.async{ca_or_cg}.shared.global{cache_hint_inst}{prefetch_inst} [%0], [%1], %2{cache_hint_operand};\\n"
    :: "r"(dst_addr), "l"(src), "n"({cp_size}){cache_hint_arg}
  );
}}
"""  # noqa: E501
            return cuda_func_call(func_name, dst_ptr, src_ptr, source_code=source_code)
        else:
            # predicate is true, copy the src to dst
            cache_hint_operand = ", %4" if cache_hint != "" else ""
            source_code = f"""
__forceinline__ __device__ void {func_name}(void* dst, void* src, int predicate) {{
  unsigned int dst_addr = __cvta_generic_to_shared(dst);
  __asm__ __volatile__(
    "{{\\n"
    " .reg .pred p;\\n"
    " setp.eq.u32 p, %3, 1;\\n"
    " @p cp.async{ca_or_cg}.shared.global{cache_hint_inst}{prefetch_inst} [%0], [%1], %2{cache_hint_operand};\\n"
    "}}\\n"
    :: "r"(dst_addr), "l"(src), "n"({cp_size}), "r"(predicate){cache_hint_arg}
  );
}}
"""  # noqa: E501
            return cuda_func_call(func_name, dst_ptr, src_ptr, predicate, source_code=source_code)


@register_codegen("ptx_cp_async_commit_group")
def codegen_ptx_cp_async_commit_group():
    func_name = "tvm_builtin_ptx_cp_async_commit_group"
    source_code = f"""
__forceinline__ __device__ void {func_name}() {{
    asm volatile("cp.async.commit_group;");
}}
"""
    return cuda_func_call(func_name, source_code=source_code)


@register_codegen("ptx_cp_async_wait_group")
def codegen_ptx_cp_async_wait_group(n):
    n = int(n)
    func_name = "tvm_builtin_ptx_cp_async_wait_group" + f"_{n}"
    source_code = f"""
__forceinline__ __device__ void {func_name}() {{
    asm volatile("cp.async.wait_group %0;" :: "n"({n}));
}}
"""
    return cuda_func_call(func_name, source_code=source_code)


#################### TMA (cp.async.bulk.tensor)


def _codegen_cp_async_bulk_tensor_global_to_cluster(
    dim, dst_ptr, bar, tensormap, *args, tile_mode="tile"
):
    dim = int(dim)
    cta_mask, cta_group, cache_hint = args[0], int(args[1]), args[2]
    coords = args[3:]
    if tile_mode == "tile_gather4":
        if dim != 2 or len(coords) != 5:
            raise ValueError(
                "tile_gather4 requires dim=2 and 5 coordinate expressions "
                f"(got dim={dim}, coords={len(coords)})."
            )
    elif len(coords) != dim:
        raise ValueError(
            f"Number of coordinate expressions ({len(coords)}) does not match dimension ({dim})."
        )
    if cache_hint != "":
        cache_hint = parse_str(cache_hint)

    # Check if multicast is enabled (popcount(cta_mask) > 1)
    # cta_mask=0 means no mask, cta_mask=1 means unicast (only CTA 0) — both are non-multicast.
    # Multicast requires cluster launch mode, so only emit .multicast::cluster when multiple CTAs
    # are targeted (i.e., more than one bit set).
    is_unicast = isinstance(cta_mask, tvm.tirx.IntImm) and bin(int(cta_mask)).count("1") <= 1
    is_multicast = not is_unicast

    func_name = (
        f"ptx_cp_async_bulk_tensor_global_to_cluster_{tile_mode}_{dim}d"
        + ("_multicast" if is_multicast else "")
        + (f"_cta_group_{cta_group}" if cta_group == 2 else "")
        + (f"_{cache_hint}" if cache_hint != "" else "")
    )
    coord_count = len(coords) if tile_mode == "tile_gather4" else dim
    coord_arg_list = ", ".join([f"int coord{i}" for i in range(coord_count)])

    # The operand indices are different for unicast vs. multicast
    coord_arg_start = 3
    if is_multicast:
        coord_arg_start += 1
    if cache_hint != "":
        coord_arg_start += 1
    if tile_mode == "tile_gather4":
        coord_arg_template = "{%" + ", %".join([str(coord_arg_start + i) for i in range(coord_count)]) + "}"
        coord_list_constraints = ", ".join([f'"r"(coord{i})' for i in range(coord_count)])
    else:
        coord_arg_template = "{%" + ", %".join([str(coord_arg_start + i) for i in range(dim)]) + "}"
        coord_list_constraints = ", ".join([f'"r"(coord{i})' for i in range(dim)])

    def is_sm100_or_higher():
        target = tvm.target.Target.current()
        if target is None:
            return False
        arch = target.arch[3:]
        if not arch[-1].isdigit():
            arch = arch[:-1]
        return int(arch) >= 100

    is_sm100_or_higher = is_sm100_or_higher()

    if cta_group == 2 or (cta_group != -1 and is_sm100_or_higher):
        cta_group_str = f".cta_group::{cta_group}"
    else:
        cta_group_str = ""

    if cache_hint != "":
        cache_hint_str = ".L2::cache_hint"
    else:
        cache_hint_str = ""

    tile_modifier = ".tile::gather4" if tile_mode == "tile_gather4" else ""

    if is_multicast:
        cache_hint_operand = ", %4" if cache_hint != "" else ""
        cache_hint_value = f', "n"({CacheHint[cache_hint]})' if cache_hint != "" else ""
        source_code = f"""
__forceinline__ __device__ void {func_name}(void* dst, void* bar, const CUtensorMap& tensormap, int cta_mask_arg, {coord_arg_list}) {{
  unsigned int dst_addr = __cvta_generic_to_shared(dst);
  unsigned int bar_addr = __cvta_generic_to_shared(bar);
  uint64_t tensormap_addr = reinterpret_cast<uint64_t>(&tensormap);
  uint16_t cta_mask = static_cast<uint16_t>(cta_mask_arg);
  __asm__ __volatile__(
    "cp.async.bulk.tensor.{dim}d.shared::cluster.global{tile_modifier}.mbarrier::complete_tx::bytes.multicast::cluster{cta_group_str}{cache_hint_str}"
    " [%0], [%1, {coord_arg_template}], [%2], %3{cache_hint_operand};"
    :
    : "r"(dst_addr), "l"(tensormap_addr), "r"(bar_addr), "h"(cta_mask){cache_hint_value},
        {coord_list_constraints}
    : "memory"
  );
}}
"""  # noqa: E501
    else:
        cache_hint_operand = ", %3" if cache_hint != "" else ""
        cache_hint_value = f', "n"({CacheHint[cache_hint]})' if cache_hint != "" else ""
        source_code = f"""
__forceinline__ __device__ void {func_name}(void* dst, void* bar, const CUtensorMap& tensormap, {coord_arg_list}) {{
  unsigned int dst_addr = __cvta_generic_to_shared(dst);
  unsigned int bar_addr = __cvta_generic_to_shared(bar);
  uint64_t tensormap_addr = reinterpret_cast<uint64_t>(&tensormap);
  __asm__ __volatile__(
    "cp.async.bulk.tensor.{dim}d.shared::cluster.global{tile_modifier}.mbarrier::complete_tx::bytes{cta_group_str}{cache_hint_str}"
    " [%0], [%1, {coord_arg_template}], [%2]{cache_hint_operand};"
    :
    : "r"(dst_addr), "l"(tensormap_addr), "r"(bar_addr){cache_hint_value},
        {coord_list_constraints}
    : "memory"
  );
}}
"""  # noqa: E501

    if is_multicast:
        return cuda_func_call(
            func_name, dst_ptr, bar, tensormap, cta_mask, *coords, source_code=source_code
        )
    else:
        return cuda_func_call(func_name, dst_ptr, bar, tensormap, *coords, source_code=source_code)


@register_codegen("ptx_cp_async_bulk_tensor_global_to_cluster")
def codegen_ptx_cp_async_bulk_tensor_global_to_cluster(dim, dst_ptr, bar, tensormap, *args):
    return _codegen_cp_async_bulk_tensor_global_to_cluster(dim, dst_ptr, bar, tensormap, *args)


@register_codegen("ptx_cp_async_bulk_tensor_tile_gather4_global_to_cluster")
def codegen_ptx_cp_async_bulk_tensor_tile_gather4_global_to_cluster(
    dim, dst_ptr, bar, tensormap, *args
):
    return _codegen_cp_async_bulk_tensor_global_to_cluster(
        dim, dst_ptr, bar, tensormap, *args, tile_mode="tile_gather4"
    )


@register_codegen("ptx_cp_async_bulk_tensor_shared_to_global")
def codegen_ptx_cp_async_bulk_tensor_shared_to_global(dim, src_ptr, tensormap, *args):
    dim = int(dim)
    cache_hint = args[0]
    coords = args[1:]
    if len(coords) != dim:
        raise ValueError(
            f"Number of coordinate expressions ({len(coords)}) does not match dimension ({dim})."
        )
    if cache_hint != "":
        cache_hint = parse_str(cache_hint)

    func_name = f"ptx_cp_async_bulk_tensor_shared_to_global_{dim}d" + (
        f"_{cache_hint}" if cache_hint != "" else ""
    )
    coord_arg_list = ", ".join([f"int coord{i}" for i in range(dim)])

    coord_indices = (
        [str(2 + i) for i in range(dim)] if cache_hint == "" else [str(3 + i) for i in range(dim)]
    )
    arg_template = "{%" + ", %".join(coord_indices) + "}"

    coord_list_constraints = ", ".join([f'"r"(coord{i})' for i in range(dim)])

    if cache_hint != "":
        cache_hint_str = ".L2::cache_hint"
    else:
        cache_hint_str = ""

    cache_hint_operand = ", %2" if cache_hint != "" else ""
    cache_hint_value = f', "n"({CacheHint[cache_hint]})' if cache_hint != "" else ""

    source_code = f"""
__forceinline__ __device__ void {func_name}(void* src, const CUtensorMap& tensormap, {coord_arg_list}) {{
  unsigned int src_addr = __cvta_generic_to_shared(src);
  uint64_t tensormap_addr = reinterpret_cast<uint64_t>(&tensormap);
  __asm__ __volatile__(
    "cp.async.bulk.tensor.{dim}d.global.shared::cta.tile.bulk_group{cache_hint_str}"
    " [%0, {arg_template}], [%1]{cache_hint_operand};"
    :
    : "l"(tensormap_addr), "r"(src_addr){cache_hint_value},
      {coord_list_constraints}
    : "memory"
  );
}}
"""  # noqa: E501
    return cuda_func_call(func_name, src_ptr, tensormap, *coords, source_code=source_code)


@register_codegen("ptx_cp_async_bulk_tensor_global_to_cluster_prefetch")
def codegen_ptx_cp_async_bulk_tensor_global_to_cluster_prefetch(dim, tensormap, *args):
    dim = int(dim)
    cache_hint = args[0]
    coords = args[1:]
    if len(coords) != dim:
        raise ValueError(
            f"Number of coordinate expressions ({len(coords)}) does not match dimension ({dim})."
        )
    if cache_hint != "":
        cache_hint = parse_str(cache_hint)

    func_name = f"ptx_cp_async_bulk_tensor_global_to_cluster_prefetch_{dim}d" + (
        f"_{cache_hint}" if cache_hint != "" else ""
    )
    coord_arg_list = ", ".join([f"int coord{i}" for i in range(dim)])

    coord_indices = (
        [str(1 + i) for i in range(dim)] if cache_hint == "" else [str(2 + i) for i in range(dim)]
    )
    arg_template = "{%" + ", %".join(coord_indices) + "}"

    coord_list_constraints = ", ".join([f'"r"(coord{i})' for i in range(dim)])

    if cache_hint != "":
        cache_hint_str = ".L2::cache_hint"
    else:
        cache_hint_str = ""

    cache_hint_operand = ", %1" if cache_hint != "" else ""
    cache_hint_value = f', "n"({CacheHint[cache_hint]})' if cache_hint != "" else ""

    source_code = f"""
__forceinline__ __device__ void {func_name}(const CUtensorMap& tensormap, {coord_arg_list}) {{
  unsigned int src_addr = __cvta_generic_to_shared(src);
  uint64_t tensormap_addr = reinterpret_cast<uint64_t>(&tensormap);
  __asm__ __volatile__(
    "cp.async.bulk.prefetch.tensor.{dim}d.L2.global.tile{cache_hint_str}"
    " [%0, {arg_template}]{cache_hint_operand};"
    :
    : "l"(tensormap_addr){cache_hint_value},
      {coord_list_constraints}
    : "memory"
  );
}}
"""
    return cuda_func_call(func_name, tensormap, *coords, source_code=source_code)


@register_codegen("ptx_cp_async_bulk_tensor_shared_to_global_reduce")
def codegen_ptx_cp_async_bulk_tensor_shared_to_global_reduce(dim, src_ptr, tensormap, *args):
    dim = int(dim)
    cache_hint, red_op = args[0], args[1]
    coords = args[2:]
    if len(coords) != dim:
        raise ValueError(
            f"Number of coordinate expressions ({len(coords)}) does not match dimension ({dim})."
        )
    if cache_hint != "":
        cache_hint = parse_str(cache_hint)
    red_op = parse_str(red_op)

    func_name = f"ptx_cp_async_bulk_tensor_shared_to_global_reduce_{dim}d" + (
        f"_{cache_hint}" if cache_hint != "" else ""
    )
    coord_arg_list = ", ".join([f"int coord{i}" for i in range(dim)])

    coord_indices = (
        [str(2 + i) for i in range(dim)] if cache_hint == "" else [str(3 + i) for i in range(dim)]
    )
    arg_template = "{%" + ", %".join(coord_indices) + "}"

    coord_list_constraints = ", ".join([f'"r"(coord{i})' for i in range(dim)])

    if cache_hint != "":
        cache_hint_str = ".L2::cache_hint"
    else:
        cache_hint_str = ""

    cache_hint_operand = ", %2" if cache_hint != "" else ""
    cache_hint_value = f', "n"({CacheHint[cache_hint]})' if cache_hint != "" else ""
    source_code = f"""
__forceinline__ __device__ void {func_name}(void* src, const CUtensorMap& tensormap, {coord_arg_list}) {{
  unsigned int src_addr = __cvta_generic_to_shared(src);
  uint64_t tensormap_addr = reinterpret_cast<uint64_t>(&tensormap);
  __asm__ __volatile__(
    "cp.reduce.async.bulk.tensor.{dim}d.global.shared::cta.{red_op}.tile.bulk_group{cache_hint_str}"
    " [%0, {arg_template}], [%1]{cache_hint_operand};"
    :
    : "l"(tensormap_addr), "r"(src_addr){cache_hint_value},
      {coord_list_constraints}
    : "memory"
  );
}}
"""  # noqa: E501
    return cuda_func_call(func_name, src_ptr, tensormap, *coords, source_code=source_code)


#################### Non-TMA bulk copy (shared::cta -> shared::cluster)


@register_codegen("ptx_cp_async_bulk_shared_to_cluster")
def codegen_ptx_cp_async_bulk_shared_to_cluster(dst_ptr, src_ptr, size, mbar):
    func_name = "ptx_cp_async_bulk_shared_to_cluster"
    # dst_ptr and mbar are shared::cluster addresses (uint64) from mapa instruction.
    # src_ptr is a generic shared::cta pointer that needs __cvta_generic_to_shared.
    source_code = f"""
__forceinline__ __device__ void {func_name}(uint64_t dst, void* src, int size, uint64_t mbar) {{
  unsigned int dst_addr = static_cast<unsigned int>(dst);
  unsigned int src_addr = __cvta_generic_to_shared(src);
  unsigned int mbar_addr = static_cast<unsigned int>(mbar);
  __asm__ __volatile__(
    "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes"
    " [%0], [%1], %2, [%3];\\n"
    :
    : "r"(dst_addr), "r"(src_addr), "r"(size), "r"(mbar_addr)
    : "memory"
  );
}}
"""
    return cuda_func_call(func_name, dst_ptr, src_ptr, size, mbar, source_code=source_code)


#################### Commit/Wait groups


@register_codegen("ptx_cp_async_bulk_commit_group")
def codegen_ptx_cp_async_bulk_tensor_commit_group():
    func_name = "ptx_cp_async_bulk_tensor_commit_group"
    source_code = f"""
__forceinline__ __device__ void {func_name}() {{
  asm volatile("cp.async.bulk.commit_group;");
}}
"""
    return cuda_func_call(func_name, source_code=source_code)


@register_codegen("ptx_cp_async_bulk_wait_group")
def codegen_ptx_cp_async_bulk_wait_group(n, read):
    n = int(n)
    read_str = ".read" if bool(read) else ""
    func_name = "ptx_cp_async_bulk_wait_group" + ("_read" if bool(read) else "") + f"_{n}"
    source_code = f"""
__forceinline__ __device__ void {func_name}() {{
  asm volatile("cp.async.bulk.wait_group{read_str} %0;" :: "n"({n}): "memory");
}}
"""
    return cuda_func_call(func_name, source_code=source_code)
