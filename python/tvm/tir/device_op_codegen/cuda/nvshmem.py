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
"""NVSHMEM related operations."""
from tvm.tir.op import cuda_func_call

from .registry import register_codegen


@register_codegen("nvshmem_my_pe")
def codegen_nvshmem_my_pe():
    func_name = "tvm_builtin_nvshmem_my_pe"
    source_code = R"""
__forceinline__ __device__ int32_t {func_name}() {{
    return nvshmem_my_pe();
}}
"""
    source_code = source_code.format(func_name=func_name)
    return cuda_func_call(func_name, source_code=source_code, return_type="int32"), ["nvshmem"]


@register_codegen("nvshmem_n_pes")
def codegen_nvshmem_n_pes():
    func_name = "tvm_builtin_nvshmem_n_pes"
    source_code = R"""
__forceinline__ __device__ int32_t {func_name}() {{
    return nvshmem_n_pes();
}}
"""
    source_code = source_code.format(func_name=func_name)
    return cuda_func_call(func_name, source_code=source_code, return_type="int32"), ["nvshmem"]


@register_codegen("nvshmem_getmem_nbi")
def codegen_nvshmem_getmem_nbi(dst, src, nelems, pe):
    func_name = "tvm_builtin_nvshmem_getmem_nbi"
    source_code = R"""
__forceinline__ __device__ void {func_name}(void *dest, const void *source, size_t nelems, int pe) {{
    nvshmem_getmem_nbi(dest, source, nelems, pe);
}}
"""
    source_code = source_code.format(func_name=func_name)
    return cuda_func_call(func_name, dst, src, nelems, pe, source_code=source_code), ["nvshmem"]


@register_codegen("nvshmem_putmem_nbi")
def codegen_nvshmem_putmem_nbi(dst, src, nelems, pe):
    func_name = "tvm_builtin_nvshmem_putmem_nbi"
    source_code = R"""
__forceinline__ __device__ void {func_name}(void *dest, const void *source, size_t nelems, int pe) {{
    nvshmem_putmem_nbi(dest, source, nelems, pe);
}}
"""
    source_code = source_code.format(func_name=func_name)
    return cuda_func_call(func_name, dst, src, nelems, pe, source_code=source_code), ["nvshmem"]


@register_codegen("nvshmem_getmem_nbi_warp")
def codegen_nvshmem_getmem_nbi_warp(dst, src, nelems, pe):
    func_name = "tvm_builtin_nvshmem_getmem_nbi_warp"
    source_code = R"""
__forceinline__ __device__ void {func_name}(void *dest, const void *source, size_t nelems, int pe) {{
    nvshmemx_getmem_nbi_warp(dest, source, nelems, pe);
}}
"""
    source_code = source_code.format(func_name=func_name)
    return cuda_func_call(func_name, dst, src, nelems, pe, source_code=source_code), ["nvshmem"]


@register_codegen("nvshmem_putmem_nbi_warp")
def codegen_nvshmem_putmem_nbi_warp(dst, src, nelems, pe):
    func_name = "tvm_builtin_nvshmem_putmem_nbi_warp"
    source_code = R"""
__forceinline__ __device__ void {func_name}(void *dest, const void *source, size_t nelems, int pe) {{
    nvshmemx_putmem_nbi_warp(dest, source, nelems, pe);
}}
"""
    source_code = source_code.format(func_name=func_name)
    return cuda_func_call(func_name, dst, src, nelems, pe, source_code=source_code), ["nvshmem"]


@register_codegen("nvshmem_getmem_nbi_block")
def codegen_nvshmem_getmem_nbi_block(dst, src, nelems, pe):
    func_name = "tvm_builtin_nvshmem_getmem_nbi_block"
    source_code = R"""
__forceinline__ __device__ void {func_name}(void *dest, const void *source, size_t nelems, int pe) {{
    nvshmemx_getmem_nbi_block(dest, source, nelems, pe);
}}
"""
    source_code = source_code.format(func_name=func_name)
    return cuda_func_call(func_name, dst, src, nelems, pe, source_code=source_code), ["nvshmem"]


@register_codegen("nvshmem_putmem_nbi_block")
def codegen_nvshmem_putmem_nbi_block(dst, src, nelems, pe):
    func_name = "tvm_builtin_nvshmem_putmem_nbi_block"
    source_code = R"""
__forceinline__ __device__ void {func_name}(void *dest, const void *source, size_t nelems, int pe) {{
    nvshmemx_putmem_nbi_block(dest, source, nelems, pe);
}}
"""
    source_code = source_code.format(func_name=func_name)
    return cuda_func_call(func_name, dst, src, nelems, pe, source_code=source_code), ["nvshmem"]


@register_codegen("nvshmem_signal_op")
def codegen_nvshmem_signal_op(sig_addr, signal, sig_op, pe):
    if not isinstance(sig_op, str):
        sig_op = sig_op.value

    func_name = "tvm_builtin_nvshmem_signal_op"
    source_code = R"""
__forceinline__ __device__ void {func_name}(uint64_t *sig_addr, uint64_t signal, int sig_op, int pe) {{
    nvshmemx_signal_op(sig_addr, signal, sig_op, pe);
}}
"""
    source_code = source_code.format(func_name=func_name)
    sig_op_val = {"set": 0, "add": 1}
    assert sig_op in sig_op_val, f"Unsupported signal operation in nvshmem_signal_op: {sig_op}"
    return cuda_func_call(
        func_name, sig_addr, signal, sig_op_val.get(sig_op), pe, source_code=source_code
    ), ["nvshmem"]


@register_codegen("nvshmem_wait_until")
def codegen_nvshmem_wait_until(ivar, cmp, cmp_value, type):
    if not isinstance(cmp, str):
        cmp = cmp.value
    if not isinstance(type, str):
        type = type.value
    type_val = {"uint64_t": ("uint64_t", "uint64"), "uint64": ("uint64_t", "uint64")}
    assert type in type_val, f"Unsupported type for nvshmem_wait_until: {type}"
    TYPE = type_val.get(type)[1]

    func_name = f"tvm_builtin_nvshmem_{TYPE}_wait_until"
    source_code = R"""
__forceinline__ __device__ void {func_name}({type} *ivar, int cmp, {type} cmp_value) {{
    nvshmem_{TYPE}_wait_until(ivar, cmp, cmp_value);
}}
"""
    source_code = source_code.format(func_name=func_name, type=type_val.get(type)[0], TYPE=TYPE)
    cmp_val = {"eq": 0, "ne": 1, "gt": 2, "ge": 3, "lt": 4, "le": 5}
    assert cmp in cmp_val, f"Unsupported cmp operation in nvshmem_wait_until: {cmp}"
    return cuda_func_call(func_name, ivar, cmp_val.get(cmp), cmp_value, source_code=source_code), [
        "nvshmem"
    ]


@register_codegen("nvshmem_quiet")
def codegen_nvshmem_quiet():
    func_name = "tvm_builtin_nvshmem_quiet"
    source_code = R"""
__forceinline__ __device__ void {func_name}() {{
    nvshmem_quiet();
}}
"""
    source_code = source_code.format(func_name=func_name)
    return cuda_func_call(func_name, source_code=source_code), ["nvshmem"]


@register_codegen("nvshmem_putmem_signal_nbi")
def codegen_nvshmem_putmem_signal_nbi(dest, source, nelems, sig_addr, signal, sig_op, pe):
    if not isinstance(sig_op, str):
        sig_op = sig_op.value

    func_name = "tvm_builtin_nvshmem_putmem_signal_nbi"
    source_code = R"""
__forceinline__ __device__ void {func_name}(void *dest, const void *source, size_t nelems, uint64_t *sig_addr, uint64_t signal, int sig_op, int pe) {{
    nvshmem_putmem_signal_nbi(dest, source, nelems, sig_addr, signal, sig_op, pe);
}}
"""
    source_code = source_code.format(func_name=func_name)
    sig_op_val = {"set": 0, "add": 1}
    assert sig_op in sig_op_val, f"Unsupported signal operation in nvshmem_signal_op: {sig_op}"
    return cuda_func_call(
        func_name,
        dest,
        source,
        nelems,
        sig_addr,
        signal,
        sig_op_val.get(sig_op),
        pe,
        source_code=source_code,
    ), ["nvshmem"]


@register_codegen("nvshmem_putmem_signal_nbi_warp")
def codegen_nvshmem_putmem_signal_nbi_warp(dest, source, nelems, sig_addr, signal, sig_op, pe):
    if not isinstance(sig_op, str):
        sig_op = sig_op.value

    func_name = "tvm_builtin_nvshmem_putmem_signal_nbi_warp"
    source_code = R"""
__forceinline__ __device__ void {func_name}(void *dest, const void *source, size_t nelems, uint64_t *sig_addr, uint64_t signal, int sig_op, int pe) {{
    nvshmemx_putmem_signal_nbi_warp(dest, source, nelems, sig_addr, signal, sig_op, pe);
}}
"""
    source_code = source_code.format(func_name=func_name)
    sig_op_val = {"set": 0, "add": 1}
    assert sig_op in sig_op_val, f"Unsupported signal operation in nvshmem_signal_op: {sig_op}"
    return cuda_func_call(
        func_name,
        dest,
        source,
        nelems,
        sig_addr,
        signal,
        sig_op_val.get(sig_op),
        pe,
        source_code=source_code,
    ), ["nvshmem"]


@register_codegen("nvshmem_putmem_signal_nbi_block")
def codegen_nvshmem_putmem_signal_nbi_block(dest, source, nelems, sig_addr, signal, sig_op, pe):
    if not isinstance(sig_op, str):
        sig_op = sig_op.value

    func_name = "tvm_builtin_nvshmem_putmem_signal_nbi_block"
    source_code = R"""
__forceinline__ __device__ void {func_name}(void *dest, const void *source, size_t nelems, uint64_t *sig_addr, uint64_t signal, int sig_op, int pe) {{
    nvshmemx_putmem_signal_nbi_block(dest, source, nelems, sig_addr, signal, sig_op, pe);
}}
"""
    source_code = source_code.format(func_name=func_name)
    sig_op_val = {"set": 0, "add": 1}
    assert sig_op in sig_op_val, f"Unsupported signal operation in nvshmem_signal_op: {sig_op}"
    return cuda_func_call(
        func_name,
        dest,
        source,
        nelems,
        sig_addr,
        signal,
        sig_op_val.get(sig_op),
        pe,
        source_code=source_code,
    ), ["nvshmem"]


@register_codegen("nvshmem_fence")
def codegen_nvshmem_fence():
    func_name = "tvm_builtin_nvshmem_fence"
    source_code = R"""
__forceinline__ __device__ void {func_name}() {{
    nvshmem_fence();
}}
"""
    source_code = source_code.format(func_name=func_name)
    return cuda_func_call(func_name, source_code=source_code), ["nvshmem"]


@register_codegen("nvshmem_barrier_all")
def codegen_nvshmem_barrier_all():
    func_name = "tvm_builtin_nvshmem_barrier_all"
    source_code = R"""
__forceinline__ __device__ void {func_name}() {{
    nvshmem_barrier_all();
}}
"""
    source_code = source_code.format(func_name=func_name)
    return cuda_func_call(func_name, source_code=source_code), ["nvshmem"]
