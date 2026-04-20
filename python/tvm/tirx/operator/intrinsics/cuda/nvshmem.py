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
"""NVSHMEM intrinsics.

Simple RMA / barrier / PE-query helpers are declared via
``cuda_helper_intrinsic(...)`` at the top. Ops with string attrs that map to
NVSHMEM integer constants (``sig_op`` ∈ {"set","add"}, ``cmp`` ∈ {"eq","ne",...})
are hand-written below.
"""

from tvm.tirx.op import cuda_func_call

from .._schema import cuda_helper_intrinsic
from .registry import register_codegen

_NVSHMEM = ("nvshmem",)

# =============================================================================
# Schema-declared helpers.
# =============================================================================

# -----------------------------------------------------------------------------
# No-arg helpers: PE queries, quiet, fence, barrier_all.
# -----------------------------------------------------------------------------
for _op, _call, _ret, _tvm_ret in [
    ("nvshmem_my_pe", "nvshmem_my_pe", "int32_t", "int32"),
    ("nvshmem_n_pes", "nvshmem_n_pes", "int32_t", "int32"),
    ("nvshmem_quiet", "nvshmem_quiet", "void", None),
    ("nvshmem_fence", "nvshmem_fence", "void", None),
    ("nvshmem_barrier_all", "nvshmem_barrier_all", "void", None),
]:
    cuda_helper_intrinsic(
        op_name=_op,
        c_body=(f"return {_call}();" if _ret != "void" else f"{_call}();"),
        return_type=_ret,
        tvm_return_type=_tvm_ret,
        extra_deps=_NVSHMEM,
    )
del _op, _call, _ret, _tvm_ret


# -----------------------------------------------------------------------------
# RMA get/put (thread/warp/block): all share the same 4-arg signature.
# -----------------------------------------------------------------------------
_RMA_SIG = "(void *dest, const void *source, size_t nelems, int pe)"
for _op, _backend_call in [
    ("nvshmem_getmem_nbi", "nvshmem_getmem_nbi"),
    ("nvshmem_putmem_nbi", "nvshmem_putmem_nbi"),
    ("nvshmem_getmem_nbi_warp", "nvshmemx_getmem_nbi_warp"),
    ("nvshmem_putmem_nbi_warp", "nvshmemx_putmem_nbi_warp"),
    ("nvshmem_getmem_nbi_block", "nvshmemx_getmem_nbi_block"),
    ("nvshmem_putmem_nbi_block", "nvshmemx_putmem_nbi_block"),
]:
    cuda_helper_intrinsic(
        op_name=_op,
        c_signature=_RMA_SIG,
        c_body=f"{_backend_call}(dest, source, nelems, pe);",
        extra_deps=_NVSHMEM,
    )
del _op, _backend_call


# =============================================================================
# Hand-written helpers — string-attr → NVSHMEM integer-constant mapping.
# =============================================================================


_SIG_OP_VAL = {"set": 0, "add": 1}
_CMP_VAL = {"eq": 0, "ne": 1, "gt": 2, "ge": 3, "lt": 4, "le": 5}


def _resolve_sig_op(sig_op) -> int:
    s = sig_op if isinstance(sig_op, str) else sig_op.value
    if s not in _SIG_OP_VAL:
        raise ValueError(f"Unsupported signal op: {s}")
    return _SIG_OP_VAL[s]


def _resolve_cmp(cmp) -> int:
    s = cmp if isinstance(cmp, str) else cmp.value
    if s not in _CMP_VAL:
        raise ValueError(f"Unsupported cmp operation: {s}")
    return _CMP_VAL[s]


@register_codegen("nvshmem_signal_op")
def codegen_nvshmem_signal_op(sig_addr, signal, sig_op, pe):
    func_name = "tvm_builtin_nvshmem_signal_op"
    source_code = f"""
__forceinline__ __device__ void {func_name}(uint64_t *sig_addr, uint64_t signal, int sig_op, int pe) {{
    nvshmemx_signal_op(sig_addr, signal, sig_op, pe);
}}
"""  # noqa: E501
    return cuda_func_call(
        func_name, sig_addr, signal, _resolve_sig_op(sig_op), pe, source_code=source_code
    ), ["nvshmem"]


@register_codegen("nvshmem_wait_until")
def codegen_nvshmem_wait_until(ivar, cmp, cmp_value, type):
    type_str = type if isinstance(type, str) else type.value
    type_map = {"uint64_t": ("uint64_t", "uint64"), "uint64": ("uint64_t", "uint64")}
    if type_str not in type_map:
        raise ValueError(f"Unsupported type for nvshmem_wait_until: {type_str}")
    c_type, type_suffix = type_map[type_str]

    func_name = f"tvm_builtin_nvshmem_{type_suffix}_wait_until"
    source_code = f"""
__forceinline__ __device__ void {func_name}({c_type} *ivar, int cmp, {c_type} cmp_value) {{
    nvshmem_{type_suffix}_wait_until(ivar, cmp, cmp_value);
}}
"""
    return cuda_func_call(func_name, ivar, _resolve_cmp(cmp), cmp_value, source_code=source_code), [
        "nvshmem"
    ]


def _putmem_signal_helper(scope_suffix, backend_call):
    func_name = f"tvm_builtin_nvshmem_putmem_signal_nbi{scope_suffix}"
    source_code = f"""
__forceinline__ __device__ void {func_name}(void *dest, const void *source, size_t nelems, uint64_t *sig_addr, uint64_t signal, int sig_op, int pe) {{
    {backend_call}(dest, source, nelems, sig_addr, signal, sig_op, pe);
}}
"""  # noqa: E501
    return func_name, source_code


def _codegen_putmem_signal(
    scope_suffix, backend_call, dest, source, nelems, sig_addr, signal, sig_op, pe
):
    func_name, source_code = _putmem_signal_helper(scope_suffix, backend_call)
    return cuda_func_call(
        func_name,
        dest,
        source,
        nelems,
        sig_addr,
        signal,
        _resolve_sig_op(sig_op),
        pe,
        source_code=source_code,
    ), ["nvshmem"]


@register_codegen("nvshmem_putmem_signal_nbi")
def codegen_nvshmem_putmem_signal_nbi(dest, source, nelems, sig_addr, signal, sig_op, pe):
    return _codegen_putmem_signal(
        "", "nvshmem_putmem_signal_nbi", dest, source, nelems, sig_addr, signal, sig_op, pe
    )


@register_codegen("nvshmem_putmem_signal_nbi_warp")
def codegen_nvshmem_putmem_signal_nbi_warp(dest, source, nelems, sig_addr, signal, sig_op, pe):
    return _codegen_putmem_signal(
        "_warp",
        "nvshmemx_putmem_signal_nbi_warp",
        dest,
        source,
        nelems,
        sig_addr,
        signal,
        sig_op,
        pe,
    )


@register_codegen("nvshmem_putmem_signal_nbi_block")
def codegen_nvshmem_putmem_signal_nbi_block(dest, source, nelems, sig_addr, signal, sig_op, pe):
    return _codegen_putmem_signal(
        "_block",
        "nvshmemx_putmem_signal_nbi_block",
        dest,
        source,
        nelems,
        sig_addr,
        signal,
        sig_op,
        pe,
    )
