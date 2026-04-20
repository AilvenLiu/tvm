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
"""Miscellaneous PTX intrinsics.

Schema-expressible ops (setmaxnreg, return-value math, map_shared_rank) at the
top. Ops with runtime dtype dispatch or `#if __CUDA_ARCH__` predicates are
hand-written below.
"""

from tvm.tirx.op import cuda_func_call

from .._schema import Bool, Derived, IntAttr, Operand, Return, ptx_intrinsic
from .registry import register_codegen
from .utils import parse_str

# =============================================================================
# Schema-declared ops.
# =============================================================================

# -----------------------------------------------------------------------------
# setmaxnreg — warp-wide max register hint.
# Python API:
#     Tx.ptx.setmaxnreg(inc, nreg)
#         inc=True  → setmaxnreg.inc.sync.aligned.u32 <nreg>
#         inc=False → setmaxnreg.dec.sync.aligned.u32 <nreg>
# nreg is a compile-time immediate; each (inc, nreg) pair gets its own helper.
# -----------------------------------------------------------------------------
ptx_intrinsic(
    op_name="ptx_setmaxnreg",
    attrs=[Bool("inc"), IntAttr("nreg")],
    derived=[Derived("action", from_=lambda a: "inc" if a.inc else "dec")],
    ptx_template="setmaxnreg.{action}.sync.aligned.u32 {nreg};",
    helper_name_template="tvm_builtin_ptx_setmaxnreg_{action}_{nreg}",
)


# -----------------------------------------------------------------------------
# Return-value PTX math ops (sm_100a+).
# Python API:
#     Tx.ptx.exp2(x)                          -> float
#     Tx.ptx.rcp(x)                           -> float
#     Tx.ptx.reduce3_max_f32(a, b, c)         -> float
#     Tx.ptx.reduce3_min_f32(a, b, c)         -> float
# -----------------------------------------------------------------------------
ptx_intrinsic(
    op_name="ptx_exp2",
    operands=[Operand("x", c_type="float", asm_constraint="f")],
    returns=Return(c_type="float", asm_constraint="=f"),
    ptx_template="ex2.approx.ftz.f32 %0, %1;",
    helper_name="tvm_builtin_ptx_exp2",
)
ptx_intrinsic(
    op_name="ptx_rcp",
    operands=[Operand("x", c_type="float", asm_constraint="f")],
    returns=Return(c_type="float", asm_constraint="=f"),
    ptx_template="rcp.approx.ftz.f32 %0, %1;",
    helper_name="tvm_builtin_ptx_rcp",
)
ptx_intrinsic(
    op_name="ptx_reduce3_max_f32",
    operands=[
        Operand("a", c_type="float", asm_constraint="f"),
        Operand("b", c_type="float", asm_constraint="f"),
        Operand("c", c_type="float", asm_constraint="f"),
    ],
    returns=Return(c_type="float", asm_constraint="=f"),
    ptx_template="max.f32 %0, %1, %2, %3;",
    helper_name="tvm_builtin_ptx_reduce3_max_f32",
)
ptx_intrinsic(
    op_name="ptx_reduce3_min_f32",
    operands=[
        Operand("a", c_type="float", asm_constraint="f"),
        Operand("b", c_type="float", asm_constraint="f"),
        Operand("c", c_type="float", asm_constraint="f"),
    ],
    returns=Return(c_type="float", asm_constraint="=f"),
    ptx_template="min.f32 %0, %1, %2, %3;",
    helper_name="tvm_builtin_ptx_reduce3_min_f32",
)


# -----------------------------------------------------------------------------
# ptx_map_shared_rank — map generic SMEM pointer to rank's SMEM.
# Python API:
#     Tx.ptx.map_shared_rank(ptr, rank) -> uint64
# -----------------------------------------------------------------------------
ptx_intrinsic(
    op_name="ptx_map_shared_rank",
    operands=[
        Operand("addr", c_type="void*", asm_constraint="l"),
        Operand("rank", c_type="uint32_t", asm_constraint="r"),
    ],
    returns=Return(c_type="uint64_t", asm_constraint="=l"),
    ptx_template="mapa.u64 %0, %1, %2;",
    helper_name="tvm_builtin_ptx_map_shared_rank",
)


# =============================================================================
# Hand-written ops.
#
# Note: ptx_any_sync is a CUDA intrinsic (no inline asm); it lives here for
# historical naming reasons and should be relocated to the CUDA helper track
# (renamed to cuda_any_sync) in a follow-up.
# =============================================================================


_LD_GLOBAL_ACQUIRE_DTYPES = {
    "uint32": ("uint32_t", "b32", "r"),
    "int32": ("int32_t", "b32", "r"),
    "uint64": ("uint64_t", "b64", "l"),
    "int64": ("int64_t", "b64", "l"),
}


@register_codegen("ptx_ld_global_acquire")
def codegen_ptx_ld_global_acquire(res, addr):
    dtype = str(res.dtype)
    if dtype not in _LD_GLOBAL_ACQUIRE_DTYPES:
        raise ValueError(f"Unsupported data type for ld.global.acquire: {dtype}")
    dtype_str, type_str, specifier = _LD_GLOBAL_ACQUIRE_DTYPES[dtype]

    func_name = f"tvm_builtin_ptx_ld_global_acquire_{type_str}"
    source_code = f"""
__forceinline__ __device__ void {func_name}({dtype_str}& res,{dtype_str}* addr) {{
  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  asm volatile ("ld.global.acquire.gpu.{type_str} %0, [%1];\\n" : "={specifier}"(res) : "l"(addr));
  #else
  asm volatile ("ld.global.cg.{type_str} %0, [%1];\\n" : "={specifier}"(res) : "l"(addr));
  #endif
}}
"""
    return cuda_func_call(func_name, res, addr, source_code=source_code)


@register_codegen("ptx_fetch_register")
def codegen_ptx_fetch_register(bits, reg):
    bits = int(bits)
    reg = parse_str(reg)

    if bits not in [32, 64]:
        raise ValueError(f"Only support 32/64 bits for ptx_fetch_register, but got {bits}.")

    func_name_safe_reg = reg.replace(".", "_")

    func_name = f"tvm_builtin_ptx_fetch_register_{func_name_safe_reg}"
    source_code = f"""
__forceinline__ __device__ int{bits}_t {func_name}() {{
  uint{bits}_t x;
  asm volatile("mov.u{bits} %0, %{reg};\\n" : "=r"(x) : "r"(reg));
  return (int{bits}_t)x;
}}
"""
    return cuda_func_call(func_name, source_code=source_code, return_type=f"int{bits}")


@register_codegen("ptx_any_sync")
def codegen_ptx_any_sync(mask, pred):
    """Warp-wide any predicate using __any_sync intrinsic."""
    func_name = "tvm_builtin_ptx_any_sync"
    source_code = f"""
__forceinline__ __device__ int {func_name}(unsigned mask, int pred) {{
    return __any_sync(mask, pred);
}}
"""
    return cuda_func_call(func_name, mask, pred, source_code=source_code, return_type="int32")
