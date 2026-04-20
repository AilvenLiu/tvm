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
# pylint: disable=too-many-instance-attributes, too-few-public-methods, invalid-name
"""Declarative schema framework for TIRx PTX intrinsic codegen.

Writing `@register_codegen` functions by hand mixes validation, helper-name
generation, C-signature construction, asm template rendering, and operand
constraint wiring into one imperative block. This framework separates those
concerns into a data declaration.

A minimal declaration:

    ptx_intrinsic(
        op_name="ptx_tcgen05_fence_before_thread_sync",
        attrs=(),
        operands=(),
        ptx_template="tcgen05.fence::before_thread_sync;",
        helper_name="ptx_tcgen05_fence_before_thread_sync",
    )

The declaration registers a codegen function under `op_name`. The Python
wrapper for users stays in `tvm.tirx.op` for now (unchanged); this module
only replaces the backend codegen side.

Template substitution language (intentionally small):

    {attr}       -> str(attr_value)
    {attr|safe}  -> str(attr_value) sanitized for identifiers (::→_, .→_)
    {.attr?}     -> attr.ptx_suffix if attr truthy, else ""  (Bool/Choice only)
    {%N}         -> asm positional placeholder (kept verbatim, N is int)
    {regs:op}    -> asm placeholder list for a reg_array operand (e.g. "%1, %2")

Helper-name tokens (helper_name_template):

    {attr}         -> str(value)
    {attr|safe}    -> sanitized
    {_attr?}       -> "_<value>" if truthy else ""
    {_attr?suffix} -> "_suffix" if truthy else ""

For ops whose codegen is too irregular (ptx_mma's register-count arithmetic,
ptx_cp_async_bulk_tensor_*'s conditional operand structure) the schema has
an `overrides` hook — the author can provide a custom `build_source` callable
that receives the resolved attrs + operand slots and returns the source-code
string. This keeps the schema declarative for 70% of ops without forcing a
universal framework.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from tvm.tirx.op import cuda_func_call
from tvm.tirx.operator.intrinsics.cuda.registry import register_codegen

# ---------------------------------------------------------------------------
# Attr kinds.
# ---------------------------------------------------------------------------


_MISSING = object()


@dataclass
class Attr:
    """Base class for compile-time attributes."""

    name: str
    default: Any = _MISSING

    @property
    def has_default(self) -> bool:
        return self.default is not _MISSING


@dataclass
class Choice(Attr):
    """An attr constrained to a finite set of values."""

    choices: tuple = ()
    ptx_suffix: str | None = None  # format e.g. ".{value}" or ".cta_group::{value}"

    def render_ptx_suffix(self, value) -> str:
        if self.ptx_suffix is None:
            return ""
        if value == "" or value is None:
            return ""
        return self.ptx_suffix.format(value=value)


@dataclass
class Bool(Attr):
    """A boolean attr, optionally with a PTX suffix when True."""

    default: bool = False
    ptx_suffix: str | None = None


@dataclass
class IntAttr(Attr):
    """An integer attr."""

    choices: tuple | None = None
    ptx_format: str | None = None  # e.g. "{value}" or "::{value}"


@dataclass
class Dtype(Attr):
    """A dtype string attr."""


@dataclass
class Derived:
    """Attr value computed from other attrs at codegen time."""

    name: str
    from_: Callable[[Any], Any]


# ---------------------------------------------------------------------------
# Operand kinds.
# ---------------------------------------------------------------------------


@dataclass
class Operand:
    """A single runtime operand."""

    name: str
    c_type: str = "void*"
    asm_constraint: str = "r"  # "r", "l", "f", "=r", ...

    # For pointer-to-register-array operands (e.g. ldmatrix's local_ptr):
    reg_array: bool = False
    count_from: str | None = None  # attr/derived name
    reg_elem_c_type: str = "uint32_t"

    # For TMEM address operands that need get_tmem_addr(base, row, col):
    tmem_addr: bool = False

    # For generic pointers that must be converted to a shared-memory address via
    # __cvta_generic_to_shared(ptr) before use in the asm. Emits a prelude
    # "unsigned int <name>_addr = __cvta_generic_to_shared(<name>);" and feeds
    # the converted value into the asm input as "r"(<name>_addr).
    cvta_to_shared: bool = False


@dataclass
class VariadicOperand:
    """Tail variadic — N distinct runtime values."""

    name: str
    c_type: str = "uint32_t"
    asm_constraint: str = "r"
    count_from: str | None = None


@dataclass
class Return:
    """Declare that the helper has a non-void return value produced by the asm.

    The framework emits a local variable of `c_type` named `var_name`, uses it
    as an asm output operand with `asm_constraint` (e.g. ``"=f"`` for float,
    ``"=r"`` for uint), and returns it at the end. The output placeholder
    occupies asm slot %0; subsequent operands get %1, %2, ...
    """

    c_type: str  # e.g. "float", "uint32_t"
    asm_constraint: str = "=f"  # "=f", "=r", "=l"
    var_name: str = "result"


# ---------------------------------------------------------------------------
# Schema.
# ---------------------------------------------------------------------------


@dataclass
class IntrinsicSchema:
    op_name: str
    operands: tuple
    attrs: tuple
    derived: tuple
    ptx_template: str
    helper_name: str | None  # literal name (no template)
    helper_name_template: str | None
    verifier: Callable | None
    build_source: Callable | None  # optional escape hatch
    extra_deps: tuple
    python_doc: str
    returns: Return | None  # non-void return; None means void helper

    attr_by_name: dict = field(default_factory=dict)
    derived_by_name: dict = field(default_factory=dict)

    def __post_init__(self):
        self.attr_by_name = {a.name: a for a in self.attrs}
        self.derived_by_name = {d.name: d for d in self.derived}


# ---------------------------------------------------------------------------
# Attr resolution.
# ---------------------------------------------------------------------------


class _AttrsNS:
    """Read-only namespace for attrs + derived values."""

    def __init__(self, values: dict):
        object.__setattr__(self, "_values", values)

    def __getattr__(self, name):
        values = object.__getattribute__(self, "_values")
        if name in values:
            return values[name]
        raise AttributeError(name)

    def __setattr__(self, name, value):
        raise AttributeError("_AttrsNS is read-only")

    def __contains__(self, name):
        return name in self._values

    def as_dict(self) -> dict:
        return dict(self._values)


def _coerce_attr(attr: Attr, value: Any):
    """Coerce a raw TIR arg value to the appropriate Python type."""
    if isinstance(attr, Bool):
        return bool(int(value)) if hasattr(value, "value") else bool(value)
    if isinstance(attr, IntAttr):
        return int(value)
    if isinstance(attr, Choice | Dtype):
        if value is None:
            return value
        # StringImm prints as '"foo"' (with quotes); raw str is 'foo'. parse_str()
        # strips the outer quotes on StringImm. Detect by checking if the repr
        # already starts with quote or not.
        s = str(value)
        if s.startswith('"') and s.endswith('"') and len(s) >= 2:
            return s[1:-1]
        return s
    return value


def _resolve_attrs(schema: IntrinsicSchema, raw_values: dict) -> _AttrsNS:
    values = {}
    for attr in schema.attrs:
        if attr.name in raw_values:
            v = _coerce_attr(attr, raw_values[attr.name])
        elif attr.has_default:
            v = attr.default
        else:
            raise ValueError(f"{schema.op_name}: missing attr '{attr.name}'")

        if isinstance(attr, Choice) and v not in attr.choices:
            raise ValueError(
                f"{schema.op_name}: invalid {attr.name}={v!r}; expected one of {attr.choices}"
            )
        if isinstance(attr, IntAttr) and attr.choices is not None and v not in attr.choices:
            raise ValueError(
                f"{schema.op_name}: invalid {attr.name}={v!r}; expected one of {attr.choices}"
            )
        values[attr.name] = v

    ns = _AttrsNS(values)
    for d in schema.derived:
        values[d.name] = d.from_(ns)

    if schema.verifier is not None:
        schema.verifier(ns)

    return _AttrsNS(values)


# ---------------------------------------------------------------------------
# Template rendering.
# ---------------------------------------------------------------------------


_PTX_TOKEN = re.compile(r"\{([^{}]+?)\}")
_NAME_TOKEN = re.compile(r"\{([^{}]+?)\}")


def _render_ptx(
    template: str, schema: IntrinsicSchema, ns: _AttrsNS, operand_slots: dict, reg_arrays: dict
) -> str:
    def sub(match):
        tok = match.group(1)

        # {.attr?} — optional suffix for Bool/Choice
        if tok.startswith(".") and tok.endswith("?"):
            name = tok[1:-1]
            attr = schema.attr_by_name.get(name)
            if attr is None:
                return ""
            v = getattr(ns, name, None)
            if isinstance(attr, Bool):
                return attr.ptx_suffix if v else ""
            if isinstance(attr, Choice):
                return attr.render_ptx_suffix(v)
            return ""

        # {regs:opname} — reg-array placeholder list
        if tok.startswith("regs:"):
            op_name = tok[len("regs:") :]
            return reg_arrays.get(op_name, "")

        # {name|safe}
        if "|" in tok:
            name, fn = tok.split("|", 1)
            v = getattr(ns, name, None)
            if v is None:
                v = operand_slots.get(name)
            if v is None:
                return ""
            if fn == "safe":
                return str(v).replace("::", "_").replace(".", "_")
            return str(v)

        # {op_name} — asm placeholder for an operand
        if tok in operand_slots:
            return f"%{operand_slots[tok]}"

        # Plain attr/derived
        v = getattr(ns, tok, None)
        if v is None:
            return ""
        # If it's a Choice with a ptx_suffix, render via suffix
        attr = schema.attr_by_name.get(tok)
        if isinstance(attr, Choice) and attr.ptx_suffix is not None:
            return attr.render_ptx_suffix(v)
        return str(v)

    return _PTX_TOKEN.sub(sub, template)


def _render_helper_name(template: str, ns: _AttrsNS) -> str:
    def sub(match):
        tok = match.group(1)
        # {_attr?suffix} or {_attr?t:f} or {_attr?}
        if tok.startswith("_") and "?" in tok:
            inner = tok[1:]
            name, _, body = inner.partition("?")
            v = getattr(ns, name, None)
            if ":" in body:
                t, _, f = body.partition(":")
                return f"_{t}" if v else f"_{f}"
            if body:
                return f"_{body}" if v else ""
            return f"_{v}" if v else ""
        if "|" in tok:
            name, fn = tok.split("|", 1)
            v = getattr(ns, name, None)
            if v is None:
                return ""
            if fn == "safe":
                return str(v).replace("::", "_").replace(".", "_")
            return str(v)
        v = getattr(ns, tok, None)
        return str(v) if v is not None else ""

    return _NAME_TOKEN.sub(sub, template)


# ---------------------------------------------------------------------------
# Codegen function builder.
# ---------------------------------------------------------------------------


def _arity(schema: IntrinsicSchema) -> int:
    """Number of positional IR args before the variadic tail."""
    n = 0
    for o in schema.operands:
        if isinstance(o, VariadicOperand):
            break
        n += 3 if o.tmem_addr else 1
    n += len(schema.attrs)
    return n


def _build_codegen(schema: IntrinsicSchema):
    """Register a codegen function for this schema."""
    has_variadic = any(isinstance(o, VariadicOperand) for o in schema.operands)

    def codegen(*args):
        # Parse args: operands (with tmem_addr expanding to 3), attrs, variadic.
        op_values = {}
        idx = 0
        for o in schema.operands:
            if isinstance(o, VariadicOperand):
                break
            if o.tmem_addr:
                op_values[o.name] = (args[idx], args[idx + 1], args[idx + 2])
                idx += 3
            else:
                op_values[o.name] = args[idx]
                idx += 1

        attr_raw = {}
        for a in schema.attrs:
            attr_raw[a.name] = args[idx]
            idx += 1

        variadic_values = list(args[idx:]) if has_variadic else []
        for o in schema.operands:
            if isinstance(o, VariadicOperand):
                op_values[o.name] = variadic_values
                break

        ns = _resolve_attrs(schema, attr_raw)

        # If user provided a build_source escape hatch, defer to it.
        if schema.build_source is not None:
            return schema.build_source(ns, op_values, schema)

        # Otherwise: auto-build source using template.
        operand_slots = {}
        reg_arrays = {}
        c_sig_parts = []
        asm_outputs = []
        asm_inputs = []
        helper_call_args = []
        c_prelude_lines = []

        slot = 0
        uses_tmem_addr = False

        # If schema declares a return value, the output occupies asm slot %0.
        ret = schema.returns
        if ret is not None:
            asm_outputs.append(f'"{ret.asm_constraint}"({ret.var_name})')
            slot = 1  # operand slots start after the output

        for o in schema.operands:
            if isinstance(o, VariadicOperand):
                continue
            if o.reg_array:
                cnt = getattr(ns, o.count_from)
                c_sig_parts.append(f"void* {o.name}")
                placeholders = ", ".join(f"%{slot + i}" for i in range(cnt))
                reg_arrays[o.name] = placeholders
                constraint = o.asm_constraint
                target = asm_outputs if constraint.startswith("=") else asm_inputs
                for i in range(cnt):
                    target.append(f'"{constraint}"(*(({o.reg_elem_c_type}*){o.name} + {i}))')
                helper_call_args.append(op_values[o.name])
                slot += cnt
            elif o.tmem_addr:
                uses_tmem_addr = True
                c_sig_parts.append(f"uint32_t {o.name}")
                c_sig_parts.append(f"int {o.name}_row")
                c_sig_parts.append(f"int {o.name}_col")
                operand_slots[o.name] = slot
                asm_inputs.append(f'"r"(get_tmem_addr({o.name}, {o.name}_row, {o.name}_col))')
                base, row, col = op_values[o.name]
                helper_call_args.extend([base, row, col])
                slot += 1
            elif o.cvta_to_shared:
                # Emit a prelude converting the generic pointer to SMEM address,
                # then pass the SMEM uint32 address into asm.
                c_sig_parts.append(f"{o.c_type} {o.name}")
                c_prelude_lines.append(
                    f"    unsigned int {o.name}_addr = __cvta_generic_to_shared({o.name});"
                )
                operand_slots[o.name] = slot
                asm_inputs.append(f'"r"({o.name}_addr)')
                helper_call_args.append(op_values[o.name])
                slot += 1
            else:
                c_sig_parts.append(f"{o.c_type} {o.name}")
                operand_slots[o.name] = slot
                asm_inputs.append(f'"{o.asm_constraint}"({o.name})')
                helper_call_args.append(op_values[o.name])
                slot += 1

        for o in schema.operands:
            if isinstance(o, VariadicOperand):
                vals = op_values[o.name]
                for i, v in enumerate(vals):
                    arg_name = f"{o.name}{i}"
                    c_sig_parts.append(f"{o.c_type} {arg_name}")
                    asm_inputs.append(f'"{o.asm_constraint}"({arg_name})')
                    helper_call_args.append(v)
                    slot += 1

        ptx_line = _render_ptx(schema.ptx_template, schema, ns, operand_slots, reg_arrays)

        if schema.helper_name is not None:
            helper_name = schema.helper_name
        else:
            helper_name = _render_helper_name(schema.helper_name_template, ns)

        sig = ", ".join(c_sig_parts)
        out_str = ", ".join(asm_outputs)
        in_str = ", ".join(asm_inputs)

        # Match the exact whitespace conventions used by the old hand-written
        # codegens so that substring assertions in tests still match.
        out_part = f" {out_str} " if out_str else " "
        in_part = f" {in_str} " if in_str else " "
        prelude = ("\n" + "\n".join(c_prelude_lines)) if c_prelude_lines else ""

        # Build C helper body: void vs typed return.
        if ret is not None:
            ret_decl = f"    {ret.c_type} {ret.var_name};"
            ret_stmt = f"    return {ret.var_name};"
            return_c_type = ret.c_type
            prelude = f"\n{ret_decl}" + prelude
            tail = f"\n{ret_stmt}\n"
        else:
            return_c_type = "void"
            tail = "\n"

        if c_sig_parts or ret is not None:
            sig_effective = sig if c_sig_parts else ""
            source_code = f"""
__forceinline__ __device__ {return_c_type} {helper_name}({sig_effective}) {{{prelude}
    asm volatile("{ptx_line}" :{out_part}:{in_part}: "memory");{tail}}}
"""
        else:
            # Nullary void.
            source_code = f"""
__forceinline__ __device__ void {helper_name}() {{
    asm volatile("{ptx_line}" ::: "memory");
}}
"""

        deps = list(schema.extra_deps)
        if uses_tmem_addr and "get_tmem_addr" not in deps:
            deps.append("get_tmem_addr")

        cfc_kwargs = {"source_code": source_code}
        if ret is not None:
            # cuda_func_call's return_type expects a TVM dtype string (e.g.
            # "float32", "uint32"), not the C type — map common cases.
            _c_to_tvm_dtype = {
                "float": "float32",
                "double": "float64",
                "uint32_t": "uint32",
                "int32_t": "int32",
                "uint64_t": "uint64",
                "int64_t": "int64",
                "bool": "bool",
            }
            cfc_kwargs["return_type"] = _c_to_tvm_dtype.get(ret.c_type, ret.c_type)
        result = cuda_func_call(helper_name, *helper_call_args, **cfc_kwargs)
        if deps:
            return result, deps
        return result

    codegen.__name__ = f"codegen_{schema.op_name}"
    register_codegen(schema.op_name)(codegen)
    return codegen


# ---------------------------------------------------------------------------
# Public declaration API.
# ---------------------------------------------------------------------------


def ptx_intrinsic(
    *,
    op_name: str,
    operands: tuple | list = (),
    attrs: tuple | list = (),
    derived: tuple | list = (),
    ptx_template: str = "",
    helper_name: str | None = None,
    helper_name_template: str | None = None,
    verifier: Callable | None = None,
    build_source: Callable | None = None,
    extra_deps: tuple = (),
    python_doc: str = "",
    returns: Return | None = None,
) -> IntrinsicSchema:
    """Register a PTX intrinsic codegen from a declarative schema.

    Exactly one of `helper_name` (literal) or `helper_name_template` must be set
    unless `build_source` is provided (in which case both are ignored).

    See module docstring for template language and examples. Ops whose codegen
    can't be expressed as a schema use ``@register_codegen(op_name)`` directly
    on a hand-written function in the same family module.
    """
    if build_source is None:
        if helper_name is None and helper_name_template is None:
            raise ValueError(
                f"{op_name}: must provide helper_name or helper_name_template "
                f"(or build_source for full override)"
            )

    schema = IntrinsicSchema(
        op_name=op_name,
        operands=tuple(operands),
        attrs=tuple(attrs),
        derived=tuple(derived),
        ptx_template=ptx_template,
        helper_name=helper_name,
        helper_name_template=helper_name_template,
        verifier=verifier,
        build_source=build_source,
        extra_deps=tuple(extra_deps),
        python_doc=python_doc,
        returns=returns,
    )
    _build_codegen(schema)
    return schema


# ---------------------------------------------------------------------------
# cuda_helper_intrinsic — schema for non-asm CUDA helper intrinsics.
# ---------------------------------------------------------------------------
#
# These are ops whose "body" is a plain C/C++ statement or helper call rather
# than inline PTX: ``__syncthreads()``, ``__any_sync(...)``, ``atomicAdd(...)``,
# ``__nanosleep(t)``, NVSHMEM RMA calls, etc. No asm, no constraints to wire,
# no helper-name suffixing driven by instruction modifiers.
#
# The common emitted shape is::
#
#     [template <typename T>]
#     __forceinline__ __device__ <ret> <name>(<params>) { <body> }
#
# A declaration names the C signature and body template directly. Operands are
# passed positionally into ``cuda_func_call``; runtime return type (for TVM
# side) can be a literal dtype string or a callable that derives it from the
# operand values (e.g. ``atomic_add`` mirrors the value operand's dtype).
#
# Ops whose dispatch is too gnarly for this shape (templated printf, runtime
# dtype switching with different bodies) use ``@register_codegen(op_name)``
# directly on a hand-written function in the same family module.


def cuda_helper_intrinsic(
    *,
    op_name: str,
    helper_name: str | None = None,
    c_signature: str = "()",
    c_body: str = "",
    return_type: str = "void",
    tvm_return_type: str | Callable | None = None,
    templated: bool = False,
    extra_deps: tuple = (),
    python_doc: str = "",
) -> None:
    """Declarative schema for CUDA helper (non-asm) intrinsics.

    Parameters
    ----------
    op_name : str
        Registry key (e.g. ``"cuda_warp_sync"``).
    helper_name : str, optional
        C function name emitted into device source. Defaults to ``op_name``
        prefixed with ``tvm_builtin_``.
    c_signature : str
        C parameter list *including* the outer parens, e.g. ``"(int predicate)"``.
    c_body : str
        Body text placed between the generated ``{`` and ``}``. No indentation
        normalization — write it how you want it to appear. May contain
        ``{helper_name}`` as a placeholder.
    return_type : str
        C return type for the helper signature. Default ``"void"``.
    tvm_return_type : str | callable, optional
        TVM dtype for the intrinsic result (for non-void helpers). A callable
        receives ``(*args)`` — same args as codegen — and returns a dtype str,
        useful when the return type mirrors an operand's dtype.
    templated : bool
        If True, prefix emitted source with ``template <typename T>``.
    extra_deps : tuple
        Helper tags forwarded as the second element of the codegen return.
    """
    if helper_name is None:
        helper_name = f"tvm_builtin_{op_name}"

    tpl_prefix = "template <typename T>\n" if templated else ""
    body_fmt = c_body.format(helper_name=helper_name) if "{helper_name}" in c_body else c_body
    source_code = (
        f"\n{tpl_prefix}__forceinline__ __device__ {return_type} "
        f"{helper_name}{c_signature} {{\n    {body_fmt}\n}}\n"
    )
    deps_list = list(extra_deps)

    def _codegen(*args):
        kwargs = {"source_code": source_code}
        if tvm_return_type is not None:
            rt = tvm_return_type(*args) if callable(tvm_return_type) else tvm_return_type
            kwargs["return_type"] = rt
        result = cuda_func_call(helper_name, *args, **kwargs)
        return (result, deps_list) if deps_list else result

    _codegen.__name__ = f"codegen_{op_name}"
    register_codegen(op_name)(_codegen)
