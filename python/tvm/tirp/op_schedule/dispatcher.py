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
"""Rich dispatcher for TIRp operator schedules.

This module adds a structured dispatch table with predicates and
debug/strict reporting.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Any

from tvm.ir import Op
from tvm.tir import PrimFunc
from tvm.tir.stmt import OpCall
from tvm.tirp.operator import get_tirp_op
from .schedule_context import ScheduleContext


class DispatchFail(RuntimeError):
    """Raised by variants or predicates to provide a reasoned failure."""


@dataclass
class Predicate:
    """A named predicate. The callable can return:

    - bool
    - (bool, str) where the second element is an optional reason on failure
    - raise DispatchFail(reason)
    """

    name: str
    fn: Callable[[OpCall, ScheduleContext], Any]

    def evaluate(self, op_call: OpCall, sctx: ScheduleContext) -> Tuple[bool, Optional[str]]:
        try:
            out = self.fn(op_call, sctx)
            if isinstance(out, tuple):
                ok, reason = out
                return bool(ok), (str(reason) if not ok and reason is not None else None)
            return bool(out), None if out else None
        except DispatchFail as e:  # surface explicit failure reasons
            return False, str(e)
        except Exception as e:  # unexpected predicate exception
            return False, f"predicate exception: {type(e).__name__}: {e}"


def predicate(name: str, fn: Callable[[OpCall, ScheduleContext], Any]) -> Predicate:
    """Wrap a callable into a named predicate."""

    return Predicate(name=name, fn=fn)


def fail(reason: str) -> None:
    """Helper for schedule variants to explain why they decline to handle the op."""

    raise DispatchFail(reason)


@dataclass
class DispatchCase:
    variant: str
    priority: int
    preds: List[Predicate]
    impl: Callable[[OpCall, ScheduleContext], Optional[PrimFunc]]


# Keyed by (Op, target_kind)
_DISPATCH_TABLE: Dict[Tuple[Op, str], List[DispatchCase]] = {}


def register_dispatch(
    op_name: str,
    target_kind: str,
    *,
    variant: str,
    priority: int = 0,
    when: Optional[List[Predicate]] = None,
):
    """Decorator to add a dispatch case for an op/target pair.

    Cases with higher priority run earlier. When list predicates must all pass.
    The impl should return a PrimFunc on success, or None to decline.
    Use `fail("reason")` to add structured decline reasons.
    """

    op = get_tirp_op(op_name)

    def decorator(impl: Callable[[OpCall, ScheduleContext], Optional[PrimFunc]]):
        cases = _DISPATCH_TABLE.setdefault((op, target_kind), [])
        cases.append(DispatchCase(variant=variant, priority=priority, preds=when or [], impl=impl))
        return impl

    return decorator


def list_registered_schedules() -> Dict[str, Dict[str, List[str]]]:
    """Return a mapping: op_name -> target_kind -> [variant names]."""

    out: Dict[str, Dict[str, List[str]]] = {}
    for (op, tgt), cases in _DISPATCH_TABLE.items():
        name = op.name
        out.setdefault(name, {}).setdefault(tgt, [])
        # keep insertion order by default; sort by priority desc for readability
        for c in sorted(cases, key=lambda x: (-x.priority, x.variant)):
            out[name][tgt].append(c.variant)
    return out


def _env_truthy(name: str) -> bool:
    val = os.environ.get(name, "0").strip().lower()
    return val in ("1", "true", "yes", "on")


def _match_trace_filter(op_call: OpCall, sctx: ScheduleContext) -> bool:
    ops = os.environ.get("TVM_TIRP_SCHED_TRACE_FILTER", "").strip()
    if ops:
        allow = {x.strip() for x in ops.split(",") if x.strip()}
        if op_call.op.name not in allow:
            return False
    tgt = os.environ.get("TVM_TIRP_SCHED_TRACE_TARGET", "").strip()
    if tgt and str(sctx.target.kind) != tgt:
        return False
    return True


def run_dispatch(op_call: OpCall, sctx: ScheduleContext) -> Optional[PrimFunc]:
    """Run structured dispatch if cases exist; return PrimFunc or None.

    If `TVM_TIRP_SCHED_TRACE=1`, prints a detailed report when all cases fail.
    If `TVM_TIRP_SCHED_STRICT=1`, raises an aggregated error when all cases fail.
    """

    key = (op_call.op, str(sctx.target.kind))
    cases = _DISPATCH_TABLE.get(key)
    if not cases:
        # No registered variants; honor TRACE/STRICT for visibility
        trace = _env_truthy("TVM_TIRP_SCHED_TRACE") and _match_trace_filter(op_call, sctx)
        strict = _env_truthy("TVM_TIRP_SCHED_STRICT")
        if trace or strict:
            header = (
                f"TIRp schedule dispatch failed: op={op_call.op.name} target={sctx.target.kind}"
            )
            report = "\n".join([header, "- no registered variants for this op/target"])
            if trace:
                print(report)
            if strict:
                raise RuntimeError(report)
        return None

    trace = _env_truthy("TVM_TIRP_SCHED_TRACE") and _match_trace_filter(op_call, sctx)
    strict = _env_truthy("TVM_TIRP_SCHED_STRICT")

    failures: List[str] = []

    # If explicit dispatch is set, filter to that variant only
    forced_variant = getattr(op_call, "dispatch", None)
    if forced_variant is not None:
        cases = [c for c in cases if c.variant == forced_variant]
        if not cases:
            msg = (
                f"TIRp schedule dispatch failed: op={op_call.op.name} target={sctx.target.kind}"
                f"\n- no variant named '{forced_variant}' is registered"
            )
            if _env_truthy("TVM_TIRP_SCHED_TRACE") and _match_trace_filter(op_call, sctx):
                print(msg)
            if _env_truthy("TVM_TIRP_SCHED_STRICT"):
                raise RuntimeError(msg)
            return None

    for case in sorted(cases, key=lambda c: (-c.priority, c.variant)):
        # evaluate predicates
        pred_ok = True
        pred_msgs: List[str] = []
        for pred in case.preds:
            ok, reason = pred.evaluate(op_call, sctx)
            if not ok:
                pred_ok = False
                msg = f"rejected: {pred.name}"
                if reason:
                    msg += f" — {reason}"
                pred_msgs.append(msg)
        if not pred_ok:
            failures.append(f"- variant={case.variant} (prio={case.priority}): " + "; ".join(pred_msgs))
            continue

        # run impl
        try:
            res = case.impl(op_call, sctx)
            if res is not None:
                return res
            failures.append(f"- variant={case.variant} (prio={case.priority}): impl returned None")
        except DispatchFail as e:
            failures.append(
                f"- variant={case.variant} (prio={case.priority}): declined — {str(e)}"
            )
        except Exception as e:  # keep searching other variants
            msg = f"- variant={case.variant} (prio={case.priority}): exception — {type(e).__name__}: {e}"
            failures.append(msg)

    # no success
    if trace or strict:
        header = (
            f"TIRp schedule dispatch failed: op={op_call.op.name} target={sctx.target.kind}"
        )
        report = "\n".join([header, *failures])
        if trace:
            print(report)
        if strict:
            raise RuntimeError(report)
    return None
