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

import pytest


def _import_and_register():
    # Ensure all schedule registrations (legacy + dispatcher variants) are loaded
    import tvm.tirp.op_schedule as _  # noqa: F401


class _DummyKind:
    def __init__(self, name: str):
        self.name = name

    def __str__(self) -> str:  # used in messages
        return self.name


class _DummyTarget:
    def __init__(self, kind_name: str):
        self.kind = _DummyKind(kind_name)


class _DummyExecScope:
    def __init__(self, name: str):
        self.name = name


class _DummySctx:
    def __init__(self, target_kind: str, exec_scope: str):
        self.target = _DummyTarget(target_kind)
        self.exec_scope = _DummyExecScope(exec_scope)


def test_dispatch_prints_predicate_reasons(monkeypatch, capsys):
    """Validate TRACE mode prints per-variant predicate failure reasons."""
    _import_and_register()
    from tvm.ir import Op
    from tvm.tirp.op_schedule.dispatcher import run_dispatch

    # Enable TRACE and focus on TRN copy only
    monkeypatch.setenv("TVM_TIRP_SCHED_TRACE", "1")
    monkeypatch.setenv("TVM_TIRP_SCHED_TRACE_TARGET", "trn")
    # Filter matches the full op name used by dispatcher
    monkeypatch.setenv("TVM_TIRP_SCHED_TRACE_FILTER", "tirp.copy")

    class _OpCall:
        def __init__(self, op):
            self.op = op
            self.args = []  # not used by the tested predicates

    # Use TRN copy; predicate requires exec_scope == "kernel".
    op_call = _OpCall(Op.get("tirp.copy"))
    sctx = _DummySctx(target_kind="trn", exec_scope="warp")  # intentionally wrong

    res = run_dispatch(op_call, sctx)
    assert res is None

    out = capsys.readouterr().out
    # Header + per-variant reason must be printed
    assert "TIRp schedule dispatch failed: op=tirp.copy target=trn" in out
    assert "variant=default" in out
    assert "rejected: exec_scope" in out


def test_dispatch_raises_with_aggregated_reasons(monkeypatch):
    """Validate STRICT mode raises aggregated error message with reasons."""
    _import_and_register()
    from tvm.ir import Op
    from tvm.tirp.op_schedule.dispatcher import run_dispatch

    # Enable STRICT and TRACE to exercise both code paths
    monkeypatch.setenv("TVM_TIRP_SCHED_TRACE", "1")
    monkeypatch.setenv("TVM_TIRP_SCHED_STRICT", "1")

    class _OpCall:
        def __init__(self, op):
            self.op = op
            self.args = []

    # Use TRN compose_op; variant implementation raises NotImplementedError
    op_call = _OpCall(Op.get("tirp.compose_op"))
    sctx = _DummySctx(target_kind="trn", exec_scope="kernel")

    with pytest.raises(RuntimeError) as e:
        run_dispatch(op_call, sctx)

    msg = str(e.value)
    assert "TIRp schedule dispatch failed: op=tirp.compose_op target=trn" in msg
    assert "variant=default" in msg
    assert "exception — NotImplementedError" in msg
