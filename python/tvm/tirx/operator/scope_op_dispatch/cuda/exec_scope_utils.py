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

"""Execution scope utilities for CUDA op dispatches."""

import functools
import operator
from collections.abc import Callable

from tvm.script import tirx as Tx
from tvm.tirx import PrimExpr, PrimFunc
from tvm.tirx.exec_scope import ExecScopeSlice
from tvm.tirx.operator.scope_op_dispatch import DispatchContext
from tvm.tirx.stmt import ScopeOpCall


def macro_or_prim_func(macro: Callable, need_macro: bool = False) -> Callable:
    """Convert a macro to a prim_func."""

    if need_macro:
        return macro

    @Tx.prim_func(tirx=True, check_well_formed=False)
    def func():
        macro()

    return func


def thread_selector(sctx: DispatchContext, inner_impl, macro=False) -> Callable:
    """Select a single thread from the given exec scope.

    For a certain scope, it should return a deterministic thread index, i.e. the
    same thread is elected every time. This is vital for the correctness of many
    synchronization primitives. PTX's elect_sync() is one example.

    Parameters
    ----------
    sctx : DispatchContext
        The dispatch context.

    inner_impl : Tx.inline
        The inner implementation.

    macro : bool
        Whether return a macro or a prim_func.

    Returns
    -------
    thread_selector : a macro or a prim_func
        The inner implementation wrapped by a thread selector in the given exec scope.
    """
    assert not isinstance(inner_impl, PrimFunc), (
        "inner_impl should be a macro rather than a PrimFunc"
    )

    exec_scope = sctx.exec_scope
    tx = sctx.launch_params["threadIdx.x"]
    # currently don't support multi-dimensional thread binding
    assert "threadIdx.y" not in sctx.launch_params and "threadIdx.z" not in sctx.launch_params

    if exec_scope.name == "cta":
        assert not isinstance(exec_scope, ExecScopeSlice)

        @Tx.inline()
        def impl():
            with Tx.thread()[0:1]:
                inner_impl()

        return macro_or_prim_func(impl, need_macro=macro)

    elif exec_scope.name == "warp":
        if isinstance(exec_scope, ExecScopeSlice) and not isinstance(exec_scope.slices, PrimExpr):
            # slice of multiple warps
            warp_selector = [slice(r.min, r.min + 1) for r in exec_scope.slices]

            @Tx.inline()
            def impl():
                with Tx.warp()[tuple(warp_selector)]:
                    with Tx.thread(parent="warp")[Tx.ptx.elect_sync()]:
                        inner_impl()

            return macro_or_prim_func(impl, need_macro=macro)
        else:
            # a single warp
            @Tx.inline()
            def impl():
                with Tx.thread(parent="warp")[Tx.ptx.elect_sync()]:
                    inner_impl()

            return macro_or_prim_func(impl, need_macro=macro)
    elif exec_scope.name == "warpgroup":
        if isinstance(exec_scope, ExecScopeSlice) and not isinstance(exec_scope.slices, PrimExpr):
            # slice of multiple warpgroups
            warpgroup_selector = [slice(r.min, r.min + 1) for r in exec_scope.slices]

            @Tx.inline()
            def impl():
                with Tx.warpgroup()[tuple(warpgroup_selector)]:
                    with Tx.warp(parent="warpgroup")[(tx // 32) % 4 == 0]:
                        with Tx.thread(parent="warp")[Tx.ptx.elect_sync()]:
                            inner_impl()

            return macro_or_prim_func(impl, need_macro=macro)
        else:
            # a single warpgroup
            @Tx.inline()
            def impl():
                with Tx.warp(parent="warpgroup")[(tx // 32) % 4 == 0]:
                    with Tx.thread(parent="warp")[Tx.ptx.elect_sync()]:
                        inner_impl()

            return macro_or_prim_func(impl, need_macro=macro)
    elif exec_scope.name == "thread":
        # already a single thread, just return the inner_impl
        return macro_or_prim_func(inner_impl, need_macro=macro)
    else:
        raise ValueError(
            f"Currently exec scope {exec_scope} is not supported to select a single thread within"
        )


def single_thread(op_call: ScopeOpCall, sctx: DispatchContext) -> bool:
    return (
        sctx.exec_scope.name == "thread"
        and isinstance(sctx.exec_scope, ExecScopeSlice)
        and (
            isinstance(sctx.exec_scope.slices, PrimExpr)
            or functools.reduce(operator.mul, [s.extent for s in sctx.exec_scope.slices], 1) == 1
        )
    )


def exec_scope_ok(
    op_call: ScopeOpCall, sctx: DispatchContext, expected_scopes: list[str]
) -> tuple[bool, str | None]:
    ok = sctx.exec_scope.name in expected_scopes
    return (ok, None if ok else f"unsupported exec_scope {sctx.exec_scope.name}")
