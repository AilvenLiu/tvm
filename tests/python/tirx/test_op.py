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
import tvm
from tvm.ir import Op
from tvm.tir.buffer import decl_buffer
from tvm.script import tir as T
from tvm.script import tirx as Tx
from tvm.tir.stmt import OpCall


def _test(op: str, *args):
    return OpCall(*args, op=Op.get("tirx." + op), workspace={}, config={})


def test_copy():
    # test argsanitizer
    A = decl_buffer((64, 64), "float32", scope="global")
    A_sm = decl_buffer((64, 64), "float32", scope="shared")

    _test("copy", A[0:64, 0:64], A_sm[0:64, 0:64])
    with pytest.raises(AssertionError):
        _test("copy", A[0:64, 0:64], A_sm[0:64, 0:64], 1)
    with pytest.raises(AssertionError):
        _test("copy", 1, A_sm[0:64, 0:64])
    with pytest.raises(AssertionError):
        _test("copy", A[0:64, 0:64], A_sm)


def test_fill():
    # test argsanitizer
    A = decl_buffer((64, 64), "float32", scope="global")

    _test("fill", A[0:64, 0:64], 1.0)
    with pytest.raises(AssertionError):
        _test("fill", A[0:64, 0:64], 1.0, 1)
    with pytest.raises(AssertionError):
        _test("fill", 1, 1.0)


def test_gemm():
    # test argsanitizer
    A = decl_buffer((64, 64), "float32", scope="global")
    B = decl_buffer((64, 64), "float32", scope="global")
    C = decl_buffer((64, 64), "float32", scope="global")
    D = decl_buffer((64, 64), "float32", scope="global")

    _test("gemm", D[:, :], A[:, :], B[:, :], C[:, :], True, False, 1.0, 0.0)
    with pytest.raises(AssertionError):
        _test("gemm", D[:, :], A[:, :], B[:, :], C[:, :], True, False, 1.0, 0.0, 1)


def test_generic_op_creates_op():
    """GenericOp auto-registers unknown ops."""
    from tvm.tirx.operator.op import GenericOp

    A = decl_buffer((64,), "float32", scope="global")
    B = decl_buffer((64,), "float32", scope="global")

    op_call = GenericOp(B[0:64], A[0:64], op_name="my_custom_op_1")
    assert op_call.op == Op.get("tirx.my_custom_op_1")
    assert len(op_call.args) == 2


def test_generic_op_reuses_registered_op():
    """GenericOp reuses already-registered ops without error."""
    from tvm.tirx.operator.op import GenericOp

    A = decl_buffer((64,), "float32", scope="global")
    B = decl_buffer((64,), "float32", scope="global")

    # Create twice with same name — should not error
    op1 = GenericOp(B[0:64], A[0:64], op_name="my_custom_op_2")
    op2 = GenericOp(B[0:64], A[0:64], op_name="my_custom_op_2")
    assert op1.op == op2.op


def test_generic_op_with_existing_tirx_op():
    """GenericOp works with already-registered tirx ops (e.g., tirx.copy)."""
    from tvm.tirx.operator.op import GenericOp

    A = decl_buffer((64,), "float32", scope="global")
    B = decl_buffer((64,), "float32", scope="global")

    op_call = GenericOp(B[0:64], A[0:64], op_name="copy")
    assert op_call.op == Op.get("tirx.copy")


def test_tx_dynamic_op_module_getattr():
    """Tx.some_undefined_op resolves via module __getattr__."""
    fn = Tx.my_dynamic_test_op
    assert callable(fn)
    assert fn.__name__ == "my_dynamic_test_op"


def test_tx_dynamic_op_in_prim_func():
    """Tx.copy_and_cast(...) works inside a prim_func without pre-registration."""

    @T.prim_func(tirx=True)
    def func(A_ptr: T.handle, B_ptr: T.handle):
        A = T.match_buffer(A_ptr, [64], "float32", scope="global")
        B = T.match_buffer(B_ptr, [64], "float16", scope="global")
        with T.kernel():
            Tx.copy_and_cast(B, A)

    # Walk IR to find OpCall with op="tirx.copy_and_cast"
    found = [False]

    def visit(stmt):
        if isinstance(stmt, OpCall) and stmt.op == Op.get("tirx.copy_and_cast"):
            found[0] = True

    tvm.tir.stmt_functor.post_order_visit(func.body, visit)
    assert found[0], "Expected OpCall with tirx.copy_and_cast not found"


def test_tx_dynamic_op_with_workspace():
    """Tx.some_op(..., workspace={...}) passes workspace to OpCall."""

    @T.prim_func(tirx=True)
    def func(A_ptr: T.handle, B_ptr: T.handle, W_ptr: T.handle):
        A = T.match_buffer(A_ptr, [64], "float32", scope="global")
        B = T.match_buffer(B_ptr, [64], "float32", scope="global")
        W = T.match_buffer(W_ptr, [64], "float32", scope="shared")
        with T.kernel():
            Tx.custom_with_ws(B, A, workspace={"tmp": W})

    found = [False]

    def visit(stmt):
        if isinstance(stmt, OpCall) and stmt.op == Op.get("tirx.custom_with_ws"):
            assert "tmp" in stmt.workspace
            found[0] = True

    tvm.tir.stmt_functor.post_order_visit(func.body, visit)
    assert found[0], "Expected OpCall with workspace not found"


def test_tx_existing_op_not_overridden():
    """Existing Tx.copy still dispatches to the registered copy op, not __getattr__."""

    @T.prim_func(tirx=True)
    def func(A_ptr: T.handle, B_ptr: T.handle):
        A = T.match_buffer(A_ptr, [64], "float32", scope="global")
        B = T.match_buffer(B_ptr, [64], "float32", scope="global")
        with T.kernel():
            Tx.copy(B, A)

    found = [False]

    def visit(stmt):
        if isinstance(stmt, OpCall) and stmt.op == Op.get("tirx.copy"):
            found[0] = True

    tvm.tir.stmt_functor.post_order_visit(func.body, visit)
    assert found[0], "Expected OpCall with tirx.copy not found"


def test_opcall_downcast_tolerant():
    """OpCall.downcast returns instance as-is for unknown ops."""
    from tvm.tirx.operator.op import GenericOp

    A = decl_buffer((64,), "float32", scope="global")
    B = decl_buffer((64,), "float32", scope="global")

    op_call = GenericOp(B[0:64], A[0:64], op_name="totally_unknown_op")
    # downcast should not raise
    result = OpCall.downcast(op_call)
    assert result is not None


if __name__ == "__main__":
    test_copy()
    test_fill()
    test_gemm()
    test_generic_op_creates_op()
    test_generic_op_reuses_registered_op()
    test_generic_op_with_existing_tirx_op()
    test_tx_dynamic_op_module_getattr()
    test_tx_dynamic_op_in_prim_func()
    test_tx_dynamic_op_with_workspace()
    test_tx_existing_op_not_overridden()
    test_opcall_downcast_tolerant()
