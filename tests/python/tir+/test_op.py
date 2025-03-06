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
from tvm.script import tirp as Tp


def _test(op: str, *args):
    f = tvm.get_global_func("tir.OpCall")
    assert isinstance(op, str)
    op = Op.get("tirp." + op)
    f(op, args, {}, {})


def test_copy():
    # test argsanitizer
    A = decl_buffer((64, 64), "float32", scope="global")
    A_sm = decl_buffer((64, 64), "float32", scope="shared")

    _test("copy", A[0:64, 0:64], A_sm[0:64, 0:64])
    with pytest.raises(Exception):
        _test("copy", A[0:64, 0:64], A_sm[0:64, 0:64], 1)
    with pytest.raises(Exception):
        _test("copy", 1, A_sm[0:64, 0:64])
    with pytest.raises(Exception):
        _test("copy", A[0:64, 0:64], A_sm)


def test_fill():
    # test argsanitizer
    A = decl_buffer((64, 64), "float32", scope="global")

    _test("fill", A[0:64, 0:64], 1.0)
    with pytest.raises(Exception):
        _test("fill", A[0:64, 0:64], 1.0, 1)
    with pytest.raises(Exception):
        _test("fill", 1, 1.0)


def test_gemm():
    # test argsanitizer
    A = decl_buffer((64, 64), "float32", scope="global")
    B = decl_buffer((64, 64), "float32", scope="global")
    C = decl_buffer((64, 64), "float32", scope="global")
    D = decl_buffer((64, 64), "float32", scope="global")

    _test("gemm", D[:, :], A[:, :], B[:, :], C[:, :], True, False, 1.0, 0.0)
    with pytest.raises(Exception):
        _test("gemm", D[:, :], A[:, :], B[:, :], C[:, :], True, False, 1.0, 0.0, 1)


if __name__ == "__main__":
    test_copy()
    test_fill()
    test_gemm()
