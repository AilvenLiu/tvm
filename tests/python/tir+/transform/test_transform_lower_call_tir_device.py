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

import tvm
import tvm.testing
from tvm import relax
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tir as T, tirp as Tp


def test_call_tir_device():
    @I.ir_module(tirp=True)
    class Before:
        @T.prim_func(tirp=True, private=True)
        def test1(A: T.Buffer((2048,), "float32"), m: T.int32):
            with T.cta():
                thread_id = T.thread_id([128], parent="cta")
                Tp.fill(A[m * 128 : m * 128 + 128], 0.0)

        @T.prim_func(tirp=True, private=True)
        def test2(A: T.Buffer((2048, 2048), "float32"), m: T.int32, n: T.int32):
            with T.cta():
                thread_id = T.thread_id([128], parent="cta")
                Tp.fill(A[m * 128 : m * 128 + 128, n * 128 : n * 128 + 128], 0.0)

        @R.function
        def main():  # type: ignore
            cls = Before
            with R.dataflow():
                x = R.call_tir_device(
                    cls.test1,
                    [],
                    R.Tensor((2048,), "float32"),
                    0,
                    (16,),
                )
                y = R.call_tir_device(
                    cls.test2,
                    [],
                    R.Tensor((2048, 2048), "float32"),
                    0,
                    (16, 16),
                )
                R.output(x, y)
            return x, y

    @I.ir_module(tirp=True)
    class After:
        @T.prim_func(tirp=True, private=True)
        def test1_kernel(A: T.Buffer((2048,), "float32")):
            with T.kernel():
                m = T.cta_id([16], parent="kernel")
                with T.cta():
                    thread_id = T.thread_id([128], parent="cta")
                    Tp.fill(A[m * 128 : m * 128 + 128], 0.0)

        @T.prim_func(tirp=True, private=True)
        def test2_kernel(A: T.Buffer((2048, 2048), "float32")):
            with T.kernel():
                m, n = T.cta_id([16, 16], parent="kernel")
                with T.cta():
                    thread_id = T.thread_id([128], parent="cta")
                    Tp.fill(A[m * 128 : m * 128 + 128, n * 128 : n * 128 + 128], 0.0)

        @R.function
        def main():  # type: ignore
            cls = After
            with R.dataflow():
                x = R.call_tir(cls.test1_kernel, [], R.Tensor((2048,), "float32"))
                y = R.call_tir(cls.test2_kernel, [], R.Tensor((2048, 2048), "float32"))
                R.output(x, y)
            return x, y

    mod = Before
    mod = relax.transform.LowerCallTIRDevice()(mod)
    tvm.ir.assert_structural_equal(mod, After)


if __name__ == "__main__":
    test_call_tir_device()
