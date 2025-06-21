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
import numpy as np
import pytest

import tvm
from tvm.script import tir as T
from tvm.script import tirp as Tp
import tvm.testing


DEV = tvm.device("cuda")


def _get_source(func: tvm.tir.PrimFunc) -> str:
    target = tvm.target.Target("cuda")
    mod = tvm.IRModule({"main": func})
    mod = tvm.compile(mod, target=target, tir_pipeline="tirp")
    src = mod.mod.imported_modules[0].get_source()
    return src, mod


def test_cuda_func_call():
    def test_add_one():
        add_one = """
__device__ int32_t add_one(int32_t a) {
    return a + 1;
}
"""

        @T.prim_func(tirp=True)
        def main(a: T.Buffer((16, 16), "int32"), b: T.Buffer((16, 16), "int32")):
            with T.kernel():
                bx = T.cta_id([1], parent="kernel")
                tx = T.thread_id([32], parent="cta")
                with T.thread()[tx == 0]:
                    for i, j in T.grid(16, 16):
                        b[i, j] = T.cuda.func_call(
                            "add_one",
                            a[i, j],
                            source_code=add_one,
                            return_type="int32",
                        )

        src, mod = _get_source(main)
        A = np.random.randint(0, 10, (16, 16)).astype("int32")
        B = np.zeros((16, 16), dtype="int32")
        A_tvm = tvm.nd.array(A, device=DEV)
        B_tvm = tvm.nd.array(B, device=DEV)
        mod["main"](A_tvm, B_tvm)
        np.testing.assert_allclose(B_tvm.numpy(), A + 1)
        print(src)

    test_add_one()

    def test_print():
        print_func = """
__device__ void print(int32_t a) {
    printf("%d\\n", a);
}
"""

        @T.prim_func(tirp=True)
        def main(a: T.Buffer((16, 16), "int32")):
            with T.kernel():
                bx = T.cta_id([1], parent="kernel")
                tx = T.thread_id([32], parent="cta")
                with T.thread()[tx == 0]:
                    for i, j in T.grid(16, 16):
                        T.cuda.func_call("print", a[i, j], source_code=print_func)

        src, mod = _get_source(main)
        A = np.random.randint(0, 10, (16, 16)).astype("int32")
        A_tvm = tvm.nd.array(A, device=DEV)
        mod["main"](A_tvm)
        print(src)

    test_print()


if __name__ == "__main__":
    test_cuda_func_call()
