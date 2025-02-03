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
from tvm.tir.layout import TileLayout
import numpy as np
import tvm.testing
from tvm.script import ir as I
from tvm.script import tir as T
from tvm.script import tirp as Tp


@pytest.mark.parametrize(
    "input",
    [
        ######### basic test #########
        (
            (32, 32,), # g_shape_a
            (32,), # g_shape_b
            (0, 0,), # st_a
            (0,), # st_b
            (32, 32), # extent_a
            (32,), # extent_b
            32, # thread_cnt
            tvm.cuda(0), # dev
        ),
        ######### large size #########
        (
            (8, 16, 2, 22,), # g_shape_a
            (8, 16,), # g_shape_b
            (0, 0, 0, 0,), # st_a
            (0, 0,), # st_b
            (8, 16, 2, 22,), # extent_a
            (8, 16,), # extent_b
            128, # thread_cnt
            tvm.cuda(0), # dev
        ),
        # ######### small size #########
        (
            (32, 7,), # g_shape_a
            (32,), # g_shape_b
            (0, 0,), # st_a
            (0,), # st_b
            (32, 7,), # extent_a
            (32,), # extent_b
            32, # thread_cnt
            tvm.cuda(0), # dev
        ),
        ######### offset test #########
        (
            (32, 32,), # g_shape_a
            (32,), # g_shape_b
            (1, 1), # st_a
            (2,), # st_b
            (5, 8), # extent_a
            (5,), # extent_b
            32, # thread_cnt
            tvm.cuda(0), # dev
        ),
    ],
)
@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_reduction_op(input, dtype):
    g_shape_a, g_shape_b, st_a, st_b, extent_a, extent_b, thread_cnt, dev = input

    s_shape_a = g_shape_a
    s_shape_b = g_shape_b
    copy_slice_a = list(slice(None) for i in range(len(g_shape_a)))
    copy_slice_b = list(slice(None) for i in range(len(g_shape_b)))
    reduce_slice_a = list(slice(st_a[i], st_a[i] + extent_a[i]) for i in range(len(g_shape_a)))
    reduce_slice_b = list(slice(st_b[i], st_b[i] + extent_b[i]) for i in range(len(g_shape_b)))
    g_layout_a = s_layout_a = TileLayout.from_nested_tuple(g_shape_a)
    g_layout_b = s_layout_b = TileLayout.from_nested_tuple(g_shape_b)

    # fmt: off
    @T.prim_func(tirp=True)
    def sum_reduction(A_ptr: T.handle, B_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, g_shape_a, dtype, layout=g_layout_a)
        B = T.match_buffer(B_ptr, g_shape_b, dtype, layout=g_layout_b)
        
        with T.kernel():
            bx = T.cta_id([1], parent="kernel")
            tx = T.thread_id([thread_cnt], parent="cta")

            with T.cta():
                A_smem = T.alloc_buffer(s_shape_a, dtype, scope="shared", layout=s_layout_a)
                B_smem = T.alloc_buffer(s_shape_b, dtype, scope="shared", layout=s_layout_b)

                Tp.copy(A_smem[*copy_slice_a], A[*copy_slice_a])
                Tp.reduce(B_smem[*reduce_slice_b], A_smem[*reduce_slice_a])
                Tp.copy(B[*copy_slice_b], B_smem[*copy_slice_b])
    # fmt: on

    target = tvm.target.Target.from_device(dev)
    with target:
        mod = tvm.IRModule({"main": sum_reduction})
        mod = tvm.tir.transform.LowerTIRp()(mod)
        mod = tvm.build(mod, target=target)
        print(f"compiled source code: {mod.imported_modules[0].get_source()}")

        np.random.seed(0)
        A_np = np.random.rand(*g_shape_a).astype(dtype)
        B_np = np.zeros(g_shape_b, dtype=dtype)
        A = tvm.nd.array(A_np, dev)
        B = tvm.nd.array(B_np, dev)
        mod(A, B)
        D = len(A.shape) - len(B.shape)

        B_ref = A.asnumpy()[*reduce_slice_a].sum(axis=tuple(range(-D, 0)))
        atol = 1e-5 if dtype == "float32" else 1e-1
        tvm.testing.assert_allclose(B_ref, B.asnumpy()[*reduce_slice_b], atol=atol)


if __name__ == "__main__":
    tvm.testing.main()
