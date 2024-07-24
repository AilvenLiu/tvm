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
from tvm.tir.layout import TileLayout, SwizzleLayout
import numpy as np
import tvm.testing
from tvm.script import ir as I
from tvm.script import tir as T
from tvm.script import tirp as Tp


@pytest.mark.parametrize(
    "input",
    [
        ################ A[0:8, 0:8] -> A_smem[0:8, 0:8] -> B[0:8, 0:8] ################
        (
            (16, 16),  # g_shape
            (8, 8),  # s_shape
            (0, 0),  # g_st
            (8, 8),  # g_extent
            8,  # thread_cnt
            TileLayout.from_nested_tuple((16, 16)),  # layoutA
            TileLayout.from_nested_tuple((16, 16)),  # layoutB
            TileLayout.from_nested_tuple((8, 8)),  # layoutS
            tvm.cuda(0),
        ),
        ################ A[0:128, 0:32] -> A_smem[0:128, 0:32] -> B[0:128, 0:32] ################
        (
            (128, 32),  # g_shape
            (128, 32),  # s_shape
            (0, 0),  # g_st
            (128, 32),  # g_extent
            32,  # thread_cnt
            TileLayout.from_nested_tuple((128, 32)),  # layoutA
            TileLayout.from_nested_tuple((128, 32)),  # layoutB
            TileLayout.from_nested_tuple((128, 32)),  # layoutS
            tvm.cuda(0),
        ),
        ################ A[32:64, 32:64] -> A_smem[0:32, 0:32] -> B[32:64, 32:64] ################
        (
            (64, 64),  # g_shape
            (32, 32),  # s_shape
            (32, 0),  # g_st
            (32, 32),  # g_extent
            32,  # thread_cnt
            TileLayout.from_nested_tuple((64, 64)),  # layoutA
            TileLayout.from_nested_tuple((64, 64)),  # layoutB
            TileLayout.from_nested_tuple((32, 32)),  # layoutS
            tvm.cuda(0),
        ),
        ################ A[0:1, 0:32, 0:32] -> A_smem[0:32, 0:32] -> B[0:1, 0:32, 0:32] ################
        (
            (4, 32, 32),  # g_shape
            (32, 32),  # s_shape
            (0, 0, 0),  # g_st
            (1, 32, 32),  # g_extent
            32,  # thread_cnt
            TileLayout.from_nested_tuple((4, 32, 32)),  # layoutA
            TileLayout.from_nested_tuple((4, 32, 32)),  # layoutB
            TileLayout.from_nested_tuple((32, 32)),  # layoutS
            tvm.cuda(0),
        ),
        ################ A[0:128, 0:32] -> A_smem[0:128, 0:32] -> B[0:128, 0:32] ################
        (
            (128, 32),  # g_shape
            (128, 32),  # s_shape
            (0, 0),  # g_st
            (128, 32),  # g_extent
            32,  # thread_cnt
            TileLayout.from_nested_tuple((128, 32)),  # layoutA
            TileLayout.from_nested_tuple((128, 32)),  # layoutB
            SwizzleLayout(3, 3, 3),  # layoutS
            tvm.cuda(0),
        ),
    ],
)
@pytest.mark.parametrize("dtype", ["float32", "float16"])
@pytest.mark.parametrize("sync", [True, False])
def test_copy_global_to_shared(input, dtype, sync):
    g_shape, s_shape, g_st, g_extent, thread_cnt, layoutA, layoutB, layoutS, dev = input

    r_smem = list(slice(None) for i in range(len(s_shape)))
    r_gmem = list(slice(g_st[i], g_st[i] + g_extent[i]) for i in range(len(g_shape)))

    # fmt: off
    @T.prim_func(tirp=True)
    def copy_sync(A_ptr: T.handle, B_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, g_shape, dtype, layout=layoutA)
        B = T.match_buffer(B_ptr, g_shape, dtype, layout=layoutB)
        
        with T.kernel():
            bx = T.cta_id([1], parent="kernel")
            tx = T.thread_id([thread_cnt], parent="cta")

            with T.cta():
                A_smem = T.alloc_buffer(s_shape, dtype, scope="shared", layout=layoutS)
                
                Tp.copy(A_smem[*r_smem], A[*r_gmem])
                Tp.copy(B[*r_gmem], A_smem[*r_smem])

    @T.prim_func(tirp=True)
    def copy_async(A_ptr: T.handle, B_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, g_shape, dtype, layout=layoutA)
        B = T.match_buffer(B_ptr, g_shape, dtype, layout=layoutB)
        
        with T.kernel():
            bx = T.cta_id([1], parent="kernel")
            tx = T.thread_id([thread_cnt], parent="cta")

            with T.cta():
                A_smem = T.alloc_buffer(s_shape, dtype, scope="shared", layout=layoutS)

                pipeline = Tp.alloc_pipeline("cta", depth=0, specialize=False)
                pipeline.producer_copy_async(A_smem[*r_smem], A[*r_gmem])
                pipeline.producer_commit_stage()
                pipeline.consumer_wait(0)
                Tp.copy(B[*r_gmem], A_smem[*r_smem])
    # fmt: on

    target = tvm.target.Target.from_device(dev)
    with target:
        mod = tvm.IRModule({"main": copy_sync if sync else copy_async})
        mod = tvm.tir.transform.LowerTIRp()(mod)
        mod = tvm.build(mod, target=target)

        np.random.seed(0)
        A_np = np.random.rand(*g_shape).astype(dtype)
        B_np = np.zeros(g_shape, dtype=dtype)
        A = tvm.nd.array(A_np, dev)
        B = tvm.nd.array(B_np, dev)
        mod(A, B)

        B_ref = B_np.copy()
        B_ref[*r_gmem] = A_np[*r_gmem]
        tvm.testing.assert_allclose(B_ref, B.asnumpy())


if __name__ == "__main__":
    tvm.testing.main()
