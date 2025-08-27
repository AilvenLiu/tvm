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

import functools
from typing import Tuple

import numpy as np

import tvm
import tvm.testing
from tvm import relax
from tvm.relax.transform import TileScheduler
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tir as T
from tvm.script import tirp as Tp
from tvm.script.ir_builder import IRBuilder


def test_two_stage_reduction():

    M = 1024
    N = 1024

    BLOCK_M = 64
    BLOCK_N = 64

    NUM_BLOCK_M = M // BLOCK_M
    NUM_BLOCK_N = N // BLOCK_N

    SM_CNT = 148

    # fmt: off
    @I.ir_module(tirp=True)
    class Before:
        @T.prim_func(tirp=True, private=True)
        def stage_1(A: T.handle, B: T.handle, m: T.int32, n: T.int32):
            A_ptr = T.match_buffer(A, (M, N), "float32")
            B_ptr = T.match_buffer(B, (M, NUM_BLOCK_N), "float32")
            with T.cta():
                tx = T.thread_id([1024], parent="cta")
                A_smem = T.alloc_buffer([BLOCK_M, BLOCK_N], "float32", scope="shared")
                B_smem = T.alloc_buffer([BLOCK_M, 1], "float32", scope="shared")
                Tp.copy(A_smem, A_ptr[m * BLOCK_M: (m + 1) * BLOCK_M, n * BLOCK_N: (n + 1) * BLOCK_N])                
                Tp.sum(B_smem, A_smem)

                Tp.copy(B_ptr[m * BLOCK_M: (m + 1) * BLOCK_M, n], B_smem)

        @T.prim_func(tirp=True, private=True)
        def stage_2(B: T.handle, C: T.handle, m: T.int32):
            B_ptr = T.match_buffer(B, (M, NUM_BLOCK_N), "float32")
            C_ptr = T.match_buffer(C, (M, 1), "float32")
            with T.cta():
                tx = T.thread_id([1024], parent="cta")
                B_smem = T.alloc_buffer(
                    [BLOCK_M, NUM_BLOCK_N], "float32", scope="shared"
                )
                C_smem = T.alloc_buffer([BLOCK_M, 1], "float32", scope="shared")
                Tp.copy(B_smem, B_ptr[m * BLOCK_M : (m + 1) * BLOCK_M, :])
                Tp.sum(C_smem, B_smem)
                Tp.copy(C_ptr[m * BLOCK_M : (m + 1) * BLOCK_M, 0], C_smem)

        @R.function
        def mega_kernel(A: R.Tensor((M, N), "float32"), event: R.Tensor((NUM_BLOCK_M, ), "int32")):
            cls = Before
            with R.dataflow():
                B = R.call_tir_device(cls.stage_1,  A, R.Tensor((M, NUM_BLOCK_N), "float32"), (NUM_BLOCK_M, NUM_BLOCK_N), out_events=[event], out_deps=[lambda x, y: x])
                C = R.call_tir_device(cls.stage_2,  B, R.Tensor((M, 1), "float32"), (NUM_BLOCK_M, ), in_events=[event], in_deps=[lambda x: x])
                R.output(C)
            return C
    # fmt: on
    class SpatialTileScheduler(TileScheduler):
        @staticmethod
        def int_var():
            return T.alloc_buffer([1], "int32", scope="local", align=4)

        def __init__(self):
            self.sm_cnt = SM_CNT
            self.division = 132
            self.m_idx = self.int_var()
            self.n_idx = self.int_var()
            self.type = self.int_var()
            self.linear_idx = self.int_var()
            IRBuilder.current().name("m_idx", self.m_idx)
            IRBuilder.current().name("n_idx", self.n_idx)
            IRBuilder.current().name("linear_idx", self.linear_idx)
            IRBuilder.current().name("type", self.type)

        @T.macro
        def update_current_m_n_idx(self):
            if self.type[0] == 0:
                self.m_idx[0] = self.linear_idx[0] // NUM_BLOCK_N
                self.n_idx[0] = self.linear_idx[0] % NUM_BLOCK_N
            else:
                self.m_idx[0] = self.linear_idx[0]

        def get_idx_and_task_type(self):
            return [self.m_idx[0], self.n_idx[0]], self.type[0]

        @T.macro
        def init(self, value):
            if value < self.division:
                self.type[0] = 0
                self.linear_idx[0] = value
            else:
                self.type[0] = 1
                self.linear_idx[0] = value - self.division
            self.update_current_m_n_idx()

        @T.macro
        def next_tile(self):
            if self.type[0] == 0:
                self.linear_idx[0] = self.linear_idx[0] + self.division
            else:
                self.linear_idx[0] = self.linear_idx[0] + (self.sm_cnt - self.division)
            self.update_current_m_n_idx()

        def valid(self):
            return T.if_then_else(
                self.type[0] == 1,
                self.linear_idx[0] < NUM_BLOCK_M,
                self.linear_idx[0] < NUM_BLOCK_M * NUM_BLOCK_N,
            )

    mod = relax.transform.StaticHorizontalFusion(SM_CNT, {"mega_kernel": SpatialTileScheduler})(
        Before
    )
    mod_std = relax.transform.LowerCallTIRDevice()(Before)

    # testing correctness
    A_np = np.random.randn(M, N).astype(np.float32)
    DEV = tvm.cuda(0)
    target = tvm.target.Target("cuda")

    A_tvm = tvm.nd.array(A_np, device=DEV)

    # we initialize event on host instead of device because of Tp.fill not correctly implemented
    sem_tvm = tvm.nd.array(np.full((NUM_BLOCK_M,), NUM_BLOCK_N, dtype=np.int32), device=DEV)
    with target:

        std_mod = tvm.compile(mod_std, target=target, tir_pipeline="tirp")
        vm_std = tvm.relax.VirtualMachine(std_mod, DEV)
        C_tvm_std = vm_std["mega_kernel"](A_tvm, sem_tvm)
        ret_std = C_tvm_std.numpy()

        fused_mod = mod
        fused_mod = tvm.compile(fused_mod, target=target, tir_pipeline="tirp")
        vm = tvm.relax.VirtualMachine(fused_mod, DEV)
        C_tvm_fused = vm["mega_kernel"](A_tvm, sem_tvm)
        ret_fused = C_tvm_fused.numpy()
        tvm.testing.assert_allclose(ret_fused, ret_std, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    test_two_stage_reduction()
