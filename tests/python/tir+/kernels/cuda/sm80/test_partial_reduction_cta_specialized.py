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
from tvm.script import tir as T
from tvm.script import tirp as Tp
from tvm.script.ir_builder import IRBuilder
from tvm.tir.event import EventImpl

M = 1024
N = 1024

BLOCK_M = 64
BLOCK_N = 64

NUM_BLOCK_M = M // BLOCK_M
NUM_BLOCK_N = N // BLOCK_N


# fmt: off
class Semaphore:
    def __init__(self, cnt, buffer):
        self.cnt = cnt
        self.sem = buffer
        self.state = T.alloc_buffer([1], "int32", scope="local", align=4)
        IRBuilder.current().name("semaphore_state", self.state)
        
    @T.macro
    def semaphore_wait(self, *coord):
        with T.thread():
            while 1:
                T.ptx.ld_global_acquire(self.state[0], self.sem.access_ptr("r", offset=self.sem.offset_of_p(coord)))
                if T.cuda.syncthreads_and(self.state[0] == self.cnt):
                    break
                T.cuda.nano_sleep(40)
                
    @T.macro 
    def semaphore_notify(self, *coord):
        with T.thread():
            T.tvm_storage_sync("shared")
            with T.thread()[0:1]:
                T.cuda.atomic_add(self.sem.access_ptr("rw", offset=self.sem.offset_of_p(coord)), 1)
            T.cuda.thread_fence()

# reduction on N
@T.prim_func(tirp=True)
def partial_reduction_ref_stage1(A: T.handle, B: T.handle):
    A_ptr = T.match_buffer(A, (M, N), "float32")
    B_ptr = T.match_buffer(B, (M, NUM_BLOCK_N), "float32")
    
    with T.kernel():
        bx, by = T.cta_id([NUM_BLOCK_M, NUM_BLOCK_N], parent="kernel")
        tx = T.thread_id([1024], parent="cta")
        
        with T.cta():
            A_smem = T.alloc_buffer([BLOCK_M, BLOCK_N], "float32", scope="shared")
            B_smem = T.alloc_buffer([BLOCK_M, 1], "float32", scope="shared")
            Tp.copy(A_smem, A_ptr[bx * BLOCK_M: (bx + 1) * BLOCK_M, by * BLOCK_N: (by + 1) * BLOCK_N])                
            Tp.sum(B_smem, A_smem)
            
            Tp.copy(B_ptr[bx * BLOCK_M: (bx + 1) * BLOCK_M, by], B_smem)
            

@T.prim_func(tirp=True)
def partial_reduction_ref_stage2(B: T.handle, C: T.handle):
    B_ptr = T.match_buffer(B, (M, NUM_BLOCK_N), "float32")
    C_ptr = T.match_buffer(C, (M, 1), "float32")
    
    with T.kernel():
        bx = T.cta_id([NUM_BLOCK_M], parent="kernel")
        tx = T.thread_id([1024], parent="cta")
        with T.cta():
            B_smem = T.alloc_buffer([BLOCK_M, NUM_BLOCK_N], "float32", scope="shared")
            C_smem = T.alloc_buffer([BLOCK_M, 1], "float32", scope="shared")
            Tp.copy(B_smem, B_ptr[bx * BLOCK_M: (bx + 1) * BLOCK_M, :])
            Tp.sum(C_smem, B_smem)
            Tp.copy(C_ptr[bx * BLOCK_M: (bx + 1) * BLOCK_M, 0], C_smem)
            
STAGE_1_SM_CNT = 116
STAGE_2_SM_CNT = 16
TOTAL_SM_CNT = STAGE_1_SM_CNT + STAGE_2_SM_CNT
            
class SpatialTileScheduler:
    @staticmethod
    def int_var():
        return T.alloc_buffer([1], "int32", scope="local", align=4)
    
    def __init__(self, prefix: str, tile_num: Tuple[int, int], sm_cnt: int):
        self.tile_num = tile_num
        self.sm_cnt = sm_cnt
        self.m_idx = self.int_var()
        self.n_idx = self.int_var()
        self.linear_idx = self.int_var()
        IRBuilder.current().name(prefix + "_m_idx", self.m_idx)
        IRBuilder.current().name(prefix + "_n_idx", self.n_idx)
        IRBuilder.current().name(prefix + "_linear_idx", self.linear_idx)
        
    def get_current_m_n_idx(self, linear_idx):
        row = linear_idx // self.tile_num[1]
        col = linear_idx % self.tile_num[1]
        return row, col
    
    @T.macro
    def init(self, linear_init):
        self.linear_idx[0] = linear_init
        self.m_idx[0] = self.get_current_m_n_idx(linear_init)[0]
        self.n_idx[0] = self.get_current_m_n_idx(linear_init)[1]

    @T.macro
    def next_tile(self):
        self.linear_idx[0] = self.linear_idx[0] + self.sm_cnt
        self.m_idx[0] = self.get_current_m_n_idx(self.linear_idx[0])[0]
        self.n_idx[0] = self.get_current_m_n_idx(self.linear_idx[0])[1]
        
    def valid(self):
        return self.linear_idx[0] < functools.reduce(lambda x, y: x * y, self.tile_num)

            
@T.prim_func(tirp=True)
def partial_reduction_fused(A: T.handle, B: T.handle, C: T.handle, semaphore: T.handle):
    A_ptr = T.match_buffer(A, (M, N), "float32")
    B_ptr = T.match_buffer(B, (M, NUM_BLOCK_N), "float32")
    C_ptr = T.match_buffer(C, (M, 1), "float32")
    sem_ptr = T.match_buffer(semaphore, (NUM_BLOCK_M, ), "int32")
    
    with T.kernel():
        bx = T.cta_id([TOTAL_SM_CNT], parent="kernel")
        tx = T.thread_id([1024], parent="cta")
        sem = T.meta_var(Semaphore(NUM_BLOCK_N, sem_ptr))
        with T.cta()[0: STAGE_1_SM_CNT]:
            A_smem = T.alloc_buffer([BLOCK_M, BLOCK_N], "float32", scope="shared")
            B_smem = T.alloc_buffer([BLOCK_M, 1], "float32", scope="shared")
            stage1_scheduler = T.meta_var(SpatialTileScheduler("stage1", (NUM_BLOCK_M, NUM_BLOCK_N), STAGE_1_SM_CNT))
            stage1_scheduler.init(bx)
            while stage1_scheduler.valid():
                m_idx = T.meta_var(stage1_scheduler.m_idx[0])
                n_idx = T.meta_var(stage1_scheduler.n_idx[0])
                Tp.copy(A_smem, A_ptr[m_idx * BLOCK_M: (m_idx + 1) * BLOCK_M, n_idx * BLOCK_N: (n_idx + 1) * BLOCK_N])                
                Tp.sum(B_smem, A_smem)
                Tp.copy(B_ptr[m_idx * BLOCK_M: (m_idx + 1) * BLOCK_M, n_idx], B_smem)
                sem.semaphore_notify(m_idx)
                stage1_scheduler.next_tile()
        with T.cta()[STAGE_1_SM_CNT: TOTAL_SM_CNT]:
            B_smem = T.alloc_buffer([BLOCK_M, NUM_BLOCK_N], "float32", scope="shared")
            C_smem = T.alloc_buffer([BLOCK_M, 1], "float32", scope="shared")
            stage2_scheduler = T.meta_var(SpatialTileScheduler("stage2", (NUM_BLOCK_M, 1), STAGE_2_SM_CNT))
            stage2_scheduler.init(bx - STAGE_1_SM_CNT)
            while stage2_scheduler.valid():
                m_idx = T.meta_var(stage2_scheduler.m_idx[0])
                sem.semaphore_wait(m_idx)
                Tp.copy(B_smem, B_ptr[m_idx * BLOCK_M: (m_idx + 1) * BLOCK_M, :])
                Tp.sum(C_smem, B_smem)
                Tp.copy(C_ptr[m_idx * BLOCK_M: (m_idx + 1) * BLOCK_M, 0], C_smem)
                stage2_scheduler.next_tile()

 
@T.prim_func(tirp=True)
def partial_reduction_fused_event(A: T.handle, B: T.handle, C: T.handle, semaphore: T.handle):
    A_ptr = T.match_buffer(A, (M, N), "float32")
    B_ptr = T.match_buffer(B, (M, NUM_BLOCK_N), "float32")
    C_ptr = T.match_buffer(C, (M, 1), "float32")
    sem_buf = T.match_buffer(semaphore, (NUM_BLOCK_M, ), scope="global", dtype="int32")

    with T.kernel():
        bx = T.cta_id([TOTAL_SM_CNT], parent="kernel")
        tx = T.thread_id([1024], parent="cta")

        state = T.alloc_local((1,), "int32")
        sem = Tp.alloc_semaphore_event_tensor(EventImpl.kGlobalSemaphore, state=[sem_buf, state], shape=[NUM_BLOCK_M])

        with T.cta()[0: STAGE_1_SM_CNT]:
            A_smem = T.alloc_buffer([BLOCK_M, BLOCK_N], "float32", scope="shared")
            B_smem = T.alloc_buffer([BLOCK_M, 1], "float32", scope="shared")
            stage1_scheduler = T.meta_var(SpatialTileScheduler("stage1", (NUM_BLOCK_M, NUM_BLOCK_N), STAGE_1_SM_CNT))
            stage1_scheduler.init(bx)
            while stage1_scheduler.valid():
                m_idx = T.meta_var(stage1_scheduler.m_idx[0])
                n_idx = T.meta_var(stage1_scheduler.n_idx[0])
                Tp.copy(A_smem, A_ptr[m_idx * BLOCK_M: (m_idx + 1) * BLOCK_M, n_idx * BLOCK_N: (n_idx + 1) * BLOCK_N])                
                Tp.sum(B_smem, A_smem)
                Tp.copy(B_ptr[m_idx * BLOCK_M: (m_idx + 1) * BLOCK_M, n_idx], B_smem)
                sem[m_idx].commit()
                stage1_scheduler.next_tile()
        with T.cta()[STAGE_1_SM_CNT: TOTAL_SM_CNT]:
            B_smem = T.alloc_buffer([BLOCK_M, NUM_BLOCK_N], "float32", scope="shared")
            C_smem = T.alloc_buffer([BLOCK_M, 1], "float32", scope="shared")
            stage2_scheduler = T.meta_var(SpatialTileScheduler("stage2", (NUM_BLOCK_M, 1), STAGE_2_SM_CNT))
            stage2_scheduler.init(bx - STAGE_1_SM_CNT)
            while stage2_scheduler.valid():
                m_idx = T.meta_var(stage2_scheduler.m_idx[0])
                sem[m_idx].wait()
                Tp.copy(B_smem, B_ptr[m_idx * BLOCK_M: (m_idx + 1) * BLOCK_M, :])
                Tp.sum(C_smem, B_smem)
                Tp.copy(C_ptr[m_idx * BLOCK_M: (m_idx + 1) * BLOCK_M, 0], C_smem)
                stage2_scheduler.next_tile()                
# fmt: on


@tvm.testing.requires_cuda_compute_version(8)
def test_partial_reduction():
    A_np = np.random.randn(M, N).astype(np.float32)
    B_np = np.zeros((M, NUM_BLOCK_N), dtype=np.float32)
    C_np = np.zeros((M, 1), dtype=np.float32)
    DEV = tvm.cuda(0)
    target = tvm.target.Target("cuda")

    A_tvm = tvm.nd.array(A_np, device=DEV)
    B_tvm = tvm.nd.array(B_np, device=DEV)
    C_tvm = tvm.nd.array(C_np, device=DEV)
    C_tvm_fused = tvm.nd.array(C_np, device=DEV)
    sem_tvm = tvm.nd.array(np.zeros((NUM_BLOCK_M,), dtype=np.int32), device=DEV)
    with target:

        ref_mod_stage_1 = tvm.IRModule({"main": partial_reduction_ref_stage1})
        ref_mod_stage_1 = tvm.compile(ref_mod_stage_1, target=target, tir_pipeline="tirp")
        ref_mod_stage_1(A_tvm, B_tvm)

        ref_mod_stage_2 = tvm.IRModule({"main": partial_reduction_ref_stage2})
        ref_mod_stage_2 = tvm.compile(ref_mod_stage_2, target=target, tir_pipeline="tirp")
        ref_mod_stage_2(B_tvm, C_tvm)
        ret_ref_stage_2 = C_tvm.numpy()

        fused_mod = tvm.IRModule({"main": partial_reduction_fused})
        fused_mod = tvm.compile(fused_mod, target=target, tir_pipeline="tirp")
        fused_mod(A_tvm, B_tvm, C_tvm_fused, sem_tvm)
        ret_fused = C_tvm_fused.numpy()
        tvm.testing.assert_allclose(ret_fused, ret_ref_stage_2, rtol=1e-3, atol=1e-3)


@tvm.testing.requires_cuda_compute_version(8)
def test_partial_reduction_event():
    A_np = np.random.randn(M, N).astype(np.float32)
    B_np = np.zeros((M, NUM_BLOCK_N), dtype=np.float32)
    C_np = np.zeros((M, 1), dtype=np.float32)
    DEV = tvm.cuda(0)
    target = tvm.target.Target("cuda")

    A_tvm = tvm.nd.array(A_np, device=DEV)
    B_tvm = tvm.nd.array(B_np, device=DEV)
    C_tvm = tvm.nd.array(C_np, device=DEV)
    C_tvm_fused = tvm.nd.array(C_np, device=DEV)
    # we initialize event on host instead of device because of Tp.fill not correctly implemented
    sem_tvm = tvm.nd.array(np.full((NUM_BLOCK_M,), NUM_BLOCK_N, dtype=np.int32), device=DEV)
    with target:

        ref_mod_stage_1 = tvm.IRModule({"main": partial_reduction_ref_stage1})
        ref_mod_stage_1 = tvm.compile(ref_mod_stage_1, target=target, tir_pipeline="tirp")
        ref_mod_stage_1(A_tvm, B_tvm)

        ref_mod_stage_2 = tvm.IRModule({"main": partial_reduction_ref_stage2})
        ref_mod_stage_2 = tvm.compile(ref_mod_stage_2, target=target, tir_pipeline="tirp")
        ref_mod_stage_2(B_tvm, C_tvm)
        ret_ref_stage_2 = C_tvm.numpy()

        fused_mod = tvm.IRModule({"main": partial_reduction_fused_event})
        fused_mod = tvm.compile(fused_mod, target=target, tir_pipeline="tirp")
        fused_mod(A_tvm, B_tvm, C_tvm_fused, sem_tvm)
        ret_fused = C_tvm_fused.numpy()
        tvm.testing.assert_allclose(ret_fused, ret_ref_stage_2, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    test_partial_reduction()
    test_partial_reduction_event()
