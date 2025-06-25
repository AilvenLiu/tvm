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
import tvm
import tvm.testing
from tvm.script import tir as T
from tvm.script import tirp as Tp
from tvm.script.ir_builder import IRBuilder
from typing import Tuple
import functools
import enum

@tvm.testing.requires_cuda_compute_version(8)
def test_partial_reduction():

    M = 1024
    N = 1024

    BLOCK_M = 64
    BLOCK_N = 64

    NUM_BLOCK_M = M // BLOCK_M
    NUM_BLOCK_N = N // BLOCK_N


    # fmt: off

    def int_var(name: str, scope="local"):
        buf = T.alloc_buffer([1], "int32", scope=scope, align=4)
        IRBuilder.current().name(name, buf)
        return buf
    
    class TaskType(enum.Enum):
        PARTIAL = 0
        REDUCE = 1
    
    class MPMCQueue:
        def __init__(self, capacity: int, task_types: T.Buffer, task_idxs: T.Buffer, head: T.Buffer, tail: T.Buffer, num_tot_tasks: int):
            if capacity & (capacity - 1):
                raise ValueError("capacity must be a power-of-two")
            self.capacity = capacity
            self.mask = capacity - 1
            self.task_types = task_types
            self.task_idxs = task_idxs
            self.head = head
            self.tail = tail
            self.head_r = int_var("head_r")
            self.tail_r = int_var("tail_r")
            self.pos = int_var("pos")
            self.masked_pos = int_var("masked_pos")
            self.num_tot_tasks = num_tot_tasks
        @T.macro
        def enqueue(self, task_type: int, *task_idx: int):
            self.tail_r[0] = T.cuda.atomic_add(self.tail.access_ptr("rw", offset=self.tail.offset_of_p([T.int32(0)])), 1)
            # TODO: wait if tail - head > capacity
            self.masked_pos[0] = self.tail_r[0] & self.mask
            self.task_types[self.masked_pos[0]] = task_type
            self.task_idxs[self.masked_pos[0], 0] = task_idx[0]
            self.task_idxs[self.masked_pos[0], 1] = task_idx[1]
            T.cuda.thread_fence()                                     # publish data

        @T.macro
        def dequeue(self, fetched_task_type: T.Buffer, fetched_task_idx: T.Buffer):
            self.head_r[0] = T.cuda.atomic_add(self.head.access_ptr("rw", offset=self.head.offset_of_p([T.int32(0)])), 1)
            if self.head_r[0] < self.num_tot_tasks:
              self.masked_pos[0] = self.head_r[0] & self.mask
              T.ptx.ld_global_acquire(fetched_task_type[0], self.task_types.access_ptr("r", offset=self.task_types.offset_of_p([self.masked_pos[0]])))
              while fetched_task_type[0] < 0:
                  T.cuda.nano_sleep(40)
                  T.ptx.ld_global_acquire(fetched_task_type[0], self.task_types.access_ptr("r", offset=self.task_types.offset_of_p([self.masked_pos[0]])))
              self.task_types[self.masked_pos[0]] = -1
              T.cuda.thread_fence()
              for i in T.vectorized(2):
                fetched_task_idx[i] = self.task_idxs[self.masked_pos[0], i]
            else:
              fetched_task_type[0] = -1
        
    class DynamicTileScheduler:
        def __init__(self, queue: MPMCQueue):
            self.queue = queue
            self.fetched_task_type = int_var("fetched_task_type", scope="shared")
            self.fetched_task_idx = T.alloc_buffer([2], "int32", scope="shared", layout="default")
            IRBuilder.current().name("fetched_task_idx", self.fetched_task_idx)
            
        @T.macro
        def _fetch_from_queue(self):
          with T.thread()[0:1]:
            self.queue.dequeue(self.fetched_task_type, self.fetched_task_idx)
          T.tvm_storage_sync("shared")
            
        @T.macro
        def init(self, linear_init):
            self._fetch_from_queue()
            
        @T.macro
        def next_tile(self):
            self._fetch_from_queue()
            
        def valid(self):
            return self.fetched_task_type[0] >= 0
            
    class Semaphore:
        def __init__(self, cnt: int, buffer: T.Buffer, queue: MPMCQueue):
            self.cnt = cnt
            self.sem = buffer
            self.state = int_var("state")
            self.queue = queue
        @T.macro 
        def semaphore_notify(self, *coord):
            with T.thread():
                T.tvm_storage_sync("shared")
                with T.thread()[0:1]:
                    # add 1 because atomic_add returns the old value
                    self.state[0] = T.cuda.atomic_add(self.sem.access_ptr("rw", offset=self.sem.offset_of_p(coord)), 1) + 1
                    if self.state[0] == self.cnt:
                        self.queue.enqueue(TaskType.REDUCE.value, coord[0], 0)
                T.cuda.thread_fence()
                
    
    # reduction on N
    @T.prim_func(tirp=True)
    def partial_reduction_ref_stage1(A: T.handle, B: T.handle):
        A_ptr = T.match_buffer(A, (M, N), "float32", layout="default")
        B_ptr = T.match_buffer(B, (M, NUM_BLOCK_N), "float32", layout="default")
        
        with T.kernel():
            bx, by = T.cta_id([NUM_BLOCK_M, NUM_BLOCK_N], parent="kernel")
            tx = T.thread_id([1024], parent="cta")
            
            with T.cta():
                A_smem = T.alloc_buffer([BLOCK_M, BLOCK_N], "float32", scope="shared", layout="default")
                B_smem = T.alloc_buffer([BLOCK_M, 1], "float32", scope="shared", layout="default")
                Tp.copy(A_smem, A_ptr[bx * BLOCK_M: (bx + 1) * BLOCK_M, by * BLOCK_N: (by + 1) * BLOCK_N])                
                Tp.sum(B_smem, A_smem)
                Tp.copy(B_ptr[bx * BLOCK_M: (bx + 1) * BLOCK_M, by], B_smem)
                

    @T.prim_func(tirp=True)
    def partial_reduction_ref_stage2(B: T.handle, C: T.handle):
        B_ptr = T.match_buffer(B, (M, NUM_BLOCK_N), "float32", layout="default")
        C_ptr = T.match_buffer(C, (M, 1), "float32", layout="default")
        
        with T.kernel():
            bx = T.cta_id([NUM_BLOCK_M], parent="kernel")
            tx = T.thread_id([1024], parent="cta")
            with T.cta():
                B_smem = T.alloc_buffer([BLOCK_M, NUM_BLOCK_N], "float32", scope="shared", layout="default")
                C_smem = T.alloc_buffer([BLOCK_M, 1], "float32", scope="shared", layout="default")
                Tp.copy(B_smem, B_ptr[bx * BLOCK_M: (bx + 1) * BLOCK_M, :])
                Tp.sum(C_smem, B_smem)
                Tp.copy(C_ptr[bx * BLOCK_M: (bx + 1) * BLOCK_M, 0], C_smem)
                
    TOTAL_SM_CNT = 132
    CAPACITY = 1024
    TASK_IDX_LEN = 2
                
    @T.prim_func(tirp=True)
    def partial_reduction_fused(A: T.handle, B: T.handle, C: T.handle, semaphore: T.handle, task_types: T.handle, task_idxs: T.handle, head: T.handle, tail: T.handle):
        A_ptr = T.match_buffer(A, (M, N), "float32", layout="default")
        B_ptr = T.match_buffer(B, (M, NUM_BLOCK_N), "float32", layout="default")
        C_ptr = T.match_buffer(C, (M, 1), "float32", layout="default")
        sem_ptr = T.match_buffer(semaphore, (NUM_BLOCK_M, ), "int32", layout="default")
        task_types_ptr = T.match_buffer(task_types, (CAPACITY, ), "int32", layout="default")
        task_idxs_ptr = T.match_buffer(task_idxs, (CAPACITY, 2), "int32", layout="default")
        head_ptr = T.match_buffer(head, (1, ), "int32", layout="default")
        tail_ptr = T.match_buffer(tail, (1, ), "int32", layout="default")
        with T.kernel():
            bx = T.cta_id([TOTAL_SM_CNT], parent="kernel")
            tx = T.thread_id([1024], parent="cta")
            queue = T.meta_var(MPMCQueue(CAPACITY, task_types_ptr, task_idxs_ptr, head_ptr, tail_ptr, NUM_BLOCK_M * NUM_BLOCK_N + NUM_BLOCK_M))
            sem = T.meta_var(Semaphore(NUM_BLOCK_N, sem_ptr, queue))
            with T.cta():
                A_smem = T.alloc_buffer([BLOCK_M, BLOCK_N], "float32", scope="shared", layout="default")
                B_smem_1 = T.alloc_buffer([BLOCK_M, 1], "float32", scope="shared", layout="default")
                B_smem_2 = T.alloc_buffer([BLOCK_M, NUM_BLOCK_N], "float32", scope="shared", layout="default")
                C_smem = T.alloc_buffer([BLOCK_M, 1], "float32", scope="shared", layout="default")
                tile_scheduler = T.meta_var(DynamicTileScheduler(queue))
                tile_scheduler.init(bx)
                while tile_scheduler.valid():
                    if tile_scheduler.fetched_task_type[0] == TaskType.PARTIAL.value:
                        m_idx = T.meta_var(tile_scheduler.fetched_task_idx[0])
                        n_idx = T.meta_var(tile_scheduler.fetched_task_idx[1])
                        Tp.copy(A_smem, A_ptr[m_idx * BLOCK_M: (m_idx + 1) * BLOCK_M, n_idx * BLOCK_N: (n_idx + 1) * BLOCK_N])                
                        Tp.sum(B_smem_1, A_smem)
                        Tp.copy(B_ptr[m_idx * BLOCK_M: (m_idx + 1) * BLOCK_M, n_idx], B_smem_1)
                        sem.semaphore_notify(m_idx)
                    elif tile_scheduler.fetched_task_type[0] == TaskType.REDUCE.value:
                        m_idx = T.meta_var(tile_scheduler.fetched_task_idx[0])
                        Tp.copy(B_smem_2, B_ptr[m_idx * BLOCK_M: (m_idx + 1) * BLOCK_M, :])
                        Tp.sum(C_smem, B_smem_2)
                        Tp.copy(C_ptr[m_idx * BLOCK_M: (m_idx + 1) * BLOCK_M, 0], C_smem)
                    tile_scheduler.next_tile()

    # fmt: on

    A_np = np.random.randn(M, N).astype(np.float32)
    B_np = np.zeros((M, NUM_BLOCK_N), dtype=np.float32)
    C_np = np.zeros((M, 1), dtype=np.float32)
    DEV = tvm.cuda(0)
    target = tvm.target.Target("cuda")
    
    class MPMCQueueHost:
        def __init__(self, capacity: int):
            self.capacity = capacity
            self.task_types = np.full((capacity,), -1, dtype=np.int32)
            self.task_idxs = np.zeros((capacity, 2), dtype=np.int32)
            self.head = np.zeros((1,), dtype=np.int32)
            self.tail = np.zeros((1,), dtype=np.int32)
            
        def init(self):
            self.head[0] = 0
            self.tail[0] = 0
            
        def enqueue(self, task_type: TaskType, *task_idx: int):
            pos = self.tail[0] & (self.capacity - 1)
            self.task_types[pos] = task_type.value
            for i in range(TASK_IDX_LEN):
                self.task_idxs[pos, i] = task_idx[i]
            self.tail[0] = self.tail[0] + 1
            

    A_tvm = tvm.nd.array(A_np, device=DEV)
    B_tvm = tvm.nd.array(B_np, device=DEV)
    C_tvm = tvm.nd.array(C_np, device=DEV)
    C_tvm_fused = tvm.nd.array(C_np, device=DEV)
    sem_tvm = tvm.nd.array(np.zeros((NUM_BLOCK_M,), dtype=np.int32), device=DEV)
    mpmc_queue = MPMCQueueHost(CAPACITY)
    mpmc_queue.init()
    for i in range(NUM_BLOCK_M):
        for j in range(NUM_BLOCK_N):
            mpmc_queue.enqueue(TaskType.PARTIAL, i, j)
    task_types_tvm = tvm.nd.array(mpmc_queue.task_types, device=DEV)
    task_idxs_tvm = tvm.nd.array(mpmc_queue.task_idxs, device=DEV)
    head_tvm = tvm.nd.array(mpmc_queue.head, device=DEV)
    tail_tvm = tvm.nd.array(mpmc_queue.tail, device=DEV)
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
        fused_mod(A_tvm, B_tvm, C_tvm_fused, sem_tvm, task_types_tvm, task_idxs_tvm, head_tvm, tail_tvm)
        ret_fused = C_tvm_fused.numpy()
        tvm.testing.assert_allclose(ret_fused, ret_ref_stage_2, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    test_partial_reduction()
