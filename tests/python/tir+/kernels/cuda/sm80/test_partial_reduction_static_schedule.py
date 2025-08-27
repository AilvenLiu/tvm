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
import enum
import functools
from typing import Tuple

import numpy as np

import tvm
import tvm.testing
from tvm.script import tir as T
from tvm.script import tirp as Tp
from tvm.script.ir_builder import IRBuilder


def extract_values(filename):
    # List to store all extracted values
    extracted_values = []

    # Open and read the file
    with open(filename, "r") as file:
        for line in file:
            # Use string manipulation to extract values
            parts = line.strip().split(", ")

            # Extract value a from "block_id: a"
            a_part = parts[0].split(": ")[1]
            a = int(a_part)

            # Extract value b from "fetched_task_type[0]: b"
            b_part = parts[1].split(": ")[1]
            b = int(b_part)

            # Extract value c from "fetched_task_idx[0]: c"
            c_part = parts[2].split(": ")[1]
            c = int(c_part)

            # Extract value d from "fetched_task_idx[1]: d"
            d_part = parts[3].split(": ")[1]
            d = int(d_part)

            # Add the extracted values as a tuple to the result list
            extracted_values.append((a, b, c, d))

    return extracted_values


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
    
        
    class StaticTileScheduler:
        def __init__(self, tasks_indptr: T.Buffer, task_types: T.Buffer, task_indices: T.Buffer):
           self.linear_idx = int_var("linear_idx")
           self.linear_lim = int_var("linear_lim")
           self.task_types = task_types
           self.task_indices = task_indices
           self.tasks_indptr = tasks_indptr
           self.task_type = int_var("task_type")
           self.task_idx = T.alloc_buffer([2], "int32", scope="local")
           IRBuilder.current().name("task_idx", self.task_idx)

        @T.macro
        def get_block_coord(self):
            self.task_type[0] = self.task_types[self.linear_idx[0]]
            self.task_idx[0] = self.task_indices[self.linear_idx[0], 0]
            self.task_idx[1] = self.task_indices[self.linear_idx[0], 1]
            
        @T.macro
        def init(self, linear_init):
            self.linear_idx[0] = self.tasks_indptr[linear_init]
            self.linear_lim[0] = self.tasks_indptr[linear_init + 1]
            self.get_block_coord()
            
        @T.macro
        def next_tile(self):
            self.linear_idx[0] = self.linear_idx[0] + 1
            self.get_block_coord()

        def valid(self):
            return self.linear_idx[0] < self.linear_lim[0]
            
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
                
    TOTAL_SM_CNT = 132
    CAPACITY = 1024
    TASK_IDX_LEN = 2
                
    @T.prim_func(tirp=True)
    def partial_reduction_fused(A: T.handle, B: T.handle, C: T.handle, semaphore: T.handle, task_types: T.handle, task_indices: T.handle, tasks_indptr: T.handle):
        A_ptr = T.match_buffer(A, (M, N), "float32")
        B_ptr = T.match_buffer(B, (M, NUM_BLOCK_N), "float32")
        C_ptr = T.match_buffer(C, (M, 1), "float32")
        sem_ptr = T.match_buffer(semaphore, (NUM_BLOCK_M, ), "int32")
        task_types_ptr = T.match_buffer(task_types, (NUM_BLOCK_M * NUM_BLOCK_N + NUM_BLOCK_M, ), "int32")
        task_indices_ptr = T.match_buffer(task_indices, (NUM_BLOCK_M * NUM_BLOCK_N + NUM_BLOCK_M, 2), "int32")
        tasks_indptr_ptr = T.match_buffer(tasks_indptr, (TOTAL_SM_CNT + 1, ), "int32")
        with T.kernel():
            bx = T.cta_id([TOTAL_SM_CNT], parent="kernel")
            tx = T.thread_id([1024], parent="cta")
            sem = T.meta_var(Semaphore(NUM_BLOCK_N, sem_ptr))
            with T.cta():
                A_smem = T.alloc_buffer([BLOCK_M, BLOCK_N], "float32", scope="shared")
                B_smem_1 = T.alloc_buffer([BLOCK_M, 1], "float32", scope="shared")
                B_smem_2 = T.alloc_buffer([BLOCK_M, NUM_BLOCK_N], "float32", scope="shared")
                C_smem = T.alloc_buffer([BLOCK_M, 1], "float32", scope="shared")
                tile_scheduler = T.meta_var(StaticTileScheduler(tasks_indptr_ptr, task_types_ptr, task_indices_ptr))
                tile_scheduler.init(bx)
                while tile_scheduler.valid():
                    if tile_scheduler.task_type[0] == TaskType.PARTIAL.value:
                        m_idx = T.meta_var(tile_scheduler.task_idx[0])
                        n_idx = T.meta_var(tile_scheduler.task_idx[1])
                        Tp.copy(A_smem, A_ptr[m_idx * BLOCK_M: (m_idx + 1) * BLOCK_M, n_idx * BLOCK_N: (n_idx + 1) * BLOCK_N])                
                        Tp.sum(B_smem_1, A_smem)
                        Tp.copy(B_ptr[m_idx * BLOCK_M: (m_idx + 1) * BLOCK_M, n_idx], B_smem_1)
                        sem.semaphore_notify(m_idx)
                    elif tile_scheduler.task_type[0] == TaskType.REDUCE.value:
                        m_idx = T.meta_var(tile_scheduler.task_idx[0])
                        sem.semaphore_wait(m_idx)
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

    A_tvm = tvm.nd.array(A_np, device=DEV)
    B_tvm = tvm.nd.array(B_np, device=DEV)
    C_tvm = tvm.nd.array(C_np, device=DEV)
    C_tvm_fused = tvm.nd.array(C_np, device=DEV)
    sem_tvm = tvm.nd.array(np.zeros((NUM_BLOCK_M,), dtype=np.int32), device=DEV)
    from pathlib import Path

    script_directory = Path(__file__).parent.resolve()
    file_path = script_directory / "partial_reduction_sm_log.txt"
    sm_trace = extract_values(file_path)
    sm_trace = sorted(sm_trace, key=lambda x: x[0])
    indices = np.zeros((TOTAL_SM_CNT + 1,), dtype=np.int32)
    for num in sm_trace:
        indices[num[0] + 1] += 1
    indptr = np.cumsum(indices)
    task_types = np.array(list(map(lambda x: x[1], sm_trace)), dtype=np.int32)
    task_indices = np.array(list(map(lambda x: x[2:], sm_trace)), dtype=np.int32)
    task_types_tvm = tvm.nd.array(task_types, device=DEV)
    task_indices_tvm = tvm.nd.array(task_indices, device=DEV)
    task_indptr_tvm = tvm.nd.array(np.array(indptr, dtype=np.int32), device=DEV)

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
        fused_mod(
            A_tvm, B_tvm, C_tvm_fused, sem_tvm, task_types_tvm, task_indices_tvm, task_indptr_tvm
        )
        ret_fused = C_tvm_fused.numpy()
        tvm.testing.assert_allclose(ret_fused, ret_ref_stage_2, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    test_partial_reduction()
