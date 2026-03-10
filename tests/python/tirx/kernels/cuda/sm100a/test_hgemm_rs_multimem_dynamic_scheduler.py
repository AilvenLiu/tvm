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

import math
import os
import sys
import tempfile

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.runtime import ShapeTuple
from tvm.runtime import disco as di
from tvm.tirx.bench.utils import export_to_perfetto_trace

sys.path.insert(0, os.path.join(os.environ.get("TIRX_KERNELS_PATH", os.path.expanduser("~/tirx-kernels/kernels")), "gemm"))
import hgemm_rs_multimem_dynamic_scheduler as kernel  # noqa: E402
from hgemm_rs_multimem_dynamic_scheduler import (  # noqa: E402
    CAPACITY,
    CUDA_EVENT_PROFILER,
    GEMM_M_CLUSTERS,
    GEMM_N_CLUSTERS,
    GROUP_SIZE,
    K,
    LOCAL_M,
    M,
    N,
    PROFILER_BUFFER_SIZE,
    PROFILER_ON,
    RS_M_CLUSTERS,
    RS_N_CLUSTERS,
    SM_NUMBER,
    TASK_IDX_LEN,
    TILE_M,
    TILE_N,
    TOTAL_ITERS,
    VALIDATE,
    WARMUP_ITERS,
    WORLD_SIZE,
    TaskType,
    a_type,
    b_type,
    d_type,
    event_type_names,
    test_mma_ss_tma_2sm_persistent,
)


@pytest.mark.skip()
@tvm.testing.requires_cuda_compute_version(10, exact=True)
def test_hgemm_rs():
    devices = list(np.arange(WORLD_SIZE))
    sess = di.ProcessSession(num_workers=WORLD_SIZE)
    sess.init_ccl(tvm.get_global_func("runtime.disco.compiled_ccl")(), *devices)
    f_init_nvshmem_uid = tvm.get_global_func("runtime.disco.nvshmem.init_nvshmem_uid")
    uid = f_init_nvshmem_uid()
    init_dfunc = sess.get_global_func("runtime.disco.nvshmem.init_nvshmem")
    init_dfunc(uid, WORLD_SIZE, 0)
    sess.sync_worker_0()

    # prepare test data
    print("begin preparing test data ...")
    np.random.seed(42)
    DEV = tvm.cuda(0)
    A_np = np.random.uniform(-1, 1, (WORLD_SIZE, M, K)).astype(a_type)
    B_np = np.random.uniform(-1, 1, (WORLD_SIZE, N, K)).astype(b_type)
    A_tvm = tvm.runtime.tensor(A_np, device=DEV)
    B_tvm = tvm.runtime.tensor(B_np, device=DEV)

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

    gemm_mpmc_queue = MPMCQueueHost(CAPACITY)
    gemm_mpmc_queue.init()
    # push in initial tasks for stage 1, because they are ready
    for g in range(math.ceil(GEMM_N_CLUSTERS / GROUP_SIZE)):
        for i in range(GEMM_M_CLUSTERS):
            for j in range(g * GROUP_SIZE, min((g + 1) * GROUP_SIZE, GEMM_N_CLUSTERS)):
                gemm_mpmc_queue.enqueue(TaskType.GEMM, i, j)
    rs_mpmc_queue = MPMCQueueHost(CAPACITY)
    rs_mpmc_queue.init()

    A_array_all = sess.empty((WORLD_SIZE, M, K), a_type, worker0_only=True)
    B_array_all = sess.empty((WORLD_SIZE, N, K), b_type, worker0_only=True)
    sess.copy_to_worker_0(A_tvm, A_array_all)
    sess.copy_to_worker_0(B_tvm, B_array_all)

    nvshmem_malloc_hook = sess.get_global_func("runtime.disco.nvshmem.empty")
    args_dict = {
        "A_array": sess.empty((M, K), a_type),
        "B_array": sess.empty((N, K), b_type),
        "gemm_out_array": nvshmem_malloc_hook(ShapeTuple((M, N)), d_type, None),
        "out_array": sess.empty((LOCAL_M, N), d_type),
    }
    for i in range(TOTAL_ITERS):
        args_dict[f"semaphore_array_{i}"] = nvshmem_malloc_hook(
            ShapeTuple((LOCAL_M // TILE_M, N // TILE_N)), "uint64", None
        )
        args_dict[f"gemm_task_types_array_{i}"] = sess.empty((CAPACITY,), "int32")
        args_dict[f"gemm_task_idxs_array_{i}"] = sess.empty((CAPACITY, 2), "int32")
        args_dict[f"gemm_head_array_{i}"] = sess.empty((1,), "int32")
        args_dict[f"gemm_tail_array_{i}"] = sess.empty((1,), "int32")
        args_dict[f"rs_task_types_array_{i}"] = nvshmem_malloc_hook(
            ShapeTuple((CAPACITY,)), "int32", None
        )
        args_dict[f"rs_task_idxs_array_{i}"] = nvshmem_malloc_hook(
            ShapeTuple((CAPACITY, 2)), "int32", None
        )
        args_dict[f"rs_head_array_{i}"] = nvshmem_malloc_hook(ShapeTuple((1,)), "int32", None)
        args_dict[f"rs_tail_array_{i}"] = nvshmem_malloc_hook(ShapeTuple((1,)), "int32", None)
        args_dict[f"profiler_buffer_array_{i}"] = sess.empty((PROFILER_BUFFER_SIZE,), "uint64")

    res_dict = {
        "gemm_out_res": sess.empty((WORLD_SIZE, M, N), d_type, worker0_only=True),
        "out_res": sess.empty((WORLD_SIZE, LOCAL_M, N), d_type, worker0_only=True),
        "profiler_buffer_res": sess.empty(
            (WORLD_SIZE, PROFILER_BUFFER_SIZE), "uint64", worker0_only=True
        ),
        "gemm_out_host": tvm.runtime.empty((WORLD_SIZE, M, N), d_type, device=DEV),
        "out_host": tvm.runtime.empty((WORLD_SIZE, LOCAL_M, N), d_type, device=DEV),
        "profiler_buffer_host": tvm.runtime.empty(
            (WORLD_SIZE, PROFILER_BUFFER_SIZE), "uint64", device=DEV
        ),
    }

    for i in range(TOTAL_ITERS):
        sess.broadcast(gemm_mpmc_queue.task_types, args_dict[f"gemm_task_types_array_{i}"])
        sess.broadcast(gemm_mpmc_queue.task_idxs, args_dict[f"gemm_task_idxs_array_{i}"])
        sess.broadcast(gemm_mpmc_queue.head, args_dict[f"gemm_head_array_{i}"])
        sess.broadcast(gemm_mpmc_queue.tail, args_dict[f"gemm_tail_array_{i}"])
        sess.broadcast(rs_mpmc_queue.task_types, args_dict[f"rs_task_types_array_{i}"])
        sess.broadcast(rs_mpmc_queue.task_idxs, args_dict[f"rs_task_idxs_array_{i}"])
        sess.broadcast(rs_mpmc_queue.head, args_dict[f"rs_head_array_{i}"])
        sess.broadcast(rs_mpmc_queue.tail, args_dict[f"rs_tail_array_{i}"])

    sess.scatter_from_worker0(A_array_all, args_dict["A_array"])
    sess.scatter_from_worker0(B_array_all, args_dict["B_array"])
    sess.sync_worker_0()
    print("Data prepared successfully")

    with tempfile.TemporaryDirectory() as tmpdir:
        target = tvm.target.Target("cuda")
        path = tmpdir + "/test.so"
        mod = tvm.compile(test_mma_ss_tma_2sm_persistent, target=target, tir_pipeline="tirx")
        print(mod.mod.imports[0].inspect_source())
        mod.export_library(path)

        print("Begin kernel execution...")
        rt_mod = sess.load_vm_module(path)
        barrier_dfunc = sess.get_global_func("runtime.disco.nvshmem.barrier_all_on_current_stream")
        if CUDA_EVENT_PROFILER:
            timer_create_dfunc = sess.get_global_func("profiling.cuda.event.create")
            timer_start_dfunc = sess.get_global_func("profiling.cuda.event.start")
            timer_stop_dfunc = sess.get_global_func("profiling.cuda.event.stop")
            timer_result_dfunc = sess.get_global_func("profiling.cuda.event.elapsed")
            timer = timer_create_dfunc()
        sess._sync_all()
        barrier_dfunc()

        for itr in range(TOTAL_ITERS):
            if CUDA_EVENT_PROFILER and itr == WARMUP_ITERS:
                timer_start_dfunc(timer)
            rt_mod["test_mma_ss_tma_2sm_persistent"](
                args_dict["A_array"],
                args_dict["B_array"],
                args_dict["gemm_out_array"],
                args_dict[f"semaphore_array_{itr}"],
                args_dict["out_array"],
                args_dict[f"profiler_buffer_array_{itr}"],
                args_dict[f"gemm_task_types_array_{itr}"],
                args_dict[f"gemm_task_idxs_array_{itr}"],
                args_dict[f"gemm_head_array_{itr}"],
                args_dict[f"gemm_tail_array_{itr}"],
                args_dict[f"rs_task_types_array_{itr}"],
                args_dict[f"rs_task_idxs_array_{itr}"],
                args_dict[f"rs_head_array_{itr}"],
                args_dict[f"rs_tail_array_{itr}"],
            )

        # get results
        if CUDA_EVENT_PROFILER:
            timer_stop_dfunc(timer)
            timer_res = timer_result_dfunc(timer)
        sess._sync_all()

        if CUDA_EVENT_PROFILER:
            timer_res_np = np.zeros((WORLD_SIZE,), dtype=np.float64)
            for rank in range(WORLD_SIZE):
                timer_res_np[rank] = (
                    timer_res.debug_get_from_remote(rank) / (TOTAL_ITERS - WARMUP_ITERS) / 1e6
                )
            print(f"GEMM RS duration: {timer_res_np.max():.5f} ms")
            for rank in range(WORLD_SIZE):
                print(f"rank {rank}: {timer_res_np[rank]:.5f} ms")

        sess.gather_to_worker0(args_dict["gemm_out_array"], res_dict["gemm_out_res"])
        sess.copy_from_worker_0(res_dict["gemm_out_host"], res_dict["gemm_out_res"])
        sess.gather_to_worker0(args_dict["out_array"], res_dict["out_res"])
        sess.copy_from_worker_0(res_dict["out_host"], res_dict["out_res"])
        sess.gather_to_worker0(
            args_dict[f"profiler_buffer_array_{TOTAL_ITERS - 1}"], res_dict["profiler_buffer_res"]
        )
        sess.copy_from_worker_0(res_dict["profiler_buffer_host"], res_dict["profiler_buffer_res"])

        # sync all workers to make sure the temporary files are cleaned up after all workers
        # finish the execution
        sess._sync_all()
        print("Kernel execution finished.")

    finalize_dfunc = sess.get_global_func("runtime.disco.nvshmem.finalize_nvshmem")
    finalize_dfunc()
    sess.sync_worker_0()

    if VALIDATE:
        print("Validating results...")

        import torch

        gemm_out_torch = torch.zeros((WORLD_SIZE, M, N), dtype=torch.float16, device="cuda")
        gemm_out_torch_sum = torch.zeros((M, N), dtype=torch.float16, device="cuda")
        for i in range(WORLD_SIZE):
            print(f"rank {i} validating...")
            A_torch = torch.tensor(A_np[i], dtype=torch.float16, device="cuda")
            B_torch = torch.tensor(B_np[i], dtype=torch.float16, device="cuda")
            gemm_out_torch[i] = torch.matmul(A_torch, B_torch.T)
            gemm_out_res = res_dict["gemm_out_host"].numpy()[i]
            np.testing.assert_allclose(
                gemm_out_res, gemm_out_torch[i].cpu().numpy(), atol=1e-3, rtol=1e-3
            )

        gemm_out_torch_sum = torch.sum(gemm_out_torch, dim=0)
        out_res = res_dict["out_host"].numpy().reshape(-1, N)
        np.testing.assert_allclose(out_res, gemm_out_torch_sum.cpu().numpy(), atol=1e-3, rtol=1e-3)

        print("Results all correct.")

    # profiler results
    if PROFILER_ON:
        for rank in range(WORLD_SIZE):
            export_to_perfetto_trace(
                res_dict["profiler_buffer_host"].numpy()[rank],
                f"dyn-schedule-hgemm-RS-rank{rank}.perfetto-trace",
                event_type_names,
            )


if __name__ == "__main__":
    test_hgemm_rs()
