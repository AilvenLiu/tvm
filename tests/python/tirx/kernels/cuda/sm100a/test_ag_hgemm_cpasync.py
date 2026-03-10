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
import tempfile

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.runtime import ShapeTuple
from tvm.runtime import disco as di
from tvm.tirx.bench.utils import export_to_perfetto_trace

import os
import sys

sys.path.insert(0, os.path.join(os.environ.get("TIRX_KERNELS_PATH", os.path.expanduser("~/tirx-kernels/kernels")), "gemm"))
import ag_hgemm_cpasync as kernel  # noqa: E402
from ag_hgemm_cpasync import (  # noqa: E402
    CAPACITY,
    CUDA_EVENT_PROFILER,
    GEMM_M_CLUSTERS,
    GEMM_N_CLUSTERS,
    GROUP_SIZE,
    K,
    LOCAL_GEMM_M_CLUSTERS,
    LOCAL_M,
    LOCAL_N,
    M,
    N,
    PROFILER_BUFFER_SIZE,
    PROFILER_ON,
    TASK_IDX_LEN,
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
def test_ag_hgemm():
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
    import torch

    torch.manual_seed(42)
    DEV = tvm.cuda(0)
    A_torch = torch.randn([WORLD_SIZE, LOCAL_M, K], dtype=torch.float16)
    B_torch = torch.randn([WORLD_SIZE, LOCAL_N, K], dtype=torch.float16)

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

    nvshmem_malloc_hook = sess.get_global_func("runtime.disco.nvshmem.empty")
    args_dict = {
        "A_array": sess.empty((LOCAL_M, K), a_type),
        "B_array": sess.empty((LOCAL_N, K), b_type),
        "out_array": sess.empty((M, LOCAL_N), d_type),
    }
    for i in range(TOTAL_ITERS):
        args_dict[f"ag_out_array_{i}"] = nvshmem_malloc_hook(ShapeTuple((M, K)), a_type, None)
        args_dict[f"semaphore_array_{i}"] = nvshmem_malloc_hook(
            ShapeTuple((WORLD_SIZE,)), "uint64", None
        )
        args_dict[f"gemm_task_types_array_{i}"] = sess.empty((CAPACITY,), "int32")
        args_dict[f"gemm_task_idxs_array_{i}"] = sess.empty((CAPACITY, 2), "int32")
        args_dict[f"gemm_head_array_{i}"] = sess.empty((1,), "int32")
        args_dict[f"gemm_tail_array_{i}"] = sess.empty((1,), "int32")
        args_dict[f"profiler_buffer_array_{i}"] = sess.empty((PROFILER_BUFFER_SIZE,), "uint64")

    res_dict = {
        "ag_out_res": sess.empty((WORLD_SIZE, M, K), a_type, worker0_only=True),
        "out_res": sess.empty((WORLD_SIZE, M, LOCAL_N), d_type, worker0_only=True),
        "profiler_buffer_res": sess.empty(
            (WORLD_SIZE, PROFILER_BUFFER_SIZE), "uint64", worker0_only=True
        ),
        "ag_out_host": tvm.runtime.empty((WORLD_SIZE, M, K), a_type, device=DEV),
        "out_host": tvm.runtime.empty((WORLD_SIZE, M, LOCAL_N), d_type, device=DEV),
        "profiler_buffer_host": tvm.runtime.empty(
            (WORLD_SIZE, PROFILER_BUFFER_SIZE), "uint64", device=DEV
        ),
    }

    A_tvm = tvm.runtime.tensor(A_torch, device=DEV)
    B_tvm = tvm.runtime.tensor(B_torch, device=DEV)
    A_array = sess.empty((WORLD_SIZE, LOCAL_M, K), a_type, worker0_only=True)
    B_array = sess.empty((WORLD_SIZE, LOCAL_N, K), b_type, worker0_only=True)
    sess.copy_to_worker_0(A_tvm, A_array)
    sess.copy_to_worker_0(B_tvm, B_array)
    sess.scatter_from_worker0(A_array, args_dict["A_array"])
    sess.scatter_from_worker0(B_array, args_dict["B_array"])

    task_types_np = np.empty((WORLD_SIZE, CAPACITY), dtype=np.int32)
    task_idxs_np = np.empty((WORLD_SIZE, CAPACITY, 2), dtype=np.int32)
    head_np = np.empty((WORLD_SIZE,), dtype=np.int32)
    tail_np = np.empty((WORLD_SIZE,), dtype=np.int32)
    for rank in range(WORLD_SIZE):
        gemm_mpmc_queue = MPMCQueueHost(CAPACITY)
        gemm_mpmc_queue.init()
        offset = rank * LOCAL_GEMM_M_CLUSTERS
        for g in range(math.ceil(GEMM_M_CLUSTERS / GROUP_SIZE)):
            for i in range(GEMM_N_CLUSTERS):
                for j in range(g * GROUP_SIZE, min((g + 1) * GROUP_SIZE, GEMM_M_CLUSTERS)):
                    gemm_mpmc_queue.enqueue(TaskType.GEMM, (offset + j) % GEMM_M_CLUSTERS, i)

        task_types_np[rank, :] = gemm_mpmc_queue.task_types
        task_idxs_np[rank, :, :] = gemm_mpmc_queue.task_idxs
        head_np[rank] = gemm_mpmc_queue.head[0]
        tail_np[rank] = gemm_mpmc_queue.tail[0]

    task_types_tvm = tvm.runtime.tensor(task_types_np, device=DEV)
    task_idxs_tvm = tvm.runtime.tensor(task_idxs_np, device=DEV)
    head_tvm = tvm.runtime.tensor(head_np, device=DEV)
    tail_tvm = tvm.runtime.tensor(tail_np, device=DEV)

    task_types_array = sess.empty((WORLD_SIZE, CAPACITY), "int32", worker0_only=True)
    task_idxs_array = sess.empty((WORLD_SIZE, CAPACITY, 2), "int32", worker0_only=True)
    head_array = sess.empty((WORLD_SIZE,), "int32", worker0_only=True)
    tail_array = sess.empty((WORLD_SIZE,), "int32", worker0_only=True)
    sess.copy_to_worker_0(task_types_tvm, task_types_array)
    sess.copy_to_worker_0(task_idxs_tvm, task_idxs_array)
    sess.copy_to_worker_0(head_tvm, head_array)
    sess.copy_to_worker_0(tail_tvm, tail_array)

    for i in range(TOTAL_ITERS):
        sess.scatter_from_worker0(task_types_array, args_dict[f"gemm_task_types_array_{i}"])
        sess.scatter_from_worker0(task_idxs_array, args_dict[f"gemm_task_idxs_array_{i}"])
        sess.scatter_from_worker0(head_array, args_dict[f"gemm_head_array_{i}"])
        sess.scatter_from_worker0(tail_array, args_dict[f"gemm_tail_array_{i}"])

    sess.sync_worker_0()
    print("Data prepared successfully")

    with tempfile.TemporaryDirectory() as tmpdir:
        target = tvm.target.Target("cuda")
        path = tmpdir + "/test.so"
        mod = tvm.compile(test_mma_ss_tma_2sm_persistent, target=target, tir_pipeline="tirx")
        # print(mod.mod.imports[0].inspect_source())
        mod.export_library(path)

        print("Begin kernel execution...")
        rt_mod = sess.load_vm_module(path)
        barrier_dfunc = sess.get_global_func("runtime.disco.nvshmem.barrier_all_on_stream")
        transfer_to_peers_dfunc = sess.get_global_func("runtime.disco.transfer_to_peers_all_gather")
        stream_create_dfunc = sess.get_global_func("runtime.disco.stream_create")
        d_stream = stream_create_dfunc()
        stream_sync_dfunc = sess.get_global_func("runtime.disco.stream_sync")
        cur_stream = sess.get_global_func("runtime.get_cuda_stream")()
        if CUDA_EVENT_PROFILER:
            timer_create_dfunc = sess.get_global_func("profiling.cuda.event.create")
            timer_start_dfunc = sess.get_global_func("profiling.cuda.event.start")
            timer_stop_dfunc = sess.get_global_func("profiling.cuda.event.stop")
            timer_result_dfunc = sess.get_global_func("profiling.cuda.event.elapsed")
            timer = timer_create_dfunc()
        sess._sync_all()

        for itr in range(TOTAL_ITERS):
            if CUDA_EVENT_PROFILER and itr == WARMUP_ITERS:
                timer_start_dfunc(timer)
            barrier_dfunc(cur_stream)
            stream_sync_dfunc(cur_stream, d_stream)
            transfer_to_peers_dfunc(
                args_dict[f"semaphore_array_{itr}"],
                args_dict["A_array"],
                args_dict[f"ag_out_array_{itr}"],
                d_stream,
                M,
                K,
                WORLD_SIZE,
            )
            rt_mod["test_mma_ss_tma_2sm_persistent"](
                args_dict["A_array"],
                args_dict["B_array"],
                args_dict[f"ag_out_array_{itr}"],
                args_dict[f"semaphore_array_{itr}"],
                args_dict["out_array"],
                args_dict[f"profiler_buffer_array_{itr}"],
                args_dict[f"gemm_task_types_array_{itr}"],
                args_dict[f"gemm_task_idxs_array_{itr}"],
                args_dict[f"gemm_head_array_{itr}"],
                args_dict[f"gemm_tail_array_{itr}"],
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
            print(f"AG GEMM duration: {timer_res_np.max():.5f} ms")
            for rank in range(WORLD_SIZE):
                print(f"rank {rank}: {timer_res_np[rank]:.5f} ms")

        sess.gather_to_worker0(args_dict[f"ag_out_array_{TOTAL_ITERS - 1}"], res_dict["ag_out_res"])
        sess.copy_from_worker_0(res_dict["ag_out_host"], res_dict["ag_out_res"])
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

        ag_out_ref = A_torch.reshape(-1, K)
        for rank in range(WORLD_SIZE):
            print(f"validating rank: {rank}")
            ag_out_res = res_dict["ag_out_host"].numpy()[rank]
            ag_out_res[rank * LOCAL_M : (rank + 1) * LOCAL_M, :] = ag_out_ref[
                rank * LOCAL_M : (rank + 1) * LOCAL_M, :
            ]
            np.testing.assert_equal(ag_out_ref, ag_out_res)

            ref = torch.matmul(ag_out_ref.cuda(), (B_torch[rank].T).cuda())
            out_res = res_dict["out_host"].numpy()[rank]
            np.testing.assert_allclose(out_res, ref.cpu().numpy(), atol=1e-3, rtol=1e-3)

        print("Results all correct.")

    # profiler results
    if PROFILER_ON:
        for rank in range(WORLD_SIZE):
            export_to_perfetto_trace(
                res_dict["profiler_buffer_host"].numpy()[rank],
                f"dyn-schedule-AG-hgemm-rank{rank}.perfetto-trace",
                event_type_names,
            )


if __name__ == "__main__":
    test_ag_hgemm()
