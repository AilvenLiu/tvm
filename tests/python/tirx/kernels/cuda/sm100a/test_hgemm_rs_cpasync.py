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

import tempfile

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.runtime import ShapeTuple
from tvm.runtime import disco as di

import os
import sys

sys.path.insert(0, os.path.join(os.environ.get("TIRX_KERNELS_PATH", os.path.expanduser("~/tirx-kernels/kernels")), "gemm"))
from hgemm_rs_cpasync import (  # noqa: E402
    BLK_M,
    BLK_N,
    BLK_N_RS,
    BLK_M_RS,
    GEMM_SMS,
    K,
    LOCAL_M,
    M,
    N,
    N_REPEAT,
    PROFILER_BUFFER_SIZE,
    ReduceScatter,
    SM_COUNT,
    WORLD_SIZE,
    a_type,
    b_type,
    d_type,
    event_type_names,
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
    # A_np = np.random.rand(WORLD_SIZE, M, K).astype(a_type)
    # B_np = np.random.rand(WORLD_SIZE, N, K).astype(b_type)
    A_np = np.random.uniform(-1, 1, (WORLD_SIZE, M, K)).astype(a_type)
    B_np = np.random.uniform(-1, 1, (WORLD_SIZE, N, K)).astype(b_type)
    A_tvm = tvm.runtime.tensor(A_np, device=DEV)
    B_tvm = tvm.runtime.tensor(B_np, device=DEV)
    semaphore_np = np.zeros((WORLD_SIZE,), dtype="uint64")
    profiler_buffer_np = np.zeros((PROFILER_BUFFER_SIZE,), dtype="uint64")

    A_array_all = sess.empty((WORLD_SIZE, M, K), a_type, worker0_only=True)
    B_array_all = sess.empty((WORLD_SIZE, N, K), b_type, worker0_only=True)
    sess.copy_to_worker_0(A_tvm, A_array_all)
    sess.copy_to_worker_0(B_tvm, B_array_all)

    transfer_to_peers_dfunc = sess.get_global_func("runtime.disco.transfer_to_peers_reduce_scatter")
    stream_create_dfunc = sess.get_global_func("runtime.disco.stream_create")
    d_stream = stream_create_dfunc()
    stream_sync_dfunc = sess.get_global_func("runtime.disco.stream_sync")
    cur_stream = sess.get_global_func("runtime.get_cuda_stream")()
    nvshmem_malloc_hook = sess.get_global_func("runtime.disco.nvshmem.empty")
    args_dict = {
        "A_array": sess.empty((M, K), a_type),
        "B_array": sess.empty((N, K), b_type),
        "gemm_out_array": nvshmem_malloc_hook(ShapeTuple((M, N)), d_type, None),
        "semaphore_array": nvshmem_malloc_hook(ShapeTuple((WORLD_SIZE,)), "uint64", None),
        "staging_buffer_array": nvshmem_malloc_hook(
            ShapeTuple((WORLD_SIZE, LOCAL_M, N)), d_type, None
        ),
        "out_array": sess.empty((LOCAL_M, N), d_type),
        "profiler_buffer_array": sess.empty((PROFILER_BUFFER_SIZE,), "uint64"),
    }

    res_dict = {
        "gemm_out_res": sess.empty((WORLD_SIZE, M, N), d_type, worker0_only=True),
        "buffer_res": sess.empty(
            (WORLD_SIZE, M // BLK_M, N // BLK_N, BLK_M, BLK_N), d_type, worker0_only=True
        ),
        "out_res": sess.empty((WORLD_SIZE, LOCAL_M, N), d_type, worker0_only=True),
        "profiler_buffer_res": sess.empty(
            (WORLD_SIZE, PROFILER_BUFFER_SIZE), "uint64", worker0_only=True
        ),
        "staging_buffer_res": sess.empty(
            (WORLD_SIZE, WORLD_SIZE, LOCAL_M, N), d_type, worker0_only=True
        ),
        "gemm_out_host": tvm.runtime.empty((WORLD_SIZE, M, N), d_type, device=DEV),
        "buffer_host": tvm.runtime.empty(
            (WORLD_SIZE, M // BLK_M, N // BLK_N, BLK_M, BLK_N), d_type, device=DEV
        ),
        "out_host": tvm.runtime.empty((WORLD_SIZE, LOCAL_M, N), d_type, device=DEV),
        "profiler_buffer_host": tvm.runtime.empty(
            (WORLD_SIZE, PROFILER_BUFFER_SIZE), "uint64", device=DEV
        ),
        "staging_buffer_host": tvm.runtime.empty(
            (WORLD_SIZE, WORLD_SIZE, LOCAL_M, N), d_type, device=DEV
        ),
    }

    sess.scatter_from_worker0(A_array_all, args_dict["A_array"])
    sess.scatter_from_worker0(B_array_all, args_dict["B_array"])
    sess.sync_worker_0()
    print("Data prepared successfully")

    with tempfile.TemporaryDirectory() as tmpdir:
        target = tvm.target.Target("cuda")
        path = tmpdir + "/test.so"
        mod = tvm.compile(ReduceScatter, target=target, tir_pipeline="tirx")
        print(mod.mod.imports[0].inspect_source())
        mod.export_library(path)

        rt_mod = sess.load_vm_module(path)
        barrier_dfunc = sess.get_global_func("runtime.disco.nvshmem.barrier_all_on_stream")
        print("Begin kernel execution...")
        sess._sync_all()

        for itr in range(N_REPEAT):
            barrier_dfunc(cur_stream)
            stream_sync_dfunc(cur_stream, d_stream)
            rt_mod["test_mma_ss_tma_2sm_persistent"](
                args_dict["A_array"],
                args_dict["B_array"],
                args_dict["gemm_out_array"],
                args_dict["semaphore_array"],
                args_dict["out_array"],
                args_dict["profiler_buffer_array"],
            )
            transfer_to_peers_dfunc(
                args_dict["semaphore_array"],
                args_dict["gemm_out_array"],
                args_dict["staging_buffer_array"],
                d_stream,
                M,
                N,
                BLK_M,
                BLK_N,
                WORLD_SIZE,
            )
            barrier_dfunc(d_stream)
            stream_sync_dfunc(d_stream, cur_stream)
            rt_mod["reduce_sum"](
                args_dict["staging_buffer_array"],
                args_dict["out_array"],
            )
            sess._sync_all()
            if itr < N_REPEAT - 1:
                sess.broadcast(semaphore_np, args_dict["semaphore_array"])
                sess.broadcast(profiler_buffer_np, args_dict["profiler_buffer_array"])
                sess._sync_all()

        # validate results
        sess.gather_to_worker0(args_dict["gemm_out_array"], res_dict["gemm_out_res"])
        sess.copy_from_worker_0(res_dict["gemm_out_host"], res_dict["gemm_out_res"])
        sess.gather_to_worker0(args_dict["out_array"], res_dict["out_res"])
        sess.copy_from_worker_0(res_dict["out_host"], res_dict["out_res"])
        sess.gather_to_worker0(args_dict["profiler_buffer_array"], res_dict["profiler_buffer_res"])
        sess.copy_from_worker_0(res_dict["profiler_buffer_host"], res_dict["profiler_buffer_res"])
        sess.gather_to_worker0(args_dict["staging_buffer_array"], res_dict["staging_buffer_res"])
        sess.copy_from_worker_0(res_dict["staging_buffer_host"], res_dict["staging_buffer_res"])

        print(args_dict["semaphore_array"].debug_get_from_remote(0))

        # sync all workers to make sure the temporary files are cleaned up after all workers
        # finish the execution
        sess._sync_all()
        print("Kernel execution finished.")

    finalize_dfunc = sess.get_global_func("runtime.disco.nvshmem.finalize_nvshmem")
    finalize_dfunc()
    sess.sync_worker_0()

    # validate results
    print("Validating results...")

    import torch

    gemm_out_torch = torch.zeros((WORLD_SIZE, M, N), dtype=torch.float16, device="cuda")
    gemm_out_torch_sum = torch.zeros((M, N), dtype=torch.float16, device="cuda")
    for i in range(WORLD_SIZE):
        print(f"rank {i} validating...")
        A_torch = torch.tensor(A_np[i], dtype=torch.float16, device="cuda")
        B_torch = torch.tensor(B_np[i], dtype=torch.float16, device="cuda")
        gemm_out_torch[i] = torch.matmul(A_torch, B_torch.T)
        # gemm_out_res = res_dict["gemm_out_host"].numpy()[i]
        # np.testing.assert_allclose(gemm_out_res, gemm_out_torch[i].cpu().numpy(), atol=1e-3, rtol=1e-3)  # noqa: E501

    # staging_buffer_torch = gemm_out_torch.reshape(WORLD_SIZE, WORLD_SIZE, LOCAL_M, N).transpose(0, 1)  # noqa: E501
    # np.testing.assert_allclose(staging_buffer_torch.cpu().numpy(), res_dict["staging_buffer_host"].numpy(), atol=1e-3, rtol=1e-3)  # noqa: E501

    gemm_out_torch_sum = torch.sum(gemm_out_torch, dim=0)
    out_res = res_dict["out_host"].numpy().reshape(-1, N)
    gemm_out_torch_sum.cpu().numpy()

    np.testing.assert_allclose(out_res, gemm_out_torch_sum.cpu().numpy(), atol=6e-2, rtol=6e-2)

    print("Results all correct.")

    # # profiler results
    # for rank in range(WORLD_SIZE):
    #     if rank == 7:
    #         export_to_perfetto_trace(
    #             res_dict["profiler_buffer_host"].numpy()[rank],
    #             f"hgemm-RS-rank{rank}.perfetto-trace",
    #             event_type_names,
    #         )


@pytest.mark.skip()
def test_reduce():
    import torch

    torch.manual_seed(42)
    DEV = tvm.cuda(0)
    target = tvm.target.Target("cuda")
    input_torch = torch.randn((WORLD_SIZE, LOCAL_M, N), dtype=torch.float16, device="cuda")
    out_torch = torch.zeros((LOCAL_M, N), dtype=torch.float16, device="cuda")
    input_tvm = tvm.runtime.tensor(input_torch.cpu(), device=DEV)
    out_tvm = tvm.runtime.tensor(out_torch.cpu(), device=DEV)
    with target:
        mod = ReduceScatter
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
        print(mod.mod.imports[0].inspect_source())
        mod["reduce_sum"](input_tvm, out_tvm)
        out_std = torch.sum(input_torch, dim=0)
        np.testing.assert_allclose(out_tvm.numpy(), out_std.cpu().numpy(), atol=6e-2, rtol=6e-2)


if __name__ == "__main__":
    test_hgemm_rs()
    # test_reduce()
