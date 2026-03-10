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
import hgemm_rs_tilelink as kernel  # noqa: E402
from hgemm_rs_tilelink import (  # noqa: E402
    BLK_M,
    BLK_N,
    GEMM_SMS,
    K,
    LOCAL_M,
    M,
    M_SEG,
    N,
    N_REPEAT,
    PROFILER_BUFFER_SIZE,
    RS_BLK_M,
    RS_BLK_N,
    SM_COUNT,
    WORLD_SIZE,
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
    # A_np = np.random.randint(0, 2, (WORLD_SIZE, M, K)).astype(a_type)
    # B_np = np.random.randint(0, 2, (WORLD_SIZE, N, K)).astype(b_type)
    A_tvm = tvm.runtime.tensor(A_np, device=DEV)
    B_tvm = tvm.runtime.tensor(B_np, device=DEV)
    semaphore_np = np.zeros((M // BLK_M, N // BLK_N), dtype="int32")
    buffer_np = np.zeros((M // RS_BLK_M, N // RS_BLK_N, RS_BLK_M, RS_BLK_N), dtype=d_type)
    sig_addr_np = np.zeros((WORLD_SIZE, SM_COUNT), dtype="uint32")
    profiler_buffer_np = np.zeros((PROFILER_BUFFER_SIZE,), dtype="uint64")

    A_array_all = sess.empty((WORLD_SIZE, M, K), a_type, worker0_only=True)
    B_array_all = sess.empty((WORLD_SIZE, N, K), b_type, worker0_only=True)
    sess.copy_to_worker_0(A_tvm, A_array_all)
    sess.copy_to_worker_0(B_tvm, B_array_all)

    nvshmem_malloc_hook = sess.get_global_func("runtime.disco.nvshmem.empty")
    args_dict = {
        "A_array": sess.empty((M, K), a_type),
        "B_array": sess.empty((N, K), b_type),
        "gemm_out_array": sess.empty((M, N), d_type),
        "semaphore_array": sess.empty((M // BLK_M, N // BLK_N), "int32"),
        "buffer_array": nvshmem_malloc_hook(
            ShapeTuple((M // RS_BLK_M, N // RS_BLK_N, RS_BLK_M, RS_BLK_N)), d_type, None
        ),
        "sig_addr_array": nvshmem_malloc_hook(ShapeTuple((WORLD_SIZE, SM_COUNT)), "uint32", None),
        "out_array": sess.empty((LOCAL_M, N), d_type),
        "profiler_buffer_array": sess.empty((PROFILER_BUFFER_SIZE,), "uint64"),
    }

    res_dict = {
        "gemm_out_res": sess.empty((WORLD_SIZE, M, N), d_type, worker0_only=True),
        "buffer_res": sess.empty(
            (WORLD_SIZE, M // RS_BLK_M, N // RS_BLK_N, RS_BLK_M, RS_BLK_N),
            d_type,
            worker0_only=True,
        ),
        "out_res": sess.empty((WORLD_SIZE, LOCAL_M, N), d_type, worker0_only=True),
        "profiler_buffer_res": sess.empty(
            (WORLD_SIZE, PROFILER_BUFFER_SIZE), "uint64", worker0_only=True
        ),
        "gemm_out_host": tvm.runtime.empty((WORLD_SIZE, M, N), d_type, device=DEV),
        "buffer_host": tvm.runtime.empty(
            (WORLD_SIZE, M // RS_BLK_M, N // RS_BLK_N, RS_BLK_M, RS_BLK_N), d_type, device=DEV
        ),
        "out_host": tvm.runtime.empty((WORLD_SIZE, LOCAL_M, N), d_type, device=DEV),
        "profiler_buffer_host": tvm.runtime.empty(
            (WORLD_SIZE, PROFILER_BUFFER_SIZE), "uint64", device=DEV
        ),
    }

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

        rt_mod = sess.load_vm_module(path)
        barrier_dfunc = sess.get_global_func("runtime.disco.nvshmem.barrier_all_on_current_stream")
        print("Begin kernel execution...")
        sess._sync_all()
        for itr in range(N_REPEAT):
            barrier_dfunc()
            rt_mod["test_mma_ss_tma_2sm_persistent"](
                args_dict["A_array"],
                args_dict["B_array"],
                args_dict["gemm_out_array"],
                args_dict["semaphore_array"],
                args_dict["buffer_array"],
                args_dict["sig_addr_array"],
                args_dict["out_array"],
                args_dict["profiler_buffer_array"],
            )
            sess._sync_all()
            if itr < N_REPEAT - 1:
                sess.broadcast(semaphore_np, args_dict["semaphore_array"])
                sess.broadcast(buffer_np, args_dict["buffer_array"])
                sess.broadcast(sig_addr_np, args_dict["sig_addr_array"])
                sess.broadcast(profiler_buffer_np, args_dict["profiler_buffer_array"])
                sess._sync_all()

        print("Kernel execution finished.")
        # validate results
        sess.gather_to_worker0(args_dict["gemm_out_array"], res_dict["gemm_out_res"])
        sess.copy_from_worker_0(res_dict["gemm_out_host"], res_dict["gemm_out_res"])
        sess.gather_to_worker0(args_dict["buffer_array"], res_dict["buffer_res"])
        sess.copy_from_worker_0(res_dict["buffer_host"], res_dict["buffer_res"])
        sess.gather_to_worker0(args_dict["out_array"], res_dict["out_res"])
        sess.copy_from_worker_0(res_dict["out_host"], res_dict["out_res"])
        sess.gather_to_worker0(args_dict["profiler_buffer_array"], res_dict["profiler_buffer_res"])
        sess.copy_from_worker_0(res_dict["profiler_buffer_host"], res_dict["profiler_buffer_res"])

        # sync all workers to make sure the temporary files are cleaned up after all workers
        # finish the execution
        sess._sync_all()

    finalize_dfunc = sess.get_global_func("runtime.disco.nvshmem.finalize_nvshmem")
    finalize_dfunc()
    sess.sync_worker_0()

    # validate results
    print("Validating results...")

    import torch

    gemm_out_torch = torch.zeros((WORLD_SIZE, M, N), dtype=torch.float16, device="cuda")
    gemm_out_torch_sum = torch.zeros((M, N), dtype=torch.float16, device="cuda")
    for i in range(WORLD_SIZE):
        A_torch = torch.tensor(A_np[i], dtype=torch.float16, device="cuda")
        B_torch = torch.tensor(B_np[i], dtype=torch.float16, device="cuda")
        gemm_out_torch[i] = torch.matmul(A_torch, B_torch.T)
        gemm_out_res = res_dict["gemm_out_host"].numpy()[i]
        assert (gemm_out_res == gemm_out_torch[i].cpu().numpy()).all()

    for i in range(WORLD_SIZE):
        start_rank = (i + WORLD_SIZE - 1) % WORLD_SIZE
        for j in range(WORLD_SIZE):
            rank = (start_rank + WORLD_SIZE - j) % WORLD_SIZE
            gemm_out_torch_sum[i * LOCAL_M : (i + 1) * LOCAL_M, :] += gemm_out_torch[rank][
                i * LOCAL_M : (i + 1) * LOCAL_M, :
            ]
    out_res = res_dict["out_host"].numpy().reshape(-1, N)
    np.testing.assert_allclose(out_res, gemm_out_torch_sum.cpu().numpy(), atol=1e-3)

    print("Results all correct.")

    # # profiler results
    # for rank in range(WORLD_SIZE):
    #     if rank == 7:
    #         export_to_perfetto_trace(
    #             res_dict["profiler_buffer_host"].numpy()[rank],
    #             f"hgemm-RS-rank{rank}.perfetto-trace",
    #             event_type_names,
    #         )


if __name__ == "__main__":
    test_hgemm_rs()
