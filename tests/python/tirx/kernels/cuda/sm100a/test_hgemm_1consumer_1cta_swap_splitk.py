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

import os
import sys

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.tirx.bench.utils import ProtonContext, bench, export_to_perfetto_trace

sys.path.insert(0, os.path.join(os.environ.get("TIRX_KERNELS_PATH", os.path.expanduser("~/tirx-kernels/kernels")), "gemm"))
from hgemm_1consumer_1cta_swap_splitk import (  # noqa: E402
    PROFILER_BUFFER_SIZE,
    PROFILER_ON,
    event_type_names,
    flops,
    get_hgemm_kernel,
    prepare_data,
)


@tvm.testing.requires_cuda_compute_version(10, exact=True)
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16, 32, 64, 128])
def test_hgemm_1consumer_1cta_swap_splitk(batch_size):
    N, K = 8192, 8192
    A_bf16, B_bf16, C_bf16 = prepare_data(batch_size, N, K)

    def tir_gemm(A_bf16, B_bf16, C_bf16):
        hgemm, reduce, TILE_K_NUM = get_hgemm_kernel(N, K)
        DEV = tvm.cuda(0)
        A_tvm = tvm.runtime.tensor(A_bf16, device=DEV)
        B_tvm = tvm.runtime.tensor(B_bf16, device=DEV)
        C_tvm = tvm.runtime.tensor(C_bf16, device=DEV)
        partial_sum_tvm = tvm.runtime.tensor(
            np.zeros((TILE_K_NUM, batch_size, N), dtype=np.float32), device=DEV
        )
        # Always allocate profiler buffer; it is unused when disabled
        profiler_buffer = np.zeros((PROFILER_BUFFER_SIZE,), dtype=np.uint64)
        profiler_buffer_tvm = tvm.runtime.tensor(profiler_buffer, DEV)
        target = tvm.target.Target("cuda")
        with target:
            mod_hgemm = tvm.ir.IRModule({"main": hgemm})
            mod_reduce = tvm.ir.IRModule({"main": reduce})
            mod_hgemm = tvm.compile(mod_hgemm, target=target, tir_pipeline="tirx")
            mod_reduce = tvm.compile(mod_reduce, target=target, tir_pipeline="tirx")

            def func():
                mod_hgemm(A_tvm, B_tvm, partial_sum_tvm, profiler_buffer_tvm)
                mod_reduce(partial_sum_tvm, C_tvm)

            ms = bench(func, warmup=10, repeat=30, proton_name="tir")
            print(f"TIR flops: {flops(batch_size, N, K, ms) / 1e12} TFLOPS, time: {ms:.3f} ms")
            if PROFILER_ON:
                export_to_perfetto_trace(
                    profiler_buffer_tvm.numpy(),
                    f"hgemm-{batch_size}-{N}-{K}-1consumer-1cta.perfetto-trace",
                    event_type_names,
                )

        return partial_sum_tvm.numpy(), C_tvm.numpy()

    def cublas_gemm(A_bf16, B_bf16):
        import torch

        torch_dev = torch.device("cuda")
        A_torch = A_bf16.to(torch_dev)
        B_torch = B_bf16.to(torch_dev)

        def func():
            return torch.matmul(A_torch, B_torch.T)

        C_torch = func()
        ms = bench(func, warmup=10, repeat=30, proton_name="cublas")
        print(f"CUBLAS flops: {flops(batch_size, N, K, ms) / 1e12} TFLOPS, time: {ms:.3f} ms")
        return C_torch.cpu().numpy()

    with ProtonContext("blackwell_gemm"):
        C_cublas = cublas_gemm(A_bf16, B_bf16)
        partial_sum_tvm, C_tvm = tir_gemm(A_bf16, B_bf16, C_bf16)

    np.testing.assert_allclose(
        partial_sum_tvm.sum(axis=0).astype(np.float16), C_cublas, rtol=1e-3, atol=1e-2
    )
    np.testing.assert_allclose(C_tvm, C_cublas, rtol=1e-3, atol=1e-2)


if __name__ == "__main__":
    for batch_size in [8192]:
        test_hgemm_1consumer_1cta_swap_splitk(batch_size)
