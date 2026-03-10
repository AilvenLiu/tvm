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

import tvm
import tvm.testing
from tvm.tirx.bench.utils import ProtonContext, bench

sys.path.insert(0, os.path.join(os.environ.get("TIRX_KERNELS_PATH", os.path.expanduser("~/tirx-kernels/kernels")), "gemm"))
from hgemm import (  # noqa: E402
    BLK_N,
    CTA_GROUP,
    DEBUG,
    M,
    N,
    K,
    NUM_CONSUMER,
    SM_NUMBER,
    flops,
    get_source,
    hgemm,
    prepare_data,
)


@tvm.testing.requires_cuda_compute_version(10, exact=True)
def test_hgemm():
    A_bf16, B_bf16, C_bf16 = prepare_data()

    def tir_gemm(A_bf16, B_bf16, C_bf16):
        DEV = tvm.cuda(0)
        A_tvm = tvm.runtime.tensor(A_bf16, device=DEV)
        B_tvm = tvm.runtime.tensor(B_bf16, device=DEV)
        C_tvm = tvm.runtime.tensor(C_bf16, device=DEV)
        target = tvm.target.Target("cuda")
        with target:
            src, mod = get_source(hgemm)

            def func():
                return mod(A_tvm, B_tvm, C_tvm)

            ms = bench(func, warmup=0, repeat=30, proton_name="tir", debug=DEBUG)
            print(f"TIR flops: {flops(ms) / 1e12} TFLOPS, time: {ms:.3f} ms")

        return C_tvm.numpy()

    def cublas_gemm(A_bf16, B_bf16):
        import torch

        torch_dev = torch.device("cuda")
        A_torch = A_bf16.to(torch_dev)
        B_torch = B_bf16.to(torch_dev)

        def func():
            return torch.matmul(A_torch, B_torch.T)

        ms = bench(func, warmup=0, repeat=30, proton_name="cublas", debug=DEBUG)
        print(f"CUBLAS flops: {flops(ms) / 1e12} TFLOPS, time: {ms:.3f} ms")
        C_torch = func()
        return C_torch.cpu().numpy()

    def cutedsl_gemm(A_bf16, B_bf16):
        import cutlass
        import torch
        from cutlass.cute.runtime import from_dlpack
        from tvm.tirx.bench.CuTeDSL.dense_gemm_persistent import run

        def create_cutlass_tensor(
            tensor, dtype, is_dynamic_layout=True, assumed_align=16, leading_dim=1
        ):
            cute_tensor = from_dlpack(tensor, assumed_align=assumed_align)
            cute_tensor.element_type = dtype
            if is_dynamic_layout:
                cute_tensor = cute_tensor.mark_layout_dynamic(leading_dim=leading_dim)
            return cute_tensor

        A_torch = A_bf16.to(torch.device("cuda")).reshape(1, M, K).permute(1, 2, 0)
        B_torch = B_bf16.to(torch.device("cuda")).reshape(1, N, K).permute(1, 2, 0)
        C_torch = torch.zeros_like(
            C_bf16.reshape(1, M, N).permute(1, 2, 0), device=torch.device("cuda")
        )
        func = run(
            mnkl=(M, N, K, 1),
            ab_dtype=cutlass.Float16,
            c_dtype=cutlass.Float16,
            acc_dtype=cutlass.Float32,
            a_major="k",
            b_major="k",
            c_major="n",
            skip_ref_check=True,
            A_torch=A_torch,
            B_torch=B_torch,
            C_torch=C_torch,
        )
        ms = bench(func, warmup=10, repeat=30, proton_name="cutedsl", debug=DEBUG)
        print(f"CuTeDSL flops: {flops(ms) / 1e12} TFLOPS, time: {ms:.3f} ms")
        return C_torch.cpu().numpy().reshape(M, N)

    with ProtonContext("blackwell_gemm", debug=DEBUG):
        C_tvm = tir_gemm(A_bf16, B_bf16, C_bf16)
        C_cublas = cublas_gemm(A_bf16, B_bf16)
        C_cutedsl = cutedsl_gemm(A_bf16, B_bf16)

    np.testing.assert_allclose(C_tvm, C_cublas, rtol=1e-3, atol=1e-2)
    np.testing.assert_allclose(C_cutedsl, C_cublas, rtol=1e-3, atol=1e-2)


if __name__ == "__main__":
    test_hgemm()
