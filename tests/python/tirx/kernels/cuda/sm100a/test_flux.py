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
from tvm.tirx.bench.utils import bench

sys.path.insert(
    0,
    os.path.join(
        os.environ.get("TIRX_KERNELS_PATH", os.path.expanduser("~/tirx-kernels/kernels")), "gemm"
    ),
)
from flux import (
    DEBUG,
    K,
    M,
    N,
    flops,
    flux_gate_kernel,
    flux_gelu_kernel,
    get_source,
)


@tvm.testing.requires_cuda_compute_version(10, exact=True)
def test_flux_gelu():
    import torch

    # --- Prepare data ---
    A_data = torch.randn((M, K), dtype=torch.float16)
    B_data = torch.randn((N, K), dtype=torch.float16)
    bias_data = torch.randn((N,), dtype=torch.float16)

    def ref_flux_gelu(A_data, B_data, bias_data):
        A_f32 = A_data.float()
        B_f32 = B_data.float()
        bias_f32 = bias_data.float()
        C = A_f32 @ B_f32.T + bias_f32.unsqueeze(0)
        # GELU via sigmoid approximation: x * sigmoid(1.5957691*x + 0.07106856*x^3)
        C = C * torch.sigmoid(C * (1.5957691 + 0.07106856 * C * C))
        return C.half().numpy()

    def tir_flux_gelu(A_data, B_data, bias_data):
        DEV = tvm.cuda(0)
        A_tvm = tvm.runtime.tensor(A_data.numpy(), device=DEV)
        B_tvm = tvm.runtime.tensor(B_data.numpy(), device=DEV)
        bias_tvm = tvm.runtime.tensor(bias_data.numpy(), device=DEV)
        D_tvm = tvm.runtime.tensor(np.zeros((M, N), dtype="float16"), device=DEV)
        target = tvm.target.Target("cuda")
        with target:
            src, mod = get_source(flux_gelu_kernel)
            print(src)

            def func():
                return mod(A_tvm, B_tvm, bias_tvm, D_tvm)

            ms = bench(func, warmup=0, repeat=30, proton_name="tir_flux_gelu", debug=DEBUG)
            print(f"TIR flux_gelu flops: {flops(ms) / 1e12} TFLOPS, time: {ms:.3f} ms")
        return D_tvm.numpy()

    C_ref = ref_flux_gelu(A_data, B_data, bias_data)
    C_tir = tir_flux_gelu(A_data, B_data, bias_data)
    np.testing.assert_allclose(C_tir, C_ref, rtol=1e-2, atol=1e-1)
    print("flux_gelu: PASSED")


@tvm.testing.requires_cuda_compute_version(10, exact=True)
def test_flux_gate():
    import torch

    # --- Prepare data ---
    A_data = torch.randn((M, K), dtype=torch.float16)
    B_data = torch.randn((N, K), dtype=torch.float16)
    bias_data = torch.randn((N,), dtype=torch.float16)
    gate_data = torch.randn((N,), dtype=torch.float16)
    Y_data = torch.randn((M, N), dtype=torch.float16)

    def ref_flux_gate(A_data, B_data, bias_data, gate_data, Y_data):
        A_f32 = A_data.float()
        B_f32 = B_data.float()
        bias_f32 = bias_data.float()
        gate_f32 = gate_data.float()
        Y_f32 = Y_data.float()
        C = (A_f32 @ B_f32.T + bias_f32.unsqueeze(0)) * gate_f32.unsqueeze(0) + Y_f32
        return C.half().numpy()

    def tir_flux_gate(A_data, B_data, bias_data, gate_data, Y_data):
        DEV = tvm.cuda(0)
        A_tvm = tvm.runtime.tensor(A_data.numpy(), device=DEV)
        B_tvm = tvm.runtime.tensor(B_data.numpy(), device=DEV)
        bias_tvm = tvm.runtime.tensor(bias_data.numpy(), device=DEV)
        gate_tvm = tvm.runtime.tensor(gate_data.numpy(), device=DEV)
        Y_tvm = tvm.runtime.tensor(Y_data.numpy(), device=DEV)
        D_tvm = tvm.runtime.tensor(np.zeros((M, N), dtype="float16"), device=DEV)
        target = tvm.target.Target("cuda")
        with target:
            src, mod = get_source(flux_gate_kernel)
            print(src)

            def func():
                return mod(A_tvm, B_tvm, bias_tvm, gate_tvm, Y_tvm, D_tvm)

            ms = bench(func, warmup=0, repeat=30, proton_name="tir_flux_gate", debug=DEBUG)
            print(f"TIR flux_gate flops: {flops(ms) / 1e12} TFLOPS, time: {ms:.3f} ms")
        return D_tvm.numpy()

    C_ref = ref_flux_gate(A_data, B_data, bias_data, gate_data, Y_data)
    C_tir = tir_flux_gate(A_data, B_data, bias_data, gate_data, Y_data)
    np.testing.assert_allclose(C_tir, C_ref, rtol=1e-2, atol=1e-1)
    print("flux_gate: PASSED")


if __name__ == "__main__":
    test_flux_gelu()
    test_flux_gate()
