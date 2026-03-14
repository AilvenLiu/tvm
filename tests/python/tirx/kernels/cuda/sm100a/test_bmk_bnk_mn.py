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

import torch

import tvm
import tvm.testing
from tvm.tirx.bench.utils import ProtonContext, bench

sys.path.insert(
    0,
    os.path.join(
        os.environ.get("TIRX_KERNELS_PATH", os.path.expanduser("~/tirx-kernels/kernels")), "gemm"
    ),
)
from bmk_bnk_mn import (
    S_DIM,
    K,
    M,
    N,
    bmk_bnk_mn_gemm,
    calc_diff,
    flops,
    get_source,
    prepare_data,
)


@tvm.testing.requires_cuda_compute_version(10, exact=True)
def test_bmk_bnk_mn():
    A, B, A_flat, B_flat, D_out, D_ref = prepare_data()

    target = tvm.target.Target("cuda")
    with target:
        src, mod = get_source(bmk_bnk_mn_gemm)

        def run():
            D_out.zero_()
            mod(A_flat, B_flat, D_out)

        ms = bench(run, warmup=10, repeat=30, proton_name="tir")
        tflops = flops(ms) / 1e12
        print(f"TIR: {tflops:.2f} TFLOPS, time: {ms:.3f} ms")

    diff = calc_diff(D_out, D_ref.to("cuda"))
    print(f"calc_diff: {diff:.6f}")
    assert diff < 2e-3, f"Correctness check failed: calc_diff={diff}"
    print("Test passed!")


def bench_bmk_bnk_mn():
    import deep_gemm

    A = torch.randn((S_DIM, M, K), dtype=torch.bfloat16, device="cuda")
    B = torch.randn((S_DIM, N, K), dtype=torch.bfloat16, device="cuda")
    A_flat = A.reshape(S_DIM * M, K).contiguous()
    B_flat = B.reshape(S_DIM * N, K).contiguous()

    target = tvm.target.Target("cuda")
    with target:
        src, mod = get_source(bmk_bnk_mn_gemm)

    def std():
        # DeepGEMM einsum 'bmk,bnk->mn': c=out enables f32 accumulation
        out = torch.zeros((M, N), dtype=torch.float32, device="cuda")

        def run():
            out.zero_()
            deep_gemm.einsum("bmk,bnk->mn", A, B, out, c=out)

        ms = bench(run, warmup=50, repeat=50, proton_name="std")
        out.zero_()
        deep_gemm.einsum("bmk,bnk->mn", A, B, out, c=out)
        return ms, out

    def tir():
        D_out = torch.zeros((M, N), dtype=torch.float32, device="cuda")
        ms = bench(
            lambda: (D_out.zero_(), mod(A_flat, B_flat, D_out))[-1],
            warmup=50,
            repeat=50,
            proton_name="tir",
        )
        D_out.zero_()
        mod(A_flat, B_flat, D_out)
        return ms, D_out

    with ProtonContext():
        tir_ms, tir_out = tir()
        print(f"TIR flops: {flops(tir_ms) / 1e12:.2f} TFLOPS, time: {tir_ms:.3f} ms")
        std_ms, std_out = std()
        print(f"Std flops: {flops(std_ms) / 1e12:.2f} TFLOPS, time: {std_ms:.3f} ms")
        diff = calc_diff(tir_out, std_out)
        print(f"calc_diff(tir, std): {diff:.6f}")
        assert diff < 2e-3, f"Correctness check failed: calc_diff={diff}"
        print("Benchmark passed!")


if __name__ == "__main__":
    bench_bmk_bnk_mn()
