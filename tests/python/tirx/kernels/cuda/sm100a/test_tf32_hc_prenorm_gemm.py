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

import deep_gemm
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
from tf32_hc_prenorm_gemm import (
    K,
    M,
    N,
    bf16_to_tvm,
    calc_diff,
    flops,
    get_source,
    kNumSplits,
    tf32_hc_prenorm_gemm,
)


@tvm.testing.requires_cuda_compute_version(10, exact=True)
def test_tf32_hc_prenorm_gemm():
    DEV = tvm.cuda(0)

    # Generate test data
    A_torch = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B_torch = torch.randn(N, K, dtype=torch.float32, device="cuda")

    # Reference computation
    D_ref = A_torch.float() @ B_torch.T  # (M, N) f32
    sqr_sum_ref = (A_torch.float() ** 2).sum(dim=1)  # (M,) f32

    # Convert to TVM tensors
    A_tvm = bf16_to_tvm(A_torch, DEV)
    B_tvm = tvm.runtime.tensor(B_torch.cpu().numpy(), device=DEV)

    # Output buffers (split-K: D is (kNumSplits, M, N), sqr_sum is (kNumSplits * M,))
    D_out_splits = torch.zeros(kNumSplits, M, N, dtype=torch.float32, device="cuda")
    sqr_sum_out_splits = torch.zeros(kNumSplits * M, dtype=torch.float32, device="cuda")

    # Compile
    target = tvm.target.Target("cuda")
    with target:
        src, mod = get_source(tf32_hc_prenorm_gemm)

    # Run
    mod(A_tvm, B_tvm, D_out_splits, sqr_sum_out_splits)

    # Reduce splits
    D_out = D_out_splits.sum(dim=0)  # (M, N)
    sqr_sum_out = sqr_sum_out_splits.view(kNumSplits, M).sum(dim=0)  # (M,)

    # Verify D
    d_diff = calc_diff(D_out.cpu(), D_ref.cpu())
    print(f"D calc_diff: {d_diff:.6e}")
    assert d_diff < 1e-3, f"D diff too large: {d_diff}"

    # Verify sqr_sum
    sqr_diff = calc_diff(sqr_sum_out.cpu(), sqr_sum_ref.cpu())
    print(f"sqr_sum calc_diff: {sqr_diff:.6e}")
    assert sqr_diff < 1e-3, f"sqr_sum diff too large: {sqr_diff}"

    # Cross-check with DeepGEMM
    D_dg = torch.empty(M, N, dtype=torch.float32, device="cuda")
    sqr_sum_dg = torch.empty(M, dtype=torch.float32, device="cuda")
    deep_gemm.tf32_hc_prenorm_gemm(A_torch, B_torch, D_dg, sqr_sum_dg, num_splits=None)

    dg_d_diff = calc_diff(D_out.cpu(), D_dg.cpu())
    dg_sqr_diff = calc_diff(sqr_sum_out.cpu(), sqr_sum_dg.cpu())
    print(f"vs DeepGEMM: D diff={dg_d_diff:.6e}, sqr_sum diff={dg_sqr_diff:.6e}")
    assert dg_d_diff < 1e-3, f"D diff vs DeepGEMM too large: {dg_d_diff}"
    assert dg_sqr_diff < 1e-3, f"sqr_sum diff vs DeepGEMM too large: {dg_sqr_diff}"

    print("Test passed!")


# ---------------------------------------------------------------------------
# Benchmark: TIRX vs DeepGEMM
# ---------------------------------------------------------------------------


def bench_tf32_hc_prenorm_gemm():
    DEV = tvm.cuda(0)

    A_torch = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B_torch = torch.randn(N, K, dtype=torch.float32, device="cuda")

    # TVM tensors
    A_tvm = bf16_to_tvm(A_torch, DEV)
    B_tvm = tvm.runtime.tensor(B_torch.cpu().numpy(), device=DEV)

    # Compile
    target = tvm.target.Target("cuda")
    with target:
        src, mod = get_source(tf32_hc_prenorm_gemm)

    def tir():
        D_out_splits = torch.zeros(kNumSplits, M, N, dtype=torch.float32, device="cuda")
        sqr_sum_out_splits = torch.zeros(kNumSplits * M, dtype=torch.float32, device="cuda")
        ms = bench(
            lambda: mod(A_tvm, B_tvm, D_out_splits, sqr_sum_out_splits),
            warmup=50,
            repeat=50,
            proton_name="tir",
        )
        D_out = D_out_splits.sum(dim=0)
        sqr_sum_out = sqr_sum_out_splits.view(kNumSplits, M).sum(dim=0)
        return ms, D_out, sqr_sum_out

    def std():
        D_dg = torch.empty(M, N, dtype=torch.float32, device="cuda")
        sqr_sum_dg = torch.empty(M, dtype=torch.float32, device="cuda")
        ms = bench(
            lambda: deep_gemm.tf32_hc_prenorm_gemm(
                A_torch, B_torch, D_dg, sqr_sum_dg, num_splits=None
            ),
            warmup=50,
            repeat=50,
            proton_name="std",
        )
        return ms, D_dg, sqr_sum_dg

    with ProtonContext():
        tir_ms, tir_D, tir_sqr = tir()
        print(f"TIR flops: {flops(tir_ms) / 1e12:.2f} TFLOPS, time: {tir_ms:.3f} ms")

        std_ms, std_D, std_sqr = std()
        print(f"Std flops: {flops(std_ms) / 1e12:.2f} TFLOPS, time: {std_ms:.3f} ms")

        d_diff = calc_diff(tir_D.cpu(), std_D.cpu())
        sqr_diff = calc_diff(tir_sqr.cpu(), std_sqr.cpu())
        print(f"TIR vs DeepGEMM: D diff={d_diff:.6e}, sqr_sum diff={sqr_diff:.6e}")
        assert d_diff < 2e-3, f"D diff too large: {d_diff}"
        assert sqr_diff < 2e-3, f"sqr_sum diff too large: {sqr_diff}"
        print("Benchmark passed!")


if __name__ == "__main__":
    bench_tf32_hc_prenorm_gemm()
