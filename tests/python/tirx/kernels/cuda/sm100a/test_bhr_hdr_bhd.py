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
from bhr_hdr_bhd import (
    B_DIM,
    D_DIM,
    H_DIM,
    R_DIM,
    bhr_hdr_bhd_gemm,
    calc_diff,
    flops,
    get_source,
    prepare_data,
)


@tvm.testing.requires_cuda_compute_version(10, exact=True)
def test_bhr_hdr_bhd():
    A_flat, B_flat, D_flat, D_ref_flat = prepare_data()

    target = tvm.target.Target("cuda")
    with target:
        src, mod = get_source(bhr_hdr_bhd_gemm)

        def run():
            mod(A_flat, B_flat, D_flat)

        ms = bench(run, warmup=10, repeat=30, proton_name="tir")
        tflops = flops(ms) / 1e12
        print(f"TIR: {tflops:.2f} TFLOPS, time: {ms:.3f} ms")

    diff = calc_diff(D_flat, D_ref_flat)
    print(f"calc_diff: {diff:.6f}")
    assert diff < 2e-3, f"Correctness check failed: calc_diff={diff}"
    print("Test passed!")


def bench_bhr_hdr_bhd():
    import deep_gemm

    A = torch.randn((B_DIM, H_DIM, R_DIM), dtype=torch.bfloat16, device="cuda")
    B_mat = torch.randn((H_DIM, D_DIM, R_DIM), dtype=torch.bfloat16, device="cuda")
    A_flat = A.permute(1, 0, 2).reshape(H_DIM * B_DIM, R_DIM).contiguous()
    B_flat = B_mat.reshape(H_DIM * D_DIM, R_DIM).contiguous()

    target = tvm.target.Target("cuda")
    with target:
        src, mod = get_source(bhr_hdr_bhd_gemm)

    def std():
        out = torch.empty((B_DIM, H_DIM, D_DIM), dtype=torch.bfloat16, device="cuda")
        ms = bench(
            lambda: deep_gemm.einsum("bhr,hdr->bhd", A, B_mat, out),
            warmup=50,
            repeat=50,
            proton_name="std",
        )
        deep_gemm.einsum("bhr,hdr->bhd", A, B_mat, out)
        out_flat = out.permute(1, 0, 2).reshape(H_DIM * B_DIM, D_DIM).contiguous()
        return ms, out_flat

    def tir():
        D_out = torch.zeros((H_DIM * B_DIM, D_DIM), dtype=torch.bfloat16, device="cuda")
        ms = bench(
            lambda: mod(A_flat, B_flat, D_out),
            warmup=50,
            repeat=50,
            proton_name="tir",
        )
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
    bench_bhr_hdr_bhd()
