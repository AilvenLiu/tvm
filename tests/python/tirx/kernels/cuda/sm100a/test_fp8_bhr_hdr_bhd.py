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

import ml_dtypes
import torch

import tvm
import tvm.testing
from tvm.tirx.bench.utils import ProtonContext, bench

sys.path.insert(0, os.path.join(os.environ.get("TIRX_KERNELS_PATH", os.path.expanduser("~/tirx-kernels/kernels")), "gemm"))
from fp8_bhr_hdr_bhd import (  # noqa: E402
    B_DIM,
    D_DIM,
    H_DIM,
    R_DIM,
    calc_diff,
    ceildiv,
    flops,
    fp8_bhr_hdr_bhd_gemm,
    get_source,
    prepare_data,
)


@tvm.testing.requires_cuda_compute_version(10, exact=True)
def test_fp8_bhr_hdr_bhd():
    DEV = tvm.cuda(0)
    A_fp8, B_fp8, sfa_pack, sfb_pack, D_ref = prepare_data()

    A_tvm = tvm.runtime.tensor(
        A_fp8.view(torch.int8).numpy().view(ml_dtypes.float8_e4m3fn), device=DEV
    )
    B_tvm = tvm.runtime.tensor(
        B_fp8.view(torch.int8).numpy().view(ml_dtypes.float8_e4m3fn), device=DEV
    )
    sfa_tvm = tvm.runtime.tensor(sfa_pack.numpy(), device=DEV)
    sfb_tvm = tvm.runtime.tensor(sfb_pack.numpy(), device=DEV)
    D_out = torch.empty((B_DIM * H_DIM, D_DIM), dtype=torch.bfloat16, device="cuda")

    target = tvm.target.Target("cuda")
    with target:
        src, mod = get_source(fp8_bhr_hdr_bhd_gemm)

        def run():
            mod(A_tvm, B_tvm, D_out, sfa_tvm, sfb_tvm)

        ms = bench(run, warmup=10, repeat=30, proton_name="tir")
        tflops = flops(ms) / 1e12
        print(f"TIR: {tflops:.2f} TFLOPS, time: {ms:.3f} ms")

    D_ref_cuda = D_ref.to("cuda")
    diff = calc_diff(D_out, D_ref_cuda)
    print(f"calc_diff: {diff:.6f}")
    assert diff < 2e-3, f"Correctness check failed: calc_diff={diff}"
    print("Test passed!")


def bench_fp8_bhr_hdr_bhd():
    import deep_gemm
    from deep_gemm.utils.math import per_block_cast_to_fp8, per_token_cast_to_fp8

    DEV = tvm.cuda(0)
    A_fp8, B_fp8, sfa_pack, sfb_pack, D_ref = prepare_data()

    A_tvm = tvm.runtime.tensor(
        A_fp8.view(torch.int8).numpy().view(ml_dtypes.float8_e4m3fn), device=DEV
    )
    B_tvm = tvm.runtime.tensor(
        B_fp8.view(torch.int8).numpy().view(ml_dtypes.float8_e4m3fn), device=DEV
    )
    sfa_tvm = tvm.runtime.tensor(sfa_pack.numpy(), device=DEV)
    sfb_tvm = tvm.runtime.tensor(sfb_pack.numpy(), device=DEV)

    target = tvm.target.Target("cuda")
    with target:
        src, mod = get_source(fp8_bhr_hdr_bhd_gemm)

    def std():
        # DeepGEMM fp8_einsum with same quantization as test_einsum.py
        A_bf16 = torch.randn((B_DIM, H_DIM, R_DIM), dtype=torch.bfloat16, device="cuda")
        B_bf16 = torch.randn((H_DIM, D_DIM, R_DIM), dtype=torch.bfloat16, device="cuda")
        x_fp8 = per_token_cast_to_fp8(A_bf16.view(-1, R_DIM), use_ue8m0=True)
        x_fp8 = x_fp8[0].view(B_DIM, H_DIM, R_DIM), x_fp8[1].view(
            B_DIM, H_DIM, ceildiv(R_DIM, 128)
        )
        y_fp8 = (
            torch.empty_like(B_bf16, dtype=torch.float8_e4m3fn),
            torch.empty(
                (H_DIM, ceildiv(D_DIM, 128), ceildiv(R_DIM, 128)),
                device="cuda",
                dtype=torch.float,
            ),
        )
        for i in range(H_DIM):
            y_fp8[0][i], y_fp8[1][i] = per_block_cast_to_fp8(B_bf16[i], use_ue8m0=True)
        z = torch.empty((B_DIM, H_DIM, D_DIM), dtype=torch.bfloat16, device="cuda")
        ms = bench(
            lambda: deep_gemm.fp8_einsum("bhr,hdr->bhd", x_fp8, y_fp8, z),
            warmup=50,
            repeat=50,
            proton_name="std",
        )
        return ms

    def tir():
        D_out = torch.empty((B_DIM * H_DIM, D_DIM), dtype=torch.bfloat16, device="cuda")
        ms = bench(
            lambda: mod(A_tvm, B_tvm, D_out, sfa_tvm, sfb_tvm),
            warmup=50,
            repeat=50,
            proton_name="tir",
        )
        D_ref_cuda = D_ref.to("cuda")
        diff = calc_diff(D_out, D_ref_cuda)
        return ms, diff

    with ProtonContext():
        tir_ms, tir_diff = tir()
        print(f"TIR flops: {flops(tir_ms) / 1e12:.2f} TFLOPS, time: {tir_ms:.3f} ms")
        print(f"calc_diff(tir, ref): {tir_diff:.6f}")
        std_ms = std()
        print(f"Std flops: {flops(std_ms) / 1e12:.2f} TFLOPS, time: {std_ms:.3f} ms")
        assert tir_diff < 2e-3, f"Correctness check failed: calc_diff={tir_diff}"
        print("Benchmark passed!")


if __name__ == "__main__":
    bench_fp8_bhr_hdr_bhd()
