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
import ml_dtypes
import torch
from deep_gemm.utils.math import per_block_cast_to_fp8, per_token_cast_to_fp8

import tvm
import tvm.testing
from tvm.tirx.bench.utils import ProtonContext, bench

sys.path.insert(0, os.path.join(os.environ.get("TIRX_KERNELS_PATH", os.path.expanduser("~/tirx-kernels/kernels")), "gemm"))
from deepgemm import (  # noqa: E402
    M,
    N,
    K,
    calc_diff,
    deepgemm,
    flops,
    get_source,
    prepare_data,
)


@tvm.testing.requires_cuda_compute_version(10, exact=True)
def test_deepgemm():
    DEV = tvm.cuda(0)
    A_fp8, B_fp8, sfa_pack, sfb_pack, C_ref, A_origin, B_origin = prepare_data()
    A_tvm = tvm.runtime.tensor(
        A_fp8.view(torch.int8).numpy().view(ml_dtypes.float8_e4m3fn), device=DEV
    )
    B_tvm = tvm.runtime.tensor(
        B_fp8.view(torch.int8).numpy().view(ml_dtypes.float8_e4m3fn), device=DEV
    )
    sfa_tvm = tvm.runtime.tensor(sfa_pack.numpy(), device=DEV)
    sfb_tvm = tvm.runtime.tensor(sfb_pack.numpy(), device=DEV)
    C_tvm = torch.empty((M, N), dtype=torch.bfloat16, device="cuda")
    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": deepgemm})
        src, mod = get_source(deepgemm)

    def std():
        a = per_token_cast_to_fp8(A_origin.to(torch.bfloat16).to("cuda"), use_ue8m0=True)
        b = per_block_cast_to_fp8(B_origin.to(torch.bfloat16).to("cuda"), use_ue8m0=True)
        out = torch.empty((M, N), dtype=torch.bfloat16, device="cuda")
        ms = bench(
            lambda: deep_gemm.fp8_gemm_nt(a, b, out, c=None, disable_ue8m0_cast=False, recipe=None),
            warmup=50,
            repeat=50,
            proton_name="std",
        )
        return ms, out

    def tir():
        ms = bench(
            lambda: mod(A_tvm, B_tvm, C_tvm, sfa_tvm, sfb_tvm),
            warmup=50,
            repeat=50,
            proton_name="tir",
        )
        return ms, C_tvm

    # It seems that the tir and std profiling will interfere with each other
    # And also the value of warmup and repeat affect the profiling result abnormally
    # May need to find a better way to do the profiling
    with ProtonContext():
        tir_ms, tir_out = tir()
        print(f"TIR flops: {flops(tir_ms) / 1e12} TFLOPS, time: {tir_ms:.3f} ms")
        std_ms, std_out = std()
        print(f"Std flops: {flops(std_ms) / 1e12} TFLOPS, time: {std_ms:.3f} ms")
        # np.testing.assert_allclose(C_tvm.numpy(), C_ref, rtol=1e-3, atol=1e-2)
        assert calc_diff(std_out, tir_out) < 2e-3
        assert calc_diff(std_out, C_ref.to("cuda")) < 2e-3
        print("Test passed!")


if __name__ == "__main__":
    test_deepgemm()
