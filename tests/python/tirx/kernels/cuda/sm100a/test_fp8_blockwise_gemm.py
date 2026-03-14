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

"""Pytest wrapper for kernels/sm100/fp8_blockwise_gemm.py (Tx.gemm_async based)."""

import os
import sys

import pytest
import torch
import torch.nn.functional as F

import tvm
import tvm.testing

sys.path.insert(
    0,
    os.path.join(
        os.environ.get("TIRX_KERNELS_PATH", os.path.expanduser("~/tirx-kernels/kernels")), "gemm"
    ),
)
fp8_test = pytest.importorskip("fp8_blockwise_gemm")


@pytest.mark.parametrize("M,N,K", [(1024, 1024, 1024), (4096, 4096, 4096)])
def test_fp8_blockwise_gemm_correctness(M, N, K):
    A_fp8, B_fp8, _sfa, _sfb, sfa_pack, sfb_pack, C_ref, _A_orig, _B_orig = fp8_test.prepare_data(
        M, N, K
    )
    kernel = fp8_test.tir_kernel(M, N, K)

    out = torch.zeros_like(C_ref, device="cuda", dtype=torch.bfloat16)

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": kernel})
        ex = tvm.compile(mod, target=target, tir_pipeline="tirx")

    ex(A_fp8, B_fp8, out, sfa_pack, sfb_pack)
    torch.cuda.synchronize()

    cosine_sim = F.cosine_similarity(
        out.float().reshape(-1), C_ref.to("cuda").float().reshape(-1), dim=0
    )
    assert cosine_sim.item() > 0.97, f"cosine similarity {cosine_sim.item():.6f} < 0.97"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
