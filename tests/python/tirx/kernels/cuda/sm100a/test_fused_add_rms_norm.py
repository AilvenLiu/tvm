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

import pytest
import torch

import tvm
from tvm.tirx.bench.utils import ProtonContext, bench

sys.path.insert(0, os.path.join(os.environ.get("TIRX_KERNELS_PATH", os.path.expanduser("~/tirx-kernels/kernels")), "activation"))
import fused_add_rms_norm  # noqa: E402


# Re-export for backward compatibility (qwen3_layer.py, qwen3_model.py import from here)
get_fused_add_rmsnorm_kernel = fused_add_rms_norm.get_fused_add_rmsnorm_kernel


@pytest.mark.parametrize("hidden_size", [5120])
@pytest.mark.parametrize("batch_size", [4113, 1, 2, 4, 8, 16, 32, 64, 128])
def test_fused_add_rmsnorm(hidden_size, batch_size):
    def test_dynamic_batch(batch_size, mod):
        x, residual, weight = fused_add_rms_norm.prepare_data(hidden_size, batch_size)

        def naive():
            x_naive = x.to(torch.float32)
            x_naive = x_naive + residual.to(torch.float32)
            residual_naive = x_naive.to(torch.float16)
            variance = x_naive.pow(2).mean(dim=-1, keepdim=True)
            x_naive = x_naive * torch.rsqrt(variance + fused_add_rms_norm.EPS)
            x_naive = (x_naive * weight.float()).to(torch.float16)
            return x_naive.cpu().numpy(), residual_naive.cpu().numpy()

        def flashinfer():
            import flashinfer

            def func():
                return flashinfer.norm.fused_add_rmsnorm(
                    x.clone(), residual.clone(), weight, fused_add_rms_norm.EPS, enable_pdl=False
                )

            ms = bench(func, warmup=10, repeat=30, proton_name="flashinfer")
            print(f"flashinfer time: {ms:.3f} ms")
            x_fused = x.clone()
            residual_fused = residual.clone()
            flashinfer.norm.fused_add_rmsnorm(
                x_fused, residual_fused, weight, fused_add_rms_norm.EPS, enable_pdl=False
            )
            return x_fused.cpu().numpy(), residual_fused.cpu().numpy()

        def tir():
            DEV = tvm.cuda(0)
            weight_tvm = tvm.runtime.tensor(weight.cpu().numpy(), DEV)

            def func():
                return mod(
                    tvm.runtime.tensor(x.cpu().numpy(), DEV),
                    tvm.runtime.tensor(residual.cpu().numpy(), DEV),
                    weight_tvm,
                )

            ms = bench(func, warmup=10, repeat=30, proton_name="tir")
            print(f"tir time: {ms:.3f} ms")
            x_tvm = tvm.runtime.tensor(x.cpu().numpy(), DEV)
            residual_tvm = tvm.runtime.tensor(residual.cpu().numpy(), DEV)
            mod(x_tvm, residual_tvm, weight_tvm)
            return x_tvm.numpy(), residual_tvm.numpy()

        x_naive, residual_native = naive()
        x_fused, residual_fused = flashinfer()
        x_tir, residual_tir = tir()

        torch.testing.assert_close(residual_fused, residual_native, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(x_fused, x_naive, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(residual_tir, residual_native, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(x_tir, x_naive, rtol=1e-3, atol=1e-3)

    # compile tir kernel
    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": fused_add_rms_norm.get_fused_add_rmsnorm_kernel(hidden_size)})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
        src = mod.mod.imports[0].inspect_source()
        print(src)

    with ProtonContext("rms_norm"):
        test_dynamic_batch(batch_size, mod)


if __name__ == "__main__":
    import itertools

    hidden_size_list = [5120]  # , 128]
    batch_size_list = [1, 128, 4096]  # 2, 4, 8, 16, 32, 64, 128, 4113]
    for hidden_size, batch_size in itertools.product(hidden_size_list, batch_size_list):
        test_fused_add_rmsnorm(hidden_size, batch_size)
