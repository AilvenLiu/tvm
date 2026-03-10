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
import rmsnorm  # noqa: E402
from rmsnorm import EPS, get_rmsnorm_kernel, prepare_data  # noqa: E402, F401


@pytest.mark.parametrize("hidden_size", [5120, 128])
@pytest.mark.parametrize("batch_size", [4113, 1, 2, 4, 8, 16, 32, 64, 128])
def test_rmsnorm(hidden_size, batch_size):
    def test_dynamic_batch(batch_size, mod):
        x, weight = rmsnorm.prepare_data(hidden_size, batch_size)

        def naive():
            x_naive = x.to(torch.float32)
            variance = x_naive.pow(2).mean(dim=-1, keepdim=True)
            x_naive = x_naive * torch.rsqrt(variance + rmsnorm.EPS)
            x_naive = (x_naive * weight.float()).to(torch.float16)
            return x_naive.cpu().numpy()

        def flashinfer():
            import flashinfer

            out = torch.empty_like(x)

            def func():
                return flashinfer.norm.rmsnorm(x.clone(), weight, rmsnorm.EPS, enable_pdl=False, out=out)

            ms = bench(func, warmup=10, repeat=30, proton_name="flashinfer")
            print(f"flashinfer time: {ms:.3f} ms")
            return out.cpu().numpy()

        def tir():
            DEV = tvm.cuda(0)
            x_tvm = tvm.runtime.tensor(x.cpu().numpy(), DEV)
            weight_tvm = tvm.runtime.tensor(weight.cpu().numpy(), DEV)
            out_tvm = tvm.runtime.empty((batch_size, hidden_size), dtype="float16", device=DEV)

            def func():
                return mod(
                    x_tvm,
                    weight_tvm,
                    out_tvm,
                )

            ms = bench(func, warmup=10, repeat=30, proton_name="tir")
            print(f"tir time: {ms:.3f} ms")
            mod(x_tvm, weight_tvm, out_tvm)
            return out_tvm.numpy()

        out_naive = naive()
        out_fused = flashinfer()
        out_tir = tir()
        torch.testing.assert_close(out_fused, out_naive, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(out_tir, out_naive, rtol=1e-3, atol=1e-3)

    # compile tir kernel
    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": rmsnorm.get_rmsnorm_kernel(hidden_size)})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")

    with ProtonContext("rms_norm"):
        test_dynamic_batch(batch_size, mod)


if __name__ == "__main__":
    import itertools

    hidden_size_list = [5120, 128]
    batch_size_list = [1, 2, 4, 8, 16, 32, 64, 128, 4113]
    for hidden_size, batch_size in itertools.product(hidden_size_list, batch_size_list):
        test_rmsnorm(hidden_size, batch_size)
