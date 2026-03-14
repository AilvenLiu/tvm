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
import pytest

import tvm
from tvm.tirx.bench.utils import ProtonContext, bench

sys.path.insert(
    0,
    os.path.join(
        os.environ.get("TIRX_KERNELS_PATH", os.path.expanduser("~/tirx-kernels/kernels")),
        "activation",
    ),
)
import fused_split_silu_multiply

# Re-export for backward compatibility (qwen3_layer.py, qwen3_model.py import from here)
get_fused_split_silu_multiply_kernel_cp_sync = (
    fused_split_silu_multiply.get_fused_split_silu_multiply_kernel_cp_sync
)
get_fused_split_silu_multiply_kernel_cp_async = (
    fused_split_silu_multiply.get_fused_split_silu_multiply_kernel_cp_async
)
get_fused_split1_silu1_multiply1_kernel = (
    fused_split_silu_multiply.get_fused_split1_silu1_multiply1_kernel
)


@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16, 32, 64, 128, 4133])
def test(batch_size):
    out_dim = 25600
    input_cat = fused_split_silu_multiply.perpare_data(batch_size, out_dim)

    def naive():
        import torch

        input_cat_naive = input_cat.clone().to("cuda")

        def func():
            return input_cat_naive[..., out_dim:] * torch.nn.functional.silu(
                input_cat_naive[..., :out_dim]
            )

        ms = bench(func, warmup=0, repeat=30, proton_name="naive")
        print(f"torch time: {ms:.3f} ms")

        return func().cpu().numpy()

    def tir_cp_async():
        DEV = tvm.cuda(0)
        input_cat_tvm = tvm.runtime.tensor(input_cat.clone(), device=DEV)
        output_tvm = tvm.runtime.empty((batch_size, out_dim), dtype="float16", device=DEV)
        target = tvm.target.Target("cuda")
        with target:
            mod = tvm.IRModule(
                {
                    "main": fused_split_silu_multiply.get_fused_split_silu_multiply_kernel_cp_async(
                        out_dim
                    )
                }
            )
            mod = tvm.compile(mod, target=target, tir_pipeline="tirx")

            def func():
                return mod(input_cat_tvm, output_tvm)

            ms = bench(func, warmup=0, repeat=30, proton_name="tir_cp_async")
            print(f"TIR time: {ms:.3f} ms")

        return output_tvm.numpy()

    def tir_cp_sync():
        DEV = tvm.cuda(0)
        input_cat_tvm = tvm.runtime.tensor(input_cat.clone(), device=DEV)
        output_tvm = tvm.runtime.empty((batch_size, out_dim), dtype="float16", device=DEV)
        target = tvm.target.Target("cuda")
        with target:
            mod = tvm.IRModule(
                {
                    "main": fused_split_silu_multiply.get_fused_split_silu_multiply_kernel_cp_sync(
                        out_dim
                    )
                }
            )
            mod = tvm.compile(mod, target=target, tir_pipeline="tirx")

            def func():
                return mod(input_cat_tvm, output_tvm)

            ms = bench(func, warmup=0, repeat=30, proton_name="tir_cp_sync")
            print(f"TIR time: {ms:.3f} ms")
        return output_tvm.numpy()

    def tir_old():
        DEV = tvm.cuda(0)
        input_cat_tvm = tvm.runtime.tensor(
            input_cat.clone().reshape(1, batch_size, out_dim * 2), device=DEV
        )
        output_tvm = tvm.runtime.empty((1, batch_size, out_dim), dtype="float16", device=DEV)
        target = tvm.target.Target("cuda")
        with target:
            mod = tvm.IRModule(
                {"main": fused_split_silu_multiply.get_fused_split1_silu1_multiply1_kernel(out_dim)}
            )
            mod = tvm.compile(mod, target=target)

            def func():
                return mod(input_cat_tvm, output_tvm)

            ms = bench(func, warmup=0, repeat=30, proton_name="tir_old")
            print(f"TIR time: {ms:.3f} ms")

        return output_tvm.numpy().reshape(batch_size, out_dim)

    def flashinfer():
        import flashinfer
        import torch

        input_cat_flashinfer = input_cat.clone().to("cuda")
        out = torch.empty((batch_size, out_dim), dtype=torch.float16, device="cuda")

        def func():
            return flashinfer.activation.silu_and_mul(input_cat_flashinfer, out)

        ms = bench(func, warmup=0, repeat=30, proton_name="flashinfer")
        print(f"FlashInfer time: {ms:.3f} ms")
        return out.cpu().numpy()

    with ProtonContext("fused_split_silu_multiply"):
        output_naive = naive()
        output_tir_cp_async = tir_cp_async()
        output_tir_cp_sync = tir_cp_sync()
        output_tir_old = tir_old()
        output_flashinfer = flashinfer()

    np.testing.assert_allclose(output_naive, output_tir_cp_async, rtol=5e-3, atol=5e-3)
    np.testing.assert_allclose(output_tir_cp_async, output_tir_cp_sync, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(output_tir_cp_async, output_tir_old, rtol=5e-3, atol=5e-3)
    np.testing.assert_allclose(output_tir_cp_async, output_flashinfer, rtol=5e-3, atol=5e-3)


if __name__ == "__main__":
    for batch_size in [1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128]:
        test(batch_size)
