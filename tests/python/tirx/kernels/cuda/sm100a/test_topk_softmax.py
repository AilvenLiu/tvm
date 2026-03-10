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
from tvm.tirx.bench.utils import ProtonContext, bench
from tvm.tirx.megakernel.utils.utils import get_source

sys.path.insert(0, os.path.join(os.environ.get("TIRX_KERNELS_PATH", os.path.expanduser("~/tirx-kernels/kernels")), "loss"))
import topk_softmax  # noqa: E402


@tvm.testing.requires_cuda_compute_version(10, exact=False)
def test(kernel, num_tokens, num_experts, topk, dtype):
    arg_dict = topk_softmax.prepare_data(num_tokens, num_experts, topk, dtype)

    def tir(arg_dict):
        DEV = tvm.cuda(0)
        tvm_arg_dict = {}
        target = tvm.target.Target("cuda")

        for key, value in arg_dict.items():
            tvm_arg_dict[key] = tvm.runtime.tensor(value, device=DEV)

        with target:

            def func():
                kernel(
                    tvm_arg_dict["gating_output"],
                    tvm_arg_dict["topk_weights"],
                    tvm_arg_dict["topk_indices"],
                )

            ms = bench(func, warmup=10, repeat=30, proton_name="tir")
            print(f"TIR time: {ms:.3f} ms")

            return tvm_arg_dict["topk_weights"].numpy(), tvm_arg_dict["topk_indices"].numpy()

    def sglang(arg_dict):
        import torch
        from sgl_kernel import topk_softmax as topk_softmax_sglang

        torch_dev = torch.device("cuda")
        std_arg_dict = {}

        for key, value in arg_dict.items():
            std_arg_dict[key] = value.to(torch_dev)

        def func():
            topk_softmax_sglang(
                topk_weights=std_arg_dict["topk_weights"],
                topk_ids=std_arg_dict["topk_indices"],
                gating_output=std_arg_dict["gating_output"],
            )

        ms = bench(func, warmup=10, repeat=30, proton_name="sglang")
        print(f"sglang time: {ms:.3f} ms")

        return (
            std_arg_dict["topk_weights"].cpu().numpy(),
            std_arg_dict["topk_indices"].cpu().numpy(),
        )

    with ProtonContext("blackwell_benchmark"):
        weights_tir, indices_tir = tir(arg_dict)
        weights_sglang, indices_sglang = sglang(arg_dict)

    np.testing.assert_allclose(weights_tir, weights_sglang, rtol=1e-3, atol=1e-3)
    np.testing.assert_equal(indices_tir, indices_sglang)


if __name__ == "__main__":
    itr = 0
    for dtype in ["float16", "float32"]:
        for num_experts in [32, 64, 128, 256]:
            if num_experts == 256 and dtype == "float32":
                continue
            mega_kernel_wrapper_static = topk_softmax.TopkSoftmaxKernel()
            mega_static_module = mega_kernel_wrapper_static.get_module_static(num_experts, dtype)
            src, lib_static = get_source(mega_static_module)
            # print(src)

            for num_tokens in [128, 512, 1024, 2048, 4096, 8192, 16384, 32768]:
                for topk in [1, 2, 4, 8]:
                    n_bytes_dict = {"float16": 2, "bfloat16": 2, "float32": 4}
                    VPT = 16 // n_bytes_dict.get(dtype)
                    ROWS_PER_WARP = (VPT * 32) // num_experts
                    num_blocks = ((num_tokens + ROWS_PER_WARP - 1) // ROWS_PER_WARP + 8 - 1) // 8
                    print(
                        f"experiment {itr}: if nonpersistent, would need <<<{num_blocks}, 256>>>, num_tokens {num_tokens}, num_experts {num_experts}, topk {topk}, dtype {dtype}"  # noqa: E501
                    )
                    test(lib_static["main"], num_tokens, num_experts, topk, dtype)
                    itr += 1
