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

import flashinfer.fused_moe as fused_moe
import numpy as np
import nvtx
import pytest
import torch
from flashinfer.autotuner import autotune
from flashinfer.testing.utils import bench_gpu_time
from torch.nn import functional as F

sys.path.insert(0, os.path.join(os.environ.get("TIRX_KERNELS_PATH", os.path.expanduser("~/tirx-kernels/kernels")), "moe"))
import fused_moe as fused_moe_kernel  # noqa: E402


@pytest.mark.parametrize("config", fused_moe_kernel.test_configs)
def bench_fused_moe(config):
    batch_size = config["batch_size"]
    hidden_size = config["hidden_size"]
    num_experts = config["num_experts"]
    top_k = config["top_k"]
    intermediate_size = config["intermediate_size"]

    torch.manual_seed(42)

    hidden_states, w13, w31, w2, router_logits, routing_weights, selected_experts = (
        fused_moe_kernel.gen_input(batch_size, hidden_size, num_experts, top_k, intermediate_size)
    )

    def print_result(backend, median_ms):
        print(f"backend: {backend}")
        print(f"{'input':<15} {'weight1':<20} {'weight2':<20} {'time(ms)'}")
        print(
            f"{tuple(hidden_states.shape)!s:<15} "
            f"{tuple(w13.shape)!s:<20} "
            f"{tuple(w2.shape)!s:<20} "
            f"{median_ms:.3f}"
        )

    def naive():
        results = torch.zeros_like(hidden_states)
        for expert_id in range(num_experts):
            mask = selected_experts == expert_id
            if not mask.sum():
                continue
            batch_idx, nth_expert = torch.where(mask)
            w31_expert = w31[expert_id]  # [2 * intermediate_size, hidden_size]
            w2_expert = w2[expert_id]  # [hidden_size, intermediate_size]
            # Split w13 into w1 and w3
            w3_expert, w1_expert = torch.chunk(w31_expert, 2, dim=0)
            expert_inputs = hidden_states[batch_idx]
            inter = F.silu(expert_inputs @ w1_expert.t()) * (expert_inputs @ w3_expert.t())
            output = inter @ w2_expert.t()
            results[batch_idx] += routing_weights[batch_idx, nth_expert, None] * output
        return results.view_as(hidden_states).cpu().numpy()

    def flashinfer():
        output = torch.zeros_like(hidden_states)

        def func():
            with nvtx.annotate("flashinfer"):
                return fused_moe.cutlass_fused_moe(
                    hidden_states,
                    selected_experts.to(torch.int),
                    routing_weights,
                    w31,
                    w2,
                    hidden_states.dtype,
                    quant_scales=[],
                    output=output,
                )

        # Warmup
        with nvtx.annotate("flashinfer-warmup"):
            for _ in range(3):
                with torch.inference_mode(), autotune(True):
                    _ = func()

        ms_list = bench_gpu_time(func)
        median_ms = np.median(ms_list)
        print_result("flashinfer", median_ms)
        return output.cpu().numpy()

    def sglang():
        def func():
            with nvtx.annotate("sglang"):
                return fused_moe_kernel.fused_moe_sglang(
                    hidden_states,
                    w13,
                    w2,
                    router_logits,
                    routing_weights,
                    selected_experts.to(torch.int),
                )

        # Warmup
        with nvtx.annotate("sglang-warmup"):
            for _ in range(3):
                with torch.inference_mode(), autotune(True):
                    _ = func()

        ms_list = bench_gpu_time(func)
        median_ms = np.median(ms_list)
        print_result("sglang", median_ms)
        return func().cpu().numpy()

    # output_naive = naive()
    output_sglang = sglang()
    output_flashinfer = flashinfer()
    np.testing.assert_allclose(output_sglang, output_flashinfer, rtol=1e-2, atol=5e-2)


if __name__ == "__main__":
    for config in fused_moe_kernel.test_configs:
        bench_fused_moe(config)
