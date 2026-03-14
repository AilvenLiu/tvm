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

import functools
import os
import sys

import numpy as np
import pytest
import torch
from sglang.srt.layers.moe.fused_moe_triton.fused_moe_triton_config import (
    try_get_optimal_moe_config,
)
from sglang.srt.layers.moe.fused_moe_triton.fused_moe_triton_kernels import invoke_fused_moe_kernel
from triton import language as tl

import tvm
import tvm.testing
from tvm.tirx.bench.utils import ProtonContext, bench

sys.path.insert(
    0,
    os.path.join(
        os.environ.get("TIRX_KERNELS_PATH", os.path.expanduser("~/tirx-kernels/kernels")), "gemm"
    ),
)
from group_gemm import (
    MAX_BLK_M,
    gen_input,
    get_group_gemm_kernel,
    prepare_group_gemm,
)

test_configs = [
    {
        "batch_size": b,  # M
        "hidden_size": 2048,  # K
        "num_experts": 128,  # E
        "top_k": 8,  # Top-K
        "intermediate_size": 768,  # N
    }
    for b in [2048, 512, 128, 1]
]

DEBUG = False


@pytest.mark.skip(reason="Tensor Memory Leak")
@pytest.mark.parametrize("task", test_configs)
def test_group_gemm(task):
    batch_size = task["batch_size"]
    hidden_size = task["hidden_size"]
    num_experts = task["num_experts"]
    top_k = task["top_k"]
    intermediate_size = task["intermediate_size"]

    (
        x1,
        w13,
        routing_weights,
        selected_experts,
        x2,
        w2,
    ) = gen_input(batch_size, hidden_size, num_experts, top_k, intermediate_size)

    def tir():
        def torch_to_tvm(tensor):
            return tvm.runtime.from_dlpack(torch.utils.dlpack.to_dlpack(tensor))

        target = tvm.target.Target("cuda")
        dev = tvm.cuda(0)
        sorted_token_ids, expert_ids, num_tokens_post_padded = prepare_group_gemm(
            MAX_BLK_M, num_experts, selected_experts
        )
        safe_index = torch.clamp(sorted_token_ids, 0, batch_size * top_k - 1)[
            : num_tokens_post_padded.item()
        ]
        x1_tvm = torch_to_tvm(x1[safe_index // top_k])
        x2_tvm = torch_to_tvm(x2[safe_index])
        w13_tvm = torch_to_tvm(w13)
        routing_weights_tvm = torch_to_tvm(
            routing_weights.view(
                batch_size * top_k,
            )
        )
        w2_tvm = torch_to_tvm(w2)
        out1_tvm = tvm.runtime.empty(
            (num_tokens_post_padded.item(), 2 * intermediate_size), dtype="float16", device=dev
        )
        out2_tvm = tvm.runtime.empty(
            (num_tokens_post_padded.item(), hidden_size), dtype="float16", device=dev
        )

        sorted_token_ids_tvm = torch_to_tvm(sorted_token_ids[: num_tokens_post_padded.item()])
        valid_num_tokens = torch.empty(
            num_tokens_post_padded.item() // MAX_BLK_M, dtype=torch.int32
        )
        for i in range(num_tokens_post_padded.item() // MAX_BLK_M):
            valid_num_tokens[i] = torch.sum(
                (sorted_token_ids[i * MAX_BLK_M : (i + 1) * MAX_BLK_M] >= 0)
                & (sorted_token_ids[i * MAX_BLK_M : (i + 1) * MAX_BLK_M] < batch_size * top_k)
            )
        valid_num_tokens_tvm = torch_to_tvm(valid_num_tokens.cuda())
        expert_ids_tvm = torch_to_tvm(expert_ids)
        num_tokens_post_padded_tvm = torch_to_tvm(num_tokens_post_padded)
        grp_gemm1 = get_group_gemm_kernel(
            hidden_size,
            num_experts,
            top_k,
            intermediate_size * 2,
            acc_output=False,
            low_batch=batch_size < 2048,
        )
        grp_gemm2 = get_group_gemm_kernel(
            intermediate_size,
            num_experts,
            top_k,
            hidden_size,
            acc_output=True,
            low_batch=batch_size < 2048,
        )
        with target:
            mod_1 = tvm.IRModule({"main": grp_gemm1})
            mod_1 = tvm.compile(mod_1, target=target, tir_pipeline="tirx")
            mod_2 = tvm.IRModule({"main": grp_gemm2})
            mod_2 = tvm.compile(mod_2, target=target, tir_pipeline="tirx")

            def func1():
                mod_1(
                    x1_tvm,
                    w13_tvm,
                    out1_tvm,
                    expert_ids_tvm,
                    sorted_token_ids_tvm,
                    routing_weights_tvm,
                    valid_num_tokens_tvm,
                    num_tokens_post_padded_tvm,
                )

            def func2():
                mod_2(
                    x2_tvm,
                    w2_tvm,
                    out2_tvm,
                    expert_ids_tvm,
                    sorted_token_ids_tvm,
                    routing_weights_tvm,
                    valid_num_tokens_tvm,
                    num_tokens_post_padded_tvm,
                )

            bench(func1, warmup=0, repeat=30, proton_name="tir1")
            bench(func2, warmup=0, repeat=30, proton_name="tir2")
            # reset out2 to get the correct result
            out2_tvm = tvm.runtime.empty(
                (num_tokens_post_padded.item(), hidden_size), dtype="float16", device=dev
            )
            mod_2(
                x2_tvm,
                w2_tvm,
                out2_tvm,
                expert_ids_tvm,
                sorted_token_ids_tvm,
                routing_weights_tvm,
                valid_num_tokens_tvm,
                num_tokens_post_padded_tvm,
            )

            def scatter_out(out_tvm, *shape):
                tmp_out = torch.from_numpy(out_tvm.numpy())
                out = torch.zeros(shape, dtype=torch.float16)
                index_for_scatter = sorted_token_ids[: num_tokens_post_padded.item()].cpu()
                for i in range(num_tokens_post_padded.item()):
                    if index_for_scatter[i] >= 0 and index_for_scatter[i] < shape[0]:
                        out[index_for_scatter[i]] = tmp_out[i]
                return out

            out1 = scatter_out(out1_tvm, batch_size * top_k, intermediate_size * 2).view(
                batch_size, top_k, intermediate_size * 2
            )
            out2 = out2_tvm.numpy()[:batch_size, :]
            return out1, out2

    def sglang():
        def get_config(batch_size):
            get_config_func = functools.partial(
                try_get_optimal_moe_config,
                w13.shape,
                (w2.shape[0], w2.shape[1], w2.shape[2]),
                selected_experts.shape[1],
                "float16",
                block_shape=None,
            )
            return get_config_func(batch_size)

        sgL_config = get_config(batch_size)
        print(f"sgL_config: {sgL_config}")
        sorted_token_ids, expert_ids, num_tokens_post_padded = prepare_group_gemm(
            sgL_config["BLOCK_SIZE_M"], num_experts, selected_experts
        )

        out1 = torch.empty(
            (batch_size, top_k, 2 * intermediate_size), dtype=torch.float16, device="cuda"
        )
        out2 = torch.empty((batch_size, top_k, hidden_size), dtype=torch.float16, device="cuda")

        def func1():
            return invoke_fused_moe_kernel(
                x1,
                w13,
                None,  # bias
                out1,
                None,  # A_scale
                None,  # B_scale
                None,  # B_zp
                topk_weights=routing_weights,
                topk_ids=selected_experts,
                sorted_token_ids=sorted_token_ids,
                expert_ids=expert_ids,
                num_tokens_post_padded=num_tokens_post_padded,
                mul_routed_weight=False,
                top_k=top_k,
                config=sgL_config,
                compute_type=tl.float16,
                use_fp8_w8a8=False,
                use_int8_w8a8=False,
                use_int8_w8a16=False,
                use_int4_w4a16=False,
                per_channel_quant=False,
                block_shape=None,
            )

        ms1 = bench(func1, warmup=3, repeat=30, proton_name="sglang gemm1", debug=DEBUG)
        print("sglang gemm1", ms1)

        def func2():
            return invoke_fused_moe_kernel(
                x2,
                w2,
                None,  # bias
                out2,
                None,  # A_scale
                None,  # B_scale
                None,  # B_zp
                topk_weights=routing_weights,
                topk_ids=selected_experts,
                sorted_token_ids=sorted_token_ids,
                expert_ids=expert_ids,
                num_tokens_post_padded=num_tokens_post_padded,
                mul_routed_weight=True,
                top_k=1,
                config=sgL_config,
                compute_type=tl.float16,
                use_fp8_w8a8=False,
                use_int8_w8a8=False,
                use_int8_w8a16=False,
                use_int4_w4a16=False,
                per_channel_quant=False,
                block_shape=None,
            )

        ms2 = bench(func2, warmup=3, repeat=30, proton_name="sglang gemm2", debug=DEBUG)
        print("sglang gemm2", ms2)
        return out1.cpu().numpy(), out2.sum(dim=1).cpu().numpy()

    with ProtonContext("group_gemm", debug=DEBUG):
        out1_tir, out2_tir = tir()
        out1_sglang, out2_sglang = sglang()
    try:
        np.testing.assert_allclose(out1_tir, out1_sglang, rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(out2_tir, out2_sglang, rtol=2e-2, atol=2e-2)
    except Exception as e:
        print(e)


if __name__ == "__main__":
    for config in test_configs:
        print(f"testing config: {config}", flush=True)
        test_group_gemm(config)
