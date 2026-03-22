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

import argparse

import numpy as np
import pytest
from tirx_kernels.megakernel.moe_full_layer import (
    MegaKernelMOEFullLayer,
    prepare_data,
)

import tvm
import tvm.testing
from tvm.tirx.bench.utils import ProtonContext, bench, export_to_perfetto_trace
from tvm.tirx.megakernel.utils.config import (
    event_type_names,
)
from tvm.tirx.megakernel.utils.utils import get_source


@tvm.testing.requires_cuda_compute_version(10, exact=True)
@pytest.mark.skip
def test(batch_size, seq_len, mega_kernel_static, mega_kernel_dynamic, mega_kernel_wrapper, sess):
    """Test function combining attention and MoE"""
    arg_dict = prepare_data(batch_size, seq_len, mega_kernel_wrapper)

    def tir(arg_dict, mk: MegaKernelMOEFullLayer, scheduler: Literal["static", "dynamic"]):
        """Run TIR kernel (supports both static and dynamic scheduler)"""
        REPEAT = 100
        DEV = tvm.cuda(0)
        tvm_arg_dict = {}
        target = tvm.target.Target("cuda")

        # Prepare execution queue based on scheduler type
        if scheduler == "static":
            # Static scheduler needs exec queue with attention task info
            exec_queue = generate_exec_queue(
                batch_size,
                arg_dict["attn_task_num"].item(),
                mk.config,
                mk.world_size,
                mk.num_etensors[False],
                "static",
            )
            tvm_arg_dict["exec_queue"] = tvm.runtime.tensor(exec_queue, DEV)
        else:
            # Dynamic scheduler
            exec_queue = generate_exec_queue(
                None, None, mk.config, mk.world_size, mk.num_etensors[True], "dynamic"
            )
            for i in range(REPEAT):
                tvm_arg_dict[f"queue_tasks_{i}"] = tvm.runtime.tensor(exec_queue.tasks, DEV)
                tvm_arg_dict[f"queue_head_{i}"] = tvm.runtime.tensor(exec_queue.head, DEV)
                tvm_arg_dict[f"queue_tail_{i}"] = tvm.runtime.tensor(exec_queue.tail, DEV)

        # Prepare attention intermediate buffers
        tvm_arg_dict["o_partial_attn"] = tvm.runtime.tensor(
            np.zeros(
                [mk.MAX_NUM_KV_SPLITS * mk.NUM_KEY_VALUE_HEADS * mk.HEAD_DIM], dtype=np.float32
            ),
            DEV,
        )
        tvm_arg_dict["lse_partial"] = tvm.runtime.tensor(
            np.zeros([mk.MAX_NUM_KV_SPLITS * mk.NUM_KEY_VALUE_HEADS], dtype=np.float32), DEV
        )
        tvm_arg_dict["partial_qkv"] = tvm.runtime.tensor(
            np.zeros(
                [
                    mk.SPLIT_QKV_PROJECT,
                    batch_size,
                    (mk.NUM_ATTENTION_HEADS + 2 * mk.NUM_KEY_VALUE_HEADS) * mk.HEAD_DIM,
                ],
                dtype=np.float32,
            ),
            DEV,
        )
        tvm_arg_dict["partial_o"] = tvm.runtime.tensor(
            np.zeros([mk.SPLIT_O_PROJECT, batch_size, mk.HIDDEN_SIZE], dtype=np.float32), DEV
        )
        tvm_arg_dict["before_o_allreduce"] = tvm.runtime.tensor(
            np.zeros([batch_size, mk.HIDDEN_SIZE], dtype=np.float16), DEV
        )

        # Get inverse plan info
        res = get_inverse_plan_info(
            batch_size,
            mk.NUM_KEY_VALUE_HEADS,
            arg_dict["q_indptr"],
            arg_dict["kv_head_idx"],
            arg_dict["attn_task_num"].item(),
        )
        tvm_arg_dict["inverse_indptr"], tvm_arg_dict["inverse_indices"] = res

        # Process append_pos (different from flashinfer)
        append_pos = arg_dict["append_pos"].clone()
        for b in range(batch_size):
            append_pos[b] = (
                arg_dict["page_kv_indices"][
                    (arg_dict["page_kv_indptr"][b] * mk.PAGE_SIZE + append_pos[b]) // mk.PAGE_SIZE
                ]
                * mk.PAGE_SIZE
                + append_pos[b] % mk.PAGE_SIZE
            )

        # Copy all arg_dict tensors to device
        for key, value in arg_dict.items():
            tvm_arg_dict[key] = tvm.runtime.tensor(value, device=DEV)
        tvm_arg_dict["append_pos"] = tvm.runtime.tensor(append_pos, device=DEV)

        tvm_arg_dict["etensor_workspace"] = tvm.runtime.tensor(
            np.zeros([mk.ETENSOR_WORKSPACE_SIZE], dtype=np.int32), device=DEV
        )

        # Prepare per-iteration buffers
        for i in range(REPEAT):
            tvm_arg_dict[f"residual_{i}"] = tvm.runtime.tensor(
                arg_dict["residual"].to(torch.float32), device=DEV
            )

        tvm_arg_dict["profiler_buffer"] = tvm.runtime.tensor(
            np.zeros([mk.PROFILER_BUFFER_SIZE], dtype=np.uint64), device=DEV
        )

        # Run kernel
        with target:
            iter = 0
            if scheduler == "static":
                kernel = mega_kernel_static["main"]
            else:
                kernel = mega_kernel_dynamic["main"]
            work_arg_dict = tvm_arg_dict

            if scheduler == "static":

                def func():
                    nonlocal iter
                    kernel(
                        # Input and output
                        work_arg_dict["hidden_state"],
                        work_arg_dict[f"residual_{iter}"],
                        work_arg_dict["output"],
                        # Attention weights
                        work_arg_dict["qkv_proj_weight"],
                        work_arg_dict["o_proj_weight"],
                        work_arg_dict["q_rms_wight"],
                        work_arg_dict["k_rms_wight"],
                        work_arg_dict["attn_add_rms_weight"],
                        work_arg_dict["mlp_add_rms_weight"],
                        # MoE weights
                        work_arg_dict["gate_weight"],
                        work_arg_dict["shuffled_grp_gate_up_weight"],
                        work_arg_dict["grp_down_weight"],
                        # Attention cache and plan
                        work_arg_dict["cos_sin_cache"],
                        work_arg_dict["rope_pos"],
                        work_arg_dict["kv_cache"],
                        work_arg_dict["append_pos"],
                        work_arg_dict["q_indptr"],
                        work_arg_dict["kv_indptr"],
                        work_arg_dict["partial_indptr"],
                        work_arg_dict["page_kv_indices"],
                        work_arg_dict["q_len"],
                        work_arg_dict["kv_len"],
                        work_arg_dict["q_start"],
                        work_arg_dict["kv_start"],
                        work_arg_dict["kv_end"],
                        work_arg_dict["kv_head_idx"],
                        work_arg_dict["work_indptr"],
                        work_arg_dict["len_kv_chunk"],
                        work_arg_dict["num_qo_len"],
                        work_arg_dict["merge_indptr"],
                        work_arg_dict["merge_o_indices"],
                        work_arg_dict["inverse_indptr"],
                        work_arg_dict["inverse_indices"],
                        # Attention intermediate
                        work_arg_dict["partial_qkv"],
                        work_arg_dict["qkv"],
                        work_arg_dict["o"],
                        work_arg_dict["o_partial_attn"],
                        work_arg_dict["lse_partial"],
                        work_arg_dict["partial_o"],
                        work_arg_dict["before_o_allreduce"],
                        work_arg_dict["hidden_state_attn_mlp"],
                        # MoE intermediate
                        work_arg_dict["gating_output"],
                        work_arg_dict["topk_weights"],
                        work_arg_dict["topk_indices"],
                        work_arg_dict["sorted_token_ids"],
                        work_arg_dict["expert_ids"],
                        work_arg_dict["num_valid_tokens"],
                        work_arg_dict["num_tokens_post_pad"],
                        work_arg_dict["cumsum_buffer"],
                        work_arg_dict["reordered_hidden_state"],
                        work_arg_dict["gate_up_output"],
                        work_arg_dict["silu_mul_output"],
                        # Event tensors
                        work_arg_dict["etensor_workspace"],
                        # Execution queue (static)
                        work_arg_dict["exec_queue"],
                        work_arg_dict["profiler_buffer"],
                    )
                    iter += 1

            else:

                def func():
                    nonlocal iter
                    kernel(
                        # Input and output
                        work_arg_dict["hidden_state"],
                        work_arg_dict[f"residual_{iter}"],
                        work_arg_dict["output"],
                        # Attention weights
                        work_arg_dict["qkv_proj_weight"],
                        work_arg_dict["o_proj_weight"],
                        work_arg_dict["q_rms_wight"],
                        work_arg_dict["k_rms_wight"],
                        work_arg_dict["attn_add_rms_weight"],
                        work_arg_dict["mlp_add_rms_weight"],
                        # MoE weights
                        work_arg_dict["gate_weight"],
                        work_arg_dict["shuffled_grp_gate_up_weight"],
                        work_arg_dict["grp_down_weight"],
                        # Attention cache and plan
                        work_arg_dict["cos_sin_cache"],
                        work_arg_dict["rope_pos"],
                        work_arg_dict["kv_cache"],
                        work_arg_dict["append_pos"],
                        work_arg_dict["q_indptr"],
                        work_arg_dict["kv_indptr"],
                        work_arg_dict["partial_indptr"],
                        work_arg_dict["page_kv_indices"],
                        work_arg_dict["q_len"],
                        work_arg_dict["kv_len"],
                        work_arg_dict["q_start"],
                        work_arg_dict["kv_start"],
                        work_arg_dict["kv_end"],
                        work_arg_dict["kv_head_idx"],
                        work_arg_dict["work_indptr"],
                        work_arg_dict["len_kv_chunk"],
                        work_arg_dict["num_qo_len"],
                        work_arg_dict["merge_indptr"],
                        work_arg_dict["merge_o_indices"],
                        work_arg_dict["inverse_indptr"],
                        work_arg_dict["inverse_indices"],
                        # Attention intermediate
                        work_arg_dict["partial_qkv"],
                        work_arg_dict["qkv"],
                        work_arg_dict["o"],
                        work_arg_dict["o_partial_attn"],
                        work_arg_dict["lse_partial"],
                        work_arg_dict["partial_o"],
                        work_arg_dict["before_o_allreduce"],
                        work_arg_dict["hidden_state_attn_mlp"],
                        # MoE intermediate
                        work_arg_dict["gating_output"],
                        work_arg_dict["topk_weights"],
                        work_arg_dict["topk_indices"],
                        work_arg_dict["sorted_token_ids"],
                        work_arg_dict["expert_ids"],
                        work_arg_dict["num_valid_tokens"],
                        work_arg_dict["num_tokens_post_pad"],
                        work_arg_dict["cumsum_buffer"],
                        work_arg_dict["reordered_hidden_state"],
                        work_arg_dict["gate_up_output"],
                        work_arg_dict["silu_mul_output"],
                        # Event tensors
                        work_arg_dict["etensor_workspace"],
                        # Execution queue (dynamic)
                        work_arg_dict[f"queue_tasks_{iter}"],
                        work_arg_dict[f"queue_head_{iter}"],
                        work_arg_dict[f"queue_tail_{iter}"],
                        work_arg_dict["profiler_buffer"],
                    )
                    iter += 1

            ms = bench(func, warmup=1, repeat=7, proton_name=f"tir-{scheduler}")
            print(f"TIR ({scheduler}) time: {ms:.3f} ms")
            if mk.profiler_on:
                export_to_perfetto_trace(
                    tvm_arg_dict["profiler_buffer"].numpy(),
                    f"{scheduler}-moe-full-layer-bs{batch_size}-tp{mk.world_size}.perfetto-trace",
                    event_type_names,
                )
            return tvm_arg_dict["output"].numpy(), tvm_arg_dict["residual_0"].numpy().astype(
                np.float16
            )

    def std(arg_dict, use_prefill, mk: MegaKernelMOEFullLayer):
        """Standard reference implementation combining attention and MoE"""
        import flashinfer

        torch_dev = torch.device("cuda")
        std_arg_dict = {}

        def func():
            for key, value in arg_dict.items():
                std_arg_dict[key] = value.clone().to(torch_dev)

            # ===== Attention part (from test_layer.py) =====
            out_f = torch.zeros(
                batch_size, mk.NUM_ATTENTION_HEADS, mk.HEAD_DIM, dtype=torch.float16, device="cuda"
            )
            lse_f = torch.zeros(
                batch_size, mk.NUM_ATTENTION_HEADS, dtype=torch.float32, device="cuda"
            )

            if use_prefill:
                workspace_buffer = torch.empty(1024 * 1024 * 1024, dtype=torch.int8).to(0)
                wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace_buffer, "HND")
                wrapper.plan(
                    torch.arange(0, batch_size + 1, dtype=torch.int32).to(0),
                    std_arg_dict["page_kv_indptr"],
                    std_arg_dict["page_kv_indices"],
                    std_arg_dict["page_kv_last_page_len"],
                    mk.NUM_ATTENTION_HEADS,
                    mk.NUM_KEY_VALUE_HEADS,
                    mk.HEAD_DIM,
                    mk.PAGE_SIZE,
                    pos_encoding_mode="NONE",
                    kv_data_type=torch.float16,
                    q_data_type=torch.float16,
                )
            else:
                wrapper = flashinfer.BatchAttention("HND")
                wrapper.plan(
                    torch.arange(0, batch_size + 1, dtype=torch.int32).to(0),
                    std_arg_dict["page_kv_indptr"],
                    std_arg_dict["page_kv_indices"],
                    torch.tensor([seq_len] * batch_size, dtype=torch.int32).to(0),
                    mk.NUM_ATTENTION_HEADS,
                    mk.NUM_KEY_VALUE_HEADS,
                    mk.HEAD_DIM,
                    mk.HEAD_DIM,
                    mk.PAGE_SIZE,
                    kv_data_type=torch.float16,
                    q_data_type=torch.float16,
                )

            # QKV projection
            qkv = torch.matmul(
                std_arg_dict["hidden_state"], std_arg_dict["qkv_proj_weight"].T
            ).reshape(batch_size, -1, mk.HEAD_DIM)
            q, k, v = torch.split(
                qkv,
                [mk.NUM_ATTENTION_HEADS, mk.NUM_KEY_VALUE_HEADS, mk.NUM_KEY_VALUE_HEADS],
                dim=1,
            )

            # RMS norm
            q = flashinfer.norm.rmsnorm(
                input=q.reshape(-1, mk.HEAD_DIM),
                weight=std_arg_dict["q_rms_wight"],
                eps=mk.RMS_NORM_EPS,
                enable_pdl=False,
            ).reshape(batch_size, mk.NUM_ATTENTION_HEADS, mk.HEAD_DIM)
            k = flashinfer.norm.rmsnorm(
                input=k.reshape(-1, mk.HEAD_DIM),
                weight=std_arg_dict["k_rms_wight"],
                eps=mk.RMS_NORM_EPS,
                enable_pdl=False,
            ).reshape(batch_size, mk.NUM_KEY_VALUE_HEADS, mk.HEAD_DIM)

            # RoPE
            q, k = flashinfer.rope.apply_rope_with_cos_sin_cache(
                positions=std_arg_dict["rope_pos"],
                query=q.reshape(batch_size, -1),
                key=k.reshape(batch_size, -1),
                head_size=mk.HEAD_DIM,
                cos_sin_cache=std_arg_dict["cos_sin_cache"],
                is_neox=True,
            )

            # Append KV cache
            flashinfer.page.append_paged_kv_cache(
                append_key=k.reshape(batch_size, mk.NUM_KEY_VALUE_HEADS, mk.HEAD_DIM),
                append_value=v,
                batch_indices=torch.arange(batch_size, dtype=torch.int32, device=torch_dev),
                positions=std_arg_dict["append_pos"],
                paged_kv_cache=std_arg_dict["kv_cache"],
                kv_indices=std_arg_dict["page_kv_indices"],
                kv_indptr=std_arg_dict["page_kv_indptr"],
                kv_last_page_len=std_arg_dict["page_kv_last_page_len"],
                kv_layout="HND",
            )

            # Attention
            if use_prefill:
                out_f = wrapper.run(
                    q.reshape(batch_size, mk.NUM_ATTENTION_HEADS, mk.HEAD_DIM),
                    std_arg_dict["kv_cache"],
                )
            else:
                wrapper.run(
                    q.reshape(batch_size, mk.NUM_ATTENTION_HEADS, mk.HEAD_DIM),
                    std_arg_dict["kv_cache"],
                    out_f,
                    lse_f,
                )

            # O projection
            hidden_state_attn_mlp = torch.matmul(
                out_f.reshape(batch_size, mk.NUM_ATTENTION_HEADS * mk.HEAD_DIM),
                std_arg_dict["o_proj_weight"].T,
            )

            # Add residual + RMS norm
            flashinfer.norm.fused_add_rmsnorm(
                input=hidden_state_attn_mlp,
                residual=std_arg_dict["residual"],
                weight=std_arg_dict["attn_add_rms_weight"],
                eps=mk.RMS_NORM_EPS,
                enable_pdl=False,
            )

            # ===== MoE part (from test_moe.py) =====
            gating_output = hidden_state_attn_mlp @ std_arg_dict["gate_weight"].T
            topk_softmax(
                gating_output=gating_output,
                topk_weights=std_arg_dict["topk_weights"],
                topk_ids=std_arg_dict["topk_indices"],
                renormalize=True,
            )

            # Prepare MoE outputs
            out1 = torch.empty(
                (batch_size, mk.NUM_EXPERTS_PER_TOK, 2 * mk.INTERMEDIATE_SIZE),
                dtype=torch.float16,
                device="cuda",
            )
            out2 = torch.empty(
                (batch_size, mk.NUM_EXPERTS_PER_TOK, mk.HIDDEN_SIZE),
                dtype=torch.float16,
                device="cuda",
            )

            def get_config(batch_size):
                get_config_func = functools.partial(
                    try_get_optimal_moe_config,
                    std_arg_dict["grp_gate_up_weight"].shape,
                    std_arg_dict["grp_down_weight"].shape,
                    std_arg_dict["topk_indices"].shape[1],
                    "float16",
                    block_shape=None,
                )
                return get_config_func(batch_size)

            sgL_config = get_config(batch_size)
            sorted_ids_std, expert_ids_std, num_tokens_post_pad_std = moe_align_block_size(
                std_arg_dict["topk_indices"],
                sgL_config["BLOCK_SIZE_M"],
                mk.NUM_EXPERTS,
            )

            # Gate-up projection
            invoke_fused_moe_kernel(
                hidden_state_attn_mlp,
                std_arg_dict["grp_gate_up_weight"],
                None,  # bias
                out1,
                None,  # A_scale
                None,  # B_scale
                None,  # B_zp
                topk_weights=std_arg_dict["topk_weights"],
                topk_ids=std_arg_dict["topk_indices"],
                sorted_token_ids=sorted_ids_std,
                expert_ids=expert_ids_std,
                num_tokens_post_padded=num_tokens_post_pad_std,
                mul_routed_weight=False,
                top_k=mk.NUM_EXPERTS_PER_TOK,
                config=sgL_config,
                compute_type=tl.float16,
                use_fp8_w8a8=False,
                use_int8_w8a8=False,
                use_int8_w8a16=False,
                use_int4_w4a16=False,
                per_channel_quant=False,
                block_shape=None,
            )

            # SiLU + multiply
            silu_mul_out = flashinfer.activation.silu_and_mul(
                out1.view(-1, 2 * mk.INTERMEDIATE_SIZE)
            )

            # Down projection
            invoke_fused_moe_kernel(
                silu_mul_out,
                std_arg_dict["grp_down_weight"],
                None,  # bias
                out2,
                None,  # A_scale
                None,  # B_scale
                None,  # B_zp
                topk_weights=std_arg_dict["topk_weights"],
                topk_ids=std_arg_dict["topk_indices"],
                sorted_token_ids=sorted_ids_std,
                expert_ids=expert_ids_std,
                num_tokens_post_padded=num_tokens_post_pad_std,
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

            # Reduce topk outputs
            output = out2.sum(dim=1)

            # Final add residual + RMS norm
            flashinfer.norm.fused_add_rmsnorm(
                input=output,
                residual=std_arg_dict["residual"],
                weight=std_arg_dict["mlp_add_rms_weight"],
                eps=mk.RMS_NORM_EPS,
                enable_pdl=False,
            )

            return output.cpu().numpy(), std_arg_dict["residual"].cpu().numpy()

        output = func()
        ms = bench(func, warmup=10, repeat=30, proton_name=f"std-use_prefill={use_prefill}")
        print(f"Standard (use_prefill={use_prefill}) time: {ms:.3f} ms")
        return output

    def run():
        if mega_kernel_dynamic["main"] is not None:
            output_tir_dynamic, residual_tir_dynamic = tir(arg_dict, mega_kernel_wrapper, "dynamic")
            print("Dynamic TIR finish", flush=True)
        if mega_kernel_static["main"] is not None:
            output_tir_static, residual_tir_static = tir(arg_dict, mega_kernel_wrapper, "static")
            print("Static TIR finish", flush=True)
        output_std1, residual_std1 = std(arg_dict, use_prefill=True, mk=mega_kernel_wrapper)
        output_std2, residual_std2 = std(arg_dict, use_prefill=False, mk=mega_kernel_wrapper)

        if mega_kernel_dynamic["main"] is not None:
            try:
                np.testing.assert_allclose(output_tir_dynamic, output_std1, rtol=1e-3, atol=1e-2)
                np.testing.assert_allclose(
                    residual_tir_dynamic, residual_std1, rtol=1e-3, atol=1e-2
                )
                print("✓ Dynamic scheduler PASSED", flush=True)
            except Exception as e:
                print(f"✗ Dynamic scheduler FAILED: {e}")
        if mega_kernel_static["main"] is not None:
            try:
                np.testing.assert_allclose(output_tir_static, output_std1, rtol=1e-3, atol=1e-2)
                np.testing.assert_allclose(residual_tir_static, residual_std1, rtol=1e-3, atol=1e-2)
                print("✓ Static scheduler PASSED", flush=True)
            except Exception as e:
                print(f"✗ Static scheduler FAILED: {e}")

    if mega_kernel_wrapper.world_size == 1:
        with ProtonContext("moe_full_layer"):
            run()
    else:
        run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MegaKernel MoE Full Layer testing script.")
    parser.add_argument(
        "--scheduler",
        type=str,
        nargs="+",
        default=["static", "dynamic"],
        choices=["static", "dynamic", "none"],
        help="Scheduler to test (static, dynamic, or none).",
    )
    parser.add_argument(
        "--world-size", type=int, default=1, help="Number of devices (only 1 supported)."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        nargs="+",
        default=[1, 3, 7, 15, 31, 63, 127, 128],
        help="List of batch sizes to test.",
    )
    parser.add_argument(
        "--seq-len", type=int, nargs="+", default=[512], help="List of sequence lengths to test."
    )
    parser.add_argument("--profiler-on", action="store_true", help="Enable the profiler.")
    args = parser.parse_args()

    assert args.world_size == 1, "Currently only world_size=1 is supported"

    testing_scheduler = set(args.scheduler)
    mega_kernel_wrapper = MegaKernelMOEFullLayer(
        config=qwen3_30b_a3b_config, world_size=args.world_size, profiler_on=args.profiler_on
    )
    if "static" in testing_scheduler:
        print("\nCompiling static scheduler module...")
        mega_static_module = mega_kernel_wrapper.get_module("static")
        src, lib_static = get_source(mega_static_module)
        print(src)
        print("Compilation complete")
    else:
        lib_static = {"main": None}

    if "dynamic" in testing_scheduler:
        print("\nCompiling dynamic scheduler module...")
        mega_dynamic_module = mega_kernel_wrapper.get_module("dynamic")
        from tvm.tirx.megakernel.utils.common import get_source

        src, lib_dynamic = get_source(mega_dynamic_module)
        print(src)
        print("Compilation complete")
    else:
        lib_dynamic = {"main": None}

    sess = None  # No multi-GPU support yet

    for batch_size in args.batch_size:
        for seq_len in args.seq_len:
            print(f"\n{'=' * 60}")
            print(f"Testing: batch_size={batch_size}, seq_len={seq_len}")
            print(f"{'=' * 60}")
            test(batch_size, seq_len, lib_static, lib_dynamic, mega_kernel_wrapper, sess)
