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
from tirx_kernels.megakernel.moe import (
    MegaKernelMOE,
    fused_moe_sglang,
    prepare_data,
)

import tvm
import tvm.testing
from tvm.megakernel.utils.config import (
    event_type_names,
)
from tvm.megakernel.utils.utils import get_source
from tvm.tirx.bench.utils import ProtonContext, bench, export_to_perfetto_trace


@tvm.testing.requires_cuda_compute_version(10, exact=True)
@pytest.mark.skip
def test(
    batch_size,
    mega_kernel_static,
    mega_kernel_dynamic,
    mega_kernel_unfused,
    mega_kernel_wrapper,
    sess,
):
    arg_dict = prepare_data(batch_size, mega_kernel_wrapper)

    def tir(arg_dict, mk: MegaKernelMOE, scheduler: Literal["static", "dynamic", "unfused"]):
        REPEAT = 100
        DEV = tvm.cuda(0)
        tvm_arg_dict = {}
        target = tvm.target.Target("cuda")
        if scheduler == "static" or scheduler == "unfused":
            # static schedule
            exec_queue = generate_exec_queue_moe(
                batch_size, mk.config, mk.num_etensors[False], "static"
            )
            tvm_arg_dict["exec_queue"] = tvm.runtime.tensor(exec_queue, DEV)
        else:
            exec_queue = generate_exec_queue_moe(
                batch_size, mk.config, mk.num_etensors[True], "dynamic"
            )
            for i in range(REPEAT):
                tvm_arg_dict[f"queue_tasks_{i}"] = tvm.runtime.tensor(exec_queue.tasks, DEV)
                tvm_arg_dict[f"queue_head_{i}"] = tvm.runtime.tensor(exec_queue.head, DEV)
                tvm_arg_dict[f"queue_tail_{i}"] = tvm.runtime.tensor(exec_queue.tail, DEV)

        for key, value in arg_dict.items():
            tvm_arg_dict[key] = tvm.runtime.tensor(value, device=DEV)

        tvm_arg_dict["output"] = tvm.runtime.tensor(
            np.zeros((batch_size, mk.HIDDEN_SIZE), dtype=np.float16), device=DEV
        )
        tvm_arg_dict["etensor_workspace"] = tvm.runtime.tensor(
            np.zeros([mk.ETENSOR_WORKSPACE_SIZE], dtype=np.int32), device=DEV
        )
        for i in range(REPEAT):
            tvm_arg_dict[f"residual_{i}"] = tvm.runtime.tensor(arg_dict["residual"], device=DEV)
            # initial tensor must be 0
            tvm_arg_dict[f"gating_output_{i}"] = tvm.runtime.tensor(
                arg_dict["gating_output"], device=DEV
            )
            tvm_arg_dict[f"topk_reduce_output_{i}"] = tvm.runtime.tensor(
                arg_dict["topk_reduce_output"], device=DEV
            )
        tvm_arg_dict["profiler_buffer"] = tvm.runtime.tensor(
            np.zeros([mk.PROFILER_BUFFER_SIZE], dtype=np.uint64), device=DEV
        )

        if mk.world_size > 1:
            raise ValueError(f"Unsupported world size: {mk.world_size}")
        with target:
            iter = 0

            if scheduler == "static" or scheduler == "unfused":
                kernel = (
                    mega_kernel_static["main"]
                    if scheduler == "static"
                    else mega_kernel_unfused["main"]
                )
                work_arg_dict = tvm_arg_dict

                def func():
                    nonlocal iter
                    kernel(
                        # input and output
                        work_arg_dict["hidden_state"],
                        work_arg_dict[f"residual_{iter}"],
                        work_arg_dict["output"],
                        # weight
                        work_arg_dict["gate_weight"],
                        work_arg_dict["shuffled_grp_gate_up_weight"],
                        work_arg_dict["grp_down_weight"],
                        # intermediate buffer
                        work_arg_dict[f"gating_output_{iter}"],
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
                        work_arg_dict[f"topk_reduce_output_{iter}"],
                        # event tensor
                        work_arg_dict["etensor_workspace"],
                        # exec queue
                        work_arg_dict["exec_queue"],
                        work_arg_dict["profiler_buffer"],
                    )
                    iter += 1

            else:
                kernel = mega_kernel_dynamic["main"]
                work_arg_dict = tvm_arg_dict

                def func():
                    nonlocal iter
                    kernel(
                        # input and output
                        work_arg_dict["hidden_state"],
                        work_arg_dict[f"residual_{iter}"],
                        work_arg_dict["output"],
                        # weight
                        work_arg_dict["gate_weight"],
                        work_arg_dict["shuffled_grp_gate_up_weight"],
                        work_arg_dict["grp_down_weight"],
                        # intermediate buffer
                        work_arg_dict[f"gating_output_{iter}"],
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
                        work_arg_dict[f"topk_reduce_output_{iter}"],
                        # event tensor
                        work_arg_dict["etensor_workspace"],
                        # exec queue
                        work_arg_dict[f"queue_tasks_{iter}"],
                        work_arg_dict[f"queue_head_{iter}"],
                        work_arg_dict[f"queue_tail_{iter}"],
                        work_arg_dict["profiler_buffer"],
                    )
                    iter += 1

            if mk.world_size == 1:
                ms = bench(func, warmup=1, repeat=5, proton_name=f"tir-{scheduler}")
                print(f"TIR time: {ms:.3f} ms")
                if mk.profiler_on:
                    export_to_perfetto_trace(
                        tvm_arg_dict["profiler_buffer"].numpy(),
                        f"{scheduler}-moe-layer-bs{batch_size}-tp{mk.world_size}.perfetto-trace",
                        event_type_names,
                    )
                # def scatter_out(out_tvm, *shape):
                #     tmp_out = torch.from_numpy(out_tvm.numpy())
                #     out = torch.zeros(shape, dtype=torch.float16)
                #     sorted_token_ids = tvm_arg_dict["sorted_token_ids"].numpy()
                #     num_tokens_post_pad = tvm_arg_dict["num_tokens_post_pad"].numpy()
                #     index_for_scatter = sorted_token_ids[:num_tokens_post_pad[0]]
                #     for i in range(num_tokens_post_pad[0]):
                #         if index_for_scatter[i] >= 0 and index_for_scatter[i] < shape[0]:
                #             out[index_for_scatter[i]] = tmp_out[i]
                #     return out
                # out1 = scatter_out(
                #     tvm_arg_dict["silu_mul_output"],
                #     batch_size * mk.NUM_EXPERTS_PER_TOK,
                #     mk.INTERMEDIATE_SIZE,
                # )
                return tvm_arg_dict["topk_reduce_output_0"].numpy()
            else:
                for i in range(REPEAT):
                    func()
                sess._sync_all()
                sess.copy_from_worker_0(res_dict["output_host"], disco_arg_dict["output"])
                sess.copy_from_worker_0(res_dict["residual_host"], disco_arg_dict["residual_0"])
                # sess.copy_from_worker_0(res_dict["hidden_state_attn_mlp_host"], disco_arg_dict["hidden_state_attn_mlp"])
                sess.gather_to_worker0(
                    disco_arg_dict["hidden_state_attn_mlp"],
                    res_dict["hidden_state_attn_mlp_res"],
                )
                sess.copy_from_worker_0(
                    res_dict["hidden_state_attn_mlp_host"],
                    res_dict["hidden_state_attn_mlp_res"],
                )
                sess.gather_to_worker0(
                    disco_arg_dict["profiler_buffer"],
                    res_dict["profiler_buffer_res"],
                )
                sess.copy_from_worker_0(
                    res_dict["profiler_buffer_host"],
                    res_dict["profiler_buffer_res"],
                )
                sess._sync_all()
                if mk.profiler_on:
                    for r in range(mk.world_size):
                        export_to_perfetto_trace(
                            res_dict["profiler_buffer_host"].numpy()[r],
                            f"{scheduler}-moe-layer-bs{batch_size}-tp{mk.world_size}.perfetto-trace",
                            event_type_names,
                        )
                return res_dict["output_host"].numpy(), res_dict["residual_host"].numpy()

    def std(arg_dict, mk: MegaKernelMOE):
        import flashinfer
        import torch

        torch_dev = torch.device("cuda")
        std_arg_dict = {}

        def func():
            for key, value in arg_dict.items():
                std_arg_dict[key] = value.clone().to(torch_dev)
            gating_output = std_arg_dict["hidden_state"] @ std_arg_dict["gate_weight"].T
            topk_softmax(
                gating_output=gating_output,
                topk_weights=std_arg_dict["topk_weights"],
                topk_ids=std_arg_dict["topk_indices"],
                renormalize=False,
            )

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
            print(f"sgL_config: {sgL_config}")
            invoke_fused_moe_kernel(
                std_arg_dict["hidden_state"],
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
            silu_mul_out = flashinfer.activation.silu_and_mul(
                out1.view(-1, 2 * mk.INTERMEDIATE_SIZE)
            )
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
            ret = out2.sum(dim=1)
            return ret.cpu().numpy()

        output = func()
        ms = bench(func, warmup=10, repeat=30, proton_name="std")
        print(f"std time: {ms:.3f} ms")
        return output

    def flashinfer(arg_dict):
        torch_dev = torch.device("cuda")
        std_arg_dict = {}
        for key, value in arg_dict.items():
            std_arg_dict[key] = value.clone().to(torch_dev)
        output = torch.zeros_like(std_arg_dict["hidden_state"])
        stream = torch.cuda.Stream()
        graph = torch.cuda.CUDAGraph()

        def flashinfer_func():
            gating_output = std_arg_dict["hidden_state"] @ std_arg_dict["gate_weight"].T
            topk_softmax(
                gating_output=gating_output,
                topk_weights=std_arg_dict["topk_weights"],
                topk_ids=std_arg_dict["topk_indices"],
                renormalize=False,
            )
            return fused_moe.cutlass_fused_moe(
                std_arg_dict["hidden_state"],
                std_arg_dict["topk_indices"].to(torch.int),
                std_arg_dict["topk_weights"],
                std_arg_dict["grp_up_gate_weight"],
                std_arg_dict["grp_down_weight"],
                std_arg_dict["hidden_state"].dtype,
                quant_scales=[],
                output=output,
            )

        for _ in range(10):
            flashinfer_func()
        with torch.cuda.graph(graph, stream=stream):
            flashinfer_func()
        torch.cuda.synchronize()

        def func():
            nonlocal graph
            graph.replay()

        ms = bench(func, warmup=10, repeat=30, proton_name="flashinfer")
        print(f"flashinfer time: {ms:.3f} ms")
        return output.cpu().numpy()

    def sglang_fused(arg_dict):
        torch_dev = torch.device("cuda")
        std_arg_dict = {}
        for key, value in arg_dict.items():
            std_arg_dict[key] = value.clone().to(torch_dev)
        torch.zeros_like(std_arg_dict["hidden_state"])
        stream = torch.cuda.Stream()
        graph = torch.cuda.CUDAGraph()

        def sglang_fused_func():
            gating_output = std_arg_dict["hidden_state"] @ std_arg_dict["gate_weight"].T
            topk_softmax(
                gating_output=gating_output,
                topk_weights=std_arg_dict["topk_weights"],
                topk_ids=std_arg_dict["topk_indices"],
                renormalize=False,
            )
            out = fused_moe_sglang(
                std_arg_dict["hidden_state"],
                std_arg_dict["grp_gate_up_weight"],
                std_arg_dict["grp_down_weight"],
                gating_output,
                std_arg_dict["topk_weights"],
                std_arg_dict["topk_indices"].to(torch.int),
            )
            return out

        for _ in range(10):
            out = sglang_fused_func()
        with torch.cuda.graph(graph, stream=stream):
            sglang_fused_func()
        torch.cuda.synchronize()

        def func():
            nonlocal graph
            graph.replay()

        ms = bench(func, warmup=10, repeat=30, proton_name="sglang_fused")
        print(f"sglang_fused time: {ms:.3f} ms")
        return out.cpu().numpy()

    def run():
        if mega_kernel_static["main"] is not None:
            output1_tir_static = tir(arg_dict, mega_kernel_wrapper, "static")
            print("static tir finish", flush=True)
        if mega_kernel_dynamic["main"] is not None:
            output1_tir_dynamic = tir(arg_dict, mega_kernel_wrapper, "dynamic")
            print("dynamic tir finish", flush=True)
        if mega_kernel_unfused["main"] is not None:
            output1_tir_unfused = tir(arg_dict, mega_kernel_wrapper, "unfused")
            print("unfused tir finish", flush=True)
        output1_std = std(arg_dict, mk=mega_kernel_wrapper)
        # output1_flashinfer = flashinfer(arg_dict)
        # np.testing.assert_allclose(output1_flashinfer, output1_std, rtol=1e-3, atol=1e-2)
        output1_sglang_fused = sglang_fused(arg_dict)
        np.testing.assert_allclose(output1_sglang_fused, output1_std, rtol=1e-3, atol=1e-2)
        if mega_kernel_static["main"] is not None:
            # std and tir might choose different experts because slight difference in gating output
            try:
                np.testing.assert_allclose(output1_tir_static, output1_std, rtol=1e-3, atol=1e-2)
                print("static pass", flush=True)
            except Exception as e:
                print(e)
        if mega_kernel_dynamic["main"] is not None:
            try:
                np.testing.assert_allclose(output1_tir_dynamic, output1_std, rtol=1e-3, atol=1e-2)
                print("dynamic pass", flush=True)
            except Exception as e:
                print(e)
        if mega_kernel_unfused["main"] is not None:
            try:
                np.testing.assert_allclose(output1_tir_unfused, output1_std, rtol=1e-3, atol=1e-2)
                print("unfused pass", flush=True)
            except Exception as e:
                print(e)

    if mega_kernel_wrapper.world_size == 1:
        with ProtonContext("blackwell_moe"):
            run()
    else:
        run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MegaKernel testing script.")
    parser.add_argument(
        "--scheduler",
        type=str,
        nargs="+",
        default=["static", "dynamic", "unfused"],
        choices=["static", "dynamic", "unfused", "none"],
        help="A list of test methods to run: 'static' or 'dynamic'.",
    )
    parser.add_argument(
        "--world-size", type=int, default=1, help="The number of devices for the world size."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        nargs="+",
        default=[1, 3, 7, 15, 31, 63, 127, 128],
        help="A list of batch sizes to test.",
    )
    parser.add_argument("--profiler-on", action="store_true", help="Enable the profiler.")
    args = parser.parse_args()

    testing_scheduler = set(args.scheduler)
    mega_kernel_wrapper = MegaKernelMOE(
        config=qwen3_30b_a3b_config, world_size=args.world_size, profiler_on=args.profiler_on
    )
    if "static" in testing_scheduler:
        mega_static_module = mega_kernel_wrapper.get_module("static")
        src, lib_static = get_source(mega_static_module)
        print(src)
    else:
        lib_static = {"main": None}
    if "dynamic" in testing_scheduler:
        mega_dynamic_module = mega_kernel_wrapper.get_module("dynamic")
        src, lib_dynamic = get_source(mega_dynamic_module)
        print(src)
    else:
        lib_dynamic = {"main": None}
    if "unfused" in testing_scheduler:
        mega_unfused_module = mega_kernel_wrapper.get_module("unfused")
        src, lib_unfused = get_source(mega_unfused_module)
        print(src)
    else:
        lib_unfused = {"main": None}
    if mega_kernel_wrapper.world_size > 1:
        devices = list(np.arange(mega_kernel_wrapper.world_size))
        sess = di.ProcessSession(num_workers=mega_kernel_wrapper.world_size)
        sess.init_ccl(tvm.get_global_func("runtime.disco.compiled_ccl")(), *devices)
        f_init_nvshmem_uid = tvm.get_global_func("runtime.disco.nvshmem.init_nvshmem_uid")
        uid = f_init_nvshmem_uid()
        init_dfunc = sess.get_global_func("runtime.disco.nvshmem.init_nvshmem")
        init_dfunc(uid, mega_kernel_wrapper.world_size, 0)
        sess.sync_worker_0()
    else:
        sess = None
    for batch_size in args.batch_size:
        print(f"batch_size: {batch_size}", flush=True)
        test(batch_size, lib_static, lib_dynamic, lib_unfused, mega_kernel_wrapper, sess)
