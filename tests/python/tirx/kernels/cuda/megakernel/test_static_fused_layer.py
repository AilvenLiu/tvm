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
import os
import sys

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.tirx.bench.utils import ProtonContext, bench, export_to_perfetto_trace
from tvm.tirx.megakernel.utils import static_scheduler
from tvm.tirx.megakernel.utils.config import (
    event_type_names,
)

sys.path.insert(
    0,
    os.path.join(
        os.environ.get("TIRX_KERNELS_PATH", os.path.expanduser("~/tirx-kernels/kernels")),
        "megakernel",
    ),
)
from static_fused_layer import (
    MegaKernel,
    get_packed_info,
    prepare_data,
)


@tvm.testing.requires_cuda_compute_version(10, exact=True)
@pytest.mark.skip
def test(batch_size, seq_len, vm, mega_kernel_wrapper, profile_on=False):
    arg_dict = prepare_data(batch_size, seq_len, mega_kernel_wrapper)

    def tir(vm, arg_dict, batch_size, mk: MegaKernel):
        dev = tvm.cuda()
        hidden_state = tvm.runtime.tensor(arg_dict["hidden_state"], device=dev)
        # residual = tvm.runtime.tensor(arg_dict["residual"].to(torch.float32), device=dev)

        if mk.GATE_UP_PROJ_SPLIT_K_FACTOR == 1:
            # reorder the gate_up_weight for the fusion of gate_up projection and silu
            new_order_indices = np.stack(
                (
                    np.arange(mk.INTERMEDIATE_SIZE).reshape(-1, 16),
                    np.arange(mk.INTERMEDIATE_SIZE, mk.INTERMEDIATE_SIZE * 2).reshape(-1, 16),
                ),
                axis=1,
            ).reshape(-1)
            if mk.world_size > 1:
                gate_up_weight = arg_dict["gate_up_weight"][:, new_order_indices, :]
            else:
                gate_up_weight = arg_dict["gate_up_weight"][new_order_indices, :]

        weights = [
            arg_dict["qkv_proj_weight"],
            arg_dict["q_rms_wight"],
            arg_dict["k_rms_wight"],
            arg_dict["o_proj_weight"],
            arg_dict["attn_add_rms_weight"],
            gate_up_weight,
            arg_dict["down_weight"],
            arg_dict["mlp_add_rms_weight"],
        ]
        weights = [tvm.runtime.tensor(weight, device=dev) for weight in weights]
        packed_info = get_packed_info(arg_dict, mk)
        # events = prepare_events(arg_dict, batch_size, max_batch_size=128, mk=mk)
        workspace = tvm.runtime.tensor(
            np.zeros((mk.EVT_WORKSPACE_SIZE,), dtype=np.int32), device=dev
        )
        iter = 0

        def func():
            nonlocal iter
            iter += 1
            residual = tvm.runtime.tensor(arg_dict["residual"].to(torch.float32), device=dev)
            res = vm["megakernel"](hidden_state, residual, packed_info, weights, workspace)
            return res

        res = bench(func, warmup=3, repeat=10, proton_name="tir")
        res = func()
        if profile_on:
            export_to_perfetto_trace(res[2].numpy(), "layer.perfetto-trace", event_type_names)
        return res[0].numpy(), res[1].numpy().astype(np.float16)

    def std(arg_dict, batch_size, use_prefill, mk: MegaKernel):
        import flashinfer
        import torch

        mk.INTERMEDIATE_SIZE * mk.world_size
        FULL_NUM_ATTENTION_HEADS = mk.NUM_ATTENTION_HEADS * mk.world_size
        FULL_NUM_KEY_VALUE_HEADS = mk.NUM_KEY_VALUE_HEADS * mk.world_size

        torch_dev = torch.device("cuda")
        std_arg_dict = {}

        def func():
            for key, value in arg_dict.items():
                if mk.world_size > 1:
                    if key == "qkv_proj_weight":
                        split_sizes = [
                            mk.NUM_ATTENTION_HEADS,
                            mk.NUM_KEY_VALUE_HEADS,
                            mk.NUM_KEY_VALUE_HEADS,
                        ]
                        value = value.reshape(mk.world_size, -1, mk.HEAD_DIM, mk.HIDDEN_SIZE)
                        q_weight, k_weight, v_weight = torch.split(value, split_sizes, dim=1)
                        q_weight = q_weight.reshape(-1, mk.HIDDEN_SIZE)
                        k_weight = k_weight.reshape(-1, mk.HIDDEN_SIZE)
                        v_weight = v_weight.reshape(-1, mk.HIDDEN_SIZE)
                        value = torch.cat([q_weight, k_weight, v_weight], dim=0)
                    elif key == "gate_up_weight":
                        value = value.reshape(-1, *value.shape[2:])
                    elif key == "o_proj_weight" or key == "down_weight":
                        value = value.transpose(0, 1)
                        value = value.reshape(value.shape[0], -1)
                    elif key == "kv_cache":
                        value = value.movedim(0, 2)
                        value = value.reshape(value.shape[0], value.shape[1], -1, *value.shape[4:])
                std_arg_dict[key] = value.clone().to(torch_dev)
            out_f = torch.zeros(
                batch_size,
                FULL_NUM_ATTENTION_HEADS,
                mk.HEAD_DIM,
                dtype=torch.float16,
                device="cuda",
            )
            lse_f = torch.zeros(
                batch_size, FULL_NUM_ATTENTION_HEADS, dtype=torch.float32, device="cuda"
            )
            if use_prefill:
                workspace_buffer = torch.empty(1024 * 1024 * 1024, dtype=torch.int8).to(0)
                wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace_buffer, "HND")
                wrapper.plan(
                    torch.arange(0, batch_size + 1, dtype=torch.int32).to(0),
                    std_arg_dict["page_kv_indptr"],
                    std_arg_dict["page_kv_indices"],
                    std_arg_dict["page_kv_last_page_len"],
                    FULL_NUM_ATTENTION_HEADS,
                    FULL_NUM_KEY_VALUE_HEADS,
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
                    FULL_NUM_ATTENTION_HEADS,
                    FULL_NUM_KEY_VALUE_HEADS,
                    mk.HEAD_DIM,
                    mk.HEAD_DIM,
                    mk.PAGE_SIZE,
                    kv_data_type=torch.float16,
                    q_data_type=torch.float16,
                )
            qkv = torch.matmul(
                std_arg_dict["hidden_state"], std_arg_dict["qkv_proj_weight"].T
            ).reshape(batch_size, -1, mk.HEAD_DIM)
            q, k, v = torch.split(
                qkv,
                [FULL_NUM_ATTENTION_HEADS, FULL_NUM_KEY_VALUE_HEADS, FULL_NUM_KEY_VALUE_HEADS],
                dim=1,
            )
            q = flashinfer.norm.rmsnorm(
                input=q.reshape(-1, mk.HEAD_DIM),
                weight=std_arg_dict["q_rms_wight"],
                eps=mk.RMS_NORM_EPS,
                enable_pdl=False,
            ).reshape(batch_size, FULL_NUM_ATTENTION_HEADS, mk.HEAD_DIM)
            k = flashinfer.norm.rmsnorm(
                input=k.reshape(-1, mk.HEAD_DIM),
                weight=std_arg_dict["k_rms_wight"],
                eps=mk.RMS_NORM_EPS,
                enable_pdl=False,
            ).reshape(batch_size, FULL_NUM_KEY_VALUE_HEADS, mk.HEAD_DIM)
            q, k = flashinfer.rope.apply_rope_with_cos_sin_cache(
                positions=std_arg_dict["rope_pos"],
                query=q.reshape(batch_size, -1),
                key=k.reshape(batch_size, -1),
                head_size=mk.HEAD_DIM,
                cos_sin_cache=std_arg_dict["cos_sin_cache"],
                is_neox=True,
            )
            flashinfer.page.append_paged_kv_cache(
                append_key=k.reshape(batch_size, FULL_NUM_KEY_VALUE_HEADS, mk.HEAD_DIM),
                append_value=v,
                batch_indices=torch.arange(batch_size, dtype=torch.int32, device=torch_dev),
                positions=std_arg_dict["append_pos"],
                paged_kv_cache=std_arg_dict["kv_cache"],
                kv_indices=std_arg_dict["page_kv_indices"],
                kv_indptr=std_arg_dict["page_kv_indptr"],
                kv_last_page_len=std_arg_dict["page_kv_last_page_len"],
                kv_layout="HND",
            )
            if use_prefill:
                out_f = wrapper.run(
                    q.reshape(batch_size, FULL_NUM_ATTENTION_HEADS, mk.HEAD_DIM),
                    std_arg_dict["kv_cache"],
                )
            else:
                wrapper.run(
                    q.reshape(batch_size, FULL_NUM_ATTENTION_HEADS, mk.HEAD_DIM),
                    std_arg_dict["kv_cache"],
                    out_f,
                    lse_f,
                )
            hidden_state_attn_mlp = torch.matmul(
                out_f.reshape(batch_size, FULL_NUM_ATTENTION_HEADS * mk.HEAD_DIM),
                std_arg_dict["o_proj_weight"].T,
            )
            flashinfer.norm.fused_add_rmsnorm(
                input=hidden_state_attn_mlp,
                residual=std_arg_dict["residual"],
                weight=std_arg_dict["attn_add_rms_weight"],
                eps=mk.RMS_NORM_EPS,
                enable_pdl=False,
            )
            out_gate_up_proj = torch.matmul(hidden_state_attn_mlp, std_arg_dict["gate_up_weight"].T)
            out_silu_multiply = flashinfer.activation.silu_and_mul(
                input=out_gate_up_proj,
            )
            output = torch.matmul(out_silu_multiply, std_arg_dict["down_weight"].T)
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
        print(f"std time: {ms:.3f} ms")
        return output

    with ProtonContext("blackwell_attn"):
        fused_res, fused_residual = tir(vm, arg_dict, batch_size, mega_kernel_wrapper)
        std_res, std_residual = std(arg_dict, batch_size, True, mega_kernel_wrapper)

    np.testing.assert_allclose(fused_res, std_res, atol=1e-2, rtol=1e-3)
    np.testing.assert_allclose(fused_residual, std_residual, atol=1e-2, rtol=1e-3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MegaKernel testing script.")
    parser.add_argument(
        "--scheduler",
        type=str,
        nargs="+",
        default=["static"],
        choices=["static", "dynamic"],
        help="A list of test methods to run.",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        choices=[1],
        help="The number of devices for the world size, now only support 1.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        nargs="+",
        default=[1, 3, 7, 15, 31, 63, 127, 128],
        help="A list of batch sizes to test.",
    )
    parser.add_argument(
        "--seq-len", type=int, nargs="+", default=[512], help="A list of sequence lengths to test."
    )
    parser.add_argument("--profiler-on", action="store_true", help="Enable the profiler.")
    args = parser.parse_args()

    tile_scheduler_class_map = {
        "static": static_scheduler.StaticTileScheduler,
        "dynamic": dynamic_scheduler.DynamicTileScheduler,
    }
    semaphore_class_map = {
        "static": static_scheduler.Semaphore,
        "dynamic": dynamic_scheduler.Semaphore,
    }
    for scheduler in args.scheduler:
        print(f"Testing with {scheduler} tile scheduler...", flush=True)
        megakernel_wrapper = MegaKernel()
        mod = megakernel_wrapper.get_mod(max_batch_size=128, profile_on=args.profiler_on)
        mod.show()
        mod = rx.transform.StaticHorizontalFusion(
            ["megakernel_blkm32", "megakernel_blkm64", "megakernel_blkm128"],
            strategy=scheduler,
            tile_scheduler_class=tile_scheduler_class_map[scheduler],
            semaphore_class=semaphore_class_map[scheduler],
            profiler_on=args.profiler_on,
        )(mod)
        mod.show()
        ex = rx.build(mod, target="cuda", tir_pipeline="tirx")
        src = ex.mod.imports[0].imports[0].inspect_source()
        print(src)
        vm = rx.VirtualMachine(ex, tvm.cuda())
        for batch_size in args.batch_size:
            print(f"batch_size: {batch_size}", flush=True)
            for seq_len in args.seq_len:
                print(f"seq_len: {seq_len}", flush=True)
                test(batch_size, seq_len, vm, megakernel_wrapper, profile_on=args.profiler_on)
