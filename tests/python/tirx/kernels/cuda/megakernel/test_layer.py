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
import tempfile

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.tirx.bench.utils import ProtonContext, bench, export_to_perfetto_trace
from tvm.tirx.megakernel.utils.config import (
    event_type_names,
)
from tvm.tirx.megakernel.utils.utils import get_source

sys.path.insert(
    0,
    os.path.join(
        os.environ.get("TIRX_KERNELS_PATH", os.path.expanduser("~/tirx-kernels/kernels")),
        "megakernel",
    ),
)
from layer import (
    MegaKernelDenseLayer,
    prepare_data,
)


@tvm.testing.requires_cuda_compute_version(10, exact=True)
@pytest.mark.skip
def test(batch_size, seq_len, mega_kernel_static, mega_kernel_dynamic, mega_kernel_wrapper, sess):
    arg_dict = prepare_data(batch_size, seq_len, mega_kernel_wrapper)

    def tir(arg_dict, mk: MegaKernelDenseLayer, scheduler: Literal["static", "dynamic"]):
        import torch

        REPEAT = 100
        DEV = tvm.cuda(0)
        tvm_arg_dict = {}
        target = tvm.target.Target("cuda")

        # preprocess the buffer used in the tir kernel
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
        tvm_arg_dict["partial_out_gate_up_proj"] = tvm.runtime.tensor(
            np.zeros(
                [mk.GATE_UP_PROJ_SPLIT_K_FACTOR, batch_size, mk.INTERMEDIATE_SIZE * 2],
                dtype=np.float32,
            ),
            DEV,
        )
        tvm_arg_dict["before_down_proj_allreduce"] = tvm.runtime.tensor(
            np.zeros([batch_size, mk.HIDDEN_SIZE], dtype=np.float16), DEV
        )
        res = get_inverse_plan_info(
            batch_size,
            mk.NUM_KEY_VALUE_HEADS,
            arg_dict["q_indptr"],
            arg_dict["kv_head_idx"],
            arg_dict["attn_task_num"].item(),
        )
        tvm_arg_dict["inverse_indptr"], tvm_arg_dict["inverse_indices"] = res

        if scheduler == "static":
            exec_queue = generate_exec_queue(
                batch_size, arg_dict["attn_task_num"].item(), mk.config, mk.world_size, 20, "static"
            )
            tvm_arg_dict["exec_queue"] = tvm.runtime.tensor(exec_queue, DEV)
        else:
            exec_queue = generate_exec_queue(None, None, mk.config, mk.world_size, 20, "dynamic")
            for i in range(REPEAT):
                tvm_arg_dict[f"queue_tasks_{i}"] = tvm.runtime.tensor(exec_queue.tasks, DEV)
                tvm_arg_dict[f"queue_head_{i}"] = tvm.runtime.tensor(exec_queue.head, DEV)
                tvm_arg_dict[f"queue_tail_{i}"] = tvm.runtime.tensor(exec_queue.tail, DEV)

        # append_pos here is different from flashinfer
        append_pos = arg_dict["append_pos"].clone()
        for b in range(batch_size):
            append_pos[b] = (
                arg_dict["page_kv_indices"][
                    (arg_dict["page_kv_indptr"][b] * mk.PAGE_SIZE + append_pos[b]) // mk.PAGE_SIZE
                ]
                * mk.PAGE_SIZE
                + append_pos[b] % mk.PAGE_SIZE
            )
        for key, value in arg_dict.items():
            tvm_arg_dict[key] = tvm.runtime.tensor(value, device=DEV)
        tvm_arg_dict["append_pos"] = tvm.runtime.tensor(append_pos, device=DEV)
        if mk.GATE_UP_PROJ_SPLIT_K_FACTOR == 1:
            tvm_arg_dict["gate_up_weight"] = tvm.runtime.tensor(gate_up_weight, device=DEV)
        tvm_arg_dict["etensor_workspace"] = tvm.runtime.tensor(
            np.zeros([mk.ETENSOR_WORKSPACE_SIZE], dtype=np.int32), device=DEV
        )
        for i in range(REPEAT):
            if mk.world_size == 1:
                tvm_arg_dict[f"residual_{i}"] = tvm.runtime.tensor(
                    arg_dict["residual"].to(torch.float32), device=DEV
                )
            else:
                tvm_arg_dict[f"residual_{i}"] = tvm.runtime.tensor(arg_dict["residual"], device=DEV)
        tvm_arg_dict["profiler_buffer"] = tvm.runtime.tensor(
            np.zeros([mk.PROFILER_BUFFER_SIZE], dtype=np.uint64), device=DEV
        )

        if mk.world_size > 1:
            nvshmem_malloc_hook = sess.get_global_func("runtime.disco.nvshmem.empty")
            tensor_to_gather = [
                "qkv_proj_weight",
                "o_proj_weight",
                "gate_up_weight",
                "down_weight",
                "kv_cache",
            ]
            disco_arg_dict = {}
            for key, value in tvm_arg_dict.items():
                if not isinstance(value, tvm.runtime.Tensor):
                    continue
                if key in tensor_to_gather:
                    disco_arg_dict[key] = sess.empty(value.shape[1:], value.dtype)
                elif "etensor" in key or "queue" in key:
                    disco_arg_dict[key] = nvshmem_malloc_hook(
                        ShapeTuple([*value.shape]), str(value.dtype), None
                    )
                else:
                    disco_arg_dict[key] = sess.empty(value.shape, value.dtype)
            disco_arg_dict["before_o_allreduce"] = nvshmem_malloc_hook(
                ShapeTuple((batch_size, mk.HIDDEN_SIZE)), "float16", None
            )
            disco_arg_dict["hidden_state_attn_mlp"] = nvshmem_malloc_hook(
                ShapeTuple((batch_size, mk.HIDDEN_SIZE)), "float16", None
            )
            disco_arg_dict["before_down_proj_allreduce"] = nvshmem_malloc_hook(
                ShapeTuple((batch_size, mk.HIDDEN_SIZE)), "float16", None
            )
            disco_arg_dict["output"] = nvshmem_malloc_hook(
                ShapeTuple((batch_size, mk.HIDDEN_SIZE)), "float16", None
            )

            res_dict = {
                "output_host": tvm.runtime.empty(
                    (batch_size, mk.HIDDEN_SIZE), "float16", device=DEV
                ),
                "residual_host": tvm.runtime.empty(
                    (batch_size, mk.HIDDEN_SIZE), "float16", device=DEV
                ),
                "hidden_state_attn_mlp_host": tvm.runtime.empty(
                    (mk.world_size, batch_size, mk.HIDDEN_SIZE), "float16", device=DEV
                ),
                "hidden_state_attn_mlp_res": sess.empty(
                    (mk.world_size, batch_size, mk.HIDDEN_SIZE), "float16", worker0_only=True
                ),
                "profiler_buffer_host": tvm.runtime.empty(
                    (mk.world_size, mk.PROFILER_BUFFER_SIZE), "uint64", device=DEV
                ),
                "profiler_buffer_res": sess.empty(
                    (mk.world_size, mk.PROFILER_BUFFER_SIZE), "uint64", worker0_only=True
                ),
            }
            # init disco weight/input args
            gathered_arg_dict = {}
            for key, value in tvm_arg_dict.items():
                if key in tensor_to_gather:
                    gathered_arg_dict[key] = sess.empty(value.shape, value.dtype, worker0_only=True)
                    sess.copy_to_worker_0(value, gathered_arg_dict[key])
                    sess.scatter_from_worker0(gathered_arg_dict[key], disco_arg_dict[key])
                elif key in disco_arg_dict:
                    sess.broadcast(value, disco_arg_dict[key])
            with tempfile.TemporaryDirectory() as tmpdir:
                path = tmpdir + "/test.so"
                (
                    mega_kernel_static if scheduler == "static" else mega_kernel_dynamic
                ).export_library(path)
                rt_mod = sess.load_vm_module(path)
                sess._sync_all()

        # run
        with target:
            iter = 0

            if scheduler == "static":
                kernel = mega_kernel_static["main"] if mk.world_size == 1 else rt_mod["main"]
                work_arg_dict = tvm_arg_dict if mk.world_size == 1 else disco_arg_dict

                def func():
                    nonlocal iter
                    kernel(
                        # input and output
                        work_arg_dict["hidden_state"],
                        work_arg_dict[f"residual_{iter}"],
                        work_arg_dict["output"],
                        # weight
                        work_arg_dict["qkv_proj_weight"],
                        work_arg_dict["o_proj_weight"],
                        work_arg_dict["q_rms_wight"],
                        work_arg_dict["k_rms_wight"],
                        work_arg_dict["gate_up_weight"],
                        work_arg_dict["down_weight"],
                        work_arg_dict["attn_add_rms_weight"],
                        work_arg_dict["mlp_add_rms_weight"],
                        # page cache, cos_sin cache and plan info
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
                        # intermediate buffer
                        work_arg_dict["partial_qkv"],
                        work_arg_dict["qkv"],
                        work_arg_dict["o"],
                        work_arg_dict["o_partial_attn"],
                        work_arg_dict["lse_partial"],
                        work_arg_dict["partial_o"],
                        work_arg_dict["before_o_allreduce"],
                        work_arg_dict["hidden_state_attn_mlp"],
                        work_arg_dict["partial_out_gate_up_proj"],
                        work_arg_dict["out_gate_up_proj"],
                        work_arg_dict["out_silu_multiply"],
                        work_arg_dict["partial_sum_down_proj"],
                        work_arg_dict["before_down_proj_allreduce"],
                        # event tensor
                        work_arg_dict["etensor_workspace"],
                        # exec queue
                        work_arg_dict["exec_queue"],
                        work_arg_dict["profiler_buffer"],
                    )
                    iter += 1

            else:
                kernel = mega_kernel_dynamic["main"] if mk.world_size == 1 else rt_mod["main"]
                work_arg_dict = tvm_arg_dict if mk.world_size == 1 else disco_arg_dict

                def func():
                    nonlocal iter
                    kernel(
                        # input and output
                        work_arg_dict["hidden_state"],
                        work_arg_dict[f"residual_{iter}"],
                        work_arg_dict["output"],
                        # weight
                        work_arg_dict["qkv_proj_weight"],
                        work_arg_dict["o_proj_weight"],
                        work_arg_dict["q_rms_wight"],
                        work_arg_dict["k_rms_wight"],
                        work_arg_dict["gate_up_weight"],
                        work_arg_dict["down_weight"],
                        work_arg_dict["attn_add_rms_weight"],
                        work_arg_dict["mlp_add_rms_weight"],
                        # page cache, cos_sin cache and plan info
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
                        # intermediate buffer
                        work_arg_dict["partial_qkv"],
                        work_arg_dict["qkv"],
                        work_arg_dict["o"],
                        work_arg_dict["o_partial_attn"],
                        work_arg_dict["lse_partial"],
                        work_arg_dict["partial_o"],
                        work_arg_dict["before_o_allreduce"],
                        work_arg_dict["hidden_state_attn_mlp"],
                        work_arg_dict["partial_out_gate_up_proj"],
                        work_arg_dict["out_gate_up_proj"],
                        work_arg_dict["out_silu_multiply"],
                        work_arg_dict["partial_sum_down_proj"],
                        work_arg_dict["before_down_proj_allreduce"],
                        # event tensor
                        work_arg_dict["etensor_workspace"],
                        # exec queue
                        work_arg_dict[f"queue_tasks_{iter}"],
                        work_arg_dict[f"queue_head_{iter}"],
                        work_arg_dict[f"queue_tail_{iter}"],
                        work_arg_dict["profiler_buffer"],
                    )
                    iter += 1

            # post process
            if mk.world_size == 1:
                ms = bench(func, warmup=1, repeat=3, proton_name=f"tir-{scheduler}")
                print(f"TIR time: {ms:.3f} ms")
                if mk.profiler_on:
                    export_to_perfetto_trace(
                        tvm_arg_dict["profiler_buffer"].numpy(),
                        f"{scheduler}-layer-bs{batch_size}-tp{mk.world_size}.perfetto-trace",
                        event_type_names,
                    )
                return tvm_arg_dict["output"].numpy(), tvm_arg_dict["residual_0"].numpy().astype(
                    np.float16
                )
            else:
                for i in range(REPEAT):
                    func()
                sess._sync_all()
                sess.copy_from_worker_0(res_dict["output_host"], disco_arg_dict["output"])
                sess.copy_from_worker_0(res_dict["residual_host"], disco_arg_dict["residual_0"])
                # sess.copy_from_worker_0(res_dict["hidden_state_attn_mlp_host"], disco_arg_dict["hidden_state_attn_mlp"])
                sess.gather_to_worker0(
                    disco_arg_dict["hidden_state_attn_mlp"], res_dict["hidden_state_attn_mlp_res"]
                )
                sess.copy_from_worker_0(
                    res_dict["hidden_state_attn_mlp_host"], res_dict["hidden_state_attn_mlp_res"]
                )
                sess.gather_to_worker0(
                    disco_arg_dict["profiler_buffer"], res_dict["profiler_buffer_res"]
                )
                sess.copy_from_worker_0(
                    res_dict["profiler_buffer_host"], res_dict["profiler_buffer_res"]
                )
                sess._sync_all()
                if mk.profiler_on:
                    for r in range(mk.world_size):
                        export_to_perfetto_trace(
                            res_dict["profiler_buffer_host"].numpy()[r],
                            f"{scheduler}-layer-bs{batch_size}-tp{mk.world_size}.perfetto-trace",
                            event_type_names,
                        )
                return res_dict["output_host"].numpy(), res_dict["residual_host"].numpy()

    def std(arg_dict, use_prefill, mk: MegaKernelDenseLayer):
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
            if mk.MODEL_NAME == "qwen3_32b":
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
            elif mk.MODEL_NAME == "llama3_1b":
                q, k = flashinfer.rope.apply_llama31_rope_pos_ids(
                    q=q.reshape(batch_size, FULL_NUM_ATTENTION_HEADS, mk.HEAD_DIM),
                    k=k.reshape(batch_size, FULL_NUM_KEY_VALUE_HEADS, mk.HEAD_DIM),
                    pos_ids=std_arg_dict["rope_pos"],
                    rope_scale=mk.ROPE_SCALING["FACTOR"],
                    rope_theta=mk.ROPE_THETA,
                    low_freq_factor=mk.ROPE_SCALING["LOW_FREQ_FACTOR"],
                    high_freq_factor=mk.ROPE_SCALING["HIGH_FREQ_FACTOR"],
                    old_context_len=mk.ROPE_SCALING["ORIGINAL_MAX_POSITION_EMBEDDINGS"],
                )
            else:
                raise ValueError(f"Unsupported model name: {mk.MODEL_NAME}")
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

    def run():
        if mega_kernel_static["main"] is not None:
            output_tir_static, residual_tir_static = tir(arg_dict, mega_kernel_wrapper, "static")
            print("static tir finish", flush=True)
        if mega_kernel_dynamic["main"] is not None:
            output_tir_dynamic, residual_tir_dynamic = tir(arg_dict, mega_kernel_wrapper, "dynamic")
            print("dynamic tir finish", flush=True)
        output_std1, residual_std1 = std(arg_dict, use_prefill=True, mk=mega_kernel_wrapper)
        output_std2, residual_std2 = std(arg_dict, use_prefill=False, mk=mega_kernel_wrapper)

        # this assert will fail on latest flashinfer version
        # np.testing.assert_allclose(output_std1, output_std2, rtol=1e-3, atol=1e-2)
        # np.testing.assert_allclose(residual_std1, residual_std2, rtol=1e-3, atol=1e-2)
        if mega_kernel_static["main"] is not None:
            np.testing.assert_allclose(output_tir_static, output_std1, rtol=1e-3, atol=1e-2)
            np.testing.assert_allclose(residual_tir_static, residual_std1, rtol=1e-3, atol=1e-2)
            print("static pass", flush=True)
        if mega_kernel_dynamic["main"] is not None:
            np.testing.assert_allclose(output_tir_dynamic, output_std1, rtol=1e-3, atol=1e-2)
            np.testing.assert_allclose(residual_tir_dynamic, residual_std1, rtol=1e-3, atol=1e-2)
            print("dynamic pass", flush=True)

    if mega_kernel_wrapper.world_size == 1:
        with ProtonContext("blackwell_layer"):
            run()
    else:
        run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MegaKernel testing script.")
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3_32b",
        choices=["qwen3_32b", "llama3_1b"],
        help="The supporting model.",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        nargs="+",
        default=["static", "dynamic"],
        choices=["static", "dynamic", "none"],
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
    parser.add_argument(
        "--seq-len", type=int, nargs="+", default=[512], help="A list of sequence lengths to test."
    )
    parser.add_argument("--profiler-on", action="store_true", help="Enable the profiler.")
    args = parser.parse_args()

    testing_scheduler = set(args.scheduler)
    mega_kernel_wrapper = MegaKernelDenseLayer(
        config=get_model_config(args.model),
        world_size=args.world_size,
        profiler_on=args.profiler_on,
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
        for seq_len in args.seq_len:
            print(f"seq_len: {seq_len}", flush=True)
            test(batch_size, seq_len, lib_static, lib_dynamic, mega_kernel_wrapper, sess)
