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
import numpy as np

import tvm
from tvm.tirx.bench.utils import ProtonContext, bench
import flashinfer
import tvm_ffi

sys.path.insert(0, os.path.join(os.environ.get("TIRX_KERNELS_PATH", os.path.expanduser("~/tirx-kernels/kernels")), "attention"))
import batch_attention  # noqa: E402
from batch_attention import *  # noqa: E402, F403


@pytest.mark.parametrize("num_heads", [(64, 8)])
@pytest.mark.parametrize("seq_len", [512, 2077, 4033])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("batch_size", [1, 11, 25, 128])
@pytest.mark.parametrize("seed", [42])
def test(num_heads, seq_len, head_dim, batch_size, seed):
    PAGE_SIZE = 16
    MAX_PAGE_NUM = 32768
    qo_heads, kv_heads = num_heads

    torch.manual_seed(seed)

    Q, KV_data, KV_indptr, KV_last_page_len, KV_indices = perpare_data(
        batch_size, qo_heads, kv_heads, seq_len, head_dim, PAGE_SIZE, MAX_PAGE_NUM
    )
    kv_indptr_f = KV_indptr.to(0)
    kv_indices_f = KV_indices.to(0)
    qo_indptr_f = torch.arange(0, batch_size + 1, dtype=torch.int32).to(0)
    wrapper = flashinfer.BatchAttention(KV_LAYOUT)
    wrapper.plan(
        qo_indptr_f,
        kv_indptr_f,
        kv_indices_f,
        torch.tensor([seq_len] * batch_size, dtype=torch.int32).to(0),
        qo_heads,
        kv_heads,
        head_dim,
        head_dim,
        PAGE_SIZE,
        kv_data_type=torch.float16,
        q_data_type=torch.float16,
    )
    plan_info = wrapper._plan_info

    def get_id(i):
        return plan_info[i].item()

    def tensor_from_bytes(
        byte_tensor: torch.Tensor, offset: int, shape, data_type: torch.dtype
    ) -> torch.Tensor:
        if byte_tensor.dtype != torch.uint8 or byte_tensor.dim() != 1:
            raise ValueError("Input must be a 1D torch.uint8 tensor.")

        num_elements = shape
        element_byte_size = torch.tensor([], dtype=data_type).element_size()
        required_bytes = num_elements * element_byte_size

        if offset + required_bytes > byte_tensor.numel():
            raise ValueError("The requested offset and shape are out of bounds.")

        byte_slice = byte_tensor[offset : offset + required_bytes]

        return byte_slice.view(data_type)

    def get_tensor(offset, shape, data_type):
        if data_type == torch.int32:
            return tensor_from_bytes(wrapper.int_workspace_buffer, offset, shape, data_type)
        elif data_type in [torch.float16, torch.float32]:
            return tensor_from_bytes(wrapper.float_workspace_buffer, offset, shape, data_type)
        else:
            raise ValueError(f"Unsupported data type: {data_type}")

    NUM_TASK_ARGS = 10
    NUM_TASKS = 2
    print(f"MAX_TOTAL_NUM_WORKERS: {MAX_TOTAL_NUM_WORKERS}")
    for i in range(0, NUM_TASKS):
        q_indptr = get_tensor(get_id(i * NUM_TASK_ARGS + 2), MAX_TOTAL_NUM_WORKERS, torch.int32)
        kv_indptr = get_tensor(get_id(i * NUM_TASK_ARGS + 3), MAX_TOTAL_NUM_WORKERS, torch.int32)
        partial_indptr = get_tensor(
            get_id(i * NUM_TASK_ARGS + 4), MAX_TOTAL_NUM_WORKERS, torch.int32
        )
        q_len = get_tensor(get_id(i * NUM_TASK_ARGS + 5), MAX_TOTAL_NUM_WORKERS, torch.int32)
        kv_len = get_tensor(get_id(i * NUM_TASK_ARGS + 6), MAX_TOTAL_NUM_WORKERS, torch.int32)
        q_start = get_tensor(get_id(i * NUM_TASK_ARGS + 7), MAX_TOTAL_NUM_WORKERS, torch.int32)
        kv_start = get_tensor(get_id(i * NUM_TASK_ARGS + 8), MAX_TOTAL_NUM_WORKERS, torch.int32)
        kv_end = get_tensor(get_id(i * NUM_TASK_ARGS + 9), MAX_TOTAL_NUM_WORKERS, torch.int32)
        kv_head_idx = get_tensor(get_id(i * NUM_TASK_ARGS + 10), MAX_TOTAL_NUM_WORKERS, torch.int32)
        work_indptr = get_tensor(get_id(i * NUM_TASK_ARGS + 11), MAX_TOTAL_NUM_WORKERS, torch.int32)

    len_kv_chunk = get_tensor(get_id(1 * NUM_TASK_ARGS + 12), NUM_TASKS, torch.int32)
    partial_o = get_tensor(
        get_id(1 * NUM_TASK_ARGS + 13), MAX_NUM_KV_SPLITS * head_dim * kv_heads, torch.float16
    )
    partial_lse = get_tensor(
        get_id(1 * NUM_TASK_ARGS + 14), MAX_NUM_KV_SPLITS * kv_heads, torch.float32
    )
    merge_indptr = get_tensor(get_id(1 * NUM_TASK_ARGS + 15), MAX_NUM_KV_SPLITS, torch.int32)
    merge_o_indices = get_tensor(get_id(1 * NUM_TASK_ARGS + 16), MAX_NUM_KV_SPLITS, torch.int32)
    num_qo_len = get_tensor(get_id(1 * NUM_TASK_ARGS + 17), 1, torch.int32)

    out_f = torch.zeros(batch_size, qo_heads, head_dim, dtype=torch.float16, device="cuda")
    lse_f = torch.zeros(batch_size, qo_heads, dtype=torch.float32, device="cuda")

    def print_work(work_idx):
        print("q_indptr", q_indptr[work_idx].cpu().numpy())
        print("kv_indptr", kv_indptr[work_idx].cpu().numpy())
        print("partial_indptr", partial_indptr[work_idx].cpu().numpy())
        print("q_len", q_len[work_idx].cpu().numpy())
        print("kv_len", kv_len[work_idx].cpu().numpy())
        print("q_start", q_start[work_idx].cpu().numpy())
        print("kv_start", kv_start[work_idx].cpu().numpy())
        print("kv_end", kv_end[work_idx].cpu().numpy())
        print("kv_head_idx", kv_head_idx[work_idx].cpu().numpy())
        print("len_kv_chunk", len_kv_chunk[work_idx].cpu().numpy())

    def flashinfer_batch_attention():
        q_f = Q.to(0).reshape(batch_size, qo_heads, head_dim)
        kv_data_f = KV_data.to(0)

        def func():
            return wrapper.run(q_f, kv_data_f, out_f, lse_f)

        ms = bench(func, warmup=10, repeat=30, proton_name="flashinfer_batch_attention")
        func()
        print(f"FlashInfer BatchAttention time: {ms:.3f} ms")

        # return partial_o.cpu().numpy(), partial_lse.cpu().numpy(), out_f.cpu().numpy()
        return out_f.cpu().numpy()

    def flashinfer_batch_prefill():
        q_f = Q.to(0).reshape(batch_size, qo_heads, head_dim)
        kv_data_f = KV_data.to(0)
        kv_indptr_f = KV_indptr.to(0)
        kv_last_page_len_f = KV_last_page_len.to(0)
        kv_indices_f = KV_indices.to(0)

        qo_indptr_f = torch.arange(0, batch_size + 1, dtype=torch.int32).to(0)
        workspace_buffer = torch.empty(1024 * 1024 * 1024, dtype=torch.int8).to(0)
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace_buffer, KV_LAYOUT)
        wrapper.plan(
            qo_indptr_f,
            kv_indptr_f,
            kv_indices_f,
            kv_last_page_len_f,
            qo_heads,
            kv_heads,
            head_dim,
            PAGE_SIZE,
            pos_encoding_mode="NONE",
            kv_data_type=torch.float16,
            q_data_type=torch.float16,
        )
        o, lse = wrapper.run_return_lse(q_f, kv_data_f)

        def func():
            return wrapper.run(q_f, kv_data_f)

        ms = bench(func, warmup=10, repeat=30, proton_name="flashinfer_batch_prefill")
        func()
        print(f"FlashInfer BatchPrefillWithPagedKVCacheWrapper time: {ms:.3f} ms")

        return o.reshape(batch_size, qo_heads, head_dim).cpu().numpy()

    def tir():
        def torch_to_tvm(tensor):
            return tvm_ffi.from_dlpack(torch.to_dlpack(tensor))

        DEV = tvm.cuda(0)
        q_tvm = tvm.runtime.tensor(Q, DEV)
        kv_data_tvm = tvm.runtime.tensor(KV_data, DEV)
        q_indptr_tvm = torch_to_tvm(q_indptr)
        kv_indptr_tvm = torch_to_tvm(kv_indptr)
        partial_indptr_tvm = torch_to_tvm(partial_indptr)
        kv_indices_tvm = tvm.runtime.tensor(KV_indices, DEV)
        q_len_tvm = torch_to_tvm(q_len)
        kv_len_tvm = torch_to_tvm(kv_len)
        q_start_tvm = torch_to_tvm(q_start)
        kv_start_tvm = torch_to_tvm(kv_start)
        kv_end_tvm = torch_to_tvm(kv_end)
        kv_head_idx_tvm = torch_to_tvm(kv_head_idx)
        work_indptr_tvm = torch_to_tvm(work_indptr)
        len_kv_chunk_tvm = torch_to_tvm(len_kv_chunk)
        o_tvm = torch_to_tvm(
            torch.zeros(batch_size, qo_heads, head_dim, dtype=torch.float16, device="cuda")
        )
        partial_o_tvm = torch_to_tvm(partial_o)
        partial_lse_tvm = torch_to_tvm(partial_lse)
        merge_indptr_tvm = torch_to_tvm(merge_indptr)
        merge_o_indices_tvm = torch_to_tvm(merge_o_indices)
        num_qo_len_tvm = torch_to_tvm(num_qo_len)

        attention, merge = get_batch_attention_kernel(qo_heads, kv_heads, head_dim, PAGE_SIZE)
        mod_attn = tvm.IRModule({"main": attention})
        mod_merge = tvm.IRModule({"main": merge})
        target = tvm.target.Target("cuda")
        with target:
            mod_attn = tvm.compile(mod_attn, target=target, tir_pipeline="tirx")
            mod_merge = tvm.compile(mod_merge, target=target, tir_pipeline="tirx")

        def func():
            mod_attn(
                q_tvm,
                kv_data_tvm,
                q_indptr_tvm,
                kv_indptr_tvm,
                partial_indptr_tvm,
                kv_indices_tvm,
                q_len_tvm,
                kv_len_tvm,
                q_start_tvm,
                kv_start_tvm,
                kv_end_tvm,
                kv_head_idx_tvm,
                work_indptr_tvm,
                len_kv_chunk_tvm,
                o_tvm,
                partial_o_tvm,
                partial_lse_tvm,
            )
            mod_merge(
                partial_o_tvm,
                o_tvm,
                partial_lse_tvm,
                num_qo_len_tvm,
                merge_indptr_tvm,
                merge_o_indices_tvm,
            )

        ms = bench(func, warmup=10, repeat=30, proton_name="tir")
        func()
        print(f"TIR time: {ms:.3f} ms")

        # return partial_o_tvm.numpy(), partial_lse_tvm.numpy(), o_tvm.numpy()
        return o_tvm.numpy()

    with ProtonContext("batch_attention"):
        print(
            f"qo_heads: {qo_heads}, kv_heads: {kv_heads}, seq_len: {seq_len}, head_dim: {head_dim}, batch_size: {batch_size}, seed: {seed}"  # noqa: E501
        )
        print("Flashinfer BatchAttention Start", flush=True)
        O_flashinfer_attention = flashinfer_batch_attention()
        print("Flashinfer BatchPrefill Start", flush=True)
        O_flashinfer_prefill = flashinfer_batch_prefill()
        print("TIR BatchAttention Start", flush=True)
        O_tir = tir()

    np.testing.assert_allclose(O_flashinfer_prefill, O_flashinfer_attention, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(O_tir, O_flashinfer_attention, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    import itertools

    num_heads_list = [(64, 8)]
    seq_len_list = [512, 2077, 4033]
    head_dim_list = [128]
    batch_size_list = [1, 11, 25, 128]
    seed_list = [42]

    for num_heads, seq_len, head_dim, batch_size, seed in itertools.product(
        num_heads_list, seq_len_list, head_dim_list, batch_size_list, seed_list
    ):
        test(num_heads, seq_len, head_dim, batch_size, seed)
