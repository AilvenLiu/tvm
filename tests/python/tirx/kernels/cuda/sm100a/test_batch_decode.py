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

sys.path.insert(0, os.path.join(os.environ.get("TIRX_KERNELS_PATH", os.path.expanduser("~/tirx-kernels/kernels")), "attention"))
import batch_decode  # noqa: E402
from batch_decode import PlanInfo, decode_attn_plan, get_decode_kernel, perpare_data  # noqa: E402

KV_LAYOUT = batch_decode.KV_LAYOUT


@pytest.mark.parametrize("num_heads", [(64, 8)])
@pytest.mark.parametrize("seq_len", [512])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16, 32, 64, 128, 256])
def test(num_heads, seq_len, head_dim, batch_size):
    PAGE_SIZE = 16
    MAX_PAGE_NUM = 32768
    qo_heads, kv_heads = num_heads
    plan_info = PlanInfo(qo_heads, kv_heads, head_dim)

    def test_dynamic_batch_size(batch_size):
        Q, KV_data, KV_indptr, KV_last_page_len, KV_indices = perpare_data(
            batch_size, qo_heads, kv_heads, seq_len, head_dim, PAGE_SIZE, MAX_PAGE_NUM
        )

        def tir():
            DEV = tvm.cuda(0)
            q_tvm = tvm.runtime.tensor(Q, DEV)
            kv_data_tvm = tvm.runtime.tensor(KV_data, DEV)
            kv_indptr_tvm = tvm.runtime.tensor(KV_indptr, DEV)
            kv_last_page_len_tvm = tvm.runtime.tensor(KV_last_page_len, DEV)
            kv_indices_tvm = tvm.runtime.tensor(KV_indices, DEV)
            plan_info.plan(batch_size, KV_indptr.numpy().tolist(), PAGE_SIZE, MAX_PAGE_NUM)

            decode, merge = get_decode_kernel(plan_info, PAGE_SIZE)
            mod_decode = tvm.IRModule({"main": decode})
            mod_merge = tvm.IRModule({"main": merge})
            target = tvm.target.Target("cuda")
            with target:
                mod_decode = tvm.compile(mod_decode, target=target, tir_pipeline="tirx")
                mod_merge = tvm.compile(mod_merge, target=target, tir_pipeline="tirx")

            def func():
                if plan_info.split_kv:
                    mod_decode(
                        q_tvm,
                        kv_data_tvm,
                        plan_info.tmp_lse_tvm,
                        kv_indptr_tvm,
                        kv_last_page_len_tvm,
                        kv_indices_tvm,
                        plan_info.request_indices_tvm,
                        plan_info.kv_tile_indices_tvm,
                        plan_info.max_chunk_size,
                        plan_info.tmp_o_tvm,
                    )
                    mod_merge(
                        plan_info.tmp_o_tvm,
                        plan_info.o_indptr_tvm,
                        plan_info.o_tvm,
                        plan_info.tmp_lse_tvm,
                        plan_info.lse_tvm,
                    )
                else:
                    mod_decode(
                        q_tvm,
                        kv_data_tvm,
                        plan_info.lse_tvm,
                        kv_indptr_tvm,
                        kv_last_page_len_tvm,
                        kv_indices_tvm,
                        plan_info.request_indices_tvm,
                        plan_info.kv_tile_indices_tvm,
                        plan_info.max_chunk_size,
                        plan_info.o_tvm,
                    )

            ms = bench(func, warmup=10, repeat=30, proton_name="tir")
            func()
            print(f"TIR time: {ms:.3f} ms")

            return plan_info.o_tvm.numpy(), plan_info.lse_tvm.numpy()

        def flashinfer_batch_decode():
            import flashinfer
            import torch

            q_f = Q.to(0)
            kv_data_f = KV_data.to(0)
            kv_indptr_f = KV_indptr.to(0)
            kv_last_page_len_f = KV_last_page_len.to(0)
            kv_indices_f = KV_indices.to(0)

            workspace_buffer = torch.empty(1024 * 1024 * 1024, dtype=torch.int8).to(0)
            wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace_buffer, KV_LAYOUT)
            wrapper.plan(
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

            ms = bench(func, warmup=10, repeat=30, proton_name="flashinfer_batch_decode")
            func()
            print(f"FlashInfer BatchDecodeWithPagedKVCacheWrapper time: {ms:.3f} ms")

            return (
                o.reshape(batch_size, qo_heads, head_dim).cpu().numpy(),
                lse.reshape(batch_size, qo_heads).cpu().numpy(),
            )

        def flashinfer_batch_decode_tensor_cores():
            import flashinfer
            import torch

            q_f = Q.to(0)
            kv_data_f = KV_data.to(0)
            kv_indptr_f = KV_indptr.to(0)
            kv_last_page_len_f = KV_last_page_len.to(0)
            kv_indices_f = KV_indices.to(0)

            workspace_buffer = torch.empty(1024 * 1024 * 1024, dtype=torch.int8).to(0)
            wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace_buffer, KV_LAYOUT)
            wrapper.plan(
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

            ms = bench(
                func, warmup=10, repeat=30, proton_name="flashinfer_batch_decode_tensor_cores"
            )
            func()
            print(
                f"FlashInfer BatchDecodeWithPagedKVCacheWrapper(use_tensor_cores=True) time: {ms:.3f} ms"  # noqa: E501
            )

            return (
                o.reshape(batch_size, qo_heads, head_dim).cpu().numpy(),
                lse.reshape(batch_size, qo_heads).cpu().numpy(),
            )

        def flashinfer_batch_prefill():
            import flashinfer
            import torch

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

            return (
                o.reshape(batch_size, qo_heads, head_dim).cpu().numpy(),
                lse.reshape(batch_size, qo_heads).cpu().numpy(),
            )

        def flashinfer_batch_attention():
            import flashinfer
            import torch

            q_f = Q.to(0).reshape(batch_size, qo_heads, head_dim)
            kv_data_f = KV_data.to(0)
            kv_indptr_f = KV_indptr.to(0)
            KV_last_page_len.to(0)
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
            o, lse = wrapper.run(q_f, kv_data_f)

            def func():
                return wrapper.run(q_f, kv_data_f)

            ms = bench(func, warmup=10, repeat=30, proton_name="flashinfer_batch_attention")
            func()
            print(f"FlashInfer BatchAttention time: {ms:.3f} ms")

            return (
                o.reshape(batch_size, qo_heads, head_dim).cpu().numpy(),
                lse.reshape(batch_size, qo_heads).cpu().numpy(),
            )

        with ProtonContext("batch_decode"):
            print(
                f">>>>>>>>>>>>>>>>>>>>>>>>> Testing (B,(H_qo,H_kv),N,D) = ({batch_size},({qo_heads},{kv_heads}),{seq_len},{head_dim})"  # noqa: E501
            )
            O_flashinfer_batch_decode, lse_flashinfer_batch_decode = flashinfer_batch_decode()
            O_flashinfer_batch_decode_tensor_cores, lse_flashinfer_batch_decode_tensor_cores = (
                flashinfer_batch_decode_tensor_cores()
            )
            O_flashinfer_batch_prefill, lse_flashinfer_batch_prefill = flashinfer_batch_prefill()
            O_flashinfer_batch_attention, lse_flashinfer_batch_attention = (
                flashinfer_batch_attention()
            )
            O_tir, lse_tir = tir()

            np.testing.assert_allclose(O_tir, O_flashinfer_batch_decode, rtol=1e-3, atol=1e-3)
            np.testing.assert_allclose(lse_tir, lse_flashinfer_batch_decode, rtol=1e-3, atol=1e-3)
            np.testing.assert_allclose(
                O_tir, O_flashinfer_batch_decode_tensor_cores, rtol=1e-3, atol=1e-3
            )
            np.testing.assert_allclose(
                lse_tir, lse_flashinfer_batch_decode_tensor_cores, rtol=1e-3, atol=1e-3
            )
            np.testing.assert_allclose(O_tir, O_flashinfer_batch_prefill, rtol=1e-3, atol=1e-3)
            np.testing.assert_allclose(lse_tir, lse_flashinfer_batch_prefill, rtol=1e-3, atol=1e-3)
            np.testing.assert_allclose(O_tir, O_flashinfer_batch_attention, rtol=1e-3, atol=1e-3)
            np.testing.assert_allclose(
                lse_tir, lse_flashinfer_batch_attention, rtol=1e-3, atol=1e-3
            )

    test_dynamic_batch_size(batch_size)


if __name__ == "__main__":
    import itertools

    num_heads_list = [(32, 8)]
    seq_len_list = [512, 3456]
    head_dim_list = [128]
    batch_size_list = [1, 128]

    for num_heads, seq_len, head_dim, batch_size in itertools.product(
        num_heads_list, seq_len_list, head_dim_list, batch_size_list
    ):
        test(num_heads, seq_len, head_dim, batch_size)
