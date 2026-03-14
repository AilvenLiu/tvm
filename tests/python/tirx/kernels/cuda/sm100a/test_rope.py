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
import torch

import tvm
from tvm.tirx.bench.utils import ProtonContext, bench

sys.path.insert(
    0,
    os.path.join(
        os.environ.get("TIRX_KERNELS_PATH", os.path.expanduser("~/tirx-kernels/kernels")),
        "activation",
    ),
)
import rope

# Re-export for backward compatibility (qwen3_layer.py, qwen3_model.py import from here)
get_cos_sin_cache_kernel = rope.get_cos_sin_cache_kernel
get_rope_kernel = rope.get_rope_kernel


@pytest.mark.parametrize("rotary_dim", [128])
@pytest.mark.parametrize("max_position_embeddings", [4096])
@pytest.mark.parametrize("base", [1000000])
def test_cos_sin_cache(rotary_dim, max_position_embeddings, base):
    cache_ref = rope.prepare_cos_sin_cache(rotary_dim, max_position_embeddings, base)

    cache_np = np.zeros((max_position_embeddings, rotary_dim), dtype=np.float32)
    cache_tvm = tvm.runtime.tensor(cache_np, tvm.cuda(0))

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": rope.get_cos_sin_cache_kernel(rotary_dim, base)})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
        mod["cos_sin_cache"](cache_tvm)
    print(cache_ref.cpu().numpy())
    print(cache_tvm.numpy())
    np.testing.assert_allclose(cache_ref.cpu().numpy(), cache_tvm.numpy(), atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("num_heads", [8, 64])
@pytest.mark.parametrize("seq_len", [1])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16, 32, 64, 128, 256, 1024])
def test_rope(num_heads, seq_len, head_dim, batch_size):
    rotary_dim = head_dim  # can be different
    max_position_embeddings = 4096
    base = 1000000

    def test_dynamic_batch(num_heads, seq_len, head_dim, batch_size, mod):
        cos_sin_cache, pos_ids, query, key = rope.prepare_data(
            rotary_dim, max_position_embeddings, num_heads, seq_len, head_dim, batch_size, base
        )

        def naive():
            pos_ids_naive, query_naive, key_naive = pos_ids, query.clone(), key.clone()
            pos_ids_naive = pos_ids_naive.flatten()
            num_tokens = pos_ids_naive.shape[0]
            cos_sin = cos_sin_cache.index_select(0, pos_ids_naive)
            query_naive = query_naive.to(torch.float32)
            key_naive = key_naive.to(torch.float32)
            cos, sin = cos_sin.chunk(2, dim=-1)

            def rotary_emb(x):
                cos_r = cos.unsqueeze(-2).to(x.dtype)
                sin_r = sin.unsqueeze(-2).to(x.dtype)
                x1, x2 = torch.chunk(x, 2, dim=-1)
                o1 = x1 * cos_r - x2 * sin_r
                o2 = x2 * cos_r + x1 * sin_r
                return torch.cat((o1, o2), dim=-1)

            query_shape = query_naive.shape
            query_naive = query_naive.view(num_tokens, -1, head_dim)
            query_rot = query_naive[..., :rotary_dim]
            query_pass = query_naive[..., rotary_dim:]
            query_rot = rotary_emb(query_rot)
            query_naive = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

            key_shape = key_naive.shape
            key_naive = key_naive.view(num_tokens, -1, head_dim)
            key_rot = key_naive[..., :rotary_dim]
            key_pass = key_naive[..., rotary_dim:]
            key_rot = rotary_emb(key_rot)
            key_naive = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)

            query_naive = query_naive.to(torch.float16)
            key_naive = key_naive.to(torch.float16)
            return query_naive.cpu().numpy(), key_naive.cpu().numpy()

        def flashinfer():
            import flashinfer

            pos_ids_flashinfer, query_flashinfer, key_flashinfer = (
                pos_ids,
                query.clone(),
                key.clone(),
            )

            def func():
                return flashinfer.rope.apply_rope_with_cos_sin_cache(
                    positions=pos_ids_flashinfer,
                    query=query_flashinfer,
                    key=key_flashinfer,
                    head_size=head_dim,
                    cos_sin_cache=cos_sin_cache,
                    is_neox=True,
                )

            ms = bench(func, warmup=10, repeat=30, proton_name="flashinfer")
            print(f"flashinfer time: {ms:.3f} ms")
            q_out, k_out = func()
            return q_out.cpu().numpy(), k_out.cpu().numpy()

        def tir():
            DEV = tvm.cuda(0)
            pos_ids_tvm = tvm.runtime.tensor(pos_ids.cpu().numpy(), DEV)
            query_tvm = tvm.runtime.tensor(
                query.cpu().numpy().reshape(-1, num_heads, head_dim), DEV
            )
            key_tvm = tvm.runtime.tensor(key.cpu().numpy().reshape(-1, num_heads, head_dim), DEV)
            query_out_tvm = tvm.runtime.empty(
                (batch_size, num_heads, head_dim), dtype="float16", device=DEV
            )
            key_out_tvm = tvm.runtime.empty(
                (batch_size, num_heads, head_dim), dtype="float16", device=DEV
            )
            cos_sin_cache_tvm = tvm.runtime.tensor(cos_sin_cache.cpu().numpy(), DEV)

            def func():
                mod(query_tvm, cos_sin_cache_tvm, pos_ids_tvm, query_out_tvm)
                mod(key_tvm, cos_sin_cache_tvm, pos_ids_tvm, key_out_tvm)

            ms = bench(func, warmup=10, repeat=30, proton_name="tir")
            print(f"tir time: {ms:.3f} ms")
            return query_out_tvm.numpy().reshape(
                -1, num_heads * head_dim
            ), key_out_tvm.numpy().reshape(-1, num_heads * head_dim)

        q_n, k_n = naive()
        # q_f, k_f = flashinfer()
        q_t, k_t = tir()

        # torch.testing.assert_close(q_n, q_f, rtol=1e-3, atol=1e-3)
        # torch.testing.assert_close(k_n, k_f, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(q_n, q_t, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(k_n, k_t, rtol=1e-3, atol=1e-3)

    # compile tir kernel
    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": rope.get_rope_kernel(head_dim)})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")

    with ProtonContext("rope"):
        test_dynamic_batch(num_heads, seq_len, head_dim, batch_size, mod)


if __name__ == "__main__":
    num_heads_list = [8]
    seq_len_list = [1]
    head_dim_list = [128]
    batch_size_list = [1, 2, 4, 8, 16, 32, 64, 128, 256]

    import itertools

    for num_heads, seq_len, head_dim, batch_size in itertools.product(
        num_heads_list, seq_len_list, head_dim_list, batch_size_list
    ):
        test_rope(num_heads, seq_len, head_dim, batch_size)
