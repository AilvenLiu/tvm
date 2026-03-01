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

import numpy as np
import pytest
import torch

import tvm
from tvm.script import tirx as Tx
from tvm.tirx.bench.utils import ProtonContext, bench

F16_BYTES = 2
F32_BYTES = 4
SM_COUNT = 148
MAX_BLK_PER_SM = 32


def ceildiv(a, b):
    return (a + b - 1) // b


def find_power_of_two(n):
    assert n > 0 and (n & (n - 1)) == 0
    return n.bit_length() - 1


def get_cos_sin_cache_kernel(rotary_dim, base):
    assert rotary_dim % 2 == 0

    # fmt: off
    @Tx.prim_func(tirx=True)
    def cos_sin_cache(cos_sin_cache: Tx.handle):
        max_seq_len = Tx.int32()
        cos_sin_cache_global = Tx.match_buffer(cos_sin_cache, [max_seq_len, rotary_dim], "float32", scope="global")  # noqa: E501

        with Tx.kernel():
            bx = Tx.cta_id([SM_COUNT], parent="kernel")
            tx = Tx.thread_id([1024], parent="cta")

            with Tx.thread():
                idx = Tx.alloc_local([1], "int32")

                idx[0] = bx * 1024 + tx
                while idx[0] < max_seq_len * rotary_dim:
                    row = Tx.meta_var(idx[0] // rotary_dim)
                    col = Tx.meta_var(idx[0] % rotary_dim)

                    if col < rotary_dim // 2:
                        cos_sin_cache_global[row, col] = Tx.cos(Tx.float64(row) / Tx.pow(base, Tx.float64(col * 2) / Tx.float64(rotary_dim)))  # noqa: E501
                    else:
                        cos_sin_cache_global[row, col] = Tx.sin(Tx.float64(row) / Tx.pow(base, Tx.float64(col * 2 - rotary_dim) / Tx.float64(rotary_dim)))  # noqa: E501

                    idx[0] += SM_COUNT * 1024
    # fmt: on

    return cos_sin_cache


def get_rope_kernel(head_dim):
    rotary_dim = head_dim  # can be different
    vec_size = max(16 // F16_BYTES, rotary_dim // 32)
    bdx = rotary_dim // vec_size
    num_threads = max(256, bdx)
    bdy = num_threads // bdx

    # fmt: off
    @Tx.prim_func(tirx=True)
    def rope_with_cos_sin_cache(q: Tx.handle, cos_sin_cache: Tx.handle, pos_ids: Tx.handle, q_rope: Tx.handle):  # noqa: E501
        nnz = Tx.int32()
        num_heads = Tx.int32()
        max_seq_len = Tx.int32()
        q_global = Tx.match_buffer(q, [nnz, num_heads, head_dim], "float16", scope="global")
        q_rope_global = Tx.match_buffer(q_rope, [nnz, num_heads, head_dim], "float16", scope="global")  # noqa: E501
        cos_sin_cache_global = Tx.match_buffer(cos_sin_cache, [max_seq_len, rotary_dim], "float32", scope="global")  # noqa: E501
        pos_ids_global = Tx.match_buffer(pos_ids, [nnz], "int32", scope="global")
        half_rotary_dim: Tx.let = rotary_dim // 2

        with Tx.kernel():
            bx = Tx.cta_id([SM_COUNT], parent="kernel")
            tx, ty = Tx.thread_id([bdx, bdy], parent="cta")

            stx = Tx.meta_var(tx * vec_size)

            with Tx.thread():
                cos = Tx.alloc_local([vec_size], "float32")
                sin = Tx.alloc_local([vec_size], "float32")
                qk_vec = Tx.alloc_local([vec_size], "float16")
                qk_vec32 = Tx.alloc_local([vec_size], "float32")
                qk_vec32_other = Tx.alloc_local([vec_size], "float32")
                idx = Tx.alloc_local([1], "int32")
                pos = Tx.alloc_local([1], "int32")
                head = Tx.alloc_local([1], "int32")

                @Tx.inline
                def compute_rope(global_in, global_out):
                    Tx.copy(qk_vec[:], global_in[pos[0], head[0], stx:stx + vec_size])
                    Tx.cast(qk_vec32[:], qk_vec[:])
                    if stx < half_rotary_dim:
                        Tx.copy(qk_vec[:], global_in[pos[0], head[0], stx + half_rotary_dim:stx + half_rotary_dim + vec_size])  # noqa: E501
                    else:
                        Tx.copy(qk_vec[:], global_in[pos[0], head[0], stx - half_rotary_dim:stx - half_rotary_dim + vec_size])  # noqa: E501
                    Tx.cast(qk_vec32_other[:], qk_vec[:])
                    if stx < half_rotary_dim:
                        for kv in Tx.unroll(vec_size):
                            qk_vec32[kv] = qk_vec32[kv] * cos[kv] - qk_vec32_other[kv] * sin[kv]
                    else:
                        for kv in Tx.unroll(vec_size):
                            qk_vec32[kv] = qk_vec32[kv] * cos[kv] + qk_vec32_other[kv] * sin[kv]
                    Tx.cast(qk_vec[:], qk_vec32[:])
                    Tx.copy(global_out[pos[0], head[0], stx:stx + vec_size], qk_vec[:])

                idx[0] = bx * bdy + ty
                while idx[0] < nnz * num_heads:
                    pos[0] = idx[0] % nnz
                    head[0] = idx[0] // nnz
                    cache_stx = Tx.meta_var(stx % half_rotary_dim)
                    Tx.copy(cos[:], cos_sin_cache_global[pos_ids_global[pos[0]], cache_stx:cache_stx + vec_size])  # noqa: E501
                    Tx.copy(sin[:], cos_sin_cache_global[pos_ids_global[pos[0]], cache_stx + half_rotary_dim:cache_stx + half_rotary_dim + vec_size])  # noqa: E501
                    compute_rope(q_global, q_rope_global)
                    idx[0] += SM_COUNT * bdy
    return rope_with_cos_sin_cache


def prepare_cos_sin_cache(rotary_dim, max_position_embeddings, base):
    inv_freq = 1.0 / (
        base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float, device="cuda") / rotary_dim)
    )
    t = torch.arange(max_position_embeddings, dtype=torch.float, device="cuda")
    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()
    cos_sin_cache = torch.cat((cos, sin), dim=-1)  # shape: [max_position_embeddings, rotary_dim]
    return cos_sin_cache


def prepare_data(
    rotary_dim, max_position_embeddings, num_heads, seq_len, head_dim, batch_size, base
):
    pos_ids = torch.arange(seq_len, device="cuda", dtype=torch.int32).repeat(
        batch_size
    )  # shape: [nnz]
    query = torch.randn(
        batch_size * seq_len, num_heads * head_dim, dtype=torch.float16, device="cuda"
    )  # shape: [nnz, num_heads * head_dim]
    key = torch.randn(
        batch_size * seq_len, num_heads * head_dim, dtype=torch.float16, device="cuda"
    )  # shape: [nnz, num_heads * head_dim]
    return prepare_cos_sin_cache(rotary_dim, max_position_embeddings, base), pos_ids, query, key


@pytest.mark.parametrize("rotary_dim", [128])
@pytest.mark.parametrize("max_position_embeddings", [4096])
@pytest.mark.parametrize("base", [1000000])
def test_cos_sin_cache(rotary_dim, max_position_embeddings, base):
    cache_ref = prepare_cos_sin_cache(rotary_dim, max_position_embeddings, base)

    cache_np = np.zeros((max_position_embeddings, rotary_dim), dtype=np.float32)
    cache_tvm = tvm.runtime.tensor(cache_np, tvm.cuda(0))

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": get_cos_sin_cache_kernel(rotary_dim, base)})
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
        cos_sin_cache, pos_ids, query, key = prepare_data(
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
        mod = tvm.IRModule({"main": get_rope_kernel(head_dim)})
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
