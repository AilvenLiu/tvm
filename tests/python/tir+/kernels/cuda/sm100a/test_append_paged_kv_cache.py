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
import random

import numpy as np
import pytest

import tvm
from tvm.script import tir as T
from tvm.script import tirp as Tp
from tvm.tirp.bench.utils import ProtonContext, bench


def perpare_data(batch_size, max_page_num, page_size, num_heads, num_tokens, head_dim):
    import torch

    torch.manual_seed(42)

    cache = torch.randn(max_page_num, 2, num_heads, page_size, head_dim).half()
    k = torch.randn(batch_size, num_tokens, num_heads, head_dim).half()
    v = torch.randn(batch_size, num_tokens, num_heads, head_dim).half()
    pos = random.sample(range(max_page_num * page_size), batch_size * num_tokens)
    pos_map = torch.tensor(pos, dtype=torch.int32).reshape(batch_size, num_tokens)

    return cache, k, v, pos_map


def get_append_paged_kv_cache_kernel(num_heads, num_tokens, head_dim):
    # Problem config
    NUM_HEADS = num_heads
    NUM_TOKENS = num_tokens
    HEAD_DIM = head_dim

    # Paged kv-cache config
    KV_LAYOUT = "HND"

    # HW config
    SM_COUNT = 148
    SMEM_SIZE = 232448

    # Other
    F16_BYTE = 2
    F32_BYTE = 4

    # Kernel config
    VEC_SIZE = max(16 // F16_BYTE, HEAD_DIM // 32)
    THREAD_NUM = 256
    BDX = HEAD_DIM // VEC_SIZE
    BDY = THREAD_NUM // BDX
    # fmt: off
    @T.prim_func(tirp=True)
    def append_paged_kv_cache(cache_ptr: T.handle, k_ptr: T.handle, v_ptr: T.handle, pos_map_ptr: T.handle):
        batch_size = T.int32()
        max_page_num = T.int32()
        page_size = T.int32()

        cache_global = T.match_buffer(cache_ptr, (max_page_num, 2, NUM_HEADS, page_size, HEAD_DIM), "float16", scope="global")
        k_global = T.match_buffer(k_ptr, (batch_size, NUM_TOKENS, NUM_HEADS, HEAD_DIM), "float16", scope="global")
        v_global = T.match_buffer(v_ptr, (batch_size, NUM_TOKENS, NUM_HEADS, HEAD_DIM), "float16", scope="global")
        pos_map_global = T.match_buffer(pos_map_ptr, (batch_size, NUM_TOKENS), "int32", scope="global", offset_factor=1)

        with T.kernel():
            tx, ty = T.thread_id([BDX, BDY], parent="cta")
            bx = T.cta_id([SM_COUNT], parent="kernel")

            stx = T.meta_var(tx * VEC_SIZE)

            with T.thread():
                idx = T.alloc_local([1], "int32")
                real_idx = T.alloc_local([1], "int32")
                batch_id = T.alloc_local([1], "int32")
                token_id = T.alloc_local([1], "int32")
                head_id = T.alloc_local([1], "int32")
                pos = T.alloc_local([1], "int32")
                kv_id = T.alloc_local([1], "int32")
                vec = T.alloc_local([VEC_SIZE], "float16")

                idx[0] = bx * BDY + ty
                while idx[0] < batch_size * NUM_TOKENS * NUM_HEADS * 2:
                    kv_id[0] = idx[0] % 2
                    head_id[0] = (idx[0] // 2) % NUM_HEADS
                    token_id[0] = (idx[0] // (2 * NUM_HEADS)) % NUM_TOKENS
                    batch_id[0] = (idx[0] // (2 * NUM_HEADS * NUM_TOKENS))
                    pos[0] = pos_map_global[batch_id[0], token_id[0]]
                    if kv_id[0] == 0:
                        Tp.copy(vec[:], k_global[batch_id[0], token_id[0], head_id[0], stx:stx + VEC_SIZE])
                    else:
                        Tp.copy(vec[:], v_global[batch_id[0], token_id[0], head_id[0], stx:stx + VEC_SIZE])
                    Tp.copy(cache_global[pos[0] // page_size, kv_id[0], head_id[0], pos[0] % page_size, stx:stx + VEC_SIZE], vec[:])
                    idx[0] += SM_COUNT * BDY

    return append_paged_kv_cache


# fmt: on


@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16, 32, 64, 128, 256, 499])
def test(batch_size):
    max_page_num = 1024
    page_size = 16
    num_heads = 16
    num_tokens = 1
    head_dim = 128
    cache, k, v, pos_map = perpare_data(
        batch_size, max_page_num, page_size, num_heads, num_tokens, head_dim
    )

    def naive():
        import torch

        cache_naive = cache.clone().to("cuda")
        k_naive = k.to("cuda")
        v_naive = v.to("cuda")
        pos_map_naive = pos_map.to("cuda")

        def func():
            page_indices = torch.div(pos_map_naive, page_size, rounding_mode="floor")
            offsets_in_page = torch.remainder(pos_map_naive, page_size)
            cache_naive[page_indices, 0, :, offsets_in_page, :] = k_naive.permute(
                0, 2, 1, 3
            ).reshape(cache_naive[page_indices, 0, :, offsets_in_page, :].shape)
            cache_naive[page_indices, 1, :, offsets_in_page, :] = v_naive.permute(
                0, 2, 1, 3
            ).reshape(cache_naive[page_indices, 1, :, offsets_in_page, :].shape)

        ms = bench(func, warmup=0, repeat=5, proton_name="naive")
        print(f"torch time: {ms:.3f} ms")

        return cache_naive.cpu().numpy()

    def tir():
        DEV = tvm.cuda(0)
        cache_tvm = tvm.nd.array(cache.clone(), device=DEV)
        k_tvm = tvm.nd.array(k, device=DEV)
        v_tvm = tvm.nd.array(v, device=DEV)
        pos_map_tvm = tvm.nd.array(pos_map, device=DEV)
        target = tvm.target.Target("cuda")
        with target:
            mod = tvm.IRModule(
                {"main": get_append_paged_kv_cache_kernel(num_heads, num_tokens, head_dim)}
            )
            mod = tvm.compile(mod, target=target, tir_pipeline="tirp")
            func = lambda: mod(cache_tvm, k_tvm, v_tvm, pos_map_tvm)
            # func()
            ms = bench(func, warmup=0, repeat=30, proton_name="tir")
            print(f"TIR time: {ms:.3f} ms")

        return cache_tvm.numpy()

    with ProtonContext("append_kv"):
        cache_naive = naive()
        cache_tir = tir()

    np.testing.assert_allclose(cache_naive, cache_tir, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    for batch_size in [1, 2, 4, 8, 16, 32, 64, 128, 256, 499]:
        test(batch_size)
