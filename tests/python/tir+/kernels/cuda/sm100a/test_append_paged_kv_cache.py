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
import tvm
from tvm.script import tir as T
from tvm.script import tirp as Tp
from ..utils import bench, ProtonContext

# Problem config
BATCH_SIZE = 32
NUM_HEADS = 16
NUM_TOKENS = 1
HEAD_DIM = 128
KV_SHAPE = BATCH_SIZE, NUM_TOKENS, NUM_HEADS, HEAD_DIM

# Paged kv-cache config
KV_LAYOUT = "HND"
PAGE_SIZE = 16
MAX_PAGE_NUM = 4
CACHE_SHAPE = MAX_PAGE_NUM, 2, NUM_HEADS, PAGE_SIZE, HEAD_DIM

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


def perpare_data():
    import torch

    torch.manual_seed(42)

    cache = torch.randn(CACHE_SHAPE).half()
    k = torch.randn(KV_SHAPE).half()
    v = torch.randn(KV_SHAPE).half()
    pos = random.sample(range(MAX_PAGE_NUM * PAGE_SIZE), BATCH_SIZE * NUM_TOKENS)
    pos_map = torch.tensor(pos, dtype=torch.int32).reshape(BATCH_SIZE, NUM_TOKENS)

    return cache, k, v, pos_map


# fmt: off
@T.prim_func(tirp=True)
def append_paged_kv_cache(cache_ptr: T.handle, k_ptr: T.handle, v_ptr: T.handle, pos_map_ptr: T.handle):
    cache_global = T.match_buffer(cache_ptr, CACHE_SHAPE, "float16", scope="global", layout="default")
    k_global = T.match_buffer(k_ptr, KV_SHAPE, "float16", scope="global", layout="default")
    v_global = T.match_buffer(v_ptr, KV_SHAPE, "float16", scope="global", layout="default")
    pos_map_global = T.match_buffer(pos_map_ptr, [BATCH_SIZE, NUM_TOKENS], "int32", scope="global", layout="default")

    with T.kernel():
        tx, ty = T.thread_id([BDX, BDY], parent="cta")
        bx = T.cta_id([SM_COUNT], parent="kernel")

        stx = T.meta_var(tx * VEC_SIZE)

        with T.thread():
            idx = T.alloc_local([1], "int32", layout="default")
            real_idx = T.alloc_local([1], "int32", layout="default")
            batch_id = T.alloc_local([1], "int32", layout="default")
            token_id = T.alloc_local([1], "int32", layout="default")
            head_id = T.alloc_local([1], "int32", layout="default")
            pos = T.alloc_local([1], "int32", layout="default")
            vec = T.alloc_local([VEC_SIZE], "float16", layout="default")

            idx[0] = bx
            while idx[0] < BATCH_SIZE * NUM_TOKENS * NUM_HEADS * 2 // BDY:
                real_idx[0] = (idx[0] * BDY // 2) + (ty % (BDY // 2))
                batch_id[0] = real_idx[0] // (NUM_TOKENS * NUM_HEADS)
                token_id[0] = (real_idx[0] % (NUM_TOKENS * NUM_HEADS)) // NUM_HEADS
                head_id[0] = (real_idx[0] % (NUM_TOKENS * NUM_HEADS)) % NUM_HEADS
                pos[0] = pos_map_global[batch_id[0], token_id[0]]

                if ty < BDY // 2:
                    Tp.copy(vec[:], k_global[batch_id[0], token_id[0], head_id[0], stx:stx + VEC_SIZE])
                    Tp.copy(cache_global[pos[0] // PAGE_SIZE, 0, head_id[0], pos[0] % PAGE_SIZE, stx:stx + VEC_SIZE], vec[:])
                else:
                    Tp.copy(vec[:], v_global[batch_id[0], token_id[0], head_id[0], stx:stx + VEC_SIZE])
                    Tp.copy(cache_global[pos[0] // PAGE_SIZE, 1, head_id[0], pos[0] % PAGE_SIZE, stx:stx + VEC_SIZE], vec[:])
                
                idx[0] += SM_COUNT
# fmt: on


def test():

    cache, k, v, pos_map = perpare_data()

    def naive():
        import torch

        cache_naive = cache.clone().to("cuda")
        k_naive = k.to("cuda")
        v_naive = v.to("cuda")
        pos_map_naive = pos_map.to("cuda")

        def func():
            page_indices = torch.div(pos_map_naive, PAGE_SIZE, rounding_mode="floor")
            offsets_in_page = torch.remainder(pos_map_naive, PAGE_SIZE)
            cache_naive[page_indices, 0, :, offsets_in_page, :] = k_naive.permute(
                0, 2, 1, 3
            ).reshape(cache_naive[page_indices, 0, :, offsets_in_page, :].shape)
            cache_naive[page_indices, 1, :, offsets_in_page, :] = v_naive.permute(
                0, 2, 1, 3
            ).reshape(cache_naive[page_indices, 1, :, offsets_in_page, :].shape)

        ms = bench(func, warmup=0, repeat=30, proton_name="naive")
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
            mod = tvm.IRModule({"main": append_paged_kv_cache})
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
    test()
