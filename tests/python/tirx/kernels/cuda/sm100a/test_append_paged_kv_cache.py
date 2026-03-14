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

sys.path.insert(
    0,
    os.path.join(
        os.environ.get("TIRX_KERNELS_PATH", os.path.expanduser("~/tirx-kernels/kernels")),
        "attention",
    ),
)
import append_paged_kv_cache


@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16, 32, 64, 128, 256, 499])
def test(batch_size):
    max_page_num = 1024
    page_size = 16
    num_heads = 16
    num_tokens = 1
    head_dim = 128
    cache, k, v, pos_map = append_paged_kv_cache.perpare_data(
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
        cache_tvm = tvm.runtime.tensor(cache.clone(), device=DEV)
        k_tvm = tvm.runtime.tensor(k, device=DEV)
        v_tvm = tvm.runtime.tensor(v, device=DEV)
        pos_map_tvm = tvm.runtime.tensor(pos_map, device=DEV)
        target = tvm.target.Target("cuda")
        with target:
            mod = tvm.IRModule(
                {
                    "main": append_paged_kv_cache.get_append_paged_kv_cache_kernel(
                        num_heads, num_tokens, head_dim
                    )
                }
            )
            mod = tvm.compile(mod, target=target, tir_pipeline="tirx")

            def func():
                return mod(cache_tvm, k_tvm, v_tvm, pos_map_tvm)

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
