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
import numpy as np

import tvm
from tvm.tirx.bench.utils import ProtonContext, bench

sys.path.insert(0, os.path.join(os.environ.get("TIRX_KERNELS_PATH", os.path.expanduser("~/tirx-kernels/kernels")), "attention"))
import linear_attention  # noqa: E402
from linear_attention import *  # noqa: E402, F403


@pytest.mark.parametrize(
    "batch,heads,seq_len", [(1, 8, 256), (1, 8, 1024), (2, 4, 512), (10, 16, 128)]
)
def test_linear_attention(batch, heads, seq_len):
    q, k, v, slopes = prepare_data(batch, heads, seq_len)
    o_ref = naive_linear_attention(q, k, v, slopes)

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": get_linear_attention_kernel()})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")

    DEV = tvm.cuda(0)
    BH = batch * heads
    L = seq_len
    o_np = np.zeros((BH, L, D), dtype=np.float16)
    q_tvm = tvm.runtime.tensor(q.cpu().numpy(), DEV)
    k_tvm = tvm.runtime.tensor(k.cpu().numpy(), DEV)
    v_tvm = tvm.runtime.tensor(v.cpu().numpy(), DEV)
    slopes_tvm = tvm.runtime.tensor(slopes.cpu().numpy(), DEV)
    o_tvm = tvm.runtime.tensor(o_np, DEV)
    mod(q_tvm, k_tvm, v_tvm, slopes_tvm, o_tvm)

    o_tir = o_tvm.numpy()
    o_ref_np = o_ref.cpu().numpy()
    np.testing.assert_allclose(o_tir, o_ref_np, rtol=5e-2, atol=5e-2)
    print(f"PASSED: batch={batch}, heads={heads}, seq_len={seq_len}")


def bench_linear_attention():
    """Benchmark linear attention kernel at TK-equivalent dimensions."""
    batch, heads = 8, 16
    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": get_linear_attention_kernel()})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")

    DEV = tvm.cuda(0)
    BH = batch * heads

    # FLOPS: Q@K^T (2*BH*N*CHUNK*F) + att@V (2*BH*N*D*CHUNK) + Q@kvs (2*BH*N*F*D)
    #        + state_update (2*BH*N*F*D) per chunk pair
    # Simplified: count all matmul ops
    def flops(ms, seq_len):
        nc = seq_len // CHUNK
        # Phase 1: Q@K^T  [64,F] @ [F,64] = 2*64*64*F per chunk
        # Phase 2: att@V_lo [64,64] @ [64,64] = 2*64*64*64 per chunk
        # Phase 3: att@V_hi [64,64] @ [64,64] = 2*64*64*64 per chunk
        # Consumer: Q@kvs [1,F] @ [F,D] = 2*F*D per row, 64 rows = 2*64*F*D per chunk
        # Consumer: state_update K^T@V = 2*64*F*D per chunk (but outer product)
        mma_flops = nc * BH * (2 * 64 * 64 * F + 2 * 2 * 64 * 64 * 64)
        consumer_flops = nc * BH * (2 * 64 * F * D + 2 * 64 * F * D)
        return (mma_flops + consumer_flops) / (ms * 1e-3)

    print(f"\n{'=' * 60}")
    print(f"Linear Attention Benchmark (B={batch}, H={heads})")
    print(f"{'=' * 60}")

    with ProtonContext("linear_attention"):
        for seq_len in [1024, 2048, 4096, 8192]:
            q, k, v, slopes = prepare_data(batch, heads, seq_len)
            L = seq_len
            o_np = np.zeros((BH, L, D), dtype=np.float16)
            q_tvm = tvm.runtime.tensor(q.cpu().numpy(), DEV)
            k_tvm = tvm.runtime.tensor(k.cpu().numpy(), DEV)
            v_tvm = tvm.runtime.tensor(v.cpu().numpy(), DEV)
            slopes_tvm = tvm.runtime.tensor(slopes.cpu().numpy(), DEV)
            o_tvm = tvm.runtime.tensor(o_np, DEV)

            func = lambda: mod(q_tvm, k_tvm, v_tvm, slopes_tvm, o_tvm)  # noqa: E731
            ms = bench(func, warmup=100, repeat=300, proton_name=f"linear_attn_N{seq_len}")
            tflops = flops(ms, seq_len) / 1e12
            print(f"  N={seq_len:>5d}: {tflops:.2f} TFLOPS, {ms:.3f} ms")

    print()


if __name__ == "__main__":
    test_linear_attention(1, 8, 256)
    bench_linear_attention()
