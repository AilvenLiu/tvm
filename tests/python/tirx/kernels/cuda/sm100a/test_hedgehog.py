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

sys.path.insert(0, os.path.join(os.environ.get("TIRX_KERNELS_PATH", os.path.expanduser("~/tirx-kernels/kernels")), "attention"))
import hedgehog  # noqa: E402
from hedgehog import *  # noqa: E402, F403


@pytest.mark.parametrize(
    "B,H,N", [(1, 2, 128), (2, 2, 256), (2, 2, 512), (1, 4, 1024), (10, 16, 128)]
)
def test_hedgehog(B, H, N):
    q, k, v, qmap, kmap, alphas, betas = prepare_data(B, H, N)
    o_ref, kv_ref, k_ref = naive_hedgehog(q, k, v, qmap, kmap, alphas, betas)

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": get_hedgehog_kernel()})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
        src = mod.mod.imports[0].inspect_source()
        print(src[:3000])

    DEV = tvm.cuda(0)
    BH = B * H
    o_np = np.zeros((BH, N, ATTN_D), dtype=np.float16)
    kv_np = np.zeros((BH, ATTN_F, ATTN_D), dtype=np.float32)
    k_np = np.zeros((BH, ATTN_F), dtype=np.float32)

    q_tvm = tvm.runtime.tensor(
        q.to(torch.float16).reshape(BH, N, ATTN_D).contiguous().cpu().numpy(), DEV
    )
    k_tvm = tvm.runtime.tensor(
        k.to(torch.float16).reshape(BH, N, ATTN_D).contiguous().cpu().numpy(), DEV
    )
    v_tvm = tvm.runtime.tensor(
        v.to(torch.float16).reshape(BH, N, ATTN_D).contiguous().cpu().numpy(), DEV
    )
    qmap_tvm = tvm.runtime.tensor(qmap.to(torch.float16).cpu().numpy(), DEV)
    kmap_tvm = tvm.runtime.tensor(kmap.to(torch.float16).cpu().numpy(), DEV)
    alphas_tvm = tvm.runtime.tensor(alphas.cpu().numpy(), DEV)
    betas_tvm = tvm.runtime.tensor(betas.cpu().numpy(), DEV)
    o_tvm = tvm.runtime.tensor(o_np, DEV)
    kv_tvm = tvm.runtime.tensor(kv_np, DEV)
    k_tvm_out = tvm.runtime.tensor(k_np, DEV)

    mod(q_tvm, k_tvm, v_tvm, qmap_tvm, kmap_tvm, alphas_tvm, betas_tvm, o_tvm, kv_tvm, k_tvm_out)

    o_tir = o_tvm.numpy().reshape(B, H, N, ATTN_D)
    o_ref_np = o_ref.to(torch.float16).cpu().numpy()
    np.testing.assert_allclose(o_tir, o_ref_np, rtol=1e-1, atol=1e-1)
    print(f"PASSED output: B={B}, H={H}, N={N}")

    kv_tir = kv_tvm.numpy().reshape(B, H, ATTN_F, ATTN_D)
    kv_ref_np = kv_ref.cpu().numpy()
    np.testing.assert_allclose(kv_tir, kv_ref_np, rtol=1e-1, atol=1e-1)
    print(f"PASSED kv_state: B={B}, H={H}, N={N}")

    k_tir = k_tvm_out.numpy().reshape(B, H, ATTN_F)
    k_ref_np = k_ref.cpu().numpy()
    np.testing.assert_allclose(k_tir, k_ref_np, rtol=1e-1, atol=1e-1)
    print(f"PASSED k_state: B={B}, H={H}, N={N}")


def bench_hedgehog():
    """Benchmark hedgehog attention kernel at TK-equivalent dimensions."""
    batch, heads = 16, 16
    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": get_hedgehog_kernel()})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")

    DEV = tvm.cuda(0)
    BH = batch * heads

    # FLOPS from TK harness.impl:
    # map_flops = 2 * 2*128*64 * BH * N  (Q,K maps: BH*N*128 @ 128x64)
    # sliding_flops = 2*(96*128 + 96*4 + 96*128) * BH * N  (avg window len 96)
    # linear_flops = (128*128*2 + 128*128*2 + 128*4*2 + 128*4*2) * BH * N
    def flops(ms, N):
        map_flops = 2 * 2 * 128 * 64 * BH * N
        sliding_flops = 2 * (96 * 128 + 96 * 4 + 96 * 128) * BH * N
        linear_flops = (128 * 128 * 2 + 128 * 128 * 2 + 128 * 4 * 2 + 128 * 4 * 2) * BH * N
        return (map_flops + sliding_flops + linear_flops) / (ms * 1e-3)

    print(f"\n{'=' * 60}")
    print(f"Hedgehog Attention Benchmark (B={batch}, H={heads})")
    print(f"{'=' * 60}")

    with ProtonContext("hedgehog"):
        for N in [1024, 2048, 4096, 8192]:
            q = torch.randn(batch, heads, N, ATTN_D, dtype=torch.bfloat16, device="cuda")
            k = torch.randn(batch, heads, N, ATTN_D, dtype=torch.bfloat16, device="cuda")
            v = torch.randn(batch, heads, N, ATTN_D, dtype=torch.bfloat16, device="cuda")
            qmap = torch.randn(heads, ATTN_D, HALF_F, dtype=torch.bfloat16, device="cuda")
            kmap = torch.randn(heads, ATTN_D, HALF_F, dtype=torch.bfloat16, device="cuda")
            alphas = torch.ones(heads, dtype=torch.float32, device="cuda")
            betas = torch.ones(heads, dtype=torch.float32, device="cuda")

            o_np = np.zeros((BH, N, ATTN_D), dtype=np.float16)
            kv_np = np.zeros((BH, ATTN_F, ATTN_D), dtype=np.float32)
            k_np = np.zeros((BH, ATTN_F), dtype=np.float32)

            q_tvm = tvm.runtime.tensor(
                q.to(torch.float16).reshape(BH, N, ATTN_D).contiguous().cpu().numpy(), DEV
            )
            k_tvm = tvm.runtime.tensor(
                k.to(torch.float16).reshape(BH, N, ATTN_D).contiguous().cpu().numpy(), DEV
            )
            v_tvm = tvm.runtime.tensor(
                v.to(torch.float16).reshape(BH, N, ATTN_D).contiguous().cpu().numpy(), DEV
            )
            qmap_tvm = tvm.runtime.tensor(qmap.to(torch.float16).cpu().numpy(), DEV)
            kmap_tvm = tvm.runtime.tensor(kmap.to(torch.float16).cpu().numpy(), DEV)
            alphas_tvm = tvm.runtime.tensor(alphas.cpu().numpy(), DEV)
            betas_tvm = tvm.runtime.tensor(betas.cpu().numpy(), DEV)
            o_tvm = tvm.runtime.tensor(o_np, DEV)
            kv_tvm = tvm.runtime.tensor(kv_np, DEV)
            k_tvm_out = tvm.runtime.tensor(k_np, DEV)

            func = lambda: mod(  # noqa: E731
                q_tvm,
                k_tvm,
                v_tvm,
                qmap_tvm,
                kmap_tvm,
                alphas_tvm,
                betas_tvm,
                o_tvm,
                kv_tvm,
                k_tvm_out,
            )
            ms = bench(func, warmup=100, repeat=300, proton_name=f"hedgehog_N{N}")
            tflops = flops(ms, N) / 1e12
            print(f"  N={N:>5d}: {tflops:.2f} TFLOPS, {ms:.3f} ms")

    print()


if __name__ == "__main__":
    test_hedgehog(1, 2, 128)
    bench_hedgehog()
