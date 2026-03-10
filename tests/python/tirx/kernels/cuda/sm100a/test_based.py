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
import tvm.testing
from tvm.tirx.bench.utils import ProtonContext, bench

sys.path.insert(0, os.path.join(os.environ.get("TIRX_KERNELS_PATH", os.path.expanduser("~/tirx-kernels/kernels")), "ssm"))
import based  # noqa: E402


@pytest.mark.parametrize(
    "batch,heads,seq_len", [(1, 1, 64), (1, 1, 256), (1, 8, 1024), (10, 16, 128)]
)
def test_based(batch, heads, seq_len):
    q, k, v = based.prepare_data(batch, heads, seq_len)
    o_ref, a0_ref, a1_ref, a2_ref = based.naive_based(q, k, v)

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": based.get_based_kernel()})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
        src = mod.mod.imports[0].inspect_source()
        print(src[:2000])

    DEV = tvm.cuda(0)
    BH = batch * heads
    L = seq_len
    o_np = np.zeros((BH, L, based.D_VO), dtype=np.float16)
    a0_np = np.zeros((BH, based.D_VO), dtype=np.float16)
    a1_np = np.zeros((BH, based.D_QK, based.D_VO), dtype=np.float16)
    a2_np = np.zeros((BH, based.D_QK * based.D_QK, based.D_VO), dtype=np.float16)

    q_tvm = tvm.runtime.tensor(q.cpu().numpy(), DEV)
    k_tvm = tvm.runtime.tensor(k.cpu().numpy(), DEV)
    v_tvm = tvm.runtime.tensor(v.cpu().numpy(), DEV)
    o_tvm = tvm.runtime.tensor(o_np, DEV)
    a0_tvm = tvm.runtime.tensor(a0_np, DEV)
    a1_tvm = tvm.runtime.tensor(a1_np, DEV)
    a2_tvm = tvm.runtime.tensor(a2_np, DEV)
    mod(q_tvm, k_tvm, v_tvm, o_tvm, a0_tvm, a1_tvm, a2_tvm)

    # Check O
    o_tir = o_tvm.numpy()
    o_ref_np = o_ref.cpu().numpy()
    abs_diff = np.abs(o_tir.astype(np.float32) - o_ref_np.astype(np.float32))
    abs_ref = np.abs(o_ref_np.astype(np.float32))
    print(
        "O:  "
        f"avg_ref={abs_ref.mean():.6f}, "
        f"avg_diff={abs_diff.mean():.6f}, "
        f"max_diff={abs_diff.max():.6f}"
    )
    # f16 accumulation error grows with nc (chunks); scale O tolerance accordingly
    nc = seq_len // 64
    o_atol = max(0.3, 0.01 * nc)
    np.testing.assert_allclose(o_tir, o_ref_np, rtol=o_atol, atol=o_atol)

    # Check a0
    a0_tir = a0_tvm.numpy()
    a0_ref_np = a0_ref.cpu().numpy()
    print(
        f"a0: avg_ref={np.abs(a0_ref_np).mean():.6f}, "
        f"max_diff={np.abs(a0_tir.astype(np.float32) - a0_ref_np.astype(np.float32)).max():.6f}"
    )
    np.testing.assert_allclose(a0_tir, a0_ref_np, rtol=1e-1, atol=1e-1)

    # Check a1
    a1_tir = a1_tvm.numpy()
    a1_ref_np = a1_ref.cpu().numpy()
    print(
        f"a1: avg_ref={np.abs(a1_ref_np).mean():.6f}, "
        f"max_diff={np.abs(a1_tir.astype(np.float32) - a1_ref_np.astype(np.float32)).max():.6f}"
    )
    # Scale tolerance with sequence length — f16 accumulation error grows with chunks.
    # Large BH also increases statistical chance of outlier diffs, so add a small margin.
    nc = seq_len // 64
    state_atol = max(0.2, 0.04 * nc)
    np.testing.assert_allclose(a1_tir, a1_ref_np, rtol=state_atol, atol=state_atol)

    # Check a2
    a2_tir = a2_tvm.numpy()
    a2_ref_np = a2_ref.cpu().numpy()
    print(
        f"a2: avg_ref={np.abs(a2_ref_np).mean():.6f}, "
        f"max_diff={np.abs(a2_tir.astype(np.float32) - a2_ref_np.astype(np.float32)).max():.6f}"
    )
    np.testing.assert_allclose(a2_tir, a2_ref_np, rtol=state_atol, atol=state_atol)

    print(f"PASSED (O + states): batch={batch}, heads={heads}, seq_len={seq_len}")


def bench_based():
    """Benchmark BASED kernel at TK-equivalent dimensions."""
    batch, heads = 16, 16
    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": based.get_based_kernel()})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")

    DEV = tvm.cuda(0)
    BH = batch * heads

    # FLOPS formula from TK: expanded_dim = D_QK^2 + D_QK + 1 = 273
    # f = 2*B*N*H*expanded_dim (feature map Q & K)
    #   + 4 * B*N*H*D_VO*expanded_dim (kv, cumsum, q*kv, sum)
    EXPANDED_DIM = based.D_QK * based.D_QK + based.D_QK + 1  # 273

    def flops(ms, seq_len):
        f = 2 * batch * seq_len * heads * EXPANDED_DIM
        f += 4 * batch * seq_len * heads * based.D_VO * EXPANDED_DIM
        return f / (ms * 1e-3)

    print(f"\n{'=' * 60}")
    print(f"BASED Benchmark (B={batch}, H={heads})")
    print(f"{'=' * 60}")

    with ProtonContext("based"):
        for seq_len in [1024, 2048, 4096, 8192]:
            q, k, v = (
                torch.randn(BH, seq_len, based.D_QK, dtype=torch.float16, device="cuda")
                / (based.D_QK**0.5),
                torch.randn(BH, seq_len, based.D_QK, dtype=torch.float16, device="cuda")
                / (based.D_QK**0.5),
                torch.randn(BH, seq_len, based.D_VO, dtype=torch.float16, device="cuda"),
            )
            o_np = np.zeros((BH, seq_len, based.D_VO), dtype=np.float16)
            a0_np = np.zeros((BH, based.D_VO), dtype=np.float16)
            a1_np = np.zeros((BH, based.D_QK, based.D_VO), dtype=np.float16)
            a2_np = np.zeros((BH, based.D_QK * based.D_QK, based.D_VO), dtype=np.float16)

            q_tvm = tvm.runtime.tensor(q.cpu().numpy(), DEV)
            k_tvm = tvm.runtime.tensor(k.cpu().numpy(), DEV)
            v_tvm = tvm.runtime.tensor(v.cpu().numpy(), DEV)
            o_tvm = tvm.runtime.tensor(o_np, DEV)
            a0_tvm = tvm.runtime.tensor(a0_np, DEV)
            a1_tvm = tvm.runtime.tensor(a1_np, DEV)
            a2_tvm = tvm.runtime.tensor(a2_np, DEV)

            def func():
                mod(q_tvm, k_tvm, v_tvm, o_tvm, a0_tvm, a1_tvm, a2_tvm)

            ms = bench(func, warmup=100, repeat=300, proton_name=f"based_N{seq_len}")
            tflops = flops(ms, seq_len) / 1e12
            print(f"  N={seq_len:>5d}: {tflops:.2f} TFLOPS, {ms:.3f} ms")

    print()


if __name__ == "__main__":
    test_based(1, 1, 64)
    bench_based()
