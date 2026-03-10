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
import mamba2  # noqa: E402


@pytest.mark.parametrize("batch,heads,seq_len", [(1, 16, 256), (2, 8, 512), (10, 16, 128)])
def test_mamba2(batch, heads, seq_len):
    q, k, v, a = mamba2.prepare_data(batch, heads, seq_len)
    o_ref = mamba2.naive_mamba2(q, k, v, a)

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": mamba2.get_mamba2_kernel()})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
        src = mod.mod.imports[0].inspect_source()
        print(src[:2000])

    DEV = tvm.cuda(0)
    BH = batch * heads
    L = seq_len
    o_np = np.zeros((BH, L, mamba2.D_MODEL), dtype=np.float16)
    q_tvm = tvm.runtime.tensor(q.cpu().numpy(), DEV)
    k_tvm = tvm.runtime.tensor(k.cpu().numpy(), DEV)
    v_tvm = tvm.runtime.tensor(v.cpu().numpy(), DEV)
    a_tvm = tvm.runtime.tensor(a.cpu().numpy(), DEV)
    o_tvm = tvm.runtime.tensor(o_np, DEV)
    mod(q_tvm, k_tvm, v_tvm, a_tvm, o_tvm)

    o_tir = o_tvm.numpy()
    o_ref_np = o_ref.cpu().numpy()
    np.testing.assert_allclose(o_tir, o_ref_np, rtol=5e-2, atol=5e-2)
    print(f"PASSED: batch={batch}, heads={heads}, seq_len={seq_len}")


def bench_mamba2():
    """Benchmark Mamba2 SSD kernel at TK-equivalent dimensions."""
    batch, heads = 16, 16
    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": mamba2.get_mamba2_kernel()})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")

    DEV = tvm.cuda(0)
    BH = batch * heads

    # FLOPS from TK Mamba2 benchmark (SSD-specific):
    # chunk=64, state=64, ngroups=1
    # mask/decay + center blocks + low-rank factors + inter-chunk + output
    def flops(ms, seq_len):
        chunk = mamba2.CHUNK
        state = mamba2.D_MODEL
        ngroups = 1
        num_chunks = seq_len // chunk
        f = 0
        # Mask: cumsum + segsum + exp
        f += seq_len * heads + seq_len * heads * chunk
        # Center blocks: QK^T and QKV
        f += 2 * num_chunks * ngroups * chunk * chunk * state
        f += num_chunks * heads * chunk * chunk
        f += 2 * num_chunks * heads * chunk * chunk * mamba2.D_MODEL
        # Low-rank: decay states + state computation
        f += num_chunks * heads * chunk
        f += 2 * num_chunks * heads * chunk * mamba2.D_MODEL * state
        # Inter-chunk: state update
        f += 2 * num_chunks * heads * chunk * mamba2.D_MODEL * state
        # Output
        f += num_chunks * heads * chunk
        f += 2 * num_chunks * heads * chunk * mamba2.D_MODEL * state
        f += num_chunks * heads * state
        f += num_chunks * heads * chunk * mamba2.D_MODEL
        return batch * f / (ms * 1e-3)

    print(f"\n{'=' * 60}")
    print(f"Mamba2 SSD Benchmark (B={batch}, H={heads})")
    print(f"{'=' * 60}")

    with ProtonContext("mamba2"):
        for seq_len in [1024, 2048, 4096, 8192]:
            q, k, v, a = mamba2.prepare_data(batch, heads, seq_len)
            L = seq_len
            o_np = np.zeros((BH, L, mamba2.D_MODEL), dtype=np.float16)
            q_tvm = tvm.runtime.tensor(q.cpu().numpy(), DEV)
            k_tvm = tvm.runtime.tensor(k.cpu().numpy(), DEV)
            v_tvm = tvm.runtime.tensor(v.cpu().numpy(), DEV)
            a_tvm = tvm.runtime.tensor(a.cpu().numpy(), DEV)
            o_tvm = tvm.runtime.tensor(o_np, DEV)

            func = lambda: mod(q_tvm, k_tvm, v_tvm, a_tvm, o_tvm)  # noqa: E731
            ms = bench(func, warmup=100, repeat=300, proton_name=f"mamba2_N{seq_len}")
            tflops = flops(ms, seq_len) / 1e12
            print(f"  N={seq_len:>5d}: {tflops:.2f} TFLOPS, {ms:.3f} ms")

    print()


if __name__ == "__main__":
    test_mamba2(1, 16, 256)
    bench_mamba2()
