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

import math
import os
import sys

import numpy as np
import pytest
import torch

import tvm
import tvm.testing
from tvm.tirx.bench.utils import ProtonContext, bench

sys.path.insert(0, os.path.join(os.environ.get("TIRX_KERNELS_PATH", os.path.expanduser("~/tirx-kernels/kernels")), "ssm"))
import fftconv  # noqa: E402


@pytest.mark.parametrize("batch,heads", [(2, 4), (1, 8), (10, 16)])
def test_fftconv(batch, heads):
    (
        u_f16,
        kf_real,
        kf_imag,
        f_real,
        f_imag,
        finv_real,
        finv_imag,
        tw_real,
        tw_imag,
        twinv_real,
        twinv_imag,
        u_orig,
        k_orig,
    ) = fftconv.prepare_data(batch, heads)

    # Reference
    o_ref = fftconv.ref_fftconv(u_orig, k_orig)
    o_ref_np = o_ref.float().cpu().numpy()

    # Compile
    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": fftconv.get_fftconv_kernel()})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
        src = mod.mod.imports[0].inspect_source()
        print(src[:2000])

    # Run
    DEV = tvm.cuda(0)
    o_np = np.zeros((batch, heads, fftconv.N1, fftconv.N1), dtype=np.float16)

    x_tvm = tvm.runtime.tensor(u_f16.cpu().numpy(), DEV)
    kf_real_tvm = tvm.runtime.tensor(kf_real.cpu().numpy(), DEV)
    kf_imag_tvm = tvm.runtime.tensor(kf_imag.cpu().numpy(), DEV)
    f_real_tvm = tvm.runtime.tensor(f_real.cpu().numpy(), DEV)
    f_imag_tvm = tvm.runtime.tensor(f_imag.cpu().numpy(), DEV)
    finv_real_tvm = tvm.runtime.tensor(finv_real.cpu().numpy(), DEV)
    finv_imag_tvm = tvm.runtime.tensor(finv_imag.cpu().numpy(), DEV)
    tw_real_tvm = tvm.runtime.tensor(tw_real.cpu().numpy(), DEV)
    tw_imag_tvm = tvm.runtime.tensor(tw_imag.cpu().numpy(), DEV)
    twinv_real_tvm = tvm.runtime.tensor(twinv_real.cpu().numpy(), DEV)
    twinv_imag_tvm = tvm.runtime.tensor(twinv_imag.cpu().numpy(), DEV)
    o_tvm = tvm.runtime.tensor(o_np, DEV)

    mod(
        x_tvm,
        kf_real_tvm,
        kf_imag_tvm,
        f_real_tvm,
        f_imag_tvm,
        finv_real_tvm,
        finv_imag_tvm,
        tw_real_tvm,
        tw_imag_tvm,
        twinv_real_tvm,
        twinv_imag_tvm,
        o_tvm,
    )

    o_tir = o_tvm.numpy().reshape(batch, heads, fftconv.N).astype(np.float32)
    o_ref_flat = o_ref_np.reshape(batch, heads, fftconv.N)

    np.testing.assert_allclose(o_tir, o_ref_flat, rtol=0.05, atol=2.0)
    print(f"PASSED: batch={batch}, heads={heads}")


def bench_fftconv():
    """Benchmark FFT convolution kernel at TK-equivalent dimensions."""
    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": fftconv.get_fftconv_kernel()})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")

    DEV = tvm.cuda(0)

    # FLOPS from TK: 2 * (10 * N * log2(N) * d_model * B)
    # N=4096, d_model = heads * 1 (each head processes one channel)
    def flops(ms, batch, heads):
        return 2 * (10 * fftconv.N * math.log2(fftconv.N) * heads * batch) / (ms * 1e-3)

    print(f"\n{'=' * 60}")
    print(f"FFTConv Benchmark (N={fftconv.N}={fftconv.N1}x{fftconv.N1})")
    print(f"{'=' * 60}")

    with ProtonContext("fftconv"):
        for batch, heads in [(1, 128), (2, 64), (4, 32)]:
            (
                u_f16,
                kf_real,
                kf_imag,
                f_real,
                f_imag,
                finv_real,
                finv_imag,
                tw_real,
                tw_imag,
                twinv_real,
                twinv_imag,
                _,
                _,
            ) = fftconv.prepare_data(batch, heads)

            o_np = np.zeros((batch, heads, fftconv.N1, fftconv.N1), dtype=np.float16)
            x_tvm = tvm.runtime.tensor(u_f16.cpu().numpy(), DEV)
            kf_real_tvm = tvm.runtime.tensor(kf_real.cpu().numpy(), DEV)
            kf_imag_tvm = tvm.runtime.tensor(kf_imag.cpu().numpy(), DEV)
            f_real_tvm = tvm.runtime.tensor(f_real.cpu().numpy(), DEV)
            f_imag_tvm = tvm.runtime.tensor(f_imag.cpu().numpy(), DEV)
            finv_real_tvm = tvm.runtime.tensor(finv_real.cpu().numpy(), DEV)
            finv_imag_tvm = tvm.runtime.tensor(finv_imag.cpu().numpy(), DEV)
            tw_real_tvm = tvm.runtime.tensor(tw_real.cpu().numpy(), DEV)
            tw_imag_tvm = tvm.runtime.tensor(tw_imag.cpu().numpy(), DEV)
            twinv_real_tvm = tvm.runtime.tensor(twinv_real.cpu().numpy(), DEV)
            twinv_imag_tvm = tvm.runtime.tensor(twinv_imag.cpu().numpy(), DEV)
            o_tvm = tvm.runtime.tensor(o_np, DEV)

            func = lambda: mod(  # noqa: E731
                x_tvm,
                kf_real_tvm,
                kf_imag_tvm,
                f_real_tvm,
                f_imag_tvm,
                finv_real_tvm,
                finv_imag_tvm,
                tw_real_tvm,
                tw_imag_tvm,
                twinv_real_tvm,
                twinv_imag_tvm,
                o_tvm,
            )
            ms = bench(func, warmup=100, repeat=300, proton_name=f"fftconv_B{batch}_H{heads}")
            tflops = flops(ms, batch, heads) / 1e12
            print(f"  B={batch:>2d}, H={heads:>4d}: {tflops:.2f} TFLOPS, {ms:.3f} ms")

    print()


if __name__ == "__main__":
    test_fftconv(2, 4)
    bench_fftconv()
