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

import argparse

import numpy as np
import torch
from tirx_kernels.megakernel.lm_head import LMHeadLayer, prepare_data

import tvm
from tvm.megakernel.utils.utils import get_source_func
from tvm.tirx.bench import ProtonContext, bench


def test(batch_size, N, K, mod):
    A, B = prepare_data(batch_size, N, K)
    target = tvm.target.Target("cuda")

    def std():
        out = torch.empty((batch_size, N), dtype=torch.float16).to("cuda")
        ms = bench(
            lambda: torch.matmul(A.to("cuda"), B.to("cuda").T, out=out),
            warmup=10,
            repeat=30,
            proton_name="std",
        )
        print(f"std: {ms:.3f} ms")
        return out.cpu().numpy()

    def tir():
        DEV = tvm.cuda(0)
        A_tvm = tvm.runtime.tensor(A.numpy(), device=DEV)
        B_tvm = tvm.runtime.tensor(B.numpy(), device=DEV)
        out_tvm = tvm.runtime.tensor(np.empty((batch_size, N), dtype="float16"), device=DEV)
        with target:
            ms = bench(lambda: mod(A_tvm, B_tvm, out_tvm), warmup=10, repeat=30, proton_name="tir")
        print(f"tir: {ms:.3f} ms")
        return out_tvm.numpy()

    with ProtonContext():
        out_std = std()
        out_tir = tir()
        np.testing.assert_allclose(out_std, out_tir, rtol=1e-3, atol=1e-2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MegaKernel testing script.")
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3_32b",
        choices=["qwen3_32b", "llama3_1b"],
        help="The supporting model.",
    )
    parser.add_argument("--N", type=int, default=128256, help="The N dimension.")
    parser.add_argument("--K", type=int, default=2048, help="The K dimension.")
    parser.add_argument(
        "--batch-size",
        type=int,
        nargs="+",
        default=[1, 3, 7, 15, 31, 63, 127, 128],
        help="The batch size.",
    )
    args = parser.parse_args()

    lm_head = LMHeadLayer(args.N, args.K)
    src, mod = get_source_func(lm_head.get_func())
    for bs in args.batch_size:
        print(f"batch={bs}, N={args.N}, K={args.K}")
        test(bs, args.N, args.K, mod)
        print("Pass!")
