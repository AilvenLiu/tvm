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

import tvm
from tvm.script import tirx as Tx
from tvm.tirx.bench.utils import ProtonContext, bench
from tvm.tirx.megakernel.kernels import GemmTile
from tvm.tirx.megakernel.utils.base import SmemManager
from tvm.tirx.megakernel.utils.config import KernelConfig
from tvm.tirx.megakernel.utils.utils import ceildiv, get_source_func


def prepare_data(batch_size, N, K):
    A = torch.randn((batch_size, K), dtype=torch.float16)
    B = torch.randn((N, K), dtype=torch.float16)
    return A, B


class LMHeadLayer:
    def __init__(self, N, K):
        self.N = N
        self.K = K

    @Tx.meta_class
    class TileScheduler:
        def __init__(self, n):
            self.n = n
            self.task_num = ceildiv(n, GemmTile.BLK_N)

        def _alloc_buffer(self):
            self.idx = Tx.alloc_local([1], "int32", name="idx")

        @Tx.inline
        def init(self, bx):
            self._alloc_buffer()
            self.idx[0] = bx

        @Tx.inline
        def next(self):
            self.idx[0] += KernelConfig.SM_NUMBER

        def valid(self):
            return self.idx[0] < self.task_num

    @Tx.inline
    def body(self, A, B, out, blk_m):
        gemm_tile = GemmTile(
            self.N, self.K, "float16", "float16", split_k_factor=1, BLK_M=blk_m, MMA_M=blk_m
        )
        gemm_tile.host_init()
        with Tx.kernel():
            bx = Tx.cta_id([KernelConfig.SM_NUMBER], parent="kernel")
            with Tx.cta():
                buf = Tx.alloc_buffer([KernelConfig.MAX_SMEM_SIZE], "uint8", scope="shared.dyn")
                smem_manager = SmemManager(KernelConfig.MAX_SMEM_SIZE, 16384, buf.data)
                smem_manager.set_tile(gemm_tile)
                gemm_tile.init(smem_manager)
                smem_manager.set_tile(gemm_tile.__class__)
                gemm_tile.class_init(smem_manager)
                scheduler = self.TileScheduler(self.N)
                scheduler.init(bx)
                smem_manager.init()
                while scheduler.valid():
                    smem_manager.enter_tile_runtime(gemm_tile)
                    gemm_tile.run(0, scheduler.idx[0], 0, A, B, out)
                    smem_manager.exit_tile_runtime()
                    scheduler.next()
                gemm_tile.__class__.class_finalize()

    def get_func(self):
        @Tx.prim_func(tirx=True)
        def lm_head_gemm(A_ptr: Tx.handle, B_ptr: Tx.handle, out_ptr: Tx.handle):
            batch_size = Tx.int32()
            A_global = Tx.match_buffer(A_ptr, [batch_size, self.K], dtype="float16", scope="global")
            B_global = Tx.match_buffer(B_ptr, [self.N, self.K], dtype="float16", scope="global")
            out_global = Tx.match_buffer(
                out_ptr, [batch_size, self.N], dtype="float16", scope="global"
            )
            if batch_size <= 32:
                self.body(A_global, B_global, out_global, 32)
            elif batch_size <= 64:
                self.body(A_global, B_global, out_global, 64)
            else:
                self.body(A_global, B_global, out_global, 128)

        return lm_head_gemm


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
