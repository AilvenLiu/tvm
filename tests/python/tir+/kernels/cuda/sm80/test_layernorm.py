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
import numpy as np

import tvm
import tvm.testing
from tvm.script import tir as T
from tvm.script import tirp as Tp
from tvm.tir.async_structs import CopyPipeline
from ..utils import bench, ProtonContext


@tvm.testing.requires_cuda_compute_version(8)
def test_layernorm():
    DEV = tvm.cuda(0)
    target = tvm.target.Target("nvidia/nvidia-h100")

    f16_bytes = 2
    bf16_bytes = 2
    f32_bytes = 4
    dtype = "float16"  # doesn't support bfloat16 yet
    ATTN_B, ATTN_N, ATTN_D = 4, 1024, 1024  # attention batch size, seq length, feature dim
    PIPELINE_DEPTH = 2
    NUM_WORKERS = 2
    N_PER_TILE = 2

    # NOTE 1: this kernel is applied on attention output:
    # Step 1. compute dropout(inp) - omitted in inference
    # Step 2. compute x = dropout(inp) + inp_resid (residual connection)
    #         write x to out_resid
    # Step 3. compute layernorm(x)
    #         layernorm(x) = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta
    #         write result to out

    # NOTE 2: this kernel follows CTA-level prog style, where we assume all warps
    #         in a CTA participate in each op.

    # fmt: off
    @T.prim_func(tirp=True)
    def layernorm(inp_ptr: T.handle, inp_resid_ptr: T.handle, norm_weight_ptr: T.handle, norm_bias_ptr: T.handle,
                  out_ptr: T.handle, out_resid_ptr: T.handle) -> None:
        inp = T.match_buffer(inp_ptr, (ATTN_B, 1, ATTN_N, ATTN_D), dtype, scope="global", layout=T.TileLayout.from_tuple((ATTN_B, 1, ATTN_N, ATTN_D)))
        out = T.match_buffer(out_ptr, (ATTN_B, 1, ATTN_N, ATTN_D), dtype, scope="global", layout=T.TileLayout.from_tuple((ATTN_B, 1, ATTN_N, ATTN_D)))
        inp_resid = T.match_buffer(inp_resid_ptr, (ATTN_B, 1, ATTN_N, ATTN_D), dtype, scope="global", layout=T.TileLayout.from_tuple((ATTN_B, 1, ATTN_N, ATTN_D)))
        out_resid = T.match_buffer(out_resid_ptr, (ATTN_B, 1, ATTN_N, ATTN_D), dtype, scope="global", layout=T.TileLayout.from_tuple((ATTN_B, 1, ATTN_N, ATTN_D)))
        norm_weight = T.match_buffer(norm_weight_ptr, (ATTN_D,), dtype, scope="global", layout=T.TileLayout.from_tuple((ATTN_D,))) # gamma
        norm_bias = T.match_buffer(norm_bias_ptr, (ATTN_D,), dtype, scope="global", layout=T.TileLayout.from_tuple((ATTN_D,))) # beta

        with T.kernel():
            bx, by = T.cta_id([T.ceildiv(ATTN_N, N_PER_TILE), ATTN_B], parent="kernel")
            tid = T.thread_id([NUM_WORKERS * 32], parent="cta")
            warp_id = T.warp_id([NUM_WORKERS], parent="cta")
            lane_id = T.thread_id([32], parent="warp")

            with T.cta():
                x_smem = T.alloc_buffer([NUM_WORKERS, PIPELINE_DEPTH, ATTN_D], dtype, scope="shared", layout=T.TileLayout.from_tuple((NUM_WORKERS, PIPELINE_DEPTH, ATTN_D)))
                resid_smem = T.alloc_buffer([NUM_WORKERS, PIPELINE_DEPTH, ATTN_D], dtype, scope="shared", layout=T.TileLayout.from_tuple((NUM_WORKERS, PIPELINE_DEPTH, ATTN_D)))
                norm_weight_smem = T.alloc_buffer([ATTN_D,], dtype, scope="shared", layout=T.TileLayout.from_tuple((ATTN_D,)))
                norm_bias_smem = T.alloc_buffer([ATTN_D,], dtype, scope="shared", layout=T.TileLayout.from_tuple((ATTN_D,)))
                mean = T.alloc_buffer([NUM_WORKERS, 1, 1], dtype, scope="shared", layout=T.TileLayout.from_tuple((NUM_WORKERS, 1, 1)), align=8)
                var = T.alloc_buffer([NUM_WORKERS, 1, 1], dtype, scope="shared", layout=T.TileLayout.from_tuple((NUM_WORKERS, 1, 1)), align=8)

                Tp.copy(norm_bias_smem[:], norm_bias[:])
                Tp.copy(norm_weight_smem[:], norm_weight[:])

                # two cp.async
                pipeline = Tp.alloc_copy_pipeline("cta", depth=0, separate_pc=False,
                                                  schedule_config={CopyPipeline.StrategyKind.IMPL: CopyPipeline.Impl.VEC_LOAD})
                pipeline.copy(x_smem[:, 0, :], inp[by, 0, slice(bx * NUM_WORKERS, (bx + 1) * NUM_WORKERS), :])
                pipeline.producer_commit()
                pipeline.copy(resid_smem[:, 0, :], inp_resid[by, 0, slice(bx * NUM_WORKERS, (bx + 1) * NUM_WORKERS), :])
                pipeline.producer_commit()
                T.tvm_storage_sync("shared")

                # main loop
                n_loops = T.meta_var(T.ceildiv(N_PER_TILE, NUM_WORKERS))
                for k in T.serial(n_loops):
                    curr_idx = T.meta_var((k + 0) * NUM_WORKERS)
                    next_idx = T.meta_var((k + 1) * NUM_WORKERS)

                    if k < n_loops - 1:
                        # TODO(@kathy): support pipeline token flip when pipeline is long
                        pipeline.copy(x_smem[:, 1, :], inp[by, 0, slice(bx * NUM_WORKERS + next_idx, (bx + 1) * NUM_WORKERS + next_idx), :])
                        pipeline.producer_commit()
                        pipeline.copy(resid_smem[:, 1, :], inp_resid[by, 0, slice(bx * NUM_WORKERS + next_idx, (bx + 1) * NUM_WORKERS + next_idx), :])
                        pipeline.producer_commit()
                    pipeline.consumer_wait(0)
                    T.tvm_storage_sync("shared")

                    # store residual
                    Tp.add(resid_smem[:, 0, :], resid_smem[:, 0, :], x_smem[:, 0, :])
                    Tp.copy(out_resid[by, 0, slice(bx * NUM_WORKERS + curr_idx, (bx + 1) * NUM_WORKERS + curr_idx), :], resid_smem[:, 0, :])
                    # numerator
                    Tp.sum(mean[:, 0, 0], resid_smem[:, 0, :])
                    Tp.fdiv(mean[:, 0, 0], mean[:, 0, 0], T.float16(ATTN_D))
                    Tp.sub(resid_smem[:, 0, :], resid_smem[:, 0, :], mean[:, 0, 0])
                    # denominator
                    Tp.mul(x_smem[:, 0, :], resid_smem[:, 0, :], resid_smem[:, 0, :])
                    Tp.sum(var[:, 0, 0], x_smem[:, 0, :])
                    Tp.fdiv(var[:, 0, 0], var[:, 0, 0], T.float16(ATTN_D))
                    Tp.add(var[:, 0, 0], var[:, 0, 0], T.float16(1e-5))
                    Tp.sqrt(var[:, 0, 0], var[:, 0, 0])
                    # layernorm
                    Tp.fdiv(resid_smem[:, 0, :], resid_smem[:, 0, :], var[:, 0, 0])
                    Tp.mul(resid_smem[:, 0, :], resid_smem[:, 0, :], norm_weight_smem[:])
                    Tp.add(resid_smem[:, 0, :], resid_smem[:, 0, :], norm_bias_smem[:])
                    # store result
                    Tp.copy(out[by, 0, slice(bx * NUM_WORKERS + curr_idx, (bx + 1) * NUM_WORKERS + curr_idx), :], resid_smem[:, 0, :])
    # fmt: on

    # get inputs
    inp_np = np.random.randn(ATTN_B, 1, ATTN_N, ATTN_D).astype(dtype)
    inp_resid_np = np.random.randn(ATTN_B, 1, ATTN_N, ATTN_D).astype(dtype)
    norm_weight_np = np.random.randn(
        ATTN_D,
    ).astype(dtype)
    norm_bias_np = np.random.randn(
        ATTN_D,
    ).astype(dtype)
    out_np = np.zeros((ATTN_B, 1, ATTN_N, ATTN_D)).astype(dtype)
    out_resid_np = np.zeros((ATTN_B, 1, ATTN_N, ATTN_D)).astype(dtype)
    inp_tvm = tvm.nd.array(inp_np, DEV)
    inp_resid_tvm = tvm.nd.array(inp_resid_np, DEV)
    norm_weight_tvm = tvm.nd.array(norm_weight_np, DEV)
    norm_bias_tvm = tvm.nd.array(norm_bias_np, DEV)
    out_tvm = tvm.nd.array(out_np, DEV)
    out_resid_tvm = tvm.nd.array(out_resid_np, DEV)

    def tir_layernorm():
        with target:
            mod = tvm.IRModule({"main": layernorm})
            mod = tvm.build(mod, target=target, pipeline="tirp")
            func = lambda: mod(
                inp_tvm, inp_resid_tvm, norm_weight_tvm, norm_bias_tvm, out_tvm, out_resid_tvm
            )
            ms = bench(func, warmup=2, repeat=10, proton_name="tir")
            print(f"TIR layernorm time: {ms:.3f} ms")

        return out_tvm.asnumpy(), out_resid_tvm.asnumpy()

    def torch_layernorm():
        # https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/layer_norm_kernel.cu

        import torch
        from torch import nn

        torch_dev = torch.device("cuda")
        inp_torch = torch.tensor(inp_np, device=torch_dev)
        inp_resid_torch = torch.tensor(inp_resid_np, device=torch_dev)
        norm_weight_torch = torch.tensor(norm_weight_np, device=torch_dev)
        norm_bias_torch = torch.tensor(norm_bias_np, device=torch_dev)
        out_torch = torch.zeros((ATTN_B, 1, ATTN_N, ATTN_D), device=torch_dev)
        out_resid_torch = torch.zeros((ATTN_B, 1, ATTN_N, ATTN_D), device=torch_dev)
        ln_func = nn.LayerNorm(
            [
                ATTN_D,
            ],
            device=torch_dev,
            dtype=torch.float16,
        )
        ln_func.weight = nn.Parameter(norm_weight_torch)
        ln_func.bias = nn.Parameter(norm_bias_torch)

        func = lambda: ln_func(inp_torch + inp_resid_torch)
        ms = bench(func, warmup=2, repeat=10, proton_name="torch")
        print(f"Torch time: {ms:.3f} ms")
        out_torch = func().detach().cpu().numpy()
        out_resid_torch = (inp_torch + inp_resid_torch).detach().cpu().numpy()
        return out_torch, out_resid_torch

    with ProtonContext("layernorm"):
        out_tvm, out_resid_tvm = tir_layernorm()
        out_torch, out_resid_torch = torch_layernorm()

    tvm.testing.assert_allclose(out_resid_tvm, out_resid_torch, atol=1e-8)
    tvm.testing.assert_allclose(out_tvm, out_torch, atol=2e-2)


if __name__ == "__main__":
    test_layernorm()
