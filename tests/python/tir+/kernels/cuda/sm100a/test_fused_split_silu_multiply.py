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
import numpy as np
import tvm
from tvm.script import tir as T
from tvm.script import tirp as Tp
from ..utils import bench, ProtonContext
import pytest


def ceildiv(a, b):
    return (a + b - 1) // b


# HW config
SM_COUNT = 148
SMEM_SIZE = 232448

# Other
F16_BYTE = 2
F32_BYTE = 4


def perpare_data(batch_size, dim):
    import torch

    torch.manual_seed(42)
    input_cat = torch.randn(batch_size, dim * 2).half()
    return input_cat


def get_fused_split_silu_multiply_kernel(out_dim):
    INTERMEDIATE_SIZE = out_dim

    # Kernel config
    VEC_SIZE = math.gcd(16 // F16_BYTE, INTERMEDIATE_SIZE)
    THREAD_NUM = min(256, INTERMEDIATE_SIZE // VEC_SIZE)
    BDX = 32
    BDY = THREAD_NUM // BDX

    # fmt: off
    @T.prim_func(tirp=True)
    def fused_split_silu_multiply(input_cat_ptr: T.handle, output_ptr: T.handle):
        batch_size = T.int32()

        input_cat_global = T.match_buffer(input_cat_ptr, [batch_size, INTERMEDIATE_SIZE * 2], "float16", scope="global", layout="default")    
        output_global = T.match_buffer(output_ptr, [batch_size, INTERMEDIATE_SIZE], "float16", scope="global", layout="default")

        with T.kernel():
            bx = T.cta_id([SM_COUNT], parent="kernel")
            tx, ty = T.thread_id([BDX, BDY], parent="cta")
            thread_id = T.meta_var(ty * BDX + tx)
            
            with T.thread():
                idx = T.alloc_local([1], "int32", layout="default")
                vec1 = T.alloc_local([VEC_SIZE], "float16", layout="default")
                vec2 = T.alloc_local([VEC_SIZE], "float16", layout="default")
                
                idx[0] = bx * BDX * BDY + thread_id
                while idx[0] * VEC_SIZE < batch_size * INTERMEDIATE_SIZE:
                    intermediate_idx = T.meta_var((idx[0] * VEC_SIZE) % INTERMEDIATE_SIZE)
                    batch_idx = T.meta_var((idx[0] * VEC_SIZE) // INTERMEDIATE_SIZE)
                    for kv in T.vectorized(VEC_SIZE):
                        vec1[kv] = input_cat_global[batch_idx, intermediate_idx + kv]
                    for kv in T.vectorized(VEC_SIZE):
                        vec2[kv] = input_cat_global[batch_idx, INTERMEDIATE_SIZE + intermediate_idx + kv]
                    for kv in T.serial(VEC_SIZE):
                        vec1[kv] = vec1[kv] * T.sigmoid(vec1[kv]) * vec2[kv]
                    for kv in T.vectorized(VEC_SIZE):
                        output_global[batch_idx, intermediate_idx + kv] = vec1[kv]
                    idx[0] += SM_COUNT * BDX * BDY
    # fmt: on
    return fused_split_silu_multiply


def get_fused_split1_silu1_multiply1_kernel(out_dim):
    INTERMEDIATE_SIZE = out_dim

    # fmt: off
    @T.prim_func
    def fused_split1_silu1_multiply1(p_fastertransformer_gemm_fp16_int514: T.handle, p_output0: T.handle):
        T.func_attr({"tir.is_scheduled": True, "tir.noalias": True})
        seq_len = T.int64()
        fastertransformer_gemm_fp16_int514 = T.match_buffer(p_fastertransformer_gemm_fp16_int514, (T.int64(1), seq_len, T.int64(INTERMEDIATE_SIZE * 2)), "float16")
        T_multiply_intermediate_1 = T.match_buffer(p_output0, (T.int64(1), seq_len, T.int64(INTERMEDIATE_SIZE)), "float16")
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(seq_len * T.int64(INTERMEDIATE_SIZE // 1024), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_multiply_1"):
                    v0 = T.axis.spatial(seq_len, (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) // T.int64(INTERMEDIATE_SIZE))
                    v1 = T.axis.spatial(T.int64(INTERMEDIATE_SIZE), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) % T.int64(INTERMEDIATE_SIZE))
                    T.reads(fastertransformer_gemm_fp16_int514[T.int64(0), v0, v1:v1 + T.int64(INTERMEDIATE_SIZE + 1)])
                    T.writes(T_multiply_intermediate_1[T.int64(0), v0, v1])
                    T_multiply_intermediate_1[T.int64(0), v0, v1] = fastertransformer_gemm_fp16_int514[T.int64(0), v0, v1] * T.sigmoid(fastertransformer_gemm_fp16_int514[T.int64(0), v0, v1]) * fastertransformer_gemm_fp16_int514[T.int64(0), v0, v1 + T.int64(INTERMEDIATE_SIZE)]

    # fmt: on
    return fused_split1_silu1_multiply1


@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16, 32, 64, 128, 4133])
def test(batch_size):
    out_dim = 25600
    input_cat = perpare_data(batch_size, out_dim)

    def naive():
        import torch

        input_cat_naive = input_cat.clone().to("cuda")

        def func():
            return input_cat_naive[..., out_dim:] * torch.nn.functional.silu(
                input_cat_naive[..., :out_dim]
            )

        ms = bench(func, warmup=0, repeat=30, proton_name="naive")
        print(f"torch time: {ms:.3f} ms")

        return func().cpu().numpy()

    def tir():
        DEV = tvm.cuda(0)
        input_cat_tvm = tvm.nd.array(input_cat.clone(), device=DEV)
        output_tvm = tvm.nd.empty((batch_size, out_dim), dtype="float16", device=DEV)
        target = tvm.target.Target("cuda")
        with target:
            mod = tvm.IRModule({"main": get_fused_split_silu_multiply_kernel(out_dim)})
            mod = tvm.compile(mod, target=target, tir_pipeline="tirp")
            func = lambda: mod(input_cat_tvm, output_tvm)
            ms = bench(func, warmup=0, repeat=30, proton_name="tir")
            print(f"TIR time: {ms:.3f} ms")

        return output_tvm.numpy()

    def tir2():
        DEV = tvm.cuda(0)
        input_cat_tvm = tvm.nd.array(
            input_cat.clone().reshape(1, batch_size, out_dim * 2), device=DEV
        )
        output_tvm = tvm.nd.empty((1, batch_size, out_dim), dtype="float16", device=DEV)
        target = tvm.target.Target("cuda")
        with target:
            mod = tvm.IRModule({"main": get_fused_split1_silu1_multiply1_kernel(out_dim)})
            mod = tvm.compile(mod, target=target)
            func = lambda: mod(input_cat_tvm, output_tvm)
            ms = bench(func, warmup=0, repeat=30, proton_name="tir2")
            print(f"TIR time: {ms:.3f} ms")

        return output_tvm.numpy().reshape(batch_size, out_dim)

    def flashinfer():
        import torch
        import flashinfer

        input_cat_flashinfer = input_cat.clone().to("cuda")
        out = torch.empty((batch_size, out_dim), dtype=torch.float16, device="cuda")
        func = lambda: flashinfer.activation.silu_and_mul(input_cat_flashinfer, out)
        ms = bench(func, warmup=0, repeat=30, proton_name="flashinfer")
        print(f"FlashInfer time: {ms:.3f} ms")
        return out.cpu().numpy()

    with ProtonContext("fused_split_silu_multiply"):
        output_naive = naive()
        output_tir = tir()
        output_tir2 = tir2()
        output_flashinfer = flashinfer()

    np.testing.assert_allclose(output_naive, output_tir, rtol=5e-3, atol=5e-3)
    np.testing.assert_allclose(output_tir, output_tir2, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(output_tir, output_flashinfer, rtol=5e-3, atol=5e-3)


if __name__ == "__main__":
    for batch_size in [32, 128, 4096]:
        test(batch_size)
