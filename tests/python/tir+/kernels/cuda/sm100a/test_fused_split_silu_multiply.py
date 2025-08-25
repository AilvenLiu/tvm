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
from tvm.tirp.op_schedule import EventImpl
from tvm.tirp.bench.utils import ProtonContext, bench
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


def get_fused_split_silu_multiply_kernel_cp_sync(out_dim):
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


def get_fused_split_silu_multiply_kernel_cp_async(out_dim):
    INTERMEDIATE_SIZE = out_dim

    # Kernel config
    VEC_SIZE = math.gcd(16 // F16_BYTE, INTERMEDIATE_SIZE)
    THREAD_NUM = min(256, INTERMEDIATE_SIZE // VEC_SIZE)

    PIPE_DEPTH = 10

    # fmt: off
    @T.prim_func(tirp=True)
    def fused_split_silu_multiply(input_cat_ptr: T.handle, output_ptr: T.handle):
        batch_size = T.int32()

        input_cat_global = T.match_buffer(input_cat_ptr, [batch_size, INTERMEDIATE_SIZE * 2], "float16", scope="global", layout="default")    
        output_global = T.match_buffer(output_ptr, [batch_size, INTERMEDIATE_SIZE], "float16", scope="global", layout="default")

        with T.kernel():
            bx = T.cta_id([SM_COUNT], parent="kernel")
            tx = T.thread_id([THREAD_NUM], parent="cta")
            
            with T.thread():
                idx = T.alloc_local([1], "int32", layout="default")
                shared_buf = T.alloc_buffer([PIPE_DEPTH, 2, THREAD_NUM, VEC_SIZE], "float16", layout="default", scope="shared.dyn")
                vec1 = T.alloc_local([VEC_SIZE], "float16", layout="default")
                vec2 = T.alloc_local([VEC_SIZE], "float16", layout="default")
                evt = Tp.alloc_bulk_group_event(EventImpl.kCpAsync)
                idx[0] = 0
                real_idx = T.meta_var(idx[0] * SM_COUNT * THREAD_NUM + bx * THREAD_NUM + tx)
                while idx[0] < PIPE_DEPTH - 1:
                    intermediate_idx = T.meta_var((real_idx * VEC_SIZE) % INTERMEDIATE_SIZE)
                    batch_idx = T.meta_var((real_idx * VEC_SIZE) // INTERMEDIATE_SIZE)
                    if real_idx * VEC_SIZE < batch_size * INTERMEDIATE_SIZE:
                        Tp.copy_async(shared_buf[idx[0], 0, tx, :], input_cat_global[batch_idx, intermediate_idx:intermediate_idx + VEC_SIZE], evt, schedule_config={"vec_len": VEC_SIZE})
                        Tp.copy_async(shared_buf[idx[0], 1, tx, :], input_cat_global[batch_idx, INTERMEDIATE_SIZE + intermediate_idx:INTERMEDIATE_SIZE + intermediate_idx + VEC_SIZE], evt, schedule_config={"vec_len": VEC_SIZE})
                    evt.commit()
                    idx[0] += 1
                
                idx[0] = 0
                while real_idx * VEC_SIZE < batch_size * INTERMEDIATE_SIZE:
                    intermediate_idx = T.meta_var((real_idx * VEC_SIZE) % INTERMEDIATE_SIZE)
                    batch_idx = T.meta_var((real_idx * VEC_SIZE) // INTERMEDIATE_SIZE)
                    idx_to_prefetch = T.meta_var(idx[0] + PIPE_DEPTH - 1)
                    real_idx_to_prefetch = T.meta_var(idx_to_prefetch * SM_COUNT * THREAD_NUM + bx * THREAD_NUM + tx)
                    intermediate_idx_to_prefetch = T.meta_var((real_idx_to_prefetch * VEC_SIZE) % INTERMEDIATE_SIZE)
                    batch_idx_to_prefetch = T.meta_var((real_idx_to_prefetch * VEC_SIZE) // INTERMEDIATE_SIZE)
                    if real_idx_to_prefetch * VEC_SIZE < batch_size * INTERMEDIATE_SIZE:
                        Tp.copy_async(shared_buf[T.truncmod(idx[0] + PIPE_DEPTH - 1, PIPE_DEPTH), 0, tx, :], input_cat_global[batch_idx_to_prefetch, intermediate_idx_to_prefetch:intermediate_idx_to_prefetch + VEC_SIZE], evt, schedule_config={"vec_len": VEC_SIZE})
                        Tp.copy_async(shared_buf[T.truncmod(idx[0] + PIPE_DEPTH - 1, PIPE_DEPTH), 1, tx, :], input_cat_global[batch_idx_to_prefetch, INTERMEDIATE_SIZE + intermediate_idx_to_prefetch:INTERMEDIATE_SIZE + intermediate_idx_to_prefetch + VEC_SIZE], evt, schedule_config={"vec_len": VEC_SIZE})
                    evt.commit()
                    evt.wait(PIPE_DEPTH - 1)
                    for kv in T.vectorized(VEC_SIZE):
                        vec1[kv] = shared_buf[T.truncmod(idx[0], PIPE_DEPTH), 0, tx, kv]
                    for kv in T.vectorized(VEC_SIZE):
                        vec2[kv] = shared_buf[T.truncmod(idx[0], PIPE_DEPTH), 1, tx, kv]
                    for kv in T.serial(VEC_SIZE):
                        vec1[kv] = vec1[kv] * T.sigmoid(vec1[kv]) * vec2[kv]
                    for kv in T.vectorized(VEC_SIZE):
                        output_global[batch_idx, intermediate_idx + kv] = vec1[kv]
                    idx[0] += 1
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

    def tir_cp_async():
        DEV = tvm.cuda(0)
        input_cat_tvm = tvm.nd.array(input_cat.clone(), device=DEV)
        output_tvm = tvm.nd.empty((batch_size, out_dim), dtype="float16", device=DEV)
        target = tvm.target.Target("cuda")
        with target:
            mod = tvm.IRModule({"main": get_fused_split_silu_multiply_kernel_cp_async(out_dim)})
            mod = tvm.compile(mod, target=target, tir_pipeline="tirp")
            func = lambda: mod(input_cat_tvm, output_tvm)
            ms = bench(func, warmup=0, repeat=30, proton_name="tir_cp_async")
            print(f"TIR time: {ms:.3f} ms")

        return output_tvm.numpy()
    
    def tir_cp_sync():
        DEV = tvm.cuda(0)
        input_cat_tvm = tvm.nd.array(input_cat.clone(), device=DEV)
        output_tvm = tvm.nd.empty((batch_size, out_dim), dtype="float16", device=DEV)
        target = tvm.target.Target("cuda")
        with target:
            mod = tvm.IRModule({"main": get_fused_split_silu_multiply_kernel_cp_sync(out_dim)})
            mod = tvm.compile(mod, target=target, tir_pipeline="tirp")
            func = lambda: mod(input_cat_tvm, output_tvm)
            ms = bench(func, warmup=0, repeat=30, proton_name="tir_cp_sync")
            print(f"TIR time: {ms:.3f} ms")
        return output_tvm.numpy()

    def tir_old():
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
            ms = bench(func, warmup=0, repeat=30, proton_name="tir_old")
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
        output_tir_cp_async = tir_cp_async()
        output_tir_cp_sync = tir_cp_sync()
        output_tir_old = tir_old()
        output_flashinfer = flashinfer()

    np.testing.assert_allclose(output_naive, output_tir_cp_async, rtol=5e-3, atol=5e-3)
    np.testing.assert_allclose(output_tir_cp_async, output_tir_cp_sync, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(output_tir_cp_async, output_tir_old, rtol=5e-3, atol=5e-3)
    np.testing.assert_allclose(output_tir_cp_async, output_flashinfer, rtol=5e-3, atol=5e-3)


if __name__ == "__main__":
    for batch_size in [1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128]:
        test(batch_size)
