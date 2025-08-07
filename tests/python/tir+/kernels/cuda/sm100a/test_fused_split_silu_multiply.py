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
import random
import numpy as np
import tvm
from tvm.script import tir as T
from tvm.script import tirp as Tp
from ..utils import bench, ProtonContext

def ceildiv(a, b):
    return (a + b - 1) // b

# Problem config
SEQ_LEN = 128
INTERMEDIATE_SIZE = 14336

# HW config
SM_COUNT = 148
SMEM_SIZE = 232448

# Other
F16_BYTE = 2
F32_BYTE = 4

# Kernel config
VEC_SIZE = math.gcd(16 // F16_BYTE, INTERMEDIATE_SIZE)
THREAD_NUM = min(256, INTERMEDIATE_SIZE // VEC_SIZE)
BDX = 32
BDY = THREAD_NUM // BDX

def perpare_data():
    import torch
    torch.manual_seed(42)
    input_cat = torch.randn(SEQ_LEN, INTERMEDIATE_SIZE * 2).half()
    return input_cat

@T.prim_func(tirp=True)
def fused_split_silu_multiply(input_cat_ptr: T.handle, output_ptr: T.handle):
    input_cat_global = T.match_buffer(input_cat_ptr, [SEQ_LEN, INTERMEDIATE_SIZE * 2], "float16", scope="global", layout="default")
    output_global = T.match_buffer(output_ptr, [SEQ_LEN, INTERMEDIATE_SIZE], "float16", scope="global", layout="default")

    with T.kernel():
        bx = T.cta_id([SM_COUNT], parent="kernel")
        tx, ty = T.thread_id([BDX, BDY], parent="cta")
        thread_id = T.meta_var(ty * BDX + tx)
        
        with T.thread():
            idx = T.alloc_local([1], "int32", layout="default")
            vec1 = T.alloc_local([VEC_SIZE], "float16", layout="default")
            vec2 = T.alloc_local([VEC_SIZE], "float16", layout="default")
            
            idx[0] = bx
            while idx[0] < ceildiv(SEQ_LEN * INTERMEDIATE_SIZE , THREAD_NUM * VEC_SIZE):
                real_idx = T.meta_var((idx[0] * THREAD_NUM + thread_id) * VEC_SIZE)
                token_idx = T.meta_var(real_idx // INTERMEDIATE_SIZE)
                offset_imme = T.meta_var((real_idx % INTERMEDIATE_SIZE) % INTERMEDIATE_SIZE)
                for kv in T.serial(VEC_SIZE):
                    vec1[kv] = input_cat_global[token_idx, offset_imme + kv]
                for kv in T.serial(VEC_SIZE):
                    vec2[kv] = input_cat_global[token_idx, INTERMEDIATE_SIZE + offset_imme + kv]
                for kv in T.serial(VEC_SIZE):
                    vec1[kv] = vec1[kv] * T.sigmoid(vec1[kv]) * vec2[kv]
                for kv in T.serial(VEC_SIZE):
                    output_global[token_idx, offset_imme + kv] = vec1[kv]
                idx[0] += SM_COUNT

@T.prim_func
def fused_split1_silu1_multiply1(p_fastertransformer_gemm_fp16_int514: T.handle, p_output0: T.handle):
    T.func_attr({"tir.is_scheduled": True, "tir.noalias": True})
    seq_len = T.int64()
    fastertransformer_gemm_fp16_int514 = T.match_buffer(p_fastertransformer_gemm_fp16_int514, (T.int64(1), seq_len, T.int64(INTERMEDIATE_SIZE * 2)), "float16")
    T_multiply_intermediate_1 = T.match_buffer(p_output0, (T.int64(1), seq_len, T.int64(INTERMEDIATE_SIZE)), "float16")
    # with T.block("root"):
    for ax0_ax1_fused_0 in T.thread_binding(seq_len * T.int64(14), thread="blockIdx.x"):
        for ax0_ax1_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
            with T.block("T_multiply_1"):
                v0 = T.axis.spatial(seq_len, (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) // T.int64(INTERMEDIATE_SIZE))
                v1 = T.axis.spatial(T.int64(INTERMEDIATE_SIZE), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) % T.int64(INTERMEDIATE_SIZE))
                T.reads(fastertransformer_gemm_fp16_int514[T.int64(0), v0, v1:v1 + T.int64(INTERMEDIATE_SIZE + 1)])
                T.writes(T_multiply_intermediate_1[T.int64(0), v0, v1])
                T_multiply_intermediate_1[T.int64(0), v0, v1] = fastertransformer_gemm_fp16_int514[T.int64(0), v0, v1] * T.sigmoid(fastertransformer_gemm_fp16_int514[T.int64(0), v0, v1]) * fastertransformer_gemm_fp16_int514[T.int64(0), v0, v1 + T.int64(INTERMEDIATE_SIZE)]
    
def test():

    input_cat = perpare_data()

    def naive():
        import torch
        input_cat_naive = input_cat.clone().to("cuda")

        def func():
            return input_cat_naive[..., INTERMEDIATE_SIZE:] * torch.nn.functional.silu(input_cat_naive[..., :INTERMEDIATE_SIZE])

        ms = bench(func, warmup=0, repeat=30, proton_name="naive")
        print(f"torch time: {ms:.3f} ms")

        return func().cpu().numpy()

    def tir():
        DEV = tvm.cuda(0)
        input_cat_tvm = tvm.nd.array(input_cat.clone(), device=DEV)
        output_tvm = tvm.nd.array(np.zeros((SEQ_LEN, INTERMEDIATE_SIZE), dtype=np.float16), device=DEV)
        target = tvm.target.Target("cuda")
        with target:
            mod = tvm.IRModule({"main": fused_split_silu_multiply})
            mod = tvm.compile(mod, target=target, tir_pipeline="tirp")
            func = lambda: mod(input_cat_tvm, output_tvm)
            ms = bench(func, warmup=0, repeat=30, proton_name="tir")
            print(f"TIR time: {ms:.3f} ms")

        return output_tvm.numpy()
    
    def tir2():
        DEV = tvm.cuda(0)
        input_cat_tvm = tvm.nd.array(input_cat.clone().reshape(1, SEQ_LEN, INTERMEDIATE_SIZE * 2), device=DEV)
        output_tvm = tvm.nd.array(np.zeros((1, SEQ_LEN, INTERMEDIATE_SIZE), dtype=np.float16), device=DEV)
        target = tvm.target.Target("cuda")
        with target:
            mod = tvm.IRModule({"main": fused_split1_silu1_multiply1})
            mod = tvm.compile(mod, target=target)
            func = lambda: mod(input_cat_tvm, output_tvm)
            ms = bench(func, warmup=0, repeat=30, proton_name="tir2")
            print(f"TIR time: {ms:.3f} ms")

        return output_tvm.numpy().reshape(SEQ_LEN, INTERMEDIATE_SIZE)

    with ProtonContext("fused_split_silu_multiply"):
        output_naive = naive()
        output_tir = tir()
        output_tir2 = tir2()

    # np.testing.assert_allclose(output_naive, output_tir, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(output_tir, output_tir2, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    test()