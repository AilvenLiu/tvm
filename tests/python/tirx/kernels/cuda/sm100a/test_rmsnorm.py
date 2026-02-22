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

import pytest
import torch

import tvm
from tvm.script import tirx as Tx
from tvm.tirx.bench.utils import ProtonContext, bench

EPS = 1e-6
F16_BYTES = 2
F32_BYTES = 4
SM_COUNT = 148


def ceildiv(a, b):
    return (a + b - 1) // b


def find_power_of_two(n):
    assert n > 0 and (n & (n - 1)) == 0
    return n.bit_length() - 1


def prepare_data(hidden_size, batch_size):
    x = torch.randn(batch_size, hidden_size, dtype=torch.float16, device="cuda")
    weight = torch.randn(hidden_size, dtype=torch.float16, device="cuda")
    return x, weight


def get_rmsnorm_kernel(hidden_size):
    vec_size = math.gcd(16 // F16_BYTES, hidden_size)
    block_size = min(256, hidden_size // vec_size)
    bdx = 32
    bdy = ceildiv(block_size, 32)
    smem_size = (bdy + hidden_size) * F32_BYTES

    # fmt: off
    @Tx.prim_func(tirx=True)
    def rmsnorm(input_ptr: Tx.handle, weight_ptr: Tx.handle, out_ptr: Tx.handle):
        batch_size = Tx.int32()
        input_global = Tx.match_buffer(input_ptr, [batch_size, hidden_size], "float16", scope="global")
        weight_global = Tx.match_buffer(weight_ptr, [hidden_size], "float16", scope="global")
        out_global = Tx.match_buffer(out_ptr, [batch_size, hidden_size], "float16", scope="global")

        with Tx.kernel():
            bx = Tx.cta_id([SM_COUNT], parent="kernel")
            tx, ty = Tx.thread_id([bdx, bdy], parent="cta")
            thread_id = Tx.meta_var(ty * bdx + tx)

            with Tx.cta():
                pool = Tx.PoolAllocator()
                x_smem = pool.alloc([hidden_size], "float32")
                sum_sq_smem = pool.alloc([bdy], "float32")
                pool.commit()

                with Tx.thread():
                    input_vec = Tx.alloc_local([vec_size], "float16")
                    weight_vec = Tx.alloc_local([vec_size], "float16")
                    input_vec_f32 = Tx.alloc_local([vec_size], "float32")
                    weight_vec_f32 = Tx.alloc_local([vec_size], "float32")
                    x_vec = Tx.alloc_local([vec_size], "float32")
                    x_tmp = Tx.alloc_local([1], "float32")
                    sum_sq = Tx.alloc_local([1], "float32")
                    rms_norm = Tx.alloc_local([1], "float32")
                    idx = Tx.alloc_local([1], "int32")

                    idx[0] = bx
                    while idx[0] < batch_size:
                        # add & sum square
                        sum_sq[0] = 0.0
                        for ki in Tx.serial(ceildiv(hidden_size, vec_size * bdx * bdy)):
                            for kv in Tx.unroll(vec_size):
                                input_vec[kv] = 0.0
                                x_vec[kv] = 0.0
                            st = Tx.meta_var((ki * bdx * bdy + thread_id) * vec_size)
                            if st < hidden_size:
                                Tx.copy(input_vec[:], input_global[idx[0], st:st + vec_size])
                                Tx.cast(input_vec_f32[:], input_vec[:])
                                for kv in Tx.unroll(vec_size):
                                    x_tmp[0] = input_vec_f32[kv]
                                    sum_sq[0] += x_tmp[0] * x_tmp[0]
                                    x_vec[kv] = x_tmp[0]
                                Tx.copy(x_smem[st:st + vec_size], x_vec[:])

                        # warp reduce sum
                        for kr in Tx.unroll(find_power_of_two(bdx // 2) + 1):
                            sum_sq[0] = sum_sq[0] + Tx.tvm_warp_shuffle_xor(0xFFFFFFFF, sum_sq[0], (bdx // 2) >> kr, 32, 32)
                        sum_sq_smem[ty] = sum_sq[0]
                        Tx.ptx.bar.sync(1, bdx * bdy)
                        Tx.ptx.fence.proxy_async("shared::cta")
                        # reduce sum through different warps
                        if ty == 0:
                            if tx < bdy:
                                sum_sq[0] = sum_sq_smem[tx]
                            else:
                                sum_sq[0] = 0.0
                            for kr in Tx.unroll(find_power_of_two(bdx // 2) + 1):
                                sum_sq[0] = sum_sq[0] + Tx.tvm_warp_shuffle_xor(0xFFFFFFFF, sum_sq[0], (bdx // 2) >> kr, 32, 32)
                            sum_sq_smem[0] = sum_sq[0]
                        Tx.ptx.bar.sync(1, bdx * bdy)
                        Tx.ptx.fence.proxy_async("shared::cta")
                        # rms norm
                        rms_norm[0] = Tx.rsqrt(sum_sq_smem[0] / hidden_size + EPS)

                        # handle the weight
                        for ki in Tx.serial(ceildiv(hidden_size, vec_size * bdx * bdy)):
                            for kv in Tx.unroll(vec_size):
                                input_vec[kv] = 0.0
                                weight_vec_f32[kv] = 0.0
                                x_vec[kv] = 0.0
                            st = Tx.meta_var((ki * bdx * bdy + thread_id) * vec_size)
                            if st < hidden_size:
                                Tx.copy(weight_vec[:], weight_global[st:st + vec_size])
                                Tx.copy(x_vec[:], x_smem[st:st + vec_size])
                                Tx.cast(weight_vec_f32[:], weight_vec[:])
                            for kv in Tx.unroll(vec_size):
                                input_vec_f32[kv] = x_vec[kv] * rms_norm[0] * weight_vec_f32[kv]
                            if st < hidden_size:
                                Tx.cast(input_vec[:], input_vec_f32[:])
                                Tx.copy(out_global[idx[0], st:st + vec_size], input_vec[:])

                        Tx.ptx.bar.sync(1, bdx * bdy)
                        idx[0] += SM_COUNT
    # fmt: on
    return rmsnorm


@pytest.mark.parametrize("hidden_size", [5120, 128])
@pytest.mark.parametrize("batch_size", [4113, 1, 2, 4, 8, 16, 32, 64, 128])
def test_rmsnorm(hidden_size, batch_size):
    def test_dynamic_batch(batch_size, mod):
        x, weight = prepare_data(hidden_size, batch_size)

        def naive():
            x_naive = x.to(torch.float32)
            variance = x_naive.pow(2).mean(dim=-1, keepdim=True)
            x_naive = x_naive * torch.rsqrt(variance + EPS)
            x_naive = (x_naive * weight.float()).to(torch.float16)
            return x_naive.cpu().numpy()

        def flashinfer():
            import flashinfer

            out = torch.empty_like(x)
            func = lambda: flashinfer.norm.rmsnorm(
                x.clone(), weight, EPS, enable_pdl=False, out=out
            )
            ms = bench(func, warmup=10, repeat=30, proton_name="flashinfer")
            print(f"flashinfer time: {ms:.3f} ms")
            return out.cpu().numpy()

        def tir():
            DEV = tvm.cuda(0)
            x_tvm = tvm.runtime.tensor(x.cpu().numpy(), DEV)
            weight_tvm = tvm.runtime.tensor(weight.cpu().numpy(), DEV)
            out_tvm = tvm.runtime.empty((batch_size, hidden_size), dtype="float16", device=DEV)
            func = lambda: mod(
                x_tvm,
                weight_tvm,
                out_tvm,
            )
            ms = bench(func, warmup=10, repeat=30, proton_name="tir")
            print(f"tir time: {ms:.3f} ms")
            mod(x_tvm, weight_tvm, out_tvm)
            return out_tvm.numpy()

        out_naive = naive()
        out_fused = flashinfer()
        out_tir = tir()
        torch.testing.assert_close(out_fused, out_naive, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(out_tir, out_naive, rtol=1e-3, atol=1e-3)

    # compile tir kernel
    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": get_rmsnorm_kernel(hidden_size)})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
        # src = mod_decode_no_split_kv.mod.imports[0].inspect_source()
        # print(src)

    with ProtonContext("rms_norm"):
        test_dynamic_batch(batch_size, mod)


if __name__ == "__main__":
    import itertools

    hidden_size_list = [5120, 128]
    batch_size_list = [1, 2, 4, 8, 16, 32, 64, 128, 4113]
    for hidden_size, batch_size in itertools.product(hidden_size_list, batch_size_list):
        test_rmsnorm(hidden_size, batch_size)
