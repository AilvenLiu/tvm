import math

import pytest
import torch

import tvm
from tvm.script import tir as T
from tvm.script import tirp as Tp
from tvm.tirp.bench.utils import ProtonContext, bench

EPS = 1e-6
F16_BYTES = 2
F32_BYTES = 4
SM_COUNT = 148
SMEM_SIZE = 232448


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
    @T.prim_func(tirp=True)
    def rmsnorm(input_ptr: T.handle, weight_ptr: T.handle, out_ptr: T.handle):
        batch_size = T.int32()
        input_global = T.match_buffer(input_ptr, [batch_size, hidden_size], "float16", scope="global")
        weight_global = T.match_buffer(weight_ptr, [hidden_size], "float16", scope="global")
        out_global = T.match_buffer(out_ptr, [batch_size, hidden_size], "float16", scope="global")

        with T.kernel():
            bx = T.cta_id([SM_COUNT], parent="kernel")
            tx, ty = T.thread_id([bdx, bdy], parent="cta")
            thread_id = T.meta_var(ty * bdx + tx)

            with T.cta():
                buf = T.alloc_buffer([SMEM_SIZE], "uint8", scope="shared.dyn")
                pool = T.meta_var(Tp.PoolAllocator(buf.data))

                x_smem = pool.alloc([hidden_size], "float32")
                sum_sq_smem = pool.alloc([bdy], "float32")

                with T.thread():
                    input_vec = T.alloc_local([vec_size], "float16")
                    weight_vec = T.alloc_local([vec_size], "float16")
                    input_vec_f32 = T.alloc_local([vec_size], "float32")
                    weight_vec_f32 = T.alloc_local([vec_size], "float32")
                    x_vec = T.alloc_local([vec_size], "float32")
                    x_tmp = T.alloc_local([1], "float32")
                    sum_sq = T.alloc_local([1], "float32")
                    rms_norm = T.alloc_local([1], "float32")
                    idx = T.alloc_local([1], "int32")

                    idx[0] = bx
                    while idx[0] < batch_size:
                        # add & sum square
                        sum_sq[0] = 0.0
                        for ki in T.serial(ceildiv(hidden_size, vec_size * bdx * bdy)):
                            for kv in T.unroll(vec_size):
                                input_vec[kv] = 0.0
                                x_vec[kv] = 0.0
                            st = T.meta_var((ki * bdx * bdy + thread_id) * vec_size)
                            if st < hidden_size:
                                Tp.copy(input_vec[:], input_global[idx[0], st:st + vec_size])
                                Tp.cast(input_vec_f32[:], input_vec[:])
                                for kv in T.unroll(vec_size):
                                    x_tmp[0] = input_vec_f32[kv]
                                    sum_sq[0] += x_tmp[0] * x_tmp[0]
                                    x_vec[kv] = x_tmp[0]
                                Tp.copy(x_smem[st:st + vec_size], x_vec[:])

                        # warp reduce sum
                        for kr in T.unroll(find_power_of_two(bdx // 2) + 1):
                            sum_sq[0] = sum_sq[0] + T.tvm_warp_shuffle_xor(0xFFFFFFFF, sum_sq[0], (bdx // 2) >> kr, 32, 32)
                        sum_sq_smem[ty] = sum_sq[0]
                        T.ptx.bar.sync(1, bdx * bdy)
                        T.ptx.fence.proxy("shared")
                        # reduce sum through different warps
                        if ty == 0:
                            if tx < bdy:
                                sum_sq[0] = sum_sq_smem[tx]
                            else:
                                sum_sq[0] = 0.0
                            for kr in T.unroll(find_power_of_two(bdx // 2) + 1):
                                sum_sq[0] = sum_sq[0] + T.tvm_warp_shuffle_xor(0xFFFFFFFF, sum_sq[0], (bdx // 2) >> kr, 32, 32)
                            sum_sq_smem[0] = sum_sq[0]
                        T.ptx.bar.sync(1, bdx * bdy)
                        T.ptx.fence.proxy("shared")
                        # rms norm
                        rms_norm[0] = T.rsqrt(sum_sq_smem[0] / hidden_size + EPS)

                        # handle the weight
                        for ki in T.serial(ceildiv(hidden_size, vec_size * bdx * bdy)):
                            for kv in T.unroll(vec_size):
                                input_vec[kv] = 0.0
                                weight_vec_f32[kv] = 0.0
                                x_vec[kv] = 0.0
                            st = T.meta_var((ki * bdx * bdy + thread_id) * vec_size)
                            if st < hidden_size:
                                Tp.copy(weight_vec[:], weight_global[st:st + vec_size])
                                Tp.copy(x_vec[:], x_smem[st:st + vec_size])
                                Tp.cast(weight_vec_f32[:], weight_vec[:])
                            for kv in T.unroll(vec_size):
                                input_vec_f32[kv] = x_vec[kv] * rms_norm[0] * weight_vec_f32[kv]
                            if st < hidden_size:
                                Tp.cast(input_vec[:], input_vec_f32[:])
                                Tp.copy(out_global[idx[0], st:st + vec_size], input_vec[:])

                        T.ptx.bar.sync(1, bdx * bdy)
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
        mod = tvm.compile(mod, target=target, tir_pipeline="tirp")
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
