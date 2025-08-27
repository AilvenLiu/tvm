from typing import Any, Dict

from tvm.script import tir as T
from tvm.script import tirp as Tp
from tvm.script.ir_builder import IRBuilder

from .common import F16_BYTES, KernelConfig, Tile, ceildiv, find_power_of_two, rsqrt


class AddRMSNormTile(Tile):
    vec_size = 16 // F16_BYTES

    @classmethod
    def class_config_init(cls, problem_config: Dict[str, Any]):
        cls.EPS = problem_config["rms_norm_eps"]
        cls.loop_inner = 2

    # inplace add rms norm
    def __init__(self, input, residual, weight):
        self.input = input
        self.residual = residual
        self.weight = weight
        self.batch_size = input.shape[0]
        self.hidden_size = input.shape[1]
        self.dtype = input.dtype
        self.bdx = 32
        self.bdy = KernelConfig.NUM_THREADS // self.bdx

    def init(self, pool_allocator: Tp.PoolAllocator):
        self.x_smem = pool_allocator.alloc([self.loop_inner * self.hidden_size], "float32").buffer
        self.sum_sq_smem = pool_allocator.alloc([self.loop_inner, self.bdy], "float32").buffer
        self.input_vec = T.alloc_local([self.loop_inner, self.vec_size], "float16")
        self.residual_vec = T.alloc_local([self.loop_inner, self.vec_size], "float16")
        self.weight_vec = T.alloc_local([self.loop_inner, self.vec_size], "float16")
        self.input_vec_f32 = T.alloc_local([self.loop_inner, self.vec_size], "float32")
        self.residual_vec_f32 = T.alloc_local([self.loop_inner, self.vec_size], "float32")
        self.weight_vec_f32 = T.alloc_local([self.loop_inner, self.vec_size], "float32")
        self.x_vec = T.alloc_local([self.loop_inner, self.vec_size], "float32")
        self.x_tmp = T.alloc_local([self.loop_inner, 1], "float32")
        self.sum_sq = T.alloc_local([self.loop_inner, 1], "float32")
        self.rms_norm = T.alloc_local([1], "float32")
        IRBuilder.current().name("input_vec", self.input_vec)
        IRBuilder.current().name("residual_vec", self.residual_vec)
        IRBuilder.current().name("weight_vec", self.weight_vec)
        IRBuilder.current().name("input_vec_f32", self.input_vec_f32)
        IRBuilder.current().name("residual_vec_f32", self.residual_vec_f32)
        IRBuilder.current().name("weight_vec_f32", self.weight_vec_f32)
        IRBuilder.current().name("x_vec", self.x_vec)
        IRBuilder.current().name("x_tmp", self.x_tmp)
        IRBuilder.current().name("sum_sq", self.sum_sq)
        IRBuilder.current().name("rms_norm", self.rms_norm)

    # fmt: off
    @T.macro
    def run(self, m_idx, n_idx, k_idx):
        with T.cta():
            tid = T.thread_id([KernelConfig.NUM_THREADS], parent="cta")
            warp_id_in_cta = T.warp_id([self.bdy], parent="cta")
            lane_id = T.thread_id([self.bdx], parent="warp")
            # add & sum square
            with T.thread():
                for kl in T.unroll(self.loop_inner):
                    self.sum_sq[kl, 0] = 0.0
                self.rms_norm[0] = 0.0
                for ki in T.serial(ceildiv(self.hidden_size, self.vec_size * KernelConfig.NUM_THREADS * self.loop_inner)):
                    for kl in T.unroll(self.loop_inner):
                        for kv in T.unroll(self.vec_size):
                            self.input_vec[kl, kv] = 0.0
                            self.residual_vec[kl, kv] = 0.0
                            self.x_vec[kl, kv] = 0.0

                    st = T.meta_var((ki * KernelConfig.NUM_THREADS + tid) * self.vec_size * self.loop_inner)

                    for kl in T.unroll(self.loop_inner):
                        if st < self.hidden_size - kl * self.vec_size:
                            Tp.copy(self.input_vec[kl, :], self.input[m_idx, st + kl * self.vec_size : st + kl * self.vec_size + self.vec_size])
                            Tp.copy(self.residual_vec[kl, :], self.residual[m_idx, st + kl * self.vec_size : st + kl * self.vec_size + self.vec_size])

                    for kl in T.unroll(self.loop_inner):
                        Tp.cast(self.input_vec_f32[kl, :], self.input_vec[kl, :])
                        Tp.cast(self.residual_vec_f32[kl, :], self.residual_vec[kl, :])
                    for kl in T.unroll(self.loop_inner):
                        for kv in T.unroll(self.vec_size):
                            self.x_tmp[kl, 0] = self.input_vec_f32[kl, kv] + self.residual_vec_f32[kl, kv]
                            self.sum_sq[kl, 0] += self.x_tmp[kl, 0] * self.x_tmp[kl, 0]
                            self.residual_vec[kl, kv] = T.cast(self.x_tmp[kl, 0], "float16")
                            self.x_vec[kl, kv] = self.x_tmp[kl, 0]
                    for kl in T.unroll(self.loop_inner):
                        if st < self.hidden_size - kl * self.vec_size:
                            Tp.copy(self.residual[m_idx, st + kl * self.vec_size : st + kl * self.vec_size + self.vec_size], self.residual_vec[kl, :])
                            Tp.copy(self.x_smem[st + kl * self.vec_size : st + kl * self.vec_size + self.vec_size], self.x_vec[kl, :])

                # warp reduce sum
                for kl in T.unroll(self.loop_inner):
                    for kr in T.unroll(find_power_of_two(self.bdx // 2) + 1):
                        self.sum_sq[kl, 0] = self.sum_sq[kl, 0] + T.tvm_warp_shuffle_xor(
                            0xFFFFFFFF, self.sum_sq[kl, 0], (self.bdx // 2) >> kr, 32, 32
                        )
                for kl in T.unroll(self.loop_inner):
                    self.sum_sq_smem[kl, warp_id_in_cta] = self.sum_sq[kl, 0]
                T.tvm_storage_sync("shared")
                
                # reduce sum through different warps
                if warp_id_in_cta == 0:
                    for kl in T.unroll(self.loop_inner):
                        if lane_id < self.bdy:
                            self.sum_sq[kl, 0] = self.sum_sq_smem[kl, lane_id]
                        else:
                            self.sum_sq[kl, 0] = 0.0
                    for kl in T.unroll(self.loop_inner):
                        for kr in T.unroll(find_power_of_two(self.bdx // 2) + 1):
                            self.sum_sq[kl, 0] = self.sum_sq[kl, 0] + T.tvm_warp_shuffle_xor(
                                0xFFFFFFFF, self.sum_sq[kl, 0], (self.bdx // 2) >> kr, 32, 32
                            )
                    for kl in T.unroll(self.loop_inner):
                        self.sum_sq_smem[kl, 0] = self.sum_sq[kl, 0]
                T.tvm_storage_sync("shared")

                # rms norm
                for kl in T.unroll(self.loop_inner):
                    self.rms_norm[0] += self.sum_sq_smem[kl, 0]
                self.rms_norm[0] = rsqrt(self.rms_norm[0] / self.hidden_size + self.EPS)

                # handle the weight
                for ki in T.serial(ceildiv(self.hidden_size, self.vec_size * KernelConfig.NUM_THREADS)):
                    for kl in T.unroll(self.loop_inner):
                        for kv in T.unroll(self.vec_size):
                            self.input_vec[kl, kv] = 0.0
                            self.weight_vec_f32[kl, kv] = 0.0
                            self.x_vec[kl, kv] = 0.0
                    st = T.meta_var((ki * KernelConfig.NUM_THREADS + tid) * self.vec_size * self.loop_inner)
                    for kl in T.unroll(self.loop_inner):
                        if st < self.hidden_size - kl * self.vec_size:
                            Tp.copy(self.weight_vec[kl, :], self.weight[st + kl * self.vec_size : st + kl * self.vec_size + self.vec_size])
                            Tp.copy(self.x_vec[kl, :], self.x_smem[st + kl * self.vec_size : st + kl * self.vec_size + self.vec_size])
                    for kl in T.unroll(self.loop_inner):
                        Tp.cast(self.weight_vec_f32[kl, :], self.weight_vec[kl, :])
                    for kl in T.unroll(self.loop_inner):
                        for kv in T.unroll(self.vec_size):
                            self.input_vec_f32[kl, kv] = self.x_vec[kl, kv] * self.rms_norm[0] * self.weight_vec_f32[kl, kv]
                    for kl in T.unroll(self.loop_inner):
                        Tp.cast(self.input_vec[kl, :], self.input_vec_f32[kl, :])
                    for kl in T.unroll(self.loop_inner):
                        if st < self.hidden_size - kl * self.vec_size:
                            Tp.copy(self.input[m_idx, st + kl * self.vec_size : st + kl * self.vec_size + self.vec_size], self.input_vec[kl, :])
    # fmt: on
