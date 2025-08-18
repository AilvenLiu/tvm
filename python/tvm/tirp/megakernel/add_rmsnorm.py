from tvm.script import tir as T
from tvm.script import tirp as Tp
from tvm.script.ir_builder import IRBuilder

from .common import F16_BYTES, KernelConfig, Tile, ceildiv, find_power_of_two


class AddRMSNormTile(Tile):
    vec_size = 16 // F16_BYTES
    EPS = 1e-6

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

    def init(self, pool_allocator):
        self.x_smem = pool_allocator.alloc([self.hidden_size], "float32", layout="default").buffer
        self.sum_sq_smem = pool_allocator.alloc([self.bdy], "float32", layout="default").buffer
        self.input_vec = T.alloc_local([self.vec_size], "float16", layout="default")
        self.residual_vec = T.alloc_local([self.vec_size], "float16", layout="default")
        self.weight_vec = T.alloc_local([self.vec_size], "float16", layout="default")
        self.input_vec_f32 = T.alloc_local([self.vec_size], "float32", layout="default")
        self.residual_vec_f32 = T.alloc_local([self.vec_size], "float32", layout="default")
        self.weight_vec_f32 = T.alloc_local([self.vec_size], "float32", layout="default")
        self.x_vec = T.alloc_local([self.vec_size], "float32", layout="default")
        self.x_tmp = T.alloc_local([1], "float32", layout="default")
        self.sum_sq = T.alloc_local([1], "float32", layout="default")
        self.rms_norm = T.alloc_local([1], "float32", layout="default")
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

    @T.macro
    def run(self, m_idx, n_idx, k_idx):
        with T.cta():
            tid = T.thread_id([KernelConfig.NUM_THREADS], parent="cta")
            warp_id_in_cta = T.warp_id([self.bdy], parent="cta")
            lane_id = T.thread_id([self.bdx], parent="warp")
            # add & sum square
            self.sum_sq[0] = 0.0
            with T.thread():
                for ki in T.serial(
                    ceildiv(self.hidden_size, self.vec_size * KernelConfig.NUM_THREADS)
                ):
                    for kv in T.unroll(self.vec_size):
                        self.input_vec[kv] = 0.0
                        self.residual_vec[kv] = 0.0
                        self.x_vec[kv] = 0.0
                    st = T.meta_var((ki * KernelConfig.NUM_THREADS + tid) * self.vec_size)
                    if st < self.hidden_size:
                        Tp.copy(self.input_vec[:], self.input[m_idx, st : st + self.vec_size])
                        Tp.copy(self.residual_vec[:], self.residual[m_idx, st : st + self.vec_size])
                        Tp.cast(self.input_vec_f32[:], self.input_vec[:])
                        Tp.cast(self.residual_vec_f32[:], self.residual_vec[:])
                        for kv in T.unroll(self.vec_size):
                            self.x_tmp[0] = self.input_vec_f32[kv] + self.residual_vec_f32[kv]
                            self.sum_sq[0] += self.x_tmp[0] * self.x_tmp[0]
                            self.residual_vec[kv] = T.cast(self.x_tmp[0], "float16")
                            self.x_vec[kv] = self.x_tmp[0]
                        if st < self.hidden_size:
                            Tp.copy(
                                self.residual[m_idx, st : st + self.vec_size], self.residual_vec[:]
                            )
                            Tp.copy(self.x_smem[st : st + self.vec_size], self.x_vec[:])

                # warp reduce sum
                for kr in T.unroll(find_power_of_two(self.bdx // 2) + 1):
                    self.sum_sq[0] = self.sum_sq[0] + T.tvm_warp_shuffle_xor(
                        0xFFFFFFFF, self.sum_sq[0], (self.bdx // 2) >> kr, 32, 32
                    )
                self.sum_sq_smem[warp_id_in_cta] = self.sum_sq[0]
                T.tvm_storage_sync("shared")
                # reduce sum through different warps
                if warp_id_in_cta == 0:
                    if lane_id < self.bdy:
                        self.sum_sq[0] = self.sum_sq_smem[lane_id]
                    else:
                        self.sum_sq[0] = 0.0
                    for kr in T.unroll(find_power_of_two(self.bdx // 2) + 1):
                        self.sum_sq[0] = self.sum_sq[0] + T.tvm_warp_shuffle_xor(
                            0xFFFFFFFF, self.sum_sq[0], (self.bdx // 2) >> kr, 32, 32
                        )
                    self.sum_sq_smem[0] = self.sum_sq[0]
                T.tvm_storage_sync("shared")
                # rms norm
                self.rms_norm[0] = T.rsqrt(self.sum_sq_smem[0] / self.hidden_size + self.EPS)

                # handle the weight
                for ki in T.serial(
                    ceildiv(self.hidden_size, self.vec_size * KernelConfig.NUM_THREADS)
                ):
                    for kv in T.unroll(self.vec_size):
                        self.input_vec[kv] = 0.0
                        self.weight_vec_f32[kv] = 0.0
                        self.x_vec[kv] = 0.0
                    st = T.meta_var((ki * KernelConfig.NUM_THREADS + tid) * self.vec_size)
                    if st < self.hidden_size:
                        Tp.copy(self.weight_vec[:], self.weight[st : st + self.vec_size])
                        Tp.copy(self.x_vec[:], self.x_smem[st : st + self.vec_size])
                        Tp.cast(self.weight_vec_f32[:], self.weight_vec[:])
                    for kv in T.unroll(self.vec_size):
                        self.input_vec_f32[kv] = (
                            self.x_vec[kv] * self.rms_norm[0] * self.weight_vec_f32[kv]
                        )
                    if st < self.hidden_size:
                        Tp.cast(self.input_vec[:], self.input_vec_f32[:])
                        Tp.copy(self.input[m_idx, st : st + self.vec_size], self.input_vec[:])
