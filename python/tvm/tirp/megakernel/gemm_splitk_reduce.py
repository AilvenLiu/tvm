from tvm.script import tir as T
from tvm.script.ir_builder import IRBuilder

from .common import F32_BYTES, KernelConfig, Tile, ceildiv, float22half2


class SplitKReduceTile(Tile):

    VEC_SIZE = 16 // F32_BYTES
    N_UNIT = 128
    N_REPEAT = 1

    def __init__(self, input, output):
        self.input = input
        self.output = output
        self.split_k_factor = input.shape[0]
        self.M = input.shape[1]
        self.N = input.shape[2]
        self.dtype = output.dtype
        self.M_split = T.min(ceildiv(KernelConfig.SM_NUMBER, self.N // self.N_UNIT), self.M)
        self.M_TILE = ceildiv(self.M, self.M_split)
        self.M_split = ceildiv(self.M, self.M_TILE)
        self.N_TILE = self.N_UNIT

    def init(self, pool_allocator):
        self.idx = T.local_cell("int32", name="idx")
        self.vec_32 = T.alloc_local([self.VEC_SIZE], "float32", layout="default")
        self.tmp = T.alloc_local([self.VEC_SIZE], "float32", layout="default")
        self.vec_16 = T.alloc_local([self.VEC_SIZE], "float16", layout="default")
        IRBuilder.current().name("vec_32", self.vec_32)
        IRBuilder.current().name("tmp", self.tmp)
        IRBuilder.current().name("vec_16", self.vec_16)

    @T.macro
    def run(self, m_idx, n_idx, k_idx):
        with T.cta():
            tid = T.thread_id([KernelConfig.NUM_THREADS], parent="cta")
            self.idx = tid * self.VEC_SIZE
            while (
                self.idx < self.M_TILE * self.N_TILE
                and m_idx * self.M_TILE + self.idx // self.N_TILE < self.M
            ):
                thread_m_idx = T.meta_var(m_idx * self.M_TILE + self.idx // self.N_TILE)
                thread_n_idx = T.meta_var(n_idx * self.N_TILE + self.idx % self.N_TILE)
                for kv in T.unroll(self.VEC_SIZE):
                    self.vec_32[kv] = 0.0
                for kt in T.serial(self.split_k_factor):
                    for kv in T.vectorized(self.VEC_SIZE):
                        self.tmp[kv] = self.input[kt, thread_m_idx, thread_n_idx + kv]
                    for kv in T.unroll(self.VEC_SIZE):
                        self.vec_32[kv] += self.tmp[kv]
                for kv in T.unroll(self.VEC_SIZE // 2):
                    float22half2(
                        T.address_of(self.vec_16[kv * 2]), T.address_of(self.vec_32[kv * 2])
                    )
                for kv in T.vectorized(self.VEC_SIZE):
                    self.output[thread_m_idx, thread_n_idx + kv] = self.vec_16[kv]
                self.idx += self.VEC_SIZE * KernelConfig.NUM_THREADS
