from tvm.script import tirx as Tx

from tvm.tirx.megakernel.utils.base import Tile
from tvm.tirx.megakernel.utils.utils import ceildiv
from tvm.tirx.megakernel.utils.config import KernelConfig, F32_BYTES


class SplitKReduceTile(Tile):

    VEC_SIZE = 16 // F32_BYTES
    N_UNIT = 128
    N_REPEAT = 1

    def __init__(self, M, N, dtype, split_k_factor):
        super().__init__()
        self.N = N
        self.dtype = dtype
        self.split_k_factor = split_k_factor
        self.N_TILE = self.N_UNIT

        self.M = M
        self.M_split = Tx.min(KernelConfig.SM_NUMBER // (self.N // self.N_UNIT), self.M)
        self.M_TILE = ceildiv(self.M, self.M_split)
        self.M_split = ceildiv(self.M, self.M_TILE)

    def _alloc_local(self):
        self.idx = Tx.local_cell("int32", name="idx")
        self.vec_32 = Tx.alloc_local([self.VEC_SIZE], "float32", name="vec_32")
        self.tmp = Tx.alloc_local([self.VEC_SIZE], "float32", name="tmp")
        self.vec_16 = Tx.alloc_local([self.VEC_SIZE], "float16", name="vec_16")

    @Tx.macro
    def run(self, m_idx, n_idx, k_idx, input, output):
        with Tx.cta():
            self._alloc_local()
            tid = Tx.thread_id([KernelConfig.NUM_THREADS], parent="cta")
            self.idx = tid * self.VEC_SIZE
            while (
                self.idx < self.M_TILE * self.N_TILE
                and m_idx * self.M_TILE + self.idx // self.N_TILE < self.M
            ):
                thread_m_idx = Tx.meta_var(m_idx * self.M_TILE + self.idx // self.N_TILE)
                thread_n_idx = Tx.meta_var(n_idx * self.N_TILE + self.idx % self.N_TILE)
                for kv in Tx.unroll(self.VEC_SIZE):
                    self.vec_32[kv] = 0.0
                for kt in Tx.serial(self.split_k_factor):
                    for kv in Tx.vectorized(self.VEC_SIZE):
                        self.tmp[kv] = input[kt, thread_m_idx, thread_n_idx + kv]
                    for kv in Tx.unroll(self.VEC_SIZE):
                        self.vec_32[kv] += self.tmp[kv]
                with Tx.thread():
                    Tx.cast(self.vec_16[:], self.vec_32[:], vec_len=self.VEC_SIZE)
                for kv in Tx.vectorized(self.VEC_SIZE):
                    output[thread_m_idx, thread_n_idx + kv] = self.vec_16[kv]
                self.idx += self.VEC_SIZE * KernelConfig.NUM_THREADS

class MOETopKReduceTile(SplitKReduceTile):
    def __init__(self, M, N, dtype, top_k):
        super().__init__(M, N, dtype, top_k)


    @Tx.macro
    def run(self, m_idx, n_idx, k_idx, input, output):
        with Tx.cta():
            self._alloc_local()
            tid = Tx.thread_id([KernelConfig.NUM_THREADS], parent="cta")
            self.idx = tid * self.VEC_SIZE
            while (
                self.idx < self.M_TILE * self.N_TILE
                and m_idx * self.M_TILE + self.idx // self.N_TILE < self.M
            ):
                thread_m_idx = Tx.meta_var(m_idx * self.M_TILE + self.idx // self.N_TILE)
                thread_n_idx = Tx.meta_var(n_idx * self.N_TILE + self.idx % self.N_TILE)
                for kv in Tx.unroll(self.VEC_SIZE):
                    self.vec_32[kv] = 0.0
                for kt in Tx.serial(self.split_k_factor):
                    for kv in Tx.vectorized(self.VEC_SIZE):
                        self.tmp[kv] = input[thread_m_idx, kt, thread_n_idx + kv]
                    for kv in Tx.unroll(self.VEC_SIZE):
                        self.vec_32[kv] += self.tmp[kv]
                with Tx.thread():
                    Tx.cast(self.vec_16[:], self.vec_32[:], vec_len=self.VEC_SIZE)
                for kv in Tx.vectorized(self.VEC_SIZE):
                    output[thread_m_idx, thread_n_idx + kv] = self.vec_16[kv]
                self.idx += self.VEC_SIZE * KernelConfig.NUM_THREADS
