from tvm.script import tir as T

from .common import SmemManager, Tile, KernelConfig, F16_BYTES, silu
from .dynamic_scheduler import DynamicTileScheduler

class SiluMultiplyTile(Tile):
    
    TILE_SIZE = 128
    VEC_SIZE = 16 // F16_BYTES

    def __init__(self, batch_size, intermediate_size, dtype):
        super().__init__()
        self.batch_size = batch_size
        self.intermediate_size = intermediate_size
        self.dtype = dtype

    def init(self, smem_manager: SmemManager):
        self.vec1 = T.alloc_local([self.VEC_SIZE], self.dtype, name="vec1")
        self.vec2 = T.alloc_local([self.VEC_SIZE], self.dtype, name="vec2")
        self.idx = T.local_cell("int32", name="idx")
        self.prefetch_round = self.batch_size // 64

    @T.macro
    def run(self, m_idx, n_idx, k_idx, input, output, tile_scheduler):
        with T.cta():
            tid = T.thread_id([KernelConfig.NUM_THREADS], parent="cta")
            self.idx = tid * self.VEC_SIZE
            while (
                self.idx < self.batch_size * self.TILE_SIZE
                - self.prefetch_round * self.VEC_SIZE * KernelConfig.NUM_THREADS
            ):
                token_idx = T.meta_var(self.idx // self.TILE_SIZE)
                offset_imme = T.meta_var(m_idx * self.TILE_SIZE + self.idx % self.TILE_SIZE)
                for kv in T.serial(self.VEC_SIZE):
                    self.vec1[kv] = input[token_idx, offset_imme + kv]
                for kv in T.serial(self.VEC_SIZE):
                    self.vec2[kv] = input[token_idx, self.intermediate_size + offset_imme + kv]
                for kv in T.serial(self.VEC_SIZE):
                    self.vec1[kv] = silu(self.vec1[kv]) * self.vec2[kv]
                for kv in T.serial(self.VEC_SIZE):
                    output[token_idx, offset_imme + kv] = self.vec1[kv]
                self.idx += self.VEC_SIZE * KernelConfig.NUM_THREADS
            if isinstance(tile_scheduler, DynamicTileScheduler):
                if self.prefetch_round > 0:
                    if tid // 32 == tile_scheduler.scheduler_warp:
                        tile_scheduler.prefetch()
            while self.idx < self.batch_size * self.TILE_SIZE:
                token_idx = T.meta_var(self.idx // self.TILE_SIZE)
                offset_imme = T.meta_var(m_idx * self.TILE_SIZE + self.idx % self.TILE_SIZE)
                for kv in T.serial(self.VEC_SIZE):
                    self.vec1[kv] = input[token_idx, offset_imme + kv]
                for kv in T.serial(self.VEC_SIZE):
                    self.vec2[kv] = input[token_idx, self.intermediate_size + offset_imme + kv]
                for kv in T.serial(self.VEC_SIZE):
                    self.vec1[kv] = self.vec1[kv] * T.sigmoid(self.vec1[kv]) * self.vec2[kv]
                for kv in T.serial(self.VEC_SIZE):
                    output[token_idx, offset_imme + kv] = self.vec1[kv]
                self.idx += self.VEC_SIZE * KernelConfig.NUM_THREADS
