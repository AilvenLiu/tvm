from tvm.script import tir as T
from tvm.script import tirp as Tp
from tvm.script.ir_builder import IRBuilder

from .common import F16_BYTES, KernelConfig, Tile, silu
from .dynamic_scheduler import DynamicTileScheduler


class SiluMultiplyTile(Tile):

    TILE_SIZE = 128
    VEC_SIZE = 16 // F16_BYTES

    def __init__(self, input, output):
        self.input = input
        self.output = output
        self.seq_len = input.shape[0]
        self.intermediate_size = output.shape[1]
        assert self.intermediate_size * 2 == input.shape[1]
        self.dtype = input.dtype

    def init(self, pool_allocator):
        self.vec1 = T.alloc_local([self.VEC_SIZE], self.dtype)
        self.vec2 = T.alloc_local([self.VEC_SIZE], self.dtype)
        IRBuilder.current().name("vec1", self.vec1)
        IRBuilder.current().name("vec2", self.vec2)
        self.idx = T.local_cell("int32", name="idx")
        self.prefetch_round = self.seq_len // 64

    @T.macro
    def run(self, m_idx, n_idx, k_idx, tile_scheduler):
        with T.cta():
            tid = T.thread_id([KernelConfig.NUM_THREADS], parent="cta")
            self.idx = tid * self.VEC_SIZE
            while (
                self.idx
                < self.seq_len * self.TILE_SIZE
                - self.prefetch_round * self.VEC_SIZE * KernelConfig.NUM_THREADS
            ):
                token_idx = T.meta_var(self.idx // self.TILE_SIZE)
                offset_imme = T.meta_var(m_idx * self.TILE_SIZE + self.idx % self.TILE_SIZE)
                for kv in T.serial(self.VEC_SIZE):
                    self.vec1[kv] = self.input[token_idx, offset_imme + kv]
                for kv in T.serial(self.VEC_SIZE):
                    self.vec2[kv] = self.input[token_idx, self.intermediate_size + offset_imme + kv]
                for kv in T.serial(self.VEC_SIZE):
                    self.vec1[kv] = silu(self.vec1[kv]) * self.vec2[kv]
                for kv in T.serial(self.VEC_SIZE):
                    self.output[token_idx, offset_imme + kv] = self.vec1[kv]
                self.idx += self.VEC_SIZE * KernelConfig.NUM_THREADS
            if isinstance(tile_scheduler, DynamicTileScheduler):
                if self.prefetch_round > 0:
                    if tid // 32 == tile_scheduler.scheduler_warp:
                        tile_scheduler.prefetch()
            while self.idx < self.seq_len * self.TILE_SIZE:
                token_idx = T.meta_var(self.idx // self.TILE_SIZE)
                offset_imme = T.meta_var(m_idx * self.TILE_SIZE + self.idx % self.TILE_SIZE)
                for kv in T.serial(self.VEC_SIZE):
                    self.vec1[kv] = self.input[token_idx, offset_imme + kv]
                for kv in T.serial(self.VEC_SIZE):
                    self.vec2[kv] = self.input[token_idx, self.intermediate_size + offset_imme + kv]
                for kv in T.serial(self.VEC_SIZE):
                    self.vec1[kv] = self.vec1[kv] * T.sigmoid(self.vec1[kv]) * self.vec2[kv]
                for kv in T.serial(self.VEC_SIZE):
                    self.output[token_idx, offset_imme + kv] = self.vec1[kv]
                self.idx += self.VEC_SIZE * KernelConfig.NUM_THREADS
