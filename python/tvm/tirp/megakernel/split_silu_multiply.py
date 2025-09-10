from tvm.script import tir as T
from tvm.script import tirp as Tp
from tvm.tir.event import EventImpl

from .common import SmemManager, Tile, KernelConfig, F16_BYTES, silu
from .dynamic_scheduler import DynamicTileScheduler

class SiluMultiplyTile(Tile):
    
    TILE_SIZE = 128
    VEC_SIZE = 16 // F16_BYTES
    PIPE_DEPTH = 10

    def __init__(self, batch_size, intermediate_size, dtype):
        super().__init__()
        self.batch_size = batch_size
        self.intermediate_size = intermediate_size
        self.dtype = dtype
        
    def _alloc_buffer(self, smem_manager: SmemManager):
        self.smem_manager = smem_manager
        # allocate smem
        self.buf = smem_manager.alloc([self.PIPE_DEPTH, 2, KernelConfig.NUM_THREADS, self.VEC_SIZE], dtype=self.dtype).buffer
        # allocate register
        self.vec1 = T.alloc_local([self.VEC_SIZE], self.dtype, name="vec1")
        self.vec2 = T.alloc_local([self.VEC_SIZE], self.dtype, name="vec2")
        self.idx = T.local_cell("int32", name="idx")

    def init(self, smem_manager: SmemManager):
        self._alloc_buffer(smem_manager)
        self.prefetch_round = self.batch_size // 64

    @T.macro
    def run(self, m_idx, n_idx, k_idx, input, output, tile_scheduler):
        with T.cta():
            warp_id = T.warp_id([KernelConfig.WARP_NUMBER * KernelConfig.WG_NUMBER], parent="cta")
            tid = T.thread_id([KernelConfig.NUM_THREADS], parent="cta")
            lane_id = T.thread_id([32], parent="warp")            
            evt = Tp.alloc_bulk_group_event(EventImpl.kCpAsync)

            with T.thread():              
                self.idx = tid * self.VEC_SIZE
                
                if warp_id == 0:
                    self.smem_manager.wait_all(lane_id)
                T.tvm_storage_sync("shared")
                
                for ki in T.unroll(self.PIPE_DEPTH - 1):
                    token_idx = T.meta_var(self.idx // self.TILE_SIZE)
                    offset_imme = T.meta_var(m_idx * self.TILE_SIZE + self.idx % self.TILE_SIZE)
                    if self.idx < self.batch_size * self.TILE_SIZE:
                        Tp.copy_async(self.buf[ki, 0, tid, :], input[token_idx, offset_imme:offset_imme + self.VEC_SIZE], 
                                      evt, schedule_config={"vec_len": self.VEC_SIZE})
                        Tp.copy_async(self.buf[ki, 1, tid, :], input[token_idx, self.intermediate_size + offset_imme:self.intermediate_size + offset_imme + self.VEC_SIZE], 
                                      evt, schedule_config={"vec_len": self.VEC_SIZE})
                    evt.commit()
                    self.idx += self.VEC_SIZE * KernelConfig.NUM_THREADS
    
                while self.idx < self.batch_size * self.TILE_SIZE + (self.PIPE_DEPTH - 1) * self.VEC_SIZE * KernelConfig.NUM_THREADS:
                    token_idx = T.meta_var(self.idx // self.TILE_SIZE)
                    offset_imme = T.meta_var(m_idx * self.TILE_SIZE + self.idx % self.TILE_SIZE)
                    if self.idx < self.batch_size * self.TILE_SIZE:
                        cp_pipe_idx = T.meta_var((self.idx // (self.VEC_SIZE * KernelConfig.NUM_THREADS)) % self.PIPE_DEPTH)
                        Tp.copy_async(self.buf[cp_pipe_idx, 0, tid, :], input[token_idx, offset_imme:offset_imme + self.VEC_SIZE], 
                                      evt, schedule_config={"vec_len": self.VEC_SIZE})
                        Tp.copy_async(self.buf[cp_pipe_idx, 1, tid, :], input[token_idx, self.intermediate_size + offset_imme:self.intermediate_size + offset_imme + self.VEC_SIZE], 
                                      evt, schedule_config={"vec_len": self.VEC_SIZE})
                    evt.commit()
                    evt.wait(self.PIPE_DEPTH - 1)
                    pipe_idx = T.meta_var((self.idx // (self.VEC_SIZE * KernelConfig.NUM_THREADS) - (self.PIPE_DEPTH - 1)) % self.PIPE_DEPTH)
                    for kv in T.vectorized(self.VEC_SIZE):
                        self.vec1[kv] = self.buf[pipe_idx, 0, tid, kv]
                    for kv in T.vectorized(self.VEC_SIZE):
                        self.vec2[kv] = self.buf[pipe_idx, 1, tid, kv]
                    for kv in T.unroll(self.VEC_SIZE):
                        self.vec1[kv] = self.vec1[kv] * T.sigmoid(self.vec1[kv]) * self.vec2[kv]
                    token_idx_write = T.meta_var((self.idx - (self.PIPE_DEPTH - 1) * self.VEC_SIZE * KernelConfig.NUM_THREADS) // self.TILE_SIZE)
                    offset_imme_write = T.meta_var(m_idx * self.TILE_SIZE + (self.idx - (self.PIPE_DEPTH - 1) * self.VEC_SIZE * KernelConfig.NUM_THREADS) % self.TILE_SIZE)
                    for kv in T.vectorized(self.VEC_SIZE):
                        output[token_idx_write, offset_imme_write + kv] = self.vec1[kv]
                    self.idx += self.VEC_SIZE * KernelConfig.NUM_THREADS
                
                if warp_id == 0:
                    self.smem_manager.arrive_all(lane_id)
                self.smem_manager.advance()
                
                # TODO: prefetch
                # if isinstance(tile_scheduler, DynamicTileScheduler):
                # if self.prefetch_round > 0:
                #     if tid // 32 == tile_scheduler.scheduler_warp:
                #         tile_scheduler.prefetch()
