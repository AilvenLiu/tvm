from typing import Any, Dict

from tvm.script import tir as T
from tvm.script import tirp as Tp
from tvm.script.ir_builder import IRBuilder

from .common import *


class AppendKVTile(Tile):

    # kv_cache_tvm: [max_page_num, 2, kv_heads, page_size, head_dim]
    # qkv_tvm: [batch_size, qo_heads + 2 * kv_heads, head_dim]
    # kv_indptr_tvm: [batch_size + 1]
    # kv_indices_tvm: [total_page_num]
    # kv_last_page_len_tvm: [batch_size]
    # pos_map_tvm: [batch_size]

    @classmethod
    def class_config_init(cls, problem_config: Dict[str, Any]):
        cls.loop_inner = 1
        cls.qo_heads = problem_config["num_attention_heads"]
        cls.kv_heads = problem_config["num_key_value_heads"]
        cls.head_dim = problem_config["head_dim"]
        cls.page_size = problem_config["page_size"]
        cls.vec_size = max(16 // F16_BYTES, cls.head_dim // 32)
        cls.bdx = cls.head_dim // cls.vec_size
        cls.bdy = KernelConfig.NUM_THREADS // cls.bdx
        cls.min_bdy = 1

    def __init__(self, kv_cache_tvm: T.handle, qkv_tvm: T.handle, pos_map_tvm: T.handle):
        self.kv_cache_global = kv_cache_tvm
        self.qkv_global = qkv_tvm
        self.pos_map_global = pos_map_tvm
        self.batch_size = qkv_tvm.shape[0]
        self.m_split = T.min(
            ceildiv(KernelConfig.SM_NUMBER, self.qo_heads + 2 * self.kv_heads), self.batch_size
        )
        self.m_tile = ceildiv(self.batch_size, self.m_split)
        self.m_split = ceildiv(self.batch_size, self.m_tile)
        self.h_tile = 1
        assert kv_cache_tvm.shape[1] == 2
        assert kv_cache_tvm.shape[2] == self.kv_heads
        assert kv_cache_tvm.shape[3] == self.page_size
        assert kv_cache_tvm.shape[4] == self.head_dim
        assert qkv_tvm.shape[1] == self.qo_heads + 2 * self.kv_heads
        assert qkv_tvm.shape[2] == self.head_dim
        assert pos_map_tvm.shape[0] == self.batch_size

    def alloc_buffer(self, pool_allocator: Tp.PoolAllocator):
        self.idx = T.alloc_local([1], "int32")
        self.pos = T.alloc_local([1], "int32")
        self.vec = T.alloc_local([self.vec_size], "float16")
        IRBuilder.current().name("idx", self.idx)
        IRBuilder.current().name("pos", self.pos)
        IRBuilder.current().name("vec", self.vec)

    @T.macro
    def init(self, pool_allocator: Tp.PoolAllocator):
        self.alloc_buffer(pool_allocator)

    @T.macro
    def run(self, m_idx, n_idx, k_idx):
        with T.cta():
            tid = T.thread_id([KernelConfig.NUM_THREADS], parent="cta")
            tx = T.meta_var(tid % self.bdx)
            ty = T.meta_var(tid // self.bdx)
            stx = T.meta_var(tx * self.vec_size)
            with T.thread():
                self.idx[0] = ty

                while (
                    self.idx[0] < self.m_tile * self.h_tile
                    and m_idx * self.m_tile + self.idx[0] // self.h_tile < self.batch_size
                ):
                    batch_idx = T.meta_var(m_idx * self.m_tile + self.idx[0] // self.h_tile)
                    head_idx = T.meta_var(n_idx * self.h_tile + self.idx[0] % self.h_tile)
                    if batch_idx < self.batch_size and head_idx < self.kv_heads:

                        self.pos[0] = T.cuda.ldg(
                            T.address_of(self.pos_map_global[batch_idx]), "int32"
                        )
                        page_id = T.meta_var(self.pos[0] // self.page_size)
                        offset = T.meta_var(self.pos[0] % self.page_size)
                        for vec in T.vectorized(self.vec_size):
                            self.vec[vec] = self.qkv_global[
                                batch_idx,
                                self.qo_heads + k_idx * self.kv_heads + head_idx,
                                stx + vec,
                            ]
                        for vec in T.vectorized(self.vec_size):
                            self.kv_cache_global[page_id, k_idx, head_idx, offset, stx + vec] = (
                                self.vec[vec]
                            )

                    self.idx[0] += self.bdy
