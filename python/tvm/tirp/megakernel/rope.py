from typing import Any, Dict

import tvm.script.tir as T
import tvm.script.tirp as Tp
from tvm.script.ir_builder import IRBuilder

from .common import *


class RopeTile(Tile):

    # qk_tvm: [batch_size, qo_heads + 2 * kv_heads, head_dim]
    # cos_sin_cache_tvm: [1, head_dim]

    @classmethod
    def class_config_init(cls, problem_config: Dict[str, Any]):
        cls.qo_heads = problem_config["num_attention_heads"]
        cls.kv_heads = problem_config["num_key_value_heads"]
        cls.head_dim = problem_config["head_dim"]
        cls.vec_size = max(16 // F16_BYTES, cls.head_dim // 32)
        cls.bdx = cls.head_dim // cls.vec_size
        cls.bdy = KernelConfig.NUM_THREADS // cls.bdx
        cls.min_bdy = 1

    def __init__(self, qkv_tvm: T.handle, cos_sin_cache_tvm: T.handle, rope_pos_tvm: T.handle):
        self.qkv_global = qkv_tvm
        self.cos_sin_cache_global = cos_sin_cache_tvm
        self.rope_pos_global = rope_pos_tvm
        self.batch_size = qkv_tvm.shape[0]
        self.m_split = T.min(
            ceildiv(KernelConfig.SM_NUMBER, self.qo_heads + 2 * self.kv_heads), self.batch_size
        )
        self.m_tile = ceildiv(self.batch_size, self.m_split)
        self.h_tile = 1
        assert self.qo_heads + 2 * self.kv_heads == qkv_tvm.shape[1]
        assert self.head_dim == qkv_tvm.shape[2]
        assert self.cos_sin_cache_global.shape[1] == self.head_dim
        assert self.rope_pos_global.shape[0] == self.batch_size

    def alloc_buffer(self, pool_allocator: Tp.PoolAllocator):
        self.idx = T.alloc_local([1], "int32", layout="default")
        self.cos = T.alloc_local([self.vec_size], "float32", layout="default")
        self.sin = T.alloc_local([self.vec_size], "float32", layout="default")
        self.qk_vec = T.alloc_local([self.vec_size], "float16", layout="default")
        self.qk_vec32 = T.alloc_local([self.vec_size], "float32", layout="default")
        self.qk_vec32_other = T.alloc_local([self.vec_size], "float32", layout="default")
        self.mask = T.alloc_local([1], "uint32", layout="default")
        IRBuilder.current().name("idx", self.idx)
        IRBuilder.current().name("cos", self.cos)
        IRBuilder.current().name("sin", self.sin)
        IRBuilder.current().name("qk_vec", self.qk_vec)
        IRBuilder.current().name("qk_vec32", self.qk_vec32)
        IRBuilder.current().name("qk_vec32_other", self.qk_vec32_other)
        IRBuilder.current().name("mask", self.mask)

    @T.macro
    def init(self, pool_allocator):
        self.alloc_buffer(pool_allocator)

    @T.macro
    def run(self, m_idx, n_idx, k_idx):
        with T.cta():
            tid = T.thread_id([KernelConfig.NUM_THREADS], parent="cta")
            tx = T.meta_var(tid % self.bdx)
            ty = T.meta_var(tid // self.bdx)
            half_dim = self.head_dim // 2

            with T.thread():
                self.idx[0] = ty

                while (
                    self.idx[0] < self.m_tile * self.h_tile
                    and m_idx * self.m_tile + self.idx[0] // self.h_tile < self.batch_size
                ):
                    batch_idx = T.meta_var(m_idx * self.m_tile + self.idx[0] // self.h_tile)
                    head_idx = T.meta_var(n_idx * self.h_tile + self.idx[0] % self.h_tile)
                    stx = T.meta_var(tx * self.vec_size)
                    cache_stx = T.meta_var(stx % half_dim)
                    pos = T.meta_var(self.rope_pos_global[batch_idx])

                    if batch_idx < self.batch_size and head_idx < self.qo_heads + self.kv_heads:
                        # load cache
                        for kv in T.unroll(self.vec_size):
                            self.cos[kv] = self.cos_sin_cache_global[pos, cache_stx + kv]
                        for kv in T.unroll(self.vec_size):
                            self.sin[kv] = self.cos_sin_cache_global[pos, half_dim + cache_stx + kv]

                        # load qk
                        for kv in T.unroll(self.vec_size):
                            self.qk_vec[kv] = self.qkv_global[batch_idx, head_idx, stx + kv]
                        for kv in T.unroll(self.vec_size // 2):
                            half22float2(
                                T.address_of(self.qk_vec32[kv * 2]),
                                T.address_of(self.qk_vec[kv * 2]),
                            )

                        # shuffle qk value
                        if ty % 2 == 0 and (
                            batch_idx + 1 == self.batch_size
                            or self.idx[0] // self.h_tile + 1 == self.m_tile
                        ):
                            self.mask[0] = 0xFFFF
                        else:
                            self.mask[0] = 0xFFFFFFFF
                        for kv in T.serial(self.vec_size):
                            self.qk_vec32_other[kv] = T.tvm_warp_shuffle_xor(
                                self.mask[0], self.qk_vec32[kv], self.bdx // 2, 32, 32
                            )

                        # compute rope
                        if stx < half_dim:
                            for kv in T.unroll(self.vec_size):
                                self.qk_vec32[kv] = (
                                    self.qk_vec32[kv] * self.cos[kv]
                                    - self.qk_vec32_other[kv] * self.sin[kv]
                                )
                        else:
                            for kv in T.unroll(self.vec_size):
                                self.qk_vec32[kv] = (
                                    self.qk_vec32[kv] * self.cos[kv]
                                    + self.qk_vec32_other[kv] * self.sin[kv]
                                )

                        # store qk
                        for kv in T.unroll(self.vec_size // 2):
                            float22half2(
                                T.address_of(self.qk_vec[kv * 2]),
                                T.address_of(self.qk_vec32[kv * 2]),
                            )
                        for kv in T.unroll(self.vec_size):
                            self.qkv_global[batch_idx, head_idx, stx + kv] = self.qk_vec[kv]

                    self.idx[0] += self.bdy
