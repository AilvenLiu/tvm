from typing import Any, Dict

from tvm.script import tir as T
from tvm.script import tirp as Tp
from tvm.script.ir_builder import IRBuilder

from .common import (
    F16_BYTES,
    KernelConfig,
    Tile,
    ceildiv,
    find_power_of_two,
    float22half2,
    half22float2,
    rsqrt,
)


class RMSnormTile(Tile):

    # weight_tvm: [num_heads]
    # qk_tvm: [batch_size, num_heads, head_dim]

    @classmethod
    def class_config_init(cls, problem_config: Dict[str, Any]):
        cls.loop_inner = 1
        cls.rms_norm_eps = problem_config["rms_norm_eps"]
        cls.qo_heads = problem_config["num_attention_heads"]
        cls.kv_heads = problem_config["num_key_value_heads"]
        cls.head_dim = problem_config["head_dim"]
        cls.vec_size = max(16 // F16_BYTES, cls.head_dim // 32)
        cls.bdx = cls.head_dim // cls.vec_size
        cls.bdy = KernelConfig.NUM_THREADS // cls.bdx
        cls.min_bdy = 1

    def __init__(self, q_weight_tvm: T.handle, k_weight_tvm: T.handle, qkv_tvm: T.handle):
        self.q_weight_global = q_weight_tvm
        self.k_weight_global = k_weight_tvm
        self.qkv_global = qkv_tvm
        self.batch_size = qkv_tvm.shape[0]  # dynamic shape
        self.m_split = T.min(
            ceildiv(KernelConfig.SM_NUMBER, self.qo_heads + 2 * self.kv_heads), self.batch_size
        )
        self.m_tile = ceildiv(self.batch_size, self.m_split)
        self.m_split = ceildiv(self.batch_size, self.m_tile)
        self.h_tile = 1
        assert self.qo_heads + 2 * self.kv_heads == qkv_tvm.shape[1]
        assert self.head_dim == qkv_tvm.shape[2]
        assert self.head_dim == q_weight_tvm.shape[0]
        assert self.head_dim == k_weight_tvm.shape[0]

    def alloc_buffer(self, pool_allocator: Tp.PoolAllocator):
        self.idx = T.alloc_local([1], "int32")
        self.input_vec = T.alloc_local([self.vec_size], "float16")
        self.weight_vec = T.alloc_local([self.vec_size], "float16")
        self.input_vec_f32 = T.alloc_local([self.vec_size], "float32")
        self.weight_vec_f32 = T.alloc_local([self.vec_size], "float32")
        self.sum_sq = T.alloc_local([1], "float32")
        self.rms_norm = T.alloc_local([1], "float32")
        self.mask = T.alloc_local([1], "uint32")
        IRBuilder.current().name("idx", self.idx)
        IRBuilder.current().name("input_vec", self.input_vec)
        IRBuilder.current().name("weight_vec", self.weight_vec)
        IRBuilder.current().name("input_vec_f32", self.input_vec_f32)
        IRBuilder.current().name("weight_vec_f32", self.weight_vec_f32)
        IRBuilder.current().name("sum_sq", self.sum_sq)
        IRBuilder.current().name("rms_norm", self.rms_norm)
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

            with T.thread():

                self.idx[0] = ty
                while (
                    self.idx[0] < self.m_tile * self.h_tile
                    and m_idx * self.m_tile + self.idx[0] // self.h_tile < self.batch_size
                ):
                    batch_idx = T.meta_var(m_idx * self.m_tile + self.idx[0] // self.h_tile)
                    head_idx = T.meta_var(n_idx * self.h_tile + self.idx[0] % self.h_tile)
                    st = T.meta_var(tx * self.vec_size)

                    # add & sum square
                    self.sum_sq[0] = 0.0
                    if batch_idx < self.batch_size and head_idx < self.kv_heads + self.qo_heads:
                        for kv in T.unroll(self.vec_size):
                            self.input_vec[kv] = self.qkv_global[batch_idx, head_idx, st + kv]
                        for kv in T.unroll(self.vec_size // 2):
                            half22float2(
                                T.address_of(self.input_vec_f32[kv * 2]),
                                T.address_of(self.input_vec[kv * 2]),
                            )
                        for kv in T.unroll(self.vec_size):
                            self.sum_sq[0] += self.input_vec_f32[kv] * self.input_vec_f32[kv]

                        # warp reduce sum
                        if ty % 2 == 0 and (
                            batch_idx + 1 == self.batch_size
                            or self.idx[0] // self.h_tile + 1 == self.m_tile
                        ):
                            self.mask[0] = 0xFFFF
                        else:
                            self.mask[0] = 0xFFFFFFFF
                        for kr in T.unroll(find_power_of_two(self.bdx // 2) + 1):
                            self.sum_sq[0] = self.sum_sq[0] + T.tvm_warp_shuffle_xor(
                                self.mask[0], self.sum_sq[0], (self.bdx // 2) >> kr, 32, 32
                            )
                        # rms norm
                        self.rms_norm[0] = rsqrt(self.sum_sq[0] / self.head_dim + self.rms_norm_eps)

                        # handle the weight
                        if n_idx * self.h_tile < self.qo_heads:
                            for kv in T.unroll(self.vec_size):
                                self.weight_vec[kv] = self.q_weight_global[st + kv]
                        else:
                            for kv in T.unroll(self.vec_size):
                                self.weight_vec[kv] = self.k_weight_global[st + kv]
                        for kv in T.unroll(self.vec_size // 2):
                            half22float2(
                                T.address_of(self.weight_vec_f32[kv * 2]),
                                T.address_of(self.weight_vec[kv * 2]),
                            )
                        for kv in T.unroll(self.vec_size):
                            self.input_vec_f32[kv] = (
                                self.input_vec_f32[kv] * self.rms_norm[0] * self.weight_vec_f32[kv]
                            )
                        for kv in T.unroll(self.vec_size // 2):
                            float22half2(
                                T.address_of(self.input_vec[kv * 2]),
                                T.address_of(self.input_vec_f32[kv * 2]),
                            )
                        for kv in T.unroll(self.vec_size):
                            self.qkv_global[batch_idx, head_idx, st + kv] = self.input_vec[kv]

                    self.idx[0] += self.bdy
