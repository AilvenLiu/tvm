from typing import Any, Dict

from tvm.script import tir as T
from tvm.script import tirp as Tp
from tvm.script.ir_builder import IRBuilder
from tvm.tir.event import EventImpl

from .common import F32_BYTES, KernelConfig, Tile, ceildiv, exp2


class DecodeMergeTile(Tile):

    @classmethod
    def class_config_init(cls, problem_config: Dict[str, Any]):
        cls.qo_heads = problem_config["num_attention_heads"]
        cls.kv_heads = problem_config["num_key_value_heads"]
        cls.head_dim = problem_config["head_dim"]
        cls.vec_size = max(16 // F32_BYTES, cls.head_dim // 32)
        cls.bdx = cls.head_dim // cls.vec_size
        cls.bdz = 1  # handle bdz qo heads in one task
        assert cls.bdz <= cls.qo_heads // cls.kv_heads
        cls.bdy = KernelConfig.NUM_THREADS // (cls.bdx * cls.bdz)
        assert cls.bdy > 0
        cls.pipe_depth = 1

    def __init__(
        self,
        o_indptr_tvm: T.handle,
        o_tmp_tvm: T.handle,
        o_tvm: T.handle,
        lse_tmp_tvm: T.handle,
        lse_tvm: T.handle,
    ):
        self.o_indptr_global = o_indptr_tvm
        self.o_tmp_global = o_tmp_tvm
        self.o_global = o_tvm
        self.lse_tmp_global = lse_tmp_tvm
        self.lse_global = lse_tvm
        self.batch_size = o_tvm.shape[0]
        self.new_batch_size = o_tmp_tvm.shape[0]
        # assert o_indptr_tvm.shape[0] == self.batch_size + 1
        assert o_tmp_tvm.shape[1] == self.qo_heads
        assert o_tmp_tvm.shape[2] == self.head_dim
        assert o_tvm.shape[1] == self.qo_heads
        assert o_tvm.shape[2] == self.head_dim
        assert lse_tmp_tvm.shape[0] == self.new_batch_size
        assert lse_tmp_tvm.shape[1] == self.qo_heads
        assert lse_tvm.shape[0] == self.batch_size
        assert lse_tvm.shape[1] == self.qo_heads

    def alloc_buffer(self, pool_allocator: Tp.PoolAllocator):
        # allocate the smem
        offset = pool_allocator.offset
        self.o_tmp_smem = pool_allocator.alloc(
            [self.bdz, self.pipe_depth, self.bdy, self.head_dim], "float32", align=16
        ).buffer
        self.lse_tmp_smem_load = pool_allocator.alloc(
            [self.bdz, self.bdy, self.bdx], "float32"
        ).buffer
        self.lse_tmp_smem_use = Tp.reshape(
            self.lse_tmp_smem_load, [self.bdz, self.bdx, self.bdy]
        ).buffer
        pool_allocator.move_base_to(offset)
        self.o_epi_smem = pool_allocator.alloc(
            [self.bdz, self.bdy, self.head_dim], "float32"
        ).buffer
        self.lse_epi_smem = pool_allocator.alloc([self.bdz, self.bdy], "float32").buffer

        # allocate the reg
        self.new_beg_batch_idx = T.alloc_local([1], "int32")
        self.num = T.alloc_local([1], "int32")
        self.tmp = T.alloc_local([self.vec_size], "float16")
        self.o = T.alloc_local([self.vec_size], "float32")
        self.m = T.alloc_local([2], "float32")
        self.d = T.alloc_local([2], "float32")
        self.m_tmp = T.alloc_local([1], "float32")
        self.o_tmp = T.alloc_local([self.vec_size], "float32")
        IRBuilder.current().name("new_beg_batch_idx", self.new_beg_batch_idx)
        IRBuilder.current().name("num", self.num)
        IRBuilder.current().name("tmp", self.tmp)
        IRBuilder.current().name("o", self.o)
        IRBuilder.current().name("m", self.m)
        IRBuilder.current().name("d", self.d)
        IRBuilder.current().name("m_tmp", self.m_tmp)
        IRBuilder.current().name("o_tmp", self.o_tmp)

    @T.macro
    def init(self, pool_allocator: Tp.PoolAllocator):
        self.alloc_buffer(pool_allocator)

    @T.macro
    def run(self, m_idx, n_idx, k_idx):
        with T.cta():
            tid = T.thread_id([KernelConfig.NUM_THREADS], parent="cta")
            tx = T.meta_var(tid % self.bdx)
            ty = T.meta_var((tid // self.bdx) % self.bdy)
            tz = T.meta_var(tid // (self.bdx * self.bdy))

            evt = Tp.alloc_bulk_group_event(EventImpl.kCpAsync)
            tx_start = T.meta_var(tx * self.vec_size)

            with T.thread():
                batch_idx = T.meta_var(m_idx)
                head_idx = T.meta_var(n_idx * self.bdz + tz)

                if batch_idx < self.batch_size and head_idx < self.qo_heads:
                    self.new_beg_batch_idx[0] = self.o_indptr_global[batch_idx]
                    self.num[0] = self.o_indptr_global[batch_idx + 1] - self.new_beg_batch_idx[0]

                    if self.num[0] == 1:
                        if ty == 0:
                            Tp.copy(
                                self.o[:],
                                self.o_tmp_global[
                                    self.new_beg_batch_idx[0],
                                    head_idx,
                                    tx_start : tx_start + self.vec_size,
                                ],
                            )
                            Tp.cast(self.tmp[:], self.o[:])
                            Tp.copy(
                                self.o_global[
                                    batch_idx, head_idx, tx_start : tx_start + self.vec_size
                                ],
                                self.tmp[:],
                            )
                            self.lse_global[batch_idx, head_idx] = self.lse_tmp_global[
                                self.new_beg_batch_idx[0], head_idx
                            ]

                    # pipeline
                    for kp in T.unroll(self.pipe_depth):
                        if kp * self.bdy + ty < self.num[0]:
                            Tp.copy_async(
                                self.o_tmp_smem[tz, kp, ty, tx_start : tx_start + self.vec_size],
                                self.o_tmp_global[
                                    self.new_beg_batch_idx[0] + kp * self.bdy + ty,
                                    head_idx,
                                    tx_start : tx_start + self.vec_size,
                                ],
                                evt,
                                schedule_config={"vec_len": self.vec_size},
                            )
                        evt.commit()

                    # initialize the value
                    self.m[0] = T.float32("-inf")
                    self.d[0] = 1.0
                    for kv in T.unroll(self.vec_size):
                        self.o[kv] = 0.0

                    for ki in T.serial(ceildiv(self.num[0], self.bdy)):
                        if ki % self.bdx == 0:
                            # load lse
                            if ki * self.bdy + ty * self.bdx + tx < self.num[0]:
                                self.lse_tmp_smem_load[tz, ty, tx] = self.lse_tmp_global[
                                    self.new_beg_batch_idx[0] + ki * self.bdy + ty * self.bdx + tx,
                                    head_idx,
                                ]
                            else:
                                self.lse_tmp_smem_load[tz, ty, tx] = 0.0
                            T.ptx.bar.sync(2, KernelConfig.NUM_THREADS)

                        evt.wait(self.pipe_depth - 1)
                        T.ptx.bar.sync(2, KernelConfig.NUM_THREADS)

                        for kv in T.serial(self.vec_size):
                            self.o_tmp[kv] = self.o_tmp_smem[
                                tz, ki % self.pipe_depth, ty, tx * self.vec_size + kv
                            ]
                        if ki * self.bdy + ty < self.num[0]:
                            self.m_tmp[0] = self.lse_tmp_smem_use[tz, ki % self.bdx, ty]
                            m1 = self.m[0]
                            d1 = self.d[0]
                            self.m[0] = T.max(m1, self.m_tmp[0])
                            self.d[0] = d1 * exp2(m1 - self.m[0]) + exp2(self.m_tmp[0] - self.m[0])
                            for kv in T.unroll(self.vec_size):
                                self.o[kv] = self.o[kv] * exp2(m1 - self.m[0]) + self.o_tmp[
                                    kv
                                ] * exp2(self.m_tmp[0] - self.m[0])
                        T.ptx.bar.sync(2, KernelConfig.NUM_THREADS)
                        if (self.pipe_depth + ki) * self.bdy + ty < self.num[0]:
                            Tp.copy_async(
                                self.o_tmp_smem[
                                    tz,
                                    ki % self.pipe_depth,
                                    ty,
                                    tx_start : tx_start + self.vec_size,
                                ],
                                self.o_tmp_global[
                                    self.new_beg_batch_idx[0]
                                    + (ki + self.pipe_depth) * self.bdy
                                    + ty,
                                    head_idx,
                                    tx_start : tx_start + self.vec_size,
                                ],
                                evt,
                                schedule_config={"vec_len": self.vec_size},
                            )
                        evt.commit()
                    evt.wait(0)
                    T.ptx.bar.sync(2, KernelConfig.NUM_THREADS)
                    # normalize
                    for kv in T.unroll(self.vec_size):
                        self.o[kv] = self.o[kv] / self.d[0]

                    # reduce
                    for kv in T.serial(self.vec_size):
                        self.o_epi_smem[tz, ty, tx * self.vec_size + kv] = self.o[kv]
                    self.lse_epi_smem[tz, ty] = self.m[0] + T.log2(self.d[0])
                    self.m[0] = T.float32("-inf")
                    self.d[0] = 1.0
                    for kv in T.serial(self.vec_size):
                        self.o[kv] = 0.0
                    T.ptx.bar.sync(2, KernelConfig.NUM_THREADS)
                    if ty == 0:
                        for ky in T.serial(self.bdy):
                            self.m_tmp[0] = self.lse_epi_smem[tz, ky]
                            for kv in T.serial(self.vec_size):
                                self.o_tmp[kv] = self.o_epi_smem[tz, ky, tx * self.vec_size + kv]
                            self.m[1] = self.m[0]
                            self.d[1] = self.d[0]
                            self.m[0] = T.max(self.m[1], self.m_tmp[0])
                            self.d[0] = self.d[1] * exp2(self.m[1] - self.m[0]) + exp2(
                                self.m_tmp[0] - self.m[0]
                            )
                            for kv in T.unroll(self.vec_size):
                                self.o[kv] = self.o[kv] * exp2(self.m[1] - self.m[0]) + self.o_tmp[
                                    kv
                                ] * exp2(self.m_tmp[0] - self.m[0])

                        for kv in T.unroll(self.vec_size):
                            self.o[kv] = self.o[kv] / self.d[0]

                        # store to global mem
                        Tp.cast(self.tmp[:], self.o[:])
                        Tp.copy(
                            self.o_global[batch_idx, head_idx, tx_start : tx_start + self.vec_size],
                            self.tmp[:],
                        )
                        if tx == 0:
                            self.lse_global[batch_idx, head_idx] = self.m[0] + T.log2(self.d[0])
