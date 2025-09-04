from tvm.script import tir as T
from tvm.script import tirp as Tp
from tvm.script.ir_builder import IRBuilder

from .common import (
    F16_BYTES,
    F32_BYTES,
    F128_BYTES,
    Barriers,
    KernelConfig,
    Tile,
    ceildiv,
    float22half2,
    warp_sync,
)


class BarTMA2MMA(Barriers):

    @T.macro
    def arrive(self, idx, expected_bytes):
        T.ptx.mbarrier.arrive.expect_tx(self.mbar.ptr_to([idx]), expected_bytes)

    @T.macro
    def arrive_only(self, idx):
        T.ptx.mbarrier.arrive(self.mbar.ptr_to([idx]))


class BarMMA2LD(Barriers):

    @T.macro
    def arrive(self, idx):
        T.ptx.tcgen05.commit(self.mbar.ptr_to([idx]), cta_group=KernelConfig.CTA_GROUP)


class BarMMA2TMA(Barriers):

    @T.macro
    def arrive(self, idx):
        T.ptx.tcgen05.commit(self.mbar.ptr_to([idx]), cta_group=KernelConfig.CTA_GROUP)


class BarLD2MMA(Barriers):

    @T.macro
    def arrive(self, idx):
        T.ptx.mbarrier.arrive(self.mbar.ptr_to([idx]), cta_id=0, pred=True)


@T.macro
def trap_when_assert_failed(cond):
    T.cuda.func_call(
        "trap_when_assert_fail",
        cond,
        source_code=f"""
__forceinline__ __device__ void trap_when_assert_fail(bool cond) {{
    do {{
        if (not (cond))
            asm("trap;");
    }} while (0);
}}
    """,
    )


class GemmTile(Tile):
    SMEM_PIPE_DEPTH = 6
    TMEM_PIPE_DEPTH = 2
    MAX_BLK_M, BLK_N, BLK_K = 128, 128, 64
    MMA_N, MMA_K = 128, 16
    EPI_TILE = 32
    TMEM_LD_SIZE = 8
    N_COLS = 512
    SWIZZLE = 3
    SMEM_SIZE = (
        SMEM_PIPE_DEPTH * MAX_BLK_M * BLK_K * F16_BYTES
        + SMEM_PIPE_DEPTH * BLK_N * BLK_K * F16_BYTES
        + TMEM_PIPE_DEPTH * EPI_TILE * MMA_N * F32_BYTES
        + 1024
    )

    assert SMEM_SIZE <= 232448
    assert TMEM_PIPE_DEPTH * MMA_N <= 512

    A_layout = T.ComposeLayout(
        T.SwizzleLayout(3, 3, 3, swizzle_inner=True),
        T.TileLayout(shard=((SMEM_PIPE_DEPTH, MAX_BLK_M, BLK_K), (MAX_BLK_M * BLK_K, BLK_K, 1))),
    )
    B_layout = T.ComposeLayout(
        T.SwizzleLayout(3, 3, 3, swizzle_inner=True),
        T.TileLayout(shard=((SMEM_PIPE_DEPTH, BLK_N, BLK_K), (BLK_N * BLK_K, BLK_K, 1))),
    )
    D_layout = T.TileLayout(
        shard=((TMEM_PIPE_DEPTH, EPI_TILE, MMA_N), (EPI_TILE * MMA_N, MMA_N, 1))
    )

    # idx of current gemm tile (no matter which shape it is)
    tile_idx = None

    def __init__(self, N, K, a_type, b_type, out_type, split_k_factor):
        self.N = N
        self.K = K
        self.a_type = a_type
        self.b_type = b_type
        self.out_type = out_type
        self.split_k_factor = split_k_factor
        self.TILE_K = ceildiv(ceildiv(self.K, self.split_k_factor), self.BLK_K) * self.BLK_K
        self.PIPE_CIRCLE_NUM = (self.TILE_K // self.BLK_K) // self.SMEM_PIPE_DEPTH
        self.PIPE_REMAIN_NUM = (self.TILE_K // self.BLK_K) % self.SMEM_PIPE_DEPTH
    
        
    def _alloc_buffer(self, pool_allocator: Tp.PoolAllocator):
        # alloc shared memory
        self.A_smem = pool_allocator.alloc(
            (self.SMEM_PIPE_DEPTH, self.MAX_BLK_M, self.BLK_K),
            self.a_type,
            layout=self.A_layout,
            align=1024,
        ).buffer
        self.B_smem = pool_allocator.alloc(
            (self.SMEM_PIPE_DEPTH, self.BLK_N, self.BLK_K),
            self.b_type,
            layout=self.B_layout,
            align=1024,
        ).buffer
        self.output_smem = pool_allocator.alloc(
            (self.TMEM_PIPE_DEPTH, self.EPI_TILE, self.MMA_N),
            self.out_type,
            layout=self.D_layout,
            align=1024,
        ).buffer

        # alloc local memory
        self.reg = T.alloc_buffer((self.TMEM_LD_SIZE,), "float32", scope="local", name="reg")
        if self.out_type == "float16":
            self.reg_fp16 = T.alloc_buffer((self.TMEM_LD_SIZE,), self.out_type, scope="local", name="reg_fp16")
        self.tmem_idx = T.local_cell("int32", name="tmem_idx")
        self.tmem_phase = T.local_cell("int32", name="tmem_phase")
        self.stage = T.local_cell("int32", name="stage")


    @classmethod
    def _alloc_buffer_class_member(cls, pool_allocator: Tp.PoolAllocator):
        # alloc shared memory
        cls.tmem_addr = pool_allocator.alloc([1], "uint32").buffer
        cls.tma2mma_bar = BarTMA2MMA(pool_allocator, cls.SMEM_PIPE_DEPTH, True)
        cls.mma2tma_bar = BarMMA2TMA(pool_allocator, cls.SMEM_PIPE_DEPTH, False)
        cls.mma2ld_bar = BarMMA2LD(pool_allocator, cls.TMEM_PIPE_DEPTH, True)
        cls.ld2mma_bar = BarLD2MMA(pool_allocator, cls.TMEM_PIPE_DEPTH, False)
        # alloc local memory
        cls.tile_idx = T.local_cell("int32", "tile_idx")
        cls.phase = T.alloc_buffer((1,), "int32", scope="local", name="phase")

    @classmethod
    @T.macro
    def class_init(cls, pool_allocator):
        cls._alloc_buffer_class_member(pool_allocator)
        cls.tile_idx = 0
        # alloc TMEM
        with T.warp()[0:1]:
            T.ptx.tcgen05.alloc(T.address_of(cls.tmem_addr[0]), n_cols=cls.N_COLS, cta_group=1)
            warp_sync()
        # init mbarrier and phase
        cls.tma2mma_bar.init(1)
        cls.mma2ld_bar.init(1)
        cls.mma2tma_bar.init(1)
        cls.ld2mma_bar.init(KernelConfig.CTA_GROUP * 128)
        cls.phase[0] = 0

        # sync
        T.ptx.fence.proxy("shared")
        T.ptx.fence.mbarrier_init()
        T.tvm_storage_sync("shared")

    @classmethod
    @T.macro
    def class_finalize(cls):
        T.tvm_storage_sync("shared")
        # dealloc TMEM
        with T.warp()[0:1]:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
            T.ptx.tcgen05.dealloc(cls.tmem_addr[0], n_cols=cls.N_COLS, cta_group=1)
        T.tvm_storage_sync("shared")

    @T.macro
    def init(self, pool_allocator):
        self._alloc_buffer(pool_allocator)

    def set_tensor_map(self, A_tensor_map, B_tensor_map, output_tensor_map, A, B, output):
        self.A_tensor_map = A_tensor_map
        self.B_tensor_map = B_tensor_map
        self.output_tensor_map = output_tensor_map
        self.M = A.shape[0]
        self.A = A
        self.B = B
        self.output = output
        assert B.shape[1] == self.K
        assert A.shape[1] == self.K
        assert output.shape[-1] == self.N
        assert self.a_type == A.dtype
        assert self.b_type == B.dtype
        assert self.out_type == output.dtype

    @T.macro
    def host_init(self):
        BLK_M = T.if_then_else(self.M <= 32, 32, T.if_then_else(self.M <= 64, 64, 128))
        T.call_packed("runtime.cuTensorMapEncodeTiled", self.A_tensor_map, self.a_type, 2, self.A.data, 
                      self.K, self.M, self.K * F16_BYTES, self.BLK_K, BLK_M, 1, 1, 0, self.SWIZZLE, 0, 0)
        T.call_packed("runtime.cuTensorMapEncodeTiled", self.B_tensor_map, self.b_type, 2, self.B.data, 
                      self.K, self.N, self.K * F16_BYTES, self.BLK_K, self.BLK_N, 1, 1, 0, self.SWIZZLE, 0, 0)
        if self.split_k_factor > 1:
            T.call_packed("runtime.cuTensorMapEncodeTiled", self.output_tensor_map, self.out_type, 3, self.output.data, 
                          self.N, self.M, self.split_k_factor, self.N * F32_BYTES, self.N * self.M * F32_BYTES, self.MMA_N, self.EPI_TILE, 1, 1, 1, 1, 0, 0, 0, 0)
        else:
            T.call_packed("runtime.cuTensorMapEncodeTiled", self.output_tensor_map, self.out_type, 2, self.output.data, 
                          self.N, self.M, self.N * F16_BYTES, self.MMA_N, self.EPI_TILE, 1, 1, 0, 0, 0, 0)

    @T.macro
    def _run(self, m_idx, n_idx, k_idx, BLK_M, MMA_M):
        with T.cta():
            wg_id = T.warpgroup_id([KernelConfig.WG_NUMBER], parent="cta")
            warp_id = T.warp_id([KernelConfig.WARP_NUMBER], parent="warpgroup")
            lane_id = T.thread_id([32], parent="warp")
            tid = T.thread_id([KernelConfig.NUM_THREADS], parent="cta")

            with T.cta():
                T.block_attr({"tirp.scope_partition": True})
                with T.warpgroup()[1:2]:
                    if warp_id == 3:
                        if T.ptx.elect_sync():
                            # main inner tma loop
                            k_offset = k_idx * self.TILE_K
                            for ko in T.serial(self.PIPE_CIRCLE_NUM):
                                for ks in T.unroll(self.SMEM_PIPE_DEPTH):
                                    # GMEM -> SMEM  (tma)
                                    self.stage = ko * self.SMEM_PIPE_DEPTH + ks
                                    self.mma2tma_bar.wait(ks, self.phase[0])
                                    T.ptx.cp_async.bulk.tensor.g2c(
                                        2,
                                        self.A_smem.ptr_to([ks, 0, 0]),
                                        self.tma2mma_bar.mbar.ptr_to([ks]),
                                        self.A_tensor_map,
                                        self.stage * self.BLK_K + k_offset,
                                        m_idx * BLK_M,
                                        cta_group=KernelConfig.CTA_GROUP,
                                        cache_hint="evict_last",
                                    )
                                    T.ptx.cp_async.bulk.tensor.g2c(
                                        2,
                                        self.B_smem.ptr_to([ks, 0, 0]),
                                        self.tma2mma_bar.mbar.ptr_to([ks]),
                                        self.B_tensor_map,
                                        self.stage * self.BLK_K + k_offset,
                                        n_idx * self.BLK_N,
                                        cta_group=KernelConfig.CTA_GROUP,
                                        cache_hint="evict_first",
                                    )
                                    self.tma2mma_bar.arrive(
                                        ks,
                                        KernelConfig.CTA_GROUP
                                        * self.BLK_K
                                        * (BLK_M + self.BLK_N)
                                        * F16_BYTES,
                                    )
                                self.phase[0] = self.phase[0] ^ 1

                            if self.PIPE_REMAIN_NUM > 0:
                                # last remained loop
                                for ks in T.unroll(self.PIPE_REMAIN_NUM):
                                    # GMEM -> SMEM  (tma)
                                    self.stage = self.PIPE_CIRCLE_NUM * self.SMEM_PIPE_DEPTH + ks
                                    self.mma2tma_bar.wait(ks, self.phase[0])
                                    T.ptx.cp_async.bulk.tensor.g2c(
                                        2,
                                        self.A_smem.ptr_to([ks, 0, 0]),
                                        self.tma2mma_bar.mbar.ptr_to([ks]),
                                        self.A_tensor_map,
                                        self.stage * self.BLK_K + k_offset,
                                        m_idx * BLK_M,
                                        cta_group=KernelConfig.CTA_GROUP,
                                        cache_hint="evict_last",
                                    )
                                    T.ptx.cp_async.bulk.tensor.g2c(
                                        2,
                                        self.B_smem.ptr_to([ks, 0, 0]),
                                        self.tma2mma_bar.mbar.ptr_to([ks]),
                                        self.B_tensor_map,
                                        self.stage * self.BLK_K + k_offset,
                                        n_idx * self.BLK_N,
                                        cta_group=KernelConfig.CTA_GROUP,
                                        cache_hint="evict_first",
                                    )
                                    self.tma2mma_bar.arrive(
                                        ks,
                                        KernelConfig.CTA_GROUP
                                        * self.BLK_K
                                        * (BLK_M + self.BLK_N)
                                        * F16_BYTES,
                                    )
                                # for unaligned cases
                                for ks in T.unroll(self.PIPE_REMAIN_NUM, self.SMEM_PIPE_DEPTH):
                                    self.mma2tma_bar.wait(ks, self.phase[0])
                                    self.tma2mma_bar.arrive_only(ks)

                                self.phase[0] = self.phase[0] ^ 1

                    elif warp_id == 0:
                        descA = T.local_cell("uint64")
                        descB = T.local_cell("uint64")
                        descI = T.local_cell("uint32")
                        T.ptx.tcgen05.encode_instr_descriptor(
                            T.address_of(descI),
                            "float32",
                            self.a_type,
                            self.b_type,
                            self.MMA_N,
                            MMA_M,
                            self.MMA_K,
                            False,
                            False,
                            KernelConfig.CTA_GROUP,
                        )

                        if T.ptx.elect_sync():
                            self.tmem_idx = self.tile_idx % self.TMEM_PIPE_DEPTH
                            self.tmem_phase = (self.tile_idx // self.TMEM_PIPE_DEPTH) & 1

                            # wait for the tmem result to be consumed
                            self.ld2mma_bar.wait(self.tmem_idx, self.tmem_phase)
                            T.ptx.tcgen05.fence.after_thread_sync()

                            # main inner mma loop
                            for ko in T.serial(self.PIPE_CIRCLE_NUM):
                                for ks in T.unroll(self.SMEM_PIPE_DEPTH):

                                    # wait tma and sf-transpose arrival
                                    self.tma2mma_bar.wait(ks, self.phase[0])
                                    T.ptx.tcgen05.fence.after_thread_sync()

                                    # issue mma
                                    for ki in T.unroll(self.BLK_K // self.MMA_K):
                                        T.ptx.tcgen05.encode_matrix_descriptor(
                                            T.address_of(descA),
                                            self.A_smem.ptr_to([ks, 0, ki * self.MMA_K]),
                                            ldo=1,
                                            sdo=8 * self.BLK_K * F16_BYTES // F128_BYTES,
                                            swizzle=self.SWIZZLE,
                                        )
                                        T.ptx.tcgen05.encode_matrix_descriptor(
                                            T.address_of(descB),
                                            self.B_smem.ptr_to([ks, 0, ki * self.MMA_K]),
                                            ldo=1,
                                            sdo=8 * self.BLK_K * F16_BYTES // F128_BYTES,
                                            swizzle=self.SWIZZLE,
                                        )

                                        if ko == 0 and ks == 0 and ki == 0:
                                            T.ptx.tcgen05.mma(
                                                "float32",
                                                self.a_type,
                                                self.b_type,
                                                self.tmem_idx * MMA_M,
                                                descB,
                                                descA,
                                                descI,
                                                False,
                                                KernelConfig.CTA_GROUP,
                                                False,
                                            )
                                        else:
                                            T.ptx.tcgen05.mma(
                                                "float32",
                                                self.a_type,
                                                self.b_type,
                                                self.tmem_idx * MMA_M,
                                                descB,
                                                descA,
                                                descI,
                                                False,
                                                KernelConfig.CTA_GROUP,
                                                True,
                                            )
                                    self.mma2tma_bar.arrive(ks)
                                self.phase[0] = self.phase[0] ^ 1

                            if self.PIPE_REMAIN_NUM > 0:
                                # last remained loop
                                for ks in T.unroll(self.PIPE_REMAIN_NUM):

                                    # wait tma and sf-transpose arrival
                                    self.tma2mma_bar.wait(ks, self.phase[0])
                                    T.ptx.tcgen05.fence.after_thread_sync()

                                    # issue mma
                                    for ki in T.unroll(self.BLK_K // self.MMA_K):
                                        T.ptx.tcgen05.encode_matrix_descriptor(
                                            T.address_of(descA),
                                            self.A_smem.ptr_to([ks, 0, ki * self.MMA_K]),
                                            ldo=1,
                                            sdo=8 * self.BLK_K * F16_BYTES // F128_BYTES,
                                            swizzle=self.SWIZZLE,
                                        )
                                        T.ptx.tcgen05.encode_matrix_descriptor(
                                            T.address_of(descB),
                                            self.B_smem.ptr_to([ks, 0, ki * self.MMA_K]),
                                            ldo=1,
                                            sdo=8 * self.BLK_K * F16_BYTES // F128_BYTES,
                                            swizzle=self.SWIZZLE,
                                        )
                                        if self.PIPE_CIRCLE_NUM == 0 and ks == 0 and ki == 0:
                                            T.ptx.tcgen05.mma(
                                                "float32",
                                                self.a_type,
                                                self.b_type,
                                                self.tmem_idx * MMA_M,
                                                descB,
                                                descA,
                                                descI,
                                                False,
                                                KernelConfig.CTA_GROUP,
                                                True,
                                            )
                                        else:
                                            T.ptx.tcgen05.mma(
                                                "float32",
                                                self.a_type,
                                                self.b_type,
                                                self.tmem_idx * MMA_M,
                                                descB,
                                                descA,
                                                descI,
                                                False,
                                                KernelConfig.CTA_GROUP,
                                                True,
                                            )
                                    self.mma2tma_bar.arrive(ks)

                                # ensure that all mma is issued
                                self.mma2ld_bar.arrive(self.tmem_idx)

                                # for unaligned cases
                                for ks in T.unroll(self.PIPE_REMAIN_NUM, self.SMEM_PIPE_DEPTH):
                                    self.tma2mma_bar.wait(ks, self.phase[0])
                                    self.mma2tma_bar.arrive(ks)

                                self.phase[0] = self.phase[0] ^ 1
                            else:
                                # ensure that all mma is issued
                                self.mma2ld_bar.arrive(self.tmem_idx)
                        self.tile_idx += 1

                with T.warpgroup()[0:1]:
                    trap_when_assert_failed(self.tmem_addr[0] == 0)
                    self.phase[0] = 0
                    self.tmem_idx = self.tile_idx % self.TMEM_PIPE_DEPTH
                    self.tmem_phase = (self.tile_idx // self.TMEM_PIPE_DEPTH) & 1

                    # flush previous tma
                    # wait for the completion of all the mma of the same tile
                    self.mma2ld_bar.wait(self.tmem_idx, self.tmem_phase)
                    T.ptx.tcgen05.fence.after_thread_sync()

                    for ko in T.unroll(MMA_M // self.EPI_TILE):
                        self.stage = (
                            self.tile_idx * MMA_M // self.EPI_TILE + ko
                        ) % self.TMEM_PIPE_DEPTH
                        # wait the smem to be free
                        if ko >= self.TMEM_PIPE_DEPTH:
                            if lane_id == 0 and warp_id == 0:
                                T.ptx.cp_async.bulk.wait_group(self.TMEM_PIPE_DEPTH - 1)
                            T.ptx.bar.sync(10, 128)

                        # tmem -> rf (ld) -> smem
                        for ki in T.unroll(self.EPI_TILE // self.TMEM_LD_SIZE):
                            T.ptx.tcgen05.ld(
                                0 + self.tmem_idx * MMA_M + ko * self.EPI_TILE,
                                warp_id * 32,
                                ki * self.TMEM_LD_SIZE,
                                "32x32b",
                                self.TMEM_LD_SIZE,
                                False,
                                *[self.reg[j] for j in range(self.TMEM_LD_SIZE)],
                            )
                            T.ptx.tcgen05.wait.ld()
                            if self.out_type == "float16":
                                for vec in range(self.TMEM_LD_SIZE // 2):
                                    float22half2(
                                        T.address_of(self.reg_fp16[vec * 2]),
                                        T.address_of(self.reg[vec * 2]),
                                    )

                                for vec in range(self.TMEM_LD_SIZE):
                                    self.output_smem[
                                        self.stage,
                                        ki * self.TMEM_LD_SIZE + vec,
                                        warp_id * 32 + lane_id,
                                    ] = self.reg_fp16[vec]
                            else:
                                for vec in range(self.TMEM_LD_SIZE):
                                    self.output_smem[
                                        self.stage,
                                        ki * self.TMEM_LD_SIZE + vec,
                                        warp_id * 32 + lane_id,
                                    ] = self.reg[vec]
                        # the tmem can be overwritten
                        if ko == MMA_M // self.EPI_TILE - 1:
                            T.ptx.tcgen05.fence.before_thread_sync()
                            self.ld2mma_bar.arrive(self.tmem_idx)

                        T.ptx.fence.proxy(scope="shared")
                        T.ptx.bar.sync(10, 128)
                        # smem -> gmem
                        if lane_id == 0 and warp_id == 0:
                            if self.split_k_factor > 1:
                                T.ptx.cp_async.bulk.tensor.s2g(
                                    3,
                                    self.output_smem.ptr_to([self.stage, 0, 0]),
                                    self.output_tensor_map,
                                    n_idx * self.BLK_N,
                                    m_idx * BLK_M + ko * self.EPI_TILE,
                                    k_idx,
                                    cache_hint="evict_last",
                                )
                            else:
                                T.ptx.cp_async.bulk.tensor.s2g(
                                    2,
                                    self.output_smem.ptr_to([self.stage, 0, 0]),
                                    self.output_tensor_map,
                                    n_idx * self.BLK_N,
                                    m_idx * BLK_M + ko * self.EPI_TILE,
                                )
                            T.ptx.cp_async.bulk.commit_group()
                    if lane_id == 0 and warp_id == 0:
                        T.ptx.cp_async.bulk.wait_group(0)
                    T.ptx.bar.sync(10, 128)
                    self.tile_idx += 1

    @T.macro
    def run(self, m_idx, n_idx, k_idx):
        if self.M <= 32:
            self._run(m_idx, n_idx, k_idx, 32, 32)
        elif self.M <= 64:
            self._run(m_idx, n_idx, k_idx, 64, 64)
        else:
            self._run(m_idx, n_idx, k_idx, 128, 128)
