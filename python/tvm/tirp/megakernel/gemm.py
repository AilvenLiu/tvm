from tvm.script import tir as T
from tvm.script import tirp as Tp
from tvm.tirp.bench.utils import CudaProfiler

from tvm.tirp.megakernel.common import (
    F16_BYTES,
    F32_BYTES,
    F128_BYTES,
    Barriers,
    KernelConfig,
    ProfileEventType,
    SmemManager,
    Tile,
    ceildiv,
    float22half2,
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

mbarrier_try_wait = """
__forceinline__ __device__ bool tvm_builtin_ptx_mbarrier_try_wait(void* barrier, int phase) {
    uint32_t smem_int_ptr = __cvta_generic_to_shared(barrier);
  uint32_t waitComplete;

  asm volatile(
      "{\\n\\t"
      ".reg .pred P1; \\n\\t"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%1], %2; \\n\\t"
      "selp.b32 %0, 1, 0, P1; \\n\\t"
      "}"
      : "=r"(waitComplete)
      : "r"(smem_int_ptr), "r"(phase));

  return static_cast<bool>(waitComplete);
}

"""

class GemmTile(Tile):
    SMEM_PIPE_DEPTH = 6
    TMEM_PIPE_DEPTH = 2
    MAX_BLK_M, MAX_BLK_N, BLK_N, BLK_K = 128, 128, 128, 64
    MMA_N, MMA_K = 128, 16
    EPI_TILE = 32
    TMEM_LD_SIZE = 8
    N_COLS = 512
    SWIZZLE = 3
    SMEM_SIZE = (
        SMEM_PIPE_DEPTH * MAX_BLK_M * BLK_K * F16_BYTES
        + SMEM_PIPE_DEPTH * MAX_BLK_N * BLK_K * F16_BYTES
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
        T.TileLayout(shard=((SMEM_PIPE_DEPTH, MAX_BLK_N, BLK_K), (MAX_BLK_N * BLK_K, BLK_K, 1))),
    )
    D_layout = T.TileLayout(
        shard=((TMEM_PIPE_DEPTH, EPI_TILE, MMA_N), (EPI_TILE * MMA_N, MMA_N, 1))
    )

    # idx of current gemm tile (no matter which shape it is)
    tile_idx = None

    def __init__(self, N, K, a_type, b_type, split_k_factor, BLK_M, MMA_M, out_type=None, use_tma_reduce=False, low_batch=True, blk_n=128, prefetch_on=False, profiler_on=False):
        super().__init__()
        self.BLK_M = BLK_M
        self.BLK_N = blk_n
        self.MMA_M = MMA_M
        self.N = N
        self.K = K
        self.a_type = a_type
        self.b_type = b_type
        assert a_type == "float16", "only float16 is supported for now"
        assert b_type == "float16", "only float16 is supported for now"
        assert not (use_tma_reduce and split_k_factor == 1), "use_tma_reduce when split_k_factor == 1 is not supported"
        if out_type is None:
            self.out_type = "float32" if split_k_factor > 1 or use_tma_reduce else "float16"
        else:
            self.out_type = out_type
        self.split_k_factor = split_k_factor
        self.use_tma_reduce = use_tma_reduce
        self.low_batch = low_batch
        self.prefetch_on = prefetch_on
        self.profiler_on = profiler_on
        self.TILE_K = ceildiv(ceildiv(self.K, self.split_k_factor), self.BLK_K) * self.BLK_K
        self.PIPE_CIRCLE_NUM = (self.TILE_K // self.BLK_K) // self.SMEM_PIPE_DEPTH
        self.PIPE_REMAIN_NUM = (self.TILE_K // self.BLK_K) % self.SMEM_PIPE_DEPTH
        self.M_pad_size = BLK_M

    def _alloc_buffer(self, smem_manager: SmemManager):
        self.smem_manager = smem_manager
        # alloc shared memory
        self.A_smem = smem_manager.alloc(
            (self.SMEM_PIPE_DEPTH, self.MAX_BLK_M, self.BLK_K),
            self.a_type,
            layout=self.A_layout,
            align=1024,
            split=self.SMEM_PIPE_DEPTH,
            method="exclusive",
        ).buffer
        self.B_smem = smem_manager.alloc(
            (self.SMEM_PIPE_DEPTH, self.MAX_BLK_N, self.BLK_K),
            self.b_type,
            layout=self.B_layout,
            align=1024,
            split=self.SMEM_PIPE_DEPTH,
            method="exclusive",
        ).buffer
        self.output_smem = smem_manager.alloc(
            (self.TMEM_PIPE_DEPTH, self.EPI_TILE, self.MMA_N),
            self.out_type,
            layout=self.D_layout,
            align=1024,
            method="exclusive",
        ).buffer

    def _alloc_local(self, m_idx):
        # alloc local memory
        self.reg = T.alloc_buffer((self.TMEM_LD_SIZE,), "float32", scope="local", name="reg")
        if self.out_type == "float16":
            self.reg_fp16 = T.alloc_buffer(
                (self.TMEM_LD_SIZE,), self.out_type, scope="local", name="reg_fp16"
            )
        self.tmem_idx = T.local_cell("int32", name="tmem_idx")
        self.tmem_phase = T.local_cell("int32", name="tmem_phase")
        self.stage = T.local_cell("int32", name="stage")
        self.wait_complete = T.local_cell("bool", name="wait_complete")

    @classmethod
    def _alloc_buffer_class_member(cls, smem_manager: SmemManager):
        # alloc shared memory
        # use GemmTile instead of cls to avoid re-allocating memory for different subclasses
        # TODO: this cannot be generalized if there are multiple subclasses of GemmTile
        #       we need to delete these members in class_finalize, and only alloc when there are no members
        GemmTile.tmem_addr = smem_manager.alloc([1], "uint32", method="persistent").buffer
        GemmTile.tma2mma_bar = BarTMA2MMA(smem_manager, cls.SMEM_PIPE_DEPTH, True)
        GemmTile.mma2tma_bar = BarMMA2TMA(smem_manager, cls.SMEM_PIPE_DEPTH, False)
        GemmTile.mma2ld_bar = BarMMA2LD(smem_manager, cls.TMEM_PIPE_DEPTH, True)
        GemmTile.ld2mma_bar = BarLD2MMA(smem_manager, cls.TMEM_PIPE_DEPTH, False)
        # alloc local memory
        GemmTile.tile_idx = T.local_cell("int32", "tile_idx")
        GemmTile.phase = T.alloc_buffer((1,), "int32", scope="local", name="phase")

    @classmethod
    @T.macro
    def class_init(cls, smem_manager: SmemManager):
        cls._alloc_buffer_class_member(smem_manager)
        cls.tile_idx = 0
        # alloc TMEM
        with T.warp()[0:1]:
            T.ptx.tcgen05.alloc(T.address_of(cls.tmem_addr[0]), n_cols=cls.N_COLS, cta_group=1)
            T.cuda.warp_sync()
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
    def init(self, smem_manager: SmemManager):
        self._alloc_buffer(smem_manager)

    def set_tensor_map(self, A_tensor_map, B_tensor_map, output_tensor_map, A, B, output):
        self.A_tensor_map = A_tensor_map
        self.B_tensor_map = B_tensor_map
        self.output_tensor_map = output_tensor_map
        self.M = A.shape[0]
        self.A = A
        self.B = B
        self.output = output
        assert self.a_type == A.dtype
        assert self.b_type == B.dtype

    @T.macro
    def host_init(self):
        T.call_packed("runtime.cuTensorMapEncodeTiled", self.A_tensor_map, self.a_type, 2, self.A.data,
                      self.K, self.M, self.K * F16_BYTES, self.BLK_K, self.BLK_M, 1, 1, 0, self.SWIZZLE, 0, 0)
        T.call_packed("runtime.cuTensorMapEncodeTiled", self.B_tensor_map, self.b_type, 2, self.B.data,
                      self.K, self.N, self.K * F16_BYTES, self.BLK_K, self.BLK_N, 1, 1, 0, self.SWIZZLE, 0, 0)
        if self.split_k_factor > 1:
            if not self.use_tma_reduce:
                T.call_packed("runtime.cuTensorMapEncodeTiled", self.output_tensor_map, self.out_type, 3, self.output.data,
                            self.N, self.M, self.split_k_factor, self.N * F32_BYTES, self.N * self.M * F32_BYTES, self.BLK_N, self.EPI_TILE, 1, 1, 1, 1, 0, 0, 0, 0)
            else:
                T.call_packed("runtime.cuTensorMapEncodeTiled", self.output_tensor_map, self.out_type, 2, self.output.data,
                          self.N, self.M, self.N * F32_BYTES, self.BLK_N, self.EPI_TILE, 1, 1, 0, 0, 0, 0)
        else:
            T.call_packed("runtime.cuTensorMapEncodeTiled", self.output_tensor_map, self.out_type, 2, self.output.data,
                          self.N, self.M, self.N * F16_BYTES, self.BLK_N, self.EPI_TILE, 1, 1, 0, 0, 0, 0)

    # call by warp 7 (tmp load warp)
    @T.macro
    def prefetch(self, m_idx, n_idx, k_idx, profiler: CudaProfiler):
        self._alloc_local(m_idx)
        with T.cta():
            wg_id = T.warpgroup_id([KernelConfig.WG_NUMBER], parent="cta")
            warp_id = T.warp_id([KernelConfig.WARP_NUMBER], parent="warpgroup")
            lane_id = T.thread_id([32], parent="warp")
            if wg_id == 1 and warp_id == 3:
                k_offset = k_idx * self.TILE_K
                if self.PIPE_CIRCLE_NUM > 0:
                    for ks in T.unroll(self.SMEM_PIPE_DEPTH):
                        # GMEM -> SMEM  (tma)
                        self.stage = ks
                        self.smem_manager.wait_specific(lane_id, self.B_smem, ks)
                        if self.profiler_on:
                            profiler.start(ProfileEventType.TMA, lane_id == 0)
                        if T.ptx.elect_sync():
                            self._fetch_B(ks, self.stage * self.BLK_K + k_offset, n_idx)
                        if self.profiler_on:
                            profiler.end(ProfileEventType.TMA, lane_id == 0)
    @T.macro
    def _fetch_B(self, ks, k_offset, n_idx):
        cache_hint = T.meta_var("evict_first" if self.low_batch else "")
        T.ptx.cp_async.bulk.tensor.g2c(
            2,
            self.B_smem.ptr_to([ks, 0, 0]),
            self.tma2mma_bar.mbar.ptr_to([ks]),
            self.B_tensor_map,
            k_offset,
            n_idx * self.BLK_N,
            cta_group=KernelConfig.CTA_GROUP,
            cache_hint=cache_hint,
        )

    @T.macro
    def _consumer_wg(self, m_idx, n_idx, k_idx, profiler: CudaProfiler):
        with T.cta():
            tid_in_wg = T.thread_id([128], parent="warpgroup")
            warp_id = T.warp_id([KernelConfig.WARP_NUMBER], parent="warpgroup")
            lane_id = T.thread_id([32], parent="warp")
            trap_when_assert_failed(self.tmem_addr[0] == 0)
            if warp_id == 0:
                self.smem_manager.wait_specific(lane_id, self.output_smem, 0)
            T.ptx.bar.sync(10, 128)
            self.phase[0] = 0
            self.tmem_idx = self.tile_idx % self.TMEM_PIPE_DEPTH
            self.tmem_phase = (self.tile_idx // self.TMEM_PIPE_DEPTH) & 1

            # flush previous tma
            # wait for the completion of all the mma of the same tile
            self.mma2ld_bar.wait(self.tmem_idx, self.tmem_phase)
            T.ptx.tcgen05.fence.after_thread_sync()

            for ko in T.unroll(self.MMA_M // self.EPI_TILE):
                self.stage = (
                    self.tile_idx * self.MMA_M // self.EPI_TILE + ko
                ) % self.TMEM_PIPE_DEPTH
                # wait the smem to be free
                if ko >= self.TMEM_PIPE_DEPTH:
                    if lane_id == 0 and warp_id == 0:
                        T.ptx.cp_async.bulk.wait_group(self.TMEM_PIPE_DEPTH - 1)
                    T.ptx.bar.sync(10, 128)

                # tmem -> rf (ld) -> smem
                for ki in T.unroll(self.EPI_TILE // self.TMEM_LD_SIZE):
                    T.ptx.tcgen05.ld(
                        0 + self.tmem_idx * self.M_pad_size + ko * self.EPI_TILE,
                        warp_id * 32,
                        ki * self.TMEM_LD_SIZE,
                        "32x32b",
                        self.TMEM_LD_SIZE,
                        False,
                        *[self.reg[j] for j in range(self.TMEM_LD_SIZE)],
                    )
                    T.ptx.tcgen05.wait.ld()
                    if self.out_type == "float16":
                        with T.thread():
                            Tp.cast(self.reg_fp16[:], self.reg[:])
                            Tp.copy(self.output_smem[self.stage, 
                                                     ki * self.TMEM_LD_SIZE:(ki + 1) * self.TMEM_LD_SIZE, 
                                                     warp_id * 32 + lane_id], 
                                    self.reg_fp16[:])
                    else:
                        with T.thread():
                            Tp.copy(self.output_smem[self.stage, 
                                                     ki * self.TMEM_LD_SIZE:(ki + 1) * self.TMEM_LD_SIZE, 
                                                     warp_id * 32 + lane_id], 
                                    self.reg[:])
                # the tmem can be overwritten
                if ko == self.MMA_M // self.EPI_TILE - 1:
                    T.ptx.tcgen05.fence.before_thread_sync()
                    self.ld2mma_bar.arrive(self.tmem_idx)

                T.ptx.fence.proxy(scope="shared")
                T.ptx.bar.sync(10, 128)
                # smem -> gmem
                if tid_in_wg == 0:
                    cache_hint = T.meta_var("evict_last" if self.low_batch else "")
                    if self.split_k_factor > 1:
                        if not self.use_tma_reduce:
                            T.ptx.cp_async.bulk.tensor.s2g(
                                3,
                                self.output_smem.ptr_to([self.stage, 0, 0]),
                                self.output_tensor_map,
                                n_idx * self.BLK_N,
                                m_idx * self.M_pad_size + ko * self.EPI_TILE,
                                k_idx,
                                cache_hint=cache_hint,
                            )
                        else:
                            T.ptx.cp_async.bulk.tensor.s2g_reduce(
                                2,
                                self.output_smem.ptr_to([self.stage, 0, 0]),
                                self.output_tensor_map,
                                n_idx * self.BLK_N,
                                m_idx * self.M_pad_size + ko * self.EPI_TILE,
                                cache_hint=cache_hint,
                                red_op="add"
                            )
                    else:
                        T.ptx.cp_async.bulk.tensor.s2g(
                            2,
                            self.output_smem.ptr_to([self.stage, 0, 0]),
                            self.output_tensor_map,
                            n_idx * self.BLK_N,
                            m_idx * self.M_pad_size + ko * self.EPI_TILE,
                        )
                T.ptx.cp_async.bulk.commit_group()
            if lane_id == 0 and warp_id == 0:
                T.ptx.cp_async.bulk.wait_group(0)
            T.ptx.bar.sync(10, 128)
            self.tile_idx += 1
            if warp_id == 0:
                self.smem_manager.arrive_specific(lane_id, self.output_smem, 0)

    @T.macro
    def _run(self, m_idx, n_idx, k_idx, profiler: CudaProfiler):
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
                            cache_hint = T.meta_var("evict_last" if self.low_batch else "")
                            for ko in T.serial(self.PIPE_CIRCLE_NUM):
                                for ks in T.unroll(self.SMEM_PIPE_DEPTH):
                                    # GMEM -> SMEM  (tma)
                                    self.stage = ko * self.SMEM_PIPE_DEPTH + ks
                                    if ko > 0:
                                        self.mma2tma_bar.wait(ks, self.phase[0])
                                    if self.profiler_on:
                                        profiler.start(ProfileEventType.TMA, lane_id == 0)
                                    if ko == 0:
                                        self.smem_manager.wait_specific_one_thread(self.A_smem, ks)
                                    T.ptx.cp_async.bulk.tensor.g2c(
                                        2,
                                        self.A_smem.ptr_to([ks, 0, 0]),
                                        self.tma2mma_bar.mbar.ptr_to([ks]),
                                        self.A_tensor_map,
                                        self.stage * self.BLK_K + k_offset,
                                        m_idx * self.M_pad_size,
                                        cta_group=KernelConfig.CTA_GROUP,
                                        cache_hint=cache_hint,
                                    )
                                    if not self.prefetch_on and ko == 0:
                                        self.smem_manager.wait_specific_one_thread(self.B_smem, ks)
                                    if not self.prefetch_on or ko > 0:
                                        # ko = 0 is prefetched before
                                        self._fetch_B(ks, self.stage * self.BLK_K + k_offset, n_idx)
                                    if self.profiler_on:
                                        profiler.end(ProfileEventType.TMA, lane_id == 0)
                                    self.tma2mma_bar.arrive(
                                        ks,
                                        KernelConfig.CTA_GROUP
                                        * self.BLK_K
                                        * (self.BLK_M + self.BLK_N)
                                        * F16_BYTES,
                                    )
                                self.phase[0] = self.phase[0] ^ 1

                            if self.PIPE_REMAIN_NUM > 0:
                                # last remained loop
                                for ks in T.unroll(self.PIPE_REMAIN_NUM):
                                    # GMEM -> SMEM  (tma)
                                    self.stage = self.PIPE_CIRCLE_NUM * self.SMEM_PIPE_DEPTH + ks
                                    self.mma2tma_bar.wait(ks, self.phase[0])
                                    if self.profiler_on:
                                        profiler.start(ProfileEventType.TMA, lane_id == 0)
                                    T.ptx.cp_async.bulk.tensor.g2c(
                                        2,
                                        self.A_smem.ptr_to([ks, 0, 0]),
                                        self.tma2mma_bar.mbar.ptr_to([ks]),
                                        self.A_tensor_map,
                                        self.stage * self.BLK_K + k_offset,
                                        m_idx * self.M_pad_size,
                                        cta_group=KernelConfig.CTA_GROUP,
                                        cache_hint=cache_hint,
                                    )
                                    self._fetch_B(ks, self.stage * self.BLK_K + k_offset, n_idx)
                                    if self.profiler_on:
                                        profiler.end(ProfileEventType.TMA, lane_id == 0)
                                    self.tma2mma_bar.arrive(
                                        ks,
                                        KernelConfig.CTA_GROUP
                                        * self.BLK_K
                                        * (self.BLK_M + self.BLK_N)
                                        * F16_BYTES,
                                    )
                                # for unaligned cases
                                for ks in T.unroll(self.PIPE_REMAIN_NUM, self.SMEM_PIPE_DEPTH):
                                    self.mma2tma_bar.wait(ks, self.phase[0])
                                    self.tma2mma_bar.arrive_only(ks)

                                self.phase[0] = self.phase[0] ^ 1

                        T.ptx.bar.sync(13, 64) # notify warp 6 to release smem chunks

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
                            self.MMA_M,
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
                            self.wait_complete = T.cuda.func_call(
                                "tvm_builtin_ptx_mbarrier_try_wait",
                                self.tma2mma_bar.mbar.ptr_to([0]),
                                self.tma2mma_bar.init_phase ^ self.phase[0],
                                source_code=mbarrier_try_wait,
                                return_type="bool",
                            )

                            # main inner mma loop
                            for ko in T.serial(self.PIPE_CIRCLE_NUM):
                                for ks in T.unroll(self.SMEM_PIPE_DEPTH):

                                    # wait tma and sf-transpose arrival
                                    if not self.wait_complete:
                                        self.tma2mma_bar.wait(ks, self.phase[0])
                                        # T.ptx.tcgen05.fence.after_thread_sync()
                                    if self.PIPE_REMAIN_NUM > 0 or ko != self.PIPE_REMAIN_NUM - 1 or ks != self.SMEM_PIPE_DEPTH - 1:
                                        if ks != self.SMEM_PIPE_DEPTH - 1:
                                            self.wait_complete = T.cuda.func_call(
                                                "tvm_builtin_ptx_mbarrier_try_wait",
                                                self.tma2mma_bar.mbar.ptr_to([(ks+1) % self.SMEM_PIPE_DEPTH]),
                                                self.tma2mma_bar.init_phase ^ self.phase[0],
                                                source_code=mbarrier_try_wait,
                                                return_type="bool",
                                            )
                                        else:
                                            self.wait_complete = T.cuda.func_call(
                                                "tvm_builtin_ptx_mbarrier_try_wait",
                                                self.tma2mma_bar.mbar.ptr_to(
                                                    [0]
                                                ),
                                                (self.tma2mma_bar.init_phase ^ 1) ^ self.phase[0],
                                                source_code=mbarrier_try_wait,
                                                return_type="bool",
                                            )

                                    # issue mma
                                    if self.profiler_on:
                                        profiler.start(ProfileEventType.MMA, lane_id == 0)
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
                                                self.tmem_idx * self.M_pad_size,
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
                                                self.tmem_idx * self.M_pad_size,
                                                descB,
                                                descA,
                                                descI,
                                                False,
                                                KernelConfig.CTA_GROUP,
                                                True,
                                            )
                                    if self.profiler_on:
                                        profiler.end(ProfileEventType.MMA, lane_id == 0)
                                    self.mma2tma_bar.arrive(ks)
                                self.phase[0] = self.phase[0] ^ 1

                            if self.PIPE_REMAIN_NUM > 0:
                                # last remained loop
                                for ks in T.unroll(self.PIPE_REMAIN_NUM):

                                    # wait tma and sf-transpose arrival
                                    if not self.wait_complete:
                                        self.tma2mma_bar.wait(ks, self.phase[0])
                                        # T.ptx.tcgen05.fence.after_thread_sync()
                                    if ks != self.PIPE_REMAIN_NUM - 1:
                                        self.wait_complete = T.cuda.func_call(
                                            "tvm_builtin_ptx_mbarrier_try_wait",
                                            self.tma2mma_bar.mbar.ptr_to([(ks+1) % self.SMEM_PIPE_DEPTH]),
                                            self.tma2mma_bar.init_phase ^ self.phase[0],
                                            source_code=mbarrier_try_wait,
                                            return_type="bool",
                                        )
                                    # issue mma
                                    if self.profiler_on:
                                        profiler.start(ProfileEventType.MMA, lane_id == 0)
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
                                                self.tmem_idx * self.M_pad_size,
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
                                                self.tmem_idx * self.M_pad_size,
                                                descB,
                                                descA,
                                                descI,
                                                False,
                                                KernelConfig.CTA_GROUP,
                                                True,
                                            )
                                    if self.profiler_on:
                                        profiler.end(ProfileEventType.MMA, lane_id == 0)
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

                    elif warp_id == 1:
                        self.smem_manager.wait_unused(lane_id, self)
                        self.smem_manager.arrive_unused(lane_id, self)
                    elif warp_id == 2:
                        self.phase[0] = self.phase[0] ^ (self.PIPE_CIRCLE_NUM & 1)
                        if self.PIPE_REMAIN_NUM > 0:
                            self.phase[0] = self.phase[0] ^ 1
                        T.ptx.bar.sync(13, 64) # wait warp 7 to finish
                        for ks in T.unroll(self.SMEM_PIPE_DEPTH):
                            self.mma2tma_bar.wait(ks, self.phase[0])
                            self.smem_manager.arrive_specific(lane_id, self.B_smem, ks)
                            self.smem_manager.arrive_specific(lane_id, self.A_smem, ks)

                with T.warpgroup()[0:1]:
                    self._consumer_wg(m_idx, n_idx, k_idx, profiler)

    @T.macro
    def run(self, m_idx, n_idx, k_idx, profiler: CudaProfiler = None):
        self._alloc_local(m_idx)
        self._run(m_idx, n_idx, k_idx, profiler)
        self.smem_manager.advance()
