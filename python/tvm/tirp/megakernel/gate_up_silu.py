from tvm.script import tir as T
from tvm.script import tirp as Tp

from tvm.tirp.bench.utils import CudaProfiler
from tvm.tirp.megakernel.gemm import GemmTile, Barriers
from tvm.tirp.megakernel.common import (
    F16_BYTES,
    KernelConfig,
    ProfileEventType,
    SmemManager,
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


def silu(val):
    return T.cuda.func_call(
        "silu",
        val,
        source_code=f"""
__forceinline__ __device__ float silu(float val) {{
  return val / (1.0f + __expf(-val));
}}
    """, return_type="float32"
    )

# A: [batch_size, hidden_size]
# B: [intermediate_size * 2, hidden_size]
# output: [batch_size, intermediate_size]
# tile: [blk_m, k] @ [blk_n, k] -> [blk_m, blk_n // 2]

class GateUpSiluTile(GemmTile):

    need_init = False
    D_layout = T.TileLayout(
        shard=((GemmTile.TMEM_PIPE_DEPTH, GemmTile.EPI_TILE, GemmTile.MMA_N // 2), (GemmTile.EPI_TILE * GemmTile.MMA_N // 2, GemmTile.MMA_N // 2, 1))
    )

    def _alloc_buffer(self, smem_manager: SmemManager):
        self.smem_manager = smem_manager
        # alloc shared memory
        self.A_smem = smem_manager.alloc(
            (self.SMEM_PIPE_DEPTH, self.MAX_BLK_M, self.BLK_K),
            self.a_type,
            layout=self.A_layout,
            align=1024,
            split=self.SMEM_PIPE_DEPTH,
            name="A_smem",
            method="exclusive",
        )
        self.B_smem = smem_manager.alloc(
            (self.SMEM_PIPE_DEPTH, self.BLK_N, self.BLK_K),
            self.b_type,
            layout=self.B_layout,
            align=1024,
            split=self.SMEM_PIPE_DEPTH,
            name="B_smem",
            method="exclusive",
        )
        self.output_smem = smem_manager.alloc(
            (self.TMEM_PIPE_DEPTH, self.EPI_TILE, self.MMA_N // 2),
            "float16",
            layout=self.D_layout,
            align=1024,
            name="output_smem",
            method="exclusive",
        )

    def _alloc_local(self, m_idx):
        # alloc local memory
        self.reg = T.alloc_buffer((self.TMEM_LD_SIZE,), "float32", scope="local", name="reg")
        self.reg_fp16 = T.alloc_buffer(
            (self.TMEM_LD_SIZE // 2,), "float16", scope="local", name="reg_fp16"
        )
        self.tmem_idx = T.local_cell("int32", name="tmem_idx")
        self.tmem_phase = T.local_cell("int32", name="tmem_phase")
        self.stage = T.local_cell("int32", name="stage")
        self.wait_complete = T.local_cell("bool", name="wait_complete")
        self.off = T.local_cell("int32", name="off")

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
        assert B.shape[1] == self.K
        assert A.shape[1] == self.K
        assert output.shape[0] == self.M
        assert output.shape[1] == self.N // 2
        assert self.a_type == A.dtype
        assert self.b_type == B.dtype
        assert self.out_type == output.dtype

    @T.macro
    def host_init(self):
        T.call_packed("runtime.cuTensorMapEncodeTiled", self.A_tensor_map, self.a_type, 2, self.A.data,
                      self.K, self.M, self.K * F16_BYTES, self.BLK_K, self.BLK_M, 1, 1, 0, self.SWIZZLE, 0, 0)
        T.call_packed("runtime.cuTensorMapEncodeTiled", self.B_tensor_map, self.b_type, 2, self.B.data,
                      self.K, self.N, self.K * F16_BYTES, self.BLK_K, self.BLK_N, 1, 1, 0, self.SWIZZLE, 0, 0)
        T.call_packed("runtime.cuTensorMapEncodeTiled", self.output_tensor_map, "float16", 2, self.output.data,
                        self.N // 2, self.M, self.N // 2 * F16_BYTES, self.MMA_N // 2, self.EPI_TILE, 1, 1, 0, 0, 0, 0)


    @T.macro
    def _consumer_wg(self, m_idx, n_idx, k_idx, profiler: CudaProfiler):
        with T.thread():
            warp_id = T.warp_id([KernelConfig.WARP_NUMBER], parent="warpgroup")
            lane_id = T.thread_id([32], parent="warp")
            T.cuda.trap_when_assert_failed(self.tmem_addr[0] == 0)
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

            SILU_HANDLE_UNIT = T.meta_var(self.TMEM_LD_SIZE // 2)
            self.off = T.if_then_else(lane_id < 16, SILU_HANDLE_UNIT, 0)
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
                    if self.profiler_on:
                        profiler.start(ProfileEventType.SILU_MUL, lane_id == 0)
                    # for each warp, lane 0~15 holds the gate output, lane 16~31 holds the up output
                    for kv in T.unroll(SILU_HANDLE_UNIT):
                        self.reg[self.off + kv] = T.tvm_warp_shuffle_xor(0xffffffff, self.reg[self.off + kv], 16, 32, 32)
                    for kv in T.unroll(SILU_HANDLE_UNIT):
                        self.reg[kv] = silu(self.reg[kv]) * self.reg[SILU_HANDLE_UNIT + kv]
                    Tp.cast(self.reg_fp16[:], self.reg[0:SILU_HANDLE_UNIT], vec=SILU_HANDLE_UNIT)
                    st = T.meta_var(ki * self.TMEM_LD_SIZE + (lane_id // 16) * SILU_HANDLE_UNIT)
                    Tp.copy(self.output_smem[self.stage, st:st + SILU_HANDLE_UNIT, warp_id * 16 + lane_id % 16], self.reg_fp16[:], vec=SILU_HANDLE_UNIT)
                    if self.profiler_on:
                        profiler.end(ProfileEventType.SILU_MUL, lane_id == 0)
                                            
                # the tmem can be overwritten
                if ko == self.MMA_M // self.EPI_TILE - 1:
                    T.ptx.tcgen05.fence.before_thread_sync()
                    self.ld2mma_bar.arrive(self.tmem_idx)

                T.ptx.fence.proxy(scope="shared")
                T.ptx.bar.sync(10, 128)
                # smem -> gmem
                if lane_id == 0 and warp_id == 0:
                    T.ptx.cp_async.bulk.tensor.s2g(
                        2,
                        self.output_smem.ptr_to([self.stage, 0, 0]),
                        self.output_tensor_map,
                        n_idx * self.BLK_N // 2,
                        m_idx * self.M_pad_size + ko * self.EPI_TILE,
                    )
                    T.ptx.cp_async.bulk.commit_group()
            if lane_id == 0 and warp_id == 0:
                T.ptx.cp_async.bulk.wait_group(0)
            T.ptx.bar.sync(10, 128)
            self.tile_idx += 1
            if warp_id == 0:
                self.smem_manager.arrive_specific(lane_id, self.output_smem, 0)
