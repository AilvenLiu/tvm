from tvm.script import tir as T
from tvm.script import tirp as Tp
from tvm.script.ir_builder import IRBuilder
from tvm.tirp.megakernel.common import ceildiv, F16_BYTES, F32_BYTES, KernelConfig

from tvm.tirp.megakernel.gemm import GemmTile, trap_when_assert_failed, float22half2


class GroupGEMMTile(GemmTile):

    def __init__(
        self,
        N,
        K,
        num_experts,
        top_k,
        a_type,
        b_type,
        BLK_M,
        MMA_M,
        reorder_output=False,
        prefetch_on=False,
        profiler_on=False,
    ):
        super().__init__(
            N,
            K,
            a_type,
            b_type,
            1,
            BLK_M,
            MMA_M,
            "float16" if not reorder_output else "float32",
            prefetch_on,
            profiler_on,
        )
        self.num_experts = num_experts
        self.top_k = top_k
        self.reorder_output = reorder_output
        self.VEC_LEN = 16 // F16_BYTES

    @T.macro
    def host_init(self):
        T.call_packed("runtime.cuTensorMapEncodeTiled", self.A_tensor_map, self.a_type, 2, self.A.data, 
                      self.K, self.M, self.K * F16_BYTES, self.BLK_K, self.BLK_M, 1, 1, 0, self.SWIZZLE, 0, 0)
        T.call_packed("runtime.cuTensorMapEncodeTiled", self.B_tensor_map, self.b_type, 3, self.B.data, 
                      self.K, self.N, self.num_experts, self.K * F16_BYTES, self.K * self.N * F16_BYTES, self.BLK_K, self.BLK_N, 1, 1, 1, 1, 0, self.SWIZZLE, 0, 0)
        if not self.reorder_output:
            T.call_packed("runtime.cuTensorMapEncodeTiled", self.output_tensor_map, self.out_type, 2, self.output.data, 
                            self.N, self.M, self.N * F16_BYTES, self.MMA_N, self.EPI_TILE, 1, 1, 0, 0, 0, 0)

    def set_moe_info(self, expert_ids, routing_weights, sorted_token_ids):
        self.expert_ids = expert_ids
        self.routing_weights = routing_weights
        self.sorted_token_ids = sorted_token_ids

    def _alloc_local(self, m_idx):
        super()._alloc_local(m_idx)
        self.eid = T.local_cell("int32", name="eid")
        T.buffer_store(self.eid.buffer, self.expert_ids[m_idx], 0)

    @T.macro
    def _fetch_B(self, ks, k_offset, n_idx):
        T.ptx.cp_async.bulk.tensor.g2c(
            3,
            self.B_smem.ptr_to([ks, 0, 0]),
            self.tma2mma_bar.mbar.ptr_to([ks]),
            self.B_tensor_map,
            k_offset,
            n_idx * self.BLK_N,
            self.eid,
            cta_group=KernelConfig.CTA_GROUP,
            cache_hint="evict_first",
        )

    @T.macro
    def _consumer_wg(self, m_idx, n_idx, k_idx):
        if not self.reorder_output:
            GemmTile._consumer_wg(self, m_idx, n_idx, k_idx)
        else:
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
                    # tmem -> rf (ld) -> smem
                    for ki in T.unroll(self.EPI_TILE // self.TMEM_LD_SIZE):
                        T.ptx.tcgen05.ld(
                            0 + self.tmem_idx * self.MMA_M + ko * self.EPI_TILE,
                            warp_id * 32,
                            ki * self.TMEM_LD_SIZE,
                            "32x32b",
                            self.TMEM_LD_SIZE,
                            False,
                            *[self.reg[j] for j in range(self.TMEM_LD_SIZE)],
                        )
                        T.ptx.tcgen05.wait.ld()
                        for vec in range(self.TMEM_LD_SIZE):
                            self.output_smem[
                                self.stage,
                                ki * self.TMEM_LD_SIZE + vec,
                                warp_id * 32 + lane_id,
                            ] = self.reg[vec]
                    # the tmem can be overwritten
                    if ko == self.MMA_M // self.EPI_TILE - 1:
                        T.ptx.tcgen05.fence.before_thread_sync()
                        self.ld2mma_bar.arrive(self.tmem_idx)

                    T.ptx.fence.proxy(scope="shared")
                    T.ptx.bar.sync(10, 128)
                    # smem -> gmem
                    for i in range(self.EPI_TILE * self.BLK_N // (128 * self.VEC_LEN)):
                        row_idx = (i * 128 + tid_in_wg) * self.VEC_LEN // self.BLK_N
                        col_idx = (i * 128 + tid_in_wg) * self.VEC_LEN % self.BLK_N
                        reordered_row_idx = self.sorted_token_ids[
                            m_idx * self.BLK_M + ko * self.EPI_TILE + row_idx
                        ]
                        if reordered_row_idx >= self.routing_weights.shape[0]:
                            continue
                        routing_weight = self.routing_weights[reordered_row_idx]
                        # TODO: vectorize this
                        for v in range(self.VEC_LEN):
                            self.output[reordered_row_idx, n_idx * self.BLK_N + col_idx + v] = T.cast(
                                self.output_smem[self.stage, row_idx, col_idx + v] *routing_weight, "float16"
                            )
                T.ptx.bar.sync(10, 128)
                self.tile_idx += 1
                if warp_id == 0:
                    self.smem_manager.arrive_specific(lane_id, self.output_smem, 0)

    @T.macro
    def run(self, m_idx, n_idx, k_idx,  expert_ids, routing_weights, sorted_token_ids, profiler = None):
        self.set_moe_info(expert_ids, routing_weights, sorted_token_ids)
        GemmTile.run(self, m_idx, n_idx, k_idx, profiler)
