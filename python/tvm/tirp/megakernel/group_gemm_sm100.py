from tvm.script import tir as T
from tvm.script import tirp as Tp
from tvm.script.ir_builder import IRBuilder
from tvm.tirp.megakernel.common import ceildiv, F16_BYTES, F32_BYTES, KernelConfig, SmemManager

from tvm.tirp.megakernel.gemm import GemmTile, trap_when_assert_failed, float22half2


red = """
__forceinline__ __device__ void red_f16_v4(half* address, half* reg) {
    uint16_t* h_reg = (uint16_t*) reg;
    asm volatile("red.global.v4.f16.add.noftz [%0], {%1, %2, %3, %4};"
                 :
                 : "l"(address), "h"(h_reg[0]), "h"(h_reg[1]), "h"(h_reg[2]), "h"(h_reg[3])
                 : "memory");
}

"""

class GroupGEMMTile(GemmTile):

    def __init__(
        self,
        N,
        K,
        num_experts,
        top_k,
        numel,
        a_type,
        b_type,
        low_batch=True,
        acc_output=False,
        prefetch_on=False,
        profiler_on=False,
    ):
        super().__init__(
            N,
            K,
            a_type,
            b_type,
            1,
            BLK_M=-1, # does not matter because we will set it later
            MMA_M=-1, # does not matter because we will set it later
            out_type="float16" if not acc_output else "float32",
            low_batch=low_batch,
            prefetch_on=prefetch_on,
            profiler_on=profiler_on,
        )
        self.num_experts = num_experts
        self.top_k = top_k
        self.numel = numel
        self.acc_output = acc_output
        self.VEC_LEN = 16 // F32_BYTES
        self.BLK_M_candidate = [128, 64, 32]
        self.A_tensor_maps = {}
        self.output_tensor_maps = {}
        self.M_pad_size = max(self.BLK_M_candidate)

    @T.macro
    def _init_A_and_output_tensor_maps(self, BLK_M):
        A_tensor_map = T.meta_var(self.A_tensor_maps[BLK_M])
        output_tensor_map = T.meta_var(self.output_tensor_maps[BLK_M])
        T.call_packed("runtime.cuTensorMapEncodeTiled", A_tensor_map, self.a_type, 2, self.A.data, 
                      self.K, self.M, self.K * F16_BYTES, self.BLK_K, BLK_M, 1, 1, 0, self.SWIZZLE, 0, 0)
        if not self.acc_output:
            T.call_packed("runtime.cuTensorMapEncodeTiled", output_tensor_map, self.out_type, 2, self.output.data, 
                        self.N, self.M, self.N * F16_BYTES, self.MMA_N, self.EPI_TILE, 1, 1, 0, 0, 0, 0)

    def _init_A_and_output_helper(self):
        for BLK_M in self.BLK_M_candidate:
            self._init_A_and_output_tensor_maps(BLK_M)

    @T.macro
    def host_init(self):
        self._init_A_and_output_helper()
        T.call_packed("runtime.cuTensorMapEncodeTiled", self.B_tensor_map, self.b_type, 3, self.B.data, 
                      self.K, self.N, self.num_experts, self.K * F16_BYTES, self.K * self.N * F16_BYTES, self.BLK_K, self.BLK_N, 1, 1, 1, 1, 0, self.SWIZZLE, 0, 0)

    def set_tensor_map(self, A_tensor_maps, B_tensor_map, output_tensor_maps, A, B, output):        
        assert len(A_tensor_maps) >= len(self.BLK_M_candidate)
        assert len(output_tensor_maps) >= len(self.BLK_M_candidate)
        for i, BLK_M in enumerate(self.BLK_M_candidate[:len(A_tensor_maps)]):    
            self.A_tensor_maps[BLK_M] = A_tensor_maps[i]
            self.output_tensor_maps[BLK_M] = output_tensor_maps[i]
        self.B_tensor_map = B_tensor_map
        self.M = A.shape[0]
        self.A = A
        self.B = B
        self.output = output
        assert self.a_type == A.dtype
        assert self.b_type == B.dtype

    def set_moe_info(self, expert_ids, routing_weights, sorted_token_ids):
        self.expert_ids = expert_ids
        self.routing_weights = routing_weights
        self.sorted_token_ids = sorted_token_ids

    def _alloc_local(self, m_idx):
        super()._alloc_local(m_idx)
        self.num_tokens_in_block = T.local_cell("int32", name="num_tokens_in_block")
        self.eid = T.local_cell("int32", name="eid")
        T.buffer_store(self.eid.buffer, self.expert_ids[m_idx], 0)

    @property   
    def A_tensor_map(self):
        return self.A_tensor_maps[self.BLK_M]

    @property
    def output_tensor_map(self):
        return self.output_tensor_maps[self.BLK_M]

    @T.macro
    def _fetch_B(self, ks, k_offset, n_idx):
        cache_hint = T.meta_var("evict_first" if self.low_batch else "")
        T.ptx.cp_async.bulk.tensor.g2c(
            3,
            self.B_smem.ptr_to([ks, 0, 0]),
            self.tma2mma_bar.mbar.ptr_to([ks]),
            self.B_tensor_map,
            k_offset,
            n_idx * self.BLK_N,
            self.eid,
            cta_group=KernelConfig.CTA_GROUP,
            cache_hint=cache_hint,
        )
        
    @classmethod
    def class_init(cls, smem_manager: SmemManager):
        super().class_init(smem_manager)
        cls.smem_sorted_token_ids = smem_manager.alloc([cls.MAX_BLK_M], "int32", method="persistent").buffer
        cls.smem_routing_weights = smem_manager.alloc([cls.MAX_BLK_M], "float32", method="persistent").buffer

    @T.macro
    def _consumer_wg(self, m_idx, n_idx, k_idx):
        if not self.acc_output:
            GemmTile._consumer_wg(self, m_idx, n_idx, k_idx)
        else:
            with T.cta():
                tid_in_wg = T.thread_id([128], parent="warpgroup")
                warp_id = T.warp_id([KernelConfig.WARP_NUMBER], parent="warpgroup")
                lane_id = T.thread_id([32], parent="warp")
                trap_when_assert_failed(self.tmem_addr[0] == 0)
                if tid_in_wg < self.M_pad_size:
                    idx = self.sorted_token_ids[m_idx * self.M_pad_size + tid_in_wg]
                    self.smem_sorted_token_ids[tid_in_wg] = idx
                    self.smem_routing_weights[tid_in_wg] = self.routing_weights[idx]
                T.ptx.bar.sync(10, 128)
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
                            0 + self.tmem_idx * self.M_pad_size + ko * self.EPI_TILE,
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
                        reordered_row_idx = self.smem_sorted_token_ids[ko * self.EPI_TILE + row_idx]
                        if reordered_row_idx >= self.numel:
                            break
                        routing_weight = self.smem_routing_weights[ko * self.EPI_TILE + row_idx]
                        # TODO: vectorize this
                        o_reg_f32 = T.alloc_buffer([self.VEC_LEN], "float32", scope="local")
                        o_reg_f16 = T.alloc_buffer([self.VEC_LEN], "float16", scope="local")
                        for v in range(self.VEC_LEN):
                            o_reg_f32[v] = self.output_smem[self.stage, row_idx, col_idx + v]
                        for v in T.unroll(self.VEC_LEN):
                            o_reg_f16[v] = T.cast(o_reg_f32[v] * routing_weight, "float16")
                        T.cuda.func_call("red_f16_v4", T.address_of(self.output[reordered_row_idx // self.top_k, n_idx * self.BLK_N + col_idx]), T.address_of(o_reg_f16[0]), source_code=red)
                T.ptx.bar.sync(10, 128)
                self.tile_idx += 1
                if warp_id == 0:
                    self.smem_manager.arrive_specific(lane_id, self.output_smem, 0)

    def set_BLK_M(self, BLK_M):
        assert BLK_M in self.BLK_M_candidate
        self.BLK_M = BLK_M
        self.MMA_M = BLK_M

    @T.macro
    def run(self, m_idx, n_idx, k_idx,  expert_ids, routing_weights, sorted_token_ids, valid_num_tokens, profiler = None):
        self.set_moe_info(expert_ids, routing_weights, sorted_token_ids)
        self._alloc_local(m_idx)
        if valid_num_tokens is not None:
            self.num_tokens_in_block = valid_num_tokens[m_idx]
        num_tokens_in_block = T.meta_var(self.num_tokens_in_block if valid_num_tokens is not None else 32 if self.low_batch else self.M_pad_size)
        with T.cta():
            tid = T.thread_id([256], parent="cta")
            if num_tokens_in_block <= 32:
                self.set_BLK_M(32)
                GemmTile._run(self, m_idx, n_idx, k_idx, profiler)
            elif num_tokens_in_block <= 64:
                self.set_BLK_M(64)
                GemmTile._run(self, m_idx, n_idx, k_idx, profiler)
            else:
                self.set_BLK_M(128)
                GemmTile._run(self, m_idx, n_idx, k_idx, profiler)
            self.smem_manager.advance()
