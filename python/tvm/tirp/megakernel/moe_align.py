from tvm.script import tir as T
from tvm.script import tirp as Tp

from .common import F32_BYTES, F16_BYTES, KernelConfig, SmemManager, Tile, ceildiv, float22half2

def next_power_of_two(x):
    return 1 << (x-1).bit_length()

class MOEAlignTile(Tile):

    def __init__(self, num_experts, numel, block_size, input_dtype="int32", pad_sorted_token_ids=False):
        super().__init__()
        self.num_experts = num_experts
        self.scan_size = next_power_of_two(num_experts)
        self.numel = numel # numel = num_tokens * num_experts
        self.block_size = block_size
        self.pad_sorted_token_ids = pad_sorted_token_ids

    def _alloc_buffer(self, smem_manager: SmemManager):
        self.smem_manager = smem_manager
        # alloc shared memory
        self.shared_counts = smem_manager.alloc([self.num_experts], "int32").buffer
        self.prefix = smem_manager.alloc([self.num_experts+1], "int32").buffer
        self.scan_buf = smem_manager.alloc([self.scan_size], "int32").buffer
        self.warp_sums = smem_manager.alloc([KernelConfig.WARP_NUMBER * KernelConfig.WG_NUMBER], "int32").buffer
        self.s_total_tokens_post_pad = smem_manager.alloc([1], "int32").buffer


    def init(self, smem_manager: SmemManager):
        self._alloc_buffer(smem_manager)
    
    @T.macro
    def warp_exclusive_scan(self, v, output, mask=0xffffffff):
        # offset = T.alloc_cell("int32", name="offset")
        # original = T.alloc_cell("int32", name="original")
        # with T.warp():
        #     lane_id = T.thread_id(32, parent="warp")
        #     offset = 1
        #     original = v
        #     while offset < 32:
        #         n = T.tvm_warp_shuffle_up(mask, v, offset)
        #         if lane_id >= offset:
        #             v += n
        #         offset = offset << 1
        #     output = v - original
        #     v = original
        output[0] = T.cuda.func_call("warp_exclusive_scan", v, mask,
                                  source_code="""
            __device__ __forceinline__ int warp_exclusive_scan(int v, unsigned mask = 0xffffffffu) {
            int original = v;
            #pragma unroll
            for (int offset = 1; offset < 32; offset <<= 1) {
                int n = __shfl_up_sync(mask, v, offset);
                if ((threadIdx.x & (32 - 1)) >= offset) v += n;
            }
            return v - original;
            }
                                  """, return_type="int32")

    @T.macro
    def run(self, m_idx, n_idx, k_idx, topk_ids, sorted_token_ids, expert_ids, total_tokens_post_pad, cumsum_buffer):
        idx = T.alloc_local([1], "int32", name="idx")
        pre = T.alloc_local([1], "int32", name="pre")
        sum_val = T.alloc_local([1], "int32", name="sum_val")
        with T.cta():
            tid = T.thread_id([KernelConfig.NUM_THREADS], parent="cta")
            lane_id = T.thread_id([32], parent="warp")
            warp_id = T.warp_id([KernelConfig.WARP_NUMBER * KernelConfig.WG_NUMBER], parent="cta")
            if tid < self.num_experts:
                self.shared_counts[tid] = 0
            T.tvm_storage_sync("shared")
            idx[0] = tid
            while idx[0] < self.numel:
                expert_id = topk_ids[idx[0]] # TODO: the reference expert use topk_ids[idx[0]] + 1. Is it correct?
                T.cuda.atomic_add(T.address_of(self.shared_counts[expert_id]), 1)
                idx[0] += KernelConfig.NUM_THREADS
            T.tvm_storage_sync("shared")
            if tid < self.num_experts:
                padded_count = ceildiv(self.shared_counts[tid], self.block_size) * self.block_size
                self.scan_buf[tid] = padded_count
            elif tid < self.scan_size:
                self.scan_buf[tid] = 0
            T.tvm_storage_sync("shared")
            v = T.if_then_else(tid < self.num_experts, self.scan_buf[tid], 0)
            self.warp_exclusive_scan(v, pre)
            if lane_id == 31:
                self.warp_sums[warp_id] = pre[0] + v
            T.tvm_storage_sync("shared")
            num_warps_for_scan = T.meta_var(ceildiv(self.scan_size, 32))
            if warp_id == 0:
                val = T.if_then_else(lane_id < num_warps_for_scan, self.warp_sums[lane_id], 0)
                self.warp_exclusive_scan(val, sum_val)
                if lane_id == num_warps_for_scan - 1:
                    self.prefix[self.num_experts] = sum_val[0] + val
                    self.s_total_tokens_post_pad[0] = sum_val[0] + val
                    total_tokens_post_pad[0] = self.s_total_tokens_post_pad[0]
                self.warp_sums[lane_id] = sum_val[0]
            T.tvm_storage_sync("shared")
            if tid < self.scan_size:
                self.scan_buf[tid] = pre[0] + self.warp_sums[warp_id]
            if tid < self.num_experts:
                self.prefix[tid] = self.scan_buf[tid]
            if tid <= self.num_experts:
                cumsum_buffer[tid] = self.prefix[tid]
            T.tvm_storage_sync("shared")
            num_blocks = self.s_total_tokens_post_pad[0] // self.block_size
            idx[0] = tid
            while idx[0] < num_blocks:
                block_start = idx[0] * self.block_size
                left = T.alloc_local([1], "int32", name="left")
                right = T.alloc_local([1], "int32", name="right")
                with T.thread():
                    left[0] = 0
                    right[0] = self.num_experts
                    while left[0] < right[0]:
                        mid = (left[0] + right[0]) // 2
                        if self.prefix[mid] <= block_start:
                            left[0] = mid + 1
                        else:
                            right[0] = mid
                expert_ids[idx[0]] = left[0] - 1 # TODO: the reference expert use left - 2
                idx[0] += KernelConfig.NUM_THREADS
            if self.pad_sorted_token_ids:
                VEC_SIZE = 4
                idx[0] = tid * VEC_SIZE
                out_ptr = sorted_token_ids.view(-1)
                fill_vec = T.alloc_buffer([4], "int32", scope="local")
                for vec in T.vectorized(VEC_SIZE):
                    fill_vec[vec] = self.numel
                while idx[0] < self.s_total_tokens_post_pad[0]:
                    for vec in T.vectorized(VEC_SIZE):
                        out_ptr[idx[0] + vec] = fill_vec[vec]
                    idx[0] += VEC_SIZE * KernelConfig.NUM_THREADS

class CountAndSortExpertTokens(Tile):
    
    def __init__(self, numel):
        super().__init__()
        self.numel = numel # numel = num_tokens * num_experts
    
    @T.macro
    def run(self, m_idx, n_idx, k_idx, topk_ids, sorted_token_ids, cumsum_buffer):
        idx = T.alloc_local([1], "int32", name="idx")
        with T.cta():
            tid = T.thread_id([KernelConfig.NUM_THREADS], parent="cta")
            idx[0] = m_idx * KernelConfig.NUM_THREADS + tid
            while idx[0] < self.numel:
                expert_id = topk_ids[idx[0]]
                rank_post_pad = T.cuda.atomic_add(T.address_of(cumsum_buffer[expert_id]), 1)
                sorted_token_ids[rank_post_pad] = idx[0]
                idx[0] += KernelConfig.NUM_THREADS * KernelConfig.SM_NUMBER


