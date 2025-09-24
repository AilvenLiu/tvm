import math
from enum import Enum

import ml_dtypes
import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.ir import PointerType, PrimType
from tvm.script import tir as T
from tvm.script.ir_builder import IRBuilder
from tvm.tirp.bench.utils import ProtonContext, bench, export_to_perfetto_trace

# cluster: [2, 1], cta_num = 2
# warpgroup:
#   wg_id = 0: ld & to GMEM
#   wg_id = 1:
#       warp_id = 0: mma
#       warp_id = 3: tma


def ceildiv(a, b):
    return (a + b - 1) // b


SM_NUMBER = 148


class ProfileEventType(Enum):
    IssueTMA = 0
    IssueMMA = 1
    TMEMLD = 2
    WRITEBACK = 3
    WAIT = 4
    INIT = 5


event_type_names = ["issue-tma", "issue-mma", "tmem-ld", "writeback", "wait", "init"]
NUM_GROUPS = 3
PROFILER_BUFFER_SIZE = int(2e6)
PROFILER_WRITE_STRIDE = SM_NUMBER * NUM_GROUPS
PROFILER_ON = False


def prepare_data(M, N, K):
    import torch

    A_bf16 = torch.randn((M, K), dtype=torch.float16)
    B_bf16 = torch.randn((N, K), dtype=torch.float16)
    # C_ref = torch.matmul(A_bf16, B_bf16.T)
    C_empty = torch.zeros((M, N), dtype=torch.float16)

    return A_bf16, B_bf16, C_empty


def flops(M, N, K, ms):
    return M * N * K * 2 / (ms * 1e-3)


def get_hgemm_kernel(dim_n, dim_k):
    M_CLUSTER = 1
    N_CLUSTER = 1
    WG_NUMBER = 2
    WARP_NUMBER = 4
    NUM_THREADS = (32 * WARP_NUMBER) * WG_NUMBER

    SMEM_PIPE_DEPTH = 6
    TMEM_PIPE_DEPTH = 2

    F16_BYTES = 2
    F32_BYTES = 4
    F128_BYTES = 16
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    N, K = dim_n, dim_k
    TILE_K = 4096
    BLK_M, BLK_N, BLK_K = 128, 128, 64
    MMA_M, MMA_N, MMA_K = 128, 128, 16
    PIPE_CIRCLE_NUM = (TILE_K // BLK_K) // SMEM_PIPE_DEPTH
    PIPE_REMAIN_NUM = (TILE_K // BLK_K) % SMEM_PIPE_DEPTH
    TILE_K_NUM = ceildiv(K, TILE_K)
    EPI_TILE = 32
    TMEM_LD_SIZE = 8
    N_COLS = 512
    CTA_GROUP = M_CLUSTER
    SWIZZLE = 3
    SMEM_SIZE = (
        SMEM_PIPE_DEPTH * BLK_M * BLK_K * F16_BYTES
        + SMEM_PIPE_DEPTH * BLK_N * BLK_K * F16_BYTES
        + TMEM_PIPE_DEPTH * EPI_TILE * MMA_N * F32_BYTES
        + 1024
    )

    assert SMEM_SIZE <= 232448
    assert TMEM_PIPE_DEPTH * MMA_N <= 512

    TILE_GROUPS_ROW_SIZE = 16
    # assert M % (BLK_M * CTA_GROUP) == 0
    assert N % (BLK_N * CTA_GROUP) == 0
    TILE_N_NUM = ceildiv(N, BLK_N * CTA_GROUP)

    class TileScheduler:

        def __init__(self, prefix: str, M):
            self.m_idx = T.local_cell("int32", name=prefix + "_m_idx")
            self.n_idx = T.local_cell("int32", name=prefix + "_n_idx")
            self.k_idx = T.local_cell("int32", name=prefix + "_k_idx")
            self.linear_idx = T.local_cell("int32", name=prefix + "_linear_idx")
            self.tile_idx = T.local_cell("int32", name="tile_idx")
            self.TILE_M_NUM = ceildiv(M, BLK_M * CTA_GROUP)

        @T.macro
        def update_current_m_n_idx(self, linear_idx):
            TILE_GROUPS_NUM = self.TILE_M_NUM // TILE_GROUPS_ROW_SIZE
            TILE_GROUPS_SIZE = TILE_GROUPS_ROW_SIZE * TILE_N_NUM * TILE_K_NUM
            TILE_FINAL_ROWS = self.TILE_M_NUM - (TILE_GROUPS_NUM * TILE_GROUPS_ROW_SIZE)
            if linear_idx < TILE_GROUPS_NUM * TILE_GROUPS_SIZE and TILE_GROUPS_NUM > 0:
                self.m_idx = linear_idx // TILE_GROUPS_SIZE * TILE_GROUPS_ROW_SIZE + (
                    linear_idx % TILE_GROUPS_ROW_SIZE
                )
                self.n_idx = (linear_idx // TILE_GROUPS_ROW_SIZE) % TILE_N_NUM
                self.k_idx = (linear_idx % TILE_GROUPS_SIZE) // (TILE_GROUPS_ROW_SIZE * TILE_N_NUM)
            elif TILE_FINAL_ROWS > 0:
                remainder_idx = linear_idx - TILE_GROUPS_SIZE * TILE_GROUPS_NUM
                self.m_idx = (
                    TILE_GROUPS_NUM * TILE_GROUPS_ROW_SIZE + remainder_idx % TILE_FINAL_ROWS
                )
                self.n_idx = (remainder_idx // TILE_FINAL_ROWS) % TILE_N_NUM
                self.k_idx = remainder_idx // (TILE_FINAL_ROWS * TILE_N_NUM)

        @T.macro
        def init(self, linear_init):
            self.linear_idx = linear_init
            self.tile_idx = 0
            self.update_current_m_n_idx(linear_init)

        @T.macro
        def next_tile(self):
            self.linear_idx = self.linear_idx + SM_NUMBER
            self.tile_idx += 1
            self.update_current_m_n_idx(self.linear_idx)

        def valid(self):
            return self.linear_idx < self.TILE_M_NUM * TILE_N_NUM * TILE_K_NUM

    atomic_add_system_uint64 = f"""
    __forceinline__ __device__ void atomic_add_system_uint64(uint64_t* addr, uint64_t value) {{
        asm volatile("red.async.release.global.gpu.add.u64 [%0], %1;" ::"l"(addr), "l"(value)
                    : "memory");
    }}
    """

    class Semaphore:
        def __init__(self, cnt, buffer):
            self.cnt = cnt
            self.sem = buffer
            self.state = T.alloc_buffer([1], "uint64", scope="local", align=4)
            IRBuilder.current().name("semaphore_state", self.state)

        @T.macro
        def semaphore_wait(self, *coord):
            with T.thread():
                while 1:
                    T.ptx.ld_global_acquire(
                        self.state[0],
                        self.sem.access_ptr("r", offset=self.sem.elem_offset_of(coord)),
                    )
                    if T.cuda.syncthreads_and(self.state[0] == self.cnt):
                        break
                    T.cuda.nano_sleep(40)

        @T.macro
        def semaphore_notify(self, tid, *coord):
            # wg is synced
            with T.thread():
                if tid % 128 == 0:
                    T.cuda.func_call(
                        "atomic_add_system_uint64",
                        self.sem.access_ptr("rw", offset=self.sem.elem_offset_of(coord)),
                        T.uint64(1),
                        source_code=atomic_add_system_uint64,
                    )

    class Barriers:

        def __init__(self, shared_buffer_base, shared_buffer_offs, pipe_depth, is_p2c):
            self.mbar: tvm.tir.Buffer = T.decl_buffer(
                (pipe_depth,), "uint64", shared_buffer_base, elem_offset=shared_buffer_offs
            ).buffer
            self.init_phase = 0 if is_p2c else 1
            self.pipe_depth = pipe_depth

        @T.macro
        def init(self, threads_num_wait):
            with T.thread()[0:1]:
                for i in T.serial(self.pipe_depth):
                    T.ptx.mbarrier.init(self.mbar.ptr_to([i]), threads_num_wait)

        @T.macro
        def wait(self, idx, phase):
            T.ptx.mbarrier.try_wait(self.mbar.ptr_to([idx]), self.init_phase ^ phase)

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
            T.ptx.tcgen05.commit(self.mbar.ptr_to([idx]), cta_group=CTA_GROUP)

    class BarMMA2TMA(Barriers):

        @T.macro
        def arrive(self, idx):
            T.ptx.tcgen05.commit(self.mbar.ptr_to([idx]), cta_group=CTA_GROUP)

    class BarLD2MMA(Barriers):

        @T.macro
        def arrive(self, idx):
            T.ptx.mbarrier.arrive(self.mbar.ptr_to([idx]), cta_id=0, pred=True)

    @T.macro
    def warp_sync():
        T.cuda.func_call(
            "sync_warp",
            source_code=f"""
    __forceinline__ __device__ void sync_warp() {{
        __syncwarp();
    }}
        """,
        )

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

    @T.macro
    def float22half2(dst, src):
        T.cuda.func_call(
            "float22half2",
            dst,
            src,
            source_code=f"""
    __forceinline__ __device__ void float22half2(void* dst, void* src) {{
        half2* dst_p = (half2*) dst;
        float2* src_p = (float2*) src;
        *dst_p = __float22half2_rn(*src_p);
    }}
        """,
        )

    A_layout = T.ComposeLayout(
        T.SwizzleLayout(3, 3, 3, swizzle_inner=True),
        T.TileLayout(shard=((SMEM_PIPE_DEPTH, BLK_M, BLK_K), (BLK_M * BLK_K, BLK_K, 1))),
    )
    B_layout = T.ComposeLayout(
        T.SwizzleLayout(3, 3, 3, swizzle_inner=True),
        T.TileLayout(shard=((SMEM_PIPE_DEPTH, BLK_N, BLK_K), (BLK_N * BLK_K, BLK_K, 1))),
    )
    D_layout = T.TileLayout(
        shard=((TMEM_PIPE_DEPTH, EPI_TILE, MMA_N), (EPI_TILE * MMA_N, MMA_N, 1))
    )

    # fmt: off
    @T.prim_func(tirp=True)
    def hgemm(A_ptr: T.handle, B: T.Buffer((N, K), b_type), partial_sum_ptr: T.handle):
        # profiler_buffer: T.Buffer((PROFILER_BUFFER_SIZE,), "uint64")):
        M = T.int32()
        A = T.match_buffer(A_ptr, [M, K], a_type)
        partial_sum = T.match_buffer(partial_sum_ptr, [TILE_K_NUM, M, N], "float32")
        A_tensor_map: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
        B_tensor_map: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
        partial_sum_tensor_map: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
        T.call_packed("runtime.cuTensorMapEncodeTiled", A_tensor_map, a_type, 2, A.data,
                      K, M, K * F16_BYTES, BLK_K, BLK_M, 1, 1, 0, SWIZZLE, 0, 0)
        T.call_packed("runtime.cuTensorMapEncodeTiled", B_tensor_map, b_type, 2, B.data, 
                      K, N, K * F16_BYTES, BLK_K, BLK_N, 1, 1, 0, SWIZZLE, 0, 0)
        T.call_packed("runtime.cuTensorMapEncodeTiled", partial_sum_tensor_map, "float32", 3, partial_sum.data,
                      N, M, TILE_K_NUM, N * F32_BYTES, N * M * F32_BYTES, MMA_N, EPI_TILE, 1, 1, 1, 1, 0, 0, 0, 0)
        with T.kernel():
            # cbx, cby = T.cta_id([M_CLUSTER, N_CLUSTER], parent="cluster")
            bx = T.cta_id([SM_NUMBER], parent="kernel")
            wg_id = T.warpgroup_id([WG_NUMBER], parent="cta")
            warp_id = T.warp_id([WARP_NUMBER], parent="warpgroup")
            lane_id = T.thread_id([32], parent="warp")
            tid = T.thread_id([NUM_THREADS], parent="cta")
            with T.cta():
                # alloc shared memory
                buf = T.alloc_buffer([SMEM_SIZE], "uint8", scope="shared.dyn")
                tmem_addr = T.decl_cell("uint32", buf.data, scope="shared.dyn", elem_offset=0)
                A_smem = T.decl_buffer((SMEM_PIPE_DEPTH, BLK_M, BLK_K), a_type, buf.data, layout=A_layout,
                                        elem_offset=1024 // F16_BYTES)
                B_smem = T.decl_buffer((SMEM_PIPE_DEPTH, BLK_N, BLK_K), b_type, buf.data, layout=B_layout,
                                        elem_offset=1024 // F16_BYTES + SMEM_PIPE_DEPTH * BLK_M * BLK_K)
                D_smem = T.decl_buffer((TMEM_PIPE_DEPTH, EPI_TILE, MMA_N), "float32", buf.data, layout=D_layout,
                                        elem_offset=1024 // F32_BYTES + SMEM_PIPE_DEPTH * (BLK_M + BLK_N) * BLK_K // 2)
               
                # alloc local memory
                reg = T.alloc_buffer((TMEM_LD_SIZE,), "float32", scope="local")
                stage = T.local_cell("int32", name="stage")
                phase = T.alloc_buffer((1, ), "int32", scope="local")
                descA = T.local_cell("uint64")
                descB = T.local_cell("uint64")
                descI = T.local_cell("uint32")
                profiler_write_offset = T.alloc_buffer([1], "uint32", scope="local", align=8)
                profiler_tag = T.alloc_buffer([1], "uint64", scope="local", align=8)
                # initialize
                tma2mma_bar = T.meta_var(BarTMA2MMA(buf.data, 6, SMEM_PIPE_DEPTH, True))
                mma2tma_bar = T.meta_var(BarMMA2TMA(buf.data, 6 + 2 * SMEM_PIPE_DEPTH, SMEM_PIPE_DEPTH, False))
                mma2ld_bar = T.meta_var(BarMMA2LD(buf.data, 6 + 3 * SMEM_PIPE_DEPTH, TMEM_PIPE_DEPTH, True))
                ld2mma_bar = T.meta_var(BarLD2MMA(buf.data, 6 + 3 * SMEM_PIPE_DEPTH + TMEM_PIPE_DEPTH, TMEM_PIPE_DEPTH, False))
                tile_scheduler = T.meta_var(TileScheduler("tile_scheduler", M))
                tma2mma_bar.init(1)
                mma2ld_bar.init(1)
                mma2tma_bar.init(1)
                ld2mma_bar.init(CTA_GROUP * 128)
                tile_scheduler.init(bx)
                # alloc TMEM
                with T.warp()[0:1]:
                    T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=N_COLS, cta_group=1)
                    warp_sync()
                
                # sync
                T.ptx.fence.proxy("shared")
                T.ptx.fence.mbarrier_init()
                T.tvm_storage_sync("shared")

                with T.cta():
                    T.block_attr({"tirp.scope_partition": True})
                    with T.warpgroup()[1:2]:
                        if warp_id == 3: 
                            if PROFILER_ON:
                                T.timer_init_cuda(profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, NUM_GROUPS, 0)
                            phase[0] = 0
                            while tile_scheduler.valid():
                                m_idx = T.meta_var(tile_scheduler.m_idx)
                                n_idx = T.meta_var(tile_scheduler.n_idx)
                                k_idx = T.meta_var(tile_scheduler.k_idx)
                                if T.ptx.elect_sync():
                                    # main inner tma loop
                                    k_offset = k_idx * TILE_K
                                    for ko in T.serial(PIPE_CIRCLE_NUM):
                                        for ks in T.unroll(SMEM_PIPE_DEPTH):
                                            # GMEM -> SMEM  (tma)    
                                            stage = (ko * SMEM_PIPE_DEPTH + ks)
                                            if PROFILER_ON:
                                                T.timer_start_cuda(ProfileEventType.WAIT, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id== 0)
                                            mma2tma_bar.wait(ks, phase[0])
                                            if PROFILER_ON:
                                                T.timer_end_cuda(ProfileEventType.WAIT, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id== 0)
                                                T.timer_start_cuda(ProfileEventType.IssueTMA, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id== 0)
                                            T.ptx.cp_async.bulk.tensor.g2c(2, A_smem.ptr_to([ks, 0, 0]), tma2mma_bar.mbar.ptr_to([ks]),
                                                                        A_tensor_map, stage * BLK_K + k_offset, m_idx * BLK_M, cta_group=CTA_GROUP, cache_hint="evict_last")
                                            T.ptx.cp_async.bulk.tensor.g2c(2, B_smem.ptr_to([ks, 0, 0]), tma2mma_bar.mbar.ptr_to([ks]),
                                                                        B_tensor_map, stage * BLK_K + k_offset, n_idx * BLK_N, cta_group=CTA_GROUP, cache_hint="evict_first")
                                            tma2mma_bar.arrive(ks, CTA_GROUP * BLK_K * (BLK_M + BLK_N) * F16_BYTES)
                                            if PROFILER_ON:
                                                T.timer_end_cuda(ProfileEventType.IssueTMA, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id== 0)
                                        phase[0] = phase[0] ^ 1

                                    if PIPE_REMAIN_NUM > 0:
                                        # last remained loop
                                        for ks in T.unroll(PIPE_REMAIN_NUM):
                                            # GMEM -> SMEM  (tma)    
                                            stage = PIPE_CIRCLE_NUM * SMEM_PIPE_DEPTH + ks
                                            if PROFILER_ON:
                                                T.timer_start_cuda(ProfileEventType.WAIT, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id== 0)
                                            mma2tma_bar.wait(ks, phase[0])
                                            if PROFILER_ON:
                                                T.timer_end_cuda(ProfileEventType.WAIT, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id== 0)
                                                T.timer_start_cuda(ProfileEventType.IssueTMA, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id== 0)
                                            T.ptx.cp_async.bulk.tensor.g2c(2, A_smem.ptr_to([ks, 0, 0]), tma2mma_bar.mbar.ptr_to([ks]),
                                                                        A_tensor_map, stage * BLK_K + k_offset, m_idx * BLK_M, cta_group=CTA_GROUP, cache_hint="evict_last")
                                            T.ptx.cp_async.bulk.tensor.g2c(2, B_smem.ptr_to([ks, 0, 0]), tma2mma_bar.mbar.ptr_to([ks]),
                                                                        B_tensor_map, stage * BLK_K + k_offset, n_idx * BLK_N, cta_group=CTA_GROUP, cache_hint="evict_first")
                                            tma2mma_bar.arrive(ks, CTA_GROUP * BLK_K * (BLK_M + BLK_N) * F16_BYTES)
                                            if PROFILER_ON:
                                                T.timer_end_cuda(ProfileEventType.IssueTMA, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id== 0)
                                        # for unaligned cases   
                                        for ks in T.unroll(PIPE_REMAIN_NUM, SMEM_PIPE_DEPTH):
                                            if PROFILER_ON:
                                                T.timer_start_cuda(ProfileEventType.WAIT, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id== 0)
                                            mma2tma_bar.wait(ks, phase[0])
                                            if PROFILER_ON:
                                                T.timer_end_cuda(ProfileEventType.WAIT, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id== 0)
                                            tma2mma_bar.arrive_only(ks)
                                    
                                        phase[0] = phase[0] ^ 1
                                tile_scheduler.next_tile()
                        
                        elif warp_id == 0:
                            if PROFILER_ON:
                                T.timer_init_cuda(profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, NUM_GROUPS, 1)
                            tmem_idx = T.local_cell("int32", "tmem_idx")
                            tmem_phase = T.local_cell("int32", "tmem_phase")
                            T.ptx.tcgen05.encode_instr_descriptor(T.address_of(descI), "float32", a_type, b_type, MMA_N, MMA_M, MMA_K, False, False, CTA_GROUP)
                            phase[0] = 0
                            while tile_scheduler.valid():
                                m_idx = T.meta_var(tile_scheduler.m_idx)
                                n_idx = T.meta_var(tile_scheduler.n_idx)
                                k_idx = T.meta_var(tile_scheduler.k_idx)
                                if T.ptx.elect_sync():
                                    tmem_idx = tile_scheduler.tile_idx % TMEM_PIPE_DEPTH
                                    tmem_phase = (tile_scheduler.tile_idx // TMEM_PIPE_DEPTH) & 1

                                    # wait for the tmem result to be consumed
                                    ld2mma_bar.wait(tmem_idx, tmem_phase)
                                    T.ptx.tcgen05.fence.after_thread_sync()

                                    # main inner mma loop
                                    for ko in T.serial(PIPE_CIRCLE_NUM):
                                        for ks in T.unroll(SMEM_PIPE_DEPTH):
                                        
                                            # wait tma and sf-transpose arrival
                                            tma2mma_bar.wait(ks, phase[0])
                                            if PROFILER_ON:
                                                T.timer_start_cuda(ProfileEventType.IssueMMA, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id== 0)
                                            T.ptx.tcgen05.fence.after_thread_sync()

                                            # issue mma                         
                                            for ki in T.unroll(BLK_K // MMA_K):
                                                T.ptx.tcgen05.encode_matrix_descriptor(T.address_of(descA), A_smem.ptr_to([ks, 0, ki * MMA_K]), 
                                                                                    ldo=1, sdo=8 * BLK_K * F16_BYTES // F128_BYTES, swizzle=SWIZZLE)
                                                T.ptx.tcgen05.encode_matrix_descriptor(T.address_of(descB), B_smem.ptr_to([ks, 0, ki * MMA_K]), 
                                                                                    ldo=1, sdo=8 * BLK_K * F16_BYTES // F128_BYTES, swizzle=SWIZZLE)          
                                                            
                                                if ko == 0 and ks == 0 and ki == 0:
                                                    T.ptx.tcgen05.mma("float32", a_type, b_type, tmem_idx * MMA_M, descB, descA, descI, False, CTA_GROUP, False)
                                                else:
                                                    T.ptx.tcgen05.mma("float32", a_type, b_type, tmem_idx * MMA_M, descB, descA, descI, False, CTA_GROUP, True)
                                            mma2tma_bar.arrive(ks)
                                            if PROFILER_ON:
                                                T.timer_end_cuda(ProfileEventType.IssueMMA, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id==0)
                                        phase[0] = phase[0] ^ 1

                                    if PIPE_REMAIN_NUM > 0:
                                        # last remained loop
                                        for ks in T.unroll(PIPE_REMAIN_NUM):

                                            # wait tma and sf-transpose arrival
                                            tma2mma_bar.wait(ks, phase[0])
                                            T.ptx.tcgen05.fence.after_thread_sync()

                                            # issue mma                        
                                            for ki in T.unroll(BLK_K // MMA_K):
                                                T.ptx.tcgen05.encode_matrix_descriptor(T.address_of(descA), A_smem.ptr_to([ks, 0, ki * MMA_K]), 
                                                                                    ldo=1, sdo=8 * BLK_K * F16_BYTES // F128_BYTES, swizzle=SWIZZLE)
                                                T.ptx.tcgen05.encode_matrix_descriptor(T.address_of(descB), B_smem.ptr_to([ks, 0, ki * MMA_K]), 
                                                                                    ldo=1, sdo=8 * BLK_K * F16_BYTES // F128_BYTES, swizzle=SWIZZLE)                            
                                                if PIPE_CIRCLE_NUM == 0 and ks == 0 and ki == 0:
                                                    T.ptx.tcgen05.mma("float32", a_type, b_type, tmem_idx * MMA_M, descB, descA, descI, False, CTA_GROUP, True)
                                                else:
                                                    T.ptx.tcgen05.mma("float32", a_type, b_type, tmem_idx * MMA_M, descB, descA, descI, False, CTA_GROUP, True)
                                            mma2tma_bar.arrive(ks)

                                        # ensure that all mma is issued
                                        mma2ld_bar.arrive(tmem_idx)

                                        # for unaligned cases   
                                        for ks in T.unroll(PIPE_REMAIN_NUM, SMEM_PIPE_DEPTH):
                                            tma2mma_bar.wait(ks, phase[0])
                                            mma2tma_bar.arrive(ks)
                                        
                                        phase[0] = phase[0] ^ 1
                                    else:
                                        # ensure that all mma is issued
                                        mma2ld_bar.arrive(tmem_idx)

                                tile_scheduler.next_tile()
                                    
                    with T.warpgroup()[0:1]:
                        if PROFILER_ON:
                            T.timer_init_cuda(profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, NUM_GROUPS, 2)
                        trap_when_assert_failed(tmem_addr == 0)
                        tmem_idx = T.local_cell("int32", "tmem_idx")
                        tmem_phase = T.local_cell("int32", "tmem_phase")
                        phase[0] = 0
                        while tile_scheduler.valid():
                            m_idx = T.meta_var(tile_scheduler.m_idx)
                            n_idx = T.meta_var(tile_scheduler.n_idx)
                            k_idx = T.meta_var(tile_scheduler.k_idx)
                            tmem_idx = tile_scheduler.tile_idx % TMEM_PIPE_DEPTH
                            tmem_phase = (tile_scheduler.tile_idx // TMEM_PIPE_DEPTH) & 1

                            # flush previous tma
                            # wait for the completion of all the mma of the same tile
                            mma2ld_bar.wait(tmem_idx, tmem_phase)
                            T.ptx.tcgen05.fence.after_thread_sync()
                            
                            for ko in T.unroll(MMA_M // EPI_TILE):
                                stage = (tile_scheduler.tile_idx * MMA_M // EPI_TILE + ko) % TMEM_PIPE_DEPTH
                                # wait the smem to be free
                                if ko >= TMEM_PIPE_DEPTH:
                                    if lane_id == 0 and warp_id == 0:
                                        T.ptx.cp_async.bulk.wait_group(TMEM_PIPE_DEPTH - 1)
                                    T.ptx.bar.sync(10, 128)
                                if PROFILER_ON:
                                    T.timer_start_cuda(ProfileEventType.TMEMLD, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id== 0 and warp_id == 0)

                                # tmem -> rf (ld) -> smem
                                for ki in T.unroll(EPI_TILE // TMEM_LD_SIZE):
                                    T.ptx.tcgen05.ld(0 + tmem_idx * MMA_M + ko * EPI_TILE, 
                                                     warp_id * 32, ki * TMEM_LD_SIZE, "32x32b", 
                                                     TMEM_LD_SIZE, False, *[reg[j] for j in range(TMEM_LD_SIZE)])
                                    T.ptx.tcgen05.wait.ld()
                            
                                    for vec in range(TMEM_LD_SIZE):
                                        D_smem[stage, ki * TMEM_LD_SIZE + vec,  warp_id * 32 + lane_id] = reg[vec]
                                if PROFILER_ON:
                                    T.timer_end_cuda(ProfileEventType.TMEMLD, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id== 0 and warp_id == 0)
                                # the tmem can be overwritten
                                if ko == MMA_M // EPI_TILE - 1:
                                    T.ptx.tcgen05.fence.before_thread_sync()
                                    ld2mma_bar.arrive(tmem_idx)

                                T.ptx.fence.proxy(scope="shared")
                                T.ptx.bar.sync(10, 128)
                                if PROFILER_ON:
                                    T.timer_start_cuda(ProfileEventType.WRITEBACK, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id== 0 and warp_id == 0)
                                # smem -> gmem
                                if lane_id == 0 and warp_id == 0:
                                    T.ptx.cp_async.bulk.tensor.s2g(3, D_smem.ptr_to([stage, 0, 0]), 
                                                                partial_sum_tensor_map, n_idx* BLK_N,
                                                                m_idx * BLK_M + ko * EPI_TILE, k_idx, cache_hint="evict_last")
                                    T.ptx.cp_async.bulk.commit_group()
                                if PROFILER_ON:
                                    T.timer_end_cuda(ProfileEventType.WRITEBACK, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id== 0 and warp_id == 0)
                            if lane_id == 0 and warp_id == 0:
                                T.ptx.cp_async.bulk.wait_group(0)
                            T.ptx.bar.sync(10, 128)
                            tile_scheduler.next_tile()
                      
                # dealloc TMEM
                with T.warp()[0:1]:
                    T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
                    T.ptx.tcgen05.dealloc(tmem_addr, n_cols=N_COLS, cta_group=1)

                T.tvm_storage_sync("shared")
    # fmt: on

    VEC_SIZE = math.gcd(16 // F32_BYTES, N)
    NUM_THREADS = 512
    BDX = 32
    BDY = NUM_THREADS // BDX

    # fmt: off
    @T.prim_func(tirp=True)
    def reduce(partial_sum_ptr: T.handle,
               D_ptr: T.handle):
        M = T.int32()
        partial_sum = T.match_buffer(partial_sum_ptr, [TILE_K_NUM, M, N], "float32")
        D = T.match_buffer(D_ptr, [M, N], d_type)
        with T.kernel():
            tx, ty = T.thread_id([BDX, BDY], parent="cta")
            bx = T.cta_id([SM_NUMBER], parent="kernel")

            with T.thread():
                idx = T.alloc_local([1], "int32")
                vec_32 = T.alloc_local([VEC_SIZE], "float32")
                tmp = T.alloc_local([VEC_SIZE], "float32")
                vec_16 = T.alloc_local([VEC_SIZE], "float16") 
                
                idx[0] = bx * NUM_THREADS * VEC_SIZE + (ty * BDX + tx) * VEC_SIZE
                while idx[0] < M * N:
                    m_idx = T.meta_var(idx[0] // N)
                    n_idx = T.meta_var(idx[0] % N)
                    for kv in T.unroll(VEC_SIZE):
                        vec_32[kv] = 0.0
                    for kt in T.serial(TILE_K_NUM):
                        for kv in T.vectorized(VEC_SIZE):
                            tmp[kv] = partial_sum[kt, m_idx, n_idx + kv]
                        for kv in T.unroll(VEC_SIZE):
                            vec_32[kv] += tmp[kv]
                    for kv in T.unroll(VEC_SIZE // 2):
                        float22half2(T.address_of(vec_16[kv * 2]), T.address_of(vec_32[kv * 2]))
                        # vec_16[kv] = T.cast(vec_32[kv], "float16")
                    for kv in T.vectorized(VEC_SIZE):
                        D[m_idx, n_idx + kv] = vec_16[kv]                    
                    idx[0] += SM_NUMBER * NUM_THREADS * VEC_SIZE
    # fmt: on
    return hgemm, reduce, TILE_K_NUM


@tvm.testing.requires_cuda_compute_version(10, exact=True)
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16, 32, 64, 128])
def test(batch_size):
    N, K = 8192, 8192
    A_bf16, B_bf16, C_bf16 = prepare_data(batch_size, N, K)

    def tir_gemm(A_bf16, B_bf16, C_bf16):
        hgemm, reduce, TILE_K_NUM = get_hgemm_kernel(N, K)
        DEV = tvm.cuda(0)
        A_tvm = tvm.runtime.tensor(A_bf16, device=DEV)
        B_tvm = tvm.runtime.tensor(B_bf16, device=DEV)
        C_tvm = tvm.runtime.tensor(C_bf16, device=DEV)
        partial_sum_tvm = tvm.runtime.tensor(
            np.zeros((TILE_K_NUM, batch_size, N), dtype=np.float32), device=DEV
        )
        if PROFILER_ON:
            profiler_buffer = np.zeros((PROFILER_BUFFER_SIZE,), dtype=np.uint64)
            profiler_buffer_tvm = tvm.runtime.tensor(profiler_buffer, DEV)
        target = tvm.target.Target("cuda")
        with target:
            mod_hgemm = tvm.ir.IRModule({"main": hgemm})
            mod_reduce = tvm.ir.IRModule({"main": reduce})
            mod_hgemm = tvm.compile(mod_hgemm, target=target, tir_pipeline="tirp")
            mod_reduce = tvm.compile(mod_reduce, target=target, tir_pipeline="tirp")

            def func():
                if PROFILER_ON:
                    mod_hgemm(A_tvm, B_tvm, partial_sum_tvm, profiler_buffer_tvm)
                else:
                    mod_hgemm(A_tvm, B_tvm, partial_sum_tvm)
                mod_reduce(partial_sum_tvm, C_tvm)

            ms = bench(func, warmup=10, repeat=30, proton_name="tir")
            print(f"TIR flops: {flops(batch_size, N, K, ms) / 1e12} TFLOPS, time: {ms:.3f} ms")
            if PROFILER_ON:
                export_to_perfetto_trace(
                    profiler_buffer_tvm.numpy(),
                    f"hgemm-{batch_size}-{N}-{K}-1consumer-1cta.perfetto-trace",
                    event_type_names,
                )

        return partial_sum_tvm.numpy(), C_tvm.numpy()

    def cublas_gemm(A_bf16, B_bf16):
        import torch

        torch_dev = torch.device("cuda")
        A_torch = A_bf16.to(torch_dev)
        B_torch = B_bf16.to(torch_dev)
        func = lambda: torch.matmul(A_torch, B_torch.T)
        C_torch = func()
        ms = bench(func, warmup=10, repeat=30, proton_name="cublas")
        print(f"CUBLAS flops: {flops(batch_size, N, K, ms) / 1e12} TFLOPS, time: {ms:.3f} ms")
        return C_torch.cpu().numpy()

    with ProtonContext("blackwell_gemm"):
        C_cublas = cublas_gemm(A_bf16, B_bf16)
        partial_sum_tvm, C_tvm = tir_gemm(A_bf16, B_bf16, C_bf16)

    np.testing.assert_allclose(
        partial_sum_tvm.sum(axis=0).astype(np.float16), C_cublas, rtol=1e-3, atol=1e-2
    )
    np.testing.assert_allclose(C_tvm, C_cublas, rtol=1e-3, atol=1e-2)


if __name__ == "__main__":
    for batch_size in [8192]:
        test(batch_size)
