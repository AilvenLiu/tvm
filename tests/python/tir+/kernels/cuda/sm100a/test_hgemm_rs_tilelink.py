# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import tempfile
from enum import Enum

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.ir.type import PointerType, PrimType
from tvm.runtime import ShapeTuple
from tvm.runtime import disco as di
from tvm.script import tir as T
from tvm.script.ir_builder import IRBuilder
from tvm.tirp.bench.utils import export_to_perfetto_trace


class ProfileEventType(Enum):
    GEMM = 0
    RS = 1
    WAIT = 2
    ACCUM = 3
    PUT = 4
    SIGNAL = 5
    LOAD = 6


event_type_names = [
    "gemm",
    "rs",
    "wait",
    "accum",
    "put",
    "signal",
    "load",
]


d_type, a_type, b_type = "float16", "float16", "float16"
nbytes = 2
WORLD_SIZE = 8
M, N, K = 16384, 12288, 49152 // WORLD_SIZE
TOTAL_K = K * WORLD_SIZE
LOCAL_M = M // WORLD_SIZE
BLK_M, BLK_N, BLK_K = 128, 256, 64
assert LOCAL_M * WORLD_SIZE == M, "M must be divisible by WORLD_SIZE"
assert LOCAL_M % BLK_M == 0, "LOCAL_M must be divisible by BLK_M"
MMA_M, MMA_N, MMA_K = 256, 256, 16
GROUP_SIZE = 8
SM_COUNT = 148
NUM_THREADS = 32 * 4 * 3
N_COLS = 512
REPEAT_NUM = 1
EPI_TILE = 64
PIPE_DEPTH = 4
NUM_CONSUMER = 2
SMEM_SIZE = (
    PIPE_DEPTH * NUM_CONSUMER * BLK_K * BLK_M
    + BLK_K * BLK_N // 2 * PIPE_DEPTH
    + NUM_CONSUMER * BLK_M * EPI_TILE
) * 2 + 1024
TMEM_LD_SIZE = 128
CLUSTER_M, CLUSTER_N = 2, 1
SWIZZLE = 3
cta_group = 2
ldo, sdo = 1, 64

# special parameters for RS
RS_SMS = 20
GEMM_SMS = SM_COUNT - RS_SMS
N_REPEAT = 15
VEC_SIZE = 8
RS_PIPE_DEPTH = 6
M_SEG = 4
RS_BLK_M = BLK_M // M_SEG
RS_BLK_N = BLK_N

# profiling
NUM_GROUPS = 5
PROFILER_BUFFER_SIZE = int(1e6)
PROFILER_WRITE_STRIDE = SM_COUNT * NUM_GROUPS


class GemmTileScheduler:
    def __init__(self, prefix: str, m_clusters: int, n_clusters: int, group_size: int):
        self.m_idx = T.local_cell("int32", name=prefix + "_m_idx")
        self.n_idx = T.local_cell("int32", name=prefix + "_n_idx")
        self.linear_idx = T.local_cell("int32", name=prefix + "_linear_idx")
        self.m_clusters = m_clusters
        self.n_clusters = n_clusters
        self.my_rank = T.nvshmem.my_pe()
        assert self.m_clusters % WORLD_SIZE == 0
        self.group_size = group_size

    @T.macro
    def update_current_m_n_idx(self, linear_idx):
        remote_m_clusters = self.m_clusters - self.m_clusters // WORLD_SIZE
        group_rows = ((remote_m_clusters) // self.group_size) * self.group_size
        final_rows = remote_m_clusters - group_rows
        group_repeat = self.group_size * self.n_clusters
        # FIXME: use group_rows > 0 to avoid constant folding bug
        if linear_idx < group_rows * self.n_clusters and group_rows > 0:
            self.m_idx = (
                linear_idx // group_repeat * self.group_size
                + (linear_idx % self.group_size)
                + (self.my_rank + 1) * self.m_clusters // WORLD_SIZE
            ) % self.m_clusters
            self.n_idx = linear_idx % group_repeat // self.group_size
        elif linear_idx < remote_m_clusters * self.n_clusters:
            remainder_idx = linear_idx - group_rows * self.n_clusters
            self.m_idx = (
                group_rows
                + remainder_idx % final_rows
                + (self.my_rank + 1) * self.m_clusters // WORLD_SIZE
            ) % self.m_clusters
            self.n_idx = remainder_idx // final_rows
        else:
            remainder_idx = linear_idx - remote_m_clusters * self.n_clusters
            self.m_idx = (
                remote_m_clusters
                + remainder_idx % (self.m_clusters // WORLD_SIZE)
                + (self.my_rank + 1) * self.m_clusters // WORLD_SIZE
            ) % self.m_clusters
            self.n_idx = remainder_idx // (self.m_clusters // WORLD_SIZE)

    @T.macro
    def init(self, linear_init):
        self.linear_idx = linear_init
        self.update_current_m_n_idx(linear_init)

    @T.macro
    def next_tile(self, stride: int):
        self.linear_idx = self.linear_idx + stride
        self.update_current_m_n_idx(self.linear_idx)

    def valid(self):
        return self.linear_idx < self.m_clusters * self.n_clusters


class RSTileScheduler:
    def __init__(self, prefix: str, m_clusters: int, n_clusters: int, group_size: int):
        self.m_idx = T.local_cell("int32", name=prefix + "_m_idx")
        self.n_idx = T.local_cell("int32", name=prefix + "_n_idx")
        self.linear_idx = T.local_cell("int32", name=prefix + "_linear_idx")
        self.m_clusters = m_clusters
        self.n_clusters = n_clusters
        self.group_size = group_size

    @T.macro
    def update_current_m_n_idx(self, linear_idx):
        group_rows = (self.m_clusters // self.group_size) * self.group_size
        final_rows = self.m_clusters - group_rows
        group_repeat = self.group_size * self.n_clusters
        # FIXME: use group_rows > 0 to avoid constant folding bug
        if linear_idx < group_rows * self.n_clusters and group_rows > 0:
            self.m_idx = linear_idx // group_repeat * self.group_size + (
                linear_idx % self.group_size
            )
            self.n_idx = linear_idx % group_repeat // self.group_size
        elif final_rows > 0:
            remainder_idx = linear_idx - group_rows * self.n_clusters
            self.m_idx = group_rows + remainder_idx % final_rows
            self.n_idx = remainder_idx // final_rows

    @T.macro
    def init(self, linear_init):
        self.linear_idx = linear_init
        self.update_current_m_n_idx(linear_init)

    @T.macro
    def next_tile(self, stride: int):
        self.linear_idx = self.linear_idx + stride
        self.update_current_m_n_idx(self.linear_idx)

    def valid(self):
        return self.linear_idx < self.m_clusters * self.n_clusters


wait_str = """
__forceinline__ __device__ void wait(void* sem_addr, int value) {
    int state = -1;
    while (true) {
        if (threadIdx.x % 32 == 0) {
            asm volatile ("ld.global.acquire.gpu.b32 %0, [%1];\\n" : "=r"(state) : "l"(sem_addr));
        }
        if (__any_sync(0xffffffff, state == value)) {
            break;
        }
        __nanosleep(40);
    }
}

"""


class Semaphore:
    def __init__(self, cnt, buffer):
        self.cnt = cnt
        self.sem = buffer
        self.state = T.alloc_buffer([1], "int32", scope="local", align=4)
        IRBuilder.current().name("semaphore_state", self.state)

    @T.macro
    def semaphore_wait(self, *coord):
        T.cuda.func_call("wait", self.sem.ptr_to(coord), self.cnt, source_code=wait_str)

    @T.macro
    def semaphore_notify(self, cbx, wg_id, tid, m_idx, n_idx):
        # wg is synced
        with T.thread():
            if tid % 128 == 0:
                T.cuda.atomic_add(
                    self.sem.access_ptr(
                        "rw", offset=self.sem.offset_of_p((m_idx * 4 + wg_id * 2 + cbx, n_idx))
                    ),
                    1,
                )
            T.cuda.thread_fence()


spin_wait = """
__forceinline__ __device__ void spin_wait(uint32_t *flag_addr) {
    uint32_t flag;
    do {
        asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(flag) : "l"(flag_addr));
    } while (flag == 0);
}
"""


signal_fp16_ptx = """
__device__ __forceinline__ void signal_fp16(
    void* flag_addr, int dst_rank, uint32_t seq)
{
    __threadfence_system();
    uint32_t *remote_flag = (uint32_t*) nvshmem_ptr(flag_addr, dst_rank);
    asm volatile("st.release.sys.global.u32 [%1], %0;"
                    :: "r"(seq), "l"(remote_flag) : "memory");
}
"""

write_remote_tma = """
__device__ __forceinline__ void write_remote_tma(
    void* dst_addr, const void* src_addr,
    int dst_rank, int BLK_M, int BLK_N)
{
    int4* r = (int4*) nvshmem_ptr(dst_addr, dst_rank);
    int4* l = (int4*) src_addr;
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" :: "l"(r), "l"(l), "r"(BLK_M * BLK_N * 2));
}
"""


class Pipeline:
    def __init__(
        self,
        shared_buf,
        base_offset,
        pipeline_depth: int,
        pipeline_num: int,
        p_single_cta: bool = False,
        c_single_cta: bool = False,
    ):
        self.pipeline_depth = pipeline_depth
        self.pipeline_num = pipeline_num
        self.mbar_p2c = T.decl_buffer(
            (pipeline_depth, pipeline_num), "uint64", shared_buf, elem_offset=base_offset
        ).buffer
        self.mbar_c2p = T.decl_buffer(
            (pipeline_depth, pipeline_num),
            "uint64",
            shared_buf,
            elem_offset=base_offset + pipeline_depth * pipeline_num,
        ).buffer
        self.idx = T.local_cell("int32", name="pipeline_idx")
        self.p2c_phase = T.local_cell("int32", name="pipeline_p2c_phase")
        self.c2p_phase = T.local_cell("int32", name="pipeline_c2p_phase")
        self.p_single_cta = p_single_cta
        self.c_single_cta = c_single_cta

    @T.macro
    def init(self, p2c_thread_count: int = 1, c2p_thread_count: int = 1):
        self.idx = 0
        self.p2c_phase = 0
        self.c2p_phase = 1
        with T.thread()[0:1]:
            for cbx in T.thread_binding(CLUSTER_M, "clusterCtaIdx.x"):
                for i in T.serial(0, self.pipeline_depth):
                    for j in T.serial(0, self.pipeline_num):
                        if not self.c_single_cta or cbx == 0:
                            T.ptx.mbarrier.init(self.mbar_p2c.ptr_to([i, j]), p2c_thread_count)
                        if not self.p_single_cta or cbx == 0:
                            T.ptx.mbarrier.init(self.mbar_c2p.ptr_to([i, j]), c2p_thread_count)
        T.ptx.fence.proxy("shared")

    @T.macro
    def advance(self):
        self.idx = (self.idx + 1) % self.pipeline_depth
        if self.idx == 0:
            self.p2c_phase = self.p2c_phase ^ 1
            self.c2p_phase = self.c2p_phase ^ 1

    @T.macro
    def producer_wait(self, pipeline_idx):
        for cbx in T.thread_binding(CLUSTER_M, "clusterCtaIdx.x"):
            if not self.p_single_cta or cbx == 0:
                T.ptx.mbarrier.try_wait(
                    self.mbar_c2p.ptr_to([self.idx, pipeline_idx]), self.c2p_phase
                )

    @T.macro
    def consumer_wait(self, pipeline_idx):
        for cbx in T.thread_binding(CLUSTER_M, "clusterCtaIdx.x"):
            if not self.c_single_cta or cbx == 0:
                T.ptx.mbarrier.try_wait(
                    self.mbar_p2c.ptr_to([self.idx, pipeline_idx]), self.p2c_phase
                )


class TMA2MMAPipeline(Pipeline):

    @T.macro
    def consumer_release(self, pipeline_idx):
        for cbx in T.thread_binding(CLUSTER_M, "clusterCtaIdx.x"):
            for tx in T.thread_binding(NUM_THREADS, "threadIdx.x"):
                if tx % 32 == 0:
                    if not self.c_single_cta:
                        T.ptx.tcgen05.commit(self.mbar_c2p.ptr_to([self.idx, pipeline_idx]), 1)
                    elif cbx == 0:
                        T.ptx.tcgen05.commit(
                            self.mbar_c2p.ptr_to([self.idx, pipeline_idx]), 2, cta_mask=3
                        )


class MMA2LDpipeline(Pipeline):
    @T.macro
    def consumer_release(self, pipeline_idx):
        for cbx in T.thread_binding(CLUSTER_M, "clusterCtaIdx.x"):
            if not self.c_single_cta or cbx == 0:
                T.ptx.mbarrier.arrive(
                    self.mbar_c2p.ptr_to([self.idx, pipeline_idx]), cta_id=0, pred=True
                )


class ReducePipe(Pipeline):

    @T.macro
    def consumer_release(self, pipeline_idx: int):
        T.ptx.mbarrier.arrive(self.mbar_c2p.ptr_to([self.idx, pipeline_idx]))


@pytest.mark.skip()
@tvm.testing.requires_cuda_compute_version(10, exact=True)
def test_hgemm_rs():
    A_layout = T.ComposeLayout(
        T.SwizzleLayout(3, 3, 3, swizzle_inner=True),
        T.TileLayout(
            shard=(
                (PIPE_DEPTH, NUM_CONSUMER, BLK_M, 1, 64),
                (BLK_M * 64 * NUM_CONSUMER, BLK_M * 64, 64, BLK_M * 64, 1),
            )
        ),
    )
    B_layout = T.ComposeLayout(
        T.SwizzleLayout(3, 3, 3, swizzle_inner=True),
        T.TileLayout(
            shard=((PIPE_DEPTH, BLK_N // 2, 1, 64), (BLK_N // 2 * 64, 64, BLK_N // 2 * 64, 1))
        ),
    )
    D_layout = T.ComposeLayout(
        T.SwizzleLayout(3, 3, 3, swizzle_inner=True),
        T.TileLayout(shard=((NUM_CONSUMER, BLK_M, EPI_TILE), (BLK_M * EPI_TILE, EPI_TILE, 1))),
    )

    # fmt: off
    @T.prim_func(tirp=True)
    def test_mma_ss_tma_2sm_persistent(A: T.Buffer((M, K), a_type, layout="default"), B: T.Buffer((N, K), b_type, layout="default"), gemm_out: T.Buffer((M, N), d_type, layout="default"),
                                       semaphore: T.Buffer((M // BLK_M, N // BLK_N), "int32", layout="default"), buffer: T.Buffer((M // RS_BLK_M, N // RS_BLK_N, RS_BLK_M, RS_BLK_N), d_type, layout="default"),
                                       sig_addr: T.Buffer((WORLD_SIZE, SM_COUNT), "uint32", layout="default"), out: T.Buffer((LOCAL_M, N), d_type, layout="default"), profiler_buffer: T.Buffer((PROFILER_BUFFER_SIZE,), "uint64")):
        A_tensor_map: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
        B_tensor_map: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
        C_tensor_map: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
        buffer_tensor_map: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
        gemm_out_tensor_map: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
        T.call_packed("runtime.cuTensorMapEncodeTiled", A_tensor_map, "float16", 2, A.data, K, M, K * 2, BLK_K, BLK_M, 1, 1, 0, 3, 0, 0)
        T.call_packed("runtime.cuTensorMapEncodeTiled", B_tensor_map, "float16", 2, B.data, K, N, K * 2, BLK_K, BLK_N // 2, 1, 1, 0, 3, 0, 0)
        T.call_packed("runtime.cuTensorMapEncodeTiled", C_tensor_map, "float16", 2, gemm_out.data, N, M, N * 2, EPI_TILE, BLK_M, 1, 1, 0, 3, 0, 0)
        T.call_packed("runtime.cuTensorMapEncodeTiled", buffer_tensor_map, "float16", 4, buffer.data, RS_BLK_N, RS_BLK_M, N // RS_BLK_N, M // RS_BLK_M, RS_BLK_N * 2, RS_BLK_N * RS_BLK_M * 2, N * RS_BLK_M * 2, RS_BLK_N, RS_BLK_M, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0)
        T.call_packed("runtime.cuTensorMapEncodeTiled", gemm_out_tensor_map, "float16", 2, gemm_out.data, N, M, N * 2, RS_BLK_N, RS_BLK_M, 1, 1, 0, 0, 0, 0)
        with T.kernel():
            cbx, cby = T.cta_id([CLUSTER_M, CLUSTER_N], parent="cluster")
            bx = T.cta_id([SM_COUNT], parent="kernel")
            wg_id = T.warpgroup_id([NUM_CONSUMER+1], parent="cta")
            warp_id = T.warp_id([4], parent="warpgroup")
            lane_id = T.thread_id([32], parent="warp")
            tid = T.thread_id([NUM_THREADS], parent="cta")
            tid_in_wg = T.thread_id([128], parent="warpgroup")
            rank = T.nvshmem.my_pe()
            sem = T.meta_var(Semaphore(cnt=1, buffer=semaphore))
            with T.cta():
                buf = T.alloc_buffer([SMEM_SIZE], "uint8", scope="shared.dyn")
                profiler_write_offset = T.alloc_buffer([1], "uint32", scope="local", align=8)
                profiler_tag = T.alloc_buffer([1], "uint64", scope="local", align=8)
                
                reg_fp16 = T.alloc_buffer((BLK_N,), d_type, scope="local")
                if bx < GEMM_SMS:
                    T.timer_init_cuda(profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, NUM_GROUPS, 0)
                    T.timer_start_cuda(ProfileEventType.GEMM, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0)
                    tmem_addr = T.decl_cell("uint32", buf.data, scope="shared.dyn", elem_offset=0)
                    A_smem = T.decl_buffer((PIPE_DEPTH, NUM_CONSUMER,BLK_M, BLK_K), a_type, buf.data, elem_offset=512, layout=A_layout)
                    B_smem = T.decl_buffer((PIPE_DEPTH, BLK_N // 2, BLK_K), b_type, buf.data, elem_offset=512 + BLK_K * BLK_M * NUM_CONSUMER * PIPE_DEPTH, layout=B_layout)
                    C_smem = T.decl_buffer((NUM_CONSUMER, BLK_M, EPI_TILE), d_type, buf.data, elem_offset=512 + BLK_K * BLK_M * NUM_CONSUMER * PIPE_DEPTH + BLK_K * BLK_N // 2 * PIPE_DEPTH, layout=D_layout)
                    reg = T.alloc_buffer((TMEM_LD_SIZE,), "float32", scope="local")
                    descA = T.local_cell("uint64")
                    descB = T.local_cell("uint64")
                    descI = T.local_cell("uint32")
                    base_desc_A = T.local_cell("uint64")
                    base_desc_B = T.local_cell("uint64")
                    tma2mma_pipe = T.meta_var(TMA2MMAPipeline(buf.data, 1, PIPE_DEPTH, 1, p_single_cta=False, c_single_cta=True))
                    mma2ld_pipe = T.meta_var(MMA2LDpipeline(buf.data, 1 + PIPE_DEPTH * 2, 1, NUM_CONSUMER, p_single_cta=True, c_single_cta=False))
                    mma2ld_pipe.init(c2p_thread_count=128 * 2, p2c_thread_count=2)
                    tma2mma_pipe.init(c2p_thread_count=NUM_CONSUMER)
                    ptr: T.Var(name="ptr", dtype=PointerType(PrimType("uint64"))) = T.reinterpret("handle", T.ptx.map_shared_rank(tma2mma_pipe.mbar_p2c.ptr_to([0, 0]), 0))
                    tma_finished = T.decl_buffer([PIPE_DEPTH], "uint64", data=ptr, scope="shared")
                    m_clusters = T.meta_var((M + BLK_M - 1) // BLK_M // CLUSTER_M // NUM_CONSUMER)
                    n_clusters = T.meta_var((N + BLK_N - 1) // BLK_N // CLUSTER_N)
                    gemm_tile_scheduler = T.meta_var(GemmTileScheduler("gemm_tile_scheduler", m_clusters=m_clusters, n_clusters=n_clusters, group_size=GROUP_SIZE))
                    gemm_tile_scheduler.init(bx//2)
                    # alloc TMEM
                    with T.warp()[0:1]:
                        T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=N_COLS, cta_group=cta_group)
                    T.ptx.tcgen05.encode_instr_descriptor(T.address_of(descI), "float32", a_type, b_type, MMA_M, MMA_N, MMA_K, trans_a=False, trans_b=False, n_cta_groups=cta_group)
                    T.tvm_storage_sync("shared")
                    # reset RF
                    with T.cta():
                        T.block_attr({"tirp.scope_partition": True})
                        with T.warpgroup()[NUM_CONSUMER:NUM_CONSUMER + 1]:
                            T.ptx.setmaxnreg(False, 56)
                            if warp_id == 3:
                                while gemm_tile_scheduler.valid():
                                    m_idx = T.meta_var(gemm_tile_scheduler.m_idx) # represent cluster task id
                                    n_idx = T.meta_var(gemm_tile_scheduler.n_idx)
                                    for ko in range(K // BLK_K):
                                        # GMEM -> SMEM
                                        if lane_id == 0:
                                            tma2mma_pipe.producer_wait(0)
                                            T.ptx.cp_async.bulk.tensor.g2c(2, A_smem.ptr_to([tma2mma_pipe.idx, 0, 0, 0]), tma_finished.ptr_to([tma2mma_pipe.idx]), A_tensor_map, ko * BLK_K, (m_idx * 4 + cbx) * BLK_M, cta_group=2)
                                            T.ptx.cp_async.bulk.tensor.g2c(2, A_smem.ptr_to([tma2mma_pipe.idx, 1, 0, 0]), tma_finished.ptr_to([tma2mma_pipe.idx]), A_tensor_map, ko * BLK_K, (m_idx * 4 + 2 + cbx) * BLK_M, cta_group=2)
                                            T.ptx.cp_async.bulk.tensor.g2c(2, B_smem.ptr_to([tma2mma_pipe.idx, 0, 0]), tma_finished.ptr_to([tma2mma_pipe.idx]), B_tensor_map, ko * BLK_K, n_idx * BLK_N + cbx * BLK_N // 2, cta_group=2)
                                            if cbx == 0:
                                                # notify the mma stage that tma load is finished
                                                T.ptx.mbarrier.arrive.expect_tx(tma_finished.ptr_to([tma2mma_pipe.idx]), (BLK_K * BLK_M * 2 * 2 + BLK_K * BLK_N) * 2)
                                            tma2mma_pipe.advance()
                                    gemm_tile_scheduler.next_tile(stride=GEMM_SMS // 2)
                                for i in range(PIPE_DEPTH):
                                    # wait for the completion of all the mma of the last tile
                                    if lane_id == 0:
                                        tma2mma_pipe.producer_wait(0)
                                        tma2mma_pipe.advance()
                            elif warp_id < NUM_CONSUMER:
                                while gemm_tile_scheduler.valid():
                                    m_idx = T.meta_var(gemm_tile_scheduler.m_idx) # represent cluster task id
                                    n_idx = T.meta_var(gemm_tile_scheduler.n_idx)
                                    with T.thread():
                                        # MMA
                                        if lane_id == 0 and cbx == 0:
                                            # wait for the last tmem result to be consumed
                                            mma2ld_pipe.producer_wait(warp_id)
                                            for ko in T.serial(0, K // BLK_K):
                                                # wait for tma load to finish
                                                tma2mma_pipe.consumer_wait(0)
                                                T.ptx.tcgen05.encode_matrix_descriptor(T.address_of(base_desc_A), A_smem.ptr_to([tma2mma_pipe.idx, warp_id, 0, 0]), ldo=ldo, sdo=sdo, swizzle=SWIZZLE)
                                                T.ptx.tcgen05.encode_matrix_descriptor(T.address_of(base_desc_B), B_smem.ptr_to([tma2mma_pipe.idx, 0, 0]), ldo=ldo, sdo=sdo, swizzle=SWIZZLE)    
                                                for ki in range(BLK_K // MMA_K):
                                                    descA = base_desc_A + ((ki * MMA_K * 2) >> 0x4)
                                                    descB = base_desc_B + ((ki * MMA_K * 2) >> 0x4)
                                                    if ki == 0 and ko == 0:
                                                        T.ptx.tcgen05.mma("float32", a_type, b_type, tmem_addr + warp_id * MMA_N, descA, descB, descI, use_a_tmem=False, cta_group=cta_group, enable_input_d=False)
                                                    else:
                                                        T.ptx.tcgen05.mma("float32", a_type, b_type, tmem_addr + warp_id * MMA_N, descA, descB, descI, use_a_tmem=False, cta_group=cta_group, enable_input_d=True)
                                                tma2mma_pipe.consumer_release(0)
                                                tma2mma_pipe.advance()
                                            T.ptx.tcgen05.commit(mma2ld_pipe.mbar_p2c.ptr_to([0, 0]), cta_group=2, cta_mask=3)
                                            mma2ld_pipe.advance()
                                    gemm_tile_scheduler.next_tile(stride=GEMM_SMS // 2)
                        with T.warpgroup()[0:NUM_CONSUMER]:
                            T.ptx.setmaxnreg(True, 224)
                            while gemm_tile_scheduler.valid():
                                m_idx = T.meta_var(gemm_tile_scheduler.m_idx) # represent cluster task id
                                n_idx = T.meta_var(gemm_tile_scheduler.n_idx)
                                # wait for the completion of all the mma of the same tile
                                mma2ld_pipe.consumer_wait(0)
                                # TMEM -> RF
                                for i in range(BLK_N // TMEM_LD_SIZE):
                                    T.ptx.tcgen05.ld(tmem_addr + wg_id * MMA_N, warp_id * 32, i * TMEM_LD_SIZE, "32x32b", TMEM_LD_SIZE, False, *[reg[j] for j in range(TMEM_LD_SIZE)])
                                    T.ptx.tcgen05.wait.ld()
                                    for j in range(TMEM_LD_SIZE):
                                        reg_fp16[i * TMEM_LD_SIZE + j] = T.cast(reg[j], "float16")
                                # the tmem can be overwritten by the next tile
                                mma2ld_pipe.consumer_release(wg_id)
                                # RF -> GMEM
                                for i in range(BLK_N // EPI_TILE):
                                    for it in range(EPI_TILE // 8):
                                        for vec in T.vectorized(8):
                                            C_smem[wg_id, warp_id * 32 + lane_id, it * 8 + vec] = reg_fp16[i * EPI_TILE + it * 8 + vec]
                                    T.ptx.bar.sync(wg_id+1, 128)
                                    T.ptx.fence.proxy(scope="shared")
                                    if lane_id == 0 and warp_id == 0:
                                        T.ptx.cp_async.bulk.tensor.s2g(2, C_smem.ptr_to([wg_id, 0, 0]), C_tensor_map, n_idx * BLK_N + i * EPI_TILE, (m_idx * 4 + wg_id * 2 + cbx) * BLK_M)
                                        T.ptx.cp_async.bulk.commit_group()
                                        T.ptx.cp_async.bulk.wait_group(0)
                                    T.ptx.bar.sync(wg_id+1, 128)
                                # notify RS ready
                                sem.semaphore_notify(cbx, wg_id, tid, m_idx, n_idx) # notify a single tile
                                mma2ld_pipe.advance()
                                gemm_tile_scheduler.next_tile(stride=GEMM_SMS // 2)
                    T.ptx.barrier.cluster.arrive(aligned=True)
                    T.ptx.barrier.cluster.wait(aligned=True)
                    # dealloc TMEM
                    with T.warp()[0:1]:
                        T.ptx.tcgen05.relinquish_alloc_permit(cta_group=cta_group)
                        T.ptx.tcgen05.dealloc(tmem_addr, n_cols=N_COLS, cta_group=cta_group)
                    T.timer_end_cuda(ProfileEventType.GEMM, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0)
                # reduce scatter
                else:
                    T.timer_init_cuda(profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, NUM_GROUPS, wg_id)
                    buffer_load_smem = T.decl_buffer((RS_PIPE_DEPTH, RS_BLK_M, RS_BLK_N), dtype=d_type, data=buf.data, scope="shared", elem_offset=512)
                    gemm_out_load_smem = T.decl_buffer((RS_PIPE_DEPTH, RS_BLK_M, RS_BLK_N), dtype=d_type, data=buf.data, scope="shared", elem_offset=512 + RS_BLK_M * RS_BLK_N * RS_PIPE_DEPTH)
                    out_smem = T.decl_buffer((2, RS_BLK_M, RS_BLK_N), dtype=d_type, data=buf.data, scope="shared", elem_offset=512 + RS_BLK_M * RS_BLK_N * RS_PIPE_DEPTH * 2)
                    load_pipe = T.meta_var(
                        ReducePipe(
                            buf.data, 128, RS_PIPE_DEPTH, 1, p_single_cta=False, c_single_cta=False
                        )
                    )
                    local_buffer = T.alloc_buffer((VEC_SIZE), d_type, scope="local")
                    local_acc = T.alloc_buffer((VEC_SIZE,), d_type, scope="local")
                    dst_rank = T.meta_var((rank + WORLD_SIZE - 1) % WORLD_SIZE)
                    m_clusters = T.meta_var((LOCAL_M + BLK_M - 1) // BLK_M)
                    n_clusters = T.meta_var((N + BLK_N - 1) // BLK_N)
                    rs_tile_scheduler = T.meta_var(RSTileScheduler("rs_tile_scheduler", m_clusters=m_clusters, n_clusters=n_clusters, group_size=GROUP_SIZE * 4))
                    load_pipe.init(c2p_thread_count=256)
                    T.tvm_storage_sync("shared")
                    for stage in range(WORLD_SIZE):
                        rs_tile_scheduler.init(bx - GEMM_SMS)
                        if stage != 0:
                            in_flag_addr = T.meta_var(sig_addr.access_ptr("r", offset=sig_addr.offset_of_p([stage - 1, bx - GEMM_SMS])))
                            T.timer_start_cuda(ProfileEventType.WAIT, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid_in_wg == 0)
                            T.cuda.func_call("spin_wait", in_flag_addr, source_code=spin_wait)
                            T.timer_end_cuda(ProfileEventType.WAIT, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid_in_wg == 0)
                        if wg_id == 2:
                            while rs_tile_scheduler.valid():
                                m_idx = T.meta_var(rs_tile_scheduler.m_idx)
                                n_idx = T.meta_var(rs_tile_scheduler.n_idx)
                                seg = T.meta_var((rank + stage + 1) % WORLD_SIZE)
                                m_idx_global = T.meta_var(m_idx + seg * LOCAL_M // BLK_M)
                                if warp_id == 0:
                                    sem.semaphore_wait(m_idx_global, n_idx)
                                    if lane_id == 0:
                                        T.timer_start_cuda(ProfileEventType.LOAD, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid_in_wg == 0)
                                        for m_o in range(M_SEG):
                                            load_pipe.producer_wait(0)
                                            if stage != 0:
                                                T.ptx.cp_async.bulk.tensor.g2c(4, buffer_load_smem.ptr_to([load_pipe.idx, 0, 0]), load_pipe.mbar_p2c.ptr_to([load_pipe.idx, 0]), buffer_tensor_map, 0, 0, n_idx, m_idx_global * M_SEG + m_o)
                                                T.ptx.cp_async.bulk.tensor.g2c(2, gemm_out_load_smem.ptr_to([load_pipe.idx, 0, 0]), load_pipe.mbar_p2c.ptr_to([load_pipe.idx, 0]), gemm_out_tensor_map, n_idx * BLK_N, m_idx_global * BLK_M + m_o * RS_BLK_M)
                                                T.ptx.mbarrier.arrive.expect_tx(load_pipe.mbar_p2c.ptr_to([load_pipe.idx, 0]), RS_BLK_M * RS_BLK_N * 2 * 2)
                                            else:
                                                T.ptx.cp_async.bulk.tensor.g2c(2, gemm_out_load_smem.ptr_to([load_pipe.idx, 0, 0]), load_pipe.mbar_p2c.ptr_to([load_pipe.idx, 0]), gemm_out_tensor_map, n_idx * BLK_N, m_idx_global * BLK_M + m_o * RS_BLK_M)
                                                T.ptx.mbarrier.arrive.expect_tx(load_pipe.mbar_p2c.ptr_to([load_pipe.idx, 0]), RS_BLK_M * RS_BLK_N * 2)
                                            load_pipe.advance()
                                        T.timer_end_cuda(ProfileEventType.LOAD, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid_in_wg == 0)
                                rs_tile_scheduler.next_tile(stride=RS_SMS)
                        else:
                            while rs_tile_scheduler.valid():
                                m_idx = T.meta_var(rs_tile_scheduler.m_idx)
                                n_idx = T.meta_var(rs_tile_scheduler.n_idx)
                                seg = T.meta_var((rank + stage + 1) % WORLD_SIZE)
                                m_idx_global = T.meta_var(m_idx + seg * LOCAL_M // BLK_M)
                                T.timer_start_cuda(ProfileEventType.ACCUM, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid_in_wg == 0)
                                for m_o in range(M_SEG):
                                    load_pipe.consumer_wait(0)
                                    n_elems = T.meta_var(RS_BLK_M * RS_BLK_N)
                                    n_threads = T.meta_var(256)
                                    if tid == 0:
                                        T.ptx.cp_async.bulk.wait_group(1)
                                    T.ptx.bar.sync(1, 256)
                                    for k in T.serial(n_elems // (n_threads * VEC_SIZE)):
                                        idx = T.meta_var((k * n_threads + tid) * VEC_SIZE)
                                        for vec in T.vectorized(VEC_SIZE):
                                            m_i = T.meta_var((idx + vec) // RS_BLK_N)
                                            n_i = T.meta_var((idx + vec) % RS_BLK_N)
                                            local_acc[vec] = gemm_out_load_smem[load_pipe.idx, m_i, n_i]
                                        if stage != 0:
                                            for vec in T.vectorized(VEC_SIZE):
                                                m_i = T.meta_var((idx + vec) // RS_BLK_N)
                                                n_i = T.meta_var((idx + vec) % RS_BLK_N)
                                                local_buffer[vec] = buffer_load_smem[load_pipe.idx, m_i, n_i]
                                            for vec in T.unroll(VEC_SIZE):
                                                local_acc[vec] += local_buffer[vec]
                                        if stage == WORLD_SIZE - 1:
                                            for vec in T.vectorized(VEC_SIZE):
                                                m_i = T.meta_var((idx + vec) // RS_BLK_N)
                                                n_i = T.meta_var((idx + vec) % RS_BLK_N)
                                                out[m_i + m_idx * BLK_M + m_o * RS_BLK_M, n_i + n_idx * RS_BLK_N] = local_acc[vec]
                                        else:
                                            for vec in T.vectorized(VEC_SIZE):
                                                m_i = T.meta_var((idx + vec) // RS_BLK_N)
                                                n_i = T.meta_var((idx + vec) % RS_BLK_N)
                                                out_smem[m_o % 2, m_i, n_i] = local_acc[vec]
                                    T.ptx.bar.sync(1, 256)
                                    T.ptx.fence.proxy(scope="shared")
                                    if tid == 0 and stage != WORLD_SIZE - 1:
                                        T.cuda.func_call("write_remote_tma",buffer.ptr_to([m_idx_global * M_SEG + m_o, n_idx, 0, 0]), out_smem.ptr_to([m_o % 2, 0, 0]), dst_rank, RS_BLK_M, RS_BLK_N, source_code=write_remote_tma)
                                        T.ptx.cp_async.bulk.commit_group()
                                    load_pipe.consumer_release(0)
                                    load_pipe.advance()
                                T.timer_end_cuda(ProfileEventType.ACCUM, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid_in_wg == 0)
                                rs_tile_scheduler.next_tile(stride=RS_SMS)
                            T.ptx.bar.sync(1, 256)
                            flag_addr = T.meta_var(sig_addr.access_ptr("w", offset=sig_addr.offset_of_p([stage, bx - GEMM_SMS])))
                            if tid == 0 :
                                T.ptx.cp_async.bulk.wait_group(0)
                                T.cuda.func_call("signal_fp16", flag_addr, dst_rank, 1, source_code=signal_fp16_ptx)
    # fmt: on

    devices = list(np.arange(WORLD_SIZE))
    sess = di.ProcessSession(num_workers=WORLD_SIZE)
    sess.init_ccl(tvm.get_global_func("runtime.disco.compiled_ccl")(), *devices)
    f_init_nvshmem_uid = tvm.get_global_func("runtime.disco.nvshmem.init_nvshmem_uid")
    uid = f_init_nvshmem_uid()
    init_dfunc = sess.get_global_func("runtime.disco.nvshmem.init_nvshmem")
    init_dfunc(uid, WORLD_SIZE, 0)
    sess.sync_worker_0()

    # prepare test data
    print(f"begin preparing test data ...")
    np.random.seed(42)
    DEV = tvm.cuda(0)
    A_np = np.random.uniform(-1, 1, (WORLD_SIZE, M, K)).astype(a_type)
    B_np = np.random.uniform(-1, 1, (WORLD_SIZE, N, K)).astype(b_type)
    # A_np = np.random.randint(0, 2, (WORLD_SIZE, M, K)).astype(a_type)
    # B_np = np.random.randint(0, 2, (WORLD_SIZE, N, K)).astype(b_type)
    A_tvm = tvm.nd.array(A_np, device=DEV)
    B_tvm = tvm.nd.array(B_np, device=DEV)
    semaphore_np = np.zeros((M // BLK_M, N // BLK_N), dtype="int32")
    buffer_np = np.zeros((M // RS_BLK_M, N // RS_BLK_N, RS_BLK_M, RS_BLK_N), dtype=d_type)
    sig_addr_np = np.zeros((WORLD_SIZE, SM_COUNT), dtype="uint32")
    profiler_buffer_np = np.zeros((PROFILER_BUFFER_SIZE,), dtype="uint64")

    A_array_all = sess.empty((WORLD_SIZE, M, K), a_type, worker0_only=True)
    B_array_all = sess.empty((WORLD_SIZE, N, K), b_type, worker0_only=True)
    sess.copy_to_worker_0(A_tvm, A_array_all)
    sess.copy_to_worker_0(B_tvm, B_array_all)

    nvshmem_malloc_hook = sess.get_global_func("runtime.disco.nvshmem.empty")
    args_dict = {
        "A_array": sess.empty((M, K), a_type),
        "B_array": sess.empty((N, K), b_type),
        "gemm_out_array": sess.empty((M, N), d_type),
        "semaphore_array": sess.empty((M // BLK_M, N // BLK_N), "int32"),
        "buffer_array": nvshmem_malloc_hook(
            ShapeTuple((M // RS_BLK_M, N // RS_BLK_N, RS_BLK_M, RS_BLK_N)), d_type, None
        ),
        "sig_addr_array": nvshmem_malloc_hook(ShapeTuple((WORLD_SIZE, SM_COUNT)), "uint32", None),
        "out_array": sess.empty((LOCAL_M, N), d_type),
        "profiler_buffer_array": sess.empty((PROFILER_BUFFER_SIZE,), "uint64"),
    }

    res_dict = {
        "gemm_out_res": sess.empty((WORLD_SIZE, M, N), d_type, worker0_only=True),
        "buffer_res": sess.empty(
            (WORLD_SIZE, M // RS_BLK_M, N // RS_BLK_N, RS_BLK_M, RS_BLK_N),
            d_type,
            worker0_only=True,
        ),
        "out_res": sess.empty((WORLD_SIZE, LOCAL_M, N), d_type, worker0_only=True),
        "profiler_buffer_res": sess.empty(
            (WORLD_SIZE, PROFILER_BUFFER_SIZE), "uint64", worker0_only=True
        ),
        "gemm_out_host": tvm.nd.empty((WORLD_SIZE, M, N), d_type, device=DEV),
        "buffer_host": tvm.nd.empty(
            (WORLD_SIZE, M // RS_BLK_M, N // RS_BLK_N, RS_BLK_M, RS_BLK_N), d_type, device=DEV
        ),
        "out_host": tvm.nd.empty((WORLD_SIZE, LOCAL_M, N), d_type, device=DEV),
        "profiler_buffer_host": tvm.nd.empty(
            (WORLD_SIZE, PROFILER_BUFFER_SIZE), "uint64", device=DEV
        ),
    }

    sess.scatter_from_worker0(A_array_all, args_dict["A_array"])
    sess.scatter_from_worker0(B_array_all, args_dict["B_array"])
    sess.sync_worker_0()
    print(f"Data prepared successfully")

    with tempfile.TemporaryDirectory() as tmpdir:
        target = tvm.target.Target("cuda")
        path = tmpdir + "/test.so"
        mod = tvm.compile(test_mma_ss_tma_2sm_persistent, target=target, tir_pipeline="tirp")
        print(mod.mod.imports[0].inspect_source())
        mod.export_library(path)

        rt_mod = sess.load_vm_module(path)
        barrier_dfunc = sess.get_global_func("runtime.disco.nvshmem.barrier_all_on_current_stream")
        print("Begin kernel execution...")
        sess._sync_all()
        for itr in range(N_REPEAT):
            barrier_dfunc()
            rt_mod["test_mma_ss_tma_2sm_persistent"](
                args_dict["A_array"],
                args_dict["B_array"],
                args_dict["gemm_out_array"],
                args_dict["semaphore_array"],
                args_dict["buffer_array"],
                args_dict["sig_addr_array"],
                args_dict["out_array"],
                args_dict["profiler_buffer_array"],
            )
            sess._sync_all()
            if itr < N_REPEAT - 1:
                sess.broadcast(semaphore_np, args_dict["semaphore_array"])
                sess.broadcast(buffer_np, args_dict["buffer_array"])
                sess.broadcast(sig_addr_np, args_dict["sig_addr_array"])
                sess.broadcast(profiler_buffer_np, args_dict["profiler_buffer_array"])
                sess._sync_all()

        print("Kernel execution finished.")
        # validate results
        sess.gather_to_worker0(args_dict["gemm_out_array"], res_dict["gemm_out_res"])
        sess.copy_from_worker_0(res_dict["gemm_out_host"], res_dict["gemm_out_res"])
        sess.gather_to_worker0(args_dict["buffer_array"], res_dict["buffer_res"])
        sess.copy_from_worker_0(res_dict["buffer_host"], res_dict["buffer_res"])
        sess.gather_to_worker0(args_dict["out_array"], res_dict["out_res"])
        sess.copy_from_worker_0(res_dict["out_host"], res_dict["out_res"])
        sess.gather_to_worker0(args_dict["profiler_buffer_array"], res_dict["profiler_buffer_res"])
        sess.copy_from_worker_0(res_dict["profiler_buffer_host"], res_dict["profiler_buffer_res"])

        # sync all workers to make sure the temporary files are cleaned up after all workers
        # finish the execution
        sess._sync_all()

    finalize_dfunc = sess.get_global_func("runtime.disco.nvshmem.finalize_nvshmem")
    finalize_dfunc()
    sess.sync_worker_0()

    # validate results
    print("Validating results...")

    import torch

    gemm_out_torch = torch.zeros((WORLD_SIZE, M, N), dtype=torch.float16, device="cuda")
    gemm_out_torch_sum = torch.zeros((M, N), dtype=torch.float16, device="cuda")
    for i in range(WORLD_SIZE):
        A_torch = torch.tensor(A_np[i], dtype=torch.float16, device="cuda")
        B_torch = torch.tensor(B_np[i], dtype=torch.float16, device="cuda")
        gemm_out_torch[i] = torch.matmul(A_torch, B_torch.T)
        gemm_out_res = res_dict["gemm_out_host"].numpy()[i]
        assert (gemm_out_res == gemm_out_torch[i].cpu().numpy()).all()

    for i in range(WORLD_SIZE):
        start_rank = (i + WORLD_SIZE - 1) % WORLD_SIZE
        for j in range(WORLD_SIZE):
            rank = (start_rank + WORLD_SIZE - j) % WORLD_SIZE
            gemm_out_torch_sum[i * LOCAL_M : (i + 1) * LOCAL_M, :] += gemm_out_torch[rank][
                i * LOCAL_M : (i + 1) * LOCAL_M, :
            ]
    out_res = res_dict["out_host"].numpy().reshape(-1, N)
    np.testing.assert_allclose(out_res, gemm_out_torch_sum.cpu().numpy(), atol=1e-3)

    print("Results all correct.")

    # profiler results
    for rank in range(WORLD_SIZE):
        if rank == 7:
            export_to_perfetto_trace(
                res_dict["profiler_buffer_host"].numpy()[rank],
                f"hgemm-RS-rank{rank}.perfetto-trace",
                event_type_names,
            )


if __name__ == "__main__":
    test_hgemm_rs()
