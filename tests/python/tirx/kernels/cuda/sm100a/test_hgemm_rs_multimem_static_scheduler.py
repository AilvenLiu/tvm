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

# 1.84us
# gemm perf can be optimized

import math
import tempfile
from enum import Enum

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.ir.type import PointerType, PrimType
from tvm.runtime import ShapeTuple
from tvm.runtime import disco as di
from tvm.script import tirx as Tx
from tvm.script.ir_builder import IRBuilder
from tvm.tirx.bench.utils import export_to_perfetto_trace, CudaProfiler
from tvm.tir.layout import TileLayout, tid_in_wg, TLane, TCol, S


class JobType(Enum):
    END = 0
    GEMM = 1
    RS = 2


class ProfileEventType(Enum):
    RS = 0
    GEMM = 1
    PREPARE = 2


event_type_names = [
    "rs",
    "gemm",
    "prepare",
]


d_type, a_type, b_type = "float16", "float16", "float16"
nbytes = 2
WORLD_SIZE = 8
M, N, K = 16384, 12288, 49152 // 8
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
TILE_M, TILE_N = BLK_M, BLK_N
cta_group = 2
ldo, sdo = 1, 64

# RS
RS_SMS = 20
GEMM_SMS = SM_COUNT - RS_SMS
GEMM_M_CLUSTERS = (M + BLK_M - 1) // BLK_M // CLUSTER_M // NUM_CONSUMER
GEMM_N_CLUSTERS = (N + BLK_N - 1) // BLK_N // CLUSTER_N
RS_M_CLUSTERS = (LOCAL_M + TILE_M - 1) // TILE_M
RS_N_CLUSTERS = (N + TILE_N - 1) // TILE_N

# profiling
NUM_GROUPS = 5
PROFILER_BUFFER_SIZE = int(1e7)
PROFILER_WRITE_STRIDE = SM_COUNT * NUM_GROUPS
PROFILER_ON = True
N_REPEAT = 30


atomic_add_system_uint64 = f"""
__forceinline__ __device__ uint64_t atomic_add_system_uint64(uint64_t* addr, uint64_t value) {{
    return atomicAdd_system(reinterpret_cast<unsigned long long*>(addr), value);
}}
"""


@Tx.meta_class
class Semaphore:
    def __init__(self, cnt, buffer):
        self.cnt = cnt
        self.sem = buffer
        self.state = Tx.alloc_buffer([1], "uint64", scope="local", align=4, name="semaphore_state")

    @Tx.inline
    def semaphore_wait(self, *coord):
        with Tx.thread():
            while 1:
                Tx.ptx.ld_global_acquire(
                    self.state[0], self.sem.access_ptr("r", offset=self.sem.elem_offset_of(coord))
                )
                if Tx.cuda.syncthreads_and(self.state[0] == self.cnt):
                    break
                Tx.cuda.nano_sleep(40)

    @Tx.inline
    def semaphore_notify(self, tid, m_idx, n_idx):
        # wg is synced
        with Tx.thread():
            if tid % 128 == 0:
                Tx.cuda.func_call(
                    "atomic_add_system_uint64",
                    self.sem.access_ptr("rw", offset=self.sem.elem_offset_of((m_idx, n_idx))),
                    Tx.uint64(1),
                    source_code=atomic_add_system_uint64,
                )
            Tx.cuda.thread_fence()


ld_reduce_8xfp16 = """
__forceinline__ __device__ void ld_reduce_8_fp16(void* src_addr, void* dst_addr) {
    int4* source = (int4*) nvshmemx_mc_ptr(NVSHMEM_TEAM_WORLD, src_addr);
    int4* dest = (int4*) dst_addr;
    constexpr int UNROLL = 1;
    union {
        uint16_t u2[8 * UNROLL];
        uint64_t u8[2 * UNROLL];
    };
    for (int u = 0; u < UNROLL; u++) {
        asm("multimem.ld_reduce.global.add.v8.f16 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
            : "=h"(u2[8 * u]), "=h"(u2[8 * u + 1]), "=h"(u2[8 * u + 2]), "=h"(u2[8 * u + 3]), "=h"(u2[8 * u + 4]), "=h"(u2[8 * u + 5]), "=h"(u2[8 * u + 6]), "=h"(u2[8 * u + 7])
            : "l"(source + u));
    }
    for (int u = 0; u < UNROLL; u++) {
        asm("st.global.v2.b64 [%0], {%1, %2};" ::"l"(dest + u), "l"(u8[2 * u]),
            "l"(u8[2 * u + 1]));
    }
}
"""


@Tx.meta_class
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
        self.mbar_p2c = Tx.decl_buffer(
            (pipeline_depth, pipeline_num), "uint64", shared_buf, elem_offset=base_offset
        )
        self.mbar_c2p = Tx.decl_buffer(
            (pipeline_depth, pipeline_num),
            "uint64",
            shared_buf,
            elem_offset=base_offset + pipeline_depth * pipeline_num,
        ).buffer
        self.idx = Tx.local_scalar("int32", name="pipeline_idx")
        self.p2c_phase = Tx.local_scalar("int32", name="pipeline_p2c_phase")
        self.c2p_phase = Tx.local_scalar("int32", name="pipeline_c2p_phase")
        self.p_single_cta = p_single_cta
        self.c_single_cta = c_single_cta

    @Tx.inline
    def init(self, p2c_thread_count: int = 1, c2p_thread_count: int = 1):
        self.idx = 0
        self.p2c_phase = 0
        self.c2p_phase = 1
        with Tx.thread()[0:1]:
            for cbx in Tx.thread_binding(CLUSTER_M, "clusterCtaIdx.x"):
                for i in Tx.serial(0, self.pipeline_depth):
                    for j in Tx.serial(0, self.pipeline_num):
                        if not self.c_single_cta or cbx == 0:
                            Tx.ptx.mbarrier.init(self.mbar_p2c.ptr_to([i, j]), p2c_thread_count)
                        if not self.p_single_cta or cbx == 0:
                            Tx.ptx.mbarrier.init(self.mbar_c2p.ptr_to([i, j]), c2p_thread_count)
        Tx.ptx.fence.proxy("shared")

    @Tx.inline
    def advance(self):
        self.idx = (self.idx + 1) % self.pipeline_depth
        if self.idx == 0:
            self.p2c_phase = self.p2c_phase ^ 1
            self.c2p_phase = self.c2p_phase ^ 1

    @Tx.inline
    def producer_wait(self, pipeline_idx):
        for cbx in Tx.thread_binding(CLUSTER_M, "clusterCtaIdx.x"):
            if not self.p_single_cta or cbx == 0:
                Tx.ptx.mbarrier.try_wait(
                    self.mbar_c2p.ptr_to([self.idx, pipeline_idx]), self.c2p_phase
                )

    @Tx.inline
    def consumer_wait(self, pipeline_idx):
        for cbx in Tx.thread_binding(CLUSTER_M, "clusterCtaIdx.x"):
            if not self.c_single_cta or cbx == 0:
                Tx.ptx.mbarrier.try_wait(
                    self.mbar_p2c.ptr_to([self.idx, pipeline_idx]), self.p2c_phase
                )


class TMA2MMAPipeline(Pipeline):

    @Tx.inline
    def consumer_release(self, pipeline_idx):
        for cbx in Tx.thread_binding(CLUSTER_M, "clusterCtaIdx.x"):
            for tx in Tx.thread_binding(NUM_THREADS, "threadIdx.x"):
                if tx % 32 == 0:
                    if not self.c_single_cta:
                        Tx.ptx.tcgen05.commit(self.mbar_c2p.ptr_to([self.idx, pipeline_idx]), 1)
                    elif cbx == 0:
                        Tx.ptx.tcgen05.commit(
                            self.mbar_c2p.ptr_to([self.idx, pipeline_idx]), 2, cta_mask=3
                        )


class MMA2LDpipeline(Pipeline):
    @Tx.inline
    def consumer_release(self, pipeline_idx):
        for cbx in Tx.thread_binding(CLUSTER_M, "clusterCtaIdx.x"):
            if not self.c_single_cta or cbx == 0:
                Tx.ptx.mbarrier.arrive(
                    self.mbar_c2p.ptr_to([self.idx, pipeline_idx]), cta_id=0, pred=True
                )


# RS SMS do GEMM in their idle times, fetch GEMM sequentially from the beginning
def get_tasks():
    alloc_size = GEMM_M_CLUSTERS * GEMM_N_CLUSTERS + RS_M_CLUSTERS * RS_N_CLUSTERS
    exec_queue = np.zeros((SM_COUNT, alloc_size, 3), dtype="int32")
    next_task_id = np.zeros(SM_COUNT, dtype="int32")
    assert GEMM_N_CLUSTERS == RS_N_CLUSTERS
    assert GEMM_M_CLUSTERS * GROUP_SIZE % (GEMM_SMS // 2) == 0
    n_idle_gemm_groups = math.ceil(GEMM_N_CLUSTERS / GROUP_SIZE) - 1

    gemm_group_rows = (GEMM_N_CLUSTERS // GROUP_SIZE) * GROUP_SIZE
    gemm_final_rows = GEMM_N_CLUSTERS - gemm_group_rows
    gemm_group_repeat = GROUP_SIZE * GEMM_M_CLUSTERS
    n_gemm_rounds = math.ceil(GEMM_M_CLUSTERS * GROUP_SIZE / (GEMM_SMS / 2))
    rs_group_rows = (RS_N_CLUSTERS // GROUP_SIZE) * GROUP_SIZE
    if rs_group_rows == RS_N_CLUSTERS:
        rs_group_rows -= GROUP_SIZE
    rs_final_rows = RS_N_CLUSTERS - rs_group_rows
    rs_group_repeat = GROUP_SIZE * RS_M_CLUSTERS

    def schedule_gemm(linear_idx, bx):
        if linear_idx < gemm_group_rows * GEMM_M_CLUSTERS and gemm_group_rows > 0:
            n_idx = linear_idx // gemm_group_repeat * GROUP_SIZE + (linear_idx % GROUP_SIZE)
            m_idx = linear_idx % gemm_group_repeat // GROUP_SIZE
        elif gemm_final_rows > 0:
            remainder_idx = linear_idx - gemm_group_rows * GEMM_M_CLUSTERS
            n_idx = gemm_group_rows + remainder_idx % gemm_final_rows
            m_idx = remainder_idx // gemm_final_rows
        task_id = next_task_id[bx]
        exec_queue[bx, task_id, 0] = m_idx
        exec_queue[bx, task_id, 1] = n_idx
        exec_queue[bx, task_id, 2] = JobType.GEMM.value
        next_task_id[bx] += 1

    def schedule_rs(linear_idx, bx):
        if linear_idx < rs_group_rows * RS_M_CLUSTERS and rs_group_rows > 0:
            n_idx = linear_idx // rs_group_repeat * GROUP_SIZE + (linear_idx % GROUP_SIZE)
            m_idx = linear_idx % rs_group_repeat // GROUP_SIZE
        elif rs_final_rows > 0:
            remainder_idx = linear_idx - rs_group_rows * RS_M_CLUSTERS
            n_idx = rs_group_rows + remainder_idx % rs_final_rows
            m_idx = remainder_idx // rs_final_rows
        task_id = next_task_id[bx]
        exec_queue[bx, task_id, 0] = m_idx
        exec_queue[bx, task_id, 1] = n_idx
        exec_queue[bx, task_id, 2] = JobType.RS.value
        next_task_id[bx] += 1

    gemm_linear_idx = 0
    rs_linear_idx = 0
    for g in range(n_idle_gemm_groups):
        # idle GEMM
        if g == 0:
            for r in range(n_gemm_rounds):
                for bx in range(SM_COUNT):
                    schedule_gemm(gemm_linear_idx, bx)
                    if bx % 2 == 1:
                        gemm_linear_idx += 1
        else:
            for bx in range(SM_COUNT):
                schedule_gemm(gemm_linear_idx, bx)
                if bx % 2 == 1:
                    gemm_linear_idx += 1
        # normal GEMM
        for r in range(n_gemm_rounds - 1):
            for bx in range(GEMM_SMS):
                if gemm_linear_idx >= GEMM_M_CLUSTERS * GEMM_N_CLUSTERS:
                    break
                schedule_gemm(gemm_linear_idx, bx)
                if bx % 2 == 1:
                    gemm_linear_idx += 1
        # RS
        while rs_linear_idx < (g + 1) * RS_M_CLUSTERS * GROUP_SIZE:
            for bx in range(GEMM_SMS, SM_COUNT):
                schedule_rs(rs_linear_idx, bx)
                rs_linear_idx += 1
                if rs_linear_idx >= (g + 1) * RS_M_CLUSTERS * GROUP_SIZE:
                    break
    # tail RS
    while rs_linear_idx < RS_M_CLUSTERS * RS_N_CLUSTERS:
        for bx in range(SM_COUNT):
            schedule_rs(rs_linear_idx, bx)
            rs_linear_idx += 1
            if rs_linear_idx >= RS_M_CLUSTERS * RS_N_CLUSTERS:
                break
    max_task_id = next_task_id.max() + 1
    exec_queue = exec_queue[:, :max_task_id, :]
    return max_task_id, np.stack([exec_queue for _ in range(WORLD_SIZE)], axis=0).astype("int32")


MAX_TASKS, exec_queue_np = get_tasks()
SMEM_OFFSET = SMEM_SIZE
SMEM_SIZE += MAX_TASKS * 3 * 4


@pytest.mark.skip()
@tvm.testing.requires_cuda_compute_version(10, exact=True)
def test_hgemm_rs():
    A_layout = Tx.ComposeLayout(
        Tx.SwizzleLayout(3, 3, 3, swizzle_inner=True),
        Tx.TileLayout(Tx.S[(PIPE_DEPTH, NUM_CONSUMER, BLK_M, 1, 64) : (BLK_M * 64 * NUM_CONSUMER, BLK_M * 64, 64, BLK_M * 64, 1)]),
    )
    B_layout = Tx.ComposeLayout(
        Tx.SwizzleLayout(3, 3, 3, swizzle_inner=True),
        Tx.TileLayout(Tx.S[(PIPE_DEPTH, BLK_N // 2, 1, 64) : (BLK_N // 2 * 64, 64, BLK_N // 2 * 64, 1)]),
    )
    D_layout = Tx.ComposeLayout(
        Tx.SwizzleLayout(3, 3, 3, swizzle_inner=True),
        Tx.TileLayout(Tx.S[(NUM_CONSUMER, BLK_M, EPI_TILE) : (BLK_M * EPI_TILE, EPI_TILE, 1)]),
    )

    # fmt: off
    @Tx.prim_func(tirx=True)
    def test_mma_ss_tma_2sm_persistent(A: Tx.Buffer((M, K), a_type), B: Tx.Buffer((N, K), b_type), gemm_out: Tx.Buffer((M // TILE_M, N // TILE_N, TILE_M, TILE_N), d_type),
                                       semaphore: Tx.Buffer((LOCAL_M // TILE_M, N // TILE_N), "uint64"), out: Tx.Buffer((LOCAL_M, N), d_type),
                                       exec_queue: Tx.Buffer((SM_COUNT, MAX_TASKS, 3), "int32"), profiler_buffer: Tx.Buffer((PROFILER_BUFFER_SIZE,), "uint64")):
        A_tensor_map: Tx.let[Tx.handle("tensormap")] = Tx.tvm_stack_alloca("tensormap", 1)
        B_tensor_map: Tx.let[Tx.handle("tensormap")] = Tx.tvm_stack_alloca("tensormap", 1)
        C_tensor_map: Tx.let[Tx.handle("tensormap")] = Tx.tvm_stack_alloca("tensormap", 1)
        Tx.call_packed("runtime.cuTensorMapEncodeTiled", A_tensor_map, "float16", 2, A.data, K, M, K * 2, BLK_K, BLK_M, 1, 1, 0, 3, 0, 0)
        Tx.call_packed("runtime.cuTensorMapEncodeTiled", B_tensor_map, "float16", 2, B.data, K, N, K * 2, BLK_K, BLK_N // 2, 1, 1, 0, 3, 0, 0)
        Tx.call_packed("runtime.cuTensorMapEncodeTiled", C_tensor_map, "float16", 4, gemm_out.data, TILE_N, TILE_M, N // TILE_N, M // TILE_M, TILE_N * 2, (TILE_N * TILE_M) * 2, (N * TILE_M) * 2, EPI_TILE, BLK_M, 1, 1, 1, 1, 1, 1, 0, 3, 0, 0)
        with Tx.kernel():
            cbx, cby = Tx.cta_id([CLUSTER_M, CLUSTER_N], parent="cluster")
            bx = Tx.cta_id([SM_COUNT], parent="kernel")
            wg_id = Tx.warpgroup_id([NUM_CONSUMER+1], parent="cta")
            warp_id = Tx.warp_id([4], parent="warpgroup")
            lane_id = Tx.thread_id([32], parent="warp")
            tid = Tx.thread_id([NUM_THREADS], parent="cta")
            rank = Tx.nvshmem.my_pe()
            tid_in_wg = Tx.thread_id([128], parent="warpgroup")
            with Tx.cta():
                buf = Tx.alloc_buffer([SMEM_SIZE], "uint8", scope="shared.dyn")
                tmem_addr = Tx.decl_scalar("uint32", buf.data, scope="shared.dyn", elem_offset=0)
                A_smem = Tx.decl_buffer((PIPE_DEPTH, NUM_CONSUMER,BLK_M, BLK_K), a_type, buf.data, elem_offset=512, layout=A_layout)
                B_smem = Tx.decl_buffer((PIPE_DEPTH, BLK_N // 2, BLK_K), b_type, buf.data, elem_offset=512 + BLK_K * BLK_M * NUM_CONSUMER * PIPE_DEPTH, layout=B_layout)
                C_smem = Tx.decl_buffer((NUM_CONSUMER, BLK_M, EPI_TILE), d_type, buf.data, elem_offset=512 + BLK_K * BLK_M * NUM_CONSUMER * PIPE_DEPTH + BLK_K * BLK_N // 2 * PIPE_DEPTH, layout=D_layout)
                reg = Tx.alloc_buffer((TMEM_LD_SIZE,), "float32", scope="local")
                reg_wg = reg.view(128, TMEM_LD_SIZE, layout=TileLayout(S[(128, TMEM_LD_SIZE) : (1@tid_in_wg, 1)]))
                reg_fp16 = Tx.alloc_buffer((BLK_N,), d_type, scope="local")
                descA: Tx.uint64
                descB: Tx.uint64
                descI: Tx.uint32
                base_desc_A: Tx.uint64
                base_desc_B: Tx.uint64
                tma2mma_pipe = Tx.meta_var(TMA2MMAPipeline(buf.data, 1, PIPE_DEPTH, 1, p_single_cta=False, c_single_cta=True))
                mma2ld_pipe = Tx.meta_var(MMA2LDpipeline(buf.data, 1 + PIPE_DEPTH * 2, 1, NUM_CONSUMER, p_single_cta=True, c_single_cta=False))
                mma2ld_pipe.init(c2p_thread_count=128 * 2, p2c_thread_count=2)
                tma2mma_pipe.init(c2p_thread_count=NUM_CONSUMER)
                ptr: Tx.let[Tx.Var(name="ptr", dtype=PointerType(PrimType("uint64")))] = Tx.reinterpret("handle", Tx.ptx.map_shared_rank(tma2mma_pipe.mbar_p2c.ptr_to([0, 0]), 0))
                tma_finished = Tx.decl_buffer([PIPE_DEPTH], "uint64", data=ptr, scope="shared")
                sem = Tx.meta_var(Semaphore(cnt=WORLD_SIZE, buffer=semaphore))
                offset: Tx.int32
                task_id: Tx.int32
                task_id = 0
                task_smem = Tx.decl_buffer((MAX_TASKS, 3), "int32", buf.data, elem_offset=SMEM_OFFSET // 4)
                profiler = CudaProfiler(
                    profiler_buffer,
                    write_stride=PROFILER_WRITE_STRIDE,
                    num_groups=NUM_GROUPS,
                    profiler_enabled=PROFILER_ON,
                )
                profiler.init(wg_id)
                # alloc TMEM
                with Tx.warp()[0:1]:
                    Tx.ptx.tcgen05.alloc(Tx.address_of(tmem_addr), n_cols=N_COLS, cta_group=cta_group)
                Tx.ptx.tcgen05.encode_instr_descriptor(Tx.address_of(descI), "float32", a_type, b_type, MMA_M, MMA_N, MMA_K, trans_a=False, trans_b=False, n_cta_groups=cta_group)
                # adjust registers count
                with Tx.cta():
                    Tx.attr({"tirx.scope_partition": True})
                    with Tx.warpgroup()[NUM_CONSUMER:NUM_CONSUMER + 1]:
                        Tx.ptx.setmaxnreg(False, 56)
                    with Tx.warpgroup()[0:NUM_CONSUMER]:
                        Tx.ptx.setmaxnreg(True, 224)
                # prefetch task_ids
                with Tx.thread():
                    for k in Tx.serial(Tx.ceildiv(MAX_TASKS * 3, NUM_THREADS)):
                        idx = Tx.meta_var(k * NUM_THREADS + tid)
                        if idx < MAX_TASKS * 3:
                            task_smem[idx // 3, idx % 3] = exec_queue[bx, idx // 3, idx % 3]
                Tx.cuda.cta_sync()
                Tx.cuda.trap_when_assert_failed(tmem_addr == 0)
                tmem = Tx.decl_buffer((128, N_COLS), "float32", scope="tmem", allocated_addr=0,
                                     layout=TileLayout(S[(128, N_COLS) : (1@TCol, 1@TLane)]))

                while task_id < MAX_TASKS and task_smem[task_id, 2] != JobType.END.value:
                    # GEMM
                    if task_smem[task_id, 2] == JobType.GEMM.value:
                        profiler.start(ProfileEventType.GEMM, tid == 0)
                        with Tx.cta():
                            Tx.attr({"tirx.scope_partition": True})
                            with Tx.warpgroup()[NUM_CONSUMER:NUM_CONSUMER + 1]:
                                if warp_id == 3:
                                    m_idx = task_smem[task_id, 0]
                                    n_idx = task_smem[task_id, 1]
                                    for ko in range(K // BLK_K):
                                        # GMEM -> SMEM
                                        if lane_id == 0:
                                            tma2mma_pipe.producer_wait(0)
                                            Tx.ptx.cp_async.bulk.tensor.g2c(2, A_smem.ptr_to([tma2mma_pipe.idx, 0, 0, 0]), tma_finished.ptr_to([tma2mma_pipe.idx]), A_tensor_map, ko * BLK_K, (m_idx * 4 + cbx) * BLK_M, cta_group=2)
                                            Tx.ptx.cp_async.bulk.tensor.g2c(2, A_smem.ptr_to([tma2mma_pipe.idx, 1, 0, 0]), tma_finished.ptr_to([tma2mma_pipe.idx]), A_tensor_map, ko * BLK_K, (m_idx * 4 + 2 + cbx) * BLK_M, cta_group=2)
                                            Tx.ptx.cp_async.bulk.tensor.g2c(2, B_smem.ptr_to([tma2mma_pipe.idx, 0, 0]), tma_finished.ptr_to([tma2mma_pipe.idx]), B_tensor_map, ko * BLK_K, n_idx * BLK_N + cbx * BLK_N // 2, cta_group=2)
                                            if cbx == 0:
                                                Tx.ptx.mbarrier.arrive.expect_tx(tma_finished.ptr_to([tma2mma_pipe.idx]), (BLK_K * BLK_M * 2 * 2 + BLK_K * BLK_N) * 2)
                                            tma2mma_pipe.advance()
                                elif warp_id < NUM_CONSUMER:
                                    m_idx = task_smem[task_id, 0]
                                    n_idx = task_smem[task_id, 1]
                                    with Tx.thread():
                                        # MMA
                                        if lane_id == 0 and cbx == 0:
                                            mma2ld_pipe.producer_wait(warp_id)
                                            for ko in Tx.serial(0, K // BLK_K):
                                                tma2mma_pipe.consumer_wait(0)
                                                Tx.ptx.tcgen05.encode_matrix_descriptor(Tx.address_of(base_desc_A), A_smem.ptr_to([tma2mma_pipe.idx, warp_id, 0, 0]), ldo=ldo, sdo=sdo, swizzle=SWIZZLE)
                                                Tx.ptx.tcgen05.encode_matrix_descriptor(Tx.address_of(base_desc_B), B_smem.ptr_to([tma2mma_pipe.idx, 0, 0]), ldo=ldo, sdo=sdo, swizzle=SWIZZLE)
                                                for ki in range(BLK_K // MMA_K):
                                                    descA = base_desc_A + ((ki * MMA_K * 2) >> 0x4)
                                                    descB = base_desc_B + ((ki * MMA_K * 2) >> 0x4)
                                                    if ki == 0 and ko == 0:
                                                        Tx.ptx.tcgen05.mma("float32", a_type, b_type, tmem_addr + warp_id * MMA_N, descA, descB, descI, use_a_tmem=False, cta_group=cta_group, enable_input_d=False)
                                                    else:
                                                        Tx.ptx.tcgen05.mma("float32", a_type, b_type, tmem_addr + warp_id * MMA_N, descA, descB, descI, use_a_tmem=False, cta_group=cta_group, enable_input_d=True)
                                                tma2mma_pipe.consumer_release(0)
                                                tma2mma_pipe.advance()
                                            Tx.ptx.tcgen05.commit(mma2ld_pipe.mbar_p2c.ptr_to([0, 0]), cta_group=2, cta_mask=3)
                                            mma2ld_pipe.advance()
                            with Tx.warpgroup()[0:NUM_CONSUMER]:
                                m_idx = task_smem[task_id, 0]
                                n_idx = task_smem[task_id, 1]
                                mma2ld_pipe.consumer_wait(0)
                                # TMEM -> RF
                                for i in range(BLK_N // TMEM_LD_SIZE):
                                    col_st = Tx.meta_var(wg_id * MMA_N + i * TMEM_LD_SIZE)
                                    Tx.copy(reg_wg[:, :], tmem[:, col_st : col_st + TMEM_LD_SIZE])
                                    with Tx.thread():
                                        Tx.cast(reg_fp16[i * TMEM_LD_SIZE : (i + 1) * TMEM_LD_SIZE], reg[:])
                                mma2ld_pipe.consumer_release(wg_id)
                                # RF -> GMEM
                                for i in range(BLK_N // EPI_TILE):
                                    for it in range(EPI_TILE // 8):
                                        for vec in Tx.vectorized(8):
                                            C_smem[wg_id, warp_id * 32 + lane_id, it * 8 + vec] = reg_fp16[i * EPI_TILE + it * 8 + vec]
                                    Tx.cuda.warpgroup_sync(wg_id+1)
                                    Tx.ptx.fence.proxy(scope="shared")
                                    if lane_id == 0 and warp_id == 0:
                                        Tx.ptx.cp_async.bulk.tensor.s2g(4, C_smem.ptr_to([wg_id, 0, 0]), C_tensor_map, i * EPI_TILE, 0, n_idx, m_idx * 4 + wg_id * 2 + cbx)
                                        Tx.ptx.cp_async.bulk.commit_group()
                                        Tx.ptx.cp_async.bulk.wait_group(0)
                                    Tx.cuda.warpgroup_sync(wg_id+1)
                                # notify RS ready
                                comm_m_idx = Tx.meta_var(m_idx * 4 + wg_id * 2 + cbx)
                                comm_m_idx_local = Tx.meta_var(comm_m_idx % (LOCAL_M // TILE_M))
                                signal_rank = Tx.meta_var(comm_m_idx // (LOCAL_M // TILE_M))
                                if signal_rank == rank:
                                    sem.semaphore_notify(tid, comm_m_idx_local, n_idx) # notify self
                                else:
                                    if tid % 128 == 0:
                                        Tx.nvshmem.signal_op(sem.sem.ptr_to([comm_m_idx_local, n_idx]), 1, "add", signal_rank)
                                mma2ld_pipe.advance()
                        profiler.end(ProfileEventType.GEMM, tid == 0)

                    elif task_smem[task_id, 2] == JobType.RS.value:
                        Tx.cuda.cta_sync()
                        m_idx = task_smem[task_id, 0]
                        n_idx = task_smem[task_id, 1]
                        sem.semaphore_wait(m_idx, n_idx)
                        profiler.start(ProfileEventType.RS, tid == 0)
                        offset = tid
                        while True:
                            if offset < TILE_M * TILE_N // 8:
                                m_start = Tx.meta_var(offset // (TILE_N // 8))
                                n_start = Tx.meta_var(offset % (TILE_N // 8) * 8)
                                Tx.cuda.func_call(
                                    "ld_reduce_8_fp16",
                                    gemm_out.ptr_to([rank * (LOCAL_M // TILE_M) + m_idx, n_idx, m_start, n_start]),
                                    out.ptr_to([TILE_M * m_idx + m_start, TILE_N * n_idx + n_start]),
                                    source_code=ld_reduce_8xfp16,
                                )
                                offset += NUM_THREADS
                            else:
                                break
                        profiler.end(ProfileEventType.RS, tid == 0)
                    task_id += 1

                # dealloc TMEM
                with Tx.warp()[0:1]:
                    Tx.ptx.tcgen05.relinquish_alloc_permit(cta_group=cta_group)
                    Tx.ptx.tcgen05.dealloc(tmem_addr, n_cols=N_COLS, cta_group=cta_group)
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
    A_tvm = tvm.runtime.tensor(A_np, device=DEV)
    B_tvm = tvm.runtime.tensor(B_np, device=DEV)
    exec_queue_tvm = tvm.runtime.tensor(exec_queue_np, device=DEV)
    semaphore_np = np.zeros((LOCAL_M // TILE_M, N // TILE_N), dtype="uint64")
    profiler_buffer_np = np.zeros((PROFILER_BUFFER_SIZE,), dtype="uint64")

    A_array_all = sess.empty((WORLD_SIZE, M, K), a_type, worker0_only=True)
    B_array_all = sess.empty((WORLD_SIZE, N, K), b_type, worker0_only=True)
    exec_queue_all = sess.empty((WORLD_SIZE, SM_COUNT, MAX_TASKS, 3), "int32", worker0_only=True)
    sess.copy_to_worker_0(A_tvm, A_array_all)
    sess.copy_to_worker_0(B_tvm, B_array_all)
    sess.copy_to_worker_0(exec_queue_tvm, exec_queue_all)

    nvshmem_malloc_hook = sess.get_global_func("runtime.disco.nvshmem.empty")
    args_dict = {
        "A_array": sess.empty((M, K), a_type),
        "B_array": sess.empty((N, K), b_type),
        "exec_queue_array": sess.empty((SM_COUNT, MAX_TASKS, 3), "int32"),
        "gemm_out_array": nvshmem_malloc_hook(
            ShapeTuple((M // TILE_M, N // TILE_N, TILE_M, TILE_N)), d_type, None
        ),
        "semaphore_array": nvshmem_malloc_hook(
            ShapeTuple((LOCAL_M // TILE_M, N // TILE_N)), "uint64", None
        ),
        "out_array": sess.empty((LOCAL_M, N), d_type),
        "profiler_buffer_array": sess.empty((PROFILER_BUFFER_SIZE,), "uint64"),
    }

    res_dict = {
        "gemm_out_res": sess.empty(
            (WORLD_SIZE, M // TILE_M, N // TILE_N, TILE_M, TILE_N), d_type, worker0_only=True
        ),
        "buffer_res": sess.empty(
            (WORLD_SIZE, M // BLK_M, N // BLK_N, BLK_M, BLK_N), d_type, worker0_only=True
        ),
        "out_res": sess.empty((WORLD_SIZE, LOCAL_M, N), d_type, worker0_only=True),
        "profiler_buffer_res": sess.empty(
            (WORLD_SIZE, PROFILER_BUFFER_SIZE), "uint64", worker0_only=True
        ),
        "gemm_out_host": tvm.runtime.empty(
            (WORLD_SIZE, M // TILE_M, N // TILE_N, TILE_M, TILE_N), d_type, device=DEV
        ),
        "buffer_host": tvm.runtime.empty(
            (WORLD_SIZE, M // BLK_M, N // BLK_N, BLK_M, BLK_N), d_type, device=DEV
        ),
        "out_host": tvm.runtime.empty((WORLD_SIZE, LOCAL_M, N), d_type, device=DEV),
        "profiler_buffer_host": tvm.runtime.empty(
            (WORLD_SIZE, PROFILER_BUFFER_SIZE), "uint64", device=DEV
        ),
    }

    sess.scatter_from_worker0(A_array_all, args_dict["A_array"])
    sess.scatter_from_worker0(B_array_all, args_dict["B_array"])
    sess.scatter_from_worker0(exec_queue_all, args_dict["exec_queue_array"])
    sess.sync_worker_0()
    print(f"Data prepared successfully")

    with tempfile.TemporaryDirectory() as tmpdir:
        target = tvm.target.Target("cuda")
        path = tmpdir + "/test.so"
        mod = tvm.compile(test_mma_ss_tma_2sm_persistent, target=target, tir_pipeline="tirx")
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
                args_dict["out_array"],
                args_dict["exec_queue_array"],
                args_dict["profiler_buffer_array"],
            )
            sess._sync_all()
            if itr < N_REPEAT - 1:
                sess.broadcast(semaphore_np, args_dict["semaphore_array"])
                sess.broadcast(profiler_buffer_np, args_dict["profiler_buffer_array"])
                sess._sync_all()

        # validate results
        sess.gather_to_worker0(args_dict["gemm_out_array"], res_dict["gemm_out_res"])
        sess.copy_from_worker_0(res_dict["gemm_out_host"], res_dict["gemm_out_res"])
        sess.gather_to_worker0(args_dict["out_array"], res_dict["out_res"])
        sess.copy_from_worker_0(res_dict["out_host"], res_dict["out_res"])
        sess.gather_to_worker0(args_dict["profiler_buffer_array"], res_dict["profiler_buffer_res"])
        sess.copy_from_worker_0(res_dict["profiler_buffer_host"], res_dict["profiler_buffer_res"])

        # sync all workers to make sure the temporary files are cleaned up after all workers
        # finish the execution
        sess._sync_all()
        print("Kernel execution finished.")

    finalize_dfunc = sess.get_global_func("runtime.disco.nvshmem.finalize_nvshmem")
    finalize_dfunc()
    sess.sync_worker_0()

    # validate results
    print("Validating results...")

    import torch

    gemm_out_torch = torch.zeros((WORLD_SIZE, M, N), dtype=torch.float16, device="cuda")
    gemm_out_torch_sum = torch.zeros((M, N), dtype=torch.float16, device="cuda")
    for i in range(WORLD_SIZE):
        print(f"rank {i} validating...")
        A_torch = torch.tensor(A_np[i], dtype=torch.float16, device="cuda")
        B_torch = torch.tensor(B_np[i], dtype=torch.float16, device="cuda")
        gemm_out_torch[i] = torch.matmul(A_torch, B_torch.T)
        gemm_out_res = np.transpose(res_dict["gemm_out_host"].numpy()[i], (0, 2, 1, 3)).reshape(
            M, N
        )
        np.testing.assert_allclose(
            gemm_out_res, gemm_out_torch[i].cpu().numpy(), atol=1e-3, rtol=1e-3
        )

    gemm_out_torch_sum = torch.sum(gemm_out_torch, dim=0)
    out_res = res_dict["out_host"].numpy().reshape(-1, N)
    np.testing.assert_allclose(out_res, gemm_out_torch_sum.cpu().numpy(), atol=1e-3, rtol=1e-3)

    print("Results all correct.")

    # profiler results
    for rank in range(WORLD_SIZE):
        if rank == 7:
            export_to_perfetto_trace(
                res_dict["profiler_buffer_host"].numpy()[rank],
                f"static-schedule-hgemm-RS-rank{rank}-rs-sms-{RS_SMS}.perfetto-trace",
                event_type_names,
            )


if __name__ == "__main__":
    test_hgemm_rs()
