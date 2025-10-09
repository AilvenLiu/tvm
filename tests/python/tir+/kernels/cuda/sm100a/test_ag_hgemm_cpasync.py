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
from tvm.script import tir as T
from tvm.script import tirp as Tp
from tvm.tir.event import EventImpl
from tvm.tir.layout import TileLayout
from tvm.script.ir_builder import IRBuilder
from tvm.tirp.bench.utils import export_to_perfetto_trace, CudaProfiler


class TaskType(Enum):
    GEMM = 0
    AG = 1


class ProfileEventType(Enum):
    GEMM = 0
    AG = 1
    FETCH = 2


event_type_names = [
    "gemm",
    "ag",
    "fetch",
]

# M, N, K = 16384, 49152, 12288
M, N, K = 8192, 8192 * 8, 8192

M_CLUSTER = 2
N_CLUSTER = 1
WG_NUMBER = 3
WARP_NUMBER = 4
NUM_CONSUMER = 2
NUM_THREADS = (32 * WARP_NUMBER) * WG_NUMBER
SM_NUMBER = 148

PIPELINE_DEPTH = 4

F16_BYTES = 2
F32_BYTES = 4
F128_BYTES = 16

d_type, a_type, b_type = "float16", "float16", "float16"
WORLD_SIZE = 8
LOCAL_M = M // WORLD_SIZE
LOCAL_N = N // WORLD_SIZE
BLK_M, BLK_N, BLK_K = 128, 128, 64
assert LOCAL_M * WORLD_SIZE == M, "M must be divisible by WORLD_SIZE"
assert LOCAL_M % BLK_M == 0, "LOCAL_M must be divisible by BLK_M"
assert LOCAL_N * WORLD_SIZE == N, "N must be divisible by WORLD_SIZE"
assert LOCAL_N % BLK_N == 0, "LOCAL_N must be divisible by BLK_N"

MMA_M, MMA_N, MMA_K = 256, 256, 16
EPI_TILE = 64
SWIZZLE = 3
SMEM_SIZE = (
    PIPELINE_DEPTH * NUM_CONSUMER * BLK_M * BLK_K * F16_BYTES
    + PIPELINE_DEPTH * BLK_N * BLK_K * F16_BYTES
    + NUM_CONSUMER * BLK_M * EPI_TILE * F16_BYTES
    + 1024
)
assert SMEM_SIZE <= 232448

TMEM_LD_SIZE = 64
N_COLS = 512
CTA_GROUP = 2

PIPE_CYCLE = (K // BLK_K) // PIPELINE_DEPTH
PIPE_REMAIN_NUM = (K // BLK_K) % PIPELINE_DEPTH
assert PIPELINE_DEPTH == 4

GROUP_SIZE = min(8, LOCAL_M // (BLK_M * NUM_CONSUMER * CTA_GROUP))
assert M % (NUM_CONSUMER * BLK_M * CTA_GROUP) == 0
assert N % (BLK_N * CTA_GROUP) == 0
GEMM_M_CLUSTERS = M // (NUM_CONSUMER * BLK_M * CTA_GROUP)  # gemm tile m: 512
GEMM_N_CLUSTERS = LOCAL_N // (BLK_N * CTA_GROUP)  # gemm tile n: 256
LOCAL_GEMM_M_CLUSTERS = GEMM_M_CLUSTERS // WORLD_SIZE

# dyn scheduling
CAPACITY = 2048
TASK_IDX_LEN = 2
ENABLE_WARP_BROADCAST = False
C2P_THREAD_COUNT = 12 * 2 if ENABLE_WARP_BROADCAST else NUM_THREADS * 2

# profiling
WARMUP_ITERS = 5
# WARMUP_ITERS = 0
TOTAL_ITERS = 30
# TOTAL_ITERS = 1

PROFILER_ON = False
NUM_GROUPS = 13
PROFILER_BUFFER_SIZE = int(1e7)
PROFILER_WRITE_STRIDE = SM_NUMBER * NUM_GROUPS
CUDA_EVENT_PROFILER = False
if CUDA_EVENT_PROFILER:
    PROFILER_ON = False
VALIDATE = True


pack_values = """
__forceinline__ __device__ void pack_values(int32_t rem, int32_t task_type, int32_t task_idx0, int32_t task_idx1, uint64_t* dst_addr) {
    asm volatile("st.shared::cluster.v4.b32 [%0], {%1, %2, %3, %4};"
                 :
                 : "l"(dst_addr), "r"(rem), "r"(task_type), "r"(task_idx0), "r"(task_idx1)
                 : "memory");
}
"""

unpack_values = """
__forceinline__ __device__ void unpack_values(uint64_t* src_addr, int32_t* rem, int32_t* task_type, int32_t* task_idx0, int32_t* task_idx1) {
    asm volatile("ld.shared::cluster.v4.b32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(*rem), "=r"(*task_type), "=r"(*task_idx0), "=r"(*task_idx1)
                 : "l"(src_addr)
                 : "memory");

}
"""

semaphore_notify_remote = """
__forceinline__ __device__ uint64_t semaphore_notify_remote(int32_t signal_rank, uint64_t* addr, uint64_t signal_value) {
    auto dst_addr = reinterpret_cast<unsigned long long*>(nvshmem_ptr(addr, signal_rank));
    return atomicAdd_system(dst_addr, signal_value);
}
"""

enqueue_remote = """
__forceinline__ __device__ void enqueue_remote(int32_t* task_types, int32_t* task_idxs, int32_t* tail, int32_t mask,
                                               int32_t signal_rank, int32_t task_type, int32_t task_idx0, int32_t task_idx1) {
    int32_t* remote_task_types = (int32_t*)nvshmem_ptr(task_types, signal_rank);
    int32_t* remote_task_idxs = (int32_t*)nvshmem_ptr(task_idxs, signal_rank);
    int32_t* remote_tail = (int32_t*)nvshmem_ptr(tail, signal_rank);
    int32_t tail_r = atomicAdd(&(remote_tail[0]), 1);
    int32_t masked_pos = tail_r & mask;
    remote_task_types[masked_pos] = task_type;
    remote_task_idxs[masked_pos * 2] = task_idx0;
    remote_task_idxs[masked_pos * 2 + 1] = task_idx1;
    __threadfence();
}
"""

ld_global_acquire = f"""
__forceinline__ __device__ int32_t ld_global_acquire(int32_t* addr) {{
  int32_t res;
  asm volatile ("ld.global.acquire.gpu.b32 %0, [%1];\\n" : "=r"(res) : "l"(addr));
  return res;
}}
"""

while_ld_global_acquire = f"""
__forceinline__ __device__ int32_t while_ld_global_acquire(int32_t* addr) {{
  int32_t res;
  asm volatile ("ld.global.acquire.gpu.b32 %0, [%1];\\n" : "=r"(res) : "l"(addr));
  while (res < 0) {{
    __nanosleep(40);
    asm volatile ("ld.global.acquire.gpu.b32 %0, [%1];\\n" : "=r"(res) : "l"(addr));
  }}
  return res;
}}
"""

warp_broadcast = """
__forceinline__ __device__ void warp_broadcast(int32_t* rem, int32_t* task_type, int32_t* task_idx0, int32_t* task_idx1) {{
    *rem = __shfl_sync(0xFFFFFFFF, *rem, 0);
    *task_type = __shfl_sync(0xFFFFFFFF, *task_type, 0);
    *task_idx0 = __shfl_sync(0xFFFFFFFF, *task_idx0, 0);
    *task_idx1 = __shfl_sync(0xFFFFFFFF, *task_idx1, 0);
}}
"""


class Barriers:

    def __init__(self, shared_buffer_base, shared_buffer_offs, pipe_depth, pipe_width, is_p2c):
        self.mbar: tvm.tir.Buffer = T.decl_buffer(
            (pipe_depth, pipe_width), "uint64", shared_buffer_base, elem_offset=shared_buffer_offs
        ).buffer
        self.init_phase = 0 if is_p2c else 1
        self.pipe_depth = pipe_depth
        self.pipe_width = pipe_width

    @T.macro
    def init(self, threads_num_wait):
        with T.thread()[0:1]:
            for i in T.serial(self.pipe_depth):
                for j in T.serial(self.pipe_width):
                    T.ptx.mbarrier.init(self.mbar.ptr_to([i, j]), threads_num_wait)

    @T.macro
    def wait(self, idx_d, idx_w, phase):
        T.ptx.mbarrier.try_wait(self.mbar.ptr_to([idx_d, idx_w]), self.init_phase ^ phase)


class BarTMA2MMA(Barriers):

    @T.macro
    def arrive(self, idx, expected_bytes):
        T.ptx.mbarrier.arrive.expect_tx(self.mbar.ptr_to([idx, 0]), expected_bytes)

    @T.macro
    def arrive_only(self, idx):
        T.ptx.mbarrier.arrive(self.mbar.ptr_to([idx, 0]))


class BarMMA2LD(Barriers):

    @T.macro
    def arrive(self, idx):
        T.ptx.tcgen05.commit(self.mbar.ptr_to([0, idx]), cta_group=CTA_GROUP, cta_mask=3)


class BarMMA2TMA(Barriers):

    @T.macro
    def arrive(self, idx):
        T.ptx.tcgen05.commit(self.mbar.ptr_to([idx, 0]), cta_group=CTA_GROUP, cta_mask=3)


class BarLD2MMA(Barriers):

    @T.macro
    def arrive(self, idx):
        T.ptx.mbarrier.arrive(self.mbar.ptr_to([0, idx]), cta_id=0, pred=True)


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
            for cbx in T.thread_binding(M_CLUSTER, "clusterCtaIdx.x"):
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
        for cbx in T.thread_binding(M_CLUSTER, "clusterCtaIdx.x"):
            if not self.p_single_cta or cbx == 0:
                T.ptx.mbarrier.try_wait(
                    self.mbar_c2p.ptr_to([self.idx, pipeline_idx]), self.c2p_phase
                )

    @T.macro
    def consumer_wait(self, pipeline_idx):
        for cbx in T.thread_binding(M_CLUSTER, "clusterCtaIdx.x"):
            if not self.c_single_cta or cbx == 0:
                T.ptx.mbarrier.try_wait(
                    self.mbar_p2c.ptr_to([self.idx, pipeline_idx]), self.p2c_phase
                )


def int_var(name: str, scope="local", dtype="int32", align=4):
    buf = T.alloc_buffer([1], dtype, scope=scope, align=align, name=name)
    return buf


class Semaphore:
    def __init__(self, cnt, buffer):
        self.cnt = cnt
        self.sem = buffer
        self.state = T.alloc_buffer([1], "uint64", scope="local", align=4, name="semaphore_state")

    @T.macro
    def semaphore_wait(self, *coord):
        with T.thread():
            while 1:
                T.ptx.ld_global_acquire(
                    self.state[0], self.sem.access_ptr("r", offset=self.sem.elem_offset_of(coord))
                )
                if self.state[0] == self.cnt:
                    break
                T.cuda.nano_sleep(40)


class MPMCQueue:
    def __init__(
        self,
        capacity: int,
        task_types: T.Buffer,
        task_idxs: T.Buffer,
        head: T.Buffer,
        tail: T.Buffer,
        num_tot_tasks: int,
    ):
        if capacity & (capacity - 1):
            raise ValueError("capacity must be a power-of-two")
        self.capacity = capacity
        self.mask = capacity - 1
        self.task_types = task_types
        self.task_idxs = task_idxs
        self.head = head
        self.tail = tail
        self.head_r = int_var("head_r")
        self.tail_r = int_var("tail_r")
        self.pos = int_var("pos")
        self.masked_pos = int_var("masked_pos")
        self.num_tot_tasks = num_tot_tasks

    @T.macro
    def enqueue(self, signal_rank: int, task_type: int, *task_idx: int):
        T.cuda.func_call(
            "enqueue_remote",
            self.task_types.ptr_to([0]),
            self.task_idxs.ptr_to([0, 0]),
            self.tail.ptr_to([0]),
            self.mask,
            signal_rank,
            task_type,
            *task_idx,
            source_code=enqueue_remote,
        )


class GEMMMPMCQueue(MPMCQueue):

    @T.macro
    def dequeue(
        self,
        fetched_task_type: T.Buffer,
        fetched_task_idx0: T.Buffer,
        fetched_task_idx1: T.Buffer,
        sem: Semaphore,
        cbx,
        bx,
        rank,
    ):
        self.head_r[0] = T.cuda.atomic_add(
            self.head.access_ptr("rw", offset=self.head.elem_offset_of([T.int32(0)])), 1
        )
        if self.head_r[0] < self.num_tot_tasks:
            # TODO: modify the wait logic to make it faster
            remote_rank = (
                rank + (self.head_r[0] // (LOCAL_GEMM_M_CLUSTERS * GEMM_N_CLUSTERS))
            ) % WORLD_SIZE
            if remote_rank != rank:
                sem.semaphore_wait(remote_rank)

            self.masked_pos[0] = self.head_r[0] & self.mask
            fetched_task_type[0] = T.cuda.func_call(
                "while_ld_global_acquire",
                self.task_types.access_ptr(
                    "r", offset=self.task_types.elem_offset_of([self.masked_pos[0]])
                ),
                source_code=while_ld_global_acquire,
                return_type="int32",
            )
            self.task_types[self.masked_pos[0]] = -1
            fetched_task_idx0[0] = self.task_idxs[self.masked_pos[0], 0]
            fetched_task_idx1[0] = self.task_idxs[self.masked_pos[0], 1]
        else:
            fetched_task_type[0] = -1


# fmt: off
@T.macro
def consumer_fetch(
    sch_pipe,
    packed_value,
    rs_rem,
    fetched_task_type,
    fetched_task_idx0,
    fetched_task_idx1,
):
    sch_pipe.consumer_wait(0)
    T.cuda.func_call(
        "unpack_values",
        packed_value.ptr_to([0]),
        rs_rem.ptr_to([0]),
        fetched_task_type.ptr_to([0]),
        fetched_task_idx0.ptr_to([0]),
        fetched_task_idx1.ptr_to([0]),
        source_code=unpack_values,
    )
    T.ptx.mbarrier.arrive(sch_pipe.mbar_c2p.ptr_to([sch_pipe.idx, 0]), cta_id=0, pred=True)
    sch_pipe.p2c_phase = sch_pipe.p2c_phase ^ 1
# fmt: on


class SingleDynamicTileScheduler:
    def __init__(
        self,
        queue: MPMCQueue,
        packed_value: T.Buffer,
        sch_pipe: Pipeline,
        sem: Semaphore,
    ):
        self.queue = queue
        self.sch_pipe = sch_pipe
        self.fetched_task_type = int_var("fetched_task_type")
        self.fetched_task_idx0 = int_var("fetched_task_idx0")
        self.fetched_task_idx1 = int_var("fetched_task_idx1")
        self.sem = sem
        self.rs_rem = int_var("rs_rem")
        self.packed_value = packed_value
        IRBuilder.current().name("packed_value", self.packed_value)

    # fmt: off
    @T.macro
    def _fetch_from_queue(
        self,
        cbx,
        bx,
        rank,
        warp_id_in_cta,
        lane_id,
    ):
        # fetch from GEMM queue
        if warp_id_in_cta == 11 and lane_id == 0:
            if cbx == 0:
                self.sch_pipe.producer_wait(0)
                self.queue.dequeue(self.fetched_task_type, self.fetched_task_idx0, self.fetched_task_idx1, self.sem, cbx, bx, rank)
                T.cuda.func_call(
                    "pack_values",
                    self.rs_rem[0],
                    self.fetched_task_type[0],
                    self.fetched_task_idx0[0],
                    self.fetched_task_idx1[0],
                    self.packed_value.ptr_to([0]),
                    source_code=pack_values,
                )
                # T.cuda.thread_fence()
                T.ptx.mbarrier.arrive(self.sch_pipe.mbar_p2c.ptr_to([self.sch_pipe.idx, 0]), cta_id=0, pred=True)
                T.ptx.mbarrier.arrive(self.sch_pipe.mbar_p2c.ptr_to([self.sch_pipe.idx, 0]), cta_id=1, pred=True)
                self.sch_pipe.c2p_phase = self.sch_pipe.c2p_phase ^ 1
        if ENABLE_WARP_BROADCAST:
            if lane_id == 0:
                consumer_fetch(self.sch_pipe, self.packed_value, self.rs_rem, self.fetched_task_type, self.fetched_task_idx0, self.fetched_task_idx1)
            T.cuda.func_call(
                "warp_broadcast",
                self.rs_rem.ptr_to([0]),
                self.fetched_task_type.ptr_to([0]),
                self.fetched_task_idx0.ptr_to([0]),
                self.fetched_task_idx1.ptr_to([0]),
                source_code=warp_broadcast,
            )
        else:
            consumer_fetch(self.sch_pipe, self.packed_value, self.rs_rem, self.fetched_task_type, self.fetched_task_idx0, self.fetched_task_idx1)

    @T.macro
    def init(self, cbx, bx, rank, warp_id_in_cta, lane_id):
        self.rs_rem[0] = -1
        self._fetch_from_queue(cbx, bx, rank, warp_id_in_cta, lane_id)

    @T.macro
    def next_tile(self, cbx, bx, rank, warp_id_in_cta, lane_id):
        self._fetch_from_queue(cbx, bx, rank, warp_id_in_cta, lane_id)

    def valid(self):
        return tvm.tir.any(self.fetched_task_type[0] >= 0, self.rs_rem[0] >= 0)
    # fmt: on


@T.macro
def skip():
    pass


@pytest.mark.skip()
@tvm.testing.requires_cuda_compute_version(10, exact=True)
def test_ag_hgemm():
    A_layout = T.ComposeLayout(
        T.SwizzleLayout(3, 3, 3, swizzle_inner=True),
        T.TileLayout(
            shard=(
                (PIPELINE_DEPTH, NUM_CONSUMER, BLK_M, BLK_K),
                (NUM_CONSUMER * BLK_M * BLK_K, BLK_M * BLK_K, BLK_K, 1),
            )
        ),
    )
    B_layout = T.ComposeLayout(
        T.SwizzleLayout(3, 3, 3, swizzle_inner=True),
        T.TileLayout(shard=((PIPELINE_DEPTH, BLK_N, BLK_K), (BLK_N * BLK_K, BLK_K, 1))),
    )
    D_layout = T.ComposeLayout(
        T.SwizzleLayout(3, 3, 3, swizzle_inner=True),
        T.TileLayout(shard=((NUM_CONSUMER, BLK_M, EPI_TILE), (BLK_M * EPI_TILE, EPI_TILE, 1))),
    )

    # fmt: off
    @T.prim_func(tirp=True)
    def test_mma_ss_tma_2sm_persistent(A: T.Buffer((LOCAL_M, K), a_type), B: T.Buffer((LOCAL_N, K), b_type), ag_out: T.Buffer((M, K), a_type),
                                       semaphore: T.Buffer((WORLD_SIZE,), "uint64"), out: T.Buffer((M, LOCAL_N), d_type), profiler_buffer: T.Buffer((PROFILER_BUFFER_SIZE,), "uint64"),
                                       gemm_task_types: T.Buffer((CAPACITY,), "int32"), gemm_task_idxs: T.Buffer((CAPACITY, 2), "int32"), gemm_head: T.Buffer((1,), "int32"), gemm_tail: T.Buffer((1,), "int32")):
        with T.kernel():
            cbx, cby = T.cta_id([M_CLUSTER, N_CLUSTER], parent="cluster")
            bx = T.cta_id([SM_NUMBER], parent="kernel")
            wg_id = T.warpgroup_id([WG_NUMBER], parent="cta")
            warp_id = T.warp_id([WARP_NUMBER], parent="warpgroup")
            warp_id_in_cta = T.warp_id([WG_NUMBER * WARP_NUMBER], parent="cta")
            lane_id = T.thread_id([32], parent="warp")
            tid = T.thread_id([NUM_THREADS], parent="cta")
            rank = T.nvshmem.my_pe()
            with T.cta():
                # alloc shared memory
                buf = T.alloc_buffer([SMEM_SIZE], "uint8", scope="shared.dyn")
                tmem_addr = T.decl_cell("uint32", buf.data, scope="shared.dyn", elem_offset=0)
                A_smem = T.decl_buffer((PIPELINE_DEPTH, NUM_CONSUMER, BLK_M, BLK_K), a_type, buf.data, layout=A_layout,
                                        elem_offset=1024 // F16_BYTES)
                B_smem = T.decl_buffer((PIPELINE_DEPTH, BLK_N, BLK_K), b_type, buf.data, layout=B_layout,
                                        elem_offset=1024 // F16_BYTES + PIPELINE_DEPTH * NUM_CONSUMER * BLK_M * BLK_K)
                D_smem = T.decl_buffer((NUM_CONSUMER, BLK_M, EPI_TILE), d_type, buf.data, layout=D_layout,
                                        elem_offset=1024 // F16_BYTES + PIPELINE_DEPTH * (NUM_CONSUMER * BLK_M + BLK_N) * BLK_K)

                # alloc local memory
                descA = T.local_cell("uint64")
                descB = T.local_cell("uint64")
                descI = T.local_cell("uint32")
                phase = T.alloc_buffer((1,), "int32", scope="local")
                phase_tmem = T.alloc_buffer((1,), "int32", scope="local")
                stage = T.local_cell("int32")

                # ag + gemm
                sem = T.meta_var(Semaphore(cnt=1, buffer=semaphore))
                gemm_queue = T.meta_var(GEMMMPMCQueue(CAPACITY, gemm_task_types, gemm_task_idxs, gemm_head, gemm_tail, GEMM_M_CLUSTERS * GEMM_N_CLUSTERS))
                packed_buf = T.decl_buffer((1,), "uint64", buf.data, elem_offset=64)
                packed_ptr: T.Var(name="packed_ptr", dtype=PointerType(PrimType("uint64"))) = T.reinterpret("handle", T.ptx.map_shared_rank(packed_buf.ptr_to([0]), 0)) # rank: 0
                packed_value = T.decl_buffer([1,], "uint64", data=packed_ptr, scope="shared")
                sch_pipe = T.meta_var(Pipeline(buf.data, 64 + 4, pipeline_depth=1, pipeline_num=1, p_single_cta=True, c_single_cta=False))
                tile_scheduler = T.meta_var(SingleDynamicTileScheduler(gemm_queue, packed_value, sch_pipe, sem))
                profiler = T.meta_var(
                    CudaProfiler(
                        profiler_buffer,
                        write_stride=PROFILER_WRITE_STRIDE,
                        num_groups=NUM_GROUPS,
                        profiler_enabled=PROFILER_ON,
                    )
                )

                # initialize
                profiler.init(warp_id_in_cta)
                tma2mma = T.meta_var(BarTMA2MMA(buf.data, 4, PIPELINE_DEPTH, 1, is_p2c=True))
                mma2tma = T.meta_var(BarMMA2TMA(buf.data, 4 + PIPELINE_DEPTH, PIPELINE_DEPTH, 1, is_p2c=False))
                mma2ld = T.meta_var(BarMMA2LD(buf.data, 4 + 2 * PIPELINE_DEPTH, 1, NUM_CONSUMER, is_p2c=True))
                ld2mma = T.meta_var(BarLD2MMA(buf.data, 4 + 2 * PIPELINE_DEPTH + NUM_CONSUMER, 1, NUM_CONSUMER, is_p2c=False))
                tma2mma.init(1)
                mma2tma.init(NUM_CONSUMER)
                mma2ld.init(1)
                ld2mma.init(128 * NUM_CONSUMER)
                ptr: T.Var(name="ptr", dtype=PointerType(PrimType("uint64"))) = T.reinterpret("handle", T.ptx.map_shared_rank(tma2mma.mbar.ptr_to([0, 0]), 0))
                tma_finished = T.decl_buffer([PIPELINE_DEPTH], "uint64", data=ptr, scope="shared")
                phase[0] = 0
                phase_tmem[0] = 0
                sch_pipe.init(c2p_thread_count=C2P_THREAD_COUNT, p2c_thread_count=1)
                T.ptx.tcgen05.encode_instr_descriptor(T.address_of(descI), "float32", a_type, b_type, MMA_M, MMA_N, MMA_K, False, False, CTA_GROUP)

                # define events
                tma_event = Tp.alloc_semaphore_event_tensor(EventImpl.kTMALoad, state=[tma_finished, None, None], shape=[PIPELINE_DEPTH])
                wb_event = Tp.alloc_bulk_group_event(EventImpl.kTMAStore)

                # alloc TMEM
                with T.warp()[0:1]:
                    T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=N_COLS, cta_group=2)

                T.ptx.barrier.cluster.arrive()
                T.ptx.barrier.cluster.wait()
                T.cuda.cta_sync()
                T.ptx.fence.proxy("shared")
                T.ptx.fence.mbarrier_init()
                tile_scheduler.init(cbx, bx, rank, warp_id_in_cta, lane_id)

                T.cuda.trap_when_assert_failed(tmem_addr == 0)
                tmem = T.decl_buffer((128, N_COLS), "float32", scope="tmem", allocated_addr=0, layout=TileLayout(([128, N_COLS], [(1, "TCol"), (1, "TLane")])))
                
                @T.macro
                def paritioned_loop(main_loop, epilogue1, epilogue2):
                    for ko in T.serial(PIPE_CYCLE):
                        for ks in T.unroll(PIPELINE_DEPTH):
                            stage = ko * PIPELINE_DEPTH + ks
                            main_loop(False, ks)
                        phase[0] = phase[0] ^ 1
                    if PIPE_REMAIN_NUM > 0:
                        # last remained loop
                        for ks in T.unroll(PIPE_REMAIN_NUM):
                            stage = PIPE_CYCLE * PIPELINE_DEPTH + ks
                            main_loop(True, ks)
                        epilogue1()
                        # for unaligned cases
                        for ks in T.unroll(PIPE_REMAIN_NUM, PIPELINE_DEPTH):
                            epilogue2(ks)
                        phase[0] = phase[0] ^ 1
                    else:
                        epilogue1()

                with T.cta():
                    while tile_scheduler.valid():
                        if tile_scheduler.fetched_task_type[0] == TaskType.GEMM.value:
                            profiler.start(ProfileEventType.GEMM, tid == 0)
                            with T.cta():
                                m_idx = T.meta_var(tile_scheduler.fetched_task_idx0[0])
                                n_idx = T.meta_var(tile_scheduler.fetched_task_idx1[0])

                                T.block_attr({"tirp.scope_partition": True})
                                with T.warpgroup()[NUM_CONSUMER:NUM_CONSUMER + 1]:
                                    T.ptx.setmaxnreg(False, 56)
                                    if warp_id == 3: 
                                        # GMEM -> SMEM  (tma)
                                        with T.thread()[T.ptx.elect_sync()]:
                                            n_start = T.meta_var((n_idx * CTA_GROUP + cbx) * BLK_N)

                                            @T.macro
                                            def tma_load(is_remain, ks):
                                                stage_k = T.meta_var(stage * BLK_K)
                                                mma2tma.wait(ks, 0, phase[0])
                                                if rank * LOCAL_GEMM_M_CLUSTERS <= m_idx and m_idx < (rank + 1) * LOCAL_GEMM_M_CLUSTERS:
                                                    m_start0 = T.meta_var(((m_idx % LOCAL_GEMM_M_CLUSTERS) * NUM_CONSUMER * CTA_GROUP + cbx) * BLK_M)
                                                    m_start1 = T.meta_var(((m_idx % LOCAL_GEMM_M_CLUSTERS) * NUM_CONSUMER * CTA_GROUP + CTA_GROUP + cbx) * BLK_M)
                                                    Tp.copy_async(A_smem[ks, 0, :, :], A[m_start0 : m_start0 + BLK_M, stage_k : stage_k + BLK_K], evt=tma_event[ks], cta_group=2)
                                                    Tp.copy_async(A_smem[ks, 1, :, :], A[m_start1 : m_start1 + BLK_M, stage_k : stage_k + BLK_K], evt=tma_event[ks], cta_group=2)
                                                else:
                                                    m_start0 = T.meta_var((m_idx * NUM_CONSUMER * CTA_GROUP + cbx) * BLK_M)
                                                    m_start1 = T.meta_var((m_idx * NUM_CONSUMER * CTA_GROUP + CTA_GROUP + cbx) * BLK_M)
                                                    Tp.copy_async(A_smem[ks, 0, :, :], ag_out[m_start0 : m_start0 + BLK_M, stage_k : stage_k + BLK_K], evt=tma_event[ks], cta_group=2)
                                                    Tp.copy_async(A_smem[ks, 1, :, :], ag_out[m_start1 : m_start1 + BLK_M, stage_k : stage_k + BLK_K], evt=tma_event[ks], cta_group=2)
                                                Tp.copy_async(B_smem[ks, :, :], B[n_start : n_start + BLK_N, stage_k : stage_k + BLK_K], evt=tma_event[ks], cta_group=2)
                                                if cbx == 0:
                                                    tma2mma.arrive(ks, NUM_CONSUMER * BLK_K * (BLK_M * NUM_CONSUMER + BLK_N) * F16_BYTES)

                                            @T.macro
                                            def tma_load_epilogue(ks):
                                                mma2tma.wait(ks, 0, phase[0])
                                                if cbx == 0:
                                                    tma2mma.arrive_only(ks)

                                            paritioned_loop(tma_load, skip, tma_load_epilogue)
                            
                                    elif warp_id < 2 and cbx == 0:
                                        with T.thread():
                                            if T.ptx.elect_sync():
                                                ld2mma.wait(0, warp_id, phase_tmem[0])
                                                T.ptx.tcgen05.fence.after_thread_sync()

                                                @T.macro
                                                def mma(is_remain, ks):
                                                    # wait tma
                                                    tma2mma.wait(ks, 0, phase[0])
                                                    for ki in T.unroll(BLK_K // MMA_K):
                                                        T.ptx.tcgen05.encode_matrix_descriptor(T.address_of(descA), A_smem.ptr_to([ks, warp_id, 0, ki * MMA_K]), 
                                                                                                ldo=1, sdo=8 * BLK_K * F16_BYTES // F128_BYTES, swizzle=SWIZZLE)
                                                        T.ptx.tcgen05.encode_matrix_descriptor(T.address_of(descB), B_smem.ptr_to([ks, 0, ki * MMA_K]), 
                                                                                                ldo=1, sdo=8 * BLK_K * F16_BYTES // F128_BYTES, swizzle=SWIZZLE)
                                                        
                                                        if (stage == 0 and ki == 0) and ((not is_remain) or (is_remain and PIPE_CYCLE == 0)):
                                                            T.ptx.tcgen05.mma("float32", a_type, b_type, warp_id * MMA_N, descA, descB, 
                                                                                descI, False, CTA_GROUP, False)
                                                        else:
                                                            T.ptx.tcgen05.mma("float32", a_type, b_type, warp_id * MMA_N, descA, descB, 
                                                                                descI, False, CTA_GROUP, True)
                                                    mma2tma.arrive(ks)

                                                @T.macro
                                                def mma_epilogue1():
                                                    mma2ld.arrive(warp_id)

                                                @T.macro
                                                def mma_epilogue2(ks):
                                                    tma2mma.wait(ks, 0, phase[0])
                                                    mma2tma.arrive(ks)

                                                paritioned_loop(mma, mma_epilogue1, mma_epilogue2)
                                                phase_tmem[0] = phase_tmem[0] ^ 1

                                with T.warpgroup()[0:NUM_CONSUMER]:
                                    T.ptx.setmaxnreg(True, 224)

                                    reg = T.alloc_buffer((TMEM_LD_SIZE,), "float32", scope="local")
                                    reg_wg = reg.view(128, TMEM_LD_SIZE, layout=TileLayout(([128, TMEM_LD_SIZE], [(1, "tid_in_wg"), (1, "m")])))
                                    reg_fp16 = T.alloc_buffer((BLK_N * CTA_GROUP,), d_type, scope="local")

                                    mma2ld.wait(0, wg_id, phase_tmem[0])
                                    phase_tmem[0] = phase_tmem[0] ^ 1
                                    T.ptx.tcgen05.fence.after_thread_sync()
                                    # TMEM -> RF (ld)
                                    for i in T.unroll(MMA_N // TMEM_LD_SIZE): # load (MMA_M // 2, MMA_N)
                                        col_st = T.meta_var(wg_id * MMA_N + i * TMEM_LD_SIZE)
                                        Tp.copy(reg_wg[:, :], tmem[:, col_st : col_st + TMEM_LD_SIZE])
                                        with T.thread():
                                            Tp.cast(reg_fp16[i * TMEM_LD_SIZE : (i + 1) * TMEM_LD_SIZE], reg[:])

                                    # the tmem can be overwritten by the next tile
                                    ld2mma.arrive(wg_id)
                                    # # RF -> GMEM
                                    for i in T.unroll(NUM_CONSUMER * BLK_N // EPI_TILE):
                                        for it in T.unroll(EPI_TILE // 8):
                                            for vec in T.vectorized(8):
                                                D_smem[wg_id, warp_id * 32 + lane_id, it * 8 + vec] = reg_fp16[i * EPI_TILE + it * 8 + vec]
                                        T.ptx.bar.sync(wg_id, 128)
                                        T.ptx.fence.proxy(scope="shared")
                                        # st to gmem
                                        with T.thread()[lane_id == 0 and warp_id == 0]:
                                            m_st = T.meta_var((m_idx * NUM_CONSUMER * CTA_GROUP + wg_id * CTA_GROUP + cbx) * BLK_M)
                                            n_st = T.meta_var(n_idx * BLK_N * CTA_GROUP + i * EPI_TILE)
                                            Tp.copy_async(out[m_st : m_st + BLK_M, n_st : n_st + EPI_TILE], D_smem[wg_id, :, :], evt=wb_event)
                                            wb_event.commit()
                                            wb_event.wait(0)
                                        T.ptx.bar.sync(wg_id, 128)

                            profiler.end(ProfileEventType.GEMM, tid == 0)

                        tile_scheduler.next_tile(cbx, bx, rank, warp_id_in_cta, lane_id)

                # dealloc TMEM
                with T.warp()[0:1]:
                    T.ptx.tcgen05.relinquish_alloc_permit(cta_group=2)
                    T.ptx.tcgen05.dealloc(tmem_addr, n_cols=N_COLS, cta_group=2)

                T.ptx.barrier.cluster.arrive()
                T.ptx.barrier.cluster.wait()

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
    import torch

    torch.manual_seed(42)
    DEV = tvm.cuda(0)
    A_torch = torch.randn([WORLD_SIZE, LOCAL_M, K], dtype=torch.float16)
    B_torch = torch.randn([WORLD_SIZE, LOCAL_N, K], dtype=torch.float16)

    class MPMCQueueHost:
        def __init__(self, capacity: int):
            self.capacity = capacity
            self.task_types = np.full((capacity,), -1, dtype=np.int32)
            self.task_idxs = np.zeros((capacity, 2), dtype=np.int32)
            self.head = np.zeros((1,), dtype=np.int32)
            self.tail = np.zeros((1,), dtype=np.int32)

        def init(self):
            self.head[0] = 0
            self.tail[0] = 0

        def enqueue(self, task_type: TaskType, *task_idx: int):
            pos = self.tail[0] & (self.capacity - 1)
            self.task_types[pos] = task_type.value
            for i in range(TASK_IDX_LEN):
                self.task_idxs[pos, i] = task_idx[i]
            self.tail[0] = self.tail[0] + 1

    nvshmem_malloc_hook = sess.get_global_func("runtime.disco.nvshmem.empty")
    args_dict = {
        "A_array": sess.empty((LOCAL_M, K), a_type),
        "B_array": sess.empty((LOCAL_N, K), b_type),
        "out_array": sess.empty((M, LOCAL_N), d_type),
    }
    for i in range(TOTAL_ITERS):
        args_dict[f"ag_out_array_{i}"] = nvshmem_malloc_hook(ShapeTuple((M, K)), a_type, None)
        args_dict[f"semaphore_array_{i}"] = nvshmem_malloc_hook(
            ShapeTuple((WORLD_SIZE,)), "uint64", None
        )
        args_dict[f"gemm_task_types_array_{i}"] = sess.empty((CAPACITY,), "int32")
        args_dict[f"gemm_task_idxs_array_{i}"] = sess.empty((CAPACITY, 2), "int32")
        args_dict[f"gemm_head_array_{i}"] = sess.empty((1,), "int32")
        args_dict[f"gemm_tail_array_{i}"] = sess.empty((1,), "int32")
        args_dict[f"profiler_buffer_array_{i}"] = sess.empty((PROFILER_BUFFER_SIZE,), "uint64")

    res_dict = {
        "ag_out_res": sess.empty((WORLD_SIZE, M, K), a_type, worker0_only=True),
        "out_res": sess.empty((WORLD_SIZE, M, LOCAL_N), d_type, worker0_only=True),
        "profiler_buffer_res": sess.empty(
            (WORLD_SIZE, PROFILER_BUFFER_SIZE), "uint64", worker0_only=True
        ),
        "ag_out_host": tvm.runtime.empty((WORLD_SIZE, M, K), a_type, device=DEV),
        "out_host": tvm.runtime.empty((WORLD_SIZE, M, LOCAL_N), d_type, device=DEV),
        "profiler_buffer_host": tvm.runtime.empty(
            (WORLD_SIZE, PROFILER_BUFFER_SIZE), "uint64", device=DEV
        ),
    }

    A_tvm = tvm.runtime.tensor(A_torch, device=DEV)
    B_tvm = tvm.runtime.tensor(B_torch, device=DEV)
    A_array = sess.empty((WORLD_SIZE, LOCAL_M, K), a_type, worker0_only=True)
    B_array = sess.empty((WORLD_SIZE, LOCAL_N, K), b_type, worker0_only=True)
    sess.copy_to_worker_0(A_tvm, A_array)
    sess.copy_to_worker_0(B_tvm, B_array)
    sess.scatter_from_worker0(A_array, args_dict["A_array"])
    sess.scatter_from_worker0(B_array, args_dict["B_array"])

    task_types_np = np.empty((WORLD_SIZE, CAPACITY), dtype=np.int32)
    task_idxs_np = np.empty((WORLD_SIZE, CAPACITY, 2), dtype=np.int32)
    head_np = np.empty((WORLD_SIZE,), dtype=np.int32)
    tail_np = np.empty((WORLD_SIZE,), dtype=np.int32)
    for rank in range(WORLD_SIZE):
        gemm_mpmc_queue = MPMCQueueHost(CAPACITY)
        gemm_mpmc_queue.init()
        offset = rank * LOCAL_GEMM_M_CLUSTERS
        for g in range(math.ceil(GEMM_M_CLUSTERS / GROUP_SIZE)):
            for i in range(GEMM_N_CLUSTERS):
                for j in range(g * GROUP_SIZE, min((g + 1) * GROUP_SIZE, GEMM_M_CLUSTERS)):
                    gemm_mpmc_queue.enqueue(TaskType.GEMM, (offset + j) % GEMM_M_CLUSTERS, i)

        task_types_np[rank, :] = gemm_mpmc_queue.task_types
        task_idxs_np[rank, :, :] = gemm_mpmc_queue.task_idxs
        head_np[rank] = gemm_mpmc_queue.head[0]
        tail_np[rank] = gemm_mpmc_queue.tail[0]

    task_types_tvm = tvm.runtime.tensor(task_types_np, device=DEV)
    task_idxs_tvm = tvm.runtime.tensor(task_idxs_np, device=DEV)
    head_tvm = tvm.runtime.tensor(head_np, device=DEV)
    tail_tvm = tvm.runtime.tensor(tail_np, device=DEV)

    task_types_array = sess.empty((WORLD_SIZE, CAPACITY), "int32", worker0_only=True)
    task_idxs_array = sess.empty((WORLD_SIZE, CAPACITY, 2), "int32", worker0_only=True)
    head_array = sess.empty((WORLD_SIZE,), "int32", worker0_only=True)
    tail_array = sess.empty((WORLD_SIZE,), "int32", worker0_only=True)
    sess.copy_to_worker_0(task_types_tvm, task_types_array)
    sess.copy_to_worker_0(task_idxs_tvm, task_idxs_array)
    sess.copy_to_worker_0(head_tvm, head_array)
    sess.copy_to_worker_0(tail_tvm, tail_array)

    for i in range(TOTAL_ITERS):
        sess.scatter_from_worker0(task_types_array, args_dict[f"gemm_task_types_array_{i}"])
        sess.scatter_from_worker0(task_idxs_array, args_dict[f"gemm_task_idxs_array_{i}"])
        sess.scatter_from_worker0(head_array, args_dict[f"gemm_head_array_{i}"])
        sess.scatter_from_worker0(tail_array, args_dict[f"gemm_tail_array_{i}"])

    sess.sync_worker_0()
    print(f"Data prepared successfully")

    with tempfile.TemporaryDirectory() as tmpdir:
        target = tvm.target.Target("cuda")
        path = tmpdir + "/test.so"
        mod = tvm.compile(test_mma_ss_tma_2sm_persistent, target=target, tir_pipeline="tirp")
        # print(mod.mod.imports[0].inspect_source())
        mod.export_library(path)

        print("Begin kernel execution...")
        rt_mod = sess.load_vm_module(path)
        barrier_dfunc = sess.get_global_func("runtime.disco.nvshmem.barrier_all_on_stream")
        transfer_to_peers_dfunc = sess.get_global_func("runtime.disco.transfer_to_peers_all_gather")
        stream_create_dfunc = sess.get_global_func("runtime.disco.stream_create")
        d_stream = stream_create_dfunc()
        stream_sync_dfunc = sess.get_global_func("runtime.disco.stream_sync")
        cur_stream = sess.get_global_func("runtime.get_cuda_stream")()
        if CUDA_EVENT_PROFILER:
            timer_create_dfunc = sess.get_global_func("profiling.cuda.event.create")
            timer_start_dfunc = sess.get_global_func("profiling.cuda.event.start")
            timer_stop_dfunc = sess.get_global_func("profiling.cuda.event.stop")
            timer_result_dfunc = sess.get_global_func("profiling.cuda.event.elapsed")
            timer = timer_create_dfunc()
        sess._sync_all()

        for itr in range(TOTAL_ITERS):
            if CUDA_EVENT_PROFILER and itr == WARMUP_ITERS:
                timer_start_dfunc(timer)
            barrier_dfunc(cur_stream)
            stream_sync_dfunc(cur_stream, d_stream)
            transfer_to_peers_dfunc(
                args_dict[f"semaphore_array_{itr}"],
                args_dict["A_array"],
                args_dict[f"ag_out_array_{itr}"],
                d_stream,
                M,
                K,
                WORLD_SIZE,
            )
            rt_mod["test_mma_ss_tma_2sm_persistent"](
                args_dict["A_array"],
                args_dict["B_array"],
                args_dict[f"ag_out_array_{itr}"],
                args_dict[f"semaphore_array_{itr}"],
                args_dict["out_array"],
                args_dict[f"profiler_buffer_array_{itr}"],
                args_dict[f"gemm_task_types_array_{itr}"],
                args_dict[f"gemm_task_idxs_array_{itr}"],
                args_dict[f"gemm_head_array_{itr}"],
                args_dict[f"gemm_tail_array_{itr}"],
            )

        # get results
        if CUDA_EVENT_PROFILER:
            timer_stop_dfunc(timer)
            timer_res = timer_result_dfunc(timer)
        sess._sync_all()

        if CUDA_EVENT_PROFILER:
            timer_res_np = np.zeros((WORLD_SIZE,), dtype=np.float64)
            for rank in range(WORLD_SIZE):
                timer_res_np[rank] = (
                    timer_res.debug_get_from_remote(rank) / (TOTAL_ITERS - WARMUP_ITERS) / 1e6
                )
            print(f"AG GEMM duration: {timer_res_np.max():.5f} ms")
            for rank in range(WORLD_SIZE):
                print(f"rank {rank}: {timer_res_np[rank]:.5f} ms")

        sess.gather_to_worker0(args_dict[f"ag_out_array_{TOTAL_ITERS-1}"], res_dict["ag_out_res"])
        sess.copy_from_worker_0(res_dict["ag_out_host"], res_dict["ag_out_res"])
        sess.gather_to_worker0(args_dict["out_array"], res_dict["out_res"])
        sess.copy_from_worker_0(res_dict["out_host"], res_dict["out_res"])
        sess.gather_to_worker0(
            args_dict[f"profiler_buffer_array_{TOTAL_ITERS-1}"], res_dict["profiler_buffer_res"]
        )
        sess.copy_from_worker_0(res_dict["profiler_buffer_host"], res_dict["profiler_buffer_res"])

        # sync all workers to make sure the temporary files are cleaned up after all workers
        # finish the execution
        sess._sync_all()
        print("Kernel execution finished.")

    finalize_dfunc = sess.get_global_func("runtime.disco.nvshmem.finalize_nvshmem")
    finalize_dfunc()
    sess.sync_worker_0()

    if VALIDATE:
        print("Validating results...")

        ag_out_ref = A_torch.reshape(-1, K)
        for rank in range(WORLD_SIZE):
            print(f"validating rank: {rank}")
            ag_out_res = res_dict["ag_out_host"].numpy()[rank]
            ag_out_res[rank * LOCAL_M : (rank + 1) * LOCAL_M, :] = ag_out_ref[
                rank * LOCAL_M : (rank + 1) * LOCAL_M, :
            ]
            np.testing.assert_equal(ag_out_ref, ag_out_res)

            ref = torch.matmul(ag_out_ref.cuda(), (B_torch[rank].T).cuda())
            out_res = res_dict["out_host"].numpy()[rank]
            np.testing.assert_allclose(out_res, ref.cpu().numpy(), atol=1e-3, rtol=1e-3)

        print("Results all correct.")

    # profiler results
    if PROFILER_ON:
        for rank in range(WORLD_SIZE):
            export_to_perfetto_trace(
                res_dict["profiler_buffer_host"].numpy()[rank],
                f"dyn-schedule-AG-hgemm-rank{rank}.perfetto-trace",
                event_type_names,
            )


if __name__ == "__main__":
    test_ag_hgemm()
