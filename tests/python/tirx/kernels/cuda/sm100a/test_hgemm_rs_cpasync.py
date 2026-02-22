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
from tvm.script import ir as I
from tvm.script import tirx as Tx
from tvm.script.ir_builder import IRBuilder
from tvm.tirx.bench.utils import export_to_perfetto_trace, CudaProfiler
from tvm.tirx.tile_scheduler import RankAwareGroupMajorTileScheduler
from tvm.tir.layout import TileLayout, tid_in_wg, TLane, TCol, S


class ProfileEventType(Enum):
    GEMM = 0
    RS = 1
    WAIT = 2
    ACCUM = 3
    PUT = 4
    SIGNAL = 5


event_type_names = [
    "gemm",
    "rs",
    "wait",
    "accum",
    "put",
    "signal",
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

# special parameters for RS
GEMM_SMS = SM_COUNT
N_REPEAT = 15
RS_LOAD_PIPE_DEPTH = 6
BLK_N_RS = 128
BLK_M_RS = 128


# profiling
NUM_GROUPS = 5
PROFILER_BUFFER_SIZE = int(1e7)
PROFILER_WRITE_STRIDE = SM_COUNT * NUM_GROUPS


atomic_add_system_uint64 = f"""
__forceinline__ __device__ uint64_t atomic_add_system_uint64(uint64_t* addr, uint64_t value) {{
    return atomicAdd(reinterpret_cast<unsigned long long*>(addr), value);
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
    def semaphore_notify(self, tid, *coord):
        # wg is synced
        with Tx.thread():
            if tid % 128 == 0:
                Tx.cuda.func_call(
                    "atomic_add_system_uint64",
                    self.sem.access_ptr("rw", offset=self.sem.elem_offset_of(coord)),
                    Tx.uint64(1),
                    source_code=atomic_add_system_uint64,
                )
            Tx.cuda.thread_fence()


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
            (pipeline_depth, pipeline_num),
            "uint64",
            shared_buf,
            elem_offset=base_offset,
            name="mbar_p2c",
        )
        self.mbar_c2p = Tx.decl_buffer(
            (pipeline_depth, pipeline_num),
            "uint64",
            shared_buf,
            elem_offset=base_offset + pipeline_depth * pipeline_num,
            name="mbar_c2p",
        )
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
        Tx.ptx.fence.proxy_async("shared::cta")

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


class ReducePipe(Pipeline):

    @Tx.inline
    def consumer_release(self, pipeline_idx: int):
        Tx.ptx.mbarrier.arrive(self.mbar_c2p.ptr_to([self.idx, pipeline_idx]))


half8tofloat8 = """
__forceinline__ __device__ void half8tofloat8(void* src_addr, void* dst_addr) {
    half2* source = (half2*) src_addr;
    float2* dest = (float2*) dst_addr;
    for (int i = 0; i < 4; i++) {
        dest[i] = __half22float2(source[i]);
    }
}
"""
float8tohalf8 = """
__forceinline__ __device__ void float8tohalf8(void* src_addr, void* dst_addr) {
    float2* source = (float2*) src_addr;
    half2* dest = (half2*) dst_addr;
    for (int i = 0; i < 4; i++) {
        dest[i] = __float22half2_rn(source[i]);
    }
}
"""


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
@I.ir_module(tirx=True)
class ReduceScatter:
    @Tx.prim_func(tirx=True)
    def test_mma_ss_tma_2sm_persistent(A: Tx.Buffer((M, K), a_type), B: Tx.Buffer((N, K), b_type), gemm_out: Tx.Buffer((M, N), d_type),
                                    semaphore: Tx.Buffer((WORLD_SIZE, ), "uint64"),
                                    out: Tx.Buffer((LOCAL_M, N), d_type), profiler_buffer: Tx.Buffer((PROFILER_BUFFER_SIZE,), "uint64")):
        A_tensor_map: Tx.let[Tx.handle("tensormap")] = Tx.tvm_stack_alloca("tensormap", 1)
        B_tensor_map: Tx.let[Tx.handle("tensormap")] = Tx.tvm_stack_alloca("tensormap", 1)
        C_tensor_map: Tx.let[Tx.handle("tensormap")] = Tx.tvm_stack_alloca("tensormap", 1)
        Tx.call_packed("runtime.cuTensorMapEncodeTiled", A_tensor_map, "float16", 2, A.data, K, M, K * 2, BLK_K, BLK_M, 1, 1, 0, 3, 0, 0)
        Tx.call_packed("runtime.cuTensorMapEncodeTiled", B_tensor_map, "float16", 2, B.data, K, N, K * 2, BLK_K, BLK_N // 2, 1, 1, 0, 3, 0, 0)
        Tx.call_packed("runtime.cuTensorMapEncodeTiled", C_tensor_map, "float16", 2, gemm_out.data, N, M, N * 2, EPI_TILE, BLK_M, 1, 1, 0, 3, 0, 0)
        with Tx.kernel():
            cbx, cby = Tx.cta_id([CLUSTER_M, CLUSTER_N], parent="cluster")
            bx = Tx.cta_id([SM_COUNT], parent="kernel")
            wg_id = Tx.warpgroup_id([NUM_CONSUMER+1], parent="cta")
            warp_id = Tx.warp_id([4], parent="warpgroup")
            lane_id = Tx.thread_id([32], parent="warp")
            tid = Tx.thread_id([NUM_THREADS], parent="cta")
            rank = Tx.nvshmem.my_pe()
            sem = Semaphore(cnt=WORLD_SIZE, buffer=semaphore)
            with Tx.cta():
                buf = Tx.alloc_buffer([SMEM_SIZE], "uint8", scope="shared.dyn")
                profiler = CudaProfiler(
                    profiler_buffer,
                    write_stride=PROFILER_WRITE_STRIDE,
                    num_groups=NUM_GROUPS,
                )
                profiler.init(0)
                with Tx.cta()[0:GEMM_SMS]:
                    profiler.start(ProfileEventType.GEMM, lane_id == 0)
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
                    m_clusters = Tx.meta_var((M + BLK_M - 1) // BLK_M // CLUSTER_M // NUM_CONSUMER)
                    n_clusters = Tx.meta_var((N + BLK_N - 1) // BLK_N // CLUSTER_N)
                    gemm_tile_scheduler = RankAwareGroupMajorTileScheduler("gemm_tile_scheduler", m_clusters, n_clusters, GROUP_SIZE, WORLD_SIZE)
                    gemm_tile_scheduler.init(bx//2)
                    # alloc TMEM
                    with Tx.warp()[0:1]:
                        Tx.ptx.tcgen05.alloc(Tx.address_of(tmem_addr), n_cols=N_COLS, cta_group=cta_group)
                    Tx.ptx.tcgen05.encode_instr_descriptor(Tx.address_of(descI), "float32", a_type, b_type, MMA_M, MMA_N, MMA_K, trans_a=False, trans_b=False, n_cta_groups=cta_group)
                    Tx.cuda.cta_sync()
                    Tx.cuda.trap_when_assert_failed(tmem_addr == 0)
                    tmem = Tx.decl_buffer((128, N_COLS), "float32", scope="tmem", allocated_addr=0,
                                         layout=TileLayout(S[(128, N_COLS) : (1@TCol, 1@TLane)]))
                    # reset RF
                    with Tx.cta():
                        Tx.attr({"tirx.scope_partition": True})
                        with Tx.warpgroup()[NUM_CONSUMER:NUM_CONSUMER + 1]:
                            Tx.ptx.setmaxnreg(False, 56)
                            if warp_id == 3:
                                while gemm_tile_scheduler.valid():
                                    m_idx = Tx.meta_var(gemm_tile_scheduler.m_idx) # represent cluster task id
                                    n_idx = Tx.meta_var(gemm_tile_scheduler.n_idx)
                                    for ko in range(K // BLK_K):
                                        # GMEM -> SMEM
                                        if lane_id == 0:
                                            tma2mma_pipe.producer_wait(0)
                                            Tx.ptx.cp_async.bulk.tensor.g2c(2, A_smem.ptr_to([tma2mma_pipe.idx, 0, 0, 0]), tma_finished.ptr_to([tma2mma_pipe.idx]), A_tensor_map, ko * BLK_K, (m_idx * 4 + cbx) * BLK_M, cta_group=2)
                                            Tx.ptx.cp_async.bulk.tensor.g2c(2, A_smem.ptr_to([tma2mma_pipe.idx, 1, 0, 0]), tma_finished.ptr_to([tma2mma_pipe.idx]), A_tensor_map, ko * BLK_K, (m_idx * 4 + 2 + cbx) * BLK_M, cta_group=2)
                                            Tx.ptx.cp_async.bulk.tensor.g2c(2, B_smem.ptr_to([tma2mma_pipe.idx, 0, 0]), tma_finished.ptr_to([tma2mma_pipe.idx]), B_tensor_map, ko * BLK_K, n_idx * BLK_N + cbx * BLK_N // 2, cta_group=2)
                                            if cbx == 0:
                                                # notify the mma stage that tma load is finished
                                                Tx.ptx.mbarrier.arrive.expect_tx(tma_finished.ptr_to([tma2mma_pipe.idx]), (BLK_K * BLK_M * 2 * 2 + BLK_K * BLK_N) * 2)
                                            tma2mma_pipe.advance()
                                    gemm_tile_scheduler.next_tile(stride=GEMM_SMS // 2)
                                for i in range(PIPE_DEPTH):
                                    # wait for the completion of all the mma of the last tile
                                    if lane_id == 0:
                                        tma2mma_pipe.producer_wait(0)
                                        tma2mma_pipe.advance()
                            elif warp_id < NUM_CONSUMER:
                                while gemm_tile_scheduler.valid():
                                    m_idx = Tx.meta_var(gemm_tile_scheduler.m_idx) # represent cluster task id
                                    n_idx = Tx.meta_var(gemm_tile_scheduler.n_idx)
                                    with Tx.thread():
                                        # MMA
                                        if lane_id == 0 and cbx == 0:
                                            # wait for the last tmem result to be consumed
                                            mma2ld_pipe.producer_wait(warp_id)
                                            for ko in Tx.serial(0, K // BLK_K):
                                                # wait for tma load to finish
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
                                    gemm_tile_scheduler.next_tile(stride=GEMM_SMS // 2)
                        with Tx.warpgroup()[0:NUM_CONSUMER]:
                            Tx.ptx.setmaxnreg(True, 224)
                            while gemm_tile_scheduler.valid():
                                m_idx = Tx.meta_var(gemm_tile_scheduler.m_idx) # represent cluster task id
                                n_idx = Tx.meta_var(gemm_tile_scheduler.n_idx)
                                # wait for the completion of all the mma of the same tile
                                mma2ld_pipe.consumer_wait(0)
                                # TMEM -> RF
                                for i in range(BLK_N // TMEM_LD_SIZE):
                                    col_st = Tx.meta_var(wg_id * MMA_N + i * TMEM_LD_SIZE)
                                    Tx.copy(reg_wg[:, :], tmem[:, col_st : col_st + TMEM_LD_SIZE])
                                    with Tx.thread():
                                        Tx.cast(reg_fp16[i * TMEM_LD_SIZE : (i + 1) * TMEM_LD_SIZE], reg[:])
                                # the tmem can be overwritten by the next tile
                                mma2ld_pipe.consumer_release(wg_id)
                                # RF -> GMEM
                                for i in range(BLK_N // EPI_TILE):
                                    with Tx.thread():
                                        Tx.copy(C_smem[wg_id, warp_id * 32 + lane_id, :], reg_fp16[i * EPI_TILE : (i + 1) * EPI_TILE])
                                    Tx.cuda.warpgroup_sync(wg_id+1)
                                    Tx.ptx.fence.proxy_async("shared::cta")
                                    if lane_id == 0 and warp_id == 0:
                                        Tx.ptx.cp_async.bulk.tensor.s2g(2, C_smem.ptr_to([wg_id, 0, 0]), C_tensor_map, n_idx * BLK_N + i * EPI_TILE, (m_idx * 4 + wg_id * 2 + cbx) * BLK_M)
                                        Tx.ptx.cp_async.bulk.commit_group()
                                        Tx.ptx.cp_async.bulk.wait_group(0)
                                    Tx.cuda.warpgroup_sync(wg_id+1)
                                # notify RS ready
                                comm_m_idx = Tx.meta_var(m_idx * 4 + wg_id * 2 + cbx)
                                signal_rank = Tx.meta_var(comm_m_idx // (LOCAL_M // BLK_M))
                                sem.semaphore_notify(tid, signal_rank)
                                mma2ld_pipe.advance()
                                gemm_tile_scheduler.next_tile(stride=GEMM_SMS // 2)
                    # dealloc TMEM
                    with Tx.warp()[0:1]:
                        Tx.ptx.tcgen05.relinquish_alloc_permit(cta_group=cta_group)
                        Tx.ptx.tcgen05.dealloc(tmem_addr, n_cols=N_COLS, cta_group=cta_group)
                    profiler.end(ProfileEventType.GEMM, lane_id == 0)


    @Tx.prim_func(tirx=True)
    def reduce_sum(
        staging_buffer: Tx.Buffer((WORLD_SIZE, LOCAL_M, N), "float16"),
        out: Tx.Buffer((LOCAL_M, N), d_type),
    ):
        src_tensor_map: Tx.let[Tx.handle("tensormap")] = Tx.tvm_stack_alloca("tensormap", 1)
        dst_tensor_map: Tx.let[Tx.handle("tensormap")] = Tx.tvm_stack_alloca("tensormap", 1)
        Tx.call_packed(
            "runtime.cuTensorMapEncodeTiled",
            src_tensor_map,
            "float16",
            3,
            staging_buffer.data,
            N,
            LOCAL_M,
            WORLD_SIZE,
            N * 2,
            LOCAL_M * N * 2,
            BLK_N_RS,
            BLK_M_RS,
            1,
            1,
            1,
            1,
            0,
            0,
            0,
            0,
        )
        Tx.call_packed(
            "runtime.cuTensorMapEncodeTiled",
            dst_tensor_map,
            "float16",
            2,
            out.data,
            N,
            LOCAL_M,
            N * 2,
            BLK_N_RS,
            BLK_M_RS,
            1,
            1,
            0,
            0,
            0,
            0,
        )
        with Tx.kernel():
            bx = Tx.cta_id([SM_COUNT], parent="kernel")
            wg_id = Tx.warpgroup_id([2], parent="cta")
            warp_id = Tx.warp_id([4], parent="warpgroup")
            lane_id = Tx.thread_id([32], parent="warp")
            tid_in_wg = Tx.thread_id([128], parent="warpgroup")
            with Tx.cta():
                buf = Tx.alloc_buffer([SMEM_SIZE], "uint8", scope="shared.dyn")
                load_pipe = ReducePipe(
                    buf.data, 0, RS_LOAD_PIPE_DEPTH, 1, p_single_cta=False, c_single_cta=False
                )
                input_smem = Tx.decl_buffer(
                    (RS_LOAD_PIPE_DEPTH, BLK_M_RS, BLK_N_RS),
                    d_type,
                    buf.data,
                    elem_offset=512,
                )
                output_smem = Tx.decl_buffer(
                    (BLK_M_RS, BLK_N_RS),
                    d_type,
                    buf.data,
                    elem_offset=512 + RS_LOAD_PIPE_DEPTH * BLK_M_RS * BLK_N_RS,
                )
                reg_fp16 = Tx.alloc_buffer((8, ), "float16", scope="local")
                reg_fp32_tmp = Tx.alloc_buffer((8), "float32", scope="local")
                reg_fp32 = Tx.alloc_buffer((BLK_M_RS * BLK_N_RS // 8 // 128, 8), "float32", scope="local")
                iter: Tx.int32
                iter = 0
                load_pipe.init(c2p_thread_count=128)
                tile_id = Tx.meta_var(iter * SM_COUNT + bx)
                Tx.tvm_storage_sync("shared")
                if warp_id == 0 and wg_id == 0:
                    while tile_id < LOCAL_M // BLK_M_RS * N // BLK_N_RS:
                        m_idx = Tx.meta_var(tile_id // (N // BLK_N_RS))
                        n_idx = Tx.meta_var(tile_id % (N // BLK_N_RS))
                        if lane_id == 0:
                            for i in range(WORLD_SIZE):
                                load_pipe.producer_wait(0)
                                Tx.ptx.cp_async.bulk.tensor.g2c(
                                    3,
                                    input_smem.ptr_to([load_pipe.idx, 0, 0]),
                                    load_pipe.mbar_p2c.ptr_to([load_pipe.idx, 0]),
                                    src_tensor_map,
                                    n_idx * BLK_N_RS,
                                    m_idx * BLK_M_RS,
                                    i,
                                )
                                Tx.ptx.mbarrier.arrive.expect_tx(
                                    load_pipe.mbar_p2c.ptr_to([load_pipe.idx, 0]),
                                    BLK_M_RS * BLK_N_RS * 2,
                                )
                                load_pipe.advance()
                        iter += 1
                elif wg_id == 1:
                    while tile_id < LOCAL_M // BLK_M_RS * N // BLK_N_RS:
                        m_idx = Tx.meta_var(tile_id // (N // BLK_N_RS))
                        n_idx = Tx.meta_var(tile_id % (N // BLK_N_RS))
                        for rank in range(WORLD_SIZE):
                            load_pipe.consumer_wait(0)
                            for i in range(BLK_M_RS * BLK_N_RS // 8 // 128):
                                m_in_smem = Tx.meta_var((i * 128 + tid_in_wg) // (BLK_N_RS // 8))
                                n_in_smem = Tx.meta_var((i * 128 + tid_in_wg) % (BLK_N_RS // 8))
                                for j in Tx.vectorized(8):
                                    reg_fp16[j] = input_smem[
                                        load_pipe.idx, m_in_smem, n_in_smem * 8 + j
                                    ]
                                if rank > 0:
                                    Tx.cuda.half8tofloat8(reg_fp16.data, reg_fp32_tmp.data)
                                    for j in Tx.vectorized(8):
                                        reg_fp32[i, j] += reg_fp32_tmp[j]
                                else:
                                    Tx.cuda.half8tofloat8(reg_fp16.data, reg_fp32.ptr_to([i, 0]))
                            load_pipe.consumer_release(0)
                            load_pipe.advance()
                        for i in range(BLK_M_RS * BLK_N_RS // 8 // 128):
                            m_in_smem = Tx.meta_var((i * 128 + tid_in_wg) // (BLK_N_RS // 8))
                            n_in_smem = Tx.meta_var((i * 128 + tid_in_wg) % (BLK_N_RS // 8))
                            Tx.cuda.float8tohalf8(reg_fp32.ptr_to([i, 0]), reg_fp16.data)
                            for j in Tx.vectorized(8):
                                output_smem[m_in_smem, n_in_smem * 8 + j] = reg_fp16[j]
                        Tx.cuda.warpgroup_sync(1)
                        Tx.ptx.fence.proxy_async("shared::cta")
                        if tid_in_wg == 0:
                            Tx.ptx.cp_async.bulk.tensor.s2g(
                                2,
                                output_smem.ptr_to([0, 0]),
                                dst_tensor_map,
                                n_idx * BLK_N_RS,
                                m_idx * BLK_M_RS,
                            )
                            Tx.ptx.cp_async.bulk.commit_group()
                            Tx.ptx.cp_async.bulk.wait_group(0)
                        Tx.cuda.warpgroup_sync(1)
                        iter += 1
# fmt: on


@pytest.mark.skip()
@tvm.testing.requires_cuda_compute_version(10, exact=True)
def test_hgemm_rs():
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
    # A_np = np.random.rand(WORLD_SIZE, M, K).astype(a_type)
    # B_np = np.random.rand(WORLD_SIZE, N, K).astype(b_type)
    A_np = np.random.uniform(-1, 1, (WORLD_SIZE, M, K)).astype(a_type)
    B_np = np.random.uniform(-1, 1, (WORLD_SIZE, N, K)).astype(b_type)
    A_tvm = tvm.runtime.tensor(A_np, device=DEV)
    B_tvm = tvm.runtime.tensor(B_np, device=DEV)
    semaphore_np = np.zeros((WORLD_SIZE,), dtype="uint64")
    profiler_buffer_np = np.zeros((PROFILER_BUFFER_SIZE,), dtype="uint64")

    A_array_all = sess.empty((WORLD_SIZE, M, K), a_type, worker0_only=True)
    B_array_all = sess.empty((WORLD_SIZE, N, K), b_type, worker0_only=True)
    sess.copy_to_worker_0(A_tvm, A_array_all)
    sess.copy_to_worker_0(B_tvm, B_array_all)

    transfer_to_peers_dfunc = sess.get_global_func("runtime.disco.transfer_to_peers_reduce_scatter")
    stream_create_dfunc = sess.get_global_func("runtime.disco.stream_create")
    d_stream = stream_create_dfunc()
    stream_sync_dfunc = sess.get_global_func("runtime.disco.stream_sync")
    cur_stream = sess.get_global_func("runtime.get_cuda_stream")()
    nvshmem_malloc_hook = sess.get_global_func("runtime.disco.nvshmem.empty")
    args_dict = {
        "A_array": sess.empty((M, K), a_type),
        "B_array": sess.empty((N, K), b_type),
        "gemm_out_array": nvshmem_malloc_hook(ShapeTuple((M, N)), d_type, None),
        "semaphore_array": nvshmem_malloc_hook(ShapeTuple((WORLD_SIZE,)), "uint64", None),
        "staging_buffer_array": nvshmem_malloc_hook(
            ShapeTuple((WORLD_SIZE, LOCAL_M, N)), d_type, None
        ),
        "out_array": sess.empty((LOCAL_M, N), d_type),
        "profiler_buffer_array": sess.empty((PROFILER_BUFFER_SIZE,), "uint64"),
    }

    res_dict = {
        "gemm_out_res": sess.empty((WORLD_SIZE, M, N), d_type, worker0_only=True),
        "buffer_res": sess.empty(
            (WORLD_SIZE, M // BLK_M, N // BLK_N, BLK_M, BLK_N), d_type, worker0_only=True
        ),
        "out_res": sess.empty((WORLD_SIZE, LOCAL_M, N), d_type, worker0_only=True),
        "profiler_buffer_res": sess.empty(
            (WORLD_SIZE, PROFILER_BUFFER_SIZE), "uint64", worker0_only=True
        ),
        "staging_buffer_res": sess.empty(
            (WORLD_SIZE, WORLD_SIZE, LOCAL_M, N), d_type, worker0_only=True
        ),
        "gemm_out_host": tvm.runtime.empty((WORLD_SIZE, M, N), d_type, device=DEV),
        "buffer_host": tvm.runtime.empty(
            (WORLD_SIZE, M // BLK_M, N // BLK_N, BLK_M, BLK_N), d_type, device=DEV
        ),
        "out_host": tvm.runtime.empty((WORLD_SIZE, LOCAL_M, N), d_type, device=DEV),
        "profiler_buffer_host": tvm.runtime.empty(
            (WORLD_SIZE, PROFILER_BUFFER_SIZE), "uint64", device=DEV
        ),
        "staging_buffer_host": tvm.runtime.empty(
            (WORLD_SIZE, WORLD_SIZE, LOCAL_M, N), d_type, device=DEV
        ),
    }

    sess.scatter_from_worker0(A_array_all, args_dict["A_array"])
    sess.scatter_from_worker0(B_array_all, args_dict["B_array"])
    sess.sync_worker_0()
    print(f"Data prepared successfully")

    with tempfile.TemporaryDirectory() as tmpdir:
        target = tvm.target.Target("cuda")
        path = tmpdir + "/test.so"
        mod = tvm.compile(ReduceScatter, target=target, tir_pipeline="tirx")
        print(mod.mod.imports[0].inspect_source())
        mod.export_library(path)

        rt_mod = sess.load_vm_module(path)
        barrier_dfunc = sess.get_global_func("runtime.disco.nvshmem.barrier_all_on_stream")
        print("Begin kernel execution...")
        sess._sync_all()

        for itr in range(N_REPEAT):
            barrier_dfunc(cur_stream)
            stream_sync_dfunc(cur_stream, d_stream)
            rt_mod["test_mma_ss_tma_2sm_persistent"](
                args_dict["A_array"],
                args_dict["B_array"],
                args_dict["gemm_out_array"],
                args_dict["semaphore_array"],
                args_dict["out_array"],
                args_dict["profiler_buffer_array"],
            )
            transfer_to_peers_dfunc(
                args_dict["semaphore_array"],
                args_dict["gemm_out_array"],
                args_dict["staging_buffer_array"],
                d_stream,
                M,
                N,
                BLK_M,
                BLK_N,
                WORLD_SIZE,
            )
            barrier_dfunc(d_stream)
            stream_sync_dfunc(d_stream, cur_stream)
            rt_mod["reduce_sum"](
                args_dict["staging_buffer_array"],
                args_dict["out_array"],
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
        sess.gather_to_worker0(args_dict["staging_buffer_array"], res_dict["staging_buffer_res"])
        sess.copy_from_worker_0(res_dict["staging_buffer_host"], res_dict["staging_buffer_res"])

        print(args_dict["semaphore_array"].debug_get_from_remote(0))

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
        # gemm_out_res = res_dict["gemm_out_host"].numpy()[i]
        # np.testing.assert_allclose(gemm_out_res, gemm_out_torch[i].cpu().numpy(), atol=1e-3, rtol=1e-3)

    # staging_buffer_torch = gemm_out_torch.reshape(WORLD_SIZE, WORLD_SIZE, LOCAL_M, N).transpose(0, 1)
    # np.testing.assert_allclose(staging_buffer_torch.cpu().numpy(), res_dict["staging_buffer_host"].numpy(), atol=1e-3, rtol=1e-3)

    gemm_out_torch_sum = torch.sum(gemm_out_torch, dim=0)
    out_res = res_dict["out_host"].numpy().reshape(-1, N)
    out_std = gemm_out_torch_sum.cpu().numpy()

    np.testing.assert_allclose(out_res, gemm_out_torch_sum.cpu().numpy(), atol=6e-2, rtol=6e-2)

    print("Results all correct.")

    # # profiler results
    # for rank in range(WORLD_SIZE):
    #     if rank == 7:
    #         export_to_perfetto_trace(
    #             res_dict["profiler_buffer_host"].numpy()[rank],
    #             f"hgemm-RS-rank{rank}.perfetto-trace",
    #             event_type_names,
    #         )


@pytest.mark.skip()
def test_reduce():
    import torch

    torch.manual_seed(42)
    DEV = tvm.cuda(0)
    target = tvm.target.Target("cuda")
    input_torch = torch.randn((WORLD_SIZE, LOCAL_M, N), dtype=torch.float16, device="cuda")
    out_torch = torch.zeros((LOCAL_M, N), dtype=torch.float16, device="cuda")
    input_tvm = tvm.runtime.tensor(input_torch.cpu(), device=DEV)
    out_tvm = tvm.runtime.tensor(out_torch.cpu(), device=DEV)
    with target:
        mod = ReduceScatter
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
        print(mod.mod.imports[0].inspect_source())
        mod["reduce_sum"](input_tvm, out_tvm)
        out_std = torch.sum(input_torch, dim=0)
        np.testing.assert_allclose(out_tvm.numpy(), out_std.cpu().numpy(), atol=6e-2, rtol=6e-2)


if __name__ == "__main__":
    test_hgemm_rs()
    # test_reduce()
