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

# Mostly the same schedule as Thunderkitten's kernel
# 6% slower than that
# maybe due to more register spills
import numpy as np
import pytest

import tvm
from tvm.ir.type import PointerType, PrimType
from tvm.script import tir as T
from tvm.script import tirp as Tp
import tvm.testing
from tvm.script.ir_builder import IRBuilder
from tvm.tir.async_structs import CopyPipeline


def _get_source(func: tvm.tir.PrimFunc) -> str:
    target = tvm.target.Target("cuda")
    mod = tvm.IRModule({"main": func})
    mod = tvm.compile(mod, target=target, tir_pipeline="tirp")
    src = mod.mod.imported_modules[0].get_source()
    return src, mod


def ceildiv(a, b):
    return (a + b - 1) // b


d_type, a_type, b_type = "float16", "float16", "float16"
M, N, K = 8192, 8192, 8192
BLK_M, BLK_N, BLK_K = 128, 256, 64
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


class TileScheduler:
    m_clusters = (M + BLK_M - 1) // BLK_M // CLUSTER_M // NUM_CONSUMER
    n_clusters = (N + BLK_N - 1) // BLK_N // CLUSTER_N

    def __init__(self, prefix: str):
        self.m_idx = T.local_cell("int32", name=prefix + "_m_idx")
        self.n_idx = T.local_cell("int32", name=prefix + "_n_idx")
        self.linear_idx = T.local_cell("int32", name=prefix + "_linear_idx")

    @T.macro
    def update_current_m_n_idx(self, linear_idx):
        group_rows = (self.m_clusters // GROUP_SIZE) * GROUP_SIZE
        final_rows = self.m_clusters - group_rows
        group_repeat = GROUP_SIZE * self.n_clusters
        # FIXME: use group_rows > 0 to avoid constant folding bug
        if linear_idx < group_rows * self.n_clusters and group_rows > 0:
            self.m_idx = linear_idx // group_repeat * GROUP_SIZE + (linear_idx % GROUP_SIZE)
            self.n_idx = linear_idx % group_repeat // GROUP_SIZE
        elif final_rows > 0:
            remainder_idx = linear_idx - group_rows * self.n_clusters
            self.m_idx = group_rows + remainder_idx % final_rows
            self.n_idx = remainder_idx // final_rows

    @T.macro
    def init(self, linear_init):
        self.linear_idx = linear_init
        self.update_current_m_n_idx(linear_init)

    @T.macro
    def next_tile(self):
        self.linear_idx = self.linear_idx + SM_COUNT // 2
        self.update_current_m_n_idx(self.linear_idx)

    def valid(self):
        return self.linear_idx < self.m_clusters * self.n_clusters


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
                            T.ptx.mbarrier.init(
                                self.mbar_p2c.access_ptr(
                                    "rw", offset=self.mbar_p2c.offset_of_p([i, j])
                                ),
                                p2c_thread_count,
                            )
                        if not self.p_single_cta or cbx == 0:
                            T.ptx.mbarrier.init(
                                self.mbar_c2p.access_ptr(
                                    "rw", offset=self.mbar_c2p.offset_of_p([i, j])
                                ),
                                c2p_thread_count,
                            )
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
                    self.mbar_c2p.access_ptr(
                        "rw", offset=self.mbar_c2p.offset_of_p([self.idx, pipeline_idx])
                    ),
                    self.c2p_phase,
                )

    @T.macro
    def consumer_wait(self, pipeline_idx):
        for cbx in T.thread_binding(CLUSTER_M, "clusterCtaIdx.x"):
            if not self.c_single_cta or cbx == 0:
                T.ptx.mbarrier.try_wait(
                    self.mbar_p2c.access_ptr(
                        "rw", offset=self.mbar_p2c.offset_of_p([self.idx, pipeline_idx])
                    ),
                    self.p2c_phase,
                )


class TMA2MMAPipeline(Pipeline):

    @T.macro
    def consumer_release(self, pipeline_idx):
        for cbx in T.thread_binding(CLUSTER_M, "clusterCtaIdx.x"):
            for tx in T.thread_binding(NUM_THREADS, "threadIdx.x"):
                if tx % 32 == 0:
                    if not self.c_single_cta:
                        T.ptx.tcgen05.commit(
                            self.mbar_c2p.access_ptr(
                                "rw", offset=self.mbar_c2p.offset_of_p([self.idx, pipeline_idx])
                            ),
                            1,
                        )
                    elif cbx == 0:
                        T.ptx.tcgen05.commit(
                            self.mbar_c2p.access_ptr(
                                "rw", offset=self.mbar_c2p.offset_of_p([self.idx, pipeline_idx])
                            ),
                            2,
                            cta_mask=3,
                        )


class MMA2LDpipeline(Pipeline):
    @T.macro
    def consumer_release(self, pipeline_idx):
        for cbx in T.thread_binding(CLUSTER_M, "clusterCtaIdx.x"):
            if not self.c_single_cta or cbx == 0:
                T.ptx.mbarrier.arrive(
                    self.mbar_c2p.access_ptr(
                        "rw", offset=self.mbar_c2p.offset_of_p([self.idx, pipeline_idx])
                    ),
                    cta_id=0,
                    pred=True,
                )


class NameBarrier:
    CONSUMER_0 = 1
    CONSUMER_1 = 2


@tvm.testing.requires_cuda_compute_version(10, exact=True)
def test_tcgen05_mma_ss_tma():

    SWIZZLE = 3
    cta_group = 2

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
    ldo, sdo = 1, 64

    def get_reg_lists(reg):
        return [reg[j] for j in range(TMEM_LD_SIZE)]

    # fmt: off
    @T.prim_func(tirp=True)
    def test_mma_ss_tma_2sm_persistent(A: T.Buffer((M, K), a_type, layout="default"), B: T.Buffer((N, K), b_type, layout="default"), C: T.Buffer((M, N), d_type, layout="default")):
        A_tensor_map: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
        B_tensor_map: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
        C_tensor_map: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
        T.call_packed("runtime.cuTensorMapEncodeTiled", A_tensor_map, "float16", 2, A.data, K, M, K * 2, BLK_K, BLK_M, 1, 1, 0, 3, 0, 0)
        T.call_packed("runtime.cuTensorMapEncodeTiled", B_tensor_map, "float16", 2, B.data, K, N, K * 2, BLK_K, BLK_N // 2, 1, 1, 0, 3, 0, 0)
        T.call_packed("runtime.cuTensorMapEncodeTiled", C_tensor_map, "float16", 2, C.data, N, M, N * 2, EPI_TILE, BLK_M, 1, 1, 0, 3, 0, 0)
        with T.kernel():
            cbx, cby = T.cta_id([CLUSTER_M, CLUSTER_N], parent="cluster")
            bx = T.cta_id([SM_COUNT], parent="kernel")
            wg_id = T.warpgroup_id([NUM_CONSUMER+1], parent="cta")
            warp_id = T.warp_id([4], parent="warpgroup")
            lane_id = T.thread_id([32], parent="warp")
            with T.cta():
                buf = T.alloc_buffer([SMEM_SIZE], "uint8", scope="shared.dyn")
                # tmem_addr = T.decl_buffer((1,), "uint32", buf.data, elem_offset=0)
                tmem_addr = T.decl_cell("uint32", buf.data, scope="shared.dyn", elem_offset=0)
                A_smem = T.decl_buffer((PIPE_DEPTH, NUM_CONSUMER,BLK_M, BLK_K), a_type, buf.data, elem_offset=512, layout=A_layout)
                B_smem = T.decl_buffer((PIPE_DEPTH, BLK_N // 2, BLK_K), b_type, buf.data, elem_offset=512 + BLK_K * BLK_M * NUM_CONSUMER * PIPE_DEPTH, layout=B_layout)
                C_smem = T.decl_buffer((NUM_CONSUMER, BLK_M, EPI_TILE), d_type, buf.data, elem_offset=512 + BLK_K * BLK_M * NUM_CONSUMER * PIPE_DEPTH + BLK_K * BLK_N // 2 * PIPE_DEPTH, layout=D_layout)
                reg = T.alloc_buffer((TMEM_LD_SIZE,), "float32", scope="local")
                reg_fp16 = T.alloc_buffer((BLK_N,), d_type, scope="local")
                # descA = T.alloc_buffer((1,), "uint64", scope="local")
                # descB = T.alloc_buffer((1,), "uint64", scope="local")
                # descI = T.alloc_buffer((1,), "uint32", scope="local")
                descA = T.local_cell("uint64")
                descB = T.local_cell("uint64")
                descI = T.local_cell("uint32")
                tma2mma_pipe = T.meta_var(TMA2MMAPipeline(buf.data, 1, PIPE_DEPTH, 1, p_single_cta=False, c_single_cta=True))
                mma2ld_pipe = T.meta_var(MMA2LDpipeline(buf.data, 1 + PIPE_DEPTH * 2, 1, NUM_CONSUMER, p_single_cta=True, c_single_cta=False))
                mma2ld_pipe.init(c2p_thread_count=128 * 2)
                tma2mma_pipe.init(c2p_thread_count=NUM_CONSUMER)
                ptr: T.Var(name="ptr", dtype=PointerType(PrimType("uint64"))) = T.reinterpret(
                    "handle",
                    T.ptx.map_shared_rank(
                        tma2mma_pipe.mbar_p2c.access_ptr("rw", offset=tma2mma_pipe.mbar_p2c.offset_of_p([0, 0])),
                        0,
                    ),
                )
                tma_finished = T.decl_buffer([PIPE_DEPTH], "uint64", data=ptr, scope="shared")
                tile_scheduler = T.meta_var(TileScheduler("tile_scheduler"))
                tile_scheduler.init(bx//2)
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
                            # first_iter = T.alloc_buffer((1,), "bool", scope="local")
                            first_iter = T.local_cell("bool")
                            first_iter = True
                            while tile_scheduler.valid():
                                m_idx = T.meta_var(tile_scheduler.m_idx) # represent cluster task id
                                n_idx = T.meta_var(tile_scheduler.n_idx)
                                for ko in range(K // BLK_K):
                                    # GMEM -> SMEM
                                    if lane_id == 0:
                                        tma2mma_pipe.producer_wait(0)
                                        if not first_iter and ko == PIPE_DEPTH - 1:
                                            # wait for the completion of all the mma of the same tile
                                            # and then notify the epilogue stage to load the result
                                            T.ptx.mbarrier.arrive(mma2ld_pipe.mbar_p2c.access_ptr("rw", offset=mma2ld_pipe.mbar_p2c.offset_of_p([0, 0])))
                                        T.ptx.cp_async.bulk.tensor.g2c(2, 
                                                                        A_smem.access_ptr("rw", offset=A_smem.offset_of_p([tma2mma_pipe.idx, 0, 0, 0])),
                                                                        tma_finished.access_ptr("rw", offset=tma_finished.offset_of_p([tma2mma_pipe.idx])),
                                                                        A_tensor_map,
                                                                        ko * BLK_K,
                                                                        (m_idx * 4 + cbx) * BLK_M,
                                                                        cta_group=2)
                                        T.ptx.cp_async.bulk.tensor.g2c(2, 
                                                                        A_smem.access_ptr("rw", offset=A_smem.offset_of_p([tma2mma_pipe.idx, 1, 0, 0])),
                                                                        tma_finished.access_ptr("rw", offset=tma_finished.offset_of_p([tma2mma_pipe.idx])),
                                                                        A_tensor_map,
                                                                        ko * BLK_K, 
                                                                        (m_idx * 4 + 2 + cbx) * BLK_M,
                                                                        cta_group=2)
                                        T.ptx.cp_async.bulk.tensor.g2c(2, 
                                                                        B_smem.access_ptr("rw", offset=B_smem.offset_of_p([tma2mma_pipe.idx, 0, 0])),
                                                                        tma_finished.access_ptr("rw", offset=tma_finished.offset_of_p([tma2mma_pipe.idx])),
                                                                        B_tensor_map, 
                                                                        ko * BLK_K, 
                                                                        n_idx * BLK_N + cbx * BLK_N // 2, 
                                                                        cta_group=2)
                                        if cbx == 0:
                                            # notify the mma stage that tma load is finished
                                            T.ptx.mbarrier.arrive.expect_tx(tma_finished.access_ptr("rw", offset=tma_finished.offset_of_p([tma2mma_pipe.idx])), (BLK_K * BLK_M * 2 * 2 + BLK_K * BLK_N) * 2)
                                        tma2mma_pipe.advance()
                                first_iter = False  
                                tile_scheduler.next_tile()
                            for i in range(PIPE_DEPTH):
                                # wait for the completion of all the mma of the last tile
                                tma2mma_pipe.producer_wait(0)
                                tma2mma_pipe.advance()
                            if lane_id == 0:
                                T.ptx.mbarrier.arrive(mma2ld_pipe.mbar_p2c.access_ptr("rw", offset=mma2ld_pipe.mbar_p2c.offset_of_p([0, 0])))
                        elif warp_id < NUM_CONSUMER:
                            while tile_scheduler.valid():
                                m_idx = T.meta_var(tile_scheduler.m_idx) # represent cluster task id
                                n_idx = T.meta_var(tile_scheduler.n_idx)
                                with T.thread():
                                    # MMA
                                    if lane_id == 0 and cbx == 0:
                                        # wait for the last tmem result to be consumed
                                        mma2ld_pipe.producer_wait(warp_id)
                                        for ko in T.serial(0, K // BLK_K):
                                            # wait for tma load to finish
                                            tma2mma_pipe.consumer_wait(0)
                                            for ki in range(BLK_K // MMA_K):
                                                T.ptx.tcgen05.encode_matrix_descriptor(T.address_of(descA), A_smem.access_ptr("r", offset=A_smem.offset_of_p([tma2mma_pipe.idx, warp_id, 0, ki * MMA_K])), ldo=ldo, sdo=sdo, swizzle=SWIZZLE)
                                                T.ptx.tcgen05.encode_matrix_descriptor(T.address_of(descB), B_smem.access_ptr("r", offset=B_smem.offset_of_p([tma2mma_pipe.idx, 0, ki * MMA_K])), ldo=ldo, sdo=sdo, swizzle=SWIZZLE)
                                                if ki == 0 and ko == 0:
                                                    T.ptx.tcgen05.mma("float32", a_type, b_type, tmem_addr + warp_id * MMA_N, descA, descB, descI, use_a_tmem=False, cta_group=cta_group, enable_input_d=False)
                                                else:
                                                    T.ptx.tcgen05.mma("float32", a_type, b_type, tmem_addr + warp_id * MMA_N, descA, descB, descI, use_a_tmem=False, cta_group=cta_group, enable_input_d=True)
                                            tma2mma_pipe.consumer_release(0)
                                            tma2mma_pipe.advance()
                                        mma2ld_pipe.advance()
                                tile_scheduler.next_tile()

                    with T.warpgroup()[0:NUM_CONSUMER]:
                        T.ptx.setmaxnreg(True, 224)
                        while tile_scheduler.valid():
                            m_idx = T.meta_var(tile_scheduler.m_idx) # represent cluster task id
                            n_idx = T.meta_var(tile_scheduler.n_idx)
                            # wait for the completion of all the mma of the same tile
                            mma2ld_pipe.consumer_wait(0)
                            # TMEM -> RF
                            for i in range(BLK_N // TMEM_LD_SIZE):
                                #FIXME: meta_var does not support list comprehension
                                regs = T.meta_var(get_reg_lists(reg))
                                T.ptx.tcgen05.ld(tmem_addr + wg_id * MMA_N, warp_id * 32, i * TMEM_LD_SIZE, "32x32b", TMEM_LD_SIZE, False, *regs)
                                T.ptx.tcgen05.wait.ld()
                                for j in range(TMEM_LD_SIZE):
                                    reg_fp16[i * TMEM_LD_SIZE + j] = T.cast(reg[j], "float16")
                            # the tmem can be overwritten by the next tile
                            mma2ld_pipe.consumer_release(wg_id)
                            # RF -> GMEM
                            for i in range(BLK_N // EPI_TILE):
                                for iter in range(EPI_TILE // 8):
                                    for vec in T.vectorized(8):
                                        C_smem[wg_id, warp_id * 32 + lane_id, iter * 8 + vec] = reg_fp16[i * EPI_TILE + iter * 8 + vec]
                                T.ptx.bar.sync(wg_id, 128)
                                T.ptx.fence.proxy(scope="shared")
                                if lane_id == 0 and warp_id == 0:
                                    T.ptx.cp_async.bulk.tensor.s2g(2, 
                                                                C_smem.access_ptr("r", offset=C_smem.offset_of_p([wg_id,0, 0])),
                                                                C_tensor_map,
                                                                n_idx * BLK_N + i * EPI_TILE,
                                                                (m_idx * 4 + wg_id * 2 + cbx) * BLK_M,
                                                                )
                                    T.ptx.cp_async.bulk.commit_group()
                                    T.ptx.cp_async.bulk.wait_group(0)
                                T.ptx.bar.sync(wg_id, 128)
                            mma2ld_pipe.advance()
                            tile_scheduler.next_tile()
                # dealloc TMEM
                with T.warp()[0:1]:
                    T.ptx.tcgen05.relinquish_alloc_permit(cta_group=cta_group)
                    T.ptx.tcgen05.dealloc(tmem_addr, n_cols=N_COLS, cta_group=cta_group)
    # fmt: on
    import torch

    torch.manual_seed(42)
    DEV = tvm.cuda(0)
    target = tvm.target.Target("cuda")
    with target:
        src, mod = _get_source(test_mma_ss_tma_2sm_persistent)
        print(src)
        A_torch = torch.randn((M, K), dtype=torch.float16)
        B_torch = torch.randn((N, K), dtype=torch.float16)
        C_torch = torch.zeros((M, N), dtype=torch.float16)
        A = tvm.nd.array(A_torch, device=DEV)
        B = tvm.nd.array(B_torch, device=DEV)
        C = tvm.nd.array(C_torch, device=DEV)
        mod(A, B, C)
        ref = torch.matmul(A_torch, B_torch.T)
        np.testing.assert_allclose(C.numpy(), ref.numpy(), rtol=1e-3, atol=1e-2)


if __name__ == "__main__":
    test_tcgen05_mma_ss_tma()
