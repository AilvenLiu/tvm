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

import numpy as np

import tvm
import tvm.testing
from tvm.ir import PointerType, PrimType
from tvm.script import tirx as Tx
from tvm.tirx.bench.utils import ProtonContext, bench
from tvm.tirx.tile_scheduler import ClusterPersistentScheduler2D
from tvm.tir.layout import TileLayout, tid_in_wg, TLane, TCol, S

# cluster: [2, 1], cta_num = 2
# warpgroup:
#   wg_id = 0: ld & to GMEM
#   wg_id = 1:
#       warp_id = 0: mma
#       warp_id = 3: tma

M_CLUSTER = 2
N_CLUSTER = 1
WG_NUMBER = 2
WARP_NUMBER = 4
NUM_THREADS = (32 * WARP_NUMBER) * WG_NUMBER
SM_NUMBER = 148

SMEM_PIPE_DEPTH = 6
TMEM_PIPE_DEPTH = 2


F16_BYTES = 2
F32_BYTES = 4
F128_BYTES = 16
a_type = tvm.DataType("float16")
b_type = tvm.DataType("float16")
d_type = tvm.DataType("float16")
M, N, K = 1024, 1024, 1024
BLK_M, BLK_N, BLK_K = 128, 128, 64
MMA_M, MMA_N, MMA_K = 256, 256, 16
PIPE_CIRCLE_NUM = (K // BLK_K) // SMEM_PIPE_DEPTH
PIPE_REMAIN_NUM = (K // BLK_K) % SMEM_PIPE_DEPTH
EPI_TILE = 64
TMEM_LD_SIZE = 8
N_COLS = 512
CTA_GROUP = 2
SWIZZLE = 3
SMEM_SIZE = (
    SMEM_PIPE_DEPTH * BLK_M * BLK_K * F16_BYTES
    + SMEM_PIPE_DEPTH * BLK_N * BLK_K * F16_BYTES
    + TMEM_PIPE_DEPTH * BLK_M * EPI_TILE * F16_BYTES
    + 1024
)

assert SMEM_SIZE <= 232448
assert TMEM_PIPE_DEPTH * MMA_N <= 512


def get_source(func: tvm.tir.PrimFunc) -> str:
    target = tvm.target.Target("cuda")
    mod = tvm.IRModule({"main": func})
    mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
    src = mod.mod.imports[0].inspect_source()
    return src, mod


def ceildiv(a, b):
    return (a + b - 1) // b


def flops(ms):
    return M * N * K * 2 / (ms * 1e-3)


@Tx.inline
def skip():
    pass


TILE_GROUPS_ROW_SIZE = 16
assert M % (BLK_M * CTA_GROUP) == 0
assert N % (BLK_N * CTA_GROUP) == 0
TILE_M_NUM = M // (BLK_M * CTA_GROUP)
TILE_N_NUM = N // (BLK_N * CTA_GROUP)


@Tx.meta_class
class Barriers:
    def __init__(self, shared_buffer_base, shared_buffer_offs, pipe_depth, is_p2c):
        self.mbar: tvm.tir.Buffer = Tx.decl_buffer(
            (pipe_depth,), "uint64", shared_buffer_base, elem_offset=shared_buffer_offs
        )
        self.init_phase = 0 if is_p2c else 1
        self.pipe_depth = pipe_depth

    @Tx.inline
    def init(self, threads_num_wait):
        with Tx.thread()[0:1]:
            for i in Tx.serial(self.pipe_depth):
                Tx.ptx.mbarrier.init(self.mbar.ptr_to([i]), threads_num_wait)

    @Tx.inline
    def wait(self, idx, phase):
        Tx.ptx.mbarrier.try_wait(self.mbar.ptr_to([idx]), self.init_phase ^ phase)


class BarTMA2MMA(Barriers):
    @Tx.inline
    def arrive(self, idx, expected_bytes):
        Tx.ptx.mbarrier.arrive.expect_tx(self.mbar.ptr_to([idx]), expected_bytes)

    @Tx.inline
    def arrive_only(self, idx):
        Tx.ptx.mbarrier.arrive(self.mbar.ptr_to([idx]))


class BarMMA2LD(Barriers):
    @Tx.inline
    def arrive(self, idx):
        Tx.ptx.tcgen05.commit(self.mbar.ptr_to([idx]), cta_group=CTA_GROUP, cta_mask=3)


class BarMMA2TMA(Barriers):
    @Tx.inline
    def arrive(self, idx):
        Tx.ptx.tcgen05.commit(self.mbar.ptr_to([idx]), cta_group=CTA_GROUP, cta_mask=3)


class BarLD2MMA(Barriers):
    @Tx.inline
    def arrive(self, idx):
        Tx.ptx.mbarrier.arrive(self.mbar.ptr_to([idx]), cta_id=0, pred=True)


def prepare_data():
    import torch

    A_bf16 = torch.randn((M, K), dtype=torch.float16)
    B_bf16 = torch.randn((N, K), dtype=torch.float16)
    C_empty = torch.zeros((M, N), dtype=torch.float16)

    return A_bf16, B_bf16, C_empty


@tvm.testing.requires_cuda_compute_version(10, exact=True)
def test_hgemm_1consumer():

    A_layout = Tx.ComposeLayout(
        Tx.SwizzleLayout(3, 3, 3, swizzle_inner=True),
        Tx.TileLayout(Tx.S[(SMEM_PIPE_DEPTH, BLK_M, BLK_K) : (BLK_M * BLK_K, BLK_K, 1)]),
    )
    B_layout = Tx.ComposeLayout(
        Tx.SwizzleLayout(3, 3, 3, swizzle_inner=True),
        Tx.TileLayout(Tx.S[(SMEM_PIPE_DEPTH, BLK_N, BLK_K) : (BLK_N * BLK_K, BLK_K, 1)]),
    )
    D_layout = Tx.ComposeLayout(
        Tx.SwizzleLayout(3, 3, 3, swizzle_inner=True),
        Tx.TileLayout(Tx.S[(TMEM_PIPE_DEPTH, BLK_M, EPI_TILE) : (BLK_M * EPI_TILE, EPI_TILE, 1)]),
    )

    # fmt: off
    @Tx.prim_func(tirx=True)
    def hgemm(A: Tx.Buffer((M, K), a_type), B: Tx.Buffer((N, K), b_type),
                D: Tx.Buffer((M, N), d_type)):
        with Tx.kernel():
            cbx, cby = Tx.cta_id([M_CLUSTER, N_CLUSTER], parent="cluster")
            bx = Tx.cta_id([SM_NUMBER], parent="kernel")
            wg_id = Tx.warpgroup_id([WG_NUMBER], parent="cta")
            warp_id = Tx.warp_id([WARP_NUMBER], parent="warpgroup")
            lane_id = Tx.thread_id([32], parent="warp")
            with Tx.cta():
                # alloc shared memory
                buf = Tx.alloc_buffer([SMEM_SIZE], "uint8", scope="shared.dyn")
                tmem_addr = Tx.decl_scalar("uint32", buf.data, scope="shared.dyn", elem_offset=0)
                A_smem = Tx.decl_buffer((SMEM_PIPE_DEPTH, BLK_M, BLK_K), a_type, buf.data, layout=A_layout,
                                        elem_offset=1024 // F16_BYTES)
                B_smem = Tx.decl_buffer((SMEM_PIPE_DEPTH, BLK_N, BLK_K), b_type, buf.data, layout=B_layout,
                                        elem_offset=1024 // F16_BYTES + SMEM_PIPE_DEPTH * BLK_M * BLK_K)
                D_smem = Tx.decl_buffer((TMEM_PIPE_DEPTH, BLK_M, EPI_TILE), d_type, buf.data, layout=D_layout,
                                        elem_offset=1024 // F16_BYTES + SMEM_PIPE_DEPTH * (BLK_M + BLK_N) * BLK_K)

                # alloc local memory
                reg = Tx.alloc_buffer((TMEM_LD_SIZE,), "float32", scope="local")
                reg_wg = reg.view(128, TMEM_LD_SIZE, layout=TileLayout(S[(128, TMEM_LD_SIZE) : (1@tid_in_wg, 1)]))
                reg_fp16 = Tx.alloc_buffer((TMEM_LD_SIZE,), d_type, scope="local")
                stage: Tx.int32
                phase = Tx.alloc_buffer((1, ), "int32", scope="local")
                descA: Tx.uint64
                descB: Tx.uint64
                descI: Tx.uint32

                # initialize
                tma2mma_bar = BarTMA2MMA(buf.data, 6, SMEM_PIPE_DEPTH, True)
                mma2tma_bar = BarMMA2TMA(buf.data, 6 + 2 * SMEM_PIPE_DEPTH, SMEM_PIPE_DEPTH, False)
                mma2ld_bar = BarMMA2LD(buf.data, 6 + 3 * SMEM_PIPE_DEPTH, TMEM_PIPE_DEPTH, True)
                ld2mma_bar = BarLD2MMA(buf.data, 6 + 3 * SMEM_PIPE_DEPTH + TMEM_PIPE_DEPTH, TMEM_PIPE_DEPTH, False)
                tile_scheduler = ClusterPersistentScheduler2D("tile_scheduler", num_m_tiles=TILE_M_NUM, num_n_tiles=TILE_N_NUM, num_clusters=SM_NUMBER // 2, l2_group_size=TILE_GROUPS_ROW_SIZE)
                tma2mma_bar.init(1)
                mma2ld_bar.init(1)
                mma2tma_bar.init(1)
                ld2mma_bar.init(CTA_GROUP * 128)
                tile_scheduler.init(bx // 2)
                ptr: Tx.let[Tx.Var(name="ptr", dtype=PointerType(PrimType("uint64")))] = Tx.reinterpret("handle", Tx.ptx.map_shared_rank(tma2mma_bar.mbar.ptr_to([0]), 0))
                tma_finished = Tx.decl_buffer([SMEM_PIPE_DEPTH], "uint64", data=ptr, scope="shared")

                # alloc TMEM
                with Tx.warp()[0:1]:
                    Tx.ptx.tcgen05.alloc(Tx.address_of(tmem_addr), n_cols=N_COLS, cta_group=2)
                    Tx.cuda.warp_sync()

                # sync
                Tx.ptx.fence.proxy("shared")
                Tx.ptx.fence.mbarrier_init()
                Tx.cuda.cluster_sync()
                Tx.cuda.trap_when_assert_failed(tmem_addr == 0)
                tmem = Tx.decl_buffer((128, N_COLS), "float32", scope="tmem", allocated_addr=0, layout=TileLayout(S[(128, N_COLS) : (1@TLane, 1@TCol)]))

                @Tx.inline
                def paritioned_loop(main_loop, epilogue1, epilogue2):
                    for ko in Tx.serial(PIPE_CIRCLE_NUM):
                        for ks in Tx.unroll(SMEM_PIPE_DEPTH):
                            stage = ko * SMEM_PIPE_DEPTH + ks
                            main_loop(False, ks)
                        phase[0] = phase[0] ^ 1
                    if PIPE_REMAIN_NUM > 0:
                        # last remained loop
                        for ks in Tx.unroll(PIPE_REMAIN_NUM):
                            stage = PIPE_CIRCLE_NUM * SMEM_PIPE_DEPTH + ks
                            main_loop(True, ks)
                        epilogue1()
                        # for unaligned cases
                        for ks in Tx.unroll(PIPE_REMAIN_NUM, SMEM_PIPE_DEPTH):
                            epilogue2(ks)
                        phase[0] = phase[0] ^ 1
                    else:
                        epilogue1()

                with Tx.cta():
                    Tx.attr({"tirx.scope_partition": True})
                    with Tx.warpgroup()[1:2]:
                        if warp_id == 3:
                            phase[0] = 0
                            while tile_scheduler.valid():
                                m_idx = Tx.meta_var(tile_scheduler.m_idx)
                                n_idx = Tx.meta_var(tile_scheduler.n_idx)
                                m_start = Tx.meta_var((m_idx * CTA_GROUP + cbx) * BLK_M)
                                n_start = Tx.meta_var((n_idx * CTA_GROUP + cbx) * BLK_N)
                                with Tx.thread()[Tx.ptx.elect_sync()]:
                                    k_start = Tx.meta_var(stage * BLK_K)

                                    @Tx.inline
                                    def tma_load(is_remain, ks):
                                        # GMEM -> SMEM  (tma)
                                        mma2tma_bar.wait(ks, phase[0])
                                        tma_copy = Tx.meta_var({"dispatch": "tma", "mbar": tma_finished.ptr_to([ks]), "cta_group": CTA_GROUP})
                                        Tx.copy_async(A_smem[ks, :, :], A[m_start : m_start + BLK_M, k_start : k_start + BLK_K], **tma_copy)
                                        Tx.copy_async(B_smem[ks, :, :], B[n_start : n_start + BLK_N, k_start : k_start + BLK_K], **tma_copy)
                                        if cbx == 0:
                                            tma2mma_bar.arrive(ks, CTA_GROUP * BLK_K * (BLK_M + BLK_N) * F16_BYTES)

                                    @Tx.inline
                                    def tma_load_epilogue(ks):
                                        mma2tma_bar.wait(ks, phase[0])
                                        if cbx == 0:
                                            tma2mma_bar.arrive_only(ks)

                                    paritioned_loop(tma_load, skip, tma_load_epilogue)
                                tile_scheduler.next_tile()

                        elif warp_id == 0:
                            if cbx == 0:
                                tmem_idx = Tx.local_scalar("int32", "tmem_idx")
                                tmem_phase = Tx.local_scalar("int32", "tmem_phase")
                                Tx.ptx.tcgen05.encode_instr_descriptor(Tx.address_of(descI), "float32", a_type, b_type, MMA_M, MMA_N, MMA_K, False, False, CTA_GROUP)
                                phase[0] = 0
                                while tile_scheduler.valid():
                                    m_idx = Tx.meta_var(tile_scheduler.m_idx)
                                    n_idx = Tx.meta_var(tile_scheduler.n_idx)
                                    if Tx.ptx.elect_sync():
                                        tmem_idx = tile_scheduler.tile_idx % TMEM_PIPE_DEPTH
                                        tmem_phase = (tile_scheduler.tile_idx // TMEM_PIPE_DEPTH) & 1

                                        # wait for the tmem result to be consumed
                                        ld2mma_bar.wait(tmem_idx, tmem_phase)
                                        Tx.ptx.tcgen05.fence.after_thread_sync()

                                        @Tx.inline
                                        def mma(is_remain, ks):
                                            # wait tma and sf-transpose arrival
                                            tma2mma_bar.wait(ks, phase[0])
                                            Tx.ptx.tcgen05.fence.after_thread_sync()
                                            # issue mma
                                            for ki in Tx.unroll(BLK_K // MMA_K):
                                                Tx.ptx.tcgen05.encode_matrix_descriptor(Tx.address_of(descA), A_smem.ptr_to([ks, 0, ki * MMA_K]),
                                                                                        ldo=1, sdo=8 * BLK_K * F16_BYTES // F128_BYTES, swizzle=SWIZZLE)
                                                Tx.ptx.tcgen05.encode_matrix_descriptor(Tx.address_of(descB), B_smem.ptr_to([ks, 0, ki * MMA_K]),
                                                                                        ldo=1, sdo=8 * BLK_K * F16_BYTES // F128_BYTES, swizzle=SWIZZLE)
                                                if (stage == 0 and ks == 0 and ki == 0) and ((not is_remain) or (is_remain and PIPE_CIRCLE_NUM == 0)):
                                                    Tx.ptx.tcgen05.mma("float32", a_type, b_type, tmem_idx * MMA_N, descA, descB, descI, False, CTA_GROUP, False)
                                                else:
                                                    Tx.ptx.tcgen05.mma("float32", a_type, b_type, tmem_idx * MMA_N, descA, descB, descI, False, CTA_GROUP, True)
                                            mma2tma_bar.arrive(ks)

                                        @Tx.inline
                                        def mma_epilogue1():
                                            # ensure that all mma is issued
                                            mma2ld_bar.arrive(tmem_idx)

                                        @Tx.inline
                                        def mma_epilogue2(ks):
                                            tma2mma_bar.wait(ks, phase[0])
                                            mma2tma_bar.arrive(ks)

                                        paritioned_loop(mma, mma_epilogue1, mma_epilogue2)

                                    tile_scheduler.next_tile()

                    with Tx.warpgroup()[0:1]:
                        Tx.cuda.trap_when_assert_failed(tmem_addr == 0)
                        tmem_idx = Tx.local_scalar("int32", "tmem_idx")
                        tmem_phase = Tx.local_scalar("int32", "tmem_phase")
                        phase[0] = 0
                        while tile_scheduler.valid():
                            m_idx = Tx.meta_var(tile_scheduler.m_idx)
                            n_idx = Tx.meta_var(tile_scheduler.n_idx)
                            tmem_idx = tile_scheduler.tile_idx % TMEM_PIPE_DEPTH
                            tmem_phase = (tile_scheduler.tile_idx // TMEM_PIPE_DEPTH) & 1

                            # flush previous tma
                            if lane_id == 0 and warp_id == 0:
                                Tx.ptx.cp_async.bulk.wait_group(0)
                            Tx.cuda.warpgroup_sync(10)
                            # wait for the completion of all the mma of the same tile
                            mma2ld_bar.wait(tmem_idx, tmem_phase)
                            Tx.ptx.tcgen05.fence.after_thread_sync()

                            for ko in Tx.unroll(MMA_N // EPI_TILE):
                                stage = (tile_scheduler.tile_idx * MMA_N // EPI_TILE + ko) % TMEM_PIPE_DEPTH

                                # wait the smem to be free
                                if ko >= TMEM_PIPE_DEPTH:
                                    if lane_id == 0 and warp_id == 0:
                                        Tx.ptx.cp_async.bulk.wait_group(TMEM_PIPE_DEPTH - 1)
                                    Tx.cuda.warpgroup_sync(10)

                                # tmem -> rf (ld) -> smem
                                for ki in Tx.unroll(EPI_TILE // TMEM_LD_SIZE):
                                    col_st = Tx.meta_var(tmem_idx * MMA_N + ko * EPI_TILE + ki * TMEM_LD_SIZE)
                                    Tx.copy(reg_wg[:, :], tmem[:, col_st : col_st + TMEM_LD_SIZE])
                                    with Tx.thread():
                                        Tx.cast(reg_fp16[:], reg[:])
                                        Tx.copy(D_smem[stage, warp_id * 32 + lane_id, ki * TMEM_LD_SIZE : (ki + 1) * TMEM_LD_SIZE], reg_fp16[:])

                                # the tmem can be overwritten
                                if ko == MMA_N // EPI_TILE - 1:
                                    Tx.ptx.tcgen05.fence.before_thread_sync()
                                    ld2mma_bar.arrive(tmem_idx)

                                Tx.ptx.fence.proxy(scope="shared")
                                Tx.cuda.warpgroup_sync(10)

                                # smem -> gmem
                                with Tx.thread()[lane_id == 0 and warp_id == 0]:
                                    m_start = Tx.meta_var((m_idx * CTA_GROUP + cbx) * BLK_M)
                                    n_start = Tx.meta_var(n_idx * CTA_GROUP * BLK_N + ko * EPI_TILE)
                                    Tx.copy_async(D[m_start : m_start + BLK_M, n_start : n_start + EPI_TILE], D_smem[stage, :, :], dispatch="tma")
                                    Tx.ptx.cp_async.bulk.commit_group()

                            tile_scheduler.next_tile()

                        with Tx.thread()[lane_id == 0 and warp_id == 0]:
                            Tx.ptx.cp_async.bulk.wait_group(0)
                        Tx.cuda.warpgroup_sync(10)

                # dealloc TMEM
                with Tx.warp()[0:1]:
                    Tx.ptx.tcgen05.relinquish_alloc_permit(cta_group=2)
                    Tx.ptx.tcgen05.dealloc(tmem_addr, n_cols=N_COLS, cta_group=2)

                Tx.cuda.cluster_sync()
    # fmt: on

    A_bf16, B_bf16, C_bf16 = prepare_data()

    def tir_gemm(A_bf16, B_bf16, C_bf16):
        DEV = tvm.cuda(0)
        A_tvm = tvm.runtime.tensor(A_bf16, device=DEV)
        B_tvm = tvm.runtime.tensor(B_bf16, device=DEV)
        C_tvm = tvm.runtime.tensor(C_bf16, device=DEV)
        target = tvm.target.Target("cuda")
        with target:
            src, mod = get_source(hgemm)
            print(src)
            func = lambda: mod(A_tvm, B_tvm, C_tvm)
            ms = bench(func, warmup=0, repeat=30, proton_name="tir")
            print(f"TIR flops: {flops(ms) / 1e12} TFLOPS, time: {ms:.3f} ms")

        return C_tvm.numpy()

    def cublas_gemm(A_bf16, B_bf16):
        import torch

        torch_dev = torch.device("cuda")
        A_torch = A_bf16.to(torch_dev)
        B_torch = B_bf16.to(torch_dev)
        func = lambda: torch.matmul(A_torch, B_torch.T)
        ms = bench(func, warmup=0, repeat=30, proton_name="cublas")
        print(f"CUBLAS flops: {flops(ms) / 1e12} TFLOPS, time: {ms:.3f} ms")
        C_torch = func()
        return C_torch.cpu().numpy()

    with ProtonContext("blackwell_gemm"):
        C_tvm = tir_gemm(A_bf16, B_bf16, C_bf16)
        C_cublas = cublas_gemm(A_bf16, B_bf16)

    np.testing.assert_allclose(C_tvm, C_cublas, rtol=1e-3, atol=1e-2)


if __name__ == "__main__":
    test_hgemm_1consumer()
