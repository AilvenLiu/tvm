import math
from enum import Enum

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.script import tirx as Tx
from tvm.script.ir_builder import IRBuilder
from tvm.tirx.bench.utils import ProtonContext, bench, export_to_perfetto_trace, CudaProfiler
from tvm.tirx.tile_scheduler import GroupMajor3D
from tvm.tir.layout import TileLayout, tid_in_wg, TLane, TCol, S

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


@Tx.inline
def skip():
    pass


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

    atomic_add_system_uint64 = f"""
    __forceinline__ __device__ void atomic_add_system_uint64(uint64_t* addr, uint64_t value) {{
        asm volatile("red.async.release.global.gpu.add.u64 [%0], %1;" ::"l"(addr), "l"(value)
                    : "memory");
    }}
    """

    @Tx.meta_class
    class Semaphore:
        def __init__(self, cnt, buffer):
            self.cnt = cnt
            self.sem = buffer
            self.state = Tx.alloc_buffer([1], "uint64", scope="local", align=4)
            IRBuilder.current().name("semaphore_state", self.state)

        @Tx.inline
        def semaphore_wait(self, *coord):
            with Tx.thread():
                while 1:
                    Tx.ptx.ld_global_acquire(
                        self.state[0],
                        self.sem.access_ptr("r", offset=self.sem.elem_offset_of(coord)),
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
            Tx.ptx.tcgen05.commit(self.mbar.ptr_to([idx]), cta_group=CTA_GROUP)

    class BarMMA2TMA(Barriers):

        @Tx.inline
        def arrive(self, idx):
            Tx.ptx.tcgen05.commit(self.mbar.ptr_to([idx]), cta_group=CTA_GROUP)

    class BarLD2MMA(Barriers):

        @Tx.inline
        def arrive(self, idx):
            Tx.ptx.mbarrier.arrive(self.mbar.ptr_to([idx]), cta_id=0, pred=True)

    A_layout = Tx.ComposeLayout(
        Tx.SwizzleLayout(3, 3, 3, swizzle_inner=True),
        Tx.TileLayout(Tx.S[(SMEM_PIPE_DEPTH, BLK_M, BLK_K) : (BLK_M * BLK_K, BLK_K, 1)]),
    )
    B_layout = Tx.ComposeLayout(
        Tx.SwizzleLayout(3, 3, 3, swizzle_inner=True),
        Tx.TileLayout(Tx.S[(SMEM_PIPE_DEPTH, BLK_N, BLK_K) : (BLK_N * BLK_K, BLK_K, 1)]),
    )
    D_layout = Tx.TileLayout(Tx.S[(TMEM_PIPE_DEPTH, EPI_TILE, MMA_N) : (EPI_TILE * MMA_N, MMA_N, 1)])

    # fmt: off
    @Tx.prim_func(tirx=True)
    def hgemm(
        A_ptr: Tx.handle,
        B: Tx.Buffer((N, K), b_type),
        partial_sum_ptr: Tx.handle,
        profiler_buffer: Tx.Buffer((PROFILER_BUFFER_SIZE,), "uint64"),
    ):
        M = Tx.int32()
        A = Tx.match_buffer(A_ptr, [M, K], a_type)
        partial_sum = Tx.match_buffer(partial_sum_ptr, [TILE_K_NUM, M, N], "float32")
        with Tx.kernel():
            # cbx, cby = Tx.cta_id([M_CLUSTER, N_CLUSTER], parent="cluster")
            bx = Tx.cta_id([SM_NUMBER], parent="kernel")
            wg_id = Tx.warpgroup_id([WG_NUMBER], parent="cta")
            warp_id = Tx.warp_id([WARP_NUMBER], parent="warpgroup")
            lane_id = Tx.thread_id([32], parent="warp")
            tid = Tx.thread_id([NUM_THREADS], parent="cta")
            with Tx.cta():
                # alloc shared memory
                buf = Tx.alloc_buffer([SMEM_SIZE], "uint8", scope="shared.dyn")
                tmem_addr = Tx.decl_scalar("uint32", buf.data, scope="shared.dyn", elem_offset=0)
                A_smem = Tx.decl_buffer((SMEM_PIPE_DEPTH, BLK_M, BLK_K), a_type, buf.data, layout=A_layout,
                                        elem_offset=1024 // F16_BYTES)
                B_smem = Tx.decl_buffer((SMEM_PIPE_DEPTH, BLK_N, BLK_K), b_type, buf.data, layout=B_layout,
                                        elem_offset=1024 // F16_BYTES + SMEM_PIPE_DEPTH * BLK_M * BLK_K)
                D_smem = Tx.decl_buffer((TMEM_PIPE_DEPTH, EPI_TILE, MMA_N), "float32", buf.data, layout=D_layout,
                                        elem_offset=1024 // F32_BYTES + SMEM_PIPE_DEPTH * (BLK_M + BLK_N) * BLK_K // 2)

                # alloc local memory
                reg = Tx.alloc_buffer((TMEM_LD_SIZE,), "float32", scope="local")
                reg_wg = reg.view(128, TMEM_LD_SIZE, layout=TileLayout(S[(128, TMEM_LD_SIZE) : (1@tid_in_wg, 1)]))
                stage: Tx.int32
                phase = Tx.alloc_buffer((1, ), "int32", scope="local")
                descA: Tx.uint64
                descB: Tx.uint64
                descI: Tx.uint32
                profiler = Tx.meta_var(
                    CudaProfiler(
                        profiler_buffer,
                        write_stride=PROFILER_WRITE_STRIDE,
                        num_groups=NUM_GROUPS,
                        profiler_enabled=PROFILER_ON,
                    )
                )
                # initialize
                tma2mma_bar = BarTMA2MMA(buf.data, 6, SMEM_PIPE_DEPTH, True)
                mma2tma_bar = BarMMA2TMA(buf.data, 6 + 2 * SMEM_PIPE_DEPTH, SMEM_PIPE_DEPTH, False)
                mma2ld_bar = BarMMA2LD(buf.data, 6 + 3 * SMEM_PIPE_DEPTH, TMEM_PIPE_DEPTH, True)
                ld2mma_bar = BarLD2MMA(buf.data, 6 + 3 * SMEM_PIPE_DEPTH + TMEM_PIPE_DEPTH, TMEM_PIPE_DEPTH, False)
                m_tiles_expr: Tx.let = Tx.truncdiv(M + BLK_M * CTA_GROUP - 1, BLK_M * CTA_GROUP)
                tile_scheduler = GroupMajor3D(
                    "tile_scheduler",
                    m_tiles=m_tiles_expr,
                    n_tiles=TILE_N_NUM,
                    k_tiles=TILE_K_NUM,
                    group_rows=TILE_GROUPS_ROW_SIZE,
                    step=SM_NUMBER,
                )
                tma2mma_bar.init(1)
                mma2ld_bar.init(1)
                mma2tma_bar.init(1)
                ld2mma_bar.init(CTA_GROUP * 128)
                tile_scheduler.init(bx)
                # alloc TMEM
                with Tx.warp()[0:1]:
                    Tx.ptx.tcgen05.alloc(Tx.address_of(tmem_addr), n_cols=N_COLS, cta_group=1)
                    Tx.cuda.warp_sync()

                # sync
                Tx.ptx.fence.proxy("shared")
                Tx.ptx.fence.mbarrier_init()
                Tx.cuda.cta_sync()
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
                            profiler.init(0)
                            phase[0] = 0
                            while tile_scheduler.valid():
                                m_idx = Tx.meta_var(tile_scheduler.m_idx)
                                n_idx = Tx.meta_var(tile_scheduler.n_idx)
                                k_idx = Tx.meta_var(tile_scheduler.k_idx)
                                with Tx.thread()[Tx.ptx.elect_sync()]:
                                    # main inner tma loop
                                    k_offset = Tx.meta_var(k_idx * TILE_K)
                                    m_st = Tx.meta_var(m_idx * BLK_M)
                                    n_st = Tx.meta_var(n_idx * BLK_N)
                                    k_start = Tx.meta_var(stage * BLK_K + k_offset)

                                    @Tx.inline
                                    def tma_load(is_remain, ks):
                                        profiler.start(ProfileEventType.WAIT, lane_id == 0)
                                        mma2tma_bar.wait(ks, phase[0])
                                        profiler.end(ProfileEventType.WAIT, lane_id == 0)
                                        profiler.start(ProfileEventType.IssueTMA, lane_id == 0)
                                        tma_copy = Tx.meta_var({"dispatch": "tma", "mbar": tma2mma_bar.mbar.ptr_to([ks]), "cta_group": CTA_GROUP})
                                        Tx.copy_async(A_smem[ks, :, :], A[m_st : m_st + BLK_M, k_start : k_start + BLK_K], **tma_copy)
                                        Tx.copy_async(B_smem[ks, :, :], B[n_st : n_st + BLK_N, k_start : k_start + BLK_K], **tma_copy)
                                        tma2mma_bar.arrive(ks, CTA_GROUP * BLK_K * (BLK_M + BLK_N) * F16_BYTES)
                                        profiler.end(ProfileEventType.IssueTMA, lane_id == 0)

                                    @Tx.inline
                                    def tma_load_epilogue(ks):
                                        profiler.start(ProfileEventType.WAIT, lane_id == 0)
                                        mma2tma_bar.wait(ks, phase[0])
                                        profiler.end(ProfileEventType.WAIT, lane_id == 0)
                                        tma2mma_bar.arrive_only(ks)

                                    paritioned_loop(tma_load, skip, tma_load_epilogue)
                                tile_scheduler.next_tile()

                        elif warp_id == 0:
                            profiler.init(1)
                            tmem_idx = Tx.local_scalar("int32", "tmem_idx")
                            tmem_phase = Tx.local_scalar("int32", "tmem_phase")
                            Tx.ptx.tcgen05.encode_instr_descriptor(Tx.address_of(descI), "float32", a_type, b_type, MMA_N, MMA_M, MMA_K, False, False, CTA_GROUP)
                            phase[0] = 0
                            while tile_scheduler.valid():
                                m_idx = Tx.meta_var(tile_scheduler.m_idx)
                                n_idx = Tx.meta_var(tile_scheduler.n_idx)
                                k_idx = Tx.meta_var(tile_scheduler.k_idx)
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
                                        profiler.start(ProfileEventType.IssueMMA, lane_id == 0)
                                        Tx.ptx.tcgen05.fence.after_thread_sync()
                                        # issue mma
                                        for ki in Tx.unroll(BLK_K // MMA_K):
                                            Tx.ptx.tcgen05.encode_matrix_descriptor(Tx.address_of(descA), A_smem.ptr_to([ks, 0, ki * MMA_K]),
                                                                                ldo=1, sdo=8 * BLK_K * F16_BYTES // F128_BYTES, swizzle=SWIZZLE)
                                            Tx.ptx.tcgen05.encode_matrix_descriptor(Tx.address_of(descB), B_smem.ptr_to([ks, 0, ki * MMA_K]),
                                                                                ldo=1, sdo=8 * BLK_K * F16_BYTES // F128_BYTES, swizzle=SWIZZLE)
                                            if (stage == 0 and ks == 0 and ki == 0) and ((not is_remain) or (is_remain and PIPE_CIRCLE_NUM == 0)):
                                                Tx.ptx.tcgen05.mma("float32", a_type, b_type, tmem_idx * MMA_M, descB, descA, descI, False, CTA_GROUP, False)
                                            else:
                                                Tx.ptx.tcgen05.mma("float32", a_type, b_type, tmem_idx * MMA_M, descB, descA, descI, False, CTA_GROUP, True)
                                        mma2tma_bar.arrive(ks)
                                        profiler.end(ProfileEventType.IssueMMA, lane_id == 0)

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
                        profiler.init(2)
                        Tx.cuda.trap_when_assert_failed(tmem_addr == 0)
                        tmem_idx = Tx.local_scalar("int32", "tmem_idx")
                        tmem_phase = Tx.local_scalar("int32", "tmem_phase")
                        phase[0] = 0
                        while tile_scheduler.valid():
                            m_idx = Tx.meta_var(tile_scheduler.m_idx)
                            n_idx = Tx.meta_var(tile_scheduler.n_idx)
                            k_idx = Tx.meta_var(tile_scheduler.k_idx)
                            tmem_idx = tile_scheduler.tile_idx % TMEM_PIPE_DEPTH
                            tmem_phase = (tile_scheduler.tile_idx // TMEM_PIPE_DEPTH) & 1

                            # flush previous tma
                            # wait for the completion of all the mma of the same tile
                            mma2ld_bar.wait(tmem_idx, tmem_phase)
                            Tx.ptx.tcgen05.fence.after_thread_sync()

                            for ko in Tx.unroll(MMA_M // EPI_TILE):
                                stage = (tile_scheduler.tile_idx * MMA_M // EPI_TILE + ko) % TMEM_PIPE_DEPTH
                                # wait the smem to be free
                                if ko >= TMEM_PIPE_DEPTH:
                                    if lane_id == 0 and warp_id == 0:
                                        Tx.ptx.cp_async.bulk.wait_group(TMEM_PIPE_DEPTH - 1)
                                    Tx.cuda.warpgroup_sync(10)
                                profiler.start(ProfileEventType.TMEMLD, lane_id == 0 and warp_id == 0)

                                # tmem -> rf (ld) -> smem
                                for ki in Tx.unroll(EPI_TILE // TMEM_LD_SIZE):
                                    col_st = Tx.meta_var(tmem_idx * MMA_M + ko * EPI_TILE + ki * TMEM_LD_SIZE)
                                    Tx.copy(reg_wg[:, :], tmem[:, col_st : col_st + TMEM_LD_SIZE])
                                    with Tx.thread():
                                        Tx.copy(D_smem[stage, ki * TMEM_LD_SIZE : (ki + 1) * TMEM_LD_SIZE, warp_id * 32 + lane_id], reg[:])
                                profiler.end(ProfileEventType.TMEMLD, lane_id == 0 and warp_id == 0)
                                # the tmem can be overwritten
                                if ko == MMA_M // EPI_TILE - 1:
                                    Tx.ptx.tcgen05.fence.before_thread_sync()
                                    ld2mma_bar.arrive(tmem_idx)

                                Tx.ptx.fence.proxy(scope="shared")
                                Tx.cuda.warpgroup_sync(10)
                                profiler.start(ProfileEventType.WRITEBACK, lane_id == 0 and warp_id == 0)
                                # smem -> gmem
                                with Tx.thread()[lane_id == 0 and warp_id == 0]:
                                    m_start = Tx.meta_var(m_idx * BLK_M + ko * EPI_TILE)
                                    n_start = Tx.meta_var(n_idx * BLK_N)
                                    Tx.copy_async(partial_sum[k_idx, m_start : m_start + EPI_TILE, n_start : n_start + BLK_N], D_smem[stage, :, :], dispatch="tma", cache_hint="evict_last")
                                    Tx.ptx.cp_async.bulk.commit_group()
                                profiler.end(ProfileEventType.WRITEBACK, lane_id == 0 and warp_id == 0)
                            with Tx.thread()[lane_id == 0 and warp_id == 0]:
                                Tx.ptx.cp_async.bulk.wait_group(0)
                            Tx.cuda.warpgroup_sync(10)
                            tile_scheduler.next_tile()

                # dealloc TMEM
                with Tx.warp()[0:1]:
                    Tx.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
                    Tx.ptx.tcgen05.dealloc(tmem_addr, n_cols=N_COLS, cta_group=1)

                Tx.cuda.cta_sync()
    # fmt: on

    VEC_SIZE = math.gcd(16 // F32_BYTES, N)
    NUM_THREADS = 512
    BDX = 32
    BDY = NUM_THREADS // BDX

    # fmt: off
    @Tx.prim_func(tirx=True)
    def reduce(partial_sum_ptr: Tx.handle,
               D_ptr: Tx.handle):
        M = Tx.int32()
        partial_sum = Tx.match_buffer(partial_sum_ptr, [TILE_K_NUM, M, N], "float32")
        D = Tx.match_buffer(D_ptr, [M, N], d_type)
        with Tx.kernel():
            tx, ty = Tx.thread_id([BDX, BDY], parent="cta")
            bx = Tx.cta_id([SM_NUMBER], parent="kernel")

            with Tx.thread():
                idx = Tx.alloc_local([1], "int32")
                vec_32 = Tx.alloc_local([VEC_SIZE], "float32")
                tmp = Tx.alloc_local([VEC_SIZE], "float32")
                vec_16 = Tx.alloc_local([VEC_SIZE], "float16")

                idx[0] = bx * NUM_THREADS * VEC_SIZE + (ty * BDX + tx) * VEC_SIZE
                while idx[0] < M * N:
                    m_idx = Tx.meta_var(idx[0] // N)
                    n_idx = Tx.meta_var(idx[0] % N)
                    for kv in Tx.unroll(VEC_SIZE):
                        vec_32[kv] = 0.0
                    for kt in Tx.serial(TILE_K_NUM):
                        for kv in Tx.vectorized(VEC_SIZE):
                            tmp[kv] = partial_sum[kt, m_idx, n_idx + kv]
                        for kv in Tx.unroll(VEC_SIZE):
                            vec_32[kv] += tmp[kv]
                    for kv in Tx.unroll(VEC_SIZE // 2):
                        Tx.cuda.float22half2(Tx.address_of(vec_16[kv * 2]), Tx.address_of(vec_32[kv * 2]))
                        # vec_16[kv] = Tx.cast(vec_32[kv], "float16")
                    for kv in Tx.vectorized(VEC_SIZE):
                        D[m_idx, n_idx + kv] = vec_16[kv]
                    idx[0] += SM_NUMBER * NUM_THREADS * VEC_SIZE
    # fmt: on
    return hgemm, reduce, TILE_K_NUM


@tvm.testing.requires_cuda_compute_version(10, exact=True)
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16, 32, 64, 128])
def test_hgemm_1consumer_1cta_swap_splitk(batch_size):
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
        # Always allocate profiler buffer; it is unused when disabled
        profiler_buffer = np.zeros((PROFILER_BUFFER_SIZE,), dtype=np.uint64)
        profiler_buffer_tvm = tvm.runtime.tensor(profiler_buffer, DEV)
        target = tvm.target.Target("cuda")
        with target:
            mod_hgemm = tvm.ir.IRModule({"main": hgemm})
            mod_reduce = tvm.ir.IRModule({"main": reduce})
            mod_hgemm = tvm.compile(mod_hgemm, target=target, tir_pipeline="tirx")
            mod_reduce = tvm.compile(mod_reduce, target=target, tir_pipeline="tirx")

            def func():
                mod_hgemm(A_tvm, B_tvm, partial_sum_tvm, profiler_buffer_tvm)
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
        test_hgemm_1consumer_1cta_swap_splitk(batch_size)
