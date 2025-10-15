import ml_dtypes
import numpy as np

import tvm
import tvm.testing
from tvm.script import tir as T
from tvm.script import tirp as Tp
from tvm.tirp.tile_scheduler import GroupMajor2D
from tvm.tirp.bench.utils import bench
from tvm.tir.layout import TileLayout

# can get 3250 TFLOPs on a single B200 with (m, n, k) = (8192, 8064, 8192), which is aligned to DeepSeek's

# cluster: [2, 1], cta_num = 2
# warpgroup:
#   wg_id = 0: ld & to GMEM
#   wg_id = 1:
#       warp_id = 0: cp & mma
#       warp_id = 2: transpose
#       warp_id = 3: tma

M_CLUSTER = 2
N_CLUSTER = 1
WG_NUMBER = 2
WARP_NUMBER = 4
NUM_THREADS = (32 * WARP_NUMBER) * WG_NUMBER
SM_NUMBER = 148

SMEM_PIPE_DEPTH = 6
TMEM_PIPE_DEPTH = 2


F8_BYTES = 1
F16_BYTES = 2
F32_BYTES = 4
F128_BYTES = 16

a_type = tvm.DataType("float8_e4m3fn")
b_type = tvm.DataType("float8_e4m3fn")
d_type = tvm.DataType("float16")
sfa_type = tvm.DataType("float8_e8m0fnu")
sfb_type = tvm.DataType("float8_e8m0fnu")
M, N, K = 8192, 8064, 8192
BLK_M, BLK_N, BLK_K = 128, 112, 128
MMA_M, MMA_N, MMA_K = 256, 224, 32
BLK_SFA, BLK_SFB = 128, 256
PIPE_CIRCLE_NUM = (K // BLK_K) // SMEM_PIPE_DEPTH
PIPE_REMAIN_NUM = (K // BLK_K) % SMEM_PIPE_DEPTH
EPI_TILE = 32
TMEM_LD_SIZE = 8
N_COLS = 512
SFA_TMEM_START_COL = TMEM_PIPE_DEPTH * MMA_N
SFB_TMEM_START_COL = TMEM_PIPE_DEPTH * MMA_N + TMEM_PIPE_DEPTH * BLK_SFA // 32
CTA_GROUP = 2
QUANT_SIZE = BLK_K
SWIZZLE = 3
SMEM_SIZE = (
    SMEM_PIPE_DEPTH * BLK_M * BLK_K * F8_BYTES
    + SMEM_PIPE_DEPTH * BLK_N * BLK_K * F8_BYTES
    + TMEM_PIPE_DEPTH * BLK_M * EPI_TILE * F16_BYTES
    + SMEM_PIPE_DEPTH * BLK_SFA * F32_BYTES
    + SMEM_PIPE_DEPTH * BLK_SFB * F32_BYTES
    + 1024
)

assert SMEM_SIZE <= 232448
assert TMEM_PIPE_DEPTH * (MMA_N + BLK_SFA // 32 + BLK_SFB // 32) <= 512


def get_source(func: tvm.tir.PrimFunc) -> str:
    target = tvm.target.Target("cuda")
    mod = tvm.IRModule({"main": func})
    mod = tvm.compile(mod, target=target, tir_pipeline="tirp")
    src = mod.mod.imports[0].inspect_source()
    return src, mod


def ceildiv(a, b):
    return (a + b - 1) // b


def flops(ms):
    return M * N * K * 2 / (ms * 1e-3)


TILE_GROUPS_ROW_SIZE = 16
assert M % (BLK_M * CTA_GROUP) == 0
assert N % (BLK_N * CTA_GROUP) == 0
TILE_M_NUM = M // (BLK_M * CTA_GROUP)
TILE_N_NUM = N // (BLK_N * CTA_GROUP)


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


class BarTRANS2MMA(Barriers):

    @T.macro
    def arrive(self, idx):
        T.ptx.mbarrier.arrive(self.mbar.ptr_to([idx]), cta_id=0, pred=True)


class BarMMA2LD(Barriers):

    @T.macro
    def arrive(self, idx):
        T.ptx.tcgen05.commit(self.mbar.ptr_to([idx]), cta_group=CTA_GROUP, cta_mask=3)


class BarMMA2TMA(Barriers):

    @T.macro
    def arrive(self, idx):
        T.ptx.tcgen05.commit(self.mbar.ptr_to([idx]), cta_group=CTA_GROUP, cta_mask=3)


class BarLD2MMA(Barriers):

    @T.macro
    def arrive(self, idx):
        T.ptx.mbarrier.arrive(self.mbar.ptr_to([idx]), cta_id=0, pred=True)


@T.macro
def skip():
    pass


def prepare_data():
    A_origin = np.random.randn(M, K).astype(np.float32)
    B_origin = np.random.randn(N, K).astype(np.float32)
    # A_origin = np.ones((M, K), dtype=np.float32)  # For debugging, we can use ones to simplify the test case
    # B_origin = np.ones((N, K), dtype=np.float32)  # For debugging, we can use ones to simplify the test case

    A_fp8 = np.empty((M, K), dtype=ml_dtypes.float8_e4m3fn)
    B_fp8 = np.empty((N, K), dtype=ml_dtypes.float8_e4m3fn)
    A_scale = np.empty((M, K // QUANT_SIZE), dtype=ml_dtypes.float8_e8m0fnu)
    B_scale = np.empty((N, K // QUANT_SIZE), dtype=ml_dtypes.float8_e8m0fnu)

    # Vectorized Quantization
    # For A
    A_abs = np.abs(A_origin)
    A_abs_reshaped = A_abs.reshape(M, K // QUANT_SIZE, QUANT_SIZE)
    A_scale_vals = np.max(A_abs_reshaped, axis=2)
    A_scale[:] = A_scale_vals.astype(ml_dtypes.float8_e8m0fnu)
    A_scale_safe = np.where(A_scale_vals == 0, 1.0, A_scale_vals)
    A_origin_reshaped = A_origin.reshape(M, K // QUANT_SIZE, QUANT_SIZE)
    A_fp8_vals = A_origin_reshaped / A_scale_safe[:, :, None]
    A_fp8_vals = np.where(A_scale_vals[:, :, None] == 0, 0.0, A_fp8_vals)
    A_fp8[:] = A_fp8_vals.reshape(M, K).astype(ml_dtypes.float8_e4m3fn)

    # For B
    B_abs = np.abs(B_origin)
    B_abs_reshaped = B_abs.reshape(N, K // QUANT_SIZE, QUANT_SIZE)
    B_scale_vals = np.max(B_abs_reshaped, axis=2)
    B_scale[:] = B_scale_vals.astype(ml_dtypes.float8_e8m0fnu)
    B_scale_safe = np.where(B_scale_vals == 0, 1.0, B_scale_vals)
    B_origin_reshaped = B_origin.reshape(N, K // QUANT_SIZE, QUANT_SIZE)
    B_fp8_vals = B_origin_reshaped / B_scale_safe[:, :, None]
    B_fp8_vals = np.where(B_scale_vals[:, :, None] == 0, 0.0, B_fp8_vals)
    B_fp8[:] = B_fp8_vals.reshape(N, K).astype(ml_dtypes.float8_e4m3fn)

    # pack the scales
    A_scale = np.ascontiguousarray(A_scale)
    B_scale = np.ascontiguousarray(B_scale)
    A_scale_packed = A_scale.view(np.uint32).T
    B_scale_packed = B_scale.view(np.uint32).T

    # Dequantization
    A_fp8_de = A_fp8.astype(np.float32)
    B_fp8_de = B_fp8.astype(np.float32)
    A_scale_de = A_scale.astype(np.float32)
    B_scale_de = B_scale.astype(np.float32)
    A_de = np.empty((M, K), dtype=np.float32)
    B_de = np.empty((N, K), dtype=np.float32)

    # Vectorized dequantization for A
    A_de = (A_fp8_de.reshape(M, K // QUANT_SIZE, QUANT_SIZE) * A_scale_de[:, :, None]).reshape(M, K)
    # Vectorized dequantization for B
    B_de = (B_fp8_de.reshape(N, K // QUANT_SIZE, QUANT_SIZE) * B_scale_de[:, :, None]).reshape(N, K)

    C_standard = np.matmul(A_origin, B_origin.T).astype(np.float16)
    C_ref = np.matmul(A_de, B_de.T).astype(np.float16)
    C_empty = np.empty((M, N), dtype=np.float16)

    return A_fp8, B_fp8, A_scale_packed, B_scale_packed, C_empty, C_standard, C_ref


@tvm.testing.requires_cuda_compute_version(10, exact=True)
def test_deepgemm():
    A_layout = T.ComposeLayout(
        T.SwizzleLayout(4, 3, 3, swizzle_inner=True),
        T.TileLayout(shard=((SMEM_PIPE_DEPTH, BLK_M, BLK_K), (BLK_M * BLK_K, BLK_K, 1))),
    )
    B_layout = T.ComposeLayout(
        T.SwizzleLayout(4, 3, 3, swizzle_inner=True),
        T.TileLayout(shard=((SMEM_PIPE_DEPTH, BLK_N, BLK_K), (BLK_N * BLK_K, BLK_K, 1))),
    )
    D_layout = T.ComposeLayout(
        T.SwizzleLayout(3, 2, 3, swizzle_inner=True),
        T.TileLayout(shard=((TMEM_PIPE_DEPTH, BLK_M, EPI_TILE), (BLK_M * EPI_TILE, EPI_TILE, 1))),
    )

    SFA_layout = T.TileLayout(shard=((SMEM_PIPE_DEPTH, BLK_SFA // 32, 32), (BLK_SFA, 32, 1)))
    SFB_layout = T.TileLayout(shard=((SMEM_PIPE_DEPTH, BLK_SFB // 32, 32), (BLK_SFB, 32, 1)))

    @T.prim_func(tirp=True)
    def deepgemm(
        A: T.Buffer((M, K), a_type),
        B: T.Buffer((N, K), b_type),
        D: T.Buffer((M, N), d_type),
        SFA: T.Buffer((ceildiv(K, QUANT_SIZE) // 4, M), "uint32"),
        SFB: T.Buffer((ceildiv(K, QUANT_SIZE) // 4, N), "uint32"),
    ):
        # fmt: off
        T.func_attr({"global_symbol": "main"})
        D_tensor_map: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
        T.call_packed("runtime.cuTensorMapEncodeTiled", D_tensor_map, d_type, 2, D.data, N, M, N * F16_BYTES, EPI_TILE, BLK_M, 1, 1, 0, 2, 0, 0)

        with T.kernel():
            cbx, cby = T.cta_id([M_CLUSTER, N_CLUSTER], parent="cluster")
            bx = T.cta_id([SM_NUMBER], parent="kernel")
            wg_id = T.warpgroup_id([WG_NUMBER], parent="cta")
            warp_id = T.warp_id([WARP_NUMBER], parent="warpgroup")
            lane_id = T.thread_id([32], parent="warp")
            with T.cta():
                # alloc shared memory
                buf = T.alloc_buffer([SMEM_SIZE], "uint8", scope="shared.dyn")
                pool = T.meta_var(Tp.PoolAllocator(buf.data))
                tmem_addr = T.decl_cell("uint32", buf.data, scope="shared.dyn", elem_offset=0)
                pool.move_base_to(1024)
                A_smem = pool.alloc((SMEM_PIPE_DEPTH, BLK_M, BLK_K), a_type, layout=A_layout)
                B_smem = pool.alloc((SMEM_PIPE_DEPTH, BLK_N, BLK_K), b_type, layout=B_layout)
                D_smem = pool.alloc((TMEM_PIPE_DEPTH, BLK_M, EPI_TILE), d_type, layout=D_layout)
                SFA_smem = pool.alloc((SMEM_PIPE_DEPTH, BLK_SFA // 32, 32), "uint32", layout=SFA_layout)
                SFB_smem = pool.alloc((SMEM_PIPE_DEPTH, BLK_SFB // 32, 32), "uint32", layout=SFB_layout)
                SFA_smem_2d = SFA_smem.view(SMEM_PIPE_DEPTH, BLK_SFA)
                SFB_smem_2d = SFB_smem.view(SMEM_PIPE_DEPTH, BLK_SFB)

                # alloc local memory
                reg = T.alloc_buffer((TMEM_LD_SIZE,), "float32", scope="local")
                reg_wg = reg.view(128, TMEM_LD_SIZE, layout=TileLayout(([128, TMEM_LD_SIZE], [(1, "tid_in_wg"), (1, "m")])))
                reg_fp16 = T.alloc_buffer((BLK_N * CTA_GROUP,), d_type, scope="local")
                stage = T.local_cell("int32")
                descA = T.local_cell("uint64")
                descB = T.local_cell("uint64")
                descSFA = T.local_cell("uint64")
                descSFB = T.local_cell("uint64")
                descI = T.local_cell("uint32")

                phase = T.local_cell("int32")

                # initialize
                tma2trans_bar = T.meta_var(Barriers(buf.data, 6, SMEM_PIPE_DEPTH, True))
                trans2mma_bar = T.meta_var(BarTRANS2MMA(buf.data, 6 + SMEM_PIPE_DEPTH, SMEM_PIPE_DEPTH, True))
                mma2tma_bar = T.meta_var(BarMMA2TMA(buf.data, 6 + 2 * SMEM_PIPE_DEPTH, SMEM_PIPE_DEPTH, False))
                mma2ld_bar = T.meta_var(BarMMA2LD(buf.data, 6 + 3 * SMEM_PIPE_DEPTH, TMEM_PIPE_DEPTH, True))
                ld2mma_bar = T.meta_var(BarLD2MMA(buf.data, 6 + 3 * SMEM_PIPE_DEPTH + TMEM_PIPE_DEPTH, TMEM_PIPE_DEPTH, False))
                tile_scheduler = T.meta_var(GroupMajor2D("tile_scheduler", m_tiles=TILE_M_NUM, n_tiles=TILE_N_NUM, group_rows=TILE_GROUPS_ROW_SIZE, step=SM_NUMBER // 2))

                tma2trans_bar.init(1)
                trans2mma_bar.init(CTA_GROUP * 32)
                mma2ld_bar.init(1)
                mma2tma_bar.init(1)
                ld2mma_bar.init(CTA_GROUP * 128)
                tile_scheduler.init(bx // 2)

                # alloc TMEM
                with T.warp()[0:1]:
                    T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=N_COLS, cta_group=2)
                    T.cuda.warp_sync()

                # sync
                T.ptx.fence.proxy("shared")
                T.ptx.fence.mbarrier_init()
                T.cuda.cluster_sync()
                T.cuda.trap_when_assert_failed(tmem_addr == 0)
                tmem = T.decl_buffer((128, N_COLS), "float32", scope="tmem", allocated_addr=0, layout=TileLayout(([128, N_COLS], [(1, "TLane"), (1, "TCol")])))

                @T.macro
                def paritioned_loop(main_loop, epilogue1, epilogue2):
                    for ko in T.serial(PIPE_CIRCLE_NUM):
                        for ks in T.unroll(SMEM_PIPE_DEPTH):
                            stage = ko * SMEM_PIPE_DEPTH + ks
                            main_loop(ks)
                        phase = phase ^ 1
                    if PIPE_REMAIN_NUM > 0:
                        # last remained loop
                        for ks in T.unroll(PIPE_REMAIN_NUM):
                            stage = PIPE_CIRCLE_NUM * SMEM_PIPE_DEPTH + ks
                            main_loop(ks)
                        epilogue1()
                        # for unaligned cases
                        for ks in T.unroll(PIPE_REMAIN_NUM, SMEM_PIPE_DEPTH):
                            epilogue2(ks)
                        phase = phase ^ 1
                    else:
                        epilogue1()

                with T.cta():
                    T.block_attr({"tirp.scope_partition": True})
                    with T.warpgroup()[wg_id == 1]:
                        T.block_attr({"tirp.scope_partition": True})
                        with T.warp(parent="warpgroup")[warp_id == 3]:
                            phase = 0
                            while tile_scheduler.valid():
                                m_idx = T.meta_var(tile_scheduler.m_idx)
                                n_idx = T.meta_var(tile_scheduler.n_idx)
                                m_start = T.meta_var((m_idx * CTA_GROUP + cbx) * BLK_M)
                                n_start = T.meta_var((n_idx * CTA_GROUP + cbx) * BLK_N)
                                n_start_sf = T.meta_var(n_idx * CTA_GROUP * BLK_N)
                                k_start = T.meta_var(stage * BLK_K)

                                @T.macro
                                def tma_load(ks):
                                    mma2tma_bar.wait(ks, phase)
                                    tma_copy = T.meta_var({"dispatch": "tma", "mbar": tma2trans_bar.mbar.ptr_to([ks]), "cta_group": CTA_GROUP})
                                    with T.thread()[T.ptx.elect_sync()]:
                                        Tp.copy_async(A_smem[ks, :, :], A[m_start: m_start + BLK_M, k_start: k_start + BLK_K], **tma_copy)
                                        Tp.copy_async(B_smem[ks, :, :], B[n_start: n_start + BLK_N, k_start: k_start + BLK_K], **tma_copy)
                                        if stage % 4 == 0:
                                            Tp.copy_async(SFA_smem_2d[ks, :], SFA[stage // 4, m_start: m_start + BLK_M], **tma_copy)
                                            Tp.copy_async(SFB_smem_2d[ks, 0:BLK_N * CTA_GROUP], SFB[stage // 4, n_start_sf: n_start_sf + BLK_N * CTA_GROUP], **tma_copy)
                                        AB_bytes = T.meta_var(BLK_M * BLK_K * F8_BYTES + BLK_N * BLK_K * F8_BYTES)
                                        SFAB_bytes = T.meta_var((BLK_N * CTA_GROUP + BLK_M) * F32_BYTES)
                                        T.ptx.mbarrier.arrive.expect_tx(tma2trans_bar.mbar.ptr_to([ks]), T.if_then_else(stage % 4 == 0, AB_bytes + SFAB_bytes, AB_bytes))

                                @T.macro
                                def tma_load_epilogue(ks):
                                    mma2tma_bar.wait(ks, phase)
                                    with T.thread()[T.ptx.elect_sync()]:
                                        T.ptx.mbarrier.arrive.expect_tx(tma2trans_bar.mbar.ptr_to([ks]), 0)

                                paritioned_loop(tma_load, skip, tma_load_epilogue)
                                tile_scheduler.next_tile()

                        with T.warp(parent="warpgroup")[warp_id == 2]:
                            # transpose
                            phase = 0
                            # reg_trans = T.alloc_buffer((4,), "uint32", scope="local")
                            while tile_scheduler.valid():
                                m_idx = T.meta_var(tile_scheduler.m_idx)
                                n_idx = T.meta_var(tile_scheduler.n_idx)

                                @T.macro
                                def transpose(ks):
                                    # wait for sf has been prepared
                                    T.ptx.mbarrier.try_wait(tma2trans_bar.mbar.ptr_to([ks]), phase)
                                    if stage % 4 == 0:
                                        Tp.permute_dims(SFA_smem[ks], [0, 2, 1])
                                        Tp.permute_dims(SFB_smem[ks, :4], [0, 2, 1])
                                        Tp.permute_dims(SFB_smem[ks, 4:], [0, 2, 1])
                                        # for ki in T.unroll(0, BLK_SFA // 128):
                                        #     for vec in T.vectorized(4):
                                        #         reg_trans[vec] = SFA_smem[ks, ki * 4 + vec, lane_id]
                                        #     T.cuda.warp_sync()
                                        #     for vec in T.vectorized(4):
                                        #         SFA_smem[ks, ki * 4 + (4 * lane_id + vec) // 32, (4 * lane_id + vec) % 32] = reg_trans[vec]
                                        #     T.cuda.warp_sync()
                                        # for ki in T.unroll(0, BLK_SFB // 128):
                                        #     for vec in T.vectorized(4):
                                        #         reg_trans[vec] = SFB_smem[ks, ki * 4 + vec, lane_id]
                                        #     T.cuda.warp_sync()
                                        #     for vec in T.vectorized(4):
                                        #         SFB_smem[ks, ki * 4 + (4 * lane_id + vec) // 32, (4 * lane_id + vec) % 32] = reg_trans[vec]
                                        #     T.cuda.warp_sync()
                                        T.ptx.fence.proxy("shared")
                                    # mark that transpose is completed
                                    trans2mma_bar.arrive(ks)

                                @T.macro
                                def transpose_epilogue(ks):
                                    T.ptx.mbarrier.try_wait(tma2trans_bar.mbar.ptr_to([ks]), phase)
                                    trans2mma_bar.arrive(ks)

                                paritioned_loop(transpose, skip, transpose_epilogue)
                                tile_scheduler.next_tile()

                        with T.warp(parent="warpgroup")[warp_id == 0]:
                            if cbx == 0:
                                tmem_idx = T.local_cell("int32", "tmem_idx")
                                tmem_phase = T.local_cell("int32", "tmem_phase")
                                T.ptx.tcgen05.encode_instr_descriptor_block_scaled(T.address_of(descI), "float32", a_type, b_type, sfa_type, sfb_type,
                                                                                    0, 0, MMA_M, MMA_N, MMA_K, False, False, CTA_GROUP)
                                phase = 0
                                while tile_scheduler.valid():
                                    m_idx = T.meta_var(tile_scheduler.m_idx)
                                    n_idx = T.meta_var(tile_scheduler.n_idx)
                                    with T.thread()[T.ptx.elect_sync()]:
                                        tmem_idx = tile_scheduler.tile_idx % TMEM_PIPE_DEPTH
                                        tmem_phase = (tile_scheduler.tile_idx // TMEM_PIPE_DEPTH) & 1

                                        # wait for the tmem result to be consumed
                                        ld2mma_bar.wait(tmem_idx, tmem_phase)
                                        T.ptx.tcgen05.fence.after_thread_sync()

                                        @T.macro
                                        def mma(ks):
                                            # wait tma and sf-transpose arrival
                                            trans2mma_bar.wait(ks, phase)
                                            T.ptx.tcgen05.fence.after_thread_sync()

                                            # copy sf to tmem
                                            if stage % 4 == 0:
                                                for ki in T.unroll(0, BLK_SFA // 128):
                                                    T.ptx.tcgen05.encode_matrix_descriptor(
                                                        T.address_of(descSFA), SFA_smem.ptr_to([ks, ki * 4, 0]),
                                                        ldo=16, sdo=8 * 4 * F32_BYTES // F128_BYTES, swizzle=0
                                                    )
                                                    T.ptx.tcgen05.cp(0, 0, SFA_TMEM_START_COL + tmem_idx * BLK_SFA // 32 + ki * 4,
                                                                        descSFA, "32x128b", "uint32", "uint32", CTA_GROUP, "warpx4")
                                                for ki in T.unroll(0, BLK_SFB // 128):
                                                    T.ptx.tcgen05.encode_matrix_descriptor(
                                                        T.address_of(descSFB), SFB_smem.ptr_to([ks, ki * 4, 0]),
                                                        ldo=16, sdo=8 * 4 * F32_BYTES // F128_BYTES, swizzle=0
                                                    )
                                                    T.ptx.tcgen05.cp(0, 0, SFB_TMEM_START_COL + tmem_idx * BLK_SFB // 32 + ki * 4,
                                                                        descSFB, "32x128b", "uint32", "uint32", CTA_GROUP, "warpx4")

                                            # issue mma
                                            T.cuda.runtime_instr_desc(T.address_of(descI), stage % 4)
                                            for ki in T.unroll(BLK_K // MMA_K):
                                                T.ptx.tcgen05.encode_matrix_descriptor(T.address_of(descA), A_smem.ptr_to([ks, 0, ki * MMA_K]),
                                                                                    ldo=1, sdo=8 * BLK_K * F8_BYTES // F128_BYTES, swizzle=SWIZZLE)
                                                T.ptx.tcgen05.encode_matrix_descriptor(T.address_of(descB), B_smem.ptr_to([ks, 0, ki * MMA_K]),
                                                                                    ldo=1, sdo=8 * BLK_K * F8_BYTES // F128_BYTES, swizzle=SWIZZLE)

                                                if stage == 0 and ki == 0:
                                                    T.ptx.tcgen05.mma.block_scale("float32", a_type, b_type, sfa_type, sfb_type, tmem_idx * MMA_N, descA, descB,
                                                                                    SFA_TMEM_START_COL + tmem_idx * BLK_SFA // 32,
                                                                                    SFB_TMEM_START_COL + tmem_idx * BLK_SFB // 32,
                                                                                    descI, False, CTA_GROUP, False)
                                                else:
                                                    T.ptx.tcgen05.mma.block_scale("float32", a_type, b_type, sfa_type, sfb_type, tmem_idx * MMA_N, descA, descB,
                                                                                    SFA_TMEM_START_COL + tmem_idx * BLK_SFA // 32,
                                                                                    SFB_TMEM_START_COL + tmem_idx * BLK_SFB // 32,
                                                                                    descI, False, CTA_GROUP, True)
                                            mma2tma_bar.arrive(ks)

                                        @T.macro
                                        def mma_epilogue1():
                                            mma2ld_bar.arrive(tmem_idx)

                                        @T.macro
                                        def mma_epilogue2(ks):
                                            trans2mma_bar.wait(ks, phase)
                                            mma2tma_bar.arrive(ks)

                                        paritioned_loop(mma, mma_epilogue1, mma_epilogue2)

                                    tile_scheduler.next_tile()

                    with T.warpgroup()[wg_id == 0]:
                        T.cuda.trap_when_assert_failed(tmem_addr == 0)
                        tmem_idx = T.local_cell("int32", "tmem_idx")
                        tmem_phase = T.local_cell("int32", "tmem_phase")
                        phase = 0
                        while tile_scheduler.valid():
                            m_idx = T.meta_var(tile_scheduler.m_idx)
                            n_idx = T.meta_var(tile_scheduler.n_idx)
                            tmem_idx = tile_scheduler.tile_idx % TMEM_PIPE_DEPTH
                            tmem_phase = (tile_scheduler.tile_idx // TMEM_PIPE_DEPTH) & 1

                            # flush previous tma
                            T.ptx.cp_async.bulk.wait_group(0)
                            T.cuda.warpgroup_sync(10)
                            # wait for the completion of all the mma of the same tile
                            mma2ld_bar.wait(tmem_idx, tmem_phase)
                            T.ptx.tcgen05.fence.after_thread_sync()

                            for ko in T.unroll(MMA_N // EPI_TILE):
                                stage = (tile_scheduler.tile_idx * MMA_N // EPI_TILE + ko) % TMEM_PIPE_DEPTH

                                # wait the smem to be free
                                if ko >= TMEM_PIPE_DEPTH:
                                    T.ptx.cp_async.bulk.wait_group(TMEM_PIPE_DEPTH - 1)
                                    T.cuda.warpgroup_sync(10)

                                # tmem -> rf (ld) -> smem
                                for ki in T.unroll(EPI_TILE // TMEM_LD_SIZE):
                                    col_st = T.meta_var(tmem_idx * MMA_N + ko * EPI_TILE + ki * TMEM_LD_SIZE)
                                    Tp.copy(reg_wg[:, :], tmem[:, col_st : col_st + TMEM_LD_SIZE])
                                    with T.thread():
                                        st = T.meta_var(ki * TMEM_LD_SIZE)
                                        Tp.cast(reg_fp16[st : st + TMEM_LD_SIZE], reg[:])
                                        Tp.copy(D_smem[stage, warp_id * 32 + lane_id, st : st + TMEM_LD_SIZE], reg_fp16[st : st + TMEM_LD_SIZE])

                                # the tmem can be overwritten
                                if ko == MMA_N // EPI_TILE - 1:
                                    T.ptx.tcgen05.fence.before_thread_sync()
                                    ld2mma_bar.arrive(tmem_idx)

                                T.ptx.fence.proxy(scope="shared")
                                T.cuda.warpgroup_sync(10)

                                # smem -> gmem
                                m_start = (m_idx * CTA_GROUP + cbx) * BLK_M
                                n_start = n_idx * CTA_GROUP * BLK_N + ko * EPI_TILE
                                Tp.copy_async(D[m_start: m_start + BLK_M, n_start: n_start + EPI_TILE], D_smem[stage, :, :], dispatch="tma")
                                T.ptx.cp_async.bulk.commit_group()

                            tile_scheduler.next_tile()

                        T.ptx.cp_async.bulk.wait_group(0)
                        T.cuda.warpgroup_sync(10)

                # dealloc TMEM
                with T.warp()[0:1]:
                    T.ptx.tcgen05.relinquish_alloc_permit(cta_group=2)
                    T.ptx.tcgen05.dealloc(tmem_addr, n_cols=N_COLS, cta_group=2)

                T.cuda.cluster_sync()

    # fmt: on

    DEV = tvm.cuda(0)
    A_fp8, B_fp8, A_scale, B_scale, C, C_standard, C_ref = prepare_data()
    A_tvm = tvm.runtime.tensor(A_fp8, device=DEV)
    B_tvm = tvm.runtime.tensor(B_fp8, device=DEV)
    A_scale_tvm = tvm.runtime.tensor(A_scale, device=DEV)
    B_scale_tvm = tvm.runtime.tensor(B_scale, device=DEV)
    C_tvm = tvm.runtime.tensor(C, device=DEV)
    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": deepgemm})
        # mod.show()
        mod = tvm.tir.transform.LowerTIRp()(mod)
        # mod.show()
        src, mod = get_source(deepgemm)
        # print(src)
        # mod(A_tvm, B_tvm, C_tvm, A_scale_tvm, B_scale_tvm)
        func = lambda: mod(A_tvm, B_tvm, C_tvm, A_scale_tvm, B_scale_tvm)
        ms = bench(func, warmup=0, repeat=100, proton_name="tir")
        print(f"TIR flops: {flops(ms) / 1e12} TFLOPS, time: {ms:.3f} ms")
        np.testing.assert_allclose(C_tvm.numpy(), C_ref, rtol=1e-3, atol=1e-2)


if __name__ == "__main__":
    test_deepgemm()
