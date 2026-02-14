import numpy as np

import tvm
import tvm.testing
from tvm.ir.type import PointerType, PrimType
from tvm.script import tirx as Tx
from tvm.tir.layout import TileLayout, TLane, TCol, tid_in_wg
from tvm.tirx.bench.utils import ProtonContext, bench
from tvm.tirx.tile_scheduler import ClusterPersistentScheduler2D
from tvm.tirx.bench.CuTeDSL.dense_gemm_persistent import run

# cluster: [2, 1], cta_num = 2
# warpgroup:
#   wg_id = 0,1: ld & to GMEM
#   wg_id = 2:
#       warp_id = 0,1: mma
#       warp_id = 3: tma

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

a_type = tvm.DataType("float16")
b_type = tvm.DataType("float16")
d_type = tvm.DataType("float16")
M, N, K = 8192, 8192, 8192
BLK_M, BLK_N, BLK_K = 128, 128, 64
MMA_M, MMA_N, MMA_K = 256, 256, 16
EPI_TILE = 64
QUANT_SIZE = BLK_K
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

DEBUG = False


@Tx.macro
def skip():
    pass


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


TILE_GROUPS_ROW_SIZE = 8
assert M % (NUM_CONSUMER * BLK_M * CTA_GROUP) == 0
assert N % (BLK_N * CTA_GROUP) == 0
TILE_M_NUM = M // (NUM_CONSUMER * BLK_M * CTA_GROUP)
TILE_N_NUM = N // (BLK_N * CTA_GROUP)


class Barriers:

    def __init__(self, shared_buffer_base, shared_buffer_offs, pipe_depth, pipe_width, is_p2c):
        self.mbar: tvm.tir.Buffer = Tx.decl_buffer(
            (pipe_depth, pipe_width), "uint64", shared_buffer_base, elem_offset=shared_buffer_offs
        ).buffer
        self.init_phase = 0 if is_p2c else 1
        self.pipe_depth = pipe_depth
        self.pipe_width = pipe_width

    @Tx.macro
    def init(self, threads_num_wait):
        with Tx.thread()[0:1]:
            for i in Tx.serial(self.pipe_depth):
                for j in Tx.serial(self.pipe_width):
                    Tx.ptx.mbarrier.init(self.mbar.ptr_to([i, j]), threads_num_wait)

    @Tx.macro
    def wait(self, idx_d, idx_w, phase):
        Tx.ptx.mbarrier.try_wait(self.mbar.ptr_to([idx_d, idx_w]), self.init_phase ^ phase)


class BarTMA2MMA(Barriers):

    @Tx.macro
    def arrive(self, idx, expected_bytes):
        Tx.ptx.mbarrier.arrive.expect_tx(self.mbar.ptr_to([idx, 0]), expected_bytes)

    @Tx.macro
    def arrive_only(self, idx):
        Tx.ptx.mbarrier.arrive(self.mbar.ptr_to([idx, 0]))


class BarMMA2LD(Barriers):

    @Tx.macro
    def arrive(self, idx):
        Tx.ptx.tcgen05.commit(self.mbar.ptr_to([0, idx]), cta_group=CTA_GROUP, cta_mask=3)


class BarMMA2TMA(Barriers):

    @Tx.macro
    def arrive(self, idx):
        Tx.ptx.tcgen05.commit(self.mbar.ptr_to([idx, 0]), cta_group=CTA_GROUP, cta_mask=3)


class BarLD2MMA(Barriers):

    @Tx.macro
    def arrive(self, idx):
        Tx.ptx.mbarrier.arrive(self.mbar.ptr_to([0, idx]), cta_id=0, pred=True)


def prepare_data():
    import torch

    A_bf16 = torch.randn((M, K), dtype=torch.float16)
    B_bf16 = torch.randn((N, K), dtype=torch.float16)
    C_empty = torch.zeros((M, N), dtype=torch.float16)

    return A_bf16, B_bf16, C_empty


@tvm.testing.requires_cuda_compute_version(10, exact=True)
def test_hgemm():
    A_layout = Tx.ComposeLayout(
        Tx.SwizzleLayout(3, 3, 3, swizzle_inner=True),
        Tx.TileLayout(
            shard=(
                (PIPELINE_DEPTH, NUM_CONSUMER, BLK_M, BLK_K),
                (NUM_CONSUMER * BLK_M * BLK_K, BLK_M * BLK_K, BLK_K, 1),
            )
        ),
    )
    B_layout = Tx.ComposeLayout(
        Tx.SwizzleLayout(3, 3, 3, swizzle_inner=True),
        Tx.TileLayout(shard=((PIPELINE_DEPTH, BLK_N, BLK_K), (BLK_N * BLK_K, BLK_K, 1))),
    )
    D_layout = Tx.ComposeLayout(
        Tx.SwizzleLayout(3, 3, 3, swizzle_inner=True),
        Tx.TileLayout(shard=((NUM_CONSUMER, BLK_M, EPI_TILE), (BLK_M * EPI_TILE, EPI_TILE, 1))),
    )

    @Tx.prim_func(tirx=True)
    def hgemm(
        A: Tx.Buffer((M, K), a_type), B: Tx.Buffer((N, K), b_type), D: Tx.Buffer((M, N), d_type)
    ):
        # fmt: off
        with Tx.kernel():
            cbx, cby = Tx.cta_id([M_CLUSTER, N_CLUSTER], parent="cluster")
            bx = Tx.cta_id([SM_NUMBER], parent="kernel")
            wg_id = Tx.warpgroup_id([WG_NUMBER], parent="cta")
            warp_id = Tx.warp_id([WARP_NUMBER], parent="warpgroup")
            lane_id = Tx.thread_id([32], parent="warp")
            with Tx.cta():
                # alloc shared memory
                buf = Tx.alloc_buffer([SMEM_SIZE], "uint8", scope="shared.dyn")
                tmem_addr = Tx.decl_cell("uint32", buf.data, scope="shared.dyn", elem_offset=0)
                A_smem = Tx.decl_buffer((PIPELINE_DEPTH, NUM_CONSUMER, BLK_M, BLK_K), a_type, buf.data, layout=A_layout,
                                        elem_offset=1024 // F16_BYTES)
                B_smem = Tx.decl_buffer((PIPELINE_DEPTH, BLK_N, BLK_K), b_type, buf.data, layout=B_layout,
                                        elem_offset=1024 // F16_BYTES + PIPELINE_DEPTH * NUM_CONSUMER * BLK_M * BLK_K)
                D_smem = Tx.decl_buffer((NUM_CONSUMER, BLK_M, EPI_TILE), d_type, buf.data, layout=D_layout,
                                        elem_offset=1024 // F16_BYTES + PIPELINE_DEPTH * (NUM_CONSUMER * BLK_M + BLK_N) * BLK_K)

                # alloc local memory
                descI = Tx.local_cell("uint32")
                descA = Tx.local_cell("uint64")
                descB = Tx.local_cell("uint64")
                phase = Tx.alloc_buffer((1,), "int32", scope="local")
                stage = Tx.local_cell("int32")

                # barriers
                tma2mma = Tx.meta_var(BarTMA2MMA(buf.data, 4, PIPELINE_DEPTH, 1, is_p2c=True))
                mma2tma = Tx.meta_var(BarMMA2TMA(buf.data, 4 + PIPELINE_DEPTH, PIPELINE_DEPTH, 1, is_p2c=False))
                mma2ld = Tx.meta_var(BarMMA2LD(buf.data, 4 + 2 * PIPELINE_DEPTH, 1, NUM_CONSUMER, is_p2c=True))
                ld2mma = Tx.meta_var(BarLD2MMA(buf.data, 4 + 2 * PIPELINE_DEPTH + NUM_CONSUMER, 1, NUM_CONSUMER, is_p2c=False))

                tma2mma.init(1)
                mma2tma.init(NUM_CONSUMER)
                mma2ld.init(1)
                ld2mma.init(128 * NUM_CONSUMER)

                ptr: Tx.Var(name="ptr", dtype=PointerType(PrimType("uint64"))) = Tx.reinterpret("handle", Tx.ptx.map_shared_rank(tma2mma.mbar.ptr_to([0, 0]), 0))
                tma_finished = Tx.decl_buffer([PIPELINE_DEPTH], "uint64", data=ptr, scope="shared")

                # Tile scheduler
                tile_scheduler = Tx.meta_var(ClusterPersistentScheduler2D("tile_scheduler", num_m_tiles=TILE_M_NUM, num_n_tiles=TILE_N_NUM, num_clusters=SM_NUMBER // 2, l2_group_size=TILE_GROUPS_ROW_SIZE))
                tile_scheduler.init(bx // 2)
                m_idx = Tx.meta_var(tile_scheduler.m_idx)
                n_idx = Tx.meta_var(tile_scheduler.n_idx)

                # alloc TMEM
                with Tx.warp()[0:1]:
                    Tx.ptx.tcgen05.alloc(Tx.address_of(tmem_addr), n_cols=N_COLS, cta_group=CTA_GROUP)

                Tx.cuda.cluster_sync()
                Tx.cuda.cta_sync()
                Tx.ptx.fence.proxy("shared")
                Tx.ptx.fence.mbarrier_init()
                Tx.cuda.trap_when_assert_failed(tmem_addr == 0)
                tmem = Tx.decl_buffer((128, N_COLS), "float32", scope="tmem", allocated_addr=0, layout=TileLayout(([128, N_COLS], [1@TLane, 1@TCol])))

                @Tx.macro
                def paritioned_loop(main_loop, epilogue1, epilogue2):
                    for ko in Tx.serial(PIPE_CYCLE):
                        for ks in Tx.unroll(PIPELINE_DEPTH):
                            stage = ko * PIPELINE_DEPTH + ks
                            main_loop(False, ks)
                        phase[0] = phase[0] ^ 1
                    if PIPE_REMAIN_NUM > 0:
                        # last remained loop
                        for ks in Tx.unroll(PIPE_REMAIN_NUM):
                            stage = PIPE_CYCLE * PIPELINE_DEPTH + ks
                            main_loop(True, ks)
                        epilogue1()
                        # for unaligned cases
                        for ks in Tx.unroll(PIPE_REMAIN_NUM, PIPELINE_DEPTH):
                            epilogue2(ks)
                        phase[0] = phase[0] ^ 1
                    else:
                        epilogue1()

                with Tx.cta():
                    Tx.scope_attr({"tirx.scope_partition": True})
                    with Tx.warpgroup()[NUM_CONSUMER:NUM_CONSUMER + 1]:
                        Tx.ptx.setmaxnreg(False, 56)
                        if warp_id == 3:
                            phase[0] = 0
                            while tile_scheduler.valid():
                                # GMEM -> SMEM  (tma)
                                with Tx.thread(parent="warp")[Tx.ptx.elect_sync()]:
                                    # precompute base offsets; k_start depends on stage
                                    m_start0 = Tx.meta_var((m_idx * NUM_CONSUMER * CTA_GROUP + cbx) * BLK_M)
                                    m_start1 = Tx.meta_var((m_idx * NUM_CONSUMER * CTA_GROUP + CTA_GROUP + cbx) * BLK_M)
                                    n_start = Tx.meta_var((n_idx * CTA_GROUP + cbx) * BLK_N)
                                    k_start = Tx.meta_var(stage * BLK_K)

                                    @Tx.macro
                                    def tma_load(is_remain, ks):
                                        tma_copy = Tx.meta_var({"dispatch": "tma", "mbar": tma_finished.ptr_to([ks]), "cta_group": CTA_GROUP})
                                        mma2tma.wait(ks, 0, phase[0])
                                        Tx.copy_async(A_smem[ks, 0, :, :], A[m_start0 : m_start0 + BLK_M, k_start : k_start + BLK_K], **tma_copy)
                                        Tx.copy_async(A_smem[ks, 1, :, :], A[m_start1 : m_start1 + BLK_M, k_start : k_start + BLK_K], **tma_copy)
                                        Tx.copy_async(B_smem[ks, :, :], B[n_start : n_start + BLK_N, k_start : k_start + BLK_K], **tma_copy)
                                        if cbx == 0:
                                            tma2mma.arrive(ks, NUM_CONSUMER * BLK_K * (BLK_M * NUM_CONSUMER + BLK_N) * F16_BYTES)

                                    @Tx.macro
                                    def tma_load_epilogue(ks):
                                        mma2tma.wait(ks, 0, phase[0])
                                        if cbx == 0:
                                            tma2mma.arrive_only(ks)

                                    paritioned_loop(tma_load, skip, tma_load_epilogue)
                                tile_scheduler.next_tile()

                        elif warp_id < 2 and cbx == 0:
                            phase_tmem = Tx.local_cell("int32")
                            phase_tmem = 0
                            phase[0] = 0

                            Tx.ptx.tcgen05.encode_instr_descriptor(Tx.address_of(descI), "float32", a_type, b_type, MMA_M, MMA_N, MMA_K, False, False, CTA_GROUP)

                            while tile_scheduler.valid():
                                m_idx = Tx.meta_var(tile_scheduler.m_idx)
                                n_idx = Tx.meta_var(tile_scheduler.n_idx)
                                with Tx.thread()[Tx.ptx.elect_sync()]:
                                    ld2mma.wait(0, warp_id, phase_tmem)
                                    Tx.ptx.tcgen05.fence.after_thread_sync()

                                    @Tx.macro
                                    def mma(is_remain, ks):
                                        # wait tma
                                        tma2mma.wait(ks, 0, phase[0])
                                        Tx.gemm_async(tmem[:, warp_id * MMA_N: warp_id * MMA_N + MMA_N], A_smem[ks, warp_id, :, :], B_smem[ks, :, :], dispatch="tcgen05", cta_group=CTA_GROUP, descI=descI,
                                                        accum=tvm.tir.Not(stage == 0 and ((not is_remain) or (is_remain and PIPE_CYCLE == 0))))
                                        mma2tma.arrive(ks)

                                    @Tx.macro
                                    def mma_epilogue1():
                                        mma2ld.arrive(warp_id)

                                    @Tx.macro
                                    def mma_epilogue2(ks):
                                        tma2mma.wait(ks, 0, phase[0])
                                        mma2tma.arrive(ks)

                                    paritioned_loop(mma, mma_epilogue1, mma_epilogue2)
                                    phase_tmem = phase_tmem ^ 1
                                tile_scheduler.next_tile()

                    with Tx.warpgroup()[0:NUM_CONSUMER]:
                        Tx.ptx.setmaxnreg(True, 224)

                        reg = Tx.alloc_buffer((TMEM_LD_SIZE,), "float32", scope="local")
                        reg_wg = reg.view(128, TMEM_LD_SIZE, layout=TileLayout(([128, TMEM_LD_SIZE], [1@tid_in_wg, 1])))
                        reg_fp16 = Tx.alloc_buffer((BLK_N * CTA_GROUP,), d_type, scope="local")
                        phase_tmem = Tx.local_cell("int32")

                        phase_tmem = 0
                        while tile_scheduler.valid():
                            mma2ld.wait(0, wg_id, phase_tmem)
                            phase_tmem = phase_tmem ^ 1
                            Tx.ptx.tcgen05.fence.after_thread_sync()
                            # TMEM -> RF (ld)
                            for i in Tx.unroll(MMA_N // TMEM_LD_SIZE):  # load (MMA_M // 2, MMA_N)
                                col_st = Tx.meta_var(wg_id * MMA_N + i * TMEM_LD_SIZE)
                                Tx.copy(reg_wg[:, :], tmem[:, col_st : col_st + TMEM_LD_SIZE])
                                with Tx.thread():
                                    Tx.cast(reg_fp16[i * TMEM_LD_SIZE : (i + 1) * TMEM_LD_SIZE], reg[:])

                            # the tmem can be overwritten by the next tile
                            ld2mma.arrive(wg_id)
                            # RF -> GMEM
                            for i in Tx.unroll(NUM_CONSUMER * BLK_N // EPI_TILE):
                                with Tx.thread():
                                    Tx.copy(D_smem[wg_id, warp_id * 32 + lane_id, :], reg_fp16[i * EPI_TILE : (i + 1) * EPI_TILE])
                                Tx.cuda.warpgroup_sync(wg_id)
                                Tx.ptx.fence.proxy(scope="shared")
                                # st to gmem via event
                                with Tx.thread()[lane_id == 0 and warp_id == 0]:
                                    m_start = (m_idx * NUM_CONSUMER * CTA_GROUP + wg_id * CTA_GROUP + cbx) * BLK_M
                                    n_start = n_idx * BLK_N * CTA_GROUP + i * EPI_TILE
                                    Tx.copy_async(D[m_start : m_start + BLK_M, n_start : n_start + EPI_TILE], D_smem[wg_id, :, :], dispatch="tma")
                                    Tx.ptx.cp_async.bulk.commit_group()
                                    Tx.ptx.cp_async.bulk.wait_group(0)
                                Tx.cuda.warpgroup_sync(wg_id)
                            tile_scheduler.next_tile()

                # dealloc TMEM
                with Tx.warp()[0:1]:
                    Tx.ptx.tcgen05.relinquish_alloc_permit(cta_group=CTA_GROUP)
                    Tx.ptx.tcgen05.dealloc(tmem_addr, n_cols=N_COLS, cta_group=CTA_GROUP)

                Tx.cuda.cluster_sync()

    A_bf16, B_bf16, C_bf16 = prepare_data()

    def tir_gemm(A_bf16, B_bf16, C_bf16):
        DEV = tvm.cuda(0)
        A_tvm = tvm.runtime.tensor(A_bf16, device=DEV)
        B_tvm = tvm.runtime.tensor(B_bf16, device=DEV)
        C_tvm = tvm.runtime.tensor(C_bf16, device=DEV)
        target = tvm.target.Target("cuda")
        with target:
            src, mod = get_source(hgemm)
            func = lambda: mod(A_tvm, B_tvm, C_tvm)
            ms = bench(func, warmup=0, repeat=30, proton_name="tir", debug=DEBUG)
            print(f"TIR flops: {flops(ms) / 1e12} TFLOPS, time: {ms:.3f} ms")

        return C_tvm.numpy()

    def cublas_gemm(A_bf16, B_bf16):
        import torch

        torch_dev = torch.device("cuda")
        A_torch = A_bf16.to(torch_dev)
        B_torch = B_bf16.to(torch_dev)
        func = lambda: torch.matmul(A_torch, B_torch.T)
        ms = bench(func, warmup=0, repeat=30, proton_name="cublas", debug=DEBUG)
        print(f"CUBLAS flops: {flops(ms) / 1e12} TFLOPS, time: {ms:.3f} ms")
        C_torch = func()
        return C_torch.cpu().numpy()

    def cutedsl_gemm(A_bf16, B_bf16):
        import cutlass
        import cutlass.cute as cute
        import cuda.bindings.driver as cuda
        import torch
        from cutlass.cute.runtime import from_dlpack

        def create_cutlass_tensor(
            tensor, dtype, is_dynamic_layout=True, assumed_align=16, leading_dim=1
        ):
            cute_tensor = from_dlpack(tensor, assumed_align=assumed_align)
            cute_tensor.element_type = dtype
            if is_dynamic_layout:
                cute_tensor = cute_tensor.mark_layout_dynamic(leading_dim=leading_dim)
            return cute_tensor

        A_torch = A_bf16.to(torch.device("cuda")).reshape(1, M, K).permute(1, 2, 0)
        B_torch = B_bf16.to(torch.device("cuda")).reshape(1, N, K).permute(1, 2, 0)
        C_torch = torch.zeros_like(
            C_bf16.reshape(1, M, N).permute(1, 2, 0), device=torch.device("cuda")
        )
        func = run(
            mnkl=(M, N, K, 1),
            ab_dtype=cutlass.Float16,
            c_dtype=cutlass.Float16,
            acc_dtype=cutlass.Float32,
            a_major="k",
            b_major="k",
            c_major="n",
            skip_ref_check=True,
            A_torch=A_torch,
            B_torch=B_torch,
            C_torch=C_torch,
        )
        ms = bench(func, warmup=10, repeat=30, proton_name="cutedsl", debug=DEBUG)
        print(f"CuTeDSL flops: {flops(ms) / 1e12} TFLOPS, time: {ms:.3f} ms")
        return C_torch.cpu().numpy().reshape(M, N)

    with ProtonContext("blackwell_gemm", debug=DEBUG):
        C_tvm = tir_gemm(A_bf16, B_bf16, C_bf16)
        C_cublas = cublas_gemm(A_bf16, B_bf16)
        C_cutedsl = cutedsl_gemm(A_bf16, B_bf16)

    np.testing.assert_allclose(C_tvm, C_cublas, rtol=1e-3, atol=1e-2)
    np.testing.assert_allclose(C_cutedsl, C_cublas, rtol=1e-3, atol=1e-2)


if __name__ == "__main__":
    test_hgemm()
