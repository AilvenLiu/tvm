import numpy as np

import tvm
from tvm.ir.type import PointerType, PrimType
from tvm.script import tir as T
import tvm.testing
from ..utils import bench, ProtonContext


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
assert (K // BLK_K) % PIPELINE_DEPTH == 0
assert PIPELINE_DEPTH == 4

@T.macro
def warp_sync():
    T.cuda.func_call("sync_warp", source_code = f"""
__forceinline__ __device__ void sync_warp() {{
    __syncwarp();
}}
    """)   

@T.macro
def trap_when_assert_failed(cond):
    T.cuda.func_call("trap_when_assert_fail", cond, source_code=f"""
__forceinline__ __device__ void trap_when_assert_fail(bool cond) {{
    do {{
        if (not (cond))
            asm("trap;");
    }} while (0);
}}
    """)




def get_source(func: tvm.tir.PrimFunc) -> str:
    target = tvm.target.Target("cuda")
    mod = tvm.IRModule({"main": func})
    mod = tvm.compile(mod, target=target, tir_pipeline="tirp")
    src = mod.mod.imported_modules[0].get_source()
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


class TileScheduler:

    def __init__(self, prefix: str):
        self.m_idx = T.local_cell("int32", name=prefix + "_m_idx")
        self.n_idx = T.local_cell("int32", name=prefix + "_n_idx")
        self.linear_idx = T.local_cell("int32", name=prefix + "_linear_idx")

    @T.macro
    def update_current_m_n_idx(self, linear_idx):
        TILE_GROUPS_NUM = TILE_M_NUM // TILE_GROUPS_ROW_SIZE
        TILE_GROUPS_SIZE = TILE_GROUPS_ROW_SIZE * TILE_N_NUM
        TILE_FINAL_ROWS = TILE_M_NUM - (TILE_GROUPS_NUM * TILE_GROUPS_ROW_SIZE)
        if linear_idx < TILE_GROUPS_NUM * TILE_GROUPS_SIZE and TILE_GROUPS_NUM > 0:
            self.m_idx = linear_idx // TILE_GROUPS_SIZE * TILE_GROUPS_ROW_SIZE + (
                linear_idx % TILE_GROUPS_ROW_SIZE
            )
            self.n_idx = (linear_idx % TILE_GROUPS_SIZE) // TILE_GROUPS_ROW_SIZE
        elif TILE_FINAL_ROWS > 0:
            remainder_idx = linear_idx - TILE_GROUPS_SIZE * TILE_GROUPS_NUM
            self.m_idx = TILE_GROUPS_NUM * TILE_GROUPS_ROW_SIZE + remainder_idx % TILE_FINAL_ROWS
            self.n_idx = remainder_idx // TILE_FINAL_ROWS

    @T.macro
    def init(self, linear_init):
        self.linear_idx = linear_init
        self.update_current_m_n_idx(linear_init)

    @T.macro
    def next_tile(self):
        self.linear_idx = self.linear_idx + SM_NUMBER // 2
        self.update_current_m_n_idx(self.linear_idx)

    def valid(self):
        return self.linear_idx < TILE_M_NUM * TILE_N_NUM


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


def prepare_data():
    import torch
    A_bf16 = torch.randn((M, K), dtype=torch.float16)
    B_bf16 = torch.randn((N, K), dtype=torch.float16)
    C_empty = torch.zeros((M, N), dtype=torch.float16)

    return A_bf16, B_bf16, C_empty



@tvm.testing.requires_cuda_compute_version(10, exact=True)
def test():

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
    def hgemm(A: T.Buffer((M, K), a_type, layout="default"), B: T.Buffer((N, K), b_type, layout="default"), 
                D: T.Buffer((M, N), d_type, layout="default")):
        
        A_tensor_map: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
        B_tensor_map: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
        D_tensor_map: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
        T.call_packed("runtime.cuTensorMapEncodeTiled", A_tensor_map, a_type, 2, A.data,
                      K, M, K * F16_BYTES, BLK_K, BLK_M, 1, 1, 0, SWIZZLE, 0, 0)
        T.call_packed("runtime.cuTensorMapEncodeTiled", B_tensor_map, b_type, 2, B.data, 
                      K, N, K * F16_BYTES, BLK_K, BLK_N, 1, 1, 0, SWIZZLE, 0, 0)
        T.call_packed("runtime.cuTensorMapEncodeTiled", D_tensor_map, d_type, 2, D.data,
                      N, M, N * F16_BYTES, EPI_TILE, BLK_M, 1, 1, 0, SWIZZLE, 0, 0)
        with T.kernel():
            cbx, cby = T.cta_id([M_CLUSTER, N_CLUSTER], parent="cluster")
            bx = T.cta_id([SM_NUMBER], parent="kernel")
            wg_id = T.warpgroup_id([WG_NUMBER], parent="cta")
            warp_id = T.warp_id([WARP_NUMBER], parent="warpgroup")
            lane_id = T.thread_id([32], parent="warp")
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
                reg = T.alloc_buffer((TMEM_LD_SIZE,), "float32", scope="local")
                reg_fp16 = T.alloc_buffer((BLK_N * CTA_GROUP,), d_type, scope="local")
                descA = T.local_cell("uint64")
                descB = T.local_cell("uint64")
                descI = T.local_cell("uint32")
                phase = T.alloc_buffer((1,), "int32", scope="local")
                stage = T.local_cell("int32", name="stage")
                
                # initialize
                tma2mma = T.meta_var(BarTMA2MMA(buf.data, 4, PIPELINE_DEPTH, 1, is_p2c=True))
                mma2tma = T.meta_var(BarMMA2TMA(buf.data, 4 + PIPELINE_DEPTH, PIPELINE_DEPTH, 1, is_p2c=False))
                mma2ld = T.meta_var(BarMMA2LD(buf.data, 4 + 2 * PIPELINE_DEPTH, 1, NUM_CONSUMER, is_p2c=True))
                ld2mma = T.meta_var(BarLD2MMA(buf.data, 4 + 2 * PIPELINE_DEPTH + NUM_CONSUMER, 1, NUM_CONSUMER, is_p2c=False))
                tile_scheduler = T.meta_var(TileScheduler("tile_scheduler"))
                tma2mma.init(1)
                mma2tma.init(NUM_CONSUMER)
                mma2ld.init(1)
                ld2mma.init(128 * NUM_CONSUMER)
                tile_scheduler.init(bx // 2)
                ptr: T.Var(name="ptr", dtype=PointerType(PrimType("uint64"))) = T.reinterpret("handle", T.ptx.map_shared_rank(tma2mma.mbar.ptr_to([0, 0]), 0))
                tma_finished = T.decl_buffer([PIPELINE_DEPTH], "uint64", data=ptr, scope="shared")

                # alloc TMEM
                with T.warp()[0:1]:
                    T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=N_COLS, cta_group=1)

                T.ptx.barrier.cluster.arrive()
                T.ptx.barrier.cluster.wait()
                T.tvm_storage_sync("shared")
                T.ptx.fence.proxy("shared")
                T.ptx.fence.mbarrier_init()
                
                with T.cta():
                    T.block_attr({"tirp.scope_partition": True})
                    with T.warpgroup()[NUM_CONSUMER:NUM_CONSUMER + 1]:
                        T.ptx.setmaxnreg(False, 56)
                        if warp_id == 3: 
                            phase[0] = 0
                            while tile_scheduler.valid():
                                m_idx = T.meta_var(tile_scheduler.m_idx)
                                n_idx = T.meta_var(tile_scheduler.n_idx)

                                # GMEM -> SMEM  (tma)
                                for ko in T.serial(PIPE_CYCLE):
                                    for ks in T.unroll(PIPELINE_DEPTH):
                                        stage = ko * PIPELINE_DEPTH + ks
                                        mma2tma.wait(ks, 0, phase[0])
                                        if T.ptx.elect_sync():
                                            T.ptx.cp_async.bulk.tensor.g2c(2, A_smem.ptr_to([ks, 0, 0, 0]), tma_finished.ptr_to([ks]),
                                                                            A_tensor_map, stage * BLK_K, (m_idx * NUM_CONSUMER * CTA_GROUP + cbx) * BLK_M, cta_group=2)
                                            T.ptx.cp_async.bulk.tensor.g2c(2, A_smem.ptr_to([ks, 1, 0, 0]), tma_finished.ptr_to([ks]),
                                                                            A_tensor_map, stage * BLK_K, (m_idx * NUM_CONSUMER * CTA_GROUP + CTA_GROUP + cbx) * BLK_M, cta_group=2)
                                            T.ptx.cp_async.bulk.tensor.g2c(2, B_smem.ptr_to([ks, 0, 0]), tma_finished.ptr_to([ks]),
                                                                            B_tensor_map, stage * BLK_K, (n_idx * CTA_GROUP + cbx) * BLK_N, cta_group=2)

                                            if cbx == 0:
                                                tma2mma.arrive(ks, NUM_CONSUMER * BLK_K * (BLK_M * NUM_CONSUMER + BLK_N) * F16_BYTES)
                                    phase[0] = phase[0] ^ 1
                                
                                tile_scheduler.next_tile()

                
                        elif warp_id < 2 and cbx == 0:
                            phase_tmem = T.alloc_buffer((1,), "int32", scope="local")
                            phase_tmem[0] = 0
                            phase[0] = 0
                            T.ptx.tcgen05.encode_instr_descriptor(T.address_of(descI), "float32", a_type, b_type, MMA_M, MMA_N, MMA_K, False, False, CTA_GROUP)
                            while tile_scheduler.valid():
                                m_idx = T.meta_var(tile_scheduler.m_idx)
                                n_idx = T.meta_var(tile_scheduler.n_idx)
                                with T.thread():
                                    ld2mma.wait(0, warp_id, phase_tmem[0])
                                    T.ptx.tcgen05.fence.after_thread_sync()

                                    for ko in T.serial(PIPE_CYCLE):
                                        for ks in T.unroll(PIPELINE_DEPTH):
                                            stage = ko * PIPELINE_DEPTH + ks

                                            # wait tma
                                            tma2mma.wait(ks, 0, phase[0])
                                            if T.ptx.elect_sync():
                                                for ki in T.unroll(BLK_K // MMA_K):
                                                    T.ptx.tcgen05.encode_matrix_descriptor(T.address_of(descA), A_smem.ptr_to([ks, warp_id, 0, ki * MMA_K]), 
                                                                                        ldo=1, sdo=8 * BLK_K * F16_BYTES // F128_BYTES, swizzle=SWIZZLE)
                                                    T.ptx.tcgen05.encode_matrix_descriptor(T.address_of(descB), B_smem.ptr_to([ks, 0, ki * MMA_K]), 
                                                                                        ldo=1, sdo=8 * BLK_K * F16_BYTES // F128_BYTES, swizzle=SWIZZLE)
                                                    
                                                    if stage == 0 and ki == 0:
                                                        T.ptx.tcgen05.mma("float32", a_type, b_type, warp_id * MMA_N, descA, descB, 
                                                                            descI, False, CTA_GROUP, False)
                                                    else:
                                                        T.ptx.tcgen05.mma("float32", a_type, b_type, warp_id * MMA_N, descA, descB, 
                                                                            descI, False, CTA_GROUP, True)
                                          
                                                mma2tma.arrive(ks)
                                        phase[0] = phase[0] ^ 1
                                    if T.ptx.elect_sync():
                                        mma2ld.arrive(warp_id)
                                    phase_tmem[0] = phase_tmem[0] ^ 1
                                tile_scheduler.next_tile()

                    with T.warpgroup()[0:NUM_CONSUMER]:
                        T.ptx.setmaxnreg(True, 224)
                        trap_when_assert_failed(tmem_addr == 0)
                        phase_tmem = T.alloc_buffer((1,), "int32", scope="local")
                        phase_tmem[0] = 0
                        while tile_scheduler.valid():
                            m_idx = T.meta_var(tile_scheduler.m_idx)
                            n_idx = T.meta_var(tile_scheduler.n_idx)
                            mma2ld.wait(0, wg_id, phase_tmem[0])
                            phase_tmem[0] = phase_tmem[0] ^ 1
                            T.ptx.tcgen05.fence.after_thread_sync()
                            # TMEM -> RF (ld)
                            for i in T.unroll(MMA_N // TMEM_LD_SIZE): # load (MMA_M // 2, MMA_N)
                                T.ptx.tcgen05.ld(wg_id * MMA_N, warp_id * 32, i * TMEM_LD_SIZE, "32x32b", TMEM_LD_SIZE, False, *[reg[j] for j in range(TMEM_LD_SIZE)])
                                T.ptx.tcgen05.wait.ld()
                                for j in range(TMEM_LD_SIZE):
                                    reg_fp16[i * TMEM_LD_SIZE + j] = T.cast(reg[j], "float16")

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
                                if lane_id == 0 and warp_id == 0:
                                    T.ptx.cp_async.bulk.tensor.s2g(2, D_smem.ptr_to([wg_id, 0, 0]), 
                                                                D_tensor_map, n_idx * BLK_N * CTA_GROUP + i * EPI_TILE,
                                                                (m_idx * NUM_CONSUMER * CTA_GROUP + wg_id * CTA_GROUP + cbx) * BLK_M)
                                    T.ptx.cp_async.bulk.commit_group()
                                    T.ptx.cp_async.bulk.wait_group(0)
                                T.ptx.bar.sync(wg_id, 128)
                            tile_scheduler.next_tile()

                # dealloc TMEM
                with T.warp()[0:1]:
                    T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
                    T.ptx.tcgen05.dealloc(tmem_addr, n_cols=N_COLS, cta_group=1)
                
                T.ptx.barrier.cluster.arrive()
                T.ptx.barrier.cluster.wait()


    A_bf16, B_bf16, C_bf16 = prepare_data()


    def tir_gemm(A_bf16, B_bf16, C_bf16):
        DEV = tvm.cuda(0)
        A_tvm = tvm.nd.array(A_bf16, device=DEV)
        B_tvm = tvm.nd.array(B_bf16, device=DEV)
        C_tvm = tvm.nd.array(C_bf16, device=DEV)
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
    test()