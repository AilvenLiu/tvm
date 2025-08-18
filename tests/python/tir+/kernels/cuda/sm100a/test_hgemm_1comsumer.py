import ml_dtypes
import numpy as np

import tvm
from tvm.ir import PointerType, PrimType
from tvm.script import tir as T
import tvm.testing
from tvm.tirp.bench.utils import ProtonContext, bench

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
M, N, K = 8192, 8192, 4096
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
    mod = tvm.compile(mod, target=target, tir_pipeline="tirp")
    src = mod.mod.imported_modules[0].get_source()
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


class TileScheduler:

    def __init__(self, prefix: str):
        self.m_idx = T.local_cell("int32", name=prefix + "_m_idx")
        self.n_idx = T.local_cell("int32", name=prefix + "_n_idx")
        self.linear_idx = T.local_cell("int32", name=prefix + "_linear_idx")
        self.tile_idx = T.local_cell("int32", name="tile_idx")

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
        self.tile_idx = 0
        self.update_current_m_n_idx(linear_init)

    @T.macro
    def next_tile(self):
        self.linear_idx = self.linear_idx + SM_NUMBER // 2
        self.tile_idx += 1
        self.update_current_m_n_idx(self.linear_idx)

    def valid(self):
        return self.linear_idx < TILE_M_NUM * TILE_N_NUM


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


class BarTMA2MMA(Barriers):

    @T.macro
    def arrive(self, idx, expected_bytes):
        T.ptx.mbarrier.arrive.expect_tx(self.mbar.ptr_to([idx]), expected_bytes)

    @T.macro
    def arrive_only(self, idx):
        T.ptx.mbarrier.arrive(self.mbar.ptr_to([idx]))


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
def warp_sync():
    T.cuda.func_call(
        "sync_warp",
        source_code=f"""
__forceinline__ __device__ void sync_warp() {{
    __syncwarp();
}}
    """,
    )


@T.macro
def trap_when_assert_failed(cond):
    T.cuda.func_call(
        "trap_when_assert_fail",
        cond,
        source_code=f"""
__forceinline__ __device__ void trap_when_assert_fail(bool cond) {{
    do {{
        if (not (cond))
            asm("trap;");
    }} while (0);
}}
    """,
    )


@T.macro
def float22half2(dst, src):
    T.cuda.func_call(
        "float22half2",
        dst,
        src,
        source_code=f"""
__forceinline__ __device__ void float22half2(void* dst, void* src) {{
    half2* dst_p = (half2*) dst;
    float2* src_p = (float2*) src;
    *dst_p = __float22half2_rn(*src_p);
}}
    """,
    )


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
        T.TileLayout(shard=((SMEM_PIPE_DEPTH, BLK_M, BLK_K), (BLK_M * BLK_K, BLK_K, 1))),
    )
    B_layout = T.ComposeLayout(
        T.SwizzleLayout(3, 3, 3, swizzle_inner=True),
        T.TileLayout(shard=((SMEM_PIPE_DEPTH, BLK_N, BLK_K), (BLK_N * BLK_K, BLK_K, 1))),
    )
    D_layout = T.ComposeLayout(
        T.SwizzleLayout(3, 3, 3, swizzle_inner=True),
        T.TileLayout(shard=((TMEM_PIPE_DEPTH, BLK_M, EPI_TILE), (BLK_M * EPI_TILE, EPI_TILE, 1))),
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
                A_smem = T.decl_buffer((SMEM_PIPE_DEPTH, BLK_M, BLK_K), a_type, buf.data, layout=A_layout,
                                        elem_offset=1024 // F16_BYTES)
                B_smem = T.decl_buffer((SMEM_PIPE_DEPTH, BLK_N, BLK_K), b_type, buf.data, layout=B_layout,
                                        elem_offset=1024 // F16_BYTES + SMEM_PIPE_DEPTH * BLK_M * BLK_K)
                D_smem = T.decl_buffer((TMEM_PIPE_DEPTH, BLK_M, EPI_TILE), d_type, buf.data, layout=D_layout,
                                        elem_offset=1024 // F16_BYTES + SMEM_PIPE_DEPTH * (BLK_M + BLK_N) * BLK_K)
               
                # alloc local memory
                reg = T.alloc_buffer((TMEM_LD_SIZE,), "float32", scope="local")
                reg_fp16 = T.alloc_buffer((TMEM_LD_SIZE,), d_type, scope="local")
                stage = T.local_cell("int32", name="stage")
                phase = T.alloc_buffer((1, ), "int32", scope="local")
                descA = T.local_cell("uint64")
                descB = T.local_cell("uint64")
                descI = T.local_cell("uint32")
                
                # initialize
                tma2mma_bar = T.meta_var(BarTMA2MMA(buf.data, 6, SMEM_PIPE_DEPTH, True))
                mma2tma_bar = T.meta_var(BarMMA2TMA(buf.data, 6 + 2 * SMEM_PIPE_DEPTH, SMEM_PIPE_DEPTH, False))
                mma2ld_bar = T.meta_var(BarMMA2LD(buf.data, 6 + 3 * SMEM_PIPE_DEPTH, TMEM_PIPE_DEPTH, True))
                ld2mma_bar = T.meta_var(BarLD2MMA(buf.data, 6 + 3 * SMEM_PIPE_DEPTH + TMEM_PIPE_DEPTH, TMEM_PIPE_DEPTH, False))
                tile_scheduler = T.meta_var(TileScheduler("tile_scheduler"))
                tma2mma_bar.init(1)
                mma2ld_bar.init(1)
                mma2tma_bar.init(1)
                ld2mma_bar.init(CTA_GROUP * 128)
                tile_scheduler.init(bx // 2)
                ptr: T.Var(name="ptr", dtype=PointerType(PrimType("uint64"))) = T.reinterpret("handle", T.ptx.map_shared_rank(tma2mma_bar.mbar.ptr_to([0]), 0))
                tma_finished = T.decl_buffer([SMEM_PIPE_DEPTH], "uint64", data=ptr, scope="shared")

                # alloc TMEM
                with T.warp()[0:1]:
                    T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=N_COLS, cta_group=1)
                    warp_sync()
                
                # sync
                T.ptx.fence.proxy("shared")
                T.ptx.fence.mbarrier_init()
                T.ptx.barrier.cluster.arrive()
                T.ptx.barrier.cluster.wait()

                with T.cta():
                    T.block_attr({"tirp.scope_partition": True})
                    with T.warpgroup()[1:2]:
                        if warp_id == 3: 
                            phase[0] = 0
                            while tile_scheduler.valid():
                                m_idx = T.meta_var(tile_scheduler.m_idx)
                                n_idx = T.meta_var(tile_scheduler.n_idx)
                                if T.ptx.elect_sync():
                                    # main inner tma loop
                                    for ko in T.serial(PIPE_CIRCLE_NUM):
                                        for ks in T.unroll(SMEM_PIPE_DEPTH):
                                            # GMEM -> SMEM  (tma)    
                                            stage = (ko * SMEM_PIPE_DEPTH + ks)
                                            mma2tma_bar.wait(ks, phase[0])
                                            T.ptx.cp_async.bulk.tensor.g2c(2, A_smem.ptr_to([ks, 0, 0]), tma_finished.ptr_to([ks]),
                                                                        A_tensor_map, stage * BLK_K, (m_idx * CTA_GROUP + cbx) * BLK_M, cta_group=2)
                                            T.ptx.cp_async.bulk.tensor.g2c(2, B_smem.ptr_to([ks, 0, 0]), tma_finished.ptr_to([ks]),
                                                                        B_tensor_map, stage * BLK_K, (n_idx * CTA_GROUP + cbx) * BLK_N, cta_group=2)
                                            if cbx == 0:
                                                tma2mma_bar.arrive(ks, CTA_GROUP * BLK_K * (BLK_M + BLK_N) * F16_BYTES)
                                        phase[0] = phase[0] ^ 1

                                    if PIPE_REMAIN_NUM > 0:
                                        # last remained loop
                                        for ks in T.unroll(PIPE_REMAIN_NUM):
                                            # GMEM -> SMEM  (tma)    
                                            stage = PIPE_CIRCLE_NUM * SMEM_PIPE_DEPTH + ks
                                            mma2tma_bar.wait(ks, phase[0])
                                            T.ptx.cp_async.bulk.tensor.g2c(2, A_smem.ptr_to([ks, 0, 0]), tma_finished.ptr_to([ks]),
                                                                        A_tensor_map, stage * BLK_K, (m_idx * CTA_GROUP + cbx) * BLK_M, cta_group=2)
                                            T.ptx.cp_async.bulk.tensor.g2c(2, B_smem.ptr_to([ks, 0, 0]), tma_finished.ptr_to([ks]),
                                                                        B_tensor_map, stage * BLK_K, (n_idx * CTA_GROUP + cbx) * BLK_N, cta_group=2)
                                            if cbx == 0:
                                                tma2mma_bar.arrive(ks, CTA_GROUP * BLK_K * (BLK_M + BLK_N) * F16_BYTES)
                                        
                                        # for unaligned cases   
                                        for ks in T.unroll(PIPE_REMAIN_NUM, SMEM_PIPE_DEPTH):
                                            mma2tma_bar.wait(ks, phase[0])
                                            if cbx == 0:
                                                tma2mma_bar.arrive_only(ks)
                                    
                                        phase[0] = phase[0] ^ 1
                                tile_scheduler.next_tile()
                        
                        elif warp_id == 0:
                            if cbx == 0:
                                tmem_idx = T.local_cell("int32", "tmem_idx")
                                tmem_phase = T.local_cell("int32", "tmem_phase")
                                T.ptx.tcgen05.encode_instr_descriptor(T.address_of(descI), "float32", a_type, b_type, MMA_M, MMA_N, MMA_K, False, False, CTA_GROUP)
                                phase[0] = 0
                                while tile_scheduler.valid():
                                    m_idx = T.meta_var(tile_scheduler.m_idx)
                                    n_idx = T.meta_var(tile_scheduler.n_idx)
                                    if T.ptx.elect_sync():
                                        tmem_idx = tile_scheduler.tile_idx % TMEM_PIPE_DEPTH
                                        tmem_phase = (tile_scheduler.tile_idx // TMEM_PIPE_DEPTH) & 1

                                        # wait for the tmem result to be consumed
                                        ld2mma_bar.wait(tmem_idx, tmem_phase)
                                        T.ptx.tcgen05.fence.after_thread_sync()

                                        # main inner mma loop
                                        for ko in T.serial(PIPE_CIRCLE_NUM):
                                            for ks in T.unroll(SMEM_PIPE_DEPTH):
                                            
                                                # wait tma and sf-transpose arrival
                                                tma2mma_bar.wait(ks, phase[0])
                                                T.ptx.tcgen05.fence.after_thread_sync()

                                                # issue mma                         
                                                for ki in T.unroll(BLK_K // MMA_K):
                                                    T.ptx.tcgen05.encode_matrix_descriptor(T.address_of(descA), A_smem.ptr_to([ks, 0, ki * MMA_K]), 
                                                                                        ldo=1, sdo=8 * BLK_K * F16_BYTES // F128_BYTES, swizzle=SWIZZLE)
                                                    T.ptx.tcgen05.encode_matrix_descriptor(T.address_of(descB), B_smem.ptr_to([ks, 0, ki * MMA_K]), 
                                                                                        ldo=1, sdo=8 * BLK_K * F16_BYTES // F128_BYTES, swizzle=SWIZZLE)          
                                                                
                                                    if ko == 0 and ks == 0 and ki == 0:
                                                        T.ptx.tcgen05.mma("float32", a_type, b_type, tmem_idx * MMA_N, descA, descB, descI, False, CTA_GROUP, False)
                                                    else:
                                                        T.ptx.tcgen05.mma("float32", a_type, b_type, tmem_idx * MMA_N, descA, descB, descI, False, CTA_GROUP, True)
                                                mma2tma_bar.arrive(ks)
                                            phase[0] = phase[0] ^ 1

                                        if PIPE_REMAIN_NUM > 0:
                                            # last remained loop
                                            for ks in T.unroll(PIPE_REMAIN_NUM):

                                                # wait tma and sf-transpose arrival
                                                tma2mma_bar.wait(ks, phase[0])
                                                T.ptx.tcgen05.fence.after_thread_sync()

                                                # issue mma                        
                                                for ki in T.unroll(BLK_K // MMA_K):
                                                    T.ptx.tcgen05.encode_matrix_descriptor(T.address_of(descA), A_smem.ptr_to([ks, 0, ki * MMA_K]), 
                                                                                        ldo=1, sdo=8 * BLK_K * F16_BYTES // F128_BYTES, swizzle=SWIZZLE)
                                                    T.ptx.tcgen05.encode_matrix_descriptor(T.address_of(descB), B_smem.ptr_to([ks, 0, ki * MMA_K]), 
                                                                                        ldo=1, sdo=8 * BLK_K * F16_BYTES // F128_BYTES, swizzle=SWIZZLE)                            
                                                    if PIPE_CIRCLE_NUM == 0 and ks == 0 and ki == 0:
                                                        T.ptx.tcgen05.mma("float32", a_type, b_type, tmem_idx * MMA_N, descA, descB, descI, False, CTA_GROUP, True)
                                                    else:
                                                        T.ptx.tcgen05.mma("float32", a_type, b_type, tmem_idx * MMA_N, descA, descB, descI, False, CTA_GROUP, True)
                                                mma2tma_bar.arrive(ks)

                                            # ensure that all mma is issued
                                            mma2ld_bar.arrive(tmem_idx)

                                            # for unaligned cases   
                                            for ks in T.unroll(PIPE_REMAIN_NUM, SMEM_PIPE_DEPTH):
                                                tma2mma_bar.wait(ks, phase[0])
                                                mma2tma_bar.arrive(ks)
                                            
                                            phase[0] = phase[0] ^ 1
                                        else:
                                            # ensure that all mma is issued
                                            mma2ld_bar.arrive(tmem_idx)

                                    tile_scheduler.next_tile()
                                    
                    with T.warpgroup()[0:1]:
                        trap_when_assert_failed(tmem_addr == 0)
                        tmem_idx = T.local_cell("int32", "tmem_idx")
                        tmem_phase = T.local_cell("int32", "tmem_phase")
                        phase[0] = 0
                        while tile_scheduler.valid():
                            m_idx = T.meta_var(tile_scheduler.m_idx)
                            n_idx = T.meta_var(tile_scheduler.n_idx)
                            tmem_idx = tile_scheduler.tile_idx % TMEM_PIPE_DEPTH
                            tmem_phase = (tile_scheduler.tile_idx // TMEM_PIPE_DEPTH) & 1

                            # flush previous tma
                            if lane_id == 0 and warp_id == 0:
                                T.ptx.cp_async.bulk.wait_group(0)
                            T.ptx.bar.sync(10, 128)
                            # wait for the completion of all the mma of the same tile
                            mma2ld_bar.wait(tmem_idx, tmem_phase)
                            T.ptx.tcgen05.fence.after_thread_sync()
                            
                            for ko in T.unroll(MMA_N // EPI_TILE):
                                stage = (tile_scheduler.tile_idx * MMA_N // EPI_TILE + ko) % TMEM_PIPE_DEPTH
                                
                                # wait the smem to be free
                                if ko >= TMEM_PIPE_DEPTH:
                                    if lane_id == 0 and warp_id == 0:
                                        T.ptx.cp_async.bulk.wait_group(TMEM_PIPE_DEPTH - 1)
                                    T.ptx.bar.sync(10, 128)

                                # tmem -> rf (ld) -> smem
                                for ki in T.unroll(EPI_TILE // TMEM_LD_SIZE):
                                    T.ptx.tcgen05.ld(0 + tmem_idx * MMA_N + ko * EPI_TILE, 
                                                     warp_id * 32, ki * TMEM_LD_SIZE, "32x32b", 
                                                     TMEM_LD_SIZE, False, *[reg[j] for j in range(TMEM_LD_SIZE)])
                                    T.ptx.tcgen05.wait.ld()
                                    
                                    for vec in range(TMEM_LD_SIZE // 2):
                                        float22half2(T.address_of(reg_fp16[vec * 2]), T.address_of(reg[vec * 2]))
                                    for vec in T.vectorized(TMEM_LD_SIZE):
                                        D_smem[stage, warp_id * 32 + lane_id, ki * TMEM_LD_SIZE + vec] = reg_fp16[vec]
                            
                                # the tmem can be overwritten
                                if ko == MMA_N // EPI_TILE - 1:
                                    T.ptx.tcgen05.fence.before_thread_sync()
                                    ld2mma_bar.arrive(tmem_idx)

                                T.ptx.fence.proxy(scope="shared")
                                T.ptx.bar.sync(10, 128)
                                    
                                # smem -> gmem
                                if lane_id == 0 and warp_id == 0:
                                    T.ptx.cp_async.bulk.tensor.s2g(2, D_smem.ptr_to([stage, 0, 0]), 
                                                                D_tensor_map, n_idx * CTA_GROUP * BLK_N + ko * EPI_TILE,
                                                                (m_idx * CTA_GROUP + cbx) * BLK_M)
                                    T.ptx.cp_async.bulk.commit_group()

                            tile_scheduler.next_tile()

                        if lane_id == 0 and warp_id == 0:
                            T.ptx.cp_async.bulk.wait_group(0)
                        T.ptx.bar.sync(10, 128)
                                
                # dealloc TMEM
                with T.warp()[0:1]:
                    T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
                    T.ptx.tcgen05.dealloc(tmem_addr, n_cols=N_COLS, cta_group=1)

                T.ptx.barrier.cluster.arrive()
                T.ptx.barrier.cluster.wait()
    # fmt: on

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
