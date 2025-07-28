import ml_dtypes
import numpy as np

import tvm
from tvm.script import tir as T
import tvm.testing
from ..utils import bench

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


class BarTMA2TRANS(Barriers):

    @T.macro
    def arrive(self, idx, expected_bytes):
        T.ptx.mbarrier.arrive.expect_tx(self.mbar.ptr_to([idx]), expected_bytes)

    @T.macro
    def arrive_only(self, idx):
        T.ptx.mbarrier.arrive(self.mbar.ptr_to([idx]))


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
def make_runtime_instr_desc(desc, sf_id):
    T.cuda.func_call(
        "runtime_instr_desc",
        desc,
        sf_id,
        source_code=f"""
__forceinline__ __device__ void runtime_instr_desc(uint32_t* desc, const uint32_t& sf_id) {{
    *desc = (*desc & ~0x60000030) | ((sf_id << 29) | (sf_id << 4));
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
        T.SwizzleLayout(3, 2, 3, swizzle_inner=True),
        T.TileLayout(shard=((TMEM_PIPE_DEPTH, BLK_M, EPI_TILE), (BLK_M * EPI_TILE, EPI_TILE, 1))),
    )

    SFA_layout = T.TileLayout(shard=((SMEM_PIPE_DEPTH, BLK_SFA // 32, 32), (BLK_SFA, 32, 1)))
    SFB_layout = T.TileLayout(shard=((SMEM_PIPE_DEPTH, BLK_SFB // 32, 32), (BLK_SFB, 32, 1)))

    @T.prim_func(tirp=True)
    def deepgemm(
        A: T.Buffer((M, K), a_type, layout="default"),
        B: T.Buffer((N, K), b_type, layout="default"),
        D: T.Buffer((M, N), d_type, layout="default"),
        SFA: T.Buffer((ceildiv(K, QUANT_SIZE) // 4, M), "uint32", layout="default"),
        SFB: T.Buffer((ceildiv(K, QUANT_SIZE) // 4, N), "uint32", layout="default"),
    ):
        # fmt: off
        T.func_attr({"global_symbol": "main"})
        A_tensor_map: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
        B_tensor_map: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
        D_tensor_map: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
        SFA_tensor_map: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
        SFB_tensor_map: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
        T.call_packed("runtime.cuTensorMapEncodeTiled", A_tensor_map, a_type, 2, A.data,
                      K, M, K * F8_BYTES, BLK_K, BLK_M, 1, 1, 0, SWIZZLE, 0, 0)
        T.call_packed("runtime.cuTensorMapEncodeTiled", B_tensor_map, b_type, 2, B.data, 
                      K, N, K * F8_BYTES, BLK_K, BLK_N, 1, 1, 0, SWIZZLE, 0, 0)
        T.call_packed("runtime.cuTensorMapEncodeTiled", D_tensor_map, d_type, 2, D.data,
                      N, M, N * F16_BYTES, EPI_TILE, BLK_M, 1, 1, 0, 2, 0, 0)
        T.call_packed("runtime.cuTensorMapEncodeTiled", SFA_tensor_map, "uint32", 2, SFA.data, 
                      M, ceildiv(K, QUANT_SIZE) // 4, M * F32_BYTES, BLK_M, 1, 1, 1, 0, 0, 0, 0)
        T.call_packed("runtime.cuTensorMapEncodeTiled", SFB_tensor_map, "uint32", 2, SFB.data, 
                      N, ceildiv(K, QUANT_SIZE) // 4, N * F32_BYTES, BLK_N * CTA_GROUP, 1, 1, 1, 0, 0, 0, 0)
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
                                        elem_offset=1024 // F8_BYTES)
                B_smem = T.decl_buffer((SMEM_PIPE_DEPTH, BLK_N, BLK_K), b_type, buf.data, layout=B_layout,
                                        elem_offset=1024 // F8_BYTES + SMEM_PIPE_DEPTH * BLK_M * BLK_K)
                D_smem = T.decl_buffer((TMEM_PIPE_DEPTH, BLK_M, EPI_TILE), d_type, buf.data, layout=D_layout,
                                        elem_offset=1024 // F16_BYTES + SMEM_PIPE_DEPTH * (BLK_M + BLK_N) * BLK_K * F8_BYTES // F16_BYTES)
                SFA_smem = T.decl_buffer((SMEM_PIPE_DEPTH, BLK_SFA // 32, 32), "uint32", buf.data, layout=SFA_layout,
                                        elem_offset=1024 // F32_BYTES + SMEM_PIPE_DEPTH * (BLK_M + BLK_N) * BLK_K * F8_BYTES // F32_BYTES
                                                    + TMEM_PIPE_DEPTH * BLK_M * EPI_TILE * F16_BYTES // F32_BYTES)
                SFB_smem = T.decl_buffer((SMEM_PIPE_DEPTH, BLK_SFB // 32, 32), "uint32", buf.data, layout=SFB_layout,
                                        elem_offset=1024 // F32_BYTES + SMEM_PIPE_DEPTH * (BLK_M + BLK_N) * BLK_K * F8_BYTES // F32_BYTES
                                                    + TMEM_PIPE_DEPTH * BLK_M * EPI_TILE * F16_BYTES // F32_BYTES + SMEM_PIPE_DEPTH * BLK_SFA)
               
                # alloc local memory
                reg = T.alloc_buffer((TMEM_LD_SIZE,), "float32", scope="local")
                reg_fp16 = T.alloc_buffer((BLK_N * CTA_GROUP,), d_type, scope="local")
                stage = T.local_cell("int32")
                phase = T.local_cell("int32")
                descA = T.local_cell("uint64")
                descB = T.local_cell("uint64")
                descSFA = T.local_cell("uint64")
                descSFB = T.local_cell("uint64")
                descI = T.local_cell("uint32")
                
                # initialize
                tma2trans_bar = T.meta_var(BarTMA2TRANS(buf.data, 6, SMEM_PIPE_DEPTH, True))
                trans2mma_bar = T.meta_var(BarTRANS2MMA(buf.data, 6 + SMEM_PIPE_DEPTH, SMEM_PIPE_DEPTH, True))
                mma2tma_bar = T.meta_var(BarMMA2TMA(buf.data, 6 + 2 * SMEM_PIPE_DEPTH, SMEM_PIPE_DEPTH, False))
                mma2ld_bar = T.meta_var(BarMMA2LD(buf.data, 6 + 3 * SMEM_PIPE_DEPTH, TMEM_PIPE_DEPTH, True))
                ld2mma_bar = T.meta_var(BarLD2MMA(buf.data, 6 + 3 * SMEM_PIPE_DEPTH + TMEM_PIPE_DEPTH, TMEM_PIPE_DEPTH, False))
                tile_scheduler = T.meta_var(TileScheduler("tile_scheduler"))
                tma2trans_bar.init(1)
                trans2mma_bar.init(CTA_GROUP * 32)
                mma2ld_bar.init(1)
                mma2tma_bar.init(1)
                ld2mma_bar.init(CTA_GROUP * 128)
                tile_scheduler.init(bx // 2)

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
                        T.block_attr({"tirp.scope_partition": True})
                        with T.warp(parent="warpgroup")[warp_id == 3]:
                            phase = 0
                            while tile_scheduler.valid():
                                m_idx = T.meta_var(tile_scheduler.m_idx)
                                n_idx = T.meta_var(tile_scheduler.n_idx)
                                
                                @T.macro
                                def tma_load(ks):
                                    mma2tma_bar.wait(ks, phase)
                                    with T.thread()[T.ptx.elect_sync()]:
                                        T.ptx.cp_async.bulk.tensor.g2c(2, A_smem.ptr_to([ks, 0, 0]), tma2trans_bar.mbar.ptr_to([ks]),
                                                                    A_tensor_map, stage * BLK_K, (m_idx * CTA_GROUP + cbx) * BLK_M, cta_group=1)
                                        T.ptx.cp_async.bulk.tensor.g2c(2, B_smem.ptr_to([ks, 0, 0]), tma2trans_bar.mbar.ptr_to([ks]),
                                                                    B_tensor_map, stage * BLK_K, (n_idx * CTA_GROUP + cbx) * BLK_N, cta_group=1)
                                        if stage % 4 == 0:
                                            T.ptx.cp_async.bulk.tensor.g2c(2, SFA_smem.ptr_to([ks, 0, 0]),
                                                                        tma2trans_bar.mbar.ptr_to([ks]),
                                                                        SFA_tensor_map, (m_idx * CTA_GROUP + cbx) * BLK_M, stage // 4, cta_group=1)
                                            T.ptx.cp_async.bulk.tensor.g2c(2, SFB_smem.ptr_to([ks, 0, 0]),
                                                                        tma2trans_bar.mbar.ptr_to([ks]),
                                                                        SFB_tensor_map, n_idx * CTA_GROUP * BLK_N, stage // 4, cta_group=1)
                                        
                                    with T.thread()[T.ptx.elect_sync()]: 
                                        # notify the mma stage that tma load is finished
                                        if stage % 4 == 0:
                                            tma2trans_bar.arrive(ks, BLK_K * (BLK_M + BLK_N) * F8_BYTES + (BLK_M + BLK_N * CTA_GROUP) * F32_BYTES)
                                        else:
                                            tma2trans_bar.arrive(ks, BLK_K * (BLK_M + BLK_N) * F8_BYTES)

                                # main inner tma loop
                                for ko in T.serial(PIPE_CIRCLE_NUM):
                                    for ks in T.unroll(SMEM_PIPE_DEPTH):
                                        # GMEM -> SMEM  (tma)    
                                        stage = ko * SMEM_PIPE_DEPTH + ks
                                        tma_load(ks)
                                    phase = phase ^ 1
                                
                                if PIPE_REMAIN_NUM > 0:
                                    # last remained loop
                                    for ks in T.unroll(PIPE_REMAIN_NUM):
                                        # GMEM -> SMEM  (tma)    
                                        stage = PIPE_CIRCLE_NUM * SMEM_PIPE_DEPTH + ks
                                        tma_load(ks)

                                    # for unaligned cases   
                                    for ks in T.unroll(PIPE_REMAIN_NUM, SMEM_PIPE_DEPTH):
                                        mma2tma_bar.wait(ks, phase)
                                        with T.thread()[T.ptx.elect_sync()]:
                                            tma2trans_bar.arrive_only(ks)
                                
                                    phase = phase ^ 1
                                tile_scheduler.next_tile()
                        
                        with T.warp(parent="warpgroup")[warp_id == 2]:
                            # transpose
                            phase = 0
                            reg_trans = T.alloc_buffer((4,), "uint32", scope="local")
                            while tile_scheduler.valid():
                                m_idx = T.meta_var(tile_scheduler.m_idx)
                                n_idx = T.meta_var(tile_scheduler.n_idx)

                                @T.macro
                                def transpose(ks):
                                    # wait for sf has been prepared
                                    tma2trans_bar.wait(ks, phase)
                                    if stage % 4 == 0:
                                        for ki in T.unroll(0, BLK_SFA // 128):
                                            for vec in T.vectorized(4):
                                                reg_trans[vec] = SFA_smem[ks, ki * 4 + vec, lane_id]
                                            warp_sync()
                                            for vec in T.vectorized(4):
                                                SFA_smem[ks, ki * 4 + (4 * lane_id + vec) // 32, (4 * lane_id + vec) % 32] = reg_trans[vec]
                                            warp_sync()
                                        for ki in T.unroll(0, BLK_SFB // 128):
                                            for vec in T.vectorized(4):
                                                reg_trans[vec] = SFB_smem[ks, ki * 4 + vec, lane_id]
                                            warp_sync()
                                            for vec in T.vectorized(4):
                                                SFB_smem[ks, ki * 4 + (4 * lane_id + vec) // 32, (4 * lane_id + vec) % 32] = reg_trans[vec]
                                            warp_sync()
                                        T.ptx.fence.proxy("shared")
                                    # mark that transpose is completed
                                    trans2mma_bar.arrive(ks)

                                # main inner transopse loop
                                for ko in T.serial(PIPE_CIRCLE_NUM):
                                    for ks in T.unroll(SMEM_PIPE_DEPTH):
                                        stage = ko * SMEM_PIPE_DEPTH + ks
                                        transpose(ks)
                                    phase = phase ^ 1
                                
                                if PIPE_REMAIN_NUM > 0:
                                    # last remained loop
                                    for ks in T.unroll(PIPE_REMAIN_NUM):
                                        stage = PIPE_CIRCLE_NUM * SMEM_PIPE_DEPTH + ks
                                        transpose(ks)
                                    
                                    # for unaligned cases   
                                    for ks in T.unroll(PIPE_REMAIN_NUM, SMEM_PIPE_DEPTH):
                                        tma2trans_bar.wait(ks, phase)
                                        trans2mma_bar.arrive(ks)

                                    phase = phase ^ 1
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
                                            make_runtime_instr_desc(T.address_of(descI), stage % 4)
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

                                        # main inner mma loop
                                        for ko in T.serial(PIPE_CIRCLE_NUM):
                                            for ks in T.unroll(SMEM_PIPE_DEPTH):
                                                stage = ko * SMEM_PIPE_DEPTH + ks
                                                mma(ks)
                                            phase = phase ^ 1

                                        if PIPE_REMAIN_NUM > 0:
                                            # last remained loop
                                            for ks in T.unroll(PIPE_REMAIN_NUM):
                                                stage = PIPE_CIRCLE_NUM * SMEM_PIPE_DEPTH + ks
                                                mma(ks)
                                            # ensure that all mma is issued
                                            mma2ld_bar.arrive(tmem_idx)

                                            # for unaligned cases   
                                            for ks in T.unroll(PIPE_REMAIN_NUM, SMEM_PIPE_DEPTH):
                                                trans2mma_bar.wait(ks, phase)
                                                mma2tma_bar.arrive(ks)
                                            
                                            phase = phase ^ 1
                                        else:
                                            # ensure that all mma is issued
                                            mma2ld_bar.arrive(tmem_idx)

                                    tile_scheduler.next_tile()
                                    
                    with T.warpgroup()[0:1]:
                        trap_when_assert_failed(tmem_addr == 0)
                        tmem_idx = T.local_cell("int32", "tmem_idx")
                        tmem_phase = T.local_cell("int32", "tmem_phase")
                        phase = 0
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
                                        float22half2(T.address_of(reg_fp16[ki * TMEM_LD_SIZE + vec * 2]), T.address_of(reg[vec * 2]))
                                    for vec in T.vectorized(TMEM_LD_SIZE):
                                        D_smem[stage, warp_id * 32 + lane_id, ki * TMEM_LD_SIZE + vec] = reg_fp16[ki * TMEM_LD_SIZE + vec]
                            
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

    DEV = tvm.cuda(0)
    A_fp8, B_fp8, A_scale, B_scale, C, C_standard, C_ref = prepare_data()
    A_tvm = tvm.nd.array(A_fp8, device=DEV)
    B_tvm = tvm.nd.array(B_fp8, device=DEV)
    A_scale_tvm = tvm.nd.array(A_scale, device=DEV)
    B_scale_tvm = tvm.nd.array(B_scale, device=DEV)
    C_tvm = tvm.nd.array(C, device=DEV)
    target = tvm.target.Target("cuda")
    with target:
        src, mod = get_source(deepgemm)
        # print(src)
        # mod(A_tvm, B_tvm, C_tvm, A_scale_tvm, B_scale_tvm)
        func = lambda: mod(A_tvm, B_tvm, C_tvm, A_scale_tvm, B_scale_tvm)
        ms = bench(func, warmup=0, repeat=100, proton_name="tir")
        print(f"TIR flops: {flops(ms) / 1e12} TFLOPS, time: {ms:.3f} ms")
        np.testing.assert_allclose(C_tvm.numpy(), C_ref, rtol=1e-3, atol=1e-2)


if __name__ == "__main__":
    test()
