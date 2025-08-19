from enum import Enum

import tvm
from tvm.script import tir as T


def ceildiv(a, b):
    return (a + b - 1) // b


class JobType(Enum):
    GEMM_GATE_UP_PROJ = 0
    SPLIT_SILU_MULTIPLY = 1
    GEMM_DOWN_PROJ = 2
    DOWN_PROJ_REDUCE = 3
    MLP_ADD_RMS_NORM = 4
    GEMM_QKV_PROJ = 5
    GEMM_QKV_REDUCE = 6
    RMSNORM = 7
    ROPE = 8
    APPEND_KV = 9
    BATCH_DECODE_NO_SPLIT = 10
    BATCH_DECODE_SPLIT = 11
    DECODE_MERGE = 12
    GEMM_O_PROJ = 13
    GEMM_O_REDUCE = 14
    ATTN_ADD_RMS_NORM = 15
    K_RMSNORM_ROPE_APPEND_KV = 16
    Q_RMSNORM_ROPE = 17
    V_APPEND_KV = 18
    O_ALLREDUCE = 19
    DOWN_PROJ_ALLREDUCE = 20
    END = 99


class KernelConfig:
    # global constant
    M_CLUSTER = 1
    N_CLUSTER = 1
    WG_NUMBER = 2
    WARP_NUMBER = 4
    NUM_THREADS = (32 * WARP_NUMBER) * WG_NUMBER
    SM_NUMBER = 148
    CTA_GROUP = M_CLUSTER
    MAX_SMEM_SIZE = 232448


F16_BYTES = 2
F32_BYTES = 4
F128_BYTES = 16


def find_power_of_two(n):
    assert n > 0 and (n & (n - 1)) == 0
    return n.bit_length() - 1


class Tile:
    @classmethod
    def class_config_init(cls, config):
        pass

    @classmethod
    def class_init(cls, pool_allocator):
        pass

    @classmethod
    def class_finalize(cls):
        pass

    def init(self, pool_allocator):
        pass

    def host_init(self):
        pass

    def run(self, m_idx, n_idx, k_idx):
        raise NotImplementedError("run is not implemented")


class Barriers:

    def __init__(self, pool_allocator, pipe_depth, is_p2c):
        self.mbar = pool_allocator.alloc((pipe_depth,), "uint64").buffer
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


@T.macro
def half22float2(dst, src):
    T.cuda.func_call(
        "half22float2",
        dst,
        src,
        source_code=f"""
__forceinline__ __device__ void half22float2(void* dst, void* src) {{
    float2* dst_p = (float2*) dst;
    half2* src_p = (half2*) src;
    *dst_p = __half22float2(*src_p);
}}
    """,
    )


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

def rsqrt(x):
    return T.cuda.func_call(
        "_rsqrt", x, source_code=f"""
__forceinline__ __device__ float _rsqrt(float x) {{
  float y;
  asm volatile("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}}
""", return_type="float32"
    )

def exp2(x):
    return T.cuda.func_call(
        "ptx_exp2", x, source_code=f"""
    __forceinline__ __device__ float ptx_exp2(float x) {{
  float y;
  asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}}
""", return_type="float32"
    )

def silu(x):
    return T.cuda.func_call(
        "silu", x, source_code=f"""
    __forceinline__ __device__ float silu(float x) {{
  return x / (1.0f + __expf(-x));
}}
""", return_type="float32"
    )


def get_source(func: "tvm.tir.PrimFunc") -> str:
    target = tvm.target.Target("cuda")
    mod = tvm.IRModule({"main": func})
    mod = tvm.compile(mod, target=target, tir_pipeline="tirp")
    src = mod.mod.imported_modules[0].get_source()
    return src, mod


class ProfileEventType(Enum):
    GEMM_GATE_UP_PROJ = 0
    SPLIT_SILU_MULTIPLY = 1
    GEMM_DOWN_PROJ = 2
    DOWN_PROJ_REDUCE = 3
    MLP_ADD_RMS_NORM = 4
    FETCH = 5
    GEMM_QKV_PROJ = 6
    GEMM_QKV_REDUCE = 7
    RMSNORM = 8
    ROPE = 9
    APPEND_KV = 10
    BATCH_DECODE_NO_SPLIT = 11
    BATCH_DECODE_SPLIT = 12
    DECODE_MERGE = 13
    GEMM_O_PROJ = 14
    GEMM_O_REDUCE = 15
    ATTN_ADD_RMS_NORM = 16
    Q_RMSNORM_ROPE = 17
    K_RMSNORM_ROPE_APPEND_KV = 18
    V_APPEND_KV = 19
    PUSH = 20


event_type_names = [
    "GEMM_GATE_UP_PROJ",
    "SPLIT_SILU_MULTIPLY",
    "GEMM_DOWN_PROJ",
    "DOWN_PROJ_REDUCE",
    "MLP_ADD_RMS_NORM",
    "FETCH",
    "GEMM_QKV_PROJ",
    "GEMM_QKV_REDUCE",
    "RMSNORM",
    "ROPE",
    "APPEND_KV",
    "BATCH_DECODE_NO_SPLIT",
    "BATCH_DECODE_SPLIT",
    "DECODE_MERGE",
    "GEMM_O_PROJ",
    "GEMM_O_REDUCE",
    "ATTN_ADD_RMS_NORM",
    "Q_RMSNORM_ROPE",
    "K_RMSNORM_ROPE_APPEND_KV",
    "V_APPEND_KV",
    "PUSH",
]
