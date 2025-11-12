from enum import Enum
from typing import List, Literal, Optional, Tuple, Type, Union
import numpy as np

import tvm
from tvm.script import tir as T
from tvm.script import tirp as Tp
from tvm.tir.expr import Var
from tvm.tir import PrimExpr

def ceildiv(a, b):
    if isinstance(a, PrimExpr) or isinstance(b, PrimExpr):
        return T.truncdiv(a + b - 1, b)
    return (a + b - 1) // b


class JobType(Enum):
    V_REDUCE_APPEND = 0
    K_REDUCE_RMS_ROPE_APPEND = 1
    Q_REDUCE_RMS_ROPE = 2
    BATCH_ATTENTION = 3
    BATCH_ATTENTION_MERGE = 4
    GATE_UP_PROJ_REDUCE = 5
    DOWN_PROJ_ALLREDUCE = 6
    O_ALLREDUCE = 7
    ATTN_ADD_RMS_NORM = 8
    GEMM_O_REDUCE = 9
    GEMM_O_PROJ = 10
    GEMM_QKV_PROJ = 11
    MLP_ADD_RMS_NORM = 12
    DOWN_PROJ_REDUCE = 13
    GEMM_DOWN_PROJ = 14
    SPLIT_SILU_MULTIPLY = 15
    GEMM_GATE_UP_PROJ = 16
    GATE_UP_SILU = 17
    MOE_GATING = 18
    MOE_TOPK_SOFTMAX = 19
    MOE_ALIGN = 20
    MOE_COUNT_AND_SORT = 21
    MOE_GROUP_GEMM_GATE_UP = 22
    MOE_SILU_MULTIPLY = 23
    MOE_GROUP_GEMM_DOWN = 24
    MOE_TOPK_REDUCE = 25
    MOE_GROUP_GEMM_GATE_UP_SILU = 26

    # end
    END = 31


# task_type: [0:5], m_idx: [5:18], n_idx: [18:28], k_idx: [28:32]
MAX_TASK_TYPE = 1 << 5
MAX_M_IDX = 1 << 13
MAX_N_IDX = 1 << 10
MAX_K_IDX = 1 << 4
def pack_into_32bit(m_idx, n_idx, k_idx, task_type, host=True, debug=False):
    if host:
        if debug:
            assert task_type < MAX_TASK_TYPE and m_idx < MAX_M_IDX and n_idx < MAX_N_IDX and k_idx < MAX_K_IDX
        return np.int64([task_type | (m_idx << 5) | (n_idx << 18) | (k_idx << 28)]).astype(np.int32)
    else:
        if debug:
            trap_when_assert_failed(task_type < MAX_TASK_TYPE)
            trap_when_assert_failed(m_idx < MAX_M_IDX)
            trap_when_assert_failed(n_idx < MAX_N_IDX)
            trap_when_assert_failed(k_idx < MAX_K_IDX)
        return task_type | (m_idx << 5) | (n_idx << 18) | (k_idx << 28)

unpack_from_32bit_code = """
__forceinline__ __device__ void unpack_from_32bit(int32_t task_info, int32_t* task_type_ptr, int32_t* m_idx_ptr, int32_t* n_idx_ptr, int32_t* k_idx_ptr) {
    *task_type_ptr = task_info & 0b11111;
    *m_idx_ptr = (task_info >> 5) & 0b1111111111111;
    *n_idx_ptr = (task_info >> 18) & 0b1111111111;
    *k_idx_ptr = (task_info >> 28) & 0b1111;
}
"""


@T.macro
def unpack_from_32bit(task_info, task_type_ptr, m_idx_ptr, n_idx_ptr, k_idx_ptr):
    T.cuda.func_call(
        "unpack_from_32bit",
        task_info,
        task_type_ptr,
        m_idx_ptr,
        n_idx_ptr,
        k_idx_ptr,
        source_code=unpack_from_32bit_code,
    )



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


def is_power_of_two(n: T.int32):
    return tvm.tir.all(n > 0, T.bitwise_and(n, n - 1) == 0)


def find_power_of_two(n):
    assert n > 0 and (n & (n - 1)) == 0
    return n.bit_length() - 1


class Tile:
    need_init = True

    @classmethod
    def class_init(cls, smem_manager):
        pass

    @classmethod
    def class_finalize(cls):
        pass

    @classmethod
    def __str__(cls):
        return cls.__name__

    def __init__(self):
        self._instance_id = id(self)

    def __str__(self):
        class_name = self.__class__.__name__
        return f"{class_name}-{self._instance_id:x}"

    def init(self, smem_manager):
        pass

    def host_init(self):
        pass

    def run(self, m_idx, n_idx, k_idx):
        raise NotImplementedError("run is not implemented")


class Barriers:

    def __init__(self, smem_manager, pipe_depth, is_p2c, persistent=True):
        self.mbar = smem_manager.alloc((pipe_depth,), "uint64", method="persistent" if persistent else "shared", name="mbarrier")
        self.init_phase = 0 if is_p2c else 1
        self.pipe_depth = pipe_depth

    @T.macro
    def init(self, threads_num_wait):
        if self.pipe_depth == 1:
            with T.thread()[0:1]:
                T.ptx.mbarrier.init(self.mbar.ptr_to([0]), threads_num_wait)
        else:
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



def rsqrt(x):
    return T.cuda.func_call(
        "_rsqrt",
        x,
        source_code=f"""
__forceinline__ __device__ float _rsqrt(float x) {{
  float y;
  asm volatile("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}}
""",
        return_type="float32",
    )


def exp2(x):
    return T.cuda.func_call(
        "ptx_exp2",
        x,
        source_code=f"""
    __forceinline__ __device__ float ptx_exp2(float x) {{
  float y;
  asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}}
""",
        return_type="float32",
    )


def silu(x):
    return T.cuda.func_call(
        "silu",
        x,
        source_code=f"""
    __forceinline__ __device__ float silu(float x) {{
  return x / (1.0f + __expf(-x));
}}
""",
        return_type="float32",
    )

def threadfence_block():
    return T.cuda.func_call(
        "threadfence_block", source_code="""
__forceinline__ __device__ void threadfence_block() {
  __threadfence_block();
}
""")

def any_sync(mask, pred):
    return T.cuda.func_call(
        "any_sync", mask, pred, source_code=f"""
__forceinline__ __device__ int any_sync(unsigned mask, int pred) {{
  return __any_sync(mask, pred);
}}
""", return_type="int32"
    )

def syncthreads_or(pred):
    return T.cuda.func_call(
        "syncthreads_or", pred, source_code=f"""
__forceinline__ __device__ int syncthreads_or(int pred) {{
  return __syncthreads_or(pred);
}}
""", return_type="int32"
    )

def any_sync(mask, pred):
    return T.cuda.func_call(
        "any_sync", mask, pred, source_code=f"""
__forceinline__ __device__ int any_sync(unsigned mask, int pred) {{
  return __any_sync(mask, pred);
}}
""", return_type="int32"
    )

@T.macro
def block_fence():
    T.cuda.func_call(
        "block_fence",
        source_code=f"""
__forceinline__ __device__ void block_fence() {{
  __threadfence_block();
}}
""")

@T.macro
def grid_sync():
    cta_id = T.cluster_id([KernelConfig.SM_NUMBER], parent="kernel")
    T.cuda.func_call(
        "grid_sync",
        source_code=f"""
__forceinline__ __device__ void grid_sync() {{
  auto g = cooperative_groups::this_thread_block();
  g.sync();
}}
""")


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

def get_source(module: "tvm.ir.IRModule"):
    target = tvm.target.Target("cuda")
    lib = tvm.compile(module, target, tir_pipeline="tirp")
    src = lib.mod.imports[0].inspect_source()
    return src, lib

def get_source_func(func: "tvm.tir.PrimFunc"):
    target = tvm.target.Target("cuda")
    mod = tvm.IRModule({"main": func})
    mod = tvm.compile(mod, target=target, tir_pipeline="tirp")
    src = mod.mod.imports[0].inspect_source()
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
    O_ALLREDUCE = 21
    DOWN_PROJ_ALLREDUCE = 22
    GATE_UP_PROJ_REDUCE = 23
    BATCH_ATTENTION = 24
    BATCH_ATTENTION_MERGE = 25
    PREFETCH = 26
    TMA = 27
    MMA = 28
    ATTN_INIT = 29
    ATTN_LOAD_Q = 30
    ATTN_LOOP_BODY = 31
    ATTN_COMPUTE_QKV = 32
    ATTN_WRITE_BACK = 33
    Q_REDUCE_RMSNORM_ROPE = 34
    K_REDUCE_RMSNORM_ROPE_APPEND = 35
    V_REDUCE_APPEND = 36
    GATE_UP_SILU = 37
    MOE_GATING = 38
    TOPK_SOFTMAX = 39
    MOE_ALIGN = 40
    COUNT_AND_SORT = 41
    GROUP_GEMM_GATE_UP = 42
    SILU_MUL = 43
    GROUP_GEMM_DOWN = 44
    TOPK_REDUCE = 45
    EP_DISPATCH_PRECOMPUTE = 46
    EP_DISPATCH_SEND = 47
    EP_DISPATCH_RECV = 48
    EP_COMBINE_SEND = 49
    EP_COMBINE_RECV = 50
    GROUP_GEMM_GATE_UP_SILU = 51
    END = 52

map_job_type_to_profile_event_type = {
    JobType.GEMM_GATE_UP_PROJ.value: ProfileEventType.GEMM_GATE_UP_PROJ,
    JobType.SPLIT_SILU_MULTIPLY.value: ProfileEventType.SPLIT_SILU_MULTIPLY,
    JobType.GEMM_DOWN_PROJ.value: ProfileEventType.GEMM_DOWN_PROJ,
    JobType.DOWN_PROJ_REDUCE.value: ProfileEventType.DOWN_PROJ_REDUCE,
    JobType.MLP_ADD_RMS_NORM.value: ProfileEventType.MLP_ADD_RMS_NORM,
    JobType.GEMM_QKV_PROJ.value: ProfileEventType.GEMM_QKV_PROJ,
    JobType.GEMM_O_PROJ.value: ProfileEventType.GEMM_O_PROJ,
    JobType.GEMM_O_REDUCE.value: ProfileEventType.GEMM_O_REDUCE,
    JobType.ATTN_ADD_RMS_NORM.value: ProfileEventType.ATTN_ADD_RMS_NORM,
    JobType.O_ALLREDUCE.value: ProfileEventType.O_ALLREDUCE,
    JobType.DOWN_PROJ_ALLREDUCE.value: ProfileEventType.DOWN_PROJ_ALLREDUCE,
    JobType.GATE_UP_PROJ_REDUCE.value: ProfileEventType.GATE_UP_PROJ_REDUCE,
    JobType.BATCH_ATTENTION.value: ProfileEventType.BATCH_ATTENTION,
    JobType.BATCH_ATTENTION_MERGE.value: ProfileEventType.BATCH_ATTENTION_MERGE,
    JobType.Q_REDUCE_RMS_ROPE.value: ProfileEventType.Q_REDUCE_RMSNORM_ROPE,
    JobType.K_REDUCE_RMS_ROPE_APPEND.value: ProfileEventType.K_REDUCE_RMSNORM_ROPE_APPEND,
    JobType.V_REDUCE_APPEND.value: ProfileEventType.V_REDUCE_APPEND,
    JobType.GATE_UP_SILU.value: ProfileEventType.GATE_UP_SILU,
    JobType.END.value: ProfileEventType.END,
}

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
    "O_ALLREDUCE",
    "DOWN_PROJ_ALLREDUCE",
    "GATE_UP_PROJ_REDUCE",
    "BATCH_ATTENTION",
    "BATCH_ATTENTION_MERGE",
    "PREFETCH",
    "TMA",
    "MMA",
    "ATTN_INIT",
    "ATTN_LOAD_Q",
    "ATTN_LOOP_BODY",
    "ATTN_COMPUTE_QKV",
    "ATTN_WRITE_BACK",
    "Q_REDUCE_RMSNORM_ROPE",
    "K_REDUCE_RMSNORM_ROPE_APPEND",
    "V_REDUCE_APPEND",
    "GATE_UP_SILU",
    "MOE_GATING",
    "TOPK_SOFTMAX",
    "MOE_ALIGN",
    "COUNT_AND_SORT",
    "GROUP_GEMM_GATE_UP",
    "SILU_MUL",
    "GROUP_GEMM_DOWN",
    "TOPK_REDUCE",
    "EP_DISPATCH_PRECOMPUTE",
    "EP_DISPATCH_SEND",
    "EP_DISPATCH_RECV",
    "EP_COMBINE_SEND",
    "EP_COMBINE_RECV",
    "GROUP_GEMM_GATE_UP_SILU",
]


class SmemManager:
    def __init__(self, smem_max_bytes, chunk_size, ptr: Var, fusion_mode=False):
        self.smem_max_bytes = smem_max_bytes
        self.chunk_size = chunk_size
        self.chunk_num = smem_max_bytes // chunk_size
        assert self.chunk_num <= 32
        self.ptr = ptr
        self.reguler_pool_allocator = Tp.PoolAllocator(ptr)
        self.persistent_pool_allocator = Tp.PoolAllocator(None if fusion_mode else ptr)
        self.tiles = {} # tile id -> [max used chunk id for the tile, {list of exclusive/other buf}, [arrival count for each chunk]]
        self.runtime_tile_chunk_count = {}
        self.bufs = {} # buf -> (split, beg, size, method)
        self.persistent_bufs = {} # persistent buf -> (beg, end)
        self.cur_tile_name = ""
        self.persistent_pool_allocator.move_base_to(self.chunk_size * self.chunk_num)
        self.exist_bufs = {}
        self.fusion_mode = fusion_mode
        self.pool_allocator = {"persistent": self.persistent_pool_allocator, "shared": self.reguler_pool_allocator, "exclusive": self.reguler_pool_allocator}

    def _inner_alloc(self):
        # notes: these smem will never be overwritten by any tasks
        self.mbar = self.alloc((self.chunk_num,), "uint64", name="mbar", method="persistent")
        self.shared_count = self.alloc((1,), "int32", name="shared_count", method="persistent")
        if self.fusion_mode:
            self.cur_phase = T.alloc_local([1], "int32", scope="local.persistent", name="cur_phase")
            self.reg_count = T.alloc_local([1], "int32", scope="local.persistent", name="reg_count")
        else:
            self.cur_phase = T.alloc_local([1], "int32", name="cur_phase")
            self.reg_count = T.alloc_local([1], "int32", name="reg_count")
       
        
    @T.macro
    def init(self):
        self.check_smem_well_formed(debug=False)
        self._inner_alloc()
        self.cur_phase[0] = 1
        with T.thread()[0:1]:
            for i in T.serial(self.chunk_num):
                T.ptx.mbarrier.init(self.mbar.ptr_to([i]), 1)
            self.shared_count[0] = 0
        T.tvm_storage_sync("shared")
        T.ptx.fence.mbarrier_init()
        T.ptx.fence.proxy("shared")

    # wrapper for pool allocator
    # method: "shared" -> wait_all / arrive_all
    #         "exclusive" -> wait_specific / arrive_specific + wait_unused / arrive_unused, buffer will be exclusive on corresponding pages
    #         "shared" -> wait_specific / arrive_specific + wait_unused / arrive_unused, buffer will share the corresponding pages
    #         "persistent" -> persistent smem, cannot be wait / arrive
    def alloc(self, shape, dtype="float32", strides=None, scope="shared.dyn", align=0, buffer_type="",
              axis_separators=None, layout="default", split=1, name=None, method: Literal["shared", "exclusive", "persistent"]="shared"):
        # avoid name conflict
        if name is not None:
            if name in self.exist_bufs:
                self.exist_bufs[name] += 1
                name = name + str(self.exist_bufs[name] - 1)
            else:
                self.exist_bufs[name] = 1
        assert "shared" in scope
        pool_allocator = self.pool_allocator[method]
        beg = pool_allocator.offset
        if align > 0:
            beg = (beg + align - 1) // align * align
        if self.fusion_mode and method == "persistent":
            scope = "shared.persistent"
        buf = pool_allocator.alloc(shape, dtype, strides, scope, align, buffer_type,
                                    axis_separators, layout, name)
        end = pool_allocator.offset
        size = end - beg
        assert size % split == 0
        if method == "persistent":
            self.persistent_bufs[buf] = (beg, end)
        else:
            # check the validity of the method
            if method == "shared":
                assert len(self.tiles[self.cur_tile_name][1]["exclusive"]) == 0, "Cannot use both shared and shared/exclusive methods at the same time"
            elif method == "exclusive":
                assert len(self.tiles[self.cur_tile_name][1]["shared"]) == 0, "Cannot use both shared and shared/exclusive methods at the same time"
            # allocation info
            buf_info = (split, beg, size, method)
            self.tiles[self.cur_tile_name][0] = max(self.tiles[self.cur_tile_name][0], (end - 1) // self.chunk_size)
            self.tiles[self.cur_tile_name][1][method].append(buf_info)
            self.bufs[buf] = buf_info
            if method == "exclusive":
                for split_idx in range(split):
                    beg_chunk_id = (beg + size // split * split_idx) // self.chunk_size
                    end_chunk_id = (beg + size // split * (split_idx + 1) - 1) // self.chunk_size
                    for chunk_id in range(beg_chunk_id, end_chunk_id + 1):
                        self.tiles[self.cur_tile_name][2][chunk_id] += 1
        return buf

    # call before kernel compilation
    def check_smem_well_formed(self, debug=False):
        for _, buf_info_dict, _ in self.tiles.values():
            # check the exclusive smem and confirm no overlap in each tile
            checked_exclusive = []
            check_overlap = []
            for method in ["shared", "exclusive"]:
                for (split, beg, size, _) in buf_info_dict[method]:
                    end = beg + size
                    # check the max smem size
                    assert end <= self.chunk_num * self.chunk_size
                    # confirm no overlap in each tile
                    for beg_other, end_other in check_overlap:
                        assert beg >= end_other or beg_other >= end, "Overlap detected in smem allocation"
                    check_overlap.append((beg, end))
                    # check the exclusive smem
                    if method == "exclusive":
                        for split_idx in range(split):
                            beg_chunk_id = (beg + size // split * split_idx) // self.chunk_size
                            end_chunk_id = (beg + size // split * (split_idx + 1) - 1) // self.chunk_size
                            for (beg_id, end_id) in checked_exclusive:
                                assert beg_id > end_chunk_id or end_id < beg_chunk_id, "Exclusive chunk overlap detected"
                            checked_exclusive.append((beg_chunk_id, end_chunk_id))
                    else:
                        beg_chunk_id = beg // self.chunk_size
                        end_chunk_id = (end - 1) // self.chunk_size
                        checked_exclusive.append((beg_chunk_id, end_chunk_id))

        # confirm that no overlap between persistent smem and other smem
        for (beg_persistent, end_persistent) in self.persistent_bufs.values():
            assert beg_persistent >= self.chunk_size * self.chunk_num and end_persistent <= self.smem_max_bytes
            for (_, beg, size, _) in self.bufs.values():
                assert beg >= end_persistent or beg_persistent >= beg + size
        persistent_buf_list = list(self.persistent_bufs.values())
        for i in range(len(persistent_buf_list)):
            beg_i, end_i = persistent_buf_list[i]
            for j in range(i + 1, len(persistent_buf_list)):
                beg_j, end_j = persistent_buf_list[j]
                assert beg_i >= end_j or beg_j >= end_i
        if debug:
            self._debug_print()

    def _debug_print(self):
        for k, v in self.tiles.items():
            print(k, v)
        for k, v in self.bufs.items():
            print(k, v)
        for k, v in self.persistent_bufs.items():
            print(k, v)

    # call before each op-kernel compilation
    def set_tile(self, cur_tile: Optional[Tile]):
        if cur_tile is None:
            self.cur_tile_name = "default"
        else:
            self.cur_tile_name = str(cur_tile)
        self.tiles[self.cur_tile_name] = [-1, {"exclusive": [], "shared": []}, [0 for _ in range(self.chunk_num)]]
        self.runtime_tile_chunk_count[self.cur_tile_name] = [[0 for _ in range(self.chunk_num)] for _ in range(2)] # wait count, arrival count
        self.reguler_pool_allocator.move_base_to(0)

    def _assert_cond(self, cond):
        assert cond

    @T.macro
    def advance(self):
        self.cur_phase[0] = self.cur_phase[0] ^ 1

    def enter_tile_runtime(self, cur_tile: Tile):
        self.cur_tile_name = str(cur_tile)

    def exit_tile_runtime(self):
        self._check_runtime()
        self.cur_tile_name = ""

    def _check_runtime(self):
        pass # TODO: not support now

    # wait all the chunks, call at the beginning of the task, cta level interface
    @T.macro
    def wait_all(self, level: Literal["cta", "warpgroup"] = "cta"):
        # self._assert_cond(len(self.tiles[self.cur_tile_name][1]["exclusive"]) == 0)
        # wait the mbarrier
        with T.thread():
            lane_id = T.thread_id([32], parent="warp")
            warp_id = T.warp_id([KernelConfig.WARP_NUMBER * KernelConfig.WG_NUMBER], parent="cta")
            wg_id = T.warpgroup_id([KernelConfig.WG_NUMBER], parent="cta")
            if level == "cta":
                if warp_id == 0:
                    if lane_id < self.chunk_num:
                        T.ptx.mbarrier.try_wait(self.mbar.ptr_to([lane_id]), self.cur_phase[0])
                T.tvm_storage_sync("shared")
            elif level == "warpgroup":
                if warp_id % KernelConfig.WARP_NUMBER == 0:
                    if lane_id < self.chunk_num:
                        T.ptx.mbarrier.try_wait(self.mbar.ptr_to([lane_id]), self.cur_phase[0])
                T.ptx.bar.sync(6 + wg_id, 128)

    # wait the specific chunk, call before use the corresponding smem, warp-level interface
    @T.macro
    def wait_specific(self, lane_id, buffer, split_idx: int):
        self._assert_cond(buffer in self.bufs and buffer not in self.persistent_bufs)
        self._assert_cond(self.bufs[buffer][3] == "exclusive") # must be exclusive
        # wait the mbarrier
        beg_chunk_id = (self.bufs[buffer][1] + self.bufs[buffer][2] // self.bufs[buffer][0] * split_idx) // self.chunk_size
        end_chunk_id = (self.bufs[buffer][1] + self.bufs[buffer][2] // self.bufs[buffer][0] * (split_idx + 1) - 1) // self.chunk_size
        if lane_id >= beg_chunk_id and lane_id <= end_chunk_id:
            T.ptx.mbarrier.try_wait(self.mbar.ptr_to([lane_id]), self.cur_phase[0])

    # wait the unused chunk, warp-level interface
    @T.macro
    def wait_unused(self, lane_id, cur_tile: Tile):
        self._assert_cond(len(self.tiles[self.cur_tile_name][1]["shared"]) == 0) # must be exclusive
        # wait the mbarrier
        if lane_id < self.chunk_num and lane_id > self.tiles[str(cur_tile)][0]:
            T.ptx.mbarrier.try_wait(self.mbar.ptr_to([lane_id]), self.cur_phase[0])

    # wait the specific chunk, thread-level interface
    @T.macro
    def wait_chunk(self, chunk_id):
        T.ptx.mbarrier.try_wait(self.mbar.ptr_to([chunk_id]), self.cur_phase[0])

    # wait the specific chunk, call before use the corresponding smem, thread-level interface
    @T.macro
    def wait_specific_one_thread(self, buffer, split_idx: int):
        self._assert_cond(buffer in self.bufs and buffer not in self.persistent_bufs)
        self._assert_cond(self.bufs[buffer][3] == "exclusive") # must be exclusive
        # wait the mbarrier
        beg_chunk_id = (self.bufs[buffer][1] + self.bufs[buffer][2] // self.bufs[buffer][0] * split_idx) // self.chunk_size
        end_chunk_id = (self.bufs[buffer][1] + self.bufs[buffer][2] // self.bufs[buffer][0] * (split_idx + 1) - 1) // self.chunk_size
        for idx in T.serial(0, end_chunk_id - beg_chunk_id + 1):
            T.ptx.mbarrier.try_wait(self.mbar.ptr_to([beg_chunk_id + idx]), self.cur_phase[0])

    # arrive all the chunks, call at the end of the task, cta level interface
    @T.macro
    def arrive_all(self, level: Literal["cta", "warpgroup"] = "cta"):
        # self._assert_cond(len(self.tiles[self.cur_tile_name][1]["exclusive"]) == 0)
        # arrive the mbarrier
        with T.thread():
            lane_id = T.thread_id([32], parent="warp")
            warp_id = T.warp_id([KernelConfig.WARP_NUMBER * KernelConfig.WG_NUMBER], parent="cta")
            wg_id = T.warpgroup_id([KernelConfig.WG_NUMBER], parent="cta")
            if level == "cta":
                T.tvm_storage_sync("shared")
                if warp_id == 0:
                    if lane_id < self.chunk_num:
                        T.ptx.mbarrier.arrive(self.mbar.ptr_to([lane_id]))
            elif level == "warpgroup":
                self.reg_count[0] = 0
                T.ptx.bar.sync(6 + wg_id, 128)
                if warp_id % KernelConfig.WARP_NUMBER == 0:
                    if lane_id == 0:
                        self.reg_count[0] = T.cuda.atomic_add(T.address_of(self.shared_count[0]), 1) + 1
                        if self.reg_count[0] == KernelConfig.WG_NUMBER:
                            T.cuda.atomic_add(T.address_of(self.shared_count[0]), -KernelConfig.WG_NUMBER)
                    self.reg_count[0] = T.tvm_warp_shuffle(0xffffffff, self.reg_count[0], 0, 32, 32)
                    if self.reg_count[0] == KernelConfig.WG_NUMBER:
                        if lane_id < self.chunk_num:
                            T.ptx.mbarrier.arrive(self.mbar.ptr_to([lane_id]))

    # arrive the specific chunk, call after the buffer can ben released, warp level interface
    @T.macro
    def arrive_specific(self, lane_id, buffer, split_idx: int):
        self._assert_cond(buffer in self.bufs and buffer not in self.persistent_bufs)
        self._assert_cond(self.bufs[buffer][3] == "exclusive") # must be exclusive
        # arrive the mbarrier
        beg_chunk_id = (self.bufs[buffer][1] + self.bufs[buffer][2] // self.bufs[buffer][0] * split_idx) // self.chunk_size
        end_chunk_id = (self.bufs[buffer][1] + self.bufs[buffer][2] // self.bufs[buffer][0] * (split_idx + 1) - 1) // self.chunk_size
        if lane_id >= beg_chunk_id and lane_id <= end_chunk_id:
            T.ptx.mbarrier.arrive(self.mbar.ptr_to([lane_id]))

    # arrive the unused chunk, call at the end of the task, warp level interface
    @T.macro
    def arrive_unused(self, lane_id, cur_tile: Tile):
        self._assert_cond(len(self.tiles[self.cur_tile_name][1]["shared"]) == 0) # must be exclusive
        if lane_id < self.chunk_num and lane_id > self.tiles[str(cur_tile)][0]:
            T.ptx.mbarrier.arrive(self.mbar.ptr_to([lane_id]))

    # arrive the specific chunk, thread-level interface
    @T.macro
    def arrive_chunk(self, chunk_id):
        T.ptx.mbarrier.arrive(self.mbar.ptr_to([chunk_id]))
        
class SemaphoreBase:
    """Abstract base class for semaphore."""
    def __init__(self):
        pass
    
    def semaphore_wait(self, *coord, level, mask):
        raise NotImplementedError
    
    def semaphore_notify(self, *coord, rank=-1):
        raise NotImplementedError
        
class TileSchedulerBase:
    """Abstract base class for tile schedulers."""
    MAX_TASKS = 128

    def __init__(self):
        pass
    
    def get_idx_and_task_type(self) -> Tuple[List[PrimExpr], PrimExpr]:
        raise NotImplementedError

    @T.macro
    def init(self):
        raise NotImplementedError

    @T.macro
    def next_tile(self):
        raise NotImplementedError
        
    @T.macro
    def wait(self, evt, *coord, wait_level, mask):
        raise NotImplementedError
            
    @T.macro
    def notify(self, evt, notify_num, func_notify, scope, scope_id):
        raise NotImplementedError
    
    @T.macro
    def pre_notify_and_push(self, evt, notify_num, func_notify, func_trigger, push_level, scope, scope_id):
        # for dynamic scheduler
        pass
    
    def valid(self):
        raise NotImplementedError
