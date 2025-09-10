from enum import Enum
import threading
from typing import List

import tvm
from tvm.script import tir as T
from tvm.script import tirp as Tp
from tvm.tirp.bench.utils import export_to_perfetto_trace


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
    GATE_UP_PROJ_REDUCE = 21
    BATCH_ATTENTION = 22
    BATCH_ATTENTION_MERGE = 23
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
        self.mbar = smem_manager.alloc((pipe_depth,), "uint64", persistent=persistent).buffer
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

def get_source(module: "tvm.ir.IRModule"):
    target = tvm.target.Target("cuda")
    lib = tvm.compile(module, target, tir_pipeline="tirp")
    src = lib.mod.imports[0].inspect_source()
    return src, lib

def get_source_func(func: "tvm.tir.PrimFunc") -> str:
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
    PREFETCH_SMEM = 26
    TMA = 27
    MMA = 28
    ATTN_INIT = 29  
    ATTN_LOAD_Q = 30
    ATTN_LOOP_BODY = 31
    ATTN_COMPUTE_QKV = 32
    ATTN_WRITE_BACK = 33


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
    "PREFETCH_SMEM",
    "TMA",
    "MMA",
    "ATTN_INIT",
    "ATTN_LOAD_Q",
    "ATTN_LOOP_BODY",
    "ATTN_COMPUTE_QKV",
    "ATTN_WRITE_BACK",
]


class SmemManager:
    def __init__(self, smem_max_bytes, chunk_size, pool_allocator: Tp.PoolAllocator):
        self.smem_max_bytes = smem_max_bytes
        self.chunk_size = chunk_size
        self.chunk_num = ceildiv(smem_max_bytes, chunk_size)
        assert self.chunk_num <= 32
        self.pool_allocator = pool_allocator
        self.tiles = {} # tile id -> (max used chunk id for the tile, list of buf)
        self.bufs = {} # buf -> (split, exclusive, beg, size)
        self.persistent_bufs = {} # persistent buf -> (beg, end)
        self.cur_tile_name = ""

    def _alloc_buffer(self):
        # notes: these smem will never be overwritten by any tasks
        self.mbar = self.pool_allocator.alloc((self.chunk_num,), "uint64").buffer
        self.cur_phase = T.alloc_local([1], "int32", layout="default", name="cur_phase")
        
    @T.macro
    def init(self):
        self.check_smem_well_formed()
        self._alloc_buffer()
        self.cur_phase[0] = 1
        with T.thread()[0:1]:
            for i in T.serial(self.chunk_num):
                T.ptx.mbarrier.init(self.mbar.ptr_to([i]), 1)
        T.tvm_storage_sync("shared")
        T.ptx.fence.mbarrier_init()
        T.ptx.fence.proxy("shared")

    # wrapper for pool allocator
    def alloc(self, shape, dtype="float32", strides=None, scope="global", align=0, buffer_type="",
              axis_separators=None, layout="default", 
              persistent=False, split=1, exclusive=False):
        beg = self.pool_allocator.offset
        if align > 0:
            beg = (beg + align - 1) // align * align
        buf = self.pool_allocator.alloc(shape, dtype, strides, scope, align, buffer_type,
                                        axis_separators, layout)
        end = self.pool_allocator.offset
        size = end - beg
        assert size % split == 0
        if not persistent:
            # allocation info
            buf_info = (split, exclusive, beg, size)
            self.tiles[self.cur_tile_name][0] = max(self.tiles[self.cur_tile_name][0], (end - 1) // self.chunk_size)
            self.tiles[self.cur_tile_name][1].append(buf_info)
            self.bufs[buf.buffer] = buf_info
        else:
            self.persistent_bufs[buf.buffer] = (beg, end)
        return buf
    
    # call before kernel compilation
    def check_smem_well_formed(self):
        for _, buf_info_list in self.tiles.values():
            # check the exclusive smem
            checked_exclusive = []
            for (split, exclusive, beg, size) in buf_info_list:
                if not exclusive:
                    for split_idx in range(split):
                        beg_chunk_id = (beg + size // split * split_idx) // self.chunk_size
                        end_chunk_id = (beg + size // split * (split_idx + 1) - 1) // self.chunk_size
                        checked_exclusive.append((beg_chunk_id, end_chunk_id))
            for (split, exclusive, beg, size) in buf_info_list:
                if exclusive:
                    for split_idx in range(split):
                        beg_chunk_id = (beg + size // split * split_idx) // self.chunk_size
                        end_chunk_id = (beg + size // split * (split_idx + 1) - 1) // self.chunk_size
                        for (beg_id, end_id) in checked_exclusive:
                            assert beg_id > end_chunk_id or end_id < beg_chunk_id
                        checked_exclusive.append((beg_chunk_id, end_chunk_id))
            # confirm no overlap in each tile
            check_overlap = []
            for (_, _, beg, size) in buf_info_list:
                end = beg + size
                for beg_other, end_other in check_overlap:
                    assert beg >= end_other or beg_other >= end
                check_overlap.append((beg, end))
            
        # check the persistent smem
        for (beg_persistent, end_persistent) in self.persistent_bufs.values():
            for (_, _, beg, size) in self.bufs.values():
                assert beg >= end_persistent or beg_persistent >= beg + size 
        persistent_buf_list = list(self.persistent_bufs.values())
        for i in range(len(persistent_buf_list)):
            beg_i, end_i = persistent_buf_list[i]
            for j in range(i + 1, len(persistent_buf_list)):
                beg_j, end_j = persistent_buf_list[j]
                assert beg_i >= end_j or beg_j >= end_i

    # call before each op-kernel compilation
    def set_tile(self, cur_tile: Tile):
        self.cur_tile_name = str(cur_tile)
        self.tiles[self.cur_tile_name] = [-1, []]
        
    def assert_cond(self,cond):
        assert cond

    @T.macro
    def advance(self):
        self.cur_phase[0] = self.cur_phase[0] ^ 1

    # wait all the chunks, warp level interface
    @T.macro
    def wait_all(self, lane_id):
        # wait the mbarrier
        if lane_id < self.chunk_num:
            T.ptx.mbarrier.try_wait(self.mbar.ptr_to([lane_id]), self.cur_phase[0])

    # wait the specific chunk, call before use the corresponding smem, warp-level interface
    @T.macro
    def wait_specific(self, lane_id, buffer, split_idx: int):
        self.assert_cond(buffer in self.bufs and buffer not in self.persistent_bufs)
        self.assert_cond(self.bufs[buffer][1]) # must be exclusive now
        # wait the mbarrier
        beg_chunk_id = (self.bufs[buffer][2] + self.bufs[buffer][3] // self.bufs[buffer][0] * split_idx) // self.chunk_size
        end_chunk_id = (self.bufs[buffer][2] + self.bufs[buffer][3] // self.bufs[buffer][0] * (split_idx + 1) - 1) // self.chunk_size
        if lane_id >= beg_chunk_id and lane_id <= end_chunk_id:
            T.ptx.mbarrier.try_wait(self.mbar.ptr_to([lane_id]), self.cur_phase[0])

    # wait the unused chunk, warp-level interface
    @T.macro
    def wait_unused(self, lane_id, cur_tile: Tile):
        # wait the mbarrier
        if lane_id < self.chunk_num and lane_id > self.tiles[str(cur_tile)][0]:
            T.ptx.mbarrier.try_wait(self.mbar.ptr_to([lane_id]), self.cur_phase[0])

    # wait the specific chunk, call before use the corresponding smem, thread-level interface
    @T.macro
    def wait_specific_one_thread(self, buffer, split_idx: int):
        self.assert_cond(buffer in self.bufs and buffer not in self.persistent_bufs)
        self.assert_cond(self.bufs[buffer][1]) # must be exclusive now
        # wait the mbarrier
        beg_chunk_id = (self.bufs[buffer][2] + self.bufs[buffer][3] // self.bufs[buffer][0] * split_idx) // self.chunk_size
        end_chunk_id = (self.bufs[buffer][2] + self.bufs[buffer][3] // self.bufs[buffer][0] * (split_idx + 1) - 1) // self.chunk_size
        for idx in T.serial(beg_chunk_id, end_chunk_id + 1):
            T.ptx.mbarrier.try_wait(self.mbar.ptr_to([idx]), self.cur_phase[0])

    # arrive all the chunks, call at the end of the task, warp level interface
    @T.macro
    def arrive_all(self, lane_id):
        if lane_id < self.chunk_num:
            T.ptx.mbarrier.arrive(self.mbar.ptr_to([lane_id]))

    # arrive the specific chunk, call after the buffer can ben released, warp level interface
    @T.macro
    def arrive_specific(self, lane_id, buffer, split_idx: int):
        self.assert_cond(buffer in self.bufs and buffer not in self.persistent_bufs)
        self.assert_cond(self.bufs[buffer][1]) # must be exclusive now
        # arrive the mbarrier
        beg_chunk_id = (self.bufs[buffer][2] + self.bufs[buffer][3] // self.bufs[buffer][0] * split_idx) // self.chunk_size
        end_chunk_id = (self.bufs[buffer][2] + self.bufs[buffer][3] // self.bufs[buffer][0] * (split_idx + 1) - 1) // self.chunk_size
        if lane_id >= beg_chunk_id and lane_id <= end_chunk_id:
            T.ptx.mbarrier.arrive(self.mbar.ptr_to([lane_id]))
    
    # arrive the unused chunk, call at the end of the task, warp level interface
    @T.macro
    def arrive_unused(self, lane_id, cur_tile: Tile):
        if lane_id < self.chunk_num and lane_id > self.tiles[str(cur_tile)][0]:
            T.ptx.mbarrier.arrive(self.mbar.ptr_to([lane_id]))


class ProfilerHandler:
    def __init__(self, profiler_on, trigger_count, profiler_layer_id: List[int] = [], dir_path: str = ""):
        self.counter = 0
        self.profiler_on = profiler_on
        self.trigger_count = trigger_count
        self.profiler_layer_id = profiler_layer_id
        self.lock = threading.Lock()
        self.dir_path = dir_path

    def export_trace(self, profiler_buffer):
        if self.profiler_on:
            with self.lock:
                self.counter += 1
                current_run = self.counter    
            if current_run == self.trigger_count:        
                for layer_id in self.profiler_layer_id:
                    export_to_perfetto_trace(
                        profiler_buffer[layer_id].numpy(),
                        f"{self.dir_path}/qwen3-model-mega-layer{layer_id}.perfetto-trace",
                        event_type_names,
                    )
    
    def initialize(self):
        tvm.register_func("megakernel.export_trace", self.export_trace)
