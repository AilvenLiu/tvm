from typing import Literal
import numpy as np

from tvm.script import tir as T
from tvm.tir.op import ceildiv

from .common import Barriers, JobType, KernelConfig, warp_sync, trap_when_assert_failed

while_ld_global_acquire = f"""
__forceinline__ __device__ void while_ld_global_acquire(int32_t* addr, int32_t* task_type, int32_t* m_idx, int32_t* n_idx, int32_t* k_idx) {{
  asm volatile ("ld.global.acquire.gpu.v4.b32 {{%0, %1, %2, %3}}, [%4];\\n" : "=r"(*task_type), "=r"(*m_idx), "=r"(*n_idx), "=r"(*k_idx) : "l"(addr) : "memory");
  while (*task_type < 0) {{
    __nanosleep(800);
    asm volatile ("ld.global.acquire.gpu.v4.b32 {{%0, %1, %2, %3}}, [%4];\\n" : "=r"(*task_type), "=r"(*m_idx), "=r"(*n_idx), "=r"(*k_idx) : "l"(addr) : "memory");
  }}
}}
"""

sts_v4 = """
__forceinline__ __device__ void sts_v4(int32_t v1, int32_t v2, int32_t v3, int32_t v4, void* dst_addr) {
    asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                 :
                 : "l"(dst_addr), "r"(v1), "r"(v2), "r"(v3), "r"(v4)
                 : "memory");
}
"""

stg_v4 = """
__forceinline__ __device__ void stg_v4(int32_t v1, int32_t v2, int32_t v3, int32_t v4, void* dst_addr, int32_t pe) {
    if (pe >= 0) {
        void* ptr = nvshmem_ptr(dst_addr, pe);
        asm volatile("st.global.release.sys.v4.b32 [%0], {%1, %2, %3, %4};"
                     :
                     : "l"(ptr), "r"(v1), "r"(v2), "r"(v3), "r"(v4)
                     : "memory");
    } else {
        asm volatile("st.global.release.gpu.v4.b32 [%0], {%1, %2, %3, %4};"
                     :
                     : "l"(dst_addr), "r"(v1), "r"(v2), "r"(v3), "r"(v4)
                     : "memory");
    }
}
"""

stg_v4_local = """
__forceinline__ __device__ void stg_v4(int32_t v1, int32_t v2, int32_t v3, int32_t v4, void* dst_addr, int32_t pe) {
    asm volatile("st.global.release.gpu.v4.b32 [%0], {%1, %2, %3, %4};"
                 :
                 : "l"(dst_addr), "r"(v1), "r"(v2), "r"(v3), "r"(v4)
                 : "memory");
}
"""


lds_v4 = """
__forceinline__ __device__ void lds_v4(void* src_addr, int32_t* v1, int32_t* v2, int32_t* v3, int32_t* v4) {
    asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(*v1), "=r"(*v2), "=r"(*v3), "=r"(*v4)
                 : "l"(src_addr)
                 : "memory");

}
"""

atomic_add_system = """
__forceinline__ __device__ int32_t atomic_add_system(int32_t* addr, int32_t value, int32_t pe) {
    if (pe >= 0) {
        void* ptr = nvshmem_ptr(addr, pe);
        return atomicAdd((int32_t*)ptr, value);
    }
    else {
        return atomicAdd(addr, value);
    }
}
"""

atomic_add = """
__forceinline__ __device__ int32_t atomic_add_system(int32_t* addr, int32_t value, int32_t pe) {
    return atomicAdd(addr, value);
}
"""


class Semaphore:
    def __init__(self, cnt, buffer, decrement=False, use_nvshmem=False):
        self.cnt = cnt
        self.sem = buffer
        self.state = T.local_cell("int32", name="semaphore_state")
        self.decrement = decrement
        if use_nvshmem:
            self.stg_v4 = stg_v4
            self.atomic_add = atomic_add_system
        else:
            self.stg_v4 = stg_v4_local
            self.atomic_add = atomic_add

    @T.macro
    def semaphore_notify(self, *coord, rank=-1):
        if not self.decrement:
            self.state = (
                T.cuda.func_call(
                    "atomic_add_system",
                    self.sem.ptr_to(coord),
                    1,
                    rank,
                    source_code=self.atomic_add,
                    return_type="int32",
                )
                + 1
            )
        else:
            self.state = (
                T.cuda.func_call(
                    "atomic_add_system",
                    self.sem.ptr_to(coord),
                    -1,
                    rank,
                    source_code=self.atomic_add,
                    return_type="int32",
                )
                - 1
            )

    def is_triggered(self):
        if not self.decrement:
            return self.state == self.cnt
        else:
            return self.state == 0


class SchedulerBarrier(Barriers):
    def __init__(self, smem_manager, is_p2c):
        super().__init__(smem_manager, 1, is_p2c)

    @T.macro
    def arrive(self):
        T.ptx.mbarrier.arrive(self.mbar.ptr_to([0]))


class MPMCQueue:

    def __init__(
        self,
        capacity: int,
        tasks: T.Buffer,
        head: T.Buffer,
        tail: T.Buffer,
        smem_manager,
        use_nvshmem=False,
        debug=False
    ):
        # TODO: we currently assume that the queue is infinitely large.
        if capacity & (capacity - 1):
            raise ValueError("capacity must be a power-of-two")
        self.capacity = capacity
        self.mask = capacity - 1
        self.tasks = tasks  # an array of (task_type, m_idx, n_idx, k_idx)
        self.head = head
        self.tail = tail
        self.head_r = T.local_cell(dtype="int32", name="head_r")
        self.tail_r = T.local_cell(dtype="int32", name="tail_r")
        self.tail_smem = smem_manager.alloc((KernelConfig.WARP_NUMBER * KernelConfig.WG_NUMBER,), "int32", method="persistent").buffer
        self.masked_pos = T.local_cell(dtype="int32", name="masked_pos")
        self.idx = T.local_cell(dtype="int32", name="idx")
        if use_nvshmem:
            self.stg_v4 = stg_v4
            self.atomic_add = atomic_add_system
        else:
            self.stg_v4 = stg_v4_local
            self.atomic_add = atomic_add
        self.debug = debug
            
            
    @T.macro
    def enqueue(self, rank, enqueue_num, func_push, level:Literal["thread", "warp", "warpgroup", "cta"]):
        if level == "thread":
            with T.thread():
                if self.debug:
                    trap_when_assert_failed(enqueue_num == 1) # notes: enqueue_num must be 1
                self.tail_r = T.cuda.func_call(
                    "atomic_add_system",
                    self.tail.access_ptr("rw", offset=self.tail.elem_offset_of([T.int32(0)])),
                    1,
                    rank,
                    source_code=self.atomic_add,
                    return_type="int32",
                )
                self.masked_pos = self.tail_r & self.mask
                task_coord = T.meta_var(func_push(0))
                task_type, m_idx, n_idx, k_idx = T.meta_var(task_coord)
                T.cuda.func_call(
                    "stg_v4",
                    task_type,
                    m_idx,
                    n_idx,
                    k_idx,
                    self.tasks.access_ptr("rw", offset=self.tasks.elem_offset_of([self.masked_pos, 0])),
                    rank,
                    source_code=self.stg_v4,
                )
        else:
            with T.cta():
                lane_id = T.thread_id([32], parent="warp")
                tid_in_wg = T.thread_id([KernelConfig.NUM_THREADS // KernelConfig.WG_NUMBER], parent="warpgroup")
                tid = T.thread_id([KernelConfig.NUM_THREADS], parent="cta")
                wg_id = T.warpgroup_id([KernelConfig.WG_NUMBER], parent="cta")
                warp_id = T.warp_id([KernelConfig.WARP_NUMBER * KernelConfig.WG_NUMBER], parent="cta")
                idx_map = T.meta_var({"warp": (warp_id, lane_id, 32), "warpgroup": (wg_id, tid_in_wg, KernelConfig.NUM_THREADS // KernelConfig.WG_NUMBER),
                                        "cta": (0, tid, KernelConfig.NUM_THREADS)})
                scope_idx, tid_in_scope, tid_stride = idx_map[level]
                if level == "warp":
                    if tid_in_scope == 0:
                        self.tail_r = T.cuda.func_call(
                            "atomic_add_system",
                            self.tail.access_ptr("rw", offset=self.tail.elem_offset_of([T.int32(0)])),
                            enqueue_num,
                            rank,
                            source_code=self.atomic_add,
                            return_type="int32",
                        )
                    self.tail_r = T.tvm_warp_shuffle(0xffffffff, self.tail_r, 0, 32, 32)
                else:
                    if tid_in_scope == 0:
                        self.tail_smem[scope_idx] = T.cuda.func_call(
                            "atomic_add_system",
                            self.tail.access_ptr("rw", offset=self.tail.elem_offset_of([T.int32(0)])),
                            enqueue_num,
                            rank,
                            source_code=self.atomic_add,
                            return_type="int32",
                        )
                    if level == "warpgroup":
                        T.ptx.bar.sync(6 + wg_id, 128)
                    elif level == "cta":
                        T.tvm_storage_sync("shared")
                    self.tail_r = self.tail_smem[scope_idx]
                    
                self.idx = tid_in_scope
                while self.idx < enqueue_num:
                    self.masked_pos = (self.tail_r + self.idx) & self.mask
                    task_coord = T.meta_var(func_push(self.idx))
                    task_type, m_idx, n_idx, k_idx = T.meta_var(task_coord)
                    T.cuda.func_call(
                        "stg_v4",
                        task_type,
                        m_idx,
                        n_idx,
                        k_idx,
                        self.tasks.access_ptr("rw", offset=self.tasks.elem_offset_of([self.masked_pos, 0])),
                        rank,
                        source_code=self.stg_v4,
                    )
                    self.idx += tid_stride

    @T.macro
    def dequeue(
        self,
        fetched_task_type,
        fetched_m_idx,
        fetched_n_idx,
        fetched_k_idx,
        has_prefetched,
    ):
        if not has_prefetched:
            self.head_r = T.cuda.atomic_add(
                self.head.access_ptr("rw", offset=self.head.elem_offset_of([T.int32(0)])), 1
            )
        self.masked_pos = self.head_r & self.mask
        T.cuda.func_call(
            "while_ld_global_acquire",
            self.tasks.access_ptr("r", offset=self.tasks.elem_offset_of([self.masked_pos, 0])),
            T.address_of(fetched_task_type),
            T.address_of(fetched_m_idx),
            T.address_of(fetched_n_idx),
            T.address_of(fetched_k_idx),
            source_code=while_ld_global_acquire,
        )
        # FIXME: enable this when we consider capacity issue
        # self.tasks[self.masked_pos, 0] = -1

    @T.macro
    def prefetch(
        self,
        shared_ptr,
    ):
        self.head_r = T.cuda.atomic_add(
            self.head.access_ptr("rw", offset=self.head.elem_offset_of([T.int32(0)])), 1
        )
        self.masked_pos = self.head_r & self.mask
        T.ptx.cp_async(shared_ptr, self.tasks.ptr_to([self.masked_pos, 0]), 16)
        T.ptx.cp_async.commit_group()


class DynamicTileScheduler:

    MAX_TASKS = 8192
    scheduler_warp = 7

    def __init__(
        self,
        tasks: T.Buffer,
        head: T.Buffer,
        tail: T.Buffer,
        smem_manager,
        use_nvshmem=False,
        debug=False,
    ):
        self.queue = MPMCQueue(
            capacity=self.MAX_TASKS,
            tasks=tasks,
            head=head,
            tail=tail,
            smem_manager=smem_manager,
            use_nvshmem=use_nvshmem,
        )
        self.task_type = T.local_cell(dtype="int32", name="task_type")
        self.m_idx = T.local_cell(dtype="int32", name="m_idx")
        self.n_idx = T.local_cell(dtype="int32", name="n_idx")
        self.k_idx = T.local_cell(dtype="int32", name="k_idx")
        self.packed_value = smem_manager.alloc((4,), "int32", align=16, method="persistent").buffer
        self.dequeue_phase = T.local_cell(dtype="int32", name="dequeue_phase")
        self.p2c_dequeue_barrier = SchedulerBarrier(smem_manager, is_p2c=True)
        self.c2p_dequeue_barrier = SchedulerBarrier(smem_manager, is_p2c=False)
        self.has_prefetched = T.local_cell(dtype="bool", name="has_prefetched")
        self.idx = T.local_cell(dtype="int32", name="idx")
        self.semaphore_state = smem_manager.alloc((KernelConfig.NUM_THREADS,), "int32", method="persistent").buffer
        self.debug = debug

    @T.macro
    def _dequeue_and_store_packed(self):

        self.queue.dequeue(self.task_type, self.m_idx, self.n_idx, self.k_idx, self.has_prefetched)
        T.cuda.func_call(
            "sts_v4",
            self.task_type,
            self.m_idx,
            self.n_idx,
            self.k_idx,
            self.packed_value.ptr_to([0]),
            source_code=sts_v4,
        )

    @T.macro
    def _fetch_from_queue(self):
        with T.cta():
            warp_id = T.warp_id([KernelConfig.WARP_NUMBER * KernelConfig.WG_NUMBER], parent="cta")
            # fetch from GEMM queue
            if warp_id == self.scheduler_warp:
                if T.ptx.elect_sync():
                    if self.has_prefetched:
                        T.ptx.cp_async.wait_group(0)
                        if self.packed_value[0] < 0:
                            self._dequeue_and_store_packed()
                    else:
                        self.c2p_dequeue_barrier.wait(0, self.dequeue_phase)
                        self._dequeue_and_store_packed()
                    self.p2c_dequeue_barrier.arrive()
                self.has_prefetched = 0
            self.p2c_dequeue_barrier.wait(0, self.dequeue_phase)
            T.cuda.func_call(
                "lds_v4",
                self.packed_value.ptr_to([0]),
                T.address_of(self.task_type),
                T.address_of(self.m_idx),
                T.address_of(self.n_idx),
                T.address_of(self.k_idx),
                source_code=lds_v4,
            )
            self.c2p_dequeue_barrier.arrive()
            self.dequeue_phase = self.dequeue_phase ^ 1

    @T.macro
    def init(self):
        self.dequeue_phase = 0
        self.has_prefetched = 0
        with T.thread()[0:1]:
            self.p2c_dequeue_barrier.init(1)
            self.c2p_dequeue_barrier.init(KernelConfig.NUM_THREADS)
        T.tvm_storage_sync("shared")
        T.ptx.fence.proxy("shared")
        T.ptx.fence.mbarrier_init()
        self._fetch_from_queue()

    @T.macro
    def next_tile(self):
        self._fetch_from_queue()

    
    @T.macro
    def push_task(
        self, evt: Semaphore, notify_num, func_trigger, 
        push_level: Literal["thread", "warp", "warpgroup", "cta"],
        scope: Literal["thread", "warp", "warpgroup", "cta"],
        enqueue_thread = 0, enqueue_warp = 0, enqueue_wg = 0
    ):
        # Notes: For push_level = "thread", we assume that the notify threads and the push threads will exactly be the same.
        #        In this way, the syncronization won't be necessary.
        # Notes: Here assume that each thread will notify only at most one time，
        #        and the tids of the threads involved among scope in the notification process start from 0 and increment sequentially.
        # Notes: (rank, num_enqueue, func_push) = func_trigger(trigger_idx), rank=-1 for the local rank
        #        (task_type, m_idx, n_idx, k_idx) = func_push(push_idx)
        
        max_notify_num_map = T.meta_var({"thread": 1, "warp": 32, "warpgroup": KernelConfig.NUM_THREADS // KernelConfig.WG_NUMBER, "cta": KernelConfig.NUM_THREADS})
        
        with T.cta():
            wg_id = T.warpgroup_id([KernelConfig.WG_NUMBER], parent="cta")
            warp_id = T.warp_id([KernelConfig.WARP_NUMBER * KernelConfig.WG_NUMBER], parent="cta")
            tid = T.thread_id([KernelConfig.NUM_THREADS], parent="cta")
            tid_in_wg = T.thread_id([KernelConfig.NUM_THREADS // KernelConfig.WG_NUMBER], parent="warpgroup")
            warp_id_in_wg = T.warp_id([KernelConfig.WARP_NUMBER], parent="warpgroup")
            lane_id = T.thread_id([32], parent="warp")
            idx_in_scope_map = T.meta_var({"thread": {"thread": 0}, "warp": {"thread": lane_id, "warp": 0},
                                            "warpgroup": {"thread": tid_in_wg, "warp": warp_id_in_wg, "warpgroup": 0},
                                            "cta": {"thread": tid, "warp": warp_id, "warpgroup": wg_id, "cta": 0}})
            stride_in_scope_map = T.meta_var({"warp": {"warp": 1}, "warpgroup": {"warp": KernelConfig.WARP_NUMBER, "warpgroup": 1},
                                            "cta": {"warp": KernelConfig.WARP_NUMBER * KernelConfig.WG_NUMBER, "warpgroup": KernelConfig.WG_NUMBER, "cta": 1}})
            if self.debug:
                trap_when_assert_failed(notify_num <= max_notify_num_map[scope])
            if scope == "thread":
                if tid == enqueue_thread:
                    if push_level == "thread": 
                        if evt.is_triggered():
                            rank, num_enqueue, func_push = T.meta_var(func_trigger(0))
                            self.queue.enqueue(rank, num_enqueue, func_push, push_level)
                    else:
                        assert False                        
            elif scope == "warp":
                if warp_id == enqueue_warp:
                    if push_level == "thread":
                        if lane_id < notify_num:
                            if evt.is_triggered():
                                rank, num_enqueue, func_push = T.meta_var(func_trigger(lane_id))
                                self.queue.enqueue(rank, num_enqueue, func_push, push_level)
                    elif push_level == "warp":
                        self.semaphore_state[tid] = evt.state
                        warp_sync()
                        self.idx = idx_in_scope_map[scope][push_level]
                        while self.idx < notify_num:
                            evt.state = self.semaphore_state[enqueue_warp * 32 + self.idx]
                            if evt.is_triggered():
                                rank, num_enqueue, func_push = T.meta_var(func_trigger(self.idx))
                                self.queue.enqueue(rank, num_enqueue, func_push, push_level)
                            self.idx += stride_in_scope_map[scope][push_level]
                    else:
                        assert False
            elif scope == "warpgroup":
                if wg_id == enqueue_wg:
                    if push_level == "thread":
                        if tid_in_wg < notify_num:
                            if evt.is_triggered():
                                rank, num_enqueue, func_push = T.meta_var(func_trigger(tid_in_wg))
                                self.queue.enqueue(rank, num_enqueue, func_push, push_level)
                    elif push_level == "warp" or push_level == "warpgroup":
                        self.semaphore_state[tid] = evt.state
                        T.ptx.bar.sync(6 + wg_id, 128)
                        self.idx = idx_in_scope_map[scope][push_level]
                        while self.idx < notify_num:
                            evt.state = self.semaphore_state[enqueue_wg * KernelConfig.NUM_THREADS // KernelConfig.WG_NUMBER + self.idx]
                            if evt.is_triggered():
                                rank, num_enqueue, func_push = T.meta_var(func_trigger(self.idx))
                                self.queue.enqueue(rank, num_enqueue, func_push, push_level)
                            self.idx += stride_in_scope_map[scope][push_level]
                    else:
                        assert False
            elif scope == "cta":
                if push_level == "thread":
                    if tid < notify_num:
                        if evt.is_triggered():
                            rank, num_enqueue, func_push = T.meta_var(func_trigger(tid))
                            self.queue.enqueue(rank, num_enqueue, func_push, push_level)
                elif push_level == "warp" or push_level == "warpgroup" or push_level == "cta":
                    self.semaphore_state[tid] = evt.state
                    T.tvm_storage_sync("shared")
                    self.idx = idx_in_scope_map[scope][push_level]
                    while self.idx < notify_num:
                        evt.state = self.semaphore_state[self.idx]
                        if evt.is_triggered():
                            rank, num_enqueue, func_push = T.meta_var(func_trigger(self.idx))
                            self.queue.enqueue(rank, num_enqueue, func_push, push_level)  
                        self.idx += stride_in_scope_map[scope][push_level]                       
                else:
                    assert False
            else:
                assert False

    def valid(self):
        return self.task_type != JobType.END.value

    @T.macro
    def prefetch(self):
        if T.ptx.elect_sync():
            self.c2p_dequeue_barrier.wait(0, self.dequeue_phase)
            self.has_prefetched = 1
            self.queue.prefetch(self.packed_value.ptr_to([0]))


class MPMCQueueHost:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tasks = np.full((capacity, 4), -1, dtype=np.int32)
        self.head = np.zeros((1,), dtype=np.int32)
        self.tail = np.zeros((1,), dtype=np.int32)
        self.head[0] = 0
        self.tail[0] = 0

    def enqueue(self, task_type, *task_idx):
        pos = self.tail[0] & (self.capacity - 1)
        self.tasks[pos, 0] = task_type
        self.tasks[pos, 1] = task_idx[0]
        self.tasks[pos, 2] = task_idx[1]
        self.tasks[pos, 3] = task_idx[2]
        self.tail[0] = self.tail[0] + 1
