import numpy as np

from tvm.script import tir as T

from .common import Barriers, JobType, KernelConfig

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

shfl_sync = """
__forceinline__ __device__ int32_t shfl_sync(int32_t val, int32_t src_lane) {
    return __shfl_sync(0xFFFFFFFF, val, src_lane);
}
"""

atomic_add_system = """
__forceinline__ __device__ int32_t atomic_add_system(int32_t* addr, int32_t value, int32_t pe) {
    if (pe >= 0) {
        void* ptr = nvshmem_ptr(addr, pe);
        return atomicAdd_system((int32_t*)ptr, value);
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
            self.state = T.cuda.func_call(
                "atomic_add_system",
                self.sem.ptr_to(coord),
                1,
                rank,
                source_code=self.atomic_add,
                return_type="int32",
            ) + 1
        else:
            self.state = T.cuda.func_call(
                "atomic_add_system",
                self.sem.ptr_to(coord),
                -1,
                rank,
                source_code=self.atomic_add,
                return_type="int32",
            ) - 1

    def is_triggered(self):
        if not self.decrement:
            return self.state == self.cnt
        else:
            return self.state == 0


class SchedulerBarrier(Barriers):
    def __init__(self, pool_allocator, is_p2c):
        super().__init__(pool_allocator, 1, is_p2c)

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
        pool_allocator,
        use_nvshmem=False,
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
        self.tail_smem = pool_allocator.alloc((1,), "int32").buffer
        self.masked_pos = T.local_cell(dtype="int32", name="masked_pos")
        self.idx = T.local_cell(dtype="int32", name="idx")
        if use_nvshmem:
            self.stg_v4 = stg_v4
            self.atomic_add = atomic_add_system
        else:
            self.stg_v4 = stg_v4_local
            self.atomic_add = atomic_add

    @T.macro
    def enqueue(self, task_type, m_idx, n_idx, k_idx, rank):

        self.tail_r = T.cuda.func_call(
            "atomic_add_system",
            self.tail.access_ptr("rw", offset=self.tail.offset_of_p([T.int32(0)])),
            1,
            rank,
            source_code=self.atomic_add,
            return_type="int32",
        )
        self.masked_pos = self.tail_r & self.mask
        T.cuda.func_call(
            "stg_v4",
            task_type,
            m_idx,
            n_idx,
            k_idx,
            self.tasks.access_ptr("rw", offset=self.tasks.offset_of_p([self.masked_pos, 0])),
            rank,
            source_code=self.stg_v4,
        )

    @T.macro
    def batch_enqueue_along_dim_warp(
        self, task_type, m_idx, n_idx, k_idx, enqueue_num, extend_dim, lane_id, rank
    ):

        if lane_id == 0:
            self.tail_r = T.cuda.func_call(
                "atomic_add_system",
                self.tail.access_ptr("rw", offset=self.tail.offset_of_p([T.int32(0)])),
                enqueue_num,
                rank,
                source_code=self.atomic_add,
                return_type="int32",
            )
        self.tail_r = T.cuda.func_call(
            "shfl_sync",
            self.tail_r,
            0,
            source_code=shfl_sync,
            return_type="int32",
        )

        self.idx = lane_id
        while self.idx < enqueue_num:
            self.masked_pos = (self.tail_r + self.idx) & self.mask
            if extend_dim == 0:
                T.cuda.func_call(
                    "stg_v4",
                    task_type,
                    m_idx + self.idx,
                    n_idx,
                    k_idx,
                    self.tasks.access_ptr(
                        "rw", offset=self.tasks.offset_of_p([self.masked_pos, 0])
                    ),
                    rank,
                    source_code=self.stg_v4,
                )
            elif extend_dim == 1:
                T.cuda.func_call(
                    "stg_v4",
                    task_type,
                    m_idx,
                    n_idx + self.idx,
                    k_idx,
                    self.tasks.access_ptr(
                        "rw", offset=self.tasks.offset_of_p([self.masked_pos, 0])
                    ),
                    rank,
                    source_code=self.stg_v4,
                )
            elif extend_dim == 2:
                T.cuda.func_call(
                    "stg_v4",
                    task_type,
                    m_idx,
                    n_idx,
                    k_idx + self.idx,
                    self.tasks.access_ptr(
                        "rw", offset=self.tasks.offset_of_p([self.masked_pos, 0])
                    ),
                    rank,
                    source_code=self.stg_v4,
                )
            self.idx += 32

    @T.macro
    def batch_enqueue_along_dim_wg(
        self, task_type, m_idx, n_idx, k_idx, enqueue_num, extend_dim, thread_id_in_wg, rank
    ):

        if thread_id_in_wg == 0:
            self.tail_smem[0] = T.cuda.func_call(
                "atomic_add_system",
                self.tail.access_ptr("rw", offset=self.tail.offset_of_p([T.int32(0)])),
                enqueue_num,
                rank,
                source_code=self.atomic_add,
                return_type="int32",
            )
        T.ptx.bar.sync(13, 128)
        self.tail_r = self.tail_smem[0]
        self.idx = thread_id_in_wg
        while self.idx < enqueue_num:
            self.masked_pos = (self.tail_r + self.idx) & self.mask
            if extend_dim == 0:
                T.cuda.func_call(
                    "stg_v4",
                    task_type,
                    m_idx + self.idx,
                    n_idx,
                    k_idx,
                    self.tasks.access_ptr(
                        "rw", offset=self.tasks.offset_of_p([self.masked_pos, 0])
                    ),
                    rank,
                    source_code=self.stg_v4,
                )
            elif extend_dim == 1:
                T.cuda.func_call(
                    "stg_v4",
                    task_type,
                    m_idx,
                    n_idx + self.idx,
                    k_idx,
                    self.tasks.access_ptr(
                        "rw", offset=self.tasks.offset_of_p([self.masked_pos, 0])
                    ),
                    rank,
                    source_code=self.stg_v4,
                )
            elif extend_dim == 2:
                T.cuda.func_call(
                    "stg_v4",
                    task_type,
                    m_idx,
                    n_idx,
                    k_idx + self.idx,
                    self.tasks.access_ptr(
                        "rw", offset=self.tasks.offset_of_p([self.masked_pos, 0])
                    ),
                    rank,
                    source_code=self.stg_v4,
                )
            self.idx += 128

    @T.macro
    def batch_enqueue_warp(self, task_type, m_idx, n_idx, k_idx, enqueue_num, lane_id, rank):

        if lane_id == 0:
            self.head_r = T.cuda.func_call(
                "atomic_add_system",
                self.head.access_ptr("rw", offset=self.head.offset_of_p([T.int32(0)])),
                enqueue_num,
                rank,
                source_code=self.atomic_add,
                return_type="int32",
            )
        self.tail_r = T.cuda.func_call(
            "shfl_sync",
            self.tail_r,
            0,
            source_code=shfl_sync,
            return_type="int32",
        )
        self.masked_pos = (self.tail_r + lane_id) & self.mask
        T.cuda.func_call(
            "stg_v4",
            task_type,
            m_idx,
            n_idx,
            k_idx,
            self.tasks.access_ptr("rw", offset=self.tasks.offset_of_p([self.masked_pos, 0])),
            rank,
            source_code=self.stg_v4,
        )

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
                self.head.access_ptr("rw", offset=self.head.offset_of_p([T.int32(0)])), 1
            )
        self.masked_pos = self.head_r & self.mask
        T.cuda.func_call(
            "while_ld_global_acquire",
            self.tasks.access_ptr("r", offset=self.tasks.offset_of_p([self.masked_pos, 0])),
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
            self.head.access_ptr("rw", offset=self.head.offset_of_p([T.int32(0)])), 1
        )
        self.masked_pos = self.head_r & self.mask
        T.ptx.cp_async(shared_ptr, self.tasks.ptr_to([self.masked_pos, 0]), 16)
        T.ptx.cp_async.commit_group()


class DynamicTileScheduler:

    MAX_TASKS = 8192
    scheduler_warp = 7
    enqueue_warp = 3
    enqueue_wg = 0

    def __init__(
        self,
        tasks: T.Buffer,
        head: T.Buffer,
        tail: T.Buffer,
        pool_allocator,
        use_nvshmem=False,
    ):
        self.queue = MPMCQueue(
            capacity=self.MAX_TASKS,
            tasks=tasks,
            head=head,
            tail=tail,
            pool_allocator=pool_allocator,
            use_nvshmem=use_nvshmem,
        )
        self.task_type = T.local_cell(dtype="int32", name="task_type")
        self.m_idx = T.local_cell(dtype="int32", name="m_idx")
        self.n_idx = T.local_cell(dtype="int32", name="n_idx")
        self.k_idx = T.local_cell(dtype="int32", name="k_idx")
        self.packed_value = pool_allocator.alloc((4,), "int32", align=16).buffer
        self.dequeue_phase = T.local_cell(dtype="int32", name="dequeue_phase")
        self.enqueue_phase = T.local_cell(dtype="int32", name="enqueue_phase")
        self.p2c_dequeue_barrier = SchedulerBarrier(pool_allocator, is_p2c=True)
        self.c2p_dequeue_barrier = SchedulerBarrier(pool_allocator, is_p2c=False)
        self.p2c_enqueue_barrier = SchedulerBarrier(pool_allocator, is_p2c=True)
        self.c2p_enqueue_barrier = SchedulerBarrier(pool_allocator, is_p2c=False)
        self.has_prefetched = T.local_cell(dtype="bool", name="has_prefetched")
        self.semaphore_state = pool_allocator.alloc((16,), "int32").buffer

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
    def _fetch_from_queue(self, is_schedule_warp):
        # fetch from GEMM queue
        if is_schedule_warp:
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
    def init(self, warp_id):
        self.dequeue_phase = 0
        self.enqueue_phase = 0
        self.has_prefetched = 0
        with T.thread()[0:1]:
            self.p2c_dequeue_barrier.init(1)
            self.c2p_dequeue_barrier.init(KernelConfig.NUM_THREADS)
            self.p2c_enqueue_barrier.init(KernelConfig.NUM_THREADS - 32)
            self.c2p_enqueue_barrier.init(1)
        T.tvm_storage_sync("shared")
        T.ptx.fence.proxy("shared")
        T.ptx.fence.mbarrier_init()
        self._fetch_from_queue(warp_id == self.scheduler_warp)

    @T.macro
    def next_tile(self, warp_id):
        self._fetch_from_queue(warp_id == self.scheduler_warp)

    # block-level operation
    @T.macro
    def push_task(
        self, task_type, m_idx, n_idx, k_idx, warp_id, semaphore, *coord, use_barrier=True, rank=-1
    ):
        if warp_id == self.enqueue_warp:
            if T.ptx.elect_sync():
                if use_barrier:
                    self.p2c_enqueue_barrier.wait(0, self.enqueue_phase)
                    self.c2p_enqueue_barrier.arrive()
                semaphore.semaphore_notify(*coord, rank=rank)
                if semaphore.is_triggered():
                    self.queue.enqueue(task_type, m_idx, n_idx, k_idx, rank)
        elif use_barrier:
            self.c2p_enqueue_barrier.wait(0, self.enqueue_phase)
            self.p2c_enqueue_barrier.arrive()
        if use_barrier:
            self.enqueue_phase = self.enqueue_phase ^ 1

    # block-level operation
    # each lane of enqueue warp can hold different task_type, m_idx, n_idx, k_idx
    @T.macro
    def push_tasks(
        self,
        task_type,
        m_idx,
        n_idx,
        k_idx,
        enqueue_num,
        warp_id,
        lane_id,
        semaphore,
        *coord,
        use_barrier=True,
        rank=-1,
    ):
        # theoretically we can define lane_id here to avoid passing it as an argument
        # but this has bug now
        if warp_id == self.enqueue_warp:
            if T.ptx.elect_sync():
                if use_barrier:
                    self.p2c_enqueue_barrier.wait(0, self.enqueue_phase)
                    self.c2p_enqueue_barrier.arrive()
                semaphore.semaphore_notify(*coord, rank=rank)
            semaphore.state = T.cuda.func_call(
                "shfl_sync",
                semaphore.state,
                0,
                source_code=shfl_sync,
                return_type="int32",
            )
            if semaphore.is_triggered():
                self.queue.batch_enqueue_warp(task_type, m_idx, n_idx, k_idx, enqueue_num, lane_id, rank)
        elif use_barrier:
            self.c2p_enqueue_barrier.wait(0, self.enqueue_phase)
            self.p2c_enqueue_barrier.arrive()
        if use_barrier:
            self.enqueue_phase = self.enqueue_phase ^ 1

    # block-level operation
    # each lane of enqueue warp must hold the same task_type, m_idx, n_idx, k_idx
    @T.macro
    def push_tasks_along_dim_warp(
        self,
        task_type,
        m_idx,
        n_idx,
        k_idx,
        enqueue_num,
        extend_dim,
        warp_id,
        lane_id,
        semaphore,
        *coord,
        use_barrier=True,
        rank=-1,
    ):
        if warp_id == self.enqueue_warp:
            if T.ptx.elect_sync():
                if use_barrier:
                    self.p2c_enqueue_barrier.wait(0, self.enqueue_phase)
                    self.c2p_enqueue_barrier.arrive()
                semaphore.semaphore_notify(*coord, rank=rank)
            semaphore.state = T.cuda.func_call(
                "shfl_sync",
                semaphore.state,
                0,
                source_code=shfl_sync,
                return_type="int32",
            )
            if semaphore.is_triggered():
                self.queue.batch_enqueue_along_dim_warp(
                    task_type, m_idx, n_idx, k_idx, enqueue_num, extend_dim, lane_id, rank
                )
        elif use_barrier:
            self.c2p_enqueue_barrier.wait(0, self.enqueue_phase)
            self.p2c_enqueue_barrier.arrive()
        if use_barrier:
            self.enqueue_phase = self.enqueue_phase ^ 1

    @T.macro
    def push_tasks_along_dim_wg(
        self,
        task_type,
        m_idx,
        n_idx,
        k_idx,
        enqueue_num,
        extend_dim,
        warp_id,
        lane_id,
        semaphore,
        *coord,
        use_barrier=True,
        rank=-1,
    ):
        if warp_id == self.enqueue_warp:
            if T.ptx.elect_sync():
                if use_barrier:
                    self.p2c_enqueue_barrier.wait(0, self.enqueue_phase)
                    self.c2p_enqueue_barrier.arrive()
                semaphore.semaphore_notify(*coord, rank=rank)
                self.semaphore_state[0] = semaphore.state
        elif use_barrier:
            self.c2p_enqueue_barrier.wait(0, self.enqueue_phase)
            self.p2c_enqueue_barrier.arrive()
        if warp_id // 4 == self.enqueue_wg:
            T.ptx.bar.sync(13, 128)
            semaphore.state = self.semaphore_state[0]
            if semaphore.is_triggered():
                self.queue.batch_enqueue_along_dim_wg(
                    task_type, m_idx, n_idx, k_idx, enqueue_num, extend_dim, warp_id * 32 + lane_id, rank
                )
        if use_barrier:
            self.enqueue_phase = self.enqueue_phase ^ 1


    @T.macro
    def push_tasks_along_dim_wg_with_extend_rank(
        self,
        task_type,
        m_idx,
        n_idx,
        k_idx,
        enqueue_num,
        extend_dim,
        warp_id,
        lane_id,
        semaphore,
        get_rank_map,
        rank_range,
        *coord,
        use_barrier=True,
    ):
        if warp_id == self.enqueue_warp:
            if use_barrier:
                if T.ptx.elect_sync():
                    self.p2c_enqueue_barrier.wait(0, self.enqueue_phase)
                    self.c2p_enqueue_barrier.arrive()
            if lane_id < rank_range:
                semaphore.semaphore_notify(*coord, rank=get_rank_map(lane_id))
                self.semaphore_state[lane_id] = semaphore.state
        elif use_barrier:
            self.c2p_enqueue_barrier.wait(0, self.enqueue_phase)
            self.p2c_enqueue_barrier.arrive()
        if warp_id < 8: # here only supports TP <= 8
            T.ptx.bar.sync(13, 256)
            if warp_id < rank_range:
                semaphore.state = self.semaphore_state[get_rank_map(warp_id)]
                if semaphore.is_triggered():
                    self.queue.batch_enqueue_along_dim_warp(
                        task_type, m_idx, n_idx, k_idx, enqueue_num, extend_dim, lane_id, get_rank_map(warp_id)
                    )
        if use_barrier:
            self.enqueue_phase = self.enqueue_phase ^ 1


    @T.macro
    def push_tasks_along_dim(
        self,
        task_type,
        m_idx,
        n_idx,
        k_idx,
        enqueue_num,
        extend_dim,
        warp_id,
        lane_id,
        semaphore,
        *coord,
        use_barrier=True,
        rank=-1,
    ):
        if enqueue_num <= 32:
            self.push_tasks_along_dim_warp(
                task_type,
                m_idx,
                n_idx,
                k_idx,
                enqueue_num,
                extend_dim,
                warp_id,
                lane_id,
                semaphore,
                *coord,
                use_barrier=use_barrier,
                rank=rank,
            )
        else:
            self.push_tasks_along_dim_wg(
                task_type,
                m_idx,
                n_idx,
                k_idx,
                enqueue_num,
                extend_dim,
                warp_id,
                lane_id,
                semaphore,
                *coord,
                use_barrier=use_barrier,
                rank=rank,
            )

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
