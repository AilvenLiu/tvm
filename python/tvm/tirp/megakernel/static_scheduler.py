
import tvm
from tvm.script import tir as T

from .common import JobType, KernelConfig, any_sync

atomic_add_int32 = f"""
__forceinline__ __device__ void atomic_add_int32(int32_t* addr, int32_t value, int32_t pe) {{
    if (pe >= 0) {{
        void* ptr = nvshmem_ptr(addr, pe);
         asm volatile("red.async.release.global.gpu.add.s32 [%0], %1;" ::"l"(ptr), "r"(value)
                       : "memory");
        //atomicAdd_system(ptr, value);
    }} else {{
        asm volatile("red.async.release.global.gpu.add.s32 [%0], %1;" ::"l"(addr), "r"(value)
                       : "memory");
    }}
}}
"""

atomic_add_int32_local = f"""
__forceinline__ __device__ void atomic_add_int32(int32_t* addr, int32_t value, int32_t pe) {{
     asm volatile("red.async.release.global.gpu.add.s32 [%0], %1;" ::"l"(addr), "r"(value)
                   : "memory");
}}
"""

nvshmem_get_ptr = """
__forceinline__ __device__ void* nvshmem_get_ptr(void* ptr, int32_t pe) {
    return nvshmem_ptr(ptr, pe);
}
"""


class Semaphore:
    def __init__(self, cnt, buffer, decrement=False, use_nvshmem=False):
        self.cnt = cnt
        self.sem = buffer
        self.state = T.alloc_buffer([1], "int32", scope="local", align=4, name="semaphore_state")
        self.decrement = decrement
        if use_nvshmem:
            self.atomic_add_int32 = atomic_add_int32
        else:
            self.atomic_add_int32 = atomic_add_int32_local
    
    # cta-level interface
    @T.macro
    def semaphore_wait(self, *coord):
        with T.thread():
            if not self.decrement:
                while 1:
                    T.ptx.ld_global_acquire(
                        self.state[0],
                        self.sem.access_ptr("r", offset=self.sem.elem_offset_of(coord)),
                    )
                    if T.cuda.syncthreads_and(self.state[0] == self.cnt):
                        break
                    T.cuda.nano_sleep(40)
            else:
                while 1:
                    T.ptx.ld_global_acquire(
                        self.state[0],
                        self.sem.access_ptr("r", offset=self.sem.elem_offset_of(coord)),
                    )
                    if T.cuda.syncthreads_and(self.state[0] == 0):
                        break
                    T.cuda.nano_sleep(40)

    # warp-level interface
    @T.macro
    def semaphore_wait_warp(self, *coord, mask=0xffffffff):
        with T.thread():
            warp_id = T.warp_id([KernelConfig.WARP_NUMBER * KernelConfig.WG_NUMBER], parent="cta")
            lane_id = T.thread_id([32], parent="warp")
            if (mask >> warp_id) & 1 == 1:
                self.state[0] = -1
                if not self.decrement:
                    while 1:
                        if lane_id == 0:
                            T.ptx.ld_global_acquire(
                                self.state[0], self.sem.access_ptr("r", offset=self.sem.elem_offset_of(coord))
                            )
                        if any_sync(0xffffffff, self.state[0] == self.cnt):
                            break
                        T.cuda.nano_sleep(40)
                else:
                    while 1:    
                        if lane_id == 0:
                            T.ptx.ld_global_acquire(
                                self.state[0], self.sem.access_ptr("r", offset=self.sem.elem_offset_of(coord))
                            )
                        if any_sync(0xffffffff, self.state[0] == 0):
                            break
                        T.cuda.nano_sleep(40)
    

    @T.macro
    def semaphore_notify(self, *coord, rank=-1):
        # wg is synced
        if self.cnt >= 0:
            T.cuda.func_call(
                "atomic_add_int32",
                self.sem.ptr_to(coord),
                1,
                rank,
                source_code=self.atomic_add_int32,
            )
        else:
            T.cuda.func_call(
                "atomic_add_int32",
                self.sem.ptr_to(coord),
                -1,
                rank,
                source_code=self.atomic_add_int32,
            )


lds_v4 = """
__forceinline__ __device__ void lds_v4(void* addr, int32_t* v1, int32_t* v2, int32_t* v3, int32_t* v4) {
    asm volatile("ld.shared::cluster.v4.b32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(*v1), "=r"(*v2), "=r"(*v3), "=r"(*v4)
                 : "l"(addr)
                 : "memory");

}
"""


class StaticTileScheduler:
    MAX_TASKS = 128

    def __init__(self, prefix: str, exec_queue, smem_manager):
        super().__init__()
        self.m_idx = T.local_cell("int32", name=prefix + "_m_idx")
        self.n_idx = T.local_cell("int32", name=prefix + "_n_idx")
        self.k_idx = T.local_cell("int32", name=prefix + "_k_idx")
        self.task_type = T.local_cell("int32", name=prefix + "_task_type")
        self.tile_idx = T.local_cell("int32", name=prefix + "_tile_idx")
        self.queue_smem = smem_manager.alloc((self.MAX_TASKS, 4), "int32", align=16, method="persistent").buffer
        self.exec_queue = exec_queue

    @T.macro
    def update_current_m_n_idx(self):
        T.cuda.func_call(
            "lds_v4",
            T.address_of(self.queue_smem[self.tile_idx, 0]),
            T.address_of(self.m_idx),
            T.address_of(self.n_idx),
            T.address_of(self.k_idx),
            T.address_of(self.task_type),
            source_code=lds_v4,
        )

    @T.macro
    def init(self, bx, tid):
        self.tile_idx = 0
        for k in T.serial(T.ceildiv(self.MAX_TASKS * 4, KernelConfig.NUM_THREADS)):
            idx = T.meta_var(k * KernelConfig.NUM_THREADS + tid)
            if idx < self.MAX_TASKS * 4:
                self.queue_smem[idx // 4, idx % 4] = self.exec_queue[bx, idx // 4, idx % 4]
        T.tvm_storage_sync("shared")
        self.update_current_m_n_idx()

    @T.macro
    def next_tile(self):
        self.tile_idx += 1
        self.update_current_m_n_idx()

    def valid(self):
        return tvm.tir.all(self.tile_idx < self.MAX_TASKS, self.task_type != JobType.END.value)
