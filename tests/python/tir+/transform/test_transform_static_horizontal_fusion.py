# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import numpy as np

import tvm
import tvm.testing
from tvm import relax
from tvm.tirp.megakernel.common import (
    JobType,
    SemaphoreBase,
    TileSchedulerBase,
    SmemManager,
    KernelConfig,
    unpack_from_32bit,
)
import tvm.tirp.megakernel.static_scheduler as static_scheduler
import tvm.tirp.megakernel.dynamic_scheduler as dynamic_scheduler
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tir as T
from tvm.script import tirp as Tp

SM_CNT = 148
NUM_THREADS = 256


class Semaphore(SemaphoreBase):
    def __init__(self, expected_cnt, buffer, decrement=False, base=(1 << 16)):
        self.expected_cnt = expected_cnt
        self.base = base
        self.sem = buffer
        self.state = T.alloc_buffer([1], "int32", scope="local", align=4, name="semaphore_state")
        self.atomic_add_int32 = f"""
__forceinline__ __device__ void atomic_add_int32(int32_t* addr, int32_t value, int32_t pe) {{
    asm volatile("red.async.release.global.gpu.add.s32 [%0], %1;" ::"l"(addr), "r"(value)
                : "memory");
}}
"""

    @T.macro
    def semaphore_wait(self, *coord, level="cta", mask=0xFFFFFFFF):
        with T.thread():
            while 1:
                T.ptx.ld_global_acquire(
                    self.state[0],
                    self.sem.access_ptr("r", offset=self.sem.elem_offset_of(coord)),
                )
                if T.cuda.syncthreads_and(self.state[0] == 0):
                    break
                T.cuda.nano_sleep(40)

    @T.macro
    def semaphore_notify(self, *coord, rank=-1):
        T.cuda.func_call(
            "atomic_add_int32",
            self.sem.ptr_to(coord),
            -(self.base + 1),
            rank,
            source_code=self.atomic_add_int32,
        )

def test_basic():

    M = 1024
    N = 1024

    BLOCK_M = 64
    BLOCK_N = 64

    NUM_BLOCK_M = M // BLOCK_M
    NUM_BLOCK_N = N // BLOCK_N

    class SpatialTileScheduler(TileSchedulerBase):
        @staticmethod
        def int_var(name):
            return T.alloc_buffer([1], "int32", scope="local", align=4, name=name)

        def __init__(self, prefix, exec_queue, smem_manager):
            self.sm_cnt = SM_CNT
            self.division = 132
            self.m_idx = self.int_var("m_idx")
            self.n_idx = self.int_var("n_idx")
            self.k_idx = self.int_var("k_idx")
            self.type = self.int_var("type")
            self.linear_idx = self.int_var("linear_idx")

        @T.macro
        def _update_current_m_n_idx(self):
            if self.type[0] == 0:
                self.m_idx[0] = self.linear_idx[0] // NUM_BLOCK_N
                self.n_idx[0] = self.linear_idx[0] % NUM_BLOCK_N
            else:
                self.m_idx[0] = self.linear_idx[0]

        def get_idx_and_task_type(self):
            return [self.m_idx[0], self.n_idx[0], self.k_idx[0]], self.type[0]

        @T.macro
        def init(self):
            with T.cta():
                bx = T.cta_id([self.sm_cnt], parent="kernel")
                if bx < self.division:
                    self.type[0] = 0
                    self.linear_idx[0] = bx
                else:
                    self.type[0] = 1
                    self.linear_idx[0] = bx - self.division
                self._update_current_m_n_idx()

        @T.macro
        def wait(self, evt: Semaphore, *coord, wait_level="cta", mask=0xFFFFFFFF):
            evt.semaphore_wait(*coord, level=wait_level, mask=mask)

        @T.macro
        def notify(self, evt: Semaphore, func_notify, scope="cta", scope_id=0):
            with T.cta():
                tid = T.thread_id([NUM_THREADS], parent="cta")
                T.cuda.cta_sync()
                notify_num, rank, *coord = func_notify(tid)
                if tid < notify_num:
                    evt.semaphore_notify(*coord, rank=rank)

        @T.macro
        def next_tile(self):
            if self.type[0] == 0:
                self.linear_idx[0] = self.linear_idx[0] + self.division
            else:
                self.linear_idx[0] = self.linear_idx[0] + (self.sm_cnt - self.division)
            self._update_current_m_n_idx()

        def valid(self):
            return T.if_then_else(
                self.type[0] == 1,
                self.linear_idx[0] < NUM_BLOCK_M,
                self.linear_idx[0] < NUM_BLOCK_M * NUM_BLOCK_N,
            )

    # fmt: off
    @I.ir_module(tirp=True)
    class Before:
        @T.prim_func(tirp=True, private=True)
        def stage_1(A: T.handle, B: T.handle, m: T.int32, n: T.int32, k: T.int32):
            T.func_attr({"megakernel.device_func": "stage1"})
            A_ptr = T.match_buffer(A, (M, N), "float32")
            B_ptr = T.match_buffer(B, (M, NUM_BLOCK_N), "float32")
            with T.cta():
                buf = T.alloc_buffer([KernelConfig.MAX_SMEM_SIZE], "uint8", scope="shared.dyn", align=16)
                smem_manager = T.meta_var(SmemManager(KernelConfig.MAX_SMEM_SIZE, 16384, buf.data, fusion_mode=True))
                smem_manager.set_tile(None)
                with T.cta():
                    T.block_attr({"tirp.megakernel.persistent.init": True})
                    smem_manager.init()
                with T.cta():
                    T.block_attr({"tirp.tile_class.run": True})
                    A_smem = smem_manager.alloc([BLOCK_M, BLOCK_N], "float32", align=16, name="A_smem")
                    B_smem = smem_manager.alloc([BLOCK_M, 1], "float32", align=16, name="B_smem")
                    smem_manager.wait_all("cta")
                    Tp.copy(A_smem, A_ptr[m * BLOCK_M: (m + 1) * BLOCK_M, n * BLOCK_N: (n + 1) * BLOCK_N])                
                    Tp.sum(B_smem, A_smem)
                    Tp.copy(B_ptr[m * BLOCK_M: (m + 1) * BLOCK_M, n], B_smem)
                    smem_manager.arrive_all("cta")
                    smem_manager.advance()

        @T.prim_func(tirp=True, private=True)
        def stage_2(B: T.handle, C: T.handle, m: T.int32, n: T.int32, k: T.int32):
            T.func_attr({"megakernel.device_func": "stage2"})
            B_ptr = T.match_buffer(B, (M, NUM_BLOCK_N), "float32")
            C_ptr = T.match_buffer(C, (M, 1), "float32")
            with T.cta():
                buf = T.alloc_buffer([KernelConfig.MAX_SMEM_SIZE], "uint8", scope="shared.dyn", align=16)
                smem_manager = T.meta_var(SmemManager(KernelConfig.MAX_SMEM_SIZE, 16384, buf.data, fusion_mode=True))
                smem_manager.set_tile(None)
                with T.cta():
                    T.block_attr({"tirp.megakernel.persistent.init": True})
                    smem_manager.init()
                with T.cta():
                    T.block_attr({"tirp.tile_class.run": True})
                    B_smem = smem_manager.alloc([BLOCK_M, NUM_BLOCK_N], "float32", align=16, name="B_smem")
                    C_smem = smem_manager.alloc([BLOCK_M, 1], "float32", align=16, name="C_smem")
                    smem_manager.wait_all("cta")
                    Tp.copy(B_smem, B_ptr[m * BLOCK_M : (m + 1) * BLOCK_M, :])
                    Tp.sum(C_smem, B_smem)
                    Tp.copy(C_ptr[m * BLOCK_M : (m + 1) * BLOCK_M, 0], C_smem)
                    smem_manager.arrive_all("cta")
                    smem_manager.advance()
                
        @R.function
        def mega_kernel(A: R.Tensor((M, N), "float32"), event_1: R.Tensor((NUM_BLOCK_M,), "int32"), event_2: R.Tensor((1,), "int32")):
            cls = Before
            
            with R.dataflow():
                B = R.call_tir_device(
                    cls.stage_1, 
                    A,
                    out_sinfo=relax.TensorStructInfo([M, NUM_BLOCK_N], "float32"),
                    job_id=0,
                    tile_num=(NUM_BLOCK_M, NUM_BLOCK_N, 1),
                    out_deps=relax.utils.Dependency(
                        event=event_1,
                        dep=lambda i, j, k, notify_idx: (1, -1, i),
                    ),
                )
                C = R.call_tir_device(
                    cls.stage_2, 
                    B,
                    out_sinfo=relax.TensorStructInfo([M, 1], "float32"),
                    job_id=1,
                    tile_num=(NUM_BLOCK_M, 1, 1),
                    in_deps=relax.utils.Dependency(
                        event=event_1,
                        dep=lambda i, j, k, wait_idx: (1, -1, i),
                    ),
                    inverse_in_deps=relax.utils.Dependency(
                        event=event_1,
                        dep=lambda rank, i, inv_idx: (1, i, inv_idx // 1, inv_idx % 1),
                    ),
                    out_deps=relax.utils.Dependency(
                        event=event_2,
                        dep=lambda i, j, k, notify_idx: (1, -1, 0),
                    ),
                )
                R.output(C)
            return C
    # fmt: on
    fused_naive_mod = relax.transform.StaticHorizontalFusion(
        "mega_kernel", "static", SpatialTileScheduler, Semaphore, "mega_kernel_"
    )(Before)
    fused_static_scheduler_mod = relax.transform.StaticHorizontalFusion(
        "mega_kernel", "static", static_scheduler.StaticTileScheduler, static_scheduler.Semaphore, "mega_kernel_"
    )(Before)
    fused_dynamic_scheduler_mod = relax.transform.StaticHorizontalFusion(
        "mega_kernel", "dynamic", dynamic_scheduler.DynamicTileScheduler, dynamic_scheduler.Semaphore, "mega_kernel_"
    )(Before)

    # testing correctness
    A_np = np.random.randn(M, N).astype(np.float32)
    DEV = tvm.cuda(0)
    target = tvm.target.Target("cuda")

    A_tvm = tvm.runtime.tensor(A_np, device=DEV)

    # we initialize event on host instead of device because of Tp.fill not correctly implemented
    with target:
        evt_1_tvm = tvm.runtime.tensor(
            np.full((NUM_BLOCK_M,), NUM_BLOCK_N * (1 + (1 << 16)), dtype=np.int32), device=DEV
        )
        evt_2_tvm = tvm.runtime.tensor(
            np.full((1,), NUM_BLOCK_M * (1 + (1 << 16)), dtype=np.int32), device=DEV
        )
        fused_naive_mod = tvm.compile(fused_naive_mod, target=target, tir_pipeline="tirp")
        vm = tvm.relax.VirtualMachine(fused_naive_mod, DEV)
        C_tvm_naive_fused = vm["mega_kernel"](A_tvm, evt_1_tvm, evt_2_tvm)
        ret_naive_fused = C_tvm_naive_fused.numpy()

        evt_1_tvm = tvm.runtime.tensor(
            np.full((NUM_BLOCK_M,), NUM_BLOCK_N * (1 + (1 << 16)), dtype=np.int32), device=DEV
        )
        evt_2_tvm = tvm.runtime.tensor(
            np.full((1,), NUM_BLOCK_M * (1 + (1 << 16)), dtype=np.int32), device=DEV
        )
        fused_static_scheduler_mod = tvm.compile(
            fused_static_scheduler_mod, target=target, tir_pipeline="tirp"
        )
        vm = tvm.relax.VirtualMachine(fused_static_scheduler_mod, DEV)
        C_tvm_static_scheduler_fused = vm["mega_kernel"](A_tvm, evt_1_tvm, evt_2_tvm)
        ret_static_scheduler_fused = C_tvm_static_scheduler_fused.numpy()
        
        evt_1_tvm = tvm.runtime.tensor(
            np.full((NUM_BLOCK_M,), NUM_BLOCK_N * (1 + (1 << 16)), dtype=np.int32), device=DEV
        )
        evt_2_tvm = tvm.runtime.tensor(
            np.full((1,), NUM_BLOCK_M * (1 + (1 << 16)), dtype=np.int32), device=DEV
        )
        fused_dynamic_scheduler_mod = tvm.compile(
            fused_dynamic_scheduler_mod, target=target, tir_pipeline="tirp"
        )
        vm = tvm.relax.VirtualMachine(fused_dynamic_scheduler_mod, DEV)
        C_tvm_dynamic_scheduler_fused = vm["mega_kernel"](A_tvm, evt_1_tvm, evt_2_tvm)
        ret_dynamic_scheduler_fused = C_tvm_dynamic_scheduler_fused.numpy()

        ret_std = np.sum(A_np, axis=1, keepdims=True)
        tvm.testing.assert_allclose(ret_naive_fused, ret_std, rtol=1e-3, atol=1e-3)
        tvm.testing.assert_allclose(ret_static_scheduler_fused, ret_std, rtol=1e-3, atol=1e-3)
        tvm.testing.assert_allclose(ret_dynamic_scheduler_fused, ret_std, rtol=1e-3, atol=1e-3)


def test_extra_args():
    M = 1024
    N = 1024

    BLOCK_M = 64
    BLOCK_N = 64

    NUM_BLOCK_M = M // BLOCK_M
    NUM_BLOCK_N = N // BLOCK_N

    # fmt: off
    @I.ir_module(tirp=True)
    class Before:
        @T.prim_func(tirp=True, private=True)
        def stage_1(A: T.handle, B: T.handle, m: T.int32, n: T.int32, k: T.int32):
            T.func_attr({"megakernel.device_func": "stage1"})
            A_ptr = T.match_buffer(A, (M, N), "float32")
            B_ptr = T.match_buffer(B, (M, NUM_BLOCK_N), "float32")
            with T.cta():
                buf = T.alloc_buffer([KernelConfig.MAX_SMEM_SIZE], "uint8", scope="shared.dyn", align=16)
                smem_manager = T.meta_var(SmemManager(KernelConfig.MAX_SMEM_SIZE, 16384, buf.data, fusion_mode=True))
                smem_manager.set_tile(None)
                with T.cta():
                    T.block_attr({"tirp.megakernel.persistent.init": True})
                    smem_manager.init()
                with T.cta():
                    T.block_attr({"tirp.tile_class.run": True})
                    A_smem = smem_manager.alloc([BLOCK_M, BLOCK_N], "float32", align=16, name="A_smem")
                    B_smem = smem_manager.alloc([BLOCK_M, 1], "float32", align=16, name="B_smem")
                    smem_manager.wait_all("cta")
                    Tp.copy(A_smem, A_ptr[m * BLOCK_M: (m + 1) * BLOCK_M, n * BLOCK_N: (n + 1) * BLOCK_N])                
                    Tp.sum(B_smem, A_smem)
                    Tp.copy(B_ptr[m * BLOCK_M: (m + 1) * BLOCK_M, n], B_smem)
                    smem_manager.arrive_all("cta")
                    smem_manager.advance()

        @T.prim_func(tirp=True, private=True)
        def stage_2(B: T.handle, C: T.handle, m: T.int32, n: T.int32, k: T.int32):
            T.func_attr({"megakernel.device_func": "stage2"})
            B_ptr = T.match_buffer(B, (M, NUM_BLOCK_N), "float32")
            C_ptr = T.match_buffer(C, (M, 1), "float32")
            with T.cta():
                buf = T.alloc_buffer([KernelConfig.MAX_SMEM_SIZE], "uint8", scope="shared.dyn", align=16)
                smem_manager = T.meta_var(SmemManager(KernelConfig.MAX_SMEM_SIZE, 16384, buf.data, fusion_mode=True))
                smem_manager.set_tile(None)
                with T.cta():
                    T.block_attr({"tirp.megakernel.persistent.init": True})
                    smem_manager.init()
                with T.cta():
                    T.block_attr({"tirp.tile_class.run": True})
                    B_smem = smem_manager.alloc([BLOCK_M, NUM_BLOCK_N], "float32", align=16, name="B_smem")
                    C_smem = smem_manager.alloc([BLOCK_M, 1], "float32", align=16, name="C_smem")
                    smem_manager.wait_all("cta")
                    Tp.copy(B_smem, B_ptr[m * BLOCK_M : (m + 1) * BLOCK_M, :])
                    Tp.sum(C_smem, B_smem)
                    Tp.copy(C_ptr[m * BLOCK_M : (m + 1) * BLOCK_M, 0], C_smem)
                    smem_manager.arrive_all("cta")
                    smem_manager.advance()
        
        @T.prim_func(tirp=True, private=True)
        def end(m: T.int32, n: T.int32, k: T.int32):
            T.func_attr({"megakernel.device_func": "end"})
            with T.cta():
                T.block_attr({"tirp.tile_class.run": True})
                T.add_to_parent(T.evaluate(0))
            
        @R.function
        def mega_kernel(A: R.Tensor((M, N), "float32"), event_1: R.Tensor((NUM_BLOCK_M,), "int32"), event_2: R.Tensor((1,), "int32"), P: R.Tensor((NUM_BLOCK_M,), "int32"), inv_P: R.Tensor((NUM_BLOCK_M,), "int32")):
            cls = Before
            
            with R.dataflow():
                B = R.call_tir_device(
                    cls.stage_1, 
                    A,
                    out_sinfo=relax.TensorStructInfo([M, NUM_BLOCK_N], "float32"),
                    job_id=0,
                    tile_num=(NUM_BLOCK_M, NUM_BLOCK_N, 1),
                    out_deps=relax.utils.Dependency(
                        event=event_1,
                        dep=lambda i, j, k, notify_idx, p: (1, -1, p[i]),
                        extra_args=[P],
                    ),
                )
                C = R.call_tir_device(
                    cls.stage_2, 
                    B,
                    out_sinfo=relax.TensorStructInfo([M, 1], "float32"),
                    job_id=1,
                    tile_num=(NUM_BLOCK_M, 1, 1),
                    in_deps=relax.utils.Dependency(
                        event=event_1,
                        dep=lambda i, j, k, wait_idx, p: (1, -1, p[i]),
                        extra_args=[P],
                    ),
                    inverse_in_deps=relax.utils.Dependency(
                        event=event_1,
                        dep=lambda rank, i, inv_idx, inv_p: (1, inv_p[i], inv_idx // 1, inv_idx % 1),
                        extra_args=[inv_P],
                    ),
                    out_deps=relax.utils.Dependency(
                        event=event_2,
                        dep=lambda i, j, k, notify_idx: (1, -1, 0),
                    ),
                )
                R.output(C)
            return C
    # fmt: on
    fused_static_scheduler_mod = relax.transform.StaticHorizontalFusion(
        "mega_kernel", "static", static_scheduler.StaticTileScheduler, static_scheduler.Semaphore, "mega_kernel_"
    )(Before)
    fused_dynamic_scheduler_mod = relax.transform.StaticHorizontalFusion(
        "mega_kernel", "dynamic", dynamic_scheduler.DynamicTileScheduler, dynamic_scheduler.Semaphore, "mega_kernel_"
    )(Before)

    # testing correctness
    A_np = np.random.randn(M, N).astype(np.float32)
    P_np = np.random.permutation(np.arange(NUM_BLOCK_M)).astype(np.int32)
    P_inv_np = np.argsort(P_np).astype(np.int32)
    DEV = tvm.cuda(0)
    target = tvm.target.Target("cuda")

    A_tvm = tvm.runtime.tensor(A_np, device=DEV)
    P_tvm = tvm.runtime.tensor(P_np, device=DEV)
    P_inv_tvm = tvm.runtime.tensor(P_inv_np, device=DEV)

    # we initialize event on host instead of device because of Tp.fill not correctly implemented
    with target:
        evt_1 = tvm.runtime.tensor(
            np.full((NUM_BLOCK_M,), NUM_BLOCK_N * (1 + (1 << 16)), dtype=np.int32), device=DEV
        )
        evt_2 = tvm.runtime.tensor(
            np.full((1,), NUM_BLOCK_M * (1 + (1 << 16)), dtype=np.int32), device=DEV
        )
        fused_static_scheduler_mod = tvm.compile(
            fused_static_scheduler_mod, target=target, tir_pipeline="tirp"
        )
        vm = tvm.relax.VirtualMachine(fused_static_scheduler_mod, DEV)
        C_tvm_static_scheduler_fused = vm["mega_kernel"](A_tvm, evt_1, evt_2, P_tvm, P_inv_tvm)
        ret_static_scheduler_fused = C_tvm_static_scheduler_fused.numpy()
        
        evt_1 = tvm.runtime.tensor(
            np.full((NUM_BLOCK_M,), NUM_BLOCK_N * (1 + (1 << 16)), dtype=np.int32), device=DEV
        )
        evt_2 = tvm.runtime.tensor(
            np.full((1,), NUM_BLOCK_M * (1 + (1 << 16)), dtype=np.int32), device=DEV
        )
        fused_dynamic_scheduler_mod = tvm.compile(
            fused_dynamic_scheduler_mod, target=target, tir_pipeline="tirp"
        )
        vm = tvm.relax.VirtualMachine(fused_dynamic_scheduler_mod, DEV)
        C_tvm_dynamic_scheduler_fused = vm["mega_kernel"](A_tvm, evt_1, evt_2, P_tvm, P_inv_tvm)
        ret_dynamic_scheduler_fused = C_tvm_dynamic_scheduler_fused.numpy()

        ret_std = np.sum(A_np, axis=1, keepdims=True)
        tvm.testing.assert_allclose(ret_static_scheduler_fused, ret_std, rtol=1e-3, atol=1e-3)
        tvm.testing.assert_allclose(ret_dynamic_scheduler_fused, ret_std, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    test_basic()
    test_extra_args()
