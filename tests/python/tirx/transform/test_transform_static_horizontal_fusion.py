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
from tvm.tirx.megakernel.utils.config import KernelConfig
from tvm.tirx.megakernel.utils.base import SmemManager
import tvm.tirx.megakernel.utils.static_scheduler as static_scheduler
import tvm.tirx.megakernel.utils.dynamic_scheduler as dynamic_scheduler
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tirx as Tx

SM_CNT = 148
NUM_THREADS = 256


def test_basic():

    M = 1024
    N = 1024

    BLOCK_M = 64
    BLOCK_N = 64

    NUM_BLOCK_M = M // BLOCK_M
    NUM_BLOCK_N = N // BLOCK_N

    # fmt: off
    @I.ir_module(tirx=True)
    class Before:
        @Tx.prim_func(tirx=True, private=True)
        def stage_1(A: Tx.handle, B: Tx.handle, m: Tx.int32, n: Tx.int32, k: Tx.int32):
            Tx.func_attr({"megakernel.device_func": "stage1"})
            A_ptr = Tx.match_buffer(A, (M, N), "float32")
            B_ptr = Tx.match_buffer(B, (M, NUM_BLOCK_N), "float32")
            with Tx.cta():
                buf = Tx.alloc_buffer([KernelConfig.MAX_SMEM_SIZE], "uint8", scope="shared.dyn", align=16)
                smem_manager = Tx.meta_var(SmemManager(KernelConfig.MAX_SMEM_SIZE, 16384, buf.data, fusion_mode=True))
                smem_manager.set_tile(None)
                with Tx.cta():
                    Tx.sblock_attr({"tirx.megakernel.persistent.init": True})
                    smem_manager.init()
                with Tx.cta():
                    Tx.sblock_attr({"tirx.tile_class.run": True})
                    A_smem = smem_manager.alloc([BLOCK_M, BLOCK_N], "float32", align=16, name="A_smem")
                    B_smem = smem_manager.alloc([BLOCK_M, 1], "float32", align=16, name="B_smem")
                    smem_manager.wait_all("cta")
                    Tx.copy(A_smem, A_ptr[m * BLOCK_M: (m + 1) * BLOCK_M, n * BLOCK_N: (n + 1) * BLOCK_N])
                    Tx.sum(B_smem, A_smem)
                    Tx.copy(B_ptr[m * BLOCK_M: (m + 1) * BLOCK_M, n], B_smem)
                    smem_manager.arrive_all("cta")
                    smem_manager.advance()

        @Tx.prim_func(tirx=True, private=True)
        def stage_2(B: Tx.handle, C: Tx.handle, m: Tx.int32, n: Tx.int32, k: Tx.int32):
            Tx.func_attr({"megakernel.device_func": "stage2"})
            B_ptr = Tx.match_buffer(B, (M, NUM_BLOCK_N), "float32")
            C_ptr = Tx.match_buffer(C, (M, 1), "float32")
            with Tx.cta():
                buf = Tx.alloc_buffer([KernelConfig.MAX_SMEM_SIZE], "uint8", scope="shared.dyn", align=16)
                smem_manager = Tx.meta_var(SmemManager(KernelConfig.MAX_SMEM_SIZE, 16384, buf.data, fusion_mode=True))
                smem_manager.set_tile(None)
                with Tx.cta():
                    Tx.sblock_attr({"tirx.megakernel.persistent.init": True})
                    smem_manager.init()
                with Tx.cta():
                    Tx.sblock_attr({"tirx.tile_class.run": True})
                    B_smem = smem_manager.alloc([BLOCK_M, NUM_BLOCK_N], "float32", align=16, name="B_smem")
                    C_smem = smem_manager.alloc([BLOCK_M, 1], "float32", align=16, name="C_smem")
                    smem_manager.wait_all("cta")
                    Tx.copy(B_smem, B_ptr[m * BLOCK_M : (m + 1) * BLOCK_M, :])
                    Tx.sum(C_smem, B_smem)
                    Tx.copy(C_ptr[m * BLOCK_M : (m + 1) * BLOCK_M, 0], C_smem)
                    smem_manager.arrive_all("cta")
                    smem_manager.advance()

        @R.function
        def mega_kernel(A: R.Tensor((M, N), "float32"), workspace: R.Tensor((100000,), "int32")):
            cls = Before

            with R.dataflow():
                event_1 = R.alloc_event_tensor(workspace, [NUM_BLOCK_M,], NUM_BLOCK_N)
                event_2 = R.alloc_event_tensor(workspace, [1,], NUM_BLOCK_M)
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
    # Before.show()
    fused_static_scheduler_mod = relax.transform.StaticHorizontalFusion(
        "mega_kernel", "static", static_scheduler.StaticTileScheduler, static_scheduler.Semaphore, "mega_kernel_"
    )(Before)
    fused_dynamic_scheduler_mod = relax.transform.StaticHorizontalFusion(
        "mega_kernel", "dynamic", dynamic_scheduler.DynamicTileScheduler, dynamic_scheduler.Semaphore, "mega_kernel_"
    )(Before)
    # fused_static_scheduler_mod.show()
    # fused_dynamic_scheduler_mod.show()

    # testing correctness
    A_np = np.random.randn(M, N).astype(np.float32)
    DEV = tvm.cuda(0)
    target = tvm.target.Target("cuda")

    A_tvm = tvm.runtime.tensor(A_np, device=DEV)
    workspace_tvm = tvm.runtime.tensor(np.zeros((100000,), dtype=np.int32), device=DEV)

    with target:
        fused_static_scheduler_mod = tvm.compile(
            fused_static_scheduler_mod, target=target, tir_pipeline="tirx"
        )
        vm = tvm.relax.VirtualMachine(fused_static_scheduler_mod, DEV)
        C_tvm_static_scheduler_fused = vm["mega_kernel"](A_tvm, workspace_tvm)
        ret_static_scheduler_fused = C_tvm_static_scheduler_fused.numpy()

        fused_dynamic_scheduler_mod = tvm.compile(
            fused_dynamic_scheduler_mod, target=target, tir_pipeline="tirx"
        )
        vm = tvm.relax.VirtualMachine(fused_dynamic_scheduler_mod, DEV)
        C_tvm_dynamic_scheduler_fused = vm["mega_kernel"](A_tvm, workspace_tvm)
        ret_dynamic_scheduler_fused = C_tvm_dynamic_scheduler_fused.numpy()

        ret_std = np.sum(A_np, axis=1, keepdims=True)
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
    @I.ir_module(tirx=True)
    class Before:
        @Tx.prim_func(tirx=True, private=True)
        def stage_1(A: Tx.handle, B: Tx.handle, m: Tx.int32, n: Tx.int32, k: Tx.int32):
            Tx.func_attr({"megakernel.device_func": "stage1"})
            A_ptr = Tx.match_buffer(A, (M, N), "float32")
            B_ptr = Tx.match_buffer(B, (M, NUM_BLOCK_N), "float32")
            with Tx.cta():
                buf = Tx.alloc_buffer([KernelConfig.MAX_SMEM_SIZE], "uint8", scope="shared.dyn", align=16)
                smem_manager = Tx.meta_var(SmemManager(KernelConfig.MAX_SMEM_SIZE, 16384, buf.data, fusion_mode=True))
                smem_manager.set_tile(None)
                with Tx.cta():
                    Tx.sblock_attr({"tirx.megakernel.persistent.init": True})
                    smem_manager.init()
                with Tx.cta():
                    Tx.sblock_attr({"tirx.tile_class.run": True})
                    A_smem = smem_manager.alloc([BLOCK_M, BLOCK_N], "float32", align=16, name="A_smem")
                    B_smem = smem_manager.alloc([BLOCK_M, 1], "float32", align=16, name="B_smem")
                    smem_manager.wait_all("cta")
                    Tx.copy(A_smem, A_ptr[m * BLOCK_M: (m + 1) * BLOCK_M, n * BLOCK_N: (n + 1) * BLOCK_N])
                    Tx.sum(B_smem, A_smem)
                    Tx.copy(B_ptr[m * BLOCK_M: (m + 1) * BLOCK_M, n], B_smem)
                    smem_manager.arrive_all("cta")
                    smem_manager.advance()

        @Tx.prim_func(tirx=True, private=True)
        def stage_2(B: Tx.handle, C: Tx.handle, m: Tx.int32, n: Tx.int32, k: Tx.int32):
            Tx.func_attr({"megakernel.device_func": "stage2"})
            B_ptr = Tx.match_buffer(B, (M, NUM_BLOCK_N), "float32")
            C_ptr = Tx.match_buffer(C, (M, 1), "float32")
            with Tx.cta():
                buf = Tx.alloc_buffer([KernelConfig.MAX_SMEM_SIZE], "uint8", scope="shared.dyn", align=16)
                smem_manager = Tx.meta_var(SmemManager(KernelConfig.MAX_SMEM_SIZE, 16384, buf.data, fusion_mode=True))
                smem_manager.set_tile(None)
                with Tx.cta():
                    Tx.sblock_attr({"tirx.megakernel.persistent.init": True})
                    smem_manager.init()
                with Tx.cta():
                    Tx.sblock_attr({"tirx.tile_class.run": True})
                    B_smem = smem_manager.alloc([BLOCK_M, NUM_BLOCK_N], "float32", align=16, name="B_smem")
                    C_smem = smem_manager.alloc([BLOCK_M, 1], "float32", align=16, name="C_smem")
                    smem_manager.wait_all("cta")
                    Tx.copy(B_smem, B_ptr[m * BLOCK_M : (m + 1) * BLOCK_M, :])
                    Tx.sum(C_smem, B_smem)
                    Tx.copy(C_ptr[m * BLOCK_M : (m + 1) * BLOCK_M, 0], C_smem)
                    smem_manager.arrive_all("cta")
                    smem_manager.advance()

        @R.function
        def mega_kernel(A: R.Tensor((M, N), "float32"), workspace: R.Tensor((100000,), "int32"), P: R.Tensor((NUM_BLOCK_M,), "int32"), inv_P: R.Tensor((NUM_BLOCK_M,), "int32")):
            cls = Before

            with R.dataflow():
                event_1 = R.alloc_event_tensor(workspace, [NUM_BLOCK_M,], NUM_BLOCK_N)
                event_2 = R.alloc_event_tensor(workspace, [1,], NUM_BLOCK_M)
                B = R.call_tir_device(
                    cls.stage_1,
                    A,
                    out_sinfo=relax.TensorStructInfo([M, NUM_BLOCK_N], "float32"),
                    job_id=0,
                    tile_num=(NUM_BLOCK_M, NUM_BLOCK_N, 1),
                    out_deps=relax.utils.Dependency(
                        event=event_1,
                        dep=lambda i, j, k, notify_idx: (1, -1, P[i]),
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
                        dep=lambda i, j, k, wait_idx: (1, -1, P[i]),
                    ),
                    inverse_in_deps=relax.utils.Dependency(
                        event=event_1,
                        dep=lambda rank, i, inv_idx: (1, inv_P[i], inv_idx // 1, inv_idx % 1),
                    ),
                    out_deps=relax.utils.Dependency(
                        event=event_2,
                        dep=lambda i, j, k, notify_idx: (1, -1, 0),
                    ),
                )
                R.output(C)
            return C
    # fmt: on
    Before.show()
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
    workspace_tvm = tvm.runtime.tensor(np.zeros((100000,), dtype=np.int32), device=DEV)
    P_tvm = tvm.runtime.tensor(P_np, device=DEV)
    P_inv_tvm = tvm.runtime.tensor(P_inv_np, device=DEV)

    with target:
        fused_static_scheduler_mod = tvm.compile(
            fused_static_scheduler_mod, target=target, tir_pipeline="tirx"
        )
        vm = tvm.relax.VirtualMachine(fused_static_scheduler_mod, DEV)
        C_tvm_static_scheduler_fused = vm["mega_kernel"](A_tvm, workspace_tvm, P_tvm, P_inv_tvm)
        ret_static_scheduler_fused = C_tvm_static_scheduler_fused.numpy()

        fused_dynamic_scheduler_mod = tvm.compile(
            fused_dynamic_scheduler_mod, target=target, tir_pipeline="tirx"
        )
        vm = tvm.relax.VirtualMachine(fused_dynamic_scheduler_mod, DEV)
        C_tvm_dynamic_scheduler_fused = vm["mega_kernel"](A_tvm, workspace_tvm, P_tvm, P_inv_tvm)
        ret_dynamic_scheduler_fused = C_tvm_dynamic_scheduler_fused.numpy()

        ret_std = np.sum(A_np, axis=1, keepdims=True)
        tvm.testing.assert_allclose(ret_static_scheduler_fused, ret_std, rtol=1e-3, atol=1e-3)
        tvm.testing.assert_allclose(ret_dynamic_scheduler_fused, ret_std, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    test_basic()
    test_extra_args()
