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
# pylint: disable=missing-function-docstring
import functools
import operator
import pytest
import copy
import numpy as np

import tvm
import tvm.testing
from tvm.ir.type import PointerType, PrimType
from tvm.script import tir as T
from tvm.script import tirp as Tp
from tvm.tir.layout import TileLayout, TLane, TCol, tid_in_wg as axis_tid_in_wg
from tvm.tirp.op_schedule.cuda.copy_async import tma_shared_layout


@pytest.mark.parametrize(
    "task",
    [
        (
            ((128, 512), "float32", [(0, 128), (256, 384)]),  # C
            ((3, 128, 64), "float16", [(1, 2), (0, 128), (0, 64)], 3),  # A
            ((3, 128, 64), "float16", [(2, 3), (0, 128), (0, 64)], 3),  # B
            False,  # transA
            False,  # transB
        ),
    ],
)
def test_gemm_tcgen05_cta_group_1(task):
    def next_power_of_2(x):
        """Return the smallest power of 2 greater than or equal to x."""
        if x <= 1:
            return 1
        return 1 << (x - 1).bit_length()

    (
        (C_shape, C_dtype, C_region),
        (A_shape, A_dtype, A_region, A_swizzle_mode),
        (B_shape, B_dtype, B_region, B_swizzle_mode),
        transA,
        transB,
    ) = task
    width = C_region[1][1] - C_region[1][0]
    assert C_shape[0] == 128
    assert C_region[0] == (0, 128)
    assert len(C_shape) == 2
    A_elem_bytes = tvm.runtime.DataType(A_dtype).bits // 8
    B_elem_bytes = tvm.runtime.DataType(B_dtype).bits // 8
    C_elem_bytes = tvm.runtime.DataType(C_dtype).bits // 8
    C_elem_32b = 4 // C_elem_bytes
    cols_alloc = max(32, next_power_of_2(C_shape[1] // C_elem_32b))
    A_layout = tma_shared_layout(A_dtype, A_swizzle_mode, A_shape)
    B_layout = tma_shared_layout(B_dtype, B_swizzle_mode, B_shape)

    r_gmem_A = list(slice(0, A_shape[i]) for i in range(len(A_shape)))
    r_gmem_B = list(slice(0, B_shape[i]) for i in range(len(B_shape)))
    total_bytes = (
        functools.reduce(operator.mul, A_shape, 1) * A_elem_bytes
        + functools.reduce(operator.mul, B_shape, 1) * B_elem_bytes
    )

    r_tmem_C = list(slice(C_region[i][0], C_region[i][1]) for i in range(len(C_shape)))
    r_smem_A = list(slice(A_region[i][0], A_region[i][1]) for i in range(len(A_shape)))
    r_smem_B = list(slice(B_region[i][0], B_region[i][1]) for i in range(len(B_shape)))

    # fmt: off
    @T.prim_func(tirp=True)
    def gemm_async(A_ptr: T.handle, B_ptr: T.handle, C_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, A_shape, A_dtype)
        B = T.match_buffer(B_ptr, B_shape, B_dtype)
        C = T.match_buffer(C_ptr, C_shape, C_dtype)

        with T.kernel():
            bx = T.cta_id([1], parent="kernel")
            wg_id = T.warpgroup_id([1], parent="cta")
            tid_in_wg = T.thread_id([128], parent="warpgroup")
            
            A_smem = T.alloc_buffer(A_shape, A_dtype, scope="shared", layout=A_layout)
            B_smem = T.alloc_buffer(B_shape, B_dtype, scope="shared", layout=B_layout)
            tmem_addr = T.alloc_shared([1], "uint32")
            tma_mbar = T.alloc_shared([1], "uint64")
            mma_mbar = T.alloc_shared([1], "uint64")

            with T.thread()[0:1]:
                T.ptx.mbarrier.init(tma_mbar.ptr_to([0]), 1)
                T.ptx.mbarrier.init(mma_mbar.ptr_to([0]), 1)
            T.ptx.fence.proxy("shared")
            T.cuda.cta_sync()

            with T.warp()[0:1]:
                T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=cols_alloc, cta_group=1)
            T.cuda.cta_sync()
            tmem = T.decl_buffer((128, C_shape[1]), C_dtype, scope="tmem", allocated_addr=tmem_addr[0],
                                 layout=TileLayout(([128, C_shape[1]], [1@TLane, 1@TCol])))

            with T.thread()[0:1]:
                tma_args = T.meta_var({"dispatch": "tma", "mbar": tma_mbar.ptr_to([0])})
                Tp.copy_async(A_smem[*r_gmem_A], A[*r_gmem_A], **tma_args)
                Tp.copy_async(B_smem[*r_gmem_B], B[*r_gmem_B], **tma_args)
                T.ptx.mbarrier.arrive.expect_tx(tma_mbar.ptr_to([0]), total_bytes)
            T.ptx.mbarrier.try_wait(tma_mbar.ptr_to([0]), 0)
            T.cuda.cta_sync()

            with T.thread()[0:1]:
                Tp.gemm_async(tmem[*r_tmem_C], A_smem[*r_smem_A], B_smem[*r_smem_B], dispatch="tcgen05")
                T.ptx.tcgen05.commit(mma_mbar.ptr_to([0]), cta_group=1)
            T.ptx.mbarrier.try_wait(mma_mbar.ptr_to([0]), 0)
            T.cuda.cta_sync()
            
            T.ptx.tcgen05.fence.after_thread_sync()
            C_reg = T.alloc_local(width, dtype=C_dtype)
            C_view = C_reg.view(128, width, layout=TileLayout(([128, width], [1@axis_tid_in_wg, 1])))
            with T.warpgroup()[0:1]:
                Tp.copy(C_view[:, :], tmem[*r_tmem_C])
            T.cuda.cta_sync()
            with T.thread():
                Tp.copy(C[tid_in_wg, C_region[1][0]:C_region[1][1]], C_reg[:])

            with T.warp()[0:1]:
                T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
                T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=cols_alloc, cta_group=1)
    # fmt: on

    dev = tvm.cuda(0)
    np.random.seed(0)

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": gemm_async})
        mod.show()
        mod = tvm.compile(mod, target=target, tir_pipeline="tirp")
        print(mod.mod.imports[0].inspect_source())

        A_np = np.random.randn(*A_shape).astype(A_dtype)
        B_np = np.random.randn(*B_shape).astype(B_dtype)
        C_np = np.zeros(C_shape, dtype=C_dtype)
        A_tvm = tvm.runtime.tensor(A_np, dev)
        B_tvm = tvm.runtime.tensor(B_np, dev)
        C_tvm = tvm.runtime.tensor(C_np, dev)
        mod["main"](A_tvm, B_tvm, C_tvm)

        C_ref = np.zeros(C_shape, dtype=C_dtype)
        A_ref = np.squeeze(A_np[*r_smem_A] if not transA else A_np[*r_smem_A].T)
        B_ref = np.squeeze(B_np[*r_smem_B] if transB else B_np[*r_smem_B].T)
        C_ref[*r_tmem_C] = A_ref @ B_ref
        np.testing.assert_allclose(C_tvm.numpy(), C_ref, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize(
    "task",
    [
        (
            ((256, 512), "float32", [(0, 128), (128, 192)]),  # C
            ((3, 256, 64), "float16", [(1, 2), (0, 128), (0, 64)], 3),  # A
            ((3, 128, 64), "float16", [(2, 3), (0, 64), (0, 64)], 3),  # B
            False,  # transA
            False,  # transB
        ),
    ],
)
def test_gemm_tcgen05_cta_group_2(task):
    def next_power_of_2(x):
        """Return the smallest power of 2 greater than or equal to x."""
        if x <= 1:
            return 1
        return 1 << (x - 1).bit_length()

    (
        (C_shape, C_dtype, C_region),
        (A_shape, A_dtype, A_region, A_swizzle_mode),
        (B_shape, B_dtype, B_region, B_swizzle_mode),
        transA,
        transB,
    ) = task
    width = (C_region[1][1] - C_region[1][0]) * 2
    assert C_shape[0] == 256
    assert C_region[0] == (0, 128)
    assert len(C_shape) == 2
    A_elem_bytes = tvm.runtime.DataType(A_dtype).bits // 8
    B_elem_bytes = tvm.runtime.DataType(B_dtype).bits // 8
    C_elem_bytes = tvm.runtime.DataType(C_dtype).bits // 8
    C_elem_32b = 4 // C_elem_bytes
    cols_alloc = max(32, next_power_of_2(C_shape[1] // C_elem_32b))

    def dim(trans):
        return -1 if trans else -2

    def get_shape_per_cta(shape, trans):
        shape_per_cta = copy.deepcopy(list(shape))
        shape_per_cta[dim(trans)] //= 2
        return shape_per_cta

    A_shape_per_cta = get_shape_per_cta(A_shape, transA)
    B_shape_per_cta = get_shape_per_cta(B_shape, transB)
    A_layout = tma_shared_layout(A_dtype, A_swizzle_mode, A_shape_per_cta)
    B_layout = tma_shared_layout(B_dtype, B_swizzle_mode, B_shape_per_cta)

    def get_global_region(shape, trans, cbx):
        r = list(slice(0, shape[i]) for i in range(len(shape)))
        d = dim(trans)
        r[d] = slice(cbx * shape[d], (cbx + 1) * shape[d])
        return r

    r_smem_A_in = list(slice(0, A_shape_per_cta[i]) for i in range(len(A_shape_per_cta)))
    r_smem_B_in = list(slice(0, B_shape_per_cta[i]) for i in range(len(B_shape_per_cta)))
    total_bytes = (
        functools.reduce(operator.mul, A_shape, 1) * A_elem_bytes
        + functools.reduce(operator.mul, B_shape, 1) * B_elem_bytes
    )

    r_tmem_C = list(slice(C_region[i][0], C_region[i][1]) for i in range(len(C_shape)))
    r_smem_A = list(slice(A_region[i][0], A_region[i][1]) for i in range(len(A_shape)))
    r_smem_B = list(slice(B_region[i][0], B_region[i][1]) for i in range(len(B_shape)))

    # fmt: off
    @T.prim_func(tirp=True)
    def gemm_async(A_ptr: T.handle, B_ptr: T.handle, C_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, A_shape, A_dtype)
        B = T.match_buffer(B_ptr, B_shape, B_dtype)
        C = T.match_buffer(C_ptr, C_shape, C_dtype)

        with T.kernel():
            cbx, cby = T.cta_id([2, 1], parent="cluster")
            bx = T.cta_id([2], parent="kernel")
            wg_id = T.warpgroup_id([1], parent="cta")
            tid_in_wg = T.thread_id([128], parent="warpgroup")

            A_smem = T.alloc_buffer(A_shape_per_cta, A_dtype, scope="shared", layout=A_layout)
            B_smem = T.alloc_buffer(B_shape_per_cta, B_dtype, scope="shared", layout=B_layout)
            tmem_addr = T.alloc_shared([1], "uint32")
            tma_mbar = T.alloc_shared([1], "uint64")
            mma_mbar = T.alloc_shared([1], "uint64")

            ptr: T.Var(name="ptr", dtype=PointerType(PrimType("uint64"))) = T.reinterpret("handle", T.ptx.map_shared_rank(tma_mbar.ptr_to([0]), 0))
            tma_mbar_cta_0 = T.decl_buffer([1], "uint64", data=ptr, scope="shared")

            with T.thread()[0:1]:
                T.ptx.mbarrier.init(tma_mbar.ptr_to([0]), 1)
                T.ptx.mbarrier.init(mma_mbar.ptr_to([0]), 1)

            with T.warp()[0:1]:
                T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=cols_alloc, cta_group=2)
            tmem = T.decl_buffer((128, C_shape[1]), C_dtype, scope="tmem", allocated_addr=tmem_addr[0],
                                 layout=TileLayout(([128, C_shape[1]], [(1, "TLane"), (1, "TCol")])))
            T.ptx.fence.mbarrier_init()
            T.ptx.fence.proxy("shared")
            T.cuda.cta_sync()
            T.cuda.cluster_sync()

            tma_args = T.meta_var({"dispatch": "tma", "mbar": tma_mbar_cta_0.ptr_to([0]), "cta_group": 2})
            with T.thread()[0:1]:
                Tp.copy_async(A_smem[*r_smem_A_in], A[*get_global_region(A_shape_per_cta, transA, cbx)], **tma_args)
                Tp.copy_async(B_smem[*r_smem_B_in], B[*get_global_region(B_shape_per_cta, transB, cbx)], **tma_args)
                if cbx == 0:
                    T.ptx.mbarrier.arrive.expect_tx(tma_mbar.ptr_to([0]), total_bytes)

            if cbx == 0:
                T.ptx.mbarrier.try_wait(tma_mbar.ptr_to([0]), 0)
                T.ptx.tcgen05.fence.after_thread_sync()
                T.cuda.cta_sync()
                with T.thread()[0:1]:
                    Tp.gemm_async(tmem[*r_tmem_C], A_smem[*r_smem_A], B_smem[*r_smem_B], dispatch="tcgen05", cta_group=2)
                    T.ptx.tcgen05.commit(mma_mbar.ptr_to([0]), cta_group=2, cta_mask=3) # signal cta 1's mbarrier
            T.ptx.mbarrier.try_wait(mma_mbar.ptr_to([0]), 0) # both cta 0 and cta 1 have done mma
            T.ptx.tcgen05.fence.after_thread_sync()
            T.cuda.cta_sync()

            C_reg = T.alloc_local(width , dtype=C_dtype)
            C_view = C_reg.view(128, width, layout=TileLayout(([128, width], [1@axis_tid_in_wg, 1])))
            with T.warpgroup()[0:1]:
                Tp.copy(C_view[:, :], tmem[C_region[0][0]:C_region[0][1], C_region[1][0]:C_region[1][0] + width])
            T.cuda.cta_sync()
            with T.thread():
                Tp.copy(C[cbx * 128 +tid_in_wg, C_region[1][0]:C_region[1][0] + width], C_reg[:])
            T.cuda.cta_sync()

            with T.warp()[0:1]:
                T.ptx.tcgen05.relinquish_alloc_permit(cta_group=2)
                T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=cols_alloc, cta_group=2)
    # fmt: on

    dev = tvm.cuda(0)
    np.random.seed(0)

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": gemm_async})
        mod.show()
        mod = tvm.compile(mod, target=target, tir_pipeline="tirp")
        print(mod.mod.imports[0].inspect_source())

        A_np = np.random.randn(*A_shape).astype(A_dtype)
        B_np = np.random.randn(*B_shape).astype(B_dtype)
        C_np = np.zeros(C_shape, dtype=C_dtype)
        A_tvm = tvm.runtime.tensor(A_np, dev)
        B_tvm = tvm.runtime.tensor(B_np, dev)
        C_tvm = tvm.runtime.tensor(C_np, dev)
        mod["main"](A_tvm, B_tvm, C_tvm)

        C_ref = np.zeros(C_shape, dtype=C_dtype)
        A_ref = np.squeeze(A_np[*r_smem_A[:-2]] if not transA else A_np[*r_smem_A[:-2]].T)
        B_ref = np.squeeze(B_np[*r_smem_B[:-2]] if transB else B_np[*r_smem_B[:-2]].T)
        C_ref[:, C_region[1][0] : C_region[1][0] + width] = A_ref @ B_ref
        np.testing.assert_allclose(C_tvm.numpy(), C_ref, atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    tvm.testing.main()
