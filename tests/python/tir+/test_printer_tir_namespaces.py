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

import tvm
from tvm import tir
from tvm.script import tir as T  # noqa: F401 - ensure namespace registration on import


def _assert_print(obj, expected):
    assert obj.script(verbose_expr=True).strip() == expected.strip()


def test_printer_cuda_namespace_printf():
    node = tir.Evaluate(tir.op.cuda_printf("x=%d", tir.IntImm("int32", 1)))
    _assert_print(node, 'T.cuda.printf("x=%d", 1)')


def test_printer_ptx_namespace_wgmma_commit_group():
    node = tir.Evaluate(tir.op.ptx_wgmma_commit_group())
    _assert_print(node, "T.ptx.wgmma.commit_group()")


def test_printer_ptx_namespace_barrier_cluster():
    node = tir.Evaluate(tir.op.ptx_barrier_cluster_arrive())
    _assert_print(node, 'T.ptx.barrier.cluster.arrive("", T.bool(True))')


def test_printer_ptx_namespace_cp_async_wait_group():
    node = tir.Evaluate(tir.op.ptx_cp_async_wait_group(tir.IntImm("int32", 0)))
    _assert_print(node, "T.ptx.cp_async.wait_group(0)")


def test_printer_nvshmem_namespace():
    node = tir.Evaluate(tir.op.nvshmem_fence())
    _assert_print(node, "T.nvshmem.fence()")


def test_printer_ptx_more():
    r = tir.Var("r", "handle")
    s = tir.Var("s", "handle")
    _assert_print(
        tir.op.ptx_ldmatrix(True, 1, ".b16", r, s),
        'r = T.handle()\n'
        's = T.handle()\n'
        'T.ptx.ldmatrix("void", T.bool(True), 1, ".b16", r, s)',
    )
    _assert_print(
        tir.op.ptx_stmatrix(1, False, s, r),
        's = T.handle()\n'
        'r = T.handle()\n'
        'T.ptx.stmatrix(1, T.bool(False), s, r)',
    )
    _assert_print(tir.op.ptx_setmaxnreg(True, 64), "T.ptx.setmaxnreg(T.bool(True), 64)")
    _assert_print(tir.op.ptx_fetch_register(32, "laneid"), 'T.ptx.fetch_register(32, "laneid")')
    _assert_print(tir.op.ptx_wgmma_fence(), "T.ptx.wgmma.fence()")
    _assert_print(tir.op.ptx_wgmma_wait_group(0), "T.ptx.wgmma.wait_group(0)")
    _assert_print(tir.op.ptx_cp_async_commit_group(), "T.ptx.cp_async.commit_group()")
    _assert_print(tir.op.ptx_cp_async_bulk_commit_group(), "T.ptx.cp_async.bulk.commit_group()")
    _assert_print(
        tir.op.ptx_cp_async_bulk_wait_group(0, True),
        "T.ptx.cp_async.bulk.wait_group(0, T.bool(True))",
    )
    _assert_print(tir.op.ptx_cp_async_mbarrier_arrive(0), "T.ptx.cp_async.mbarrier.arrive(0)")
    _assert_print(tir.op.ptx_fence_proxy("async"), 'T.ptx.fence.proxy("async")')
    _assert_print(tir.op.ptx_fence_mbarrier_init_release_cluster(), "T.ptx.fence.mbarrier_init()")
    _assert_print(tir.op.ptx_elect_sync(), "T.ptx.elect_sync(T.int64(4294967295))")
    _assert_print(
        tir.op.ptx_ld_global_acquire(r, s),
        'r = T.handle()\n'
        's = T.handle()\n'
        'T.ptx.ld_global_acquire(r, s)'
    )
    _assert_print(
        tir.op.ptx_map_shared_rank(r, 2),
        'r = T.handle()\nT.ptx.map_shared_rank(r, 2)'
    )
    _assert_print(tir.op.ptx_bar_arrive(0, 128), "T.ptx.bar.arrive(0, 128)")
    _assert_print(tir.op.ptx_bar_sync(0, 128), "T.ptx.bar.sync(0, 128)")
    _assert_print(
        tir.op.ptx_tcgen05_alloc(s, 64, 1),
        's = T.handle()\nT.ptx.tcgen05.alloc(s, 64, 1)'
    )
    _assert_print(
        tir.op.ptx_tcgen05_dealloc(s, 64, 1),
        's = T.handle()\nT.ptx.tcgen05.dealloc(s, 64, 1)'
    )
    d = tir.Var("d", "handle")
    a = tir.Var("a", "handle")
    b = tir.Var("b", "handle")
    _assert_print(
        tir.op.ptx_tcgen05_encode_matrix_descriptor(d, a, 1, 2, 0),
        'd = T.handle()\n'
        'a = T.handle()\n'
        'T.ptx.tcgen05.encode_matrix_descriptor(d, a, 1, 2, 0)'
    )
    _assert_print(
        tir.op.ptx_tcgen05_encode_instr_descriptor(d, "f16", "f16", "f16", 16, 16, 16, True, False, 1, False, False, False, False),
        'd = T.handle()\nT.ptx.tcgen05.encode_instr_descriptor(d, "f16", "f16", "f16", 16, 16, 16, T.bool(True), T.bool(False), 1, T.bool(False), T.bool(False), T.bool(False), T.bool(False))'
    )
    _assert_print(
        tir.op.ptx_tcgen05_encode_instr_descriptor_block_scaled(d, "f16", "f16", "f16", "f16", a, b, 16, 16, 16, True, False, 1, False, False, False),
        'd = T.handle()\n'
        'a = T.handle()\n'
        'b = T.handle()\n'
        'T.ptx.tcgen05.encode_instr_descriptor_block_scaled(d, "f16", "f16", "f16", "f16", a, b, 16, 16, 16, T.bool(True), T.bool(False), 1, T.bool(False), T.bool(False), T.bool(False), T.bool(False))'
    )
    _assert_print(
        tir.op.ptx_tcgen05_cp(a, 0, 0, d, "64x128b", "f16", "f16", 1, ""),
        'a = T.handle()\n'
        'd = T.handle()\n'
        'T.ptx.tcgen05.cp(a, 0, 0, d, "64x128b", "f16", "f16", 1, "")'
    )
    _assert_print(
        tir.op.ptx_tcgen05_shift(a, 1),
        'a = T.handle()\nT.ptx.tcgen05.shift(a, 1)'
    )
    _assert_print(
        tir.op.ptx_tcgen05_ld(a, 0, 0, "64x128b", 1, False, 0),
        'a = T.handle()\nT.ptx.tcgen05.ld(a, 0, 0, "64x128b", 1, T.bool(False), 0)'
    )
    _assert_print(
        tir.op.ptx_tcgen05_st(a, 0, 0, "64x128b", 1, False, 0),
        'a = T.handle()\nT.ptx.tcgen05.st(a, 0, 0, "64x128b", 1, T.bool(False), 0)'
    )
    _assert_print(tir.op.ptx_tcgen05_wait_ld(), "T.ptx.tcgen05.wait.ld()")
    _assert_print(tir.op.ptx_tcgen05_wait_st(), "T.ptx.tcgen05.wait.st()")
    _assert_print(
        tir.op.ptx_tcgen05_commit(a, 1, 0),
        'a = T.handle()\nT.ptx.tcgen05.commit(a, 1, 0)'
    )
    _assert_print(tir.op.ptx_tcgen05_relinquish_alloc_permit(1), "T.ptx.tcgen05.relinquish_alloc_permit(1)")


def test_printer_ptx_mbarrier():
    bar = tir.Var("bar", "handle")
    _assert_print(
        tir.op.ptx_mbarrier_init(bar, 32),
        'bar = T.handle()\nT.ptx.mbarrier.init(bar, 32)'
    )
    _assert_print(
        tir.op.ptx_mbarrier_arrive(bar),
        'bar = T.handle()\nT.ptx.mbarrier.arrive(bar)'
    )
    _assert_print(
        tir.op.ptx_mbarrier_arrive_expect_tx(bar, 128),
        'bar = T.handle()\nT.ptx.mbarrier.arrive.expect_tx(bar, 128)'
    )
    _assert_print(
        tir.op.ptx_mbarrier_try_wait(bar, 1),
        'bar = T.handle()\nT.ptx.mbarrier.try_wait(bar, 1)'
    )
    _assert_print(tir.op.ptx_barrier_cluster_wait(False, True), "T.ptx.barrier.cluster.wait(T.bool(False), T.bool(True))")


def test_printer_cuda_more():
    p = tir.Var("p", "handle")
    _assert_print(tir.op.cuda_thread_fence(), "T.cuda.thread_fence()")
    _assert_print(tir.op.cuda_warp_sync(), "T.cuda.warp_sync()")
    _assert_print(tir.op.cuda_block_sync(), "T.cuda.block_sync()")
    _assert_print(tir.op.cuda_grid_sync(), "T.cuda.grid_sync()")
    _assert_print(tir.op.cuda_syncthreads_and(1), "T.cuda.syncthreads_and(1)")
    _assert_print(tir.op.cuda_syncthreads_or(1), "T.cuda.syncthreads_or(1)")
    _assert_print(tir.op.cuda_nano_sleep(100), "T.cuda.nano_sleep(100)")
    _assert_print(
        tir.op.cuda_atomic_add(p, tir.IntImm("int32", 1)),
        'p = T.handle()\nT.cuda.atomic_add(p, 1)'
    )
    _assert_print(
        tir.op.cuda_atomic_cas(p, 1, 2),
        'p = T.handle()\nT.cuda.atomic_cas(p, 1, 2)'
    )
    _assert_print(
        tir.op.cuda_ldg(p, "float32"),
        'p = T.handle()\nT.cuda.ldg(p, "float32")'
    )
    _assert_print(
        tir.op.cuda_func_call("f", 1, source_code=""),
        'T.cuda.func_call("f", 1, "")'
    )


def test_printer_nvshmem_more():
    p = tir.Var("p", "handle")
    _assert_print(tir.op.nvshmem_my_pe(), "T.nvshmem.my_pe()")
    _assert_print(tir.op.nvshmem_n_pes(), "T.nvshmem.n_pes()")
    _assert_print(
        tir.op.nvshmem_signal_op(p, 1, "set", 0),
        'p = T.handle()\nT.nvshmem.signal_op(p, 1, "set", 0)'
    )
    _assert_print(
        tir.op.nvshmem_wait_until(p, "eq", 0),
        'p = T.handle()\nT.nvshmem.wait_until(p, "eq", 0, "uint64_t")'
    )
    _assert_print(tir.op.nvshmem_quiet(), "T.nvshmem.quiet()")
    _assert_print(tir.op.nvshmem_barrier_all(), "T.nvshmem.barrier_all()")
    _assert_print(
        tir.op.nvshmem_getmem_nbi(p, p, 16, 0),
        'p = T.handle()\nT.nvshmem.getmem_nbi(p, p, 16, 0)'
    )
    _assert_print(
        tir.op.nvshmem_getmem_nbi_warp(p, p, 16, 0),
        'p = T.handle()\nT.nvshmem.getmem_nbi.warp(p, p, 16, 0)'
    )
    _assert_print(
        tir.op.nvshmem_putmem_nbi_block(p, p, 16, 0),
        'p = T.handle()\nT.nvshmem.putmem_nbi.block(p, p, 16, 0)'
    )
    _assert_print(
        tir.op.nvshmem_putmem_nbi(p, p, 16, 0),
        'p = T.handle()\nT.nvshmem.putmem_nbi(p, p, 16, 0)'
    )
    _assert_print(
        tir.op.nvshmem_putmem_nbi_warp(p, p, 16, 0),
        'p = T.handle()\nT.nvshmem.putmem_nbi.warp(p, p, 16, 0)'
    )
    _assert_print(
        tir.op.nvshmem_putmem_signal_nbi(p, p, 16, p, 1, "set", 0),
        'p = T.handle()\nT.nvshmem.putmem_signal_nbi(p, p, 16, p, 1, "set", 0)'
    )
    _assert_print(
        tir.op.nvshmem_putmem_signal_nbi_warp(p, p, 16, p, 1, "set", 0),
        'p = T.handle()\nT.nvshmem.putmem_signal_nbi.warp(p, p, 16, p, 1, "set", 0)'
    )
    _assert_print(
        tir.op.nvshmem_putmem_signal_nbi_block(p, p, 16, p, 1, "set", 0),
        'p = T.handle()\nT.nvshmem.putmem_signal_nbi.block(p, p, 16, p, 1, "set", 0)'
    )


def test_printer_nki_namespace():
    A = tir.decl_buffer([1], dtype="float16", name="A")
    B = tir.decl_buffer([1], dtype="float16", name="B")
    a0 = A[0]
    b0 = B[0]
    _assert_print(
        tir.op.nki_load(a0, b0),
        'A = T.Buffer((1,), "float16", layout=None)\n'
        'B = T.Buffer((1,), "float16", layout=None)\n'
        'T.nki.load(A[0], B[0])',
    )
    _assert_print(
        tir.op.nki_store(a0, b0),
        'A = T.Buffer((1,), "float16", layout=None)\n'
        'B = T.Buffer((1,), "float16", layout=None)\n'
        'T.nki.store(A[0], B[0])',
    )
    _assert_print(
        tir.op.nki_tensor_copy(a0, b0),
        'A = T.Buffer((1,), "float16", layout=None)\n'
        'B = T.Buffer((1,), "float16", layout=None)\n'
        'T.nki.tensor_copy(A[0], B[0])',
    )
    _assert_print(
        tir.op.nki_matmul(a0, a0, b0),
        'A = T.Buffer((1,), "float16", layout=None)\n'
        'B = T.Buffer((1,), "float16", layout=None)\n'
        'T.nki.matmul(A[0], A[0], B[0], T.bool(True))',
    )
    _assert_print(
        tir.op.nki_activation(a0, b0, "relu", 0.0, 1.0),
        'A = T.Buffer((1,), "float16", layout=None)\n'
        'B = T.Buffer((1,), "float16", layout=None)\n'
        'T.nki.activation(A[0], B[0], "relu", T.float32(0.0), T.float32(1.0))',
    )
    _assert_print(
        tir.op.nki_memset(a0, 0),
        'A = T.Buffer((1,), "float16", layout=None)\n'
        'T.nki.memset(A[0], 0)',
    )
    _assert_print(
        tir.op.nki_identity(a0, 1),
        'A = T.Buffer((1,), "float16", layout=None)\n'
        'T.nki.identity(A[0], 1)',
    )
    _assert_print(
        tir.op.nki_reciprocal(a0, b0),
        'A = T.Buffer((1,), "float16", layout=None)\n'
        'B = T.Buffer((1,), "float16", layout=None)\n'
        'T.nki.reciprocal(A[0], B[0])',
    )
    _assert_print(
        tir.op.nki_tensorreduce(a0, b0, "sum", False, 0),
        'A = T.Buffer((1,), "float16", layout=None)\n'
        'B = T.Buffer((1,), "float16", layout=None)\n'
        'T.nki.tensorreduce(A[0], B[0], "sum", T.bool(False), 0)',
    )
    _assert_print(
        tir.op.nki_tensortensor(a0, a0, b0, "add"),
        'A = T.Buffer((1,), "float16", layout=None)\n'
        'B = T.Buffer((1,), "float16", layout=None)\n'
        'T.nki.tensortensor(A[0], A[0], B[0], "add")',
    )
    _assert_print(
        tir.op.nki_tensorscalar(a0, a0, 1.0, "mul", False),
        'A = T.Buffer((1,), "float16", layout=None)\n'
        'T.nki.tensorscalar(A[0], A[0], T.float32(1.0), "mul", T.bool(False))',
    )
    _assert_print(
        tir.op.nki_tensorscalar_reduce(a0, a0, 1.0, "mul", "sum", False),
        'A = T.Buffer((1,), "float16", layout=None)\n'
        'T.nki.tensorscalar_reduce(A[0], A[0], T.float32(1.0), "mul", "sum", T.bool(False), T.bool(False))',
    )
    _assert_print(
        tir.op.nki_scalar_tensor_tensor(a0, a0, 1.0, a0, "add", "add"),
        'A = T.Buffer((1,), "float16", layout=None)\n'
        'T.nki.scalar_tensor_tensor(A[0], A[0], T.float32(1.0), A[0], "add", "add", T.bool(False), T.bool(False))',
    )
    _assert_print(
        tir.op.nki_scalar_tensor_scalar(a0, a0, 1.0, 1.0, "add", "add"),
        'A = T.Buffer((1,), "float16", layout=None)\n'
        'T.nki.scalar_tensor_scalar(A[0], A[0], T.float32(1.0), T.float32(1.0), "add", "add", T.bool(False), T.bool(False))',
    )
    _assert_print(
        tir.op.nki_activation_reduce(a0, a0, b0, "relu", "sum", 0.0, 1.0),
        'A = T.Buffer((1,), "float16", layout=None)\n'
        'B = T.Buffer((1,), "float16", layout=None)\n'
        'T.nki.activation_reduce(A[0], A[0], B[0], "relu", "sum", T.float32(0.0), T.float32(1.0))',
    )
    _assert_print(
        tir.op.nki_affine_select(a0, a0, a0, 1.0),
        'A = T.Buffer((1,), "float16", layout=None)\n'
        'T.nki.affine_select(A[0], A[0], A[0], T.float32(1.0))',
    )


def test_printer_ptx_mma_and_wgmma():
    r = tir.Var("r", "handle")
    d = tir.Var("d", "handle")
    a = tir.Var("a", "handle")
    b = tir.Var("b", "handle")
    _assert_print(
        tir.op.ptx_mma("m8n8k4", "row", "row", "fp16", "fp16", "fp16", "fp16", r, r, r, 0, False),
        'r = T.handle()\nT.ptx.mma("void", "m8n8k4", "row", "row", "fp16", "fp16", "fp16", "fp16", r, r, r, 0, T.bool(False))'
    )
    _assert_print(
        tir.op.ptx_wgmma_encode_matrix_descriptor(d, a, 1, 1, 0),
        'd = T.handle()\n'
        'a = T.handle()\n'
        'T.ptx.wgmma.encode_matrix_descriptor(d, a, 1, 1, 0)'
    )
    _assert_print(tir.op.ptx_wgmma_noop_barrier(0), "T.ptx.wgmma.noop_barrier(0)")
    _assert_print(
        tir.op.ptx_wgmma_mma_async_ss(16, 16, 16, "f16", "f16", True, False, 1.0, 1.0, True, d, d, 0, 0),
        'd = T.handle()\nT.ptx.wgmma.mma_async.ss(16, 16, 16, "f16", "f16", T.bool(True), T.bool(False), T.float32(1.0), T.float32(1.0), T.bool(True), d, d, 0, 0)'
    )
    _assert_print(
        tir.op.ptx_wgmma_mma_async_rs(16, 16, 16, "f16", "f16", True, False, 1.0, 1.0, True, d, 0, 0),
        'd = T.handle()\nT.ptx.wgmma.mma_async.rs(16, 16, 16, "f16", "f16", T.bool(True), T.bool(False), T.float32(1.0), T.float32(1.0), T.bool(True), d, 0, 0)'
    )


def test_printer_ptx_cp_async_tensor():
    tmap = tir.Var("tm", "handle")
    _assert_print(
        tir.op.ptx_cp_async_bulk_tensor_global_to_cluster(2, tmap, 0, tmap, 0, 0, 0, 0, 1, ""),
        'tm = T.handle()\nT.ptx.cp_async.bulk.tensor.g2c(2, tm, 0, tm, 0, 0, 0, 0, 1, "", 0, 1, "")'
    )
    _assert_print(
        tir.op.ptx_cp_async_bulk_tensor_global_to_cluster_prefetch(2, tmap, 0, 0, 0, ""),
        'tm = T.handle()\nT.ptx.cp_async.bulk.tensor.g2c_prefetch(2, tm, 0, 0, 0, "", "")'
    )
    _assert_print(
        tir.op.ptx_cp_async_bulk_tensor_shared_to_global(2, 0, tmap, 0, 0, 0, ""),
        'tm = T.handle()\nT.ptx.cp_async.bulk.tensor.s2g(2, 0, tm, 0, 0, 0, "", "")'
    )
    _assert_print(
        tir.op.ptx_cp_async_bulk_tensor_shared_to_global_reduce(2, 0, tmap, 0, 0, 0, "", "add"),
        'tm = T.handle()\nT.ptx.cp_async.bulk.tensor.s2g_reduce(2, 0, tm, 0, 0, 0, "", "add", "", "add")'
    )


def test_printer_ptx_cp_async_call():
    sh = tir.Var("sh", "handle")
    gl = tir.Var("gl", "handle")
    _assert_print(
        tir.op.ptx_cp_async(sh, gl, 16, "", -1, -1, ""),
        'sh = T.handle()\n'
        'gl = T.handle()\n'
        'T.ptx.cp_async("void", sh, gl, 16, "", -1, -1, "")'
    )
