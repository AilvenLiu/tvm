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
import pytest
import tempfile

import tvm
import tvm.testing
from tvm.runtime import disco as di
from tvm.runtime import ShapeTuple
from tvm.script import tir as T
from tvm.exec import disco_worker as _  # pylint: disable=unused-import

NUM_WORKERS = 4
TARGET = tvm.target.Target("cuda")

# =================================================================
#  Test Class with Shared Session
# =================================================================


class TestNVSHMEM:
    """
    A test suite for NVSHMEM operations that shares a single Disco session
    among all test cases to prevent setup/teardown issues.
    """

    sess: di.Session = None

    @classmethod
    def setup_class(cls):
        """Set up a single disco session with nvshmem for all tests."""
        cls.sess = di.ProcessSession(num_workers=NUM_WORKERS)
        f_init_uid = tvm.get_global_func("runtime.disco.nvshmem.init_nvshmem_uid")
        uid = f_init_uid()
        init_func = cls.sess.get_global_func("runtime.disco.nvshmem.init_nvshmem")
        init_func(uid, NUM_WORKERS, 0)
        cls.sess.sync_worker_0()

    @classmethod
    def teardown_class(cls):
        """Tear down the shared disco session."""
        if cls.sess is not None:
            finalize_func = cls.sess.get_global_func("runtime.disco.nvshmem.finalize_nvshmem")
            finalize_func()
            cls.sess.sync_worker_0()
            cls.sess.shutdown()

    # =================================================================
    #  Helper Functions
    # =================================================================

    def run_prim_func(self, prim_func, *args):
        """Compile, export, load, and run a PrimFunc in the shared disco session."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/test.so"
            mod = tvm.compile(prim_func, target=TARGET, tir_pipeline="tirp")
            mod.export_library(path)
            rt_mod = self.sess.load_vm_module(path)
            rt_mod["main"](*args)
            self.sess._sync_all()

    def create_nvshmem_array(self, shape, dtype, init_data_fn=None, zero_out=True):
        """Create and optionally initialize an nvshmem-accessible DNDArray."""
        nvshmem_empty = self.sess.get_global_func("runtime.disco.nvshmem.empty")
        arr = nvshmem_empty(ShapeTuple(shape), dtype, None)

        if init_data_fn:
            for i in range(NUM_WORKERS):
                arr.debug_copy_from(i, init_data_fn(i, shape, dtype))
        elif zero_out:
            zero_data = np.zeros(shape, dtype=dtype)
            for i in range(NUM_WORKERS):
                arr.debug_copy_from(i, zero_data)

        return arr

    # =================================================================
    #  Test Cases
    # =================================================================

    def test_thread_info(self):
        """Tests the my_pe() and n_pes() intrinsics."""

        @T.prim_func(tirp=True)
        def main(res: T.Buffer((2,), "int32")):
            with T.kernel():
                bx = T.cta_id([1], parent="kernel")
                tx = T.thread_id([1], parent="cta")
                with T.thread():
                    res[0] = T.nvshmem.my_pe()
                    res[1] = T.nvshmem.n_pes()

        res_array = self.sess.empty((2,), "int32")
        self.run_prim_func(main, res_array)

        for i in range(NUM_WORKERS):
            res = res_array.debug_get_from_remote(i).numpy()
            np.testing.assert_equal(res, [i, NUM_WORKERS])

    @pytest.mark.parametrize(
        "scope, shape, nwarps, nelems, op_name",
        [
            ("thread", (32,), 1, 4, "getmem_nbi"),
            ("thread", (32,), 1, 4, "putmem_nbi"),
            ("warp", (64,), 2, 4 * 32, "getmem_nbi"),
            ("warp", (64,), 2, 4 * 32, "putmem_nbi"),
            ("block", (64,), 2, 4 * 64, "getmem_nbi"),
            ("block", (64,), 2, 4 * 64, "putmem_nbi"),
        ],
    )
    def test_transfer(self, scope, shape, nwarps, nelems, op_name):
        """Tests data transfer operations (get/put) at thread, warp, and block scopes."""
        dtype = "float32"
        is_get = "get" in op_name
        op_func = getattr(T.nvshmem, op_name)
        if scope != "thread":
            op_func = getattr(op_func, scope)

        @T.prim_func(tirp=True)
        def main(A: T.Buffer(shape, dtype), B: T.Buffer(shape, dtype)):
            with T.kernel():
                bx = T.cta_id([1], parent="kernel")
                warp_id = T.warp_id([nwarps], parent="cta")
                lane_id = T.thread_id([32], parent="warp")
                tid = T.thread_id([nwarps * 32], parent="cta")

                with T.thread():
                    my_pe = T.nvshmem.my_pe()
                    n_pes = T.nvshmem.n_pes()
                    offset = T.if_then_else(
                        scope == "block", 0, T.if_then_else(scope == "thread", tid, warp_id * 32)
                    )
                    op_func(
                        dst=B.access_ptr("w", offset=offset),
                        src=A.access_ptr("r", offset=offset),
                        nelems=nelems,
                        pe=(my_pe + 1) % n_pes,
                    )
                    T.nvshmem.quiet()

        init_fn = lambda i, s, d: np.arange(s[0], dtype=d) + i * 100
        A_array = self.create_nvshmem_array(shape, dtype, init_fn)
        B_array = self.create_nvshmem_array(shape, dtype)
        self.sess.sync_worker_0()
        self.run_prim_func(main, A_array, B_array)

        for i in range(NUM_WORKERS):
            if is_get:
                expected_B = A_array.debug_get_from_remote((i + 1) % NUM_WORKERS).numpy()
                actual_B = B_array.debug_get_from_remote(i).numpy()
            else:  # put
                expected_B = A_array.debug_get_from_remote(i).numpy()
                actual_B = B_array.debug_get_from_remote((i + 1) % NUM_WORKERS).numpy()
            np.testing.assert_equal(actual_B, expected_B)

    @pytest.mark.parametrize("sig_op", ["set", "add"])
    def test_signal_op(self, sig_op):
        """Tests signal_op and wait_until to implement a barrier-like pattern."""
        cmp_value = 1 if sig_op == "set" else 2

        @T.prim_func(tirp=True)
        def main(res: T.Buffer((1,), "uint64")):
            with T.kernel():
                bx = T.cta_id([1], parent="kernel")
                tid = T.thread_id([1], parent="cta")
                with T.thread():
                    my_pe = T.nvshmem.my_pe()
                    n_pes = T.nvshmem.n_pes()
                    dst_pe = (my_pe + 1) % n_pes
                    if sig_op == "add":
                        res[0] = 1
                    T.nvshmem.barrier_all()
                    T.nvshmem.signal_op(
                        sig_addr=res.access_ptr("w"),
                        signal=1,
                        sig_op=sig_op,
                        pe=dst_pe,
                    )
                    T.nvshmem.wait_until(
                        ivar=res.access_ptr("r"),
                        cmp="eq",
                        cmp_value=cmp_value,
                    )

        res_array = self.create_nvshmem_array((1,), "uint64")
        self.sess.sync_worker_0()
        self.run_prim_func(main, res_array)

        for i in range(NUM_WORKERS):
            res = res_array.debug_get_from_remote(i).numpy()
            if sig_op == "set":
                np.testing.assert_equal(res[0], 1)
            elif sig_op == "add":
                np.testing.assert_equal(res[0], 2)

    @pytest.mark.parametrize(
        "scope, shape, nwarps, nelems, cmp_value",
        [
            ("thread", (32,), 1, 4, 32),
            ("warp", (64,), 2, 4 * 32, 2),
            ("block", (64,), 2, 4 * 64, 1),
        ],
    )
    def test_put_signal(self, scope, shape, nwarps, nelems, cmp_value):
        """Tests combined data transfer and signal operations at thread/warp/block scopes."""
        dtype = "float32"
        op_func = getattr(T.nvshmem, "putmem_signal_nbi")
        if scope != "thread":
            op_func = getattr(op_func, scope)

        @T.prim_func(tirp=True)
        def main(
            A: T.Buffer(shape, dtype),
            B: T.Buffer(shape, dtype),
            signal_array: T.Buffer((1,), "uint64"),
        ):
            with T.kernel():
                bx = T.cta_id([1], parent="kernel")
                warp_id = T.warp_id([nwarps], parent="cta")
                lane_id = T.thread_id([32], parent="warp")
                tid = T.thread_id([nwarps * 32], parent="cta")

                with T.thread():
                    my_pe = T.nvshmem.my_pe()
                    n_pes = T.nvshmem.n_pes()
                    dst_pe = (my_pe + 1) % n_pes
                    offset = T.if_then_else(
                        scope == "block", 0, T.if_then_else(scope == "thread", tid, warp_id * 32)
                    )
                    op_func(
                        dst=B.access_ptr("w", offset=offset),
                        src=A.access_ptr("r", offset=offset),
                        nelems=nelems,
                        sig_addr=signal_array.access_ptr("w", offset=0),
                        signal=1,
                        sig_op="set",
                        pe=dst_pe,
                    )
                    T.nvshmem.wait_until(
                        ivar=signal_array.access_ptr("r", offset=0),
                        cmp="eq",
                        cmp_value=cmp_value,
                    )

        init_A = lambda i, s, d: np.arange(s[0], dtype=d) + i * 100
        A_array = self.create_nvshmem_array(shape, dtype, init_A)
        B_array = self.create_nvshmem_array(shape, dtype)
        signal_array = self.create_nvshmem_array((1,), "uint64")

        self.sess.sync_worker_0()
        self.run_prim_func(main, A_array, B_array, signal_array)

        for i in range(NUM_WORKERS):
            expected = A_array.debug_get_from_remote(i).numpy()
            actual = B_array.debug_get_from_remote((i + 1) % NUM_WORKERS).numpy()
            signal_np = signal_array.debug_get_from_remote(i).numpy()
            np.testing.assert_equal(actual, expected)
            np.testing.assert_equal(signal_np[0], cmp_value)

    def test_fence_barrier(self):
        """Tests fence and barrier operations."""
        shape = (64,)
        dtype = "float32"

        @T.prim_func(tirp=True)
        def main(
            A: T.Buffer(shape, dtype), B: T.Buffer(shape, dtype), res: T.Buffer((1,), "uint64")
        ):
            with T.kernel():
                bx = T.cta_id([1], parent="kernel")
                warp_id = T.warp_id([2], parent="cta")
                lane_id = T.thread_id([32], parent="warp")
                tid = T.thread_id([2 * 32], parent="cta")

                with T.thread():
                    my_pe = T.nvshmem.my_pe()
                    n_pes = T.nvshmem.n_pes()
                    dst_pe = (my_pe + 1) % n_pes
                    T.nvshmem.barrier_all()
                    T.nvshmem.putmem_nbi.block(
                        dst=B.access_ptr("w", offset=0),
                        src=A.access_ptr("r", offset=0),
                        nelems=4 * 64,
                        pe=(my_pe + 1) % n_pes,
                    )
                    T.nvshmem.fence()
                    if tid == 0:
                        T.nvshmem.signal_op(
                            sig_addr=res.access_ptr("w"),
                            signal=1,
                            sig_op="set",
                            pe=dst_pe,
                        )
                    T.nvshmem.wait_until(
                        ivar=res.access_ptr("r"),
                        cmp="eq",
                        cmp_value=1,
                    )

        init_fn = lambda i, s, d: np.arange(s[0], dtype=d) + i * 100
        A_array = self.create_nvshmem_array(shape, dtype, init_fn)
        B_array = self.create_nvshmem_array(shape, dtype)
        res_array = self.create_nvshmem_array((1,), "uint64")
        self.sess.sync_worker_0()
        self.run_prim_func(main, A_array, B_array, res_array)

        for i in range(NUM_WORKERS):
            expected_B = A_array.debug_get_from_remote(i).numpy()
            actual_B = B_array.debug_get_from_remote((i + 1) % NUM_WORKERS).numpy()
            np.testing.assert_equal(actual_B, expected_B)

    def test_ring_allgather(self):
        """Tests ring allgather algorithm."""
        A_shape = (32,)
        B_shape = (NUM_WORKERS, 32)
        res_shape = (NUM_WORKERS,)
        dtype = "float32"

        @T.prim_func(tirp=True)
        def main(
            A: T.Buffer(A_shape, dtype),
            B: T.Buffer(B_shape, dtype),
            res: T.Buffer(res_shape, "uint64"),
        ):
            with T.kernel():
                bx = T.cta_id([1], parent="kernel")
                warp_id = T.warp_id([1], parent="cta")
                lane_id = T.thread_id([32], parent="warp")
                tid = T.thread_id([32], parent="cta")

                with T.thread():
                    my_pe = T.nvshmem.my_pe()
                    n_pes = T.nvshmem.n_pes()
                    B[my_pe, tid] = A[tid]
                    T.nvshmem.barrier_all()

                    for step in range(0, NUM_WORKERS - 1):
                        dst_pe = (my_pe + NUM_WORKERS - 1) % n_pes
                        send_idx = (my_pe + step) % n_pes
                        recv_idx = (send_idx + 1) % n_pes
                        T.nvshmem.putmem_signal_nbi.block(
                            dst=B.access_ptr("w", offset=B.offset_of_p([send_idx, 0])),
                            src=B.access_ptr("r", offset=B.offset_of_p([send_idx, 0])),
                            nelems=32 * 4,
                            sig_addr=res.access_ptr("w", offset=send_idx),
                            signal=1,
                            sig_op="set",
                            pe=dst_pe,
                        )
                        T.nvshmem.wait_until(
                            ivar=res.access_ptr("r", offset=recv_idx),
                            cmp="eq",
                            cmp_value=1,
                        )

        init_fn = lambda i, s, d: np.arange(s[0], dtype=d) + i * 100
        A_array = self.create_nvshmem_array(A_shape, dtype, init_fn)
        B_array = self.create_nvshmem_array(B_shape, dtype)
        res_array = self.create_nvshmem_array(res_shape, "uint64")
        self.sess.sync_worker_0()
        self.run_prim_func(main, A_array, B_array, res_array)

        expected = np.stack([A_array.debug_get_from_remote(i).numpy() for i in range(NUM_WORKERS)])
        for i in range(NUM_WORKERS):
            output = B_array.debug_get_from_remote(i).numpy()
            np.testing.assert_equal(output, expected)


if __name__ == "__main__":
    tvm.testing.main()
