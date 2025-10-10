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

import tvm
import tvm.testing
from tvm.script import tir as T
from tvm.script import tirp as Tp
from tvm.tir.transform import LowerTIRp

from .utils import run_on_remote_and_check_correct, ssh_client

target = tvm.target.Target("aws/trn1/trn1.2xlarge")


@pytest.mark.dependency(depends=["ssh_success"])
def test_gemm2(ssh_client):
    K = 4096
    M = 4096
    N = 4096
    BLOCK_M = 1024
    BLOCK_N = 1024
    BLOCK_K = 1024
    NUM_BLOCK_M = M // BLOCK_M
    NUM_BLOCK_N = N // BLOCK_N
    NUM_BLOCK_K = K // BLOCK_K
    TILE_M = 128
    TILE_N = 512
    NUM_TILE_M = BLOCK_M // TILE_M
    NUM_TILE_N = BLOCK_N // TILE_N
    dtype = "float16"

    # fmt: off
    @T.macro
    def mm(iter_id, a_sbuf, b_sbuf, c_psum, result_tiles):
        n = T.meta_var(iter_id // (NUM_BLOCK_M * NUM_BLOCK_K))
        k = T.meta_var(iter_id % NUM_BLOCK_K)
        m = T.meta_var(iter_id // NUM_BLOCK_K % NUM_BLOCK_M)
        for mi, ni in T.grid(NUM_TILE_M, NUM_TILE_N):
            psum_bank = T.meta_var((mi*NUM_TILE_N+ni)%8)
            TILE_N_START = T.meta_var(ni * TILE_N)
            TILE_M_START = T.meta_var(mi * TILE_M)
            # FIXME: currently nki has a bug for psum initialization
            # fix this when NKI exposes psum initialization API
            Tp.gemm(c_psum[psum_bank], a_sbuf[iter_id%3, :, TILE_M_START:TILE_M_START+TILE_M], b_sbuf[iter_id%3, :, TILE_N_START:TILE_N_START+TILE_N], c_psum[psum_bank], transpose_A=True)
            if k == 0:
                Tp.copy(result_tiles[(n*NUM_BLOCK_K+m)%2,
                    TILE_M_START : TILE_M_START+TILE_M,
                    TILE_N_START : TILE_N_START + TILE_N,
                ], c_psum[psum_bank])
            else:
                Tp.add(
                    result_tiles[(n*NUM_BLOCK_K+m)%2,
                        TILE_M_START : TILE_M_START+TILE_M,
                        TILE_N_START : TILE_N_START + TILE_N,
                    ],
                    result_tiles[(n*NUM_BLOCK_K+m)%2,
                        TILE_M_START : TILE_M_START+TILE_M,
                        TILE_N_START : TILE_N_START + TILE_N,
                    ],
                    c_psum[psum_bank],
                )

    @T.macro
    def load_A(iter_id, a_sbuf, A):
        m = T.meta_var(iter_id // NUM_BLOCK_K % NUM_BLOCK_M)
        k = T.meta_var(iter_id % NUM_BLOCK_K)
        Tp.copy(
            a_sbuf[iter_id % 3], A[ k * BLOCK_K : (k + 1) * BLOCK_K, (m) * BLOCK_M : (m + 1) * BLOCK_M]
        )

    @T.macro
    def load_B(iter_id, b_sbuf, B):
        n = T.meta_var(iter_id // (NUM_BLOCK_M * NUM_BLOCK_K))
        k = T.meta_var(iter_id % NUM_BLOCK_K)
        Tp.copy(
            b_sbuf[iter_id % 3],
            B[(k) * BLOCK_K : (k + 1) * BLOCK_K, n * BLOCK_N : (n + 1) * BLOCK_N],
        )
    # do not reuse memory load, and keep result_tiles small
    @T.prim_func(tirp=True)
    def matmul2(A_ptr: T.handle, B_ptr: T.handle, C_ptr: T.handle):
        T.func_attr({"num_inputs": 2})
        A = T.match_buffer(A_ptr, (K, M), dtype)
        B = T.match_buffer(B_ptr, (K, N), dtype)
        C = T.match_buffer(C_ptr, (M, N), dtype)
        with T.kernel():
            result_tiles = T.alloc_buffer((2, BLOCK_M, BLOCK_N), "float32", scope="trn.sbuf", layout="FPF")
            b_sbuf = T.alloc_buffer((3, BLOCK_K, BLOCK_N), dtype, scope="trn.sbuf", layout="FPF")
            a_sbuf = T.alloc_buffer((3, BLOCK_K, BLOCK_M), dtype, scope="trn.sbuf", layout="FPF")
            c_psum = T.alloc_buffer((8, TILE_M, TILE_N), "float32", scope="trn.psum", layout="FPF", allocated_addr=(0, 0))
            load_A(0, a_sbuf, A)
            load_B(0, b_sbuf, B)
            for iter_id in T.serial(NUM_BLOCK_M * NUM_BLOCK_K * NUM_BLOCK_N-1):
                m = T.meta_var(iter_id // NUM_BLOCK_K % NUM_BLOCK_M)
                k = T.meta_var(iter_id % NUM_BLOCK_K)
                n = T.meta_var(iter_id // (NUM_BLOCK_M * NUM_BLOCK_K))
                load_A(iter_id+1, a_sbuf, A)
                load_B(iter_id+1, b_sbuf, B)
                mm(iter_id, a_sbuf, b_sbuf, c_psum, result_tiles)
                if k == NUM_BLOCK_K-1:
                    Tp.copy(C[m*BLOCK_M:(m+1)*BLOCK_M, n*BLOCK_N:(n+1)*BLOCK_N], result_tiles[(n*NUM_BLOCK_K+m)%2])
            mm(NUM_BLOCK_M * NUM_BLOCK_K * NUM_BLOCK_N-1, a_sbuf, b_sbuf, c_psum, result_tiles)
            Tp.copy(C[(NUM_BLOCK_M-1)*BLOCK_M:(NUM_BLOCK_M-1)*BLOCK_M+BLOCK_M, (NUM_BLOCK_N-1)*BLOCK_N:(NUM_BLOCK_N-1)*BLOCK_N+BLOCK_N], result_tiles[(NUM_BLOCK_M * NUM_BLOCK_N-1)%2])
    # fmt: on
    with target:
        mod = tvm.IRModule({"main": matmul2})
        func = mod["main"]
        run_on_remote_and_check_correct(func, lambda x, y: (x.reshape(K, M).T @ y,), target)


if __name__ == "__main__":
    test_gemm2()
