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
from tvm.script import tir as T
from tvm.script import tirp as Tp
from tvm.tir.transform import LowerTIRp
import pytest
from .utils import run_on_remote_and_check_correct, ssh_client

target = tvm.target.Target("aws/trn1/trn1.2xlarge")


@pytest.mark.dependency(depends=["ssh_success"])
def test_gemm(ssh_client):
    K = 4096
    M = 4096
    N = 4096
    BLOCK_M = 2048
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
    def mm(n, k, m, a_sbuf, b_sbuf, c_psum, result_tiles):
        for mi, ni in T.grid(NUM_TILE_M, NUM_TILE_N):
            psum_bank = T.meta_var((mi*NUM_TILE_N+ni)%8)
            TILE_N_START = T.meta_var(ni * TILE_N)
            TILE_M_START = T.meta_var(mi * TILE_M)
            # FIXME: currently nki has a bug for psum initialization
            # fix this when NKI exposes psum initialization API
            Tp.gemm(c_psum[psum_bank], a_sbuf[m%2, TILE_M_START:TILE_M_START+TILE_M, :], b_sbuf[(n*NUM_BLOCK_K+k)%2, :, TILE_N_START:TILE_N_START+TILE_N], c_psum[psum_bank])
            if k == 0:
                Tp.copy(result_tiles[
                    m * BLOCK_M + TILE_M_START : m * BLOCK_M + TILE_M_START+TILE_M,
                    TILE_N_START : TILE_N_START + TILE_N,
                ], c_psum[psum_bank])
            else:
                Tp.add(
                    result_tiles[
                        m * BLOCK_M + TILE_M_START : m * BLOCK_M + TILE_M_START+TILE_M,
                        TILE_N_START : TILE_N_START + TILE_N,
                    ],
                    result_tiles[
                        m * BLOCK_M + TILE_M_START : m * BLOCK_M + TILE_M_START+TILE_M,
                        TILE_N_START : TILE_N_START + TILE_N,
                    ],
                    c_psum[psum_bank],
                )

    @T.macro
    def load_A(n, k, m, a_sbuf, A):
        Tp.copy(
            a_sbuf[m % 2], A[(m) * BLOCK_M : (m + 1) * BLOCK_M, k * BLOCK_K : (k + 1) * BLOCK_K]
        )

    @T.macro
    def load_B(n, k, b_sbuf, B):
        Tp.copy(
            b_sbuf[(n * NUM_BLOCK_K + k) % 2],
            B[(k) * BLOCK_K : (k + 1) * BLOCK_K, n * BLOCK_N : (n + 1) * BLOCK_N],
        )

    # C = A.T @ B
    @T.prim_func(tirp=True)
    def matmul(A_ptr: T.handle, B_ptr: T.handle, C_ptr: T.handle):
        T.func_attr({"num_inputs": 2})
        A = T.match_buffer(A_ptr, (M, K), dtype, layout = T.TileLayout.from_tuple((M, K), (1, M)))
        B = T.match_buffer(B_ptr, (K, N), dtype, layout=T.TileLayout.from_tuple(K * N))
        C = T.match_buffer(C_ptr, (M, N), dtype, layout=T.TileLayout.from_tuple(N * M))
        with T.kernel():
            result_tiles = T.alloc_buffer((M, BLOCK_N), dtype, scope="trn.sbuf", layout=T.TrainiumLayout("FPF", T.TileLayout.from_tuple((M // 128, 128, BLOCK_N), (BLOCK_N, 1, 1))), allocated_addr=0)
            b_sbuf = T.alloc_buffer((2, BLOCK_K, BLOCK_N), dtype, scope="trn.sbuf", layout = T.TrainiumLayout("FFPF", T.TileLayout.from_tuple((2, BLOCK_K // 128, 128, BLOCK_N), (BLOCK_K*BLOCK_N//128,BLOCK_N, 1, 1))), allocated_addr=M*BLOCK_N//128*2)
            a_sbuf = T.alloc_buffer((2, BLOCK_M, BLOCK_K), dtype, scope="trn.sbuf", layout = T.TrainiumLayout("FFFP", T.TileLayout.from_tuple((2, BLOCK_M, BLOCK_K // 128, 128), (BLOCK_M*BLOCK_K//128, 1, BLOCK_M, 1))), allocated_addr=M*BLOCK_N//128*2 + 2*BLOCK_K*BLOCK_N//128*2)
            c_psum = T.alloc_buffer((8, TILE_M, TILE_N), "float32", scope="trn.psum", layout = T.TrainiumPSUMLayout("FPF", T.TileLayout.from_tuple((8, TILE_M, TILE_N), (TILE_N, 1, 1))), allocated_addr=(0, 0))
            for n in T.serial(NUM_BLOCK_N):
                Tp.memset(result_tiles, T.float32(0.0))
                load_B(n, 0, b_sbuf, B)
                load_A(n, 0, 0, a_sbuf, A)
                for k in T.serial(NUM_BLOCK_K-1):
                    load_B(n, k+1, b_sbuf, B)
                    load_A(n, k+1, 0, a_sbuf, A)
                    for m in T.serial(NUM_BLOCK_M-1):
                        load_A(n, k, m+1, a_sbuf, A)
                        mm(n, k, m, a_sbuf, b_sbuf, c_psum, result_tiles)
                    mm(n,k,NUM_BLOCK_M-1, a_sbuf, b_sbuf, c_psum, result_tiles)
                for m in T.serial(NUM_BLOCK_M-1):
                    load_A(n, NUM_BLOCK_K-1, m+1, a_sbuf, A)
                    mm(n, NUM_BLOCK_K - 1, m, a_sbuf, b_sbuf, c_psum, result_tiles)
                mm(n, NUM_BLOCK_K - 1, NUM_BLOCK_M - 1, a_sbuf, b_sbuf, c_psum, result_tiles)
                Tp.copy(C[0:M, n*BLOCK_N:(n+1)*BLOCK_N], result_tiles)
    # fmt: on
    with target:
        mod = tvm.IRModule({"main": matmul})
        func = mod["main"]
        run_on_remote_and_check_correct(func, lambda x, y: (x.reshape(K, M).T @ y,), target)


if __name__ == "__main__":
    test_gemm()
