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
    K = 1024
    M = 4096
    N = 2048
    BLOCK_M = 2048
    BLOCK_N = 256
    BLOCK_K = 1024
    NUM_BLOCK_M = M // BLOCK_M
    NUM_BLOCK_N = N // BLOCK_N
    NUM_BLOCK_K = K // BLOCK_K
    dtype = "float16"
    # C = A.T @ B
    @T.prim_func(tirp=True)
    def matmul(A_ptr: T.handle, B_ptr: T.handle, C_ptr: T.handle):
        T.func_attr({"num_inputs": 2})
        # todo: specify allocation of A, B, C
        A = T.match_buffer(A_ptr, (M, K), dtype, layout = T.TileLayout.from_tuple((M, K), (1, M)))
        B = T.match_buffer(B_ptr, (K, N), dtype, layout=T.TileLayout.from_tuple(K * N))
        C = T.match_buffer(C_ptr, (M, N), dtype, layout=T.TileLayout.from_tuple(N * M))
        # todo: we should allow alloc_buffer under for loops
        # todo: can we skip the first T.kernel()?
        with T.kernel():
            result_tiles = T.alloc_buffer((M, BLOCK_N), "float32", scope="trn.sbuf", layout=T.TrainiumLayout("FPF", T.TileLayout.from_tuple((M // 128, 128, BLOCK_N), (BLOCK_N, 1, 1))))
            b_sbuf = T.alloc_buffer((BLOCK_K, BLOCK_N), dtype, scope="trn.sbuf", layout = T.TrainiumLayout("FPF", T.TileLayout.from_tuple((BLOCK_K // 128, 128, BLOCK_N), (BLOCK_N, 1, 1))))
            a_sbuf = T.alloc_buffer((BLOCK_M, BLOCK_K), dtype, scope="trn.sbuf", layout = T.TrainiumLayout("FFP", T.TileLayout.from_tuple((BLOCK_M, BLOCK_K // 128, 128), (1, BLOCK_M, 1))))
            c_psum = T.alloc_buffer((BLOCK_M, BLOCK_N), "float32", scope="trn.psum", layout = T.TrainiumPSUMLayout("FPF", T.TileLayout.from_tuple((BLOCK_M // 128, 128, BLOCK_N), (BLOCK_N, 1, 1))))
            for n in T.serial(NUM_BLOCK_N):
                Tp.memset(result_tiles, T.float32(0.0))
                for k in T.serial(NUM_BLOCK_K):
                    Tp.copy(b_sbuf, B[k * BLOCK_K : (k + 1) * BLOCK_K, n * BLOCK_N : (n + 1) * BLOCK_N])
                    for m in T.serial(NUM_BLOCK_M):
                        Tp.copy(a_sbuf, A[m * BLOCK_M : (m + 1) * BLOCK_M, k * BLOCK_K : (k + 1) * BLOCK_K])
                        Tp.memset(c_psum, T.float32(0.0))
                        Tp.gemm(c_psum, a_sbuf, b_sbuf, c_psum)
                        Tp.add(result_tiles[m*BLOCK_M:(m+1)*BLOCK_M, :], result_tiles[m*BLOCK_M:(m+1)*BLOCK_M, :], c_psum)
                Tp.copy(C[0:M, n*BLOCK_N:(n+1)*BLOCK_N], result_tiles)
    with target:
        mod = tvm.IRModule({"main": matmul})
        mod = tvm.tir.transform.LowerTIRp()(mod)
        func = mod["main"]
        run_on_remote_and_check_correct(func, lambda x, y: (x.reshape(K, M).T @ y,), target)


if __name__ == "__main__":
    test_gemm()
