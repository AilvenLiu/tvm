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
import pytest
import math
import numpy as np

import tvm
from tvm.script import tir as T, ir as I
import tvm.testing
from tvm.tir.transform import LowerTIRp

def test_nki_add_1():
    # fmt: off
    @T.prim_func
    def func(A: T.Buffer((128, 512)), B: T.Buffer((128, 512))):
        A_sbuf = T.alloc_buffer((128, 512), "float32", scope="trn.sbuf",)
        B_sbuf = T.alloc_buffer((128, 512), "float32", scope="trn.sbuf",)
        with T.attr(0, "tensorized_nki_instruction", 1):
            for i in range(0, 128):
                for j in range(0, 512):
                    A_sbuf[i, j] =  A[i, j]
        with T.attr(0, "tensorized_nki_instruction", 1):
            for i in range(0, 128):
                for j in range(0, 512):
                    B_sbuf[i, j] = A_sbuf[i, j] + 1
        with T.attr(0, "tensorized_nki_instruction", 1):
            for i in range(0, 128):
                for j in range(0, 512):
                    B[i, j] =  B_sbuf[i, j]
            
    target = tvm.target.Target("aws/trn1/trn1.2xlarge")
    mod = tvm.IRModule({"main": func})
    mod = tvm.tir.transform.DecorateDeviceScope()(mod)
    with tvm.transform.PassContext(config={"tir.disable_storage_rewrite": True}):
        mod = tvm.build(mod, target=target, target_host="llvm")
        src = mod.imported_modules[0].get_source()
        print(src)

def test_nki_add_2():
    # fmt: off
    @T.prim_func
    def func(A: T.Buffer((128, 512)), B: T.Buffer((128, 512))):
        A_sbuf = T.alloc_buffer((128, 512), "float32", scope="trn.sbuf",)
        B_sbuf = T.alloc_buffer((128, 512), "float32", scope="trn.sbuf",)
        for k in range(0, 4):
            with T.attr(0, "tensorized_nki_instruction", 1):
                for i in range(0, 128):
                    for j in range(0, 512):
                        A_sbuf[i, j] =  A[i, 512*k+j]
            with T.attr(0, "tensorized_nki_instruction", 1):
                for i in range(0, 128):
                    for j in range(0, 512):
                        B_sbuf[i, j] = A_sbuf[i, j] + 1
            with T.attr(0, "tensorized_nki_instruction", 1):
                for i in range(0, 128):
                    for j in range(0, 512):
                        B[i, 512*k+j] =  B_sbuf[i, j]

            
    target = tvm.target.Target("aws/trn1/trn1.2xlarge")
    mod = tvm.IRModule({"main": func})
    mod = tvm.tir.transform.DecorateDeviceScope()(mod)
    with tvm.transform.PassContext(config={"tir.disable_storage_rewrite": True}):
        mod = tvm.build(mod, target=target, target_host="llvm")
        src = mod.imported_modules[0].get_source()
        print(src)

def test_nki_matmul_1():
    TILES_IN_BLOCK_M=16
    TILES_IN_BLOCK_N=2
    TILES_IN_BLOCK_K=8
    TILE_M = 128
    TILE_K = 128
    TILE_N = 512
    K = 1024
    M = 4096
    N = 2048
    BLOCK_M = TILE_M * TILES_IN_BLOCK_M
    BLOCK_N = TILE_N * TILES_IN_BLOCK_N
    BLOCK_K = TILE_K * TILES_IN_BLOCK_K
    # the size has to be multiple of block size
    assert M % BLOCK_M == 0
    assert N % BLOCK_N == 0
    assert K % BLOCK_K == 0

    NUM_BLOCK_M = M // BLOCK_M
    NUM_BLOCK_N = N // BLOCK_N
    NUM_BLOCK_K = K // BLOCK_K
    

    @T.prim_func
    def func(lhsT: T.Buffer((K, M), "float16"), rhs: T.Buffer((K, N), "float16"), result: T.buffer((M, N), "float16")):
        result_tiles = T.alloc_buffer((TILE_M, NUM_BLOCK_M, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N, TILE_N), "float32", scope="trn.sbuf")
        rhs_tiles = T.alloc_buffer((TILE_K, TILES_IN_BLOCK_K, BLOCK_N), "float16", scope="trn.sbuf")
        lhsT_tiles = T.alloc_buffer((TILE_K, TILES_IN_BLOCK_K, BLOCK_M), "float16", scope="trn.sbuf")
        res_tile = T.alloc_buffer((TILE_M, TILE_N), "float32", scope="trn.psum")
        result_packed = T.alloc_buffer((TILE_K, BLOCK_N), "float32", scope="trn.sbuf")
        for n in range(NUM_BLOCK_N):
            with T.attr(0, "tensorized_nki_instruction", 1):
                for i0 in range(TILE_M):
                    for i1 in range(NUM_BLOCK_M):
                        for i2 in range(TILES_IN_BLOCK_M):
                            for i3 in range(TILES_IN_BLOCK_N):
                                for i4 in range(TILE_N):
                                    result_tiles[i0, i1, i2, i3, i4] = 0 
            for k in range(NUM_BLOCK_K):
                for bk_r in range(TILES_IN_BLOCK_K):
                    with T.attr(0, "tensorized_nki_instruction", 1):
                        for i in range(TILE_K):
                            for j in range(BLOCK_N):
                                rhs_tiles[i, bk_r, j] = rhs[(TILES_IN_BLOCK_K * k + bk_r) * TILE_K + i, n * BLOCK_N + j]
                for m in range(NUM_BLOCK_M):
                    for bk_l in range(TILES_IN_BLOCK_K):
                        with T.attr(0, "tensorized_nki_instruction", 1):
                            for i in range(TILE_K):
                                for j in range(BLOCK_M):
                                    lhsT_tiles[i, bk_l, j] = lhsT[(TILES_IN_BLOCK_K * k + bk_l) * TILE_K + i, m * BLOCK_M + j]
                    for bn in range(TILES_IN_BLOCK_N):
                        for bm in range(TILES_IN_BLOCK_M):
                            with T.attr(0, "tensorized_nki_instruction", 1):
                                for i in range(TILE_M):
                                    for j in range(TILE_N):
                                        res_tile[i, j] = 0
                            for bk in range(TILES_IN_BLOCK_K):
                                with T.attr(0, "tensorized_nki_instruction", 1):
                                    for i in range(TILE_M):
                                        for j in range(TILE_N):
                                            for k in range(TILE_K):
                                                T.nki_matmul(res_tile[i, j], lhsT_tiles[k, bk, bm * TILE_M + i], rhs_tiles[k, bk, bn * TILE_N + j], 1)
                            with T.attr(0, "tensorized_nki_instruction", 1):
                                for i in range(TILE_M):
                                    for j in range(TILE_N):
                                        result_tiles[i, m, bm, bn, j] += res_tile[i, j]
            for m in range(NUM_BLOCK_M):
                for bm in range(TILES_IN_BLOCK_M):
                    for bn in range(TILES_IN_BLOCK_N):
                        with T.attr(0, "tensorized_nki_instruction", 1):
                            for i in range(TILE_K):
                                for j in range(TILE_N):
                                    result_packed[i, bn * TILE_N + j] = result_tiles[i, m, bm, bn, j]
                    with T.attr(0, "tensorized_nki_instruction", 1):
                        for i in range(TILE_K):
                            for j in range(BLOCK_N):
                                result[m * BLOCK_M + bm * TILE_M + i, n * BLOCK_N + j] = result_packed[i, j]
    
    target = tvm.target.Target("aws/trn1/trn1.2xlarge")
    mod = tvm.IRModule({"main": func})
    mod = tvm.tir.transform.DecorateDeviceScope()(mod)
    with tvm.transform.PassContext(config={"tir.disable_storage_rewrite": True}):
        mod = tvm.build(mod, target=target, target_host="llvm")
        src = mod.imported_modules[0].get_source()
        print(src)
        
    
if __name__ == "__main__":
    test_nki_add_1()
    test_nki_add_2()
    test_nki_matmul_1()
