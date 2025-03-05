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

seqlen_q=16*1024
seqlen_kv=16*1024
d=128
softmax_scale=1.0 / (d ** 0.5)
head_q = 1
head_kv = 1
p_size = 128

BLOCK_KV = 8192
NUM_BLOCKS_KV = seqlen_kv // BLOCK_KV
BLOCK_Q = 128
NUM_BLOCKS_Q = seqlen_q // BLOCK_Q

NUM_MM1_PER_BLOCK = BLOCK_KV // 512
def test_flash_attn(ssh_client):
    
    running_max_shape = (seqlen_q, 1)
    running_max_layout = T.TrainiumLayout("FP", T.TileLayout.from_tuple((seqlen_q // p_size, p_size), (1, 1)))
    
    @T.macro
    def mm1(q_loaded, k_loaded, qk, mm1_dot_partial_max):
        mm1_dot_psum = T.alloc_buffer((8, BLOCK_Q, 512), dtype="float32", scope="trn.psum", layout= T.TrainiumPSUMLayout("FPF", T.TileLayout.from_tuple((8, BLOCK_Q, 512), (512, 1, 1))), allocated_addr=(0,0))
        for i in T.serial(0, NUM_MM1_PER_BLOCK):
            Tp.gemm(mm1_dot_psum[i % 8], q_loaded, k_loaded[i * 512: (i + 1) * 512, :], mm1_dot_psum[i % 8], transpose_B=True)
            with Tp.compose_op():
                Tp.mul(qk[:, i * 512: (i + 1) * 512], mm1_dot_psum[i % 8], T.float32(softmax_scale))
                Tp.max(mm1_dot_partial_max[:, i], qk[:, i * 512: (i + 1) * 512], axes=-1)

    class SimpleSBUFAllocator:
        def __init__(self):
            self.allocated_addr = 0

        def allocate(self, size):
            addr = self.allocated_addr
            self.allocated_addr += size
            return addr

    allocator = SimpleSBUFAllocator()   
    
    @T.prim_func(tirp=True)
    def flash_attn(q_ptr: T.handle, k_ptr: T.handle, v_ptr: T.handle, out_ptr: T.handle):
        T.func_attr({"num_inputs": 3})
        q = T.match_buffer(q_ptr, (seqlen_q, head_q, d), dtype="float16", layout=T.TileLayout.from_tuple((seqlen_q, head_q, d), (1, seqlen_q * d, seqlen_q)))
        k = T.match_buffer(k_ptr, (seqlen_kv, head_kv, d), dtype="float16", layout=T.TileLayout.from_tuple((seqlen_kv, head_kv, d), (1, seqlen_kv * d, seqlen_kv)))
        v = T.match_buffer(v_ptr, (seqlen_kv, head_kv, d), dtype="float16", layout=T.TileLayout.from_tuple((seqlen_kv, head_kv, d), (d, seqlen_kv * d, 1)))
        out = T.match_buffer(out_ptr, (seqlen_q, head_q, d), dtype="float16", layout=T.TileLayout.from_tuple((seqlen_q, head_q, d), (d, seqlen_q * d, 1)))
        with T.kernel():
            identity_tensor = T.alloc_buffer((128, 128), dtype="float16", scope="trn.sbuf", allocated_addr=allocator.allocate(128*2))
            
            with T.attr(0, "tensorized_nki_instruction", 1):
                for p_loop, rhs_f_loop in T.grid(p_size, 128):
                    T.evaluate(T.nki_identity(identity_tensor[p_loop, rhs_f_loop], p_size))
            
            for head in T.serial(0, head_q):
                # running_max and prev_running_max are used to store the max value of the previous block, with the value multiplied by -1
                running_max = T.alloc_buffer(running_max_shape, dtype="float32", scope="trn.sbuf", layout=running_max_layout, allocated_addr=allocator.allocate(seqlen_q // 128 * 4))
                l = T.alloc_buffer(running_max_shape, dtype="float32", scope="trn.sbuf", layout=running_max_layout, allocated_addr=allocator.allocate(seqlen_q // 128 * 4))
                for block_kv in T.serial(0, NUM_BLOCKS_KV):
                    k_loaded = T.alloc_buffer((BLOCK_KV, d), dtype=k.dtype, scope="trn.sbuf", layout = T.TrainiumLayout("FP", T.TileLayout.from_tuple((BLOCK_KV, d), (1, 1))), allocated_addr=allocator.allocate(BLOCK_KV * 2))
                    v_loaded = T.alloc_buffer((BLOCK_KV, d), dtype=v.dtype, scope="trn.sbuf", layout = T.TrainiumLayout("PF", T.TileLayout.from_tuple((p_size, BLOCK_KV * d // p_size), (1, 1))), allocated_addr=allocator.allocate(BLOCK_KV * 2))
                    # load k and v
                    Tp.copy(k_loaded, k[block_kv * BLOCK_KV: (block_kv + 1) * BLOCK_KV, head, :])
                    Tp.copy(v_loaded, v[block_kv * BLOCK_KV: (block_kv + 1) * BLOCK_KV, head, :])
                    for block_q in T.serial(0, NUM_BLOCKS_Q):
                        q_loaded = T.alloc_buffer((BLOCK_Q, d), dtype=q.dtype, scope="trn.sbuf", layout= T.TrainiumLayout("FP", T.TileLayout.from_tuple((BLOCK_Q, d), (1, 1))), allocated_addr=allocator.allocate(d * 2))
                        qk = T.alloc_buffer((BLOCK_Q, BLOCK_KV), dtype="float32", scope="trn.sbuf", layout= T.TrainiumLayout("PF", T.TileLayout.from_tuple((BLOCK_Q, BLOCK_KV), (1, 1))), allocated_addr=allocator.allocate(BLOCK_KV*4))
                        mm1_dot_partial_max = T.alloc_buffer((BLOCK_Q, NUM_MM1_PER_BLOCK), dtype="float32", scope="trn.sbuf", layout= T.TrainiumLayout("PF", T.TileLayout.from_tuple((BLOCK_Q, NUM_MM1_PER_BLOCK), (1, 1))), allocated_addr=allocator.allocate(NUM_MM1_PER_BLOCK*4))
                        mm1_dot_max = T.alloc_buffer((BLOCK_Q, 1), dtype="float32", scope="trn.sbuf", layout= T.TrainiumLayout("PF", T.TileLayout.from_tuple((BLOCK_Q, 1), (1, 1))), allocated_addr=allocator.allocate(4))
                        scaling_factor = T.alloc_buffer((BLOCK_Q, 1), dtype="float32", scope="trn.sbuf", layout= T.TrainiumLayout("PF", T.TileLayout.from_tuple((BLOCK_Q, 1), (1, 1))), allocated_addr=allocator.allocate(4))
                        prev_running_max = T.alloc_buffer((BLOCK_Q, 1), dtype="float32", scope="trn.sbuf", layout=T.TrainiumLayout("PF", T.TileLayout.from_tuple((BLOCK_Q, 1), (1, 1))), allocated_addr=allocator.allocate(4))
                        p = T.alloc_buffer((BLOCK_Q, BLOCK_KV), dtype="float16", scope="trn.sbuf", layout=T.TrainiumLayout("PF", T.TileLayout.from_tuple((BLOCK_Q, BLOCK_KV), (1, 1))), allocated_addr=allocator.allocate(BLOCK_KV*2))
                        rowsum_p = T.alloc_buffer((BLOCK_Q, 1), dtype="float32", scope="trn.sbuf", layout=T.TrainiumLayout("PF", T.TileLayout.from_tuple((BLOCK_Q, 1), (1, 1))), allocated_addr=allocator.allocate(4))
                        p_transposed = T.alloc_buffer((BLOCK_Q, BLOCK_KV), dtype="float16", scope="trn.sbuf", layout=T.TrainiumLayout("FPF", T.TileLayout.from_tuple((BLOCK_Q, p_size, BLOCK_KV//p_size), (1, 1, BLOCK_Q))), allocated_addr=allocator.allocate(BLOCK_KV*2))
                        mm2_out = T.alloc_buffer((BLOCK_Q, d), dtype="float16", scope="trn.sbuf", layout=T.TrainiumLayout("PF", T.TileLayout.from_tuple((BLOCK_Q, d), (1, 1))), allocated_addr=allocator.allocate(d*2))
                        prev_output = T.alloc_buffer((BLOCK_Q, d), dtype="float16", scope="trn.sbuf", layout=T.TrainiumLayout("PF", T.TileLayout.from_tuple((BLOCK_Q, d), (1, 1))), allocated_addr=allocator.allocate(d*2))
                        scaled_output = T.alloc_buffer((BLOCK_Q, d), dtype="float16", scope="trn.sbuf", layout=T.TrainiumLayout("PF", T.TileLayout.from_tuple((BLOCK_Q, d), (1, 1))), allocated_addr=allocator.allocate(d*2))
                        l_reciprocal = T.alloc_buffer((BLOCK_Q, 1), dtype="float32", scope="trn.sbuf", layout=T.TrainiumLayout("PF", T.TileLayout.from_tuple((BLOCK_Q, 1), (1, 1))), allocated_addr=allocator.allocate(4))
                        # load q
                        Tp.copy(q_loaded, q[block_q * BLOCK_Q: (block_q + 1) * BLOCK_Q, head, :])
                        # Q@K.T
                        mm1(q_loaded, k_loaded, qk, mm1_dot_partial_max)
                        # mm1_dot_max = -max(Q@K.T)
                        with Tp.compose_op():
                            Tp.max(mm1_dot_max, mm1_dot_partial_max, axes=-1)
                            Tp.mul(mm1_dot_max, mm1_dot_max, T.float32(-1.0))
                        # running_max = min(-max(Q@K.T), running_max)
                        # scaling_factor = exp(-prev_running_max + running_max) (this is because we apply negate to the max value)
                        if block_kv == 0:
                            Tp.copy(running_max[block_q * BLOCK_Q: (block_q + 1) * BLOCK_Q, 0], mm1_dot_max)
                        else:
                            Tp.mul(prev_running_max, prev_running_max, T.float32(-1.0))
                            Tp.minimum(running_max[block_q * BLOCK_Q: (block_q + 1) * BLOCK_Q, 0], running_max[block_q * BLOCK_Q: (block_q + 1) * BLOCK_Q, 0], mm1_dot_max)
                            Tp.exp(scaling_factor, prev_running_max, bias=running_max[block_q * BLOCK_Q: (block_q + 1) * BLOCK_Q, 0])
                        # p = exp(Q@K.T + running_max)
                        with Tp.compose_op():
                            Tp.exp(p, qk, bias=running_max[block_q * BLOCK_Q: (block_q + 1) * BLOCK_Q, 0])
                            Tp.sum(rowsum_p, p, axes=-1)
                        # transpose p
                        Tp.copy(p_transposed, p, workspace={"identity": identity_tensor})
                        # l = sum(p) + scaling_factor * l
                        if block_kv == 0:
                            Tp.copy(l[block_q * BLOCK_Q: (block_q + 1) * BLOCK_Q, 0], rowsum_p)
                        else:
                            with Tp.compose_op():
                                Tp.mul(l[block_q * BLOCK_Q: (block_q + 1) * BLOCK_Q, 0], l[block_q * BLOCK_Q: (block_q + 1) * BLOCK_Q, 0], scaling_factor)
                                Tp.add(l[block_q * BLOCK_Q: (block_q + 1) * BLOCK_Q, 0], l[block_q * BLOCK_Q: (block_q + 1) * BLOCK_Q, 0], rowsum_p)
                        # l_reciprocal = 1 / l (this is to prepare for the computation of O / l)
                        if block_kv == NUM_BLOCKS_KV - 1:
                            Tp.reciprocal(l_reciprocal, l[block_q * BLOCK_Q: (block_q + 1) * BLOCK_Q, 0])
                        # mm2_out = P@V
                        Tp.gemm(mm2_out, p_transposed, v_loaded, mm2_out)
                        # O = l * O + mm2_out
                        if block_kv == 0:
                            Tp.copy(out[block_q * BLOCK_Q: (block_q + 1) * BLOCK_Q, head, :], mm2_out)
                        else:
                            Tp.copy(prev_output, out[block_q * BLOCK_Q: (block_q + 1) * BLOCK_Q, head, :])
                            with Tp.compose_op():
                                Tp.mul(prev_output, prev_output, scaling_factor)
                                Tp.add(scaled_output, prev_output, mm2_out)
                            if block_kv == NUM_BLOCKS_KV - 1:
                                # write O/l to hbm
                                Tp.mul(scaled_output, scaled_output, l_reciprocal)
                            Tp.copy(out[block_q * BLOCK_Q: (block_q + 1) * BLOCK_Q, head, :], scaled_output)
    # fmt: on
    with target:
        mod = tvm.IRModule({"main": flash_attn})
        func = mod["main"]
        #FIXME: the correctness is not verified due to a bug in neuron compiler
        run_on_remote_and_check_correct(func, None, target)


if __name__ == "__main__":
    test_flash_attn()
