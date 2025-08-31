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
import torch

import tvm
import tvm.testing
from tvm.script import tir as T
from tvm.script import tirp as Tp

from .utils import run_on_remote_and_check_correct, ssh_client

target = tvm.target.Target("aws/trn1/trn1.2xlarge")

seqlen_q = 8 * 1024
seqlen_kv = 8 * 1024
d = 128
softmax_scale = 1.0 / (d**0.5)
head_q = 1
head_kv = 1
p_size = 128


@pytest.mark.dependency(depends=["ssh_success"])
def test_flash_attn(ssh_client, causal=True):

    BLOCK_KV = 8192
    NUM_BLOCKS_KV = seqlen_kv // BLOCK_KV
    BLOCK_Q = 128
    NUM_BLOCKS_Q = seqlen_q // BLOCK_Q

    INST_SIZE = 512
    NUM_MM1_PER_BLOCK = BLOCK_KV // INST_SIZE

    running_max_shape = (seqlen_q, 1)
    running_max_layout = "PF"

    # fmt: off
    @T.macro
    def mm1(q_loaded, k_loaded, qk, mm1_dot_partial_max, mm1_dot_max, block_q, block_kv):
        mm1_dot_psum = T.alloc_buffer((8, BLOCK_Q, 512), dtype="float32", scope="trn.psum",layout="FPF", allocated_addr=(0,0))
        for i in T.serial(0, NUM_MM1_PER_BLOCK):
            k_mm_range = T.meta_var(T.max(0, T.min((block_q//4*4+4) * BLOCK_Q - block_kv * BLOCK_KV - i * 512, 512)) if causal else 512)
            Tp.gemm(mm1_dot_psum[i % 8, :, 0:k_mm_range], q_loaded[block_q % 2], k_loaded[i * 512: i * 512 + k_mm_range, :], mm1_dot_psum[i % 8, :, 0:k_mm_range], transpose_B=True)
            if not causal:
                Tp.binary_reduce(qk[block_q % 2, :, i * 512: i * 512 + k_mm_range], mm1_dot_partial_max[:, i], mm1_dot_psum[i % 8, :, 0:k_mm_range], T.float32(softmax_scale), binary_op="mul", reduce_op="max", reduce_axes=-1)
            else:
                if block_kv * BLOCK_KV + (i+1) * 512 < block_q * BLOCK_Q:
                    Tp.binary_reduce(qk[block_q % 2, :, i * 512: i * 512 + k_mm_range], mm1_dot_partial_max[:, i], mm1_dot_psum[i % 8, :, 0:k_mm_range], T.float32(softmax_scale), binary_op="mul", reduce_op="max", reduce_axes=-1)
                else:
                    Tp.mul(qk[block_q % 2, :, i * 512: i * 512 + k_mm_range], mm1_dot_psum[i % 8, :, 0:k_mm_range], T.float32(softmax_scale))
                    Tp.select(qk[block_q % 2, :, i * 512: i * 512 + k_mm_range], qk[block_q % 2, :, i * 512: i * 512 + k_mm_range], -9984.0, pred=lambda _, q, k: block_q * BLOCK_Q + q >= block_kv * BLOCK_KV + i * 512 + k)
                    Tp.max(mm1_dot_partial_max[:, i], qk[block_q % 2, :, i * 512: i * 512 + k_mm_range], axes=-1)
        # mm1_dot_max = -max(Q@K.T)
        Tp.reduce_negate(mm1_dot_max, mm1_dot_partial_max, reduce_op="max", reduce_axes=-1)

    # fmt: on

    def get_kv_range(block_q, block_kv, causal):
        if causal:
            # we enforce kv_range to be a multiple of 512
            # this is to prevent having f_loop/p_loop in the mask of matmul.
            # Neuron compiler does not support this in a proper way.
            return T.max(T.min((block_q // 4 * 4 + 4) * BLOCK_Q - block_kv * BLOCK_KV, BLOCK_KV), 0)
        else:
            return BLOCK_KV

    @T.macro
    def load_q(q_loaded, q, block_q, head):
        Tp.copy(q_loaded[block_q % 2], q[block_q * BLOCK_Q : (block_q + 1) * BLOCK_Q, head, :])

    @T.macro
    def update_running_max(
        running_max, mm1_dot_max, prev_running_max, scaling_factor, block_q, block_kv
    ):
        # running_max = min(-max(Q@K.T), running_max)
        # scaling_factor = exp(-prev_running_max + running_max) (this is because we apply negate to the max value)
        if block_kv == 0:
            Tp.copy(running_max[block_q * BLOCK_Q : (block_q + 1) * BLOCK_Q, 0], mm1_dot_max)
        else:
            Tp.mul(
                prev_running_max,
                running_max[block_q * BLOCK_Q : (block_q + 1) * BLOCK_Q, 0],
                T.float32(-1.0),
            )
            Tp.minimum(
                running_max[block_q * BLOCK_Q : (block_q + 1) * BLOCK_Q, 0],
                running_max[block_q * BLOCK_Q : (block_q + 1) * BLOCK_Q, 0],
                mm1_dot_max,
            )
            Tp.exp(
                scaling_factor[block_q % 2],
                prev_running_max,
                bias=running_max[block_q * BLOCK_Q : (block_q + 1) * BLOCK_Q, 0],
            )

    @T.macro
    def exp(p, qk, running_max, partial_rowsum_p, rowsum_p, block_q, block_kv):
        # p = exp(Q@K.T + running_max)
        # FIXME: this still fails to be simplified. Try to use explicit mask later
        # Most masks are of 2 kinds: 1. out-of-bound mask, 2. mask that reduce redundant computation.
        # We can set an attribute to the mask, showing that the second kind of mask can be relaxed.
        # F loop can always be relaxed to a constant, so that the mask only contains out-of-bound mask.
        if causal:
            Tp.memset(partial_rowsum_p, 0.0)
            p_reshape = p.view(BLOCK_Q, BLOCK_KV // INST_SIZE, INST_SIZE)
            qk_reshape = qk.view(2, BLOCK_Q, BLOCK_KV // INST_SIZE, INST_SIZE)
            running_max_reshape = running_max.view(seqlen_q, 1, 1)
            reduced_kv_range = T.meta_var(
                T.max(
                    T.min(
                        (block_q // 4 + 1) - block_kv * BLOCK_KV // INST_SIZE, BLOCK_KV // INST_SIZE
                    ),
                    0,
                )
            )
            Tp.unary_reduce(
                p_reshape[:, 0:reduced_kv_range, :],
                partial_rowsum_p[:, 0:reduced_kv_range],
                qk_reshape[block_q % 2, :, 0:reduced_kv_range, :],
                unary_op="exp",
                reduce_op="sum",
                bias=running_max_reshape[block_q * BLOCK_Q : (block_q + 1) * BLOCK_Q, 0, 0],
                reduce_axes=-1,
                schedule_config={"max_inst_size": INST_SIZE},
            )
            Tp.sum(rowsum_p[block_q % 2], partial_rowsum_p, axes=-1)
        else:
            Tp.unary_reduce(
                p,
                rowsum_p[block_q % 2],
                qk[block_q % 2],
                unary_op="exp",
                reduce_op="sum",
                bias=running_max[block_q * BLOCK_Q : (block_q + 1) * BLOCK_Q, 0],
                reduce_axes=-1,
            )

    @T.macro
    def transpose(p, p_transposed, block_q, block_kv):
        kv_range = T.meta_var(get_kv_range(block_q, block_kv, causal))
        Tp.copy(p_transposed[:, 0:kv_range], p[:, 0:kv_range])

    @T.macro
    def pv(mm2_out, p_transposed, v_loaded, block_q, block_kv):
        kv_range = T.meta_var(get_kv_range(block_q, block_kv, causal))
        Tp.gemm(mm2_out, p_transposed[:, 0:kv_range], v_loaded[0:kv_range, :], mm2_out)

    @T.macro
    def write_back(
        l,
        l_reciprocal,
        scaling_factor,
        rowsum_p,
        out,
        prev_output,
        scaled_output,
        mm2_out,
        head,
        block_q,
        block_kv,
    ):
        kv_range = T.meta_var(get_kv_range(block_q, block_kv, causal))
        # l = sum(p) + scaling_factor * l
        if block_kv == 0:
            Tp.copy(l[block_q * BLOCK_Q : (block_q + 1) * BLOCK_Q, 0], rowsum_p[block_q % 2])
        else:
            Tp.binary_chain(
                l[block_q * BLOCK_Q : (block_q + 1) * BLOCK_Q, 0],
                l[block_q * BLOCK_Q : (block_q + 1) * BLOCK_Q, 0],
                scaling_factor[block_q % 2],
                rowsum_p[block_q % 2],
                op0="mul",
                op1="add",
            )
        # l_reciprocal = 1 / l (this is to prepare for the computation of O / l)
        if block_kv == NUM_BLOCKS_KV - 1:
            Tp.reciprocal(l_reciprocal, l[block_q * BLOCK_Q : (block_q + 1) * BLOCK_Q, 0])
        # O = l * O + mm2_out
        if block_kv == 0:
            Tp.copy(out[block_q * BLOCK_Q : (block_q + 1) * BLOCK_Q, head, :], mm2_out)
        elif kv_range > 0 or block_kv == NUM_BLOCKS_KV - 1:
            Tp.copy(prev_output, out[block_q * BLOCK_Q : (block_q + 1) * BLOCK_Q, head, :])
            if kv_range > 0:
                Tp.binary_chain(
                    scaled_output,
                    prev_output,
                    scaling_factor[block_q % 2],
                    mm2_out,
                    op0="mul",
                    op1="add",
                )
            else:
                Tp.mul(scaled_output, prev_output, scaling_factor[block_q % 2])
            if block_kv == NUM_BLOCKS_KV - 1:
                # write O/l to hbm
                Tp.mul(scaled_output, scaled_output, l_reciprocal)
            Tp.copy(out[block_q * BLOCK_Q : (block_q + 1) * BLOCK_Q, head, :], scaled_output)

    # fmt: off
    @T.prim_func(tirp=True)
    def flash_attn(q_ptr: T.handle, k_ptr: T.handle, v_ptr: T.handle, out_ptr: T.handle):
        T.func_attr({"num_inputs": 3})
        q = T.match_buffer(q_ptr, (seqlen_q, head_q, d), dtype="float16", layout=T.TileLayout.from_tuple((seqlen_q, head_q, d), (1, seqlen_q * d, seqlen_q)))
        k = T.match_buffer(k_ptr, (seqlen_kv, head_kv, d), dtype="float16", layout=T.TileLayout.from_tuple((seqlen_kv, head_kv, d), (1, seqlen_kv * d, seqlen_kv)))
        v = T.match_buffer(v_ptr, (seqlen_kv, head_kv, d), dtype="float16", layout=T.TileLayout.from_tuple((seqlen_kv, head_kv, d), (d, seqlen_kv * d, 1)))
        out = T.match_buffer(out_ptr, (seqlen_q, head_q, d), dtype="float16", layout=T.TileLayout.from_tuple((seqlen_q, head_q, d), (d, seqlen_q * d, 1)))
        with T.kernel():
            for head in T.serial(0, head_q):
                # running_max and prev_running_max are used to store the max value of the previous block, with the value multiplied by -1
                running_max = T.alloc_buffer(running_max_shape, dtype="float32", scope="trn.sbuf", layout=running_max_layout)
                l = T.alloc_buffer(running_max_shape, dtype="float32", scope="trn.sbuf", layout=running_max_layout)
                for block_kv in T.serial(0, NUM_BLOCKS_KV):
                    k_loaded = T.alloc_buffer((BLOCK_KV, d), dtype=k.dtype, scope="trn.sbuf", layout = "FP")
                    v_loaded = T.alloc_buffer((BLOCK_KV, d), dtype=v.dtype, scope="trn.sbuf", layout = "PF")
                    #TODO: integrate neuron runtime
                    q_loaded = T.alloc_buffer((2, BLOCK_Q, d), dtype=q.dtype, scope="trn.sbuf", layout= "FFP")
                    qk = T.alloc_buffer((2, BLOCK_Q, BLOCK_KV), dtype="float32", scope="trn.sbuf", layout= "FPF")
                    mm1_dot_partial_max = T.alloc_buffer((BLOCK_Q, NUM_MM1_PER_BLOCK), dtype="float32", scope="trn.sbuf", layout= "PF")
                    mm1_dot_max = T.alloc_buffer((BLOCK_Q, 1), dtype="float32", scope="trn.sbuf", layout= "PF")
                    scaling_factor = T.alloc_buffer((2, BLOCK_Q, 1), dtype="float32", scope="trn.sbuf", layout= "FPF")
                    prev_running_max = T.alloc_buffer((BLOCK_Q, 1), dtype="float32", scope="trn.sbuf", layout= "PF")
                    p = T.alloc_buffer((BLOCK_Q, BLOCK_KV), dtype="float16", scope="trn.sbuf", layout= "PF")
                    partial_rowsum_p = T.alloc_buffer((BLOCK_Q, BLOCK_KV//INST_SIZE), dtype="float32", layout="PF",scope="trn.sbuf")
                    rowsum_p = T.alloc_buffer((2, BLOCK_Q, 1), dtype="float32", scope="trn.sbuf", layout= "FPF")
                    p_transposed = T.alloc_buffer((BLOCK_Q, BLOCK_KV), dtype="float16", scope="trn.sbuf", layout= "FP")
                    mm2_out = T.alloc_buffer((BLOCK_Q, d), dtype="float16", scope="trn.sbuf", layout= "PF")
                    prev_output = T.alloc_buffer((BLOCK_Q, d), dtype="float16", scope="trn.sbuf", layout= "PF")
                    scaled_output = T.alloc_buffer((BLOCK_Q, d), dtype="float16", scope="trn.sbuf", layout= "PF")
                    l_reciprocal = T.alloc_buffer((BLOCK_Q, 1), dtype="float32", scope="trn.sbuf", layout= "PF")
                    
                    # load k and v
                    Tp.copy(k_loaded, k[block_kv * BLOCK_KV: (block_kv + 1) * BLOCK_KV, head, :])
                    Tp.copy(v_loaded, v[block_kv * BLOCK_KV: (block_kv + 1) * BLOCK_KV, head, :])
                    
                    
                    load_q(q_loaded, q, 0, head)
                    mm1(q_loaded, k_loaded, qk, mm1_dot_partial_max, mm1_dot_max, 0, block_kv)
                    update_running_max(running_max, mm1_dot_max, prev_running_max, scaling_factor, 0, block_kv)
                    exp(p, qk, running_max, partial_rowsum_p, rowsum_p, 0, block_kv)
                    transpose(p, p_transposed, 0, block_kv)
                    load_q(q_loaded, q, 1, head)
                    mm1(q_loaded, k_loaded, qk, mm1_dot_partial_max, mm1_dot_max, 1, block_kv)
                    update_running_max(running_max, mm1_dot_max, prev_running_max, scaling_factor, 1, block_kv)
                    for block_q in T.serial(0, NUM_BLOCKS_Q-2):
                        # load q
                        load_q(q_loaded, q, block_q+2, head)
                        # p = exp(Q@K.T + running_max)
                        exp(p, qk, running_max, partial_rowsum_p, rowsum_p, block_q+1, block_kv)
                        # Q@K.T
                        mm1(q_loaded, k_loaded, qk, mm1_dot_partial_max, mm1_dot_max, block_q+2, block_kv)
                        # mm2_out = P@V
                        pv(mm2_out, p_transposed, v_loaded, block_q, block_kv)
                        # transpose p
                        transpose(p, p_transposed, block_q+1, block_kv)
                        # write back
                        write_back(l, l_reciprocal, scaling_factor, rowsum_p, out, prev_output, scaled_output, mm2_out, head, block_q, block_kv)
                        # running_max = min(-max(Q@K.T), running_max)
                        # scaling_factor = exp(-prev_running_max + running_max) (this is because we apply negate to the max value)
                        update_running_max(running_max, mm1_dot_max, prev_running_max, scaling_factor, block_q+2, block_kv)
                    pv(mm2_out, p_transposed, v_loaded, NUM_BLOCKS_Q-2, block_kv)
                    write_back(l, l_reciprocal, scaling_factor, rowsum_p, out, prev_output, scaled_output, mm2_out, head, NUM_BLOCKS_Q-2, block_kv)
                    exp(p, qk, running_max, partial_rowsum_p, rowsum_p, NUM_BLOCKS_Q-1, block_kv)
                    transpose(p, p_transposed, NUM_BLOCKS_Q-1, block_kv)
                    pv(mm2_out, p_transposed, v_loaded, NUM_BLOCKS_Q-1, block_kv)
                    write_back(l, l_reciprocal, scaling_factor, rowsum_p, out, prev_output, scaled_output, mm2_out, head, NUM_BLOCKS_Q-1, block_kv)

    # fmt: on
    with target:
        mod = tvm.IRModule({"main": flash_attn})
        func = mod["main"]

        def attn_ref(q, k, v):
            q = q.reshape(head_q, d, seqlen_q)
            k = k.reshape(head_kv, d, seqlen_kv)
            v = v.reshape(head_kv, seqlen_kv, d)
            q = q.transpose(1, 2)
            attn_scores = torch.bmm(q, k)
            attn_scores *= softmax_scale
            if causal:
                mask = torch.triu(
                    torch.ones((seqlen_q, seqlen_kv), device=q.device), diagonal=1
                ).bool()
                mask = torch.zeros_like(attn_scores, dtype=q.dtype).masked_fill(mask, float("-inf"))
                attn_scores += mask
            attn_probs = torch.softmax(attn_scores, dim=-1)
            out = torch.bmm(attn_probs, v)
            return [out.reshape(seqlen_q, head_q, d)]

        run_on_remote_and_check_correct(func, attn_ref, target)


if __name__ == "__main__":
    test_flash_attn(causal=False)
