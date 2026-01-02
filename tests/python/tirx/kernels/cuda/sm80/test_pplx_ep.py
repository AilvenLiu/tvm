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

import math
import tempfile

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.contrib.popen_pool import PopenWorker
from tvm.runtime import ShapeTuple
from tvm.runtime import disco as di
from tvm.script import tir as T
from tvm.script.ir_builder import IRBuilder


@pytest.mark.skip(reason="nvshmem doesn't work with pytest")
def test_dispatch_combine(world_size=8):
    def test_func():
        np.random.seed(42)

        # config parameters
        dtype = "float32"
        nbytes = tvm.runtime.DataType(dtype).bits // 8
        M = 128  # batch size
        N_EXPERTS = 256
        N_LOCAL_EXPERTS = N_EXPERTS // world_size
        assert N_EXPERTS % world_size == 0, "Number of experts must be divisible by world size"
        K = 8  # top K experts
        HIDDEN_DIM = 16  # TODO: change to 7168, but now it's too slow
        VEC_SIZE = 1
        assert HIDDEN_DIM % VEC_SIZE == 0, "HIDDEN_DIM must be divisible by VEC_SIZE"
        N_REPEAT = 15  # number of runs for profiling

        # data parallelism parameters
        # rank 0,1 - group 0, rank 2,3 - group 1, each GPU in the same group gets the same data
        # practical for combining with tensor parallelism
        N_GROUP = world_size
        GROUP_SIZE = world_size // N_GROUP
        assert world_size % N_GROUP == 0, "World size must be divisible by number of DP groups"
        # FIXME@(kathy): support variadic shapes in LLaMA
        # MAX_TOKENS_PER_GROUP = 10  # max batch size
        MAX_RECV_TOKENS = N_LOCAL_EXPERTS * N_GROUP * M
        MAX_SEND_TOKENS = N_LOCAL_EXPERTS * N_GROUP * M

        # comm-comp overlap parameters
        DISPATCH_SEND = True
        DISPATCH_RECV = True
        COMBINE_SEND = True
        COMBINE_RECV = True

        # hardware parameters
        DEV = tvm.cuda(0)
        N_SMS = DEV.multi_processor_count
        N_WARPS_DISPATCH = 10
        N_WARPS_COMBINE = 16
        N_BLOCKS_DISPATCH = min(N_SMS, max(math.ceil(N_EXPERTS / N_WARPS_DISPATCH), M * K))
        N_BLOCKS_COMBINE = min(N_SMS, N_LOCAL_EXPERTS * N_GROUP * M)
        N_RECV_GROUPS = N_LOCAL_EXPERTS * N_GROUP
        N_RECV_GROUPS_PER_CTA = math.ceil(N_RECV_GROUPS / N_BLOCKS_DISPATCH)

        class GridBarrier:
            def __init__(self, cnt, buffer):
                self.cnt = cnt
                self.sem = buffer
                self.state = T.alloc_buffer(
                    [1], "int32", scope="local", align=4, name="semaphore_state"
                )

            @T.macro
            def wait(self):
                with T.thread():
                    while 1:
                        T.ptx.ld_global_acquire(self.state[0], self.sem.access_ptr("r", offset=0))
                        if T.cuda.syncthreads_and(self.state[0] == self.cnt):
                            break
                        T.cuda.nano_sleep(40)

            @T.macro
            def signal(self):
                with T.thread():
                    T.cuda.cta_sync()
                    with T.thread()[0:1]:
                        T.cuda.atomic_add(self.sem.access_ptr("rw", offset=0), 1)
                    T.cuda.thread_fence()

            @T.macro
            def sync(self):
                self.signal()
                self.wait()

        def int_var():
            return T.alloc_buffer([1], "uint32", scope="local", align=4)

        @T.macro
        def zero_smem(tid, smem_buf):
            # zero out the shared memory buffer
            for k in T.serial(T.ceildiv(N_EXPERTS, N_WARPS_DISPATCH * 32)):
                idx = T.meta_var(k * N_WARPS_DISPATCH * 32 + tid)
                if idx < N_EXPERTS:
                    smem_buf[idx] = 0
            T.cuda.cta_sync()

        @T.macro
        def warp_count(lane_id, dst_expert, count, send_experts):
            # threads in a leader warp collectively counts the number of tokens to send to a dest expert
            count[0] = 0
            for k in T.serial(T.ceildiv(M * K, 32)):
                idx = T.meta_var(k * 32 + lane_id)
                if idx < M * K:
                    row = T.meta_var(idx // K)
                    col = T.meta_var(idx % K)
                    expert = send_experts[row, col]
                    if expert == dst_expert:
                        count[0] += 1
            i = int_var()
            i[0] = 16
            while i[0] >= 1:
                count[0] += T.tvm_warp_shuffle_xor(0xFFFFFFFF, count[0], i[0], 32, 32)
                i[0] = i[0] // 2

        @T.macro
        def thread_signal(group_idx, lane_id, dst_expert, count, buf_target_wait):
            # a single thread signal the count
            dst_local_expert = T.meta_var(T.int32(dst_expert % N_LOCAL_EXPERTS))
            dst_rank = T.meta_var(T.int32(dst_expert // N_LOCAL_EXPERTS))
            if lane_id == 0:
                T.nvshmem.signal_op(
                    sig_addr=buf_target_wait.access_ptr(
                        "w", offset=buf_target_wait.elem_offset_of([dst_local_expert, group_idx])
                    ),
                    signal=count[0] + 1,
                    sig_op="set",
                    pe=dst_rank,
                )

        @T.macro
        def thread_accum(tid, token_idx, send_experts, smem_buf):
            # threads collectively accum the token index counter
            if tid < K:
                dst_expert = send_experts[token_idx, tid]
                smem_buf[dst_expert] += 1

        @T.macro
        def thread_prepare_buf(n_threads, tid, token_idx, send_tokens, buf_send):
            # thread collectively prepare the send buffer for the dispatch kernel
            for k in T.serial(T.ceildiv(HIDDEN_DIM, n_threads)):
                idx = T.meta_var(k * n_threads + tid)
                if idx < HIDDEN_DIM:
                    buf_send[token_idx, idx] = send_tokens[token_idx, idx]
            if tid == 0:
                buf_send[token_idx, HIDDEN_DIM] = T.cast(token_idx, dtype)

        @T.macro
        def warp_dispatch_exp(
            group_idx,
            warp_id,
            token_idx,
            smem_buf,
            send_experts,
            buf_send,
            buf_recv,
            buf_actual_wait,
        ):
            # warps collectively send a target token, where each warp sends to one dest expert
            n_warps = T.meta_var(T.int32(N_WARPS_DISPATCH - 1))
            for k in T.serial(T.ceildiv(K, n_warps)):
                exp_idx = T.meta_var(k * n_warps + warp_id)
                if exp_idx < K:
                    dst_expert = send_experts[token_idx, exp_idx]
                    dst_local_expert = T.meta_var(T.int32(dst_expert % N_LOCAL_EXPERTS))
                    dst_rank = T.meta_var(T.int32(dst_expert // N_LOCAL_EXPERTS))
                    dst_index = T.meta_var(T.int32(smem_buf[dst_expert] - 1))
                    T.nvshmem.putmem_signal_nbi.warp(
                        dst=buf_recv.access_ptr(
                            "w",
                            offset=buf_recv.elem_offset_of(
                                [dst_local_expert, group_idx, dst_index, 0]
                            ),
                        ),
                        src=buf_send.access_ptr(
                            "r", offset=buf_send.elem_offset_of([token_idx, 0])
                        ),
                        nelems=(HIDDEN_DIM + 1) * nbytes,
                        sig_addr=buf_actual_wait.access_ptr(
                            "w",
                            offset=buf_actual_wait.elem_offset_of([dst_local_expert, group_idx]),
                        ),
                        signal=1,
                        sig_op="add",
                        pe=dst_rank,
                    )

        @T.macro
        def cta_wait(
            bx,
            tid,
            buf_target_wait,
            buf_actual_wait,
            recv_num_total,
            recv_num_per_expert,
            smem_expert_st,
            smem_token_st,
        ):
            group_offset_cta = T.meta_var(bx * N_RECV_GROUPS_PER_CTA)
            if group_offset_cta < N_RECV_GROUPS:
                group_idx = T.meta_var(tid + group_offset_cta)
                if group_idx < T.min(N_RECV_GROUPS, group_offset_cta + N_RECV_GROUPS_PER_CTA):
                    local_expert = T.meta_var(group_idx // N_GROUP)
                    group_rank = T.meta_var(group_idx % N_GROUP)
                    T.nvshmem.wait_until(
                        ivar=buf_target_wait.access_ptr(
                            "r", offset=buf_target_wait.elem_offset_of([local_expert, group_rank])
                        ),
                        cmp="ne",
                        cmp_value=0,
                    )
                    num_recv_tokens = buf_target_wait[local_expert, group_rank] - 1
                    T.nvshmem.wait_until(
                        ivar=buf_actual_wait.access_ptr(
                            "r", offset=buf_actual_wait.elem_offset_of([local_expert, group_rank])
                        ),
                        cmp="eq",
                        cmp_value=num_recv_tokens,
                    )
                    smem_expert_st[tid] = T.cuda.atomic_add(
                        recv_num_per_expert.access_ptr(
                            "rw", offset=recv_num_per_expert.elem_offset_of([local_expert])
                        ),
                        T.uint32(num_recv_tokens),
                    )
                    smem_token_st[tid] = T.cuda.atomic_add(
                        recv_num_total.access_ptr("rw", offset=0), T.uint32(num_recv_tokens)
                    )
                T.cuda.cta_sync()

        @T.macro
        def thread_compute_meta(
            bx,
            tid,
            buf_meta_index,
            buf_meta_expert,
            buf_meta_offset,
            buf_meta_group,
            buf_meta_token,
            buf_recv,
            buf_target_wait,
            smem_expert_st,
            smem_token_st,
        ):
            # threads collectively compute meta data
            n_threads = T.meta_var(N_WARPS_DISPATCH * 32)
            group_offset_cta = T.meta_var(bx * N_RECV_GROUPS_PER_CTA)
            if group_offset_cta < N_RECV_GROUPS:
                for k in T.serial(T.min(N_RECV_GROUPS_PER_CTA, N_RECV_GROUPS - group_offset_cta)):
                    group_idx = T.meta_var(k + group_offset_cta)
                    local_expert = T.meta_var(group_idx // N_GROUP)
                    group_rank = T.meta_var(group_idx % N_GROUP)
                    num_recv_tokens = buf_target_wait[local_expert, group_rank] - 1
                    for t in T.serial(T.ceildiv(num_recv_tokens, n_threads)):
                        token_idx = T.meta_var(t * n_threads + tid)
                        if token_idx < num_recv_tokens:
                            meta_idx = T.meta_var(smem_token_st[k] + token_idx)
                            buf_meta_index[meta_idx] = T.cast(
                                buf_recv[local_expert, group_rank, token_idx, HIDDEN_DIM], "uint32"
                            )
                            buf_meta_expert[meta_idx] = T.cast(local_expert, "uint32")
                            buf_meta_offset[meta_idx] = T.cast(
                                smem_expert_st[k] + token_idx, "uint32"
                            )
                            buf_meta_group[meta_idx] = T.cast(group_rank, "uint32")
                            buf_meta_token[meta_idx] = T.cast(token_idx, "uint32")

        @T.macro
        def thread_store_tokens(
            bx,
            tid,
            recv_tokens,
            buf_recv,
            recv_num_total,
            buf_meta_expert,
            buf_meta_offset,
            buf_meta_group,
            buf_meta_token,
        ):
            num_tokens = T.meta_var(T.int32(recv_num_total[0]))
            for k in T.serial(T.ceildiv(num_tokens, N_BLOCKS_DISPATCH)):
                token_idx = T.meta_var(k * N_BLOCKS_DISPATCH + bx)
                if token_idx < num_tokens:
                    meta_offset = buf_meta_offset[token_idx]
                    meta_expert = buf_meta_expert[token_idx]
                    meta_group = buf_meta_group[token_idx]
                    meta_token = buf_meta_token[token_idx]
                    n_threads = T.meta_var(N_WARPS_DISPATCH * 32)
                    for k in T.serial(T.ceildiv(HIDDEN_DIM, n_threads * VEC_SIZE)):
                        idx = T.meta_var(k * n_threads * VEC_SIZE + tid * VEC_SIZE)
                        if idx < HIDDEN_DIM:
                            for vec in T.vectorized(VEC_SIZE):
                                recv_tokens[meta_expert, meta_offset, idx + vec] = buf_recv[
                                    meta_expert, meta_group, meta_token, idx + vec
                                ]
                    T.cuda.cta_sync()

        @T.macro
        def thread_prepare_buf_back(
            bx,
            tid,
            n_threads,
            send_num_total,
            send_tokens,
            buf_send_new,
            buf_meta_expert,
            buf_meta_offset,
        ):
            # threads collectively prepare the send buffer for the combine kernel
            total_tokens = T.meta_var(T.int32(send_num_total[0]))
            for k in T.serial(T.ceildiv(total_tokens, N_BLOCKS_COMBINE)):
                token_idx = T.meta_var(k * N_BLOCKS_COMBINE + bx)
                if token_idx < total_tokens:
                    expert = T.int32(buf_meta_expert[token_idx])
                    offset = T.int32(buf_meta_offset[token_idx])
                    for k in T.serial(T.ceildiv(HIDDEN_DIM, n_threads * VEC_SIZE)):
                        idx = T.meta_var(k * n_threads * VEC_SIZE + tid * VEC_SIZE)
                        if idx < HIDDEN_DIM:
                            for vec in T.vectorized(VEC_SIZE):
                                buf_send_new[token_idx, idx + vec] = send_tokens[
                                    expert, offset, idx + vec
                                ]
                    T.cuda.cta_sync()

        @T.macro
        def warp_dispatch_dp(
            rank,
            bx,
            warp_id,
            send_num_total,
            buf_send,
            buf_recv,
            buf_actual_wait,
            buf_meta_expert,
            buf_meta_index,
            buf_meta_group,
        ):
            # warps collectively send a target token, where each warp sends to one dest dp group
            total_tokens = T.meta_var(T.int32(send_num_total[0]))
            for k in T.serial(T.ceildiv(total_tokens, N_BLOCKS_COMBINE)):
                token_idx = T.meta_var(k * N_BLOCKS_COMBINE + bx)
                if token_idx < total_tokens:
                    expert = T.int32(buf_meta_expert[token_idx])
                    index = T.int32(buf_meta_index[token_idx])
                    group = T.int32(buf_meta_group[token_idx])
                    dst_expert = T.meta_var(T.int32(rank * N_LOCAL_EXPERTS + expert))
                    for k in T.serial(T.ceildiv(GROUP_SIZE, N_WARPS_COMBINE)):
                        idx = T.meta_var(k * N_WARPS_COMBINE + warp_id)
                        if idx < GROUP_SIZE:
                            dst_rank = T.meta_var(group * GROUP_SIZE + idx)
                            T.nvshmem.putmem_signal_nbi.warp(
                                dst=buf_recv.access_ptr(
                                    "w", offset=buf_recv.elem_offset_of([dst_expert, index, 0])
                                ),
                                src=buf_send.access_ptr(
                                    "r", offset=buf_send.elem_offset_of([token_idx, 0])
                                ),
                                nelems=HIDDEN_DIM * nbytes,
                                sig_addr=buf_actual_wait.access_ptr(
                                    "w", offset=buf_actual_wait.elem_offset_of([index])
                                ),
                                signal=1,
                                sig_op="add",
                                pe=dst_rank,
                            )

        @T.macro
        def thread_combine(
            bx,
            tid,
            n_threads,
            recv_tokens,
            buf_recv,
            buf_actual_wait,
            send_experts,
            send_weights,
        ):
            # threads collectively compute the weighted sum
            for k in T.serial(T.ceildiv(M, N_BLOCKS_COMBINE)):
                token_idx = k * N_BLOCKS_COMBINE + bx
                if token_idx < M:
                    T.nvshmem.wait_until(
                        ivar=buf_actual_wait.access_ptr(
                            "r", offset=buf_actual_wait.elem_offset_of([token_idx])
                        ),
                        cmp="eq",
                        cmp_value=K,
                    )
                    sum = T.alloc_buffer([1], dtype, scope="local", align=8)
                    for k in T.serial(T.ceildiv(HIDDEN_DIM, n_threads)):
                        idx = T.meta_var(k * n_threads + tid)
                        if idx < HIDDEN_DIM:
                            sum[0] = 0
                            for exp in T.serial(K):
                                expert = send_experts[token_idx, exp]
                                weight = send_weights[token_idx, exp]
                                sum[0] += weight * buf_recv[expert, token_idx, idx]
                            recv_tokens[token_idx, idx] = sum[0]

        # fmt: off
        @T.prim_func(tirx=True)
        def dispatch_kernel(send_tokens: T.Buffer((M, HIDDEN_DIM), dtype), # input tokens to dispatch
                            send_experts: T.Buffer((M, K), "uint32"), # input tokens route to topk experts
                            recv_tokens: T.Buffer((N_LOCAL_EXPERTS, N_GROUP * M, HIDDEN_DIM), dtype), # received tokens
                            recv_num_per_expert: T.Buffer((N_LOCAL_EXPERTS), "uint32"), # number of tokens to receive per local expert
                            recv_num_total: T.Buffer((1,), "uint32"), # total number of tokens to receive
                            buf_target_wait: T.Buffer((N_LOCAL_EXPERTS, N_GROUP), "uint64"), # the number expect to wait on per expert per group
                            buf_actual_wait: T.Buffer((N_LOCAL_EXPERTS, N_GROUP), "uint64"), # the number actually wait on per expert per group
                            buf_send: T.Buffer((M, HIDDEN_DIM + 1), dtype), # send buffer for (token data, token index)
                            buf_recv: T.Buffer((N_LOCAL_EXPERTS, N_GROUP, M, HIDDEN_DIM + 1), dtype), # receive buffer for (token data, token index)
                            buf_meta_index: T.Buffer((MAX_RECV_TOKENS,), "uint32"), # index in the source send_tokens buffer
                            buf_meta_expert: T.Buffer((MAX_RECV_TOKENS,), "uint32"), # local expert index
                            buf_meta_offset: T.Buffer((MAX_RECV_TOKENS,), "uint32"), # token index in its local expert
                            buf_meta_group: T.Buffer((MAX_RECV_TOKENS,), "uint32"), # dp group index
                            buf_meta_token: T.Buffer((MAX_RECV_TOKENS,), "uint32"), # token index in its local expert and data group
                            sem_bar: T.Buffer((1,), "int32"), # semaphore for grid-level barrier
                            ):
            with T.kernel():
                bx = T.cta_id([N_BLOCKS_DISPATCH], parent="kernel")
                warp_id = T.warp_id([N_WARPS_DISPATCH], parent="cta")
                lane_id = T.thread_id([32], parent="warp")
                tid = T.thread_id([N_WARPS_DISPATCH * 32], parent="cta")
                rank = T.nvshmem.my_pe()
                grid_bar = T.meta_var(GridBarrier(N_BLOCKS_DISPATCH, sem_bar))
                group_rank = T.meta_var(T.int32(rank % GROUP_SIZE))
                group_idx = T.meta_var(T.int32(rank // GROUP_SIZE))

                with T.cta():
                    smem_buf = T.alloc_buffer([N_EXPERTS], "uint32", scope="shared", align=128) # for send token index m calculation
                    smem_expert_st = T.alloc_buffer([N_RECV_GROUPS_PER_CTA,], "uint32", scope="shared", align=4) # start index of expert for each CTA
                    smem_token_st = T.alloc_buffer([N_RECV_GROUPS_PER_CTA,], "uint32", scope="shared", align=4) # start index of token for each CTA

                    if DISPATCH_SEND:
                        zero_smem(tid, smem_buf)
                        with T.warp()[N_WARPS_DISPATCH - 1:N_WARPS_DISPATCH]:
                            for k in T.serial(T.ceildiv(N_EXPERTS, N_BLOCKS_DISPATCH * GROUP_SIZE)):
                                dst_expert = T.meta_var(T.uint32(k * N_BLOCKS_DISPATCH * GROUP_SIZE + bx * GROUP_SIZE + group_rank))
                                if dst_expert < N_EXPERTS:
                                    count = int_var()
                                    warp_count(lane_id, dst_expert, count, send_experts)
                                    thread_signal(group_idx, lane_id, dst_expert, count, buf_target_wait)
                        with T.warp()[0:N_WARPS_DISPATCH - 1]:
                            for m in range(M):
                                thread_accum(tid, m, send_experts, smem_buf)
                                if m % (N_BLOCKS_DISPATCH * GROUP_SIZE) == T.int32(bx * GROUP_SIZE + group_rank):
                                    n_threads = T.meta_var(T.int32(N_WARPS_DISPATCH - 1) * 32)
                                    thread_prepare_buf(n_threads, tid, m, send_tokens, buf_send)
                                    T.ptx.bar.sync(1, n_threads)
                                    warp_dispatch_exp(group_idx, warp_id, m, smem_buf, send_experts, buf_send, buf_recv, buf_actual_wait)
                    if DISPATCH_RECV:
                        cta_wait(bx, tid, buf_target_wait, buf_actual_wait, recv_num_total, recv_num_per_expert, smem_expert_st, smem_token_st)
                        thread_compute_meta(bx, tid, buf_meta_index, buf_meta_expert, buf_meta_offset, buf_meta_group, buf_meta_token,
                                            buf_recv, buf_target_wait, smem_expert_st, smem_token_st)
                        grid_bar.sync() # FIXME(@kathy): use fine-grained sync
                        thread_store_tokens(bx, tid, recv_tokens, buf_recv, recv_num_total, buf_meta_expert, buf_meta_offset, buf_meta_group, buf_meta_token)
        # fmt: on

        # fmt: off
        @T.prim_func(tirx=True)
        def combine_kernel(send_tokens: T.Buffer((N_LOCAL_EXPERTS, N_GROUP * M, HIDDEN_DIM), dtype), # send tokens
                        send_num_total: T.Buffer((1,), "uint32"), # total number of tokens to send back
                        send_experts: T.Buffer((M, K), "uint32"), # input tokens route to topk experts
                        send_weights: T.Buffer((M, K), dtype), # input tokens weight for topk experts
                        recv_tokens_new: T.Buffer((M, HIDDEN_DIM), dtype), # final output tokens after weighted sum
                        buf_send_new: T.Buffer((MAX_SEND_TOKENS, HIDDEN_DIM), dtype), # send buffer
                        buf_recv_new: T.Buffer((N_EXPERTS, M, HIDDEN_DIM), dtype), # recv buffer
                        buf_actual_wait_new: T.Buffer((M,), "uint64"), # the number to wait on
                        buf_meta_index: T.Buffer((MAX_RECV_TOKENS,), "uint32"), # index in the source send_tokens buffer
                        buf_meta_expert: T.Buffer((MAX_RECV_TOKENS,), "uint32"), # local expert index
                        buf_meta_offset: T.Buffer((MAX_RECV_TOKENS,), "uint32"), # token index in its local expert
                        buf_meta_group: T.Buffer((MAX_RECV_TOKENS,), "uint32"), # dp group index
                        ):
            with T.kernel():
                bx = T.cta_id([N_BLOCKS_COMBINE], parent="kernel")
                warp_id = T.warp_id([N_WARPS_COMBINE], parent="cta")
                lane_id = T.thread_id([32], parent="warp")
                tid = T.thread_id([N_WARPS_COMBINE * 32], parent="cta")
                rank = T.nvshmem.my_pe()
                n_threads = T.meta_var(N_WARPS_COMBINE * 32)

                with T.cta():
                    if COMBINE_SEND:
                        thread_prepare_buf_back(bx, tid, n_threads, send_num_total, send_tokens, buf_send_new, buf_meta_expert, buf_meta_offset)
                        warp_dispatch_dp(rank, bx, warp_id, send_num_total, buf_send_new, buf_recv_new, buf_actual_wait_new, buf_meta_expert, buf_meta_index, buf_meta_group)
                    if COMBINE_RECV:
                        thread_combine(bx, tid, n_threads, recv_tokens_new, buf_recv_new, buf_actual_wait_new, send_experts, send_weights)
        # fmt: on

        # launch disco runtime
        sess = di.ProcessSession(num_workers=world_size)
        f_init_nvshmem_uid = tvm.get_global_func("runtime.disco.nvshmem.init_nvshmem_uid")
        uid = f_init_nvshmem_uid()
        init_dfunc = sess.get_global_func("runtime.disco.nvshmem.init_nvshmem")
        init_dfunc(uid, world_size, 0)
        sess.sync_worker_0()

        # construct test data
        send_tokens_np = [np.random.rand(M, HIDDEN_DIM).astype(dtype) for _ in range(N_GROUP)]
        send_experts_np = [
            np.array(
                [np.random.choice(np.arange(N_EXPERTS), size=K, replace=False) for _ in range(M)]
            ).astype("uint32")
            for _ in range(N_GROUP)
        ]
        send_weights_np = [np.random.rand(M, K).astype(dtype) for _ in range(N_GROUP)]
        sem_bar_np = np.zeros((1,), dtype="int32")
        recv_num_per_expert_np = np.zeros((N_LOCAL_EXPERTS,), dtype="uint32")
        recv_num_total_np = np.zeros((1,), dtype="uint32")
        buf_target_wait_np = np.zeros((N_LOCAL_EXPERTS, N_GROUP), dtype="uint64")
        buf_actual_wait_np = np.zeros((N_LOCAL_EXPERTS, N_GROUP), dtype="uint64")
        buf_actual_wait_new_np = np.zeros((M,), dtype="uint64")

        # alloc test data
        nvshmem_malloc_hook = sess.get_global_func("runtime.disco.nvshmem.empty")
        args_dict = {
            "send_tokens_array": sess.empty((M, HIDDEN_DIM), dtype),
            "send_experts_array": sess.empty((M, K), "uint32"),
            "send_weights_array": sess.empty((M, K), dtype),
            "recv_tokens_array": sess.empty((N_LOCAL_EXPERTS, N_GROUP * M, HIDDEN_DIM), dtype),
            "recv_tokens_new_array": sess.empty((M, HIDDEN_DIM), dtype),
            "recv_num_per_expert_array": sess.empty((N_LOCAL_EXPERTS,), "uint32"),
            "recv_num_total_array": sess.empty((1,), "uint32"),
            "buf_meta_index_array": sess.empty((MAX_RECV_TOKENS,), "uint32"),
            "buf_meta_expert_array": sess.empty((MAX_RECV_TOKENS,), "uint32"),
            "buf_meta_offset_array": sess.empty((MAX_RECV_TOKENS,), "uint32"),
            "buf_meta_group_array": sess.empty((MAX_RECV_TOKENS,), "uint32"),
            "buf_meta_token_array": sess.empty((MAX_RECV_TOKENS,), "uint32"),
            "sem_bar_array": sess.empty((1,), "int32"),
            "buf_send_array": nvshmem_malloc_hook(ShapeTuple((M, HIDDEN_DIM + 1)), dtype, None),
            "buf_recv_array": nvshmem_malloc_hook(
                ShapeTuple((N_LOCAL_EXPERTS, N_GROUP, M, HIDDEN_DIM + 1)), dtype, None
            ),
            "buf_send_new_array": nvshmem_malloc_hook(
                ShapeTuple((MAX_SEND_TOKENS, HIDDEN_DIM)), dtype, None
            ),
            "buf_recv_new_array": nvshmem_malloc_hook(
                ShapeTuple((N_EXPERTS, M, HIDDEN_DIM)), dtype, None
            ),
            "buf_actual_wait_array": nvshmem_malloc_hook(
                ShapeTuple((N_LOCAL_EXPERTS, N_GROUP)), "uint64", None
            ),
            "buf_actual_wait_new_array": nvshmem_malloc_hook(ShapeTuple((M,)), "uint64", None),
            "buf_target_wait_array": nvshmem_malloc_hook(
                ShapeTuple((N_LOCAL_EXPERTS, N_GROUP)), "uint64", None
            ),
        }

        for i in range(world_size):
            args_dict["send_tokens_array"].debug_copy_from(i, send_tokens_np[i // GROUP_SIZE])
            args_dict["send_experts_array"].debug_copy_from(i, send_experts_np[i // GROUP_SIZE])
            args_dict["send_weights_array"].debug_copy_from(i, send_weights_np[i // GROUP_SIZE])
        sess.sync_worker_0()

        res_dict = {}
        for key in args_dict:
            res_dict[key.replace("_array", "_res")] = []

        with tempfile.TemporaryDirectory() as tmpdir:
            # compile kernels
            target = tvm.target.Target("cuda")
            dispatch_path = tmpdir + "/dispatch.so"
            dispatch_mod = tvm.compile(dispatch_kernel, target=target, tir_pipeline="tirx")
            dispatch_mod.export_library(dispatch_path)
            combine_path = tmpdir + "/combine.so"
            combine_mod = tvm.compile(combine_kernel, target=target, tir_pipeline="tirx")
            combine_mod.export_library(combine_path)

            # launch kernels
            rt_dispatch_mod = sess.load_vm_module(dispatch_path)
            rt_combine_mod = sess.load_vm_module(combine_path)
            barrier_dfunc = sess.get_global_func(
                "runtime.disco.nvshmem.barrier_all_on_current_stream"
            )
            sess._sync_all()
            for i in range(N_REPEAT):
                barrier_dfunc()
                rt_dispatch_mod["dispatch_kernel"](
                    args_dict["send_tokens_array"],
                    args_dict["send_experts_array"],
                    args_dict["recv_tokens_array"],
                    args_dict["recv_num_per_expert_array"],
                    args_dict["recv_num_total_array"],
                    args_dict["buf_target_wait_array"],
                    args_dict["buf_actual_wait_array"],
                    args_dict["buf_send_array"],
                    args_dict["buf_recv_array"],
                    args_dict["buf_meta_index_array"],
                    args_dict["buf_meta_expert_array"],
                    args_dict["buf_meta_offset_array"],
                    args_dict["buf_meta_group_array"],
                    args_dict["buf_meta_token_array"],
                    args_dict["sem_bar_array"],
                )
                barrier_dfunc()
                rt_combine_mod["combine_kernel"](
                    args_dict["recv_tokens_array"],
                    args_dict["recv_num_total_array"],
                    args_dict["send_experts_array"],
                    args_dict["send_weights_array"],
                    args_dict["recv_tokens_new_array"],
                    args_dict["buf_send_new_array"],
                    args_dict["buf_recv_new_array"],
                    args_dict["buf_actual_wait_new_array"],
                    args_dict["buf_meta_index_array"],
                    args_dict["buf_meta_expert_array"],
                    args_dict["buf_meta_offset_array"],
                    args_dict["buf_meta_group_array"],
                )
                sess._sync_all()
                if i < N_REPEAT - 1:
                    for i in range(world_size):
                        args_dict["sem_bar_array"].debug_copy_from(i, sem_bar_np)
                        args_dict["recv_num_per_expert_array"].debug_copy_from(
                            i, recv_num_per_expert_np
                        )
                        args_dict["recv_num_total_array"].debug_copy_from(i, recv_num_total_np)
                        args_dict["buf_target_wait_array"].debug_copy_from(i, buf_target_wait_np)
                        args_dict["buf_actual_wait_array"].debug_copy_from(i, buf_actual_wait_np)
                        args_dict["buf_actual_wait_new_array"].debug_copy_from(
                            i, buf_actual_wait_new_np
                        )
                    sess._sync_all()

            # validate results
            for key in args_dict:
                for i in range(world_size):
                    res_key = key.replace("_array", "_res")
                    res_dict[res_key].append(args_dict[key].debug_get_from_remote(i).numpy())
                res_dict[res_key] = np.stack(res_dict[res_key], axis=0)

            # sync all workers to make sure the temporary files are cleaned up after all workers
            # finish the execution
            sess._sync_all()

        finalize_dfunc = sess.get_global_func("runtime.disco.nvshmem.finalize_nvshmem")
        finalize_dfunc()
        sess.sync_worker_0()

        # validate results
        print("Validating results...")

        np.testing.assert_equal(
            res_dict["send_tokens_res"],
            np.repeat(np.stack(send_tokens_np, axis=0), GROUP_SIZE, axis=0),
        )
        np.testing.assert_equal(
            res_dict["send_experts_res"],
            np.repeat(np.stack(send_experts_np, axis=0), GROUP_SIZE, axis=0),
        )
        np.testing.assert_equal(
            res_dict["send_weights_res"],
            np.repeat(np.stack(send_weights_np, axis=0), GROUP_SIZE, axis=0),
        )

        i_vals = np.arange(world_size)[:, np.newaxis]
        m_vals = np.arange(M)[np.newaxis, :]
        mask = (m_vals % GROUP_SIZE) == (i_vals % GROUP_SIZE)
        buf_send_ref = np.zeros((world_size, M, HIDDEN_DIM + 1), dtype=dtype)
        buf_send_ref[:, :, :HIDDEN_DIM][mask] = res_dict["send_tokens_res"][mask]
        buf_send_ref[:, :, HIDDEN_DIM] = np.where(mask, np.arange(M), 0)
        np.testing.assert_equal(res_dict["buf_send_res"], buf_send_ref)

        buf_actual_wait_ref = np.zeros((world_size, N_LOCAL_EXPERTS, N_GROUP), dtype=np.uint64)
        for g in range(N_GROUP):
            counts = np.bincount(send_experts_np[g].ravel(), minlength=N_EXPERTS)
            buf_actual_wait_ref[:, :, g] = counts[:N_EXPERTS].reshape(world_size, N_LOCAL_EXPERTS)
        np.testing.assert_equal(res_dict["buf_actual_wait_res"], buf_actual_wait_ref)
        np.testing.assert_equal(res_dict["buf_target_wait_res"], buf_actual_wait_ref + 1)
        np.testing.assert_equal(
            res_dict["recv_num_total_res"], np.sum(buf_actual_wait_ref, axis=(1, 2))[:, np.newaxis]
        )
        np.testing.assert_equal(
            res_dict["recv_num_per_expert_res"], np.sum(buf_actual_wait_ref, axis=2)
        )

        target_vals = np.arange(N_EXPERTS).reshape(world_size, N_LOCAL_EXPERTS)
        mask = np.stack(send_experts_np)[..., np.newaxis, np.newaxis] == target_vals.reshape(
            1, 1, world_size, N_LOCAL_EXPERTS
        )
        mask = np.any(mask, axis=2).transpose(
            2, 3, 0, 1
        )  # (world_size, N_LOCAL_EXPERTS, N_GROUP, M)
        dest_row_map = np.cumsum(mask, axis=3) - 1
        ranks_idx, exps_idx, groups_idx, m_idx = np.where(mask)
        dest_row_idx = dest_row_map[ranks_idx, exps_idx, groups_idx, m_idx]
        buf_recv_ref = np.zeros((world_size, N_LOCAL_EXPERTS, N_GROUP, M, HIDDEN_DIM + 1))
        buf_recv_ref[ranks_idx, exps_idx, groups_idx, dest_row_idx, :HIDDEN_DIM] = np.stack(
            send_tokens_np
        )[groups_idx, m_idx]
        buf_recv_ref[ranks_idx, exps_idx, groups_idx, dest_row_idx, HIDDEN_DIM] = m_idx
        np.testing.assert_equal(res_dict["buf_recv_res"], buf_recv_ref)

        for i in range(world_size):
            buf_meta_res_stack0 = np.stack(
                [
                    res_dict["buf_meta_expert_res"][i],
                    res_dict["buf_meta_group_res"][i],
                    res_dict["buf_meta_token_res"][i],
                    res_dict["buf_meta_index_res"][i],
                ]
            )
            sorted_indices = np.lexsort(buf_meta_res_stack0[::-1])
            buf_meta_res_stack0 = buf_meta_res_stack0[:, sorted_indices]
            # construct ref sol
            expert_indices, group_indices = np.indices(res_dict["buf_actual_wait_res"][i].shape)
            flat_counts = res_dict["buf_actual_wait_res"][i].ravel().astype(np.int64)
            total_counts = res_dict["recv_num_total_res"][i].item()
            row0 = np.repeat(expert_indices.ravel(), flat_counts)
            row1 = np.repeat(group_indices.ravel(), flat_counts)
            group_st_idx = np.cumsum(flat_counts) - flat_counts
            row2 = np.arange(total_counts) - np.repeat(group_st_idx, flat_counts)
            row3 = buf_recv_ref[i, row0, row1, row2, HIDDEN_DIM].astype(np.uint32)
            buf_meta_ref_stack0 = np.vstack([row0, row1, row2, row3])
            buf_meta_ref_stack0 = np.pad(
                buf_meta_ref_stack0,
                pad_width=((0, 0), (MAX_RECV_TOKENS - total_counts, 0)),
                mode="constant",
                constant_values=0,
            )
            np.testing.assert_equal(buf_meta_res_stack0, buf_meta_ref_stack0)

            buf_meta_res_stack1 = np.stack(
                [res_dict["buf_meta_expert_res"][i], res_dict["buf_meta_offset_res"][i]]
            )
            sorted_indices = np.lexsort(buf_meta_res_stack1[::-1])
            buf_meta_res_stack1 = buf_meta_res_stack1[:, sorted_indices]
            # construct ref sol
            if total_counts == 0:
                # if no tokens to receive, we should not have any meta data
                buf_meta_ref_stack1 = np.zeros((2, MAX_RECV_TOKENS), dtype=np.int64)
                np.testing.assert_equal(buf_meta_res_stack1, buf_meta_ref_stack1)
                continue
            g_st = np.concatenate(([True], np.diff(row0) != 0))
            row4 = np.arange(total_counts) - np.arange(total_counts)[g_st][np.cumsum(g_st) - 1]
            buf_meta_ref_stack1 = np.vstack([row0, row4])
            buf_meta_ref_stack1 = np.pad(
                buf_meta_ref_stack1,
                pad_width=((0, 0), (MAX_RECV_TOKENS - total_counts, 0)),
                mode="constant",
                constant_values=0,
            )
            np.testing.assert_equal(buf_meta_res_stack1, buf_meta_ref_stack1)

            expert_idx = res_dict["buf_meta_expert_res"][i, :total_counts]
            group_idx = res_dict["buf_meta_group_res"][i, :total_counts]
            token_idx = res_dict["buf_meta_token_res"][i, :total_counts]
            offset_idx = res_dict["buf_meta_offset_res"][i, :total_counts]
            recv_tokens_ref = np.zeros((N_LOCAL_EXPERTS, N_GROUP * M, HIDDEN_DIM), dtype=dtype)
            recv_tokens_ref[expert_idx, offset_idx] = buf_recv_ref[
                i, expert_idx, group_idx, token_idx, :HIDDEN_DIM
            ]
            np.testing.assert_equal(res_dict["recv_tokens_res"][i], recv_tokens_ref)

            buf_send_new_ref = np.zeros((MAX_SEND_TOKENS, HIDDEN_DIM), dtype=dtype)
            buf_send_new_ref[:total_counts] = buf_recv_ref[
                i, expert_idx, group_idx, token_idx, :HIDDEN_DIM
            ]
            np.testing.assert_equal(res_dict["buf_send_new_res"][i], buf_send_new_ref)

        np.testing.assert_equal(
            res_dict["buf_actual_wait_new_res"], np.ones((world_size, M), dtype=np.uint64) * K
        )

        mask = (
            res_dict["send_experts_res"][:, np.newaxis, :, :]
            == np.arange(N_EXPERTS)[np.newaxis, :, np.newaxis, np.newaxis]
        )
        mask = np.any(mask, axis=3)
        buf_recv_new_ref = np.where(
            mask[..., np.newaxis], res_dict["send_tokens_res"][:, np.newaxis, :, :], 0
        )
        np.testing.assert_equal(res_dict["buf_recv_new_res"], buf_recv_new_ref)

        rank_idx = np.arange(world_size)[:, np.newaxis, np.newaxis]
        m_idx = np.arange(M)[np.newaxis, :, np.newaxis]
        recv_tokens_new_ref = np.sum(
            buf_recv_new_ref[rank_idx, res_dict["send_experts_res"], m_idx]
            * res_dict["send_weights_res"][..., np.newaxis],
            axis=2,
        )
        np.testing.assert_allclose(res_dict["recv_tokens_new_res"], recv_tokens_new_ref, atol=1e-6)

        print("Results all correct.")
        return True

    p = PopenWorker()
    p.send(test_func)
    assert p.recv()


if __name__ == "__main__":
    test_dispatch_combine()
