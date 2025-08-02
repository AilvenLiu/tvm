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
from ..utils import bench, ProtonContext

# TODO: split-kv; merge


def ceildiv(a, b):
    return (a + b - 1) // b

def find_power_of_two(n):
    assert n > 0 and (n & (n - 1)) == 0
    return n.bit_length() - 1

# problem size
KV_LAYOUT = "HND"
BATCH_SIZE = 32
NUM_HEADS = 37
SEQ_LEN = 4096
HEAD_DIM = 128
QO_SHAPE = BATCH_SIZE, NUM_HEADS, HEAD_DIM
# KV_SHAPE = BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM (logically)

# paged cache size
PAGE_SIZE = 16
PAGE_NUM = ceildiv(SEQ_LEN, PAGE_SIZE)
TOTAL_PAGE_NUM = PAGE_NUM * BATCH_SIZE
MAX_PAGE_NUM = 8192
KV_CACHE_SHAPE = MAX_PAGE_NUM, 2, NUM_HEADS, PAGE_SIZE, HEAD_DIM
assert TOTAL_PAGE_NUM <= MAX_PAGE_NUM

# other
F16_BYTE = 2
F32_BYTE = 4
SM_SCALE = (1 / 0.6931471805599453) * (1 / HEAD_DIM ** 0.5)
SM_COUNT = 148

def perpare_data():
    import torch

    page_last_len = PAGE_SIZE if SEQ_LEN % PAGE_SIZE == 0 else SEQ_LEN % PAGE_SIZE

    kv_indptr = torch.empty(BATCH_SIZE + 1, dtype=torch.int32).int()
    for i in range(BATCH_SIZE + 1):
        kv_indptr[i] = i * PAGE_NUM
    kv_last_page_len = torch.empty(BATCH_SIZE, dtype=torch.int32).int()
    for i in range(BATCH_SIZE):
        kv_last_page_len[i] = page_last_len
    kv_indices = torch.arange(MAX_PAGE_NUM, dtype=torch.int32).int()
    kv_indices = kv_indices[torch.randperm(MAX_PAGE_NUM)]
    kv_indices = kv_indices[:TOTAL_PAGE_NUM]
    q = torch.randn(QO_SHAPE).half()
    kv_data = torch.randn(KV_CACHE_SHAPE).half()

    return q, kv_data, kv_indptr, kv_last_page_len, kv_indices




@tvm.testing.requires_cuda_compute_version(10, exact=True)
def test():

    # kernel config
    VEC_SIZE = 8
    assert VEC_SIZE * F16_BYTE >= 16 and VEC_SIZE * 32 >= HEAD_DIM # ensure cp_async_size >= 128bit && bdx <= 32
    NUM_THREADS = 256
    BDX = HEAD_DIM // VEC_SIZE
    BDY = NUM_THREADS // BDX
    TILE_SIZE_PER_BDX = 4
    PIPE_DEPTH = 2
    SMEM_SIZE = 2 * PIPE_DEPTH * TILE_SIZE_PER_BDX * BDY * HEAD_DIM * F16_BYTE + max(VEC_SIZE, TILE_SIZE_PER_BDX) * BDY * BDX * F32_BYTE
    assert PIPE_DEPTH <= BDX

    def ldg(addr):
        return T.cuda.func_call("ldg", addr, source_code=f"""
__forceinline__ __device__ int ldg(int* src) {{
    return __ldg(src);
}}                                                        
        """, return_type="int32")

    @T.prim_func(tirp=True)
    def decode(Q_ptr: T.handle, KV_ptr: T.handle, O_ptr: T.handle, KV_indptr: T.handle, KV_last_page_len: T.handle, KV_indices: T.handle):

        Q_global = T.match_buffer(Q_ptr, QO_SHAPE, "float16", scope="global")
        KV_global = T.match_buffer(KV_ptr, KV_CACHE_SHAPE, "float16", scope="global")
        O_global = T.match_buffer(O_ptr, QO_SHAPE, "float16", scope="global")
        KV_indptr_global = T.match_buffer(KV_indptr, [BATCH_SIZE + 1], "int32", scope="global")
        KV_last_page_len_global = T.match_buffer(KV_last_page_len, [BATCH_SIZE], "int32", scope="global")
        KV_indices_global = T.match_buffer(KV_indices, [TOTAL_PAGE_NUM], "int32", scope="global")

        # f16 -> f32
        @T.macro
        def load_cast(dst: T.handle, src: T.handle, tmp: T.handle, vec_size):
            T.cuda.func_call("load_cast", dst, src, tmp, vec_size,
                            source_code=f"""
            __forceinline__ __device__ void load_cast(void* dst, void* src, void* tmp, int vec_size) {{
                for (int i = 0; i < vec_size / 8; ++i) {{
                    ((int4*)tmp)[i] = ((int4*)src)[i];
                }}
                for (int i = 0; i < vec_size / 2; ++i) {{
                    ((float2*)dst)[i] = __half22float2(((half2*)tmp)[i]);
                }}
            }}
            """)
        
        # f32 -> f16
        @T.macro
        def store_cast(dst: T.handle, src: T.handle, tmp: T.handle, vec_size):
            T.cuda.func_call("store_cast", dst, src, tmp, vec_size,
                            source_code=f"""
            __forceinline__ __device__ void store_cast(void* dst, void* src, void* tmp, int vec_size) {{
                for (int i = 0; i < vec_size / 2; ++i) {{
                    ((half2*)tmp)[i] = __float22half2_rn(((float2*)src)[i]);
                }}
                    for (int i = 0; i < vec_size / 8; ++i) {{
                    ((int4*)dst)[i] = ((int4*)tmp)[i];
                }}
            }}
            """)

        # f16 -> f32
        @T.macro
        def cast(dst: T.handle, src: T.handle, vec_size):
            T.cuda.func_call("cast", dst, src, vec_size, 
                            source_code=f"""
            __forceinline__ __device__ void cast(void* dst, void* src, int vec_size) {{
                for (int i = 0; i < vec_size / 2; ++i) {{
                    ((float2*)dst)[i] = __half22float2(((half2*)src)[i]);
                }}
            }}
            """)

        with T.kernel():
            bx, by = T.cta_id([BATCH_SIZE, NUM_HEADS], parent="kernel")
            tx, ty = T.thread_id([BDX, BDY], parent="cta")

            with T.cta():
                # allocate the memory
                buf = T.alloc_buffer([SMEM_SIZE], "uint8", scope="shared.dyn")
                k_smem = T.decl_buffer([PIPE_DEPTH, BDY, TILE_SIZE_PER_BDX, HEAD_DIM], "float16", buf.data, elem_offset=0)
                v_smem = T.decl_buffer([PIPE_DEPTH, BDY, TILE_SIZE_PER_BDX, HEAD_DIM], "float16", buf.data, elem_offset=PIPE_DEPTH * TILE_SIZE_PER_BDX * BDY * HEAD_DIM)
                kv_offset_load = T.decl_buffer([TILE_SIZE_PER_BDX, BDY, BDX], "int32", buf.data, elem_offset=2 * PIPE_DEPTH * TILE_SIZE_PER_BDX * BDY * HEAD_DIM * F16_BYTE // F32_BYTE)
                kv_offset_use = T.decl_buffer([BDX, BDY, TILE_SIZE_PER_BDX], "int32", buf.data, elem_offset=2 * PIPE_DEPTH * TILE_SIZE_PER_BDX * BDY * HEAD_DIM * F16_BYTE // F32_BYTE)
                epi_o = T.decl_buffer([BDY, BDX, VEC_SIZE], "float32", buf.data, elem_offset=0)
                epi_md = T.decl_buffer([BDY, 2], "float32", buf.data, elem_offset=BDY * BDX * VEC_SIZE)

                with T.thread():

                    # allocate the reg
                    idx = T.alloc_local([1], "int32")
                    tmp = T.alloc_local([VEC_SIZE], "float16")
                    q = T.alloc_local([VEC_SIZE], "float32")
                    k = T.alloc_local([VEC_SIZE], "float32")
                    v = T.alloc_local([VEC_SIZE], "float32")
                    s = T.alloc_local([TILE_SIZE_PER_BDX], "float32")
                    page_base = T.alloc_local([1], "int32")
                    indices = T.alloc_local([1], "int32")
                    kv_offset_cp = T.alloc_local([TILE_SIZE_PER_BDX], "int32")
                    o = T.alloc_local([VEC_SIZE], "float32")
                    m = T.alloc_local([2], "float32")
                    d = T.alloc_local([2], "float32")
                    m_tmp = T.alloc_local([1], "float32")
                    d_tmp = T.alloc_local([1], "float32")
                    o_tmp = T.alloc_local([VEC_SIZE], "float32")

                    @T.macro
                    def fetch_kv_offset(kt, idx1):
                        token_id = T.meta_var(page_base[0] + idx1)
                        p = T.meta_var(token_id // PAGE_SIZE)
                        r = T.meta_var(token_id % PAGE_SIZE)
                        indices[0] = ldg(KV_indices_global.ptr_to([p]))
                        kv_offset_load[kt, ty, tx] = indices[0] * 2 * NUM_HEADS * PAGE_SIZE * HEAD_DIM + by * PAGE_SIZE * HEAD_DIM + r * HEAD_DIM
                        
                    # fetch q
                    load_cast(T.address_of(q), Q_global.ptr_to([bx, by, tx * VEC_SIZE]), T.address_of(tmp), VEC_SIZE)

                    # fetch kv-offset
                    page_base[0] = KV_indptr_global[bx] * PAGE_SIZE 
                    for kt in T.unroll(TILE_SIZE_PER_BDX):
                       fetch_kv_offset(kt, (kt * BDY + ty) * BDX + tx)
                    T.ptx.fence.proxy("shared")
                    T.ptx.bar.sync(1, NUM_THREADS)

                    # fetch K&V
                    for kp in T.unroll(PIPE_DEPTH):
                        # get kv-offset used in cp
                        for kt in T.unroll(TILE_SIZE_PER_BDX):
                            kv_offset_cp[kt] = kv_offset_use[kp, ty, kt] + tx * VEC_SIZE

                        # fetch K
                        for kt in T.unroll(TILE_SIZE_PER_BDX):
                            for kc in T.unroll(VEC_SIZE * F16_BYTE // 16):
                                T.ptx.cp_async("float16", k_smem.ptr_to([kp, ty, kt, tx * VEC_SIZE + kc * 16 // F16_BYTE]), 0, 
                                               KV_global.data, kv_offset_cp[kt] + kc * 16 // F16_BYTE, 16)
                        T.ptx.cp_async.commit_group()

                        # fetch V
                        for kt in T.unroll(TILE_SIZE_PER_BDX):
                            for kc in T.unroll(VEC_SIZE * F16_BYTE // 16):
                                T.ptx.cp_async("float16", v_smem.ptr_to([kp, ty, kt, tx * VEC_SIZE + kc * 16 // F16_BYTE]), 0, 
                                               KV_global.data, NUM_HEADS * PAGE_SIZE * HEAD_DIM + kv_offset_cp[kt] + kc * 16 // F16_BYTE, 16)
                        T.ptx.cp_async.commit_group()

                    # initilize the value
                    idx[0] = 0
                    for kv in T.serial(VEC_SIZE):
                        o[kv] = 0.0
                    m[0] = 0.0
                    d[0] = 1.0
                    # pipeline
                    for ki in T.serial(SEQ_LEN // (TILE_SIZE_PER_BDX * BDY)):
                        # fetch new kv-offset
                        # TODO: ADD condition here
                        if ((ki + PIPE_DEPTH) % BDX == 0):
                            for kt in T.unroll(TILE_SIZE_PER_BDX):
                                fetch_kv_offset(kt, (ki + PIPE_DEPTH) * TILE_SIZE_PER_BDX * BDY + (kt * BDY + ty) * BDX + tx)
                            T.ptx.fence.proxy("shared")
                        
                        # compute qk
                        T.ptx.cp_async.wait_group(2 * PIPE_DEPTH - 1) # wait for K
                        T.ptx.bar.sync(1, NUM_THREADS)
                        m[1] = m[0]
                        for kt in T.unroll(TILE_SIZE_PER_BDX):
                            # cast k to f32
                            cast(T.address_of(k), k_smem.ptr_to([idx[0], ty, kt, VEC_SIZE * tx]), VEC_SIZE)
                            s[kt] = 0.0
                            # local gemm
                            for kv in T.unroll(VEC_SIZE):
                                s[kt] += q[kv] * k[kv]
                            # reduce from other tx's sum
                            for kr in T.unroll(find_power_of_two(BDX // 2) + 1):
                                s[kt] = s[kt] + T.tvm_warp_shuffle_xor(0xFFFFFFFF, s[kt], (BDX // 2) >> kr, 32, 32)
                            s[kt] *= SM_SCALE
                            # update max value
                            m[0] = T.max(m[0], s[kt])
                        
                        # update the sum for softmax
                        o_scale = T.meta_var(T.exp2(m[1] - m[0]))
                        d[0] *= o_scale
                        for kt in T.unroll(TILE_SIZE_PER_BDX):
                            s[kt] = T.exp2(s[kt] - m[0])
                            d[0] += s[kt]
                        for kv in T.unroll(VEC_SIZE):
                            o[kv] = o[kv] * o_scale
                        T.ptx.bar.sync(1, NUM_THREADS)

                        # get kv-offset used in cp
                        for kt in T.unroll(TILE_SIZE_PER_BDX):
                            kv_offset_cp[kt] = kv_offset_use[(ki + PIPE_DEPTH) % BDX, ty, kt] + tx * VEC_SIZE

                        # fetch K
                        if ki + PIPE_DEPTH < SEQ_LEN // (TILE_SIZE_PER_BDX * BDY):
                            for kt in T.unroll(TILE_SIZE_PER_BDX):
                                for kc in T.unroll(VEC_SIZE * F16_BYTE // 16):
                                    T.ptx.cp_async("float16", k_smem.ptr_to([idx[0], ty, kt, tx * VEC_SIZE + kc * 16 // F16_BYTE]), 0, 
                                                KV_global.data, kv_offset_cp[kt] + kc * 16 // F16_BYTE, 16)
                        T.ptx.cp_async.commit_group()

                        # calculate softmax(qk)v
                        T.ptx.cp_async.wait_group(2 * PIPE_DEPTH - 1) # wait for V
                        T.ptx.bar.sync(1, NUM_THREADS)
                        for kt in T.unroll(TILE_SIZE_PER_BDX):
                            cast(T.address_of(v), v_smem.ptr_to([idx[0], ty, kt, VEC_SIZE * tx]), VEC_SIZE)
                            for kv in T.unroll(VEC_SIZE):
                                o[kv] += s[kt] * v[kv]
                        T.ptx.bar.sync(1, NUM_THREADS)

                        # fetch V
                        if ki + PIPE_DEPTH < SEQ_LEN // (TILE_SIZE_PER_BDX * BDY):
                            for kt in T.unroll(TILE_SIZE_PER_BDX):
                                for kc in T.unroll(VEC_SIZE * F16_BYTE // 16):
                                    T.ptx.cp_async("float16", v_smem.ptr_to([idx[0], ty, kt, tx * VEC_SIZE + kc * 16 // F16_BYTE]), 0, 
                                                KV_global.data, NUM_HEADS * PAGE_SIZE * HEAD_DIM + kv_offset_cp[kt] + kc * 16 // F16_BYTE, 16)
                        T.ptx.cp_async.commit_group()
                        idx[0] = (idx[0] + 1) % PIPE_DEPTH
                    
                    T.ptx.cp_async.wait_group(0)
                    T.ptx.bar.sync(1, NUM_THREADS)

                    # prepare o,m,d in smem for merging
                    for kv in T.unroll(VEC_SIZE):
                        epi_o[ty, tx, kv] = o[kv]
                    if tx == 0:
                        epi_md[ty, 0] = m[0]
                        epi_md[ty, 1] = d[0]
                    T.ptx.fence.proxy("shared")
                    T.ptx.bar.sync(1, NUM_THREADS)
                    # merge o through different ty
                    if ty == 0:
                        m[0] = 0.0
                        d[0] = 1.0
                        for kv in T.unroll(VEC_SIZE):
                            o[kv] = 0.0
                        for ky in T.unroll(BDY):
                            m_tmp[0] = epi_md[ky, 0]
                            d_tmp[0] = epi_md[ky, 1]
                            for kv in T.unroll(VEC_SIZE):
                                o_tmp[kv] = epi_o[ky, tx, kv]
                            m[1] = m[0]
                            d[1] = d[0]
                            m[0] = T.max(m[1], m_tmp[0])
                            d[0] = d[1] * T.exp2(m[1] - m[0]) + d_tmp[0] * T.exp2(m_tmp[0] - m[0])
                            for kv in T.unroll(VEC_SIZE):
                                o[kv] = o[kv] * T.exp2(m[1] - m[0]) + o_tmp[kv] * T.exp2(m_tmp[0] - m[0])
                        # normalize
                        for kv in T.unroll(VEC_SIZE):
                            o[kv] = o[kv] / d[0]
                        # store to global mem
                        store_cast(O_global.ptr_to([bx, by, tx * VEC_SIZE]), T.address_of(o), T.address_of(tmp), VEC_SIZE)
                    T.ptx.bar.sync(1, NUM_THREADS)


    q, kv_data, kv_indptr, kv_last_page_len, kv_indices = perpare_data()

    def tir():
        DEV = tvm.cuda(0)
        q_tvm = tvm.nd.array(q, DEV)
        kv_data_tvm = tvm.nd.array(kv_data, DEV)
        kv_indptr_tvm = tvm.nd.array(kv_indptr, DEV)
        kv_last_page_len_tvm = tvm.nd.array(kv_last_page_len, DEV)
        kv_indices_tvm = tvm.nd.array(kv_indices, DEV)
        o_tvm = tvm.nd.array(np.zeros(QO_SHAPE, dtype=np.float16), DEV)

        target = tvm.target.Target("cuda")
        with target:
            target = tvm.target.Target("cuda")
            mod = tvm.IRModule({"main": decode})
            mod = tvm.compile(mod, target=target, tir_pipeline="tirp")
            src = mod.mod.imported_modules[0].get_source()
            # print(src)
            func = lambda: mod(q_tvm, kv_data_tvm, o_tvm, kv_indptr_tvm, kv_last_page_len_tvm, kv_indices_tvm)
            ms = bench(func, warmup=0, repeat=100, proton_name="tir")
            print(f"TIR time: {ms:.3f} ms")
            func()

        return o_tvm.numpy()

    def flashinfer():
        import flashinfer
        import torch

        q_f = q.to(0)
        kv_data_f = kv_data.to(0)
        kv_indptr_f = kv_indptr.to(0)
        kv_last_page_len_f = kv_last_page_len.to(0)
        kv_indices_f = kv_indices.to(0)

        workspace_buffer = torch.empty(1024 * 1024 * 1024, dtype=torch.int8).to(0)
        wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace_buffer, KV_LAYOUT)
        wrapper.plan(
            kv_indptr_f,
            kv_indices_f,
            kv_last_page_len_f,
            NUM_HEADS,
            NUM_HEADS,
            HEAD_DIM,
            PAGE_SIZE,
            pos_encoding_mode="NONE",
            data_type=torch.float16,
            q_data_type=torch.float16,
        )
        o = wrapper.run(q_f, kv_data_f)

        wrapper_tensor_cores = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            workspace_buffer, KV_LAYOUT, use_tensor_cores=True
        )
        wrapper_tensor_cores.plan(
            kv_indptr_f,
            kv_indices_f,
            kv_last_page_len_f,
            NUM_HEADS,
            NUM_HEADS,
            HEAD_DIM,
            PAGE_SIZE,
            pos_encoding_mode="NONE",
            data_type=torch.float16,
            q_data_type=torch.float16,
        )
        o_tc = wrapper_tensor_cores.run(q_f, kv_data_f)

        torch.testing.assert_close(o, o_tc, rtol=1e-3, atol=1e-3)


        func = lambda: wrapper.run(q_f, kv_data_f)
        ms = bench(func, warmup=0, repeat=100, proton_name="flashinfer")
        func()
        print(f"FlashInfer time: {ms:.3f} ms")

        return o.reshape(BATCH_SIZE, NUM_HEADS, HEAD_DIM).cpu().numpy()
    
    with ProtonContext("batch_decode"):
        O_flashinfer = flashinfer()
        O_tir = tir()
        np.testing.assert_allclose(O_tir, O_flashinfer, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    test()
