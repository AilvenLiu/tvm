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

import os
import sys

import numpy as np
import pytest
import torch

import tvm
from tvm.tirx.bench.utils import ProtonContext, bench

sys.path.insert(
    0,
    os.path.join(
        os.environ.get("TIRX_KERNELS_PATH", os.path.expanduser("~/tirx-kernels/kernels")),
        "attention",
    ),
)
from flash_attention4 import *


@pytest.mark.parametrize("seq_len", [8192, 4096, 2048, 1024])
@pytest.mark.parametrize("num_qo_heads", [32])
@pytest.mark.parametrize("num_kv_heads", [4, 8, 16, 32])
@pytest.mark.parametrize("is_causal", [False, True])
def test_flash_attention4(seq_len, num_qo_heads, num_kv_heads, is_causal):
    BATCH = 1
    SEQ_Q = seq_len
    SEQ_KV = seq_len
    NUM_QO_HEADS = num_qo_heads
    NUM_KV_HEADS = num_kv_heads
    HEAD_DIM = 128
    DEBUG = False

    def flops(ms):
        """Calculate FLOPS for Flash Attention: Q@K^T + P@V = 2 * B * H * S_q * S_k * D"""
        # For causal, effective ops is approximately half
        effective_factor = 0.5 if is_causal else 1.0
        return 4 * BATCH * NUM_QO_HEADS * SEQ_Q * SEQ_KV * HEAD_DIM * effective_factor / (ms * 1e-3)

    Q, K, V, _ = prepare_data(BATCH, SEQ_Q, SEQ_KV, NUM_QO_HEADS, NUM_KV_HEADS, HEAD_DIM)

    def get_source(func):
        target = tvm.target.Target("cuda")
        mod = tvm.IRModule({"main": func})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
        print(mod.mod.imports[0].inspect_source(), flush=True)
        return mod

    def tir_attn(Q, K, V):
        Q_tir, K_tir, V_tir = Q, K, V
        O_tir = torch.zeros_like(Q)

        prim_func = get_flash_attention4_kernel(
            BATCH, SEQ_Q, SEQ_KV, NUM_QO_HEADS, NUM_KV_HEADS, HEAD_DIM, is_causal=is_causal
        )
        mod = get_source(prim_func)

        dev = tvm.cuda(0)
        Q_tvm = tvm.runtime.tensor(Q_tir.cpu().numpy(), device=dev)
        K_tvm = tvm.runtime.tensor(K_tir.cpu().numpy(), device=dev)
        V_tvm = tvm.runtime.tensor(V_tir.cpu().numpy(), device=dev)
        O_tvm = tvm.runtime.tensor(O_tir.cpu().numpy(), device=dev)
        profiler_buffer = np.zeros((PROFILER_BUFFER_SIZE,), dtype=np.uint64)
        profiler_buffer_tvm = tvm.runtime.tensor(profiler_buffer, dev)

        def func():
            return mod(Q_tvm, K_tvm, V_tvm, O_tvm, profiler_buffer_tvm)

        ms = bench(func, warmup=100, repeat=300, proton_name="tir_fa4", debug=DEBUG)
        print(f"TIR FA4: {flops(ms) / 1e12:.2f} TFLOPS, time: {ms:.3f} ms", flush=True)
        if PROFILER_ON:
            export_to_perfetto_trace(
                profiler_buffer_tvm.numpy(),
                f"fa4-{BATCH}-{SEQ_Q}-{SEQ_KV}-{NUM_QO_HEADS}-{NUM_KV_HEADS}-{HEAD_DIM}.perfetto-trace",
                event_type_names,
            )

        mod(Q_tvm, K_tvm, V_tvm, O_tvm, profiler_buffer_tvm)
        torch.cuda.synchronize()

        O_res = O_tvm.numpy()
        return O_res

    def cutedsl_attn(Q, K, V):
        """CuTeDSL Blackwell FMHA baseline"""
        try:
            import math
            import os
            import sys

            current_dir = os.path.dirname(os.path.abspath(__file__))
            tvm_root = os.path.abspath(os.path.join(current_dir, "../../../../../../"))
            blackwell_path = os.path.join(
                tvm_root, "3rdparty/cutlass/examples/python/CuTeDSL/blackwell"
            )
            sys.path.insert(0, blackwell_path)
            import cutlass
            import cutlass.cute as cute
            import cutlass.torch as cutlass_torch
            from fmha import BlackwellFusedMultiHeadAttentionForward, MaskType
        except ImportError as e:
            print(f"CuTeDSL Blackwell FMHA not available: {e}, skipping baseline")
            return None

        Q_cute = Q.cuda()
        K_cute = K.cuda()
        V_cute = V.cuda()
        O_cute = torch.zeros_like(Q_cute)

        q_tensor, q_torch = cutlass_torch.cute_tensor_like(
            Q_cute, cutlass.Float16, is_dynamic_layout=True, assumed_align=16
        )
        k_tensor, k_torch = cutlass_torch.cute_tensor_like(
            K_cute, cutlass.Float16, is_dynamic_layout=True, assumed_align=16
        )
        v_tensor, v_torch = cutlass_torch.cute_tensor_like(
            V_cute, cutlass.Float16, is_dynamic_layout=True, assumed_align=16
        )
        o_tensor, o_torch = cutlass_torch.cute_tensor_like(
            O_cute, cutlass.Float16, is_dynamic_layout=True, assumed_align=16
        )

        q_torch.copy_(Q_cute)
        k_torch.copy_(K_cute)
        v_torch.copy_(V_cute)

        mma_tiler = (128, 128, HEAD_DIM)
        fmha = BlackwellFusedMultiHeadAttentionForward(
            cutlass.Float32,
            cutlass.Float32,
            mma_tiler,
            is_persistent=True,
            mask_type=MaskType.NO_MASK if not is_causal else MaskType.CAUSAL_MASK,
        )

        current_stream = cutlass_torch.default_stream()

        scale_softmax = 1.0 / math.sqrt(HEAD_DIM)
        log2_e = math.log2(math.exp(1.0))
        scale_softmax_log2 = scale_softmax * log2_e
        scale_output = 1.0

        problem_size = (BATCH, SEQ_Q, SEQ_KV, NUM_QO_HEADS, NUM_KV_HEADS, HEAD_DIM)
        cum_seqlen_q = None
        cum_seqlen_k = None

        compiled_fmha = cute.compile(
            fmha,
            q_tensor.iterator,
            k_tensor.iterator,
            v_tensor.iterator,
            o_tensor.iterator,
            problem_size,
            cum_seqlen_q,
            cum_seqlen_k,
            scale_softmax_log2,
            scale_output,
            current_stream,
        )

        def run_fmha():
            compiled_fmha(
                q_tensor.iterator,
                k_tensor.iterator,
                v_tensor.iterator,
                o_tensor.iterator,
                problem_size,
                cum_seqlen_q,
                cum_seqlen_k,
                scale_softmax_log2,
                scale_output,
                current_stream,
            )

        ms = bench(run_fmha, warmup=10, repeat=30, proton_name="cutedsl_fa4", debug=DEBUG)
        print(f"CuTeDSL FA: {flops(ms) / 1e12:.2f} TFLOPS, time: {ms:.3f} ms", flush=True)

        # Run once for result
        run_fmha()
        torch.cuda.synchronize()

        return o_torch.cpu().numpy()

    def flashattn_sm100(Q, K, V):
        """Flash-Attention SM100 implementation from installed flash-attn package

        Note: Requires flash-attn to be installed with:
            pip install flash-attn --no-build-isolation
        """
        try:
            import math

            import cutlass
            import cutlass.cute as cute
            import cutlass.torch as cutlass_torch
            from flash_attn.cute.flash_fwd_sm100 import FlashAttentionForwardSm100
        except ImportError as e:
            print(f"Flash-Attention SM100 not available: {e}")
            print("Install with: pip install flash-attn --no-build-isolation")
            print("Note: CuTeDSL baseline uses the same CUTLASS implementation")
            return None
        except Exception as e:
            print(f"Unexpected error loading Flash-Attention SM100: {e}")
            return None

        Q_fa = Q.cuda()
        K_fa = K.cuda()
        V_fa = V.cuda()
        O_fa = torch.zeros_like(Q_fa)

        q_tensor, q_torch = cutlass_torch.cute_tensor_like(
            Q_fa, cutlass.Float16, is_dynamic_layout=True, assumed_align=16
        )
        k_tensor, k_torch = cutlass_torch.cute_tensor_like(
            K_fa, cutlass.Float16, is_dynamic_layout=True, assumed_align=16
        )
        v_tensor, v_torch = cutlass_torch.cute_tensor_like(
            V_fa, cutlass.Float16, is_dynamic_layout=True, assumed_align=16
        )
        o_tensor, o_torch = cutlass_torch.cute_tensor_like(
            O_fa, cutlass.Float16, is_dynamic_layout=True, assumed_align=16
        )

        q_torch.copy_(Q_fa)
        k_torch.copy_(K_fa)
        v_torch.copy_(V_fa)

        fa_fwd = FlashAttentionForwardSm100(
            head_dim=HEAD_DIM,
            head_dim_v=HEAD_DIM,
            qhead_per_kvhead=NUM_QO_HEADS // NUM_KV_HEADS,  # GQA
            is_causal=is_causal,
            is_local=False,
            pack_gqa=False,
            m_block_size=128,
            n_block_size=128,
            is_persistent=True,
        )

        current_stream = cutlass_torch.default_stream()

        scale_softmax = 1.0 / math.sqrt(HEAD_DIM)

        compiled_fa = cute.compile(
            fa_fwd,
            q_tensor,
            k_tensor,
            v_tensor,
            o_tensor,
            None,  # lse_tensor
            scale_softmax,
            current_stream,
            None,  # mCuSeqlensQ
            None,  # mCuSeqlensK
            None,  # mSeqUsedQ
            None,  # mSeqUsedK
            None,  # mPageTable
            None,  # softcap
            None,  # window_size_left
            None,  # window_size_right
            None,  # learnable_sink
        )

        def run_fa():
            compiled_fa(
                q_tensor,
                k_tensor,
                v_tensor,
                o_tensor,
                None,  # lse_tensor
                scale_softmax,
                current_stream,
                None,  # mCuSeqlensQ
                None,  # mCuSeqlensK
                None,  # mSeqUsedQ
                None,  # mSeqUsedK
                None,  # mPageTable
                None,  # softcap
                None,  # window_size_left
                None,  # window_size_right
                None,  # learnable_sink
            )

        ms = bench(run_fa, warmup=100, repeat=300, proton_name="flashattn_sm100", debug=DEBUG)
        print(
            f"Flash-Attention SM100: {flops(ms) / 1e12:.2f} TFLOPS, time: {ms:.3f} ms", flush=True
        )

        run_fa()
        torch.cuda.synchronize()

        return o_torch.cpu().numpy()

    def flashinfer(Q, K, V):
        import flashinfer

        workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
        prefill_wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            workspace_buffer, "NHD", backend="cutlass"
        )
        qo_indptr = torch.tensor([0, SEQ_Q], device="cuda:0", dtype=torch.int32)
        kv_indptr = torch.tensor([0, SEQ_KV], device="cuda:0", dtype=torch.int32)
        prefill_wrapper.plan(
            qo_indptr,
            kv_indptr,
            num_qo_heads=NUM_QO_HEADS,
            num_kv_heads=NUM_KV_HEADS,
            head_dim_qk=HEAD_DIM,
        )
        q_torch = Q.clone().reshape(-1, NUM_QO_HEADS, HEAD_DIM).cuda()
        k_torch = K.clone().reshape(-1, NUM_KV_HEADS, HEAD_DIM).cuda()
        v_torch = V.clone().reshape(-1, NUM_KV_HEADS, HEAD_DIM).cuda()

        def run_flashinfer():
            o_torch = prefill_wrapper.run(q_torch, k_torch, v_torch)
            return o_torch

        bench(run_flashinfer, warmup=100, repeat=300, proton_name="flashinfer", debug=DEBUG)
        o_torch = run_flashinfer()
        return o_torch.cpu().numpy().reshape(BATCH, SEQ_Q, NUM_QO_HEADS, HEAD_DIM)

    with ProtonContext("blackwell_fa4", debug=DEBUG):
        print("\nRunning CuTeDSL FA4 baseline...")
        O_cutedsl = cutedsl_attn(Q, K, V)

        print("\nRunning Flash-Attention SM100 baseline...")
        O_flashattn = flashattn_sm100(Q, K, V)

        print("Running TIR Flash Attention...")
        O_tir = tir_attn(Q, K, V)

        print("\nRunning FlashInfer FA4 baseline...")
        # O_flashinfer = flashinfer(Q, K, V)
        O_flashinfer = None

    # Compare with CuTeDSL FA
    if O_cutedsl is not None:
        print("\n=== TIR vs CuTeDSL FA4 ===")
        diff_cute = np.abs(O_tir - O_cutedsl)
        rtol, atol = 1e-2, 1e-2

        abs_ref = np.abs(O_cutedsl)
        valid_mask = abs_ref > atol
        rel_diff_cute = np.zeros_like(diff_cute)
        if np.any(valid_mask):
            rel_diff_cute[valid_mask] = diff_cute[valid_mask] / abs_ref[valid_mask]

        max_rel_err = np.max(rel_diff_cute) if np.any(valid_mask) else 0.0
        mismatch_mask_cute = (diff_cute > atol) & (rel_diff_cute > rtol)
        num_mismatches_cute = np.sum(mismatch_mask_cute)

        print(
            f"max_abs_err={np.max(diff_cute):.6f}, max_rel_err={max_rel_err:.6f}, "
            f"mismatches={num_mismatches_cute}/{O_tir.size} ({100.0 * num_mismatches_cute / O_tir.size:.2f}%)"
        )

        np.testing.assert_allclose(O_tir, O_cutedsl, rtol=rtol, atol=atol)
        print("\nVerification passed!")
    else:
        print("\nCuTeDSL FA4 baseline not available, skipping comparison")

    # Compare with Flash-Attention4 SM100
    if O_flashattn is not None:
        print("\n=== TIR vs Flash-Attention4 SM100 ===")
        diff_fa = np.abs(O_tir - O_flashattn)
        rtol, atol = 1e-2, 1e-2

        abs_ref = np.abs(O_flashattn)
        valid_mask = abs_ref > atol
        rel_diff_fa = np.zeros_like(diff_fa)
        if np.any(valid_mask):
            rel_diff_fa[valid_mask] = diff_fa[valid_mask] / abs_ref[valid_mask]

        max_rel_err = np.max(rel_diff_fa) if np.any(valid_mask) else 0.0
        mismatch_mask_fa = (diff_fa > atol) & (rel_diff_fa > rtol)
        num_mismatches_fa = np.sum(mismatch_mask_fa)

        print(
            f"max_abs_err={np.max(diff_fa):.6f}, max_rel_err={max_rel_err:.6f}, "
            f"mismatches={num_mismatches_fa}/{O_tir.size} ({100.0 * num_mismatches_fa / O_tir.size:.2f}%)"
        )

        np.testing.assert_allclose(O_tir, O_flashattn, rtol=rtol, atol=atol)
        print("\nVerification vs Flash-Attention SM100 passed!")
    else:
        print("\nFlash-Attention SM100 baseline not available, skipping comparison")

    # Compare with FlashInfer FA4
    if O_flashinfer is not None:
        print("\n=== TIR vs FlashInfer FA4 ===")
        diff_flashinfer = np.abs(O_tir - O_flashinfer)
        rtol, atol = 1e-2, 1e-2

        abs_ref = np.abs(O_flashinfer)
        valid_mask = abs_ref > atol
        rel_diff_flashinfer = np.zeros_like(diff_flashinfer)
        if np.any(valid_mask):
            rel_diff_flashinfer[valid_mask] = diff_flashinfer[valid_mask] / abs_ref[valid_mask]

        max_rel_err = np.max(rel_diff_flashinfer) if np.any(valid_mask) else 0.0
        mismatch_mask_flashinfer = (diff_flashinfer > atol) & (rel_diff_flashinfer > rtol)
        num_mismatches_flashinfer = np.sum(mismatch_mask_flashinfer)

        print(
            f"max_abs_err={np.max(diff_flashinfer):.6f}, max_rel_err={max_rel_err:.6f}, "
            f"mismatches={num_mismatches_flashinfer}/{O_tir.size} ({100.0 * num_mismatches_flashinfer / O_tir.size:.2f}%)"
        )


if __name__ == "__main__":
    test_flash_attention4(8192, 32, 8, is_causal=False)
    test_flash_attention4(8192, 32, 8, is_causal=True)
    # TODO: causal attention for non-GQA kernel is still 10% slower than FA4.
    # likely due to register pressure issue
    test_flash_attention4(8192, 32, 32, is_causal=True)
    test_flash_attention4(8192, 32, 32, is_causal=False)
