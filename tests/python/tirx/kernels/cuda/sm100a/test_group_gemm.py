import numpy as np
import pytest
import torch
from torch.nn import functional as F
import functools

import tvm
import tvm.testing
from tvm.script import tir as T
from tvm.script import tirx as Tx
from tvm.tirx.bench.utils import ProtonContext, bench
from tvm.tirx.megakernel.group_gemm_sm100 import GroupGEMMTile
from tvm.tirx.megakernel.common import SmemManager, KernelConfig, ceildiv
from tvm.tirx.tile_scheduler import GroupMajor2D

from sglang.srt.layers.moe.fused_moe_triton.fused_moe_triton_kernels import invoke_fused_moe_kernel
from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
from sglang.srt.layers.moe.fused_moe_triton.fused_moe_triton_config import (
    try_get_optimal_moe_config,
)
from triton import language as tl


test_configs = [
    {
        "batch_size": b,  # M
        "hidden_size": 2048,  # K
        "num_experts": 128,  # E
        "top_k": 8,  # Top-K
        "intermediate_size": 768,  # N
    }
    for b in [2048, 512, 128, 1]
]

DEBUG = False
MAX_BLK_M = 128


def compute_routing(router_logits: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute routing weights and selected experts from router logits.

    Args:
        router_logits (torch.Tensor): Router logits of shape [batch_size, num_experts]
        top_k (int): Number of experts to route to per token

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - routing_weights: Expert weights of shape [batch_size, top_k]
            - selected_experts: Expert indices of shape [batch_size, top_k]
    """
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.float()
    return routing_weights, selected_experts


def prepare_group_gemm(BLK_M, num_experts, selected_experts):
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        selected_experts, BLK_M, num_experts
    )
    sorted_token_ids = sorted_token_ids[:]
    return sorted_token_ids, expert_ids, num_tokens_post_padded


def gen_input(batch_size, hidden_size, num_experts, top_k, intermediate_size):
    torch.manual_seed(42)
    e = num_experts
    m = batch_size
    n = intermediate_size
    k = hidden_size
    otype = torch.float16
    router_logits = torch.randn(m, e, dtype=otype).cuda()
    routing_weights, selected_experts = compute_routing(router_logits, top_k)

    x1 = torch.randn(m, k, dtype=otype).cuda()
    w13 = torch.randn((e, 2 * n, k), device="cuda", dtype=otype)
    x2 = torch.randn(m, top_k, n, dtype=otype).reshape(batch_size * top_k, intermediate_size).cuda()
    w2 = torch.randn((e, k, n), device="cuda", dtype=otype)

    selected_experts = selected_experts.to(torch.int)

    return (
        x1,
        w13,
        routing_weights,
        selected_experts,
        x2,
        w2,
    )


def get_group_gemm_kernel(K, E, top_k, N, acc_output=False, low_batch=True):

    # fmt: off
    @T.prim_func(tirx=True)
    def group_gemm(
        A_ptr: T.handle,
        B_ptr: T.handle,
        C_ptr: T.handle,
        expert_ids_ptr: T.handle,
        sorted_token_ids_ptr: T.handle,
        routing_weights_ptr: T.handle,
        valid_num_tokens_ptr: T.handle,
        num_tokens_post_padded: T.Buffer((1), "int32"),
    ):
        M = T.int32()
        A = T.match_buffer(A_ptr, (M, K), "float16")
        B = T.match_buffer(B_ptr, (E, N, K), "float16")
        C = T.match_buffer(C_ptr, (M, N), "float16")
        MAX_EXPERT_IDS = T.int32()
        expert_ids = T.match_buffer(expert_ids_ptr, (MAX_EXPERT_IDS), "int32")
        sorted_token_ids = T.match_buffer(sorted_token_ids_ptr, (M), "int32")
        valid_num_tokens = T.match_buffer(valid_num_tokens_ptr, (M // MAX_BLK_M), "int32")
        numel = T.int32()
        routing_weights = T.match_buffer(routing_weights_ptr, (numel), "float32")
        A_tensor_map_128: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
        A_tensor_map_64: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
        A_tensor_map_32: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
        B_tensor_map: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
        C_tensor_map_128: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
        C_tensor_map_64: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
        C_tensor_map_32: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
        group_gemm_tile = T.meta_var(
            GroupGEMMTile(
                N, K, E, top_k, numel, "float16", "float16", acc_output=acc_output, low_batch=low_batch
            )
        )
        group_gemm_tile.set_tensor_map([A_tensor_map_128, A_tensor_map_64, A_tensor_map_32], B_tensor_map, [C_tensor_map_128, C_tensor_map_64, C_tensor_map_32], A, B, C)
        group_gemm_tile.host_init()
        with T.kernel():
            cta_cnt = T.meta_var(KernelConfig.SM_NUMBER)  # persistent kernel
            bx = T.cta_id([cta_cnt], parent="kernel")
            tid = T.thread_id([KernelConfig.NUM_THREADS], parent="cta")

            buf = T.alloc_buffer([KernelConfig.MAX_SMEM_SIZE], "uint8", scope="shared.dyn")
            smem_manager = T.meta_var(SmemManager(KernelConfig.MAX_SMEM_SIZE, 16384, buf.data))
            smem_manager.set_tile(group_gemm_tile)
            group_gemm_tile.init(smem_manager)
            smem_manager.pool_allocator.move_base_to(16384*14) # FIXME: this should be fixed in smem manager
            smem_manager.set_tile(group_gemm_tile.__class__)
            GroupGEMMTile.class_init(smem_manager)
            M_TILE_CNT = T.meta_var(ceildiv(num_tokens_post_padded[0], MAX_BLK_M))
            N_TILE_CNT = ceildiv(N, GroupGEMMTile.BLK_N)
            tile_scheduler = T.meta_var(GroupMajor2D("sched", M_TILE_CNT, N_TILE_CNT, 1, step=cta_cnt))
            tile_scheduler.init(bx)
            smem_manager.init()
            while tile_scheduler.valid():
                smem_manager.enter_tile_runtime(group_gemm_tile)
                group_gemm_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, 0, expert_ids, routing_weights, sorted_token_ids, valid_num_tokens,None)
                tile_scheduler.next_tile()
            GroupGEMMTile.class_finalize()
    return group_gemm
    # fmt: on


@pytest.mark.skip(reason="Tensor Memory Leak")
@pytest.mark.parametrize("task", test_configs)
def test_group_gemm(task):
    batch_size = task["batch_size"]
    hidden_size = task["hidden_size"]
    num_experts = task["num_experts"]
    top_k = task["top_k"]
    intermediate_size = task["intermediate_size"]

    (
        x1,
        w13,
        routing_weights,
        selected_experts,
        x2,
        w2,
    ) = gen_input(batch_size, hidden_size, num_experts, top_k, intermediate_size)

    def tir():
        def torch_to_tvm(tensor):
            return tvm.runtime.from_dlpack(torch.utils.dlpack.to_dlpack(tensor))

        target = tvm.target.Target("cuda")
        dev = tvm.cuda(0)
        sorted_token_ids, expert_ids, num_tokens_post_padded = prepare_group_gemm(
            MAX_BLK_M, num_experts, selected_experts
        )
        safe_index = torch.clamp(sorted_token_ids, 0, batch_size * top_k - 1)[
            : num_tokens_post_padded.item()
        ]
        x1_tvm = torch_to_tvm(x1[safe_index // top_k])
        x2_tvm = torch_to_tvm(x2[safe_index])
        w13_tvm = torch_to_tvm(w13)
        routing_weights_tvm = torch_to_tvm(
            routing_weights.view(
                batch_size * top_k,
            )
        )
        w2_tvm = torch_to_tvm(w2)
        out1_tvm = tvm.runtime.empty(
            (num_tokens_post_padded.item(), 2 * intermediate_size), dtype="float16", device=dev
        )
        out2_tvm = tvm.runtime.empty(
            (num_tokens_post_padded.item(), hidden_size), dtype="float16", device=dev
        )

        sorted_token_ids_tvm = torch_to_tvm(sorted_token_ids[: num_tokens_post_padded.item()])
        valid_num_tokens = torch.empty(num_tokens_post_padded.item() // MAX_BLK_M, dtype=torch.int32)
        for i in range(num_tokens_post_padded.item() // MAX_BLK_M):
            valid_num_tokens[i] = torch.sum((sorted_token_ids[i * MAX_BLK_M:(i + 1) * MAX_BLK_M] >= 0) & (sorted_token_ids[i * MAX_BLK_M:(i + 1) * MAX_BLK_M] < batch_size * top_k))
        valid_num_tokens_tvm = torch_to_tvm(valid_num_tokens.cuda())
        expert_ids_tvm = torch_to_tvm(expert_ids)
        num_tokens_post_padded_tvm = torch_to_tvm(num_tokens_post_padded)
        grp_gemm1 = get_group_gemm_kernel(
            hidden_size,
            num_experts,
            top_k,
            intermediate_size * 2,
            acc_output=False,
            low_batch=batch_size<2048,
        )
        grp_gemm2 = get_group_gemm_kernel(
            intermediate_size,
            num_experts,
            top_k,
            hidden_size,
            acc_output=True,
            low_batch=batch_size < 2048,
        )
        with target:
            mod_1 = tvm.IRModule({"main": grp_gemm1})
            mod_1 = tvm.compile(mod_1, target=target, tir_pipeline="tirx")
            mod_2 = tvm.IRModule({"main": grp_gemm2})
            mod_2 = tvm.compile(mod_2, target=target, tir_pipeline="tirx")

            def func1():
                mod_1(
                    x1_tvm,
                    w13_tvm,
                    out1_tvm,
                    expert_ids_tvm,
                    sorted_token_ids_tvm,
                    routing_weights_tvm,
                    valid_num_tokens_tvm,
                    num_tokens_post_padded_tvm,
                )

            def func2():
                mod_2(
                    x2_tvm,
                    w2_tvm,
                    out2_tvm,
                    expert_ids_tvm,
                    sorted_token_ids_tvm,
                    routing_weights_tvm,
                    valid_num_tokens_tvm,
                    num_tokens_post_padded_tvm,
                )

            ms = bench(func1, warmup=0, repeat=30, proton_name="tir1")
            ms = bench(func2, warmup=0, repeat=30, proton_name="tir2")
            # reset out2 to get the correct result
            out2_tvm = tvm.runtime.empty(
                (num_tokens_post_padded.item(), hidden_size), dtype="float16", device=dev
            )
            mod_2(
                x2_tvm,
                w2_tvm,
                out2_tvm,
                expert_ids_tvm,
                sorted_token_ids_tvm,
                routing_weights_tvm,
                valid_num_tokens_tvm,
                num_tokens_post_padded_tvm,
            )

            def scatter_out(out_tvm, *shape):
                tmp_out = torch.from_numpy(out_tvm.numpy())
                out = torch.zeros(shape, dtype=torch.float16)
                index_for_scatter = sorted_token_ids[: num_tokens_post_padded.item()].cpu()
                for i in range(num_tokens_post_padded.item()):
                    if index_for_scatter[i] >= 0 and index_for_scatter[i] < shape[0]:
                        out[index_for_scatter[i]] = tmp_out[i]
                return out

            out1 = scatter_out(out1_tvm, batch_size * top_k, intermediate_size * 2).view(
                batch_size, top_k, intermediate_size * 2
            )
            out2 = out2_tvm.numpy()[:batch_size, :]
            return out1, out2

    def sglang():
        def get_config(batch_size):
            get_config_func = functools.partial(
                try_get_optimal_moe_config,
                w13.shape,
                (w2.shape[0], w2.shape[1], w2.shape[2]),
                selected_experts.shape[1],
                "float16",
                block_shape=None,
            )
            return get_config_func(batch_size)

        sgL_config = get_config(batch_size)
        print(f"sgL_config: {sgL_config}")
        sorted_token_ids, expert_ids, num_tokens_post_padded = prepare_group_gemm(
            sgL_config["BLOCK_SIZE_M"], num_experts, selected_experts
        )

        out1 = torch.empty(
            (batch_size, top_k, 2 * intermediate_size), dtype=torch.float16, device="cuda"
        )
        out2 = torch.empty((batch_size, top_k, hidden_size), dtype=torch.float16, device="cuda")

        func1 = lambda: invoke_fused_moe_kernel(
            x1,
            w13,
            None,  # bias
            out1,
            None,  # A_scale
            None,  # B_scale
            None,  # B_zp
            topk_weights=routing_weights,
            topk_ids=selected_experts,
            sorted_token_ids=sorted_token_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=num_tokens_post_padded,
            mul_routed_weight=False,
            top_k=top_k,
            config=sgL_config,
            compute_type=tl.float16,
            use_fp8_w8a8=False,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            per_channel_quant=False,
            block_shape=None,
        )
        ms1 = bench(func1, warmup=3, repeat=30, proton_name="sglang gemm1", debug=DEBUG)
        print("sglang gemm1", ms1)

        func2 = lambda: invoke_fused_moe_kernel(
            x2,
            w2,
            None,  # bias
            out2,
            None,  # A_scale
            None,  # B_scale
            None,  # B_zp
            topk_weights=routing_weights,
            topk_ids=selected_experts,
            sorted_token_ids=sorted_token_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=num_tokens_post_padded,
            mul_routed_weight=True,
            top_k=1,
            config=sgL_config,
            compute_type=tl.float16,
            use_fp8_w8a8=False,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            per_channel_quant=False,
            block_shape=None,
        )

        ms2 = bench(func2, warmup=3, repeat=30, proton_name="sglang gemm2", debug=DEBUG)
        print("sglang gemm2", ms2)
        return out1.cpu().numpy(), out2.sum(dim=1).cpu().numpy()

    with ProtonContext("group_gemm", debug=DEBUG):
        out1_tir, out2_tir = tir()
        out1_sglang, out2_sglang = sglang()
    try:
        np.testing.assert_allclose(out1_tir, out1_sglang, rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(out2_tir, out2_sglang, rtol=2e-2, atol=2e-2)
    except Exception as e:
        print(e)

if __name__ == "__main__":
    for config in test_configs:
        print(f"testing config: {config}", flush=True)
        test_group_gemm(config)
