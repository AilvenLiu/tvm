import numpy as np
import pytest
import torch

import tvm
import tvm.testing
from tvm.script import tir as T
from tvm.script import tirp as Tp
from tvm.tirp.bench.utils import ProtonContext, bench
from tvm.tirp.megakernel.moe_align import MOEAlignTile, CountAndSortExpertTokens
from tvm.tirp.megakernel.common import SmemManager, KernelConfig

from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size

NUM_EXPERTS = 128
TOPK = 8
BLOCK_SIZE = 16


def get_moe_align_kernel(pad_sorted_token_ids):
    # fmt: off
    @T.prim_func(tirp=True)
    def moe_align_kernel(
        topk_ids_ptr: T.handle,
        sorted_token_ids_ptr: T.handle,
        expert_ids_ptr: T.handle,
        num_tokens_post_pad: T.Buffer((1,), "int32"),
        cumsum_buffer: T.Buffer((NUM_EXPERTS+1,), "int32"),
    ):
        num_tokens = T.int32()    
        topk_ids = T.match_buffer(topk_ids_ptr, (num_tokens, TOPK), dtype="int64")
        max_num_tokens_padded = T.int32()
        sorted_token_ids = T.match_buffer(sorted_token_ids_ptr, (max_num_tokens_padded,), dtype="int32")
        expert_ids = T.match_buffer(expert_ids_ptr, (max_num_tokens_padded // BLOCK_SIZE,), dtype="int32")
        moe_align_tile = T.meta_var(MOEAlignTile(NUM_EXPERTS, num_tokens * TOPK, BLOCK_SIZE, pad_sorted_token_ids=pad_sorted_token_ids))
        with T.kernel():
            bx = T.cta_id([1], parent="kernel")
            tid = T.thread_id([KernelConfig.NUM_THREADS], parent="cta")
            buf = T.alloc_buffer([KernelConfig.MAX_SMEM_SIZE], "uint8", scope="shared.dyn")
            pool = T.meta_var(Tp.PoolAllocator(buf.data))
            smem_manager = T.meta_var(SmemManager(KernelConfig.MAX_SMEM_SIZE, 16384, pool))
            smem_manager.set_tile(moe_align_tile.__class__)
            moe_align_tile.init(smem_manager)
            smem_manager.init()
            topk_ids_flattened = topk_ids.view(-1)
            moe_align_tile.run(0, 0, 0, topk_ids_flattened, sorted_token_ids, expert_ids, num_tokens_post_pad, cumsum_buffer)
    return moe_align_kernel
    # fmt: on


@T.prim_func(tirp=True)
def count_and_sort_expert_tokens_kernel(
    topk_ids_ptr: T.handle,
    sorted_token_ids_ptr: T.handle,
    cumsum_buffer: T.Buffer((NUM_EXPERTS + 1,), "int32"),
):
    num_tokens = T.int32()
    topk_ids = T.match_buffer(topk_ids_ptr, (num_tokens, TOPK), dtype="int64")
    max_num_tokens_padded = T.int32()
    sorted_token_ids = T.match_buffer(sorted_token_ids_ptr, (max_num_tokens_padded,), dtype="int32")
    count_and_sort_tile = T.meta_var(CountAndSortExpertTokens(num_tokens * TOPK))
    with T.kernel():
        bx = T.cta_id([KernelConfig.SM_NUMBER], parent="kernel")
        tid = T.thread_id([KernelConfig.NUM_THREADS], parent="cta")
        topk_ids_flattened = topk_ids.view(-1)
        count_and_sort_tile.run(bx, 0, 0, topk_ids_flattened, sorted_token_ids, cumsum_buffer)


def test(num_tokens, num_experts, topk, pad_sorted_token_ids):
    topk_ids = torch.argsort(torch.rand(num_tokens, num_experts), dim=1)[:, :topk]
    max_num_tokens_padded = topk_ids.numel() + num_experts * (BLOCK_SIZE - 1)
    sorted_ids = torch.empty((max_num_tokens_padded,), dtype=torch.int32)
    if not pad_sorted_token_ids:
        sorted_ids.fill_(topk_ids.numel())
    max_num_m_blocks = max_num_tokens_padded // BLOCK_SIZE
    expert_ids = torch.zeros((max_num_m_blocks,), dtype=torch.int32)
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32)
    cumsum_buffer = torch.empty(num_experts + 1, dtype=torch.int32)

    dev = tvm.cuda()
    topk_ids_tvm = tvm.runtime.tensor(topk_ids.numpy(), device=dev)
    sorted_ids_tvm = tvm.runtime.tensor(sorted_ids.numpy(), device=dev)
    expert_ids_tvm = tvm.runtime.tensor(expert_ids.numpy(), device=dev)
    num_tokens_post_pad_tvm = tvm.runtime.tensor(num_tokens_post_pad.numpy(), device=dev)
    cumsum_buffer_tvm = tvm.runtime.tensor(cumsum_buffer.numpy(), device=dev)

    def tir():
        target = tvm.target.Target("cuda")
        with target:
            mod = tvm.IRModule(
                {
                    "moe_align_kernel": get_moe_align_kernel(pad_sorted_token_ids),
                    "count_and_sort_expert_tokens_kernel": count_and_sort_expert_tokens_kernel,
                }
            )
            mod = tvm.compile(mod, target=target, tir_pipeline="tirp")

            def func():
                mod["moe_align_kernel"](
                    topk_ids_tvm,
                    sorted_ids_tvm,
                    expert_ids_tvm,
                    num_tokens_post_pad_tvm,
                    cumsum_buffer_tvm,
                )
                mod["count_and_sort_expert_tokens_kernel"](
                    topk_ids_tvm, sorted_ids_tvm, cumsum_buffer_tvm
                )

            ms = bench(func, warmup=10, repeat=30, proton_name="tir")
        return sorted_ids_tvm, expert_ids_tvm, num_tokens_post_pad_tvm

    def std():
        def func():
            sorted_ids_std, expert_ids_std, num_tokens_post_pad_std = moe_align_block_size(
                topk_ids.to("cuda"), BLOCK_SIZE, num_experts
            )
            return sorted_ids_std, expert_ids_std, num_tokens_post_pad_std

        ms = bench(func, warmup=10, repeat=30, proton_name="std")
        return func()

    with ProtonContext("moe_align"):
        sorted_ids_tvm, expert_ids_tvm, num_tokens_post_pad_tvm = tir()
        sorted_ids_std, expert_ids_std, num_tokens_post_pad_std = std()
    tvm.testing.assert_allclose(
        num_tokens_post_pad_tvm.numpy(), num_tokens_post_pad_std.cpu().numpy()
    )
    used_blocks = num_tokens_post_pad_std.item() // BLOCK_SIZE
    tvm.testing.assert_allclose(
        expert_ids_tvm.numpy()[:used_blocks], expert_ids_std.cpu().numpy()[:used_blocks]
    )

    # Select an expert to check
    expert_idx = expert_ids_std.max().item()

    # Get the first and last block id where expert_ids_cuda == expert_idx
    matching_indices = torch.where(expert_ids_std == expert_idx)[0]
    block_sorted_start = matching_indices[0].item() * BLOCK_SIZE
    block_sorted_end = min(
        (matching_indices[-1].item() + 1) * BLOCK_SIZE, num_tokens_post_pad_std.item()
    )
    selected_sorted_ids_std = sorted_ids_std[block_sorted_start:block_sorted_end].sort()[0]
    selected_sorted_ids_tvm = torch.from_numpy(sorted_ids_tvm.numpy())[
        block_sorted_start:block_sorted_end
    ].sort()[0]
    tvm.testing.assert_allclose(
        selected_sorted_ids_std.cpu().numpy(), selected_sorted_ids_tvm.cpu().numpy()
    )


if __name__ == "__main__":
    torch.random.manual_seed(42)
    for batch_size in [1, 16, 64, 128, 4096]:
        for pad_sorted_token_ids in [True, False]:
            test(batch_size, NUM_EXPERTS, TOPK, pad_sorted_token_ids)
