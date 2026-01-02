import numpy as np
import pytest
import torch

import tvm
import tvm.testing
from tvm.script import tir as T
from tvm.script import tirx as Tx
from tvm.tirx.bench.utils import ProtonContext, bench
from tvm.tirx.megakernel.moe_align import MOEAlignTile, CountAndSortExpertTokens
from tvm.tirx.megakernel.common import SmemManager, KernelConfig

from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size


NUM_EXPERTS = 128
TOPK = 8
BLOCK_SIZE = 32
HIDDEN_SIZE = 2048

test_configs = [
    {
        "num_tokens": b,
        "hidden_size": HIDDEN_SIZE,
        "num_experts": NUM_EXPERTS,
        "top_k": TOPK,
        "pad_sorted_token_ids": pad_sorted_token_ids,
    }
    for b in [2048, 512, 128, 1]
    for pad_sorted_token_ids in [True, False]
]


def get_moe_align_kernel(pad_sorted_token_ids):
    # fmt: off
    @T.prim_func(tirx=True)
    def moe_align_kernel(
        topk_ids_ptr: T.handle,
        sorted_token_ids_ptr: T.handle,
        expert_ids_ptr: T.handle,
        num_tokens_post_pad: T.Buffer((1,), "int32"),
        num_valid_tokens_ptr: T.handle,
        cumsum_buffer: T.Buffer((NUM_EXPERTS+1,), "int32"),
    ):
        num_tokens = T.int32()
        topk_ids = T.match_buffer(topk_ids_ptr, (num_tokens, TOPK), dtype="int64")
        max_num_tokens_padded = T.int32()
        sorted_token_ids = T.match_buffer(sorted_token_ids_ptr, (max_num_tokens_padded,), dtype="int32")
        expert_ids = T.match_buffer(expert_ids_ptr, (max_num_tokens_padded // BLOCK_SIZE,), dtype="int32")
        num_valid_tokens = T.match_buffer(num_valid_tokens_ptr, (max_num_tokens_padded // BLOCK_SIZE,), dtype="int32")
        moe_align_tile = T.meta_var(MOEAlignTile(NUM_EXPERTS, num_tokens * TOPK, BLOCK_SIZE, pad_sorted_token_ids=pad_sorted_token_ids))
        with T.kernel():
            bx = T.cta_id([1], parent="kernel")
            tid = T.thread_id([KernelConfig.NUM_THREADS], parent="cta")
            buf = T.alloc_buffer([KernelConfig.MAX_SMEM_SIZE], "uint8", scope="shared.dyn")
            smem_manager = T.meta_var(SmemManager(KernelConfig.MAX_SMEM_SIZE, 16384, buf.data))
            smem_manager.set_tile(moe_align_tile)
            moe_align_tile.init(smem_manager)
            smem_manager.init()
            topk_ids_flattened = topk_ids.view(-1)
            moe_align_tile.run(0, 0, 0, topk_ids_flattened, sorted_token_ids, expert_ids, num_tokens_post_pad, cumsum_buffer, num_valid_tokens)
    return moe_align_kernel
    # fmt: on


@T.prim_func(tirx=True)
def count_and_sort_expert_tokens_kernel(
    topk_ids_ptr: T.handle,
    sorted_token_ids_ptr: T.handle,
    cumsum_buffer: T.Buffer((NUM_EXPERTS + 1,), "int32"),
    data_ptr: T.handle,
    reordered_data_ptr: T.handle,
):
    num_tokens = T.int32()
    topk_ids = T.match_buffer(topk_ids_ptr, (num_tokens, TOPK), dtype="int64")
    max_num_tokens_padded = T.int32()
    sorted_token_ids = T.match_buffer(sorted_token_ids_ptr, (max_num_tokens_padded,), dtype="int32")
    data = T.match_buffer(data_ptr, (num_tokens, HIDDEN_SIZE), dtype="float16")
    reordered_data = T.match_buffer(reordered_data_ptr, (max_num_tokens_padded, HIDDEN_SIZE), dtype="float16")
    count_and_sort_tile = T.meta_var(CountAndSortExpertTokens(num_tokens * TOPK, HIDDEN_SIZE, TOPK))
    with T.kernel():
        bx = T.cta_id([KernelConfig.SM_NUMBER], parent="kernel")
        tid = T.thread_id([KernelConfig.NUM_THREADS], parent="cta")
        topk_ids_flattened = topk_ids.view(-1)
        buf = T.alloc_buffer([KernelConfig.MAX_SMEM_SIZE], "uint8", scope="shared.dyn")
        smem_manager = T.meta_var(SmemManager(KernelConfig.MAX_SMEM_SIZE, 16384, buf.data))
        smem_manager.set_tile(count_and_sort_tile)
        count_and_sort_tile.init(smem_manager)
        smem_manager.init()
        count_and_sort_tile.run(bx, 0, 0, topk_ids_flattened, sorted_token_ids, cumsum_buffer, data, reordered_data)

@pytest.mark.parametrize("task", test_configs)
def test_moe_align(task):
    num_tokens = task["num_tokens"]
    num_experts = task["num_experts"]
    topk = task["top_k"]
    hidden_size = task["hidden_size"]
    pad_sorted_token_ids = task["pad_sorted_token_ids"]
    topk_ids = torch.argsort(torch.rand(num_tokens, num_experts), dim=1)[:, :topk]
    max_num_tokens_padded = topk_ids.numel() + num_experts * (BLOCK_SIZE - 1)
    sorted_ids = torch.empty((max_num_tokens_padded,), dtype=torch.int32)
    if not pad_sorted_token_ids:
        sorted_ids.fill_(topk_ids.numel())
    max_num_m_blocks = max_num_tokens_padded // BLOCK_SIZE
    expert_ids = torch.zeros((max_num_m_blocks,), dtype=torch.int32)
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32)
    cumsum_buffer = torch.empty(num_experts + 1, dtype=torch.int32)
    num_valid_tokens = torch.empty((max_num_m_blocks,), dtype=torch.int32)
    dev = tvm.cuda()
    topk_ids_tvm = tvm.runtime.tensor(topk_ids.numpy(), device=dev)
    sorted_ids_tvm = tvm.runtime.tensor(sorted_ids.numpy(), device=dev)
    expert_ids_tvm = tvm.runtime.tensor(expert_ids.numpy(), device=dev)
    num_tokens_post_pad_tvm = tvm.runtime.tensor(num_tokens_post_pad.numpy(), device=dev)
    cumsum_buffer_tvm = tvm.runtime.tensor(cumsum_buffer.numpy(), device=dev)
    num_valid_tokens_tvm = tvm.runtime.tensor(num_valid_tokens.numpy(), device=dev)
    data = torch.randn(num_tokens, hidden_size, dtype=torch.float16)
    data_tvm = tvm.runtime.tensor(data.numpy(), device=dev)
    reordered_data_tvm = tvm.runtime.empty((max_num_tokens_padded, hidden_size), dtype="float16", device=dev)

    def tir():
        target = tvm.target.Target("cuda")
        with target:
            mod = tvm.IRModule(
                {
                    "moe_align_kernel": get_moe_align_kernel(pad_sorted_token_ids),
                    "count_and_sort_expert_tokens_kernel": count_and_sort_expert_tokens_kernel,
                }
            )
            mod = tvm.compile(mod, target=target, tir_pipeline="tirx")

            def func():
                mod["moe_align_kernel"](
                    topk_ids_tvm,
                    sorted_ids_tvm,
                    expert_ids_tvm,
                    num_tokens_post_pad_tvm,
                    num_valid_tokens_tvm,
                    cumsum_buffer_tvm,
                )
                mod["count_and_sort_expert_tokens_kernel"](
                    topk_ids_tvm, sorted_ids_tvm, cumsum_buffer_tvm, data_tvm, reordered_data_tvm
                )

            ms = bench(func, warmup=10, repeat=30, proton_name="tir")
        return sorted_ids_tvm, expert_ids_tvm, num_tokens_post_pad_tvm, num_valid_tokens_tvm

    def std():
        def func():
            sorted_ids_std, expert_ids_std, num_tokens_post_pad_std = moe_align_block_size(
                topk_ids.to("cuda"), BLOCK_SIZE, num_experts
            )
            valid_num_tokens = torch.empty(num_tokens_post_pad_std.item() // BLOCK_SIZE, dtype=torch.int32, device=sorted_ids_std.device)
            for i in range(num_tokens_post_pad_std.item() // BLOCK_SIZE):
                valid_num_tokens[i] = torch.sum((sorted_ids_std[i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE] >= 0) & (sorted_ids_std[i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE] < num_tokens * topk))
            return sorted_ids_std, expert_ids_std, num_tokens_post_pad_std, valid_num_tokens

        ms = bench(func, warmup=10, repeat=30, proton_name="std")
        return func()

    with ProtonContext("moe_align"):
        sorted_ids_tvm, expert_ids_tvm, num_tokens_post_pad_tvm, num_valid_tokens_tvm = tir()
        sorted_ids_std, expert_ids_std, num_tokens_post_pad_std, num_valid_tokens_std = std()
    tvm.testing.assert_allclose(
        num_tokens_post_pad_tvm.numpy(), num_tokens_post_pad_std.cpu().numpy()
    )
    used_blocks = num_tokens_post_pad_std.item() // BLOCK_SIZE
    tvm.testing.assert_allclose(
        expert_ids_tvm.numpy()[:used_blocks], expert_ids_std.cpu().numpy()[:used_blocks]
    )
    tvm.testing.assert_allclose(
        num_valid_tokens_tvm.numpy()[:used_blocks], num_valid_tokens_std.cpu().numpy()[:used_blocks]
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

    index = torch.from_numpy(sorted_ids_tvm.numpy())[:num_tokens_post_pad_std.item()].cpu()
    mask = (index >= 0) & (index < num_tokens * topk)
    zeros = torch.zeros(1, data.shape[1], dtype=data.dtype, device=data.device)
    padded_data = torch.cat([data, zeros], dim=0)
    safe_index = torch.where(mask, index // topk, num_tokens)
    reordered_data_std = padded_data[safe_index]
    tvm.testing.assert_allclose(reordered_data_tvm.numpy()[:num_tokens_post_pad_std.item(), :], reordered_data_std.cpu().numpy())


if __name__ == "__main__":
    torch.random.manual_seed(42)
    for task in test_configs:
        test_moe_align(task)
