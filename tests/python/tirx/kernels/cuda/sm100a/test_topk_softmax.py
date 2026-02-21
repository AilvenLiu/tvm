import math

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.script import ir as I
from tvm.script import tirx as Tx
from tvm.tirx.bench.utils import ProtonContext, bench
from tvm.tirx.megakernel.utils.config import ProfileEventType, KernelConfig
from tvm.tirx.megakernel.utils.utils import get_source
from tvm.tirx.megakernel.kernels.topk_softmax import TopkSoftmaxTile


class TopkSoftmaxKernel:
    def __init__(self):
        self.tile_attr = {}
        self.class_list = set()

    def set_tiles(self, num_experts, num_tokens, topk, dtype="float32"):
        self.topk_softmax_tile = self._add_tile(
            TopkSoftmaxTile(num_experts, num_tokens, topk, dtype),
            ProfileEventType.TOPK_SOFTMAX,
        )

    @Tx.inline
    def run_tile(self, tile, *args):
        with Tx.cta():
            tile.run(*args)

    def _add_tile(self, tile, profiler_event_type):
        self.tile_attr[tile] = profiler_event_type
        self.class_list.add(tile.__class__)
        return tile

    def device_init_all(self):
        for tile in self.tile_attr.keys():
            tile.init()

    def get_func_static(self, num_experts, dtype="float32"):
        # fmt: off
        @Tx.prim_func(tirx=True)
        def main(gating_output_ptr: Tx.handle, topk_weights_ptr: Tx.handle, topk_indices_ptr: Tx.handle):
            Tx.func_attr({"global_symbol": "main", "target": Tx.target("cuda")})

            num_tokens = Tx.int32()
            topk = Tx.int32()

            gating_output_global = Tx.match_buffer(gating_output_ptr, [num_tokens, num_experts], dtype, scope="global")
            topk_weights_global = Tx.match_buffer(topk_weights_ptr, [num_tokens, topk], "float32", scope="global")
            topk_indices_global = Tx.match_buffer(topk_indices_ptr, [num_tokens, topk], "int32", scope="global")

            self.set_tiles(num_experts, num_tokens, topk, dtype)

            with Tx.kernel():
                bx = Tx.cta_id([self.topk_softmax_tile.PERSISTENT_SM_NUMBER], parent="kernel")
                warp_id_in_cta = Tx.warp_id([KernelConfig.WG_NUMBER * KernelConfig.WARP_NUMBER], parent="cta")
                lane_id = Tx.thread_id([self.topk_softmax_tile.bdx], parent="warp")
                with Tx.cta():
                    # no need of smem_manager
                    self.device_init_all()

                    # TODO: initialize event tensors & tile scheduler
                    self.run_tile(self.topk_softmax_tile, bx, 0, 0, gating_output_global, topk_weights_global, topk_indices_global)
        # fmt: on
        return main

    def get_module_static(self, num_experts, dtype="float32"):
        @I.ir_module(tirx=True)
        class StaticModule:
            @Tx.prim_func(tirx=True)
            def main():
                pass

        module: tvm.IRModule = StaticModule
        module.update_func(module.get_global_var("main"), self.get_func_static(num_experts, dtype))
        return module


arg_dict = {}


def prepare_data(num_tokens, num_experts, topk, dtype):
    global arg_dict
    import torch

    torch.manual_seed(42)

    dtype_dict = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    arg_dict["gating_output"] = torch.randn((num_tokens, num_experts), dtype=dtype_dict.get(dtype))
    arg_dict["topk_weights"] = torch.empty((num_tokens, topk), dtype=torch.float32)
    arg_dict["topk_indices"] = torch.empty((num_tokens, topk), dtype=torch.int32)

    return arg_dict


@tvm.testing.requires_cuda_compute_version(10, exact=False)
def test(kernel, num_tokens, num_experts, topk, dtype):
    arg_dict = prepare_data(num_tokens, num_experts, topk, dtype)

    def tir(arg_dict):
        DEV = tvm.cuda(0)
        tvm_arg_dict = {}
        target = tvm.target.Target("cuda")

        for key, value in arg_dict.items():
            tvm_arg_dict[key] = tvm.runtime.tensor(value, device=DEV)

        with target:

            def func():
                kernel(
                    tvm_arg_dict["gating_output"],
                    tvm_arg_dict["topk_weights"],
                    tvm_arg_dict["topk_indices"],
                )

            ms = bench(func, warmup=10, repeat=30, proton_name="tir")
            print(f"TIR time: {ms:.3f} ms")

            return tvm_arg_dict["topk_weights"].numpy(), tvm_arg_dict["topk_indices"].numpy()

    def sglang(arg_dict):
        from sgl_kernel import topk_softmax
        import torch

        torch_dev = torch.device("cuda")
        std_arg_dict = {}

        for key, value in arg_dict.items():
            std_arg_dict[key] = value.to(torch_dev)

        def func():
            topk_softmax(
                topk_weights=std_arg_dict["topk_weights"],
                topk_ids=std_arg_dict["topk_indices"],
                gating_output=std_arg_dict["gating_output"],
            )

        ms = bench(func, warmup=10, repeat=30, proton_name="sglang")
        print(f"sglang time: {ms:.3f} ms")

        return std_arg_dict["topk_weights"].cpu().numpy(), std_arg_dict[
            "topk_indices"
        ].cpu().numpy()

    with ProtonContext("blackwell_benchmark"):
        weights_tir, indices_tir = tir(arg_dict)
        weights_sglang, indices_sglang = sglang(arg_dict)

    np.testing.assert_allclose(weights_tir, weights_sglang, rtol=1e-3, atol=1e-3)
    np.testing.assert_equal(indices_tir, indices_sglang)


if __name__ == "__main__":
    itr = 0
    for dtype in ["float16", "float32"]:
        for num_experts in [32, 64, 128, 256]:
            if num_experts == 256 and dtype == "float32":
                continue
            mega_kernel_wrapper_static = TopkSoftmaxKernel()
            mega_static_module = mega_kernel_wrapper_static.get_module_static(num_experts, dtype)
            src, lib_static = get_source(mega_static_module)
            # print(src)

            for num_tokens in [128, 512, 1024, 2048, 4096, 8192, 16384, 32768]:
                for topk in [1, 2, 4, 8]:
                    n_bytes_dict = {"float16": 2, "bfloat16": 2, "float32": 4}
                    VPT = 16 // n_bytes_dict.get(dtype)
                    ROWS_PER_WARP = (VPT * 32) // num_experts
                    num_blocks = ((num_tokens + ROWS_PER_WARP - 1) // ROWS_PER_WARP + 8 - 1) // 8
                    print(
                        f"experiment {itr}: if nonpersistent, would need <<<{num_blocks}, 256>>>, num_tokens {num_tokens}, num_experts {num_experts}, topk {topk}, dtype {dtype}"
                    )
                    test(lib_static["main"], num_tokens, num_experts, topk, dtype)
                    itr += 1
