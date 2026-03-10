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

import argparse
import os
import sys
import tempfile

import numpy as np

import tvm
import tvm.testing
from tvm.runtime import ShapeTuple
from tvm.runtime import disco as di

sys.path.insert(0, os.path.join(os.environ.get("TIRX_KERNELS_PATH", os.path.expanduser("~/tirx-kernels/kernels")), "moe"))
import ep_dispatch_combine  # noqa: E402


@tvm.testing.requires_cuda_compute_version(10, exact=False)
def test(
    dispatch_kernel,
    combine_kernel,
    num_tokens,
    total_num_experts,
    topk,
    hidden_dim,
    in_dtype,
    out_dtype,
    world_size,
    n_dp_groups,
    debug,
):
    import ml_dtypes
    import torch

    arg_dict = ep_dispatch_combine.prepare_data(
        num_tokens,
        total_num_experts,
        topk,
        hidden_dim,
        in_dtype,
        out_dtype,
        world_size,
        n_dp_groups,
    )

    def tir(arg_dict):
        devices = list(np.arange(world_size))
        sess = di.ProcessSession(num_workers=world_size)
        sess.init_ccl(tvm.get_global_func("runtime.disco.compiled_ccl")(), *devices)
        f_init_nvshmem_uid = tvm.get_global_func("runtime.disco.nvshmem.init_nvshmem_uid")
        uid = f_init_nvshmem_uid()
        init_dfunc = sess.get_global_func("runtime.disco.nvshmem.init_nvshmem")
        init_dfunc(uid, world_size, 0)
        sess.sync_worker_0()

        # make tvm data
        tvm_arg_dict = {}
        N_REPEAT = 100
        local_num_experts = total_num_experts // world_size
        send_tokens_all_tvm = tvm.runtime.from_dlpack(torch.to_dlpack(arg_dict["send_tokens"]))
        route_experts_all_tvm = tvm.runtime.from_dlpack(torch.to_dlpack(arg_dict["route_experts"]))
        route_weights_all_tvm = tvm.runtime.from_dlpack(torch.to_dlpack(arg_dict["route_weights"]))
        send_tokens_all_workers = sess.empty(
            (world_size, num_tokens, hidden_dim), in_dtype, worker0_only=True
        )
        route_experts_all_workers = sess.empty(
            (world_size, num_tokens, topk), "uint32", worker0_only=True
        )
        route_weights_all_workers = sess.empty(
            (world_size, num_tokens, topk), out_dtype, worker0_only=True
        )
        sess.copy_to_worker_0(send_tokens_all_tvm, send_tokens_all_workers)
        sess.copy_to_worker_0(route_experts_all_tvm, route_experts_all_workers)
        sess.copy_to_worker_0(route_weights_all_tvm, route_weights_all_workers)

        nvshmem_malloc_hook = sess.get_global_func("runtime.disco.nvshmem.empty")
        tvm_arg_dict["send_tokens"] = nvshmem_malloc_hook(
            ShapeTuple((num_tokens, hidden_dim)), in_dtype, None
        )
        tvm_arg_dict["recv_tokens"] = nvshmem_malloc_hook(
            ShapeTuple((local_num_experts, world_size, num_tokens, hidden_dim)), in_dtype, None
        )
        tvm_arg_dict["buf_recv"] = nvshmem_malloc_hook(
            ShapeTuple((total_num_experts, num_tokens, hidden_dim)), out_dtype, None
        )
        tvm_arg_dict["route_experts"] = sess.empty((num_tokens, topk), "uint32")
        tvm_arg_dict["route_weights"] = sess.empty((num_tokens, topk), out_dtype)
        tvm_arg_dict["num_recv_tokens"] = sess.empty((local_num_experts, world_size), "uint32")
        tvm_arg_dict["dst_token_indices"] = sess.empty((num_tokens, topk), "int32")
        tvm_arg_dict["output"] = sess.empty((num_tokens, hidden_dim), out_dtype)
        for i in range(N_REPEAT):
            tvm_arg_dict[f"target_wait_{i}"] = nvshmem_malloc_hook(
                ShapeTuple((local_num_experts, world_size)), "uint64", None
            )
            tvm_arg_dict[f"actual_wait_{i}"] = nvshmem_malloc_hook(
                ShapeTuple((local_num_experts, world_size)), "uint64", None
            )
            tvm_arg_dict[f"buf_wait_{i}"] = nvshmem_malloc_hook(
                ShapeTuple((total_num_experts,)), "uint64", None
            )
            tvm_arg_dict[f"dst_token_idx_{i}"] = sess.empty((total_num_experts,), "int32")

        sess.scatter_from_worker0(send_tokens_all_workers, tvm_arg_dict["send_tokens"])
        sess.scatter_from_worker0(route_experts_all_workers, tvm_arg_dict["route_experts"])
        sess.scatter_from_worker0(route_weights_all_workers, tvm_arg_dict["route_weights"])
        sess.sync_worker_0()

        # make result dict
        res_dict = {}
        DEV = tvm.cuda(0)
        res_dict["output_all_workers"] = sess.empty(
            (world_size, num_tokens, hidden_dim), out_dtype, worker0_only=True
        )
        res_dict["output_all_tvm"] = tvm.runtime.empty(
            (world_size, num_tokens, hidden_dim), out_dtype, device=DEV
        )
        if debug:
            res_dict["recv_tokens_all_workers"] = sess.empty(
                (world_size, local_num_experts, world_size, num_tokens, hidden_dim),
                in_dtype,
                worker0_only=True,
            )
            res_dict["recv_tokens_all_tvm"] = tvm.runtime.empty(
                (world_size, local_num_experts, world_size, num_tokens, hidden_dim),
                in_dtype,
                device=DEV,
            )
            res_dict["num_recv_tokens_all_workers"] = sess.empty(
                (world_size, local_num_experts, world_size), "uint32", worker0_only=True
            )
            res_dict["num_recv_tokens_all_tvm"] = tvm.runtime.empty(
                (world_size, local_num_experts, world_size), "uint32", device=DEV
            )
            res_dict["dst_token_indices_all_workers"] = sess.empty(
                (world_size, num_tokens, topk), "int32", worker0_only=True
            )
            res_dict["dst_token_indices_all_tvm"] = tvm.runtime.empty(
                (world_size, num_tokens, topk), "int32", device=DEV
            )
            res_dict["buf_recv_all_workers"] = sess.empty(
                (world_size, total_num_experts, num_tokens, hidden_dim),
                out_dtype,
                worker0_only=True,
            )
            res_dict["buf_recv_all_tvm"] = tvm.runtime.empty(
                (world_size, total_num_experts, num_tokens, hidden_dim), out_dtype, device=DEV
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            target = tvm.target.Target("cuda")
            dispatch_path = tmpdir + "/dispatch.so"
            dispatch_mod = tvm.compile(dispatch_kernel, target=target, tir_pipeline="tirx")
            dispatch_mod.export_library(dispatch_path)
            print(dispatch_mod.mod.imports[0].inspect_source())

            combine_path = tmpdir + "/combine.so"
            combine_mod = tvm.compile(combine_kernel, target=target, tir_pipeline="tirx")
            combine_mod.export_library(combine_path)
            print(combine_mod.mod.imports[0].inspect_source())

            dispatch_rt_mod = sess.load_vm_module(dispatch_path)
            combine_rt_mod = sess.load_vm_module(combine_path)
            barrier_dfunc = sess.get_global_func(
                "runtime.disco.nvshmem.barrier_all_on_current_stream"
            )
            sess._sync_all()

            for itr in range(N_REPEAT):
                barrier_dfunc()
                if debug:
                    print(f"Running dispatch kernel {itr}...")
                dispatch_rt_mod["main_dispatch"](
                    tvm_arg_dict["send_tokens"],
                    tvm_arg_dict["route_experts"],
                    tvm_arg_dict["recv_tokens"],
                    tvm_arg_dict["num_recv_tokens"],
                    tvm_arg_dict[f"target_wait_{itr}"],
                    tvm_arg_dict[f"actual_wait_{itr}"],
                    tvm_arg_dict["dst_token_indices"],
                    tvm_arg_dict[f"dst_token_idx_{itr}"],
                )
                if debug:
                    print(f"Dispatch kernel {itr} finished.")
                if debug:
                    print(f"Running combine kernel {itr}...")
                barrier_dfunc()
                combine_rt_mod["main_combine"](
                    tvm_arg_dict["recv_tokens"],
                    tvm_arg_dict["route_experts"],
                    tvm_arg_dict["route_weights"],
                    tvm_arg_dict["num_recv_tokens"],
                    tvm_arg_dict["dst_token_indices"],
                    tvm_arg_dict["output"],
                    tvm_arg_dict[f"buf_wait_{itr}"],
                    tvm_arg_dict["buf_recv"],
                )
                if debug:
                    print(f"Combine kernel {itr} finished.")

            sess._sync_all()

            print("Gathering output to worker 0...")
            sess.gather_to_worker0(tvm_arg_dict["output"], res_dict["output_all_workers"])
            sess.copy_from_worker_0(res_dict["output_all_tvm"], res_dict["output_all_workers"])
            if debug:
                sess.gather_to_worker0(
                    tvm_arg_dict["recv_tokens"], res_dict["recv_tokens_all_workers"]
                )
                sess.copy_from_worker_0(
                    res_dict["recv_tokens_all_tvm"], res_dict["recv_tokens_all_workers"]
                )
                sess.gather_to_worker0(
                    tvm_arg_dict["num_recv_tokens"], res_dict["num_recv_tokens_all_workers"]
                )
                sess.copy_from_worker_0(
                    res_dict["num_recv_tokens_all_tvm"], res_dict["num_recv_tokens_all_workers"]
                )
                sess.gather_to_worker0(
                    tvm_arg_dict["dst_token_indices"], res_dict["dst_token_indices_all_workers"]
                )
                sess.copy_from_worker_0(
                    res_dict["dst_token_indices_all_tvm"], res_dict["dst_token_indices_all_workers"]
                )
                sess.gather_to_worker0(tvm_arg_dict["buf_recv"], res_dict["buf_recv_all_workers"])
                sess.copy_from_worker_0(
                    res_dict["buf_recv_all_tvm"], res_dict["buf_recv_all_workers"]
                )
            sess._sync_all()
            print("Kernel execution all finished.")

        finalize_dfunc = sess.get_global_func("runtime.disco.nvshmem.finalize_nvshmem")
        finalize_dfunc()
        sess.sync_worker_0()

        return res_dict

    res_dict = tir(arg_dict)

    out = res_dict["output_all_tvm"].numpy()  # [WORLD_SIZE, num_tokens, hidden_dim], out_dtype

    ml_dtype_dict = {
        "bfloat16": ml_dtypes.bfloat16,
        "float16": np.float16,
        "float8_e4m3fn": ml_dtypes.float8_e4m3fn,
    }
    import torch

    ref = torch.zeros_like(arg_dict["send_tokens"]).to(torch.device("cuda"))
    for k in range(topk):
        ref += arg_dict["send_tokens"] * arg_dict["route_weights"][:, :, k][..., None]
    ref = ref.view(torch.uint8).cpu().numpy().view(ml_dtype_dict[out_dtype])

    np.testing.assert_allclose(out, ref, atol=1e-1, rtol=1e-2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--total-num-experts", type=int, default=128)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--n-dp-groups", type=int, default=8)
    parser.add_argument(
        "--in-dtype",
        choices=["bfloat16", "float16", "float8_e4m3fn"],
        default="bfloat16",
    )
    parser.add_argument(
        "--out-dtype",
        choices=["bfloat16", "float16"],
        default="bfloat16",
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    num_tokens = int(args.batch_size)
    total_num_experts = int(args.total_num_experts)
    topk = int(args.topk)
    hidden_dim = int(args.hidden_dim)
    in_dtype = str(args.in_dtype)
    out_dtype = str(args.out_dtype)
    world_size = int(args.world_size)
    n_dp_groups = int(args.n_dp_groups)
    debug = args.debug

    dispatch_kernel_wrapper = ep_dispatch_combine.EPDispatchKernel()
    dispatch_static_module = dispatch_kernel_wrapper.get_module_static(
        num_tokens,
        total_num_experts,
        topk,
        hidden_dim,
        in_dtype,
        out_dtype,
        world_size,
        n_dp_groups,
    )
    combine_kernel_wrapper = ep_dispatch_combine.EPCombineKernel()
    combine_static_module = combine_kernel_wrapper.get_module_static(
        num_tokens,
        total_num_experts,
        topk,
        hidden_dim,
        in_dtype,
        out_dtype,
        world_size,
        n_dp_groups,
    )

    print(
        f"running: EP={world_size}, DP={world_size}, hidden_dim={hidden_dim}, total_num_experts={total_num_experts}, topk={topk}, num_tokens={num_tokens}, in_dtype={in_dtype}, out_dtype={out_dtype}"  # noqa: E501
    )
    test(
        dispatch_static_module,
        combine_static_module,
        num_tokens,
        total_num_experts,
        topk,
        hidden_dim,
        in_dtype,
        out_dtype,
        world_size,
        n_dp_groups,
        debug,
    )
