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
import operator

import numpy as np
import pytest
from tirx_kernels.megakernel.gemm_search_config import (
    GemmConfigSearcher,
    prepare_data,
)

import tvm
import tvm.testing
from tvm.tirx.bench.utils import ProtonContext, bench, export_to_perfetto_trace
from tvm.tirx.megakernel.kernels import SplitKReduceTile
from tvm.tirx.megakernel.utils import static_scheduler
from tvm.tirx.megakernel.utils.config import (
    JobType,
    KernelConfig,
    event_type_names,
)
from tvm.tirx.megakernel.utils.utils import ceildiv, get_source, pack_into_32bit


@tvm.testing.requires_cuda_compute_version(10, exact=True)
@pytest.mark.skip
def test(batch_size, mega_kernel_static, mega_kernel_wrapper):
    REPEAT = 100
    arg_dict = prepare_data(mega_kernel_wrapper, REPEAT)

    def tir(arg_dict, mk: GemmConfigSearcher):
        DEV = tvm.cuda(0)
        tvm_arg_dict = {}
        for key, value in arg_dict.items():
            tvm_arg_dict[key] = tvm.runtime.tensor(value.numpy(), device=DEV)
        target = tvm.target.Target("cuda")
        exec_queue = np.zeros(
            (KernelConfig.SM_NUMBER, static_scheduler.StaticTileScheduler.MAX_TASKS), dtype=np.int32
        )
        central_queue = []
        for i in range(2):
            central_queue.append((i, 0, 0, JobType.INIT_ETENSOR.value))
        for n_idx in range(mk.n // mk.blk_n):
            for k_idx in range(mk.split_k):
                central_queue.append((0, n_idx, k_idx, 0))
        for i in range(KernelConfig.SM_NUMBER):
            central_queue.append((i, 0, 0, JobType.WAIT_ETENSOR_INIT.value))
        if mk.split_k > 1 and not mk.use_tma_reduce:
            m_split = min(batch_size, KernelConfig.SM_NUMBER // (mk.n // SplitKReduceTile.N_UNIT))
            m_tile = ceildiv(batch_size, m_split)
            m_split = ceildiv(batch_size, m_tile)
            for m_idx in range(m_split):
                for n_idx in range(mk.n // SplitKReduceTile.N_UNIT):
                    central_queue.append((m_idx, n_idx, 0, 1))
        tile_idx = 0
        while len(central_queue) > 0:
            for bx in range(KernelConfig.SM_NUMBER):
                if len(central_queue) > 0:
                    exec_queue[bx, tile_idx] = pack_into_32bit(*central_queue.pop(0), debug=True)
                else:
                    exec_queue[bx, tile_idx] = pack_into_32bit(
                        -1, -1, -1, JobType.END.value, debug=True
                    )
            tile_idx += 1
        for bx in range(KernelConfig.SM_NUMBER):
            exec_queue[bx, tile_idx] = pack_into_32bit(-1, -1, -1, JobType.END.value, debug=True)
        tvm_arg_dict["exec_queue"] = tvm.runtime.tensor(exec_queue, device=DEV)
        tvm_arg_dict["profiler_buffer"] = tvm.runtime.tensor(
            np.zeros([mk.PROFILER_BUFFER_SIZE], dtype=np.uint64), device=DEV
        )

        # run
        with target:
            iter = 0
            kernel = mega_kernel_static["main"]

            def func():
                nonlocal iter
                kernel(
                    tvm_arg_dict["A"],
                    tvm_arg_dict["B"],
                    tvm_arg_dict["partial"],
                    tvm_arg_dict["output"],
                    tvm_arg_dict[f"output32_{iter}"],
                    tvm_arg_dict["etensor_workspace"],
                    tvm_arg_dict["exec_queue"],
                    tvm_arg_dict["profiler_buffer"],
                )
                iter += 1

            ms = bench(
                func,
                warmup=1,
                repeat=3,
                proton_name=f"tir-blkn{mk.blk_n}-splitk{mk.split_k}{'-tmareduce' if mk.use_tma_reduce else ''}",
            )
            print(f"TIR time: {ms:.3f} ms")
            if mk.profiler_on:
                export_to_perfetto_trace(
                    tvm_arg_dict["profiler_buffer"].numpy(),
                    f"blkn{mk.blk_n}-splitk{mk.split_k}.perfetto-trace",
                    event_type_names,
                )
            if mk.use_tma_reduce:
                return tvm_arg_dict["output32_0"].numpy().astype(np.float16), ms
            else:
                return tvm_arg_dict["output"].numpy(), ms

    def std(arg_dict):
        import torch

        torch_dev = torch.device("cuda")
        std_arg_dict = {}

        def func():
            for key, value in arg_dict.items():
                std_arg_dict[key] = value.clone().to(torch_dev)
            output = torch.matmul(std_arg_dict["A"], std_arg_dict["B"].T)
            return output.cpu().numpy()

        output = func()
        ms = bench(func, warmup=10, repeat=30, proton_name="std")
        print(f"std time: {ms:.3f} ms")
        return output

    def run():
        if mega_kernel_static["main"] is not None:
            output_tir_static, ms_tir = tir(arg_dict, mega_kernel_wrapper)
            print("static tir finish", flush=True)
        output_std = std(arg_dict)

        if mega_kernel_static["main"] is not None:
            np.testing.assert_allclose(output_tir_static, output_std, rtol=1e-3, atol=1e-2)
            print("static pass", flush=True)
        return ms_tir

    with ProtonContext("blackwell_layer"):
        ms_tir = run()
    return ms_tir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MegaKernel testing script.")
    parser.add_argument(
        "--batch-size", type=int, nargs="+", default=[1], help="A list of batch sizes to test."
    )
    parser.add_argument("--N", type=int)
    parser.add_argument("--K", type=int)
    parser.add_argument(
        "--split-k",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        help="The split k factor.",
    )
    parser.add_argument(
        "--blk-n", type=int, nargs="+", default=[16, 32, 64, 128], help="The block n size."
    )
    parser.add_argument("--profiler-on", action="store_true", help="Enable the profiler.")
    args = parser.parse_args()

    for batch_size in args.batch_size:
        print(f"batch_size: {batch_size}", flush=True)
        wrappers = {}
        for split_k in args.split_k:
            for blk_n in args.blk_n:
                for use_tma_reduce in [True, False] if split_k > 1 else [False]:
                    mega_kernel_wrapper = GemmConfigSearcher(
                        batch_size,
                        args.N,
                        args.K,
                        blk_n,
                        split_k,
                        use_tma_reduce,
                        profiler_on=args.profiler_on,
                    )
                    mega_static_module = mega_kernel_wrapper.get_module("static")
                    src, lib_static = get_source(mega_static_module)
                    wrappers[(split_k, blk_n, use_tma_reduce)] = (mega_kernel_wrapper, lib_static)

        times = {}
        for (split_k, blk_n, use_tma_reduce), (mega_kernel_wrapper, lib_static) in wrappers.items():
            print(
                f"split_k: {split_k}, blk_n: {blk_n}, use_tma_reduce: {use_tma_reduce}", flush=True
            )
            ms = test(batch_size, lib_static, mega_kernel_wrapper)
            times[(split_k, blk_n, use_tma_reduce)] = ms
        sorted_items_asc = sorted(times.items(), key=operator.itemgetter(1))
        print("Top 10 configs:")
        for (split_k, blk_n, use_tma_reduce), ms in sorted_items_asc[:10]:
            print(
                f"split_k: {split_k}, blk_n: {blk_n}, use_tma_reduce: {use_tma_reduce}, time: {ms:.3f} ms"
            )
