import argparse
import operator
import numpy as np
import pytest
from typing import Type, Literal

import tvm
import tvm.testing
from tvm.script import ir as I
from tvm.script import tirx as Tx
from tvm.tirx.bench.utils import ProtonContext, bench, export_to_perfetto_trace

from tvm.tirx.megakernel.utils.base import MegaKernelWrapper, SemaphoreBase
from tvm.tirx.megakernel.utils.utils import ceildiv, get_source, f_init_const, pack_into_32bit
from tvm.tirx.megakernel.utils.config import KernelConfig, JobType, ProfileEventType, event_type_names
from tvm.tirx.megakernel.utils import static_scheduler
from tvm.tirx.megakernel.kernels import GemmTile, SplitKReduceTile


class GemmConfigSearcher(MegaKernelWrapper):

    def __init__(self, batch_size, n, k, blk_n, split_k, use_tma_reduce, profiler_on):
        super().__init__({}, 1, profiler_on)
        self.batch_size = batch_size
        if batch_size <= 32:
            self.m = 32
        elif batch_size <= 64:
            self.m = 64
        else:
            self.m = 128
        self.n = n
        self.k = k
        self.blk_n = blk_n
        self.split_k = split_k
        self.use_tma_reduce = use_tma_reduce
        assert not use_tma_reduce or split_k > 1

    def _set_tiles(self):
        self.gemm_tile = self._add_tile(
            GemmTile(
                self.n, self.k, "float16", "float16", self.split_k, self.m, self.m,
                "float32" if self.split_k > 1 or self.use_tma_reduce else "float16",
                use_tma_reduce=self.use_tma_reduce,
                low_batch=False, prefetch_on=False, profiler_on=self.profiler_on
            ),
            ProfileEventType.GEMM_O_PROJ,
            predicate=True,
        )
        self.reduce_tile = self._add_tile(
            SplitKReduceTile(self.batch_size, self.n, "float16", self.split_k),
            ProfileEventType.GEMM_O_REDUCE,
            predicate=self.split_k > 1 and not self.use_tma_reduce,
        )

    def set_tiles(self):
        self.reset()
        self._set_tiles()
        
    def _set_events(self, Semaphore: Type[SemaphoreBase], etensor_workspace_global):
        self.evt = self.add_etensor(
            Semaphore,
            etensor_workspace_global,
            shape=[self.n // self.reduce_tile.N_TILE],
            f_init=f_init_const(self.split_k * (self.reduce_tile.N_TILE // self.gemm_tile.BLK_N))
        )
        
    def set_events(self, Semaphore: Type[static_scheduler.Semaphore], etensor_workspace_global):
        self._set_events(Semaphore, etensor_workspace_global)
        self.set_events_complete(False, Semaphore, etensor_workspace_global)

    @Tx.macro
    def task_impl_gemm(self, A, B, partial, output, output32):
        with Tx.cta():
            if self.use_tma_reduce:
                self.run_tile(
                    self.gemm_tile, self.tile_scheduler.m_idx, self.tile_scheduler.n_idx, self.tile_scheduler.k_idx,
                    A, B, output32, self.profiler,
                )
            elif self.split_k == 1:
                self.run_tile(
                    self.gemm_tile, self.tile_scheduler.m_idx, self.tile_scheduler.n_idx, self.tile_scheduler.k_idx,
                    A, B, output, self.profiler,
                )
            else:
                self.run_tile(
                    self.gemm_tile, self.tile_scheduler.m_idx, self.tile_scheduler.n_idx, self.tile_scheduler.k_idx,
                    A, B, partial, self.profiler,
                )            
            if self.split_k > 1 and not self.use_tma_reduce:
                self.tile_scheduler.notify(
                    self.evt,
                    lambda notify_idx: (1, -1, self.tile_scheduler.n_idx * self.gemm_tile.BLK_N // self.reduce_tile.N_TILE),
                    scope="warpgroup", scope_id=0,
                )

    @Tx.macro
    def task_reduce(self, partial_global, output_global):
        with Tx.cta():
            self.tile_scheduler.wait(self.evt, self.tile_scheduler.n_idx, wait_level="warp")
            self.run_tile(
                self.reduce_tile, self.tile_scheduler.m_idx, self.tile_scheduler.n_idx, self.tile_scheduler.k_idx,
                partial_global, output_global,
            )

    # fmt: off
    @Tx.macro
    def fused_body(
        self,
        A_global,
        B_global,
        partial_global,
        output_global,
        output32_global,
        etensor_workspace,
        profiler_buffer,
        exec_queue,
        Semaphore: Type[static_scheduler.Semaphore],
        Scheduler: Type[static_scheduler.StaticTileScheduler],
    ):
        # initialize tile
        self.set_tiles()
        self.host_init_all()

        with Tx.kernel():
            bx = Tx.cta_id([KernelConfig.SM_NUMBER], parent="kernel")
            warp_id = Tx.warp_id([KernelConfig.WARP_NUMBER * KernelConfig.WG_NUMBER], parent="cta")
            wg_id = Tx.warpgroup_id([KernelConfig.WG_NUMBER], parent="cta")
            tid = Tx.thread_id([KernelConfig.NUM_THREADS], parent="cta")
            tid_in_wg = Tx.thread_id([KernelConfig.NUM_THREADS // KernelConfig.WG_NUMBER], parent="warpgroup")
            lane_id = Tx.thread_id([32], parent="warp")
            self.init_profiler(profiler_buffer)
            with Tx.cta():
                buf = Tx.alloc_buffer([KernelConfig.MAX_SMEM_SIZE], "uint8", scope="shared.dyn")
                
                self.set_smem_manager(KernelConfig.MAX_SMEM_SIZE, 16384, buf.data)
                self.device_init_all(self.smem_manager)
                self.class_init_all(self.smem_manager)

                # initialize event tensors
                self.set_events(Semaphore, etensor_workspace)

                # initialize tile scheduler and smem_manager
                self.init_tile_scheduler(False, Scheduler, "gemm", exec_queue, self.smem_manager)
                self.smem_manager.init()
                
                while self.tile_scheduler.valid():
                    
                    if self.tile_scheduler.task_type == 0:
                        self.task_impl_gemm(A_global, B_global, partial_global, output_global, output32_global)
                    elif self.tile_scheduler.task_type == 1:
                        self.task_reduce(partial_global, output_global)
                    elif self.tile_scheduler.task_type == JobType.INIT_ETENSOR.value:
                        self.task_impl_init_etensor(False)
                    elif self.tile_scheduler.task_type == JobType.WAIT_ETENSOR_INITx.value:
                        self.task_impl_wait_etensor_init_complete(False)
                    else:
                        Tx.cuda.trap_when_assert_failed(False)
                    self.smem_manager.exit_tile_runtime()
                    self.tile_scheduler.next_tile()
                if self.profiler_on:
                    self.profiler.finalize(lane_id == 0)
                self.class_finalize_all()

    def get_func_static(self, unfused=False):
        # fmt: off
        @Tx.prim_func(tirx=True)
        def main(
            # input and output
            A_ptr: Tx.handle,
            B_ptr: Tx.handle,
            partial_ptr: Tx.handle,
            output_ptr: Tx.handle,
            output32_ptr: Tx.handle,
            etensor_workspace_ptr: Tx.handle,
            exec_queue_ptr: Tx.handle,
            profiler_buffer: Tx.Buffer((self.PROFILER_BUFFER_SIZE,), "uint64")
        ):
            Tx.func_attr(
                {"global_symbol": "main", "target": Tx.target("cuda")}
            )

            # match buffer
            A_global = Tx.match_buffer(A_ptr, [self.batch_size, self.k], "float16", scope="global")
            B_global = Tx.match_buffer(B_ptr, [self.n, self.k], "float16", scope="global")
            partial_global = Tx.match_buffer(partial_ptr, [self.split_k, self.batch_size, self.n], "float32", scope="global")
            output_global = Tx.match_buffer(output_ptr, [self.batch_size, self.n], "float16", scope="global")
            output32_global = Tx.match_buffer(output32_ptr, [self.batch_size, self.n], "float32", scope="global")
            
            etensor_workspace_size = Tx.int32()
            etensor_workspace_global = Tx.match_buffer(etensor_workspace_ptr, [etensor_workspace_size], "int32", scope="global", offset_factor=1)
            
            # exec queue
            exec_queue = Tx.match_buffer(exec_queue_ptr, [KernelConfig.SM_NUMBER, static_scheduler.StaticTileScheduler.MAX_TASKS], "int32", scope="global")
            
            # main
            self.fused_body(
                A_global, B_global, partial_global, output_global, output32_global, etensor_workspace_global, profiler_buffer, exec_queue,
                static_scheduler.Semaphore, static_scheduler.StaticTileScheduler
            )
   
        return main


arg_dict = {}


def prepare_data(mk: GemmConfigSearcher, repeat=100):
    global arg_dict
    import torch

    torch.manual_seed(42)

    arg_dict["A"] = torch.randn((mk.batch_size, mk.k), dtype=torch.float16)
    arg_dict["B"] = torch.randn((mk.n, mk.k), dtype=torch.float16)
    arg_dict["partial"] = torch.zeros((mk.split_k, mk.batch_size, mk.n), dtype=torch.float32)
    arg_dict["output"] = torch.zeros((mk.batch_size, mk.n), dtype=torch.float16)
    arg_dict[f"etensor_workspace"] = torch.zeros([mk.ETENSOR_WORKSPACE_SIZE], dtype=torch.int32)
    for i in range(repeat):
        arg_dict[f"output32_{i}"] = torch.zeros((mk.batch_size, mk.n), dtype=torch.float32)

    return arg_dict


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
            central_queue.append((i, 0, 0, JobType.WAIT_ETENSOR_INITx.value))
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
        tvm_arg_dict[f"profiler_buffer"] = tvm.runtime.tensor(
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
                    tvm_arg_dict[f"etensor_workspace"],
                    tvm_arg_dict["exec_queue"],
                    tvm_arg_dict[f"profiler_buffer"],
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
                    tvm_arg_dict[f"profiler_buffer"].numpy(),
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
        ms = bench(func, warmup=10, repeat=30, proton_name=f"std")
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
