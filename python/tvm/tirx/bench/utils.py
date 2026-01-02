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
import torch
import numpy as np
from enum import Enum
from typing import List, Union
import triton.testing
import triton.profiler as proton
import tvm_ffi

import argparse
import subprocess
from tvm.contrib import nvcc
import tvm
import triton.runtime as runtime
from tvm.script import tir as T


def is_running_under_pytest():
    """Check if the code is being executed within a pytest session."""
    return "PYTEST_CURRENT_TEST" in os.environ


def setup():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-ptx", type=str, help="Dump PTX code to specified file")
    parser.add_argument("--dump-source", action="store_true", help="Dump source code")
    args = parser.parse_args()

    if args.dump_ptx:

        @tvm_ffi.register_global_func("tvm_callback_cuda_compile", override=True)
        def tvm_callback_cuda_compile(code, target):
            ptx = nvcc.compile_cuda(code, target_format="ptx")
            with open(args.dump_ptx, "w", encoding="utf-8") as f:
                f.write(ptx.decode())
            return ptx

    return args


def bench_fn(func, warmup, repeat, proton_name, flush_l2_size):
    # cache = runtime.driver.active.get_empty_cache_for_benchmark()
    for _ in range(warmup):
        # runtime.driver.active.clear_cache(cache)
        torch.empty(flush_l2_size, dtype=torch.int, device="cuda").zero_()
        func()
    if not is_running_under_pytest():
        proton.activate()
        with proton.scope(proton_name, metrics={}):
            for _ in range(repeat):
                # runtime.driver.active.clear_cache(cache)
                torch.empty(flush_l2_size, dtype=torch.int, device="cuda").zero_()
                func()
        proton.deactivate()
    else:
        for _ in range(repeat):
            func()


def bench(
    func, warmup=0, repeat=10, proton_name="kernel", debug=False, flush_l2_size=int(8e8 // 4)
):
    if not debug:
        bench_fn(
            func, warmup=warmup, repeat=repeat, proton_name=proton_name, flush_l2_size=flush_l2_size
        )
    else:
        func()
    return triton.testing.do_bench(func, warmup=warmup, rep=repeat)


class ProtonContext:
    """Context manager for Proton profiling sessions."""

    def __init__(self, name="kernel", hook="triton", debug=False):
        self.name = name
        self.hook = hook
        self.debug = debug

    def __enter__(self):
        if not is_running_under_pytest() and not self.debug:
            proton.start(self.name, hook=self.hook)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not is_running_under_pytest() and not self.debug:
            proton.finalize()

            subprocess.run(
                ["proton-viewer", "-m", "avg_time/ms", f"{self.name}.hatchet"], check=True
            )
            os.remove(f"{self.name}.hatchet")


# utils for tg4perfetto profiler, adapted from https://github.com/flashinfer-ai/flashinfer


class EventType(Enum):
    kBegin = 0
    kEnd = 1
    kInstant = 2
    kFinalize = 3


def decode_tag(tag, num_groups):
    block_group_tag = tag >> 12
    event_idx = (tag >> 2) & 0x3FF
    event_type = tag & 0x3
    return (
        block_group_tag // num_groups,
        block_group_tag % num_groups,
        event_idx,
        event_type,
    )


def export_to_perfetto_trace(
    profiler_buffer: np.ndarray,
    file_name: str,
    event_type_names: List[str],
) -> None:
    if is_running_under_pytest():
        return

    import torch

    # pip install git+https://github.com/ihavnoid/tg4perfetto.git
    from tg4perfetto import TraceGenerator

    profiler_buffer_host = torch.tensor(profiler_buffer)
    num_blocks, num_groups = profiler_buffer_host[:1].view(dtype=torch.int32)
    num_blocks = int(num_blocks)
    num_groups = int(num_groups)
    tgen = TraceGenerator(file_name)

    tid_map = {}
    track_map = {}
    finish_idx = set()
    for block_idx in range(num_blocks):
        pid = tgen.create_group(f"block_{block_idx}")
        for group_idx in range(num_groups):
            tid = pid.create_group(f"group_{group_idx}")
            tid_map[(block_idx, group_idx)] = tid

    for i in range(1, len(profiler_buffer_host)):
        if profiler_buffer_host[i] == 0:
            continue
        tag, timestamp = profiler_buffer_host[i : i + 1].view(dtype=torch.uint32)
        tag = int(tag)
        timestamp = int(timestamp)
        block_idx, group_idx, event_idx, event_type = decode_tag(tag, num_groups)

        if event_type == EventType.kFinalize.value:
            finish_idx.add((block_idx, group_idx))
            if len(finish_idx) == num_blocks * num_groups:
                break
        else:
            if (block_idx, group_idx) in finish_idx:
                continue

        event = event_type_names[event_idx]
        tid = tid_map[(block_idx, group_idx)]

        if (block_idx, group_idx, event_idx) in track_map:
            track = track_map[(block_idx, group_idx, event_idx)]
        else:
            track = tid.create_track()
            track_map[(block_idx, group_idx, event_idx)] = track

        if event_type == EventType.kBegin.value:
            track.open(timestamp, event)
        elif event_type == EventType.kEnd.value:
            track.close(timestamp)
        elif event_type == EventType.kInstant.value:
            track.instant(timestamp, event)

    tgen.flush()


class CudaProfiler:
    """A lightweight wrapper around T.timer_* CUDA intrinsics.

    Stores repeated arguments used by timer_init/start/end/finalize so users can
    call concise methods in kernels. Intended to mirror Pipeline/TileScheduler helpers.

    When ``profiler_enabled`` is False (or a false-y PrimExpr), calls to
    ``init/start/end/finalize`` become no-ops. This allows constructing a
    profiler unconditionally and eliminating external ``if PROFILER_ON:`` guards.
    """

    def __init__(
        self,
        profiler_buffer: T.Buffer,
        write_stride: int,
        num_groups: int,
        default_leader: Union[None, tvm.tir.PrimExpr, bool] = None,
        profiler_enabled: Union[bool, tvm.tir.PrimExpr] = True,
    ):
        self.buffer = profiler_buffer
        self.write_stride = write_stride
        self.num_groups = num_groups
        self.default_leader = default_leader
        # Accept either a Python bool or a PrimExpr; normalize simple bools to T.bool
        # so we can use it uniformly inside macros for conditional emission.
        if isinstance(profiler_enabled, (bool, np.bool_)):
            self.profiler_enabled = T.bool(bool(profiler_enabled))
        else:
            # Assume PrimExpr-like input; use as-is
            self.profiler_enabled = profiler_enabled  # type: ignore[assignment]

        self.profiler_tag = T.alloc_buffer(
            [1], "uint64", scope="local", align=8, name="profiler_tag"
        )
        self.profiler_write_offset = T.alloc_buffer(
            [1], "uint32", scope="local", align=8, name="profiler_write_offset"
        )

    def _leader(self, leader: Union[None, tvm.tir.PrimExpr, bool]):
        if leader is not None:
            if isinstance(leader, (bool, np.bool_)):
                return T.bool(bool(leader))
            return leader
        if self.default_leader is not None:
            return self.default_leader
        return T.bool(True)

    @T.macro
    def init(self, group_id: tvm.tir.PrimExpr):
        if self.profiler_enabled:
            T.timer_init_cuda(
                self.buffer.data,
                self.profiler_tag.data,
                self.profiler_write_offset.data,
                self.num_groups,
                group_id,
            )

    @T.macro
    def start(self, event_type: Enum, leader: Union[None, tvm.tir.PrimExpr, bool] = None):
        if self.profiler_enabled:
            T.timer_start_cuda(
                event_type,
                self.buffer.data,
                self.profiler_tag.data,
                self.profiler_write_offset.data,
                self.write_stride,
                self._leader(leader),
            )

    @T.macro
    def end(self, event_type: Enum, leader: Union[None, tvm.tir.PrimExpr, bool] = None):
        if self.profiler_enabled:
            T.timer_end_cuda(
                event_type,
                self.buffer.data,
                self.profiler_tag.data,
                self.profiler_write_offset.data,
                self.write_stride,
                self._leader(leader),
            )

    @T.macro
    def finalize(self, leader: Union[None, tvm.tir.PrimExpr, bool] = None):
        if self.profiler_enabled:
            T.timer_finalize_cuda(
                self.buffer.data,
                self.profiler_tag.data,
                self.profiler_write_offset.data,
                self.write_stride,
                self._leader(leader),
            )
