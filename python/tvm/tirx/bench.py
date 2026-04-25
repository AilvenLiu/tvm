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
import re
import subprocess
import sys
import time
from enum import Enum

import numpy as np
import torch
import triton.profiler as proton
import tvm_ffi

import tvm
from tvm.contrib import nvcc
from tvm.script import tirx as Tx


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


def bench_fn(func, warmup, repeat, proton_name, flush_l2_size, nsight=False):
    for _ in range(warmup):
        torch.empty(flush_l2_size, dtype=torch.int, device="cuda").zero_()
        func()
    if not is_running_under_pytest() and not nsight:
        proton.activate()
        with proton.scope(proton_name, metrics={}):
            for _ in range(repeat):
                torch.empty(flush_l2_size, dtype=torch.int, device="cuda").zero_()
                func()
        proton.deactivate()
    else:
        for _ in range(repeat):
            torch.empty(flush_l2_size, dtype=torch.int, device="cuda").zero_()
            func()


def bench(
    func,
    warmup=0,
    repeat=10,
    proton_name="kernel",
    debug=False,
    nsight=False,
    flush_l2_size=int(8e8 // 4),
):
    if not debug:
        bench_fn(
            func,
            warmup=warmup,
            repeat=repeat,
            proton_name=proton_name,
            flush_l2_size=flush_l2_size,
            nsight=nsight,
        )


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _parse_proton_tree(text):
    """Parse proton-viewer tree output into {impl: time_ms}.

    Accepts ALL depth-1 nodes (no KNOWN_IMPLS filter). For each depth-1 impl,
    takes the slowest depth-2 child kernel time.

    Returns (impl_times, baseline_errors) where:
      impl_times: {str: float} — impl name to avg time in ms
      baseline_errors: {str: str} — impl name to error message
    """
    impl = None
    results = {}
    baseline_errors = {}
    for raw in text.splitlines():
        line = _ANSI_RE.sub("", raw).rstrip()
        if not line:
            continue
        if line.startswith("BASELINE_ERROR:"):
            parts = line.split(":", 2)
            if len(parts) >= 3:
                baseline_errors[parts[1].strip()] = parts[2].strip()
            continue
        # Depth-1 impl header: starts with tree drawing chars
        if line and line[0] in "\u251c\u2514":  # ├ └
            parts = line.split("\u2500", 1)[-1].split()  # split on ─
            if len(parts) >= 2:
                impl = parts[1]
            else:
                impl = None
            continue
        # Depth-2 kernel: contains tree drawing chars at deeper indent
        if impl and ("\u251c\u2500" in line or "\u2514\u2500" in line):  # ├─ └─
            parts = line.split("\u2500", 1)[-1].split()
            if len(parts) >= 2:
                name = parts[1]
                if (
                    "vectorized_elementwise_kernel" in name
                    or "elementwise_kernel_with_index" in name
                ):
                    continue
                try:
                    t = float(parts[0])
                    results[impl] = max(results.get(impl, 0), t)
                except ValueError:
                    pass
    return results, baseline_errors


class ProtonContext:
    """Context manager for Proton profiling sessions.

    Always captures proton-viewer output and parses impl times so that
    get_impl_times() / get_baseline_errors() work after exiting the context.

    The proton tree is printed to **stdout** by default (visible on screen
    when running kernels interactively).  When the environment variable
    ``TIRX_BENCH_JSON=1`` is set (done automatically by ``--json`` mode),
    the tree goes to **stderr** instead so it does not corrupt the JSON on
    stdout.
    """

    def __init__(self, name="kernel", hook="triton", debug=False, nsight=False):
        self.name = name
        self.hook = hook
        self.debug = debug
        self.nsight = nsight
        self._impl_times = {}
        self._baseline_errors = {}

    def __enter__(self):
        if not is_running_under_pytest() and not self.debug and not self.nsight:
            proton.start(self.name, hook=self.hook)
            proton.deactivate()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not is_running_under_pytest() and not self.debug and not self.nsight:
            proton.finalize()

            hatchet = f"{self.name}.hatchet"
            result = subprocess.run(
                ["proton-viewer", "-m", "avg_time/ms", hatchet],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                self._impl_times, self._baseline_errors = _parse_proton_tree(result.stdout)
                out = sys.stderr if os.environ.get("TIRX_BENCH_JSON") else sys.stdout
                print(result.stdout, file=out, end="")
            else:
                print(
                    f"proton-viewer failed (rc={result.returncode}): {result.stderr}",
                    file=sys.stderr,
                )

            if os.path.exists(hatchet):
                os.remove(hatchet)

    def get_impl_times(self):
        """Return {impl_name: avg_time_ms} parsed from proton-viewer output."""
        return dict(self._impl_times)

    def get_baseline_errors(self):
        """Return {impl_name: error_message} from BASELINE_ERROR lines."""
        return dict(self._baseline_errors)


def _get_l2_cache_bytes():
    """Query L2 cache size from the current CUDA device, fallback to 128MB."""
    try:
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        if hasattr(props, "l2_cache_size") and props.l2_cache_size > 0:
            return props.l2_cache_size
    except Exception:
        pass
    return 128 * 1024 * 1024  # 128MB default (B200)


def _tensor_bytes(args):
    """Sum the byte size of all torch/tvm tensors in a (possibly nested) tuple."""
    total = 0
    if isinstance(args, list | tuple):
        for a in args:
            total += _tensor_bytes(a)
    elif isinstance(args, torch.Tensor):
        total += args.nelement() * args.element_size()
    elif hasattr(args, "numpy"):  # tvm.runtime.NDArray
        total += args.numpy().nbytes
    return total


def _compute_groups(sample_input, l2_bytes=None):
    """Return the number of input groups for L2 cache eviction.

    Matches the ThunderKittens formula::

        num_groups = 1                              if input >= 3*L2
                   = floor(3*L2 / input) + 1        otherwise

    Uses 3x L2 to guarantee reliable eviction.
    """
    if l2_bytes is None:
        l2_bytes = _get_l2_cache_bytes()
    input_bytes = _tensor_bytes(sample_input)
    if input_bytes <= 0:
        return 1
    threshold = l2_bytes * 3
    if input_bytes >= threshold:
        return 1
    return int(threshold // input_bytes) + 1


def bench_tk(
    funcs,
    inputs,
    warmup=500,
    repeat=100,
    cooldown_s=1.0,
):
    """ThunderKittens-style benchmarking with CUDA events.

    Implements the benchmarking methodology from the ThunderKittens 2.0 paper
    for more accurate GPU kernel measurement.

    Parameters
    ----------
    funcs : dict[str, callable]
        Map of impl name to callable, e.g. {"tir": fn1, "flashinfer": fn2}.
        Each callable receives ``*inputs[i]`` as arguments.
    inputs : list[tuple] | callable
        If **list**: pre-built input groups used directly.
        If **callable**: a factory ``() -> tuple`` that creates one input group.
        ``bench_tk`` auto-computes the number of groups needed for L2 eviction
        (via ``_compute_groups``) and calls the factory that many times.
    warmup : int
        Number of warmup iterations (default 500 for power-steady state).
    repeat : int
        Number of timed iterations.
    cooldown_s : float
        Seconds to sleep between impls for thermal cooldown.

    Returns
    -------
    dict[str, float]
        Map of impl name to average time in milliseconds.
    """
    if callable(inputs):
        make_inputs = inputs
        sample = make_inputs()
        num_groups = _compute_groups(sample)
        inputs = [sample] + [make_inputs() for _ in range(num_groups - 1)]

    num_groups = len(inputs)
    if num_groups == 0:
        return {}

    results = {}

    for idx, (name, func) in enumerate(funcs.items()):
        # Thermal cooldown between impls (skip for first)
        if idx > 0:
            time.sleep(cooldown_s)

        # Warmup
        for i in range(warmup):
            func(*inputs[i % num_groups])

        # Pure CUDA event timing — no proton overhead
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()

        start_event.record()
        for i in range(repeat):
            func(*inputs[i % num_groups])
        end_event.record()

        torch.cuda.synchronize()
        results[name] = start_event.elapsed_time(end_event) / repeat

        # Post-measurement cooldown (matches TK)
        time.sleep(cooldown_s)

    return results


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
    event_type_names: list[str],
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


@Tx.meta_class
class CudaProfiler:
    """A lightweight wrapper around Tx.timer_* CUDA intrinsics.

    Stores repeated arguments used by timer_init/start/end/finalize so users can
    call concise methods in kernels. Intended to mirror Pipeline/TileScheduler helpers.

    When ``profiler_enabled`` is False (or a false-y PrimExpr), calls to
    ``init/start/end/finalize`` become no-ops. This allows constructing a
    profiler unconditionally and eliminating external ``if PROFILER_ON:`` guards.
    """

    def __init__(
        self,
        profiler_buffer: Tx.Buffer,
        write_stride: int,
        num_groups: int,
        default_leader: None | tvm.tirx.PrimExpr | bool = None,
        profiler_enabled: bool | tvm.tirx.PrimExpr = True,
    ):
        self.buffer = profiler_buffer
        self.write_stride = write_stride
        self.num_groups = num_groups
        self.default_leader = default_leader
        # Accept either a Python bool or a PrimExpr; normalize simple bools to Tx.bool
        # so we can use it uniformly inside macros for conditional emission.
        if isinstance(profiler_enabled, bool | np.bool_):
            self.profiler_enabled = Tx.bool(bool(profiler_enabled))
        else:
            # Assume PrimExpr-like input; use as-is
            self.profiler_enabled = profiler_enabled  # type: ignore[assignment]

        self.profiler_tag = Tx.alloc_buffer(
            [1], "uint64", scope="local", align=8, name="profiler_tag"
        )
        self.profiler_write_offset = Tx.alloc_buffer(
            [1], "uint32", scope="local", align=8, name="profiler_write_offset"
        )

    def _leader(self, leader: None | tvm.tirx.PrimExpr | bool):
        if leader is not None:
            if isinstance(leader, bool | np.bool_):
                return Tx.bool(bool(leader))
            return leader
        if self.default_leader is not None:
            return self.default_leader
        return Tx.bool(True)

    @Tx.inline
    def init(self, group_id: tvm.tirx.PrimExpr):
        if self.profiler_enabled:
            Tx.timer_init_cuda(
                self.buffer.data,
                self.profiler_tag.data,
                self.profiler_write_offset.data,
                self.num_groups,
                group_id,
            )

    @Tx.inline
    def start(self, event_type: Enum, leader: None | tvm.tirx.PrimExpr | bool = None):
        if self.profiler_enabled:
            Tx.timer_start_cuda(
                event_type,
                self.buffer.data,
                self.profiler_tag.data,
                self.profiler_write_offset.data,
                self.write_stride,
                self._leader(leader),
            )

    @Tx.inline
    def end(self, event_type: Enum, leader: None | tvm.tirx.PrimExpr | bool = None):
        if self.profiler_enabled:
            Tx.timer_end_cuda(
                event_type,
                self.buffer.data,
                self.profiler_tag.data,
                self.profiler_write_offset.data,
                self.write_stride,
                self._leader(leader),
            )

    @Tx.inline
    def finalize(self, leader: None | tvm.tirx.PrimExpr | bool = None):
        if self.profiler_enabled:
            Tx.timer_finalize_cuda(
                self.buffer.data,
                self.profiler_tag.data,
                self.profiler_write_offset.data,
                self.write_stride,
                self._leader(leader),
            )
