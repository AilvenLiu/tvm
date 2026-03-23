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
"""Tests for tvm.tirx.bench.utils — _parse_proton_tree and bench_tk."""

import pytest
import torch

import tvm.testing
from tvm.tirx.bench.utils import _compute_groups, _parse_proton_tree, bench_tk

# ── _parse_proton_tree ──────────────────────────────────────────────────────


SAMPLE_TREE = """\
├─ 1.500 tir
│  ├─ 1.500 my_kernel_fn
│  └─ 0.001 vectorized_elementwise_kernel
└─ 0.800 cublas
   └─ 0.800 sm90_xmma_gemm_f16f16
"""


def test_parse_proton_tree_basic():
    impls, errors = _parse_proton_tree(SAMPLE_TREE)
    assert impls == {"tir": 1.5, "cublas": 0.8}
    assert errors == {}


def test_parse_proton_tree_filters_elementwise():
    """vectorized_elementwise_kernel and elementwise_kernel_with_index are skipped."""
    tree = """\
├─ 0.500 tir
│  ├─ 0.500 real_kernel
│  └─ 0.001 elementwise_kernel_with_index
"""
    impls, _ = _parse_proton_tree(tree)
    assert impls == {"tir": 0.5}


def test_parse_proton_tree_slowest_child():
    """Takes the slowest depth-2 child per impl."""
    tree = """\
├─ 2.000 tir
│  ├─ 0.300 kernel_a
│  └─ 0.700 kernel_b
"""
    impls, _ = _parse_proton_tree(tree)
    assert impls == {"tir": 0.7}


def test_parse_proton_tree_baseline_errors():
    tree = """\
BASELINE_ERROR: cublas: CUDA OOM
├─ 1.000 tir
│  └─ 1.000 my_kernel
"""
    impls, errors = _parse_proton_tree(tree)
    assert impls == {"tir": 1.0}
    assert errors == {"cublas": "CUDA OOM"}


def test_parse_proton_tree_ansi_stripped():
    """ANSI color codes are stripped before parsing."""
    tree = "\x1b[32m├─ 1.000 tir\x1b[0m\n│  └─ 1.000 k\n"
    impls, _ = _parse_proton_tree(tree)
    assert impls == {"tir": 1.0}


def test_parse_proton_tree_empty():
    impls, errors = _parse_proton_tree("")
    assert impls == {}
    assert errors == {}


# ── bench_tk ────────────────────────────────────────────────────────────────


@tvm.testing.requires_cuda
def test_bench_tk_basic():
    """bench_tk returns positive times for each impl."""
    M, N = 256, 256
    A = torch.randn(M, N, device="cuda", dtype=torch.float16)
    B = torch.randn(M, N, device="cuda", dtype=torch.float16)

    funcs = {
        "matmul": lambda a, b: torch.mm(a, b),
    }
    inputs = [(A, B)]

    results = bench_tk(funcs, inputs, warmup=5, repeat=10, cooldown_s=0.0)
    assert "matmul" in results
    assert results["matmul"] > 0


@tvm.testing.requires_cuda
def test_bench_tk_multiple_impls():
    """Multiple impls each get their own timing."""
    M, N = 128, 128
    A = torch.randn(M, N, device="cuda", dtype=torch.float16)
    B = torch.randn(M, N, device="cuda", dtype=torch.float16)

    funcs = {
        "mm": lambda a, b: torch.mm(a, b),
        "addmm": lambda a, b: torch.addmm(
            torch.zeros(M, N, device="cuda", dtype=torch.float16), a, b
        ),
    }
    inputs = [(A, B)]

    results = bench_tk(funcs, inputs, warmup=5, repeat=10, cooldown_s=0.0)
    assert set(results.keys()) == {"mm", "addmm"}
    assert all(v > 0 for v in results.values())


@tvm.testing.requires_cuda
def test_bench_tk_multiple_input_groups():
    """Multiple input groups cycle correctly (L2 eviction)."""
    M, N = 128, 128
    groups = [
        (
            torch.randn(M, N, device="cuda", dtype=torch.float16),
            torch.randn(M, N, device="cuda", dtype=torch.float16),
        )
        for _ in range(4)
    ]

    funcs = {"mm": lambda a, b: torch.mm(a, b)}
    results = bench_tk(funcs, groups, warmup=5, repeat=20, cooldown_s=0.0)
    assert results["mm"] > 0


def test_bench_tk_empty_inputs():
    """Empty inputs returns empty dict."""
    funcs = {"mm": lambda a, b: torch.mm(a, b)}
    results = bench_tk(funcs, inputs=[], warmup=5, repeat=10)
    assert results == {}


# ── _compute_groups ────────────────────────────────────────────────────────


def test_compute_groups_small_tensors():
    """Small tensors need many groups to fill 3x L2."""
    # 128x128 fp16 = 32KB.  3*128MB / 32KB = 12288, +1 = 12289
    sample = (torch.empty(128, 128, dtype=torch.float16),)
    n = _compute_groups(sample, l2_bytes=128 * 1024 * 1024)
    assert n == 12289


def test_compute_groups_large_tensors():
    """Inputs >= 3x L2 need only 1 group."""
    # 16384x16384 fp32 = 1GB >> 3*128MB = 384MB
    sample = (torch.empty(16384, 16384, dtype=torch.float32),)
    n = _compute_groups(sample, l2_bytes=128 * 1024 * 1024)
    assert n == 1


def test_compute_groups_moderate_tensors():
    """Moderate tensors: floor(3*L2 / input) + 1."""
    # 8192x8192 bf16 = 128MB.  floor(384M / 128M) + 1 = 4
    sample = (torch.empty(8192, 8192, dtype=torch.bfloat16),)
    n = _compute_groups(sample, l2_bytes=128 * 1024 * 1024)
    assert n == 4


@tvm.testing.requires_cuda
def test_bench_tk_callable_inputs():
    """bench_tk accepts a factory callable and auto-computes groups."""
    M, N = 256, 256

    call_count = [0]

    def make_inputs():
        call_count[0] += 1
        return (
            torch.randn(M, N, device="cuda", dtype=torch.float16),
            torch.randn(M, N, device="cuda", dtype=torch.float16),
        )

    funcs = {"mm": lambda a, b: torch.mm(a, b)}
    results = bench_tk(funcs, make_inputs, warmup=5, repeat=10, cooldown_s=0.0)
    assert "mm" in results
    assert results["mm"] > 0
    assert call_count[0] >= 2  # at least 2 groups created


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
