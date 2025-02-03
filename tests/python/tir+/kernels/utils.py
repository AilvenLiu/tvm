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

import argparse
import subprocess
from tvm.contrib import nvcc
import triton.profiler as proton
from triton.testing import do_bench
import tvm


def is_running_under_pytest():
    """Check if the code is being executed within a pytest session."""
    return "PYTEST_CURRENT_TEST" in os.environ


def setup():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-ptx", type=str, help="Dump PTX code to specified file")
    parser.add_argument("--dump-source", action="store_true", help="Dump source code")
    args = parser.parse_args()

    if args.dump_ptx:

        @tvm.register_func("tvm_callback_cuda_compile", override=True)
        def tvm_callback_cuda_compile(code, target):
            ptx = nvcc.compile_cuda(code, target_format="ptx")
            with open(args.dump_ptx, "w", encoding="utf-8") as f:
                f.write(ptx.decode())
            return ptx

    return args


def bench(func, warmup=0, repeat=10, proton_name="kernel"):
    if not is_running_under_pytest():
        with proton.scope(proton_name, metrics={}):
            ms = do_bench(func, warmup=warmup, rep=repeat)
    else:
        ms = do_bench(func, warmup=warmup, rep=repeat)

    return ms


class ProtonContext:
    def __init__(self, name="kernel", hook="triton"):
        self.name = name
        self.hook = hook

    def __enter__(self):
        if not is_running_under_pytest():
            proton.start(self.name, hook=self.hook)
            proton.activate(0)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not is_running_under_pytest():
            proton.deactivate(0)
            proton.finalize()

            subprocess.run(
                ["proton-viewer", "-m", "avg_time/ms", f"{self.name}.hatchet"], check=True
            )
