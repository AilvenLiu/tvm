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
import pytest

import tvm
from tvm.tir.layout import TrainiumLayout, TileLayout
import numpy as np
import tvm.testing
from tvm.script import ir as I
from tvm.script import tir as T
from tvm.script import tirp as Tp
from tvm.ir import assert_structural_equal

target = tvm.target.Target("aws/trn1/trn1.2xlarge")

opcode_map = {
    "sum": "add",
    "max": "max",
    "min": "min",
}

Tp_func_map = {
    "sum": Tp.sum,
    "max": Tp.max,
    "min": Tp.min,
}


@pytest.mark.parametrize("op_type", ["sum", "max", "min"])
def test_simple_reduction(op_type):
    src_shape = [128, 512]
    src_layout = TrainiumLayout(
        dimension_types="PF", combined_1d_layout=T.TileLayout.from_tuple((128, 512), (1, 1))
    )
    dst_shape = [128, 1]
    dst_layout = TrainiumLayout(
        dimension_types="PF", combined_1d_layout=T.TileLayout.from_tuple((128, 1), (1, 1))
    )

    opcode = opcode_map[op_type]
    tp_func = Tp_func_map[op_type]
    # fmt: off
    @T.prim_func(tirp=True)
    def reduction() -> None:
        with T.kernel():
            A_sbuf = T.alloc_buffer(src_shape, "float32", scope="trn.sbuf", layout=src_layout)
            B_sbuf = T.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
            tp_func(B_sbuf, A_sbuf, axes=-1)

    @T.prim_func(tirp=True)
    def expected():
        T.func_attr({"global_symbol": "reduction"})
        with T.kernel():
            A_sbuf = T.alloc_buffer((128, 512), scope="trn.sbuf", logical_scope="kernel")
            B_sbuf = T.alloc_buffer((128, 1), scope="trn.sbuf", logical_scope="kernel")
            for b_loop in range(1):
                T.attr(0, "tensorized_nki_instruction", 1)
                for p_loop in T.serial(0, 128, annotations={"nki_dim":"P"}):
                    for f_loop in T.serial(0, 512, annotations={"nki_dim":"F"}):
                        T.nki_tensorreduce(B_sbuf[p_loop, 0], A_sbuf[p_loop, f_loop], opcode, False, -1)

    # fmt: on
    with target:
        mod = tvm.IRModule({"main": reduction})
        mod = tvm.tir.transform.LowerTIRp()(mod)
        assert_structural_equal(mod["main"], expected)


def test_reduction_with_multiple_axes():
    src_shape = [128, 512, 4]
    src_layout = TrainiumLayout(
        dimension_types="PFF",
        combined_1d_layout=T.TileLayout.from_tuple((128, 512, 4), (1, 1, 512)),
    )
    dst_shape = [128]
    dst_layout = TrainiumLayout(
        dimension_types="P", combined_1d_layout=T.TileLayout.from_tuple((128), (1))
    )

    # fmt: off
    @T.prim_func(tirp=True)
    def reduction():
        with T.kernel():
            A_sbuf = T.alloc_buffer(src_shape, "float32", scope="trn.sbuf", layout=src_layout)
            B_sbuf = T.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
            Tp.sum(B_sbuf, A_sbuf, axes=(1, 2), schedule_config={"max_inst_size": 2048})

    @T.prim_func(tirp=True)
    def expected():
        T.func_attr({"global_symbol": "reduction"})
        with T.kernel():
            A_sbuf = T.alloc_buffer((128, 2048), scope="trn.sbuf", logical_scope="kernel")
            B_sbuf = T.alloc_buffer((128, 1), scope="trn.sbuf", logical_scope="kernel")
            for b_loop in range(1):
                T.attr(0, "tensorized_nki_instruction", 1)
                for p_loop in T.serial(0, 128, annotations={"nki_dim":"P"}):
                    for f_loop in T.serial(0, 2048, annotations={"nki_dim":"F"}):
                        T.nki_tensorreduce(B_sbuf[p_loop, 0], A_sbuf[p_loop, f_loop], "add", False, -1)

    # fmt: on
    with target:
        mod = tvm.IRModule({"main": reduction})
        mod = tvm.tir.transform.LowerTIRp()(mod)
        assert_structural_equal(mod["main"], expected)


def test_reduction_in_loop():
    src_shape = [128, 512, 4]
    src_layout = TrainiumLayout(
        dimension_types="PFF", combined_1d_layout=T.TileLayout.from_tuple((128, 512, 4), (1, 4, 1))
    )
    dst_shape = [128, 4]
    dst_layout = TrainiumLayout(
        dimension_types="PF", combined_1d_layout=T.TileLayout.from_tuple((128, 4), (1, 1))
    )

    # fmt: off
    @T.prim_func(tirp=True)
    def reduction():
        with T.kernel():
            A_sbuf = T.alloc_buffer(src_shape, "float32", scope="trn.sbuf", layout=src_layout)
            B_sbuf = T.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
            for i in range(4):
                Tp.sum(B_sbuf[:, i], A_sbuf[:, :, i], axes=-2)
        
    @T.prim_func(tirp=True)
    def expected():
        T.func_attr({"global_symbol": "reduction"})
        with T.kernel():
            A_sbuf = T.alloc_buffer((128, 2048), scope="trn.sbuf", logical_scope="kernel")
            B_sbuf = T.alloc_buffer((128, 4), scope="trn.sbuf", logical_scope="kernel")
            for i, b_loop in T.grid(4, 1):
                T.attr(0, "tensorized_nki_instruction", 1)
                for p_loop in T.serial(0, 128, annotations={"nki_dim":"P"}):
                    for f_loop in T.serial(0, 512, annotations={"nki_dim":"F"}):
                        T.nki_tensorreduce(B_sbuf[p_loop, i], A_sbuf[p_loop, f_loop * 4 + i], "add", False, -1)
    # fmt: on
    with target:
        mod = tvm.IRModule({"main": reduction})
        mod = tvm.tir.transform.LowerTIRp()(mod)
        assert_structural_equal(mod["main"], expected)


def test_reduction_two_stage():
    src_shape = [128, 32, 4, 32]
    src_layout = TrainiumLayout(
        dimension_types="PF", combined_1d_layout=T.TileLayout.from_tuple((128, 32 * 32 * 4), (1, 1))
    )
    dst_shape = [128, 4]
    dst_layout = TrainiumLayout(
        dimension_types="PF", combined_1d_layout=T.TileLayout.from_tuple((128, 4), (1, 1))
    )

    # fmt: off
    @T.prim_func(tirp=True)
    def reduction():
        with T.kernel():
            A_sbuf = T.alloc_buffer(src_shape, "float32", scope="trn.sbuf", layout=src_layout)
            B_sbuf = T.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
            Tp.sum(B_sbuf, A_sbuf, axes=(1, 3))

    @T.prim_func(tirp=True)
    def expected():
        T.func_attr({"global_symbol": "reduction"})
        with T.kernel():
            intermediate_buffer = T.alloc_buffer((128, 32), logical_scope="kernel", scope="trn.sbuf")
            A_sbuf = T.alloc_buffer((128, 4096), scope="trn.sbuf", logical_scope="kernel")
            B_sbuf = T.alloc_buffer((128, 4), scope="trn.sbuf", logical_scope="kernel")
            for b_loop in range(4):
                for reduction_b_loop in range(32):
                    T.attr(0, "tensorized_nki_instruction", 1)
                    for p_loop in T.serial(0, 128, annotations={"nki_dim":"P"}):
                        for f_loop in T.serial(0, 32, annotations={"nki_dim":"F"}):
                            T.nki_tensorreduce(intermediate_buffer[p_loop, reduction_b_loop], A_sbuf[p_loop, reduction_b_loop * 128 + b_loop * 32 + f_loop], "add", False, -1)
                T.attr(0, "tensorized_nki_instruction", 1)
                for p_loop in T.serial(0, 128, annotations={"nki_dim":"P"}):
                    for f_loop in T.serial(0, 32, annotations={"nki_dim":"F"}):
                        T.nki_tensorreduce(B_sbuf[p_loop, b_loop], intermediate_buffer[p_loop, f_loop], "add", False, -1)

    # fmt: on
    with target:
        mod = tvm.IRModule({"main": reduction})
        mod = tvm.tirp.transform.PrivateBufferAlloc()(mod)
        mod = tvm.tir.transform.LowerTIRp()(mod)
        assert_structural_equal(mod["main"], expected)


def test_reduction_with_guard():
    src_shape = [512, 2048]
    src_layout = TrainiumLayout(
        dimension_types="FPF",
        combined_1d_layout=T.TileLayout.from_tuple((4, 128, 2048), (2048, 1, 1)),
    )
    dst_shape = [512, 1]
    dst_layout = TrainiumLayout(
        dimension_types="FP", combined_1d_layout=T.TileLayout.from_tuple((4, 128), (1, 1))
    )

    # fmt: off
    @T.prim_func(tirp=True)
    def reduction() -> None:
        with T.kernel():
            A_sbuf = T.alloc_buffer(src_shape, "float32", scope="trn.sbuf", layout=src_layout)
            B_sbuf = T.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
            for i in range(4):
                for j in range(4):
                    Tp.sum(B_sbuf[0: (i+1) * 128, 0], A_sbuf[0: (i+1) * 128, 0: (j+1) * 256], schedule_config={"max_inst_size": 512})

    @T.prim_func(tirp=True)
    def expected():
        T.func_attr({"global_symbol": "reduction"})
        with T.kernel():
            intermediate_buffer = T.alloc_buffer((128, 2), scope="trn.sbuf", logical_scope="kernel")
            A_sbuf = T.alloc_buffer((128, 8192), scope="trn.sbuf", logical_scope="kernel")
            B_sbuf = T.alloc_buffer((128, 4), scope="trn.sbuf", logical_scope="kernel")
            for i, j in T.grid(4, 4):
                for b_loop in range(4):
                    for reduction_b_loop in range(2):
                        T.attr(0, "tensorized_nki_instruction", 1)
                        for p_loop in T.serial(128, annotations={"nki_dim": "P"}):
                            for f_loop in T.serial(512, annotations={"nki_dim": "F"}):
                                if b_loop - i < 1 and reduction_b_loop * 512 + f_loop < j * 256 + 256:
                                    T.nki_tensorreduce(intermediate_buffer[p_loop, reduction_b_loop], A_sbuf[p_loop, b_loop * 2048 + reduction_b_loop * 512 + f_loop], "add", T.bool(False), -1)
                    T.attr(0, "tensorized_nki_instruction", 1)
                    for p_loop in T.serial(128, annotations={"nki_dim": "P"}):
                        for f_loop in T.serial(2, annotations={"nki_dim": "F"}):
                            if b_loop - i < 1 and f_loop * 2 - j < 1:
                                T.nki_tensorreduce(B_sbuf[p_loop, b_loop], intermediate_buffer[p_loop, f_loop], "add", T.bool(False), -1)
    # fmt: on
    with target:
        mod = tvm.IRModule({"main": reduction})
        mod = tvm.tirp.transform.PrivateBufferAlloc()(mod)
        mod = tvm.tir.transform.LowerTIRp()(mod)
        mod = tvm.tir.transform.Simplify()(mod)
        assert_structural_equal(mod["main"], expected)


def test_reduction_two_stage_workspace():
    src_shape = [128, 32, 4, 32]
    src_layout = TrainiumLayout(
        dimension_types="PF", combined_1d_layout=T.TileLayout.from_tuple((128, 32 * 32 * 4), (1, 1))
    )
    dst_shape = [128, 4]
    dst_layout = TrainiumLayout(
        dimension_types="PF", combined_1d_layout=T.TileLayout.from_tuple((128, 4), (1, 1))
    )

    # fmt: off
    @T.prim_func(tirp=True)
    def reduction():
        with T.kernel():
            intermediate_buffer = T.alloc_buffer((128, 64), logical_scope="kernel", scope="trn.sbuf")
            A_sbuf = T.alloc_buffer(src_shape, "float32", scope="trn.sbuf", layout=src_layout)
            B_sbuf = T.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
            Tp.sum(B_sbuf, A_sbuf, axes=(1, 3), workspace={"partial_reduce": intermediate_buffer})

    @T.prim_func(tirp=True)
    def expected():
        T.func_attr({"global_symbol": "reduction"})
        with T.kernel():
            intermediate_buffer = T.alloc_buffer((128, 64), logical_scope="kernel", scope="trn.sbuf")
            A_sbuf = T.alloc_buffer((128, 4096), scope="trn.sbuf", logical_scope="kernel")
            B_sbuf = T.alloc_buffer((128, 4), scope="trn.sbuf", logical_scope="kernel")
            for b_loop in range(4):
                for reduction_b_loop in range(32):
                    T.attr(0, "tensorized_nki_instruction", 1)
                    for p_loop in T.serial(0, 128, annotations={"nki_dim":"P"}):
                        for f_loop in T.serial(0, 32, annotations={"nki_dim":"F"}):
                            T.nki_tensorreduce(intermediate_buffer[p_loop, reduction_b_loop], A_sbuf[p_loop, reduction_b_loop * 128 + b_loop * 32 + f_loop], "add", False, -1)
                T.attr(0, "tensorized_nki_instruction", 1)
                for p_loop in T.serial(0, 128, annotations={"nki_dim":"P"}):
                    for f_loop in T.serial(0, 32, annotations={"nki_dim":"F"}):
                        T.nki_tensorreduce(B_sbuf[p_loop, b_loop], intermediate_buffer[p_loop, f_loop], "add", False, -1)

    # fmt: on
    with target:
        mod = tvm.IRModule({"main": reduction})
        mod = tvm.tir.transform.LowerTIRp()(mod)
        assert_structural_equal(mod["main"], expected)


if __name__ == "__main__":
    tvm.testing.main()
