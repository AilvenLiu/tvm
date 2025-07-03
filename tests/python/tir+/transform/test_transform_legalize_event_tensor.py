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
from tvm.tir.layout import TileLayout
import numpy as np
import tvm.testing
from tvm.script import ir as I
from tvm.script import tir as T
from tvm.script import tirp as Tp
from tvm.ir import assert_structural_equal
from tvm.tirp.transform import EventTensorLegalizer

def test_event_tensor_legalizer():
    # fmt: off
    @T.prim_func(tirp=True)
    def before(A_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, [128], "event_i64", scope="global.semaphore")
        with T.kernel():
            cta_id = T.cta_id([1], parent="kernel")
            thread_id = T.thread_id([128], parent="cta")
            with T.thread():
                A[thread_id] += T.int64(1)
            
    @T.prim_func(tirp=True)
    def after(A_ptr: T.handle) -> None:
        T.func_attr({"global_symbol": "before"})
        A = T.match_buffer(A_ptr, [128], "int64", scope="global.semaphore")
        with T.kernel():
            cta_id = T.cta_id([1], parent="kernel")
            thread_id = T.thread_id([128], parent="cta")
            with T.thread():
                A[thread_id] += T.int64(1)

    # fmt: on
    mod = tvm.IRModule({"before": before})
    mod = EventTensorLegalizer()(mod)
    assert_structural_equal(mod["before"], after)


def test_event_tensor_legalizer_2():
    # fmt: off
    @T.prim_func(tirp=True)
    def before(A_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, [128], "event_i64", scope="global.semaphore")
        with T.kernel():
            cta_id = T.cta_id([1], parent="kernel")
            thread_id = T.thread_id([128], parent="cta")
            with T.thread():
                T.cuda.atomic_add(A.access_ptr("rw", offset=A.offset_of_p([thread_id])), T.int64(1))
            
    @T.prim_func(tirp=True)
    def after(A_ptr: T.handle) -> None:
        T.func_attr({"global_symbol": "before"})
        A = T.match_buffer(A_ptr, [128], "int64", scope="global.semaphore")
        with T.kernel():
            cta_id = T.cta_id([1], parent="kernel")
            thread_id = T.thread_id([128], parent="cta")
            with T.thread():
                T.cuda.atomic_add(A.access_ptr("rw", offset=A.offset_of_p([thread_id])), T.int64(1))

    # fmt: on
    mod = tvm.IRModule({"main": before})
    mod = EventTensorLegalizer()(mod)
    print(mod["main"])
    assert_structural_equal(mod["main"], after)


if __name__ == "__main__":
    tvm.testing.main()
