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
# pylint: disable=missing-function-docstring
import pytest
import numpy as np

import tvm
import tvm.testing
from tvm.script import tir as T
from tvm.script import tirp as Tp


@pytest.mark.xfail(reason="Tp.fill is not correctly lowered")
def test_gpu_semaphore_init():
    @T.prim_func(tirp=True)
    def gpu_semaphore_init(semaphore: T.handle):
        sem = T.match_buffer(
            semaphore,
            (1024,),
            dtype="event_i32",
            scope="global.semaphore",
        )
        with T.kernel():
            cta_id = T.cta_id([1], parent="kernel")
            thread_id = T.thread_id([128], parent="cta")
            with T.cta():
                Tp.event_tensor_init(sem, init_value=128)

    target = tvm.target.Target.from_device(tvm.cuda(0))
    with target:
        mod = tvm.IRModule({"main": gpu_semaphore_init})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirp")
        semaphore = tvm.nd.array(np.zeros((1024,), dtype=np.int32), device=tvm.cuda(0))
        mod(semaphore)
        np.testing.assert_equal(semaphore.numpy(), np.full((1024,), 128, dtype=np.int32))


def test_gpu_semaphore_commit():
    @T.prim_func(tirp=True)
    def gpu_semaphore_commit(semaphore: T.handle):
        sem = T.match_buffer(
            semaphore,
            (128,),
            dtype="event_i32",
            scope="global.semaphore",
        )
        with T.kernel():
            cta_id = T.cta_id([128], parent="kernel")
            thread_id = T.thread_id([128], parent="cta")
            with T.cta():
                Tp.event_commit(sem[cta_id])

    target = tvm.target.Target.from_device(tvm.cuda(0))
    with target:
        mod = tvm.IRModule({"main": gpu_semaphore_commit})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirp")
        semaphore = tvm.nd.array(np.full((128,), 128, dtype=np.int32), device=tvm.cuda(0))
        mod(semaphore)
        np.testing.assert_equal(semaphore.numpy(), np.full((128,), 127, dtype=np.int32))


def test_gpu_semaphore_wait():
    @T.prim_func(tirp=True)
    def gpu_semaphore_wait(semaphore: T.handle):
        sem = T.match_buffer(
            semaphore,
            (64,),
            dtype="event_i32",
            scope="global.semaphore",
        )
        with T.kernel():
            cta_id = T.cta_id([128], parent="kernel")
            thread_id = T.thread_id([128], parent="cta")
            with T.cta()[0:64]:
                Tp.event_commit(sem[cta_id])
            with T.cta()[64:128]:
                Tp.event_wait(sem[cta_id - 64])

    target = tvm.target.Target.from_device(tvm.cuda(0))
    with target:
        mod = tvm.IRModule({"main": gpu_semaphore_wait})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirp")
        semaphore = tvm.nd.array(np.full((64,), 1, dtype=np.int32), device=tvm.cuda(0))
        mod(semaphore)
        np.testing.assert_equal(semaphore.numpy(), np.full((64,), 0, dtype=np.int32))


if __name__ == "__main__":
    tvm.testing.main()
