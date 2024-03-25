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

from tvm.script import tir as T
from tvm.tir.analysis import verify_tirp_well_formed as verify


def test_root_scope():
    # fmt: off
    @T.prim_func(tirp=True)
    def test1() -> None:
        with T.thread():
            pass
    
    @T.prim_func(tirp=True)
    def test2() -> None:
        with T.warp():
            pass

    @T.prim_func(tirp=True)
    def test3() -> None:
        with T.cta():
            pass

    @T.prim_func(tirp=True)
    def test4() -> None:
        with T.kernel():
            pass

    @T.prim_func(tirp=True)
    def test5() -> None:
        with T.world():
            pass
    # fmt: on

    with pytest.raises(Exception, match="invalid exec_scope thread as root"):
        verify(test1)
    with pytest.raises(Exception, match="invalid exec_scope warp as root"):
        verify(test2)
    with pytest.raises(Exception, match="invalid exec_scope cta as root"):
        verify(test3)
    verify(test4)
    verify(test5)


def test_nested_scope():
    # fmt: off
    @T.prim_func(tirp=True)
    def test1() -> None:
        with T.kernel():
            with T.cta():
                with T.warp():
                    with T.thread():
                        pass
                with T.thread():
                    pass
    
    @T.prim_func(tirp=True)
    def test2() -> None:
        with T.kernel():
                with T.thread():
                    with T.cta():
                        pass

    @T.prim_func(tirp=True)
    def test3() -> None:
        with T.kernel():
                with T.warp():
                    with T.thread():
                        with T.cta():
                            pass
    # fmt: on

    verify(test1)
    verify(test2)
    with pytest.raises(Exception, match="invalid exec_scope cta under warp"):
        verify(test3)


if __name__ == "__main__":
    test_root_scope()
    test_nested_scope()
