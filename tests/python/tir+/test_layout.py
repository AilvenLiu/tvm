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
import tvm.testing
from tvm.script import tir as T


def test_constructor_nested_tuple():
    layout = T.TileLayout.from_nested_tuple(
        data=((8, T.S(0)), (8, (T.S(1), 2))),
        strides=((16, -1), (2, (-1, 1))),
        device=(8, 4),
        from_to=("thread", "warp"),
    )

    data_leaf1 = T.IterTreeSplit(children=[], extent=8)
    data_leaf2 = T.IterTreeSplit(children=[], extent=8)
    data_leaf3 = T.IterTreeSplit(children=[], extent=8)
    data_leaf4 = T.IterTreeSplit(children=[], extent=4)
    data_leaf5 = T.IterTreeSplit(children=[], extent=2)

    device_leaf1 = T.IterTreeSplit(children=[], extent=8)
    device_leaf2 = T.IterTreeSplit(children=[], extent=4)

    layout_expected = T.TileLayout(
        data_tree=T.DataIterTree(
            root=T.IterTreeSplit(
                children=[
                    T.IterTreeSplit(
                        children=[
                            data_leaf1,
                            data_leaf2,
                        ],
                        extent=64,
                    ),
                    T.IterTreeSplit(
                        children=[
                            data_leaf3,
                            T.IterTreeSplit(
                                children=[
                                    data_leaf4,
                                    data_leaf5,
                                ],
                                extent=8,
                            ),
                        ],
                        extent=64,
                    ),
                ],
                extent=4096,
            ),
            coeff=[16, -1, 2, -1, 1],
        ),
        device_tree=T.DeviceIterTree(
            root=T.IterTreeSplit(
                children=[
                    device_leaf1,
                    device_leaf2,
                ],
                extent=32,
            ),
            attrs=[
                T.DeviceIterAttr.split(bound=1),
                T.DeviceIterAttr.split(bound=3),
            ],
        ),
        from_scope=T.ExecScope.create("thread"),
        to_scope=T.ExecScope.create("warp"),
    )

    tvm.ir.assert_structural_equal(layout, layout_expected, True)

    with pytest.raises(AssertionError):
        # device axis 1 can only either be S or E
        layout = T.TileLayout.from_nested_tuple(
            data=((8, T.S(0)), (8, (T.S(1), 2))),
            strides=((16, -1), (2, (-1, 1))),
            device=(4, 8),
            exclusive=[(1, 0)],
            from_to=("thread", "warp"),
        )
    with pytest.raises(AssertionError):
        # device axis 0 can only be bound once
        layout = T.TileLayout.from_nested_tuple(
            data=((8, T.S(0)), (8, (T.S(0), 2))),
            strides=((16, -1), (2, (-1, 1))),
            device=(4, 8),
            from_to=("thread", "warp"),
        )
    with pytest.raises(AssertionError):
        # from_to must be a tuple of length 2 if provided
        layout = T.TileLayout.from_nested_tuple(
            data=((8, T.S(0)), (8, (T.S(1), 2))),
            strides=((16, -1), (2, (-1, 1))),
            device=(4, 8),
            from_to=("thread",),
        )
    with pytest.raises(AssertionError):
        # data and strides do not match
        layout = T.TileLayout.from_nested_tuple(
            data=((8, T.S(0)), (8, (T.S(1), 2))),
            strides=((16, -1), (2, (-1, 1), 1)),
            device=(4, 8),
            from_to=("thread", "warp"),
        )
    with pytest.raises(AssertionError):
        # device index out of bound
        layout = T.TileLayout.from_nested_tuple(
            data=((8, T.S(0)), (8, (T.S(2), 2))),
            strides=((16, -1), (2, (-1, 1))),
            device=(4, 8),
            from_to=("thread", "warp"),
        )
    with pytest.raises(AssertionError):
        # device index out of bound
        layout = T.TileLayout.from_nested_tuple(
            data=((8, T.S(0)), (8, (T.S(1), 2))),
            strides=((16, -1), (2, (-1, 1))),
            device=(4, 8),
            exclusive=[(2, 0)],
            from_to=("thread", "warp"),
        )


if __name__ == "__main__":
    test_constructor_nested_tuple()
