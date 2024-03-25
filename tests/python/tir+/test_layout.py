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
import tvm
import tvm.testing
from tvm.script import tir as T, from_source


def test_nested_tuple():
    layout = T.TileLayout.from_nested_tuple(
        data=((8, T.S(0)), (8, (T.S(1), 2))),
        strides=((16, -1), (2, (-1, 1))),
        device=(4, 8),
        exclusive=((1, 0),),
    )
    print(layout)
    data_root = T.Var("", "int32")
    data_64_outer = T.Var("", "int32")
    data_64_outer_8_outer = T.Var("", "int32")
    data_64_outer_8_inner = T.Var("", "int32")
    data_64_inner = T.Var("", "int32")
    data_64_inner_8_outer = T.Var("", "int32")
    data_64_inner_8_inner = T.Var("", "int32")
    data_64_inner_8_inner_4_outer = T.Var("", "int32")
    data_64_inner_8_inner_2_inner = T.Var("", "int32")

    device_root = T.Var("", "int32")
    device_4_outer = T.Var("", "int32")
    device_8_inner = T.Var("", "int32")

    layout_expected = T.TileLayout(
        data_trees=[
            T.DataIterTree(
                root=data_root,
                splits=[
                    T.IterTreeSplit(
                        parent=data_64_inner_8_inner,
                        children=[data_64_inner_8_inner_2_inner, data_64_inner_8_inner_4_outer],
                        extents=[2, 4],
                    ),
                    T.IterTreeSplit(
                        parent=data_64_inner,
                        children=[data_64_inner_8_inner, data_64_inner_8_outer],
                        extents=[8, 8],
                    ),
                    T.IterTreeSplit(
                        parent=data_64_outer,
                        children=[data_64_outer_8_inner, data_64_outer_8_outer],
                        extents=[8, 8],
                    ),
                    T.IterTreeSplit(
                        parent=data_root, children=[data_64_inner, data_64_outer], extents=[64, 64]
                    ),
                ],
                coeff=[1, -1, 2, -1, 16],
            )
        ],
        device_trees=[
            T.DeviceIterTree(
                root=device_root,
                splits=[
                    T.IterTreeSplit(
                        parent=device_root,
                        children=[device_8_inner, device_4_outer],
                        extents=[8, 4],
                    )
                ],
                attrs=[
                    T.ScopeIdAttr(type=0, bound=data_64_outer_8_inner, owner=None),
                    T.ScopeIdAttr(type=2, bound=data_64_inner_8_inner_4_outer, owner=0),
                ],
            )
        ],
    )
    tvm.ir.assert_structural_equal(layout, layout_expected, True)


if __name__ == "__main__":
    test_nested_tuple()
