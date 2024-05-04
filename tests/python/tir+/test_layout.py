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
from tvm.ir import assert_structural_equal


def test_constructor_nested_tuple_no_device():
    layout = T.TileLayout.from_nested_tuple(
        data=((8, 8), (8, (4, 2))),
        strides=((512, 64), (8, (2, 1))),
    )

    data_leaf1 = T.IterTreeSplit(children=[], extent=8)
    data_leaf2 = T.IterTreeSplit(children=[], extent=8)
    data_leaf3 = T.IterTreeSplit(children=[], extent=8)
    data_leaf4 = T.IterTreeSplit(children=[], extent=4)
    data_leaf5 = T.IterTreeSplit(children=[], extent=2)

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
            coeff=[512, 64, 8, 2, 1],
        ),
        device_tree=None,
        from_scope=None,
        to_scope=None,
    )

    tvm.ir.assert_structural_equal(layout, layout_expected, True)

    with pytest.raises(AssertionError):
        # from_to must be None if device is None
        layout = T.TileLayout.from_nested_tuple(
            data=((8, 8), (8, (4, 2))),
            strides=((512, 64), (8, (2, 1), 1)),
            from_to=("thread", "warp"),
        )
    with pytest.raises(AssertionError):
        # exclusive must be None if device is None
        layout = T.TileLayout.from_nested_tuple(
            data=((8, 8), (8, (4, 2))),
            strides=((512, 64), (8, (2, 1))),
            exclusive=[(1, 0)],
        )


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


def test_normalize_tile_layout():
    def normalize_tile_layout(layout):
        return T.TileLayout.normalize(layout)

    # no normalization case
    layout = T.TileLayout.from_nested_tuple(
        data=((8, 8), (8, (4, 2))),
        strides=((512, 64), (8, (2, 1))),
    )
    assert_structural_equal(layout, normalize_tile_layout(layout))

    # only data tree #1
    layout = T.TileLayout.from_nested_tuple(
        data=((8, 8, 1), (8, (4, 2))),
        strides=((512, 64, 160), (8, (2, 1))),
    )
    layout_expected = T.TileLayout.from_nested_tuple(
        data=((8, 8), (8, (4, 2))), strides=((512, 64), (8, (2, 1)))
    )
    assert_structural_equal(layout_expected, normalize_tile_layout(layout))

    # only data tree #2
    layout = T.TileLayout.from_nested_tuple(
        data=((8, 8), (8, (4, 1, 1))),
        strides=((512, 64), (8, (2, 1, 1))),
    )
    layout_expected = T.TileLayout.from_nested_tuple(
        data=((8, 8), (8, 4)), strides=((512, 64), (8, 2))
    )
    assert_structural_equal(layout_expected, normalize_tile_layout(layout))

    # only device tree #3
    layout = T.TileLayout.from_nested_tuple(
        data=((8, 8), (1, 1, (1, 4, 1, 1))),
        strides=((512, 64), (1, 1, (1, 2, 1, 1))),
    )
    layout_expected = T.TileLayout.from_nested_tuple(data=((8, 8), 4), strides=((512, 64), 2))
    assert_structural_equal(layout_expected, normalize_tile_layout(layout))

    # no normalization case
    layout_normalized = T.TileLayout.from_nested_tuple(
        data=((8, T.S(0)), (8, (T.S(1), 2))),
        strides=((16, -1), (2, (-1, 1))),
        device=(8, 4),
        from_to=("thread", "warp"),
    )
    assert_structural_equal(layout_normalized, normalize_tile_layout(layout_normalized))

    # both data and device tree #1
    layout = T.TileLayout.from_nested_tuple(
        data=((8, T.S(0)), (8, (1, T.S(1), 2, 1))),
        strides=((16, -1), (2, (1, -1, 1, 1))),
        device=(8, 4),
        from_to=("thread", "warp"),
    )
    assert_structural_equal(layout_normalized, normalize_tile_layout(layout))

    # both data and device tree #2
    layout = T.TileLayout.from_nested_tuple(
        data=((8, T.S(0)), (8, (1, T.S(2), 2, T.S(1)))),
        strides=((16, -1), (2, (1, -1, 1, -1))),
        device=(8, 1, 4),
        from_to=("thread", "warp"),
    )
    assert_structural_equal(layout_normalized, normalize_tile_layout(layout))

    # both data and device tree #3
    layout = T.TileLayout.from_nested_tuple(
        data=((8, T.S(0)), (8, (1, T.S(1), 2, 1))),
        strides=((16, -1), (2, (1, -1, 1, -1))),
        device=(8, 1, 4),
        exclusive=[(2, 0)],
        from_to=("thread", "warp"),
    )
    layout_normalized = T.TileLayout.from_nested_tuple(
        data=((8, T.S(0)), (8, 2)),
        strides=((16, -1), (2, 1)),
        device=(8, 4),
        exclusive=[(1, 0)],
        from_to=("thread", "warp"),
    )
    assert_structural_equal(layout_normalized, normalize_tile_layout(layout))

    # unit layout case#1
    layout = T.TileLayout.from_nested_tuple(
        data=((1, 1), (1, (1, 1))), strides=((1, 1), (1, (1, 1)))
    )
    layout_unit = T.TileLayout.from_nested_tuple(data=1, strides=1)
    assert_structural_equal(layout_unit, normalize_tile_layout(layout))

    # unit layout case#2
    layout = T.TileLayout.from_nested_tuple(
        data=((1, T.S(0)), (T.S(1), (1, 1))),
        strides=((1, -1), (-1, (1, 1))),
        device=(1, 1),
        from_to=("thread", "warp"),
    )
    layout_unit = T.TileLayout.from_nested_tuple(
        data=1, strides=1, device=1, from_to=("thread", "warp")
    )
    assert_structural_equal(layout_unit, normalize_tile_layout(layout))


def test_tile_layout():
    layout = T.TileLayout.from_nested_tuple(data=8, strides=1)
    layout_tile = T.TileLayout.from_nested_tuple(data=((8, 8),), strides=((8, 1),))
    assert_structural_equal(layout_tile, T.TileLayout.tile(layout, layout))

    layout = T.TileLayout.from_nested_tuple(data=(8, 8), strides=(8, 1))
    layout_tile = T.TileLayout.from_nested_tuple(data=((8, 8), (8, 8)), strides=((512, 8), (64, 1)))
    assert_structural_equal(layout_tile, T.TileLayout.tile(layout, layout))

    layout2 = T.TileLayout.from_nested_tuple(data=(2, 4), strides=(1, 2))
    layout_tile = T.TileLayout.from_nested_tuple(data=((8, 2), (8, 4)), strides=((64, 1), (8, 2)))
    assert_structural_equal(layout_tile, T.TileLayout.tile(layout, layout2))

    layout3 = T.TileLayout.from_nested_tuple(data=((4, 2), (2, 4)), strides=((16, 8), (1, 2)))
    layout_tile = T.TileLayout.from_nested_tuple(
        data=((8, (4, 2)), (8, (2, 4))), strides=((512, (16, 8)), (64, (1, 2)))
    )
    assert_structural_equal(layout_tile, T.TileLayout.tile(layout, layout3))

    # Tile over a sharded layout
    layout = T.TileLayout.from_nested_tuple(
        data=((T.S(0), 1), (T.S(1), 2)),
        strides=((-1, 2), (-1, 1)),
        device=(8, 4),
        from_to=("thread", "warp"),
    )
    layout_tile = T.TileLayout.tile(
        outer=T.TileLayout.from_nested_tuple(data=(8, 8), strides=(8, 1)),
        inner=layout,
    )
    layout_expected = T.TileLayout.from_nested_tuple(
        data=((8, (T.S(0), 1)), (8, (T.S(1), 2))),
        strides=((16, (-1, 2)), (2, (-1, 1))),
        device=(8, 4),
        from_to=("thread", "warp"),
    )
    assert_structural_equal(layout_expected, layout_tile)

    with pytest.raises(Exception):
        # dims mismatch
        layout = T.TileLayout.from_nested_tuple(data=8, strides=1)
        layout2 = T.TileLayout.from_nested_tuple(data=(2, 4), strides=(1, 2))
        T.TileLayout.tile(layout, layout2)


def test_shard_layout():
    # mma layout test
    layout = T.TileLayout.from_nested_tuple(data=(1, 2), strides=(2, 1))
    layout_warp = T.TileLayout.shard(
        shape=(8, 8), mesh=(8, 4), strategy="S0S1", inner=layout, from_to=("thread", "warp")
    )
    layout_expected = T.TileLayout.from_nested_tuple(
        data=((T.S(0), 1), (T.S(1), 2)),
        strides=((-1, 2), (-1, 1)),
        device=(8, 4),
        from_to=("thread", "warp"),
    )
    assert_structural_equal(layout_expected, layout_warp)

    layout_cta = T.TileLayout.shard(
        shape=(16, 16), mesh=(2, 2), strategy="S0S1", inner=layout_warp, from_to=("warp", "cta")
    )
    layout_expected = T.TileLayout.from_nested_tuple(
        data=((T.S(0), (T.S(2), 1)), (T.S(1), (T.S(3), 2))),
        strides=((-1, (-1, 2)), (-1, (-1, 1))),
        device=(2, 2, 8, 4),
        from_to=("thread", "cta"),
    )
    assert_structural_equal(layout_expected, layout_cta)

    # quad shuffle test
    layout = T.TileLayout.from_nested_tuple(data=(1, 2), strides=(2, 1))
    layout_warp = T.TileLayout.shard(
        shape=(8, 2),
        mesh=(8, 4),
        strategy="S0E0",
        inner=layout,
        from_to=("thread", "warp"),
    )
    layout_expected = T.TileLayout.from_nested_tuple(
        data=((T.S(0), 1), 2),
        strides=((-1, 2), 1),
        device=(8, 4),
        exclusive=[(1, 0)],
        from_to=("thread", "warp"),
    )
    assert_structural_equal(layout_expected, layout_warp)

    # replicate test
    layout = T.TileLayout.from_nested_tuple(data=(64, 128), strides=(128, 1))
    layout_rep = T.TileLayout.shard(
        shape=(128, 128),
        mesh=(2, 2),
        strategy="S0R",
        inner=layout,
        from_to=("kernel", "world"),
    )
    layout_expected = T.TileLayout.from_nested_tuple(
        data=((T.S(0), 64), 128),
        strides=((-1, 128), 1),
        device=(2, 2),
        from_to=("kernel", "world"),
    )
    assert_structural_equal(layout_expected, layout_rep)

    # error case: shape, mesh and inner shape mismatch
    layout = T.TileLayout.from_nested_tuple(data=(64, 64), strides=(128, 1))
    with pytest.raises(Exception):
        layout_rep = T.TileLayout.shard(
            shape=(128, 128),
            mesh=(2, 2),
            strategy="S0R",
            inner=layout,
            from_to=("kernel", "world"),
        )


if __name__ == "__main__":
    test_constructor_nested_tuple_no_device()
    test_constructor_nested_tuple()
    test_normalize_tile_layout()
    test_tile_layout()
    test_shard_layout()
