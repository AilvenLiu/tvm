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
import itertools

import tvm
import tvm.testing
from tvm.script import tir as T
from tvm.ir import assert_structural_equal


def test_constructor_nested_tuple_no_device():
    layout = T.TileLayout.from_nested_tuple(
        data=(8, 8, 8, 4, 2),
        strides=(512, 64, 8, 2, 1),
    )

    layout_expected = T.TileLayout(
        data_iter_array=[
            T.DataIterAttr(extent=8, stride=512),
            T.DataIterAttr(extent=8, stride=64),
            T.DataIterAttr(extent=8, stride=8),
            T.DataIterAttr(extent=4, stride=2),
            T.DataIterAttr(extent=2, stride=1),
        ],
        device_iter_array=None,
        from_scope=None,
        to_scope=None,
    )

    tvm.ir.assert_structural_equal(layout, layout_expected, True)

    with pytest.raises(AssertionError):
        # from_to must be None if device is None
        layout = T.TileLayout.from_nested_tuple(
            data=(8, 8, 8, 4, 2),
            strides=(512, 64, 8, 2, 1),
            from_to=("thread", "warp"),
        )
    with pytest.raises(AssertionError):
        # exclusive must be None if device is None
        layout = T.TileLayout.from_nested_tuple(
            data=(8, 8, 8, 4, 2),
            strides=(512, 64, 8, 2, 1),
            exclusive=[(1, 0)],
        )


def test_constructor_nested_tuple():
    layout = T.TileLayout.from_nested_tuple(
        data=(8, T.S(0), 8, T.S(1), 2),
        strides=(16, -1, 2, -1, 1),
        device=(8, 4),
        from_to=("thread", "warp"),
    )

    layout_expected = T.TileLayout(
        data_iter_array=[
            T.DataIterAttr(extent=8, stride=16),
            T.DataIterAttr(extent=8, stride=-1),
            T.DataIterAttr(extent=8, stride=2),
            T.DataIterAttr(extent=4, stride=-1),
            T.DataIterAttr(extent=2, stride=1),
        ],
        device_iter_array=[
            T.DeviceIterAttr.split(extent=8, bound=1),
            T.DeviceIterAttr.split(extent=4, bound=3),
        ],
        from_scope=T.ExecScope.create("thread"),
        to_scope=T.ExecScope.create("warp"),
    )

    tvm.ir.assert_structural_equal(layout, layout_expected, True)

    with pytest.raises(AssertionError):
        # device axis 1 can only either be S or E
        layout = T.TileLayout.from_nested_tuple(
            data=(8, T.S(0), 8, T.S(1), 2),
            strides=(16, -1, 2, -1, 1),
            device=(4, 8),
            exclusive=[(1, 0)],
            from_to=("thread", "warp"),
        )
    with pytest.raises(AssertionError):
        # device axis 0 can only be bound once
        layout = T.TileLayout.from_nested_tuple(
            data=(8, T.S(0), 8, T.S(0), 2),
            strides=(16, -1, 2, -1, 1),
            device=(4, 8),
            from_to=("thread", "warp"),
        )
    with pytest.raises(AssertionError):
        # from_to must be a tuple of length 2 if provided
        layout = T.TileLayout.from_nested_tuple(
            data=(8, T.S(0), 8, T.S(1), 2),
            strides=(16, -1, 2, -1, 1),
            device=(4, 8),
            from_to=("thread",),
        )
    with pytest.raises(AssertionError):
        # data and strides do not match
        layout = T.TileLayout.from_nested_tuple(
            data=(8, T.S(0), 8, T.S(1), 2),
            strides=(16, -1, 2, -1, 1, 1),
            device=(4, 8),
            from_to=("thread", "warp"),
        )
    with pytest.raises(AssertionError):
        # device index out of bound
        layout = T.TileLayout.from_nested_tuple(
            data=(8, T.S(0), 8, T.S(2), 2),
            strides=(16, -1, 2, -1, 1),
            device=(4, 8),
            from_to=("thread", "warp"),
        )
    with pytest.raises(AssertionError):
        # device index out of bound
        layout = T.TileLayout.from_nested_tuple(
            data=(8, T.S(0), 8, T.S(1), 2),
            strides=(16, -1, 2, -1, 1),
            device=(4, 8),
            exclusive=[(2, 0)],
            from_to=("thread", "warp"),
        )

    # defualt stride
    layout = T.TileLayout.from_nested_tuple(data=(8, 4, 3, 5, 7, 2, 4))
    layout_expected = T.TileLayout.from_nested_tuple(
        data=(8, 4, 3, 5, 7, 2, 4),
        strides=(3360, 840, 280, 56, 8, 4, 1),
    )
    assert_structural_equal(layout, layout_expected)


def normalize_tile_layout(layout):
    return T.TileLayout.normalize(layout)


def test_normalize_tile_layout():

    # no unit removal case but normalize subtree
    layout = T.TileLayout.from_nested_tuple(
        data=(8, 8, 8, 4, 2),
        strides=(512, 64, 8, 2, 1),
    )
    layout_expected = T.TileLayout.from_nested_tuple(
        data=4096,
        strides=1,
    )
    assert_structural_equal(layout_expected, normalize_tile_layout(layout))

    # only data tree #1
    layout = T.TileLayout.from_nested_tuple(
        data=(8, 8, 1, 8, 4, 2),
        strides=(512, 64, 160, 8, 2, 1),
    )
    layout_expected = T.TileLayout.from_nested_tuple(
        data=4096,
        strides=1,
    )
    assert_structural_equal(layout_expected, normalize_tile_layout(layout))

    # only data tree #2
    layout = T.TileLayout.from_nested_tuple(
        data=(8, 8, 8, 4, 1, 1),
        strides=(512, 64, 8, 2, 1, 1),
    )
    layout_expected = T.TileLayout.from_nested_tuple(
        data=2048,
        strides=2,
    )
    assert_structural_equal(layout_expected, normalize_tile_layout(layout))

    # only data tree #3
    layout = T.TileLayout.from_nested_tuple(
        data=(8, 8, 1, 1, 1, 4, 1, 1),
        strides=(512, 64, 1, 1, 1, 2, 1, 1),
    )
    layout_expected = T.TileLayout.from_nested_tuple(
        data=(64, 4),
        strides=(64, 2),
    )
    assert_structural_equal(layout_expected, normalize_tile_layout(layout))

    # only data tree #4
    layout = T.TileLayout.from_nested_tuple(
        data=(1, 1, 1, 8, 1, 1, 1),
        strides=(512, 64, 160, 64, 8, 2, 1),
    )
    layout_expected = T.TileLayout.from_nested_tuple(
        data=8,
        strides=64,
    )
    assert_structural_equal(layout_expected, normalize_tile_layout(layout))

    # only data tree #5
    layout = T.TileLayout.from_nested_tuple(
        data=(2, 3, 6),
        strides=(18, 6, 1),
    )
    layout_expected = T.TileLayout.from_nested_tuple(
        data=36,
        strides=1,
    )
    assert_structural_equal(layout_expected, normalize_tile_layout(layout))

    # only data tree #6
    layout = T.TileLayout.from_nested_tuple(
        data=(8, 2, 3, 6),
        strides=(6, 18, 6, 1),
    )
    layout_expected = T.TileLayout.from_nested_tuple(
        data=(8, 36),
        strides=(6, 1),
    )
    assert_structural_equal(layout_expected, normalize_tile_layout(layout))

    # only data tree (partial norm - back) #7
    layout = T.TileLayout.from_nested_tuple(
        data=(8, 2, 3, 6),
        strides=(6, 24, 6, 1),
    )
    layout_expected = T.TileLayout.from_nested_tuple(
        data=(8, 2, 18),
        strides=(6, 24, 1),
    )
    assert_structural_equal(layout_expected, normalize_tile_layout(layout))

    # only data tree (partial norm - front) #8
    layout = T.TileLayout.from_nested_tuple(
        data=(8, 2, 4, 2, 3, 6),
        strides=(2, 1, 4, 24, 6, 1),
    )
    layout_expected = T.TileLayout.from_nested_tuple(
        data=(16, 4, 2, 18),
        strides=(1, 4, 24, 1),
    )
    assert_structural_equal(layout_expected, normalize_tile_layout(layout))

    # only data tree (partial norm - middle) #9
    layout = T.TileLayout.from_nested_tuple(
        data=(3, 4, 5, 2),
        strides=(20, 5, 1, 60),
    )
    layout_expected = T.TileLayout.from_nested_tuple(
        data=(60, 2),
        strides=(1, 60),
    )
    assert_structural_equal(layout_expected, normalize_tile_layout(layout))

    # only data tree (partial norm - middle) #10
    layout = T.TileLayout.from_nested_tuple(
        data=(18, 8, 2, 4, 2, 3, 6),
        strides=(4, 2, 1, 4, 24, 6, 1),
    )
    layout_expected = T.TileLayout.from_nested_tuple(
        data=(18, 16, 4, 2, 18),
        strides=(4, 1, 4, 24, 1),
    )
    assert_structural_equal(layout_expected, normalize_tile_layout(layout))

    # only data tree (partial norm - middle) #11
    layout = T.TileLayout.from_nested_tuple(
        data=(3, 4, 5, 2, 3, 4),
        strides=(20, 5, 1, 60, 20, 5),
    )
    layout_expected = T.TileLayout.from_nested_tuple(
        data=(60, 24),
        strides=(1, 5),
    )
    assert_structural_equal(layout_expected, normalize_tile_layout(layout))

    # no normalization case
    layout_normalized = T.TileLayout.from_nested_tuple(
        data=(8, T.S(0), 8, T.S(1), 2),
        strides=(16, -1, 2, -1, 1),
        device=(8, 4),
        from_to=("thread", "warp"),
    )
    assert_structural_equal(layout_normalized, normalize_tile_layout(layout_normalized))

    # both data and device tree #1
    layout = T.TileLayout.from_nested_tuple(
        data=(8, T.S(0), 8, 1, T.S(1), 2, 1),
        strides=(16, -1, 2, 1, -1, 1, 1),
        device=(8, 4),
        from_to=("thread", "warp"),
    )
    assert_structural_equal(layout_normalized, normalize_tile_layout(layout))

    # both data and device tree #2
    layout = T.TileLayout.from_nested_tuple(
        data=(8, T.S(0), 8, 1, T.S(2), 2, T.S(1)),
        strides=(16, -1, 2, 1, -1, 1, -1),
        device=(8, 1, 4),
        from_to=("thread", "warp"),
    )
    assert_structural_equal(layout_normalized, normalize_tile_layout(layout))

    # both data and device tree (both fused) #3
    layout = T.TileLayout.from_nested_tuple(
        data=(8, T.S(0), 8, 1, T.S(1), 2, 1),
        strides=(16, -1, 2, 1, -1, 1, -1),
        device=(8, 1, 4),
        exclusive=[(2, 0)],
        from_to=("thread", "warp"),
    )
    layout_normalized = T.TileLayout.from_nested_tuple(
        data=(8, T.S(0), 16),
        strides=(16, -1, 1),
        device=(8, 4),
        exclusive=[(1, 0)],
        from_to=("thread", "warp"),
    )
    assert_structural_equal(layout_normalized, normalize_tile_layout(layout))

    # both data and device tree #4
    layout = T.TileLayout.from_nested_tuple(
        data=(T.S(0), T.S(1), 8, 8, 16),
        strides=(-1, -1, 4, 2, 4),
        device=(8, 4),
        exclusive=[],
        from_to=("thread", "warp"),
    )
    layout_normalized = T.TileLayout.from_nested_tuple(
        data=(T.S(0), 8, 8, 16),
        strides=(-1, 4, 2, 4),
        device=(32),
        exclusive=[],
        from_to=("thread", "warp"),
    )
    assert_structural_equal(layout_normalized, normalize_tile_layout(layout))

    # both data and device tree #5
    layout = T.TileLayout.from_nested_tuple(
        data=(T.S(0), T.S(1), 8, 8, 16),
        strides=(-1, -1, 4, 2, 4),
        device=(8, 4),
        exclusive=[],
        from_to=("thread", "warp"),
    )
    layout_normalized = T.TileLayout.from_nested_tuple(
        data=(T.S(0), 8, 8, 16),
        strides=(-1, 4, 2, 4),
        device=(32),
        exclusive=[],
        from_to=("thread", "warp"),
    )
    assert_structural_equal(layout_normalized, normalize_tile_layout(layout))

    # both data and device tree #6
    layout = T.TileLayout.from_nested_tuple(
        data=(T.S(0), T.S(1), 8, 16),
        strides=(-1, -1, 2, 4),
        device=(8, 4),
        exclusive=[],
        from_to=("thread", "warp"),
    )
    layout_normalized = T.TileLayout.from_nested_tuple(
        data=(T.S(0), 8, 16),
        strides=(-1, 2, 4),
        device=(32),
        exclusive=[],
        from_to=("thread", "warp"),
    )
    assert_structural_equal(layout_normalized, normalize_tile_layout(layout))

    # both data and device tree #7
    layout = T.TileLayout.from_nested_tuple(
        data=(T.S(0), T.S(1), 8),
        strides=(-1, -1, 8),
        device=(8, 4),
        exclusive=[],
        from_to=("thread", "warp"),
    )
    layout_normalized = T.TileLayout.from_nested_tuple(
        data=(T.S(0), 8),
        strides=(-1, 8),
        device=(32),
        exclusive=[],
        from_to=("thread", "warp"),
    )
    assert_structural_equal(layout_normalized, normalize_tile_layout(layout))

    # both data and device tree #8 （Fuse-Case 1)
    layout = T.TileLayout.from_nested_tuple(
        data=(T.S(0), T.S(1), 8),
        strides=(-1, -1, 4),
        device=(8, 4),
        exclusive=[],
        from_to=("thread", "warp"),
    )
    layout_normalized = T.TileLayout.from_nested_tuple(
        data=(T.S(0), 8),
        strides=(-1, 4),
        device=(32),
        exclusive=[],
        from_to=("thread", "warp"),
    )
    assert_structural_equal(layout_normalized, normalize_tile_layout(layout))

    # both data and device tree #9 （Fuse-Case 2)
    layout = T.TileLayout.from_nested_tuple(
        data=(T.S(0), T.S(1)),
        strides=(-1, -1),
        device=(8, 4),
        exclusive=[],
        from_to=("thread", "warp"),
    )
    layout_normalized = T.TileLayout.from_nested_tuple(
        data=(T.S(0)),
        strides=(-1),
        device=(32),
        exclusive=[],
        from_to=("thread", "warp"),
    )
    assert_structural_equal(layout_normalized, normalize_tile_layout(layout))

    # both data and device tree #10 （Fuse-Case 3)
    layout = T.TileLayout.from_nested_tuple(
        data=(T.S(0), T.S(1)),
        strides=(-1, -1),
        device=(8, 4),
        exclusive=[],
        from_to=("thread", "warp"),
    )
    layout_normalized = T.TileLayout.from_nested_tuple(
        data=(T.S(0)),
        strides=(-1),
        device=(32),
        exclusive=[],
        from_to=("thread", "warp"),
    )
    assert_structural_equal(layout_normalized, normalize_tile_layout(layout))

    # both data and device tree #11 （Fuse-Case 4)
    layout = T.TileLayout.from_nested_tuple(
        data=(T.S(0), T.S(1), 8),
        strides=(-1, -1, 8),
        device=(8, 4),
        exclusive=[],
        from_to=("thread", "warp"),
    )
    layout_normalized = T.TileLayout.from_nested_tuple(
        data=(T.S(0), 8),
        strides=(-1, 8),
        device=(32),
        exclusive=[],
        from_to=("thread", "warp"),
    )
    assert_structural_equal(layout_normalized, normalize_tile_layout(layout))

    # both data and device tree #12 (Fuse-mixed)
    layout = T.TileLayout.from_nested_tuple(
        data=(T.S(0), T.S(1), 4, 8, 8, 8),
        strides=(-1, -1, 4, 8, 8, 8),
        device=(8, 4),
        exclusive=[],
        from_to=("thread", "warp"),
    )
    layout_normalized = T.TileLayout.from_nested_tuple(
        data=(T.S(0), 4, 8, 8, 8),
        strides=(-1, 4, 8, 8, 8),
        device=(32),
        exclusive=[],
        from_to=("thread", "warp"),
    )
    assert_structural_equal(layout_normalized, normalize_tile_layout(layout))

    # both data and device tree #13 (Fuse-mixed with partial)
    layout = T.TileLayout.from_nested_tuple(
        data=(T.S(0), T.S(1), 4, 8, 8, 8),
        strides=(-1, -1, 16, 2, 8, 8),
        device=(8, 4),
        exclusive=[],
        from_to=("thread", "warp"),
    )
    layout_normalized = T.TileLayout.from_nested_tuple(
        data=(T.S(0), 32, 8, 8),
        strides=(-1, 2, 8, 8),
        device=(32),
        exclusive=[],
        from_to=("thread", "warp"),
    )
    assert_structural_equal(layout_normalized, normalize_tile_layout(layout))

    # both data and device tree #14 (Fuse-mixed with partial)
    layout = T.TileLayout.from_nested_tuple(
        data=(T.S(0), T.S(1), 4, 8, 8, 4, 4, 16, 8),
        strides=(-1, -1, 16, 2, 8, 2, 16, 1, 4),
        device=(8, 4),
        exclusive=[],
        from_to=("thread", "warp"),
    )
    layout_normalized = T.TileLayout.from_nested_tuple(
        data=(T.S(0), 32, 32, 64, 8),
        strides=(-1, 2, 2, 1, 4),
        device=(32),
        exclusive=[],
        from_to=("thread", "warp"),
    )
    assert_structural_equal(layout_normalized, normalize_tile_layout(layout))

    # only data tree (partial norm - middle) #15
    layout = T.TileLayout.from_nested_tuple(
        data=(T.S(0), 3, 4, 5, 2, 3, 4),
        strides=(-1, 20, 5, 1, 60, 20, 5),
        device=(8),
        exclusive=[],
        from_to=("thread", "warp"),
    )
    layout_expected = T.TileLayout.from_nested_tuple(
        data=(T.S(0), 60, 24),
        strides=(-1, 1, 5),
        device=(8),
        exclusive=[],
        from_to=("thread", "warp"),
    )
    assert_structural_equal(layout_expected, normalize_tile_layout(layout))

    # unit layout case#1
    layout = T.TileLayout.from_nested_tuple(data=(1, 1, 1, 1, 1), strides=(1, 1, 1, 1, 1))
    layout_unit = T.TileLayout.from_nested_tuple(data=1, strides=1)
    assert_structural_equal(layout_unit, normalize_tile_layout(layout))

    # unit layout case#2
    layout = T.TileLayout.from_nested_tuple(
        data=(1, T.S(0), T.S(1), 1, 1),
        strides=(1, -1, -1, 1, 1),
        device=(1, 1),
        from_to=("thread", "warp"),
    )
    layout_unit = T.TileLayout.from_nested_tuple(
        data=1, strides=1, device=1, from_to=("thread", "warp")
    )
    assert_structural_equal(layout_unit, normalize_tile_layout(layout))

    # idempotent unit layout case#3
    layout_unit = T.TileLayout.from_nested_tuple(
        data=1, strides=1, device=1, from_to=("thread", "warp")
    )
    assert_structural_equal(layout_unit, normalize_tile_layout(normalize_tile_layout(layout_unit)))


def test_tile_layout():

    layout = T.TileLayout.from_nested_tuple(data=8, strides=1)
    layout_tile = T.TileLayout.from_nested_tuple(data=(8, 8), strides=(8, 1))
    assert_structural_equal(layout_tile, T.TileLayout.tile(layout, layout, [8], [8]))
    assert T.TileLayout.is_tile_inner(
        layout_tile, layout, [64], [8]
    ), "The layout is used for tiling layout_tile."
    assert T.TileLayout.is_tile_outer(
        layout_tile, layout, [64], [8]
    ), "The layout_tile is a tiling for layout."

    layout = T.TileLayout.from_nested_tuple(data=(8, 8), strides=(8, 1))
    layout_tile = T.TileLayout.from_nested_tuple(data=(8, 8, 8, 8), strides=(512, 8, 64, 1))
    assert_structural_equal(layout_tile, T.TileLayout.tile(layout, layout, [8, 8], [8, 8]))
    assert T.TileLayout.is_tile_inner(
        layout_tile, layout, [64, 64], [8, 8]
    ), "The layout is used for tiling layout_tile."
    assert T.TileLayout.is_tile_outer(
        layout_tile, layout, [64, 64], [8, 8]
    ), "The layout_tile is a tiling for layout."

    layout2 = T.TileLayout.from_nested_tuple(data=(2, 4), strides=(1, 2))
    layout_tile = T.TileLayout.from_nested_tuple(data=(8, 2, 8, 4), strides=(64, 1, 8, 2))
    assert_structural_equal(layout_tile, T.TileLayout.tile(layout, layout2, [8, 8], [2, 4]))
    assert T.TileLayout.is_tile_inner(
        layout_tile, layout2, [16, 32], [2, 4]
    ), "The layout2 is used for tiling layout_tile."
    assert T.TileLayout.is_tile_outer(
        layout_tile, layout, [16, 32], [8, 8]
    ), "The layout_tile is a tiling for layout."
    assert not T.TileLayout.is_tile_inner(
        layout_tile, layout, [16, 32], [8, 8]
    ), "The layout2 is used for tiling layout_tile."
    assert not T.TileLayout.is_tile_outer(
        layout_tile, layout2, [16, 32], [2, 4]
    ), "The layout_tile is a tiling for layout."

    layout3 = T.TileLayout.from_nested_tuple(data=(4, 2, 2, 4), strides=(16, 8, 1, 2))
    layout_tile = T.TileLayout.from_nested_tuple(
        data=(8, 4, 2, 8, 2, 4), strides=(512, 16, 8, 64, 1, 2)
    )
    assert_structural_equal(
        normalize_tile_layout(layout_tile), T.TileLayout.tile(layout, layout3, (8, 8), (8, 8))
    )
    assert T.TileLayout.is_tile_inner(
        layout_tile, layout3, (64, 64), (8, 8)
    ), "The layout3 is used for tiling layout_tile."
    assert T.TileLayout.is_tile_outer(
        layout_tile, layout, (64, 64), (8, 8)
    ), "The layout_tile is a tiling for layout."
    assert not T.TileLayout.is_tile_inner(
        layout_tile, layout, (64, 64), (8, 8)
    ), "The layout3 is not used for tiling layout_tile."
    assert not T.TileLayout.is_tile_outer(
        layout_tile, layout3, (64, 64), (8, 8)
    ), "The layout_tile is not a tiling for layout."

    # Tile over a sharded layout - 1
    layout = T.TileLayout.from_nested_tuple(
        data=(T.S(0), 1, T.S(1), 2),
        strides=(-1, 2, -1, 1),
        device=(8, 4),
        from_to=("thread", "warp"),
    )
    layout_tile = T.TileLayout.tile(
        outer=T.TileLayout.from_nested_tuple(data=(8, 8), strides=(8, 1)),
        inner=layout,
        outer_shape=(8, 8),
        inner_shape=(8, 8),
    )
    layout_expected = T.TileLayout.from_nested_tuple(
        data=(8, T.S(0), 1, 8, T.S(1), 2),
        strides=(16, -1, 2, 2, -1, 1),
        device=(8, 4),
        from_to=("thread", "warp"),
    )
    assert_structural_equal(normalize_tile_layout(layout_expected), layout_tile)
    assert T.TileLayout.is_tile_inner(
        layout_tile, layout, (64, 64), (8, 8)
    ), "The layout is used for tiling layout_tile."
    assert T.TileLayout.is_tile_outer(
        layout_tile, T.TileLayout.from_nested_tuple(data=(8, 8), strides=(8, 1)), (64, 64), (8, 8)
    ), "The layout is used for tiling layout_tile."
    assert not T.TileLayout.is_tile_inner(
        layout_tile, T.TileLayout.from_nested_tuple(data=(8, 8), strides=(8, 1)), (64, 64), (8, 8)
    ), "The layout is not used for tiling layout_tile."
    assert not T.TileLayout.is_tile_outer(
        layout_tile, layout, (64, 64), (8, 8)
    ), "The layout is not used for tiling layout_tile."

    # Tile over a sharded layout - 2
    layout = T.TileLayout.from_nested_tuple(
        data=(T.S(0), T.S(1)),
        strides=(-1, -1),
        device=(8, 4),
        from_to=("thread", "warp"),
    )
    layout_tile = T.TileLayout.tile(
        outer=T.TileLayout.from_nested_tuple(data=(8, 8), strides=(8, 1)),
        inner=layout,
        outer_shape=(8, 8),
        inner_shape=(8, 4),
    )
    layout_expected = T.TileLayout.from_nested_tuple(
        data=(8, T.S(0), 8, T.S(1)),
        strides=(8, -1, 1, -1),
        device=(8, 4),
        from_to=("thread", "warp"),
    )
    assert_structural_equal(layout_expected, layout_tile)
    assert T.TileLayout.is_tile_inner(
        layout_tile, layout, (64, 32), (8, 4)
    ), "The layout is used for tiling layout_tile."
    assert T.TileLayout.is_tile_outer(
        layout_tile, T.TileLayout.from_nested_tuple(data=(8, 8), strides=(8, 1)), (64, 32), (8, 8)
    ), "The layout is used for tiling layout_tile."
    assert not T.TileLayout.is_tile_inner(
        layout_tile, T.TileLayout.from_nested_tuple(data=(8, 8), strides=(8, 1)), (64, 32), (8, 8)
    ), "The layout is not used for tiling layout_tile."
    assert not T.TileLayout.is_tile_outer(
        layout_tile, layout, (64, 32), (8, 4)
    ), "The layout is not used for tiling layout_tile."

    # Tile over a complicated sharded layout - 3
    layout = T.TileLayout.from_nested_tuple(
        data=(T.S(0), 1, T.S(1), 2),
        strides=(-1, 2, -1, 1),
        device=(8, 4),
        from_to=("thread", "warp"),
    )

    outer = T.TileLayout.from_nested_tuple(data=(8, 2, 8, 4), strides=(8, 4, 2, 1))
    layout_tile = T.TileLayout.tile(
        outer=outer, inner=layout, outer_shape=(16, 32), inner_shape=(8, 8)
    )
    layout_expected = T.TileLayout.from_nested_tuple(
        data=(16, T.S(0), 1, 8, 4, T.S(1), 2),
        strides=(8, -1, 2, 4, 2, -1, 1),
        device=(8, 4),
        from_to=("thread", "warp"),
    )
    assert_structural_equal(normalize_tile_layout(layout_expected), layout_tile)
    assert T.TileLayout.is_tile_inner(
        layout_tile, layout, (128, 256), (8, 8)
    ), "The layout is used for tiling layout_tile."
    assert T.TileLayout.is_tile_outer(
        layout_tile, outer, (128, 256), (16, 32)
    ), "The layout is used for tiling layout_tile."
    assert not T.TileLayout.is_tile_inner(
        layout_tile, outer, (128, 256), (16, 32)
    ), "The layout is not used for tiling layout_tile."
    assert not T.TileLayout.is_tile_outer(
        layout_tile, layout, (128, 256), (8, 8)
    ), "The layout is not used for tiling layout_tile."

    # Normalized Tile Layout Test - 4 (tile < inner)
    outer = T.TileLayout.from_nested_tuple(data=(4, 2, 1), strides=(2, 1, 1))
    inner = T.TileLayout.from_nested_tuple(data=(2, 4, 1), strides=(2, 3, 1))
    layout_tile = T.TileLayout.tile(
        outer,
        inner,
        outer_shape=(4, 2),
        inner_shape=(2, 4),
    )

    assert T.TileLayout.is_tile_inner(
        layout_tile, inner, (8, 8), (2, 4)
    ), "The inner is used for tiling layout_tile."
    assert T.TileLayout.is_tile_inner(
        normalize_tile_layout(layout_tile), inner, (8, 8), (2, 4)
    ), "The inner is used for tiling layout_tile."
    assert not T.TileLayout.is_tile_inner(
        normalize_tile_layout(layout_tile), outer, (8, 8), (4, 2)
    ), "The outer is not used for tiling layout_tile as inner."

    # Normalized Tile Layout Test - 5 (tile = inner)
    outer = T.TileLayout.from_nested_tuple(data=(8, 2), strides=(2, 1))
    inner = T.TileLayout.from_nested_tuple(data=(2, 4), strides=(4, 1))
    layout_tile = T.TileLayout.tile(outer, inner, (8, 2), (2, 4))
    assert T.TileLayout.is_tile_inner(
        layout_tile, inner, (16, 8), (2, 4)
    ), "The inner is used for tiling layout_tile."
    assert T.TileLayout.is_tile_inner(
        normalize_tile_layout(layout_tile), inner, (16, 8), (2, 4)
    ), "The inner is used for tiling layout_tile."

    # Normalized Tile Layout Test - 6 (tile < inner)
    outer = T.TileLayout.from_nested_tuple(data=(8, 4, 1), strides=(4, 1, 4))
    inner = T.TileLayout.from_nested_tuple(data=(2, 1, 1), strides=(4, 3, 1))
    inner_tmp = T.TileLayout.from_nested_tuple(data=(8, 2, 2), strides=(4, 2, 2))
    layout_tile = T.TileLayout.tile(outer, inner, (8, 4), (2, 1))
    assert T.TileLayout.is_tile_inner(
        layout_tile, inner, (16, 4), (2, 1)
    ), "The inner is used for tiling layout_tile."
    assert T.TileLayout.is_tile_inner(
        normalize_tile_layout(layout_tile), inner, (16, 4), (2, 1)
    ), "The inner is used for tiling layout_tile."
    assert not T.TileLayout.is_tile_inner(
        normalize_tile_layout(layout_tile), normalize_tile_layout(outer), (16, 4), (8, 4)
    ), "The layout is not used for tiling layout_tile as inner."
    assert not T.TileLayout.is_tile_inner(
        normalize_tile_layout(layout_tile), inner_tmp, (16, 4), (8, 2, 2)
    ), "The layout is not used for tiling layout_tile as inner."

    # Normalized Tile Layout Test - 7 (tile = inner)
    outer = T.TileLayout.from_nested_tuple(data=(8, 8, 4), strides=(32, 4, 1))
    inner = T.TileLayout.from_nested_tuple(data=(1, 2, 1), strides=(4, 3, 1))
    inner_tmp = T.TileLayout.from_nested_tuple(data=(1, 2, 2), strides=(8, 4, 3))
    layout_tile = T.TileLayout.tile(outer, inner, outer_shape=(8, 8, 4), inner_shape=(1, 2, 1))
    assert T.TileLayout.is_tile_inner(
        layout_tile, inner, (8, 16, 4), (1, 2, 1)
    ), "The inner is used for tiling layout_tile."
    assert T.TileLayout.is_tile_inner(
        normalize_tile_layout(layout_tile), inner, (8, 16, 4), (1, 2, 1)
    ), "The inner is used for tiling layout_tile."
    assert not T.TileLayout.is_tile_inner(
        normalize_tile_layout(layout_tile), normalize_tile_layout(outer), (8, 16, 4), (8, 8, 4)
    ), "The layout is not used for tiling layout_tile as inner."
    assert not T.TileLayout.is_tile_inner(
        normalize_tile_layout(layout_tile), inner_tmp, (8, 16, 4), (1, 2, 2)
    ), "The layout is not used for tiling layout_tile as inner."

    # Normalized Tile Layout Test - 8 (tile = inner w/ device)
    outer = T.TileLayout.from_nested_tuple(data=(8, 8, 4), strides=(32, 4, 1))
    inner = T.TileLayout.from_nested_tuple(
        data=(8, T.S(0), 1, T.S(1), 2),
        strides=(4, -1, 2, -1, 1),
        device=(8, 4),
        from_to=("thread", "warp"),
    )
    layout_tile = T.TileLayout.tile(outer, inner, outer_shape=(8, 8, 4), inner_shape=(8, 8, 8))
    assert T.TileLayout.is_tile_inner(
        layout_tile, inner, (64, 64, 32), (8, 8, 8)
    ), "The inner is used for tiling layout_tile."
    assert T.TileLayout.is_tile_inner(
        normalize_tile_layout(layout_tile), inner, (64, 64, 32), (8, 8, 8)
    ), "The inner is used for tiling layout_tile."
    assert not T.TileLayout.is_tile_inner(
        normalize_tile_layout(layout_tile), normalize_tile_layout(outer), (64, 64, 32), (8, 8, 4)
    ), "The layout is not used for tiling layout_tile as inner."

    # Normalized Tile Layout Test - 9 (tile = inner w/ device + diff major-dim)
    outer = T.TileLayout.from_nested_tuple(data=(16, 8, 4), strides=(1, 64, 16))
    inner = T.TileLayout.from_nested_tuple(data=(2, 4, 2, 2), strides=(4, 1, 4, 3))
    inner_tmp = T.TileLayout.from_nested_tuple(data=(1, 2, 2), strides=(8, 4, 3))
    layout_tile = T.TileLayout.tile(outer, inner, outer_shape=(16, 8, 4), inner_shape=(8, 2, 2))
    assert T.TileLayout.is_tile_inner(
        layout_tile, inner, (128, 16, 8), (8, 2, 2)
    ), "The inner is used for tiling layout_tile."
    assert T.TileLayout.is_tile_inner(
        normalize_tile_layout(layout_tile), inner, (128, 16, 8), (8, 2, 2)
    ), "The inner is used for tiling layout_tile."
    assert not T.TileLayout.is_tile_inner(
        normalize_tile_layout(layout_tile), normalize_tile_layout(outer), (128, 16, 8), (16, 8, 4)
    ), "The layout is not used for tiling layout_tile as inner."
    assert T.TileLayout.is_tile_inner(
        normalize_tile_layout(layout_tile), inner_tmp, (128, 16, 8), (1, 2, 2)
    ), "The layout is used for tiling layout_tile as inner."

    with pytest.raises(Exception):
        # dims mismatch
        layout = T.TileLayout.from_nested_tuple(data=8, strides=1)
        layout2 = T.TileLayout.from_nested_tuple(data=(2, 4), strides=(1, 2))
        T.TileLayout.tile(layout, layout2, [8], [2, 4])

    with pytest.raises(Exception):
        # outer must not have device tree
        layout = T.TileLayout.from_nested_tuple(
            data=(T.S(0), 1, T.S(1), 2),
            strides=(-1, 2, -1, 1),
            device=(8, 4),
            from_to=("thread", "warp"),
        )
        T.TileLayout.tile(
            outer=layout,
            inner=T.TileLayout.from_nested_tuple(data=(8, 8), strides=(8, 1)),
            outer_shape=(8, 8),
            inner_shape=(8, 8),
        )


def test_vec_len_layout():

    ### only data tree
    layout_rm = T.TileLayout.from_nested_tuple(data=(8, 4), strides=(4, 1))
    layout_rm_2 = T.TileLayout.from_nested_tuple(data=(8, 2, 2), strides=(4, 2, 1))
    layout_rm_4 = T.TileLayout.from_nested_tuple(data=(8, 1, 4), strides=(4, 4, 1))
    layout_rm_double = T.TileLayout.from_nested_tuple(data=(4, 8), strides=(8, 1))
    layout_rm_3D_unit = T.TileLayout.from_nested_tuple(data=(8, 1, 4), strides=(4, 1, 1))
    layout_rm_3D_2 = T.TileLayout.from_nested_tuple(data=(8, 2, 2), strides=(4, 2, 1))
    layout_rm_3D_4 = T.TileLayout.from_nested_tuple(data=(8, 4, 1), strides=(4, 1, 1))

    layout_cm = T.TileLayout.from_nested_tuple(data=(8, 4), strides=(1, 8))
    layout_cm_2 = T.TileLayout.from_nested_tuple(data=(4, 2, 4), strides=(2, 1, 8))
    layout_cm_4 = T.TileLayout.from_nested_tuple(data=(2, 4, 4), strides=(4, 1, 8))
    layout_cm_half = T.TileLayout.from_nested_tuple(data=(4, 8), strides=(1, 4))
    layout_cm_3D_unit = T.TileLayout.from_nested_tuple(data=(8, 1, 4), strides=(1, 1, 8))
    layout_cm_3D_2 = T.TileLayout.from_nested_tuple(data=(4, 2, 4), strides=(2, 1, 8))
    layout_cm_3D_4 = T.TileLayout.from_nested_tuple(data=(2, 4, 4), strides=(4, 1, 8))

    # different dim-majors
    # vec_len(layout_rm, layout_cm) = 1
    assert T.TileLayout.find_optimal_vec_len(layout_rm, layout_cm) == 1
    assert T.TileLayout.find_optimal_vec_len(layout_rm_2, layout_cm) == 1
    assert T.TileLayout.find_optimal_vec_len(layout_rm_4, layout_cm) == 1
    assert T.TileLayout.find_optimal_vec_len(layout_rm, layout_cm_2) == 1
    assert T.TileLayout.find_optimal_vec_len(layout_rm_2, layout_cm_2) == 1
    assert T.TileLayout.find_optimal_vec_len(layout_rm_4, layout_cm_2) == 1
    assert T.TileLayout.find_optimal_vec_len(layout_rm, layout_cm_4) == 1
    assert T.TileLayout.find_optimal_vec_len(layout_rm_2, layout_cm_4) == 1
    assert T.TileLayout.find_optimal_vec_len(layout_rm_4, layout_cm_4) == 1
    assert T.TileLayout.find_optimal_vec_len(layout_rm, layout_cm_3D_unit) == 1
    assert T.TileLayout.find_optimal_vec_len(layout_rm_3D_unit, layout_cm) == 1

    # row-major - 2D
    assert T.TileLayout.find_optimal_vec_len(layout_rm, layout_rm_2) == 2
    assert T.TileLayout.find_optimal_vec_len(layout_rm, layout_rm_4) == 4
    assert T.TileLayout.find_optimal_vec_len(layout_rm, layout_rm_double) == 4
    assert T.TileLayout.find_optimal_vec_len(layout_rm_2, layout_rm_4) == 2

    # col-major - 2D
    assert T.TileLayout.find_optimal_vec_len(layout_cm, layout_cm_2) == 2
    assert T.TileLayout.find_optimal_vec_len(layout_cm, layout_cm_4) == 4
    assert T.TileLayout.find_optimal_vec_len(layout_cm, layout_cm_half) == 4
    assert T.TileLayout.find_optimal_vec_len(layout_cm_2, layout_cm_4) == 2

    ### 3D tests
    layout_3D_1 = T.TileLayout.from_nested_tuple(data=(8, 2, 4), strides=(1, 8, 16))
    layout_3D_2 = T.TileLayout.from_nested_tuple(data=(4, 2, 8), strides=(1, 4, 8))
    layout_3D_3 = T.TileLayout.from_nested_tuple(data=(16, 2, 4), strides=(1, 16, 32))
    layout_3D_4 = T.TileLayout.from_nested_tuple(data=(16, 2, 8), strides=(1, 16, 128))
    layout_3D_1_twice_coeff = T.TileLayout.from_nested_tuple(data=(8, 2, 4), strides=(2, 16, 32))

    assert T.TileLayout.find_optimal_vec_len(layout_3D_1, layout_3D_2) == 4
    assert T.TileLayout.find_optimal_vec_len(layout_3D_1, layout_3D_1_twice_coeff) == 1

    # row-major - 3D
    assert T.TileLayout.find_optimal_vec_len(layout_rm, layout_rm_3D_unit) == 4
    assert T.TileLayout.find_optimal_vec_len(layout_rm, layout_rm_3D_2) == 2
    assert T.TileLayout.find_optimal_vec_len(layout_rm, layout_rm_3D_4) == 4

    # col-major - 3D
    assert T.TileLayout.find_optimal_vec_len(layout_cm, layout_cm_3D_unit) == 8
    assert T.TileLayout.find_optimal_vec_len(layout_cm, layout_cm_3D_2) == 2
    assert T.TileLayout.find_optimal_vec_len(layout_cm, layout_cm_3D_4) == 4

    # structure not match
    with pytest.raises(Exception):
        assert T.TileLayout.find_optimal_vec_len(layout_3D_3, layout_3D_4) == 1
        assert T.TileLayout.find_optimal_vec_len(layout_3D_1, layout_3D_4) == 1
        assert T.TileLayout.find_optimal_vec_len(layout_3D_2, layout_3D_4) == 1


def test_shard_layout():
    # mma layout test
    layout = T.TileLayout.from_nested_tuple(data=(1, 2), strides=(2, 1))
    layout_warp = T.TileLayout.shard(
        shape=(8, 8), mesh=(8, 4), strategy="S0S1", inner=layout, from_to=("thread", "warp")
    )
    layout_expected = T.TileLayout.from_nested_tuple(
        data=(T.S(0), 1, T.S(1), 2),
        strides=(-1, 2, -1, 1),
        device=(8, 4),
        from_to=("thread", "warp"),
    )
    assert_structural_equal(layout_expected, layout_warp)

    layout_cta = T.TileLayout.shard(
        shape=(16, 16), mesh=(2, 2), strategy="S0S1", inner=layout_warp, from_to=("warp", "cta")
    )
    layout_expected = T.TileLayout.from_nested_tuple(
        data=(T.S(0), T.S(2), 1, T.S(1), T.S(3), 2),
        strides=(-1, -1, 2, -1, -1, 1),
        device=(2, 2, 8, 4),
        from_to=("thread", "cta"),
    )
    assert_structural_equal(normalize_tile_layout(layout_expected), layout_cta)

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
        data=(T.S(0), 1, 2),
        strides=(-1, 2, 1),
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
        data=(T.S(0), 64, 128),
        strides=(-1, 128, 1),
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


def test_size_cosize():
    ###################################################################### size
    # TileLayout
    layout = T.TileLayout.from_nested_tuple(data=(8, 8), strides=(8, 1))
    assert layout.size == 64

    # SwizzleLayout
    layout = T.SwizzleLayout(per_element=3, swizzle_len=3, atom_len=3)
    assert layout.size == 512
    layout = T.SwizzleLayout(per_element=4, swizzle_len=3, atom_len=3)
    assert layout.size == 1024

    ###################################################################### cosize
    # TileLayout
    layout = T.TileLayout.from_nested_tuple(data=(8, 8), strides=(8, 1))
    assert layout.cosize == 64
    layout = T.TileLayout.from_nested_tuple(data=(8, 6), strides=(8, 1))
    assert layout.cosize == 62
    layout = T.TileLayout.from_nested_tuple(
        data=(T.S(0), 1, T.S(1), 2),
        strides=(-1, 2, -1, 1),
        device=(8, 4),
        from_to=("thread", "warp"),
    )
    assert layout.cosize == 2

    # SwizzleLayout
    layout = T.SwizzleLayout(per_element=3, swizzle_len=3, atom_len=3)
    assert layout.cosize == 512
    layout = T.SwizzleLayout(per_element=4, swizzle_len=3, atom_len=3)
    assert layout.cosize == 1024


def test_apply():
    ################ TileLayout
    def test0():
        layout = T.TileLayout.from_nested_tuple(data=(8, 8), strides=(8, 1))
        for i, j in itertools.product(range(8), range(8)):
            assert layout.apply(i * 8 + j) == i * 8 + j * 1
        for i, j in itertools.product(range(8), range(8)):
            assert layout.apply(i, j, shape=(8, 8)) == i * 8 + j * 1
        # apply can accept coord larger than size
        for p in range(1024):
            outer = p // 64
            inner = p % 64
            i, j = inner // 8, inner % 8
            assert layout.apply(p) == outer * 64 + i * 8 + j * 1
        with pytest.raises(Exception):
            layout.apply(1, 1, 1)

    def test1():
        layout = T.TileLayout.from_nested_tuple(data=(8, 8), strides=(10, 1))
        for i, j in itertools.product(range(8), range(8)):
            assert layout.apply(i * 8 + j) == i * 10 + j * 1
        for i, j in itertools.product(range(8), range(8)):
            assert layout.apply(i, j, shape=(8, 8)) == i * 10 + j * 1

        # apply can accept coord larger than size
        for p in range(1024):
            outer = p // 64
            inner = p % 64
            i, j = inner // 8, inner % 8
            assert (
                layout.apply(
                    p,
                )
                == outer * 78 + i * 10 + j * 1
            )

    def test2():
        layout = T.TileLayout.from_nested_tuple(data=(2, 3, 4, 2, 2), strides=(1, 2, 12, 6, 48))

        def f(i0, i1):
            leaf1 = i0 // 3
            leaf2 = i0 % 3
            leaf3 = i1 // 4
            leaf4 = (i1 % 4) // 2
            leaf5 = i1 % 2
            assert (
                layout.apply(i0, i1, shape=(6, 16))
                == leaf1 * 1 + leaf2 * 2 + leaf3 * 12 + leaf4 * 6 + leaf5 * 48
            )

        for i0, i1 in itertools.product(range(6), range(16)):
            f(i0, i1)
        for i in range(6 * 16):
            f(i // 16, i % 16)

    def test3():
        layout = T.TileLayout.from_nested_tuple(
            data=(T.S(0), 1, T.S(1), 2),
            strides=(-1, 2, -1, 1),
            device=(8, 4),
            from_to=("thread", "warp"),
        )
        for i0, i1 in itertools.product(range(8), range(8)):
            assert layout.apply(i0, i1, shape=(8, 8)) == i1 % 2

    ################ Swizzle Layout
    def test4():
        layout = T.SwizzleLayout(per_element=0, swizzle_len=3, atom_len=3)
        assert layout.size == 64
        for i, j in itertools.product(range(8), range(8)):
            assert layout.apply(i * 8 + j) == i * 8 + i ^ j

    def test5():
        layout = T.SwizzleLayout(per_element=3, swizzle_len=3, atom_len=3)
        assert layout.size == 512
        for i, j, k in itertools.product(range(8), range(8), range(8)):
            assert layout.apply((i * 8 + j) * 8 + k) == (i * 8 + (i ^ j)) * 8 + k
        # apply can accept coord larger than size
        for p in range(4096):
            outer = p // 512
            inner = p % 512
            i, j, k = inner // 64, (inner % 64) // 8, inner % 8
            assert layout.apply(p) == outer * 512 + (i * 8 + (i ^ j)) * 8 + k

    def test6():
        layout = T.SwizzleLayout(per_element=0, swizzle_len=3, atom_len=3, swizzle_inner=False)
        assert layout.size == 64
        for i, j in itertools.product(range(8), range(8)):
            assert layout.apply(i * 8 + j) == (i ^ j) * 8 + j

    test0()
    test1()
    test2()
    test3()
    test4()
    test5()
    test6()


if __name__ == "__main__":
    test_constructor_nested_tuple_no_device()
    test_constructor_nested_tuple()
    test_normalize_tile_layout()
    test_tile_layout()
    test_shard_layout()
    test_size_cosize()
    test_apply()
