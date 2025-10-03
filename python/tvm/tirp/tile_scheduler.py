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
"""Reusable tile scheduler helpers for TIR tests/kernels.

These classes emit TIR via @T.macro. Construct them with T.meta_var(...)
inside a PrimFunc and call their macros to advance scheduling state.
"""

import warnings
import tvm
from tvm.script import tir as T


class BaseTileScheduler:
    """Base class for tile schedulers with common state and macros."""

    def __init__(self, prefix: str):
        self.m_idx = T.local_cell("int32", name=prefix + "_m_idx")
        self.n_idx = T.local_cell("int32", name=prefix + "_n_idx")
        self.linear_idx = T.local_cell("int32", name=prefix + "_linear_idx")

    @T.macro
    def update_current_m_n_idx(self, linear_idx):
        # To be implemented by subclasses
        pass

    @T.macro
    def init(self, linear_init):
        self.linear_idx = linear_init
        self.update_current_m_n_idx(linear_init)

    @T.macro
    def next_tile(self, step):
        self.linear_idx = self.linear_idx + step
        self.update_current_m_n_idx(self.linear_idx)

    def valid(self, total_tiles):
        return self.linear_idx < total_tiles


class GroupMajor2D(BaseTileScheduler):
    """
    Unified 2D group-major scheduler with optional tail and hierarchical inner strides.

    prefix: str
    m_tiles: int | T.ExprLike   # tiles along M (static int or runtime expr)
    n_tiles: int
    group_rows: int
    step: int = 1
    inner_m: int = 1
    inner_n: int = 1
    """

    def __init__(
        self,
        prefix: str,
        m_tiles,
        n_tiles: int,
        group_rows: int,
        step: int = 1,
        inner_m: int = 1,
        inner_n: int = 1,
    ):
        super().__init__(prefix)
        self._m_tiles = m_tiles
        self._n_tiles = n_tiles
        self._group_rows = group_rows
        self._step = step
        self._inner_m = inner_m
        self._inner_n = inner_n
        self.tile_idx = T.local_cell("int32", name=prefix + "_tile_idx")

        is_static_m = isinstance(m_tiles, int)

        tile_cols_int = (n_tiles + inner_n - 1) // inner_n
        self._TILE_COLS = tile_cols_int

        if is_static_m:
            self._TILE_ROWS = (m_tiles + inner_m - 1) // inner_m
            self._FULL_GROUPS = self._TILE_ROWS // group_rows
        else:
            # dynamic expressions kept as TIR nodes
            self._TILE_ROWS = T.truncdiv(self._m_tiles + self._inner_m - 1, self._inner_m)
            self._FULL_GROUPS = T.truncdiv(self._TILE_ROWS, self._group_rows)

        self._FINAL_ROWS = self._TILE_ROWS - self._FULL_GROUPS * self._group_rows
        self._TOTAL = self._TILE_ROWS * tile_cols_int * inner_m * inner_n

    # fmt: off
    @T.macro
    def update_current_m_n_idx(self, linear_idx):
        # peel hierarchical inner coords first
        INNER_M = T.meta_var(self._inner_m)
        INNER_N = T.meta_var(self._inner_n)

        inner_m = T.meta_var(linear_idx % INNER_M)
        t = T.meta_var(linear_idx // INNER_M)
        inner_n = T.meta_var(t % INNER_N)
        tile_linear = T.meta_var(t // INNER_N)

        # use prebuilt expressions (static or dynamic)
        TILE_COLS = T.meta_var(self._TILE_COLS)
        FULL_GROUPS = T.meta_var(self._FULL_GROUPS)
        FINAL_ROWS = T.meta_var(self._FINAL_ROWS)
        GROUP_SPAN = T.meta_var(self._group_rows * self._TILE_COLS)

        @T.macro
        def update(tile_row, tile_col):
            self.m_idx = tile_row * INNER_M + inner_m
            self.n_idx = tile_col * INNER_N + inner_n

        # group-major over tile grid with tail rows
        if (FULL_GROUPS > 0) & (tile_linear < FULL_GROUPS * GROUP_SPAN):
            tile_row = (tile_linear // GROUP_SPAN) * self._group_rows + (tile_linear % self._group_rows)
            tile_col = (tile_linear // self._group_rows) % TILE_COLS
            update(tile_row, tile_col)
        elif FINAL_ROWS > 0:
            rem = tile_linear - FULL_GROUPS * GROUP_SPAN
            tile_row = FULL_GROUPS * self._group_rows + (rem % FINAL_ROWS)
            tile_col = rem // FINAL_ROWS
            update(tile_row, tile_col)
        else:
            tile_row = 0
            tile_col = 0
            update(tile_row, tile_col)

    @T.macro
    def init(self, linear_init):
        self.linear_idx = linear_init
        self.tile_idx = 0
        self.update_current_m_n_idx(linear_init)

    @T.macro
    def next_tile(self):
        self.linear_idx = self.linear_idx + self._step
        self.tile_idx = self.tile_idx + 1
        self.update_current_m_n_idx(self.linear_idx)

    @T.macro
    def next_tile_stride(self, stride: int):
        self.linear_idx = self.linear_idx + stride
        self.tile_idx = self.tile_idx + 1
        self.update_current_m_n_idx(self.linear_idx)
    # fmt: on

    def valid(self):
        return self.linear_idx < self._TOTAL


class GroupMajor3D(BaseTileScheduler):
    """
    3D grouped-row scheduler (M,N,K) with tail handling on M.

    Args
    ----
    prefix: str
    m_tiles: int | T PrimExpr   # tiles along M (static or runtime)
    n_tiles: int                # tiles along N (static)
    k_tiles: int                # tiles along K (static)
    group_rows: int             # rows per group along M
    step: int = 1               # default stride for next_tile()
    """

    def __init__(
        self,
        prefix: str,
        m_tiles,
        n_tiles: int,
        k_tiles: int,
        group_rows: int,
        step: int = 1,
    ):
        super().__init__(prefix)
        self._step = step
        self.tile_idx = T.local_cell("int32", name=prefix + "_tile_idx")
        self.k_idx = T.local_cell("int32", name=prefix + "_k_idx")

        # ---- constants / primexprs baked once ----
        self._G = group_rows
        self._N = n_tiles
        self._K = k_tiles

        if isinstance(m_tiles, int):
            self._GROUPS = m_tiles // group_rows
            self._FINAL_ROWS = m_tiles - self._GROUPS * group_rows
            self._GROUP_SIZE = group_rows * n_tiles * k_tiles
            self._TOTAL = m_tiles * n_tiles * k_tiles
        else:
            self._GROUPS = T.truncdiv(m_tiles, group_rows)
            self._FINAL_ROWS = m_tiles - self._GROUPS * group_rows
            self._GROUP_SIZE = self._G * self._N * self._K
            self._TOTAL = m_tiles * n_tiles * k_tiles

        # handy composites used in macro
        self._FULL_BOUND = self._GROUPS * self._GROUP_SIZE
        self._HAS_FULL = self._GROUPS > 0
        self._HAS_TAIL = self._FINAL_ROWS > 0

    # fmt: off
    @T.macro
    def update_current_m_n_idx(self, linear_idx):
        # full-group formulas
        full_m = T.floordiv(linear_idx, self._GROUP_SIZE) * self._G + T.floormod(
            linear_idx, self._G
        )
        full_n = T.floormod(T.floordiv(linear_idx, self._G), self._N)
        full_k = T.floordiv(T.floormod(linear_idx, self._GROUP_SIZE), self._G * self._N)

        # tail formulas (relative to FULL_BOUND)
        rem = linear_idx - self._FULL_BOUND
        tail_m = self._GROUPS * self._G + T.floormod(rem, self._FINAL_ROWS)
        tail_n = T.floordiv(rem, self._FINAL_ROWS) % self._N
        tail_k = T.floordiv(rem, self._FINAL_ROWS * self._N)

        # choose phase
        if self._HAS_FULL & (linear_idx < self._FULL_BOUND):
            self.m_idx = full_m
            self.n_idx = full_n
            self.k_idx = full_k
        elif self._HAS_TAIL:
            self.m_idx = tail_m
            self.n_idx = tail_n
            self.k_idx = tail_k
        else:
            self.m_idx = 0
            self.n_idx = 0
            self.k_idx = 0

    @T.macro
    def init(self, linear_init):
        self.linear_idx = linear_init
        self.tile_idx = 0
        self.update_current_m_n_idx(linear_init)

    @T.macro
    def next_tile(self):
        self.linear_idx = self.linear_idx + self._step
        self.tile_idx = self.tile_idx + 1
        self.update_current_m_n_idx(self.linear_idx)

    @T.macro
    def next_tile_stride(self, stride: int):
        self.linear_idx = self.linear_idx + stride
        self.tile_idx = self.tile_idx + 1
        self.update_current_m_n_idx(self.linear_idx)
    # fmt: on

    def valid(self):
        return self.linear_idx < self._TOTAL


class RankAwareGroupMajorTileScheduler(BaseTileScheduler):
    """
    Group-major scheduler that applies a rank-aware remapping (remote rows first).
    Kept as a thin adapter because it depends on NVSHMEM rank at device-side.
    """

    def __init__(
        self, prefix: str, m_clusters: int, n_clusters: int, group_size: int, world_size: int
    ):
        super().__init__(prefix)
        self._m_clusters = m_clusters
        self._n_clusters = n_clusters
        self._group_size = group_size
        self._world_size = world_size

    @T.macro
    def update_current_m_n_idx(self, linear_idx):
        my_rank = T.nvshmem.my_pe()
        remote_m_clusters = self._m_clusters - self._m_clusters // self._world_size
        group_rows = (remote_m_clusters // self._group_size) * self._group_size
        final_rows = remote_m_clusters - group_rows
        group_repeat = self._group_size * self._n_clusters
        if linear_idx < group_rows * self._n_clusters and group_rows > 0:
            self.m_idx = (
                (linear_idx // group_repeat) * self._group_size
                + (linear_idx % self._group_size)
                + (my_rank + 1) * self._m_clusters // self._world_size
            ) % self._m_clusters
            self.n_idx = (linear_idx % group_repeat) // self._group_size
        elif linear_idx < remote_m_clusters * self._n_clusters:
            remainder_idx = linear_idx - group_rows * self._n_clusters
            self.m_idx = (
                group_rows
                + remainder_idx % final_rows
                + (my_rank + 1) * self._m_clusters // self._world_size
            ) % self._m_clusters
            self.n_idx = remainder_idx // final_rows
        else:
            remainder_idx = linear_idx - remote_m_clusters * self._n_clusters
            self.m_idx = (
                remote_m_clusters
                + remainder_idx % (self._m_clusters // self._world_size)
                + (my_rank + 1) * self._m_clusters // self._world_size
            ) % self._m_clusters
            self.n_idx = remainder_idx // (self._m_clusters // self._world_size)

    @T.macro
    def next_tile(self, stride: int):
        self.linear_idx = self.linear_idx + stride
        self.update_current_m_n_idx(self.linear_idx)

    def valid(self):
        return self.linear_idx < self._m_clusters * self._n_clusters


class IndexedTripleTileScheduler(BaseTileScheduler):
    """Scheduler that maps linear_idx to (b_idx, h_idx, q_idx) via index lists."""

    def __init__(self, prefix: str, b_indices, h_indices, q_indices, tiles_indptr):
        super().__init__(prefix)
        self.b_indices = b_indices
        self.h_indices = h_indices
        self.q_indices = q_indices
        self.tiles_indptr = tiles_indptr
        self.q_idx = T.local_cell("int32", name=prefix + "_q_idx")
        self.h_idx = T.local_cell("int32", name=prefix + "_h_idx")
        self.b_idx = T.local_cell("int32", name=prefix + "_b_idx")
        self.linear_lim = T.local_cell("int32", name=prefix + "_linear_lim")

    @T.macro
    def _load(self):
        self.q_idx = self.q_indices[self.linear_idx]
        self.h_idx = self.h_indices[self.linear_idx]
        self.b_idx = self.b_indices[self.linear_idx]

    @T.macro
    def init(self, sm):
        self.linear_idx = self.tiles_indptr[sm]
        self.linear_lim = self.tiles_indptr[sm + 1]
        self._load()

    @T.macro
    def next_tile(self):
        self.linear_idx = self.linear_idx + 1
        self._load()

    def valid(self):
        return self.linear_idx < self.linear_lim
