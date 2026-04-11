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
"""TMEM allocation helpers for TIRX kernels."""

from __future__ import annotations

import functools
import operator

from tvm import DataType
from tvm.tirx.layout import S, TCol, TileLayout, TLane

from .ir import add_to_parent, decl_buffer, evaluate, meta_class


def _default_tmem_layout(rows, cols):
    return TileLayout(S[(rows, cols) : (1 @ TLane, 1 @ TCol)])


def _emit_stmt(expr):
    add_to_parent(evaluate(expr))


def _shape_product(shape):
    return functools.reduce(operator.mul, shape, 1)


@meta_class
class TMEMRegion:
    """Parse-time staged view over a TMEM buffer."""

    def __init__(self, name, buf, col_start, width, stages=1):
        self.name = name
        self.buf = buf
        self.col_start = col_start
        self.width = width
        self.stages = stages

    def _stage_base(self, stage):
        return stage * self.width

    def __getitem__(self, item):
        if isinstance(item, tuple):
            assert len(item) == 2, "TMEMRegion expects region[stage] or region[stage, start:stop]"
            stage, col_slice = item
            assert isinstance(col_slice, slice), "TMEMRegion tuple indexing requires a slice"
            base = self._stage_base(stage)
            start = 0 if col_slice.start is None else col_slice.start
            stop = self.width if col_slice.stop is None else col_slice.stop
            return self.buf[:, base + start : base + stop : col_slice.step]
        base = self._stage_base(item)
        return self.buf[:, base : base + self.width]

    def slice(self, stage, start, stop):
        base = self._stage_base(stage)
        return self.buf[:, base + start : base + stop]

    def col(self, stage=0, offset=0):
        return self.col_start + stage * self.width + offset


@meta_class
class TMEMPool:
    """Bump allocator over TMEM columns."""

    def __init__(
        self,
        pool,
        total_cols=512,
        *,
        cta_group=1,
        warp_id=None,
        wg_id=None,
        dealloc_warp_id=None,
        dealloc_wg_id=None,
        tmem_addr=None,
    ):
        self.pool = pool
        self.total_cols = total_cols
        self.cta_group = cta_group
        self.warp_id = warp_id
        self.wg_id = wg_id
        self._alloc_warp_idx = 0 if warp_id is not None else None
        self._alloc_wg_idx = 0 if wg_id is not None else None
        self._dealloc_warp_idx = (
            self._alloc_warp_idx if dealloc_warp_id is None else dealloc_warp_id
        )
        self._dealloc_wg_idx = self._alloc_wg_idx if dealloc_wg_id is None else dealloc_wg_id
        self.offset = 0
        self.max_offset = 0
        self._committed = False
        self._addr_buf = (
            pool.alloc([1], "uint32", align=4, name="tmem_addr") if tmem_addr is None else tmem_addr
        )

    def _addr_slot(self):
        try:
            return self._addr_buf[0]
        except TypeError:
            return self._addr_buf

    @property
    def addr(self):
        return self._addr_slot()

    def _guard_pred(self, wg_idx, warp_idx):
        pred = None
        if wg_idx is not None:
            assert self.wg_id is not None, "TMEMPool guard requires wg_id to be provided"
            pred = self.wg_id == wg_idx
        if warp_idx is not None:
            assert self.warp_id is not None, "TMEMPool guard requires warp_id to be provided"
            warp_pred = self.warp_id == warp_idx
            pred = warp_pred if pred is None else pred & warp_pred
        return pred

    def _emit_warp_guard(self, Tx, pred, emit):
        if pred is None:
            with Tx.warp()[0:1]:
                emit()
            return
        with Tx.If(pred):
            with Tx.Then():
                with Tx.warp():
                    emit()

    def _resolve_cols(self, shape, dtype, cols):
        if cols is not None:
            return cols
        assert len(shape) == 2, "TMEMPool.alloc() requires cols= for non-2D TMEM buffers"
        bits = DataType(dtype).bits
        total_bits = _shape_product(shape) * bits
        rows = shape[0]
        assert total_bits % (32 * rows) == 0, (
            f"Cannot infer TMEM columns from shape={shape}, dtype={dtype!r}; "
            "please pass cols= explicitly"
        )
        return total_bits // (32 * rows)

    def alloc(self, shape, dtype="float32", *, layout=None, cols=None, name=None):
        cols = self._resolve_cols(shape, dtype, cols)
        col_start = self.offset
        col_end = col_start + cols
        assert col_end <= self.total_cols, (
            f"TMEM overflow: {col_end} > {self.total_cols} after allocating {name!r}"
        )
        if layout is None:
            assert len(shape) == 2, "TMEMPool.alloc() requires layout= for non-2D TMEM buffers"
            layout = _default_tmem_layout(shape[0], shape[1])
        res = decl_buffer(
            shape,
            dtype,
            scope="tmem",
            allocated_addr=col_start,
            layout=layout,
            name=name,
        )
        self.offset = col_end
        self.max_offset = self.offset if self.offset > self.max_offset else self.max_offset
        return res

    def move_base_to(self, col):
        self.offset = col
        self.max_offset = self.offset if self.offset > self.max_offset else self.max_offset

    def region(self, name, buf, width, stages=1):
        assert buf.allocated_addr is not None and len(buf.allocated_addr) > 0, (
            "TMEMPool.region() requires a TMEM buffer with allocated_addr"
        )
        return TMEMRegion(name, buf, buf.allocated_addr[0], width, stages)

    def commit(self):
        assert not self._committed, "TMEMPool.commit() can only be called once"
        from tvm.script import tirx as Tx

        def emit_alloc():
            _emit_stmt(
                Tx.ptx.tcgen05.alloc(
                    Tx.address_of(self.addr),
                    n_cols=self.total_cols,
                    cta_group=self.cta_group,
                )
            )
            _emit_stmt(Tx.cuda.warp_sync())

        self._emit_warp_guard(
            Tx,
            self._guard_pred(self._alloc_wg_idx, self._alloc_warp_idx),
            emit_alloc,
        )
        self._committed = True

    def dealloc(self):
        from tvm.script import tirx as Tx

        def emit_dealloc():
            _emit_stmt(Tx.ptx.tcgen05.relinquish_alloc_permit(cta_group=self.cta_group))
            _emit_stmt(Tx.ptx.tcgen05.dealloc(0, n_cols=self.total_cols, cta_group=self.cta_group))

        self._emit_warp_guard(
            Tx,
            self._guard_pred(self._dealloc_wg_idx, self._dealloc_warp_idx),
            emit_dealloc,
        )
