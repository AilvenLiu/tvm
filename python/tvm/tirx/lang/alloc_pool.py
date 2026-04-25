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
"""SMEM and TMEM bump-allocator pools for TIRX kernels."""

from __future__ import annotations

import functools
import operator

from tvm import DataType
from tvm.tirx.layout import S, TCol, TileLayout, TLane

# ---------------------------------------------------------------------------
# ir_builder helpers — imported lazily to avoid circular deps at module level
# ---------------------------------------------------------------------------

_ir = None


def _get_ir():
    global _ir
    if _ir is None:
        from tvm.script.ir_builder.tirx import ir as _mod

        _ir = _mod
    return _ir


def _get_frame():
    from tvm.script.ir_builder.tirx import frame

    return frame


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

_POOL_UNSET = object()


def _default_tmem_layout(rows, cols):
    return TileLayout(S[(rows, cols) : (1 @ TLane, 1 @ TCol)])


def _emit_stmt(expr):
    ir = _get_ir()
    ir.add_to_parent(ir.evaluate(expr))


def _shape_product(shape):
    return functools.reduce(operator.mul, shape, 1)


def _auto_swizzle_mode(dtype):
    """Select the default MMA swizzle mode for a shared-memory allocation."""
    from tvm.tirx.operator.tile_primitive_dispatch.cuda.tma_utils import SwizzleMode

    del dtype
    return SwizzleMode.SWIZZLE_128B_ATOM


# ---------------------------------------------------------------------------
# TMEMRegion
# ---------------------------------------------------------------------------


def _meta_class(cls):
    """Apply @meta_class decorator from ir_builder."""
    return _get_ir().meta_class(cls)


@_meta_class
class TMEMRegion:
    """Parse-time staged view over a TMEM buffer.

    Parameters
    ----------
    buf : Buffer
        The underlying TMEM buffer (e.g. f32 or f16 view).
    col_start : int
        First column of stage 0 in *buf*'s column space.
    width : int
        Number of columns per stage.
    stages : int
        Number of pipeline stages (default 1).
    stride : int or None
        Column distance between consecutive stages.  When *None* (default),
        equals *width* (stages are packed back-to-back).
    """

    def __init__(self, buf, col_start, width, stages=1, stride=None):
        self.buf = buf
        self.col_start = col_start
        self.width = width
        self.stages = stages
        self.stride = width if stride is None else stride

    def _stage_base(self, stage):
        return self.col_start + stage * self.stride

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


# ---------------------------------------------------------------------------
# TMEMPool
# ---------------------------------------------------------------------------


@_meta_class
class TMEMPool:
    """Bump allocator over TMEM columns."""

    def __init__(
        self,
        pool,
        total_cols=512,
        *,
        cta_group=1,
        alloc_warp=0,
        dealloc_warp=None,
        tmem_addr=None,
    ):
        # tcgen05 alloc/dealloc are warp-uniform PTX instructions: every lane
        # in the chosen warp must participate, and exactly one warp in the
        # CTA must execute them. The pool emits its own
        # ``if thread_rank() // 32 == target_warp: with Tx.warp(): tcgen05.alloc(...)``
        # guard, using ``Tx.cuda.thread_rank()`` (cooperative_groups thread
        # rank) so callers don't have to declare the CTA's thread layout.
        self.pool = pool
        self.total_cols = total_cols
        self.cta_group = cta_group
        self.alloc_warp = alloc_warp
        self.dealloc_warp = alloc_warp if dealloc_warp is None else dealloc_warp
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

    def _emit_warp_guard(self, Tx, target_warp, emit):
        with Tx.If(Tx.cuda.thread_rank() // 32 == target_warp):
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
        ir = _get_ir()
        cols = self._resolve_cols(shape, dtype, cols)
        col_start = self.offset
        col_end = col_start + cols
        assert col_end <= self.total_cols, (
            f"TMEM overflow: {col_end} > {self.total_cols} after allocating {name!r}"
        )
        if layout is None:
            assert len(shape) == 2, "TMEMPool.alloc() requires layout= for non-2D TMEM buffers"
            layout = _default_tmem_layout(shape[0], shape[1])
        res = ir.decl_buffer(
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

    def region(self, buf, col_start, width, stages=1, stride=None):
        """Create a staged region view over *buf*.

        Parameters
        ----------
        buf : Buffer
            TMEM buffer returned by ``alloc()``.
        col_start : int
            First column of stage 0 (in *buf*'s column units).
        width : int
            Columns per stage.
        stages : int
            Pipeline depth.
        stride : int or None
            Column distance between consecutive stages (default = *width*).
        """
        return TMEMRegion(buf, col_start, width, stages, stride)

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

        self._emit_warp_guard(Tx, self.alloc_warp, emit_alloc)
        self._committed = True

    def dealloc(self):
        from tvm.script import tirx as Tx

        def emit_dealloc():
            _emit_stmt(Tx.ptx.tcgen05.relinquish_alloc_permit(cta_group=self.cta_group))
            _emit_stmt(Tx.ptx.tcgen05.dealloc(0, n_cols=self.total_cols, cta_group=self.cta_group))

        self._emit_warp_guard(Tx, self.dealloc_warp, emit_dealloc)


# ---------------------------------------------------------------------------
# SMEMPool
# ---------------------------------------------------------------------------


@_meta_class
class SMEMPool:
    """Bump allocator over a contiguous shared memory region.

    Parameters
    ----------
    ptr : Var or None, optional
        If omitted, an ``alloc_buffer([0], "uint8", scope="shared.dyn")`` is
        created automatically and ``commit()`` must be called after all
        allocations to emit the size annotation.
        If a ``Var`` (or ``None`` for megakernel fusion mode) is provided,
        the caller manages the backing buffer and ``commit()`` is a no-op.
    """

    def __init__(self, ptr=_POOL_UNSET):
        ir = _get_ir()
        if ptr is _POOL_UNSET:
            buf = ir.alloc_buffer([0], "uint8", scope="shared.dyn")
            self.ptr = buf.data
            self._owns_buffer = True
        else:
            self.ptr = ptr
            self._owns_buffer = False
        self.offset = 0
        self.max_offset = 0

    def alloc(
        self,
        shape,
        dtype="float32",
        strides=None,
        scope="global",
        align=0,
        buffer_type="",
        axis_separators=None,
        layout="default",
        name=None,
    ):
        ir = _get_ir()
        if align > 0:
            self.offset = (self.offset + align - 1) // align * align
        res = ir.decl_buffer(
            shape,
            dtype,
            self.ptr,
            strides,
            None,
            self.offset,
            scope,
            align,
            0,
            buffer_type,
            axis_separators,
            layout,
            name,
        )
        self.offset += functools.reduce(lambda x, y: x * y, shape) * (DataType(dtype).bits // 8)
        if self._owns_buffer:
            self.max_offset = self.offset if self.offset > self.max_offset else self.max_offset
        return res

    def alloc_mma(
        self,
        shape,
        dtype="float16",
        swizzle_mode="auto",
        align=1024,
        name=None,
    ):
        """Allocate MMA-compatible shared memory with an inferred swizzle layout."""
        from tvm.tirx.operator.tile_primitive_dispatch.cuda.tma_utils import (
            SwizzleMode,
            mma_shared_layout,
        )

        if isinstance(swizzle_mode, str):
            if swizzle_mode == "auto":
                swizzle_mode = _auto_swizzle_mode(dtype)
            elif swizzle_mode == "none":
                swizzle_mode = SwizzleMode.SWIZZLE_NONE
            else:
                raise ValueError(
                    f"Unsupported swizzle_mode={swizzle_mode!r}; expected 'auto', 'none', "
                    "or SwizzleMode"
                )
        layout = mma_shared_layout(dtype, swizzle_mode, shape)
        return self.alloc(shape, dtype, align=align, layout=layout, name=name)

    def move_base_to(self, offset):
        self.offset = offset
        if self._owns_buffer:
            self.max_offset = self.offset if self.offset > self.max_offset else self.max_offset

    def commit(self, size=None):
        """Emit pool size annotation into the IR.

        Must be called after all ``alloc()`` / ``move_base_to()`` calls.

        Parameters
        ----------
        size : int, optional
            Explicit shared memory size in bytes.  When *None* (the default),
            the high-water mark ``max_offset`` tracked by the allocator is used.
        """
        if not self._owns_buffer:
            return
        ir = _get_ir()
        frame_mod = _get_frame()
        resolved = size if size is not None else self.max_offset
        assert resolved >= self.max_offset, (
            f"Specified smem size ({resolved}) is smaller than "
            f"the pool high-water mark ({self.max_offset})"
        )
        attr_frame = ir.attr(self.ptr, "tirx.pool_max_bytes", resolved)
        if isinstance(attr_frame, frame_mod.AttrFrame):
            from functools import partial

            attr_frame.add_callback(partial(attr_frame.__exit__, None, None, None))
            attr_frame.__enter__()


# Backward-compatible alias used by existing kernels.
PoolAllocator = SMEMPool
