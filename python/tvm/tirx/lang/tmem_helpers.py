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
"""Parse-time TMEM load/store helpers."""

from tvm.script import tirx as Tx
from tvm.tirx.layout import S, TileLayout
from tvm.tirx.layout import tid_in_wg as axis_tid_in_wg


def _row_view(buf, width):
    return buf.view(128, width, layout=TileLayout(S[(128, width) : (1 @ axis_tid_in_wg, 1)]))


@Tx.meta_class
class F16OnF32View:
    def __init__(self, size):
        self.backing = Tx.alloc_buffer([size // 2], "float32", scope="local")
        self.buf = Tx.decl_buffer((size,), "float16", data=self.backing.data)
        self.view = _row_view(self.buf, size)


def alloc_f16_on_f32(size):
    return F16OnF32View(size)


def tmem_load_row(dst_view, region_slice, width, chunk_size=32):
    for start in range(0, width, chunk_size):
        stop = min(start + chunk_size, width)
        Tx.copy_async(dst_view[:, start:stop], region_slice[:, start:stop])


def tmem_store_chunked(region_slice, src_view, chunks_and_signals):
    for start, stop, bar, stage in chunks_and_signals:
        Tx.copy_async(region_slice[:, start:stop], src_view[:, start:stop])
        if bar is not None:
            Tx.ptx.tcgen05.wait.st()
            bar.arrive(stage)


def tmem_rescale(region_slice, scale, tile_size=16, head_dim=128):
    o_row_buf = Tx.alloc_buffer([tile_size], "float32", scope="local")
    o_row_wg = _row_view(o_row_buf, tile_size)
    for d_start in range(0, head_dim, tile_size):
        Tx.copy_async(o_row_wg, region_slice[:, d_start : d_start + tile_size])
        with Tx.thread():
            Tx.mul(o_row_buf, o_row_buf, scale)
        Tx.copy_async(region_slice[:, d_start : d_start + tile_size], o_row_wg[:, 0:tile_size])
    Tx.ptx.tcgen05.wait.st()


def tmem_writeback(region_slice, smem_dst, stage, row_idx, scale, tile_size=16, head_dim=128):
    reg_f32 = Tx.alloc_buffer([tile_size], "float32", scope="local")
    reg_f32_wg = _row_view(reg_f32, tile_size)
    reg_f16 = Tx.alloc_buffer([tile_size], "float16", scope="local")
    for d_start in range(0, head_dim, tile_size):
        Tx.copy_async(reg_f32_wg, region_slice[:, d_start : d_start + tile_size])
        with Tx.thread():
            Tx.mul(reg_f32, reg_f32, scale)
        with Tx.thread():
            Tx.cast(reg_f16, reg_f32)
            Tx.copy(smem_dst[stage, row_idx, d_start : d_start + tile_size], reg_f16, vec_len=8)
    Tx.ptx.fence.proxy_async("shared::cta")
