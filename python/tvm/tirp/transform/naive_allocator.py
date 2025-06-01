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

from typing import List, Dict
import functools

from tvm.tir.stmt_functor import StmtVisitor, StmtExprMutator
from tvm.tir import BufferLoad, Block, BufferStore, OpCall, BufferRegion
from tvm.tir.buffer import Buffer
from tvm.tir.transform.function_pass import prim_func_pass
from tvm.tir import IntImm, BufferView, BufferGet
from tvm import DataType


def is_const_shape(buffer: Buffer) -> bool:
    for i in buffer.shape:
        if not isinstance(i, IntImm):
            return False
    return True


def get_buffer_size(buffer: Buffer) -> int:
    if buffer.scope() == "trn.sbuf":
        if buffer.layout is None:
            # the first dimension is partition size
            num_elem = functools.reduce(lambda x, y: x * y, buffer.shape[1:])
        else:
            par_size = buffer.layout.size("P")
            num_elem = functools.reduce(lambda x, y: x * y, buffer.shape) // par_size
    elif buffer.scope().startswith("shared"):
        num_elem = functools.reduce(lambda x, y: x * y, buffer.shape)
    else:
        return None
    if not is_const_shape(buffer):
        raise ValueError(
            f"Buffer {buffer.name} has non-constant shape. Do not know how to allocate it."
        )
    return int(num_elem * DataType(buffer.dtype).itemsize)


class AllocInfoCollector(StmtVisitor):
    def __init__(self):
        super().__init__()
        self.alloc_pool_start = 0

    def visit_block_(self, op: Block):
        for buffer in op.alloc_buffers:
            if len(buffer.allocated_addr) == 0:
                continue
            buffer_size = get_buffer_size(buffer)
            if buffer_size is None:
                continue
            self.alloc_pool_start = max(
                self.alloc_pool_start, buffer.allocated_addr[-1] + buffer_size
            )
        return super().visit_block_(op)


class BufferReplacer(StmtExprMutator):
    def __init__(self, buffer_map: Dict[Buffer, Buffer] = {}):
        super().__init__()
        self.buffer_map = buffer_map

    def visit_buffer_load_(self, op: BufferLoad):
        op = super().visit_buffer_load_(op)
        if op.buffer in self.buffer_map:
            return BufferLoad(self.buffer_map[op.buffer], op.indices)
        return op

    def visit_buffer_store_(self, op: BufferStore):
        op = super().visit_buffer_store_(op)
        if op.buffer in self.buffer_map:
            return BufferStore(self.buffer_map[op.buffer], op.value, op.indices)
        return op

    def visit_buffer_region_(self, op: BufferRegion):
        op = super().visit_buffer_region_(op)
        if op.buffer in self.buffer_map:
            return BufferRegion(self.buffer_map[op.buffer], op.region)
        return op

    def visit_op_call_(self, op):
        op = super().visit_op_call_(op)
        new_workspace = {
            key: self.buffer_map[value] if value in self.buffer_map else value
            for key, value in op.workspace.items()
        }
        return OpCall(
            *op.args, op=op.op, workspace=new_workspace, schedule_config=op.schedule_config
        )

    def visit_block_(self, op):
        op = super().visit_block_(op)
        new_buffer_views = []
        new_buffer_gets = []
        changed = False
        for buffer_view in op.buffer_views:
            if (
                not buffer_view.src_buffer in self.buffer_map
                and not buffer_view.dst_buffer in self.buffer_map
            ):
                new_buffer_views.append(buffer_view)
            else:
                new_src_buffer = (
                    self.buffer_map[buffer_view.src_buffer]
                    if buffer_view.src_buffer in self.buffer_map
                    else buffer_view.src_buffer
                )
                new_dst_buffer = (
                    self.buffer_map[buffer_view.dst_buffer]
                    if buffer_view.dst_buffer in self.buffer_map
                    else buffer_view.dst_buffer
                )
                new_buffer_views.append(
                    BufferView(new_src_buffer, buffer_view.layout, new_dst_buffer)
                )
                changed = True

        for buffer_get in op.buffer_gets:
            if (
                not buffer_get.src_buffer in self.buffer_map
                and not buffer_get.dst_buffer in self.buffer_map
            ):
                new_buffer_gets.append(buffer_get)
            else:
                new_src_buffer = (
                    self.buffer_map[buffer_get.src_buffer]
                    if buffer_get.src_buffer in self.buffer_map
                    else buffer_get.src_buffer
                )
                new_dst_buffer = (
                    self.buffer_map[buffer_get.dst_buffer]
                    if buffer_get.dst_buffer in self.buffer_map
                    else buffer_get.dst_buffer
                )
                new_buffer_gets.append(BufferGet(new_src_buffer, new_dst_buffer))
                changed = True
        if changed:
            return Block(
                op.iter_vars,
                op.reads,
                op.writes,
                op.name_hint,
                body=op.body,
                alloc_buffers=op.alloc_buffers,
                match_buffers=op.match_buffers,
                annotations=op.annotations,
                exec_scope=op.exec_scope,
                buffer_views=new_buffer_views,
                buffer_gets=new_buffer_gets,
                pipelines=op.pipelines,
            )
        return op


class AllocMutator(BufferReplacer):
    def __init__(self, alloc_pool_start: int):
        super().__init__()
        self.alloc_offset = alloc_pool_start

    def visit_block_(self, op: Block):
        changed = False
        new_alloc_buffers = []
        for buffer in op.alloc_buffers:
            buffer_size = get_buffer_size(buffer)
            if len(buffer.allocated_addr) > 0 or buffer_size is None:
                new_alloc_buffers.append(buffer)
                continue
            new_buffer = buffer.with_allocated_addr([self.alloc_offset])
            new_alloc_buffers.append(new_buffer)
            self.buffer_map[buffer] = new_buffer
            changed = True
            self.alloc_offset += buffer_size

        op = super().visit_block_(op)
        if changed:
            return Block(
                op.iter_vars,
                op.reads,
                op.writes,
                op.name_hint,
                body=op.body,
                alloc_buffers=new_alloc_buffers,
                match_buffers=op.match_buffers,
                annotations=op.annotations,
                exec_scope=op.exec_scope,
                buffer_views=op.buffer_views,
                buffer_gets=op.buffer_gets,
                pipelines=op.pipelines,
            )
        return op


@prim_func_pass(opt_level=0, name="NaiveAllocator")
class NaiveAllocator:
    def transform_function(self, func, mod, ctx):
        collector = AllocInfoCollector()
        collector(func.body)
        mutator = AllocMutator(collector.alloc_pool_start)
        new_body = mutator(func.body)
        return func.with_body(new_body)
