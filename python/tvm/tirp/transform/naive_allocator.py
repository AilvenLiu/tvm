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

from tvm.tir.stmt_functor import StmtVisitor
from tvm.tir import Block
from tvm.tir.buffer import Buffer
from tvm.tir.transform.function_pass import prim_func_pass
from tvm.tir import IntImm
from tvm import DataType
from .common import BufferReplacer


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
