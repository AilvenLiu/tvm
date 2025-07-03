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

from typing import Dict, Tuple

from tvm.ir import PointerType, PrimType
from tvm.tir import Block, Var
from tvm.tir.buffer import Buffer
from tvm.tir.transform.function_pass import prim_func_pass
from tvm import DataTypeCode
from tvm.tir.function import PrimFunc
from .common import BufferReplacer

class EventTensorReplacer(BufferReplacer):
    def __init__(self, buffer_map: Dict[Buffer, Buffer], var_map: Dict[Var, Var]):
        super().__init__(buffer_map, var_map)

    def visit_block_(self, op: Block):
        changed = False
        new_alloc_buffers = []
        for buffer in op.alloc_buffers:
            if buffer.is_event_tensor():
                new_buffer = convert_event_tensor(buffer)
                self.buffer_map[buffer] = new_buffer
                new_alloc_buffers.append(new_buffer)
                changed = True
            else:
                new_alloc_buffers.append(buffer)

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

def convert_event_tensor(buffer: Buffer) -> Tuple[Buffer, Var]:
    new_dtype = buffer.dtype.with_code(DataTypeCode.INT)
    new_buffer = buffer.with_dtype(new_dtype)
    data = buffer.data
    old_type = data.type_annotation
    new_data = Var(data.name, PointerType(PrimType(new_dtype), storage_scope=old_type.storage_scope, logical_scope=old_type.logical_scope), data.span)
    return new_buffer.with_data(new_data), new_data

@prim_func_pass(opt_level=0, name="EventTensorLegalizer")
class EventTensorLegalizer:
    def transform_function(self, func, mod, ctx):
        buffer_replace_map = {}
        var_replace_map = {}
        new_buffer_map = {}
        for var, buffer in func.buffer_map.items():
            if buffer.is_event_tensor():
                new_buffer, new_data = convert_event_tensor(buffer)
                buffer_replace_map[buffer] = new_buffer
                var_replace_map[buffer.data] = new_data
                new_buffer_map[var] = new_buffer
            else:
                new_buffer_map[var] = buffer
        legalizer = EventTensorReplacer(buffer_replace_map, var_replace_map)
        new_body = legalizer(func.body)
        new_func = PrimFunc(func.params, new_body, func.ret_type, new_buffer_map, func.attrs, func.span)
        return new_func
