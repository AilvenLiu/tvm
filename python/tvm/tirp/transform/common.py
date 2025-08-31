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

from typing import Dict, List

from tvm.tir import (
    Block,
    BufferLoad,
    BufferRegion,
    BufferStore,
    OpCall,
    PrimExpr,
    Stmt,
    Var,
)
from tvm.tir.buffer import Buffer
from tvm.tir.event import SemaphoreEventTensorItem
from tvm.tir.stmt_functor import StmtExprMutator, StmtMutator
from tvm.tirp.operator.op import KernelReplacePoint


class BufferReplacer(StmtExprMutator):
    """
    Replace buffer with another buffer.
    Also replace the data of the buffer with another var.
    """

    def __init__(self, buffer_map: Dict[Buffer, Buffer] = {}, var_map: Dict[Var, Var] = {}):
        super().__init__()
        self.buffer_map = buffer_map
        self.var_map = var_map

    def visit_var_(self, op: Var):
        op = super().visit_var_(op)
        if op in self.var_map:
            return self.var_map[op]
        return op

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

    def visit_array_prim_expr_(self, op: List[PrimExpr]):
        return [self.visit_expr(expr) for expr in op]

    def visit_op_call_(self, op):
        op = super().visit_op_call_(op)
        new_workspace = {
            key: self.buffer_map[value] if value in self.buffer_map else value
            for key, value in op.workspace.items()
        }
        args = list()
        for arg in op.args:
            if isinstance(arg, SemaphoreEventTensorItem):
                args.append(
                    SemaphoreEventTensorItem(arg.tensor, self.visit_array_prim_expr_(arg.indices))
                )
            else:
                args.append(arg)
        return OpCall(*args, op=op.op, workspace=new_workspace, schedule_config=op.schedule_config)


class KernelReplacePointSearcher(StmtMutator):
    def __init__(self, body: Stmt):
        super().__init__()
        self.body = body

    def visit_op_call_(self, op: OpCall):
        op = OpCall.downcast(op)
        if isinstance(op, KernelReplacePoint):
            return self.body
        return super().visit_op_call_(op)


def seek_kernel_replace_point(stmt: Stmt, body: Stmt) -> Stmt:
    """replace kernel replace point in stmt with body"""
    return KernelReplacePointSearcher(body)(stmt)
