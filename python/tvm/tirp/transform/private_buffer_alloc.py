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

from tvm.tir.stmt_functor import StmtVisitor, StmtMutator
from tvm.tirp.op_schedule.schedule_context import ScheduleContext
from tvm.target import Target
from tvm.ir import Range
from tvm.tir.stmt import Block, AttrStmt, For, OpCall, Stmt
from tvm.tir.buffer import Buffer
from tvm.tir.transform.function_pass import prim_func_pass
from tvm.tirp.transform.common import seek_kernel_replace_point


class PrivateAllocCollector(StmtVisitor):
    def __init__(self, target: Target):
        super().__init__()
        self.target = target
        self.exec_scope_stack_ = []
        self.launch_params = {}
        self.var_range_map = {}
        self.buffer_dict = {}
        self.private_buf_refs = {}

    def visit_block_(self, op: Block):
        if op.exec_scope is not None:
            self.exec_scope_stack_.append(op.exec_scope)
        super().visit_block_(op)
        if op.exec_scope is not None:
            self.exec_scope_stack_.pop()

    def visit_attr_(self, op: AttrStmt):
        if op.attr_key == "thread_extent":
            self.launch_params[op.node.thread_tag] = op.value
        super().visit_attr_(op)

    def visit_for_(self, op: For):
        self.var_range_map[op.loop_var] = Range.from_min_extent(op.min, op.extent)
        super().visit_for_(op)

    def visit_op_call_(self, op: OpCall):
        sctx = ScheduleContext(
            target=self.target,
            exec_scope=self.exec_scope_stack_[-1],
            launch_params=self.launch_params,
            var_range_map=self.var_range_map,
            alloc_only=True,
        )
        op = OpCall.downcast(op)
        private_buf_refs = op.get_private_buffers(self.buffer_dict, sctx)
        self.private_buf_refs[op] = private_buf_refs




class PrivateAllocMutator(StmtMutator):
    def __init__(
        self,
        alloc_buffers: List[Buffer],
        init_stmts: List[Stmt],
        added_workspace: Dict[OpCall, Dict[str, Buffer]],
    ):
        super().__init__()
        self.alloc_buffers = alloc_buffers
        self.init_stmts = init_stmts
        self.added_workspace = added_workspace
        self.is_outer_block = True

    def visit_block_(self, op: Block):
        is_outer_block = self.is_outer_block
        self.is_outer_block = False
        op = super().visit_block_(op)
        if is_outer_block:
            alloc_buffers = self.alloc_buffers + list(op.alloc_buffers)
            body = op.body
            for stmt in self.init_stmts:
                body = seek_kernel_replace_point(stmt, body)
            block = Block(
                [],
                [],
                [],
                name_hint=op.name_hint,
                body=body,
                alloc_buffers=alloc_buffers,
                match_buffers=op.match_buffers,
                annotations=op.annotations,
                exec_scope=op.exec_scope,
                buffer_views=op.buffer_views,
                buffer_gets=op.buffer_gets,
                pipelines=op.pipelines,
            )
            return block
        return op

    def visit_op_call_(self, op):
        if op not in self.added_workspace:
            return op
        new_workspace = dict(op.workspace)
        new_workspace.update(self.added_workspace[op])
        op = OpCall(*op.args, op=op.op, workspace=new_workspace, schedule_config=op.schedule_config)
        return op


def private_alloc(stmt: Stmt, target: Target) -> Stmt:
    collector = PrivateAllocCollector(target)
    collector(stmt)

    alloc_buffers = [buffer for buffer, _ in collector.buffer_dict.values()]
    init_stmts = [stmt for _, stmt in collector.buffer_dict.values() if stmt is not None]
    added_workspace = {
        op: {
            name: collector.buffer_dict[ref][0]
            for name, ref in collector.private_buf_refs[op].items()
        }
        for op in collector.private_buf_refs
    }

    mutator = PrivateAllocMutator(alloc_buffers, init_stmts, added_workspace)
    return mutator(stmt)


@prim_func_pass(opt_level=0, name="PrivateBufferAlloc")
class PrivateBufferAlloc:
    """Generate private buffer allocations for each OpCall"""

    def transform_function(self, func, mod, ctx):
        target = func.attrs.get("target", None)
        if target is None:
            target = Target.current(allow_none=False)
        new_body = private_alloc(func.body, target)
        new_func = func.with_body(new_body)
        return new_func
