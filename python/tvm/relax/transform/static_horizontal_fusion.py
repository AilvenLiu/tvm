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
"""Static horizontal fusion
The pass is written in Python for experiment, fast development.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Type

import tvm
from tvm import relax
from tvm.ir import DictAttrs
from tvm.ir.module import IRModule
from tvm.relax.expr import Expr
from tvm.relax.expr_functor import PyExprMutator, mutator
from tvm.relax.struct_info import StructInfo, TensorStructInfo, TupleStructInfo
from tvm.script import tir as T
from tvm.script import tirp as Tp
from tvm.script.ir_builder import IRBuilder
from tvm.tir import (
    Block,
    Buffer,
    BufferLoad,
    IndexMap,
    PrimExpr,
    PrimFunc,
    SeqStmt,
    Stmt,
    Var,
)
from tvm.tir.analysis import verify_tirp_well_formed
from tvm.tir.event import EventImpl, SemaphoreEventTensor, SemaphoreEventTensorItem
from tvm.tir.stmt_functor import StmtExprMutator, StmtExprVisitor
from tvm.tirp.operator import EventCommit, EventWait
from tvm.tirp.transform.common import BufferReplacer, seek_kernel_replace_point


class TileScheduler:
    """Abstract base class for tile schedulers."""

    def init(self, value: PrimExpr):
        raise NotImplementedError

    def get_idx_and_task_type(self) -> Tuple[List[PrimExpr], int]:
        raise NotImplementedError

    def next_tile(self) -> None:
        raise NotImplementedError

    def valid(self) -> bool:
        raise NotImplementedError


@tvm.transform.module_pass(opt_level=0, name="StaticHorizontalFusion")
class StaticHorizontalFusion:
    """Static horizontal fusion."""

    def __init__(self, sm_count: int, tile_schedulers: Dict[str, Type[TileScheduler]] = None):
        if tile_schedulers is None:
            tile_schedulers = {}
        self.tile_schedulers = tile_schedulers
        self.sm_count = sm_count
        self.gvar_to_remove = []

    def transform_module(self, mod: IRModule, _ctx: tvm.transform.PassContext) -> IRModule:
        """IRModule-level transformation"""
        rewriter = _Rewriter(mod, self.sm_count, self.tile_schedulers)
        mod = rewriter.transform()
        for gvar in rewriter.gvar_to_remove:
            del mod[gvar]
        return mod


class EventOpInserter(StmtExprMutator):
    def __init__(
        self,
        in_event_bufs: List[Buffer],
        out_event_bufs: List[Buffer],
        in_deps: List[IndexMap],
        out_deps: List[IndexMap],
        tile_idx: List[PrimExpr],
    ):
        super().__init__()
        self.in_event_bufs = in_event_bufs
        self.out_event_bufs = out_event_bufs
        self.in_deps = in_deps
        self.out_deps = out_deps
        self.tile_idx = tile_idx

    @staticmethod
    def rewrite(
        stmt: Stmt,
        in_event_bufs: List[Buffer],
        out_event_bufs: List[Buffer],
        in_deps: List[IndexMap],
        out_deps: List[IndexMap],
        tile_idx: List[PrimExpr],
    ) -> Stmt:
        inserter = EventOpInserter(in_event_bufs, out_event_bufs, in_deps, out_deps, tile_idx)
        return inserter.visit_stmt(stmt)

    def visit_block_(self, block: Block):
        # define state regs for event tensors
        in_state_regs = [
            T.buffer((1,), "int32", scope="local", buffer_name="event_state")
            for _ in self.in_event_bufs
        ]
        out_state_regs = [
            T.buffer((1,), "int32", scope="local", buffer_name="event_state")
            for _ in self.out_event_bufs
        ]
        # define event tensors
        in_event_tensors = [
            SemaphoreEventTensor(
                EventImpl.kGlobalSemaphore, state=[buf, state_reg], shape=buf.shape
            )
            for buf, state_reg in zip(self.in_event_bufs, in_state_regs)
        ]
        out_event_tensors = [
            SemaphoreEventTensor(
                EventImpl.kGlobalSemaphore, state=[buf, state_reg], shape=buf.shape
            )
            for buf, state_reg in zip(self.out_event_bufs, out_state_regs)
        ]
        # only process root block
        waits = [
            EventWait(SemaphoreEventTensorItem(tensor, dep.map_indices(self.tile_idx)))
            for tensor, dep in zip(in_event_tensors, self.in_deps)
        ]
        commits = [
            EventCommit(SemaphoreEventTensorItem(tensor, dep.map_indices(self.tile_idx)))
            for tensor, dep in zip(out_event_tensors, self.out_deps)
        ]

        new_body = SeqStmt(waits + [block.body] + commits)
        # Reconstruct the block with the new body, preserving all other attributes.
        return Block(
            block.iter_vars,
            block.reads,
            block.writes,
            block.name_hint,
            new_body,
            block.init,
            list(block.alloc_buffers) + in_state_regs + out_state_regs,
            block.match_buffers,
            block.annotations,
            block.span,
            block.exec_scope,
            block.buffer_views,
            block.buffer_gets,
            sem_event_tensors=in_event_tensors + out_event_tensors,
        )


class DisjointSet:
    """A compact Disjoint Set Union (DSU) data structure with path compression and union-by-size."""

    def __init__(self):
        self.parent: Dict[object, object] = {}
        self.size: Dict[object, int] = {}

    def add_element(self, element: object):
        if element not in self.parent:
            self.parent[element] = element
            self.size[element] = 1

    def find(self, element: object) -> object:
        if self.parent[element] == element:
            return element
        self.parent[element] = self.find(self.parent[element])
        return self.parent[element]

    def join(self, a: object, b: object):
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a != root_b:
            if self.size[root_a] < self.size[root_b]:
                root_a, root_b = root_b, root_a
            self.parent[root_b] = root_a
            self.size[root_a] += self.size[root_b]

    def get_size(self, a: object) -> int:
        return self.size[self.find(a)]


@dataclass(frozen=True)
class VarEntry:
    relax_var: Expr
    tir_var: Var
    tir_buffer: Buffer


@dataclass
class DeviceFuncInfo:
    func_body: Stmt
    in_var_entries: List[VarEntry]
    out_var_entries: List[VarEntry]
    tile_idx: List[PrimExpr]


@mutator
class _Rewriter(PyExprMutator):
    def __init__(
        self, mod: IRModule, sm_count: int, tile_schedulers: Dict[str, Type[TileScheduler]]
    ):
        super().__init__(mod)
        self.mod = mod
        self.call_tir_device_op = tvm.ir.Op.get("relax.call_tir_device")
        self.tile_schedulers = tile_schedulers
        self.cur_func_name = None
        self.cur_tile_scheduler = None
        self.device_func_exec_scope = None
        self.device_func_infos: Dict[relax.Call, DeviceFuncInfo] = {}
        self.event_buffers = {}
        self.sm_count = sm_count
        self.ret_value_entries: Dict[relax.Call, List[int]] = defaultdict(list)
        self.new_ret_index = []
        self.var_entry_mapping = DisjointSet()
        self.gvar_to_remove = list()

    def clear_state(self):
        self.device_func_exec_scope = None
        self.device_func_infos = {}
        self.event_buffers = {}
        self.ret_value_entries = defaultdict(list)

    def transform(self) -> IRModule:
        """Entry point"""
        for g_var, func in self.mod.functions_items():
            if isinstance(func, relax.Function) and g_var.name_hint in self.tile_schedulers:
                self.cur_func_name = g_var.name_hint
                self.cur_tile_scheduler = self.tile_schedulers[g_var.name_hint]
                self.clear_state()
                self.visit_expr(func)
                self.builder_.update_func(g_var, self.build_new_function(func))
        return self.builder_.get()

    def get_output_num(self, call: relax.Call) -> int:
        if isinstance(call.struct_info, TensorStructInfo):
            return 1
        elif isinstance(call.struct_info, TupleStructInfo):
            return len(call.struct_info.fields)
        else:
            raise ValueError(f"Unsupported struct info: {call.struct_info}")

    def flatten_sinfo(self, sinfo: StructInfo) -> List[TensorStructInfo]:
        if isinstance(sinfo, TensorStructInfo):
            return [sinfo]
        elif isinstance(sinfo, TupleStructInfo):
            return sinfo.fields
        else:
            raise ValueError(f"Unsupported struct info: {sinfo}")

    def _get_event_buffer(self, event: Expr) -> Buffer:
        if event not in self.event_buffers:
            assert isinstance(event.struct_info, TensorStructInfo), f"event {event} is not a tensor"
            shape = [int(s) for s in event.struct_info.shape.values]
            self.event_buffers[event] = T.buffer(
                shape,
                event.struct_info.dtype,
                scope="global",
                buffer_name="event",
            )
        return self.event_buffers[event]

    def visit_call_(self, call: relax.Call) -> Expr:  # pylint: disable=arguments-renamed

        if call.op == self.call_tir_device_op:
            tile_num = call.args[2]
            tir_gvar = call.args[0]
            in_events = call.args[3]
            out_events = call.args[4]
            in_deps = call.attrs.in_deps
            out_deps = call.attrs.out_deps
            prim_func = self.mod[tir_gvar]
            self.gvar_to_remove.append(tir_gvar)
            # check if this device function is well-formed and has consistent exec scope
            verify_tirp_well_formed(prim_func, device_func=True)
            exec_scope = prim_func.body.block.exec_scope
            if self.device_func_exec_scope is None:
                self.device_func_exec_scope = exec_scope.name
            else:
                assert (
                    self.device_func_exec_scope == exec_scope.name
                ), f"device function exec scope mismatch: {self.device_func_exec_scope} != {exec_scope.name}"
            # collect event tensors and insert event commit/wait
            in_event_tensors = []
            out_event_tensors = []
            for e in in_events:
                in_event_tensors.append(self._get_event_buffer(e))
            for e in out_events:
                out_event_tensors.append(self._get_event_buffer(e))
            tile_idx = prim_func.params[-len(tile_num) :]
            body = EventOpInserter.rewrite(
                prim_func.body, in_event_tensors, out_event_tensors, in_deps, out_deps, tile_idx
            )
            # gather information for merging
            in_var_entries = []
            out_var_entries = []
            for i, arg in enumerate(call.args[1]):
                assert isinstance(
                    arg, relax.Var
                ), f"call_tir_device args[1] should be a list of relax.Var, but got {arg}"
                var_entry = VarEntry(
                    arg, prim_func.params[i], prim_func.buffer_map[prim_func.params[i]]
                )
                in_var_entries.append(var_entry)
                self.var_entry_mapping.add_element(var_entry)
                binding = self.builder_.lookup_binding(arg)
                if isinstance(binding, relax.Call):
                    self.var_entry_mapping.join(
                        var_entry, self.device_func_infos[binding].out_var_entries[0]
                    )
                elif isinstance(binding, relax.TupleGetItem):
                    self.var_entry_mapping.join(
                        var_entry,
                        self.device_func_infos[binding.tuple_value].out_var_entries[binding.index],
                    )
                elif binding is None:
                    pass
                else:
                    raise ValueError(f"Unsupported binding: {binding}, var: {arg}")
            output_num = self.get_output_num(call)
            for i in range(output_num):
                var_entry = VarEntry(
                    None,
                    prim_func.params[-len(tile_num) - output_num + i],
                    prim_func.buffer_map[prim_func.params[-len(tile_num) - output_num + i]],
                )
                out_var_entries.append(var_entry)
                self.var_entry_mapping.add_element(var_entry)
            self.device_func_infos[call] = DeviceFuncInfo(
                body, in_var_entries, out_var_entries, tile_idx
            )
            return super().visit_call_(call)
        else:
            raise ValueError(
                "StaticHorizontalFusion can only be applied to relax function with only call_tir_device"
            )

    def visit_seq_expr_(self, op: relax.SeqExpr) -> Expr:
        op = super().visit_seq_expr_(op)

        def _handle_var(var: relax.Var):
            binding = self.builder_.lookup_binding(var)
            if isinstance(binding, relax.Call):
                self.ret_value_entries[binding].append(0)
            elif isinstance(binding, relax.TupleGetItem):
                call_binding = self.builder_.lookup_binding(binding.tuple_value)
                if isinstance(call_binding, relax.Call):
                    self.ret_value_entries[call_binding].append(binding.index)
                else:
                    raise ValueError(f"Unsupported binding: {call_binding}")
            else:
                raise ValueError(f"Unsupported binding: {binding}")

        if isinstance(op.body, relax.Var):
            _handle_var(op.body)
        elif isinstance(op.body, relax.Tuple):
            for field in op.body.fields:
                _handle_var(field)
        else:
            raise ValueError(f"Unsupported body: {op.body}")
        return op

    def merge_function(self):
        new_relax_params = []
        new_tir_params = []
        new_buffer_map = {}
        out_sinfo = []
        buffer_replace_map = defaultdict(dict)
        for call, func in self.device_func_infos.items():
            for entry in func.in_var_entries:
                if self.var_entry_mapping.get_size(entry) == 1:
                    # this is an input tensor after merging
                    new_relax_params.append(entry.relax_var)
                    new_tir_params.append(entry.tir_var)
                    new_buffer_map[entry.tir_var] = entry.tir_buffer
                else:
                    buffer_replace_map[call][entry.tir_buffer] = self.var_entry_mapping.find(
                        entry
                    ).tir_buffer
            out_sinfo.extend(self.flatten_sinfo(call.struct_info))
        for e in self.event_buffers:
            event_tensor_var = T.var("handle", name=e.name_hint)
            new_buffer_map[event_tensor_var] = self.event_buffers[e]
            new_tir_params.append(event_tensor_var)
            new_relax_params.append(e)
        total_input_num = len(new_relax_params)

        for call, func in self.device_func_infos.items():
            for i in self.ret_value_entries[call]:
                self.new_ret_index.append(i + len(new_tir_params) - total_input_num)
            for entry in func.out_var_entries:
                # this is an output tensor after merging
                leader_entry = self.var_entry_mapping.find(entry)
                buffer_replace_map[call][entry.tir_buffer] = leader_entry.tir_buffer
                new_tir_params.append(leader_entry.tir_var)
                new_buffer_map[leader_entry.tir_var] = leader_entry.tir_buffer

        def switch_task_type(tile_scheduler: TileScheduler):
            idxs, type = tile_scheduler.get_idx_and_task_type()
            total_type_count = len(self.device_func_infos)
            if_frames = [T.If(type == i) for i in range(total_type_count)]
            then_frames = [T.Then() for i in range(total_type_count)]
            else_frames = [T.Else() for i in range(total_type_count - 1)]
            for i, (call, func) in enumerate(self.device_func_infos.items()):
                if_frames[i].__enter__()
                needed_idx_num = len(func.tile_idx)
                d = {func.tile_idx[j]: idxs[j] for j in range(needed_idx_num)}
                print(f"replace var: {d}")
                replacer = BufferReplacer(
                    buffer_map=buffer_replace_map[call],
                    var_map={func.tile_idx[j]: idxs[j] for j in range(needed_idx_num)},
                )
                new_body = replacer.visit_stmt(func.func_body)
                with then_frames[i]:
                    T.add_to_parent(new_body)
                if i < total_type_count - 1:
                    else_frames[i].__enter__()

            for i in range(total_type_count - 1, -1, -1):
                if i < total_type_count - 1:
                    else_frames[i].__exit__(None, None, None)
                if_frames[i].__exit__(None, None, None)

        @T.prim_func(tirp=True)
        def persistent_kernel():
            with T.kernel():
                v = T.scope_id([self.sm_count], parent="kernel", cur=self.device_func_exec_scope)
                tile_scheduler = T.meta_var(self.cur_tile_scheduler())
                tile_scheduler.init(v)
                while tile_scheduler.valid():
                    switch_task_type(tile_scheduler)
                    tile_scheduler.next_tile()

        new_body = persistent_kernel.body
        new_prim_func = PrimFunc(
            new_tir_params,
            new_body,
            None,
            new_buffer_map,
            tvm.ir.make_node(
                "ir.DictAttrs",
                is_tirp=True,
                global_symbol=f"persistent_kernel_{self.cur_func_name}",
            ),
        )
        new_gvar = self.builder_.add_func(new_prim_func, f"persistent_kernel_{self.cur_func_name}")
        return relax.call_tir(new_gvar, new_relax_params, out_sinfo)

    def build_new_function(self, func: relax.Function):
        builder = relax.BlockBuilder()
        with builder.function(
            name="mega_kernel",
            params=func.params,
            attrs=dict(func.attrs),
            pure=func.is_pure,
            private=True,
        ):
            output_var = builder.emit(self.merge_function())
            if isinstance(output_var.struct_info, TensorStructInfo):
                builder.emit_func_output(output_var)
            else:
                ret_vars = []
                for i in self.new_ret_index:
                    ret_vars.append(builder.emit(relax.TupleGetItem(output_var, i)))
                if len(ret_vars) == 1:
                    ret_vars = ret_vars[0]
                builder.emit_func_output(ret_vars)
        mod = builder.finalize()
        return mod["mega_kernel"]
