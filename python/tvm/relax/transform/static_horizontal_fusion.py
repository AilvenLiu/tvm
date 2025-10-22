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
from tvm.ir import DictAttrs
from typing import Dict, List, Type, Tuple, Optional, Union
from dataclasses import dataclass

import tvm
from tvm import relax
from tvm.ir import DictAttrs, load_json, save_json
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
    Evaluate,
    BlockRealize,
    AllocBuffer,
)
from tvm.tir.analysis import verify_tirp_well_formed
from tvm.tir.stmt_functor import StmtExprMutator, StmtExprVisitor
from tvm.tirp.transform.common import seek_kernel_replace_point, BufferReplacer
from tvm.tirp.megakernel.common import KernelConfig, JobType, SmemManager, TileSchedulerBase, SemaphoreBase, pack_into_32bit
from tvm.tirp.operator import KernelReplacePoint
from tvm.tir.exec_scope import ExecScope

# FIXME: add decl_buffer for all newly generated buffers

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
    in_event_tensors: List[Buffer]
    out_event_tensors: List[Buffer]
    sym_vars: set[Var]
    in_info_list: List[Tuple[callable, callable, PrimExpr, PrimExpr]]
    out_info_list: List[Tuple[callable, callable, PrimExpr, PrimExpr]]
    func_type: str
    job_type_id: int
    wait_level: str
    notify_scope: str
    notify_scope_id: int

@tvm.transform.module_pass(opt_level=0, name="StaticHorizontalFusion")
class StaticHorizontalFusion:
    """Static horizontal fusion."""

    def __init__(self, func_name: Union[str, List[str]], tile_scheduler_class: Type[TileSchedulerBase], semaphore_class: Type[SemaphoreBase], fusion_prefix: str = "megakernel_"):
        self.func_name = func_name if isinstance(func_name, list) else [func_name]
        self.tile_scheduler_class = tile_scheduler_class
        self.semaphore_class = semaphore_class
        self.fusion_prefix = fusion_prefix
        self.gvar_to_remove = []

    def transform_module(self, mod: IRModule, _ctx: tvm.transform.PassContext) -> IRModule:
        """IRModule-level transformation"""
        rewriter = _Rewriter(mod, self.func_name, self.fusion_prefix, self.tile_scheduler_class, self.semaphore_class)
        mod = rewriter.transform()
        for gvar in rewriter.gvar_to_remove:
            del mod[gvar]
        mod = relax.transform.DeadCodeElimination()(mod)
        return mod


class EventOpInserter(StmtExprMutator):
    def __init__(
        self,
        func_info: DeviceFuncInfo,
        tile_scheduler: TileSchedulerBase,
        semaphore_class: Type[SemaphoreBase]
    ):
        super().__init__()
        self.func_info = func_info
        self.tile_scheduler = tile_scheduler
        self.semaphore_class = semaphore_class

    @staticmethod
    def rewrite(
        func_info: DeviceFuncInfo,
        tile_scheduler: TileSchedulerBase,
        semaphore_class: Type[SemaphoreBase]
    ) -> Stmt:
        inserter = EventOpInserter(func_info, tile_scheduler, semaphore_class)
        return inserter.visit_stmt(func_info.func_body)

    def visit_block_(self, block: Block): 
        if block.annotations.get("tirp.tile_class.prefetch") is not None:
            return block
        elif block.annotations.get("tirp.tile_class.run") is not None:  
            def get_semaphore_wait(idx):
                @T.prim_func(tirp=True, check_well_formed=False)
                def wait():
                    sem = T.meta_var(self.semaphore_class(-1, self.func_info.in_event_tensors[idx], decrement=True))
                    wait_func = T.meta_var(self.func_info.in_info_list[idx][0])
                    self.tile_scheduler.wait(sem, *(wait_func(T.int32(0))), wait_level=self.func_info.wait_level)
                return wait
            
            def get_semaphore_notify(idx):            
                @T.prim_func(tirp=True, check_well_formed=False)
                def notify():
                    sem = T.meta_var(self.semaphore_class(-1, self.func_info.out_event_tensors[idx], decrement=True))
                    notify_func = T.meta_var(self.func_info.out_info_list[idx][0])
                    num_func = T.meta_var(self.func_info.out_info_list[idx][1])
                    self.tile_scheduler.notify(sem, num_func()[0], notify_func, 
                                            scope=self.func_info.notify_scope, scope_id=self.func_info.notify_scope_id)
                return notify
            
            # insert wait and notify
            waits = [get_semaphore_wait(idx).body for idx in range(len(self.func_info.in_event_tensors))]
            commits = [get_semaphore_notify(idx).body for idx in range(len(self.func_info.out_event_tensors))]
            if len(waits) > 0:
                waits = SeqStmt(waits) if len(waits) > 1 else waits[0]
                for _, _, buf1, buf2 in self.func_info.in_info_list:
                    waits = AllocBuffer(buf1, waits)
                    waits = AllocBuffer(buf2, waits)
                waits = [waits]
            else:
                waits = []
            if len(commits) > 0:
                commits = SeqStmt(commits) if len(commits) > 1 else commits[0]
                for _, _, buf1, buf2 in self.func_info.out_info_list:
                    commits = AllocBuffer(buf1, commits)
                    commits = AllocBuffer(buf2, commits)
                commits = [commits]
            else:
                commits = []
            body_list = waits + [block.body] + commits
            new_body = SeqStmt(body_list) if len(body_list) > 1 else body_list[0]
            # Reconstruct the block with the new body, preserving all other attributes.
            return Block(
                block.iter_vars,
                block.reads,
                block.writes,
                block.name_hint,
                new_body,
                block.init,
                block.alloc_buffers,
                block.match_buffers,
                block.annotations,
                block.span,
                block.exec_scope,
            )
        else:
            return super().visit_block_(block)


@dataclass
class PersistentVarInfo:
    init_body: Stmt
    finalize_body: Stmt
    vars: Dict[str, List[Buffer]]


class PersistentVarCollector(StmtExprMutator):
    
    def __init__(self):
        super().__init__()
        self.persistent_vars_tile_class = defaultdict(list)
        self.persistent_var_tile_class_init = None
        self.persistent_var_tile_class_finalize = None
        self.persistent_vars_megakernel = defaultdict(list)
        self.persistent_var_megakernel_init = None
        self.persistent_var_megakernel_finalize = None
        self.visit_tile_class = False
        
    
    @staticmethod
    def rewrite(stmt: Stmt) -> Tuple[Stmt, PersistentVarInfo, PersistentVarInfo]:
        collector = PersistentVarCollector()
        return (
            collector.visit_stmt(stmt),
            PersistentVarInfo(collector.persistent_var_tile_class_init, collector.persistent_var_tile_class_finalize, collector.persistent_vars_tile_class),
            PersistentVarInfo(collector.persistent_var_megakernel_init, collector.persistent_var_megakernel_finalize, collector.persistent_vars_megakernel)
        )
    
    def visit_alloc_buffer_(self, op):
        op = super().visit_alloc_buffer_(op)
        if "persistent" in op.buffer.scope():
            cur_persistent_vars = self.persistent_vars_tile_class if self.visit_tile_class else self.persistent_vars_megakernel
            cur_persistent_vars[op.buffer.name].append(op.buffer)
            return op.body
        return op
    
    def visit_block_(self, block: Block):
        if any(block.annotations.get(key) is not None for key in ["tirp.tile_class.persistent.init", "tirp.tile_class.persistent.finalize"]):
            self.visit_tile_class = True
        elif any(block.annotations.get(key) is not None for key in ["tirp.megakernel.persistent.init", "tirp.megakernel.persistent.finalize"]):
            self.visit_tile_class = False
        block = super().visit_block_(block)
        annotation_to_var = {
            "tirp.tile_class.persistent.init": "persistent_var_tile_class_init",
            "tirp.tile_class.persistent.finalize": "persistent_var_tile_class_finalize",
            "tirp.megakernel.persistent.init": "persistent_var_megakernel_init",
            "tirp.megakernel.persistent.finalize": "persistent_var_megakernel_finalize",
        }
        for annotation, var_name in annotation_to_var.items():
            if block.annotations.get(annotation) is not None:
                setattr(self, var_name, block.body)
                break 
        if not any(block.annotations.get(key) for key in annotation_to_var.keys()):
            return block
        return Block(
            block.iter_vars,
            block.reads,
            block.writes,
            block.name_hint,
            Evaluate(0),
            exec_scope=ExecScope.create("cta")
        )

class HostStmtCollector(StmtExprMutator):
    def __init__(self):
        super().__init__()
        self.uppermost_block = None
    
    def visit_block_realize_(self, op: BlockRealize):
        self.uppermost_block = op
        return KernelReplacePoint()

    
    @staticmethod
    def rewrite(stmt: Stmt) -> Tuple[Stmt, Stmt]:
        collector = HostStmtCollector()
        host_replace_template = collector.visit_stmt(stmt)
        return collector.uppermost_block, host_replace_template

class VarCollector(StmtExprVisitor):
    
    def __init__(self):
        super().__init__()
        self.vars = set()
    
    def visit_var_(self, op):
        self.vars.add(op)
        return super().visit_var_(op)

class DynSharedMemCollector(StmtExprMutator):
    def __init__(self):
        super().__init__()
        self.dyn_shared_mem_buffer = None
        
    def visit_alloc_buffer_(self, op):
        op =super().visit_alloc_buffer_(op)
        if op.buffer.scope() == "shared.dyn":
            if self.dyn_shared_mem_buffer is None:
                self.dyn_shared_mem_buffer = op.buffer
                return op.body
            else:
                raise ValueError(f"Multiple dynamic shared memory buffers found: {self.dyn_shared_mem_buffer} and {op.buffer}")
        return op

    @staticmethod
    def rewrite(stmt: Stmt) -> Tuple[Buffer, Stmt]:
        collector = DynSharedMemCollector()
        stmt = collector.visit_stmt(stmt)
        return collector.dyn_shared_mem_buffer, stmt

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
            
    def is_connected(self, a: object, b: object) -> bool:
        return self.find(a) == self.find(b)

    def get_size(self, a: object) -> int:
        return self.size[self.find(a)]

@mutator
class _Rewriter(PyExprMutator):
    def __init__(
        self, mod: IRModule, rewrite_func_name: Union[str, List[str]], fusion_prefix: str, 
        tile_scheduler_class: Type[TileSchedulerBase], semaphore_class: Type[SemaphoreBase]
    ):
        super().__init__(mod)
        self.mod = mod
        self.call_tir_device_op = tvm.ir.Op.get("relax.call_tir_device")
        self.rewrite_func_name = set(rewrite_func_name)
        self.fusion_prefix = fusion_prefix
        self.tile_scheduler_class = tile_scheduler_class
        self.semaphore_class = semaphore_class
        self.device_func_exec_scope = None
        self.device_func_infos: Dict[relax.Call, DeviceFuncInfo] = {}
        self.persistent_var_infos: Dict[str, PersistentVarInfo] = {}
        self.host_templates: List[Stmt] = []
        self.dyn_shared_mem_buffers = []
        self.event_buffers = {}
        self.ret_value_entries: Dict[relax.Call, List[int]] = defaultdict(list)
        self.new_ret_index = {}
        self.var_entry_mapping = DisjointSet()
        self.relax_var_to_entry = {}
        self.gvar_to_remove = list()

    def clear_state(self):
        # user might want to rewrite 2 megakernels, so clear state before each rewrite
        self.device_func_exec_scope = None
        self.device_func_infos = {}
        self.persistent_var_infos = {}
        self.host_templates = []
        self.dyn_shared_mem_buffers = []
        self.event_buffers = {}
        self.ret_value_entries = defaultdict(list)
        self.new_ret_index = {}
        self.var_entry_mapping = DisjointSet()
        self.relax_var_to_entry = {}
        self.cur_rewrite_func_name = None

    def transform(self) -> IRModule:
        """Entry point"""
        for g_var, func in self.mod.functions_items():
            if isinstance(func, relax.Function) and g_var.name_hint in self.rewrite_func_name:
                self.clear_state()
                self.cur_rewrite_func_name = g_var.name_hint
                self.builder_.update_func(g_var, self.visit_expr(func))
                # self.build_new_function(func)
        return self.builder_.finalize()

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
            shape = [s for s in event.struct_info.shape.values]
            self.event_buffers[event] = T.buffer(
                shape,
                event.struct_info.dtype,
                scope="global",
                buffer_name="event",
                offset_factor=1
            )
        return self.event_buffers[event]

    def _update_persistent_var_info(self, persistent_var_info_tile_class: PersistentVarInfo, persistent_var_info_megakernel: PersistentVarInfo, class_name: Optional[str]):
        if class_name is None:
            return
        # megakernel persistent var is mapped with ""
        for name, persistent_info in zip([class_name, ""], [persistent_var_info_tile_class, persistent_var_info_megakernel]):
            if name not in self.persistent_var_infos:
                self.persistent_var_infos[name] = persistent_info
            else:
                existing_info = self.persistent_var_infos[name]
                assert len(persistent_info.vars) == 0 or len(persistent_info.vars) == len(existing_info.vars), f"non empty persistent var num mismatch: {len(persistent_info.vars)} != {len(existing_info.vars)}"
                if len(persistent_info.vars) > 0:
                    for var_name, var_list in existing_info.vars.items():
                        assert var_name in persistent_info.vars, f"var {var_name} not found in persistent_var_info"
                        var_list.extend(persistent_info.vars[var_name])
                    tvm.ir.assert_structural_equal(persistent_info.init_body, existing_info.init_body)
                    tvm.ir.assert_structural_equal(persistent_info.finalize_body, existing_info.finalize_body)
            
            
    def visit_function_(self, func: relax.Function) -> Expr:
        for i, para in enumerate(func.params):
            assert isinstance(
                para, relax.Var
            ), f"expected Var for parameter {para}, but got {type(para)}"
            if isinstance(para.struct_info, TensorStructInfo):
                tir_var = T.var("handle", name=para.name_hint)
                shape = [s for s in para.struct_info.shape.values]
                tir_buf = T.buffer(
                    shape,
                    para.struct_info.dtype,
                    scope="global",
                    buffer_name=f"func_arg{i}",
                    offset_factor=1
                )
                var_entry = VarEntry(para, tir_var, tir_buf)
                self.var_entry_mapping.add_element(var_entry)
                self.relax_var_to_entry[var_entry.relax_var] = var_entry
        return super().visit_function_(func)

    def visit_call_(self, call: relax.Call) -> Expr:  # pylint: disable=arguments-renamed
        if call.op == self.call_tir_device_op:
            tile_num = call.args[2]
            assert len(tile_num) == 3, f"tile_num dimension mismatch: {len(tile_num)} != 3"
            tir_gvar = call.args[0]
            in_events = call.attrs.in_events
            out_events = call.attrs.out_events
            in_extra_tensors = call.attrs.in_extra_tensors
            out_extra_tensors = call.attrs.out_extra_tensors
            in_extra_tir_vars = call.attrs.in_extra_tir_vars
            out_extra_tir_vars = call.attrs.out_extra_tir_vars
            in_deps = call.attrs.in_deps
            out_deps = call.attrs.out_deps
            in_nums = call.attrs.in_nums
            out_nums = call.attrs.out_nums
            in_deps_dim = call.attrs.in_deps_dim
            out_deps_dim = call.attrs.out_deps_dim
            inplace_indices = list(int(i) for i in call.attrs.inplace_indices)
            prim_func = load_json(save_json(self.mod[tir_gvar]))
            self.gvar_to_remove.append(tir_gvar)
            # check if this device function is well-formed and has consistent exec scope
            verify_tirp_well_formed(prim_func, device_func=True)
            # collect event tensors and insert event commit/wait
            in_event_tensors = []
            out_event_tensors = []
            for e in in_events:
                in_event_tensors.append(self._get_event_buffer(e))
            for e in out_events:
                out_event_tensors.append(self._get_event_buffer(e))
            tile_idx = prim_func.params[-len(tile_num) :]
            body, host_template = HostStmtCollector.rewrite(prim_func.body)
            self.host_templates.append(host_template)
            exec_scope = body.block.exec_scope
            if self.device_func_exec_scope is None:
                self.device_func_exec_scope = exec_scope.name
            else:
                assert (
                    self.device_func_exec_scope == exec_scope.name
                ), f"device function exec scope mismatch: {self.device_func_exec_scope} != {exec_scope.name}"
            body, persistent_var_info_tile_class, persistent_var_info_megakernel = PersistentVarCollector.rewrite(body)
            func_type = prim_func.attrs.get("megakernel.device_func")
            job_type_id = prim_func.attrs.get("megakernel.job_type_id")
            wait_level = prim_func.attrs.get("megakernel.wait_level")
            notify_scope = prim_func.attrs.get("megakernel.notify_scope")
            notify_scope_id = prim_func.attrs.get("megakernel.notify_scope_id")
            self._update_persistent_var_info(persistent_var_info_tile_class, persistent_var_info_megakernel, func_type)
            
            dyn_shared_mem_buffer, body = DynSharedMemCollector.rewrite(body)
            if dyn_shared_mem_buffer is not None:
                self.dyn_shared_mem_buffers.append(dyn_shared_mem_buffer)
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
                self.relax_var_to_entry[var_entry.relax_var] = var_entry
                binding = self.builder_.lookup_binding(arg)
                if isinstance(binding, relax.Call):
                    self.var_entry_mapping.join(
                        var_entry, self.device_func_infos[binding].out_var_entries[0]
                    )
                elif isinstance(binding, relax.TupleGetItem):
                    tuple_binding = self.builder_.lookup_binding(binding.tuple_value)
                    if tuple_binding in self.device_func_infos:
                        self.var_entry_mapping.join(
                            var_entry,
                            self.device_func_infos[tuple_binding].out_var_entries[binding.index],
                        )
                elif binding is None:
                    pass
                else:
                    raise ValueError(f"Unsupported binding: {binding}, var: {arg}")
            output_num = self.get_output_num(call)
            non_inplace_output_num = 0
            for i in range(output_num):
                if inplace_indices[i] != -1:
                    var_entry = VarEntry(
                        None,
                        in_var_entries[inplace_indices[i]].tir_var,
                        in_var_entries[inplace_indices[i]].tir_buffer,
                    )
                    self.var_entry_mapping.add_element(var_entry)
                    self.relax_var_to_entry[var_entry.relax_var] = var_entry
                    self.var_entry_mapping.join(
                        var_entry, in_var_entries[inplace_indices[i]]
                    )
                else:
                    var_entry = VarEntry(
                        None,
                        prim_func.params[len(in_var_entries) + non_inplace_output_num],
                        prim_func.buffer_map[prim_func.params[len(in_var_entries) + non_inplace_output_num]],
                    )
                    non_inplace_output_num += 1
                    self.var_entry_mapping.add_element(var_entry)
                    self.relax_var_to_entry[var_entry.relax_var] = var_entry
                out_var_entries.append(var_entry)
                
            # handle the dep       
            sym_vars = set()     
            def _handle_dep(func: PrimFunc, extra_tensors: List[Expr], extra_tir_vars: List[PrimExpr], dim: int, accept_dep_idx: bool):
                var_replace_map = {}
                buffer_replace_map = {}
                
                def _replace_var(src: Var, dst: Var):
                    var_replace_map[src] = dst
                    
                def _replace_buffer(src: Buffer, dst: Buffer):
                    buffer_replace_map[src] = dst
                    for i in range(len(src.shape)):
                        if isinstance(src.shape[i], Var):
                            var_replace_map[src.shape[i]] = dst.shape[i]
                            
                # handle the tile_idx
                for i, var in enumerate(tile_idx):
                    _replace_var(func.params[i], var)
                # handle extra tensors
                extra_tensor_offset = len(tile_num) + 1 if accept_dep_idx else len(tile_num)
                for i, var in enumerate(extra_tensors):
                    if var in self.relax_var_to_entry:
                        var_entry = self.relax_var_to_entry[var]
                        _replace_var(func.params[extra_tensor_offset + i], var_entry.tir_var)
                        _replace_buffer(func.buffer_map[func.params[extra_tensor_offset + i]], var_entry.tir_buffer)
                    else:
                        binding = self.builder_.lookup_binding(var)
                        assert isinstance(binding, relax.TupleGetItem), f"unknown source of call_tir_device extra_args: {var} with binding {binding}"
                        var_entry = VarEntry(
                            var,
                            func.params[extra_tensor_offset + i],
                            func.buffer_map[func.params[extra_tensor_offset + i]],
                        )
                        self.var_entry_mapping.add_element(var_entry)
                        self.relax_var_to_entry[var] = var_entry
                    in_var_entries.append(var_entry)
                # handle extra tir sym vars
                extra_tir_var_offset = extra_tensor_offset + len(extra_tensors)
                for i, var in enumerate(extra_tir_vars):
                    _replace_var(func.params[extra_tir_var_offset + i], var)
                    sym_vars.add(var)
                # handle the output buffer
                out_buf = func.buffer_map[func.params[-1]]                    
                            
                if accept_dep_idx:
                    def lambda_func(dep_idx: T.Var):
                        replacer = BufferReplacer(
                            buffer_replace_map, var_replace_map | {func.params[len(tile_num)]: dep_idx}
                        )
                        stmt = replacer.visit_stmt(func.body)
                        T.add_to_parent(stmt)
                        return [BufferLoad(out_buf, [i]) for i in range(dim.value)]
                else:
                    def lambda_func():
                        replacer = BufferReplacer(buffer_replace_map, var_replace_map)
                        stmt = replacer.visit_stmt(func.body)
                        T.add_to_parent(stmt)
                        return [BufferLoad(out_buf, [i]) for i in range(dim.value)]       
                                 
                return lambda_func, out_buf
            
            in_info_list = []
            out_info_list = []
            for idx in range(len(in_deps)):
                in_dep_func, in_dep_buf = _handle_dep(in_deps[idx], in_extra_tensors[idx], in_extra_tir_vars[idx], in_deps_dim[idx], True)
                in_num_func, in_num_buf = _handle_dep(in_nums[idx], in_extra_tensors[idx], in_extra_tir_vars[idx], T.int32(1), False)
                in_info_list.append((in_dep_func, in_num_func, in_dep_buf, in_num_buf))
            for idx in range(len(out_deps)):
                out_dep_func, out_dep_buf = _handle_dep(out_deps[idx], out_extra_tensors[idx], out_extra_tir_vars[idx], out_deps_dim[idx], True)
                out_num_func, out_num_buf = _handle_dep(out_nums[idx], out_extra_tensors[idx], out_extra_tir_vars[idx], T.int32(1), False)
                out_info_list.append((out_dep_func, out_num_func, out_dep_buf, out_num_buf))
            self.device_func_infos[call] = DeviceFuncInfo(
                body, in_var_entries, out_var_entries, tile_idx, in_event_tensors, out_event_tensors, sym_vars,
                in_info_list, out_info_list, func_type, job_type_id, wait_level, notify_scope, notify_scope_id
            )
            return super().visit_call_(call)
        else:
            raise ValueError(
                "StaticHorizontalFusion can only be applied to relax function with only call_tir_device"
            )

    def _build_persistent_kernel(self, buffer_replace_map: Dict[relax.Call, Dict[Buffer, Buffer]], var_replace_map: Dict[Var, Var]):
        class PersistentVars:
            def __init__(self, persistent_var_infos: List[PersistentVarInfo], smem_manager: SmemManager):
                nonlocal buffer_replace_map
                self.buffers = []
                self.inits = []
                self.finalizes = []
                for persistent_var_info in persistent_var_infos:
                    for buffer_name, old_buffers in persistent_var_info.vars.items():
                        scope_resolve_table = {
                            "shared.persistent": "shared.dyn",
                            "local.persistent": "local",
                        }
                        if old_buffers[0].scope() == "shared.persistent":
                            new_buffer = smem_manager.alloc(old_buffers[0].shape, old_buffers[0].dtype, scope=scope_resolve_table[old_buffers[0].scope()], align=old_buffers[0].data_alignment, layout=old_buffers[0].layout, method="persistent").buffer
                            self.buffers.append(new_buffer)
                            for old_buffer in old_buffers:
                                buffer_replace_map[old_buffer] = new_buffer
                        else:
                            new_buffer = T.alloc_buffer(old_buffers[0].shape, old_buffers[0].dtype, scope=scope_resolve_table[old_buffers[0].scope()], align=old_buffers[0].data_alignment, layout=old_buffers[0].layout, name=buffer_name)
                            self.buffers.append(new_buffer)
                            for old_buffer in old_buffers:
                                buffer_replace_map[old_buffer] = new_buffer
                for persistent_var_info in persistent_var_infos:
                    replacer = BufferReplacer(
                        buffer_map=buffer_replace_map,
                        var_map=var_replace_map,
                    )
                    self.inits.append(replacer.visit_stmt(persistent_var_info.init_body))
                    self.finalizes.append(replacer.visit_stmt(persistent_var_info.finalize_body))

            def init(self):
                for init in self.inits:
                    T.add_to_parent(init)

            def finalize(self):
                for finalize in self.finalizes:
                    T.add_to_parent(finalize)

        def switch_task_type(tile_scheduler: TileSchedulerBase):
            idxs, type = tile_scheduler.get_idx_and_task_type()
            total_type_count = len(self.device_func_infos)
            if_frames = [T.If(type == info.job_type_id) for info in self.device_func_infos.values()]
            then_frames = [T.Then() for i in range(total_type_count)]
            else_frames = [T.Else() for i in range(total_type_count - 1)]
            for i, (call, func) in enumerate(self.device_func_infos.items()):
                if_frames[i].__enter__()
                new_body = EventOpInserter.rewrite(func, tile_scheduler, self.semaphore_class)
                needed_idx_num = len(func.tile_idx)
                replacer = BufferReplacer(
                    buffer_map=buffer_replace_map,
                    var_map=var_replace_map | {func.tile_idx[j]: idxs[j] for j in range(needed_idx_num)},
                )
                new_body = replacer.visit_stmt(new_body)
                with then_frames[i]:
                    T.add_to_parent(new_body)
                if i < total_type_count - 1:
                    else_frames[i].__enter__()

            for i in range(total_type_count - 1, -1, -1):
                if i < total_type_count - 1:
                    else_frames[i].__exit__(None, None, None)
                if_frames[i].__exit__(None, None, None)

        def replace_dyn_shared_mem_buffer(new_buffer):
            for buffer in self.dyn_shared_mem_buffers:
                buffer_replace_map[buffer] = new_buffer

        @T.prim_func(tirp=True)
        def persistent_kernel(queue: T.buffer((KernelConfig.SM_NUMBER, self.tile_scheduler_class.MAX_TASKS), "int32")):
            with T.kernel():
                bx = T.scope_id([KernelConfig.SM_NUMBER], parent="kernel", cur=self.device_func_exec_scope)
                tid = T.thread_id([KernelConfig.NUM_THREADS], parent="cta")
                smem_buffer = T.alloc_buffer(KernelConfig.MAX_SMEM_SIZE, "uint8", scope="shared.dyn", align=16)
                replace_dyn_shared_mem_buffer(smem_buffer)
                smem_manager = T.meta_var(SmemManager(KernelConfig.MAX_SMEM_SIZE, 16384, smem_buffer.data))
                # TODO: remove tile scheduler from arg
                tile_scheduler = T.meta_var(self.tile_scheduler_class("mega_", queue, smem_manager))
                persistent_vars = T.meta_var(PersistentVars(list(self.persistent_var_infos.values()), smem_manager))
                persistent_vars.init()
                tile_scheduler.init()
                while tile_scheduler.valid():
                    switch_task_type(tile_scheduler)
                    tile_scheduler.next_tile()
                persistent_vars.finalize()
        new_body = persistent_kernel.body
        for template in self.host_templates:
            template = BufferReplacer(
                buffer_map=buffer_replace_map,
                var_map=var_replace_map,
            ).visit_stmt(template)
            new_body = seek_kernel_replace_point(template, new_body)
        sym_var_map = {}
        for info in self.device_func_infos.values():
            for var in info.sym_vars:
                if var not in sym_var_map:
                    sym_var_map[var] = Var("sym", var.dtype)
        new_sym_vars = [sym_var_map[var] for var in sym_var_map.keys()]
        old_sym_vars = list(sym_var_map.keys())
        new_body = BufferReplacer(var_map=sym_var_map).visit_stmt(new_body)
        return persistent_kernel.with_body(new_body), new_sym_vars, old_sym_vars

    def _build_gen_exec_queue_kernel(self):
        THREAD_NUM = 256
        assert len(self.device_func_infos) <= KernelConfig.SM_NUMBER
        var_collector = VarCollector()
        def emit_task_round_robin(queue, idx, bx, tx):
            for i, (call, device_func_info) in enumerate(self.device_func_infos.items()):
                tile_num = call.args[2]
                with T.If(bx == i):
                    with T.Then():
                        with T.If(tx == 0):
                            with T.Then():     
                                for e in tile_num.values:
                                    var_collector.visit_expr(e)
                                for_grid = T.grid(*tile_num.values)
                                vars = for_grid.__enter__()
                                packed_val = pack_into_32bit(vars[0], vars[1], vars[2], device_func_info.job_type_id, host=False)
                                T.buffer_store(queue, packed_val, [idx[0] % KernelConfig.SM_NUMBER, idx[0] // KernelConfig.SM_NUMBER])
                                T.buffer_store(idx, idx[0] + 1, 0)
                                for_grid.__exit__(None, None, None)
                    with T.Else():
                        T.buffer_store(idx, idx[0] + tile_num[0] * tile_num[1] * tile_num[2], 0)
            with T.If(bx == len(self.device_func_infos)):
                with T.Then():
                    with T.If(tx == 0):
                        with T.Then():
                            with T.serial(0, KernelConfig.SM_NUMBER):
                                packed_val = pack_into_32bit(0, 0, 0, JobType.END.value, host=False)
                                T.buffer_store(queue, packed_val, [idx[0] % KernelConfig.SM_NUMBER, idx[0] // KernelConfig.SM_NUMBER])
                                T.buffer_store(idx, idx[0] + 1, 0)

        @T.prim_func(tirp=True)
        def gen_exec_queue(queue: T.buffer((KernelConfig.SM_NUMBER, self.tile_scheduler_class.MAX_TASKS,), "int32")):
            with T.kernel():
                bx = T.cta_id([KernelConfig.SM_NUMBER], parent="kernel")
                tx = T.thread_id([THREAD_NUM], parent="cta")
                idx = T.alloc_buffer([1], "int32", scope="local")
                idx[0] = 0
                emit_task_round_robin(queue, idx, bx, tx)

        new_vars = [Var("sym", v.dtype) for v in var_collector.vars]
        replace = BufferReplacer(
            var_map={v: new_vars[i] for i, v in enumerate(var_collector.vars)},
        )
        new_body = replace.visit_stmt(gen_exec_queue.body)
        func_with_sym_vars = PrimFunc(
            params=list(gen_exec_queue.params) + new_vars,
            body=new_body,
            ret_type=gen_exec_queue.ret_type,
            buffer_map=gen_exec_queue.buffer_map,
            attrs=gen_exec_queue.attrs,
            span=gen_exec_queue.span,
        )
        return func_with_sym_vars, list(var_collector.vars)

    def merge_function(self):
        new_relax_params = []
        new_tir_params = []
        new_buffer_map = {}
        out_sinfo = []
        inplace_indices = []
        buffer_replace_map = {}
        var_replace_map = {}

        def replace_buffer(src: Buffer, dst: Buffer):
            buffer_replace_map[src] = dst
            for i in range(len(src.shape)):
                if isinstance(src.shape[i], Var):
                    var_replace_map[src.shape[i]] = dst.shape[i]

        input_var_entries_after_merge = []
        output_var_entries_after_merge = []

        def connected_with_any_entries(e, entries):
            for i, input_entry in enumerate(entries):
                if self.var_entry_mapping.is_connected(e, input_entry):
                    return i
            return None
        
        visited_entry_group = set()
        for call, func in self.device_func_infos.items():
            for entry in func.in_var_entries:
                leader_entry = self.var_entry_mapping.find(entry)
                if leader_entry not in visited_entry_group:
                    visited_entry_group.add(leader_entry)
                    # this is an input tensor after merging
                    new_relax_params.append(entry.relax_var)
                    new_tir_params.append(leader_entry.tir_var)
                    new_buffer_map[leader_entry.tir_var] = leader_entry.tir_buffer
                    input_var_entries_after_merge.append(entry)
                replace_buffer(entry.tir_buffer, leader_entry.tir_buffer)

            for entry in func.out_var_entries:
                leader_entry = self.var_entry_mapping.find(entry)
                visited_entry_group.add(leader_entry)
                
        for e in self.event_buffers:
            event_tensor_var = T.var("handle", name=e.name_hint)
            new_buffer_map[event_tensor_var] = self.event_buffers[e]
            new_tir_params.append(event_tensor_var)
            new_relax_params.append(e)
        total_input_num = len(new_relax_params)

        for call, func in self.device_func_infos.items():
            self.new_ret_index[call] = {}
            if len(self.ret_value_entries[call]) == 1 and self.ret_value_entries[call][0] == -1:
                call_ret_indices = list(range(len(func.out_var_entries)))
            else:
                call_ret_indices = self.ret_value_entries[call]
            call_out_sinfo = self.flatten_sinfo(call.struct_info)
            for i, entry in enumerate(func.out_var_entries):
                leader_entry = self.var_entry_mapping.find(entry)
                replace_buffer(entry.tir_buffer, leader_entry.tir_buffer)
                input_connected = connected_with_any_entries(entry, input_var_entries_after_merge)
                output_connected = connected_with_any_entries(entry, output_var_entries_after_merge)
                if i in call_ret_indices:
                    if output_connected is None:
                        self.new_ret_index[call][i] = len(output_var_entries_after_merge)
                    else:
                        self.new_ret_index[call][i] = output_connected
                if output_connected is None:
                    out_sinfo.append(call_out_sinfo[i])
                    output_var_entries_after_merge.append(entry)
                    new_buffer_map[leader_entry.tir_var] = leader_entry.tir_buffer
                    if input_connected is None:
                        inplace_indices.append(-1)
                        new_tir_params.append(leader_entry.tir_var)
                    else:
                        inplace_indices.append(input_connected)

        persistent_kernel, persistent_kernel_new_sym_vars, persistent_kernel_sym_vars = self._build_persistent_kernel(buffer_replace_map, var_replace_map)
        new_body = persistent_kernel.body
        new_tir_params.insert(total_input_num, persistent_kernel.params[0])
        new_tir_params.extend(persistent_kernel_new_sym_vars)
        new_buffer_map.update(persistent_kernel.buffer_map)
        new_prim_func = PrimFunc(
            new_tir_params,
            new_body,
            None,
            new_buffer_map,
            tvm.ir.make_node(
                "ir.DictAttrs",
                is_tirp=True,
                global_symbol=f"persistent_kernel_{self.cur_rewrite_func_name}",
            ),
        )
        new_gvar = self.builder_.add_func(new_prim_func, f"persistent_kernel_{self.cur_rewrite_func_name}")
        queue_buffer = list(persistent_kernel.buffer_map.values())[0]
        gen_exec_queue_tir, sym_vars = self._build_gen_exec_queue_kernel()
        gen_exec_queue_gvar = self.builder_.add_func(gen_exec_queue_tir, f"gen_exec_queue_{self.cur_rewrite_func_name}")
        exec_queue = self.builder_.emit(relax.call_tir(gen_exec_queue_gvar, [], out_sinfo=TensorStructInfo(shape=queue_buffer.shape, dtype=queue_buffer.dtype), tir_vars=sym_vars))
        new_relax_params.append(
            exec_queue
        )
        if all(i == -1 for i in inplace_indices):
            return relax.call_tir(new_gvar, new_relax_params, out_sinfo, tir_vars=persistent_kernel_sym_vars)
        return relax.call_tir_inplace(
            new_gvar, new_relax_params, inplace_indices, out_sinfo, tir_vars=persistent_kernel_sym_vars
        )

    def visit_seq_expr_(self, op: relax.SeqExpr) -> Expr:
        op = super().visit_seq_expr_(op)  
        ret_order = []
        def _handle_var(var: relax.Var):
            binding = self.builder_.lookup_binding(var)
            if isinstance(binding, relax.Call):
                self.ret_value_entries[binding].append(-1)
                ret_order.append((binding, -1))
            elif isinstance(binding, relax.TupleGetItem):
                call_binding = self.builder_.lookup_binding(binding.tuple_value)
                if isinstance(call_binding, relax.Call):
                    self.ret_value_entries[call_binding].append(binding.index)
                    ret_order.append((call_binding, binding.index))
                else:
                    raise ValueError(f"Unsupported binding: {call_binding}")
            elif isinstance(binding, relax.expr.DataflowVar):
                binding = self.builder_.lookup_binding(binding)
                if isinstance(binding, relax.Call):
                    self.ret_value_entries[binding].append(0)
                    ret_order.append((binding, -1))
                else:
                    raise ValueError(f"Unsupported binding: {binding}")
            else:
                raise ValueError(f"Unsupported binding: {binding}")

        if isinstance(op.body, relax.Var):
            _handle_var(op.body)
        elif isinstance(op.body, relax.Tuple):
            for field in op.body.fields:
                _handle_var(field)
        else:
            raise ValueError(f"Unsupported body: {op.body}")
        self.builder_._begin_binding_block()
        call_megakernel = self.builder_.emit(self.merge_function())
        if isinstance(call_megakernel.struct_info, TensorStructInfo):
            ret = call_megakernel
        else:
            ret_vars = []
            for (call, idx) in ret_order:
                if idx == -1:
                    # all the tir_call results will be returned
                    new_ret_index_call = self.new_ret_index[call]
                    assert len(new_ret_index_call) == len(self.device_func_infos[call].out_var_entries)
                    if len(new_ret_index_call) == 1:
                        ret_vars.append(self.builder_.emit(relax.TupleGetItem(call_megakernel, new_ret_index_call[0])))
                    else:
                        ret_tuple = relax.Tuple([self.builder_.emit(relax.TupleGetItem(call_megakernel, new_ret_index_call[idx])) for idx in sorted(new_ret_index_call.keys())])
                        ret_vars.append(ret_tuple)
                else:
                    if idx in self.new_ret_index[call]:
                        ret_vars.append(self.builder_.emit(relax.TupleGetItem(call_megakernel, self.new_ret_index[call][idx])))
                    else:
                        raise ValueError(f"Call {call} with index {idx} not found in new_ret_index")
            if len(ret_vars) == 1:
                ret = ret_vars[0]
            else:
                ret = relax.Tuple(ret_vars)
        new_block = self.builder_._end_block()
        return relax.SeqExpr(op.blocks + [new_block], ret)
