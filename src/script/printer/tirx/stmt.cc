/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
#include "../../../tirx/transform/ir_utils.h"  // For `GetPtrStorageScope`
#include "./utils.h"

namespace tvm {
namespace script {
namespace printer {

Doc DoConciseScoping(const ffi::Optional<ExprDoc>& lhs, const ExprDoc& rhs,
                     ffi::Array<StmtDoc>* stmts, bool concise_scoping) {
  if (concise_scoping) {
    if (lhs.defined()) {
      stmts->insert(stmts->begin(), AssignDoc(lhs.value(), rhs, std::nullopt));
    } else {
      stmts->insert(stmts->begin(), ExprStmtDoc(rhs));
    }
    return StmtBlockDoc(*stmts);
  } else {
    return ScopeDoc(lhs, rhs, *stmts);
  }
}

bool AllowConciseScoping(const IRDocsifier& d, const ObjectRef& obj) {
  if (d->cfg.defined()) {
    if (d->cfg->obj_to_annotate.count(obj)) {
      // if the object requires annotation, do not fold this frame
      return false;
    }
  }
  TVM_FFI_ICHECK(!d->frames.empty());
  if (const auto* f = d->frames.back().as<TIRFrameNode>()) {
    return f->allow_concise_scoping;
  }
  TVM_FFI_THROW(NotImplementedError) << "fragment printing";
  TVM_FFI_UNREACHABLE();
}

bool IsAncestorOfAllVarUse(const tirx::Stmt& node, const ObjectRef& var, const IRDocsifier& d) {
  if (!d->common_prefix.count(var.get())) {
    return false;
  }
  const std::vector<const Object*>& path = d->common_prefix.at(var.get());
  for (auto it = path.rbegin(); it != path.rend(); ++it) {
    if (*it == node.get()) {
      return true;
    }
  }
  return false;
}

ffi::Optional<PrimExpr> FindReturnValue(const tirx::Stmt& node) {
  auto eval = node.as<tirx::EvaluateNode>();
  if (!eval) return std::nullopt;

  auto call = eval->value.as<tirx::CallNode>();
  if (!call) return std::nullopt;

  if (!call->op.same_as(tirx::builtin::ret())) return std::nullopt;

  if (call->args.size() != 1) return std::nullopt;

  return call->args[0];
}

ExprDoc BulkGroupEventDecl(const tirx::BulkGroupEvent& event, const ffi::String& method,
                           const AccessPath& p, const IRDocsifier& d) {
  return TIRp(d, method)->Call({LiteralDoc::Int(static_cast<int64_t>(event->impl), p->Attr("impl")),
                                d->AsDoc<ExprDoc>(event->state, p->Attr("state"))});
}

ExprDoc HandleEvent(const tirx::BulkGroupEvent& event, const ffi::String& method,
                    const AccessPath& p, const IRDocsifier& d) {
  if (!d->IsVarDefined(event)) {
    if (ffi::Optional<Frame> opt_f = FindLowestVarDef(event, d)) {
      ExprDoc lhs = DefineBulkEvent(event, opt_f.value(), d);
      ExprDoc rhs = BulkGroupEventDecl(event, method, p, d);
      opt_f.value()->stmts.push_back(AssignDoc(lhs, rhs, std::nullopt));
    }
  }
  if (ffi::Optional<ExprDoc> doc = d->GetVarDoc(event)) {
    return doc.value();
  }
  LOG(FATAL) << "IndexError: Variable is not defined in the environment: " << event;
}

ExprDoc SemaphoreEventTensorDecl(const tirx::SemaphoreEventTensor& event_tensor,
                                 const ffi::String& method, const AccessPath& p,
                                 const IRDocsifier& d) {
  return TIRp(d, method)->Call({
      LiteralDoc::Int(static_cast<int64_t>(event_tensor->impl), p->Attr("impl")),
      d->AsDoc<ExprDoc>(event_tensor->state, p->Attr("state")),
      d->AsDoc<ExprDoc>(event_tensor->shape, p->Attr("shape")),
  });
}

ExprDoc HandleEventTensor(const tirx::SemaphoreEventTensor& event_tensor, const ffi::String& method,
                          const AccessPath& p, const IRDocsifier& d) {
  if (!d->IsVarDefined(event_tensor)) {
    if (ffi::Optional<Frame> opt_f = FindLowestVarDef(event_tensor, d)) {
      ExprDoc lhs = DefineSemaphoreEventTensor(event_tensor, opt_f.value(), d);
      ExprDoc rhs = SemaphoreEventTensorDecl(event_tensor, method, p, d);
      opt_f.value()->stmts.push_back(AssignDoc(lhs, rhs, std::nullopt));
    }
  }
  if (ffi::Optional<ExprDoc> doc = d->GetVarDoc(event_tensor)) {
    return doc.value();
  }
  LOG(FATAL) << "IndexError: Variable is not defined in the environment: " << event_tensor;
}

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tirx::BulkGroupEvent>("",
                                       [](tirx::BulkGroupEvent event, AccessPath p,
                                          IRDocsifier d) -> Doc {
                                         return HandleEvent(event, "BulkGroupEvent", p, d);
                                       });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tirx::SemaphoreEventTensor>(
        "", [](tirx::SemaphoreEventTensor event_tensor, AccessPath p, IRDocsifier d) -> Doc {
          return HandleEventTensor(event_tensor, "SemaphoreEventTensor", p, d);
        });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tirx::SemaphoreEventTensorItem>(
        "", [](tirx::SemaphoreEventTensorItem item, AccessPath p, IRDocsifier d) -> Doc {
          const auto& e_tensor_doc = d->AsDoc<ExprDoc>(item->tensor, p->Attr("tensor"));
          ffi::Array<Doc> indices_doc;
          for (size_t i = 0, n = item->indices.size(); i < n; ++i) {
            indices_doc.push_back(
                d->AsDoc<Doc>(item->indices[i], p->Attr("indices")->ArrayItem(i)));
          }
          return e_tensor_doc->operator[](indices_doc);
        });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tirx::tirp::OpCall>(
        "", [](tirx::tirp::OpCall op_call, AccessPath p, IRDocsifier d) -> Doc {
          static const OpAttrMap<tirx::TScriptPrinterName>& op_names =
              Op::GetAttrMap<tirx::TScriptPrinterName>("TScriptPrinterName");
          auto op = op_call->op;
          if (op_names.count(op) == 0) {
            LOG(WARNING) << "No TScriptPrinterName attribute for " << op->name;
          }

          auto print_member_function_call = [&](std::string method) {
            ffi::Array<Doc> args;
            for (size_t i = 1, n = op_call->args.size(); i < n; ++i) {
              args.push_back(d->AsDoc<Doc>(op_call->args[i], p->Attr("args")->ArrayItem(i)));
            }
            return OpCallDoc(
                AttrAccessDoc(d->AsDoc<ExprDoc>(op_call->args[0], p->Attr("args")->ArrayItem(0)),
                              method),
                args, {}, {});
          };

          static const auto& tirp_op_map = Op::GetAttrMap<Bool>("TIsTIRpOp");
          static const auto& schedule_op_map = Op::GetAttrMap<Bool>("TIsScheduleOp");
          static const auto& compose_op_map = Op::GetAttrMap<Bool>("TIsComposeOp");
          static const auto& event_op_map = Op::GetAttrMap<Bool>("TIsEventOp");
          static const auto& async_op_map = Op::GetAttrMap<Bool>("TIsAsyncOp");
          ICHECK(bool(tirp_op_map.get(op, tvm::Bool(false))))
              << "Only TIR+ ops can be used in tirx::tirp::OpCall";
          ffi::String name = op_names.get(op, op->name);
          if (bool(schedule_op_map.get(op, tvm::Bool(false))) ||
              bool(async_op_map.get(op, tvm::Bool(false)))) {
            // Schedule ops
            ffi::Array<Doc> args;
            for (size_t i = 0, n = op_call->args.size(); i < n; ++i) {
              args.push_back(d->AsDoc<Doc>(op_call->args[i], p->Attr("args")->ArrayItem(i)));
            }
            return OpCallDoc(
                TIRp(d, name), args, d->AsDoc<DictDoc>(op_call->workspace, p->Attr("workspace")),
                d->AsDoc<DictDoc>(op_call->schedule_config, p->Attr("schedule_config")));
          } else if (bool(compose_op_map.get(op, tvm::Bool(false)))) {
            // Compose ops
            With<TIRFrame> f(d, op_call);
            ffi::Array<tirx::Stmt> stmts;
            for (size_t i = 0, n = op_call->args.size(); i < n; ++i) {
              stmts.push_back(Downcast<tirx::Stmt>(op_call->args[i]));
            }
            tirx::SeqStmt seq_stmt(stmts);
            AsDocBody(seq_stmt, p->Attr("args"), f->get(), d);
            return ScopeDoc(std::nullopt,
                            TIRp(d, "compose_op")
                                ->Call({}, {"workspace", "schedule_config"},
                                       {d->AsDoc<DictDoc>(op_call->workspace, p->Attr("workspace")),
                                        d->AsDoc<DictDoc>(op_call->schedule_config,
                                                          p->Attr("schedule_config"))}),
                            (*f)->stmts);
          } else if (bool(event_op_map.get(op, tvm::Bool(false)))) {
            // Event ops
            ICHECK(op_call->args[0].as<tirx::SemaphoreEventTensorNode>() ||
                   op_call->args[0].as<tirx::BaseEventNode>())
                << "First argument must be a SemaphoreEventTensor or BulkGroupEvent";
            // event_method_name
            std::string method = std::string(name).substr(6);
            return print_member_function_call(method);
          } else {
            // Misc ops
            ffi::Array<Doc> args;
            for (size_t i = 0, n = op_call->args.size(); i < n; ++i) {
              args.push_back(d->AsDoc<Doc>(op_call->args[i], p->Attr("args")->ArrayItem(i)));
            }
            return OpCallDoc(TIRp(d, name), args, {}, {});
          }
        });
TVM_SCRIPT_REPR(tirx::tirp::OpCallNode, ReprPrintTIR);

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tirx::Evaluate>("", [](tirx::Evaluate eval, AccessPath p, IRDocsifier d) -> Doc {
      if (d->cfg->syntax_sugar) {
        if (auto return_value = FindReturnValue(eval)) {
          ExprDoc value =
              d->AsDoc<ExprDoc>(return_value.value(), p->Attr("value")->Attr("args")->ArrayItem(0));
          return ReturnDoc(value);
        }
      }

      ExprDoc value = d->AsDoc<ExprDoc>(eval->value, p->Attr("value"));
      if (eval->value->IsInstance<tirx::CallNode>()) {
        return ExprStmtDoc(value);
      }
      return ExprStmtDoc(TIR(d, "evaluate")->Call({value}));
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tirx::Bind>("", [](tirx::Bind stmt, AccessPath p, IRDocsifier d) -> Doc {
      // Step 1. Type annotation
      ICHECK(stmt->var->type_annotation.defined())
          << "Type annotation is required for variable: " << stmt->var->name_hint;
      ffi::Optional<ExprDoc> type_doc = d->AsDoc<ExprDoc>(stmt->var->type_annotation,  //
                                                          p->Attr("var")->Attr("type_annotation"));
      if (const auto* tuple_type = stmt->var->type_annotation.as<TupleTypeNode>()) {
        if (tuple_type->fields.empty()) {
          type_doc = std::nullopt;
        }
      }
      // Step 2. RHS
      ExprDoc rhs = d->AsDoc<ExprDoc>(stmt->value, p->Attr("value"));
      // Step 3. LHS - Bind is flat, define var if new, otherwise just assign
      if (!d->IsVarDefined(stmt->var)) {
        TVM_FFI_ICHECK(!d->frames.empty());
        ExprDoc lhs = DefineVar(stmt->var, d->frames.back(), d);
        return AssignDoc(lhs, rhs, type_doc);
      } else {
        ExprDoc lhs = d->AsDoc<ExprDoc>(stmt->var, p->Attr("var"));
        return AssignDoc(lhs, rhs, std::nullopt);
      }
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tirx::AssertStmt>(
        "", [](tirx::AssertStmt stmt, AccessPath p, IRDocsifier d) -> Doc {
          ExprDoc cond = d->AsDoc<ExprDoc>(stmt->condition, p->Attr("condition"));
          // Always emit the canonical tuple form: assert cond, ("Kind", ["part0", "part1", ...])
          ffi::Array<ExprDoc> parts;
          auto parts_path = p->Attr("message_parts");
          for (size_t i = 0; i < stmt->message_parts.size(); ++i) {
            parts.push_back(d->AsDoc<ExprDoc>(stmt->message_parts[i], parts_path->ArrayItem(i)));
          }
          ExprDoc kind_doc = d->AsDoc<ExprDoc>(stmt->error_kind, p->Attr("error_kind"));
          return AssertDoc(cond, TupleDoc({kind_doc, ListDoc(parts)}));
        });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tirx::While>("", [](tirx::While stmt, AccessPath p, IRDocsifier d) -> Doc {
      ExprDoc cond = d->AsDoc<ExprDoc>(stmt->condition, p->Attr("condition"));
      With<TIRFrame> f(d, stmt);
      AsDocBody(stmt->body, p->Attr("body"), f->get(), d);
      return WhileDoc(cond, (*f)->stmts);
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tirx::Break>("", [](tirx::Break stmt, AccessPath p, IRDocsifier d) -> Doc {
      return BreakDoc();
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tirx::Continue>("", [](tirx::Continue stmt, AccessPath p, IRDocsifier d) -> Doc {
      return ContinueDoc();
    });

namespace {
Doc DeclBufferDoc(tirx::DeclBuffer stmt, AccessPath p, IRDocsifier d,
                  BufferVarDefinition var_definitions) {
  ExprDoc rhs = BufferDecl(stmt->buffer, "decl_buffer", {}, p->Attr("buffer"), d->frames.back(), d,
                           var_definitions);
  ExprDoc lhs = DefineBuffer(stmt->buffer, d->frames.back(), d);
  return AssignDoc(lhs, rhs, std::nullopt);
}
}  // namespace

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tirx::DeclBuffer>(  //
        "", [](tirx::DeclBuffer stmt, AccessPath p, IRDocsifier d) -> Doc {
          return DeclBufferDoc(stmt, p, d, BufferVarDefinition::None);
        });

namespace {
Doc AllocBufferDoc(tirx::AllocBuffer stmt, AccessPath p, IRDocsifier d) {
  bool concise = AllowConciseScoping(d, stmt);
  ExprDoc rhs = BufferDecl(stmt->buffer, "alloc_buffer", {}, p->Attr("buffer"), d->frames.back(), d,
                           BufferVarDefinition::DataPointer);
  With<TIRFrame> f(d, stmt);
  ExprDoc lhs = DefineBuffer(stmt->buffer, *f, d);
  AsDocBody(stmt->body, p->Attr("body"), f->get(), d);
  return DoConciseScoping(lhs, rhs, &(*f)->stmts, concise);
}
}  // namespace

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tirx::AllocBuffer>(  //
        "", [](tirx::AllocBuffer stmt, AccessPath p, IRDocsifier d) -> Doc {
          return AllocBufferDoc(stmt, p, d);
        });

namespace {
Doc AllocBulkGroupEventDoc(tirx::AllocBulkGroupEvent stmt, AccessPath p, IRDocsifier d) {
  bool concise = AllowConciseScoping(d, stmt);
  tirx::BulkGroupEvent event = stmt->bulk_group_event;
  ExprDoc rhs = BulkGroupEventDecl(event, "alloc_bulk_group_event", p->Attr("bulk_group_event"), d);
  With<TIRFrame> f(d, stmt);
  ExprDoc lhs = DefineBulkEvent(event, *f, d);
  AsDocBody(stmt->body, p->Attr("body"), f->get(), d);
  return DoConciseScoping(lhs, rhs, &(*f)->stmts, concise);
}
}  // namespace

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tirx::AllocBulkGroupEvent>(  //
        "", [](tirx::AllocBulkGroupEvent stmt, AccessPath p, IRDocsifier d) -> Doc {
          return AllocBulkGroupEventDoc(stmt, p, d);
        });

namespace {
Doc AllocSemaphoreEventTensorDoc(tirx::AllocSemaphoreEventTensor stmt, AccessPath p, IRDocsifier d) {
  bool concise = AllowConciseScoping(d, stmt);
  tirx::SemaphoreEventTensor event_tensor = stmt->sem_event_tensor;
  ExprDoc rhs = SemaphoreEventTensorDecl(event_tensor, "alloc_semaphore_event_tensor",
                                         p->Attr("sem_event_tensor"), d);
  With<TIRFrame> f(d, stmt);
  ExprDoc lhs = DefineSemaphoreEventTensor(event_tensor, *f, d);
  AsDocBody(stmt->body, p->Attr("body"), f->get(), d);
  return DoConciseScoping(lhs, rhs, &(*f)->stmts, concise);
}
}  // namespace

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tirx::AllocSemaphoreEventTensor>(  //
        "", [](tirx::AllocSemaphoreEventTensor stmt, AccessPath p, IRDocsifier d) -> Doc {
          return AllocSemaphoreEventTensorDoc(stmt, p, d);
        });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tirx::IfThenElse>(  //
        "", [](tirx::IfThenElse stmt, AccessPath p, IRDocsifier d) -> Doc {
          ExprDoc cond = d->AsDoc<ExprDoc>(stmt->condition, p->Attr("condition"));
          ffi::Array<StmtDoc> then_branch;
          ffi::Array<StmtDoc> else_branch;
          if (stmt->then_case.defined()) {
            With<TIRFrame> f(d, stmt->then_case);
            AsDocBody(stmt->then_case, p->Attr("then_case"), f->get(), d);
            then_branch = (*f)->stmts;
          }
          if (stmt->else_case.defined()) {
            With<TIRFrame> f(d, stmt->else_case);
            AsDocBody(stmt->else_case.value(), p->Attr("else_case"), f->get(), d);
            else_branch = (*f)->stmts;
          }
          return IfDoc(cond, then_branch, else_branch);
        });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tirx::SeqStmt>("", [](tirx::SeqStmt stmt, AccessPath p, IRDocsifier d) -> Doc {
      With<TIRFrame> f(d, stmt);
      AsDocBody(stmt, p, f->get(), d);
      return StmtBlockDoc((*f)->stmts);
    });

void InsertEnvThread(const tirx::IterVar& iter_var, const AccessPath& iter_var_p,
                     const IRDocsifier& d) {
  Frame f = FindLowestVarDef(iter_var->var, d).value();
  DefineVar(iter_var->var, f, d);
  ExprDoc rhs = TIR(d, "env_thread")
                    ->Call({LiteralDoc::Str(iter_var->thread_tag,  //
                                            iter_var_p->Attr("thread_tag"))});
  ExprDoc lhs = d->AsDoc<ExprDoc>(iter_var->var, iter_var_p->Attr("var"));
  f->stmts.push_back(AssignDoc(lhs, rhs, std::nullopt));
}

ExprDoc DocsifyLaunchThread(const tirx::AttrStmt& attr_stmt, const AccessPath& attr_stmt_p,
                            ffi::Optional<tirx::Var>* define_var, const IRDocsifier& d) {
  tirx::IterVar iter_var = Downcast<tirx::IterVar>(attr_stmt->node);
  AccessPath iter_var_p = attr_stmt_p->Attr("node");

  ExprDoc var_doc{ffi::UnsafeInit()};
  if (d->IsVarDefined(iter_var->var)) {
    var_doc = d->AsDoc<ExprDoc>(iter_var->var, iter_var_p->Attr("var"));
  } else if (IsAncestorOfAllVarUse(attr_stmt, iter_var->var, d)) {
    var_doc = LiteralDoc::Str(iter_var->thread_tag, iter_var_p->Attr("thread_tag"));
    *define_var = iter_var->var;
  } else {
    InsertEnvThread(iter_var, iter_var_p, d);
    var_doc = d->AsDoc<ExprDoc>(iter_var->var, iter_var_p->Attr("var"));
  }
  return TIR(d, "launch_thread")
      ->Call({
          var_doc,
          d->AsDoc<ExprDoc>(attr_stmt->value, attr_stmt_p->Attr("value")),
      });
}

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tirx::AttrStmt>(  //
        "", [](tirx::AttrStmt stmt, AccessPath stmt_p, IRDocsifier d) -> Doc {
          bool concise = AllowConciseScoping(d, stmt);
          ffi::Optional<ExprDoc> lhs = std::nullopt;
          ffi::Optional<ExprDoc> rhs = std::nullopt;
          ffi::Optional<tirx::Var> define_var = std::nullopt;
          tirx::Stmt body = stmt->body;
          AccessPath body_p = stmt_p->Attr("body");
          if (stmt->attr_key == "thread_extent" || stmt->attr_key == "virtual_thread") {
            if (stmt->node.as<tirx::IterVarNode>()) {
              rhs = DocsifyLaunchThread(stmt, stmt_p, &define_var, d);
            }
          }
          if (!rhs.defined()) {
            rhs = TIR(d, "attr")->Call({
                d->AsDoc<ExprDoc>(stmt->node, stmt_p->Attr("node")),
                LiteralDoc::Str(stmt->attr_key, stmt_p->Attr("attr_key")),
                d->AsDoc<ExprDoc>(stmt->value, stmt_p->Attr("value")),
            });
          }
          With<TIRFrame> f(d, stmt);
          if (define_var.defined()) {
            lhs = DefineVar(define_var.value(), *f, d);
          }
          AsDocBody(body, body_p, f->get(), d);
          return DoConciseScoping(lhs, rhs.value(), &(*f)->stmts, concise);
        });

TVM_SCRIPT_REPR(tirx::BindNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tirx::AttrStmtNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tirx::AssertStmtNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tirx::WhileNode, ReprPrintTIR);
TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tirx::AllocBuffer>(  //
        "", [](tirx::AllocBuffer stmt, AccessPath p, IRDocsifier d) -> Doc {
          tirx::Buffer buffer = stmt->buffer;
          AccessPath buffer_p = p->Attr("buffer");
          Frame frame = d->frames.back();
          // Define buffer's data var inline as buffer.data
          if (!d->IsVarDefined(buffer->data)) {
            d->Define(buffer->data, frame, [buffer, buffer_p, d]() {
              return d->AsDoc<ExprDoc>(buffer, buffer_p)->Attr("data");
            });
          }
          // Build simplified T.alloc_buffer(shape, dtype, scope=...) call.
          // Only print shape, dtype, scope (and annotations if non-empty).
          ffi::Array<ExprDoc> args;
          ffi::Array<ffi::String> kwargs_keys;
          ffi::Array<ExprDoc> kwargs_values;
          // shape (positional)
          {
            int n = buffer->shape.size();
            ffi::Array<ExprDoc> shape_docs;
            shape_docs.reserve(n);
            AccessPath shape_p = buffer_p->Attr("shape");
            for (int i = 0; i < n; ++i) {
              PrimExpr e = buffer->shape[i];
              AccessPath e_p = shape_p->ArrayItem(i);
              if (!d->IsVarDefined(e) && e->IsInstance<tirx::VarNode>()) {
                ExprDoc lhs = DefineVar(Downcast<tirx::Var>(e), frame, d);
                lhs->source_paths.push_back(e_p);
                frame->stmts.push_back(
                    AssignDoc(lhs, PrintVarCreation(Downcast<tirx::Var>(e), e_p, d), std::nullopt));
              }
              shape_docs.push_back(d->AsDoc<ExprDoc>(e, e_p));
            }
            args.push_back(TupleDoc(shape_docs));
          }
          // dtype (positional, skip if default float32)
          if (buffer->dtype != d->cfg->buffer_dtype) {
            args.push_back(LiteralDoc::DataType(buffer->dtype, buffer_p->Attr("dtype")));
          }
          // scope (keyword, skip if "global")
          {
            ffi::String scope = buffer.scope();
            if (scope != "global") {
              kwargs_keys.push_back("scope");
              kwargs_values.push_back(LiteralDoc::Str(
                  scope, buffer_p->Attr("data")->Attr("type_annotation")->Attr("storage_scope")));
            }
          }
          // annotations (keyword, skip if empty)
          if (!stmt->annotations.empty()) {
            kwargs_keys.push_back("annotations");
            kwargs_values.push_back(d->AsDoc<ExprDoc>(stmt->annotations, p->Attr("annotations")));
          }
          ExprDoc rhs = TIR(d, "alloc_buffer")->Call(args, kwargs_keys, kwargs_values);
          ExprDoc lhs = DefineBuffer(stmt->buffer, frame, d);
          return AssignDoc(lhs, rhs, std::nullopt);
        });

TVM_SCRIPT_REPR(tirx::AllocBufferNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tirx::BreakNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tirx::ContinueNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tirx::DeclBufferNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tirx::SeqStmtNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tirx::IfThenElseNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tirx::EvaluateNode, ReprPrintTIR);
}  // namespace printer
}  // namespace script
}  // namespace tvm
