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
#include "./utils.h"

namespace tvm {
namespace script {
namespace printer {

Doc PrintBlock(IRDocsifier d, tirx::SBlock block, AccessPath block_p,  //
               ffi::Optional<tirx::SBlockRealize> opt_realize,
               ffi::Optional<AccessPath> opt_realize_p) {
  With<TIRFrame> frame(d, block);
  TVM_FFI_ICHECK_EQ(opt_realize.defined(), opt_realize_p.defined());
  const tirx::SBlockRealizeNode* realize =
      opt_realize.defined() ? opt_realize.value().get() : nullptr;
  AccessPath realize_p = *opt_realize_p;

  bool tirp = false;
  for (Frame f : d->frames) {
    if (const auto* tir_f = f.as<TIRFrameNode>()) {
      if (auto func = tir_f->tir.as<tir::PrimFuncNode>()) {
        if (func->attrs.defined() && func->attrs->dict.count(tvm::attr::kIsTIRp)) {
          tirp = true;
          break;
        }
      }
    }
  }

  if (block->exec_scope.defined()) {
    if (const tvm::tir::WorldScopeNode* scope = block->exec_scope.as<tvm::tir::WorldScopeNode>()) {
      ExprDoc lhs = DefineVar(scope->scope_id_def->def_ids[0], *frame, d);
      ExprDoc rhs =
          TIR(d, "kernel_id")
              ->Call({d->AsDoc<ExprDoc>(scope->scope_id_def->extents[0], block_p->Attr("exec_scope")
                                                                             ->Attr("scope_id_def")
                                                                             ->Attr("extents")
                                                                             ->ArrayItem(0))});
      (*frame)->stmts.push_back(AssignDoc(lhs, rhs, NullOpt));
    } else if (const tvm::tir::KernelScopeNode* scope =
                   block->exec_scope.as<tvm::tir::KernelScopeNode>()) {
      for (auto scope_id_def : scope->scope_id_def) {
        Array<ExprDoc> lhs;
        for (auto scope_id : scope_id_def->def_ids) {
          lhs.push_back(DefineVar(scope_id, *frame, d));
        }
        ExprDoc rhs =
            TIR(d, scope_id_def->scope->cur + "_id")
                ->Call({d->AsDoc<ExprDoc>(
                           scope_id_def->extents,
                           block_p->Attr("exec_scope")->Attr("scope_id_def")->Attr("extents"))},
                       {"parent"},
                       {LiteralDoc::Str(
                           scope_id_def->scope->parent,
                           block_p->Attr("exec_scope")->Attr("scope_id_def")->Attr("parent"))});
        (*frame)->stmts.push_back(AssignDoc(TupleDoc(lhs), rhs, NullOpt));
      }
    }
  }

  // Step 1. Handle block var and block bindings
  // Step 1.1. Obtain all loop var defined along path
  std::unordered_map<const tirx::VarNode*, tirx::For> loop_vars;
  for (Frame f : d->frames) {
    if (const auto* tir_f = f.as<TIRFrameNode>()) {
      if (auto for_loop = tir_f->tirx.as<tirx::For>()) {
        for (ffi::Optional<tirx::For> loop = for_loop; loop;
             loop = loop.value()->body.as<tirx::For>()) {
          loop_vars.insert(std::make_pair(loop.value()->loop_var.get(), loop.value()));
        }
      }
    }
  }

  std::vector<int> remap_vars_indices;
  auto add_remapped_iter_var = [&](int i) -> bool {
    if (realize && d->cfg->syntax_sugar) {
      tirx::ExprDeepEqual expr_equal;
      tirx::IterVar iter_var = block->iter_vars[i];
      PrimExpr value = realize->iter_values[i];
      if (iter_var->iter_type == tirx::IterVarType::kDataPar ||
          iter_var->iter_type == tirx::IterVarType::kCommReduce) {
        if (const auto* var = value.as<tirx::VarNode>()) {
          if (loop_vars.count(var)) {
            tirx::For for_loop = loop_vars.at(var);
            if (expr_equal(for_loop->min, iter_var->dom->min) &&
                expr_equal(for_loop->extent, iter_var->dom->extent)) {
              remap_vars_indices.push_back(i);
              return true;
            }
          }
        }
      }
    }
    return false;
  };

  auto print_single_iter_var = [&](int i) {
    tirx::IterVar iter_var = block->iter_vars[i];
    AccessPath iter_var_p = block_p->Attr("iter_var")->ArrayItem(i);
    ExprDoc rhs = TIR(d, "axis");
    if (iter_var->iter_type == tirx::IterVarType::kDataPar) {
      rhs = rhs->Attr("spatial");
    } else if (iter_var->iter_type == tirx::IterVarType::kCommReduce) {
      rhs = rhs->Attr("reduce");
    } else if (iter_var->iter_type == tirx::IterVarType::kOrdered) {
      rhs = rhs->Attr("scan");
    } else if (iter_var->iter_type == tirx::IterVarType::kOpaque) {
      rhs = rhs->Attr("opaque");
    } else {
      TVM_FFI_THROW(ValueError) << "Unknown IterVarType in block signature: "
                                << tirx::IterVarType2String(iter_var->iter_type);
    }
    ExprDoc dom{ffi::UnsafeInit()};
    if (tirx::is_zero(iter_var->dom->min)) {
      ExprDoc extent = d->AsDoc<ExprDoc>(iter_var->dom->extent,  //
                                         iter_var_p->Attr("dom")->Attr("extent"));
      dom = extent;
    } else {
      ExprDoc min = d->AsDoc<ExprDoc>(iter_var->dom->min, iter_var_p->Attr("dom")->Attr("min"));
      ExprDoc max = d->AsDoc<ExprDoc>(iter_var->dom->min + iter_var->dom->extent,
                                      iter_var_p->Attr("dom")->Attr("extent"));
      dom = TupleDoc({min, max});
    }
    if (realize) {
      ExprDoc binding = d->AsDoc<ExprDoc>(realize->iter_values[i],  //
                                          realize_p->Attr("iter_values")->ArrayItem(i));
      rhs = rhs->Call({dom, binding});
    } else {
      rhs = rhs->Call({dom});
    }
    (*frame)->stmts.push_back(AssignDoc(DefineVar(iter_var->var, *frame, d), rhs, std::nullopt));
  };

  auto print_remapped_iter_var = [&]() {
    if (remap_vars_indices.size()) {
      int m = remap_vars_indices.size();
      if (!m) {
        return;
      }
      if (m == 1) {
        print_single_iter_var(remap_vars_indices[0]);
        remap_vars_indices.clear();
        return;
      }
      ffi::Array<ExprDoc> lhs;
      ffi::Array<ExprDoc> loop_var_doc;
      lhs.reserve(m);
      loop_var_doc.reserve(m);
      std::string binding_type = "";
      ffi::Array<AccessPath> binding_paths;
      for (int i : remap_vars_indices) {
        tirx::IterVar iter_var = block->iter_vars[i];
        AccessPath iter_var_p = block_p->Attr("iter_vars")->ArrayItem(i);
        lhs.push_back(DefineVar(iter_var->var, *frame, d));
        loop_var_doc.push_back(d->AsDoc<ExprDoc>(realize->iter_values[i],
                                                 realize_p->Attr("iter_values")->ArrayItem(i)));
        binding_paths.push_back(iter_var_p->Attr("iter_type"));
        binding_type += iter_var->iter_type == tirx::IterVarType::kDataPar ? "S" : "R";
      }
      ExprDoc rhs = TIR(d, "axis")->Attr("remap");
      ExprDoc binding_str = LiteralDoc::Str(binding_type, std::nullopt);
      binding_str->source_paths = std::move(binding_paths);
      rhs = rhs->Call({binding_str, ListDoc(loop_var_doc)});
      (*frame)->stmts.push_back(AssignDoc(TupleDoc(lhs), rhs, std::nullopt));
      remap_vars_indices.clear();
    }
  };

  // Step 1.2. Construct all block var bindings
  int n_vars = block->iter_vars.size();
  for (int i = 0; i < n_vars; ++i) {
    if (!add_remapped_iter_var(i)) {
      print_remapped_iter_var();
      print_single_iter_var(i);
    }
  }
  print_remapped_iter_var();

  // Step 2. Handle block predicate
  if (realize) {
    TVM_FFI_ICHECK(realize->predicate.defined() && realize->predicate->dtype.is_bool());
    if (!tirx::is_one(realize->predicate)) {
      (*frame)->stmts.push_back(ExprStmtDoc(
          TIR(d, "where")
              ->Call({d->AsDoc<ExprDoc>(realize->predicate, realize_p->Attr("predicate"))})));
    }
  }
  // Step 3. Handle block read/write regions
  if (!tirp) {
    ffi::Array<ExprDoc> reads;
    for (int i = 0, n = block->reads.size(); i < n; ++i) {
      reads.push_back(d->AsDoc<ExprDoc>(block->reads[i], block_p->Attr("reads")->ArrayItem(i)));
    }
    (*frame)->stmts.push_back(ExprStmtDoc(TIR(d, "reads")->Call(reads)));
    ffi::Array<ExprDoc> writes;
    for (int i = 0, n = block->writes.size(); i < n; ++i) {
      writes.push_back(d->AsDoc<ExprDoc>(block->writes[i], block_p->Attr("writes")->ArrayItem(i)));
    }
    (*frame)->stmts.push_back(ExprStmtDoc(TIR(d, "writes")->Call(writes)));
  }
  // Step 4. Handle block attributes
  if (!block->annotations.empty()) {
    (*frame)->stmts.push_back(ExprStmtDoc(
        TIR(d, "sblock_attr")
            ->Call({d->AsDoc<ExprDoc>(block->annotations, block_p->Attr("annotations"))})));
  }
  // Step 5. Handle `alloc_buffer`
  for (int i = 0, n = block->alloc_buffers.size(); i < n; ++i) {
    tirx::Buffer buffer = block->alloc_buffers[i];
    AccessPath buffer_p = block_p->Attr("alloc_buffers")->ArrayItem(i);
    IdDoc lhs = DefineBuffer(buffer, *frame, d);
    ExprDoc rhs = BufferDecl(buffer, "sblock_alloc_buffer", {}, buffer_p, *frame, d,
                             BufferVarDefinition::DataPointer);
    (*frame)->stmts.push_back(AssignDoc(lhs, rhs, std::nullopt));
  }
  // Step 6. Handle `match_buffer`
  for (int i = 0, n = block->match_buffers.size(); i < n; ++i) {
    tirx::MatchBufferRegion buffer_region = block->match_buffers[i];
    AccessPath buffer_region_p = block_p->Attr("match_buffers")->ArrayItem(i);
    StmtDoc doc = d->AsDoc<StmtDoc>(buffer_region, buffer_region_p);
    (*frame)->stmts.push_back(doc);
  }

  // tir+
  for (size_t i = 0; i < block->barriers.size(); ++i) {
    tir::Barrier barrier = block->barriers[i];
    ObjectPath barrier_p = block_p->Attr("barriers")->ArrayIndex(i);
    IdDoc lhs = DefineBarrier(barrier, *frame, d);
    ExprDoc rhs =
        TIRp(d, "alloc_barrier")
            ->Call({LiteralDoc::Str(barrier->thread_scope->name, barrier_p->Attr("thread_scope")),
                    LiteralDoc::Str(barrier->name_hint, barrier_p->Attr("name"))});
    (*frame)->stmts.push_back(AssignDoc(lhs, rhs, NullOpt));
  }
  for (size_t i = 0; i < block->barrier_arrays.size(); ++i) {
    tir::BarrierArray barrier_array = block->barrier_arrays[i];
    ObjectPath barrier_array_p = block_p->Attr("barrier_arrays")->ArrayIndex(i);
    IdDoc lhs = DefineBarrierArray(barrier_array, *frame, d);
    ExprDoc rhs =
        TIRp(d, "alloc_barrier_array")
            ->Call({LiteralDoc::Str(barrier_array->thread_scope->name,
                                    barrier_array_p->Attr("thread_scope")),
                    LiteralDoc::Int(barrier_array->size, barrier_array_p->Attr("size")),
                    LiteralDoc::Str(barrier_array->name_hint, barrier_array_p->Attr("name"))});
    (*frame)->stmts.push_back(AssignDoc(lhs, rhs, NullOpt));
  }
  for (size_t i = 0; i < block->pipelines.size(); ++i) {
    tir::Pipeline pipeline = block->pipelines[i];
    if (pipeline->IsInstance<tir::CopyPipelineNode>()) {
      ObjectPath pipeline_p = block_p->Attr("pipelines")->ArrayIndex(i);
      IdDoc lhs = DefinePipeline(pipeline, *frame, d);
      ExprDoc rhs =
          TIRp(d, "alloc_copy_pipeline")
              ->Call(
                  {LiteralDoc::Str(pipeline->thread_scope->name, pipeline_p->Attr("thread_scope")),
                   LiteralDoc::Int(pipeline->depth, pipeline_p->Attr("depth")),
                   LiteralDoc::Boolean(pipeline->separate_pc, pipeline_p->Attr("separate_pc")),
                   LiteralDoc::Str(pipeline->name_hint, pipeline_p->Attr("name_hint"))});
      (*frame)->stmts.push_back(AssignDoc(lhs, rhs, NullOpt));
    } else {
      LOG(FATAL) << "ValueError: Unknown Pipeline type in block signature: "
                 << pipeline->GetTypeKey();
    }
  }
  for (size_t i = 0; i < block->buffer_views.size(); ++i) {
    tir::BufferView buffer_view = block->buffer_views[i];
    ObjectPath buffer_view_p = block_p->Attr("buffer_views")->ArrayIndex(i);

    IdDoc lhs = DefineBuffer(buffer_view->dst_buffer, *frame, d);

    ExprDoc rhs = d->AsDoc<ExprDoc>(buffer_view, buffer_view_p);
    (*frame)->stmts.push_back(AssignDoc(lhs, rhs, std::nullopt));
  }
  for (size_t i = 0; i < block->buffer_gets.size(); ++i) {
    tir::BufferGet buffer_get = block->buffer_gets[i];
    ObjectPath buffer_get_p = block_p->Attr("buffer_gets")->ArrayIndex(i);

    IdDoc lhs = DefineBuffer(buffer_get->dst_buffer, *frame, d);

    ExprDoc rhs = TIR(d, "get")->Call({
        d->AsDoc<ExprDoc>(buffer_get->src_buffer, buffer_get_p->Attr("src_buffer")),
    });
    (*frame)->stmts.push_back(AssignDoc(lhs, rhs, std::nullopt));
  }

  // Step 7. Handle init block
  if (block->init.defined()) {
    tirx::Stmt init = block->init.value();
    With<TIRFrame> init_frame(d, init);
    AsDocBody(init, block_p->Attr("init"), init_frame->get(), d);
    (*frame)->stmts.push_back(
        ScopeDoc(std::nullopt, TIR(d, "init")->Call({}), (*init_frame)->stmts));
  }
  // Step 8. Handle block body
  AsDocBody(block->body, block_p->Attr("body"), frame->get(), d);
  ffi::Array<ffi::String> kwargs_keys;
  ffi::Array<ExprDoc> kwargs_values;
  if (!realize) {
    kwargs_keys.push_back("no_realize");
    kwargs_values.push_back(LiteralDoc::Boolean(true, std::nullopt));
  }
  // tir+
  if (block->exec_scope.defined()) {
    if (auto scope = block->exec_scope.as<tvm::tir::ExecScopeSlice>()) {
      return ScopeDoc(NullOpt,
                      TIR(d, block->exec_scope.value()->name)
                          ->Call({d->AsDoc<ExprDoc>(scope.value()->slices,
                                                    block_p->Attr("exec_scope")->Attr("slices")),
                                  LiteralDoc::Str(scope.value()->parent,
                                                  block_p->Attr("exec_scope")->Attr("parent"))}),
                      (*frame)->stmts);
    }
    return ScopeDoc(std::nullopt, TIR(d, block->exec_scope.value()->name)->Call({}),
                    (*frame)->stmts);
  }
  return ScopeDoc(std::nullopt,
                  TIR(d, "sblock")  //
                      ->Call({LiteralDoc::Str(block->name_hint, block_p->Attr("name_hint"))},
                             kwargs_keys, kwargs_values),
                  (*frame)->stmts);
}

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tirx::SBlockRealize>(
        "", [](tirx::SBlockRealize realize, AccessPath p, IRDocsifier d) -> Doc {
          Doc doc = PrintBlock(d, realize->block, p->Attr("block"), realize, p);
          // since we do not have d->AsDoc for realize->block,
          // we should add possible doc decoration manually.
          AddDocDecoration<ScopeDoc>(doc, realize->block, p->Attr("block"), d->cfg);
          return doc;
        });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tirx::SBlock>("", [](tirx::SBlock block, AccessPath p, IRDocsifier d) -> Doc {
      return PrintBlock(d, block, p, std::nullopt, std::nullopt);
    });

TVM_SCRIPT_REPR(tirx::SBlockNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tirx::SBlockRealizeNode, ReprPrintTIR);

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::BufferGet>(
        "", [](tir::BufferGet buffer_get, ObjectPath p, IRDocsifier d) -> Doc {
          Doc doc = TIR(d, "get")->Call({
              d->AsDoc<ExprDoc>(buffer_get->src_buffer, p->Attr("src_buffer")),
              d->AsDoc<ExprDoc>(buffer_get->dst_buffer, p->Attr("dst_buffer")),
          });
          return doc;
        });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::BufferView>(
        "", [](tir::BufferView buffer_view, ObjectPath p, IRDocsifier d) -> Doc {
          Doc doc = TIR(d, "view")->Call(
              {d->AsDoc<ExprDoc>(buffer_view->src_buffer, p->Attr("src_buffer")),
               d->AsDoc<ExprDoc>(buffer_view->layout, p->Attr("layout")),
               d->AsDoc<ExprDoc>(buffer_view->dst_buffer->shape,
                                 p->Attr("dst_buffer")->Attr("shape"))});
          return doc;
        });
TVM_SCRIPT_REPR(tir::BufferGetNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tir::BufferViewNode, ReprPrintTIR);

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::WorldScope>(
        "", [](tir::WorldScope world, ObjectPath p, IRDocsifier d) -> Doc {
          Doc doc = TIR(d, "ExecScope")
                        ->Call({d->AsDoc<ExprDoc>(world->scope_id_def, p->Attr("scope_id_def"))});
          return doc;
        });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::KernelScope>(
        "", [](tir::KernelScope kernel, ObjectPath p, IRDocsifier d) -> Doc {
          Doc doc = TIR(d, "ExecScope")
                        ->Call({d->AsDoc<ExprDoc>(kernel->scope_id_def, p->Attr("scope_id_def"))});
          return doc;
        });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::ExecScope>(
        "", [](tir::ExecScope exec_scope, ObjectPath p, IRDocsifier d) -> Doc {
          Doc doc = TIR(d, "ExecScope")->Call({LiteralDoc::Str(exec_scope->name, p->Attr("name"))});
          return doc;
        });

TVM_SCRIPT_REPR(tir::WorldScopeNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tir::KernelScopeNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tir::ExecScopeNode, ReprPrintTIR);

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::ScopeIdDef>("", [](tir::ScopeIdDef def, ObjectPath p, IRDocsifier d) -> Doc {
      Doc doc = TIR(d, "ScopeIdDef")
                    ->Call({d->AsDoc<ExprDoc>(def->def_ids, p->Attr("def_ids")),
                            d->AsDoc<ExprDoc>(def->extents, p->Attr("extents")),
                            LiteralDoc::Str(def->scope->parent, p->Attr("parent")),
                            LiteralDoc::Str(def->scope->cur, p->Attr("cur"))});
      return doc;
    });
TVM_SCRIPT_REPR(tir::ScopeIdDefNode, ReprPrintTIR);

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::ScopePair>("", [](tir::ScopePair pair, ObjectPath p, IRDocsifier d) -> Doc {
      Doc doc = TIR(d, "ScopePair")
                    ->Call({d->AsDoc<ExprDoc>(pair->parent, p->Attr("parent")),
                            d->AsDoc<ExprDoc>(pair->cur, p->Attr("cur"))});
      return doc;
    });
TVM_SCRIPT_REPR(tir::ScopePairNode, ReprPrintTIR);

}  // namespace printer
}  // namespace script
}  // namespace tvm
