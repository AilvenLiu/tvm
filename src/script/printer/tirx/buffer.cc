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
#include <tvm/runtime/device_api.h>  // For `kAllocAlignment`

#include "./utils.h"

namespace tvm {
namespace script {
namespace printer {

ffi::Map<ffi::String, ExprDoc> BufferAttrs(tirx::Buffer buffer, const AccessPath& buffer_p,
                                           const Frame& frame, const IRDocsifier& d,
                                           BufferVarDefinition var_definitions) {
  using tvm::tirx::Var;
  using tvm::tirx::VarNode;
  ffi::Map<ffi::String, ExprDoc> kwargs;
  ffi::Array<ExprDoc> var_def_lhs;
  ffi::Array<ExprDoc> var_def_rhs;

  // Step 0. Set up statistics
  std::unordered_map<const Object*, int> use_count;
  auto update_use_count = [&](const PrimExpr& e) {
    tirx::PostOrderVisit(e, [&](const ObjectRef& n) {
      if (const VarNode* var = n.as<VarNode>()) {
        ++use_count[var];
      }
    });
  };
  update_use_count(buffer->elem_offset);
  update_use_count(buffer->data);
  for (const PrimExpr& e : buffer->strides) {
    update_use_count(e);
  }
  for (const PrimExpr& e : buffer->shape) {
    update_use_count(e);
  }
  auto is_new_var = [&](const PrimExpr& e) {
    return e->IsInstance<VarNode>() && !d->IsVarDefined(e);
  };
  auto add_out_of_line_var_def = [&](const Var& var, const AccessPath& var_p) {
    TVM_FFI_ICHECK(!d->IsVarDefined(var));
    ExprDoc lhs = DefineVar(var, frame, d);
    lhs->source_paths.push_back(var_p);
    var_def_lhs.push_back(lhs);
    var_def_rhs.push_back(PrintVarCreation(var, var_p, d));
  };
  auto try_inline_def = [&](const PrimExpr& e, const AccessPath& e_p,
                            std::function<ExprDoc()> inline_f) {
    TVM_FFI_ICHECK(is_new_var(e));
    Var var = Downcast<Var>(e);
    if (use_count[var.get()] == 1) {
      d->Define(e, frame, inline_f);
      return true;
    } else {
      add_out_of_line_var_def(var, e_p);
      return false;
    }
  };
  // Step 1. Handle `buffer.shape`
  {
    const ffi::Array<PrimExpr>& shape = buffer->shape;
    AccessPath shape_p = buffer_p->Attr("shape");
    int n = shape.size();
    ffi::Array<ExprDoc> results;
    results.reserve(n);
    for (int i = 0; i < n; ++i) {
      PrimExpr e = shape[i];
      AccessPath e_p = shape_p->ArrayItem(i);
      if (is_new_var(e)) {
        add_out_of_line_var_def(Downcast<Var>(e), e_p);
      }
      results.push_back(d->AsDoc<ExprDoc>(e, e_p));
    }
    kwargs.Set("shape", TupleDoc(results));
  }
  // Step 2. Handle `buffer.dtype`
  if (buffer->dtype != d->cfg->buffer_dtype) {
    kwargs.Set("dtype", LiteralDoc::DataType(buffer->dtype, buffer_p->Attr("dtype")));
  }
  // Step 3. Handle `buffer.data`
  bool is_inline_data = false;
  if (is_new_var(buffer->data)) {
    if (var_definitions >= BufferVarDefinition::DataPointer) {
      is_inline_data = try_inline_def(buffer->data, buffer_p->Attr("data"), [=]() {
        return d->AsDoc<ExprDoc>(buffer, buffer_p)->Attr("data");
      });
    } else {
      add_out_of_line_var_def(buffer->data, buffer_p->Attr("data"));
    }
  }
  if (!is_inline_data) {
    kwargs.Set("data", d->AsDoc<ExprDoc>(buffer->data, buffer_p->Attr("data")));
  }
  // Step 4. Handle `buffer.strides`
  if (!buffer->strides.empty()) {
    const ffi::Array<PrimExpr>& strides = buffer->strides;
    AccessPath strides_p = buffer_p->Attr("strides");
    int n = strides.size();
    ffi::Array<ExprDoc> results;
    results.reserve(n);
    for (int i = 0; i < n; ++i) {
      PrimExpr e = strides[i];
      AccessPath e_p = strides_p->ArrayItem(i);
      if (is_new_var(e)) {
        if (try_inline_def(e, e_p, [=]() {
              return d->AsDoc<ExprDoc>(buffer, buffer_p)
                  ->Attr("strides")[{LiteralDoc::Int(i, std::nullopt)}];
            })) {
          results.push_back(LiteralDoc::Str(Downcast<Var>(e)->name_hint, e_p));
          continue;
        }
      }
      results.push_back(d->AsDoc<ExprDoc>(e, e_p));
    }
    kwargs.Set("strides", TupleDoc(results));
  }
  // Step 5. Handle `buffer.elem_offset`
  bool needs_print_factor = false;
  if (const auto* int_imm = buffer->elem_offset.as<IntImmNode>()) {
    if (int_imm->value != 0) {
      kwargs.Set("elem_offset",
                 d->AsDoc<ExprDoc>(buffer->elem_offset,  //
                                   buffer_p->Attr("elem_offset")));
    }
  } else if (is_new_var(buffer->elem_offset)) {
    try_inline_def(buffer->elem_offset, buffer_p->Attr("elem_offset"),
                   [=]() { return d->AsDoc<ExprDoc>(buffer, buffer_p)->Attr("elem_offset"); });
    needs_print_factor = true;
  } else {
    kwargs.Set("elem_offset",
               d->AsDoc<ExprDoc>(buffer->elem_offset,  //
                                 buffer_p->Attr("elem_offset")));
  }
  // Step 6. Handle `buffer.scope`
  {
    ffi::String scope = buffer.scope();
    if (scope != "global") {
      kwargs.Set(
          "scope",
          LiteralDoc::Str(scope,
                          buffer_p->Attr("data")->Attr("type_annotation")->Attr("storage_scope")));
    }
  }
  // Step 7. Handle `buffer.data_alignment`
  if (buffer->data_alignment != runtime::kAllocAlignment) {
    kwargs.Set("align", LiteralDoc::Int(buffer->data_alignment, buffer_p->Attr("data_alignment")));
  }
  // Step 8. Handle `buffer.offset_factor`
  if (needs_print_factor || buffer->offset_factor != 1) {
    kwargs.Set("offset_factor",
               LiteralDoc::Int(buffer->offset_factor, buffer_p->Attr("offset_factor")));
  }
  // Step 9. Handle `buffer.buffer_type`
  if (buffer->buffer_type != tirx::BufferType::kDefault) {
    kwargs.Set("buffer_type", LiteralDoc::Str("auto", buffer_p->Attr("buffer_type")));
  }
  // Step 10. Handle `buffer.axis_separator`
  if (!buffer->axis_separators.empty()) {
    kwargs.Set("axis_separators",
               d->AsDoc<ExprDoc>(buffer->axis_separators, buffer_p->Attr("axis_separators")));
  }
  // Step 11. Handle `buffer.logical_scope`
  {
    String scope = buffer.logical_scope();
    if (scope != "") {
      kwargs.Set(
          "logical_scope",
          LiteralDoc::Str(scope,
                          buffer_p->Attr("data")->Attr("type_annotation")->Attr("logical_scope")));
    }
  }
  // Step 12. Handle `buffer.layout`
  if (const tirx::TBufferNode* tbuffer = buffer.as<tirx::TBufferNode>()) {
    kwargs.Set("layout", d->AsDoc<ExprDoc>(tbuffer->layout, buffer_p->Attr("layout")));
  }

  if (var_def_lhs.size() == 1) {
    frame->stmts.push_back(AssignDoc(var_def_lhs[0], var_def_rhs[0], std::nullopt));
  } else if (var_def_lhs.size() > 1) {
    frame->stmts.push_back(AssignDoc(TupleDoc(var_def_lhs), TupleDoc(var_def_rhs), std::nullopt));
  }
  return kwargs;
}

ExprDoc BufferCall(const ExprDoc& prefix, const ffi::Map<ffi::String, ExprDoc>& attrs,
                   ffi::Array<ExprDoc> args) {
  ffi::Array<ffi::String> kwargs_keys;
  ffi::Array<ExprDoc> kwargs_values;
  for (ffi::String s : {"shape", "dtype"}) {
    if (ffi::Optional<ExprDoc> doc = attrs.Get(s)) {
      args.push_back(doc.value());
    }
  }
  for (ffi::String s : {"data", "strides", "elem_offset", "scope", "align", "offset_factor",
                        "buffer_type", "axis_separators", "logical_scope", "layout"}) {
    if (ffi::Optional<ExprDoc> doc = attrs.Get(s)) {
      kwargs_keys.push_back(s);
      kwargs_values.push_back(doc.value());
    }
  }
  return prefix->Call(args, kwargs_keys, kwargs_values);
}

ExprDoc BufferDecl(const tirx::Buffer& buffer, const ffi::String& method,
                   const ffi::Array<ExprDoc>& args, const AccessPath& p, const Frame& frame,
                   const IRDocsifier& d, BufferVarDefinition var_definitions) {
  return BufferCall(/*prefix=*/TIR(d, method),
                    /*attrs=*/BufferAttrs(buffer, p, frame, d, var_definitions),
                    /*args=*/args);
}

ExprDoc BufferAttn(const tirx::Buffer& buffer, const AccessPath& p, const Frame& frame,
                   const IRDocsifier& d) {
  ffi::Map<ffi::String, ExprDoc> attrs =
      BufferAttrs(buffer, p, frame, d, BufferVarDefinition::DataPointer);
  ExprDoc shape = attrs.Get("shape").value();
  ExprDoc dtype =
      attrs.Get("dtype").value_or(LiteralDoc::DataType(buffer->dtype, p->Attr("dtype")));
  return TIR(d, "Buffer")->Call({shape, dtype}, {}, {});
}

ffi::Array<Doc> BufferIndices(const ffi::Array<PrimExpr>& indices, const AccessPath& p,
                              const IRDocsifier& d) {
  int n = indices.size();
  ffi::Array<Doc> indices_doc;
  indices_doc.reserve(n);
  for (int i = 0; i < n; ++i) {
    if (const auto* ramp = indices[i].as<tirx::RampNode>()) {
      if (const auto* stride = ramp->stride.as<IntImmNode>()) {
        AccessPath ramp_p = p->Attr("indices")->ArrayItem(i);
        AccessPath stride_p = ramp_p->Attr("stride");
        ExprDoc start = d->AsDoc<ExprDoc>(ramp->base,  //
                                          ramp_p->Attr("base"));
        ExprDoc stop = d->AsDoc<ExprDoc>(ramp->base + ramp->lanes * ramp->stride,  //
                                         ramp_p->Attr("lanes"));
        ffi::Optional<ExprDoc> step = std::nullopt;
        if (stride->value != 1) {
          step = d->AsDoc<ExprDoc>(ramp->stride, ramp_p->Attr("stride"));
        }
        indices_doc.push_back(SliceDoc(start, stop, step));
        continue;
      }
    }
    indices_doc.push_back(d->AsDoc<ExprDoc>(indices[i], p->Attr("indices")->ArrayItem(i)));
  }
  return indices_doc;
}

ffi::Array<Doc> BufferSlices(const ffi::Array<Range>& region, const AccessPath& p,
                             const IRDocsifier& d) {
  int n = region.size();
  ffi::Array<Doc> indices;
  indices.reserve(n);
  for (int i = 0; i < n; ++i) {
    Range range = region[i];
    AccessPath range_p = p->ArrayItem(i);
    ExprDoc min = d->AsDoc<ExprDoc>(range->min, range_p->Attr("min"));
    if (tirx::is_one(range->extent)) {
      indices.push_back(min);
    } else {
      ExprDoc max = d->AsDoc<ExprDoc>(range->min + range->extent, range_p->Attr("extent"));
      indices.push_back(SliceDoc(min, max, std::nullopt));
    }
  }
  return indices;
}

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tirx::BufferRegion>(
        "", [](tirx::BufferRegion buffer_region, AccessPath p, IRDocsifier d) -> Doc {
          ExprDoc prefix = d->AsDoc<ExprDoc>(buffer_region->buffer, p->Attr("buffer"));
          return prefix[BufferSlices(buffer_region->region, p->Attr("region"), d)];
        });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tirx::BufferStore>(  //
        "", [](tirx::BufferStore store, AccessPath p, IRDocsifier d) -> Doc {
          ExprDoc buffer = d->AsDoc<ExprDoc>(store->buffer, p->Attr("buffer"));
          ExprDoc value = d->AsDoc<ExprDoc>(store->value, p->Attr("value"));

          // Use .vstore(...) syntax when there is a predicate
          if (store->predicate.defined()) {
            ExprDoc indices = d->AsDoc<ExprDoc>(store->indices, p->Attr("indices"));
            ExprDoc predicate = d->AsDoc<ExprDoc>(store->predicate, p->Attr("predicate"));
            return ExprStmtDoc(
                buffer->Attr("vstore")->Call({indices, value}, {"predicate"}, {predicate}));
          }

          return AssignDoc(
              /*lhs=*/buffer[BufferIndices(store->indices, p->Attr("indices"), d)],
              /*rhs=*/value, std::nullopt);
        });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tirx::BufferLoad>(  //
        "", [](tirx::BufferLoad load, AccessPath p, IRDocsifier d) -> Doc {
          ExprDoc buffer = d->AsDoc<ExprDoc>(load->buffer, p->Attr("buffer"));

          // Use .vload(...) syntax when there is a predicate
          if (load->predicate.defined()) {
            ExprDoc indices = d->AsDoc<ExprDoc>(load->indices, p->Attr("indices"));
            ExprDoc predicate = d->AsDoc<ExprDoc>(load->predicate, p->Attr("predicate"));
            return buffer->Attr("vload")->Call({indices}, {"predicate"}, {predicate});
          }

          return buffer[BufferIndices(load->indices, p->Attr("indices"), d)];
        });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)  //
    .set_dispatch<tirx::Buffer>("", [](tirx::Buffer buffer, AccessPath p, IRDocsifier d) -> Doc {
      if (!d->IsVarDefined(buffer)) {
        if (ffi::Optional<Frame> opt_f = FindLowestVarDef(buffer, d)) {
          ExprDoc lhs = DefineBuffer(buffer, opt_f.value(), d);
          ExprDoc rhs = BufferDecl(buffer, "Buffer", {}, p, opt_f.value(), d,
                                   BufferVarDefinition::DataPointer);
          opt_f.value()->stmts.push_back(AssignDoc(lhs, rhs, std::nullopt));
        }
      }
      if (ffi::Optional<ExprDoc> doc = d->GetVarDoc(buffer)) {
        return doc.value();
      }
      TVM_FFI_THROW(IndexError) << "Buffer is not defined in the environment: " << buffer;
      TVM_FFI_UNREACHABLE();
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)  //
    .set_dispatch<tirx::TBuffer>("", [](tirx::TBuffer buffer, ObjectPath p, IRDocsifier d) -> Doc {
      if (!d->IsVarDefined(buffer)) {
        if (Optional<Frame> opt_f = FindLowestVarDef(buffer, d)) {
          ExprDoc lhs = DefineBuffer(buffer, opt_f.value(), d);
          ExprDoc rhs = BufferDecl(buffer, "Buffer", {}, p, opt_f.value(), d,
                                   BufferVarDefinition::DataPointer);
          opt_f.value()->stmts.push_back(AssignDoc(lhs, rhs, NullOpt));
        }
      }
      if (Optional<ExprDoc> doc = d->GetVarDoc(buffer)) {
        return doc.value();
      }
      LOG(FATAL) << "IndexError: TBuffer is not defined in the environment: " << buffer;
    });

std::pair<ExprDoc, ExprDoc> PrintDataIterTree(tirx::IterTreeBase root, const Array<PrimExpr>& coeff,
                                              const tirx::TileLayout::SplitMap& split_map,
                                              const tirx::IterTree::LeafIndexMap& data_leaf2idx,
                                              const tirx::IterTree::LeafIndexMap& device_leaf2idx,
                                              IRDocsifier d, ObjectPath coeff_p,
                                              ObjectPath root_p) {
  Array<ExprDoc> results_data, results_strides;
  const auto* split = root.as<tirx::IterTreeSplitNode>();
  ICHECK(split != nullptr) << "InternalError: Expect IterTreeSplit but get " << root->GetTypeKey();
  if (split->children.empty()) {
    // leaf node
    auto data_idx_it = data_leaf2idx.find(root);
    ICHECK(data_idx_it != data_leaf2idx.end())
        << "InternalError: Cannot find data index for leaf " << root;
    int data_idx = data_idx_it->second;
    ExprDoc stride = d->AsDoc<ExprDoc>(coeff[data_idx], coeff_p->ArrayIndex(data_idx));
    auto it = split_map.find(root);
    if (it != split_map.end()) {
      // Bound leaf, print T.S(device_index)
      auto dev_idx_it = device_leaf2idx.find(it->second);
      ICHECK(dev_idx_it != device_leaf2idx.end())
          << "InternalError: Cannot find device index for leaf " << it->second;
      int dev_idx = dev_idx_it->second;
      return {TIR(d, "S")->Call({LiteralDoc::Int(dev_idx, NullOpt)}, {}, {}), stride};
    } else {
      // Data leaf, print extent
      return {d->AsDoc<ExprDoc>(split->extent, root_p->Attr("extent")), stride};
    }
  } else {
    results_data.reserve(split->children.size());
    results_strides.reserve(split->children.size());
    for (size_t i = 0; i < split->children.size(); ++i) {
      auto [data, stride] =
          PrintDataIterTree(split->children[i], coeff, split_map, data_leaf2idx, device_leaf2idx, d,
                            coeff_p, root_p->Attr("children")->ArrayIndex(i));
      results_data.push_back(data);
      results_strides.push_back(stride);
    }
  }
  return {TupleDoc(results_data), TupleDoc(results_strides)};
}

ExprDoc PrintDeviceIterTree(tirx::IterTreeBase root, IRDocsifier d, ObjectPath p) {
  Array<ExprDoc> results;
  const auto* split = root.as<tirx::IterTreeSplitNode>();
  ICHECK(split != nullptr) << "InternalError: Expect IterTreeSplit but get " << root->GetTypeKey();
  if (split->children.empty()) {
    // leaf node
    return d->AsDoc<ExprDoc>(split->extent, p->Attr("extent"));
  } else {
    results.reserve(split->children.size());
    for (size_t i = 0; i < split->children.size(); ++i) {
      results.push_back(
          PrintDeviceIterTree(split->children[i], d, p->Attr("children")->ArrayIndex(i)));
    }
  }
  return TupleDoc(results);
}

Doc PrintTileLayout(tirx::TileLayout layout, IRDocsifier d, ObjectPath p) {
  const auto& data_leaves = layout->data_tree.GetLeaves();
  const auto& device_leaves = layout->device_tree.GetLeaves();
  tirx::DataIterTree::CoeffMap coeff_map = layout->data_tree.GetCoeffMap(data_leaves);
  tirx::DeviceIterTree::AttrMap attr_map = layout->device_tree.GetAttrMap(device_leaves);
  tirx::IterTree::LeafIndexMap device_leaf2idx = layout->device_tree.GetLeafIndexMap(device_leaves);
  tirx::IterTree::LeafIndexMap data_leaf2idx = layout->data_tree.GetLeafIndexMap(data_leaves);

  tirx::TileLayout::SplitMap split_map = layout.GetSplitMap(data_leaves, device_leaves);
  auto [data, strides] = PrintDataIterTree(
      layout->data_tree->root, layout->data_tree->coeff, split_map, data_leaf2idx, device_leaf2idx,
      d, p->Attr("data_tree")->Attr("coeff"), p->Attr("data_tree")->Attr("root"));
  auto device =
      PrintDeviceIterTree(layout->device_tree->root, d, p->Attr("device_tree")->Attr("root"));
  Array<String> keys;
  Array<ExprDoc> values;
  keys.push_back("data");
  values.push_back(data);
  keys.push_back("strides");
  values.push_back(strides);
  keys.push_back("device");
  values.push_back(device);
  // print Exclusive Attrs
  {
    Array<ExprDoc> e_docs;
    for (size_t i = 0; i < device_leaves.size(); ++i) {
      auto it = attr_map.find(device_leaves[i]);
      ICHECK(it != attr_map.end())
          << "InternalError: Cannot find device attribute for leaf " << device_leaves[i];
      const tirx::DeviceIterAttr& attr = it->second;
      if (attr.IsExclusive()) {
        ExprDoc idx = LiteralDoc::Int(i, NullOpt);
        ExprDoc owner = d->AsDoc<ExprDoc>(
            attr->owner, p->Attr("device_tree")->Attr("attrs")->ArrayIndex(i)->Attr("owner"));
        e_docs.push_back(TupleDoc({idx, owner}));
      }
    }
    if (!e_docs.empty()) {
      keys.push_back("exclusive");
      values.push_back(ListDoc(e_docs));
    }
  }
  // print from_to scope
  {
    if (layout->from.defined()) {
      ICHECK(layout->to.defined()) << "InternalError: `to` must be defined if `from` is defined";
      keys.push_back("from_to");
      values.push_back(TupleDoc({d->AsDoc<ExprDoc>(layout->from.value()->name, p->Attr("from")),
                                 d->AsDoc<ExprDoc>(layout->from.value()->name, p->Attr("to"))}));
    }
  }

  return TIR(d, "TileLayout")->Attr("from_nested_tuple")->Call({}, {keys}, {values});
}

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)  //
    .set_dispatch<tirx::TileLayout>("",
                                   [](tirx::TileLayout layout, ObjectPath p, IRDocsifier d) -> Doc {
                                     return PrintTileLayout(layout, d, p);
                                   });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)  //
    .set_dispatch<tirx::DataIterTree>(
        "", [](tirx::DataIterTree tree, ObjectPath p, IRDocsifier d) -> Doc {
          Doc doc = TIR(d, "DataIterTree")
                        ->Call({}, {"root", "coeff"},
                               {
                                   d->AsDoc<ExprDoc>(tree->root, p->Attr("root")),
                                   d->AsDoc<ExprDoc>(tree->coeff, p->Attr("coeff")),
                               });
          return doc;
        });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)  //
    .set_dispatch<tirx::DeviceIterTree>(
        "", [](tirx::DeviceIterTree tree, ObjectPath p, IRDocsifier d) -> Doc {
          Doc doc = TIR(d, "DeviceIterTree")
                        ->Call({}, {"root", "attrs"},
                               {
                                   d->AsDoc<ExprDoc>(tree->root, p->Attr("root")),
                                   d->AsDoc<ExprDoc>(tree->attrs, p->Attr("attrs")),
                               });
          return doc;
        });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)  //
    .set_dispatch<tirx::IterTreeSplit>(
        "", [](tirx::IterTreeSplit split, ObjectPath p, IRDocsifier d) -> Doc {
          Doc doc = TIR(d, "IterTreeSplit")
                        ->Call({}, {"children", "extent"},
                               {
                                   d->AsDoc<ExprDoc>(split->children, p->Attr("children")),
                                   d->AsDoc<ExprDoc>(split->extent, p->Attr("extent")),
                               });
          return doc;
        });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)  //
    .set_dispatch<tirx::DeviceIterAttr>(
        "", [](tirx::DeviceIterAttr attr, ObjectPath p, IRDocsifier d) -> Doc {
          Array<String> keys;
          Array<ExprDoc> values;
          keys.push_back("type");
          values.push_back(LiteralDoc::Int(attr->type, p->Attr("type")));
          if (attr->bound.defined()) {
            keys.push_back("bound");
            values.push_back(d->AsDoc<ExprDoc>(attr->bound, p->Attr("bound")));
          }
          if (attr->owner.defined()) {
            keys.push_back("owner");
            values.push_back(d->AsDoc<ExprDoc>(attr->owner, p->Attr("owner")));
          }
          Doc doc = TIR(d, "DeviceIterAttr")->Call({}, keys, values);
          return doc;
        });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tirx::MatchBufferRegion>(
        "", [](tirx::MatchBufferRegion stmt, AccessPath p, IRDocsifier d) -> Doc {
          Frame frame = d->frames.back();
          ExprDoc lhs = DefineBuffer(stmt->buffer, frame, d);
          ExprDoc src_buffer = d->AsDoc<ExprDoc>(stmt->source, p->Attr("source"));
          ExprDoc rhs = BufferDecl(stmt->buffer, "match_buffer", {src_buffer}, p->Attr("buffer"),
                                   d->frames.back(), d, BufferVarDefinition::MatchBuffer);
          return AssignDoc(lhs, rhs, std::nullopt);
        });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tirx::ProducerLoad>(  //
        "", [](tirx::ProducerLoad load, AccessPath p, IRDocsifier d) -> Doc {
          ExprDoc prefix = IdDoc(load->producer->GetNameHint());
          return prefix[BufferIndices(load->indices, p->Attr("indices"), d)];
        });

TVM_SCRIPT_REPR(tirx::BufferRegionNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tirx::BufferLoadNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tirx::BufferStoreNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tirx::BufferNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tirx::TBufferNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tirx::TileLayoutNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tirx::DataIterTreeNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tirx::DeviceIterTreeNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tirx::IterTreeSplitNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tirx::DeviceIterAttrNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tirx::MatchBufferRegionNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tirx::ProducerLoadNode, ReprPrintTIR);

}  // namespace printer
}  // namespace script
}  // namespace tvm
