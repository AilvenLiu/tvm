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
#include <tvm/tirx/exec_scope.h>
#include <tvm/tirx/op.h>

namespace tvm {
namespace tirx {

/******** Constructors ********/

// ScopeId
ScopeId::ScopeId(String name) {
  auto n = make_object<ScopeIdNode>();
  n->type_annotation = GetTypeFromRuntimeDataType(DataType::Int(32));
  n->dtype = DataType::Int(32);
  n->name_hint = std::move(name);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(ScopeIdNode);

TVM_REGISTER_GLOBAL("tirx.ScopeId").set_body_typed([](String name) { return ScopeId(name); });

// ScopeIdDef
ScopeIdDef::ScopeIdDef(Array<ScopeId> ids, Array<PrimExpr> extents, String parent, String cur) {
  auto n = make_object<ScopeIdDefNode>();
  ICHECK_EQ(ids.size(), extents.size()) << "Number of dimensions must match";
  n->def_ids = std::move(ids);
  n->extents = std::move(extents);
  n->parent = std::move(parent);
  n->cur = std::move(cur);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(ScopeIdDefNode);

TVM_REGISTER_GLOBAL("tirx.ScopeIdDef")
    .set_body_typed([](Array<ScopeId> vars, Array<PrimExpr> extents, String parent, String cur) {
      return ScopeIdDef(vars, extents, parent, cur);
    });

// ExecScope
ExecScope::ExecScope(String name) {
  auto n = make_object<ExecScopeNode>();
  n->name = std::move(name);
  data_ = std::move(n);
}

bool ExecScope::Is(const String& name) const { return name == this->get()->name; }

ExecScope ExecScope::Create(String name) {
  if (name == "world") {
    return WorldScope(ScopeIdDef({}, {}, "", ""));
  } else if (name == "kernel") {
    return KernelScope(Array<ScopeIdDef>({}));
  } else {
    return ExecScope(name);
  }
}

TVM_REGISTER_NODE_TYPE(ExecScopeNode);

TVM_REGISTER_GLOBAL("tirx.ExecScope").set_body_typed([](Array<PrimExpr> dims, String name) {
  return ExecScope(name);
});

// WorldScope
WorldScope::WorldScope(ScopeIdDef def) {
  auto n = make_object<WorldScopeNode>();
  n->name = "world";
  n->scope_id_def = std::move(def);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(WorldScopeNode);

TVM_REGISTER_GLOBAL("tirx.WorldScope").set_body_typed([](ScopeIdDef def) {
  return WorldScope(def);
});

// KernelScope
KernelScope::KernelScope(Array<ScopeIdDef> def) {
  auto n = make_object<KernelScopeNode>();
  n->name = "kernel";
  n->scope_id_def = std::move(def);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(KernelScopeNode);

TVM_REGISTER_GLOBAL("tirx.KernelScope").set_body_typed([](Array<ScopeIdDef> def) {
  return KernelScope(def);
});

// ExecScopeSlice
ExecScopeSlice::ExecScopeSlice(Array<ScopeId> ids, Array<Range> ranges, String name) {
  auto n = make_object<ExecScopeSliceNode>();
  ICHECK(!ids.empty()) << "ExecScopeSlice must have at least one defining ScopeId";
  if (ranges.defined()) {
    ICHECK_EQ(ids.size(), ranges.size()) << "Number of dimensions must match";
  }
  n->name = name;
  n->def_ids = std::move(ids);
  n->ranges = std::move(ranges);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(ExecScopeSliceNode);

TVM_REGISTER_GLOBAL("tirx.ExecScopeSlice")
    .set_body_typed([](Array<ScopeId> vars, Array<Range> ranges, String name) {
      return ExecScopeSlice(vars, ranges, name);
    });

/******** Helper functions ********/

static std::unordered_map<String, int> scope_order = {
    {"world", 0},      {"kernel", 1}, {"cluster", 2}, {"cta", 3},
    {"warp_group", 4}, {"warp", 5},   {"thread", 6}};

bool Higher(const ExecScope& lhs, const ExecScope& rhs) {
  ICHECK(scope_order.count(lhs->name) && scope_order.count(rhs->name))
      << "Unknown scope name: " << lhs->name << " or " << rhs->name;
  return scope_order.at(lhs->name) < scope_order.at(rhs->name);
}

bool ValideScope(const ExecScope& scope) { return scope_order.count(scope->name) > 0; }

}  // namespace tirx
}  // namespace tvm
