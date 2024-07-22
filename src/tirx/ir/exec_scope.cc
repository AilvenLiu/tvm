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
  CHECK_EQ(ids.size(), extents.size()) << "ValueError: Number of dimensions must match, got "
                                       << ids.size() << " and " << extents.size();
  CHECK(ScopeOrder.count(parent)) << "ValueError: Unknown scope name: " << parent;
  CHECK(ScopeOrder.count(cur)) << "ValueError: Unknown scope name: " << cur;
  CHECK(ScopeOrder.at(parent) < ScopeOrder.at(cur))
      << "ValueError: Parent scope must be higher than current scope";
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

ScopeIdDef::ScopeIdDef(String parent, String cur) : ScopeIdDef({}, {}, parent, cur) {}

PrimExpr ScopeIdDef::fused_extent() const {
  PrimExpr res = get()->extents[0];
  for (size_t i = 1; i < get()->extents.size(); ++i) {
    res = res * get()->extents[i];
  }
  return res;
}

Optional<ScopeIdDef> Compose(const ScopeIdDef& lhs, const ScopeIdDef& rhs) {
  if (lhs->cur == rhs->parent) {
    return ScopeIdDef(lhs->parent, rhs->cur);
  } else {
    return NullOpt;
  }
}

Optional<ScopeIdDef> Compliment(const ScopeIdDef& lhs, const ScopeIdDef& rhs) {
  if (lhs->parent == rhs->parent) {
    if (ExecScope::Create(rhs->cur).Higher(lhs->cur)) {
      return ScopeIdDef(rhs->cur, lhs->cur);
    }
  } else if (lhs->cur == rhs->cur) {
    if (ExecScope::Create(lhs->parent).Higher(rhs->parent)) {
      return ScopeIdDef(lhs->parent, rhs->parent);
    }
  }
  return NullOpt;
}

// ExecScope
ExecScope::ExecScope(String name) {
  CHECK(Valid(name)) << "ValueError: Unknown scope name: " << name;
  CHECK(name != "world" && name != "kernel") << "ValueError: Reserved scope name: " << name;
  auto n = make_object<ExecScopeNode>();
  n->name = std::move(name);
  data_ = std::move(n);
}

ExecScope ExecScope::Create(String name) {
  if (name == "world") {
    return WorldScope(ScopeIdDef({}, {}, "world", "kernel"));
  } else if (name == "kernel") {
    return KernelScope(Array<ScopeIdDef>({}));
  } else {
    return ExecScope(name);
  }
}

bool ExecScope::Valid(const String& name) { return ScopeOrder.find(name) != ScopeOrder.end(); }

bool ExecScope::Is(const String& name) const { return name == this->get()->name; }

bool ExecScope::Is(const ExecScope& other) const { return Is(other->name); }

bool ExecScope::Higher(const String& other) const {
  CHECK(Valid(this->get()->name)) << "ValueError: Unknown scope name";
  CHECK(Valid(other)) << "ValueError: Unknown scope name";
  return ScopeOrder.at(this->get()->name) < ScopeOrder.at(other);
}

bool ExecScope::Higher(const ExecScope& other) const { return Higher(other->name); }

TVM_REGISTER_NODE_TYPE(ExecScopeNode);

TVM_REGISTER_GLOBAL("tirx.ExecScope").set_body_typed([](String name) { return ExecScope(name); });

TVM_REGISTER_GLOBAL("tirx.ExecScopeCreate").set_body_typed([](String name) {
  return ExecScope::Create(name);
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
bool IsStorageBuffer(const String& storage, const String& logical) {
  return StorageToLogical.count(storage) && StorageToLogical.at(storage) == logical;
}

String StorageToLogicalScope(const String& storage) {
  ICHECK(StorageToLogical.count(storage)) << "Unknown storage type: " << storage;
  return StorageToLogical.at(storage);
}

}  // namespace tirxxx
}  // namespace tvm
