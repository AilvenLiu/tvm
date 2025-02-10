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
#include <tvm/arith/analyzer.h>
#include <tvm/tirx/exec_scope.h>
#include <tvm/tirx/op.h>

#include <queue>

namespace tvm {
namespace tirx {

/******** Definition of Execution Scope ********/
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
    return WorldScope(ScopeIdDef({}, {}, ScopePair("world", "kernel")));
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
ExecScopeSlice::ExecScopeSlice(Optional<Array<Range>> slices, Optional<PrimExpr> select_cond,
                               String parent, String cur) {
  auto n = make_object<ExecScopeSliceNode>();
  n->name = cur;
  n->parent = parent;
  CHECK(!slices.defined() || !select_cond.defined())
      << "ValueError: select_cond and slices cannot both be present";
  n->slices = std::move(slices);
  n->select_cond = std::move(select_cond);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(ExecScopeSliceNode);

TVM_REGISTER_GLOBAL("tirx.ExecScopeSlice")
    .set_body_typed([](Optional<Array<Range>> slices, Optional<PrimExpr> select_cond, String parent,
                       String cur) { return ExecScopeSlice(slices, select_cond, parent, cur); });

/******** Definition of Var ********/
// ScopePair
ScopePair::ScopePair(String parent, String cur) {
  auto n = make_object<ScopePairNode>();
  n->parent = parent;
  n->cur = cur;
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(ScopePairNode);

TVM_REGISTER_GLOBAL("tirx.Scope").set_body_typed([](String parent, String cur) {
  return ScopePair(parent, cur);
});

// ScopeIdDef
ScopeIdDef::ScopeIdDef(Array<Var> ids, Array<PrimExpr> extents, ScopePair scope) {
  auto n = make_object<ScopeIdDefNode>();
  CHECK_EQ(ids.size(), extents.size()) << "ValueError: Number of dimensions must match, got "
                                       << ids.size() << " and " << extents.size();
  n->def_ids = std::move(ids);
  n->extents = std::move(extents);
  n->scope = std::move(scope);
  data_ = std::move(n);
}

PrimExpr ScopeIdDef::fused_extent() const {
  CHECK_GT(get()->extents.size(), 0) << "ValueError: Cannot get extent of empty scope";
  PrimExpr ret = get()->extents[0];
  for (size_t i = 1; i < get()->extents.size(); ++i) {
    ret = ret * get()->extents[i];
  }
  return ret;
}

TVM_REGISTER_NODE_TYPE(ScopeIdDefNode);

TVM_REGISTER_GLOBAL("tirx.ScopeIdDef")
    .set_body_typed([](Array<Var> vars, Array<PrimExpr> extents, ScopePair scope) {
      return ScopeIdDef(vars, extents, scope);
    });

bool ScopeIdDefVerifier::Verify(const Array<ScopeIdDef>& defs) {
  id_set.clear();
  arith::Analyzer ana;

  auto get_extent = [&](const ScopeIdDef& id) {
    auto it = id_set.find(id->scope);
    CHECK(it != id_set.end()) << "Cannot resolve extent for scope " << id->scope;
    return (*it).second.fused_extent();
  };
  std::queue<ScopeIdDef> queue;
  auto insert_id = [&](const ScopeIdDef& id) {
    auto it = id_set.find(id->scope);
    if (it == id_set.end()) {
      id_set.insert({id->scope, id});
      queue.push(id);
    } else {
      CHECK(ana.CanProveEqual((*it).second.fused_extent(), id.fused_extent()))
          << "Inconsistent extents for scope " << id->scope << ": " << (*it).second.fused_extent()
          << " vs " << id.fused_extent();
    }
  };
  for (const auto& def : defs) {
    insert_id(def);
  }
  while (!queue.empty()) {
    auto head = queue.front();
    queue.pop();
    PrimExpr extent = get_extent(head);

    std::vector<ScopeIdDef> ids;
    for (const auto& [scope, def] : id_set) {
      ids.push_back(def);
    }

    for (const auto& def : ids) {
      auto def_extent = get_extent(def);
      if (auto composed = Compose(head, def)) {
        insert_id(composed.value());
      } else if (auto composed = Compose(def, head)) {
        insert_id(composed.value());
      } else if (auto compliment = Compliment(head, def)) {
        insert_id(compliment.value());
      } else if (auto compliment = Compliment(def, head)) {
        insert_id(compliment.value());
      }
    }
  }
  return true;
}

Optional<ScopeIdDef> Compose(const ScopeIdDef& lhs, const ScopeIdDef& rhs) {
  if (lhs->scope->cur == rhs->scope->parent) {
    return ScopeIdDef({Var("")}, {lhs.fused_extent() * rhs.fused_extent()},
                      ScopePair(lhs->scope->parent, rhs->scope->cur));
  } else {
    return NullOpt;
  }
}

Optional<ScopeIdDef> Compliment(const ScopeIdDef& lhs, const ScopeIdDef& rhs) {
  if (lhs->scope->parent == rhs->scope->parent) {
    if (ExecScope::Create(rhs->scope->cur).Higher(lhs->scope->cur)) {
      return ScopeIdDef({Var("")}, {FloorDiv(lhs.fused_extent(), rhs.fused_extent())},
                        ScopePair(rhs->scope->cur, lhs->scope->cur));
    }
  } else if (lhs->scope->cur == rhs->scope->cur) {
    if (ExecScope::Create(lhs->scope->parent).Higher(rhs->scope->parent)) {
      return ScopeIdDef({Var("")}, {FloorDiv(lhs.fused_extent(), rhs.fused_extent())},
                        ScopePair(lhs->scope->parent, rhs->scope->parent));
    }
  }
  return NullOpt;
}

/******** Helper functions ********/
bool IsStorageBuffer(const String& storage, const String& logical) {
  return StorageToLogical.count(storage) && StorageToLogical.at(storage) == logical;
}

String StorageToLogicalScope(const String& storage) {
  ICHECK(StorageToLogical.count(storage)) << "Unknown storage type: " << storage;
  return StorageToLogical.at(storage);
}

/******** ScopeIdResolve related functions ********/
ScopeIdResolveTable::Registry& ScopeIdResolveTable::Register(String parent, String cur,
                                                             String target_kind) {
  const auto& key = GetKey(ScopePair(parent, cur), target_kind);
  auto* table = Global();
  CHECK(table->resolve_map_.count(key) == 0) << "Duplicate registration for " << key;
  return table->resolve_map_[key];
}

Array<PrimExpr> ScopeIdResolveTable::Resolve(const ScopeIdDef& def, String target_kind,
                                             const LaunchParams& params) {
  auto table = ScopeIdResolveTable::Global();
  const auto& key = GetKey(def->scope, target_kind);
  auto it = table->resolve_map_.find(key);
  CHECK(it != table->resolve_map_.end()) << "Cannot resolve scope id for " << key;
  return it->second.func_(def, params);
}

#define TVM_SCOPEID_RESOLVE_FUNC_REG_VAR_DEF \
  static TVM_ATTRIBUTE_UNUSED ScopeIdResolveTable::Registry& res

#define TVM_REGISTER_SCOPEID_RESOLVE(parent, cur, target_kind)        \
  TVM_STR_CONCAT(TVM_SCOPEID_RESOLVE_FUNC_REG_VAR_DEF, __COUNTER__) = \
      ::tvm::tirx::ScopeIdResolveTable::Global()->Register(parent, cur, target_kind)

using LaunchParams = ScopeIdResolveTable::LaunchParams;

std::pair<PrimExpr, PrimExpr> GetThread(const std::string& tag, const LaunchParams& params,
                                        bool allow_missing = false) {
  auto it = params.find(tag);
  if (it == params.end()) {
    CHECK(allow_missing) << "Cannot find thread var: " << tag;
    return {0, 1};
  }
  return {(*it).second->var, (*it).second->dom->extent};
}

Array<PrimExpr> Trivial3DResolve(const ScopeIdDef& def, const LaunchParams& params,
                                 const std::string& prefix) {
  CHECK_LE(def->extents.size(), 3) << "ValueError: kernel->cta can only have 3 dimensions";
  Array<PrimExpr> ret;
  for (size_t i = 0; i < def->extents.size(); i++) {
    ret.push_back(GetThread(prefix + static_cast<char>('x' + i), params).first);
  }
  return std::move(ret);
}

TVM_REGISTER_SCOPEID_RESOLVE("kernel", "cta", "cuda")
    .set([](const ScopeIdDef& def, const LaunchParams& params) -> Array<PrimExpr> {
      return Trivial3DResolve(def, params, "blockIdx.");
    });

TVM_REGISTER_SCOPEID_RESOLVE("cta", "thread", "cuda")
    .set([](const ScopeIdDef& def, const LaunchParams& params) -> Array<PrimExpr> {
      return Trivial3DResolve(def, params, "threadIdx.");
    });

TVM_REGISTER_SCOPEID_RESOLVE("warp", "thread", "cuda")
    .set([](const ScopeIdDef& def, const LaunchParams& params) -> Array<PrimExpr> {
      CHECK_EQ(def->extents.size(), 1) << "ValueError: cta->warp can only have 1 dimension for now";
      PrimExpr tx, ty, tz, ex, ey, ez;
      std::tie(tx, ex) = GetThread("threadIdx.x", params, true);
      std::tie(ty, ey) = GetThread("threadIdx.y", params, true);
      std::tie(tz, ez) = GetThread("threadIdx.z", params, true);
      arith::Analyzer ana;
      return {ana.Simplify(FloorMod(tx + ty * ex + tz * ex * ey, 32))};
    });

TVM_REGISTER_SCOPEID_RESOLVE("cta", "warp", "cuda")
    .set([](const ScopeIdDef& def, const LaunchParams& params) -> Array<PrimExpr> {
      CHECK_EQ(def->extents.size(), 1) << "ValueError: cta->warp can only have 1 dimension for now";
      PrimExpr tx, ty, tz, ex, ey, ez;
      std::tie(tx, ex) = GetThread("threadIdx.x", params, true);
      std::tie(ty, ey) = GetThread("threadIdx.y", params, true);
      std::tie(tz, ez) = GetThread("threadIdx.z", params, true);
      arith::Analyzer ana;
      return {ana.Simplify(FloorDiv(tx + ty * ex + tz * ex * ey, 32))};
    });

TVM_REGISTER_SCOPEID_RESOLVE("cluster", "cta", "cuda")
    .set([](const ScopeIdDef& def, const LaunchParams& params) -> Array<PrimExpr> {
      return Trivial3DResolve(def, params, "clusterCtaIdx.");
    });

}  // namespace tirxxx
}  // namespace tvm
