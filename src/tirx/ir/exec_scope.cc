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

bool ExecScopeNode::Is(const String& name) const { return name == this->name; }

bool ExecScopeNode::Is(const ExecScope& other) const { return Is(other->name); }

bool ExecScopeNode::Higher(const String& other) const {
  CHECK(ExecScope::Valid(this->name)) << "ValueError: Unknown scope name";
  CHECK(ExecScope::Valid(other)) << "ValueError: Unknown scope name";
  return ScopeOrder.at(this->name) < ScopeOrder.at(other);
}

bool ExecScopeNode::Higher(const ExecScope& other) const { return Higher(other->name); }

TVM_REGISTER_NODE_TYPE(ExecScopeNode);

TVM_FFI_REGISTER_GLOBAL("tirx.ExecScope").set_body_typed([](String name) { return ExecScope(name); });

TVM_FFI_REGISTER_GLOBAL("tirx.ExecScopeCreate").set_body_typed([](String name) {
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

TVM_FFI_REGISTER_GLOBAL("tirx.WorldScope").set_body_typed([](ScopeIdDef def) {
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

TVM_FFI_REGISTER_GLOBAL("tirx.KernelScope").set_body_typed([](Array<ScopeIdDef> def) {
  return KernelScope(def);
});

// ExecScopeSlice
ExecScopeSlice::ExecScopeSlice(Variant<Array<Range>, PrimExpr> slice,
                               Optional<Array<PrimExpr>> extents, String parent, String cur) {
  auto n = make_object<ExecScopeSliceNode>();
  n->name = cur;
  n->parent = parent;
  n->extents = std::move(extents);
  if (extents.defined()) {
    if (auto slices = slice.as<Array<Range>>()) {
      CHECK_EQ(slices.value().size(), extents.value().size())
          << "ValueError: Number of slices must match the number of extents";
    } else if (auto cond = slice.as<PrimExpr>()) {
      CHECK_EQ(1, extents.value().size())
          << "ValueError: Number of select_cond must match the number of extents";
    }
  }
  n->slice = std::move(slice);
  data_ = std::move(n);
}

bool ExecScopeSliceNode::Is(const ExecScope& other) const {
  auto other_slice = other.as<ExecScopeSliceNode>();
  if (!other_slice) {
    return false;
  }
  return ExecScopeNode::Is(other) && StructuralEqual()(this->slice, other_slice->slice);
}

TVM_REGISTER_NODE_TYPE(ExecScopeSliceNode);

TVM_FFI_REGISTER_GLOBAL("tirx.ExecScopeSlice")
    .set_body_typed([](Variant<Array<Range>, PrimExpr> slice, Optional<Array<PrimExpr>> extents,
                       String parent,
                       String cur) { return ExecScopeSlice(slice, extents, parent, cur); });

/******** Definition of Var ********/
// ScopePair
ScopePair::ScopePair(String parent, String cur) {
  auto n = make_object<ScopePairNode>();
  n->parent = parent;
  n->cur = cur;
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(ScopePairNode);

TVM_FFI_REGISTER_GLOBAL("tirx.ScopePair").set_body_typed([](String parent, String cur) {
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

TVM_FFI_REGISTER_GLOBAL("tirx.ScopeIdDef")
    .set_body_typed([](Array<Var> vars, Array<PrimExpr> extents, ScopePair scope) {
      return ScopeIdDef(vars, extents, scope);
    });

bool ScopeIdDefVerifier::Verify(const Array<ScopeIdDef>& defs) {
  id_set.clear();
  arith::Analyzer ana;
  std::queue<ScopeIdDef> queue;

  auto insert_id = [&](const ScopeIdDef& id) {
    auto [it, inserted] = id_set.try_emplace(id->scope, id);
    if (!inserted) {
      CHECK(ana.CanProveEqual(it->second.fused_extent(), id.fused_extent()))
          << "Inconsistent extents for scope " << id->scope;
    } else {
      queue.push(id);
    }
  };

  for (const auto& def : defs) insert_id(def);

  while (!queue.empty()) {
    auto head = queue.front();
    queue.pop();

    for (const auto& [_, def] : id_set) {
      for (auto op : {Compose, Compliment}) {
        if (auto result = op(head, def)) insert_id(result.value());
        if (auto result = op(def, head)) insert_id(result.value());
      }
    }
  }
  return true;
}

Optional<ScopeIdDef> Compose(const ScopeIdDef& lhs, const ScopeIdDef& rhs) {
  return (lhs->scope->cur == rhs->scope->parent)
             ? ScopeIdDef({Var("")}, {lhs.fused_extent() * rhs.fused_extent()},
                          ScopePair(lhs->scope->parent, rhs->scope->cur))
             : Optional<ScopeIdDef>(std::nullopt);
}

Optional<ScopeIdDef> Compliment(const ScopeIdDef& lhs, const ScopeIdDef& rhs) {
  if (lhs->scope->parent == rhs->scope->parent &&
      ExecScope::Create(rhs->scope->cur)->Higher(lhs->scope->cur)) {
    return ScopeIdDef({Var("")}, {FloorDiv(lhs.fused_extent(), rhs.fused_extent())},
                      ScopePair(rhs->scope->cur, lhs->scope->cur));
  }
  if (lhs->scope->cur == rhs->scope->cur &&
      ExecScope::Create(lhs->scope->parent)->Higher(rhs->scope->parent)) {
    return ScopeIdDef({Var("")}, {FloorDiv(lhs.fused_extent(), rhs.fused_extent())},
                      ScopePair(lhs->scope->parent, rhs->scope->parent));
  }
  return std::nullopt;
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

Array<PrimExpr> ScopeIdResolveTable::Resolve(const ScopePair& scope,
                                             const Optional<Array<PrimExpr>>& extents, int out_dim,
                                             String target_kind, const LaunchParams& params) {
  auto table = ScopeIdResolveTable::Global();
  const auto& key = GetKey(scope, target_kind);
  auto it = table->resolve_map_.find(key);
  CHECK(it != table->resolve_map_.end()) << "Cannot resolve scope id for " << key;
  return it->second.func_(extents, out_dim, params);
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

Array<PrimExpr> Trivial3DResolve(const LaunchParams& params, const std::string& prefix,
                                 int out_dim) {
  Array<PrimExpr> ret;
  for (size_t i = 0; i < out_dim; i++) {
    ret.push_back(GetThread(prefix + static_cast<char>('x' + i), params).first);
  }
  return std::move(ret);
}

// Helper function to handle common thread index calculations
inline PrimExpr GetLinearThreadIndex(const LaunchParams& params) {
  PrimExpr tx, ty, tz, ex, ey, ez;
  std::tie(tx, ex) = GetThread("threadIdx.x", params, true);
  std::tie(ty, ey) = GetThread("threadIdx.y", params, true);
  std::tie(tz, ez) = GetThread("threadIdx.z", params, true);
  return tx + ty * ex + tz * ex * ey;
}

TVM_REGISTER_SCOPEID_RESOLVE("kernel", "cta", "cuda")
    .set([](const Optional<Array<PrimExpr>>& extents, int out_dim,
            const LaunchParams& params) -> Array<PrimExpr> {
      return Trivial3DResolve(params, "blockIdx.", out_dim);
    });

TVM_REGISTER_SCOPEID_RESOLVE("kernel", "cluster", "cuda")
    .set([](const Optional<Array<PrimExpr>>& extents, int out_dim,
            const LaunchParams& params) -> Array<PrimExpr> {
      CHECK_LE(out_dim, 3) << "ValueError: kernel->cluster can only have 3 dimensions for now";
      Array<PrimExpr> ret;
      for (size_t i = 0; i < out_dim; i++) {
        ret.push_back(tirx::Call(
            DataType::Int(32), builtin::ptx_fetch_register(),
            {IntImm(DataType::Int(32), 32), StringImm("clusterid." + std::string(1, 'x' + i))}));
      }
      return ret;
    });

TVM_REGISTER_SCOPEID_RESOLVE("cluster", "cta", "cuda")
    .set([](const Optional<Array<PrimExpr>>& extents, int out_dim,
            const LaunchParams& params) -> Array<PrimExpr> {
      return Trivial3DResolve(params, "clusterCtaIdx.", out_dim);
    });

// Macro to reduce boilerplate for single-dimension checks and thread calculations
#define REGISTER_1D_THREAD_SCOPE(from, to, divisor, modifier)                                     \
  TVM_REGISTER_SCOPEID_RESOLVE(from, to, "cuda")                                                  \
      .set([](const Optional<Array<PrimExpr>>& extents, int out_dim,                              \
              const LaunchParams& params) -> Array<PrimExpr> {                                    \
        CHECK_EQ(out_dim, 1) << "ValueError: " from "->" to " can only have 1 dimension for now"; \
        arith::Analyzer ana;                                                                      \
        return {ana.Simplify(modifier(FloorDiv(GetLinearThreadIndex(params), divisor)))};         \
      })

// Define common modifiers
auto identity = [](PrimExpr x) { return x; };
auto mod4 = [](PrimExpr x) { return FloorMod(x, 4); };
auto mod32 = [](PrimExpr x) { return FloorMod(x, 32); };
auto mod128 = [](PrimExpr x) { return FloorMod(x, 128); };

REGISTER_1D_THREAD_SCOPE("cta", "warpgroup", 128, identity);
REGISTER_1D_THREAD_SCOPE("cta", "warp", 32, identity);
REGISTER_1D_THREAD_SCOPE("warpgroup", "warp", 32, mod4);
REGISTER_1D_THREAD_SCOPE("warpgroup", "thread", 1, mod128);
REGISTER_1D_THREAD_SCOPE("warp", "thread", 1, mod32);

TVM_REGISTER_SCOPEID_RESOLVE("cta", "thread", "cuda")
    .set([](const Optional<Array<PrimExpr>>& extents, int out_dim,
            const LaunchParams& params) -> Array<PrimExpr> {
      return Trivial3DResolve(params, "threadIdx.", out_dim);
    });

}  // namespace tirxxx
}  // namespace tvm
