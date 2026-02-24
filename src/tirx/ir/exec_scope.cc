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
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/exec_scope.h>
#include <tvm/tirx/op.h>

#include <queue>

namespace tvm {
namespace tirx {

namespace {
/*! \brief ExecScope order from highest to lowest */
const std::unordered_map<ffi::String, int> ScopeOrder = {
    {"world", 0},     {"kernel", 1}, {"cluster", 2}, {"cta", 3},
    {"warpgroup", 4}, {"warp", 5},   {"thread", 6}};
}  // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  ExecScopeNode::RegisterReflection();
  ExecScopeSliceNode::RegisterReflection();
  ScopePairNode::RegisterReflection();
  ScopeIdDefNode::RegisterReflection();
}

/******** Definition of Execution Scope ********/
// ExecScope
ExecScope::ExecScope(ffi::String name, ffi::Array<ScopeIdDef> scope_id_def) {
  CHECK(Valid(name)) << "ValueError: Unknown scope name: " << name;
  auto n = ffi::make_object<ExecScopeNode>();
  n->name = std::move(name);
  n->scope_id_def = std::move(scope_id_def);
  data_ = std::move(n);
}

ExecScope ExecScope::Create(ffi::String name) { return ExecScope(name); }

bool ExecScope::Valid(const ffi::String& name) { return ScopeOrder.find(name) != ScopeOrder.end(); }

bool ExecScopeNode::Is(const ffi::String& name) const { return name == this->name; }

bool ExecScopeNode::Is(const ExecScope& other) const {
  // Both must be the same concrete type (both plain or both slice)
  if (this->IsInstance<ExecScopeSliceNode>() != other->IsInstance<ExecScopeSliceNode>()) {
    return false;
  }
  return Is(other->name);
}

bool ExecScopeNode::Higher(const ffi::String& other) const {
  CHECK(ExecScope::Valid(this->name)) << "ValueError: Unknown scope name";
  CHECK(ExecScope::Valid(other)) << "ValueError: Unknown scope name";
  return ScopeOrder.at(this->name) < ScopeOrder.at(other);
}

bool ExecScopeNode::Higher(const ExecScope& other) const { return Higher(other->name); }

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.ExecScope", [](ffi::String name) { return ExecScope(name); });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.ExecScopeCreate",
                        [](ffi::String name) { return ExecScope::Create(name); });
}

// ExecScopeSlice
ExecScopeSlice::ExecScopeSlice(ffi::Variant<ffi::Array<Range>, PrimExpr> slices,
                               ffi::Optional<ffi::Array<PrimExpr>> extents, ffi::String parent,
                               ffi::String cur) {
  auto n = ffi::make_object<ExecScopeSliceNode>();
  n->name = cur;
  n->parent = parent;
  n->extents = std::move(extents);
  if (extents.defined()) {
    if (auto slices_ = slices.as<ffi::Array<Range>>()) {
      CHECK_EQ(slices_.value().size(), extents.value().size())
          << "ValueError: Number of slices must match the number of extents";
    } else if (auto cond = slices.as<PrimExpr>()) {
      CHECK_EQ(1, extents.value().size())
          << "ValueError: Number of select_cond must match the number of extents";
    }
  }
  n->slices = std::move(slices);
  data_ = std::move(n);
}

bool ExecScopeSliceNode::Is(const ExecScope& other) const {
  auto other_slice = other.as<ExecScopeSliceNode>();
  if (!other_slice) {
    return false;
  }
  return ExecScopeNode::Is(other) && StructuralEqual()(this->slices, other_slice->slices);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.ExecScopeSlice", [](ffi::Variant<ffi::Array<Range>, PrimExpr> slices,
                                                 ffi::Optional<ffi::Array<PrimExpr>> extents,
                                                 ffi::String parent, ffi::String cur) {
    return ExecScopeSlice(slices, extents, parent, cur);
  });
}

/******** Definition of Var ********/
// ScopePair
ScopePair::ScopePair(ffi::String parent, ffi::String cur) {
  auto n = ffi::make_object<ScopePairNode>();
  n->parent = parent;
  n->cur = cur;
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.ScopePair",
                        [](ffi::String parent, ffi::String cur) { return ScopePair(parent, cur); });
}

// ScopeIdDef
ScopeIdDef::ScopeIdDef(ffi::Array<Var> ids, ffi::Array<PrimExpr> extents, ScopePair scope,
                        ffi::Optional<ffi::Array<PrimExpr>> preferred_extents) {
  auto n = ffi::make_object<ScopeIdDefNode>();
  CHECK_EQ(ids.size(), extents.size()) << "ValueError: Number of dimensions must match, got "
                                       << ids.size() << " and " << extents.size();
  n->def_ids = std::move(ids);
  n->extents = std::move(extents);
  n->scope = std::move(scope);
  n->preferred_extents = std::move(preferred_extents);
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

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "tirx.ScopeIdDef",
      [](ffi::Array<Var> vars, ffi::Array<PrimExpr> extents, ScopePair scope,
         ffi::Optional<ffi::Array<PrimExpr>> preferred_extents) {
        return ScopeIdDef(vars, extents, scope, preferred_extents);
      });
}

bool ScopeIdDefVerifier::Verify(const ffi::Array<ScopeIdDef>& defs) {
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

  for (const auto& def : defs) {
    if (def->preferred_extents.defined()) {
      CHECK(def->scope->parent == "cluster" && def->scope->cur == "cta")
          << "ValueError: preferred_extents is only valid for cluster→cta scope, got "
          << def->scope->parent << "→" << def->scope->cur;
      CHECK_EQ(def->preferred_extents.value().size(), def->extents.size())
          << "ValueError: preferred_extents must have the same size as extents, got "
          << def->preferred_extents.value().size() << " vs " << def->extents.size();
    }
    insert_id(def);
  }

  while (!queue.empty()) {
    auto head = queue.front();
    queue.pop();


    // Snapshot to avoid iterator invalidation on insert
    std::vector<ScopeIdDef> snapshot;
    snapshot.reserve(id_set.size());
    for (const auto& [_, def] : id_set) snapshot.push_back(def);
    for (const auto& def : snapshot) {
      for (auto op : {Compose, Compliment}) {
        if (auto result = op(head, def)) insert_id(result.value());
        if (auto result = op(def, head)) insert_id(result.value());
      }
    }
  }
  return true;
}

ffi::Optional<ScopeIdDef> Compose(const ScopeIdDef& lhs, const ScopeIdDef& rhs) {
  return (lhs->scope->cur == rhs->scope->parent)
             ? ScopeIdDef({Var("")}, {lhs.fused_extent() * rhs.fused_extent()},
                          ScopePair(lhs->scope->parent, rhs->scope->cur))
             : ffi::Optional<ScopeIdDef>(std::nullopt);
}

ffi::Optional<ScopeIdDef> Compliment(const ScopeIdDef& lhs, const ScopeIdDef& rhs) {
  if (is_zero(rhs.fused_extent())) return std::nullopt;
  arith::Analyzer ana;
  auto try_compliment = [&](PrimExpr lhs_ext, PrimExpr rhs_ext,
                            ScopePair scope) -> ffi::Optional<ScopeIdDef> {
    if (ana.CanProve(floormod(lhs_ext, rhs_ext) == 0)) {
      return ScopeIdDef({Var("")}, {floordiv(lhs_ext, rhs_ext)}, scope);
    }
    CHECK(!ana.CanProve(floormod(lhs_ext, rhs_ext) != 0))
        << "ValueError: scope " << scope << " has non-divisible extents: " << lhs_ext
        << " is not divisible by " << rhs_ext;
    return std::nullopt;
  };
  if (lhs->scope->parent == rhs->scope->parent &&
      ExecScope::Create(rhs->scope->cur)->Higher(lhs->scope->cur)) {
    return try_compliment(lhs.fused_extent(), rhs.fused_extent(),
                          ScopePair(rhs->scope->cur, lhs->scope->cur));
  }
  if (lhs->scope->cur == rhs->scope->cur &&
      ExecScope::Create(lhs->scope->parent)->Higher(rhs->scope->parent)) {
    return try_compliment(lhs.fused_extent(), rhs.fused_extent(),
                          ScopePair(lhs->scope->parent, rhs->scope->parent));
  }
  return std::nullopt;
}

/******** ScopeIdResolve related functions ********/
ScopeIdResolveTable::Registry& ScopeIdResolveTable::Register(ffi::String parent, ffi::String cur,
                                                             ffi::String target_kind) {
  const auto& key = GetKey(ScopePair(parent, cur), target_kind);
  auto* table = Global();
  CHECK(table->resolve_map_.count(key) == 0) << "Duplicate registration for " << key;
  return table->resolve_map_[key];
}

ffi::Array<PrimExpr> ScopeIdResolveTable::Resolve(
    const ScopePair& scope, const ffi::Optional<ffi::Array<PrimExpr>>& extents, int out_dim,
    ffi::String target_kind, const LaunchParams& params) {
  auto table = ScopeIdResolveTable::Global();
  const auto& key = GetKey(scope, target_kind);
  auto it = table->resolve_map_.find(key);
  CHECK(it != table->resolve_map_.end()) << "Cannot resolve scope id for " << key;
  return it->second.func_(extents, out_dim, params);
}

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


// Helper function to handle common thread index calculations
inline PrimExpr GetLinearThreadIndex(const LaunchParams& params) {
  PrimExpr tx, ty, tz, ex, ey, ez;
  std::tie(tx, ex) = GetThread("threadIdx.x", params, true);
  std::tie(ty, ey) = GetThread("threadIdx.y", params, true);
  std::tie(tz, ez) = GetThread("threadIdx.z", params, true);
  return tx + ty * ex + tz * ex * ey;
}

PrimExpr ScopeIdResolveTable::ComputeWarpIdInCta(const LaunchParams& params) {
  PrimExpr warp_id = FloorDiv(GetLinearThreadIndex(params), 32);
  PrimExpr mask = IntImm(DataType::UInt(32), 0xffffffff);
  return Call(warp_id.dtype(), builtin::tvm_warp_shuffle(),
              {mask, warp_id, IntImm(DataType::Int(32), 0), IntImm(DataType::Int(32), 32),
               IntImm(DataType::Int(32), 32)});
}

#define TVM_SCOPEID_RESOLVE_FUNC_REG_VAR_DEF \
  static TVM_ATTRIBUTE_UNUSED ScopeIdResolveTable::Registry& res

#define TVM_REGISTER_SCOPEID_RESOLVE(parent, cur, target_kind)        \
  TVM_STR_CONCAT(TVM_SCOPEID_RESOLVE_FUNC_REG_VAR_DEF, __COUNTER__) = \
      ::tvm::tirx::ScopeIdResolveTable::Global()->Register(parent, cur, target_kind)


Array<PrimExpr> Trivial3DResolve(const LaunchParams& params, const std::string& prefix,
                                 int out_dim) {
  Array<PrimExpr> ret;
  for (size_t i = 0; i < out_dim; i++) {
    ret.push_back(GetThread(prefix + static_cast<char>('x' + i), params).first);
  }
  return std::move(ret);
}



TVM_REGISTER_SCOPEID_RESOLVE("kernel", "cta", "cuda")
    .set([](const ffi::Optional<ffi::Array<PrimExpr>>& extents, int out_dim,
            const LaunchParams& params) -> Array<PrimExpr> {
      return Trivial3DResolve(params, "blockIdx.", out_dim);
    });

TVM_REGISTER_SCOPEID_RESOLVE("kernel", "cluster", "cuda")
    .set([](const ffi::Optional<ffi::Array<PrimExpr>>& extents, int out_dim,
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
    .set([](const ffi::Optional<ffi::Array<PrimExpr>>& extents, int out_dim,
            const LaunchParams& params) -> Array<PrimExpr> {
      return Trivial3DResolve(params, "clusterCtaIdx.", out_dim);
    });

// Macro to reduce boilerplate for single-dimension checks and thread calculations
#define REGISTER_1D_THREAD_SCOPE(from, to, divisor, modifier)                                     \
  TVM_REGISTER_SCOPEID_RESOLVE(from, to, "cuda")                                                  \
      .set([](const ffi::Optional<ffi::Array<PrimExpr>>& extents, int out_dim,                    \
              const LaunchParams& params) -> Array<PrimExpr> {                                    \
        CHECK_EQ(out_dim, 1) << "ValueError: " from "->" to " can only have 1 dimension for now"; \
        arith::Analyzer ana;                                                                      \
        return {ana.Simplify(modifier(FloorDiv(GetLinearThreadIndex(params), divisor)))};         \
      })

// Define common modifiers
auto identity = [](PrimExpr x) { return x; };
auto mod4 = [](PrimExpr x) { return FloorMod(x, 4); };
auto div4 = [](PrimExpr x) { return FloorDiv(x, 4); };
auto mod32 = [](PrimExpr x) { return FloorMod(x, 32); };
auto mod128 = [](PrimExpr x) { return FloorMod(x, 128); };

// Warp-related scopes: resolve from warp_id_in_cta in launch params
#define REGISTER_1D_WARP_SCOPE(from, to, modifier)                                                 \
  TVM_REGISTER_SCOPEID_RESOLVE(from, to, "cuda")                                                   \
      .set([](const ffi::Optional<ffi::Array<PrimExpr>>& extents, int out_dim,                     \
              const LaunchParams& params) -> Array<PrimExpr> {                                     \
        CHECK_EQ(out_dim, 1) << "ValueError: " from "->" to " can only have 1 dimension for now";  \
        arith::Analyzer ana;                                                                       \
        return {ana.Simplify(modifier(GetThread("warp_id_in_cta", params).first))};                \
      })

REGISTER_1D_WARP_SCOPE("cta", "warpgroup", div4);
REGISTER_1D_WARP_SCOPE("cta", "warp", identity);
REGISTER_1D_WARP_SCOPE("warpgroup", "warp", mod4);
REGISTER_1D_THREAD_SCOPE("warpgroup", "thread", 1, mod128);
REGISTER_1D_THREAD_SCOPE("warp", "thread", 1, mod32);

TVM_REGISTER_SCOPEID_RESOLVE("cta", "thread", "cuda")
    .set([](const ffi::Optional<ffi::Array<PrimExpr>>& extents, int out_dim,
            const LaunchParams& params) -> ffi::Array<PrimExpr> {
      return Trivial3DResolve(params, "threadIdx.", out_dim);
    });

}  // namespace tirxxxxxxxxxx
}  // namespace tvm
