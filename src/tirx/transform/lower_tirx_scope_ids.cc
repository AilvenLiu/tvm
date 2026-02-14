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

/*!
 * \file lower_tirx_scope_ids.cc
 * \brief Resolve scope id definitions for TIRx lowering.
 */

#include <tvm/arith/analyzer.h>
#include <tvm/runtime/logging.h>
#include <tvm/target/target.h>
#include <tvm/tirx/exec_scope.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/tirx_op.h>
#include <tvm/tirx/transform.h>

#include <unordered_map>

namespace tvm {
namespace tirx {

class ScopeIdDefGather : public StmtExprVisitor {
 public:
  static std::vector<ScopeIdDef> Gather(const Stmt& stmt) {
    ScopeIdDefGather gather;
    gather(stmt);
    return gather.scope_id_def;
  }

  void VisitStmt_(const ExecScopeStmtNode* op) override {
    StmtExprVisitor::VisitStmt_(op);
    for (const auto& def : op->exec_scope->scope_id_def) {
      scope_id_def.push_back(def);
    }
  }

  std::vector<ScopeIdDef> scope_id_def;
};

class ScopeIdDefRemover : public StmtExprMutator {
 public:
  static Stmt Remove(const Stmt& stmt) { return ScopeIdDefRemover()(stmt); }

  Stmt VisitStmt_(const ExecScopeStmtNode* op) override {
    Stmt body = StmtExprMutator::VisitStmt(op->body);
    ExecScope new_scope;
    if (const auto* slice = op->exec_scope.as<ExecScopeSliceNode>()) {
      auto n_scope = ffi::make_object<ExecScopeSliceNode>(*slice);
      n_scope->scope_id_def = {};
      new_scope = ExecScopeSlice(n_scope);
    } else if (const auto* scope = op->exec_scope.as<ExecScopeNode>()) {
      auto n_scope = ffi::make_object<ExecScopeNode>(*scope);
      n_scope->scope_id_def = {};
      new_scope = ExecScope(n_scope);
    } else {
      LOG(FATAL) << "Internal Error: unknown exec_scope type: " << op->exec_scope;
    }
    return ExecScopeStmt(new_scope, body);
  }
};

class ScopeIdDefResolver : public StmtExprMutator {
 public:
  explicit ScopeIdDefResolver(const Target& target) : target_(target) {}

  static Stmt Resolve(const Stmt& stmt, const Target& target) {
    return ScopeIdDefResolver(target)(stmt);
  }

 private:
  using LaunchParams = ScopeIdResolveTable::LaunchParams;

  Stmt VisitStmt_(const ExecScopeStmtNode* op) override {
    const auto& scope = op->exec_scope;
    if (!scope->Is("kernel")) {
      // Non-kernel ExecScopeStmt: just visit body and reconstruct
      Stmt body = VisitStmt(op->body);
      if (body.same_as(op->body)) {
        return ffi::GetRef<Stmt>(op);
      }
      return ExecScopeStmt(scope, body);
    }
    // Kernel scope: resolve scope ids
    ICHECK(!scope->Is("world")) << "TIRx Error: world scope is not supported at the moment";
    ICHECK(!kernel_launch_params_.has_value())
        << "TIRx Error: nested kernel scopes are not supported";

    // Step 0: Gather the scope id defs from all nested scopes
    Array<ScopeIdDef> scope_id_def = std::move(ScopeIdDefGather::Gather(ffi::GetRef<Stmt>(op)));

    // Step 1: Verify the ScopeIdDef is well-formed
    ScopeIdDefVerifier verifier;
    CHECK(verifier.Verify(scope_id_def)) << "Inconsistent ScopeIdDef";

    // Step 2: Extract kernel launch parameters
    LaunchParams launch_params;
    ExtractKernelLaunchParams(verifier.id_set, target_, &launch_params);

    std::unordered_map<Var, PrimExpr, ObjectPtrHash, ObjectPtrEqual> id_map;
    std::vector<std::pair<Var, PrimExpr>> scope_lets;

    // Check if any scope needs warp_id_in_cta
    bool need_warp_id = false;
    for (const auto& def : scope_id_def) {
      if (ScopeIdResolveTable::NeedWarpIdInCta(def->scope->cur)) {
        need_warp_id = true;
        break;
      }
    }

    if (need_warp_id) {
      PrimExpr shuffled = ScopeIdResolveTable::ComputeWarpIdInCta(launch_params);
      Var warp_id_in_cta_var("warp_id_in_cta", shuffled.dtype());
      scope_lets.push_back({warp_id_in_cta_var, shuffled});
      IterVar warp_iv(Range::FromMinExtent(0, 1), warp_id_in_cta_var, kThreadIndex,
                      "warp_id_in_cta");
      launch_params.insert({"warp_id_in_cta", warp_iv});
    }
    kernel_launch_params_ = launch_params;

    for (const auto& def : scope_id_def) {
      auto resolved =
          ScopeIdResolveTable::Resolve(def->scope, def->extents, def->extents.size(),
                                       target_->kind->name, kernel_launch_params_.value());
      ICHECK_EQ(resolved.size(), def->extents.size())
          << "Internal Error: Inconsistent resolved size " << resolved.size() << " vs "
          << def->extents.size();
      for (size_t i = 0; i < def->def_ids.size(); i++) {
        PrimExpr value = resolved[i];
        Var let_var(def->def_ids[i]->name_hint, value.dtype());
        id_map[def->def_ids[i]] = let_var;
        scope_lets.push_back({let_var, value});
      }
    }

    // Step 3: Substitute scope ids in body, then visit
    Stmt new_body = Substitute(op->body, id_map);
    new_body = VisitStmt(new_body);
    Stmt ret = ExecScopeStmt(scope, new_body);

    // Step 4: Remove the scope_id_def inside the scope
    ret = ScopeIdDefRemover::Remove(ret);

    // Step 5: Wrap with LetStmts for all scope id values
    for (auto it = scope_lets.rbegin(); it != scope_lets.rend(); ++it) {
      ret = LetStmt(it->first, it->second, ret);
    }

    // Step 6: Wrap with thread_extent attributes
    for (const auto& [tag, iv] : kernel_launch_params_.value()) {
      if (tag == "warp_id_in_cta") continue;
      ret = AttrStmt(iv, tirx::attr::thread_extent, iv->dom->extent, ret);
    }
    kernel_launch_params_ = std::nullopt;
    return ret;
  }

  void ExtractKernelLaunchParams(const ScopeIdDefVerifier::ScopeIdSet& id_set, const Target& target,
                                 LaunchParams* launch_params) {
    auto add_launch_param = [&](const ScopePair& pair, const std::string& prefix) {
      auto it = id_set.find(pair);
      if (it == id_set.end()) {
        return;
      }
      const auto& def = (*it).second;
      CHECK_LE(def->extents.size(), 3) << "ValueError: Only up to 3 extents are supported";
      for (size_t i = 0; i < def->extents.size(); i++) {
        std::string thread_tag = prefix + static_cast<char>('x' + i);
        IterVar iv(Range::FromMinExtent(0, def->extents[i]), Var(thread_tag),
                   IterVarType::kThreadIndex, thread_tag);
        launch_params->insert({ffi::String(prefix + static_cast<char>('x' + i)), iv});
      }
    };
    // blockIdx.x, blockIdx.y, blockIdx.z
    auto it = id_set.find(ScopePair("cluster", "cta"));
    if (it == id_set.end() || is_one((*it).second.fused_extent())) {
      // no cluster
      add_launch_param(ScopePair("kernel", "cta"), "blockIdx.");
    } else {
      // use cluster
      // clusterCtaIdx.x, clusterCtaIdx.y, clusterCtaIdx.z
      CHECK(target->kind->name == "cuda") << "ValueError: cluster is only supported in CUDA";
      CHECK_EQ(target->kind->default_device_type, kDLCUDA)
          << "ValueError: cluster is only supported in CUDA";
      add_launch_param(ScopePair("cluster", "cta"), "clusterCtaIdx.");
      add_launch_param(ScopePair("kernel", "cta"), "blockIdx.");
    }
    // threadIdx.x, threadIdx.y, threadIdx.z
    add_launch_param(ScopePair("cta", "thread"), "threadIdx.");
  }

  /*! \brief The launch params of current kernel scope */
  ffi::Optional<LaunchParams> kernel_launch_params_{std::nullopt};
  /*! \brief The arithmetic analyzer */
  arith::Analyzer ana_;
  const Target& target_;
};

namespace {
Target ResolveTarget(const PrimFunc& f) {
  auto target = f->GetAttr<Target>(tvm::attr::kTarget);
  if (!target.defined()) {
    target = Target::Current(false);
  }
  return target.value();
}
}  // namespace

namespace transform {

Pass LowerTIRxResolveScopeIds() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    Target target = ResolveTarget(f);
    auto* n = f.CopyOnWrite();
    n->body = ScopeIdDefResolver::Resolve(n->body, target);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tirx.LowerTIRxResolveScopeIds", {});
}

}  // namespace transform
}  // namespace tirxxxx
}  // namespace tvm
