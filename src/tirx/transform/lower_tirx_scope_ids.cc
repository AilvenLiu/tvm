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
  struct DefWithSourceScope {
    ScopeIdDef def;
    ffi::String source_scope;
  };

  static std::vector<DefWithSourceScope> Gather(const Stmt& stmt) {
    ScopeIdDefGather gather;
    gather(stmt);
    return gather.scope_id_def;
  }

  void VisitStmt_(const ExecScopeStmtNode* op) override {
    StmtExprVisitor::VisitStmt_(op);
    for (const auto& def : op->exec_scope->scope_id_def) {
      scope_id_def.push_back({def, op->exec_scope->name});
    }
  }

  std::vector<DefWithSourceScope> scope_id_def;
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
  class ImplicitScopeIdEvalInjector : public StmtExprMutator {
   public:
    static Stmt Inject(const Stmt& stmt,
                       const std::vector<std::pair<Var, ffi::String>>& eval_specs) {
      ImplicitScopeIdEvalInjector injector(eval_specs);
      return injector(stmt);
    }

   private:
    explicit ImplicitScopeIdEvalInjector(
        const std::vector<std::pair<Var, ffi::String>>& eval_specs) {
      for (const auto& [var, scope] : eval_specs) {
        eval_map_[scope.operator std::string()].push_back(var);
      }
    }

    Stmt VisitStmt_(const ExecScopeStmtNode* op) final {
      Stmt body = VisitStmt(op->body);
      auto it = eval_map_.find(op->exec_scope->name.operator std::string());
      if (it != eval_map_.end() && !it->second.empty()) {
        ffi::Array<Stmt> evals;
        evals.reserve(it->second.size());
        for (const Var& var : it->second) {
          evals.push_back(Evaluate(var));
        }
        body = SeqStmt::Flatten(evals, body);
        eval_map_.erase(it);
      }
      if (body.same_as(op->body)) {
        return ffi::GetRef<Stmt>(op);
      }
      return ExecScopeStmt(op->exec_scope, body);
    }

    std::unordered_map<std::string, std::vector<Var>> eval_map_;
  };

  using LaunchParams = ScopeIdResolveTable::LaunchParams;

  static bool IsImplicitScopeId(const Var& var) {
    // Parser records unnamed scope-id defs as empty-name Vars.
    return var->name_hint.empty();
  }

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
    TVM_FFI_ICHECK(!scope->Is("world")) << "TIRx Error: world scope is not supported at the moment";
    TVM_FFI_ICHECK(!kernel_launch_params_.has_value())
        << "TIRx Error: nested kernel scopes are not supported";

    // Step 0: Gather the scope id defs from all nested scopes
    std::vector<ScopeIdDefGather::DefWithSourceScope> gathered_scope_id_def =
        ScopeIdDefGather::Gather(ffi::GetRef<Stmt>(op));
    Array<ScopeIdDef> scope_id_def;
    scope_id_def.reserve(gathered_scope_id_def.size());
    for (const auto& item : gathered_scope_id_def) {
      scope_id_def.push_back(item.def);
    }

    // Step 1: Verify the ScopeIdDef is well-formed
    ScopeIdDefVerifier verifier;
    TVM_FFI_ICHECK(verifier.Verify(scope_id_def)) << "Inconsistent ScopeIdDef";

    // Step 2: Extract kernel launch parameters
    LaunchParams launch_params;
    ExtractKernelLaunchParams(verifier.id_set, target_, &launch_params);

    std::unordered_map<Var, PrimExpr, ObjectPtrHash, ObjectPtrEqual> id_map;
    std::vector<std::pair<Var, PrimExpr>> scope_binds;
    std::vector<std::pair<Var, ffi::String>> implicit_scope_id_evals;

    if (launch_params.count("threadIdx.x") > 0) {
      PrimExpr shuffled = ScopeIdResolveTable::ComputeWarpIdInCta(launch_params);
      Var warp_id_in_cta_var("warp_id_in_cta", shuffled.dtype());
      scope_binds.push_back({warp_id_in_cta_var, shuffled});
      IterVar warp_iv(Range::FromMinExtent(0, 1), warp_id_in_cta_var, kThreadIndex,
                      "warp_id_in_cta");
      launch_params.insert({"warp_id_in_cta", warp_iv});
    }
    kernel_launch_params_ = launch_params;

    for (const auto& def_with_scope : gathered_scope_id_def) {
      const ScopeIdDef& def = def_with_scope.def;
      auto resolved =
          ScopeIdResolveTable::Resolve(def->scope, def->extents, def->extents.size(),
                                       target_->kind->name, kernel_launch_params_.value());
      TVM_FFI_ICHECK_EQ(resolved.size(), def->extents.size())
          << "Internal Error: Inconsistent resolved size " << resolved.size() << " vs "
          << def->extents.size();
      for (size_t i = 0; i < def->def_ids.size(); i++) {
        PrimExpr value = resolved[i];
        Var let_var(def->def_ids[i]->name_hint, value.dtype());
        id_map[def->def_ids[i]] = let_var;
        scope_binds.push_back({let_var, value});
        if (IsImplicitScopeId(def->def_ids[i])) {
          implicit_scope_id_evals.push_back({let_var, def_with_scope.source_scope});
        }
      }
    }

    // Step 3: Substitute scope ids in body, then visit
    Stmt new_body = Substitute(op->body, id_map);
    new_body = VisitStmt(new_body);
    Stmt ret = ExecScopeStmt(scope, new_body);

    // Step 4: Remove the scope_id_def inside the scope
    ret = ScopeIdDefRemover::Remove(ret);

    // Step 5: Emit Bind statements for all scope id values.
    ffi::Array<Stmt> bind_stmts;
    bind_stmts.reserve(scope_binds.size());
    for (const auto& [var, value] : scope_binds) {
      bind_stmts.push_back(Bind(var, value));
    }
    ret = SeqStmt::Flatten(bind_stmts, ret);

    // Step 6: Wrap with thread_extent attributes
    for (const auto& [tag, iv] : kernel_launch_params_.value()) {
      if (tag == "warp_id_in_cta") continue;
      ret = AttrStmt(iv, tirx::attr::thread_extent, iv->dom->extent, ret);
    }
    ret = ImplicitScopeIdEvalInjector::Inject(ret, implicit_scope_id_evals);
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
      TVM_FFI_ICHECK_LE(def->extents.size(), 3) << "ValueError: Only up to 3 extents are supported";
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
      TVM_FFI_ICHECK(target->kind->name == "cuda")
          << "ValueError: cluster is only supported in CUDA";
      TVM_FFI_ICHECK_EQ(target->kind->default_device_type, kDLCUDA)
          << "ValueError: cluster is only supported in CUDA";
      add_launch_param(ScopePair("cluster", "cta"), "clusterCtaIdx.");
      // Preferred cluster size (CUDA 12.8+)
      const auto& cta_def = (*it).second;
      if (cta_def->preferred_extents.defined()) {
        const auto& pref = cta_def->preferred_extents.value();
        for (size_t i = 0; i < pref.size(); i++) {
          std::string tag = "preferredClusterCtaIdx." + std::string(1, 'x' + i);
          IterVar iv(Range::FromMinExtent(0, pref[i]), Var(tag), IterVarType::kThreadIndex, tag);
          launch_params->insert({ffi::String(tag), iv});
        }
      }
      add_launch_param(ScopePair("kernel", "cta"), "blockIdx.");
    }
    // threadIdx.x, threadIdx.y, threadIdx.z
    add_launch_param(ScopePair("cta", "thread"), "threadIdx.");
    if (!id_set.empty()) {
      TVM_FFI_ICHECK(launch_params->count("threadIdx.x") > 0)
          << "ValueError: kernel has no thread launch parameters. "
          << "At minimum, declare cta→thread extent (e.g., Tx.thread_id([128]))";
    }
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
}  // namespace tirx
}  // namespace tvm
