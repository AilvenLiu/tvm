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
 * \file lower_tirx_scope_slices.cc
 * \brief Resolve exec scope slices introduced during TIRx lowering.
 */

#include <tvm/runtime/logging.h>
#include <tvm/target/target.h>
#include <tvm/tirx/exec_scope.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/tirx_op.h>
#include <tvm/tirx/transform.h>
=======
#include <tvm/tirx/exec_scope.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/tirx_op.h>
#include <tvm/tirx/transform.h>
>>>>>>> e003fe6cd1 ([Refactor] Decouple exec_scope from SBlock into independent ExecScopeStmt node (#469)):src/tir/transform/lower_tirx_scope_slices.cc

namespace tvm {
namespace tirx {

class ExecScopeSliceResolver : public StmtExprMutator {
 public:
  explicit ExecScopeSliceResolver(const Target& target) : target_(target) {}

  static Stmt Resolve(const Stmt& stmt, const Target& target) {
    return ExecScopeSliceResolver(target)(stmt);
  }

 private:
  using LaunchParams = ScopeIdResolveTable::LaunchParams;

  /*!
   * \brief Check whether the body of an ExecScopeStmt contains an AttrStmt
   *        with the given key (used to detect tirx.scope_partition).
   *        If found, return the inner body (with the AttrStmt stripped).
   */
  static ffi::Optional<Stmt> StripBodyAttr(const Stmt& body, const char* attr_key) {
    // The AttrStmt may be at the outermost level of body
    if (auto attr = body.as<AttrStmtNode>()) {
      if (attr->attr_key == attr_key) {
        return attr->body;
      }
    }
    return ffi::Optional<Stmt>(std::nullopt);
  }

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == tirx::attr::thread_extent) {
      auto iv = op->node.as<IterVar>();
      ICHECK(iv.has_value()) << "Internal Error: thread_extent should annotate an IterVar";
      launch_params_[iv.value()->thread_tag] = iv.value();
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const LetStmtNode* op) final {
    // Detect warp_id_in_cta LetStmt introduced by the scope ID resolver
    if (op->var->name_hint == "warp_id_in_cta") {
      warp_id_in_cta_var_ = op->var;
      warp_id_in_cta_value_ = op->value;
      IterVar warp_iv(Range::FromMinExtent(0, 1), op->var, kThreadIndex, "warp_id_in_cta");
      launch_params_["warp_id_in_cta"] = warp_iv;
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const ExecScopeStmtNode* op) final {
    auto exec_scope = op->exec_scope;
    bool is_kernel = exec_scope->Is("kernel");
    if (is_kernel) {
      need_warp_id_in_cta_letstmt_ = false;
    }

    // Check for scope_partition AttrStmt wrapping the body
    auto stripped_body = StripBodyAttr(op->body, attr::tirx_scope_partition);
    if (stripped_body.defined()) {
      // Peel off any LetStmts wrapping the body (e.g., from scope_id resolution)
      Stmt inner_body = stripped_body.value();
      std::vector<std::pair<Var, PrimExpr>> let_stmts;
      while (auto let_node = inner_body.as<LetStmtNode>()) {
        let_stmts.push_back({let_node->var, let_node->value});
        inner_body = let_node->body;
      }

      auto seq = inner_body.as<SeqStmt>();
      CHECK(seq.has_value())
          << "TIRxError: ExecScopeStmt with scope partition has invalid body " << op->body;
      Array<Stmt> new_seq;
      for (const auto& stmt : seq.value()->seq) {
        new_seq.push_back(VisitStmt(stmt));
      }
      // Connect the IfThenElse stmts into a single IfThenElse
      Stmt body = new_seq[new_seq.size() - 1];
      for (int i = new_seq.size() - 2; i >= 0; i--) {
        auto if_then = new_seq[i].as<IfThenElse>();
        CHECK(if_then.has_value() && !if_then.value()->else_case.defined())
            << "TIRxError: ExecScopeStmt with scope partition has invalid body " << op->body;
        body = IfThenElse(if_then.value()->condition, if_then.value()->then_case, body);
      }
      // Re-wrap with peeled LetStmts (in reverse order)
      for (int i = let_stmts.size() - 1; i >= 0; i--) {
        body = LetStmt(let_stmts[i].first, let_stmts[i].second, body);
      }

      // If this node is also a scope_slice, continue to resolve the slice
      // (don't return yet — fall through to scope_slice handling below)
      auto scope_slice_opt = exec_scope.as<ExecScopeSlice>();
      if (scope_slice_opt.has_value()) {
        auto scope_slice = scope_slice_opt.value();
        auto scope = ScopePair(scope_slice->parent, scope_slice->name);
        int out_dim = scope_slice->slices.as<PrimExpr>().has_value()
                          ? 1
                          : scope_slice->slices.as<Array<Range>>().value().size();
        if (ScopeIdResolveTable::NeedWarpIdInCta(scope_slice->name) &&
            launch_params_.find("warp_id_in_cta") == launch_params_.end()) {
          PrimExpr shuffled = ScopeIdResolveTable::ComputeWarpIdInCta(launch_params_);
          warp_id_in_cta_var_ = Var("warp_id_in_cta", shuffled.dtype());
          warp_id_in_cta_value_ = shuffled;
          need_warp_id_in_cta_letstmt_ = true;
          IterVar warp_iv(Range::FromMinExtent(0, 1), warp_id_in_cta_var_.value(), kThreadIndex,
                          "warp_id_in_cta");
          launch_params_["warp_id_in_cta"] = warp_iv;
        }
        Array<PrimExpr> resolved = ScopeIdResolveTable::Resolve(
            scope, scope_slice->extents, out_dim, target_->kind->name, launch_params_);
        ICHECK_EQ(resolved.size(), out_dim);
        auto plain_scope = ExecScope::Create(scope_slice->name);
        Stmt inner = ExecScopeStmt(plain_scope, body);
        if (auto select_cond = scope_slice->slices.as<PrimExpr>()) {
          return IfThenElse(select_cond.value(), inner);
        } else {
          auto slices = scope_slice->slices.as<Array<Range>>().value();
          PrimExpr cond = Bool(true);
          for (size_t i = 0; i < slices.size(); i++) {
            cond = cond && resolved[i] >= slices[i]->min &&
                   resolved[i] < slices[i]->extent + slices[i]->min;
          }
          return IfThenElse(cond, inner);
        }
      }

      Stmt result = ExecScopeStmt(exec_scope, body);
      if (is_kernel && need_warp_id_in_cta_letstmt_) {
        auto exec_scope_stmt = result.as<ExecScopeStmtNode>();
        result = ExecScopeStmt(exec_scope_stmt->exec_scope,
                               LetStmt(warp_id_in_cta_var_.value(), warp_id_in_cta_value_.value(),
                                       exec_scope_stmt->body));
        need_warp_id_in_cta_letstmt_ = false;
      }
      return result;
    }

    // Check for scope_slice on the ExecScopeStmt
    auto scope_slice_opt = exec_scope.as<ExecScopeSlice>();
    if (!scope_slice_opt.has_value()) {
      // No scope slice — just visit body
      Stmt body = VisitStmt(op->body);
      Stmt result = ExecScopeStmt(exec_scope, body);
      if (is_kernel && need_warp_id_in_cta_letstmt_) {
        // Wrap the body with the warp_id_in_cta LetStmt
        auto exec_scope_stmt = result.as<ExecScopeStmtNode>();
        result = ExecScopeStmt(exec_scope_stmt->exec_scope,
                               LetStmt(warp_id_in_cta_var_.value(), warp_id_in_cta_value_.value(),
                                       exec_scope_stmt->body));
        need_warp_id_in_cta_letstmt_ = false;
      }
      return result;
    }
    // Has scope slice — resolve it
    auto scope_slice = scope_slice_opt.value();
    auto scope = ScopePair(scope_slice->parent, scope_slice->name);
    int out_dim = scope_slice->slices.as<PrimExpr>().has_value()
                      ? 1
                      : scope_slice->slices.as<Array<Range>>().value().size();
    if (ScopeIdResolveTable::NeedWarpIdInCta(scope_slice->name) &&
        launch_params_.find("warp_id_in_cta") == launch_params_.end()) {
      PrimExpr shuffled = ScopeIdResolveTable::ComputeWarpIdInCta(launch_params_);
      warp_id_in_cta_var_ = Var("warp_id_in_cta", shuffled.dtype());
      warp_id_in_cta_value_ = shuffled;
      need_warp_id_in_cta_letstmt_ = true;
      IterVar warp_iv(Range::FromMinExtent(0, 1), warp_id_in_cta_var_.value(), kThreadIndex,
                      "warp_id_in_cta");
      launch_params_["warp_id_in_cta"] = warp_iv;
    }
    Array<PrimExpr> resolved = ScopeIdResolveTable::Resolve(
        scope, scope_slice->extents, out_dim, target_->kind->name, launch_params_);
    ICHECK_EQ(resolved.size(), out_dim);
    auto plain_scope = ExecScope::Create(scope_slice->name);
    Stmt body = VisitStmt(op->body);
    Stmt inner = ExecScopeStmt(plain_scope, body);
    if (auto select_cond = scope_slice->slices.as<PrimExpr>()) {
      return IfThenElse(select_cond.value(), inner);
    } else {
      auto slices = scope_slice->slices.as<Array<Range>>().value();
      PrimExpr cond = Bool(true);
      for (size_t i = 0; i < slices.size(); i++) {
        cond = cond && resolved[i] >= slices[i]->min &&
               resolved[i] < slices[i]->extent + slices[i]->min;
      }
      return IfThenElse(cond, inner);
    }
  }

  LaunchParams launch_params_;
  const Target& target_;
  ffi::Optional<Var> warp_id_in_cta_var_;
  ffi::Optional<PrimExpr> warp_id_in_cta_value_;
  bool need_warp_id_in_cta_letstmt_ = false;
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

Pass LowerTIRxResolveScopeSlices() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    Target target = ResolveTarget(f);
    auto* n = f.CopyOnWrite();
    n->body = ExecScopeSliceResolver::Resolve(n->body, target);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tirx.LowerTIRxResolveScopeSlices", {});
}

}  // namespace transform
}  // namespace tirxxxx
}  // namespace tvm
