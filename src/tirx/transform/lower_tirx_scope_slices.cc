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
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/exec_scope.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/tirx_op.h>
#include <tvm/tirx/transform.h>

namespace tvm {
namespace tirx {

class ExecScopeSliceResolver : public StmtExprMutator {
 public:
  explicit ExecScopeSliceResolver(const Target& target) : target_(target) {}

  static Stmt Resolve(const Stmt& stmt, const Target& target) {
    return ExecScopeSliceResolver(target)(stmt);
  }

 private:
  using StmtExprMutator::VisitStmt_;
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
      TVM_FFI_ICHECK(iv.has_value()) << "Internal Error: thread_extent should annotate an IterVar";
      launch_params_[iv.value()->thread_tag] = iv.value();
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const BindNode* op) final {
    if (op->var->name_hint == "warp_id_in_cta") {
      IterVar warp_iv(Range::FromMinExtent(0, 1), op->var, kThreadIndex, "warp_id_in_cta");
      launch_params_["warp_id_in_cta"] = warp_iv;
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  /*! \brief Resolve an ExecScopeSlice into IfThenElse(condition, ExecScopeStmt(plain_scope, body))
   */
  Stmt ResolveSliceToIfThenElse(const ExecScopeSlice& scope_slice, const Stmt& body) {
    auto scope = ScopePair(scope_slice->parent, scope_slice->name);
    int out_dim = scope_slice->slices.as<PrimExpr>().has_value()
                      ? 1
                      : scope_slice->slices.as<Array<Range>>().value().size();
    Array<PrimExpr> resolved = ScopeIdResolveTable::Resolve(scope, scope_slice->extents, out_dim,
                                                            target_->kind->name, launch_params_);
    TVM_FFI_ICHECK_EQ(resolved.size(), out_dim);
    auto plain_scope = ExecScope::Create(scope_slice->name);
    Stmt inner = ExecScopeStmt(plain_scope, body);
    if (auto select_cond = scope_slice->slices.as<PrimExpr>()) {
      return IfThenElse(select_cond.value(), inner);
    }
    auto slices = scope_slice->slices.as<Array<Range>>().value();
    PrimExpr cond = Bool(true);
    for (size_t i = 0; i < slices.size(); i++) {
      cond =
          cond && resolved[i] >= slices[i]->min && resolved[i] < slices[i]->extent + slices[i]->min;
    }
    return IfThenElse(cond, inner);
  }

  Stmt VisitStmt_(const ExecScopeStmtNode* op) final {
    auto exec_scope = op->exec_scope;

    // Check for scope_partition AttrStmt wrapping the body
    auto stripped_body = StripBodyAttr(op->body, attr::tirx_scope_partition);
    if (stripped_body.defined()) {
      // Peel off any leading Bind stmts wrapping the body (e.g., from scope_id resolution)
      Stmt inner_body = stripped_body.value();
      ffi::Array<Stmt> leading_binds;
      if (auto seq = inner_body.as<SeqStmtNode>()) {
        size_t bind_count = 0;
        while (bind_count < seq->seq.size() && seq->seq[bind_count].as<BindNode>()) {
          leading_binds.push_back(seq->seq[bind_count]);
          ++bind_count;
        }
        if (bind_count > 0) {
          ffi::Array<Stmt> rest;
          rest.reserve(seq->seq.size() - bind_count);
          for (size_t i = bind_count; i < seq->seq.size(); ++i) {
            rest.push_back(seq->seq[i]);
          }
          inner_body = SeqStmt::Flatten(rest);
        }
      }

      auto seq = inner_body.as<SeqStmt>();
      TVM_FFI_ICHECK(seq.has_value())
          << "TIRxError: ExecScopeStmt with scope partition has invalid body " << op->body;
      Array<Stmt> new_seq;
      for (const auto& stmt : seq.value()->seq) {
        new_seq.push_back(VisitStmt(stmt));
      }
      // Connect the IfThenElse stmts into a single IfThenElse
      Stmt body = new_seq[new_seq.size() - 1];
      for (int i = new_seq.size() - 2; i >= 0; i--) {
        auto if_then = new_seq[i].as<IfThenElse>();
        TVM_FFI_ICHECK(if_then.has_value() && !if_then.value()->else_case.defined())
            << "TIRxError: ExecScopeStmt with scope partition has invalid body " << op->body;
        body = IfThenElse(if_then.value()->condition, if_then.value()->then_case, body);
      }
      // Re-wrap with peeled leading Bind stmts.
      if (!leading_binds.empty()) {
        body = SeqStmt::Flatten(leading_binds, body);
      }

      // If this node is also a scope_slice, resolve it
      auto scope_slice_opt = exec_scope.as<ExecScopeSlice>();
      if (scope_slice_opt.has_value()) {
        return ResolveSliceToIfThenElse(scope_slice_opt.value(), body);
      }
      return ExecScopeStmt(exec_scope, body);
    }

    // Check for scope_slice on the ExecScopeStmt
    auto scope_slice_opt = exec_scope.as<ExecScopeSlice>();
    if (scope_slice_opt.has_value()) {
      Stmt body = VisitStmt(op->body);
      return ResolveSliceToIfThenElse(scope_slice_opt.value(), body);
    }

    // No scope slice — just visit body
    Stmt body = VisitStmt(op->body);
    return ExecScopeStmt(exec_scope, body);
  }

  LaunchParams launch_params_;
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
}  // namespace tirx
}  // namespace tvm
