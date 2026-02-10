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
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/tirx_op.h>
#include <tvm/tirx/transform.h>
=======
#include <tvm/tirx/exec_scope.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/tirx_op.h>
#include <tvm/tirx/transform.h>
>>>>>>> 7e7a455d25 ([LowerTIRx] Generate warp-uniform warp_id when lowering scope id (#467)):src/tir/transform/lower_tirx_scope_slices.cc

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

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == tirx::attr::thread_extent) {
      auto iv = op->node.as<IterVar>();
      ICHECK(iv.has_value()) << "Internal Error: thread_extent should annotate an IterVar";
      launch_params_[iv.value()->thread_tag] = iv.value();
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const SBlockRealizeNode* op) final {
    Stmt block = this->VisitStmt(op->block);
    if (block.same_as(op->block)) {
      return ffi::GetRef<Stmt>(op);
    } else {
      if (auto* block_ptr = block.as<SBlockNode>()) {
        auto n = CopyOnWrite(op);
        n->block = ffi::GetRef<SBlock>(block_ptr);
        return Stmt(n);
      } else {
        return block;
      }
    }
  }

  Stmt VisitStmt_(const SBlockNode* op) final {
    SBlock block = ffi::GetRef<SBlock>(op);
    auto* n = block.CopyOnWrite();
    bool is_kernel = op->exec_scope.defined() && op->exec_scope.value()->Is("kernel");
    if (is_kernel) {
      need_warp_id_in_cta_letstmt_ = false;
      if (op->annotations.count("tirx.warp_id_in_cta")) {
        warp_id_in_cta_var_ = Downcast<Var>(op->annotations.at("tirx.warp_id_in_cta"));
        // Add warp_id_in_cta to launch params so resolve functions can access it
        IterVar warp_iv(Range::FromMinExtent(0, 1), warp_id_in_cta_var_.value(), kThreadIndex,
                        "warp_id_in_cta");
        launch_params_["warp_id_in_cta"] = warp_iv;
        n->annotations.erase("tirx.warp_id_in_cta");
      }
    }
    if (op->annotations.count("tirx.scope_partition")) {
      // scope partition is enabled, rewrite the body
      auto seq = block->body.as<SeqStmt>();
      if (!seq.has_value()) {
        CHECK(false) << "TIRxError: Block with scope partition at has invalid body " << block->body;
      }
      Array<Stmt> new_seq;
      for (const auto& stmt : seq.value()->seq) {
        new_seq.push_back(VisitStmt(stmt));
      }
      // Connect the IfThenElse stmts into a single IfThenElse
      Stmt body = new_seq[new_seq.size() - 1];
      for (int i = new_seq.size() - 2; i >= 0; i--) {
        auto if_then = new_seq[i].as<IfThenElse>();
        CHECK(if_then.has_value() && !if_then.value()->else_case.defined())
            << "TIRxError: Block with scope partition has invalid body " << block->body;
        body = IfThenElse(if_then.value()->condition, if_then.value()->then_case, body);
      }
      n->body = body;
      n->annotations.erase("tirx.scope_partition");
    } else {
      // no scope partition, visit the body
      n->body = VisitStmt(n->body);
    }
    if (is_kernel && need_warp_id_in_cta_letstmt_) {
      n->body = LetStmt(warp_id_in_cta_var_.value(), warp_id_in_cta_value_.value(), n->body);
      need_warp_id_in_cta_letstmt_ = false;
    }
    // no scope partition, return the block as is
    if (!op->exec_scope.defined()) {
      return std::move(block);
    }
    auto exec_scope = op->exec_scope.value();
    auto scope_slice_opt = exec_scope.as<ExecScopeSlice>();
    if (!scope_slice_opt.has_value()) {
      // no scope slice, return the block as is
      return std::move(block);
    }
    auto scope_slice = scope_slice_opt.value();
    auto scope = ScopePair(scope_slice->parent, scope_slice->name);
    int out_dim = scope_slice->slices.as<PrimExpr>().has_value()
                      ? 1
                      : scope_slice->slices.as<Array<Range>>().value().size();
    // If the scope needs warp_id_in_cta and it's not yet in launch_params_, compute it on-demand
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
    n->exec_scope = ExecScope::Create(scope_slice->name);
    if (auto select_cond = scope_slice->slices.as<PrimExpr>()) {
      return IfThenElse(select_cond.value(), SBlockRealize({}, Bool(true), block));
    } else {
      auto slices = scope_slice->slices.as<Array<Range>>().value();
      PrimExpr cond = Bool(true);
      for (size_t i = 0; i < slices.size(); i++) {
        cond = cond && resolved[i] >= slices[i]->min &&
               resolved[i] < slices[i]->extent + slices[i]->min;
      }
      return IfThenElse(cond, SBlockRealize({}, Bool(true), block));
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
}  // namespace tirxxx
}  // namespace tvm
