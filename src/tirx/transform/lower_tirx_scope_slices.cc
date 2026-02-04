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
<<<<<<<< HEAD:src/tirx/transform/lower_tirp_scope_slices.cc
#include <tvm/tirx/exec_scope.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/tirp_op.h>
#include <tvm/tirx/transform.h>
========
#include <tvm/tirx/exec_scope.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/tirx_op.h>
#include <tvm/tirx/transform.h>
>>>>>>>> bd927f10c7 (rename):src/tirx/transform/lower_tirx_scope_slices.cc

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
    auto resolved = ScopeIdResolveTable::Resolve(scope, scope_slice->extents, out_dim,
                                                 target_->kind->name, launch_params_);
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
<<<<<<<< HEAD:src/tirx/transform/lower_tirp_scope_slices.cc
  return CreatePrimFuncPass(pass_func, 0, "tirx.LowerTIRpResolveScopeSlices", {});
========
  return CreatePrimFuncPass(pass_func, 0, "tirx.LowerTIRxResolveScopeSlices", {});
>>>>>>>> bd927f10c7 (rename):src/tirx/transform/lower_tirx_scope_slices.cc
}

}  // namespace transform
}  // namespace tirxx
}  // namespace tvm
