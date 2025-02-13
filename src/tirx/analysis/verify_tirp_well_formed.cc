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
 * \file tir/analysis/verify_tirp_well_formed.cc
 * \brief Check if the TIR+ program is well-formed.
 */

#include <tvm/arith/analyzer.h>
#include <tvm/ir/op.h>
#include <tvm/runtime/registry.h>
#include <tvm/tirx/exec_scope.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/tirp_op.h>

#include <exception>
#include <optional>
#include <tuple>
#include <variant>

#include "../ir/functor_common.h"
#include "../ir/tir_visitor_with_path.h"
#include "tvm/ir/module.h"

namespace tvm {
namespace tirx {

class ExecScopeVerifier : public Verifier<ExecScopeVerifier> {
 public:
  using Verifier::Verifier;

 private:
  using Verifier::Visit;

  void VisitStmt_(const BlockNode* op, ObjectPath path) override {
    // C0: exec_scope is defined
    auto roof = cur_roof_;
    Verify(op->exec_scope != nullptr) << "TIRpError: Block at " << path << " has no exec_scope";
    auto scope = op->exec_scope.value();
    // C1: exec_scope is valid
    Verify(ExecScope::Valid(scope->name))
        << "TIRpError: Block at " << path << " has unknown exec_scope " << scope->name;
    // C2: exec_scope is valid for root
    if (scope_stack_.empty()) {
      Verify(scope.Is("world") || scope.Is("kernel"))
          << "TIRpError: Block at " << path << " has invalid exec_scope " << scope->name
          << " as root";
    } else {
      // C3: exec_scope is valid for nested scope
      if (scope_stack_.back().Higher(scope)) {
        cur_roof_ = scope_stack_.back();
      } else if (scope_stack_.back().Is(scope->name)) {
        // do nothing
      } else {
        ICHECK(cur_roof_.defined()) << "TIRpError: root scope should be the highest scope";
        Verify(!scope.Higher(cur_roof_.value()))
            << "TIRpError: Block at " << path << " has invalid exec_scope " << scope->name
            << " under " << cur_roof_.value()->name;
      }
    }
    // C4: exec_scope slice is consistent
    bool erase_slices{false}, erase_select_cond{false}, pop_scope{false};
    arith::Analyzer ana;
    auto covered = [&](const Array<Range>& l, const Array<Range>& r) -> bool {
      // r's range is covered by l's range
      if (l.size() != r.size()) {
        return false;
      }
      for (size_t i = 0; i < l.size(); ++i) {
        if (!ana.CanProve(l[i]->min <= r[i]->min &&
                          l[i]->min + l[i]->extent >= r[i]->min + r[i]->extent)) {
          return false;
        }
      }
      return true;
    };
    if (const auto* slice = scope.as<ExecScopeSliceNode>()) {
      auto it_slices = scope_slices_.find(slice->name);
      auto it_select_cond = scope_select_cond_.find(slice->name);
      if (auto cond = slice->slice.as<PrimExpr>()) {
        // current scope slice has select_cond, it should not have slices
        Verify(it_slices == scope_slices_.end())
            << "TIRpError: ExecScopeSlice at " << path << " has both slices and select_cond";
        if (it_select_cond == scope_select_cond_.end()) {
          scope_select_cond_[slice->name] = cond.value();
          erase_select_cond = true;
        }
      } else {
        // current scope slice has slices, it should not have select_cond
        Verify(it_select_cond == scope_select_cond_.end())
            << "TIRpError: ExecScopeSlice at " << path << " has both slices and select_cond";
        auto slices = slice->slice.as<Array<Range>>().value();
        if (it_slices == scope_slices_.end()) {
          scope_slices_[slice->name] = {slices};
          erase_slices = true;
        } else {
          bool consistent = true;
          for (const auto& s : it_slices->second) {
            if (!covered(s, slices) && !covered(slices, s)) {
              consistent = false;
              break;
            }
          }
          Verify(consistent) << "TIRpError: ExecScopeSlice at " << path << " is inconsistent with "
                             << it_slices->first;
          it_slices->second.push_back(slices);
          pop_scope = true;
        }
      }
    }
    scope_stack_.push_back(scope);
    Verifier::VisitStmt_(op, path);
    scope_stack_.pop_back();
    // C5: cleanup scope slice when exiting scope
    if (const auto* slice = scope.as<ExecScopeSliceNode>()) {
      if (erase_slices) {
        scope_slices_.erase(slice->name);
      }
      if (erase_select_cond) {
        scope_select_cond_.erase(slice->name);
      }
      if (pop_scope) {
        auto it_slices = scope_slices_.find(slice->name);
        ICHECK(it_slices != scope_slices_.end());
        ICHECK(!it_slices->second.empty());
        it_slices->second.pop_back();
      }
    }
    cur_roof_ = roof;
  }

  void VisitStmt_(const tirp::OpCallNode* op, ObjectPath path) override {
    static const tvm::OpAttrMap<Bool>& tirp_op_map_ = Op::GetAttrMap<Bool>("TIsTIRpOp");
    Verify(tirp_op_map_.count(op->op))
        << "TIRpError: OpCall at " << path << " has unknown TIR+ op " << op->op;
  }

  Optional<ExecScope> cur_roof_ = NullOpt;
  std::vector<ExecScope> scope_stack_;
  std::unordered_map<std::string, std::vector<Array<Range>>> scope_slices_;
  std::unordered_map<std::string, PrimExpr> scope_select_cond_;
};

class ScopeIdVerifier : public Verifier<ScopeIdVerifier> {
 public:
  using Verifier::Verifier;

 private:
  using Verifier::Visit;

  void VisitStmt_(const BlockNode* op, ObjectPath path) override {
    Verify(op->exec_scope.defined())
        << "InternalError: exec_scope is not defined for block at " << path;
    const auto& scope = op->exec_scope.value();
    if (auto opt_kernel = scope.as<KernelScope>()) {
      ScopeIdDefVerifier verifier;
      Verify(verifier.Verify(opt_kernel.value()->scope_id_def))
          << "TIRpError: Kernel at " << path << " has invalid scope_id_def";
    }
  }

  arith::Analyzer ana_;
};

class LayoutVerifier : public Verifier<LayoutVerifier> {
 public:
  using Verifier::Verifier;

 private:
  using Verifier::Visit;

  void VisitStmt_(const BlockNode* op, ObjectPath path) override {
    auto verify = [&](const Buffer& buffer) {
      if (buffer->layout.defined()) {
        Verify(buffer->layout.value()->VerifyWellFormed())
            << "TIRpError: Buffer at " << path << " has invalid layout " << buffer->layout;
        ICHECK(buffer->layout.value()->CompatibleWithShape(buffer->shape))
            << "TIRpError: Buffer at " << path << " has layout " << buffer->layout
            << " that is not compatible with shape " << buffer->shape;
      }
    };
    for (const auto& view : op->buffer_views) {
      verify(view->dst_buffer);
    }
    for (const auto& alloc : op->alloc_buffers) {
      verify(alloc);
    }
    Verifier::VisitStmt_(op, path);
  }
};

class AsyncStructsVerifier : public Verifier<AsyncStructsVerifier> {
 public:
  using Verifier::Verifier;

 private:
  using Verifier::Visit;

  void VisitStmt_(const BlockNode* op, ObjectPath path) override {
    Verify(op->exec_scope != nullptr) << "TIRpError: Block at " << path << " has no exec_scope";
    auto scope = op->exec_scope.value();
    scope_stack_.push_back(scope);
    Verifier::VisitStmt_(op, path);
    scope_stack_.pop_back();
  }

  std::vector<ExecScope> scope_stack_;
};

bool VerifyTIRpWellFormed(const PrimFunc& func, bool assert_mode) {
  if (!ExecScopeVerifier::Verify(func, assert_mode)) {
    return false;
  }
  if (!ScopeIdVerifier::Verify(func, assert_mode)) {
    return false;
  }
  if (!LayoutVerifier::Verify(func, assert_mode)) {
    return false;
  }
  if (!AsyncStructsVerifier::Verify(func, assert_mode)) {
    return false;
  }
  return true;
}

bool VerifyTIRpWellFormed(const IRModule& mod, bool assert_mode) {
  for (const auto& [gvar, base_func] : mod->functions) {
    if (auto prim_func = base_func.as<PrimFunc>()) {
      bool res = VerifyTIRpWellFormed(prim_func.value(), assert_mode);
      if (!res) {
        return false;
      }
    }
  }
  return true;
}

TVM_REGISTER_GLOBAL("tirx.analysis.VerifyTIRpWellFormed")
    .set_body_typed([](const ObjectRef& obj, bool assert_mode) {
      if (auto n = obj.as<PrimFunc>()) {
        return VerifyTIRpWellFormed(n.value(), assert_mode);
      } else if (auto n = obj.as<IRModule>()) {
        return VerifyTIRpWellFormed(n.value(), assert_mode);
      } else {
        LOG(FATAL) << "Expects PrimFunc or IRModule,  but get " << obj->GetTypeKey() << " instead.";
        return false;
      }
    });

}  // namespace tirxxx
}  // namespace tvm
