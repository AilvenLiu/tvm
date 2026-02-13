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
 * \file tir/analysis/verify_tirx_well_formed.cc
 * \brief Check if the TIRX program is well-formed.
 */

#include <tvm/arith/analyzer.h>
#include <tvm/ir/op.h>
<<<<<<<< HEAD:src/tirx/analysis/verify_tirp_well_formed.cc
#include <tvm/tirx/exec_scope.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/tirp_op.h>
========
#include <tvm/tirx/exec_scope.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/tirx_op.h>
>>>>>>>> bd927f10c7 (rename):src/tirx/analysis/verify_tirx_well_formed.cc

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

  void VisitStmt_(const SBlockNode* op, ffi::reflection::AccessPath path) override {
    if (op->annotations.count(attr::tirx_scope_partition)) {
      // scope partition is enabled, check if body is a list of ExecScopeStmt
      if (auto seq = op->body.as<SeqStmt>()) {
        ffi::Optional<ExecScopeSlice> scope_slice_chk = std::nullopt;
        for (const auto& stmt : seq.value()->seq) {
          // Handle ExecScopeStmt children
          auto exec_scope_stmt = stmt.as<ExecScopeStmt>();
          if (!exec_scope_stmt.has_value()) {
            Verify(false) << "TIRxError: Block with scope partition at " << path
                          << " has invalid body " << op->body;
          }
          auto scope = exec_scope_stmt.value()->exec_scope;
          Verifier::VisitStmt_(exec_scope_stmt.value().get(), path);
          auto scope_slice = scope.as<ExecScopeSlice>();
          Verify(scope_slice.has_value())
              << "TIRxError: Block with scope partition at " << path
              << " has invalid exec_scope " << scope;
          if (scope_slice_chk.has_value()) {
            Verify(scope_slice_chk.value()->name == scope_slice.value()->name &&
                   scope_slice_chk.value()->parent == scope_slice.value()->parent)
                << "TIRxError: Block with scope partition at " << path
                << " has invalid exec_scope " << scope;
          }
          scope_slice_chk = scope_slice;
        }
      } else {
        Verify(false) << "TIRxError: Block with scope partition at " << path << " has invalid body "
                      << op->body;
      }
      return;
    }
    Verifier::VisitStmt_(op, path);
  }

  void VisitStmt_(const tirx::OpCallNode* op, ffi::reflection::AccessPath path) override {
    static const tvm::OpAttrMap<Bool>& tirx_op_map_ = Op::GetAttrMap<Bool>("TIsTIRxOp");
    Verify(tirx_op_map_.count(op->op))
        << "TIRxError: OpCall at " << path << " has unknown TIRX op " << op->op;
  }

  void VisitStmt_(const ExecScopeStmtNode* op, ffi::reflection::AccessPath path) override {
    auto scope = op->exec_scope;
    // C1: exec_scope is valid
    Verify(ExecScope::Valid(scope->name))
        << "TIRxError: ExecScopeStmt at " << path << " has unknown exec_scope " << scope->name;
    bool is_root = false;
    if (!root_.has_value()) {
      root_ = scope;
      is_root = true;
    }
    if (!scope_stack_.empty()) {
      ICHECK(root_.has_value()) << "TIRxError: root scope should be the highest scope";
      Verify(!scope->Higher(root_.value()))
          << "TIRxError: ExecScopeStmt at " << path << " has invalid exec_scope " << scope->name
          << " under " << root_.value()->name;
    }
    // C4: exec_scope slice consistency
    bool erase_slices{false}, erase_select_cond{false}, pop_scope{false};
    arith::Analyzer ana;
    auto covered = [&](const Array<Range>& l, const Array<Range>& r) -> bool {
      if (l.size() != r.size()) return false;
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
      if (auto cond = slice->slices.as<PrimExpr>()) {
        Verify(it_slices == scope_slices_.end())
            << "TIRxError: ExecScopeSlice at " << path << " has both slices and select_cond";
        if (it_select_cond == scope_select_cond_.end()) {
          scope_select_cond_[slice->name] = cond.value();
          erase_select_cond = true;
        }
      } else {
        Verify(it_select_cond == scope_select_cond_.end())
            << "TIRxError: ExecScopeSlice at " << path << " has both slices and select_cond";
        auto slices = slice->slices.as<Array<Range>>().value();
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
          Verify(consistent) << "TIRxError: ExecScopeSlice at " << path
                             << " is inconsistent with " << it_slices->first;
          it_slices->second.push_back(slices);
          pop_scope = true;
        }
      }
    }
    scope_stack_.push_back(scope);
    Verifier::VisitStmt_(op, path);
    scope_stack_.pop_back();
    if (const auto* slice = scope.as<ExecScopeSliceNode>()) {
      if (erase_slices) scope_slices_.erase(slice->name);
      if (erase_select_cond) scope_select_cond_.erase(slice->name);
      if (pop_scope) {
        auto it_slices = scope_slices_.find(slice->name);
        ICHECK(it_slices != scope_slices_.end());
        ICHECK(!it_slices->second.empty());
        it_slices->second.pop_back();
      }
    }
    if (is_root) root_ = std::nullopt;
  }

  ffi::Optional<ExecScope> root_ = std::nullopt;
  std::vector<ExecScope> scope_stack_;
  std::unordered_map<std::string, std::vector<Array<Range>>> scope_slices_;
  std::unordered_map<std::string, PrimExpr> scope_select_cond_;
};

class ScopeIdVerifier : public Verifier<ScopeIdVerifier> {
 public:
  using Verifier::Verifier;

 private:
  using Verifier::Visit;

  void VisitStmt_(const ExecScopeStmtNode* op, ffi::reflection::AccessPath path) override {
    const auto& scope = op->exec_scope;
    auto it = scope_id_def_.end();
    scope_id_def_.insert(it, scope->scope_id_def.begin(), scope->scope_id_def.end());
    Verifier::VisitStmt_(op, path);
    if (!scope->scope_id_def.empty()) {
      ScopeIdDefVerifier verifier;
      Verify(verifier.Verify(scope_id_def_))
          << "TIRxError: Scope at " << path << " has invalid scope_id_def";
    }
    scope_id_def_.erase(scope_id_def_.end() - scope->scope_id_def.size(), scope_id_def_.end());
  }

  Array<ScopeIdDef> scope_id_def_;
  arith::Analyzer ana_;
};

class LayoutVerifier : public Verifier<LayoutVerifier> {
 public:
  using Verifier::Verifier;

 private:
  using Verifier::Visit;

  void VisitStmt_(const SBlockNode* op, ffi::reflection::AccessPath path) override {
    auto verify = [&](const Buffer& buffer) {
      if (buffer->layout.defined()) {
        Verify(buffer->layout.value()->VerifyWellFormed())
            << "TIRxError: Buffer at " << path << " has invalid layout " << buffer->layout;
        ICHECK(buffer->layout.value()->CompatibleWithShape(buffer->shape))
            << "TIRxError: Buffer at " << path << " has layout " << buffer->layout
            << " that is not compatible with shape " << buffer->shape;
      }
    };
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

  void VisitStmt_(const SBlockNode* op, ffi::reflection::AccessPath path) override {
    Verifier::VisitStmt_(op, path);
  }

  void VisitStmt_(const ExecScopeStmtNode* op, ffi::reflection::AccessPath path) override {
    scope_stack_.push_back(op->exec_scope);
    Verifier::VisitStmt_(op, path);
    scope_stack_.pop_back();
  }

  std::vector<ExecScope> scope_stack_;
};

class DeviceFuncVerifier : public Verifier<DeviceFuncVerifier> {
 public:
  using Verifier::Verifier;

 private:
  using Verifier::Visit;

  void VisitStmt_(const SBlockNode* op, ffi::reflection::AccessPath path) override {
    Verifier::VisitStmt_(op, path);
  }

  void VisitStmt_(const ExecScopeStmtNode* op, ffi::reflection::AccessPath path) override {
    if (!inside_root_scope_) {
      // At the top level: only one root scope is allowed
      Verify(!root_.has_value()) << "TIRxError: Only one root scope is allowed in device function";
      root_ = op->exec_scope;
      auto kernel_scope = ExecScope::Create("kernel");
      Verify(kernel_scope->Higher(root_.value()))
          << "TIRxError: Root scope of device function at " << path
          << " is higher than kernel scope";
      inside_root_scope_ = true;
      Verifier::VisitStmt_(op, path);
      inside_root_scope_ = false;
    } else {
      // Already inside a root scope: nested scopes are allowed
      Verifier::VisitStmt_(op, path);
    }
  }

  ffi::Optional<ExecScope> root_ = std::nullopt;
  bool inside_root_scope_ = false;
};

bool VerifyTIRxWellFormed(const PrimFunc& func, bool assert_mode, bool device_func) {
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
  if (device_func) {
    if (!DeviceFuncVerifier::Verify(func, assert_mode)) {
      return false;
    }
  }
  return true;
}

bool VerifyTIRxWellFormed(const IRModule& mod, bool assert_mode, bool device_func) {
  for (const auto& [gvar, base_func] : mod->functions) {
    if (auto prim_func = base_func.as<PrimFunc>()) {
      bool res = VerifyTIRxWellFormed(prim_func.value(), assert_mode, device_func);
      if (!res) {
        return false;
      }
    }
  }
  return true;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
<<<<<<<< HEAD:src/tirx/analysis/verify_tirp_well_formed.cc
  refl::GlobalDef().def("tirx.analysis.VerifyTIRpWellFormed",
========
  refl::GlobalDef().def("tirx.analysis.VerifyTIRxWellFormed",
>>>>>>>> bd927f10c7 (rename):src/tirx/analysis/verify_tirx_well_formed.cc
                        [](const ObjectRef& obj, bool assert_mode, bool device_func) {
                          if (auto n = obj.as<PrimFunc>()) {
                            return VerifyTIRxWellFormed(n.value(), assert_mode, device_func);
                          } else if (auto n = obj.as<IRModule>()) {
                            return VerifyTIRxWellFormed(n.value(), assert_mode, device_func);
                          } else {
                            LOG(FATAL) << "Expects PrimFunc or IRModule,  but get "
                                       << obj->GetTypeKey() << " instead.";
                            return false;
                          }
                        });
}
}  // namespace tirxx
}  // namespace tvm
