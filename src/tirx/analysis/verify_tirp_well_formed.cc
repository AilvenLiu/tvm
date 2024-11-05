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

  void Visit(const Stmt& obj, ObjectPath path) override {
    if (!scope_stack_.empty() && !scope_stack_.back().Is("thread")) {
      Verify(obj->IsInstance<BlockNode>() || obj->IsInstance<ForNode>() ||
             obj->IsInstance<BlockRealizeNode>() || obj->IsInstance<SeqStmtNode>() ||
             obj->IsInstance<tirp::OpCallNode>() || obj->IsInstance<AttrStmtNode>() ||
             obj->IsInstance<DeclBufferNode>())
          << "TIRpError: Stmt at " << path << " is not under a thread scope and has type "
          << obj->GetTypeKey() << "\n"
          << obj;
    }
    Verifier::Visit(obj, path);
  }

  void VisitStmt_(const BlockNode* op, ObjectPath path) override {
    Verify(op->exec_scope != nullptr) << "TIRpError: Block at " << path << " has no exec_scope";
    auto scope = op->exec_scope.value();
    Verify(ExecScope::Valid(scope->name))
        << "TIRpError: Block at " << path << " has unknown exec_scope " << scope->name;
    if (scope_stack_.empty()) {
      Verify(scope.Is("world") || scope.Is("kernel"))
          << "TIRpError: Block at " << path << " has invalid exec_scope " << scope->name
          << " as root";
    } else {
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
    scope_stack_.push_back(scope);
    Verifier::VisitStmt_(op, path);
    scope_stack_.pop_back();
  }

  void VisitStmt_(const tirp::OpCallNode* op, ObjectPath path) override {
    static const tvm::OpAttrMap<Bool>& tirp_op_map_ = Op::GetAttrMap<Bool>("TIsTIRpOp");
    Verify(tirp_op_map_.count(op->op))
        << "TIRpError: OpCall at " << path << " has unknown TIR+ op " << op->op;
  }

  Optional<ExecScope> cur_roof_ = NullOpt;
  std::vector<ExecScope> scope_stack_;
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
