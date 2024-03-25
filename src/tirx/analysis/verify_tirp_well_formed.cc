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

#include <tvm/runtime/registry.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>

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
    Verify(op->exec_scope != nullptr) << "Block at " << path << " has no exec_scope";
    auto scope = op->exec_scope.value();
    if (scope_stack_.empty()) {
      Verify(scope.Is("world") || scope.Is("global"))
          << "Block at " << path << " has invalid exec_scope " << scope->name;
    } else {
      if (Higher(scope_stack_.back(), scope)) {
        cur_roof_ = scope_stack_.back();
      } else if (scope_stack_.back().Is(scope->name)) {
        // do nothing
      } else {
        Verify(!Higher(scope, cur_roof_)) << "Block at " << path << " has invalid exec_scope "
                                          << scope->name << " under " << cur_roof_->name;
      }
    }
    scope_stack_.push_back(scope);
  }

  ExecScope cur_roof_{"None"};
  std::vector<ExecScope> scope_stack_;
};

bool VerifyTIRpWellFormed(const PrimFunc& func, bool assert_mode) {
  if (!ExecScopeVerifier::Verify(func, assert_mode)) {
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

}  // namespace tirx
}  // namespace tvm
