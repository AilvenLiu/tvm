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
 * \file lower_tirp_dedup_tensormap.cc
 * \brief Deduplicate identical cuTensorMap objects created by TIRp schedules.
 */

#include <tvm/arith/analyzer.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <unordered_map>

namespace tvm {
namespace tirx {

namespace {

// Helper to check if a call is to tvm.tir builtin op
inline bool IsBuiltin(const CallNode* call, const Op& op) { return call && call->op.same_as(op); }

// Is a stack allocation for a tensormap handle?
inline bool IsTensorMapAlloca(const LetStmtNode* let) {
  if (const auto* call = let->value.as<CallNode>()) {
    if (IsBuiltin(call, builtin::tvm_stack_alloca())) {
      if (call->args.size() == 2) {
        if (const auto* type_str = call->args[0].as<StringImmNode>()) {
          return type_str->value == "tensormap";
        }
      }
    }
  }
  return false;
}

// Is an Evaluate of tvm_call_packed("runtime.cuTensorMapEncodeTiled", ...)?
inline const CallNode* AsCuTensorMapEncode(const EvaluateNode* eval) {
  const CallNode* call = eval->value.as<CallNode>();
  if (!call || !call->op.same_as(builtin::tvm_call_packed())) return nullptr;
  if (call->args.empty()) return nullptr;
  if (const auto* s = call->args[0].as<StringImmNode>()) {
    if (s->value == "runtime.cuTensorMapEncodeTiled") return call;
  }
  return nullptr;
}

// Extract the tensormap var and the key (arguments after the tensormap var)
inline std::pair<ffi::Optional<Var>, ffi::Array<PrimExpr>> ExtractEncodeKey(const CallNode* call) {
  ICHECK(call->op.same_as(builtin::tvm_call_packed()));
  // args[0] is function name, args[1] is tensormap handle, rest are parameters
  if (call->args.size() < 2) return {ffi::Optional<Var>(), ffi::Array<PrimExpr>()};
  ffi::Optional<Var> tensormap;
  if (auto v = call->args[1].as<Var>()) {
    tensormap = v.value();
  } else {
    tensormap = ffi::Optional<Var>();
  }
  ffi::Array<PrimExpr> key;
  key.reserve(call->args.size() - 2);
  for (size_t i = 2; i < call->args.size(); ++i) key.push_back(call->args[i]);
  return {tensormap, key};
}

}  // namespace

// First pass: Analyze encode calls and decide canonical tensormap per-parameter set
class CuTensorMapDedupAnalyzer : public StmtExprVisitor {
 public:
  void VisitStmt_(const EvaluateNode* op) final {
    if (const CallNode* call = AsCuTensorMapEncode(op)) {
      auto [maybe_var, key] = ExtractEncodeKey(call);
      if (maybe_var.defined()) {
        const Var& v = maybe_var.value();
        // Find an existing key that is structurally equal
        bool found = false;
        for (const auto& kv : canonical_list_) {
          if (StructuralEqual()(kv.first, key)) {
            const Var& canonical = kv.second;
            if (!canonical.same_as(v)) {
              var_remap_[v] = canonical;
            }
            found = true;
            break;
          }
        }
        if (!found) canonical_list_.emplace_back(std::move(key), v);
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  const std::unordered_map<Var, Var, ObjectPtrHash, ObjectPtrEqual>& var_remap() const {
    return var_remap_;
  }

 private:
  std::vector<std::pair<ffi::Array<PrimExpr>, Var>> canonical_list_;
  std::unordered_map<Var, Var, ObjectPtrHash, ObjectPtrEqual> var_remap_;
};

// Second pass: Rewrite vars to canonical, remove duplicate allocas and duplicate encode calls
class CuTensorMapDedupRewriter : public StmtExprMutator {
 public:
  CuTensorMapDedupRewriter(std::unordered_map<Var, Var, ObjectPtrHash, ObjectPtrEqual> var_remap)
      : var_remap_(std::move(var_remap)) {}

 private:
  using StmtExprMutator::VisitExpr_;
  using StmtExprMutator::VisitStmt_;

  PrimExpr VisitExpr_(const VarNode* op) final {
    Var v = ffi::GetRef<Var>(op);
    auto it = var_remap_.find(v);
    if (it != var_remap_.end()) {
      MarkIfTensorMapVar(it->second);
      return it->second;
    }
    MarkIfTensorMapVar(v);
    return ffi::GetRef<PrimExpr>(op);
  }

  Stmt VisitStmt_(const LetStmtNode* op) final {
    // Rewrite body first (so vars inside can be remapped)
    Stmt body = VisitStmt(op->body);
    if (IsTensorMapAlloca(op)) {
      // If this Let allocates a tensormap that is remapped to a canonical var, drop it
      auto it = var_remap_.find(op->var);
      if (it != var_remap_.end()) {
        return body;
      }
      // Or if the tensormap var is unused after rewriting, drop it
      if (used_tensormap_vars_.count(op->var) == 0) {
        return body;
      }
    }
    // Otherwise, keep the Let (with potentially mutated value/body)
    PrimExpr value = VisitExpr(op->value);
    if (value.same_as(op->value) && body.same_as(op->body)) {
      return ffi::GetRef<Stmt>(op);
    }
    return LetStmt(op->var, value, body);
  }

  Stmt VisitStmt_(const EvaluateNode* op) final {
    // Default mutation
    Evaluate eval = Downcast<Evaluate>(StmtExprMutator::VisitStmt_(op));
    if (const CallNode* call = AsCuTensorMapEncode(eval.get())) {
      // Build key after var remapping
      auto [maybe_var, key] = ExtractEncodeKey(call);
      // Keep only the first occurrence for this key
      for (const auto& k : emitted_keys_) {
        if (StructuralEqual()(k, key)) {
          return Evaluate(0);
        }
      }
      emitted_keys_.push_back(std::move(key));
      return eval;
    }
    return eval;
  }

  // Map of duplicate var -> canonical var
  std::unordered_map<Var, Var, ObjectPtrHash, ObjectPtrEqual> var_remap_;
  // Track which parameter keys have already emitted an encode call
  std::vector<ffi::Array<PrimExpr>> emitted_keys_;

  // Track used tensormap vars post-remap to safely remove dead allocas
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> used_tensormap_vars_;

  void MarkIfTensorMapVar(const Var& v) {
    if (const auto* ptr = v->type_annotation.as<PointerTypeNode>()) {
      if (ptr->element_type.as<TensorMapTypeNode>()) {
        used_tensormap_vars_.insert(v);
      }
    }
  }
};

namespace transform {

Pass LowerTIRpDedupCuTensorMaps() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    // Analyze usage to find duplicates
    CuTensorMapDedupAnalyzer analyzer;
    analyzer(f->body);
    if (analyzer.var_remap().empty()) {
      return f;
    }
    auto* n = f.CopyOnWrite();
    n->body = CuTensorMapDedupRewriter(analyzer.var_remap())(n->body);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tirx.LowerTIRpDedupCuTensorMaps", {});
}

}  // namespace transform
}  // namespace tirx
}  // namespace tvm
