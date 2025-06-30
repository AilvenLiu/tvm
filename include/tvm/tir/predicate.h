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
 *//*!
 * \file tvm/tir/predicate.h
 * \brief Definition of predicate
 */

#ifndef TVM_TIR_PREDICATE_H_
#define TVM_TIR_PREDICATE_H_

#include <tvm/arith/analyzer.h>
#include <tvm/ir/module.h>
#include <tvm/runtime/object.h>
#include <tvm/tir/exec_scope.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/var.h>
namespace tvm {
namespace tir {

class PredicateNode : public Object {
 public:
  /*! \brief The variables in the predicate */
  Array<Var> vars;
  /*! \brief The predicate */
  PrimExpr pred;

  /*! \brief Replace the variables in the predicate with the given indices */
  PrimExpr Apply(const Array<PrimExpr>& indices) const;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<PredicateNode>()
        .def_ro("vars", &PredicateNode::vars)
        .def_ro("pred", &PredicateNode::pred);
  }

  bool SEqualReduce(const PredicateNode* other, SEqualReducer equal) const {
    return equal.DefEqual(vars, other->vars) && equal(pred, other->pred);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce.DefHash(vars);
    hash_reduce(pred);
  }

  static constexpr const char* _type_key = "tir.Predicate";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  static constexpr bool _type_has_method_visit_attrs = false;
  TVM_DECLARE_FINAL_OBJECT_INFO(PredicateNode, Object);
};

class Predicate : public ObjectRef {
 public:
  explicit Predicate(Array<Var> vars, PrimExpr pred);

  TVM_DEFINE_OBJECT_REF_METHODS(Predicate, ObjectRef, PredicateNode);
};

}  // namespace tir
}  // namespace tvm

#endif  // TVM_TIR_PREDICATE_H_
