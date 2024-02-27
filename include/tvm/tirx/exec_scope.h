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
 * \file tvm/tirx/block_scope.h
 * \brief Definition of execution scope
 */

#ifndef TVM_TIRX_EXEC_SCOPE_H_
#define TVM_TIRX_EXEC_SCOPE_H_

#include <tvm/ir/module.h>
#include <tvm/tirx/var.h>

namespace tvm {
namespace tirx {

class ExecScopeNode : public Object {
 public:
  /*! \brief Now support at most 3 dims */
  Array<PrimExpr> dims;
  /*! \brief scope name, used when printing */
  String name;

  size_t size() const { return dims.size(); }

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("dims", &dims);
    v->Visit("name", &name);
  }

  bool SEqualReduce(const ExecScopeNode* other, SEqualReducer equal) const {
    return equal(dims, other->dims) && equal(name, other->name);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(dims);
    hash_reduce(name);
  }

  static constexpr const char* _type_key = "tirx.ExecScope";
  TVM_DECLARE_BASE_OBJECT_INFO(ExecScopeNode, Object);
};

class ExecScope : public ObjectRef {
 public:
  TVM_DLL explicit ExecScope(Array<PrimExpr> dims, String name = "");

  TVM_DEFINE_OBJECT_REF_METHODS(ExecScope, ObjectRef, ExecScopeNode);
};

class ThreadingVarNode : public VarNode {
 public:
  /*! \brief The execution scope defining this var */
  ExecScope scope;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("scope", &scope);
    VarNode::VisitAttrs(v);
  }

  bool SEqualReduce(const ThreadingVarNode* other, SEqualReducer equal) const {
    if (!equal(scope, other->scope)) return false;
    return VarNode::SEqualReduce(other, equal);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(scope);
    VarNode::SHashReduce(hash_reduce);
  }

  static constexpr const char* _type_key = "tirx.ThreadingVar";
  TVM_DECLARE_FINAL_OBJECT_INFO(ThreadingVarNode, VarNode);
};

class ThreadingVar : public Var {
 public:
  TVM_DLL explicit ThreadingVar(ExecScope scope, String name = "");

  TVM_DEFINE_OBJECT_REF_METHODS(ThreadingVar, Var, ThreadingVarNode);
};

class SubExecScopeNode : public ExecScopeNode {
 public:
  /*! \brief defining threading vars */
  Array<ThreadingVar> def_vars;
  /*! \brief subrange of each threading vars */
  Optional<Array<Range>> ranges;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("def_vars", &def_vars);
    v->Visit("ranges", &ranges);
    ExecScopeNode::VisitAttrs(v);
  }

  bool SEqualReduce(const SubExecScopeNode* other, SEqualReducer equal) const {
    return equal(def_vars, other->def_vars) && equal(ranges, other->ranges) &&
           ExecScopeNode::SEqualReduce(other, equal);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(def_vars);
    hash_reduce(ranges);
    ExecScopeNode::SHashReduce(hash_reduce);
  }

  static constexpr const char* _type_key = "tirx.SubExecScope";
  TVM_DECLARE_FINAL_OBJECT_INFO(SubExecScopeNode, ExecScopeNode);
};

class SubExecScope : public ExecScope {
 public:
  TVM_DLL explicit SubExecScope(Array<ThreadingVar> vars, Optional<Array<Range>> ranges,
                                String name = "");

  TVM_DEFINE_OBJECT_REF_METHODS(SubExecScope, ExecScope, SubExecScopeNode);
};

}  // namespace tirx
}  // namespace tvm

#endif  // TVM_TIRX_EXEC_SCOPE_H_
