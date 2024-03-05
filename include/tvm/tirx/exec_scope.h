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

class ScopeIdNode : public VarNode {
 public:
  static constexpr const char* _type_key = "tirx.ScopeId";
  TVM_DECLARE_FINAL_OBJECT_INFO(ScopeIdNode, VarNode);
};

class ScopeId : public Var {
 public:
  TVM_DLL explicit ScopeId(String name = "");

  TVM_DEFINE_OBJECT_REF_METHODS(ScopeId, Var, ScopeIdNode);
};

class ScopeIdDefNode : public Object {
 public:
  /*! \brief The ScopeId defined */
  Array<ScopeId> def_ids;
  /*! \brief The extents of the ScopeId */
  Array<PrimExpr> extents;
  /*! \brief Parent ExecScope name */
  String parent;
  /*! \brief Current ExecScope name */
  String cur;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("def_ids", &def_ids);
    v->Visit("extents", &extents);
    v->Visit("parent", &parent);
    v->Visit("cur", &cur);
  }

  bool SEqualReduce(const ScopeIdDefNode* other, SEqualReducer equal) const {
    return equal(def_ids, other->def_ids) && equal(extents, other->extents) &&
           equal(parent, other->parent) && equal(cur, other->cur);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(def_ids);
    hash_reduce(extents);
    hash_reduce(parent);
    hash_reduce(cur);
  }
  static constexpr const char* _type_key = "tirx.ScopeIdDef";
  TVM_DECLARE_FINAL_OBJECT_INFO(ScopeIdDefNode, Object);
};

class ScopeIdDef : public ObjectRef {
 public:
  TVM_DLL ScopeIdDef(Array<ScopeId> def_ids, Array<PrimExpr> extents, String parent, String cur);

  TVM_DEFINE_OBJECT_REF_METHODS(ScopeIdDef, ObjectRef, ScopeIdDefNode);
};

class ExecScopeNode : public Object {
 public:
  /*! \brief scope name, used when printing */
  String name;

  void VisitAttrs(AttrVisitor* v) { v->Visit("name", &name); }

  bool SEqualReduce(const ExecScopeNode* other, SEqualReducer equal) const {
    return equal(name, other->name);
  }

  void SHashReduce(SHashReducer hash_reduce) const { hash_reduce(name); }

  static constexpr const char* _type_key = "tirx.ExecScope";
  TVM_DECLARE_BASE_OBJECT_INFO(ExecScopeNode, Object);
};

class ExecScope : public ObjectRef {
 public:
  TVM_DLL explicit ExecScope(String name = "");

  TVM_DEFINE_OBJECT_REF_METHODS(ExecScope, ObjectRef, ExecScopeNode);
};

// Two special ExecSope: World and Kernel
class WorldScopeNode : public ExecScopeNode {
 public:
  ScopeIdDef scope_id_def;
  static constexpr const char* _type_key = "tirx.WorldScope";
  TVM_DECLARE_FINAL_OBJECT_INFO(WorldScopeNode, ExecScopeNode);
};

class WorldScope : public ExecScope {
 public:
  TVM_DLL explicit WorldScope(ScopeIdDef scope_id_def);

  TVM_DEFINE_OBJECT_REF_METHODS(WorldScope, ExecScope, WorldScopeNode);
};

class KernelScopeNode : public ExecScopeNode {
 public:
  Array<ScopeIdDef> scope_id_def;
  static constexpr const char* _type_key = "tirx.KernelScope";
  TVM_DECLARE_FINAL_OBJECT_INFO(KernelScopeNode, ExecScopeNode);
};

class KernelScope : public ExecScope {
 public:
  TVM_DLL explicit KernelScope(Array<ScopeIdDef> scope_id_def);

  TVM_DEFINE_OBJECT_REF_METHODS(KernelScope, ExecScope, KernelScopeNode);
};

class ExecScopeSliceNode : public ExecScopeNode {
 public:
  /*! \brief defining threading vars */
  Array<ScopeId> def_ids;
  /*! \brief subrange of each threading vars */
  Array<Range> ranges;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("def_vars", &def_ids);
    v->Visit("ranges", &ranges);
    ExecScopeNode::VisitAttrs(v);
  }

  bool SEqualReduce(const ExecScopeSliceNode* other, SEqualReducer equal) const {
    return equal(def_ids, other->def_ids) && equal(ranges, other->ranges) &&
           ExecScopeNode::SEqualReduce(other, equal);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(def_ids);
    hash_reduce(ranges);
    ExecScopeNode::SHashReduce(hash_reduce);
  }

  static constexpr const char* _type_key = "tirx.ExecScopeSlice";
  TVM_DECLARE_FINAL_OBJECT_INFO(ExecScopeSliceNode, ExecScopeNode);
};

class ExecScopeSlice : public ExecScope {
 public:
  TVM_DLL explicit ExecScopeSlice(Array<ScopeId> vars, Array<Range> ranges, String name = "");

  TVM_DEFINE_OBJECT_REF_METHODS(ExecScopeSlice, ExecScope, ExecScopeSliceNode);
};

}  // namespace tirxx
}  // namespace tvm

#endif  // TVM_TIRX_EXEC_SCOPE_H_
