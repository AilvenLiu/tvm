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
  void VisitAttrs(AttrVisitor* v) { VarNode::VisitAttrs(v); }

  bool SEqualReduce(const ScopeIdNode* other, SEqualReducer equal) const {
    return VarNode::SEqualReduce(other, equal);
  }

  void SHashReduce(SHashReducer hash_reduce) const { VarNode::SHashReduce(hash_reduce); }

  static constexpr const char* _type_key = "tirx.ScopeId";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  TVM_DECLARE_FINAL_OBJECT_INFO(ScopeIdNode, VarNode);
};

class ScopeId : public Var {
 public:
  TVM_DLL explicit ScopeId(String name = "");

  TVM_DEFINE_OBJECT_REF_METHODS(ScopeId, Var, ScopeIdNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(ScopeIdNode);
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
    return equal.DefEqual(def_ids, other->def_ids) && equal(extents, other->extents) &&
           equal(parent, other->parent) && equal(cur, other->cur);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(def_ids);
    hash_reduce(extents);
    hash_reduce(parent);
    hash_reduce(cur);
  }

  static constexpr const char* _type_key = "tirx.ScopeIdDef";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  TVM_DECLARE_FINAL_OBJECT_INFO(ScopeIdDefNode, Object);
};

class ScopeIdDef : public ObjectRef {
 public:
  TVM_DLL explicit ScopeIdDef(Array<ScopeId> def_ids, Array<PrimExpr> extents, String parent,
                              String cur);

  explicit ScopeIdDef(String parent, String cur);

  PrimExpr fused_extent() const;

  // Hash and Equal for only scope comparison
  struct ScopeHash {
    size_t operator()(const ScopeIdDef& lhs) const {
      return std::hash<String>()(lhs->parent) ^ std::hash<String>()(lhs->cur);
    }
  };
  struct ScopeEqual {
    bool operator()(const ScopeIdDef& lhs, const ScopeIdDef& rhs) const {
      return lhs->parent == rhs->parent && lhs->cur == rhs->cur;
    }
  };

  TVM_DEFINE_OBJECT_REF_METHODS(ScopeIdDef, ObjectRef, ScopeIdDefNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(ScopeIdDefNode);
};

Optional<ScopeIdDef> Compose(const ScopeIdDef& lhs, const ScopeIdDef& rhs);

Optional<ScopeIdDef> Compliment(const ScopeIdDef& lhs, const ScopeIdDef& rhs);

inline std::ostream& operator<<(std::ostream& out, const ScopeIdDef& input) {
  out << input->parent << " --> " << input->cur;
  return out;
}

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
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  TVM_DECLARE_BASE_OBJECT_INFO(ExecScopeNode, Object);
};

class ExecScope : public ObjectRef {
 public:
  TVM_DLL explicit ExecScope(String name);

  static ExecScope Create(String name);

  /*! \brief scope is identified by name */
  bool Is(const String& name) const;

  TVM_DEFINE_OBJECT_REF_METHODS(ExecScope, ObjectRef, ExecScopeNode);
};

// Two special ExecSope: World and Kernel
class WorldScopeNode : public ExecScopeNode {
 public:
  ScopeIdDef scope_id_def;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("scope_id_def", &scope_id_def);
    ExecScopeNode::VisitAttrs(v);
  }

  bool SEqualReduce(const WorldScopeNode* other, SEqualReducer equal) const {
    return equal(scope_id_def, other->scope_id_def) && ExecScopeNode::SEqualReduce(other, equal);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(scope_id_def);
    ExecScopeNode::SHashReduce(hash_reduce);
  }

  static constexpr const char* _type_key = "tirx.WorldScope";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  TVM_DECLARE_FINAL_OBJECT_INFO(WorldScopeNode, ExecScopeNode);
};

class WorldScope : public ExecScope {
 public:
  TVM_DLL explicit WorldScope(ScopeIdDef scope_id_def);

  TVM_DEFINE_OBJECT_REF_METHODS(WorldScope, ExecScope, WorldScopeNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(WorldScopeNode);
};

class KernelScopeNode : public ExecScopeNode {
 public:
  Array<ScopeIdDef> scope_id_def;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("scope_id_def", &scope_id_def);
    ExecScopeNode::VisitAttrs(v);
  }

  bool SEqualReduce(const KernelScopeNode* other, SEqualReducer equal) const {
    return equal(scope_id_def, other->scope_id_def) && ExecScopeNode::SEqualReduce(other, equal);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(scope_id_def);
    ExecScopeNode::SHashReduce(hash_reduce);
  }

  static constexpr const char* _type_key = "tirx.KernelScope";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  TVM_DECLARE_FINAL_OBJECT_INFO(KernelScopeNode, ExecScopeNode);
};

class KernelScope : public ExecScope {
 public:
  TVM_DLL explicit KernelScope(Array<ScopeIdDef> scope_id_def);

  TVM_DEFINE_OBJECT_REF_METHODS(KernelScope, ExecScope, KernelScopeNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(KernelScopeNode);
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
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  TVM_DECLARE_FINAL_OBJECT_INFO(ExecScopeSliceNode, ExecScopeNode);
};

class ExecScopeSlice : public ExecScope {
 public:
  TVM_DLL explicit ExecScopeSlice(Array<ScopeId> vars, Array<Range> ranges, String name);

  TVM_DEFINE_OBJECT_REF_METHODS(ExecScopeSlice, ExecScope, ExecScopeSliceNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(ExecScopeSliceNode);
};

static const std::unordered_map<String, int> ScopeOrder = {
    {"world", 0},      {"kernel", 1}, {"cluster", 2}, {"cta", 3},
    {"warp_group", 4}, {"warp", 5},   {"thread", 6}};

static const std::unordered_map<String, String> StorageToLogical = {
    {"local", "thread"}, {"shared", "cta"}, {"global", "kernel"}};

bool Higher(const ExecScope& lhs, const ExecScope& rhs);

bool Higher(const String& lhs, const String& rhs);

bool Equal(const ExecScope& lhs, const ExecScope& rhs);

bool Equal(const String& lhs, const String& rhs);

bool ValideScope(const ExecScope& scope);

bool ValideScope(const String& scope);

bool IsStorageBuffer(const String& storage, const String& logical);

String StorageToLogicalScope(const String& storage);

}  // namespace tirx
}  // namespace tvm

#endif  // TVM_TIRX_EXEC_SCOPE_H_
