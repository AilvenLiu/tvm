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

/******** Definition of ScopeId ********/
class ScopePairNode : public Object {
 public:
  /*! \brief The parent scope */
  String parent;
  /*! \brief The current scope */
  String cur;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("parent", &parent);
    v->Visit("cur", &cur);
  }

  bool SEqualReduce(const ScopePairNode* other, SEqualReducer equal) const {
    return equal(parent, other->parent) && equal(cur, other->cur);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(parent);
    hash_reduce(cur);
  }

  static constexpr const char* _type_key = "tirx.ScopePair";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  TVM_DECLARE_FINAL_OBJECT_INFO(ScopePairNode, Object);
};

class ScopePair : public ObjectRef {
 public:
  TVM_DLL explicit ScopePair(String parent, String cur);

  struct ScopePairEqual {
    bool operator()(const ScopePair& a, const ScopePair& b) const {
      return tvm::runtime::ObjectEqual()(a->parent, b->parent) &&
             tvm::runtime::ObjectEqual()(a->cur, b->cur);
    }
  };

  struct ScopePairHash {
    size_t operator()(const ScopePair& a) const {
      return tvm::runtime::ObjectHash()(a->parent) ^ tvm::runtime::ObjectHash()(a->cur);
    }
  };

  TVM_DEFINE_OBJECT_REF_METHODS(ScopePair, ObjectRef, ScopePairNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(ScopePairNode);
};

class ScopeIdDefNode : public Object {
 public:
  /*! \brief The ScopeId defined */
  Array<Var> def_ids;
  /*! \brief The extents of the ScopeId */
  Array<PrimExpr> extents;
  /*! \brief The scope of the scope id */
  ScopePair scope;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("def_ids", &def_ids);
    v->Visit("extents", &extents);
    v->Visit("scope", &scope);
  }

  bool SEqualReduce(const ScopeIdDefNode* other, SEqualReducer equal) const {
    return equal.DefEqual(def_ids, other->def_ids) && equal(extents, other->extents) &&
           equal(scope, other->scope);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(def_ids);
    hash_reduce(extents);
    hash_reduce(scope);
  }

  static constexpr const char* _type_key = "tirx.ScopeIdDef";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  TVM_DECLARE_FINAL_OBJECT_INFO(ScopeIdDefNode, Object);
};

class ScopeIdDef : public ObjectRef {
 public:
  TVM_DLL explicit ScopeIdDef(Array<Var> def_ids, Array<PrimExpr> extents, ScopePair scope);

  PrimExpr fused_extent() const;

  TVM_DEFINE_OBJECT_REF_METHODS(ScopeIdDef, ObjectRef, ScopeIdDefNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(ScopeIdDefNode);
};

/*! \brief Compose two scope id definitions */
Optional<ScopeIdDef> Compose(const ScopeIdDef& lhs, const ScopeIdDef& rhs);

/*! \brief Compliment two scope id definitions */
Optional<ScopeIdDef> Compliment(const ScopeIdDef& lhs, const ScopeIdDef& rhs);

class ScopeIdDefVerifier {
 public:
  using ScopeIdSet = std::unordered_map<ScopePair, ScopeIdDef, ScopePair::ScopePairHash,
                                        ScopePair::ScopePairEqual>;

  /*! \brief Verify the scope id definitions are well formed */
  bool Verify(const Array<ScopeIdDef>& defs);

  /*! \brief The resovled scope id set */
  ScopeIdSet id_set;
};

class ScopeIdResolveTable {
 public:
  using ScopeIdSet = ScopeIdDefVerifier::ScopeIdSet;
  using LaunchParams = std::unordered_map<String, IterVar, ObjectHash, ObjectEqual>;

  typedef Array<PrimExpr> (*ResolveFunc)(const Optional<Array<PrimExpr>>& extents, int out_dim,
                                         const LaunchParams& params);

  static ScopeIdResolveTable* Global() {
    static ScopeIdResolveTable inst;
    return &inst;
  }

  class Registry {
   public:
    Registry& set(ResolveFunc func) {
      this->func_ = func;
      return *this;
    }

   private:
    friend class ScopeIdResolveTable;
    ResolveFunc func_;
  };

  /*! \brief Register a ScopeIdDef resolve rule */
  static Registry& Register(String parent, String cur, String target_kind);

  /*! \brief Resolve a ScopeIdDef */
  static Array<PrimExpr> Resolve(const ScopePair& scope, const Optional<Array<PrimExpr>>& extents,
                                 int out_dim, String target_kind, const LaunchParams& params);

 private:
  static std::string GetKey(const ScopePair& scope, const String& target_kind) {
    return scope->parent.operator std::string() + "__##__" + scope->cur.operator std::string() +
           "__##__" + target_kind.operator std::string();
  }

  /*! \brief The registered scope id definitions */
  std::unordered_map<std::string, Registry> resolve_map_;
};

/******** Definition of Execution Scope ********/
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

  /*! \brief create a exec scope from scope name */
  static ExecScope Create(String name);

  /*! \brief check if a scope name is valid */
  static bool Valid(const String& name);

  /*! \brief scope's are the same */
  bool Is(const ExecScope& other) const;

  /*! \brief scope is identified by name */
  bool Is(const String& name) const;

  /*! \brief scope is higher than other sope */
  bool Higher(const ExecScope& other) const;

  /*! \brief scope is higher than other sope */
  bool Higher(const String& other) const;

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
  /*! \brief slices of the execution scope */
  Optional<Array<Range>> slices;
  /*! \brief select condition */
  Optional<PrimExpr> select_cond;
  /*! \brief extents of the execution scope */
  Optional<Array<PrimExpr>> extents;
  /*! \brief parent scope name */
  String parent;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("slices", &slices);
    v->Visit("select_cond", &select_cond);
    v->Visit("extents", &extents);
    v->Visit("parent", &parent);
    ExecScopeNode::VisitAttrs(v);
  }

  bool SEqualReduce(const ExecScopeSliceNode* other, SEqualReducer equal) const {
    return equal(slices, other->slices) && equal(select_cond, other->select_cond) &&
           equal(extents, other->extents) && equal(parent, other->parent) &&
           ExecScopeNode::SEqualReduce(other, equal);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(slices);
    hash_reduce(select_cond);
    hash_reduce(extents);
    hash_reduce(parent);
    ExecScopeNode::SHashReduce(hash_reduce);
  }

  static constexpr const char* _type_key = "tirx.ExecScopeSlice";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  TVM_DECLARE_FINAL_OBJECT_INFO(ExecScopeSliceNode, ExecScopeNode);
};

class ExecScopeSlice : public ExecScope {
 public:
  TVM_DLL explicit ExecScopeSlice(Optional<Array<Range>> slices, Optional<PrimExpr> select_cond,
                                  Optional<Array<PrimExpr>> extents, String parent, String cur);

  TVM_DEFINE_OBJECT_REF_METHODS(ExecScopeSlice, ExecScope, ExecScopeSliceNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(ExecScopeSliceNode);
};

/******** Helper functions ********/
/*! \brief ExecScope order from highest to lowest */
static const std::unordered_map<String, int> ScopeOrder = {
    {"world", 0},     {"kernel", 1}, {"cluster", 2}, {"cta", 3},
    {"warpgroup", 4}, {"warp", 5},   {"thread", 6}};

/*! \brief Map from storage scope to its belonging logical scope */
static const std::unordered_map<String, String> StorageToLogical = {
    {"local", "thread"},  {"shared", "cta"},      {"shared.dyn", "cta"},
    {"global", "kernel"}, {"trn.sbuf", "kernel"}, {"trn.psum", "kernel"}};

/*! \brief Whether the storage scope belongs to the logical scope*/
bool IsStorageBuffer(const String& storage, const String& logical);

/*! \brief Return the belonging logical scope of some storage scope */
String StorageToLogicalScope(const String& storage);

}  // namespace tirxx
}  // namespace tvm

#endif  // TVM_TIRX_EXEC_SCOPE_H_
