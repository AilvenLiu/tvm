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

#include <tvm/ffi/container/variant.h>
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

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ScopePairNode>()
        .def_ro("parent", &ScopePairNode::parent)
        .def_ro("cur", &ScopePairNode::cur);
  }

  static constexpr const char* _type_key = "tirx.ScopePair";
  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
  TVM_DECLARE_FINAL_OBJECT_INFO(ScopePairNode, Object);
};

class ScopePair : public ObjectRef {
 public:
  TVM_DLL explicit ScopePair(String parent, String cur);

  struct ScopePairEqual {
    bool operator()(const ScopePair& a, const ScopePair& b) const {
      return a->parent == b->parent && a->cur == b->cur;
    }
  };

  struct ScopePairHash {
    size_t operator()(const ScopePair& a) const {
      return std::hash<String>()(a->parent) ^ std::hash<String>()(a->cur);
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

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ScopeIdDefNode>()
        .def_ro("def_ids", &ScopeIdDefNode::def_ids, refl::AttachFieldFlag::SEqHashDef())
        .def_ro("extents", &ScopeIdDefNode::extents)
        .def_ro("scope", &ScopeIdDefNode::scope);
  }

  static constexpr const char* _type_key = "tirx.ScopeIdDef";
  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
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
  using LaunchParams = std::unordered_map<String, IterVar>;

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
class ExecScope;
class ExecScopeNode : public Object {
 public:
  Array<ScopeIdDef> scope_id_def;

  /*! \brief scope name, used when printing */
  String name;

  /*! \brief scope's are the same */
  virtual bool Is(const ExecScope& other) const;

  /*! \brief scope is identified by name */
  bool Is(const String& name) const;

  /*! \brief scope is higher than other sope */
  bool Higher(const ExecScope& other) const;

  /*! \brief scope is higher than other sope */
  bool Higher(const String& other) const;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ExecScopeNode>()
        .def_ro("name", &ExecScopeNode::name)
        .def_ro("scope_id_def", &ExecScopeNode::scope_id_def);
  }

  static constexpr const char* _type_key = "tirx.ExecScope";
  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
  TVM_DECLARE_BASE_OBJECT_INFO(ExecScopeNode, Object);
};

class ExecScope : public ObjectRef {
 public:
  TVM_DLL explicit ExecScope(String name, Array<ScopeIdDef> scope_id_def = {});

  /*! \brief create a exec scope from scope name */
  static ExecScope Create(String name);

  /*! \brief check if a scope name is valid */
  static bool Valid(const String& name);

  TVM_DEFINE_OBJECT_REF_METHODS(ExecScope, ObjectRef, ExecScopeNode);
};

class ExecScopeSliceNode : public ExecScopeNode {
 public:
  /*! \brief slices or select condition of the execution scope */
  Variant<Array<Range>, PrimExpr> slices = Array<Range>({});
  /*! \brief extents of the execution scope */
  Optional<Array<PrimExpr>> extents;
  /*! \brief parent scope name */
  String parent;

  bool Is(const ExecScope& other) const final;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ExecScopeSliceNode>()
        .def_ro("slices", &ExecScopeSliceNode::slices)
        .def_ro("extents", &ExecScopeSliceNode::extents)
        .def_ro("parent", &ExecScopeSliceNode::parent)
        .def_ro("name", &ExecScopeSliceNode::name)
        .def_ro("scope_id_def", &ExecScopeSliceNode::scope_id_def);
  }

  static constexpr const char* _type_key = "tirx.ExecScopeSlice";
  TVM_DECLARE_FINAL_OBJECT_INFO(ExecScopeSliceNode, ExecScopeNode);
};

class ExecScopeSlice : public ExecScope {
 public:
  TVM_DLL explicit ExecScopeSlice(Variant<Array<Range>, PrimExpr> slices,
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

}  // namespace tirxxxxxx
}  // namespace tvm

#endif  // TVM_TIRX_EXEC_SCOPE_H_
