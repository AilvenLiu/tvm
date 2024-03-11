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
 * \file tvm/tirx/layout.h
 * \brief Definition of layout
 */

#ifndef TVM_TIRX_LAYOUT_H_
#define TVM_TIRX_LAYOUT_H_

#include <tvm/ir/module.h>
#include <tvm/tirx/exec_scope.h>
#include <tvm/tirx/var.h>

namespace tvm {
namespace tirx {

// Base class for layout
class LayoutNode : public Object {
 public:
  static constexpr const char* _type_key = "tirx.Layout";
  TVM_DECLARE_BASE_OBJECT_INFO(LayoutNode, Object);
};

class Layout : public ObjectRef {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(Layout, ObjectRef, LayoutNode);
};

// IterTreeSplit
class IterTreeSplitNode : public Object {
 public:
  /*! \brief parent node */
  Var parent;
  /*! \brief split children */
  Array<Var> children;
  /*! \brief The extent of the split */
  Array<PrimExpr> extents;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("parent", &parent);
    v->Visit("children", &children);
    v->Visit("extents", &extents);
  }

  bool SEqualReduce(const IterTreeSplitNode* other, SEqualReducer equal) const {
    return equal(parent, other->parent) && equal(children, other->children) &&
           equal(extents, other->extents);
  }

  void SHashReduce(SHashReducer hash_reducer) const {
    hash_reducer(parent);
    hash_reducer(children);
    hash_reducer(extents);
  }

  static constexpr const char* _type_key = "tirx.IterTreeSplit";
  TVM_DECLARE_FINAL_OBJECT_INFO(IterTreeSplitNode, Object);
};

class IterTreeSplit : public ObjectRef {
 public:
  TVM_DLL explicit IterTreeSplit(Var parent, Array<Var> children, Array<PrimExpr> extents);

  TVM_DEFINE_OBJECT_REF_METHODS(IterTreeSplit, ObjectRef, IterTreeSplitNode);
};

// IterTree
class IterTreeNode : public Object {
 public:
  /*! \brief root var */
  Var root;
  /*! \brief The splits in the tree */
  Array<IterTreeSplit> splits;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("root", &root);
    v->Visit("splits", &splits);
  }

  bool SEqualReduce(const IterTreeNode* other, SEqualReducer equal) const {
    return equal(root, other->root) && equal(splits, other->splits);
  }

  void SHashReduce(SHashReducer hash_reducer) const {
    hash_reducer(root);
    hash_reducer(splits);
  }

  static constexpr const char* _type_key = "tirx.IterTreeNode";
  TVM_DECLARE_BASE_OBJECT_INFO(IterTreeNode, Object);
};

class IterTree : public ObjectRef {
 public:
  TVM_DLL explicit IterTree(Var root, Array<IterTreeSplit> splits);

  TVM_DEFINE_OBJECT_REF_METHODS(IterTree, ObjectRef, IterTreeNode);
};

// CoordIterTree
class CoordIterTreeNode : public IterTreeNode {
 public:
  /*! \brief The coefficients of each leaf node */
  Array<PrimExpr> coeff;

  void VisitAttrs(AttrVisitor* v) {
    IterTreeNode::VisitAttrs(v);
    v->Visit("coeff", &coeff);
  }

  bool SEqualReduce(const CoordIterTreeNode* other, SEqualReducer equal) const {
    return IterTreeNode::SEqualReduce(other, equal) && equal(coeff, other->coeff);
  }

  void SHashReduce(SHashReducer hash_reducer) const {
    IterTreeNode::SHashReduce(hash_reducer);
    hash_reducer(coeff);
  }

  static constexpr const char* _type_key = "tirx.CoordIterTree";
  TVM_DECLARE_FINAL_OBJECT_INFO(CoordIterTreeNode, IterTreeNode);
};

class CoordIterTree : public IterTree {
 public:
  TVM_DLL explicit CoordIterTree(Var root, Array<IterTreeSplit> splits, Array<PrimExpr> coeff);

  TVM_DEFINE_OBJECT_REF_METHODS(CoordIterTree, IterTree, CoordIterTreeNode);
};

// ScopeIdAttr
enum ScopeIdType : int {
  kSplit = 0,
  kReplicate = 1,
  kExclusive = 2,
};

class ScopeIdAttrNode : public Object {
 public:
  /*! \brief type of ScopeID, can be split (S), replicate (R), exclusive (E) */
  ScopeIdType type;
  /*! \brief If type is split, the bound leaf in CoordIterTree */
  Optional<Var> bound;
  /*! \brief If type is exclusive, the id that owns the data */
  Optional<PrimExpr> owner;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("type", &type);
    v->Visit("bound", &bound);
    v->Visit("owner", &owner);
  }

  bool SEqualReduce(const ScopeIdAttrNode* other, SEqualReducer equal) const {
    return equal(type, other->type) && equal(bound, other->bound) && equal(owner, other->owner);
  }

  void SHashReduce(SHashReducer hash_reducer) const {
    hash_reducer(type);
    hash_reducer(bound);
    hash_reducer(owner);
  }

  static constexpr const char* _type_key = "tirx.ScopeIdAttr";
  TVM_DECLARE_FINAL_OBJECT_INFO(ScopeIdAttrNode, Object);
};

class ScopeIdAttr : public ObjectRef {
 public:
  TVM_DLL explicit ScopeIdAttr(ScopeIdType type, Optional<Var> bound = NullOpt,
                               Optional<PrimExpr> owner = NullOpt);

  TVM_DEFINE_OBJECT_REF_METHODS(ScopeIdAttr, ObjectRef, ScopeIdAttrNode);
};

// ScopeIdIterTree
class ScopeIdIterTreeNode : public IterTreeNode {
 public:
  /*! \brief The attributes of each leaf node */
  Array<ScopeIdAttr> attrs;

  void VisitAttrs(AttrVisitor* v) {
    IterTreeNode::VisitAttrs(v);
    v->Visit("attrs", &attrs);
  }

  bool SEqualReduce(const ScopeIdIterTreeNode* other, SEqualReducer equal) const {
    return IterTreeNode::SEqualReduce(other, equal) && equal(attrs, other->attrs);
  }

  void SHashReduce(SHashReducer hash_reducer) const {
    IterTreeNode::SHashReduce(hash_reducer);
    hash_reducer(attrs);
  }

  static constexpr const char* _type_key = "tirx.ScopeIdIterTree";
  TVM_DECLARE_FINAL_OBJECT_INFO(ScopeIdIterTreeNode, IterTreeNode);
};

class ScopeIdIterTree : public IterTree {
 public:
  TVM_DLL explicit ScopeIdIterTree(Var root, Array<IterTreeSplit> splits, Array<ScopeIdAttr> attrs);

  TVM_DEFINE_OBJECT_REF_METHODS(ScopeIdIterTree, IterTree, ScopeIdIterTreeNode);
};

// TileLayout
class TileLayoutNode : public LayoutNode {
 public:
  /*! \brief coordinate iter forest  */
  Array<CoordIterTree> coord_iter_trees;
  /*! \brief scope id iter forest */
  Array<ScopeIdIterTree> scope_id_iter_trees;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("coord_iter_trees", &coord_iter_trees);
    v->Visit("scope_id_iter_trees", &scope_id_iter_trees);
  }

  bool SEqualReduce(const TileLayoutNode* other, SEqualReducer equal) const {
    return equal(coord_iter_trees, other->coord_iter_trees) &&
           equal(scope_id_iter_trees, other->scope_id_iter_trees);
  }

  void SHashReduce(SHashReducer hash_reducer) const {
    hash_reducer(coord_iter_trees);
    hash_reducer(scope_id_iter_trees);
  }

  static constexpr const char* _type_key = "tirx.TileLayout";
  TVM_DECLARE_FINAL_OBJECT_INFO(TileLayoutNode, LayoutNode);
};

class TileLayout : public Layout {
 public:
  TVM_DLL explicit TileLayout(Array<CoordIterTree> coord_iter_trees,
                              Array<ScopeIdIterTree> scope_id_iter_trees);

  TVM_DEFINE_OBJECT_REF_METHODS(TileLayout, Layout, TileLayoutNode);
};

}  // namespace tirx
}  // namespace tvm

#endif  // TVM_TIRX_LAYOUT_H_
