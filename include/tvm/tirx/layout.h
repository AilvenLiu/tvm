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
class TLayoutNode : public Object {
 public:
  static constexpr const char* _type_key = "tirx.TLayout";
  TVM_DECLARE_BASE_OBJECT_INFO(TLayoutNode, Object);
};

class TLayout : public ObjectRef {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(TLayout, ObjectRef, TLayoutNode);
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

  static constexpr const char* _type_key = "tirx.IterTree";
  TVM_DECLARE_BASE_OBJECT_INFO(IterTreeNode, Object);
};

class IterTree : public ObjectRef {
 public:
  TVM_DLL explicit IterTree(Var root, Array<IterTreeSplit> splits);

  TVM_DEFINE_OBJECT_REF_METHODS(IterTree, ObjectRef, IterTreeNode);
};

// CoordIterTree
class DataIterTreeNode : public IterTreeNode {
 public:
  /*! \brief The coefficients of each leaf node */
  Array<PrimExpr> coeff;

  void VisitAttrs(AttrVisitor* v) {
    IterTreeNode::VisitAttrs(v);
    v->Visit("coeff", &coeff);
  }

  bool SEqualReduce(const DataIterTreeNode* other, SEqualReducer equal) const {
    return IterTreeNode::SEqualReduce(other, equal) && equal(coeff, other->coeff);
  }

  void SHashReduce(SHashReducer hash_reducer) const {
    IterTreeNode::SHashReduce(hash_reducer);
    hash_reducer(coeff);
  }

  static constexpr const char* _type_key = "tirx.DataIterTree";
  TVM_DECLARE_FINAL_OBJECT_INFO(DataIterTreeNode, IterTreeNode);
};

class DataIterTree : public IterTree {
 public:
  TVM_DLL explicit DataIterTree(Var root, Array<IterTreeSplit> splits, Array<PrimExpr> coeff);

  TVM_DEFINE_OBJECT_REF_METHODS(DataIterTree, IterTree, DataIterTreeNode);
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
  /*! \brief If type is split, the bound leaf in DataIterTree */
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

// DeviceIterTree
class DeviceIterTreeNode : public IterTreeNode {
 public:
  /*! \brief The attributes of each leaf node */
  Array<ScopeIdAttr> attrs;

  void VisitAttrs(AttrVisitor* v) {
    IterTreeNode::VisitAttrs(v);
    v->Visit("attrs", &attrs);
  }

  bool SEqualReduce(const DeviceIterTreeNode* other, SEqualReducer equal) const {
    return IterTreeNode::SEqualReduce(other, equal) && equal(attrs, other->attrs);
  }

  void SHashReduce(SHashReducer hash_reducer) const {
    IterTreeNode::SHashReduce(hash_reducer);
    hash_reducer(attrs);
  }

  static constexpr const char* _type_key = "tirx.DeviceIterTree";
  TVM_DECLARE_FINAL_OBJECT_INFO(DeviceIterTreeNode, IterTreeNode);
};

class DeviceIterTree : public IterTree {
 public:
  TVM_DLL explicit DeviceIterTree(Var root, Array<IterTreeSplit> splits, Array<ScopeIdAttr> attrs);

  TVM_DEFINE_OBJECT_REF_METHODS(DeviceIterTree, IterTree, DeviceIterTreeNode);
};

// TileLayout
class TileLayoutNode : public TLayoutNode {
 public:
  /*! \brief coordinate iter forest  */
  Array<DataIterTree> data_trees;
  /*! \brief scope id iter forest */
  Array<DeviceIterTree> device_trees;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("data_trees", &data_trees);
    v->Visit("device_trees", &device_trees);
  }

  bool SEqualReduce(const TileLayoutNode* other, SEqualReducer equal) const {
    return equal(data_trees, other->data_trees) && equal(device_trees, other->device_trees);
  }

  void SHashReduce(SHashReducer hash_reducer) const {
    hash_reducer(data_trees);
    hash_reducer(device_trees);
  }

  static constexpr const char* _type_key = "tirx.TileLayout";
  TVM_DECLARE_FINAL_OBJECT_INFO(TileLayoutNode, TLayoutNode);
};

class TileLayout : public TLayout {
 public:
  TVM_DLL explicit TileLayout(Array<DataIterTree> data_trees, Array<DeviceIterTree> device_trees);

  TVM_DEFINE_OBJECT_REF_METHODS(TileLayout, TLayout, TileLayoutNode);
};

}  // namespace tirxxx
}  // namespace tvm

#endif  // TVM_TIRX_LAYOUT_H_
