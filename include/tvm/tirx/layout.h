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
#include <tvm/runtime/object.h>
#include <tvm/tirx/exec_scope.h>
#include <tvm/tirx/var.h>

namespace tvm {
namespace tirx {

// Base class for layout
class TLayoutNode : public Object {
 public:
  /*! \brief Get the default input shape of the layout, used by BufferView to create new logical
   * buffer */
  virtual Array<PrimExpr> GetDefaultShape() const = 0;

  /*! \brief Compatible with shape */
  virtual bool CompatibleWithShape(const Array<PrimExpr>& shape) const = 0;

  /*! \brief Verify if the layout is well-formed */
  virtual bool VerifyWellFormed() const = 0;

  /*! \brief Get the size of the layout */
  virtual PrimExpr GetSize() const = 0;

  /*! \brief Get the cosize of the layout */
  virtual PrimExpr GetCosize() const = 0;

  /*! \brief Apply the input coordinate and get the mapped output */
  virtual PrimExpr Apply(const Array<PrimExpr>& coord) const = 0;

  static constexpr const char* _type_key = "tirx.TLayout";
  static constexpr const bool _type_has_method_sequal_reduce = false;
  static constexpr const bool _type_has_method_shash_reduce = false;
  TVM_DECLARE_BASE_OBJECT_INFO(TLayoutNode, Object);
};

class TLayout : public ObjectRef {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(TLayout, ObjectRef, TLayoutNode);
};

// Base class for nodes in IterTree
// IterTree is the core data structure of TileLayout
class IterTreeBaseNode : public Object {
 public:
  static constexpr const char* _type_key = "tirx.IterTreeBase";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  TVM_DECLARE_BASE_OBJECT_INFO(IterTreeBaseNode, Object);
};

class IterTreeBase : public ObjectRef {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(IterTreeBase, ObjectRef, IterTreeBaseNode);
};

// IterTreeSplit
class IterTreeSplitNode : public IterTreeBaseNode {
 public:
  /*! \brief extent of this split */
  PrimExpr extent;
  /*! \brief children of the split */
  Array<IterTreeBase> children;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("extent", &extent);
    v->Visit("children", &children);
  }

  bool SEqualReduce(const IterTreeSplitNode* other, SEqualReducer equal) const {
    return equal(extent, other->extent) && equal(children, other->children);
  }

  void SHashReduce(SHashReducer hash_reducer) const {
    hash_reducer(extent);
    hash_reducer(children);
  }

  static constexpr const char* _type_key = "tirx.IterTreeSplit";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  TVM_DECLARE_FINAL_OBJECT_INFO(IterTreeSplitNode, IterTreeBaseNode);
};

class IterTreeSplit : public IterTreeBase {
 public:
  TVM_DLL explicit IterTreeSplit(PrimExpr extent, Array<IterTreeBase> children);

  /*! \brief Whether this is a leaf node */
  bool IsLeaf() const;

  /*! \brief Get all the leaves in the tree in DFS order */
  Array<IterTreeBase> GetLeaves() const;

  TVM_DEFINE_OBJECT_REF_METHODS(IterTreeSplit, IterTreeBase, IterTreeSplitNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(IterTreeSplitNode);
};

// IterTree
class IterTreeNode : public Object {
 public:
  /*! \brief The root split */
  IterTreeSplit root;

  void VisitAttrs(AttrVisitor* v) { v->Visit("root", &root); }

  bool SEqualReduce(const IterTreeNode* other, SEqualReducer equal) const {
    return equal(root, other->root);
  }

  void SHashReduce(SHashReducer hash_reducer) const { hash_reducer(root); }

  static constexpr const char* _type_key = "tirx.IterTree";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  TVM_DECLARE_BASE_OBJECT_INFO(IterTreeNode, Object);
};

class IterTree : public ObjectRef {
 public:
  TVM_DLL explicit IterTree(IterTreeSplit root);

  using LeafIndexMap = std::unordered_map<IterTreeBase, int, ObjectPtrHash, ObjectPtrEqual>;

  /*!
   * \brief Create an IterTree from a (nested) tuple of PrimExpr.
   * \param tuple_in The input nested tuple.
   * \return A pair. The first element is the IterTree, and the second element is the leaves.
   */
  static Array<ObjectRef> FromTuple(const ObjectRef& tuple_in);

  /*! \brief Get all the leaves in the tree in DFS order */
  Array<IterTreeBase> GetLeaves() const;

  /*! \brief Get the index of each leaf in the DFS order */
  LeafIndexMap GetLeafIndexMap(Optional<Array<IterTreeBase>> opt_leaves = NullOpt) const;

  TVM_DEFINE_OBJECT_REF_METHODS(IterTree, ObjectRef, IterTreeNode);
};

// DataIterTree
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
  TVM_DLL explicit DataIterTree(IterTreeSplit root, Array<PrimExpr> coeff);

  using CoeffMap = std::unordered_map<IterTreeBase, PrimExpr, ObjectPtrHash, ObjectPtrEqual>;

  /*! \brief Get the mapping from leaf to coefficient */
  CoeffMap GetCoeffMap(Optional<Array<IterTreeBase>> opt_leaves = NullOpt) const;

  TVM_DEFINE_OBJECT_REF_METHODS(DataIterTree, IterTree, DataIterTreeNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(DataIterTreeNode);
};

// DeviceIterAttr
enum ScopeIdType : int {
  kSplit = 0,
  kReplicate = 1,
  kExclusive = 2,
};

class DeviceIterAttrNode : public Object {
 public:
  /*! \brief type of ScopeID, can be split (S), replicate (R), exclusive (E) */
  ScopeIdType type;
  /*! \brief If type is split, the bound leaf in DataIterTree */
  Optional<PrimExpr> bound;
  /*! \brief If type is exclusive, the id that owns the data */
  Optional<PrimExpr> owner;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("type", &type);
    v->Visit("bound", &bound);
    v->Visit("owner", &owner);
  }

  bool SEqualReduce(const DeviceIterAttrNode* other, SEqualReducer equal) const {
    return equal(type, other->type) && equal(bound, other->bound) && equal(owner, other->owner);
  }

  void SHashReduce(SHashReducer hash_reducer) const {
    hash_reducer(type);
    hash_reducer(bound);
    hash_reducer(owner);
  }

  static constexpr const char* _type_key = "tirx.DeviceIterAttr";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  TVM_DECLARE_FINAL_OBJECT_INFO(DeviceIterAttrNode, Object);
};

class DeviceIterAttr : public ObjectRef {
 public:
  TVM_DLL explicit DeviceIterAttr(ScopeIdType type, Optional<PrimExpr> bound = NullOpt,
                                  Optional<PrimExpr> owner = NullOpt);

  /*! \brief Create a replicate attribute */
  static DeviceIterAttr Replicate();

  /*! \brief Create a split attribute */
  static DeviceIterAttr Split(PrimExpr bound);

  /*! \brief Create an exclusive attribute */
  static DeviceIterAttr Exclusive(PrimExpr owner);

  /*! \brief Check if the attribute is replicate */
  bool IsReplicate() const;

  /*! \brief Check if the attribute is split */
  bool IsSplit() const;

  /*! \brief Check if the attribute is exclusive */
  bool IsExclusive() const;

  /*! \brief Get the bound of the split attribute */
  size_t GetIntBound() const;

  TVM_DEFINE_OBJECT_REF_METHODS(DeviceIterAttr, ObjectRef, DeviceIterAttrNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(DeviceIterAttrNode);
};

// DeviceIterTree
class DeviceIterTreeNode : public IterTreeNode {
 public:
  /*! \brief The attributes of each leaf node */
  Array<DeviceIterAttr> attrs;

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
  TVM_DLL explicit DeviceIterTree(IterTreeSplit root, Array<DeviceIterAttr> attrs);

  using AttrMap = std::unordered_map<IterTreeBase, DeviceIterAttr, ObjectPtrHash, ObjectPtrEqual>;

  /*! \brief Get the mapping from leaf to DeviceIterAttr */
  AttrMap GetAttrMap(Optional<Array<IterTreeBase>> opt_leaves = NullOpt) const;

  TVM_DEFINE_OBJECT_REF_METHODS(DeviceIterTree, IterTree, DeviceIterTreeNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(DeviceIterTreeNode);
};

// TileLayout
class TileLayoutNode : public TLayoutNode {
 public:
  /*! \brief data iter tree */
  DataIterTree data_tree;
  /*! \brief device iter tree */
  Optional<DeviceIterTree> device_tree;
  /*! \brief From exec scope */
  Optional<ExecScope> from;
  /*! \brief To exec scope */
  Optional<ExecScope> to;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("data_tree", &data_tree);
    v->Visit("device_tree", &device_tree);
    v->Visit("from", &from);
    v->Visit("to", &to);
  }

  bool SEqualReduce(const TileLayoutNode* other, SEqualReducer equal) const {
    return equal(data_tree, other->data_tree) && equal(device_tree, other->device_tree) &&
           equal(from, other->from) && equal(to, other->to);
  }

  void SHashReduce(SHashReducer hash_reducer) const {
    hash_reducer(data_tree);
    hash_reducer(device_tree);
    hash_reducer(from);
    hash_reducer(to);
  }

  /*! \brief Get the input shape of the layout */
  Array<PrimExpr> GetDefaultShape() const final;

  /*! \brief Compatible with shape */
  bool CompatibleWithShape(const Array<PrimExpr>& shape) const final;

  /*! \brief Verify if the layout is well-formed */
  bool VerifyWellFormed() const final;

  /*! \brief Get the size of the layout */
  PrimExpr GetSize() const final;

  /*! \brief Get the cosize of the layout */
  PrimExpr GetCosize() const final;

  /*! \brief Apply the input coordinate and get the mapped output */
  PrimExpr Apply(const Array<PrimExpr>& coord) const final;

  static constexpr const char* _type_key = "tirx.TileLayout";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  TVM_DECLARE_FINAL_OBJECT_INFO(TileLayoutNode, TLayoutNode);
};

class TileLayout : public TLayout {
 public:
  TVM_DLL explicit TileLayout(DataIterTree data_tree,
                              Optional<DeviceIterTree> device_tree = NullOpt,
                              Optional<ExecScope> from = NullOpt, Optional<ExecScope> to = NullOpt);

  using SplitMap = std::unordered_map<IterTreeBase, IterTreeBase, ObjectPtrHash, ObjectPtrEqual>;

  /*! \brief Get the mapping from leaf in data tree to leaf in device tree when they are bound */
  SplitMap GetSplitMap(Optional<Array<IterTreeBase>> opt_data_leaves = NullOpt,
                       Optional<Array<IterTreeBase>> opt_device_leaves = NullOpt) const;

  /*! \brief Update the tree with the given roots and mappings */
  static TileLayout FromMaps(IterTreeSplit data_root, Optional<IterTreeSplit> device_root,
                             const DataIterTree::CoeffMap& coeff,
                             const DeviceIterTree::AttrMap& attr, const SplitMap& split_map,
                             Optional<ExecScope> from = NullOpt, Optional<ExecScope> to = NullOpt);

  TVM_DEFINE_OBJECT_REF_METHODS(TileLayout, TLayout, TileLayoutNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(TileLayoutNode);
};

/*! \brief Construct a new layout by tiling the ouer layout over the inner layout */
TileLayout Tile(TileLayout outer, TileLayout inner);

/*!
 * \brief Construct a new layout to express the sharding strategy of a tensor.
 * \param shape The shape of the tensor.
 * \param mesh The device mesh
 * \param strategy The sharding strategy of the tensor.
 * \param inner The layout of the sharded partition of the tensor.
 * \param from The source scope of the layout.
 * \param to The target scope of the layout.
 */
TileLayout Shard(Array<PrimExpr> shape, IterTree mesh, String strategy, TileLayout inner,
                 ExecScope from, ExecScope to);

/*! \brief Layout normalization
    1. Deduplicate the split nodes in the tree, such that no two split nodes share the same child
   node.
    2. Remove the split nodes with extent 1.
 */
TileLayout NormalizeTileLayout(TileLayout layout);

// SwizzleLayout
class SwizzleLayoutNode : public TLayoutNode {
 public:
  int per_element;
  int swizzle_len;
  int atom_len;
  bool swizzle_inner;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("per_element", &per_element);
    v->Visit("swizzle_len", &swizzle_len);
    v->Visit("atom_len", &atom_len);
    v->Visit("swizzle_inner", &swizzle_inner);
  }

  bool SEqualReduce(const SwizzleLayoutNode* other, SEqualReducer equal) const {
    return equal(per_element, other->per_element) && equal(swizzle_len, other->swizzle_len) &&
           equal(atom_len, other->atom_len) && equal(swizzle_inner, other->swizzle_inner);
  }

  void SHashReduce(SHashReducer hash_reducer) const {
    hash_reducer(per_element);
    hash_reducer(swizzle_len);
    hash_reducer(atom_len);
    hash_reducer(swizzle_inner);
  }

  /*! \brief Get the input shape of the layout */
  Array<PrimExpr> GetDefaultShape() const final;

  /*! \brief Compatible with shape */
  bool CompatibleWithShape(const Array<PrimExpr>& shape) const final;

  /*! \brief Verify if the layout is well-formed */
  bool VerifyWellFormed() const final;

  /*! \brief Get the size of the layout */
  PrimExpr GetSize() const final;

  /*! \brief Get the cosize of the layout */
  PrimExpr GetCosize() const final;

  /*! \brief Apply the input coordinate and get the mapped output */
  PrimExpr Apply(const Array<PrimExpr>& coord) const final;

  static constexpr const char* _type_key = "tirx.SwizzleLayout";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  TVM_DECLARE_FINAL_OBJECT_INFO(SwizzleLayoutNode, TLayoutNode);

 private:
  friend class SwizzleLayout;
  int inner_mask;
  int outer_mask;
};

class SwizzleLayout : public TLayout {
 public:
  TVM_DLL explicit SwizzleLayout(int per_element, int swizzle_len, int atom_len,
                                 bool swizzle_inner);

  TVM_DEFINE_OBJECT_REF_METHODS(SwizzleLayout, TLayout, SwizzleLayoutNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(SwizzleLayoutNode);
};

/********************* Utils *********************/
bool IsTrivialLayout(const TLayout& layout);

}  // namespace tirx
}  // namespace tvm

#endif  // TVM_TIRX_LAYOUT_H_
