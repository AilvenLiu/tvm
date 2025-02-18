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

class TLayout;
class TileLayout;
// Base class for layout
class TLayoutNode : public Object {
 public:
  /*! \brief Compatible with shape */
  virtual bool CompatibleWithShape(const Array<PrimExpr>& shape) const;

  /*! \brief Verify if the layout is well-formed */
  virtual bool VerifyWellFormed() const = 0;

  /*! \brief Get the size of the layout */
  virtual PrimExpr GetSize() const = 0;

  /*! \brief Get the cosize of the layout */
  virtual PrimExpr GetCosize() const = 0;

  /*! \brief Apply layout on the input coordinate and get the mapped output */
  virtual Array<PrimExpr> Apply(const Array<PrimExpr>& coord, const Array<PrimExpr>& shape) const;

  /*! \brief Apply layout on the flattened coordinate and get the mapped output */
  virtual Array<PrimExpr> Apply(const PrimExpr& coord) const = 0;

  /*! \brief Normalize the layout */
  virtual TLayout Normalize() const;

  /*! \brief Tile the layout with an outer layout */
  virtual TLayout Tile(const TileLayout& outer, const Array<PrimExpr>& outer_shape,
                       const Array<PrimExpr>& inner_shape) const;

  /*! \brief Check if the layout is the inner layout of a tiled layout */
  virtual bool IsTileInner(const TLayout& tile_layout, const Array<PrimExpr>& tiled_shape,
                           const Array<PrimExpr>& inner_shape) const;

  /*! \brief Check if the layout is the outer layout of a tiled layout */
  virtual bool IsTileOuter(const TLayout& tile_layout, const Array<PrimExpr>& tiled_shape,
                           const Array<PrimExpr>& outer_shape) const;

  static constexpr const char* _type_key = "tirx.TLayout";
  static constexpr const bool _type_has_method_sequal_reduce = false;
  static constexpr const bool _type_has_method_shash_reduce = false;
  TVM_DECLARE_BASE_OBJECT_INFO(TLayoutNode, Object);
};

class TLayout : public ObjectRef {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(TLayout, ObjectRef, TLayoutNode);
};

// DeviceIterAttr
enum ScopeIdType : int {
  kSplit = 0,
  kReplicate = 1,
  kExclusive = 2,
};

class DataIterAttrNode : public Object {
 public:
  PrimExpr extent;
  PrimExpr stride;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("extent", &extent);
    v->Visit("stride", &stride);
  }

  bool SEqualReduce(const DataIterAttrNode* other, SEqualReducer equal) const {
    return equal(extent, other->extent) && equal(stride, other->stride);
  }

  void SHashReduce(SHashReducer hash_reducer) const {
    hash_reducer(extent);
    hash_reducer(stride);
  }

  static constexpr const char* _type_key = "tirx.DataIterAttr";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  TVM_DECLARE_FINAL_OBJECT_INFO(DataIterAttrNode, Object);
};

class DataIterAttr : public ObjectRef {
 public:
  TVM_DLL explicit DataIterAttr(PrimExpr extent, PrimExpr stride);

  TVM_DEFINE_OBJECT_REF_METHODS(DataIterAttr, ObjectRef, DataIterAttrNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(DataIterAttrNode);
};

class DeviceIterAttrNode : public Object {
 public:
  PrimExpr extent;
  /*! \brief type of ScopeID, can be split (S), replicate (R), exclusive (E) */
  ScopeIdType type;
  /*! \brief If type is split, the bound leaf in DataIterTree */
  Optional<PrimExpr> bound;
  /*! \brief If type is exclusive, the id that owns the data */
  Optional<PrimExpr> owner;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("extent", &extent);
    v->Visit("type", &type);
    v->Visit("bound", &bound);
    v->Visit("owner", &owner);
  }

  bool SEqualReduce(const DeviceIterAttrNode* other, SEqualReducer equal) const {
    return equal(extent, other->extent) && equal(type, other->type) && equal(bound, other->bound) &&
           equal(owner, other->owner);
  }

  void SHashReduce(SHashReducer hash_reducer) const {
    hash_reducer(extent);
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
  TVM_DLL explicit DeviceIterAttr(PrimExpr extent, ScopeIdType type,
                                  Optional<PrimExpr> bound = NullOpt,
                                  Optional<PrimExpr> owner = NullOpt);

  /*! \brief Create a replicate attribute */
  static DeviceIterAttr Replicate(PrimExpr extent);

  /*! \brief Create a split attribute */
  static DeviceIterAttr Split(PrimExpr extent, PrimExpr bound);

  /*! \brief Create an exclusive attribute */
  static DeviceIterAttr Exclusive(PrimExpr extent, PrimExpr owner);

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

// TileLayout
class TileLayoutNode : public TLayoutNode {
 public:
  /*! \brief data iter tree */
  Array<DataIterAttr> data_iter_array;
  /*! \brief device iter tree */
  Array<DeviceIterAttr> device_iter_array;
  /*! \brief From exec scope */
  Optional<ExecScope> from;
  /*! \brief To exec scope */
  Optional<ExecScope> to;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("data_iter_array", &data_iter_array);
    v->Visit("device_iter_array", &device_iter_array);
    v->Visit("from", &from);
    v->Visit("to", &to);
  }

  bool SEqualReduce(const TileLayoutNode* other, SEqualReducer equal) const {
    return equal(data_iter_array, other->data_iter_array) &&
           equal(device_iter_array, other->device_iter_array) && equal(from, other->from) &&
           equal(to, other->to);
  }

  void SHashReduce(SHashReducer hash_reducer) const {
    hash_reducer(data_iter_array);
    hash_reducer(device_iter_array);
    hash_reducer(from);
    hash_reducer(to);
  }
  /*! \brief Verify if the layout is well-formed */
  bool VerifyWellFormed() const final;

  /*! \brief Get the size of the layout */
  PrimExpr GetSize() const final;

  /*! \brief Get the cosize of the layout */
  PrimExpr GetCosize() const final;

  /*! \brief Apply the input coordinate and get the mapped output */
  Array<PrimExpr> Apply(const PrimExpr& coord) const final;

  /*! \brief Normalize the layout */
  TLayout Normalize() const final;

  /*! \brief Tile the layout with an outer layout */
  TLayout Tile(const TileLayout& outer, const Array<PrimExpr>& outer_shape,
               const Array<PrimExpr>& inner_shape) const final;

  /*! \brief Check if the layout is the inner layout of a tiled layout */
  bool IsTileInner(const TLayout& tile_layout, const Array<PrimExpr>& tiled_shape,
                   const Array<PrimExpr>& inner_shape) const final;

  /*! \brief Check if the layout is the outer layout of a tiled layout */
  bool IsTileOuter(const TLayout& tile_layout, const Array<PrimExpr>& tiled_shape,
                   const Array<PrimExpr>& outer_shape) const final;

  static constexpr const char* _type_key = "tirx.TileLayout";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  TVM_DECLARE_FINAL_OBJECT_INFO(TileLayoutNode, TLayoutNode);
};

class TileLayout : public TLayout {
 public:
  TVM_DLL explicit TileLayout(Array<DataIterAttr> data_iter_array,
                              Array<DeviceIterAttr> device_iter_array = {},
                              Optional<ExecScope> from = NullOpt, Optional<ExecScope> to = NullOpt);

  using SplitMap = std::unordered_map<int, int>;

  /*! \brief Get the mapping from leaf in data tree to leaf in device tree when they are bound */
  SplitMap GetSplitMap() const;

  TVM_DEFINE_OBJECT_REF_METHODS(TileLayout, TLayout, TileLayoutNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(TileLayoutNode);
};

/*!
 * \brief Construct a new layout to express the sharding strategy of a tensor.
 * \param shape The shape of the tensor.
 * \param mesh The device mesh
 * \param strategy The sharding strategy of the tensor.
 * \param inner The layout of the sharded partition of the tensor.
 * \param from The source scope of the layout.
 * \param to The target scope of the layout.
 */
TileLayout Shard(Array<PrimExpr> shape, Array<PrimExpr> mesh, String strategy, TileLayout inner,
                 ExecScope from, ExecScope to);

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

  /*! \brief Verify if the layout is well-formed */
  bool VerifyWellFormed() const final;

  /*! \brief Get the size of the layout */
  PrimExpr GetSize() const final;

  /*! \brief Get the cosize of the layout */
  PrimExpr GetCosize() const final;

  /*! \brief Apply the input coordinate and get the mapped output */
  Array<PrimExpr> Apply(const PrimExpr& coord) const final;

  /*! \brief Normalize the layout */
  TLayout Normalize() const final;

  /*! \brief Tile the layout with an outer layout */
  TLayout Tile(const TileLayout& outer, const Array<PrimExpr>& outer_shape,
               const Array<PrimExpr>& inner_shape) const final;

  /*! \brief Check if the layout is the inner layout of a tiled layout */
  bool IsTileInner(const TLayout& tile_layout, const Array<PrimExpr>& tiled_shape,
                   const Array<PrimExpr>& inner_shape) const final;

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

// ComposeLayout
class ComposeLayoutNode : public TLayoutNode {
 public:
  SwizzleLayout layout_A;
  TileLayout layout_B;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("layout_A", &layout_A);
    v->Visit("layout_B", &layout_B);
  }

  bool SEqualReduce(const ComposeLayoutNode* other, SEqualReducer equal) const {
    return equal(layout_A, other->layout_A) && equal(layout_B, other->layout_B);
  }

  void SHashReduce(SHashReducer hash_reducer) const {
    hash_reducer(layout_A);
    hash_reducer(layout_B);
  }

  /*! \brief Check if the layout is compatible with the shape */
  bool CompatibleWithShape(const Array<PrimExpr>& shape) const final;

  /*! \brief Verify if the layout is well-formed */
  bool VerifyWellFormed() const final;

  /*! \brief Get the size of the layout */
  PrimExpr GetSize() const final;

  /*! \brief Get the cosize of the layout */
  PrimExpr GetCosize() const final;

  /*! \brief Apply the input coordinate and get the mapped output */
  Array<PrimExpr> Apply(const PrimExpr& coord) const final;

  /*! \brief Normalize the layout */
  TLayout Normalize() const final;

  /*! \brief Tile the layout with an outer layout */
  TLayout Tile(const TileLayout& outer, const Array<PrimExpr>& outer_shape,
               const Array<PrimExpr>& inner_shape) const final;

  /*! \brief Check if the layout is the inner layout of a tiled layout */
  bool IsTileInner(const TLayout& tile_layout, const Array<PrimExpr>& tiled_shape,
                   const Array<PrimExpr>& inner_shape) const final;

  static constexpr const char* _type_key = "tirx.ComposeLayout";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  TVM_DECLARE_FINAL_OBJECT_INFO(ComposeLayoutNode, TLayoutNode);
};

class ComposeLayout : public TLayout {
 public:
  TVM_DLL explicit ComposeLayout(SwizzleLayout layout_A, TileLayout layout_B);

  TVM_DEFINE_OBJECT_REF_METHODS(ComposeLayout, TLayout, ComposeLayoutNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(ComposeLayoutNode);
};

// Trainium Layout

enum PhysicalDimensionType : int {
  kPartition = 0,
  kFree = 1,
};

class TrainiumLayoutNode : public TLayoutNode {
 public:
  ShapeTuple dimension_types;
  TileLayout combined_1d_layout;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("dimension_types", &dimension_types);
    v->Visit("combined_1d_layout", &combined_1d_layout);
  }

  bool SEqualReduce(const TrainiumLayoutNode* other, SEqualReducer equal) const {
    return equal(dimension_types, other->dimension_types) &&
           equal(combined_1d_layout, other->combined_1d_layout);
  }

  void SHashReduce(SHashReducer hash_reducer) const {
    hash_reducer(dimension_types);
    hash_reducer(combined_1d_layout);
  }
  /*! \brief Check if the layout is compatible with the shape */
  bool CompatibleWithShape(const Array<PrimExpr>& shape) const final;

  /*! \brief Verify if the layout is well-formed */
  virtual bool VerifyWellFormed() const;

  /*! \brief Get the size of the layout */
  PrimExpr GetSize() const;

  /*! \brief Get the cosize of the layout */
  PrimExpr GetCosize() const;

  /*! \brief Get the partition dimension size */
  PrimExpr GetPartitionSize() const;

  /*! \brief Apply the input coordinate and get the mapped output */
  Array<PrimExpr> Apply(const Array<PrimExpr>& coord, const Array<PrimExpr>& shape) const;

  virtual Array<PrimExpr> Apply(const PrimExpr& coord) const;

  /*! \brief Normalize the layout */
  TLayout Normalize() const final;

  static constexpr const char* _type_key = "tirx.TrainiumLayout";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  TVM_DECLARE_BASE_OBJECT_INFO(TrainiumLayoutNode, TLayoutNode);
};

class TrainiumLayout : public TLayout {
 public:
  TVM_DLL explicit TrainiumLayout(ShapeTuple dimension_types, TileLayout combined_1d_layout);

  TVM_DEFINE_OBJECT_REF_METHODS(TrainiumLayout, TLayout, TrainiumLayoutNode);
};

class TrainiumPSUMLayoutNode : public TrainiumLayoutNode {
 public:
  Array<PrimExpr> Apply(const PrimExpr& coord) const final;

  bool VerifyWellFormed() const final;

  static constexpr const char* _type_key = "tir.TrainiumPSUMLayout";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  TVM_DECLARE_FINAL_OBJECT_INFO(TrainiumPSUMLayoutNode, TrainiumLayoutNode);
};

class TrainiumPSUMLayout : public TrainiumLayout {
 public:
  TVM_DLL explicit TrainiumPSUMLayout(ShapeTuple dimension_types, TileLayout combined_1d_layout);

  TVM_DEFINE_OBJECT_REF_METHODS(TrainiumPSUMLayout, TrainiumLayout, TrainiumPSUMLayoutNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(TrainiumPSUMLayoutNode);
};

constexpr int kPSUMMaxElemPerBank = 512;
constexpr int kPSUMBankNum = 8;

/********************* Utils *********************/
bool IsTrivialLayout(const TLayout& layout);

}  // namespace tirxxx
}  // namespace tvm

#endif  // TVM_TIRX_LAYOUT_H_
