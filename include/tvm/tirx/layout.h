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

#include <tvm/ffi/container/tuple.h>
#include <tvm/ir/module.h>
#include <tvm/node/attr_registry_map.h>
#include <tvm/runtime/object.h>
#include <tvm/tirx/exec_scope.h>
#include <tvm/tirx/var.h>

namespace tvm {

// Forward declaration
template <typename, typename>
class AttrRegistry;

namespace tirx {
template <typename>
class AxisAttrMap;

class TLayout;
class TileLayout;
using ffi::Tuple;

// Base class for layout
class TLayoutNode : public Object {
 public:
  /*! \brief Compatible with shape */
  virtual bool CompatibleWithShape(const Array<PrimExpr>& shape) const = 0;

  /*! \brief Verify if the layout is well-formed */
  virtual bool VerifyWellFormed() const = 0;

  /*! \brief Get the size of the layout (of some axis) */
  virtual PrimExpr GetSize(Optional<String> axis_name = std::nullopt) const = 0;

  /*! \brief Get the cosize of the layout (of some axis) */
  virtual PrimExpr GetCosize(Optional<String> axis_name = std::nullopt) const = 0;

  /*! \brief Apply layout on the input coordinate and get the mapped output */
  virtual Map<String, PrimExpr> Apply(Array<PrimExpr> coord) const = 0;
  virtual Map<String, PrimExpr> Apply(PrimExpr coord) const = 0;
  Map<String, PrimExpr> Apply(const Array<PrimExpr>& coord, const Array<PrimExpr>& shape) const;

  /*! \brief Turn the layout to normalized form */
  virtual TLayout Normalize() const = 0;

  /*! \brief Tile the current layout with a given layout */
  virtual TLayout Tile(const TileLayout& outer, const Array<PrimExpr>& outer_shape,
                       const Array<PrimExpr>& inner_shape) const = 0;

  /*! \brief Check if the layout is the inner layout of a tiled layout
   * \param tile_layout The tiled layout to check
   * \param tiled_shape The shape of the tiled layout
   * \param inner_shape The shape of the inner layout
   * \return The outer layout if this layout is the inner layout of tile_layout, std::nullopt
   * otherwise
   */
  virtual Optional<TileLayout> IsTileInner(const TLayout& tile_layout,
                                           const Array<PrimExpr>& tiled_shape,
                                           const Array<PrimExpr>& inner_shape) const = 0;

  /*! \brief Check if the layout is the outer layout of a tiled layout
   * \param tile_layout The tiled layout to check
   * \param tiled_shape The shape of the tiled layout
   * \param outer_shape The shape of the outer layout
   * \return The inner layout if this layout is the outer layout of tile_layout, std::nullopt
   * otherwise
   */
  virtual Optional<TLayout> IsTileOuter(const TLayout& tile_layout,
                                        const Array<PrimExpr>& tiled_shape,
                                        const Array<PrimExpr>& outer_shape) const = 0;

  static constexpr const char* _type_key = "tirx.TLayout";
  static constexpr const bool _type_has_method_sequal_reduce = false;
  static constexpr const bool _type_has_method_shash_reduce = false;
  TVM_DECLARE_BASE_OBJECT_INFO(TLayoutNode, Object);
};

class TLayout : public ObjectRef {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(TLayout, ObjectRef, TLayoutNode);
};

// Axis
class AxisNode : public Object {
 public:
  String name;

  void VisitAttrs(AttrVisitor* v) { v->Visit("name", &name); }

  bool SEqualReduce(const AxisNode* other, SEqualReducer equal) const {
    return equal(name, other->name);
  }

  void SHashReduce(SHashReducer hash_reducer) const { hash_reducer(name); }

  /*! \brief Check if the axis is a thread axis. */
  bool IsThreadAxis() const;

  /*! \brief Check if the axis is a memory axis. */
  bool IsMemoryAxis() const;

  static constexpr const char* _type_key = "tirx.Axis";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  TVM_DECLARE_FINAL_OBJECT_INFO(AxisNode, Object);

 private:
  // Iternals necessary for AttrRegistry
  template <typename>
  friend class tvm::AttrRegistryMapContainerMap;
  template <typename, typename>
  friend class tvm::AttrRegistry;
  friend class AxisRegEntry;
  /*! \brief Program internal unique index of operator. */
  uint32_t index_{0};
  /*! \brief Return the index stored in attr registry */
  uint32_t AttrRegistryIndex() const { return index_; }
  /*! \brief Return the name stored in attr registry */
  String AttrRegistryName() const { return name; }
};

class Axis : public ObjectRef {
 public:
  Axis() = default;

  /*! \brief Get the axis object by name. */
  TVM_DLL static Axis Get(const String& name);

  /*! \brief Get the attribute map for the axis. */
  template <typename ValueType>
  inline static AxisAttrMap<ValueType> GetAttrMap(const String& attr_name);

  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(Axis, ObjectRef, AxisNode);

 private:
  // Internals necessary for AttrRegistry
  template <typename, typename>
  friend class tvm::AttrRegistry;
  friend class AxisRegEntry;
};

// AxisRegistry
class AxisRegEntry {
 public:
  /*! \brief List all axis names. */
  TVM_DLL static Array<String> ListAxisNames();

  /*! \brief Register or get the axis entry by name. */
  TVM_DLL static AxisRegEntry& RegisterOrGet(const String& name);

  /*! \brief Set the attribute for the axis. */
  template <typename ValueType>
  inline AxisRegEntry& set_attr(const String& attr_name, const ValueType& value, int plevel = 10);

 private:
  // return internal pointer to op.
  inline AxisNode* get();
  TVM_DLL void UpdateAttr(const String& key, ffi::Any value, int plevel);

  // Internals necessary for AttrRegistry
  Axis axis_;
  String name;
  explicit AxisRegEntry(uint32_t index);
  template <typename, typename>
  friend class tvm::AttrRegistry;
  friend class Axis;
};

using AxisRegistry = AttrRegistry<AxisRegEntry, Axis>;

// AxisAttrMap
template <typename ValueType>
class AxisAttrMap : public AttrRegistryMap<Axis, ValueType> {
 public:
  using TParent = AttrRegistryMap<Axis, ValueType>;
  using TParent::count;
  using TParent::get;
  using TParent::operator[];

 private:
  friend class Axis;
  explicit AxisAttrMap(const AttrRegistryMapContainerMap<Axis>& map) : TParent(map) {}
};

// Define a macro to register the axis entry.
#define TVM_AXIS_REGISTER_VAR_DEF \
  static DMLC_ATTRIBUTE_UNUSED ::tvm::tirx::AxisRegEntry& __make_##Axis

#define TVM_REGISTER_AXIS(AxisName)                        \
  TVM_STR_CONCAT(TVM_AXIS_REGISTER_VAR_DEF, __COUNTER__) = \
      ::tvm::tirx::AxisRegEntry::RegisterOrGet(AxisName)

class IterNode : public Object {
 public:
  PrimExpr extent;
  PrimExpr stride;
  Axis axis;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("extent", &extent);
    v->Visit("stride", &stride);
    v->Visit("axis", &axis);
  }

  bool SEqualReduce(const IterNode* other, SEqualReducer equal) const {
    return equal(extent, other->extent) && equal(stride, other->stride) && equal(axis, other->axis);
  }

  void SHashReduce(SHashReducer hash_reducer) const {
    hash_reducer(extent);
    hash_reducer(stride);
    hash_reducer(axis);
  }

  static constexpr const char* _type_key = "tirx.Iter";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  TVM_DECLARE_FINAL_OBJECT_INFO(IterNode, Object);
};

class Iter : public ObjectRef {
 public:
  TVM_DLL explicit Iter(PrimExpr extent, PrimExpr stride, Axis axis);
  TVM_DEFINE_OBJECT_REF_METHODS(Iter, ObjectRef, IterNode);
};

class TileLayoutNode : public TLayoutNode {
 public:
  Array<Iter> shard;
  Array<Iter> replicate;
  Array<Tuple<Iter, PrimExpr>> exclude;
  Optional<ExecScope> subscope;
  Optional<ExecScope> scope;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("shard", &shard);
    v->Visit("replicate", &replicate);
    v->Visit("exclude", &exclude);
    v->Visit("subscope", &subscope);
    v->Visit("scope", &scope);
  }

  bool SEqualReduce(const TileLayoutNode* other, SEqualReducer equal) const {
    return equal(shard, other->shard) && equal(replicate, other->replicate) &&
           equal(exclude, other->exclude) && equal(subscope, other->subscope) &&
           equal(scope, other->scope);
  }

  void SHashReduce(SHashReducer hash_reducer) const {
    hash_reducer(shard);
    hash_reducer(replicate);
    hash_reducer(exclude);
    hash_reducer(subscope);
    hash_reducer(scope);
  }

  /*! \brief Check if the layout is compatible with the shape */
  bool CompatibleWithShape(const Array<PrimExpr>& shape) const final;

  /*! \brief Verify if the layout is well-formed */
  bool VerifyWellFormed() const final;

  /*! \brief Get the size of the layout (of some axis) */
  PrimExpr GetSize(Optional<String> axis_name = std::nullopt) const final;

  /*! \brief Get the cosize of the layout (of some axis) */
  PrimExpr GetCosize(Optional<String> axis_name = std::nullopt) const final;

  /*! \brief Apply the input coordinate and get the mapped output */
  Map<String, PrimExpr> Apply(Array<PrimExpr> coord) const final;
  Map<String, PrimExpr> Apply(PrimExpr coord) const final;

  /*! \brief Turn the layout to normalized form */
  TLayout Normalize() const final;

  /*! \brief Tile the layout with an outer layout */
  TLayout Tile(const TileLayout& outer, const Array<PrimExpr>& outer_shape,
               const Array<PrimExpr>& inner_shape) const final;

  /*! \brief Check if the layout is the inner layout of a tiled layout */
  Optional<TileLayout> IsTileInner(const TLayout& tile_layout, const Array<PrimExpr>& tiled_shape,
                                   const Array<PrimExpr>& inner_shape) const final;

  /*! \brief Check if the layout is the outer layout of a tiled layout */
  Optional<TLayout> IsTileOuter(const TLayout& tile_layout, const Array<PrimExpr>& tiled_shape,
                                const Array<PrimExpr>& outer_shape) const final;

  /*! \brief Get the shape of the shard */
  Array<PrimExpr> GetShardShape() const;

  /*! \brief Is the layout trivial (pure memory, identical mapping) */
  bool IsTrivial() const;

  /*! \brief Check if the layout is trainium layout */
  bool IsTrainium() const;

  /*! \brief Has Memory Axis */
  bool HasMemoryAxis() const;

  /*! \brief Has Thread Axis */
  bool HasThreadAxis() const;

  static constexpr const char* _type_key = "tirx.TileLayout";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  TVM_DECLARE_FINAL_OBJECT_INFO(TileLayoutNode, TLayoutNode);
};

class TileLayout : public TLayout {
 public:
  TVM_DLL explicit TileLayout(Array<Iter> shard, Array<Iter> replicate,
                              Array<Tuple<Iter, PrimExpr>> exclude,
                              Optional<ExecScope> subscope = std::nullopt,
                              Optional<ExecScope> scope = std::nullopt);

  TVM_DEFINE_OBJECT_REF_METHODS(TileLayout, TLayout, TileLayoutNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(TileLayoutNode);
};

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

  /*! \brief Check if the layout is compatible with the shape */
  bool CompatibleWithShape(const Array<PrimExpr>& shape) const final;

  /*! \brief Verify if the layout is well-formed */
  bool VerifyWellFormed() const final;

  /*! \brief Get the size of the layout */
  PrimExpr GetSize(Optional<String> axis_name = std::nullopt) const final;

  /*! \brief Get the cosize of the layout */
  PrimExpr GetCosize(Optional<String> axis_name = std::nullopt) const final;

  /*! \brief Apply the input coordinate and get the mapped output */
  Map<String, PrimExpr> Apply(Array<PrimExpr> coord) const final;
  Map<String, PrimExpr> Apply(PrimExpr coord) const final;

  /*! \brief Turn the layout to normalized form */
  TLayout Normalize() const final;

  /*! \brief Tile the layout with an outer layout */
  TLayout Tile(const TileLayout& outer, const Array<PrimExpr>& outer_shape,
               const Array<PrimExpr>& inner_shape) const final;

  /*! \brief Check if the layout is the inner layout of a tiled layout */
  Optional<TileLayout> IsTileInner(const TLayout& tile_layout, const Array<PrimExpr>& tiled_shape,
                                   const Array<PrimExpr>& inner_shape) const final;

  /*! \brief Check if the layout is the outer layout of a tiled layout */
  Optional<TLayout> IsTileOuter(const TLayout& tile_layout, const Array<PrimExpr>& tiled_shape,
                                const Array<PrimExpr>& outer_shape) const final;

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

  /*! \brief Get the size (of some axis) of the layout */
  PrimExpr GetSize(Optional<String> axis_name = std::nullopt) const final;

  /*! \brief Get the cosize (of some axis) of the layout */
  PrimExpr GetCosize(Optional<String> axis_name = std::nullopt) const final;

  /*! \brief Apply the input coordinate and get the mapped output */
  Map<String, PrimExpr> Apply(Array<PrimExpr> coord) const final;
  Map<String, PrimExpr> Apply(PrimExpr coord) const final;

  /*! \brief Turn the layout to normalized form */
  TLayout Normalize() const final;

  /*! \brief Tile the layout with an outer layout */
  TLayout Tile(const TileLayout& outer, const Array<PrimExpr>& outer_shape,
               const Array<PrimExpr>& inner_shape) const final;

  /*! \brief Check if the layout is the inner layout of a tiled layout */
  Optional<TileLayout> IsTileInner(const TLayout& tile_layout, const Array<PrimExpr>& tiled_shape,
                                   const Array<PrimExpr>& inner_shape) const final;

  /*! \brief Check if the layout is the outer layout of a tiled layout */
  Optional<TLayout> IsTileOuter(const TLayout& tile_layout, const Array<PrimExpr>& tiled_shape,
                                const Array<PrimExpr>& outer_shape) const final;

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

constexpr int kPSUMMaxElemPerBank = 512;
constexpr int kPSUMBankNum = 8;

}  // namespace tirx
}  // namespace tvm

#endif  // TVM_TIRX_LAYOUT_H_
