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
 * \file tvm/ir/tensormap_type.h
 * \brief TensorMap type in the IR.
 */
#ifndef TVM_IR_TENSORMAP_TYPE_H_
#define TVM_IR_TENSORMAP_TYPE_H_

#include <tvm/ir/expr.h>
#include <tvm/tir/expr.h>

namespace tvm {

enum class TensorMapInterleavedKind : int {
  kNone = 0,
  k16B = 1,
  k32B = 2,
};

enum class TensorMapSwizzleKind : int {
  kNone = 0,
  k32B = 1,
  k64B = 2,
  k128B = 3,
  k128B_BASE32B = 4,
};

enum class TensorMapL2PromotionKind : int {
  kNone = 0,
  kL2_64B = 1,
  kL2_128B = 2,
  kL2_256B = 3,
};

enum class TensorMapOOBFillKind : int {
  kNone = 0,
  kNan = 1,
};

/*!
 * \brief TensorMap type in the IR.
 *
 * TensorMap is a type that represents a CuTensorMap object in the TVM IR.
 * \sa TensorMapTypeNode
 */
class TensorMapTypeNode : public TypeNode {
 public:
  /*! \brief The data type of the tensor. */
  runtime::DataType tensor_dtype;
  /*! \brief The shape of the global tensor. */
  Array<PrimExpr> global_shape;
  /*! \brief The strides of the global tensor */
  Array<PrimExpr> global_strides;
  /*! \brief The shape of the shared memory tensor */
  Array<PrimExpr> shared_shape;
  /*! \brief The strides of the shared memory tensor */
  Array<PrimExpr> shared_strides;
  /*! \brief The interleaved kind */
  TensorMapInterleavedKind interleaved_kind;
  /*! \brief The swizzle kind */
  TensorMapSwizzleKind swizzle_kind;
  /*! \brief The L2 promotion kind */
  TensorMapL2PromotionKind l2_promotion_kind;
  /*! \brief The OOB fill kind */
  TensorMapOOBFillKind oob_fill_kind;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("tensor_dtype", &tensor_dtype);
    v->Visit("global_shape", &global_shape);
    v->Visit("global_strides", &global_strides);
    v->Visit("shared_shape", &shared_shape);
    v->Visit("shared_strides", &shared_strides);
    v->Visit("interleaved_kind", &interleaved_kind);
    v->Visit("swizzle_kind", &swizzle_kind);
    v->Visit("l2_promotion_kind", &l2_promotion_kind);
    v->Visit("oob_fill_kind", &oob_fill_kind);
  }

  bool SEqualReduce(const TensorMapTypeNode* other, SEqualReducer equal) const {
    return equal(tensor_dtype, other->tensor_dtype) && equal(global_shape, other->global_shape) &&
           equal(global_strides, other->global_strides) &&
           equal(shared_shape, other->shared_shape) &&
           equal(shared_strides, other->shared_strides) &&
           equal(interleaved_kind, other->interleaved_kind) &&
           equal(swizzle_kind, other->swizzle_kind) &&
           equal(l2_promotion_kind, other->l2_promotion_kind) &&
           equal(oob_fill_kind, other->oob_fill_kind);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(tensor_dtype);
    hash_reduce(global_shape);
    hash_reduce(global_strides);
    hash_reduce(shared_shape);
    hash_reduce(shared_strides);
    hash_reduce(interleaved_kind);
    hash_reduce(swizzle_kind);
    hash_reduce(l2_promotion_kind);
    hash_reduce(oob_fill_kind);
  }

  static constexpr const char* _type_key = "TensorMapType";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  TVM_DECLARE_FINAL_OBJECT_INFO(TensorMapTypeNode, TypeNode);
};

/*!
 * \brief Managed reference to TensorMapTypeNode.
 * \sa TensorMapTypeNode
 */
class TensorMapType : public Type {
 public:
  TVM_DLL TensorMapType(
      runtime::DataType tensor_dtype = DataType::Void(), Array<PrimExpr> global_shape = {},
      Array<PrimExpr> global_strides = {}, Array<PrimExpr> shared_shape = {},
      Array<PrimExpr> shared_strides = {},
      TensorMapInterleavedKind interleaved_kind = TensorMapInterleavedKind::kNone,
      TensorMapSwizzleKind swizzle_kind = TensorMapSwizzleKind::kNone,
      TensorMapL2PromotionKind l2_promotion_kind = TensorMapL2PromotionKind::kNone,
      TensorMapOOBFillKind oob_fill_kind = TensorMapOOBFillKind::kNone);

  TVM_DEFINE_OBJECT_REF_METHODS_WITHOUT_DEFAULT_CONSTRUCTOR(TensorMapType, Type, TensorMapTypeNode);
};

}  // namespace tvm

#endif  // TVM_IR_TENSORMAP_TYPE_H_
