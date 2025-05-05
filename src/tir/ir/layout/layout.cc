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
#include "utils.h"

namespace tvm {
namespace tir {

bool IsTrivialLayout(const TLayout& layout) {
  auto tile_layout = layout.as<TileLayoutNode>();
  if (tile_layout == nullptr) {
    return false;
  }
  if (!tile_layout->device_iter_array.empty()) {
    return false;
  }
  ICHECK(!tile_layout->data_iter_array.empty())
      << "InternalError: The data iter array should be defined";
  arith::Analyzer ana;
  PrimExpr expected_stride = 1;
  int data_iter_size = tile_layout->data_iter_array.size();
  for (int i = data_iter_size - 1; i >= 0; --i) {
    if (!ana.CanProveEqual(tile_layout->data_iter_array[i]->stride, expected_stride)) {
      return false;
    }
    expected_stride = expected_stride * tile_layout->data_iter_array[i]->extent;
  }
  return true;
}

TVM_REGISTER_GLOBAL("tir.IsTrivialLayout").set_body_typed(IsTrivialLayout);

/**************** TLayout ****************/
bool TLayoutNode::CompatibleWithShape(const Array<PrimExpr>& shape) const {
  return true;
}

Array<PrimExpr> TLayoutNode::Apply(const Array<PrimExpr>& coord,
                                   const Array<PrimExpr>& shape) const {
  ICHECK_EQ(coord.size(), shape.size())
      << "ValueError: The size of coord and shape should be equal";
  PrimExpr flattened_coord = 0;
  for (size_t i = 0; i < coord.size(); i++) {
    flattened_coord = flattened_coord * shape[i] + coord[i];
  }
  return Apply(flattened_coord);
}

TVM_REGISTER_GLOBAL("tir.TLayoutGetSize").set_body_typed([](TLayout layout) {
  return layout->GetSize();
});

TVM_REGISTER_GLOBAL("tir.TLayoutGetCosize").set_body_typed([](TLayout layout) {
  return layout->GetCosize();
});

TVM_REGISTER_GLOBAL("tir.TLayoutApply")
    .set_body_typed([](TLayout layout, Array<PrimExpr> coord, Array<PrimExpr> shape) {
      return layout->Apply(coord, shape);
    });

/**************** DataIterAttr ****************/
DataIterAttr::DataIterAttr(PrimExpr extent, PrimExpr stride) {
  auto n = make_object<DataIterAttrNode>();
  n->extent = extent;
  n->stride = stride;
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(DataIterAttrNode);

TVM_REGISTER_GLOBAL("tir.DataIterAttr").set_body_typed([](PrimExpr extent, PrimExpr stride) {
  return DataIterAttr(extent, stride);
});

/**************** DeviceIterAttr ****************/
DeviceIterAttr::DeviceIterAttr(PrimExpr extent, ScopeIdType type, Optional<PrimExpr> bound,
                               Optional<PrimExpr> owner) {
  auto n = make_object<DeviceIterAttrNode>();
  n->extent = extent;
  n->type = type;
  n->bound = bound;
  n->owner = owner;
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(DeviceIterAttrNode);

TVM_REGISTER_GLOBAL("tir.DeviceIterAttr")
    .set_body_typed([](PrimExpr extent, int type, Optional<PrimExpr> bound,
                       Optional<PrimExpr> owner) {
      return DeviceIterAttr(extent, static_cast<ScopeIdType>(type), bound, owner);
    });

DeviceIterAttr DeviceIterAttr::Replicate(PrimExpr extent) {
  return DeviceIterAttr(extent, kReplicate, std::nullopt, std::nullopt);
}

DeviceIterAttr DeviceIterAttr::Split(PrimExpr extent, PrimExpr bound) {
  return DeviceIterAttr(extent, kSplit, bound);
}

DeviceIterAttr DeviceIterAttr::Exclusive(PrimExpr extent, PrimExpr owner) {
  return DeviceIterAttr(extent, kExclusive, std::nullopt, owner);
}

bool DeviceIterAttr::IsReplicate() const { return this->get()->type == kReplicate; }

bool DeviceIterAttr::IsSplit() const { return this->get()->type == kSplit; }

bool DeviceIterAttr::IsExclusive() const { return this->get()->type == kExclusive; }

size_t DeviceIterAttr::GetIntBound() const {
  ICHECK(this->get()->bound.defined()) << "ValueError: The bound is not defined";
  auto bound = this->get()->bound.value();
  const auto* n = bound.as<IntImmNode>();
  ICHECK(n != nullptr) << "ValueError: The bound is not an integer";
  ICHECK(n->value >= 0) << "ValueError: The bound must be non-negative";
  return n->value;
}

}  // namespace tir
}  // namespace tvm
