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

/**************** TrainiumLayout ****************/
TrainiumLayout::TrainiumLayout(ffi::Shape dimension_types, TileLayout combined_1d_layout) {
  auto n = make_object<TrainiumLayoutNode>();
  n->dimension_types = dimension_types;
  n->combined_1d_layout = combined_1d_layout;
  CHECK(n->VerifyWellFormed()) << "ValueError: The trainium layout is not well-formed";
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(TrainiumLayoutNode);

TVM_REGISTER_GLOBAL("tir.TrainiumLayout")
    .set_body_typed([](ffi::Shape dimension_types, TileLayout combined_1d_layout) {
      return TrainiumLayout(dimension_types, combined_1d_layout);
    });

constexpr int kMaxPartitionSize = 128;

bool TrainiumLayoutNode::CompatibleWithShape(const Array<PrimExpr>& shape) const {
  arith::Analyzer analyzer;
  PrimExpr prod_shape = ReduceMul(shape);
  PrimExpr prod_layout = GetSize() * GetPartitionSize();
  return analyzer.CanProveEqual(prod_shape, prod_layout);
}

bool TrainiumLayoutNode::VerifyWellFormed() const {
  if (!combined_1d_layout->VerifyWellFormed()) {
    return false;
  }
  if (dimension_types.size() != combined_1d_layout->data_iter_array.size()) {
    return false;
  }
  auto split_map = combined_1d_layout.GetSplitMap();
  std::vector<DataIterAttr> partition_data_iters;
  arith::Analyzer analyzer;
  for (size_t i = 0; i < dimension_types.size(); i++) {
    // Data Iter must be either partition or free
    if (dimension_types[i] != PhysicalDimensionType::kPartition &&
        dimension_types[i] != PhysicalDimensionType::kFree) {
      return false;
    }
    if (dimension_types[i] == PhysicalDimensionType::kPartition) {
      // partition dimension cannot be bound
      if (split_map.find(i) != split_map.end()) {
        return false;
      }
      if (!analyzer.CanProveEqual(combined_1d_layout->data_iter_array[i]->extent, 1)) {
        partition_data_iters.push_back(combined_1d_layout->data_iter_array[i]);
      }
    }
  }

  // check if the partition dimension is contiguous
  std::sort(partition_data_iters.begin(), partition_data_iters.end(),
            [](DataIterAttr a, DataIterAttr b) {
              auto stride_a = a->stride.as<IntImmNode>();
              auto stride_b = b->stride.as<IntImmNode>();
              ICHECK(stride_a && stride_b) << "ValueError: The stride must be a constant";
              return stride_a->value < stride_b->value;
            });
  for (int i = 0; i < static_cast<int>(partition_data_iters.size()) - 1; i++) {
    auto this_data_iter = partition_data_iters[i];
    auto next_data_iter = partition_data_iters[i + 1];
    if (!analyzer.CanProveEqual(this_data_iter->stride * this_data_iter->extent,
                                next_data_iter->stride)) {
      return false;
    }
  }
  if (!analyzer.CanProve(GetPartitionSize() <= kMaxPartitionSize)) {
    return false;
  }
  return true;
}

TileLayout Get1DLayout(TrainiumLayout layout, PhysicalDimensionType type) {
  Array<DataIterAttr> data_iter_array;
  Array<DeviceIterAttr> device_iter_array;
  std::unordered_map<int, int> index_map;
  for (size_t i = 0; i < layout->dimension_types.size(); i++) {
    if (layout->dimension_types[i] == type) {
      data_iter_array.push_back(layout->combined_1d_layout->data_iter_array[i]);
      index_map[i] = data_iter_array.size() - 1;
    }
  }
  for (size_t i = 0; i < layout->combined_1d_layout->device_iter_array.size(); i++) {
    auto device_iter = layout->combined_1d_layout->device_iter_array[i];
    if (device_iter.IsSplit()) {
      int bound = index_map[device_iter.GetIntBound()];
      if (index_map.find(bound) != index_map.end()) {
        device_iter_array.push_back(DeviceIterAttr::Split(device_iter->extent, index_map[bound]));
      }
    } else {
      device_iter_array.push_back(device_iter);
    }
  }
  return TileLayout(data_iter_array, device_iter_array, layout->combined_1d_layout->from,
                    layout->combined_1d_layout->to);
}

PrimExpr TrainiumLayoutNode::GetSize() const {
  // GetSize returns free dimension size
  TileLayout free_layout = Get1DLayout(GetRef<TrainiumLayout>(this), PhysicalDimensionType::kFree);
  return free_layout->GetSize();
}

PrimExpr TrainiumLayoutNode::GetCosize() const {
  // GetCosize returns the total stride of the free dimension
  TileLayout free_layout = Get1DLayout(GetRef<TrainiumLayout>(this), PhysicalDimensionType::kFree);
  return free_layout->GetCosize();
}

PrimExpr TrainiumLayoutNode::GetPartitionSize() const {
  TileLayout partition_layout =
      Get1DLayout(GetRef<TrainiumLayout>(this), PhysicalDimensionType::kPartition);
  return partition_layout->GetSize();
}

TVM_REGISTER_GLOBAL("tir.TrainiumLayoutGetPartitionSize").set_body_typed([](TrainiumLayout layout) {
  return layout->GetPartitionSize();
});

Array<PrimExpr> TrainiumLayoutNode::Apply(const Array<PrimExpr>& coord,
                                          const Array<PrimExpr>& shape) const {
  auto shape_prod = ReduceMul(shape);
  auto layout_sz = GetSize();
  arith::Analyzer analyzer;
  ICHECK(!analyzer.CanProveEqual(shape_prod, layout_sz))
      << "ValueError: The shape must match the layout size";
  auto pr = NormalizeTrainiumLayoutWithShape(GetRef<TrainiumLayout>(this), shape);
  auto grouped_layout = pr.first;
  auto seps = pr.second;
  if (!grouped_layout.defined()) {
    return TLayoutNode::Apply(coord, shape);
  }
  PrimExpr partition_coord = 0, free_coord = 0;
  auto split_map = grouped_layout->combined_1d_layout.GetSplitMap();
  for (int i = 0; i < static_cast<int>(shape.size()); i++) {
    PrimExpr cur = coord[i];
    for (int j = seps[i + 1] - 1; j >= seps[i]; j--) {
      if (split_map.find(j) != split_map.end()) {
        cur = floordiv(cur, grouped_layout->combined_1d_layout->data_iter_array[j]->extent);
        continue;
      }
      PrimExpr e = floormod(cur, grouped_layout->combined_1d_layout->data_iter_array[j]->extent) *
                   grouped_layout->combined_1d_layout->data_iter_array[j]->stride;
      if (grouped_layout->dimension_types[j] == PhysicalDimensionType::kPartition) {
        partition_coord += e;
      } else {
        free_coord += e;
      }
      cur = floordiv(cur, grouped_layout->combined_1d_layout->data_iter_array[j]->extent);
    }
  }
  return {partition_coord, free_coord};
}

Array<PrimExpr> TrainiumLayoutNode::Apply(const PrimExpr& coord) const {
  PrimExpr partition_coord = 0, free_coord = 0;
  PrimExpr cur = coord;
  auto split_map = combined_1d_layout.GetSplitMap();
  for (int i = static_cast<int>(combined_1d_layout->data_iter_array.size()) - 1; i >= 0; i--) {
    if (split_map.find(i) != split_map.end()) {
      cur = floordiv(cur, combined_1d_layout->data_iter_array[i]->extent);
      continue;
    }
    PrimExpr e = floormod(cur, combined_1d_layout->data_iter_array[i]->extent) *
                 combined_1d_layout->data_iter_array[i]->stride;
    if (dimension_types[i] == PhysicalDimensionType::kPartition) {
      partition_coord += e;
    } else {
      free_coord += e;
    }
    cur = floordiv(cur, combined_1d_layout->data_iter_array[i]->extent);
  }
  return {partition_coord, free_coord};
}
/**************** TrainiumPSUMLayout ****************/

Array<PrimExpr> TrainiumPSUMLayoutNode::Apply(const Array<PrimExpr>& coord,
                                              const Array<PrimExpr>& shape) const {
  auto indices = TrainiumLayoutNode::Apply(coord, shape);
  ICHECK_EQ(indices.size(), 2);
  return {floordiv(indices[1], kPSUMMaxElemPerBank), indices[0],
          floormod(indices[1], kPSUMMaxElemPerBank)};
}

bool TrainiumPSUMLayoutNode::VerifyWellFormed() const {
  if (!TrainiumLayoutNode::VerifyWellFormed()) {
    return false;
  }
  auto cosize = GetCosize();
  arith::Analyzer analyzer;
  if (!analyzer.CanProve(cosize <= kPSUMMaxElemPerBank * kPSUMBankNum)) {
    return false;
  }
  return true;
}

TVM_REGISTER_NODE_TYPE(TrainiumPSUMLayoutNode);

TrainiumPSUMLayout::TrainiumPSUMLayout(ffi::Shape dimension_types, TileLayout combined_1d_layout) {
  auto n = make_object<TrainiumPSUMLayoutNode>();
  n->dimension_types = dimension_types;
  n->combined_1d_layout = combined_1d_layout;
  CHECK(n->VerifyWellFormed()) << "ValueError: The trainium psum layout is not well-formed";
  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("tir.TrainiumPSUMLayout")
    .set_body_typed([](ffi::Shape dimension_types, TileLayout combined_1d_layout) {
      return TrainiumPSUMLayout(dimension_types, combined_1d_layout);
    });

}  // namespace tir
}  // namespace tvm
