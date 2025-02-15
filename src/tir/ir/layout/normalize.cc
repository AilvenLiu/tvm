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

/******** Normalization Utils ********/

TileLayout SimplifyTileLayout(TileLayout layout) {
  arith::Analyzer analyzer;
  Array<DataIterAttr> new_data_iter_array;
  auto split_map = layout.GetSplitMap();
  for (const auto& data_iter : layout->data_iter_array) {
    PrimExpr new_extent = analyzer.Simplify(data_iter->extent);
    PrimExpr new_stride = analyzer.Simplify(data_iter->stride);
    if (split_map.find(new_data_iter_array.size()) != split_map.end()) {
      new_stride = -1;
    }
    new_data_iter_array.push_back(DataIterAttr(new_extent, new_stride));
  }
  Array<DeviceIterAttr> new_device_iter_array;
  for (const auto& device_iter : layout->device_iter_array) {
    PrimExpr new_extent = analyzer.Simplify(device_iter->extent);
    new_device_iter_array.push_back(
        DeviceIterAttr(new_extent, device_iter->type, device_iter->bound, device_iter->owner));
  }
  return TileLayout(new_data_iter_array, new_device_iter_array, layout->from, layout->to);
}

void FuseNeighborShardAxis(TileLayout layout, std::unordered_set<int>* fused_data_iters,
                           std::unordered_set<int>* fused_device_iters) {
  // if both (i, j) and (i+1, j+1) exist in the split map, then we can fuse them
  auto split_map = layout.GetSplitMap();
  for (auto it = split_map.begin(); it != split_map.end(); it++) {
    auto other_it = split_map.find(it->first + 1);
    if (other_it != split_map.end() && other_it->second == it->second + 1) {
      fused_data_iters->insert(it->first);
      fused_device_iters->insert(it->second);
    }
  }
}

void FuseEquivDataIter(TileLayout layout, Optional<ShapeTuple> phys_dimension_type,
                       std::unordered_set<int>* fused_data_iters,
                       std::unordered_set<int>* fused_device_iters) {
  // if the stride of data(i) is equal to the extent of data(i+1) times the stride of data(i+1),
  // then fuse them
  arith::Analyzer analyzer;
  auto split_map = layout.GetSplitMap();
  for (int i = 0; i < static_cast<int>(layout->data_iter_array.size()) - 1; i++) {
    if (split_map.find(i) != split_map.end()) {
      continue;
    }
    auto this_data_iter = layout->data_iter_array[i];
    auto next_data_iter = layout->data_iter_array[i + 1];
    if (phys_dimension_type.defined()) {
      if (phys_dimension_type.value()[i] != phys_dimension_type.value()[i + 1]) {
        continue;
      }
    }
    PrimExpr this_stride = this_data_iter->stride;
    PrimExpr next_extent = next_data_iter->extent;
    PrimExpr next_stride = next_data_iter->stride;
    if (analyzer.CanProveEqual(this_stride, next_extent * next_stride)) {
      fused_data_iters->insert(i);
    }
  }
}

void EliminateUnitIter(TileLayout layout, std::unordered_set<int>* fused_data_iters,
                       std::unordered_set<int>* fused_device_iters) {
  // if the extent of data(i) is 1, then fuse it
  for (int i = 0; i < static_cast<int>(layout->data_iter_array.size()); i++) {
    if (is_one(layout->data_iter_array[i]->extent)) {
      fused_data_iters->insert(i);
    }
  }
  for (int i = 0; i < static_cast<int>(layout->device_iter_array.size()); i++) {
    if (is_one(layout->device_iter_array[i]->extent)) {
      fused_device_iters->insert(i);
    }
  }
}

TileLayout FuseIters(TileLayout layout, std::unordered_set<int> fused_data_iters,
                     std::unordered_set<int> fused_device_iters) {
  Array<DataIterAttr> new_data_iter_array;
  std::unordered_map<int, int> data_index_map;
  PrimExpr fused_extent = 1;
  for (int i = 0; i < static_cast<int>(layout->data_iter_array.size()); i++) {
    fused_extent = fused_extent * layout->data_iter_array[i]->extent;
    if (fused_data_iters.find(i) != fused_data_iters.end()) {
      continue;
    }
    DataIterAttr new_data_attr = DataIterAttr(fused_extent, layout->data_iter_array[i]->stride);
    new_data_iter_array.push_back(new_data_attr);
    fused_extent = 1;
    data_index_map[i] = new_data_iter_array.size() - 1;
  }
  if (new_data_iter_array.empty()) {
    new_data_iter_array.push_back(DataIterAttr(1, 1));
  }
  Array<DeviceIterAttr> new_device_iter_array;
  PrimExpr fused_device_extent = 1;
  for (int i = 0; i < static_cast<int>(layout->device_iter_array.size()); i++) {
    fused_device_extent = fused_device_extent * layout->device_iter_array[i]->extent;
    if (fused_device_iters.find(i) != fused_device_iters.end()) {
      continue;
    }
    DeviceIterAttr old_device_attr = layout->device_iter_array[i];
    int bound = -1;
    if (old_device_attr.IsSplit()) {
      bound = data_index_map[old_device_attr.GetIntBound()];
      new_device_iter_array.push_back(DeviceIterAttr::Split(fused_device_extent, bound));
    } else {
      new_device_iter_array.push_back(DeviceIterAttr(fused_device_extent, old_device_attr->type,
                                                     old_device_attr->bound,
                                                     old_device_attr->owner));
    }
    fused_device_extent = 1;
  }
  if (new_device_iter_array.empty() && !layout->device_iter_array.empty()) {
    new_device_iter_array.push_back(DeviceIterAttr::Replicate(1));
  }
  return TileLayout(new_data_iter_array, new_device_iter_array, layout->from, layout->to);
}

ShapeTuple EraseTupleElem(ShapeTuple tup, const std::unordered_set<int>& indices) {
  std::vector<int64_t> new_tup;
  for (size_t i = 0; i < tup.size(); i++) {
    if (indices.find(i) == indices.end()) {
      new_tup.push_back(tup[i]);
    }
  }
  return ShapeTuple(new_tup);
}

/******** Normalize TLayout ********/

TLayout TLayoutNode::Normalize() const { CHECK(false) << "Not implemented"; }

/******** Normalize TileLayout ********/

TLayout TileLayoutNode::Normalize() const {
  std::unordered_set<int> fused_data_iters;
  std::unordered_set<int> fused_device_iters;
  EliminateUnitIter(GetRef<TileLayout>(this), &fused_data_iters, &fused_device_iters);
  auto layout = FuseIters(GetRef<TileLayout>(this), fused_data_iters, fused_device_iters);
  layout = SimplifyTileLayout(layout);
  fused_data_iters.clear();
  fused_device_iters.clear();
  FuseNeighborShardAxis(layout, &fused_data_iters, &fused_device_iters);
  FuseEquivDataIter(layout, NullOpt, &fused_data_iters, &fused_device_iters);
  layout = FuseIters(layout, fused_data_iters, fused_device_iters);
  return SimplifyTileLayout(layout);
}

TVM_REGISTER_GLOBAL("tir.TileLayoutNormalize").set_body_typed([](TileLayout layout) {
  return layout->Normalize();
});

/******** Normalize SwizzleLayout ********/

TLayout SwizzleLayoutNode::Normalize() const { return GetRef<SwizzleLayout>(this); }

TVM_REGISTER_GLOBAL("tir.SwizzleLayoutNormalize").set_body_typed([](SwizzleLayout layout) {
  return layout->Normalize();
});

/******** Normalize ComposeLayout ********/

TLayout ComposeLayoutNode::Normalize() const {
  return ComposeLayout(layout_A, layout_B->Normalize().as<TileLayout>().value());
}

TVM_REGISTER_GLOBAL("tir.ComposeLayoutNormalize").set_body_typed([](ComposeLayout layout) {
  return layout->Normalize();
});

/******** Normalize TrainiumLayout ********/

TLayout TrainiumLayoutNode::Normalize() const {
  std::unordered_set<int> fused_data_iters;
  std::unordered_set<int> fused_device_iters;
  EliminateUnitIter(combined_1d_layout, &fused_data_iters, &fused_device_iters);
  auto combined_1d_layout =
      FuseIters(this->combined_1d_layout, fused_data_iters, fused_device_iters);
  auto dimension_types = EraseTupleElem(this->dimension_types, fused_data_iters);
  combined_1d_layout = SimplifyTileLayout(combined_1d_layout);
  fused_data_iters.clear();
  fused_device_iters.clear();
  FuseNeighborShardAxis(combined_1d_layout, &fused_data_iters, &fused_device_iters);
  FuseEquivDataIter(combined_1d_layout, dimension_types, &fused_data_iters, &fused_device_iters);
  combined_1d_layout = FuseIters(combined_1d_layout, fused_data_iters, fused_device_iters);
  dimension_types = EraseTupleElem(dimension_types, fused_data_iters);

  return TrainiumLayout(dimension_types, SimplifyTileLayout(combined_1d_layout));
}

TVM_REGISTER_GLOBAL("tir.TrainiumLayoutNormalize").set_body_typed([](TrainiumLayout layout) {
  return layout->Normalize();
});

std::pair<TrainiumLayout, std::vector<int64_t>> NormalizeTrainiumLayoutWithShape(
    TrainiumLayout layout, Array<PrimExpr> shape) {
  layout = (layout->Normalize()).as<TrainiumLayout>().value();
  auto combined_1d_layout = layout->combined_1d_layout;
  std::unordered_map<int, int> index_map;
  auto [grouped_combined_1d_layout, seps] =
      TileLayoutGroupByLogicalShape(combined_1d_layout, shape, &index_map);
  ICHECK(!seps.empty()) << "ValueError: The layout must be able to transform from logical view "
                           "shape only with split and reorder";
  std::vector<int64_t> new_dimension_types;
  for (size_t i = 0; i < grouped_combined_1d_layout->data_iter_array.size(); i++) {
    new_dimension_types.push_back(layout->dimension_types[index_map[i]]);
  }
  return {TrainiumLayout(new_dimension_types, SimplifyTileLayout(grouped_combined_1d_layout)),
          seps};
}

TVM_REGISTER_GLOBAL("tir.NormalizeTrainiumLayoutWithShape")
    .set_body_typed([](TrainiumLayout layout, Array<PrimExpr> shape) {
      auto pr = NormalizeTrainiumLayoutWithShape(layout, shape);
      return Array<ObjectRef>{pr.first, ShapeTuple(pr.second)};
    });

TVM_REGISTER_GLOBAL("tir.NormalizeTileLayoutWithShape")
    .set_body_typed([](TileLayout layout, Array<PrimExpr> shape) {
      auto pr = TileLayoutGroupByLogicalShape(layout, shape, nullptr);
      return Array<ObjectRef>{pr.first, ShapeTuple(pr.second)};
    });

}  // namespace tir
}  // namespace tvm
