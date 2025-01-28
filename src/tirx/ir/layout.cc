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
#include <tvm/arith/analyzer.h>
#include <tvm/node/serialization.h>
#include <tvm/node/structural_equal.h>
#include <tvm/tirx/layout.h>
#include <tvm/tirx/op.h>

namespace tvm {
namespace tirx {

/**************** Utils ****************/
PrimExpr ReduceMul(Array<PrimExpr> values) {
  PrimExpr result = values[0];
  for (size_t i = 1; i < values.size(); i++) {
    result = result * values[i];
  }
  return result;
}

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

TVM_REGISTER_GLOBAL("tirx.IsTrivialLayout").set_body_typed(IsTrivialLayout);

/**************** TLayout ****************/
bool TLayoutNode::CompatibleWithShape(const Array<PrimExpr>& shape) const {
  arith::Analyzer analyzer;
  return analyzer.CanProveEqual(FloorMod(ReduceMul(shape), this->GetSize()), 0);
}

PrimExpr TLayoutNode::Apply(const Array<PrimExpr>& coord, const Array<PrimExpr>& shape) const {
  ICHECK_EQ(coord.size(), shape.size())
      << "ValueError: The size of coord and shape should be equal";
  PrimExpr flattened_coord = 0;
  for (size_t i = 0; i < coord.size(); i++) {
    flattened_coord = flattened_coord * shape[i] + coord[i];
  }
  return Apply(flattened_coord);
}

TVM_REGISTER_GLOBAL("tirx.TLayoutGetSize").set_body_typed([](TLayout layout) {
  return layout->GetSize();
});

TVM_REGISTER_GLOBAL("tirx.TLayoutGetCosize").set_body_typed([](TLayout layout) {
  return layout->GetCosize();
});

TVM_REGISTER_GLOBAL("tirx.TLayoutApply")
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

TVM_REGISTER_GLOBAL("tirx.DataIterAttr").set_body_typed([](PrimExpr extent, PrimExpr stride) {
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

TVM_REGISTER_GLOBAL("tirx.DeviceIterAttr")
    .set_body_typed([](PrimExpr extent, int type, Optional<PrimExpr> bound,
                       Optional<PrimExpr> owner) {
      return DeviceIterAttr(extent, static_cast<ScopeIdType>(type), bound, owner);
    });

DeviceIterAttr DeviceIterAttr::Replicate(PrimExpr extent) {
  return DeviceIterAttr(extent, kReplicate, NullOpt, NullOpt);
}

DeviceIterAttr DeviceIterAttr::Split(PrimExpr extent, PrimExpr bound) {
  return DeviceIterAttr(extent, kSplit, bound);
}

DeviceIterAttr DeviceIterAttr::Exclusive(PrimExpr extent, PrimExpr owner) {
  return DeviceIterAttr(extent, kExclusive, NullOpt, owner);
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

PrimExpr TileLayoutNode::GetSize() const {
  PrimExpr result = data_iter_array[0]->extent;
  for (size_t i = 1; i < data_iter_array.size(); i++) {
    result = result * data_iter_array[i]->extent;
  }
  return result;
}

PrimExpr TileLayoutNode::GetCosize() const {
  auto split_map = GetRef<TileLayout>(this).GetSplitMap();
  arith::Analyzer analyzer;
  PrimExpr cosize = 1;
  for (size_t i = 0; i < data_iter_array.size(); ++i) {
    if (split_map.find(i) != split_map.end()) {
      continue;
    }
    // Check if the coefficient is non-negative
    ICHECK(analyzer.CanProveGreaterEqual(data_iter_array[i]->stride, 0))
        << "ValueError: GetCosize cannot handle case where the coefficient of a non-bound leaf is "
           "negative";
    cosize += data_iter_array[i]->stride * (data_iter_array[i]->extent - 1);
  }
  return cosize;
}

bool TileLayoutNode::VerifyWellFormed() const {
  arith::Analyzer analyzer;
  // Verify the split map
  const auto& split_map = GetRef<TileLayout>(this).GetSplitMap();
  bool result = true;
  for (const auto& kv : split_map) {
    // The extents of the data and device iter array must be the same
    result &= analyzer.CanProveEqual(data_iter_array[kv.first]->extent,
                                     device_iter_array[kv.second]->extent);
  }
  return result;
}

PrimExpr TileLayoutNode::Apply(const PrimExpr& coord) const {
  arith::Analyzer analyzer;
  PrimExpr input = analyzer.Simplify(coord);
  const auto& size = GetSize();
  const auto& cosize = GetCosize();
  auto inner = analyzer.Simplify(floormod(input, size));
  const auto& outer = analyzer.Simplify(floordiv(input, size));

  auto split_map = GetRef<TileLayout>(this).GetSplitMap();
  PrimExpr result = 0;
  for (int i = static_cast<int>(data_iter_array.size()) - 1; i >= 0; i--) {
    if (split_map.find(i) != split_map.end()) {
      continue;
    }
    result += data_iter_array[i]->stride * floormod(inner, data_iter_array[i]->extent);
    inner = floordiv(inner, data_iter_array[i]->extent);
  }
  return analyzer.Simplify(outer * cosize + result);
}

/**************** TileLayout ****************/
TileLayout::TileLayout(Array<DataIterAttr> data_iter_array, Array<DeviceIterAttr> device_iter_array,
                       Optional<ExecScope> from, Optional<ExecScope> to) {
  auto n = make_object<TileLayoutNode>();
  n->data_iter_array = std::move(data_iter_array);
  n->device_iter_array = std::move(device_iter_array);
  n->from = std::move(from);
  n->to = std::move(to);
  if (n->from.defined()) {
    ICHECK(n->to.defined()) << "ValueError: The to scope must be defined if the from scope is";
    ICHECK(n->to.value().Higher(n->from.value()))
        << "ValueError: The from scope must be higher than the to scope";
  }
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(TileLayoutNode);

TVM_REGISTER_GLOBAL("tirx.TileLayout")
    .set_body_typed([](Array<DataIterAttr> data_iter_array, Array<DeviceIterAttr> device_iter_array,
                       Optional<ExecScope> from, Optional<ExecScope> to) {
      return TileLayout(data_iter_array, device_iter_array, from, to);
    });

TileLayout::SplitMap TileLayout::GetSplitMap() const {
  auto* n = operator->();
  if (n->device_iter_array.empty()) {
    // No device array is defined, return an empty map
    return {};
  }
  SplitMap split_map;
  for (size_t i = 0; i < n->device_iter_array.size(); i++) {
    auto attr = n->device_iter_array[i];
    if (attr.IsSplit()) {
      size_t bound = attr.GetIntBound();
      ICHECK(bound < n->data_iter_array.size())
          << "ValueError: The bound of the split attribute is out of range";
      auto it = split_map.find(bound);
      ICHECK(it == split_map.end()) << "ValueError: Duplicate split bound for the same data iter";
      split_map[bound] = i;
    }
  }
  return std::move(split_map);
}

/******** Normalization ********/

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

void FuseEquivDataIter(TileLayout layout, std::unordered_set<int>* fused_data_iters,
                       std::unordered_set<int>* fused_device_iters) {
  // if the stride of data(i) is equal to the extent of data(i+1) times the stride of data(i+1),
  // then fuse them
  arith::Analyzer analyzer;
  auto split_map = layout.GetSplitMap();
  for (int i = 0; i < static_cast<int>(layout->data_iter_array.size()) - 1; i++) {
    if (split_map.find(i) != split_map.end()) {
      continue;
    }
    PrimExpr this_stride = layout->data_iter_array[i]->stride;
    PrimExpr next_extent = layout->data_iter_array[i + 1]->extent;
    PrimExpr next_stride = layout->data_iter_array[i + 1]->stride;
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

TileLayout NormalizeTileLayout(TileLayout layout) {
  std::unordered_set<int> fused_data_iters;
  std::unordered_set<int> fused_device_iters;
  EliminateUnitIter(layout, &fused_data_iters, &fused_device_iters);
  layout = FuseIters(layout, fused_data_iters, fused_device_iters);
  layout = SimplifyTileLayout(layout);
  fused_data_iters.clear();
  fused_device_iters.clear();
  FuseNeighborShardAxis(layout, &fused_data_iters, &fused_device_iters);
  FuseEquivDataIter(layout, &fused_data_iters, &fused_device_iters);
  layout = FuseIters(layout, fused_data_iters, fused_device_iters);
  return SimplifyTileLayout(layout);
}

TVM_REGISTER_GLOBAL("tirx.NormalizeTileLayout").set_body_typed(NormalizeTileLayout);

/******** Tile ********/

std::pair<TileLayout, std::vector<int>> TileLayoutGroupByLogicalShape(TileLayout layout,
                                                                      Array<PrimExpr> shape) {
  size_t shape_match_dim = 0;
  arith::Analyzer analyzer;
  PrimExpr prod = 1;
  std::vector<int> seps;
  seps.push_back(0);
  Array<DataIterAttr> new_data_iter_array;
  Array<DeviceIterAttr> new_device_iter_array;
  auto split_map = layout.GetSplitMap();
  std::unordered_map<int, std::vector<int>> split_device_iter;
  for (int i = 0; i < static_cast<int>(layout->data_iter_array.size()); i++) {
    auto split_map_it = split_map.find(i);
    prod *= layout->data_iter_array[i]->extent;
    PrimExpr cur_extent = layout->data_iter_array[i]->extent;
    PrimExpr stride = layout->data_iter_array[i]->stride;
    while (shape_match_dim < shape.size() &&
           analyzer.CanProveEqual(floormod(prod, shape[shape_match_dim]), 0)) {
      if (!analyzer.CanProveEqual(floormod(cur_extent, floordiv(prod, shape[shape_match_dim])),
                                  0)) {
        return std::make_pair(TileLayout(), std::vector<int>());
      }
      PrimExpr split_extent = floordiv(cur_extent, floordiv(prod, shape[shape_match_dim]));
      new_data_iter_array.push_back(
          DataIterAttr(split_extent, stride * floordiv(cur_extent, split_extent)));
      cur_extent = floordiv(prod, shape[shape_match_dim]);
      if (split_map_it != split_map.end()) {
        split_device_iter[split_map_it->second].push_back(new_data_iter_array.size() - 1);
      }
      prod = floordiv(prod, shape[shape_match_dim]);
      shape_match_dim++;
      seps.push_back(new_data_iter_array.size());
    }
    if (!analyzer.CanProveEqual(cur_extent, 1)) {
      new_data_iter_array.push_back(DataIterAttr(cur_extent, stride));
      if (split_map_it != split_map.end()) {
        split_device_iter[split_map_it->second].push_back(new_data_iter_array.size() - 1);
      }
    }
  }
  if (shape_match_dim != shape.size()) {
    return std::make_pair(TileLayout(), std::vector<int>());
  }
  for (int i = 0; i < static_cast<int>(layout->device_iter_array.size()); i++) {
    if (!layout->device_iter_array[i].IsSplit()) {
      new_device_iter_array.push_back(layout->device_iter_array[i]);
      continue;
    }
    auto split_device_iter_it = split_device_iter.find(i);
    ICHECK(split_device_iter_it != split_device_iter.end())
        << "ValueError: The split device iter must have a corresponding data iter";
    for (auto data_iter_index : split_device_iter_it->second) {
      new_device_iter_array.push_back(
          DeviceIterAttr::Split(new_data_iter_array[data_iter_index]->extent, data_iter_index));
    }
  }
  ICHECK(seps.size() == shape.size() + 1);
  return std::make_pair(
      TileLayout(new_data_iter_array, new_device_iter_array, layout->from, layout->to), seps);
}

TileLayout Tile(TileLayout outer, TileLayout inner, Array<PrimExpr> outer_shape,
                Array<PrimExpr> inner_shape) {
  outer = NormalizeTileLayout(outer);
  inner = NormalizeTileLayout(inner);

  ICHECK_EQ(outer_shape.size(), inner_shape.size())
      << "ValueError: The dimension of logical view shapes must be the same";
  auto [grouped_outer, outer_seps] = TileLayoutGroupByLogicalShape(outer, outer_shape);
  auto [grouped_inner, inner_seps] = TileLayoutGroupByLogicalShape(inner, inner_shape);
  outer = grouped_outer;
  inner = grouped_inner;

  ICHECK(!outer_seps.empty()) << "ValueError: The outer layout must be able to transform from "
                                 "logical view shape only with split and reorder";
  ICHECK(!inner_seps.empty())
      << "ValueError: The inner layout must be able to transform from logical view shape only with "
         "split and reorder";
  ICHECK(outer->device_iter_array.empty()) << "ValueError: The outer layout must not have device "
                                              "iterators";
  int shape_dim = static_cast<int>(outer_shape.size());

  // Get the stride in the inner layout
  arith::Analyzer analyzer;
  PrimExpr inner_stride = inner->GetCosize();
  Array<DataIterAttr> fused_data_iter_array;
  std::unordered_map<int, int> index_map;
  auto inner_split_map = inner.GetSplitMap();
  for (int i = 0; i < shape_dim; i++) {
    for (int j = outer_seps[i]; j < outer_seps[i + 1]; j++) {
      PrimExpr new_stride = inner_stride * outer->data_iter_array[j]->stride;
      fused_data_iter_array.push_back(DataIterAttr(outer->data_iter_array[j]->extent, new_stride));
    }
    for (int j = inner_seps[i]; j < inner_seps[i + 1]; j++) {
      fused_data_iter_array.push_back(inner->data_iter_array[j]);
      if (inner_split_map.find(j) != inner_split_map.end()) {
        index_map[inner_split_map[j]] = fused_data_iter_array.size() - 1;
      }
    }
  }
  Array<DeviceIterAttr> new_device_iter_array;
  for (int i = 0; i < static_cast<int>(inner->device_iter_array.size()); i++) {
    if (inner->device_iter_array[i].IsSplit()) {
      int bound = index_map[i];
      new_device_iter_array.push_back(
          DeviceIterAttr::Split(inner->device_iter_array[i]->extent, bound));
    } else {
      new_device_iter_array.push_back(inner->device_iter_array[i]);
    }
  }

  return SimplifyTileLayout(
      TileLayout(fused_data_iter_array, new_device_iter_array, inner->from, inner->to));
}

TVM_REGISTER_GLOBAL("tirx.TileLayoutTile")
    .set_body_typed([](TileLayout outer, TileLayout inner, Array<PrimExpr> outer_shape,
                       Array<PrimExpr> inner_shape) {
      return Tile(outer, inner, outer_shape, inner_shape);
    });

/******** Tile - Check Inner Layout ********/

bool CheckTileLayoutDeviceExtentEqual(TileLayout layout1, TileLayout layout2) {
  // check if two layouts have the same device extent
  if (layout1->device_iter_array.size() != layout2->device_iter_array.size()) {
    return false;
  }
  arith::Analyzer analyzer;
  for (size_t i = 0; i < layout1->device_iter_array.size(); i++) {
    auto extent1 = layout1->device_iter_array[i]->extent;
    auto extent2 = layout2->device_iter_array[i]->extent;
    if (!analyzer.CanProveEqual(extent1, extent2)) {
      return false;
    }
  }
  return true;
}

Array<PrimExpr> TileShape(Array<PrimExpr> shape, Array<PrimExpr> factor, bool is_inner) {
  int tiled_shape_dim = static_cast<int>(shape.size());
  int factor_shape_dim = static_cast<int>(factor.size());
  ICHECK(tiled_shape_dim == factor_shape_dim)
      << "ValueError: The dimension of logical view shapes must be the same";
  for (int i = 0; i < tiled_shape_dim; i++) {
    ICHECK(arith::Analyzer().CanProveEqual(floormod(shape[i], factor[i]), 0))
        << "ValueError: The shape must be divisible by the factor";
  }
  Array<PrimExpr> new_shape;
  if (is_inner) {
    for (int i = 0; i < factor_shape_dim; i++) {
      new_shape.push_back(floordiv(shape[i], factor[i]));
      new_shape.push_back(factor[i]);
    }
  } else {
    for (int i = 0; i < factor_shape_dim; i++) {
      new_shape.push_back(factor[i]);
      new_shape.push_back(floordiv(shape[i], factor[i]));
    }
  }
  return new_shape;
}

std::vector<int> GetEvenSeps(std::vector<int> seps) {
  std::vector<int> even_seps;
  for (int i = 0; i < static_cast<int>(seps.size()); i++) {
    if (i % 2 == 0) {
      even_seps.push_back(seps[i]);
    }
  }
  return even_seps;
}

bool IsTileLayout_Inner(TileLayout tiled_layout, TileLayout layout, Array<PrimExpr> tiled_shape,
                        Array<PrimExpr> inner_shape) {
  tiled_layout = NormalizeTileLayout(tiled_layout);
  layout = NormalizeTileLayout(layout);
  int tiled_shape_dim = static_cast<int>(tiled_shape.size());
  int inner_shape_dim = static_cast<int>(inner_shape.size());
  if (tiled_shape_dim != inner_shape_dim) {
    return false;
  }
  auto factored_tiled_shape = TileShape(tiled_shape, inner_shape, true);
  auto [grouped_tiled_layout, tiled_seps] =
      TileLayoutGroupByLogicalShape(tiled_layout, factored_tiled_shape);
  tiled_seps = GetEvenSeps(tiled_seps);
  auto [grouped_layout, inner_seps] = TileLayoutGroupByLogicalShape(layout, inner_shape);
  tiled_layout = grouped_tiled_layout;
  layout = grouped_layout;

  // 1. check whether device iter extent are same
  if (!CheckTileLayoutDeviceExtentEqual(tiled_layout, layout)) {
    return false;
  }
  // 2. check data iter match
  ICHECK(!tiled_seps.empty()) << "ValueError: The tiled layout must be able to transform from "
                                 "logical view shape only with split and reorder";
  ICHECK(!inner_seps.empty())
      << "ValueError: The inner layout must be able to transform from logical view shape only with "
         "split and reorder";
  arith::Analyzer analyzer;
  std::unordered_map<int, int> index_map;
  auto inner_split_map = layout.GetSplitMap();
  for (int i = 0; i < tiled_shape_dim; i++) {
    int inner_sep_range = inner_seps[i + 1] - inner_seps[i];
    int tiled_sep_range = tiled_seps[i + 1] - tiled_seps[i];
    if (inner_sep_range > tiled_sep_range) {
      return false;
    }
    for (int j = 0; j < inner_sep_range; j++) {
      DataIterAttr inner_attr = layout->data_iter_array[inner_seps[i] + j];
      DataIterAttr tiled_attr =
          tiled_layout->data_iter_array[tiled_seps[i + 1] - inner_sep_range + j];
      index_map[inner_seps[i] + j] = tiled_seps[i + 1] - inner_sep_range + j;
      // extent and stride of inner layout and the corresponding part in tiled layout must be the
      // same
      if (!analyzer.CanProveEqual(inner_attr->extent, tiled_attr->extent) ||
          (!analyzer.CanProveEqual(inner_attr->stride, tiled_attr->stride) &&
           inner_split_map.find(inner_seps[i] + j) == inner_split_map.end() &&
           !analyzer.CanProveEqual(inner_attr->extent, 1))) {
        return false;
      }
    }
  }
  // 3. check split map match
  auto tiled_split_map = tiled_layout.GetSplitMap();
  if (tiled_split_map.size() != inner_split_map.size()) {
    return false;
  }
  for (const auto& kv : inner_split_map) {
    auto it = tiled_split_map.find(index_map[kv.first]);
    if (it == tiled_split_map.end() || it->second != kv.second) {
      return false;
    }
  }
  return true;
}

TVM_REGISTER_GLOBAL("tirx.IsTileLayout_Inner")
    .set_body_typed([](TileLayout tiled_layout, TileLayout layout, Array<PrimExpr> tiled_shape,
                       Array<PrimExpr> inner_shape) {
      return IsTileLayout_Inner(tiled_layout, layout, tiled_shape, inner_shape);
    });

/******** Tile - Check Outer Layout ********/

bool IsTileLayout_Outer(TileLayout tiled_layout, TileLayout layout, Array<PrimExpr> tiled_shape,
                        Array<PrimExpr> outer_shape) {
  tiled_layout = NormalizeTileLayout(tiled_layout);
  layout = NormalizeTileLayout(layout);
  int tiled_shape_dim = static_cast<int>(tiled_shape.size());
  int outer_shape_dim = static_cast<int>(outer_shape.size());
  ICHECK(tiled_shape_dim == outer_shape_dim)
      << "ValueError: The dimension of logical view shapes must be the same";
  for (int i = 0; i < tiled_shape_dim; i++) {
    if (!arith::Analyzer().CanProveEqual(floormod(tiled_shape[i], outer_shape[i]), 0)) {
      return false;
    }
  }

  // 1. outer layout must not have device iter
  if (!layout->device_iter_array.empty()) {
    return false;
  }
  // 2. check data iter match
  auto factored_tiled_shape = TileShape(tiled_shape, outer_shape, false);
  auto [grouped_tiled_layout, tiled_seps] =
      TileLayoutGroupByLogicalShape(tiled_layout, factored_tiled_shape);
  tiled_seps = GetEvenSeps(tiled_seps);
  auto [grouped_layout, outer_seps] = TileLayoutGroupByLogicalShape(layout, outer_shape);
  tiled_layout = grouped_tiled_layout;
  layout = grouped_layout;
  ICHECK(!tiled_seps.empty()) << "ValueError: The tiled layout must be able to transform from "
                                 "logical view shape only with split and reorder";
  ICHECK(!outer_seps.empty())
      << "ValueError: The outer layout must be able to transform from logical view shape only with "
         "split and reorder";
  arith::Analyzer analyzer;
  Array<DataIterAttr> inner_data_iter_array;
  auto tiled_split_map = tiled_layout.GetSplitMap();

  for (int i = 0; i < tiled_shape_dim; i++) {
    int outer_sep_range = outer_seps[i + 1] - outer_seps[i];
    int tiled_sep_range = tiled_seps[i + 1] - tiled_seps[i];
    if (outer_sep_range > tiled_sep_range) {
      return false;
    }
    for (int j = 0; j < outer_sep_range; j++) {
      DataIterAttr outer_attr = layout->data_iter_array[outer_seps[i] + j];
      DataIterAttr tiled_attr = tiled_layout->data_iter_array[tiled_seps[i] + j];
      // data iter in outer layout cannot be bound
      if (tiled_split_map.find(tiled_seps[i] + j) != tiled_split_map.end()) {
        return false;
      }
      // extent of outer layout and the corresponding part in tiled layout must be the same
      if (!analyzer.CanProveEqual(outer_attr->extent, tiled_attr->extent)) {
        return false;
      }
    }
    for (int j = tiled_seps[i] + outer_sep_range; j < tiled_seps[i + 1]; j++) {
      if (tiled_split_map.find(j) != tiled_split_map.end()) {
        continue;
      }
      inner_data_iter_array.push_back(tiled_layout->data_iter_array[j]);
    }
  }
  TileLayout inner_layout = TileLayout(inner_data_iter_array, {});
  auto inner_cosize = inner_layout->GetCosize();
  for (int i = 0; i < tiled_shape_dim; i++) {
    int outer_sep_range = outer_seps[i + 1] - outer_seps[i];
    for (int j = 0; j < outer_sep_range; j++) {
      DataIterAttr outer_attr = layout->data_iter_array[outer_seps[i] + j];
      DataIterAttr tiled_attr = tiled_layout->data_iter_array[tiled_seps[i] + j];
      // outer layout's stride will be multipled with inner cosize after tiling
      if (!analyzer.CanProveEqual(outer_attr->stride * inner_cosize, tiled_attr->stride)) {
        return false;
      }
    }
  }
  return true;
}

TVM_REGISTER_GLOBAL("tirx.IsTileLayout_Outer")
    .set_body_typed([](TileLayout tiled_layout, TileLayout layout, Array<PrimExpr> tiled_shape,
                       Array<PrimExpr> outer_shape) {
      return IsTileLayout_Outer(tiled_layout, layout, tiled_shape, outer_shape);
    });

/******* Vector Length ********/
PrimExpr ComputeGCD(PrimExpr a, PrimExpr b, arith::Analyzer* ana) {
  while (!ana->CanProveEqual(b, 0)) {
    PrimExpr temp = b;
    b = floormod(a, b);
    a = temp;
  }
  return a;
}

std::pair<int, int> GetMajorDimension(const Array<PrimExpr>& strides) {
  arith::Analyzer analyzer;
  int major_dim = 0;
  PrimExpr min_stride = strides[0];
  PrimExpr argmin_stride = strides[0];
  for (size_t i = 0; i < strides.size(); ++i) {
    if (as_const_int(strides[i])[0] < as_const_int(min_stride)[0]) {
      major_dim = static_cast<int>(i);
      min_stride = strides[i];
    }
  }
  int min_stride_int = as_const_int(min_stride) ? static_cast<int>(*as_const_int(min_stride)) : 0;
  return {(strides.size() - major_dim) - 1, min_stride_int};
}

std::tuple<int, int, PrimExpr> ExtractLayout(TileLayout layout) {
  arith::Analyzer analyzer;
  // Get the data dimensions and strides from the layout
  Array<PrimExpr> strides;
  for (auto& iter : layout->data_iter_array) {
    strides.push_back(iter->stride);
  }

  // Get the major dimension (dimension with the largest stride)
  auto [major_dim, min_stride] = GetMajorDimension(strides);
  return {major_dim, min_stride, layout->data_iter_array[strides.size() - major_dim - 1]->extent};
}

int Vec_Len(TileLayout Ls, TileLayout Ld) {
  // Constants initialization
  arith::Analyzer analyzer;
  std::vector<int> optimal_vec_len = {1, 2, 4, 8};
  Ls = NormalizeTileLayout(Ls);
  Ld = NormalizeTileLayout(Ld);

  PrimExpr Ls_size = Ls->GetSize();
  PrimExpr Ld_size = Ld->GetSize();
  auto [major_s, min_stride_s, argmin_s] = ExtractLayout(Ls);
  auto [major_d, min_stride_d, argmin_d] = ExtractLayout(Ld);

  // Check if normalized layouts are structurally equal
  if (analyzer.CanProveEqual(Ls_size, Ld_size)) {
    if ((min_stride_s != 1) || (min_stride_d != 1) || (major_s != major_d)) {
      // Case 1 & 2: Layouts are not contiguous or min strides don't match; different major-dim
      return 1;
    } else {
      // Case 3: the same dim-major, and the same stride, return major dimension in terms of
      // optimal vector length based on extent
      return *std::lower_bound(
          optimal_vec_len.begin(), optimal_vec_len.end() - 1,
          static_cast<int>(as_const_int(ComputeGCD(argmin_s, argmin_d, &analyzer))[0]));
    }
  } else {
    // Case4: Structure doesn't match, then return the minimum of GCD stride, normally 1
    LOG(FATAL) << "Source and Destination extents don't match!";
  }
}

TVM_REGISTER_GLOBAL("tirx.Vec_Len").set_body_typed([](TileLayout tiled_layout, TileLayout layout) {
  return Vec_Len(tiled_layout, layout);
});

/******** Shard ********/
std::vector<DeviceIterAttr> ParseStrategy(String strategy, Array<PrimExpr> mesh) {
  std::vector<DeviceIterAttr> attrs;
  size_t cur = 0;
  auto f_parse_number_with_bracket = [&cur, &strategy]() {
    std::string number;
    while ((++cur) < strategy.size() && isdigit(strategy.at(cur))) {
      number.push_back(strategy.at(cur));
    }
    ICHECK_GT(number.size(), 0) << "ValueError: Invalid strategy " << strategy;
    return atoi(number.c_str());
  };
  size_t mesh_dim = 0;
  for (; cur < strategy.size() && mesh_dim < mesh.size();) {
    if (strategy.at(cur) == 'S') {
      attrs.push_back(DeviceIterAttr::Split(mesh[mesh_dim++], f_parse_number_with_bracket()));
    } else if (strategy.at(cur) == 'E') {
      attrs.push_back(DeviceIterAttr::Exclusive(mesh[mesh_dim++], f_parse_number_with_bracket()));
    } else if (strategy.at(cur) == 'R') {
      attrs.push_back(DeviceIterAttr::Replicate(mesh[mesh_dim++]));
      ++cur;
    } else {
      LOG(FATAL) << "ValueError: Invalid strategy " << strategy;
    }
  }
  return std::move(attrs);
}

TileLayout Shard(Array<PrimExpr> shape, Array<PrimExpr> mesh, String strategy, TileLayout inner,
                 ExecScope from, ExecScope to) {
  // Parse the strategy
  auto attrs = ParseStrategy(strategy, mesh);
  ICHECK_EQ(attrs.size(), mesh.size())
      << "ValueError: The number of attributes must be the same as the number of mesh leaves";
  Array<PrimExpr> inner_shape(shape);
  for (int i = 0; i < static_cast<int>(mesh.size()); i++) {
    if (attrs[i].IsSplit()) {
      auto bound = attrs[i].GetIntBound();
      inner_shape.Set(bound, floordiv(inner_shape[bound], mesh[i]));
    }
  }
  auto [grouped_inner, inner_seps] = TileLayoutGroupByLogicalShape(inner, inner_shape);
  inner = grouped_inner;
  ICHECK(!inner_seps.empty())
      << "ValueError: The inner layout must be able to transform from logical view shape only with "
         "split and reorder";

  Array<DataIterAttr> new_data_iter_array;
  std::unordered_map<int, int> index_map;
  auto split_map = inner.GetSplitMap();
  for (int i = 0; i < static_cast<int>(inner_seps.size()) - 1; i++) {
    for (size_t j = 0; j < attrs.size(); j++) {
      auto attr = attrs[j];
      if (attr.IsSplit() && static_cast<int>(attr.GetIntBound()) == i) {
        new_data_iter_array.push_back(DataIterAttr(attr->extent, -1));
        attrs[j] =
            DeviceIterAttr::Split(attr->extent, static_cast<int>(new_data_iter_array.size()) - 1);
      }
    }
    for (int j = inner_seps[i]; j < inner_seps[i + 1]; j++) {
      new_data_iter_array.push_back(inner->data_iter_array[j]);
      if (split_map.find(j) != split_map.end()) {
        index_map[j] = new_data_iter_array.size() - 1;
      }
    }
  }
  Array<DeviceIterAttr> new_device_iter_array;
  new_device_iter_array.insert(new_device_iter_array.end(), attrs.begin(), attrs.end());
  for (size_t i = 0; i < inner->device_iter_array.size(); i++) {
    auto device_iter = inner->device_iter_array[i];
    if (device_iter.IsSplit()) {
      new_device_iter_array.push_back(
          DeviceIterAttr::Split(device_iter->extent, index_map[device_iter.GetIntBound()]));
    } else {
      new_device_iter_array.push_back(device_iter);
    }
  }
  // Construct the from and to scopes
  if (!inner->device_iter_array.empty()) {
    ICHECK(inner->to.defined()) << "ValueError: The inner layout must have the to scope";
    ICHECK(inner->to.value().Is((from)))
        << "ValueError: The from scope of the inner layout must be the same as the to scope of "
           "the outer layout";
  }

  return TileLayout(new_data_iter_array, new_device_iter_array,
                    !inner->device_iter_array.empty() ? inner->from : from, to);
}

TVM_REGISTER_GLOBAL("tirx.TileLayoutShard")
    .set_body_typed([](Array<PrimExpr> shape, Array<PrimExpr> mesh, String strategy,
                       TileLayout inner, ExecScope from,
                       ExecScope to) { return Shard(shape, mesh, strategy, inner, from, to); });

/**************** SwizzleLayout ****************/
SwizzleLayout::SwizzleLayout(int per_element, int swizzle_len, int atom_len, bool swizzle_inner) {
  auto n = make_object<SwizzleLayoutNode>();
  n->per_element = per_element;
  n->swizzle_len = swizzle_len;
  n->atom_len = atom_len;
  n->swizzle_inner = swizzle_inner;
  ICHECK(n->VerifyWellFormed()) << "ValueError: The swizzle layout is not well-formed";
  int swizzle_mask = (1 << swizzle_len) - 1;
  n->inner_mask = swizzle_mask;
  n->outer_mask = swizzle_mask << atom_len;
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(SwizzleLayoutNode);

TVM_REGISTER_GLOBAL("tirx.SwizzleLayout")
    .set_body_typed([](int per_element, int swizzle_len, int atom_len, bool swizzle_inner) {
      return SwizzleLayout(per_element, swizzle_len, atom_len, swizzle_inner);
    });

bool SwizzleLayoutNode::VerifyWellFormed() const {
  return per_element >= 0 && swizzle_len >= 0 && atom_len >= swizzle_len;
}

PrimExpr SwizzleLayoutNode::GetSize() const { return 1 << (per_element + swizzle_len + atom_len); }

PrimExpr SwizzleLayoutNode::GetCosize() const { return GetSize(); }

PrimExpr SwizzleLayoutNode::Apply(const PrimExpr& coord) const {
  PrimExpr input = coord;
  auto f = [&](const PrimExpr& x) -> PrimExpr {
    if (swizzle_inner) {
      return x ^ ((x & outer_mask) >> atom_len);
    } else {
      return x ^ ((x & inner_mask) << atom_len);
    }
  };
  auto base = 1 << per_element;
  arith::Analyzer analyzer;
  // It takes more arithmetic operations to compute the result, but it is more friendly to the
  // vectorization
  return analyzer.Simplify((f(floordiv(input, base)) << per_element) + floormod(input, base));
}

}  // namespace tirxxx

}  // namespace tvm