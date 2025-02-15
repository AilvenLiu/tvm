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
#include <tvm/runtime/container/variant.h>

#include "utils.h"

namespace tvm {
namespace tir {

/******** Tile - TLayout ********/

TLayout TLayoutNode::Tile(const TileLayout& outer, const Array<PrimExpr>& outer_shape,
                          const Array<PrimExpr>& inner_shape) const {
  CHECK(false) << "Tile is not implemented for this layout";
}

/******** Tile - TileLayout ********/

TileLayout IdentityTileLayout(Array<PrimExpr> shape) {
  std::vector<DataIterAttr> data_iter_vec;
  PrimExpr stride = 1;
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; i--) {
    data_iter_vec.push_back(DataIterAttr(shape[i], stride));
    stride *= shape[i];
  }
  std::reverse(data_iter_vec.begin(), data_iter_vec.end());
  return TileLayout(data_iter_vec, {});
}

std::pair<TileLayout, std::vector<int64_t>> TileLayoutGroupByLogicalShape(
    TileLayout layout, Array<PrimExpr> shape, std::unordered_map<int, int>* index_map) {
  size_t shape_match_dim = 0;
  arith::Analyzer analyzer;
  PrimExpr prod = 1;
  std::vector<int64_t> seps;
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
        return std::make_pair(TileLayout(), std::vector<int64_t>());
      }
      PrimExpr split_extent = floordiv(cur_extent, floordiv(prod, shape[shape_match_dim]));
      new_data_iter_array.push_back(
          DataIterAttr(split_extent, stride * floordiv(cur_extent, split_extent)));
      if (index_map) {
        (*index_map)[new_data_iter_array.size() - 1] = i;
      }
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
      if (index_map) {
        (*index_map)[new_data_iter_array.size() - 1] = i;
      }
      if (split_map_it != split_map.end()) {
        split_device_iter[split_map_it->second].push_back(new_data_iter_array.size() - 1);
      }
    }
  }
  if (shape_match_dim != shape.size()) {
    return std::make_pair(TileLayout(), std::vector<int64_t>());
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

TLayout TileLayoutNode::Tile(const TileLayout& outer, const Array<PrimExpr>& outer_shape,
                             const Array<PrimExpr>& inner_shape) const {
  TileLayout outer_tile = outer->Normalize().as<TileLayout>().value();
  TileLayout inner_tile = GetRef<TileLayout>(this)->Normalize().as<TileLayout>().value();

  ICHECK_EQ(outer_shape.size(), inner_shape.size())
      << "ValueError: The dimension of logical view shapes must be the same";
  auto [grouped_outer, outer_seps] =
      TileLayoutGroupByLogicalShape(outer_tile, outer_shape, nullptr);
  auto [grouped_inner, inner_seps] =
      TileLayoutGroupByLogicalShape(inner_tile, inner_shape, nullptr);
  outer_tile = grouped_outer;
  inner_tile = grouped_inner;

  ICHECK(!outer_seps.empty()) << "ValueError: The outer layout must be able to transform from "
                                 "logical view shape only with split and reorder";
  ICHECK(!inner_seps.empty())
      << "ValueError: The inner layout must be able to transform from logical view shape only with "
         "split and reorder";
  ICHECK(outer_tile->device_iter_array.empty())
      << "ValueError: The outer layout must not have device "
         "iterators";
  int shape_dim = static_cast<int>(outer_shape.size());

  // Get the stride in the inner layout
  arith::Analyzer analyzer;
  PrimExpr inner_stride = inner_tile->GetCosize();
  Array<DataIterAttr> fused_data_iter_array;
  std::unordered_map<int, int> index_map;
  auto inner_split_map = inner_tile.GetSplitMap();
  for (int i = 0; i < shape_dim; i++) {
    for (int j = outer_seps[i]; j < outer_seps[i + 1]; j++) {
      PrimExpr new_stride = inner_stride * outer->data_iter_array[j]->stride;
      fused_data_iter_array.push_back(DataIterAttr(outer->data_iter_array[j]->extent, new_stride));
    }
    for (int j = inner_seps[i]; j < inner_seps[i + 1]; j++) {
      fused_data_iter_array.push_back(inner_tile->data_iter_array[j]);
      if (inner_split_map.find(j) != inner_split_map.end()) {
        index_map[inner_split_map[j]] = fused_data_iter_array.size() - 1;
      }
    }
  }
  Array<DeviceIterAttr> new_device_iter_array;
  for (int i = 0; i < static_cast<int>(inner_tile->device_iter_array.size()); i++) {
    if (inner_tile->device_iter_array[i].IsSplit()) {
      int bound = index_map[i];
      new_device_iter_array.push_back(
          DeviceIterAttr::Split(inner_tile->device_iter_array[i]->extent, bound));
    } else {
      new_device_iter_array.push_back(inner_tile->device_iter_array[i]);
    }
  }

  return SimplifyTileLayout(
      TileLayout(fused_data_iter_array, new_device_iter_array, inner_tile->from, inner_tile->to));
}

TVM_REGISTER_GLOBAL("tir.TileLayoutTile")
    .set_body_typed([](TileLayout outer, TileLayout inner, Array<PrimExpr> outer_shape,
                       Array<PrimExpr> inner_shape) {
      return inner->Tile(outer, outer_shape, inner_shape);
    });

/******** Tile - ComposeLayout ********/

TLayout ComposeLayoutNode::Tile(const TileLayout& outer, const Array<PrimExpr>& outer_shape,
                                const Array<PrimExpr>& inner_shape) const {
  return ComposeLayout(layout_A,
                       layout_B->Tile(outer, outer_shape, inner_shape).as<TileLayout>().value());
}

TVM_REGISTER_GLOBAL("tir.ComposeLayoutTile")
    .set_body_typed([](TileLayout outer, ComposeLayout inner, Array<PrimExpr> outer_shape,
                       Array<PrimExpr> inner_shape) {
      return inner->Tile(outer, outer_shape, inner_shape);
    });

/******** Tile - SwizzleLayout ********/

TLayout SwizzleLayoutNode::Tile(const TileLayout& outer, const Array<PrimExpr>& outer_shape,
                                const Array<PrimExpr>& inner_shape) const {
  ComposeLayout layout =
      ComposeLayout(GetRef<SwizzleLayout>(this), IdentityTileLayout(inner_shape));
  return layout->Tile(outer, outer_shape, inner_shape);
}

TVM_REGISTER_GLOBAL("tir.SwizzleLayoutTile")
    .set_body_typed([](TileLayout outer, SwizzleLayout inner, Array<PrimExpr> outer_shape,
                       Array<PrimExpr> inner_shape) {
      return inner->Tile(outer, outer_shape, inner_shape);
    });

/******** Tile - Check Inner Layout - TLayout ********/

bool TLayoutNode::IsTileInner(const TLayout& tile_layout, const Array<PrimExpr>& tiled_shape,
                              const Array<PrimExpr>& inner_shape) const {
  ICHECK(false) << "IsTileInner is not implemented for this layout";
}

/******** Tile - Check Inner Layout - TileLayout ********/

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

std::vector<int64_t> GetEvenSeps(std::vector<int64_t> seps) {
  std::vector<int64_t> even_seps;
  for (int i = 0; i < static_cast<int>(seps.size()); i++) {
    if (i % 2 == 0) {
      even_seps.push_back(seps[i]);
    }
  }
  return even_seps;
}

bool TileLayoutNode::IsTileInner(const TLayout& tile_layout, const Array<PrimExpr>& tiled_shape,
                                 const Array<PrimExpr>& inner_shape) const {
  if (auto tile = tile_layout.as<TileLayout>()) {
    TileLayout tiled_layout = tile.value()->Normalize().as<TileLayout>().value();
    TileLayout layout = GetRef<TileLayout>(this)->Normalize().as<TileLayout>().value();
    int tiled_shape_dim = static_cast<int>(tiled_shape.size());
    int inner_shape_dim = static_cast<int>(inner_shape.size());
    if (tiled_shape_dim != inner_shape_dim) {
      return false;
    }
    auto factored_tiled_shape = TileShape(tiled_shape, inner_shape, true);
    auto [grouped_tiled_layout, tiled_seps] =
        TileLayoutGroupByLogicalShape(tiled_layout, factored_tiled_shape, nullptr);
    tiled_seps = GetEvenSeps(tiled_seps);
    auto [grouped_layout, inner_seps] = TileLayoutGroupByLogicalShape(layout, inner_shape, nullptr);
    tiled_layout = grouped_tiled_layout;
    layout = grouped_layout;

    // 1. check whether device iter extent are same
    if (!CheckTileLayoutDeviceExtentEqual(tiled_layout, layout)) {
      return false;
    }
    // 2. check data iter match
    ICHECK(!tiled_seps.empty()) << "ValueError: The tiled layout must be able to transform from "
                                   "logical view shape only with split and reorder";
    ICHECK(!inner_seps.empty()) << "ValueError: The inner layout must be able to transform from "
                                   "logical view shape only with "
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
  } else {
    return false;
  }
}

TVM_REGISTER_GLOBAL("tir.TileLayoutIsTileInner")
    .set_body_typed([](TLayout tiled_layout, TileLayout layout, Array<PrimExpr> tiled_shape,
                       Array<PrimExpr> inner_shape) {
      return layout->IsTileInner(tiled_layout, tiled_shape, inner_shape);
    });

/******** Tile - Check Inner Layout - SwizzleLayout ********/

bool SwizzleLayoutNode::IsTileInner(const TLayout& tile_layout, const Array<PrimExpr>& tiled_shape,
                                    const Array<PrimExpr>& inner_shape) const {
  if (auto compose = tile_layout.as<ComposeLayout>()) {
    return StructuralEqual()(compose.value()->layout_A, GetRef<SwizzleLayout>(this));
  } else {
    return false;
  }
}

TVM_REGISTER_GLOBAL("tir.SwizzleLayoutIsTileInner")
    .set_body_typed([](TLayout tiled_layout, SwizzleLayout layout, Array<PrimExpr> tiled_shape,
                       Array<PrimExpr> inner_shape) {
      return layout->IsTileInner(tiled_layout, tiled_shape, inner_shape);
    });

/******** Tile - Check Inner Layout - ComposeLayout ********/

bool ComposeLayoutNode::IsTileInner(const TLayout& tile_layout, const Array<PrimExpr>& tiled_shape,
                                    const Array<PrimExpr>& inner_shape) const {
  if (auto compose = tile_layout.as<ComposeLayout>()) {
    return StructuralEqual()(compose.value()->layout_A, this->layout_A) &&
           this->layout_B->IsTileInner(compose.value()->layout_B, tiled_shape, inner_shape);
  } else {
    return false;
  }
}

TVM_REGISTER_GLOBAL("tir.ComposeLayoutIsTileInner")
    .set_body_typed([](TLayout tiled_layout, ComposeLayout layout, Array<PrimExpr> tiled_shape,
                       Array<PrimExpr> inner_shape) {
      return layout->IsTileInner(tiled_layout, tiled_shape, inner_shape);
    });

/******** Tile - Check Outer Layout - TLayout ********/

bool TLayoutNode::IsTileOuter(const TLayout& tile_layout, const Array<PrimExpr>& tiled_shape,
                              const Array<PrimExpr>& outer_shape) const {
  ICHECK(false) << "IsTileOuter is not implemented for this layout";
}

/******** Tile - Check Outer Layout - TileLayout ********/

bool TileLayoutNode::IsTileOuter(const TLayout& tile_layout, const Array<PrimExpr>& tiled_shape,
                                 const Array<PrimExpr>& outer_shape) const {
  if (auto tile = tile_layout.as<TileLayout>()) {
    TileLayout tiled_layout = tile.value()->Normalize().as<TileLayout>().value();
    TileLayout layout = this->Normalize().as<TileLayout>().value();
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
    if (!device_iter_array.empty()) {
      return false;
    }
    // 2. check data iter match
    auto factored_tiled_shape = TileShape(tiled_shape, outer_shape, false);
    auto [grouped_tiled_layout, tiled_seps] =
        TileLayoutGroupByLogicalShape(tiled_layout, factored_tiled_shape, nullptr);
    tiled_seps = GetEvenSeps(tiled_seps);
    auto [grouped_layout, outer_seps] = TileLayoutGroupByLogicalShape(layout, outer_shape, nullptr);
    tiled_layout = grouped_tiled_layout;
    layout = grouped_layout;
    ICHECK(!tiled_seps.empty()) << "ValueError: The tiled layout must be able to transform from "
                                   "logical view shape only with split and reorder";
    ICHECK(!outer_seps.empty()) << "ValueError: The outer layout must be able to transform from "
                                   "logical view shape only with "
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
  } else if (auto compose = tile_layout.as<ComposeLayout>()) {
    return this->IsTileOuter(compose.value()->layout_B, tiled_shape, outer_shape);
  } else {
    return false;
  }
}

TVM_REGISTER_GLOBAL("tir.TileLayoutIsTileOuter")
    .set_body_typed([](TLayout tiled_layout, TileLayout layout, Array<PrimExpr> tiled_shape,
                       Array<PrimExpr> outer_shape) {
      return layout->IsTileOuter(tiled_layout, tiled_shape, outer_shape);
    });

}  // namespace tir
}  // namespace tvm
