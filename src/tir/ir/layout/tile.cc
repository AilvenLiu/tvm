/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements. See the NOTICE file
 * distributed with this work for additional information.
 * The ASF licenses this file to you under the Apache License, Version 2.0.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <tvm/runtime/container/variant.h>

#include "utils.h"

namespace tvm {
namespace tir {

/* --------------------------------------------------------------------------
 * Helper Functions
 * -------------------------------------------------------------------------- */

// Creates a TileLayout mapping a logical shape to itself (identity).
TileLayout IdentityTileLayout(Array<PrimExpr> shape) {
  return TileLayout({DataIterAttr(ReduceMul(shape), 1)}, {});
}

// Groups a TileLayout by a given logical shape. Returns (new TileLayout, separation indices).
std::pair<TileLayout, std::vector<int64_t>> TileLayoutGroupByLogicalShape(
    TileLayout layout, Array<PrimExpr> shape, std::unordered_map<int, int>* index_map) {
  size_t shape_match_dim = 0;
  arith::Analyzer analyzer;
  PrimExpr prod = 1;
  std::vector<int64_t> seps{0};
  Array<DataIterAttr> new_data_iters;
  Array<DeviceIterAttr> new_device_iters;
  auto split_map = layout.GetSplitMap();

  // This map collects which data-iter indices are associated with a device-iter index.
  std::unordered_map<int, std::vector<int>> split_device_iter;

  for (int i = 0; i < static_cast<int>(layout->data_iter_array.size()); ++i) {
    auto split_it = split_map.find(i);
    auto extent_i = layout->data_iter_array[i]->extent;
    auto stride_i = layout->data_iter_array[i]->stride;
    prod *= extent_i;

    // Try to match dimension with shape[shape_match_dim].
    while (shape_match_dim < shape.size() &&
           analyzer.CanProveEqual(floormod(prod, shape[shape_match_dim]), 0)) {
      // If the current extent isn't cleanly divisible, fail.
      if (!analyzer.CanProveEqual(floormod(extent_i, floordiv(prod, shape[shape_match_dim])), 0)) {
        return {TileLayout(), {}};
      }
      // Split the data-iter extent.
      PrimExpr split_extent = floordiv(extent_i, floordiv(prod, shape[shape_match_dim]));
      PrimExpr new_stride = stride_i * floordiv(extent_i, split_extent);

      new_data_iters.push_back(DataIterAttr(split_extent, new_stride));
      if (index_map) {
        (*index_map)[new_data_iters.size() - 1] = i;
      }

      // Update extent_i, prod, and shape_match_dim.
      extent_i = floordiv(prod, shape[shape_match_dim]);
      prod = floordiv(prod, shape[shape_match_dim]);
      ++shape_match_dim;
      seps.push_back(new_data_iters.size());

      // Mark the new index as belonging to the original device-iter index if needed.
      if (split_it != split_map.end()) {
        split_device_iter[split_it->second].push_back(new_data_iters.size() - 1);
      }
    }

    // If still > 1, add a final data-iter for leftover factor.
    if (!analyzer.CanProveEqual(extent_i, 1)) {
      new_data_iters.push_back(DataIterAttr(extent_i, stride_i));
      if (index_map) {
        (*index_map)[new_data_iters.size() - 1] = i;
      }
      if (split_it != split_map.end()) {
        split_device_iter[split_it->second].push_back(new_data_iters.size() - 1);
      }
    }
  }

  // If not all shape dimensions were matched, fail.
  if (shape_match_dim != shape.size()) return {TileLayout(), {}};

  // Rebuild device-iter array.
  for (int i = 0; i < static_cast<int>(layout->device_iter_array.size()); ++i) {
    if (!layout->device_iter_array[i].IsSplit()) {
      new_device_iters.push_back(layout->device_iter_array[i]);
    } else {
      // For each data-iter derived from this device-iter, create a new device-iter.
      auto it = split_device_iter.find(i);
      ICHECK(it != split_device_iter.end())
          << "The split device iter must have a corresponding data iter";
      for (auto di_idx : it->second) {
        new_device_iters.push_back(DeviceIterAttr::Split(new_data_iters[di_idx]->extent, di_idx));
      }
    }
  }

  return {TileLayout(new_data_iters, new_device_iters, layout->from, layout->to), seps};
}

// Tiles a logical shape by a given factor array. If is_inner is true, factors appear in inner
// order.
Array<PrimExpr> TileShape(Array<PrimExpr> shape, Array<PrimExpr> factor, bool is_inner) {
  ICHECK_EQ(shape.size(), factor.size()) << "Shape and factor dimension must match.";
  arith::Analyzer analyzer;

  // Check divisibility.
  for (int i = 0; i < static_cast<int>(shape.size()); ++i) {
    ICHECK(analyzer.CanProveEqual(floormod(shape[i], factor[i]), 0))
        << "Shape[i] must be divisible by factor[i]";
  }

  Array<PrimExpr> new_shape;
  for (int i = 0; i < static_cast<int>(shape.size()); ++i) {
    if (is_inner) {
      new_shape.push_back(floordiv(shape[i], factor[i]));
      new_shape.push_back(factor[i]);
    } else {
      new_shape.push_back(factor[i]);
      new_shape.push_back(floordiv(shape[i], factor[i]));
    }
  }
  return new_shape;
}

// Extract every even index from seps (used to pick certain boundaries).
std::vector<int64_t> GetEvenSeps(std::vector<int64_t> seps) {
  std::vector<int64_t> even;
  for (size_t i = 0; i < seps.size(); ++i) {
    if (i % 2 == 0) {
      even.push_back(seps[i]);
    }
  }
  return even;
}

// Check if two TileLayouts have device iters of identical extent.
bool CheckTileLayoutDeviceExtentEqual(TileLayout layout1, TileLayout layout2) {
  if (layout1->device_iter_array.size() != layout2->device_iter_array.size()) return false;
  arith::Analyzer analyzer;
  for (size_t i = 0; i < layout1->device_iter_array.size(); ++i) {
    if (!analyzer.CanProveEqual(layout1->device_iter_array[i]->extent,
                                layout2->device_iter_array[i]->extent)) {
      return false;
    }
  }
  return true;
}

/* --------------------------------------------------------------------------
 * TLayout::Tile Implementations
 * -------------------------------------------------------------------------- */

// Base TLayout: default not implemented.
TLayout TLayoutNode::Tile(const TileLayout& outer, const Array<PrimExpr>& outer_shape,
                          const Array<PrimExpr>& inner_shape) const {
  LOG(FATAL) << "Tile is not implemented for this layout.";
  throw;
}

// TileLayout: tile the *inner* layout (this) by combining with an *outer* layout.
TLayout TileLayoutNode::Tile(const TileLayout& outer, const Array<PrimExpr>& outer_shape,
                             const Array<PrimExpr>& inner_shape) const {
  auto outer_tile = outer->Normalize().as<TileLayout>().value();
  auto inner_tile = GetRef<TileLayout>(this)->Normalize().as<TileLayout>().value();

  ICHECK_EQ(outer_shape.size(), inner_shape.size())
      << "The dimension of outer and inner shapes must match.";

  // Group both layouts by their respective logical shapes.
  auto [grouped_outer, outer_seps] =
      TileLayoutGroupByLogicalShape(outer_tile, outer_shape, nullptr);
  auto [grouped_inner, inner_seps] =
      TileLayoutGroupByLogicalShape(inner_tile, inner_shape, nullptr);
  outer_tile = grouped_outer;
  inner_tile = grouped_inner;

  ICHECK(!outer_seps.empty()) << "Outer layout must only use split/reorder from logical shape.";
  ICHECK(!inner_seps.empty()) << "Inner layout must only use split/reorder from logical shape.";
  // Outer layout must not have device iters in this logic.
  ICHECK(outer_tile->device_iter_array.empty()) << "Outer layout must not have device iterators.";

  arith::Analyzer analyzer;
  PrimExpr inner_stride = inner_tile->GetCosize();
  Array<DataIterAttr> fused_data_iters;
  std::unordered_map<int, int> index_map;
  auto inner_split_map = inner_tile.GetSplitMap();
  int ndim = static_cast<int>(outer_shape.size());

  // Build fused data iters: combine each dimension's outer-seps, then inner-seps.
  for (int i = 0; i < ndim; ++i) {
    for (int j = outer_seps[i]; j < outer_seps[i + 1]; ++j) {
      PrimExpr new_stride = inner_stride * outer_tile->data_iter_array[j]->stride;
      fused_data_iters.push_back(DataIterAttr(outer_tile->data_iter_array[j]->extent, new_stride));
    }
    for (int j = inner_seps[i]; j < inner_seps[i + 1]; ++j) {
      fused_data_iters.push_back(inner_tile->data_iter_array[j]);
      if (inner_split_map.count(j)) {
        index_map[inner_split_map[j]] = fused_data_iters.size() - 1;
      }
    }
  }

  // Rebuild device iter array using the fused data iter indices for inner splits.
  Array<DeviceIterAttr> new_device_iters;
  for (int i = 0; i < static_cast<int>(inner_tile->device_iter_array.size()); ++i) {
    if (inner_tile->device_iter_array[i].IsSplit()) {
      int mapped_idx = index_map[i];
      new_device_iters.push_back(
          DeviceIterAttr::Split(inner_tile->device_iter_array[i]->extent, mapped_idx));
    } else {
      new_device_iters.push_back(inner_tile->device_iter_array[i]);
    }
  }

  TileLayout fused_layout(fused_data_iters, new_device_iters, inner_tile->from, inner_tile->to);
  return SimplifyTileLayout(fused_layout);
}

// ComposeLayout: tile by composing the "layout_B" with the outer layout, then stacking layout_A.
TLayout ComposeLayoutNode::Tile(const TileLayout& outer, const Array<PrimExpr>& outer_shape,
                                const Array<PrimExpr>& inner_shape) const {
  // layout_B is first tiled with `outer`, then compose with layout_A.
  auto tiled_B = layout_B->Tile(outer, outer_shape, inner_shape).as<TileLayout>().value();
  return ComposeLayout(layout_A, tiled_B);
}

// SwizzleLayout: tile by effectively composing with Identity(inner_shape).
TLayout SwizzleLayoutNode::Tile(const TileLayout& outer, const Array<PrimExpr>& outer_shape,
                                const Array<PrimExpr>& inner_shape) const {
  // Compose(Swizzle, Identity) -> then tile with `outer`.
  auto comp = ComposeLayout(GetRef<SwizzleLayout>(this), IdentityTileLayout(inner_shape));
  return comp->Tile(outer, outer_shape, inner_shape);
}

/* --------------------------------------------------------------------------
 * Global Registration for Tile
 * -------------------------------------------------------------------------- */

TVM_REGISTER_GLOBAL("tir.TileLayoutTile")
    .set_body_typed([](TileLayout outer, TileLayout inner, Array<PrimExpr> outer_shape,
                       Array<PrimExpr> inner_shape) {
      return inner->Tile(outer, outer_shape, inner_shape);
    });

TVM_REGISTER_GLOBAL("tir.ComposeLayoutTile")
    .set_body_typed([](TileLayout outer, ComposeLayout inner, Array<PrimExpr> outer_shape,
                       Array<PrimExpr> inner_shape) {
      return inner->Tile(outer, outer_shape, inner_shape);
    });

TVM_REGISTER_GLOBAL("tir.SwizzleLayoutTile")
    .set_body_typed([](TileLayout outer, SwizzleLayout inner, Array<PrimExpr> outer_shape,
                       Array<PrimExpr> inner_shape) {
      return inner->Tile(outer, outer_shape, inner_shape);
    });

/* --------------------------------------------------------------------------
 * Checking Inner Layout
 * -------------------------------------------------------------------------- */

// Base TLayout: not implemented.
Optional<TileLayout> TLayoutNode::IsTileInner(const TLayout& /*tile_layout*/,
                                              const Array<PrimExpr>& /*tiled_shape*/,
                                              const Array<PrimExpr>& /*inner_shape*/) const {
  LOG(FATAL) << "IsTileInner is not implemented for this layout.";
  throw;
}

// TileLayout: check if `tile_layout` can be the result of applying this layout as the *inner*
// layout.
Optional<TileLayout> TileLayoutNode::IsTileInner(const TLayout& tile_layout,
                                                 const Array<PrimExpr>& tiled_shape,
                                                 const Array<PrimExpr>& inner_shape) const {
  auto maybe_tile = tile_layout.as<TileLayout>();
  if (!maybe_tile) return NullOpt;

  TileLayout tiled = maybe_tile.value()->Normalize().as<TileLayout>().value();
  TileLayout layout = GetRef<TileLayout>(this)->Normalize().as<TileLayout>().value();
  arith::Analyzer analyzer;

  // Check dimension and build a factored shape (tiled_shape -> factor = inner_shape).
  int ndim = static_cast<int>(tiled_shape.size());
  if (ndim != static_cast<int>(inner_shape.size())) return NullOpt;

  auto factored = TileShape(tiled_shape, inner_shape, true);
  auto [grouped_tiled, tiled_seps] = TileLayoutGroupByLogicalShape(tiled, factored, nullptr);
  auto [grouped_layout, inner_seps] = TileLayoutGroupByLogicalShape(layout, inner_shape, nullptr);

  if (grouped_tiled->device_iter_array.size() != grouped_layout->device_iter_array.size()) {
    // Quick device-iter check (or we do a direct extent check below).
    // But let's do the standard one:
    if (!CheckTileLayoutDeviceExtentEqual(grouped_tiled, grouped_layout)) return NullOpt;
  }

  if (tiled_seps.empty() || inner_seps.empty()) return NullOpt;

  auto tiled_seps_even = GetEvenSeps(tiled_seps);
  std::unordered_map<int, int> index_map;
  auto layout_split_map = grouped_layout.GetSplitMap();

  std::vector<DataIterAttr> outer_data_iters;

  // For each dimension i, compare the trailing portion of grouped_tiled (for that dimension)
  // with the portion in grouped_layout.
  for (int i = 0; i < ndim; ++i) {
    int inner_count = inner_seps[i + 1] - inner_seps[i];
    int tiled_count = tiled_seps_even[i + 1] - tiled_seps_even[i];
    if (inner_count > tiled_count) return NullOpt;

    // Compare extents (and stride if not splitted).
    for (int j = 0; j < inner_count; ++j) {
      DataIterAttr inner_attr = grouped_layout->data_iter_array[inner_seps[i] + j];
      DataIterAttr tiled_attr =
          grouped_tiled->data_iter_array[tiled_seps_even[i + 1] - inner_count + j];
      index_map[inner_seps[i] + j] = tiled_seps_even[i + 1] - inner_count + j;

      if (!analyzer.CanProveEqual(inner_attr->extent, tiled_attr->extent)) return NullOpt;
      bool is_splitted = (layout_split_map.find(inner_seps[i] + j) != layout_split_map.end());
      if (!is_splitted && !analyzer.CanProveEqual(inner_attr->stride, tiled_attr->stride) &&
          !analyzer.CanProveEqual(inner_attr->extent, 1)) {
        return NullOpt;
      }
    }
    for (int j = 0; j < tiled_count - inner_count; ++j) {
      outer_data_iters.push_back(grouped_tiled->data_iter_array[tiled_seps_even[i] + j]);
    }
  }

  // Also check device-iter splits match up.
  auto tiled_split_map = grouped_tiled.GetSplitMap();
  if (tiled_split_map.size() != layout_split_map.size()) return NullOpt;

  for (auto& kv : layout_split_map) {
    auto it = tiled_split_map.find(index_map[kv.first]);
    if (it == tiled_split_map.end() || it->second != kv.second) return NullOpt;
  }

  // Divide strides by inner->CoSize()
  std::vector<DataIterAttr> adjusted_data_iters;
  PrimExpr inner_cosize = grouped_layout->GetCosize();
  for (const auto& iter : outer_data_iters) {
    adjusted_data_iters.push_back(
        DataIterAttr(iter->extent, analyzer.Simplify(FloorDiv(iter->stride, inner_cosize))));
  }
  return TileLayout(adjusted_data_iters, {});
}

// SwizzleLayout: check if the given tile_layout is Compose(SwizzleLayout, anything).
Optional<TileLayout> SwizzleLayoutNode::IsTileInner(const TLayout& tile_layout,
                                                    const Array<PrimExpr>& tiled_shape,
                                                    const Array<PrimExpr>& inner_shape) const {
  // We expect tile_layout to be Compose(SwizzleLayout(this), _).
  if (auto comp = tile_layout.as<ComposeLayout>()) {
    if (StructuralEqual()(comp.value()->layout_A, GetRef<SwizzleLayout>(this))) {
      auto identity = IdentityTileLayout(inner_shape);
      return identity->IsTileInner(comp.value()->layout_B, tiled_shape, inner_shape);
    }
  }
  return NullOpt;
}

// ComposeLayout: check if tile_layout is Compose(A, B) and layout_A matches this->layout_A,
// then recursively check layout_B.
Optional<TileLayout> ComposeLayoutNode::IsTileInner(const TLayout& tile_layout,
                                                    const Array<PrimExpr>& tiled_shape,
                                                    const Array<PrimExpr>& inner_shape) const {
  if (auto comp = tile_layout.as<ComposeLayout>()) {
    if (StructuralEqual()(comp.value()->layout_A, this->layout_A)) {
      return this->layout_B->IsTileInner(comp.value()->layout_B, tiled_shape, inner_shape);
    }
  }
  return NullOpt;
}

/* Registration for Checking Inner Layout */

TVM_REGISTER_GLOBAL("tir.TileLayoutIsTileInner")
    .set_body_typed([](TLayout tiled_layout, TileLayout layout, Array<PrimExpr> tiled_shape,
                       Array<PrimExpr> inner_shape) {
      return layout->IsTileInner(tiled_layout, tiled_shape, inner_shape);
    });

TVM_REGISTER_GLOBAL("tir.SwizzleLayoutIsTileInner")
    .set_body_typed([](TLayout tiled_layout, SwizzleLayout layout, Array<PrimExpr> tiled_shape,
                       Array<PrimExpr> inner_shape) {
      return layout->IsTileInner(tiled_layout, tiled_shape, inner_shape);
    });

TVM_REGISTER_GLOBAL("tir.ComposeLayoutIsTileInner")
    .set_body_typed([](TLayout tiled_layout, ComposeLayout layout, Array<PrimExpr> tiled_shape,
                       Array<PrimExpr> inner_shape) {
      return layout->IsTileInner(tiled_layout, tiled_shape, inner_shape);
    });

/* --------------------------------------------------------------------------
 * Checking Outer Layout
 * -------------------------------------------------------------------------- */

// Base TLayout: not implemented.
bool TLayoutNode::IsTileOuter(const TLayout& /*tile_layout*/,
                              const Array<PrimExpr>& /*tiled_shape*/,
                              const Array<PrimExpr>& /*outer_shape*/) const {
  LOG(FATAL) << "IsTileOuter is not implemented for this layout.";
  throw;
}

// TileLayout: check if `tile_layout` can be the result of applying this layout as the *outer*
// layout.
bool TileLayoutNode::IsTileOuter(const TLayout& tile_layout, const Array<PrimExpr>& tiled_shape,
                                 const Array<PrimExpr>& outer_shape) const {
  auto maybe_tile = tile_layout.as<TileLayout>();
  if (!maybe_tile) {
    // Could be ComposeLayout, in which case we test layout_B of compose.
    if (auto comp = tile_layout.as<ComposeLayout>()) {
      return IsTileOuter(comp.value()->layout_B, tiled_shape, outer_shape);
    }
    return false;
  }

  auto tiled = maybe_tile.value()->Normalize().as<TileLayout>().value();
  auto layout = GetRef<TileLayout>(this)->Normalize().as<TileLayout>().value();
  arith::Analyzer analyzer;
  int ndim = static_cast<int>(tiled_shape.size());

  // Check dimension match and that each dimension of tiled_shape is multiple of outer_shape.
  ICHECK_EQ(ndim, static_cast<int>(outer_shape.size()))
      << "Outer shape and tiled shape must have same dimension.";
  for (int i = 0; i < ndim; ++i) {
    if (!analyzer.CanProveEqual(floormod(tiled_shape[i], outer_shape[i]), 0)) {
      return false;
    }
  }

  // Outer layout must not have device iters if used as the outer layout.
  if (!layout->device_iter_array.empty()) return false;

  // Factor tiled_shape by outer_shape.
  auto factored = TileShape(tiled_shape, outer_shape, /*is_inner=*/false);
  auto [grouped_tiled, tiled_seps] = TileLayoutGroupByLogicalShape(tiled, factored, nullptr);
  auto [grouped_layout, outer_seps] = TileLayoutGroupByLogicalShape(layout, outer_shape, nullptr);

  if (tiled_seps.empty() || outer_seps.empty()) return false;
  auto tiled_seps_even = GetEvenSeps(tiled_seps);

  // We'll collect leftover data iters (which belong to the "inner" part).
  Array<DataIterAttr> inner_data_iters;
  auto tiled_split_map = grouped_tiled.GetSplitMap();

  // Verify outer extents match directly.
  for (int i = 0; i < ndim; ++i) {
    int outer_count = outer_seps[i + 1] - outer_seps[i];
    int tile_count = tiled_seps_even[i + 1] - tiled_seps_even[i];
    if (outer_count > tile_count) return false;

    // Compare each "outer" chunk's extents.
    for (int j = 0; j < outer_count; ++j) {
      DataIterAttr outer_attr = grouped_layout->data_iter_array[outer_seps[i] + j];
      DataIterAttr tiled_attr = grouped_tiled->data_iter_array[tiled_seps_even[i] + j];
      if (tiled_split_map.count(tiled_seps_even[i] + j)) return false;
      if (!analyzer.CanProveEqual(outer_attr->extent, tiled_attr->extent)) return false;
    }

    // The rest belong to the "inner" side.
    for (int j = tiled_seps_even[i] + outer_count; j < tiled_seps_even[i + 1]; ++j) {
      if (!tiled_split_map.count(j)) {
        inner_data_iters.push_back(grouped_tiled->data_iter_array[j]);
      }
    }
  }

  // Check that the outer stride matches the fused stride = outer_attr->stride * (inner cosize).
  TileLayout inner_layout(inner_data_iters, {});
  auto inner_cosize = inner_layout->GetCosize();

  for (int i = 0; i < ndim; ++i) {
    int outer_count = outer_seps[i + 1] - outer_seps[i];
    for (int j = 0; j < outer_count; ++j) {
      DataIterAttr outer_attr = grouped_layout->data_iter_array[outer_seps[i] + j];
      DataIterAttr tiled_attr = grouped_tiled->data_iter_array[tiled_seps_even[i] + j];
      if (!analyzer.CanProveEqual(outer_attr->stride * inner_cosize, tiled_attr->stride)) {
        return false;
      }
    }
  }
  return true;
}

/* Registration for Checking Outer Layout */

TVM_REGISTER_GLOBAL("tir.TileLayoutIsTileOuter")
    .set_body_typed([](TLayout tiled_layout, TileLayout layout, Array<PrimExpr> tiled_shape,
                       Array<PrimExpr> outer_shape) {
      return layout->IsTileOuter(tiled_layout, tiled_shape, outer_shape);
    });

}  // namespace tir
}  // namespace tvm
