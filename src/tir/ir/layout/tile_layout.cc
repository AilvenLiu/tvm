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

/**************** Axis ****************/
// AxisNode
ObjectPtr<Object> CreateAxis(const std::string& name) {
  // Hack use ffi::Any as exchange
  auto axis = Axis::Get(name);
  ICHECK(axis.defined()) << "Cannot find axis \'" << name << '\'';
  return ffi::details::ObjectUnsafe::ObjectPtrFromObjectRef<Object>(axis);
}

TVM_REGISTER_NODE_TYPE(AxisNode)
    .set_creator(CreateAxis)
    .set_repr_bytes([](const Object* n) -> std::string {
      return static_cast<const AxisNode*>(n)->name;
    });

bool AxisNode::IsThreadAxis() const {
  static const auto& thread_attr_map = Axis::GetAttrMap<bool>("thread");
  return thread_attr_map[GetRef<Axis>(this)];
}

bool AxisNode::IsMemoryAxis() const {
  static const auto& thread_attr_map = Axis::GetAttrMap<bool>("thread");
  return !thread_attr_map[GetRef<Axis>(this)];
}

TVM_REGISTER_GLOBAL("tir.AxisIsThreadAxis").set_body_typed([](Axis axis) {
  return axis->IsThreadAxis();
});

TVM_REGISTER_GLOBAL("tir.AxisIsMemoryAxis").set_body_typed([](Axis axis) {
  return axis->IsMemoryAxis();
});

// Axis
Axis Axis::Get(const String& name) {
  const AxisRegEntry* reg = AxisRegistry::Global()->Get(name);
  CHECK(reg != nullptr) << "Axis " << name << " is not registered";
  return reg->axis_;
}

template <typename ValueType>
inline AxisAttrMap<ValueType> Axis::GetAttrMap(const String& attr_name) {
  return AxisAttrMap<ValueType>(AxisRegistry::Global()->GetAttrMap(attr_name));
}

// AxisRegEntry
inline AxisNode* AxisRegEntry::get() { return const_cast<AxisNode*>(axis_.operator->()); }

AxisRegEntry::AxisRegEntry(uint32_t index) {
  ObjectPtr<AxisNode> n = make_object<AxisNode>();
  n->index_ = index;
  axis_ = Axis(n);
}

AxisRegEntry& AxisRegEntry::RegisterOrGet(const String& name) {
  auto& entry = AxisRegistry::Global()->RegisterOrGet(name);
  entry.get()->name = name;
  return entry;
}

Array<String> AxisRegEntry::ListAxisNames() { return AxisRegistry::Global()->ListAllNames(); }

template <typename ValueType>
inline AxisRegEntry& AxisRegEntry::set_attr(const String& key, const ValueType& value, int plevel) {
  ICHECK_GT(plevel, 0) << "plevel in set_attr must be greater than 0";
  ffi::Any rv;
  rv = value;
  UpdateAttr(key, rv, plevel);
  return *this;
}

void AxisRegEntry::UpdateAttr(const String& key, ffi::Any value, int plevel) {
  AxisRegistry::Global()->UpdateAttr(key, axis_, value, plevel);
}

// register theaad axis
TVM_REGISTER_AXIS("bx").set_attr<bool>("thread", true);
TVM_REGISTER_AXIS("by").set_attr<bool>("thread", true);
TVM_REGISTER_AXIS("bz").set_attr<bool>("thread", true);
TVM_REGISTER_AXIS("cbx").set_attr<bool>("thread", true);
TVM_REGISTER_AXIS("cby").set_attr<bool>("thread", true);
TVM_REGISTER_AXIS("cbz").set_attr<bool>("thread", true);
TVM_REGISTER_AXIS("tx").set_attr<bool>("thread", true);
TVM_REGISTER_AXIS("warpid").set_attr<bool>("thread", true);
TVM_REGISTER_AXIS("laneid").set_attr<bool>("thread", true);
TVM_REGISTER_AXIS("wgid").set_attr<bool>("thread", true);
TVM_REGISTER_AXIS("tid_in_wg").set_attr<bool>("thread", true);
TVM_REGISTER_AXIS("wid_in_wg").set_attr<bool>("thread", true);

// register memory axis
TVM_REGISTER_AXIS("m").set_attr<bool>("thread", false);
TVM_REGISTER_AXIS("P").set_attr<bool>("thread", false);
TVM_REGISTER_AXIS("F").set_attr<bool>("thread", false);
TVM_REGISTER_AXIS("Bank").set_attr<bool>("thread", false);
TVM_REGISTER_AXIS("TCol").set_attr<bool>("thread", false);
TVM_REGISTER_AXIS("TLane").set_attr<bool>("thread", false);

TVM_REGISTER_GLOBAL("tir.AxisGet").set_body_typed([](String name) -> Axis {
  return Axis::Get(name);
});

/**************** Iter ****************/
TVM_REGISTER_NODE_TYPE(IterNode);

Iter::Iter(PrimExpr extent, PrimExpr stride, Axis axis) {
  auto n = make_object<IterNode>();
  n->extent = extent;
  n->stride = stride;
  n->axis = axis;
  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("tir.Iter").set_body_typed([](PrimExpr extent, PrimExpr stride, Axis axis) {
  return Iter(extent, stride, axis);
});

/**************** TileLayout ****************/

TVM_REGISTER_NODE_TYPE(TileLayoutNode);

TileLayout::TileLayout(Array<Iter> shard, Array<Iter> replicate,
                       Array<Tuple<Iter, PrimExpr>> exclude, Optional<ExecScope> subscope,
                       Optional<ExecScope> scope) {
  auto n = make_object<TileLayoutNode>();
  n->shard = shard;
  n->replicate = replicate;
  n->exclude = exclude;
  n->subscope = subscope;
  n->scope = scope;
  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("tir.TileLayout")
    .set_body_typed([](Array<Iter> shard, Array<Iter> replicate,
                       Array<Tuple<Iter, PrimExpr>> exclude, Optional<ExecScope> subscope,
                       Optional<ExecScope> scope) {
      return TileLayout(shard, replicate, exclude, subscope, scope);
    });

bool TileLayoutNode::CompatibleWithShape(const Array<PrimExpr>& shape) const { return true; }

bool VerifyCompactness(const std::vector<Iter>& iters) {
  arith::Analyzer analyzer;
  PrimExpr stride_to_find = 1;
  for (size_t i = 0; i < iters.size(); ++i) {
    auto iter = std::find_if(iters.begin(), iters.end(), [&](const Iter& iter) {
      return analyzer.CanProveEqual(iter->stride, stride_to_find);
    });
    if (iter == iters.end()) {
      return false;
    }
    stride_to_find *= (*iter)->extent;
  }
  return true;
}

bool TileLayoutNode::VerifyWellFormed() const {
  // 1. For thread axes, verify its compactness
  std::unordered_map<String, std::vector<Iter>> thread_axes;
  auto collect_thread_axis = [&thread_axes](const Iter& iter) {
    if (iter->axis->IsThreadAxis()) {
      thread_axes[iter->axis->name].push_back(iter);
    }
  };
  for (const auto& iter : shard) {
    collect_thread_axis(iter);
  }
  for (const auto& iter : replicate) {
    collect_thread_axis(iter);
  }
  for (const auto& iter_selector : exclude) {
    collect_thread_axis(iter_selector.get<0>());
  }
  for (const auto& [axis, iters] : thread_axes) {
    if (!VerifyCompactness(iters)) {
      return false;
    }
  }
  // 2. If there's any thread axis, subscope and scope must be provided
  if (!thread_axes.empty()) {
    if (!subscope.defined() || !scope.defined()) {
      return false;
    }
    if (!scope.value()->Higher(subscope.value())) {
      return false;
    }
  }
  return true;
}

PrimExpr TileLayoutNode::GetSize(Optional<String> axis_name) const {
  auto filter = [&](const Iter& iter, PrimExpr acc) {
    if (!axis_name.has_value() || iter->axis->name == axis_name.value()) {
      return acc * iter->extent;
    }
    return acc;
  };
  PrimExpr res = IntImm(shard[0]->extent->dtype, 1);
  for (const auto& iter : shard) {
    res = filter(iter, res);
  }
  for (const auto& iter : replicate) {
    res = filter(iter, res);
  }
  for (const auto& iter : exclude) {
    res = filter(iter.get<0>(), res);
  }
  return res;
}

PrimExpr TileLayoutNode::GetCosize(Optional<String> axis_name) const {
  arith::Analyzer analyzer;
  PrimExpr result = IntImm(shard[0]->extent->dtype, 0);
  auto filter = [&](const Iter& iter, PrimExpr acc) {
    if ((!axis_name.has_value() && iter->axis->IsMemoryAxis()) ||
        (axis_name.has_value() && iter->axis->name == axis_name.value())) {
      ICHECK(analyzer.CanProve(iter->stride > 0))
          << "Negative stride is not supported for memory axes currently";
      return acc + (iter->extent - 1) * iter->stride;
    }
    return acc;
  };
  for (const auto& iter : shard) {
    result = filter(iter, result);
  }
  for (const auto& iter : replicate) {
    result = filter(iter, result);
  }
  for (const auto& iter : exclude) {
    result = filter(iter.get<0>(), result);
  }
  return analyzer.Simplify(is_zero(result) ? 1 : result + 1);
}

Map<String, PrimExpr> TileLayoutNode::Apply(PrimExpr coord) const {
  return Apply(SplitCoord(coord, GetShardShape()));
}

Map<String, PrimExpr> TileLayoutNode::Apply(Array<PrimExpr> coord) const {
  arith::Analyzer analyzer;
  CHECK_EQ(coord.size(), shard.size()) << "Coordinate size must match the number of shard axes";
  std::unordered_map<String, PrimExpr> result;
  for (size_t i = 0; i < shard.size(); ++i) {
    auto it = result.find(shard[i]->axis->name);
    if (it == result.end()) {
      result[shard[i]->axis->name] = analyzer.Simplify(coord[i] * shard[i]->stride);
    } else {
      result[shard[i]->axis->name] = analyzer.Simplify(it->second + coord[i] * shard[i]->stride);
    }
  }
  return result;
}

TileLayout RemoveUnitIters(TileLayout layout) {
  auto new_layout = layout.CopyOnWrite();
  std::vector<Iter> new_shard;
  std::copy_if(layout->shard.begin(), layout->shard.end(), std::back_inserter(new_shard),
               [](const Iter& iter) { return !is_one(iter->extent); });
  // if new_shard is empty, add a unit iter
  if (new_shard.empty()) {
    // TODO(@bohan): does it matter which axis we use?
    new_shard.push_back(Iter(1, 1, layout->shard[0]->axis));
  }
  new_layout->shard = new_shard;
  return GetRef<TileLayout>(new_layout);
}

TileLayout FuseShardAxes(TileLayout layout) {
  std::vector<Iter> fused_shard;
  arith::Analyzer ana;
  const auto& shard = layout->shard;
  for (size_t cur = 0; cur < shard.size();) {
    // Find consecutive fusable axes
    PrimExpr extent = shard[cur]->extent;
    size_t next = cur + 1;
    while (next < shard.size() && shard[next]->axis.same_as(shard[cur]->axis) &&
           ana.CanProveEqual(shard[next]->extent * shard[next]->stride, shard[next - 1]->stride)) {
      extent *= shard[next]->extent;
      ++next;
    }
    if (next == cur + 1) {
      fused_shard.push_back(shard[cur]);
    } else {
      fused_shard.push_back(Iter(extent, shard[next - 1]->stride, shard[cur]->axis));
    }
    cur = next;
  }
  auto new_layout = layout.CopyOnWrite();
  new_layout->shard = fused_shard;
  return GetRef<TileLayout>(new_layout);
}

TLayout TileLayoutNode::Normalize() const {
  // 0. Remove unit iters in shard
  TileLayout res = RemoveUnitIters(GetRef<TileLayout>(this));
  // 1. Fuse shard axes
  res = FuseShardAxes(res);
  return res;
}

std::pair<TileLayout, std::vector<int64_t>> GroupByShape(TileLayout layout,
                                                         const Array<PrimExpr>& shape) {
  arith::Analyzer analyzer;
  size_t shape_idx = 0;
  PrimExpr prod = 1;

  std::vector<Iter> new_shard;
  std::vector<int64_t> seps{0};

  for (size_t i = 0; i < layout->shard.size(); ++i) {
    auto extent_i = layout->shard[i]->extent;
    auto stride_i = layout->shard[i]->stride;
    prod *= extent_i;
    while (shape_idx < shape.size() &&
           analyzer.CanProveEqual(floormod(prod, shape[shape_idx]), 0)) {
      // prod' * extent_i = prod = c * shape[shape_match_dim]
      // we split out e from extent_i such that prod' * e = shape[shape_match_dim]
      // we can prove e = extent_i / c, i.e. we split extent_i into (e, c)
      // c becomes the new extent_i, prod' is reset to 1, then we do it recursively
      PrimExpr c = floordiv(prod, shape[shape_idx]);
      CHECK(analyzer.CanProveEqual(floormod(extent_i, c), 0))
          << "layout " << layout << " can not be grouped by shape " << shape;
      new_shard.push_back(Iter(floordiv(extent_i, c), stride_i * c, layout->shard[i]->axis));
      // Update extent_i, prod and shape_match_dim
      extent_i = c;
      prod = c;
      shape_idx++;
      seps.push_back(new_shard.size());
    }
    // There's still remaining, add it to the new shard
    if (!is_one(extent_i)) {
      CHECK(shape_idx < shape.size())
          << "layout " << layout << " can not be grouped by shape " << shape;
      new_shard.push_back(Iter(extent_i, stride_i, layout->shard[i]->axis));
    }
  }

  CHECK(shape_idx == shape.size())
      << "layout " << layout << " can not be grouped by shape " << shape;

  auto* n = layout.CopyOnWrite();
  n->shard = new_shard;
  return {GetRef<TileLayout>(n), seps};
}

TVM_REGISTER_GLOBAL("tir.TileLayoutGroupByShape")
    .set_body_typed([](const TileLayout& layout, const Array<PrimExpr>& shape) {
      auto [res, seps] = GroupByShape(layout, shape);
      return Tuple<TileLayout, Array<int64_t>>{res, Array<int64_t>(seps.begin(), seps.end())};
    });

TLayout TileLayoutNode::Tile(const TileLayout& outer_in, const Array<PrimExpr>& outer_shape,
                             const Array<PrimExpr>& inner_shape) const {
  auto outer = outer_in->Normalize().as<TileLayout>().value();
  auto inner = GetRef<TileLayout>(this)->Normalize().as<TileLayout>().value();

  CHECK_EQ(outer_shape.size(), inner_shape.size()) << "Outer and inner shape size must match";

  // Group both outer/inner layouts by their respective logical shapes
  auto [grouped_outer, outer_seps] = GroupByShape(outer, outer_shape);
  auto [grouped_inner, inner_seps] = GroupByShape(inner, inner_shape);

  outer = grouped_outer;
  inner = grouped_inner;

  arith::Analyzer analyzer;

  {
    // Scale outer axis strides by inner_cosize_map
    Map<String, PrimExpr> inner_cosize_map;
    for (const auto& iter : inner->shard) {
      if (inner_cosize_map.find(iter->axis->name) == inner_cosize_map.end()) {
        inner_cosize_map.Set(iter->axis->name, inner->GetCosize(iter->axis->name));
      }
    }
    std::vector<Iter> new_shard;
    for (size_t i = 0; i < outer->shard.size(); ++i) {
      auto it = inner_cosize_map.find(outer->shard[i]->axis->name);
      if (it != inner_cosize_map.end()) {
        new_shard.push_back(Iter(outer->shard[i]->extent, outer->shard[i]->stride * (*it).second,
                                 outer->shard[i]->axis));
      } else {
        new_shard.push_back(outer->shard[i]);
      }
    }
    outer = TileLayout(new_shard, outer->replicate, outer->exclude, outer->subscope, outer->scope);
  }

  CHECK(!outer_seps.empty()) << "Outer layout must only use split/reorder from logical scope";
  CHECK(!inner_seps.empty()) << "Inner layout must only use split/reorder from logical scope";

  // Combine the shards from both layouts for each dimension
  std::vector<Iter> tile_shard;
  for (size_t i = 0; i < outer_shape.size(); ++i) {
    // Add outer layout's shards for this dimension
    tile_shard.insert(tile_shard.end(), outer->shard.begin() + outer_seps[i],
                      outer->shard.begin() + outer_seps[i + 1]);

    // Add inner layout's shards for this dimension
    tile_shard.insert(tile_shard.end(), inner->shard.begin() + inner_seps[i],
                      inner->shard.begin() + inner_seps[i + 1]);
  }

  // Combine replicate attributes from both layouts
  std::vector<Iter> tile_rep{inner->replicate.begin(), inner->replicate.end()};
  tile_rep.insert(tile_rep.end(), outer->replicate.begin(), outer->replicate.end());

  // Combine exclude attributes from both layouts
  std::vector<Tuple<Iter, PrimExpr>> tile_offset;
  for (const auto& iter_selector : inner->exclude) {
    tile_offset.push_back(iter_selector);
  }
  for (const auto& iter_selector : outer->exclude) {
    tile_offset.push_back(iter_selector);
  }

  Optional<ExecScope> tile_subscope;
  Optional<ExecScope> tile_scope;

  if (inner->subscope.defined()) {
    tile_subscope = inner->subscope;
    tile_scope = outer->subscope.defined() ? outer->scope : inner->scope;
    if (outer->subscope.defined()) {
      CHECK(outer->subscope.value()->Is(inner->scope.value()));
    }
  } else if (outer->subscope.defined()) {
    tile_subscope = outer->subscope;
    tile_scope = outer->scope;
  }
  return TileLayout(tile_shard, tile_rep, tile_offset, tile_subscope, tile_scope)->Normalize();
}

// Tiles a logical shape by a given factor array.
Array<PrimExpr> TileShape(Array<PrimExpr> shape, Array<PrimExpr> factor, bool is_inner) {
  ICHECK_EQ(shape.size(), factor.size()) << "Shape and factor dimension must match.";
  arith::Analyzer analyzer;

  Array<PrimExpr> new_shape;
  for (int i = 0; i < static_cast<int>(shape.size()); ++i) {
    ICHECK(analyzer.CanProveEqual(floormod(shape[i], factor[i]), 0))
        << "Shape[i] must be divisible by factor[i]";

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

// Extract every even index from seps
std::vector<int64_t> GetEvenSeps(std::vector<int64_t> seps) {
  std::vector<int64_t> even;
  for (size_t i = 0; i < seps.size(); i += 2) {
    even.push_back(seps[i]);
  }
  return even;
}

Optional<TileLayout> TileLayoutNode::IsTileInner(const TLayout& tile_layout,
                                                 const Array<PrimExpr>& tiled_shape,
                                                 const Array<PrimExpr>& inner_shape) const {
  auto maybe_tile = tile_layout.as<TileLayout>();
  if (!maybe_tile) return std::nullopt;

  TileLayout tiled = maybe_tile.value()->Normalize().as<TileLayout>().value();
  TileLayout layout = GetRef<TileLayout>(this)->Normalize().as<TileLayout>().value();
  arith::Analyzer analyzer;

  CHECK_EQ(tiled_shape.size(), inner_shape.size())
      << "Tiled shape size must match inner shape size";

  auto factored = TileShape(tiled_shape, inner_shape, true);
  auto [grouped_tiled, tiled_seps] = GroupByShape(tiled, factored);
  CHECK(grouped_tiled.defined() && !tiled_seps.empty())
      << "tile layout group by shape failed, layout is " << tiled << " and shape is " << factored;
  auto [grouped_layout, inner_seps] = GroupByShape(layout, inner_shape);
  CHECK(grouped_layout.defined() && !inner_seps.empty())
      << "tile layout group by shape failed, layout is " << layout << " and shape is "
      << inner_shape;

  auto tiled_seps_even = GetEvenSeps(tiled_seps);
  std::vector<Iter> outer_shard;
  PrimExpr inner_stride = grouped_layout->GetCosize();

  for (size_t i = 0; i < tiled_shape.size(); ++i) {
    int inner_count = inner_seps[i + 1] - inner_seps[i];
    int tiled_count = tiled_seps_even[i + 1] - tiled_seps_even[i];
    if (inner_count > tiled_count) return std::nullopt;

    // Compare extents (and stride if extent is not 1).
    for (int j = 0; j < inner_count; ++j) {
      Iter inner_iter = grouped_layout->shard[inner_seps[i] + j];
      Iter tiled_iter = grouped_tiled->shard[tiled_seps_even[i + 1] - inner_count + j];
      if (!analyzer.CanProveEqual(inner_iter->extent, tiled_iter->extent) ||
          (!analyzer.CanProveEqual(inner_iter->stride, tiled_iter->stride) &&
           !is_one(inner_iter->extent)) ||
          !inner_iter->axis.same_as(tiled_iter->axis)) {
        return std::nullopt;
      }
    }

    for (int j = 0; j < tiled_count - inner_count; ++j) {
      Iter outer_iter = grouped_tiled->shard[tiled_seps_even[i] + j];
      outer_shard.push_back(
          Iter(outer_iter->extent, floordiv(outer_iter->stride, inner_stride), outer_iter->axis));
    }
  }
  // TODO(@bohan): replicate and exclude should be considered here
  return TileLayout(outer_shard, this->replicate, this->exclude);
}

Optional<TLayout> TileLayoutNode::IsTileOuter(const TLayout& tile_layout,
                                              const Array<PrimExpr>& tiled_shape,
                                              const Array<PrimExpr>& outer_shape) const {
  auto maybe_tile = tile_layout.as<TileLayout>();
  if (!maybe_tile) {
    // Could be ComposeLayout, in which case we test layout_B of compose.
    if (auto comp = tile_layout.as<ComposeLayout>()) {
      auto inner_layout = IsTileOuter(comp.value()->layout_B, tiled_shape, outer_shape);
      if (!inner_layout) return std::nullopt;
      return ComposeLayout(comp.value()->layout_A, inner_layout.value().as<TileLayout>().value());
    }
    return std::nullopt;
  }
  TileLayout tiled = maybe_tile.value()->Normalize().as<TileLayout>().value();
  TileLayout layout = GetRef<TileLayout>(this)->Normalize().as<TileLayout>().value();
  arith::Analyzer analyzer;

  CHECK_EQ(tiled_shape.size(), outer_shape.size())
      << "Tiled shape size must match outer shape size";

  auto factored = TileShape(tiled_shape, outer_shape, false);
  auto [grouped_tiled, tiled_seps] = GroupByShape(tiled, factored);
  CHECK(grouped_tiled.defined() && !tiled_seps.empty())
      << "tile layout group by shape failed, layout is " << tiled << " and shape is " << factored;
  auto [grouped_layout, outer_seps] = GroupByShape(layout, outer_shape);
  CHECK(grouped_layout.defined() && !outer_seps.empty())
      << "tile layout group by shape failed, layout is " << layout << " and shape is "
      << outer_shape;

  auto tiled_seps_even = GetEvenSeps(tiled_seps);
  std::vector<Iter> inner_shard;

  for (size_t i = 0; i < tiled_shape.size(); ++i) {
    int outer_count = outer_seps[i + 1] - outer_seps[i];
    int tiled_count = tiled_seps_even[i + 1] - tiled_seps_even[i];
    if (outer_count > tiled_count) return std::nullopt;

    // Compare extents (delay checking stride since we don't know inner_stride yet)
    for (int j = 0; j < outer_count; ++j) {
      Iter outer_iter = grouped_layout->shard[outer_seps[i] + j];
      Iter tiled_iter = grouped_tiled->shard[tiled_seps_even[i] + j];
      if (!analyzer.CanProveEqual(outer_iter->extent, tiled_iter->extent) ||
          !outer_iter->axis.same_as(tiled_iter->axis)) {
        return std::nullopt;
      }
    }

    for (int j = 0; j < tiled_count - outer_count; ++j) {
      Iter inner_iter = grouped_tiled->shard[tiled_seps_even[i] + outer_count + j];
      inner_shard.push_back(inner_iter);
    }
  }
  // TODO(@bohan): replicate and exclude should be considered here
  auto res = TileLayout(inner_shard, this->replicate, this->exclude, tiled->subscope, tiled->scope);
  PrimExpr inner_stride = res->GetCosize();
  // Check if the stride of the outer shard is correct
  for (size_t i = 0; i < tiled_shape.size(); ++i) {
    for (int j = outer_seps[i]; j < outer_seps[i + 1]; ++j) {
      if (!analyzer.CanProveEqual(
              grouped_layout->shard[j]->stride * inner_stride,
              grouped_tiled->shard[tiled_seps_even[i] + j - outer_seps[i]]->stride)) {
        return std::nullopt;
      }
    }
  }
  return res;
}

Array<PrimExpr> TileLayoutNode::GetShardShape() const {
  return shard.Map([](const Iter& iter) { return iter->extent; });
}

bool TileLayoutNode::IsTrivial() const {
  if (shard.size() > 1) return false;
  if (shard.size() == 1) {
    if (!shard[0]->axis->IsMemoryAxis() || !is_one(shard[0]->stride)) return false;
  }
  return replicate.size() == 0 && exclude.size() == 0 && !subscope.defined() && !scope.defined();
}

TVM_REGISTER_GLOBAL("tir.TileLayoutIsTrivial").set_body_typed([](const TileLayout& layout) {
  return layout->Normalize().as<TileLayout>().value()->IsTrivial();
});

bool TileLayoutNode::IsTrainium() const {
  return !std::any_of(shard.begin(), shard.end(), [](const Iter& iter) {
    return iter->axis->IsMemoryAxis() && !iter->axis.same_as(Axis::Get("F")) &&
           !iter->axis.same_as(Axis::Get("P")) && !iter->axis.same_as(Axis::Get("Bank"));
  });
}

TVM_REGISTER_GLOBAL("tir.TileLayoutIsTrainium").set_body_typed([](const TileLayout& layout) {
  return layout->IsTrainium();
});

bool TileLayoutNode::HasMemoryAxis() const {
  return std::any_of(shard.begin(), shard.end(),
                     [](const Iter& iter) { return iter->axis->IsMemoryAxis(); });
}

bool TileLayoutNode::HasThreadAxis() const {
  return std::any_of(shard.begin(), shard.end(),
                     [](const Iter& iter) { return iter->axis->IsThreadAxis(); });
}

}  // namespace tir
}  // namespace tvm
