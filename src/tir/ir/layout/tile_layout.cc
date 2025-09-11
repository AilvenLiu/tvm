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

TVM_FFI_STATIC_INIT_BLOCK({
  AxisNode::RegisterReflection();
  IterNode::RegisterReflection();
  TileLayoutNode::RegisterReflection();
  SwizzleLayoutNode::RegisterReflection();
  ComposeLayoutNode::RegisterReflection();
});

/**************** Axis ****************/
// AxisNode
ObjectPtr<Object> CreateAxis(const std::string& name) {
  // Hack use ffi::Any as exchange
  auto axis = Axis::Get(name);
  ICHECK(axis.defined()) << "Cannot find axis \'" << name << '\'';
  return ffi::details::ObjectUnsafe::ObjectPtrFromObjectRef<Object>(axis);
}

bool AxisNode::IsThreadAxis() const {
  static const auto& thread_attr_map = Axis::GetAttrMap<bool>("thread");
  return thread_attr_map[ffi::GetRef<Axis>(this)];
}

bool AxisNode::IsMemoryAxis() const {
  static const auto& thread_attr_map = Axis::GetAttrMap<bool>("thread");
  return !thread_attr_map[ffi::GetRef<Axis>(this)];
}

ffi::Optional<ExecScope> AxisNode::GetScope() const {
  static const auto& scope_attr_map = Axis::GetAttrMap<ffi::Optional<ExecScope>>("scope");
  return scope_attr_map.get(ffi::GetRef<Axis>(this), std::nullopt);
}

ffi::Optional<ExecScope> AxisNode::GetSubscope() const {
  static const auto& subscope_attr_map = Axis::GetAttrMap<ffi::Optional<ExecScope>>("subscope");
  return subscope_attr_map.get(ffi::GetRef<Axis>(this), std::nullopt);
}

ffi::Optional<FAxisFuser> AxisNode::GetFuser() const {
  static const auto& fuser_attr_map = Axis::GetAttrMap<ffi::Optional<FAxisFuser>>("fuser");
  return fuser_attr_map.get(ffi::GetRef<Axis>(this), std::nullopt);
}

ffi::Optional<FAxisSplitter> AxisNode::GetSplitter() const {
  static const auto& splitter_attr_map = Axis::GetAttrMap<ffi::Optional<FAxisSplitter>>("splitter");
  return splitter_attr_map.get(ffi::GetRef<Axis>(this), std::nullopt);
}

TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tir.AxisIsThreadAxis", [](Axis axis) { return axis->IsThreadAxis(); });
});

TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tir.AxisIsMemoryAxis", [](Axis axis) { return axis->IsMemoryAxis(); });
});

TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tir.AxisGetScope", [](Axis axis) { return axis->GetScope(); });
});

TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tir.AxisGetSubscope", [](Axis axis) { return axis->GetSubscope(); });
});

// Axis
Axis Axis::Get(const ffi::String& name) {
  const AxisRegEntry* reg = AxisRegistry::Global()->Get(name);
  CHECK(reg != nullptr) << "Axis " << name << " is not registered";
  return reg->axis_;
}

template <typename ValueType>
inline AxisAttrMap<ValueType> Axis::GetAttrMap(const ffi::String& attr_name) {
  return AxisAttrMap<ValueType>(AxisRegistry::Global()->GetAttrMap(attr_name));
}

// AxisRegEntry
inline AxisNode* AxisRegEntry::get() { return const_cast<AxisNode*>(axis_.operator->()); }

AxisRegEntry::AxisRegEntry(uint32_t index) {
  ObjectPtr<AxisNode> n = ffi::make_object<AxisNode>();
  n->index_ = index;
  axis_ = Axis(n);
}

AxisRegEntry& AxisRegEntry::RegisterOrGet(const ffi::String& name) {
  auto& entry = AxisRegistry::Global()->RegisterOrGet(name);
  entry.get()->name = name;
  return entry;
}

ffi::Array<ffi::String> AxisRegEntry::ListAxisNames() {
  return AxisRegistry::Global()->ListAllNames();
}

template <typename ValueType>
inline AxisRegEntry& AxisRegEntry::set_attr(const ffi::String& key, const ValueType& value,
                                            int plevel) {
  ICHECK_GT(plevel, 0) << "plevel in set_attr must be greater than 0";
  ffi::Any rv;
  rv = value;
  UpdateAttr(key, rv, plevel);
  return *this;
}

AxisRegEntry& AxisRegEntry::set_scope(const ffi::String& scope_name, int plevel) {
  set_attr<ffi::Optional<ExecScope>>("scope", ExecScope::Create(scope_name), plevel);
  return *this;
}

AxisRegEntry& AxisRegEntry::set_subscope(const ffi::String& subscope_name, int plevel) {
  set_attr<ffi::Optional<ExecScope>>("subscope", ExecScope::Create(subscope_name), plevel);
  return *this;
}

AxisRegEntry& AxisRegEntry::set_fuser(const FAxisFuser& fuser) {
  set_attr<ffi::Optional<FAxisFuser>>("fuser", fuser);
  return *this;
}

AxisRegEntry& AxisRegEntry::set_splitter(const FAxisSplitter& splitter) {
  set_attr<ffi::Optional<FAxisSplitter>>("splitter", splitter);
  return *this;
}

void AxisRegEntry::UpdateAttr(const ffi::String& key, ffi::Any value, int plevel) {
  AxisRegistry::Global()->UpdateAttr(key, axis_, value, plevel);
}

// register theaad axis
ffi::Array<Iter> SplitterGen(const Iter& iter, const Axis& axis_outer, const Axis& axis_inner,
                             const PrimExpr& e_inner) {
  arith::Analyzer analyzer;
  if (analyzer.CanProve(iter->extent * iter->stride < e_inner)) {
    return {Iter(iter->extent, iter->stride, axis_inner)};
  } else if (analyzer.CanProveEqual(floormod(e_inner, iter->stride), 0) &&
             analyzer.CanProveEqual(floormod(iter->extent * iter->stride, e_inner), 0)) {
    const auto& d = analyzer.Simplify(floordiv(e_inner, iter->stride));
    const auto& c = analyzer.Simplify(floordiv(iter->extent, d));
    return {Iter(c, IntImm(e_inner.dtype(), 1), axis_outer), Iter(d, iter->stride, axis_inner)};
  } else if (analyzer.CanProveEqual(floormod(iter->stride, e_inner), 0)) {
    const auto& d = analyzer.Simplify(floordiv(iter->stride, e_inner));
    return {Iter(iter->extent, d, axis_outer)};
  }
  return {};
}

TVM_REGISTER_AXIS("pid").set_attr<bool>("thread", true).set_scope("world").set_subscope("kernel");
TVM_REGISTER_AXIS("bx").set_attr<bool>("thread", true).set_scope("kernel").set_subscope("cta");
TVM_REGISTER_AXIS("by").set_attr<bool>("thread", true).set_scope("kernel").set_subscope("cta");
TVM_REGISTER_AXIS("bz").set_attr<bool>("thread", true).set_scope("kernel").set_subscope("cta");
TVM_REGISTER_AXIS("cbx").set_attr<bool>("thread", true).set_scope("cluster").set_subscope("cta");
TVM_REGISTER_AXIS("cby").set_attr<bool>("thread", true).set_scope("cluster").set_subscope("cta");
TVM_REGISTER_AXIS("cbz").set_attr<bool>("thread", true).set_scope("cluster").set_subscope("cta");
TVM_REGISTER_AXIS("tx")
    .set_attr<bool>("thread", true)
    .set_scope("cta")
    .set_subscope("thread")
    .set_fuser([](Target target, ffi::String subscope, ffi::String scope,
                  Iter iter) -> ffi::Optional<Iter> {
      if (target->kind->default_device_type == kDLCUDA) {
        return std::nullopt;
      }
      return std::nullopt;
    })
    .set_splitter([](Target target, ffi::String scope, Iter iter) -> ffi::Array<Iter> {
      arith::Analyzer analyzer;
      if (target->kind->default_device_type == kDLCUDA) {
        if (scope == "warp") {
          // tx -> warpid, laneid
          return SplitterGen(iter, Axis::Get("warpid"), Axis::Get("laneid"), 32);
        } else if (scope == "warpgroup") {
          // tx -> wgid, tid_in_wg
          return SplitterGen(iter, Axis::Get("wgid"), Axis::Get("tid_in_wg"), 128);
        }
        LOG(FATAL) << "Cannot split cta->thread axis into cta->" << scope << "->thread";
      }
      return {};
    });
TVM_REGISTER_AXIS("warpid")
    .set_attr<bool>("thread", true)
    .set_scope("cta")
    .set_subscope("warp")
    .set_fuser([](Target target, ffi::String subscope, ffi::String scope,
                  Iter iter) -> ffi::Optional<Iter> {
      if (target->kind->default_device_type == kDLCUDA) {
        // cta->warp ===> cta->thread (tx)
        if (subscope == "thread" && scope == "cta") {
          return Iter(iter->extent, 32 * iter->stride, Axis::Get("tx"));
        }
        return std::nullopt;
      }
      return std::nullopt;
    })
    .set_splitter([](Target target, ffi::String scope, Iter iter) -> ffi::Array<Iter> {
      arith::Analyzer analyzer;
      if (target->kind->default_device_type == kDLCUDA) {
        if (scope == "warp") {
          // warpid -> wgid, wid_in_wg
          return SplitterGen(iter, Axis::Get("wgid"), Axis::Get("wid_in_wg"), 4);
        }
        LOG(FATAL) << "Cannot split cta->warp axis into cta->" << scope << "->warp";
      }
      return {};
    });
TVM_REGISTER_AXIS("laneid")
    .set_attr<bool>("thread", true)
    .set_scope("warp")
    .set_subscope("thread")
    .set_fuser([](Target target, ffi::String subscope, ffi::String scope,
                  Iter iter) -> ffi::Optional<Iter> {
      if (target->kind->default_device_type == kDLCUDA) {
        if (subscope == "thread" && scope == "warpgroup") {
          // warp->thread ===> warpgroup->thread (tid_in_wg)
          return Iter(iter->extent, iter->stride, Axis::Get("tid_in_wg"));
        } else if (subscope == "thread" && scope == "cta") {
          // warp->thread ===> cta->thread (tx)
          return Iter(iter->extent, iter->stride, Axis::Get("tx"));
        }
        return std::nullopt;
      }
      return std::nullopt;
    })
    .set_splitter([](Target target, ffi::String scope, Iter iter) -> ffi::Array<Iter> {
      arith::Analyzer analyzer;
      if (target->kind->default_device_type == kDLCUDA) {
        LOG(FATAL) << "laneid can not be split any more";
      }
      return {};
    });
TVM_REGISTER_AXIS("wgid")
    .set_attr<bool>("thread", true)
    .set_scope("cta")
    .set_subscope("warpgroup")
    .set_fuser([](Target target, ffi::String subscope, ffi::String scope,
                  Iter iter) -> ffi::Optional<Iter> {
      if (target->kind->default_device_type == kDLCUDA) {
        if (subscope == "thread" && scope == "cta") {
          // cta->warpgroup ===> cta->thread (tx)
          return Iter(iter->extent, iter->stride * 128, Axis::Get("tx"));
        } else if (subscope == "warp" && scope == "cta") {
          // cta->warpgroup ===> cta->warp (warpid)
          return Iter(iter->extent, iter->stride * 4, Axis::Get("wgid"));
        }
      }
      return std::nullopt;
    })
    .set_splitter([](Target target, ffi::String scope, Iter iter) -> ffi::Array<Iter> {
      arith::Analyzer analyzer;
      if (target->kind->default_device_type == kDLCUDA) {
        LOG(FATAL) << "wgid can not be split any more";
      }
      return {};
    });
TVM_REGISTER_AXIS("tid_in_wg")
    .set_attr<bool>("thread", true)
    .set_scope("warpgroup")
    .set_subscope("thread")
    .set_fuser([](Target target, ffi::String subscope, ffi::String scope,
                  Iter iter) -> ffi::Optional<Iter> {
      if (target->kind->default_device_type == kDLCUDA) {
        if (subscope == "thread" && scope == "cta") {
          // warpgroup->thread ===> cta->thread (tx)
          return Iter(iter->extent, iter->stride, Axis::Get("tx"));
        }
        return std::nullopt;
      }
      return std::nullopt;
    })
    .set_splitter([](Target target, ffi::String scope, Iter iter) -> ffi::Array<Iter> {
      arith::Analyzer analyzer;
      if (target->kind->default_device_type == kDLCUDA) {
        if (scope == "warp") {
          // tid_in_wg -> wid_in_wg, laneid
          return SplitterGen(iter, Axis::Get("wid_in_wg"), Axis::Get("laneid"), 32);
        }
        LOG(FATAL) << "Cannot split warpgroup->thread axis into warpgroup->" << scope << "->thread";
      }
      return {};
    });
TVM_REGISTER_AXIS("wid_in_wg")
    .set_attr<bool>("thread", true)
    .set_scope("warpgroup")
    .set_subscope("warp")
    .set_fuser([](Target target, ffi::String subscope, ffi::String scope,
                  Iter iter) -> ffi::Optional<Iter> {
      if (target->kind->default_device_type == kDLCUDA) {
        if (subscope == "thread" && scope == "warpgroup") {
          // warpgroup->warp ===> warpgroup->thread (tid_in_wg)
          return Iter(iter->extent, iter->stride * 32, Axis::Get("tid_in_wg"));
        } else if (subscope == "thread" && scope == "cta") {
          // warpgroup->warp ===> cta->thread (tx)
          return Iter(iter->extent, iter->stride * 32, Axis::Get("tx"));
        } else if (subscope == "warp" && scope == "cta") {
          // warpgroup->warp ===> cta->warp (warpid)
          return Iter(iter->extent, iter->stride, Axis::Get("warpid"));
        }
        return std::nullopt;
      }
      return std::nullopt;
    })
    .set_splitter([](Target target, ffi::String scope, Iter iter) -> ffi::Array<Iter> {
      arith::Analyzer analyzer;
      if (target->kind->default_device_type == kDLCUDA) {
        LOG(FATAL) << "wid_in_wg can not be split any more";
      }
      return {};
    });

// register memory axis
TVM_REGISTER_AXIS("m").set_attr<bool>("thread", false);
TVM_REGISTER_AXIS("P").set_attr<bool>("thread", false);
TVM_REGISTER_AXIS("F").set_attr<bool>("thread", false);
TVM_REGISTER_AXIS("Bank").set_attr<bool>("thread", false);
TVM_REGISTER_AXIS("TCol").set_attr<bool>("thread", false);
TVM_REGISTER_AXIS("TLane").set_attr<bool>("thread", false);

TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tir.AxisGet", [](ffi::String name) -> Axis { return Axis::Get(name); });
});

/**************** Iter ****************/
Iter::Iter(PrimExpr extent, PrimExpr stride, Axis axis) {
  auto n = ffi::make_object<IterNode>();
  n->extent = extent;
  n->stride = stride;
  n->axis = axis;
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tir.Iter", [](PrimExpr extent, PrimExpr stride, Axis axis) {
    return Iter(extent, stride, axis);
  });
});

/**************** TileLayout ****************/

TileLayout::TileLayout(ffi::Array<Iter> shard, ffi::Array<Iter> replicate,
                       ffi::Map<Axis, PrimExpr> exclude) {
  auto n = ffi::make_object<TileLayoutNode>();
  n->shard = shard;
  n->replicate = replicate;
  n->exclude = exclude;
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tir.TileLayout", [](ffi::Array<Iter> shard, ffi::Array<Iter> replicate,
                                             ffi::Map<Axis, PrimExpr> exclude) {
    return TileLayout(shard, replicate, exclude);
  });
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
  // // 1. For thread axes, verify its compactness
  // std::unordered_map<String, std::vector<Iter>> thread_axes;
  // auto collect_thread_axis = [&thread_axes](const Iter& iter) {
  //   if (iter->axis->IsThreadAxis()) {
  //     thread_axes[iter->axis->name].push_back(iter);
  //   }
  // };
  // for (const auto& iter : shard) {
  //   collect_thread_axis(iter);
  // }
  // for (const auto& iter : replicate) {
  //   collect_thread_axis(iter);
  // }
  // for (const auto& [axis, iters] : thread_axes) {
  //   if (!VerifyCompactness(iters)) {
  //     return false;
  //   }
  // }
  // 1. Check if the scope is connected
  if (!GetScope().defined() && HasThreadAxis()) {
    return false;
  }
  return true;
}

PrimExpr TileLayoutNode::GetSize(ffi::Optional<ffi::String> axis_name) const {
  auto filter = [&](const Iter& iter, PrimExpr acc) {
    if (!axis_name.has_value() || iter->axis->name == axis_name.value()) {
      return acc * iter->extent;
    }
    return acc;
  };
  PrimExpr res = 1;
  for (const auto& iter : shard) {
    res = filter(iter, res);
  }
  return res;
}

PrimExpr TileLayoutNode::GetCosize(ffi::Optional<ffi::String> axis_name) const {
  arith::Analyzer analyzer;
  PrimExpr result = 1;
  auto filter = [&](const Axis& axis) {
    return (!axis_name.has_value() && axis->IsMemoryAxis()) ||
           (axis_name.has_value() && axis->name == axis_name.value());
  };

  for (const auto& iter : shard) {
    if (filter(iter->axis)) result += (iter->extent - 1) * iter->stride;
  }
  for (const auto& iter : replicate) {
    if (filter(iter->axis)) result += (iter->extent - 1) * iter->stride;
  }
  for (const auto& [axis, offset] : exclude) {
    if (filter(axis)) result += offset;
  }
  return analyzer.Simplify(result);
}

ffi::Map<ffi::String, PrimExpr> TileLayoutNode::Apply(PrimExpr coord) const {
  return Apply(SplitCoord(coord, GetShardShape()));
}

ffi::Map<ffi::String, PrimExpr> TileLayoutNode::Apply(Array<PrimExpr> coord) const {
  arith::Analyzer analyzer;
  CHECK_EQ(coord.size(), shard.size()) << "Coordinate size must match the number of shard axes";
  std::unordered_map<ffi::String, PrimExpr> result;
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
  return ffi::GetRef<TileLayout>(new_layout);
}

TileLayout RemoveZeroOffset(TileLayout layout) {
  auto new_layout = layout.CopyOnWrite();
  ffi::Map<Axis, PrimExpr> exclude;
  for (const auto& [axis, offset] : layout->exclude) {
    if (!is_zero(offset)) {
      exclude.Set(axis, offset);
    }
  }
  new_layout->exclude = exclude;
  return ffi::GetRef<TileLayout>(new_layout);
}

TileLayout FuseShardIters(TileLayout layout) {
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
  return ffi::GetRef<TileLayout>(new_layout);
}

TileLayout TryFuseAxes(TileLayout layout) {
  // Step 1: Get the target and scope information
  auto scope_pair_opt = layout->GetScope();
  Target target = Target::Current();
  if (!scope_pair_opt.has_value() || !target.defined()) {
    return layout;
  }
  auto subscope = scope_pair_opt.value().get<0>()->name;
  auto scope = scope_pair_opt.value().get<1>()->name;

  // Step 2: Create vectors for the new layout components
  std::vector<Iter> shard;
  std::vector<Iter> replicate;
  ffi::Map<Axis, PrimExpr> exclude;

  // Step 3: Define the axis fusion function
  auto try_fuse_axis = [&](const Iter& iter) -> Iter {
    const auto& fuser = iter->axis->GetFuser();
    return fuser.has_value() ? fuser.value()(target, subscope, scope, iter).value_or(iter) : iter;
  };

  // Step 4: Process shard iterators
  for (auto iter : layout->shard) {
    shard.push_back(try_fuse_axis(iter));
  }
  // Step 5: Process replicate iterators
  for (auto iter : layout->replicate) {
    replicate.push_back(try_fuse_axis(iter));
  }
  // Step 6: Process exclude iterators
  for (auto [axis, offset] : layout->exclude) {
    Iter iter = try_fuse_axis(Iter(1, offset, axis));
    exclude.Set(iter->axis, iter->stride);
  }
  // Step 7: Create and return the new layout
  auto result = TileLayout(shard, replicate, exclude);
  return result;
}

TileLayout SortReplicateIters(TileLayout layout) {
  auto n = layout.CopyOnWrite();
  std::vector<Iter> replicate(n->replicate.begin(), n->replicate.end());
  auto hash_compare = [](const auto& a, const auto& b) {
    return StructuralHash()(a) < StructuralHash()(b);
  };
  std::sort(replicate.begin(), replicate.end(), hash_compare);
  n->replicate = std::move(replicate);
  return ffi::GetRef<TileLayout>(n);
}

TLayout TileLayoutNode::Normalize() const {
  // 0. Remove unit iters in shard
  TileLayout res = RemoveUnitIters(ffi::GetRef<TileLayout>(this));
  // 1. Remove zero offset in exclude
  res = RemoveZeroOffset(res);
  // 2. Try fuse axes
  res = TryFuseAxes(res);
  // 3. Fuse shard iters
  res = FuseShardIters(res);
  // 3. Sort replicate iters
  res = SortReplicateIters(res);
  return res;
}

std::pair<TileLayout, std::vector<int64_t>> GroupByShape(TileLayout layout,
                                                         const ffi::Array<PrimExpr>& shape) {
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
  return {ffi::GetRef<TileLayout>(n), seps};
}

TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "tir.TileLayoutGroupByShape", [](const TileLayout& layout, const Array<PrimExpr>& shape) {
        auto [res, seps] = GroupByShape(layout, shape);
        return Tuple<TileLayout, Array<int64_t>>{res, Array<int64_t>(seps.begin(), seps.end())};
      });
});

TLayout TileLayoutNode::Tile(const TileLayout& outer_in, const Array<PrimExpr>& outer_shape,
                             const Array<PrimExpr>& inner_shape) const {
  auto outer = outer_in->Normalize().as<TileLayout>().value();
  auto inner = ffi::GetRef<TileLayout>(this)->Normalize().as<TileLayout>().value();

  CHECK_EQ(outer_shape.size(), inner_shape.size()) << "Outer and inner shape size must match";

  // Group both outer/inner layouts by their respective logical shapes
  auto [grouped_outer, outer_seps] = GroupByShape(outer, outer_shape);
  auto [grouped_inner, inner_seps] = GroupByShape(inner, inner_shape);

  outer = grouped_outer;
  inner = grouped_inner;

  arith::Analyzer analyzer;

  {
    // Scale outer axis strides by inner_cosize_map
    ffi::Map<ffi::String, PrimExpr> inner_cosize_map;
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
    outer = TileLayout(new_shard, outer->replicate, outer->exclude);
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
  ffi::Map<Axis, PrimExpr> tile_offset;
  for (const auto& [axis, offset] : inner->exclude) {
    tile_offset.Set(axis, offset);
  }
  for (const auto& [axis, offset] : outer->exclude) {
    auto it = tile_offset.find(axis);
    if (it != tile_offset.end()) {
      tile_offset.Set(axis, (*it).second + offset);
    } else {
      tile_offset.Set(axis, offset);
    }
  }

  return TileLayout(tile_shard, tile_rep, tile_offset)->Normalize();
}

// Tiles a logical shape by a given factor array.
ffi::Array<PrimExpr> TileShape(ffi::Array<PrimExpr> shape, ffi::Array<PrimExpr> factor,
                               bool is_inner) {
  ICHECK_EQ(shape.size(), factor.size()) << "Shape and factor dimension must match.";
  arith::Analyzer analyzer;

  ffi::Array<PrimExpr> new_shape;
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

ffi::Array<PrimExpr> ShapeDiv(ffi::Array<PrimExpr> shape, ffi::Array<PrimExpr> factor) {
  ffi::Array<PrimExpr> new_shape;
  for (int i = 0; i < static_cast<int>(shape.size()); ++i) {
    new_shape.push_back(floordiv(shape[i], factor[i]));
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

TileLayout SplitAxes(TileLayout layout, const ffi::String& split_scope) {
  Target target = Target::Current();
  if (!target.defined()) {
    return layout;
  }
  auto split_iter = [&](const Iter& iter) -> ffi::Array<Iter> {
    const auto& splitter = iter->axis->GetSplitter();
    if (splitter.has_value()) {
      return splitter.value()(target, split_scope, iter);
    }
    return {iter};
  };

  std::vector<Iter> shard, replicate;
  ffi::Map<Axis, PrimExpr> exclude;

  for (const auto& iter : layout->shard) {
    auto split_iters = split_iter(iter);
    shard.insert(shard.end(), split_iters.begin(), split_iters.end());
  }

  for (const auto& iter : layout->replicate) {
    auto split_iters = split_iter(iter);
    replicate.insert(replicate.end(), split_iters.begin(), split_iters.end());
  }

  for (const auto& [axis, offset] : layout->exclude) {
    auto split_iters = split_iter(Iter(1, offset, axis));
    if (split_iters.size() == 1) {
      exclude.Set(split_iters[0]->axis, split_iters[0]->stride);
    } else {
      auto coord = SplitCoord(offset, {split_iters[0]->extent, split_iters[1]->extent});
      ICHECK(coord.size() == 2) << "Split coord size must be 2";
      exclude.Set(split_iters[0]->axis, coord[0] * split_iters[0]->stride);
      exclude.Set(split_iters[1]->axis, coord[1] * split_iters[1]->stride);
    }
  }

  return TileLayout(shard, replicate, exclude);
}

ffi::Optional<TileLayout> TileLayoutNode::IsTileInner(
    const TLayout& tile_layout, const ffi::Array<PrimExpr>& tiled_shape,
    const ffi::Array<PrimExpr>& inner_shape) const {
  auto maybe_tile = tile_layout.as<TileLayout>();
  if (!maybe_tile) return std::nullopt;

  TileLayout tiled = maybe_tile.value()->Normalize().as<TileLayout>().value();
  TileLayout layout = ffi::GetRef<TileLayout>(this)->Normalize().as<TileLayout>().value();

  auto tiled_scope = tiled->GetScope();
  auto inner_scope = layout->GetScope();
  if (tiled_scope.has_value() && inner_scope.has_value()) {
    if (!tiled_scope.value().get<0>()->Is(inner_scope.value().get<0>()) ||
        inner_scope.value().get<1>()->Higher(tiled_scope.value().get<1>())) {
      return std::nullopt;
    }
    if (tiled_scope.value().get<1>()->Higher(inner_scope.value().get<1>())) {
      tiled = SplitAxes(tiled, inner_scope.value().get<1>()->name);
    }
  }

  arith::Analyzer analyzer;
  // Get the cosize map of the inner layout of each axis
  ffi::Map<ffi::String, PrimExpr> inner_cosize_map;
  for (const auto& iter : layout->shard) {
    if (inner_cosize_map.find(iter->axis->name) == inner_cosize_map.end()) {
      inner_cosize_map.Set(iter->axis->name, layout->GetCosize(iter->axis->name));
    }
  }
  auto rescale_iter = [&](const Iter& iter) -> ffi::Optional<Iter> {
    auto it = inner_cosize_map.find(iter->axis->name);
    if (it != inner_cosize_map.end() && !is_one(iter->extent)) {
      if (!analyzer.CanProveEqual(floormod(iter->stride, (*it).second), 0)) {
        return std::nullopt;
      }
      return Iter(iter->extent, floordiv(iter->stride, (*it).second), iter->axis);
    }
    return iter;
  };

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

  // Gather outer shards
  std::vector<Iter> outer_shard;
  for (size_t i = 0; i < tiled_shape.size(); ++i) {
    int inner_count = inner_seps[i + 1] - inner_seps[i];
    int tiled_count = tiled_seps_even[i + 1] - tiled_seps_even[i];
    if (inner_count > tiled_count) return std::nullopt;

    // Compare extents (and stride/axis if extent is not 1).
    for (int j = 0; j < inner_count; ++j) {
      Iter inner_iter = grouped_layout->shard[inner_seps[i] + j];
      Iter tiled_iter = grouped_tiled->shard[tiled_seps_even[i + 1] - inner_count + j];
      if (!analyzer.CanProveEqual(inner_iter->extent, tiled_iter->extent) ||
          (!is_one(inner_iter->extent) &&
           !(analyzer.CanProveEqual(inner_iter->stride, tiled_iter->stride) &&
             inner_iter->axis.same_as(tiled_iter->axis)))) {
        // failure cases:
        // 1. extent doesn't match
        // 2. extent is not 1, and stride or axis doesn't match
        return std::nullopt;
      }
    }
    for (int j = 0; j < tiled_count - inner_count; ++j) {
      auto outer_iter = rescale_iter(grouped_tiled->shard[tiled_seps_even[i] + j]);
      if (!outer_iter.has_value()) return std::nullopt;
      outer_shard.push_back(outer_iter.value());
    }
  }

  // Gather outer replicate
  std::vector<Iter> outer_replicate;
  for (const auto& tiled_iter : tiled->replicate) {
    if (std::none_of(
            layout->replicate.begin(), layout->replicate.end(),
            [&](const Iter& inner_iter) { return StructuralEqual()(tiled_iter, inner_iter); })) {
      auto outer_iter = rescale_iter(tiled_iter);
      if (!outer_iter.has_value()) return std::nullopt;
      outer_replicate.push_back(outer_iter.value());
    }
  }
  // Gather outer exclude
  ffi::Map<Axis, PrimExpr> outer_exclude;
  for (const auto& [axis, offset] : tiled->exclude) {
    auto it = layout->exclude.find(axis);
    if (it != layout->exclude.end()) {
      outer_exclude.Set(axis, analyzer.Simplify(offset - (*it).second));
    } else {
      outer_exclude.Set(axis, offset);
    }
  }
  return TileLayout(outer_shard, outer_replicate, outer_exclude);
}

ffi::Optional<TLayout> TileLayoutNode::IsTileOuter(const TLayout& tile_layout,
                                                   const ffi::Array<PrimExpr>& tiled_shape,
                                                   const ffi::Array<PrimExpr>& outer_shape) const {
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
  TileLayout layout = ffi::GetRef<TileLayout>(this)->Normalize().as<TileLayout>().value();

  auto tiled_scope = tiled->GetScope();
  auto outer_scope = layout->GetScope();
  if (tiled_scope.has_value() && outer_scope.has_value()) {
    if (!tiled_scope.value().get<1>()->Is(outer_scope.value().get<1>()) ||
        tiled_scope.value().get<0>()->Higher(outer_scope.value().get<0>())) {
      return std::nullopt;
    }
    if (outer_scope.value().get<0>()->Higher(tiled_scope.value().get<0>())) {
      tiled = SplitAxes(tiled, outer_scope.value().get<0>()->name);
    }
  }

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

  // Gather inner shards
  std::vector<Iter> inner_shard;
  for (size_t i = 0; i < tiled_shape.size(); ++i) {
    int outer_count = outer_seps[i + 1] - outer_seps[i];
    int tiled_count = tiled_seps_even[i + 1] - tiled_seps_even[i];
    if (outer_count > tiled_count) return std::nullopt;

    // Failure cases:
    // 1. extent doesn't match
    // 2. extent is not 1, and axis doesn't match (we don't know inner cosize yet)
    for (int j = 0; j < outer_count; ++j) {
      Iter outer_iter = grouped_layout->shard[outer_seps[i] + j];
      Iter tiled_iter = grouped_tiled->shard[tiled_seps_even[i] + j];
      if (!analyzer.CanProveEqual(outer_iter->extent, tiled_iter->extent) ||
          (!is_one(outer_iter->extent) && !outer_iter->axis.same_as(tiled_iter->axis))) {
        return std::nullopt;
      }
    }

    for (int j = 0; j < tiled_count - outer_count; ++j) {
      Iter inner_iter = grouped_tiled->shard[tiled_seps_even[i] + outer_count + j];
      inner_shard.push_back(inner_iter);
    }
  }

  // Gather inner replicate
  std::vector<Iter> inner_replicate;
  for (const auto& tiled_iter : tiled->replicate) {
    if (std::none_of(
            layout->replicate.begin(), layout->replicate.end(),
            [&](const Iter& inner_iter) { return StructuralEqual()(tiled_iter, inner_iter); })) {
      inner_replicate.push_back(tiled_iter);
    }
  }

  // Gather inner exclude
  ffi::Map<Axis, PrimExpr> inner_exclude;
  for (const auto& [axis, offset] : tiled->exclude) {
    auto it = layout->exclude.find(axis);
    if (it != layout->exclude.end()) {
      inner_exclude.Set(axis, analyzer.Simplify(offset - (*it).second));
    } else {
      inner_exclude.Set(axis, offset);
    }
  }

  auto inner_layout = TileLayout(inner_shard, inner_replicate, inner_exclude);
  auto try_tile = inner_layout->Tile(layout, outer_shape, ShapeDiv(tiled_shape, outer_shape));
  if (StructuralEqual()(try_tile->Normalize(), tiled->Normalize())) {
    return inner_layout;
  }
  return std::nullopt;
}

ffi::Array<PrimExpr> TileLayoutNode::GetShardShape() const {
  return shard.Map([](const Iter& iter) { return iter->extent; });
}

bool TileLayoutNode::IsTrivial() const {
  if (shard.size() > 1) return false;
  if (shard.size() == 1) {
    if (!shard[0]->axis->IsMemoryAxis() || !is_one(shard[0]->stride)) return false;
  }
  return replicate.size() == 0 && exclude.size() == 0;
}

TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tir.TileLayoutIsTrivial", [](const TileLayout& layout) {
    return layout->Normalize().as<TileLayout>().value()->IsTrivial();
  });
});

bool TileLayoutNode::IsTrainium() const {
  return !std::any_of(shard.begin(), shard.end(), [](const Iter& iter) {
    return iter->axis->IsMemoryAxis() && !iter->axis.same_as(Axis::Get("F")) &&
           !iter->axis.same_as(Axis::Get("P")) && !iter->axis.same_as(Axis::Get("Bank"));
  });
}

TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tir.TileLayoutIsTrainium",
                        [](const TileLayout& layout) { return layout->IsTrainium(); });
});

bool TileLayoutNode::HasMemoryAxis() const {
  return std::any_of(shard.begin(), shard.end(),
                     [](const Iter& iter) { return iter->axis->IsMemoryAxis(); });
}

bool TileLayoutNode::HasThreadAxis() const {
  return std::any_of(shard.begin(), shard.end(),
                     [](const Iter& iter) { return iter->axis->IsThreadAxis(); });
}

ffi::Optional<ffi::Tuple<ExecScope, ExecScope>> TileLayoutNode::GetScope() const {
  if (!HasThreadAxis()) return std::nullopt;

  std::unordered_map<ffi::String, ffi::String> scope_map;
  ffi::Optional<ffi::String> inner_most;

  auto check_axis = [&](const Axis& axis) {
    if (!axis->IsThreadAxis()) return;

    auto subscope_opt = axis->GetSubscope();
    auto scope_opt = axis->GetScope();
    CHECK(subscope_opt.defined() && scope_opt.defined())
        << "Thread axis " << axis->name << " has no subscope or scope";

    ffi::String subscope = subscope_opt.value()->name;
    ffi::String scope = scope_opt.value()->name;

    if (!inner_most.has_value() ||
        ExecScope::Create(inner_most.value())->Higher(ExecScope::Create(subscope)))
      inner_most = subscope;

    auto it = scope_map.find(subscope);
    if (it == scope_map.end())
      scope_map[subscope] = scope;
    else
      CHECK_EQ(it->second, scope) << "Ill-formed tile layout: conflicting scopes for " << subscope;
  };

  for (const auto& iter : shard) check_axis(iter->axis);
  for (const auto& iter : replicate) check_axis(iter->axis);
  for (const auto& [axis, offset] : exclude) check_axis(axis);

  ffi::String outer_most = inner_most.value();
  size_t count = 0;
  for (auto it = scope_map.find(outer_most); it != scope_map.end();
       it = scope_map.find(outer_most)) {
    count++;
    outer_most = it->second;
  }

  CHECK_EQ(count, scope_map.size()) << "Ill-formed tile layout: disconnected scope chain";
  return Tuple<ExecScope, ExecScope>{ExecScope::Create(inner_most.value()),
                                     ExecScope::Create(outer_most)};
}

std::vector<PrimExpr> GetDefaultStrides(const ffi::Array<PrimExpr>& data,
                                        PrimExpr initial_stride = PrimExpr(1)) {
  if (data.empty()) {
    return {};
  }

  size_t n = data.size();
  std::vector<PrimExpr> strides(n);
  PrimExpr current_stride = initial_stride;

  for (int i = n - 1; i >= 0; --i) {
    strides[i] = current_stride;
    current_stride *= data[i];
  }

  return strides;
}

TileLayout TileLayoutNode::DefaultLayout(ffi::Array<PrimExpr> shape) {
  Array<Iter> shard;
  auto strides = GetDefaultStrides(shape);
  for (size_t i = 0; i < shape.size(); ++i) {
    shard.push_back(Iter(shape[i], strides[i], Axis::Get("m")));
  }
  return TileLayout(shard, ffi::Array<Iter>(), ffi::Map<Axis, PrimExpr>());
}

TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "tir.TileLayoutGetScope",
      [](const TileLayout& layout) -> ffi::Optional<ffi::Tuple<ExecScope, ExecScope>> {
        return layout->GetScope();
      });
});

}  // namespace tir
}  // namespace tvm
