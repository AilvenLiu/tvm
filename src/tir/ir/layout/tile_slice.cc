/*
 * Region slicing utilities for TileLayout.
 */
#include "tile_internal.h"

namespace tvm {
namespace tir {

// Slice a contiguous region [begin, begin+extent) over the grouped block (shard).
ffi::Optional<TileLayout> SlicePerGroup(TileLayout layout, PrimExpr begin, PrimExpr extent) {
  layout = layout->Canonicalize().as<TileLayout>().value();
  const auto& shard = layout->shard;
  if (shard.empty()) {
    return std::nullopt;
  }

  arith::Analyzer analyzer;

  int m = static_cast<int>(shard.size());
  std::vector<PrimExpr> B(m);
  PrimExpr acc = PrimExpr(1);
  for (int k = m - 1; k >= 0; --k) {
    B[k] = acc;
    acc = analyzer.Simplify(acc * shard[k]->extent);
  }

  std::vector<PrimExpr> d0(m);
  ffi::Map<Axis, PrimExpr> new_offset;
  for (const auto& [axis, off] : layout->offset) new_offset.Set(axis, off);

  auto add_axis_offset = [&](const Axis& axis, PrimExpr value) {
    auto it = new_offset.find(axis);
    if (it != new_offset.end()) {
      new_offset.Set(axis, analyzer.Simplify((*it).second + value));
    } else {
      new_offset.Set(axis, analyzer.Simplify(value));
    }
  };

  for (int k = 0; k < m; ++k) {
    const PrimExpr& Ek = shard[k]->extent;
    const PrimExpr& Sk = shard[k]->stride;
    const Axis& ak = shard[k]->axis;
    PrimExpr dk0 = analyzer.Simplify(floormod(floordiv(begin, B[k]), Ek));
    d0[k] = dk0;
    add_axis_offset(ak, analyzer.Simplify(dk0 * Sk));
  }

  PrimExpr rem = extent;
  std::vector<Iter> peeled_rev;
  int pivot = m - 1;
  for (; pivot >= 0; --pivot) {
    const PrimExpr& Ek = shard[pivot]->extent;
    bool peelable = analyzer.CanProveEqual(d0[pivot], 0) && analyzer.CanProveEqual(floormod(rem, Ek), 0);
    if (!peelable) break;
    peeled_rev.push_back(shard[pivot]);
    rem = analyzer.Simplify(floordiv(rem, Ek));
  }

  if (pivot < 0) {
    if (!analyzer.CanProveEqual(rem, 1)) return std::nullopt;
    std::vector<Iter> peeled_slow_to_fast(peeled_rev.rbegin(), peeled_rev.rend());
    return TileLayout(peeled_slow_to_fast, layout->replica, new_offset);
  }

  const PrimExpr& Ek = shard[pivot]->extent;
  const PrimExpr& Sk = shard[pivot]->stride;
  const Axis& ak = shard[pivot]->axis;

  if (analyzer.CanProve(d0[pivot] + rem <= Ek)) {
    std::vector<Iter> new_shard;
    new_shard.push_back(Iter(rem, Sk, ak));
    new_shard.insert(new_shard.end(), peeled_rev.rbegin(), peeled_rev.rend());
    return TileLayout(new_shard, layout->replica, new_offset);
  }

  PrimExpr two = make_const(rem.dtype(), 2);
  PrimExpr c = analyzer.Simplify(floordiv(rem, two));
  bool even = analyzer.CanProveEqual(floormod(rem, two), 0);
  bool mid = analyzer.CanProveEqual(analyzer.Simplify(d0[pivot] + c), Ek);
  bool cap = true;
  if (pivot > 0) {
    cap = analyzer.CanProve(analyzer.Simplify(d0[pivot - 1] + 1 <= shard[pivot - 1]->extent));
  }
  if (even && mid && cap) {
    if (pivot == 0 || shard[pivot - 1]->axis.same_as(ak)) {
      PrimExpr delta =
          analyzer.Simplify((pivot > 0 ? shard[pivot - 1]->stride : PrimExpr(0)) - (Ek - c) * Sk);
      std::vector<Iter> new_shard;
      new_shard.push_back(Iter(make_const(c.dtype(), 2), delta, ak));
      new_shard.push_back(Iter(c, Sk, ak));
      new_shard.insert(new_shard.end(), peeled_rev.rbegin(), peeled_rev.rend());
      return TileLayout(new_shard, layout->replica, new_offset);
    }
  }

  return std::nullopt;
}

ffi::Optional<TileLayout> TileLayoutNode::Slice(Array<PrimExpr> shape, Region region) const {
  arith::Analyzer analyzer;
  auto [grouped_layout, seps] = Group(ffi::GetRef<TileLayout>(this), shape);
  std::vector<Iter> new_shard;
  ffi::Map<Axis, PrimExpr> new_offset;
  for (size_t i = 0; i < seps.size() - 1; ++i) {
    std::vector<Iter> shard(grouped_layout->shard.begin() + seps[i],
                            grouped_layout->shard.begin() + seps[i + 1]);
    TileLayout group = TileLayout(shard, {}, {});
    auto sliced_opt = SlicePerGroup(group, region[i]->min, region[i]->extent);
    if (!sliced_opt.has_value()) return std::nullopt;
    auto sliced = sliced_opt.value();
    new_shard.insert(new_shard.end(), sliced->shard.begin(), sliced->shard.end());
    for (const auto& [axis, off] : sliced->offset) {
      auto it = new_offset.find(axis);
      if (it != new_offset.end()) {
        new_offset.Set(axis, analyzer.Simplify((*it).second + off));
      } else {
        new_offset.Set(axis, analyzer.Simplify(off));
      }
    }
  }
  return TileLayout(new_shard, grouped_layout->replica, new_offset);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "tir.TileLayoutSlice",
      [](const TileLayout& layout, Array<PrimExpr> shape,
         Region region) -> ffi::Optional<TileLayout> { return layout->Slice(shape, region); });
}

}  // namespace tir
}  // namespace tvm
