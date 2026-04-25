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
 * \file exec_context.cc
 * \brief C++ port of ExecContext (compile-time active-thread state).
 */

#include <tvm/runtime/logging.h>
#include <tvm/tirx/exec_context.h>
#include <tvm/tirx/expr.h>

#include <sstream>

namespace tvm {
namespace tirx {

bool Lane::Intersect(int64_t lo, int64_t hi, Lane* out) const {
  int64_t new_off = std::max(offset, lo);
  int64_t new_end = std::min(offset + extent, hi);
  if (new_end <= new_off) return false;
  out->extent = new_end - new_off;
  out->offset = new_off;
  return true;
}

ActiveSet InitialActiveSet(int64_t lane_ext, int64_t warp_ext, int64_t cta_ext) {
  ActiveSet A;
  A.laneid = Lane{lane_ext, 0};
  A.warpid = Lane{warp_ext, 0};
  A.cta_id = Lane{cta_ext, 0};
  return A;
}

namespace {

// Narrow a flat axis lane (laneid / warpid / cta_id) by [lo, hi).
bool NarrowFlat(Lane ActiveSet::* axis_field, const ActiveSet& A, int64_t lo, int64_t hi,
                ActiveSet* out, std::string* err) {
  Lane narrowed;
  const Lane& cur = A.*axis_field;
  if (!cur.Intersect(lo, hi, &narrowed)) {
    *err = "filter produces empty lane";
    return false;
  }
  *out = A;
  out->*axis_field = narrowed;
  return true;
}

}  // namespace

bool FilterNarrow(const ActiveSet& A, ScopeBinding binding, int64_t lo, int64_t hi, ActiveSet* out,
                  std::string* err) {
  if (lo >= hi) {
    *err = "filter range is empty or inverted";
    return false;
  }

  switch (binding) {
    case ScopeBinding::kWarpThread:
      // narrow A.laneid (the bound Var is a lane index 0..32)
      return NarrowFlat(&ActiveSet::laneid, A, lo, hi, out, err);
    case ScopeBinding::kCtaWarp:
      // narrow A.warpid (flat warp index)
      return NarrowFlat(&ActiveSet::warpid, A, lo, hi, out, err);
    case ScopeBinding::kKernelCta:
    case ScopeBinding::kClusterCta:
      // narrow A.cta_id (cluster->cta and kernel->cta target the same axis)
      return NarrowFlat(&ActiveSet::cta_id, A, lo, hi, out, err);
    case ScopeBinding::kCtaWarpgroup: {
      // narrow the outer factor of A.warpid (warpgroup index)
      const Lane& wp = A.warpid;
      if (wp.offset % kWgSize != 0 || wp.extent % kWgSize != 0) {
        *err = "filter on warpgroup_id requires warpid lane aligned to WG_SIZE";
        return false;
      }
      Lane cur_outer{wp.extent / kWgSize, wp.offset / kWgSize};
      Lane new_outer;
      if (!cur_outer.Intersect(lo, hi, &new_outer)) {
        *err = "filter on warpgroup_id produces empty lane";
        return false;
      }
      *out = A;
      out->warpid = Lane{new_outer.extent * kWgSize, new_outer.offset * kWgSize};
      return true;
    }
    case ScopeBinding::kWarpgroupWarp: {
      // narrow the inner factor of A.warpid (warp-within-wg index)
      const Lane& wp = A.warpid;
      int64_t cur_inner_off = wp.offset % kWgSize;
      if (wp.extent > kWgSize - cur_inner_off) {
        *err = "filter on warp_id_in_wg would break A's box: warpid spans multiple warpgroups";
        return false;
      }
      Lane cur_inner{wp.extent, cur_inner_off};
      Lane new_inner;
      if (!cur_inner.Intersect(lo, hi, &new_inner)) {
        *err = "filter on warp_id_in_wg produces empty lane";
        return false;
      }
      int64_t outer_base = (wp.offset / kWgSize) * kWgSize;
      *out = A;
      out->warpid = Lane{new_inner.extent, outer_base + new_inner.offset};
      return true;
    }
    case ScopeBinding::kKernelCluster:
      *err = "filter on cluster_id is not supported";
      return false;
    case ScopeBinding::kCtaThread:
    case ScopeBinding::kWarpgroupThread:
      // Flat-thread bindings span (laneid, warpid) product -- no single lane.
      *err = "filter on flat-thread binding has no single-lane representation";
      return false;
  }
  *err = "unknown ScopeBinding";
  return false;
}

namespace {

// Factor warpid (ext, off) into (wid_in_wg, wgid). Returns false on case 3.
bool FactorWarpid(const Lane& wp, Lane* wid_in_wg, Lane* wgid) {
  int64_t off = wp.offset;
  int64_t ext = wp.extent;
  int64_t wid_off = off % kWgSize;
  int64_t wgid_off = off / kWgSize;

  // case 1: aligned
  if (wid_off == 0 && ext % kWgSize == 0) {
    *wid_in_wg = Lane{kWgSize, 0};
    *wgid = Lane{ext / kWgSize, wgid_off};
    return true;
  }
  // case 2: fits in one warpgroup
  if (ext <= kWgSize - wid_off) {
    *wid_in_wg = Lane{ext, wid_off};
    *wgid = Lane{1, wgid_off};
    return true;
  }
  // case 3: crosses boundary, unaligned
  return false;
}

}  // namespace

bool ScopeSwitch(const ActiveSet& A, ScopeKind scope_kind, ExecSplit* out, std::string* err) {
  out->inter.clear();
  out->intra.clear();
  switch (scope_kind) {
    case ScopeKind::kThread:
      out->inter["laneid"] = A.laneid;
      out->inter["warpid"] = A.warpid;
      out->inter["cta_id"] = A.cta_id;
      return true;
    case ScopeKind::kWarp:
      out->intra["laneid"] = A.laneid;
      out->inter["warpid"] = A.warpid;
      out->inter["cta_id"] = A.cta_id;
      return true;
    case ScopeKind::kCta:
      out->intra["laneid"] = A.laneid;
      out->intra["warpid"] = A.warpid;
      out->inter["cta_id"] = A.cta_id;
      return true;
    case ScopeKind::kCluster:
      out->intra["laneid"] = A.laneid;
      out->intra["warpid"] = A.warpid;
      out->intra["cta_id"] = A.cta_id;
      return true;
    case ScopeKind::kWarpgroup: {
      Lane wid_in_wg, wgid;
      if (!FactorWarpid(A.warpid, &wid_in_wg, &wgid)) {
        std::ostringstream os;
        os << "scope_switch(warpgroup) failed: warpid lane (extent=" << A.warpid.extent
           << ", offset=" << A.warpid.offset << ") crosses warpgroup boundary and is not aligned";
        *err = os.str();
        return false;
      }
      out->intra["laneid"] = A.laneid;
      out->intra["wid_in_wg"] = wid_in_wg;
      out->inter["wgid"] = wgid;
      out->inter["cta_id"] = A.cta_id;
      return true;
    }
    case ScopeKind::kKernel:
      // scope_kind=kernel: inter=A, intra=empty (initial state)
      out->inter["laneid"] = A.laneid;
      out->inter["warpid"] = A.warpid;
      out->inter["cta_id"] = A.cta_id;
      return true;
    case ScopeKind::kWorld:
      *err = "scope_switch(world) is not a valid ExecContext transition";
      return false;
  }
  *err = "unknown ScopeKind";
  return false;
}

ExecContext ExecContext::AtKernelEntry(int64_t lane_ext, int64_t warp_ext, int64_t cta_ext) {
  ExecContext ctx;
  ctx.A = InitialActiveSet(lane_ext, warp_ext, cta_ext);
  ctx.scope_kind = ScopeKind::kKernel;
  std::string err;
  bool ok = ScopeSwitch(ctx.A, ctx.scope_kind, &ctx.split, &err);
  (void)ok;  // AtKernelEntry never fails for kKernel.
  return ctx;
}

bool ExecContext::WithFilter(ScopeBinding binding, int64_t lo, int64_t hi, ExecContext* out,
                             std::string* err) const {
  ActiveSet new_A;
  if (!FilterNarrow(A, binding, lo, hi, &new_A, err)) return false;
  ExecSplit new_split;
  if (!ScopeSwitch(new_A, scope_kind, &new_split, err)) return false;
  out->A = new_A;
  out->scope_kind = scope_kind;
  out->split = std::move(new_split);
  return true;
}

bool ExecContext::WithScopeSwitch(ScopeKind new_scope_kind, ExecContext* out,
                                  std::string* err) const {
  ExecSplit new_split;
  if (!ScopeSwitch(A, new_scope_kind, &new_split, err)) return false;
  out->A = A;
  out->scope_kind = new_scope_kind;
  out->split = std::move(new_split);
  return true;
}

ffi::Map<ffi::String, ffi::Array<IntImm>> EncodeSplitSide(
    const std::unordered_map<std::string, Lane>& side) {
  ffi::Map<ffi::String, ffi::Array<IntImm>> out;
  for (const auto& kv : side) {
    out.Set(ffi::String(kv.first), ffi::Array<IntImm>{IntImm(DataType::Int(64), kv.second.extent),
                                                      IntImm(DataType::Int(64), kv.second.offset)});
  }
  return out;
}

}  // namespace tirx
}  // namespace tvm
