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
 * \file tvm/tirx/exec_context.h
 * \brief Compile-time ExecContext state: the active thread set ``A`` and the
 * (inter, intra) split under the current scope kind, threaded through the IR
 * walker so per-op lowerers see the precise execution shape at each site.
 *
 * Mirrors the pure-Python implementation in python/tvm/tirx/exec_context.py.
 */
#ifndef TVM_TIRX_EXEC_CONTEXT_H_
#define TVM_TIRX_EXEC_CONTEXT_H_

#include <tvm/tirx/exec_scope.h>
#include <tvm/tirx/var.h>

#include <string>
#include <unordered_map>
#include <vector>

namespace tvm {
namespace tirx {

/*! \brief Warpgroup size in warps (hardware-fixed). */
constexpr int kWgSize = 4;

/*! \brief Integer range [offset, offset+extent) on one axis. */
struct Lane {
  int64_t extent = 1;
  int64_t offset = 0;

  /*! \brief Intersect with [lo, hi). Returns false if the result is empty. */
  bool Intersect(int64_t lo, int64_t hi, Lane* out) const;
};

/*!
 * \brief Active thread set A -- a box on (laneid, warpid, cta_id).
 *
 * Each dim is kept even when extent=1.
 */
struct ActiveSet {
  Lane laneid;
  Lane warpid;
  Lane cta_id;

  int64_t size() const { return laneid.extent * warpid.extent * cta_id.extent; }
};

/*!
 * \brief One scope_switch split. Fields are sparse dicts keyed by axis name
 * drawn from {"laneid", "warpid", "cta_id", "wid_in_wg", "wgid"}. An empty
 * map denotes the empty layout (e.g. intra under scope_kind=thread).
 */
struct ExecSplit {
  std::unordered_map<std::string, Lane> inter;
  std::unordered_map<std::string, Lane> intra;
};

/*! \brief Initial A at T.kernel() entry: all threads active, offsets zero. */
TVM_DLL ActiveSet InitialActiveSet(int64_t lane_ext, int64_t warp_ext, int64_t cta_ext);

/*!
 * \brief Narrow A on the lane bound to ``binding``.
 *
 * The ScopeBinding maps directly to which native axis (laneid/warpid/cta_id)
 * to narrow, and for warpid whether to narrow the full axis (kCtaWarp), the
 * outer factor (kCtaWarpgroup), or the inner factor (kWarpgroupWarp).
 *
 * Bindings with no single-lane representation (kKernelCluster cluster_id is
 * not a filter target; kCtaThread / kWarpgroupThread are flat products of
 * two axes) return false with reason in *err.
 */
TVM_DLL bool FilterNarrow(const ActiveSet& A, ScopeBinding binding, int64_t lo, int64_t hi,
                          ActiveSet* out, std::string* err);

/*!
 * \brief Factor A into (inter, intra) for target scope_kind.
 *
 * Returns false on factoring failure (warpgroup with warpid lane that
 * crosses a warpgroup boundary unaligned) and writes reason to *err.
 */
TVM_DLL bool ScopeSwitch(const ActiveSet& A, ScopeKind scope_kind, ExecSplit* out,
                         std::string* err);

/*! \brief Per-program-point ExecContext: active set + scope kind + split. */
struct ExecContext {
  ActiveSet A;
  ScopeKind scope_kind = ScopeKind::kKernel;
  ExecSplit split;  // (inter, intra) of current A under current scope_kind

  /*! \brief Kernel-entry ctor. */
  static ExecContext AtKernelEntry(int64_t lane_ext, int64_t warp_ext, int64_t cta_ext);

  /*! \brief Apply filter; scope_kind preserved, split recomputed. */
  bool WithFilter(ScopeBinding binding, int64_t lo, int64_t hi, ExecContext* out,
                  std::string* err) const;

  /*! \brief Apply scope_switch; A preserved, split recomputed for new scope_kind. */
  bool WithScopeSwitch(ScopeKind new_scope_kind, ExecContext* out, std::string* err) const;
};

/*!
 * \brief Encode one side of an ExecSplit (inter or intra) as the FFI map used
 * by ``DispatchContextNode::{inter, intra}``: axis name -> [extent, offset].
 */
TVM_DLL ffi::Map<ffi::String, ffi::Array<IntImm>> EncodeSplitSide(
    const std::unordered_map<std::string, Lane>& side);

}  // namespace tirx
}  // namespace tvm

#endif  // TVM_TIRX_EXEC_CONTEXT_H_
