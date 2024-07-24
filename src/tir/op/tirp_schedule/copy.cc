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
 * \file tir/op/tirp_schedule/copy.cc
 * \brief Schedule of TIR+ operators: copy.
 */

#include <tvm/tir/builtin.h>
#include <tvm/tir/tirp_op.h>

#include "schedule.h"

namespace tvm {
namespace tir {
namespace tirp {

Stmt VectorizedCopy(const BufferRegion& src_buffer_region, const BufferRegion& dst_buffer_region,
                    ScheduleContext context, CopyInstType inst_type) {
  const auto& src = src_buffer_region->buffer;
  const auto& dst = dst_buffer_region->buffer;
  const auto& src_region = src_buffer_region->region;
  const auto& dst_region = dst_buffer_region->region;
  /****************************** Arg sanitize ******************************/
  CHECK(src->layout.defined()) << "ValueError: src buffer must have layout";
  CHECK(dst->layout.defined()) << "ValueError: dst buffer must have layout";
  // Check if the region matches
  // Note that there can be regions with extent 1, which are not considered in this case
  arith::Analyzer ana;
  std::vector<PrimExpr> src_extents, dst_extents;
  for (size_t i = 0; i < src_region.size(); ++i) {
    if (!is_one(src_region[i]->extent)) {
      src_extents.push_back(src_region[i]->extent);
    }
  }
  for (size_t i = 0; i < dst_region.size(); ++i) {
    if (!is_one(dst_region[i]->extent)) {
      dst_extents.push_back(dst_region[i]->extent);
    }
  }
  CHECK(src_extents.size() == dst_extents.size())
      << "ValueError: src and dst region mismatch, got " << src_extents.size() << " and "
      << dst_extents.size();
  for (size_t i = 0; i < src_extents.size(); ++i) {
    CHECK(ana.CanProveEqual(src_extents[i], dst_extents[i]))
        << "ValueError: src and dst region mismatch, got " << src_extents[i] << " and "
        << dst_extents[i];
  }
  /****************************** Schedule Dispatch ******************************/
  // Currently only support copy between global and shared memory
  CHECK((src.scope() == "global" && dst.scope() == "shared") ||
        (src.scope() == "shared" && dst.scope() == "global"))
      << "ValueError: Unsupported copy between " << src.scope() << " and " << dst.scope();
  // Check the layouts
  const SwizzleLayoutNode* swizzle = nullptr;
  if (src.scope() == "global" && dst.scope() == "shared") {
    swizzle = dst->layout.as<SwizzleLayoutNode>();
    CHECK(IsTrivialLayout(src->layout.value())) << "ValueError: src buffer layout must be trivial";
    CHECK(IsTrivialLayout(dst->layout.value()) || swizzle != nullptr)
        << "ValueError: dst buffer layout must be trivial or swizzle";
  } else if (src.scope() == "shared" && dst.scope() == "global") {
    swizzle = src->layout.as<SwizzleLayoutNode>();
    CHECK(IsTrivialLayout(src->layout.value()) || swizzle != nullptr)
        << "ValueError: src buffer layout must be trivial or swizzle";
    CHECK(IsTrivialLayout(dst->layout.value())) << "ValueError: dst buffer layout must be trivial";
  }
  // Currently only support data copy between the same data type
  CHECK(src->dtype == dst->dtype) << "ValueError: src and dst buffer data type mismatch";
  // Currently only consider the number of elements is a multiple of cta thread size
  PrimExpr n_elements = 1;
  for (const auto& region : src_region) {
    n_elements *= region->extent;
  }
  const auto& thread_cnt = context.GetScopeExtent(ScopeIdDef{String("cta"), String("thread")});
  CHECK(ana.CanProveEqual(FloorMod(n_elements, thread_cnt), 0)) << "ValueError: n_elements must be "
                                                                << "a multiple of cta thread size";
  // Select vector length, need to verify the alignment
  const auto& src_st = src_region.back()->min;
  const auto& dst_st = dst_region.back()->min;
  const auto& src_stride = src->shape.back();
  const auto& dst_stride = dst->shape.back();
  const auto& src_region_extent = src_region.back()->extent;
  const auto& dst_region_extent = dst_region.back()->extent;
  std::vector<PrimExpr> vec_candidates = {16, 8, 4};  // copy size in bytes
  if (inst_type == CopyInstType::kBufferLoad) {
    vec_candidates.push_back(1);
  }
  if (swizzle != nullptr) {
    // swizzle layout
    auto lim = (1 << swizzle->per_element) * src->dtype.bytes();
    // filter out the candidates that are not aligned with the swizzle layout
    vec_candidates.erase(std::remove_if(vec_candidates.begin(), vec_candidates.end(),
                                        [&](const PrimExpr& candidate) {
                                          return !ana.CanProveEqual(FloorMod(lim, candidate), 0);
                                        }),
                         vec_candidates.end());
  }
  for (size_t i = 0; i < vec_candidates.size(); ++i) {
    vec_candidates[i] = ana.Simplify(FloorDiv(vec_candidates[i], src->dtype.bytes()));
  }
  Optional<PrimExpr> vec_len = NullOpt;
  for (const auto& candidate : vec_candidates) {
    if (ana.CanProveEqual(FloorMod(src_st, candidate), 0) &&
        ana.CanProveEqual(FloorMod(dst_st, candidate), 0) &&
        ana.CanProveEqual(FloorMod(src_stride, candidate), 0) &&
        ana.CanProveEqual(FloorMod(dst_stride, candidate), 0) &&
        ana.CanProveEqual(FloorMod(src_region_extent, candidate), 0) &&
        ana.CanProveEqual(FloorMod(dst_region_extent, candidate), 0) &&
        ana.CanProveEqual(FloorMod(FloorDiv(n_elements, thread_cnt), candidate), 0)) {
      vec_len = candidate;
      break;
    }
  }
  CHECK(vec_len.defined()) << "ValueError: Cannot find a valid vector length";
  // Create stmts for vectorized copy
  // TODO(@bohan): consider using schedule langauge here?
  Var s("s"), vec("vec"), threadIdx = context.GetThreadVar("threadIdx.x");
  PrimExpr s_extent = ana.Simplify(FloorDiv(n_elements, thread_cnt * vec_len.value()));
  ana.Bind(vec, Range::FromMinExtent(0, vec_len.value()));
  ana.Bind(s, Range::FromMinExtent(0, s_extent));
  ana.Bind(threadIdx, Range::FromMinExtent(0, thread_cnt));
  // Calculate src and dst indices
  auto get_indices = [&](const Region& region, CopyInstType inst_type) -> std::vector<PrimExpr> {
    PrimExpr fused = inst_type == CopyInstType::kBufferLoad
                         ? ana.Simplify((s * thread_cnt + threadIdx) * vec_len.value() + vec)
                         : ana.Simplify((s * thread_cnt + threadIdx) * vec_len.value());
    std::vector<PrimExpr> indices;
    for (int i = region.size() - 1; i >= 0; --i) {
      indices.push_back(ana.Simplify(region[i]->min + FloorMod(fused, region[i]->extent)));
      fused = FloorDiv(fused, region[i]->extent);
    }
    std::reverse(indices.begin(), indices.end());
    return std::move(indices);
  };
  auto src_indices = get_indices(src_region, inst_type);
  auto dst_indices = get_indices(dst_region, inst_type);
  // Create copy instruction
  Stmt copy_inst;
  if (inst_type == CopyInstType::kBufferLoad) {
    // vectorized copy using buffer load
    copy_inst = BufferStore(dst, BufferLoad(src, src_indices), dst_indices);
    copy_inst = For(vec, 0, vec_len.value(), ForKind::kVectorized, copy_inst);
  } else if (inst_type == CopyInstType::kCUDAcpasync) {
    // vectorized copy using CUDA cpasync
    CHECK(src.scope() == "global" && dst.scope() == "shared") << "ValueError: Unsupported copy";
    copy_inst = CallBuiltinOp(builtin::ptx_cp_async(),
                              {dst->data, dst.OffsetOf_p(dst_indices), src->data,
                               src.OffsetOf_p(src_indices), vec_len.value() * src->dtype.bytes()});
  }
  // Create loop nest
  Stmt body = For(s, 0, s_extent, ForKind::kSerial, copy_inst);
  if (inst_type == CopyInstType::kBufferLoad) {
    body = SeqStmt({body, CallBuiltinOp(builtin::tvm_storage_sync(), {StringImm("shared")})});
  }
  body = BlockRealize({}, Bool(true), Block("copy", body, ExecScope::Create("thread")));
  return body;
}

Stmt CopyOpScheduler(const Op& op, Array<ObjectRef> args, ScheduleContext context) {
  if (op.same_as(copy())) {
    // synchronized copy operator
    const auto& dst = Downcast<BufferRegion>(args[0]);
    const auto& src = Downcast<BufferRegion>(args[1]);
    return VectorizedCopy(src, dst, context, CopyInstType::kBufferLoad);
  }
  LOG(FATAL) << "ValueError: Unsupported copy operator " << op;
  throw;
}

}  // namespace tirp
}  // namespace tir
}  // namespace tvm
