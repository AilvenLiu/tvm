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
 * \file tir/op/tirp.cc
 * TIR+ built-in operators.
 */

#include <tvm/tirx/op.h>
#include <tvm/tirx/op_attr_types.h>
#include <tvm/tirx/tirp_op.h>

namespace tvm {
namespace tirx {
namespace tirp {

#define TVM_TIRP_REGISTER_OP(OpName) \
  TVM_REGISTER_OP("tirp." OpName).set_attr<TScriptPrinterName>("TScriptPrinterName", OpName)

#define TIRP_DEFINE_BUILTIN_FUNC(OpName)            \
  const Op& OpName() {                              \
    static const Op& op = Op::Get("tirp." #OpName); \
    return op;                                      \
  }                                                 \
  TVM_TIRP_REGISTER_OP(#OpName)

#define TIRP_DEFINE_OP(OpName)                                                      \
  TIRP_DEFINE_BUILTIN_FUNC(OpName)                                                  \
      .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kPure)) \
      .set_attr<Bool>("TIsTIRpOp", Bool(true))

/********************* Schedule Ops **********************/
#define TIRP_DEFINE_SCHEDULE_OP(OpName) \
  TIRP_DEFINE_OP(OpName).set_attr<Bool>("TIsScheduleOp", Bool(true))

TIRP_DEFINE_SCHEDULE_OP(copy).set_num_inputs(2).set_attr<FArgSanitizer>(
    "FArgSanitizer", [](tvm::Op op, Array<ObjectRef> args) {
      ICHECK_EQ(args.size(), 2U) << "copy() expects 2 arguments";
      ICHECK(args[0].as<BufferRegionNode>()) << "arg[0] of copy() must be BufferRegion";
      ICHECK(args[1].as<BufferRegionNode>()) << "arg[1] of copy() must be BufferRegion";
    });

TIRP_DEFINE_SCHEDULE_OP(fill).set_num_inputs(2).set_attr<FArgSanitizer>(
    "FArgSanitizer", [](tvm::Op op, Array<ObjectRef> args) {
      ICHECK_EQ(args.size(), 2U) << "fill() expects 2 arguments";
      ICHECK(args[0].as<BufferRegionNode>()) << "arg[0] of fill() must be BufferRegion";
      ICHECK(args[1].as<PrimExprNode>()) << "arg[1] of fill() must be PrimExpr";
    });

TIRP_DEFINE_SCHEDULE_OP(gemm).set_num_inputs(6).set_attr<FArgSanitizer>(
    "FArgSanitizer", [](tvm::Op op, Array<ObjectRef> args) {
      ICHECK_EQ(args.size(), 6U) << "gemm() expects 6 arguments";
      ICHECK(args[0].as<BufferRegionNode>()) << "arg[0] of gemm() must be BufferRegion";
      ICHECK(args[1].as<BufferRegionNode>()) << "arg[1] of gemm() must be BufferRegion";
      ICHECK(args[2].as<BufferRegionNode>()) << "arg[2] of gemm() must be BufferRegion";
      ICHECK(args[3].as<BufferRegionNode>()) << "arg[3] of gemm() must be BufferRegion";
      ICHECK(args[4].as<PrimExprNode>()) << "arg[4] of gemm() must be PrimExpr";
      ICHECK(args[5].as<PrimExprNode>()) << "arg[5] of gemm() must be PrimExpr";
    });

/********************* Barrier Ops **********************/
#define TIRP_DEFINE_BARRIER_OP(OpName) \
  TIRP_DEFINE_OP(OpName).set_attr<Bool>("TIsBarrierOp", Bool(true))

TIRP_DEFINE_BARRIER_OP(barrier_init)
    .set_num_inputs(2)
    .set_attr<FArgSanitizer>("FArgSanitizer", [](tvm::Op op, Array<ObjectRef> args) {
      ICHECK_EQ(args.size(), 2U) << "barrier_init() expects 2 arguments";
      ICHECK(args[0].as<BarrierNode>()) << "arg[0] of barrier_init() must be Barrier";
      ICHECK(args[1].as<PrimExprNode>()) << "arg[1] of barrier_init() must be PrimExpr";
    });

TIRP_DEFINE_BARRIER_OP(barrier_arrive)
    .set_num_inputs(1)
    .set_attr<FArgSanitizer>("FArgSanitizer", [](tvm::Op op, Array<ObjectRef> args) {
      ICHECK_EQ(args.size(), 1U) << "barrier_arrive() expects 1 argument";
      ICHECK(args[0].as<BarrierNode>()) << "arg[0] of barrier_arrive() must be Barrier";
    });

TIRP_DEFINE_BARRIER_OP(barrier_wait)
    .set_num_inputs(1)
    .set_attr<FArgSanitizer>("FArgSanitizer", [](tvm::Op op, Array<ObjectRef> args) {
      ICHECK_EQ(args.size(), 1U) << "barrier_wait() expects 1 argument";
      ICHECK(args[0].as<BarrierNode>()) << "arg[0] of barrier_wait() must be Barrier";
    });

/********************* Pipeline Ops **********************/
#define TIRP_DEFINE_PIPELINE_OP(OpName) \
  TIRP_DEFINE_OP(OpName).set_attr<Bool>("TIsPipelineOp", Bool(true))

TIRP_DEFINE_PIPELINE_OP(pipeline_producer_acquire)
    .set_num_inputs(1)
    .set_attr<FArgSanitizer>("FArgSanitizer", [](tvm::Op op, Array<ObjectRef> args) {
      ICHECK_EQ(args.size(), 1U) << "pipeline_producer_acquire() expects 1 argument";
      ICHECK(args[0].as<PipelineNode>())
          << "arg[0] of pipeline_producer_acquire() must be Pipeline";
    });

TIRP_DEFINE_PIPELINE_OP(pipeline_producer_copy_async)
    .set_num_inputs(3)
    .set_attr<FArgSanitizer>("FArgSanitizer", [](tvm::Op op, Array<ObjectRef> args) {
      ICHECK_EQ(args.size(), 3U) << "pipeline_producer_copy_async() expects 3 arguments";
      ICHECK(args[0].as<PipelineNode>())
          << "arg[0] of pipeline_producer_copy_async() must be Pipeline";
      ICHECK(args[1].as<BufferRegionNode>())
          << "arg[1] of pipeline_producer_copy_async() must be BufferRegion";

      ICHECK(args[2].as<BufferRegionNode>())
          << "arg[2] of pipeline_producer_copy_async() must be BufferRegion";
    });

TIRP_DEFINE_PIPELINE_OP(pipeline_producer_commit_stage)
    .set_num_inputs(1)
    .set_attr<FArgSanitizer>("FArgSanitizer", [](tvm::Op op, Array<ObjectRef> args) {
      ICHECK_EQ(args.size(), 1U) << "pipeline_producer_commit_stage() expects 1 argument";
      ICHECK(args[0].as<PipelineNode>())
          << "arg[0] of pipeline_producer_commit_stage() must be Pipeline";
    });

TIRP_DEFINE_PIPELINE_OP(pipeline_consumer_wait)
    .set_num_inputs(2)
    .set_attr<FArgSanitizer>("FArgSanitizer", [](tvm::Op op, Array<ObjectRef> args) {
      ICHECK_EQ(args.size(), 2U) << "pipeline_consumer_wait() expects 2 arguments";
      ICHECK(args[0].as<PipelineNode>()) << "arg[0] of pipeline_consumer_wait() must be Pipeline";
      ICHECK(args[1].as<PrimExprNode>()) << "arg[1] of pipeline_consumer_wait() must be PrimExpr";
    });

TIRP_DEFINE_PIPELINE_OP(pipeline_consumer_release)
    .set_num_inputs(1)
    .set_attr<FArgSanitizer>("FArgSanitizer", [](tvm::Op op, Array<ObjectRef> args) {
      ICHECK_EQ(args.size(), 1U) << "pipeline_consumer_release() expects 1 argument";
      ICHECK(args[0].as<PipelineNode>())
          << "arg[0] of pipeline_consumer_release() must be Pipeline";
    });

}  // namespace tirxp
}  // namespace tirx
}  // namespace tvm
