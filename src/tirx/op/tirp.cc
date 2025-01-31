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

/********************* Utils **********************/

bool IsInt(const ObjectRef& obj) { return obj.as<runtime::Int>() || obj.as<IntImmNode>(); }

bool IsFloat(const ObjectRef& obj) { return obj.as<runtime::Float>() || obj.as<FloatImmNode>(); }

bool IsIntOrFloat(const ObjectRef& obj) { return IsInt(obj) || IsFloat(obj); }

IntImm ToIntImm(const ObjectRef& obj) {
  ICHECK(IsInt(obj)) << "ValueError: Cannot convert to IntImm";
  if (const auto* int_node = obj.as<runtime::Int::ContainerType>()) {
    return IntImm(DataType::Int(32), int_node->value);
  } else {
    return Downcast<IntImm>(obj);
  }
}

FloatImm ToFloatImm(const ObjectRef& obj) {
  ICHECK(IsFloat(obj)) << "ValueError: Cannot convert to FloatImm";
  if (const auto* float_node = obj.as<runtime::Float::ContainerType>()) {
    return FloatImm(DataType::Float(32), float_node->value);
  } else {
    return Downcast<FloatImm>(obj);
  }
}

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

/********************* ScheduleContext **********************/
TVM_REGISTER_NODE_TYPE(ScheduleContextNode);

ScheduleContext::ScheduleContext(Target target, ExecScope exec_scope,
                                 Map<String, PrimExpr> launch_params) {
  auto n = make_object<ScheduleContextNode>();
  n->target = std::move(target);
  n->exec_scope = std::move(exec_scope);
  n->launch_params = std::move(launch_params);
  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("tirp.ScheduleContext")
    .set_body_typed([](Target target, ExecScope exec_scope, Map<String, PrimExpr> launch_params) {
      return ScheduleContext(target, exec_scope, launch_params);
    });

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
      ICHECK(IsIntOrFloat(args[1])) << "arg[1] of fill() must be int or float";
    });

TIRP_DEFINE_SCHEDULE_OP(gemm).set_num_inputs(6).set_attr<FArgSanitizer>(
    "FArgSanitizer", [](tvm::Op op, Array<ObjectRef> args) {
      ICHECK_EQ(args.size(), 6U) << "gemm() expects 6 arguments";
      ICHECK(args[0].as<BufferRegionNode>()) << "arg[0] of gemm() must be BufferRegion";
      ICHECK(args[1].as<BufferRegionNode>()) << "arg[1] of gemm() must be BufferRegion";
      ICHECK(args[2].as<BufferRegionNode>()) << "arg[2] of gemm() must be BufferRegion";
      ICHECK(args[3].as<BufferRegionNode>()) << "arg[3] of gemm() must be BufferRegion";
      ICHECK(IsIntOrFloat(args[4])) << "arg[4] of gemm() must be int or float";
      ICHECK(IsIntOrFloat(args[5])) << "arg[5] of gemm() must be int or float";
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

TIRP_DEFINE_PIPELINE_OP(pipeline_copy)
    .set_num_inputs(3)
    .set_attr<FArgSanitizer>("FArgSanitizer", [](tvm::Op op, Array<ObjectRef> args) {
      ICHECK_EQ(args.size(), 3U) << "pipeline_copy() expects 3 arguments";
      ICHECK(args[0].as<PipelineNode>()) << "arg[0] of pipeline_copy() must be Pipeline";
      ICHECK(args[1].as<BufferRegionNode>()) << "arg[1] of pipeline_copy() must be BufferRegion";

      ICHECK(args[2].as<BufferRegionNode>()) << "arg[2] of pipeline_copy() must be BufferRegion";
    });

TIRP_DEFINE_PIPELINE_OP(pipeline_producer_commit)
    .set_num_inputs(1)
    .set_attr<FArgSanitizer>("FArgSanitizer", [](tvm::Op op, Array<ObjectRef> args) {
      ICHECK_EQ(args.size(), 1U) << "pipeline_producer_commit() expects 1 argument";
      ICHECK(args[0].as<PipelineNode>()) << "arg[0] of pipeline_producer_commit() must be Pipeline";
    });

TIRP_DEFINE_PIPELINE_OP(pipeline_consumer_wait)
    .set_num_inputs(2)
    .set_attr<FArgSanitizer>("FArgSanitizer", [](tvm::Op op, Array<ObjectRef> args) {
      ICHECK_EQ(args.size(), 2U) << "pipeline_consumer_wait() expects 2 arguments";
      ICHECK(args[0].as<PipelineNode>()) << "arg[0] of pipeline_consumer_wait() must be Pipeline";
      ICHECK(IsInt(args[1])) << "arg[1] of pipeline_consumer_wait() must be int";
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
