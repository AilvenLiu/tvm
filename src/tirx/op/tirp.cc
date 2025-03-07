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

bool IsBool(const ObjectRef& obj) { return obj.as<runtime::Bool>() || obj.as<IntImmNode>(); }

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

template <typename Key, typename Value>
Value getOrSetDefault(Map<String, ObjectRef>& m, const Key& key, const Value& defaultValue) {
  // try_emplace inserts the defaultValue only if key does not exist.
  auto it = m.find(key);
  if (it == m.end()) {
    m.Set(key, defaultValue);
    return defaultValue;
  }
  return Downcast<Value>((*it).second);
}

void ScheduleContextNode::AddAllocBuffer(Buffer buffer) {
  auto buffers = getOrSetDefault(callbacks, callback::kPrivateAlloc, Array<Buffer>());
  buffers.push_back(buffer);
  callbacks.Set(callback::kPrivateAlloc, buffers);
}

void ScheduleContextNode::AddInitStmt(Stmt stmt, bool host) {
  auto tag = host ? callback::kHostInitStmt : callback::kDeviceInitStmt;
  auto stmts = getOrSetDefault(callbacks, tag, Array<Stmt>());
  stmts.push_back(stmt);
  callbacks.Set(tag, stmts);
}

ScheduleContext::ScheduleContext(Target target, ExecScope exec_scope,
                                 Map<String, PrimExpr> launch_params, Map<Var, Range> var_range_map,
                                 Map<String, ObjectRef> callbacks) {
  auto n = make_object<ScheduleContextNode>();
  n->target = std::move(target);
  n->exec_scope = std::move(exec_scope);
  n->launch_params = std::move(launch_params);
  n->var_range_map = std::move(var_range_map);
  n->callbacks = std::move(callbacks);
  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("tirp.ScheduleContext")
    .set_body_typed([](Target target, ExecScope exec_scope, Map<String, PrimExpr> launch_params,
                       Map<Var, Range> var_range_map, Map<String, ObjectRef> callbacks) {
      return ScheduleContext(target, exec_scope, launch_params, var_range_map, callbacks);
    });

TVM_REGISTER_GLOBAL("tirx.ScheduleContextAddAllocBuffer")
    .set_body_method<ScheduleContext>(&ScheduleContextNode::AddAllocBuffer);
TVM_REGISTER_GLOBAL("tirx.ScheduleContextAddInitStmt")
    .set_body_method<ScheduleContext>(&ScheduleContextNode::AddInitStmt);

/********************* Schedule Ops **********************/
#define TIRP_DEFINE_SCHEDULE_OP(OpName) \
  TIRP_DEFINE_OP(OpName).set_attr<Bool>("TIsScheduleOp", Bool(true))

TIRP_DEFINE_SCHEDULE_OP(zero).set_num_inputs(2).set_attr<FArgSanitizer>(
    "FArgSanitizer", [](tvm::Op op, Array<ObjectRef> args) {
      ICHECK_EQ(args.size(), 2U) << "zero() expects 2 arguments";
      ICHECK(args[0].as<BufferRegionNode>()) << "arg[0] of zero() must be BufferRegion";
      ICHECK(args[1].as<BufferRegionNode>()) << "arg[1] of zero() must be BufferRegion";
    });

TIRP_DEFINE_SCHEDULE_OP(sqrt).set_num_inputs(4).set_attr<FArgSanitizer>(
    "FArgSanitizer", [](tvm::Op op, Array<ObjectRef> args) {
      ICHECK_EQ(args.size(), 4U) << "sqrt() expects 4 arguments";
      ICHECK(args[0].as<BufferRegionNode>()) << "arg[0] of sqrt() must be BufferRegion";
      ICHECK(args[1].as<BufferRegionNode>()) << "arg[1] of sqrt() must be BufferRegion";
    });
TIRP_DEFINE_SCHEDULE_OP(exp).set_num_inputs(4).set_attr<FArgSanitizer>(
    "FArgSanitizer", [](tvm::Op op, Array<ObjectRef> args) {
      ICHECK_EQ(args.size(), 4U) << "exp() expects 4 arguments";
      ICHECK(args[0].as<BufferRegionNode>()) << "arg[0] of exp() must be BufferRegion";
      ICHECK(args[1].as<BufferRegionNode>()) << "arg[1] of exp() must be BufferRegion";
    });

TIRP_DEFINE_SCHEDULE_OP(add).set_num_inputs(3).set_attr<FArgSanitizer>(
    "FArgSanitizer", [](tvm::Op op, Array<ObjectRef> args) {
      ICHECK_EQ(args.size(), 3U) << "add() expects 3 arguments";
      ICHECK(args[0].as<BufferRegionNode>()) << "arg[0] of add() must be BufferRegion";
      ICHECK(args[1].as<BufferRegionNode>() || args[1].as<FloatImmNode>())
          << "arg[1] of add() must be BufferRegion or FloatImm";
      ICHECK(args[2].as<BufferRegionNode>() || args[2].as<FloatImmNode>())
          << "arg[2] of add() must be BufferRegion or FloatImm";
    });

TIRP_DEFINE_SCHEDULE_OP(sub).set_num_inputs(3).set_attr<FArgSanitizer>(
    "FArgSanitizer", [](tvm::Op op, Array<ObjectRef> args) {
      ICHECK_EQ(args.size(), 3U) << "sub() expects 3 arguments";
      ICHECK(args[0].as<BufferRegionNode>()) << "arg[0] of sub() must be BufferRegion";
      ICHECK(args[1].as<BufferRegionNode>() || args[1].as<FloatImmNode>())
          << "arg[1] of sub() must be BufferRegion";
      ICHECK(args[2].as<BufferRegionNode>() || args[2].as<FloatImmNode>())
          << "arg[2] of sub() must be BufferRegion or FloatImm";
    });

TIRP_DEFINE_SCHEDULE_OP(mul).set_num_inputs(3).set_attr<FArgSanitizer>(
    "FArgSanitizer", [](tvm::Op op, Array<ObjectRef> args) {
      ICHECK_EQ(args.size(), 3U) << "mul() expects 3 arguments";
      ICHECK(args[0].as<BufferRegionNode>()) << "arg[0] of mul() must be BufferRegion";
      ICHECK(args[1].as<BufferRegionNode>() || args[1].as<FloatImmNode>())
          << "arg[1] of mul() must be BufferRegion or FloatImm";
      ICHECK(args[2].as<BufferRegionNode>() || args[2].as<FloatImmNode>())
          << "arg[2] of mul() must be BufferRegion or FloatImm";
    });

TIRP_DEFINE_SCHEDULE_OP(fdiv).set_num_inputs(3).set_attr<FArgSanitizer>(
    "FArgSanitizer", [](tvm::Op op, Array<ObjectRef> args) {
      ICHECK_EQ(args.size(), 3U) << "fdiv() expects 3 arguments";
      ICHECK(args[0].as<BufferRegionNode>()) << "arg[0] of fdiv() must be BufferRegion";
      ICHECK(args[1].as<BufferRegionNode>()) << "arg[1] of fdiv() must be BufferRegion";
      ICHECK(args[2].as<BufferRegionNode>() || args[2].as<FloatImmNode>())
          << "arg[2] of fdiv() must be BufferRegion or FloatImm";
    });

TIRP_DEFINE_SCHEDULE_OP(minimum).set_num_inputs(3).set_attr<FArgSanitizer>(
    "FArgSanitizer", [](tvm::Op op, Array<ObjectRef> args) {
      ICHECK_EQ(args.size(), 3U) << "minimum() expects 3 arguments";
      ICHECK(args[0].as<BufferRegionNode>()) << "arg[0] of minimum() must be BufferRegion";
      ICHECK(args[1].as<BufferRegionNode>() || args[1].as<FloatImmNode>())
          << "arg[1] of minimum() must be BufferRegion or FloatImm";
      ICHECK(args[2].as<BufferRegionNode>() || args[2].as<FloatImmNode>())
          << "arg[2] of minimum() must be BufferRegion or FloatImm";
    });

TIRP_DEFINE_SCHEDULE_OP(maximum).set_num_inputs(3).set_attr<FArgSanitizer>(
    "FArgSanitizer", [](tvm::Op op, Array<ObjectRef> args) {
      ICHECK_EQ(args.size(), 3U) << "maximum() expects 3 arguments";
      ICHECK(args[0].as<BufferRegionNode>()) << "arg[0] of maximum() must be BufferRegion";
      ICHECK(args[1].as<BufferRegionNode>() || args[1].as<FloatImmNode>())
          << "arg[1] of maximum() must be BufferRegion or FloatImm";
      ICHECK(args[2].as<BufferRegionNode>() || args[2].as<FloatImmNode>())
          << "arg[2] of maximum() must be BufferRegion or FloatImm";
    });

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

TIRP_DEFINE_SCHEDULE_OP(gemm).set_num_inputs(8).set_attr<FArgSanitizer>(
    "FArgSanitizer", [](tvm::Op op, Array<ObjectRef> args) {
      ICHECK_EQ(args.size(), 8U) << "gemm() expects 8 arguments";
      ICHECK(args[0].as<BufferRegionNode>()) << "arg[0] of gemm() must be BufferRegion";
      ICHECK(args[1].as<BufferRegionNode>()) << "arg[1] of gemm() must be BufferRegion";
      ICHECK(args[2].as<BufferRegionNode>()) << "arg[2] of gemm() must be BufferRegion";
      ICHECK(args[3].as<BufferRegionNode>()) << "arg[3] of gemm() must be BufferRegion";
      ICHECK(IsBool(args[4])) << "arg[4] of gemm() must be int";
      ICHECK(IsBool(args[5])) << "arg[5] of gemm() must be int";
      ICHECK(IsIntOrFloat(args[6])) << "arg[6] of gemm() must be int or float";
      ICHECK(IsIntOrFloat(args[7])) << "arg[7] of gemm() must be int or float";
    });

TIRP_DEFINE_SCHEDULE_OP(reciprocal)
    .set_num_inputs(2)
    .set_attr<FArgSanitizer>("FArgSanitizer", [](tvm::Op op, Array<ObjectRef> args) {
      ICHECK_EQ(args.size(), 2U) << "reciprocal() expects 2 arguments";
      ICHECK(args[0].as<BufferRegionNode>()) << "arg[0] of reciprocal() must be BufferRegion";
      ICHECK(args[1].as<BufferRegionNode>()) << "arg[1] of reciprocal() must be BufferRegion";
    });

TIRP_DEFINE_SCHEDULE_OP(sum).set_num_inputs(4).set_attr<FArgSanitizer>(
    "FArgSanitizer", [](tvm::Op op, Array<ObjectRef> args) {
      ICHECK_EQ(args.size(), 4U) << "sum() expects 4 arguments";
      ICHECK(args[0].as<BufferRegionNode>()) << "arg[0] of sum() must be BufferRegion";
      ICHECK(args[1].as<BufferRegionNode>()) << "arg[1] of sum() must be BufferRegion";
    });

TIRP_DEFINE_SCHEDULE_OP(max).set_num_inputs(4).set_attr<FArgSanitizer>(
    "FArgSanitizer", [](tvm::Op op, Array<ObjectRef> args) {
      ICHECK_EQ(args.size(), 4U) << "max() expects 4 arguments";
      ICHECK(args[0].as<BufferRegionNode>()) << "arg[0] of max() must be BufferRegion";
      ICHECK(args[1].as<BufferRegionNode>()) << "arg[1] of max() must be BufferRegion";
    });

TIRP_DEFINE_SCHEDULE_OP(min).set_num_inputs(4).set_attr<FArgSanitizer>(
    "FArgSanitizer", [](tvm::Op op, Array<ObjectRef> args) {
      ICHECK_EQ(args.size(), 4U) << "min() expects 4 arguments";
      ICHECK(args[0].as<BufferRegionNode>()) << "arg[0] of min() must be BufferRegion";
      ICHECK(args[1].as<BufferRegionNode>()) << "arg[1] of min() must be BufferRegion";
    });

TIRP_DEFINE_SCHEDULE_OP(memset).set_num_inputs(2).set_attr<FArgSanitizer>(
    "FArgSanitizer", [](tvm::Op op, Array<ObjectRef> args) {
      ICHECK_EQ(args.size(), 2U) << "memset() expects 2 arguments";
      ICHECK(args[0].as<BufferRegionNode>()) << "arg[0] of memset() must be BufferRegion";
      ICHECK(args[1].as<FloatImmNode>()) << "arg[1] of memset() must be FloatImm";
    });

/********************* Compose Ops **********************/
#define TIRP_DEFINE_COMPOSE_OP(OpName) \
  TIRP_DEFINE_OP(OpName).set_attr<Bool>("TIsComposeOp", Bool(true))

TIRP_DEFINE_COMPOSE_OP(compose_op)
    .set_num_inputs(1)
    .set_attr<FArgSanitizer>("FArgSanitizer", [](tvm::Op op, Array<ObjectRef> args) {});

/********************* Pipeline Ops **********************/
#define TIRP_DEFINE_PIPELINE_OP(OpName) \
  TIRP_DEFINE_OP(OpName).set_attr<Bool>("TIsPipelineOp", Bool(true))

TIRP_DEFINE_PIPELINE_OP(pipeline_init)
    .set_num_inputs(1)
    .set_attr<FArgSanitizer>("FArgSanitizer", [](tvm::Op op, Array<ObjectRef> args) {
      ICHECK_EQ(args.size(), 1U) << "pipeline_init() expects 1 argument";
      ICHECK(args[0].as<PipelineNode>()) << "arg[0] of pipeline_init() must be Pipeline";
    });

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
    .set_num_inputs(2)
    .set_attr<FArgSanitizer>("FArgSanitizer", [](tvm::Op op, Array<ObjectRef> args) {
      ICHECK_EQ(args.size(), 2U) << "pipeline_producer_commit() expects 2 arguments";
      ICHECK(args[0].as<PipelineNode>()) << "arg[0] of pipeline_producer_commit() must be Pipeline";
      ICHECK(IsInt(args[1])) << "arg[1] of pipeline_producer_commit() must be int";
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

/********************* Misc Ops **********************/
TIRP_DEFINE_OP(tvm_kernel_replace_point)
    .set_num_inputs(0)
    .set_attr<FArgSanitizer>("FArgSanitizer", [](tvm::Op op, Array<ObjectRef> args) {
      ICHECK_EQ(args.size(), 0U) << "tvm_kernel_replace_point() expects 0 arguments";
    });

}  // namespace tirxp
}  // namespace tirx
}  // namespace tvm
