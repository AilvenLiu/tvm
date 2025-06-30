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

TVM_FFI_STATIC_INIT_BLOCK({ ScheduleContextNode::RegisterReflection(); });

/********************* Utils **********************/

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
                                 bool alloc_only, Map<String, ObjectRef> callbacks) {
  auto n = make_object<ScheduleContextNode>();
  n->target = std::move(target);
  n->exec_scope = std::move(exec_scope);
  n->launch_params = std::move(launch_params);
  n->var_range_map = std::move(var_range_map);
  n->alloc_only = alloc_only;
  n->callbacks = std::move(callbacks);
  data_ = std::move(n);
}

TVM_FFI_REGISTER_GLOBAL("tirp.ScheduleContext")
    .set_body_typed([](Target target, ExecScope exec_scope, Map<String, PrimExpr> launch_params,
                       Map<Var, Range> var_range_map, bool alloc_only,
                       Map<String, ObjectRef> callbacks) {
      return ScheduleContext(target, exec_scope, launch_params, var_range_map, alloc_only,
                             callbacks);
    });

TVM_FFI_REGISTER_GLOBAL("tirp.ScheduleContextAddAllocBuffer")
    .set_body_method(&ScheduleContextNode::AddAllocBuffer);
TVM_FFI_REGISTER_GLOBAL("tirp.ScheduleContextAddInitStmt")
    .set_body_method(&ScheduleContextNode::AddInitStmt);

/********************* Schedule Ops **********************/
#define TIRP_DEFINE_SCHEDULE_OP(OpName) \
  TIRP_DEFINE_OP(OpName).set_attr<Bool>("TIsScheduleOp", Bool(true))

TIRP_DEFINE_SCHEDULE_OP(zero);
TIRP_DEFINE_SCHEDULE_OP(sqrt);
TIRP_DEFINE_SCHEDULE_OP(exp);
TIRP_DEFINE_SCHEDULE_OP(add);
TIRP_DEFINE_SCHEDULE_OP(sub);
TIRP_DEFINE_SCHEDULE_OP(mul);
TIRP_DEFINE_SCHEDULE_OP(fdiv);
TIRP_DEFINE_SCHEDULE_OP(minimum);
TIRP_DEFINE_SCHEDULE_OP(maximum);
TIRP_DEFINE_SCHEDULE_OP(copy);
TIRP_DEFINE_SCHEDULE_OP(fill);
TIRP_DEFINE_SCHEDULE_OP(gemm);
TIRP_DEFINE_SCHEDULE_OP(reciprocal);
TIRP_DEFINE_SCHEDULE_OP(sum);
TIRP_DEFINE_SCHEDULE_OP(max);
TIRP_DEFINE_SCHEDULE_OP(min);
TIRP_DEFINE_SCHEDULE_OP(memset);
TIRP_DEFINE_SCHEDULE_OP(reduce_negate);
TIRP_DEFINE_SCHEDULE_OP(binary_reduce);
TIRP_DEFINE_SCHEDULE_OP(unary_reduce);
TIRP_DEFINE_SCHEDULE_OP(binary_chain);
TIRP_DEFINE_SCHEDULE_OP(select);

/********************* Compose Ops **********************/
#define TIRP_DEFINE_COMPOSE_OP(OpName) \
  TIRP_DEFINE_OP(OpName).set_attr<Bool>("TIsComposeOp", Bool(true))

TIRP_DEFINE_COMPOSE_OP(compose_op);

/********************* Pipeline Ops **********************/
#define TIRP_DEFINE_PIPELINE_OP(OpName) \
  TIRP_DEFINE_OP(OpName).set_attr<Bool>("TIsPipelineOp", Bool(true))

TIRP_DEFINE_PIPELINE_OP(pipeline_init);
TIRP_DEFINE_PIPELINE_OP(pipeline_producer_acquire);
TIRP_DEFINE_PIPELINE_OP(pipeline_copy);
TIRP_DEFINE_PIPELINE_OP(pipeline_producer_commit);
TIRP_DEFINE_PIPELINE_OP(pipeline_consumer_wait);
TIRP_DEFINE_PIPELINE_OP(pipeline_consumer_release);

/********************* Misc Ops **********************/
TIRP_DEFINE_OP(tvm_kernel_replace_point);

}  // namespace tirxxp
}  // namespace tirxx
}  // namespace tvm
