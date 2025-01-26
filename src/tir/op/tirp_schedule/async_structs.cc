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
 * \file tir/op/tirp_schedule/async_structs.cc
 * \brief Schedule of TIR+ operators: async structs.
 */

#include <tvm/tir/builtin.h>
#include <tvm/tir/tirp_op.h>

#include "schedule.h"

namespace tvm {
namespace tir {
namespace tirp {

Stmt CallBuiltinOp(const Op& op, const Array<PrimExpr>& args) {
  return Evaluate(Call(DataType::Void(), op, args));
}

/********************* Barrier Ops **********************/
Stmt BarrierOpScheduler(const Op& op, Array<ObjectRef> args, ScheduleContext context) {
  const auto& target = context->target;
  const auto& exec_scope = context->exec_scope;
  // Currently only supports CUDA target
  CHECK(IsCUDA(target)) << "ValueError: BarrierOp only supports CUDA target";
  CHECK_GT(args.size(), 0U) << "ValueError: BarrierOp expects at least 1 argument";
  // We expect all the barrier object has to be barrier array element
  auto barrier = args[0].as<BarrierArrayElemNode>();
  CHECK(barrier != nullptr)
      << "ValueError: BarrierOp expects BarrierArrayElemNode as the first argument";
  // Currently only supports barrier executes under thread scope
  CHECK(exec_scope.Is("thread")) << "ValueError: BarrierOp only supports thread as exec scope";
  CHECK(barrier->thread_scope.Is("cta"))
      << "ValueError: BarrierOp only supports cta scope as thread scope";

  if (op.same_as(barrier_init())) {
    CHECK_EQ(args.size(), 3U) << "ValueError: barrier_init expects 2 arguments";
    return CallBuiltinOp(builtin::cuda_barrier_init(),
                         {ToIntImm(args[1]), barrier->index, ToIntImm(args[2])});
  } else if (op.same_as(barrier_arrive())) {
    CHECK_EQ(args.size(), 2U) << "ValueError: barrier_arrive expects 1 argument";
    return CallBuiltinOp(builtin::cuda_barrier_arrive(), {barrier->index, ToIntImm(args[1])});
  } else if (op.same_as(barrier_wait())) {
    CHECK_EQ(args.size(), 2U) << "ValueError: barrier_wait expects 1 argument";
    return CallBuiltinOp(builtin::cuda_barrier_wait(), {barrier->index, ToIntImm(args[1])});
  } else if (op.same_as(barrier_arrive_and_wait())) {
    CHECK_EQ(args.size(), 2U) << "ValueError: barrier_arrive_and_wait expects 1 argument";
    return CallBuiltinOp(builtin::cuda_barrier_arrive_and_wait(),
                         {barrier->index, ToIntImm(args[1])});
  }
  LOG(FATAL) << "ValueError: Unsupported BarrierOp " << op;
  throw;
}

/********************* Pipeline Ops **********************/
Stmt PipelineOpScheduler(const Op& op, Array<ObjectRef> args, ScheduleContext context) {
  const auto& target = context->target;
  const auto& exec_scope = context->exec_scope;
  // Currently only supports CUDA target
  CHECK(IsCUDA(target)) << "ValueError: PipelineOp only supports CUDA target";
  CHECK_GT(args.size(), 0U) << "ValueError: PipelineOp expects at least 1 argument";
  auto pipeline = args[0].as<PipelineNode>();
  CHECK(pipeline != nullptr) << "ValueError: PipelineOp expects PipelineNode as the first argument";
  // Currently only supports pipeline executes under cta scope
  CHECK(exec_scope.Is("cta")) << "ValueError: PipelineOp only supports cta as exec scope";
  CHECK(pipeline->thread_scope.Is("cta"))
      << "ValueError: PipelineOp only supports cta scope as thread scope";
  const auto& arch_str_opt = target->GetAttr<String>("arch");
  ICHECK(arch_str_opt.defined()) << "InternalError: CUDA architecture is not defined in the target";
  std::string arch_str = arch_str_opt.value();
  // arch_str is expected to be in the format of "sm_XX"
  ICHECK(arch_str.rfind("sm_", 0) == 0)
      << "InternalError: Unsupported CUDA architecture " << arch_str;
  int sm_version = std::stoi(arch_str.substr(3));

  auto wrap = [&](const Stmt& body) -> Stmt {
    return BlockRealize({}, Bool(true), Block("pipeline", body, ExecScope::Create("thread")));
  };

  // No specialize, no pipeline depth
  if (pipeline->depth == 0 && !pipeline->specialize) {
    // Case 1: No specialize, no pipeline depth
    if (sm_version >= 80) {
      if (op.same_as(pipeline_producer_acquire())) {
        // producer_acquire is a no-op
        LOG(FATAL) << "ValueError: pipeline with depth 0 and specialize false is not supported for "
                      "producer_acquire";
      } else if (op.same_as(pipeline_producer_copy_async())) {
        // producer_copy_async
        CHECK_EQ(args.size(), 3U) << "ValueError: pipeline_producer_copy_async expects 3 arguments";
        const auto& dst = Downcast<BufferRegion>(args[1]);
        const auto& src = Downcast<BufferRegion>(args[2]);
        LOG(FATAL) << "not implemented";
      } else if (op.same_as(pipeline_producer_commit_stage())) {
        // producer_commit_stage
        CHECK_EQ(args.size(), 1U)
            << "ValueError: pipeline_producer_commit_stage expects 1 argument";
        return wrap(CallBuiltinOp(builtin::ptx_commit_group(), {}));
      } else if (op.same_as(pipeline_consumer_wait())) {
        // consumer_wait
        CHECK_EQ(args.size(), 2U) << "ValueError: pipeline_consumer_wait expects 2 arguments";
        return wrap(SeqStmt({CallBuiltinOp(builtin::ptx_wait_group(), {ToIntImm(args[1])}),
                             CallBuiltinOp(builtin::tvm_storage_sync(), {StringImm("shared")})}));
      } else if (op.same_as(pipeline_consumer_release())) {
        // consumer_release is a no-op
        LOG(FATAL) << "ValueError: pipeline with depth 0 and specialize false is not supported for "
                      "consumer_release";
      }
      LOG(FATAL) << "ValueError: Unsupported PipelineOp " << op;
      throw;
    }
    LOG(FATAL) << "ValueError: CUDA architecture " << arch_str << " " << sm_version
               << " is not supported for pipeline without specialize and depth";
    throw;
  }
  LOG(FATAL) << "ValueError: Unsupported pipeline with depth " << pipeline->depth
             << " and specialize " << pipeline->specialize;
  throw;
}

}  // namespace tirp
}  // namespace tir
}  // namespace tvm
