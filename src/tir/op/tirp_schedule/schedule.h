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
#ifndef TVM_TIR_OP_TIRP_SCHEDULE_SCHEDULE_H_
#define TVM_TIR_OP_TIRP_SCHEDULE_SCHEDULE_H_

#include <tvm/arith/analyzer.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/target/tag.h>
#include <tvm/tir/exec_scope.h>
#include <tvm/tir/layout.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/tirp_op.h>

namespace tvm {
namespace tir {
namespace tirp {

/********************* Utils **********************/
bool IsCUDA(const Target& target);

Stmt CallBuiltinOp(const Op& op, const Array<PrimExpr>& args);

/********************* Copy Ops **********************/
enum class CopyInstType { kBufferLoad, kCUDAcpasync };

Stmt VectorizedCopy(const BufferRegion& src, const BufferRegion& dst, ScheduleContext context,
                    CopyInstType inst_type);

Stmt CopyOpScheduler(const Op& op, Array<ObjectRef> args, ScheduleContext context);

/********************* Barrier Ops **********************/
Stmt BarrierOpScheduler(const Op& op, Array<ObjectRef> args, ScheduleContext context);

/********************* Pipeline Ops **********************/
Stmt PipelineOpScheduler(const Op& op, Array<ObjectRef> args, ScheduleContext context);

}  // namespace tirp
}  // namespace tir
}  // namespace tvm

#endif  // TVM_TIR_OP_TIRP_SCHEDULE_SCHEDULE_H_
