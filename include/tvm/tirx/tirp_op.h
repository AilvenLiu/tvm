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
 * \file tvm/tirx/tirp_op.h
 * \brief TIR+ built-in operators.
 */
#ifndef TVM_TIRX_TIRP_OP_H_
#define TVM_TIRX_TIRP_OP_H_

#include <tvm/ir/op.h>
#include <tvm/target/target.h>
#include <tvm/tirx/exec_scope.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/tirp_stmt.h>

namespace tvm {
namespace tirx {
namespace tirp {

/*!
 * \brief The type of the function that sanitizes the arguments of a TIR+ operator.
 * \param op The operator.
 * \param args The arguments.
 */
using FArgSanitizer = runtime::TypedPackedFunc<void(tvm::Op, Array<ObjectRef>)>;

/*!
 * \brief The type of the function that schedules a TIR+ operator.
 * \param op The operator.
 * \param target The target.
 * \param exec_scope The execution scope.
 * \param args The arguments.
 */
using FOpScheduler = runtime::TypedPackedFunc<Stmt(tvm::Op, Target, ExecScope, Array<ObjectRef>)>;

/*!
 * \brief See pesudo code below:
 *
 * Tp.copy(BufferRegion dst, BufferRegion src)
 */
TVM_DLL const Op& copy();

/*!
 * \brief See pesudo code below:
 *
 *  Tp.fill(BufferRegion dst, PrimExpr value)
 */
TVM_DLL const Op& fill();

/*!
 * \brief See pesudo code below:
 *
 * Tp.gemm(Buffer A, Buffer B, Buffer C, Buffer D, PrimExpr alpha, PrimExpr beta)
 */
TVM_DLL const Op& gemm();

/*!
 * \brief See pesudo code below:
 *
 *  barrier.init(count)
 */
TVM_DLL const Op& barrier_init();

/*!
 * \brief See pesudo code below:
 *
 *  barrier.arrive()
 */
TVM_DLL const Op& barrier_arrive();

/*!
 * \brief See pesudo code below:
 *
 *  barrier.wait()
 */
TVM_DLL const Op& barrier_wait();

/*!
 * \brief See pesudo code below:
 *
 * barrier.arrive_and_wait()
 */
TVM_DLL const Op& barrier_arrive_and_wait();

/*!
 * \brief See pesudo code below:
 *
 *  pipe.producer_acquire()
 */
TVM_DLL const Op& producer_acquire();

/*!
 * \brief See pesudo code below:
 *
 *  pipe.producer_copy_async(BufferRegion dst, BufferRegion src)
 */
TVM_DLL const Op& producer_copy_async();

/*!
 * \brief See pesudo code below:
 *
 *  pipe.producer_commit_stage()
 */
TVM_DLL const Op& producer_commit_stage();

/*!
 * \brief See pesudo code below:
 *
 *  pipe.consumer_wait(size_t num_stages)
 */
TVM_DLL const Op& consumer_wait();

/*!
 * \brief See pesudo code below:
 *
 *  pipe.consumer_release()
 */
TVM_DLL const Op& consumer_release();

}  // namespace tirxxp
}  // namespace tirxx
}  // namespace tvm

#endif  // TVM_TIRX_TIRP_OP_H_
