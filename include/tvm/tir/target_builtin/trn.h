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
 * \file tvm/tir/target_builtin/trn.h
 * \brief TIR builtin intrinsics specific to Trainium target.
 */
#ifndef TVM_TIR_TARGET_BUILTIN_TRN_H_
#define TVM_TIR_TARGET_BUILTIN_TRN_H_

#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>

namespace tvm {
namespace tir {
namespace builtin {

/*!
 * \brief nki intrinsics for load operation.
 *
 * nki_load(result, data)
 */
TVM_DLL const Op& nki_load();
/*!
 * \brief nki intrinsics for store operation.
 *
 * nki_store(result, data)
 */
TVM_DLL const Op& nki_store();
/*!
 * \brief nki intrinsics for tensor_copy operation.
 *
 * nki_tensor_copy(result, data)
 */
TVM_DLL const Op& nki_tensor_copy();
/*!
 * \brief nki intrinsics for matmul operation.
 *
 * nki_matmul(C, A, B, accum)
 *
 * equivalent to C += A.T @ B (if accum is true), or C = A.T @ B (if accum is false)
 */
TVM_DLL const Op& nki_matmul();
/*!
 * \brief nki intrinsics for transpose operation.
 *
 * nki_transpose(result, data)
 *
 * The tensor must be 128x128.
 */

TVM_DLL const Op& nki_transpose();
/*!
 * \brief nki intrinsics for sum operation.
 *
 * nki_sum(result, data, axis)
 */
TVM_DLL const Op& nki_sum();

/*!
 * \brief nki intrinsics for activation operation.
 *
 * nki_activation(result, data, opcode, bias, scale)
 */
TVM_DLL const Op& nki_activation();

/*!
 * \brief nki intrinsics for reciprocal operation.
 *
 * nki_reciprocal(result, data)
 */
TVM_DLL const Op& nki_reciprocal();

/*!
 * \brief nki intrinsics for tensortensor operation.
 *
 * nki_tensortensor(result, operand1, operand2, opcode)
 */
TVM_DLL const Op& nki_tensortensor();

/*!
 * \brief nki intrinsics for tensorscalar operation.
 *
 * nki_tensortensor(result, operand1, operand2, opcode, reorder)
 */
TVM_DLL const Op& nki_tensorscalar();

/*!
 * \brief nki intrinsics for tensorreduce operation.
 *
 * nki_tensorreduce(result, data, opcode, axes)
 */
TVM_DLL const Op& nki_tensorreduce();

/*!
 * \brief nki intrinsics for memset operation.
 *
 * nki_memset(result, value)
 */
TVM_DLL const Op& nki_memset();

/*!
 * \brief nki intrinsics for activation reduce operation.
 *
 * nki_activation_reduce(reduce_res, act_res, data, opcode, reduce_opcode)
 */
TVM_DLL const Op& nki_activation_reduce();

/*!
 * \brief nki intrinsics for tensorscalar reduce operation.
 *
 * nki_tensorscalar_reduce(reduce_res, tensorscalar_res, operand1, operand2, opcode, reduce_opcode, reorder)
 */
TVM_DLL const Op& nki_tensorscalar_reduce();


}  // namespace builtin
}  // namespace tir
}  // namespace tvm

#endif  // TVM_TIR_TARGET_BUILTIN_TRN_H_
