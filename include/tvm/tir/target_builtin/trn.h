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

}  // namespace builtin
}  // namespace tir
}  // namespace tvm

#endif  // TVM_TIR_TARGET_BUILTIN_TRN_H_
