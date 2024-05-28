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

void ArgSanitizer(tvm::Op op, Array<ObjectRef> args) {
  if (op.same_as(copy())) {
    // copy(src, dst)
    ICHECK_EQ(args.size(), 2U) << "copy() expects 2 arguments";
    ICHECK(args[0].as<BufferRegionNode>()) << "arg[0] of copy() must be BufferRegion";
    ICHECK(args[1].as<BufferRegionNode>()) << "arg[1] of copy() must be BufferRegion";
  } else if (op.same_as(fill())) {
    // fill(dst, value)
    ICHECK_EQ(args.size(), 2U) << "fill() expects 2 arguments";
    ICHECK(args[0].as<BufferRegionNode>()) << "arg[0] of fill() must be BufferRegion";
    ICHECK(args[1].as<PrimExprNode>()) << "arg[1] of fill() must be PrimExpr";
  } else if (op.same_as(gemm())) {
    // gemm(A, B, C, D, alpha, beta)
    ICHECK_EQ(args.size(), 6U) << "gemm() expects 6 arguments";
    ICHECK(args[0].as<BufferNode>()) << "arg[0] of gemm() must be Buffer";
    ICHECK(args[1].as<BufferNode>()) << "arg[1] of gemm() must be Buffer";
    ICHECK(args[2].as<BufferNode>()) << "arg[2] of gemm() must be Buffer";
    ICHECK(args[3].as<BufferNode>()) << "arg[3] of gemm() must be Buffer";
    ICHECK(args[4].as<PrimExprNode>()) << "arg[4] of gemm() must be PrimExpr";
    ICHECK(args[5].as<PrimExprNode>()) << "arg[5] of gemm() must be PrimExpr";
  } else {
    LOG(FATAL) << "Unknown TIR+ op " << op->name;
  }
}

TIRP_DEFINE_BUILTIN_FUNC(copy)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kPure))
    .set_attr<FArgSanitizer>("FArgSanitizer", ArgSanitizer)
    .set_attr<Bool>("TIsTIRpOp", Bool(true))
    .set_num_inputs(2);

TIRP_DEFINE_BUILTIN_FUNC(fill)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kPure))
    .set_attr<FArgSanitizer>("FArgSanitizer", ArgSanitizer)
    .set_attr<Bool>("TIsTIRpOp", Bool(true))
    .set_num_inputs(2);

TIRP_DEFINE_BUILTIN_FUNC(gemm)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kPure))
    .set_attr<FArgSanitizer>("FArgSanitizer", ArgSanitizer)
    .set_attr<Bool>("TIsTIRpOp", Bool(true))
    .set_num_inputs(6);

}  // namespace tirxp
}  // namespace tirx
}  // namespace tvm
