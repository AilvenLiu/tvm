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
 * \file tir/tirx_stmt.cc
 * TIRX statement nodes.
 */

<<<<<<<< HEAD:src/tirx/ir/tirp_stmt.cc
#include <tvm/tirx/op.h>
#include <tvm/tirx/op_attr_types.h>
#include <tvm/tirx/tirp_op.h>

namespace tvm {
namespace tirx {
namespace tirp {
========
#include <tvm/tirx/op.h>
#include <tvm/tirx/op_attr_types.h>
#include <tvm/tirx/tirx_op.h>

namespace tvm {
namespace tirx {
namespace tirx {
>>>>>>>> bd927f10c7 (rename):src/tir/ir/tirx_stmt.cc

TVM_FFI_STATIC_INIT_BLOCK() { OpCallNode::RegisterReflection(); }

// OpCall
OpCall::OpCall(tvm::Op op, ffi::Array<ffi::Any> args, ffi::Map<ffi::String, Buffer> workspace,
               ffi::Map<ffi::String, ffi::Any> config, ffi::Optional<ffi::String> dispatch) {
<<<<<<<< HEAD:src/tirx/ir/tirp_stmt.cc
  // Check if the op is a TIR+ op.
  static const auto& tirp_op_map = Op::GetAttrMap<Bool>("TIsTIRpOp");
  ICHECK_EQ(tirp_op_map.count(op), 1) << "Only TIR+ ops can be used in tirx::tirp::OpCall";
========
  // Check if the op is a TIRX op.
  static const auto& tirx_op_map = Op::GetAttrMap<Bool>("TIsTIRxOp");
  ICHECK_EQ(tirx_op_map.count(op), 1) << "Only TIRX ops can be used in tirx::tirx::OpCall";
>>>>>>>> bd927f10c7 (rename):src/tir/ir/tirx_stmt.cc
  // Construct the OpCall.
  ObjectPtr<OpCallNode> n = ffi::make_object<OpCallNode>();
  n->op = std::move(op);
  n->args = std::move(args);
  n->workspace = std::move(workspace);
  n->config = std::move(config);
  n->dispatch = std::move(dispatch);
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "tirx.OpCall",
      [](tvm::Op op, ffi::Array<ffi::Any> args, ffi::Map<ffi::String, Buffer> workspace,
         ffi::Map<ffi::String, ffi::Any> config, ffi::Optional<ffi::String> dispatch) {
        return OpCall(op, args, workspace, config, dispatch);
      });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.OpCallCopyHandle", [](const OpCall& op) { return OpCall(op); });
}

<<<<<<<< HEAD:src/tirx/ir/tirp_stmt.cc
}  // namespace tirxxxxxp
}  // namespace tirxxxxx
========
}  // namespace tirxx
}  // namespace tirx
>>>>>>>> bd927f10c7 (rename):src/tir/ir/tirx_stmt.cc
}  // namespace tvm
