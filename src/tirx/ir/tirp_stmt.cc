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
 * \file tir/tirp_stmt.cc
 * TIR+ statement nodes.
 */

#include <tvm/tirx/op.h>
#include <tvm/tirx/op_attr_types.h>
#include <tvm/tirx/tirp_op.h>

namespace tvm {
namespace tirx {
namespace tirp {

// OpCall
OpCall::OpCall(tvm::Op op, Array<ObjectRef> args, Map<String, Buffer> workspace, Map<String, ObjectRef> schedule_config) {
  // Check if the op is a TIR+ op.
  static const auto& tirp_op_map = Op::GetAttrMap<Bool>("TIsTIRpOp");
  ICHECK_EQ(tirp_op_map.count(op), 1) << "Only TIR+ ops can be used in tirx::tirp::OpCall";
  // Construct the OpCall.
  ObjectPtr<OpCallNode> n = make_object<OpCallNode>();
  n->op = std::move(op);
  n->args = std::move(args);
  n->workspace = std::move(workspace);
  n->schedule_config = std::move(schedule_config);
  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("tirx.OpCall").set_body_typed([](tvm::Op op, Array<ObjectRef> args, Map<String, Buffer> workspace, Map<String, ObjectRef> schedule_config) {
  return OpCall(op, args, workspace, schedule_config);
});

TVM_REGISTER_GLOBAL("tir.OpCallCopyHandle").set_body_typed([](const OpCall& op) {
  return OpCall(op);
});

TVM_REGISTER_NODE_TYPE(OpCallNode);

}  // namespace tirxxxxp
}  // namespace tirxxxx
}  // namespace tvm
