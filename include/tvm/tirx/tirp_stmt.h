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
 * \brief TIR+ statements.
 */
#ifndef TVM_TIRX_TIRP_STMT_H_
#define TVM_TIRX_TIRP_STMT_H_

#include <tvm/ir/op.h>
#include <tvm/tirx/stmt.h>

namespace tvm {
namespace tirx {
namespace tirp {

/*!
 * \brief TIR+ OpCall stmt.
 */
class OpCallNode : public StmtNode {
 public:
  // tvm::Op which corresponds to the TIR+ operator.
  tvm::Op op;

  // Arguments to the operator.
  Array<ffi::Any> args;

  // Workspace (pre-allocated buffers) for the operator.
  Map<String, Buffer> workspace;

  // Schedule config for the operator.
  Map<String, ffi::Any> schedule_config;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<OpCallNode>()
        .def_ro("op", &OpCallNode::op)
        .def_ro("args", &OpCallNode::args)
        .def_ro("workspace", &OpCallNode::workspace)
        .def_ro("schedule_config", &OpCallNode::schedule_config);
  }

  static constexpr const char* _type_key = "tirx.OpCall";
  TVM_DECLARE_FINAL_OBJECT_INFO(OpCallNode, StmtNode);
};

/*!
 * \brief Managed reference to OpCallNode
 * \sa OpCallNode
 */
class OpCall : public Stmt {
 public:
  TVM_DLL OpCall(tvm::Op op, Array<ffi::Any> args, Map<String, Buffer> workspace = {},
                 Map<String, ffi::Any> schedule_config = {});

  static bool IsValidOpCallArgType(const ffi::Any& arg);

  TVM_DEFINE_OBJECT_REF_METHODS(OpCall, Stmt, OpCallNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(OpCallNode);
};

}  // namespace tirxxxp
}  // namespace tirxxx
}  // namespace tvm

#endif  // TVM_TIRX_TIRP_STMT_H_
