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
#include <tvm/tirx/exec_scope.h>
#include <tvm/tirx/op.h>

namespace tvm {
namespace tirx {

/******** Constructors ********/

ExecScope::ExecScope(Array<PrimExpr> dims, String name) {
  auto n = make_object<ExecScopeNode>();
  ICHECK_LE(dims.size(), 3) << "Only support at most 3 dimensions";
  n->dims = std::move(dims);
  n->name = std::move(name);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(ExecScopeNode);

TVM_REGISTER_GLOBAL("tirx.ExecScope").set_body_typed([](Array<PrimExpr> dims, String name) {
  return ExecScope(dims, name);
});

ThreadingVar::ThreadingVar(ExecScope scope, String name) {
  auto n = make_object<ThreadingVarNode>();
  n->scope = std::move(scope);
  n->type_annotation = GetTypeFromRuntimeDataType(DataType::Int(32));
  n->dtype = DataType::Int(32);
  n->name_hint = std::move(name);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(ThreadingVarNode);

TVM_REGISTER_GLOBAL("tirx.ThreadingVar").set_body_typed([](ExecScope scope, String name) {
  return ThreadingVar(scope, name);
});

SubExecScope::SubExecScope(Array<ThreadingVar> vars, Optional<Array<Range>> ranges, String name) {
  auto n = make_object<SubExecScopeNode>();
  ICHECK(!vars.empty()) << "SubExecScope must have at least one threading variable";
  if (ranges.defined()) {
    ICHECK_EQ(vars.size(), ranges.value().size()) << "Number of dimensions must match";
  }
  ICHECK_EQ(vars[0]->scope->size(), vars.size()) << "Number of dimensions must match";
  n->def_vars = std::move(vars);
  n->ranges = std::move(ranges);
  n->name = std::move(name);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(SubExecScopeNode);

TVM_REGISTER_GLOBAL("tirx.SubExecScope")
    .set_body_typed([](Array<ThreadingVar> vars, Optional<Array<Range>> ranges, String name) {
      return SubExecScope(vars, ranges, name);
    });

}  // namespace tirx
}  // namespace tvm
