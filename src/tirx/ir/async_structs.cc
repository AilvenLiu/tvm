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
 * \file async_structs.cc
 */

#include <tvm/tirx/async_structs.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/tirp_op.h>

namespace tvm {
namespace tirx {

/*************************** Pipeline ***************************/
TVM_REGISTER_NODE_TYPE(PipelineNode);

Pipeline::Pipeline(ExecScope thread_scope, size_t depth, bool separate_pc, String name_hint,
                   Map<String, Buffer> workspace, Map<String, ffi::Any> schedule_config) {
  auto n = make_object<PipelineNode>();
  n->thread_scope = std::move(thread_scope);
  n->name_hint = std::move(name_hint);
  n->depth = depth;
  n->separate_pc = separate_pc;
  n->workspace = std::move(workspace);
  n->schedule_config = std::move(schedule_config);
  data_ = std::move(n);
}

TVM_FFI_REGISTER_GLOBAL("tirx.Pipeline")
    .set_body_typed([](ExecScope thread_scope, size_t depth, bool separate_pc, String name_hint,
                       Map<String, Buffer> workspace, Map<String, ffi::Any> schedule_config) {
      return Pipeline(thread_scope, depth, separate_pc, name_hint, workspace, schedule_config);
    });

/*************************** CopyPipeline ***************************/
TVM_REGISTER_NODE_TYPE(CopyPipelineNode);

CopyPipeline::CopyPipeline(ExecScope thread_scope, size_t depth, bool separate_pc, String name_hint,
                           Map<String, Buffer> workspace, Map<String, ffi::Any> schedule_config) {
  auto n = make_object<CopyPipelineNode>();
  n->thread_scope = std::move(thread_scope);
  n->name_hint = std::move(name_hint);
  n->depth = depth;
  n->separate_pc = separate_pc;
  n->workspace = std::move(workspace);
  n->schedule_config = std::move(schedule_config);
  data_ = std::move(n);
}

TVM_FFI_REGISTER_GLOBAL("tirx.CopyPipeline")
    .set_body_typed([](ExecScope thread_scope, size_t depth, bool separate_pc, String name_hint,
                       Map<String, Buffer> workspace, Map<String, ffi::Any> schedule_config) {
      return CopyPipeline(thread_scope, depth, separate_pc, name_hint, workspace, schedule_config);
    });

}  // namespace tirxxxxxxxx
}  // namespace tvm
