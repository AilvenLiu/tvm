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
 * \file tvm/tirx/async_structs.h
 * \brief Language structures for asynchronous execution in TIR+.
 */
#ifndef TVM_TIRX_ASYNC_STRUCTS_H_
#define TVM_TIRX_ASYNC_STRUCTS_H_

#include <tvm/ir/module.h>
#include <tvm/runtime/object.h>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/exec_scope.h>

namespace tvm {
namespace tirx {

// Pipeline
class PipelineNode : public Object {
 public:
  /*! \brief The thread scope of this pipeline */
  ExecScope thread_scope;
  /*! \brief The pipeline depth */
  size_t depth;
  /*! \brief Whether to separate producer and consumer threads */
  bool separate_pc;
  /*! \brief The name hint of the pipeline. */
  String name_hint;

  /*! \brief The workspace of the pipeline. */
  Map<String, tvm::tirx::Buffer> workspace;
  /*! \brief The schedule config of the pipeline. */
  Map<String, ObjectRef> schedule_config;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("thread_scope", &thread_scope);
    v->Visit("name_hint", &name_hint);
    v->Visit("depth", &depth);
    v->Visit("separate_pc", &separate_pc);
    v->Visit("workspace", &workspace);
    v->Visit("schedule_config", &schedule_config);
  }

  bool SEqualReduce(const PipelineNode* other, SEqualReducer equal) const {
    if (!equal(thread_scope, other->thread_scope)) return false;
    if (!equal(depth, other->depth)) return false;
    if (!equal(separate_pc, other->separate_pc)) return false;
    if (!equal(workspace, other->workspace)) return false;
    if (!equal(schedule_config, other->schedule_config)) return false;
    return equal.FreeVarEqualImpl(this, other);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(thread_scope);
    hash_reduce(depth);
    hash_reduce(separate_pc);
    hash_reduce(workspace);
    hash_reduce(schedule_config);
    hash_reduce.FreeVarHashImpl(this);
  }

  static constexpr const char* _type_key = "tirx.Pipeline";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  TVM_DECLARE_BASE_OBJECT_INFO(PipelineNode, Object);
};

class Pipeline : public ObjectRef {
 public:
  TVM_DLL explicit Pipeline(ExecScope thread_scope, size_t depth = 0, bool separate_pc = false,
                            String name_hint = "", Map<String, tvm::tirx::Buffer> workspace = {},
                            Map<String, ObjectRef> schedule_config = {});

  TVM_DEFINE_OBJECT_REF_METHODS(Pipeline, ObjectRef, PipelineNode);
};

// CopyPipeline
class CopyPipelineNode : public PipelineNode {
 public:
  static constexpr const char* _type_key = "tirx.CopyPipeline";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  TVM_DECLARE_FINAL_OBJECT_INFO(CopyPipelineNode, PipelineNode);
};

class CopyPipeline : public Pipeline {
 public:
  TVM_DLL explicit CopyPipeline(ExecScope thread_scope, size_t depth = 0, bool separate_pc = false,
                                String name_hint = "", Map<String, tvm::tirx::Buffer> workspace = {},
                                Map<String, ObjectRef> schedule_config = {});

  TVM_DEFINE_OBJECT_REF_METHODS(CopyPipeline, Pipeline, CopyPipelineNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(CopyPipelineNode);
};

}  // namespace tirxxxxx
}  // namespace tvm

#endif  // TVM_TIRX_ASYNC_STRUCTS_H_
