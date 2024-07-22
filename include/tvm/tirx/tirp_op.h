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
 * \brief The map from the scope id (defined by two scopes) to the extent of the scope id
 */
using ScopeExtentMap = Map<ScopeIdDef, PrimExpr>;

/*!
 * \brief The map from the thread variable name to the variable.
 */
using ThreadVarMap = Map<String, Var>;

/*!
 * \brief The context information of the kernel required by op schedule.
 */
class ScheduleContextNode : public Object {
 public:
  /*! \brief The target of the kernel. */
  Target target;
  /*! \brief The exec scope of the operator*/
  ExecScope exec_scope;
  /*! \brief The thread variables. */
  ThreadVarMap thread_var_map;
  /*! \brief The scope extent map. */
  ScopeExtentMap scope_extent_map;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("target", &target);
    v->Visit("exec_scope", &exec_scope);
    v->Visit("thread_var_map", &thread_var_map);
    v->Visit("scope_extent_map", &scope_extent_map);
  }

  bool SEqualReduce(const ScheduleContextNode* other, SEqualReducer equal) const {
    return equal(target, other->target) && equal(exec_scope, other->exec_scope) &&
           equal(thread_var_map, other->thread_var_map) &&
           equal(scope_extent_map, other->scope_extent_map);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(target);
    hash_reduce(exec_scope);
    hash_reduce(thread_var_map);
    hash_reduce(scope_extent_map);
  }

  static constexpr const char* _type_key = "tir.ScheduleContext";
  static constexpr bool _type_has_method_sequal_reduce = true;
  static constexpr bool _type_has_method_shash_reduce = true;
  TVM_DECLARE_FINAL_OBJECT_INFO(ScheduleContextNode, Object);
};

/*!
 * \brief Managed reference to ScheduleContextNode.
 */
class ScheduleContext : public ObjectRef {
 public:
  /*!
   * \brief Constructor.
   * \param target The target of the kernel.
   * \param exec_scope The exec scope of the operator.
   * \param thread_var_map The thread variables.
   * \param scope_extent_map The scope extent map.
   */
  TVM_DLL ScheduleContext(Target target, ExecScope exec_scope, ThreadVarMap thread_var_map,
                          ScopeExtentMap scope_extent_map);

  /*!
   * \brief Get the extent of the scope.
   * \param scope_id The scope defined by two exec scopes.
   * \return The extent of the scope.
   */
  PrimExpr GetScopeExtent(const ScopeIdDef& scope_id) const;

  /*!
   * \brief Get the thread variable.
   * \param name The name of the thread variable.
   * \return The thread variable.
   */
  Var GetThreadVar(const String& name) const;

  TVM_DEFINE_OBJECT_REF_METHODS(ScheduleContext, ObjectRef, ScheduleContextNode);
};

/*!
 * \brief The type of the function that schedules a TIR+ operator.
 * \param op The operator.
 * \param args The arguments.
 * \param context The schedule context.
 */
using FOpScheduler = runtime::TypedPackedFunc<Stmt(tvm::Op, Array<ObjectRef>, ScheduleContext)>;

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
TVM_DLL const Op& pipeline_producer_acquire();

/*!
 * \brief See pesudo code below:
 *
 *  pipe.producer_copy_async(BufferRegion dst, BufferRegion src)
 */
TVM_DLL const Op& pipeline_producer_copy_async();

/*!
 * \brief See pesudo code below:
 *
 *  pipe.producer_commit_stage()
 */
TVM_DLL const Op& pipeline_producer_commit_stage();

/*!
 * \brief See pesudo code below:
 *
 *  pipe.consumer_wait(size_t num_stages)
 */
TVM_DLL const Op& pipeline_consumer_wait();

/*!
 * \brief See pesudo code below:
 *
 *  pipe.consumer_release()
 */
TVM_DLL const Op& pipeline_consumer_release();

}  // namespace tirxxp
}  // namespace tirxx
}  // namespace tvm

#endif  // TVM_TIRX_TIRP_OP_H_
