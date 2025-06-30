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
using FArgSanitizer = ffi::TypedFunction<void(tvm::Op, Array<ObjectRef>)>;

namespace callback {
/*! \brief The buffers allocated by the operator. */
constexpr const char* kPrivateAlloc = "private_alloc";
/*! \brief The initialization statement of the operator.
 *  which will be inserted at the beginning of the kernel
 */
constexpr const char* kDeviceInitStmt = "device_init_stmt";
/*! \brief The initialization statement of the operator.
 *  which will be inserted at the beginning of the kernel
 */
constexpr const char* kHostInitStmt = "host_init_stmt";
}  // namespace callback

/*!
 * \brief The context information of the kernel required by op schedule.
 */
class ScheduleContextNode : public Object {
 public:
  /*! \brief The target of the kernel. */
  Target target;
  /*! \brief The exec scope of the operator*/
  ExecScope exec_scope;
  /*! \brief The kernel launch parameters. */
  Map<String, PrimExpr> launch_params;
  /*! \brief A map from loop variables to their ranges. */
  Map<Var, Range> var_range_map;
  /*! \brief Whether the schedule context is only used for buffer allocation. */
  bool alloc_only;
  /*! \brief Callback to be handled when the operator is scheduled. */
  Map<String, ObjectRef> callbacks;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ScheduleContextNode>()
        .def_ro("target", &ScheduleContextNode::target)
        .def_ro("exec_scope", &ScheduleContextNode::exec_scope)
        .def_ro("launch_params", &ScheduleContextNode::launch_params)
        .def_ro("var_range_map", &ScheduleContextNode::var_range_map)
        .def_ro("alloc_only", &ScheduleContextNode::alloc_only)
        .def_ro("callbacks", &ScheduleContextNode::callbacks);
  }

  /*! \brief Add a buffer to be allocated in the kernel. */
  void AddAllocBuffer(Buffer buffer);

  /*! \brief Add an initialization statement to be inserted.
   *  \param stmt The statement to be inserted.
   *  \param host Whether the statement is a host statement.
   *  If True, the statement will be added to the host code (before the kernel).
   *  If False, the statement will be added to the kernel body (at the beginning of the kernel).
   */
  void AddInitStmt(Stmt stmt, bool host = false);

  static constexpr const char* _type_key = "tirp.ScheduleContext";
  static constexpr bool _type_has_method_sequal_reduce = false;
  static constexpr bool _type_has_method_shash_reduce = false;
  static constexpr bool _type_has_method_visit_attrs = false;
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
   * \param launch_params The kernel launch parameters.
   * \param var_range_map: A map from loop variables to their ranges.
   * \param alloc_only Whether the schedule context is only used for buffer allocation.
   * \param callbacks The callbacks to be handled when the operator is scheduled.
   */
  TVM_DLL ScheduleContext(Target target, ExecScope exec_scope,
                          Map<String, PrimExpr> launch_params = {},
                          Map<Var, Range> var_range_map = {}, bool alloc_only = false,
                          Map<String, ObjectRef> callbacks = {});

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(ScheduleContext, ObjectRef, ScheduleContextNode);
};

/*!
 * \brief The type of the function that schedules a TIR+ operator.
 * \param op The operator.
 * \param args The arguments.
 * \param context The schedule context.
 */
using FOpScheduler = ffi::TypedFunction<Stmt(tvm::Op, Array<ObjectRef>, ScheduleContext)>;

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
 *  pipe.init()
 */
TVM_DLL const Op& pipeline_init();

/*!
 * \brief See pesudo code below:
 *
 *  pipe.producer_acquire()
 */
TVM_DLL const Op& pipeline_producer_acquire();

/*!
 * \brief See pesudo code below:
 *
 *  pipe.producer_commit()
 */
TVM_DLL const Op& pipeline_producer_commit();

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

/*!
 * \brief See pesudo code below:
 *
 *  pipe.copy(BufferRegion dst, BufferRegion src)
 */
TVM_DLL const Op& pipeline_copy();

/*!
 * \brief See pesudo code below:
 *
 *  tvm_kernel_replace_point()
 */
TVM_DLL const Op& tvm_kernel_replace_point();

}  // namespace tirxxp
}  // namespace tirxx
}  // namespace tvm

#endif  // TVM_TIRX_TIRP_OP_H_
