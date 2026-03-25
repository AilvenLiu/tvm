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
<<<<<<<< HEAD:include/tvm/tirx/tirp_op.h
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
========
 * \file tvm/tir/tirx_op.h
 * \brief TIRX built-in operators.
 */
#ifndef TVM_TIR_TIRX_OP_H_
#define TVM_TIR_TIRX_OP_H_

#include <tvm/ir/op.h>
#include <tvm/target/target.h>
#include <tvm/tir/exec_scope.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/tirx_stmt.h>

namespace tvm {
namespace tir {
namespace tirx {
>>>>>>>> bd927f10c7 (rename):include/tvm/tir/tirx_op.h

/*!
 * \brief The type of the function that sanitizes the arguments of a TIRX operator.
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
/*! \brief Statements to be inserted after a specific buffer's definition (DeclBuffer/AllocBuffer).
 *  Stored as Map<Buffer, Array<Stmt>>.
 */
constexpr const char* kPostBufferDefStmt = "post_buffer_def_stmt";
}  // namespace callback

/*!
 * \brief The context information of the kernel required by op dispatch.
 */
class DispatchContextNode : public Object {
 public:
  /*! \brief The target of the kernel. */
  Target target;
  /*! \brief The exec scope of the operator */
  ExecScope exec_scope;
  /*! \brief The kernel launch parameters. */
  ffi::Map<ffi::String, IterVar> launch_params;
  /*! \brief A map from loop variables to their ranges. */
  ffi::Map<Var, Range> var_range_map;
  /*! \brief Whether the dispatch context is only used for buffer allocation. */
  bool alloc_only;
  /*! \brief Callback to be handled when the operator is scheduled. */
  ffi::Map<ffi::String, ObjectRef> callbacks;
  /*! \brief Shared state that persists across dispatch calls within a single lowering pass.
   *  Used by dispatch implementations to cache and reuse descriptors/tensormaps.
   */
  ffi::Map<ffi::String, ObjectRef> shared_state;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<DispatchContextNode>()
        .def_ro("target", &DispatchContextNode::target)
        .def_ro("exec_scope", &DispatchContextNode::exec_scope)
        .def_ro("launch_params", &DispatchContextNode::launch_params)
        .def_ro("var_range_map", &DispatchContextNode::var_range_map)
        .def_ro("alloc_only", &DispatchContextNode::alloc_only)
        .def_ro("callbacks", &DispatchContextNode::callbacks)
        .def_ro("shared_state", &DispatchContextNode::shared_state);
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

  /*! \brief Add a statement to be inserted after a buffer's definition.
   *  \param buffer The buffer whose DeclBuffer/AllocBuffer scope the stmt should appear in.
   *  \param stmt The statement to be inserted.
   */
  void AddPostBufferDefStmt(Buffer buffer, Stmt stmt);

  /*! \brief Set a value in the shared state cache.
   *  \param key The cache key.
   *  \param value The value to cache.
   */
  void SharedStateSet(ffi::String key, ObjectRef value);

  /*! \brief Get a value from the shared state cache.
   *  \param key The cache key.
   *  \return The cached value, or NullOpt if not found.
   */
  ffi::Optional<ObjectRef> SharedStateGet(ffi::String key);

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.DispatchContext", DispatchContextNode, Object);
};

/*!
 * \brief Managed reference to DispatchContextNode.
 */
class DispatchContext : public ObjectRef {
 public:
  /*!
   * \brief Constructor.
   * \param target The target of the kernel.
   * \param exec_scope The exec scope of the operator.
   * \param launch_params The kernel launch parameters.
   * \param var_range_map: A map from loop variables to their ranges.
   * \param alloc_only Whether the dispatch context is only used for buffer allocation.
   * \param callbacks The callbacks to be handled when the operator is scheduled.
   * \param shared_state Shared state persisting across dispatch calls within a lowering pass.
   */
  TVM_DLL DispatchContext(Target target, ExecScope exec_scope,
                          ffi::Map<ffi::String, IterVar> launch_params = {},
                          ffi::Map<Var, Range> var_range_map = {}, bool alloc_only = false,
                          ffi::Map<ffi::String, ObjectRef> callbacks = {},
                          ffi::Map<ffi::String, ObjectRef> shared_state = {});

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(DispatchContext, ObjectRef, DispatchContextNode);
};

/*!
 * \brief The type of the function that dispatches a TIRX operator.
 * \param op The operator.
 * \param args The arguments.
 * \param context The dispatch context.
 */
using FOpDispatcher = ffi::TypedFunction<Stmt(tvm::Op, Array<ObjectRef>, DispatchContext)>;

/*!
 * \brief See pesudo code below:
 *
 * Tx.cast(BufferRegion dst, BufferRegion src)
 */
TVM_DLL const Op& cast();

/*!
 * \brief See pesudo code below:
 *
 * Tx.permute_dims(BufferRegion buffer, List order)
 */
TVM_DLL const Op& permute_dims();

/*!
 * \brief See pesudo code below:
 *
 * Tx.copy(BufferRegion dst, BufferRegion src)
 */
TVM_DLL const Op& copy();

/*!
 * \brief See pesudo code below:
 *
 * Tx.Async.copy(BufferRegion dst, BufferRegion src, BaseEvent event)
 */
TVM_DLL const Op& copy_async();

/*!
 * \brief See pesudo code below:
 *
 *  Tx.fill(BufferRegion dst, PrimExpr value)
 */
TVM_DLL const Op& fill();

/*!
 * \brief See pesudo code below:
 *
 * Tx.gemm(Buffer A, Buffer B, Buffer C, Buffer D, PrimExpr alpha, PrimExpr beta)
 */
TVM_DLL const Op& gemm();

/*!
 * \brief See pesudo code below:
 *
 * Tx.gemm_async(BufferRegion C, BufferRegion A, BufferRegion B, bool transA, bool transB,
 * bool accum)
 */
TVM_DLL const Op& gemm_async();

TVM_DLL const Op& zero();

TVM_DLL const Op& sqrt();

TVM_DLL const Op& exp();

TVM_DLL const Op& add();

TVM_DLL const Op& sub();

TVM_DLL const Op& mul();

TVM_DLL const Op& fdiv();

TVM_DLL const Op& minimum();

TVM_DLL const Op& maximum();

TVM_DLL const Op& reciprocal();

TVM_DLL const Op& sum();

TVM_DLL const Op& max();

TVM_DLL const Op& min();

TVM_DLL const Op& memset();

TVM_DLL const Op& reduce_negate();

TVM_DLL const Op& binary_reduce();

TVM_DLL const Op& unary_reduce();

TVM_DLL const Op& binary_chain();

TVM_DLL const Op& select();

TVM_DLL const Op& event_init();

TVM_DLL const Op& event_commit();

TVM_DLL const Op& event_wait();

/*!
 * \brief See pesudo code below:
 *
 *  tvm_kernel_replace_point()
 */
TVM_DLL const Op& tvm_kernel_replace_point();

<<<<<<<< HEAD:include/tvm/tirx/tirp_op.h
}  // namespace tirxxp
}  // namespace tirxx
}  // namespace tvm

#endif  // TVM_TIRX_TIRP_OP_H_
========
}  // namespace tirx
}  // namespace tir
}  // namespace tvm

#endif  // TVM_TIR_TIRX_OP_H_
>>>>>>>> bd927f10c7 (rename):include/tvm/tir/tirx_op.h
