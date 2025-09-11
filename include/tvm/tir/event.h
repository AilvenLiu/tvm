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
 * \file tvm/tir/event.h
 * \brief TIR+ lib-levelevent abstraction.
 */
#ifndef TVM_TIR_EVENT_H_
#define TVM_TIR_EVENT_H_

#include <tvm/tir/expr.h>

namespace tvm {
namespace tir {

enum class kEventImpl : int {
  kMbarrier = 0,
  kCpAsync = 1,
  kTMALoadOnly = 2,
  kTMAStore2 = 3,
  kGlobalSemaphore = 4,
};

class BaseEventNode : public Object {
 public:
  virtual kEventImpl GetImpl() const = 0;
  virtual ffi::Array<ffi::Any> GetState() const = 0;

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
  TVM_FFI_DECLARE_OBJECT_INFO("tirp.BaseEvent", BaseEventNode, Object);
};

class BaseEvent : public ObjectRef {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(BaseEvent, ObjectRef, BaseEventNode);
};

class BulkGroupEventNode : public BaseEventNode {
 public:
  ffi::String name;
  kEventImpl impl;
  ffi::Array<ffi::Any> state;

  kEventImpl GetImpl() const final;
  ffi::Array<ffi::Any> GetState() const final;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<BulkGroupEventNode>()
        .def_ro("name", &BulkGroupEventNode::name)
        .def_ro("impl", &BulkGroupEventNode::impl)
        .def_ro("state", &BulkGroupEventNode::state);
  }

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirp.BulkGroupEvent", BulkGroupEventNode, BaseEventNode);
};

class BulkGroupEvent : public BaseEvent {
 public:
  TVM_DLL BulkGroupEvent(kEventImpl impl, const ffi::Array<ffi::Any>& state,
                         const ffi::String& name);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(BulkGroupEvent, BaseEvent, BulkGroupEventNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(BulkGroupEventNode);
};

class SemaphoreEventTensorNode : public Object {
 public:
  ffi::String name;
  kEventImpl impl;
  ffi::Array<ffi::Any> state;
  ffi::Array<PrimExpr> shape;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<SemaphoreEventTensorNode>()
        .def_ro("name", &SemaphoreEventTensorNode::name)
        .def_ro("impl", &SemaphoreEventTensorNode::impl)
        .def_ro("state", &SemaphoreEventTensorNode::state)
        .def_ro("shape", &SemaphoreEventTensorNode::shape);
  }

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirp.SemaphoreEventTensor", SemaphoreEventTensorNode, Object);
};

class SemaphoreEventTensor : public ObjectRef {
 public:
  TVM_DLL SemaphoreEventTensor(const kEventImpl& impl, const ffi::Array<ffi::Any>& state,
                               const ffi::Array<PrimExpr>& shape, const ffi::String& name);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(SemaphoreEventTensor, ObjectRef,
                                             SemaphoreEventTensorNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(SemaphoreEventTensorNode);
};

class SemaphoreEventTensorItemNode : public BaseEventNode {
 public:
  SemaphoreEventTensor tensor;
  ffi::Array<PrimExpr> indices;

  kEventImpl GetImpl() const final;
  ffi::Array<ffi::Any> GetState() const final;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<SemaphoreEventTensorItemNode>()
        .def_ro("tensor", &SemaphoreEventTensorItemNode::tensor)
        .def_ro("indices", &SemaphoreEventTensorItemNode::indices);
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirp.SemaphoreEventTensorItem", SemaphoreEventTensorItemNode,
                                    BaseEventNode);
};

class SemaphoreEventTensorItem : public BaseEvent {
 public:
  TVM_DLL SemaphoreEventTensorItem(const SemaphoreEventTensor& tensor,
                                   const ffi::Array<PrimExpr>& indices);

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(SemaphoreEventTensorItem, BaseEvent,
                                             SemaphoreEventTensorItemNode);
};

}  // namespace tir
}  // namespace tvm

#endif  // TVM_TIR_EVENT_H_
