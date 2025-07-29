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
  virtual Array<ffi::Any> GetState() const = 0;

  static constexpr const char* _type_key = "tirp.BaseEvent";
  static constexpr bool _type_has_method_sequal_reduce = false;
  static constexpr bool _type_has_method_shash_reduce = false;
  static constexpr bool _type_has_method_visit_attrs = false;
  TVM_DECLARE_BASE_OBJECT_INFO(BaseEventNode, Object);
};

class BaseEvent : public ObjectRef {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(BaseEvent, ObjectRef, BaseEventNode);
};

class BulkGroupEventNode : public BaseEventNode {
 public:
  String name;
  kEventImpl impl;
  Array<ffi::Any> state;

  kEventImpl GetImpl() const final;
  Array<ffi::Any> GetState() const final;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<BulkGroupEventNode>()
        .def_ro("name", &BulkGroupEventNode::name)
        .def_ro("impl", &BulkGroupEventNode::impl)
        .def_ro("state", &BulkGroupEventNode::state);
  }

  bool SEqualReduce(const BulkGroupEventNode* other, SEqualReducer equal) const {
    if (!equal(name, other->name)) return false;
    if (!equal(impl, other->impl)) return false;
    if (!equal(state, other->state)) return false;
    return equal.FreeVarEqualImpl(this, other);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(name);
    hash_reduce(impl);
    hash_reduce(state);
    hash_reduce.FreeVarHashImpl(this);
  }

  static constexpr const char* _type_key = "tirp.BulkGroupEvent";
  static constexpr bool _type_has_method_sequal_reduce = true;
  static constexpr bool _type_has_method_shash_reduce = true;
  static constexpr bool _type_has_method_visit_attrs = false;
  TVM_DECLARE_FINAL_OBJECT_INFO(BulkGroupEventNode, BaseEventNode);
};

class BulkGroupEvent : public BaseEvent {
 public:
  TVM_DLL BulkGroupEvent(kEventImpl impl, const Array<ffi::Any>& state, const String& name);
  TVM_DEFINE_OBJECT_REF_METHODS(BulkGroupEvent, BaseEvent, BulkGroupEventNode);
};

class SemaphoreEventTensorNode : public Object {
 public:
  String name;
  kEventImpl impl;
  Array<ffi::Any> state;
  Array<PrimExpr> shape;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<SemaphoreEventTensorNode>()
        .def_ro("name", &SemaphoreEventTensorNode::name)
        .def_ro("impl", &SemaphoreEventTensorNode::impl)
        .def_ro("state", &SemaphoreEventTensorNode::state)
        .def_ro("shape", &SemaphoreEventTensorNode::shape);
  }

  bool SEqualReduce(const SemaphoreEventTensorNode* other, SEqualReducer equal) const {
    if (!equal(name, other->name)) return false;
    if (!equal(impl, other->impl)) return false;
    if (!equal(state, other->state)) return false;
    if (!equal(shape, other->shape)) return false;
    return equal.FreeVarEqualImpl(this, other);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(name);
    hash_reduce(impl);
    hash_reduce(state);
    hash_reduce(shape);
    hash_reduce.FreeVarHashImpl(this);
  }

  static constexpr const char* _type_key = "tirp.SemaphoreEventTensor";
  static constexpr bool _type_has_method_sequal_reduce = true;
  static constexpr bool _type_has_method_shash_reduce = true;
  static constexpr bool _type_has_method_visit_attrs = false;
  TVM_DECLARE_FINAL_OBJECT_INFO(SemaphoreEventTensorNode, Object);
};

class SemaphoreEventTensor : public ObjectRef {
 public:
  TVM_DLL SemaphoreEventTensor(const kEventImpl& impl, const Array<ffi::Any>& state,
                               const Array<PrimExpr>& shape, const String& name);
  TVM_DEFINE_OBJECT_REF_METHODS(SemaphoreEventTensor, ObjectRef, SemaphoreEventTensorNode);
};

class SemaphoreEventTensorItemNode : public BaseEventNode {
 public:
  SemaphoreEventTensor tensor;
  Array<PrimExpr> indices;

  kEventImpl GetImpl() const final;
  Array<ffi::Any> GetState() const final;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<SemaphoreEventTensorItemNode>()
        .def_ro("tensor", &SemaphoreEventTensorItemNode::tensor)
        .def_ro("indices", &SemaphoreEventTensorItemNode::indices);
  }

  bool SEqualReduce(const SemaphoreEventTensorItemNode* other, SEqualReducer equal) const {
    return equal(tensor, other->tensor) && equal(indices, other->indices);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(tensor);
    hash_reduce(indices);
    hash_reduce.FreeVarHashImpl(this);
  }

  static constexpr const char* _type_key = "tirp.SemaphoreEventTensorItem";
  static constexpr bool _type_has_method_sequal_reduce = true;
  static constexpr bool _type_has_method_shash_reduce = true;
  static constexpr bool _type_has_method_visit_attrs = false;
  TVM_DECLARE_FINAL_OBJECT_INFO(SemaphoreEventTensorItemNode, BaseEventNode);
};

class SemaphoreEventTensorItem : public BaseEvent {
 public:
  TVM_DLL SemaphoreEventTensorItem(const SemaphoreEventTensor& tensor,
                                   const Array<PrimExpr>& indices);

  TVM_DEFINE_OBJECT_REF_METHODS(SemaphoreEventTensorItem, BaseEvent, SemaphoreEventTensorItemNode);
};

}  // namespace tir
}  // namespace tvm

#endif  // TVM_TIR_EVENT_H_
