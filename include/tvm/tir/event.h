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
};

class BaseEventNode : public Object {
 public:
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

class SemaphoreEventNode : public BaseEventNode {
 public:
  PrimExpr expected_count;
  String name;
  kEventImpl impl;
  Array<ffi::Any> state;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<SemaphoreEventNode>()
        .def_ro("expected_count", &SemaphoreEventNode::expected_count)
        .def_ro("name", &SemaphoreEventNode::name)
        .def_ro("impl", &SemaphoreEventNode::impl)
        .def_ro("state", &SemaphoreEventNode::state);
  }

  bool SEqualReduce(const SemaphoreEventNode* other, SEqualReducer equal) const {
    if (!equal(expected_count, other->expected_count)) return false;
    if (!equal(name, other->name)) return false;
    if (!equal(impl, other->impl)) return false;
    if (!equal(state, other->state)) return false;
    return equal.FreeVarEqualImpl(this, other);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(expected_count);
    hash_reduce(name);
    hash_reduce(impl);
    hash_reduce(state);
    hash_reduce.FreeVarHashImpl(this);
  }

  static constexpr const char* _type_key = "tirp.SemaphoreEvent";
  static constexpr bool _type_has_method_sequal_reduce = true;
  static constexpr bool _type_has_method_shash_reduce = true;
  static constexpr bool _type_has_method_visit_attrs = false;
  TVM_DECLARE_FINAL_OBJECT_INFO(SemaphoreEventNode, BaseEventNode);
};

class SemaphoreEvent : public BaseEvent {
 public:
  TVM_DLL SemaphoreEvent(const PrimExpr& expected_count, kEventImpl impl,
                         const Array<ffi::Any>& state, const String& name);
  TVM_DEFINE_OBJECT_REF_METHODS(SemaphoreEvent, BaseEvent, SemaphoreEventNode);
};

class BulkGroupEventNode : public BaseEventNode {
 public:
  String name;
  kEventImpl impl;
  Array<ffi::Any> state;

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

class EventTensorNode : public Object {
 public:
  SemaphoreEvent event;
  Array<PrimExpr> shape;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<EventTensorNode>()
        .def_ro("event", &EventTensorNode::event)
        .def_ro("shape", &EventTensorNode::shape);
  }

  bool SEqualReduce(const EventTensorNode* other, SEqualReducer equal) const {
    if (!equal(event, other->event)) return false;
    if (!equal(shape, other->shape)) return false;
    return equal.FreeVarEqualImpl(this, other);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(event);
    hash_reduce(shape);
    hash_reduce.FreeVarHashImpl(this);
  }

  String name() const;

  static constexpr const char* _type_key = "tirp.EventTensor";
  static constexpr bool _type_has_method_sequal_reduce = true;
  static constexpr bool _type_has_method_shash_reduce = true;
  static constexpr bool _type_has_method_visit_attrs = false;
  TVM_DECLARE_FINAL_OBJECT_INFO(EventTensorNode, Object);
};

class EventTensor : public ObjectRef {
 public:
  TVM_DLL EventTensor(const SemaphoreEvent& event, const Array<PrimExpr>& shape);
  TVM_DEFINE_OBJECT_REF_METHODS(EventTensor, ObjectRef, EventTensorNode);
};

class EventTensorItemNode : public BaseEventNode {
 public:
  EventTensor tensor;
  Array<PrimExpr> indices;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<EventTensorItemNode>()
        .def_ro("tensor", &EventTensorItemNode::tensor)
        .def_ro("indices", &EventTensorItemNode::indices);
  }

  bool SEqualReduce(const EventTensorItemNode* other, SEqualReducer equal) const {
    return equal(tensor, other->tensor) && equal(indices, other->indices);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(tensor);
    hash_reduce(indices);
    hash_reduce.FreeVarHashImpl(this);
  }

  static constexpr const char* _type_key = "tirp.EventTensorItem";
  static constexpr bool _type_has_method_sequal_reduce = true;
  static constexpr bool _type_has_method_shash_reduce = true;
  static constexpr bool _type_has_method_visit_attrs = false;
  TVM_DECLARE_FINAL_OBJECT_INFO(EventTensorItemNode, BaseEventNode);
};

class EventTensorItem : public BaseEvent {
 public:
  TVM_DLL EventTensorItem(const EventTensor& tensor, const Array<PrimExpr>& indices);

  TVM_DEFINE_OBJECT_REF_METHODS(EventTensorItem, BaseEvent, EventTensorItemNode);
};

}  // namespace tir
}  // namespace tvm

#endif  // TVM_TIR_EVENT_H_
