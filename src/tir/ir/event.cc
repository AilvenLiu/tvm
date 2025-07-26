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

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/event.h>

namespace tvm {
namespace tir {

TVM_FFI_STATIC_INIT_BLOCK({
  SemaphoreEventNode::RegisterReflection();
  BulkGroupEventNode::RegisterReflection();
  EventTensorNode::RegisterReflection();
  EventTensorItemNode::RegisterReflection();
});

TVM_REGISTER_NODE_TYPE(BaseEventNode);

SemaphoreEvent::SemaphoreEvent(const PrimExpr& expected_count, kEventImpl impl,
                               const Array<ffi::Any>& state, const String& name) {
  ObjectPtr<SemaphoreEventNode> n = make_object<SemaphoreEventNode>();
  n->expected_count = std::move(expected_count);
  n->name = std::move(name);
  n->impl = impl;
  n->state = std::move(state);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(SemaphoreEventNode);

BulkGroupEvent::BulkGroupEvent(kEventImpl impl, const Array<ffi::Any>& state, const String& name) {
  ObjectPtr<BulkGroupEventNode> n = make_object<BulkGroupEventNode>();
  n->name = std::move(name);
  n->impl = impl;
  n->state = std::move(state);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(BulkGroupEventNode);

EventTensor::EventTensor(const SemaphoreEvent& event, const Array<PrimExpr>& shape) {
  ObjectPtr<EventTensorNode> n = make_object<EventTensorNode>();
  ICHECK(!event->IsInstance<EventTensorItemNode>());
  n->event = std::move(event);
  n->shape = std::move(shape);
  data_ = std::move(n);
}

String EventTensorNode::name() const {
  if (const auto* sem_event = event.as<SemaphoreEventNode>()) {
    return sem_event->name;
  } else if (const auto* bulk_event = event.as<BulkGroupEventNode>()) {
    return bulk_event->name;
  } else {
    LOG(FATAL) << "Unsupported event type: " << event->GetTypeKey();
  }
}

EventTensorItem::EventTensorItem(const EventTensor& tensor, const Array<PrimExpr>& indices) {
  ObjectPtr<EventTensorItemNode> n = make_object<EventTensorItemNode>();
  n->tensor = std::move(tensor);
  n->indices = std::move(indices);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(EventTensorNode);

TVM_REGISTER_NODE_TYPE(EventTensorItemNode);

TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("tirp.SemaphoreEvent",
           [](PrimExpr expected_count, kEventImpl impl, Array<ffi::Any> state, String name) {
             return SemaphoreEvent(expected_count, impl, state, name);
           })
      .def("tirp.BulkGroupEvent", [](kEventImpl impl, Array<ffi::Any> state,
                                     String name) { return BulkGroupEvent(impl, state, name); })
      .def("tirp.EventTensor",
           [](SemaphoreEvent event, Array<PrimExpr> shape) { return EventTensor(event, shape); })
      .def("tirp.EventTensorItem", [](EventTensor tensor, Array<PrimExpr> indices) {
        return EventTensorItem(tensor, indices);
      });
});

}  // namespace tir
}  // namespace tvm