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
  BulkGroupEventNode::RegisterReflection();
  SemaphoreEventTensorNode::RegisterReflection();
  SemaphoreEventTensorItemNode::RegisterReflection();
});

kEventImpl BulkGroupEventNode::GetImpl() const { return impl; }
Array<ffi::Any> BulkGroupEventNode::GetState() const { return state; }

BulkGroupEvent::BulkGroupEvent(kEventImpl impl, const Array<ffi::Any>& state, const String& name) {
  ObjectPtr<BulkGroupEventNode> n = make_object<BulkGroupEventNode>();
  n->name = std::move(name);
  n->impl = impl;
  n->state = std::move(state);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(BulkGroupEventNode);

SemaphoreEventTensor::SemaphoreEventTensor(const kEventImpl& impl, const Array<ffi::Any>& state,
                                           const Array<PrimExpr>& shape, const String& name) {
  ObjectPtr<SemaphoreEventTensorNode> n = make_object<SemaphoreEventTensorNode>();
  n->name = std::move(name);
  n->impl = impl;
  n->state = std::move(state);
  n->shape = std::move(shape);
  data_ = std::move(n);
}

kEventImpl SemaphoreEventTensorItemNode::GetImpl() const { return tensor->impl; }
Array<ffi::Any> SemaphoreEventTensorItemNode::GetState() const { return tensor->state; }

SemaphoreEventTensorItem::SemaphoreEventTensorItem(const SemaphoreEventTensor& tensor,
                                                   const Array<PrimExpr>& indices) {
  ObjectPtr<SemaphoreEventTensorItemNode> n = make_object<SemaphoreEventTensorItemNode>();
  n->tensor = std::move(tensor);
  n->indices = std::move(indices);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(SemaphoreEventTensorNode);
TVM_REGISTER_NODE_TYPE(SemaphoreEventTensorItemNode);

TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("tirp.BulkGroupEvent", [](kEventImpl impl, Array<ffi::Any> state,
                                     String name) { return BulkGroupEvent(impl, state, name); })
      .def("tirp.SemaphoreEventTensor",
           [](kEventImpl impl, Array<ffi::Any> state, Array<PrimExpr> shape, String name) {
             return SemaphoreEventTensor(impl, state, shape, name);
           })
      .def("tirp.SemaphoreEventTensorItem",
           [](SemaphoreEventTensor tensor, Array<PrimExpr> indices) {
             return SemaphoreEventTensorItem(tensor, indices);
           })
      .def("tirp.BaseEventImplGet", [](BaseEvent event) { return event->GetImpl(); })
      .def("tirp.BaseEventStateGet", [](BaseEvent event) { return event->GetState(); });
});

}  // namespace tir
}  // namespace tvm