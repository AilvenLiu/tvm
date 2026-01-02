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
 * \file lower_tirx.cc
 * \brief Compose the TIRx lowering pipeline from individual passes.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/transform.h>
#include <tvm/tirx/transform.h>

namespace tvm {
namespace tirx {
namespace transform {

Pass LowerTIRx() {
  std::vector<tvm::transform::Pass> passes = {LowerTIRxResolveScopeIds(), LowerTIRxScheduleOps()};
  if (const char* env = std::getenv("TVM_PRINT_AFTER_TIRX_SCHEDULE_OPS")) {
    passes.push_back(tvm::transform::PrintIR());
  }
<<<<<<<< HEAD:src/tirx/transform/lower_tirp.cc
  passes.push_back(LowerTIRpResolveScopeSlices());
  passes.push_back(LowerTIRpDedupCuTensorMaps());
  passes.push_back(LowerTIRpCleanup());
  return tvm::transform::Sequential(passes, "tirx.LowerTIRp");
========
  passes.push_back(LowerTIRxResolveScopeSlices());
  passes.push_back(LowerTIRxDedupCuTensorMaps());
  passes.push_back(LowerTIRxCleanup());
  return tvm::transform::Sequential(passes, "tirx.LowerTIRx");
>>>>>>>> bd927f10c7 (rename):src/tirx/transform/lower_tirx.cc
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
<<<<<<<< HEAD:src/tirx/transform/lower_tirp.cc
      .def("tirx.transform.LowerTIRpResolveScopeIds", LowerTIRpResolveScopeIds)
      .def("tirx.transform.LowerTIRpScheduleOps", LowerTIRpScheduleOps)
      .def("tirx.transform.LowerTIRpResolveScopeSlices", LowerTIRpResolveScopeSlices)
      .def("tirx.transform.LowerTIRpDedupCuTensorMaps", LowerTIRpDedupCuTensorMaps)
      .def("tirx.transform.LowerTIRpCleanup", LowerTIRpCleanup)
      .def("tirx.transform.LowerTIRp", LowerTIRp);
========
      .def("tirx.transform.LowerTIRxResolveScopeIds", LowerTIRxResolveScopeIds)
      .def("tirx.transform.LowerTIRxScheduleOps", LowerTIRxScheduleOps)
      .def("tirx.transform.LowerTIRxResolveScopeSlices", LowerTIRxResolveScopeSlices)
      .def("tirx.transform.LowerTIRxDedupCuTensorMaps", LowerTIRxDedupCuTensorMaps)
      .def("tirx.transform.LowerTIRxCleanup", LowerTIRxCleanup)
      .def("tirx.transform.LowerTIRx", LowerTIRx);
>>>>>>>> bd927f10c7 (rename):src/tirx/transform/lower_tirx.cc
}

}  // namespace transform
}  // namespace tirxxxxxxxxxxxxxxxxxx
}  // namespace tvm
