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
#include "utils.h"

namespace tvm {
namespace tir {

/**************** ComposeLayout ****************/
ComposeLayout::ComposeLayout(SwizzleLayout layout_A, TileLayout layout_B) {
  auto n = make_object<ComposeLayoutNode>();
  n->layout_A = layout_A;
  n->layout_B = layout_B;
  CHECK(n->VerifyWellFormed()) << "ValueError: The compose layout is not well-formed";

  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(ComposeLayoutNode);

TVM_REGISTER_GLOBAL("tir.ComposeLayout")
    .set_body_typed([](SwizzleLayout layout_A, TileLayout layout_B) {
      return ComposeLayout(layout_A, layout_B);
    });

bool ComposeLayoutNode::VerifyWellFormed() const {
  if (!layout_A->VerifyWellFormed() || !layout_B->VerifyWellFormed()) {
    return false;
  }
  arith::Analyzer analyzer;
  if (!analyzer.CanProve(FloorMod(layout_B->GetCosize(), layout_A->GetSize()) == 0)) {
    return false;
  }
  return true;
}

PrimExpr ComposeLayoutNode::GetSize() const { return layout_B->GetSize(); }

PrimExpr ComposeLayoutNode::GetCosize() const { return layout_B->GetCosize(); }

Array<PrimExpr> ComposeLayoutNode::Apply(const PrimExpr& coord) const {
  return layout_A->Apply(layout_B->Apply(coord)[0]);
}

}  // namespace tir
}  // namespace tvm
