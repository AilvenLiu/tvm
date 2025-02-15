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

/**************** SwizzleLayout ****************/
SwizzleLayout::SwizzleLayout(int per_element, int swizzle_len, int atom_len, bool swizzle_inner) {
  auto n = make_object<SwizzleLayoutNode>();
  n->per_element = per_element;
  n->swizzle_len = swizzle_len;
  n->atom_len = atom_len;
  n->swizzle_inner = swizzle_inner;
  CHECK(n->VerifyWellFormed()) << "ValueError: The swizzle layout is not well-formed";
  int swizzle_mask = (1 << swizzle_len) - 1;
  n->inner_mask = swizzle_mask;
  n->outer_mask = swizzle_mask << atom_len;
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(SwizzleLayoutNode);

TVM_REGISTER_GLOBAL("tir.SwizzleLayout")
    .set_body_typed([](int per_element, int swizzle_len, int atom_len, bool swizzle_inner) {
      return SwizzleLayout(per_element, swizzle_len, atom_len, swizzle_inner);
    });

bool SwizzleLayoutNode::VerifyWellFormed() const {
  return per_element >= 0 && swizzle_len >= 0 && atom_len >= swizzle_len;
}

PrimExpr SwizzleLayoutNode::GetSize() const { return 1 << (per_element + swizzle_len + atom_len); }

PrimExpr SwizzleLayoutNode::GetCosize() const { return GetSize(); }

Array<PrimExpr> SwizzleLayoutNode::Apply(const PrimExpr& coord) const {
  PrimExpr input = coord;
  auto f = [&](const PrimExpr& x) -> PrimExpr {
    if (swizzle_inner) {
      return x ^ ((x & outer_mask) >> atom_len);
    } else {
      return x ^ ((x & inner_mask) << atom_len);
    }
  };
  auto base = 1 << per_element;
  arith::Analyzer analyzer;
  // It takes more arithmetic operations to compute the result, but it is more friendly to the
  // vectorization
  return {analyzer.Simplify((f(floordiv(input, base)) << per_element) + floormod(input, base))};
}

}  // namespace tir
}  // namespace tvm
