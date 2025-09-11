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
  auto n = ffi::make_object<ComposeLayoutNode>();
  n->layout_A = layout_A;
  n->layout_B = layout_B;
  CHECK(n->VerifyWellFormed()) << "ValueError: The compose layout is not well-formed";

  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tir.ComposeLayout", [](SwizzleLayout layout_A, TileLayout layout_B) {
    return ComposeLayout(layout_A, layout_B);
  });
});

bool ComposeLayoutNode::CompatibleWithShape(const Array<PrimExpr>& shape) const { return true; }

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

PrimExpr ComposeLayoutNode::GetSize(ffi::Optional<ffi::String> axis_name) const {
  CHECK(!axis_name.has_value()) << "ValueError: axis_name is not supported for compose layout";
  return layout_B->GetSize(axis_name);
}

PrimExpr ComposeLayoutNode::GetCosize(ffi::Optional<ffi::String> axis_name) const {
  CHECK(!axis_name.has_value()) << "ValueError: axis_name is not supported for compose layout";
  return layout_B->GetCosize(axis_name);
}

ffi::Map<ffi::String, PrimExpr> ComposeLayoutNode::Apply(ffi::Array<PrimExpr> coord) const {
  LOG(FATAL) << "ComposeLayoutNode::Apply(Array<PrimExpr>) is not implemented";
  return {};
}

ffi::Map<ffi::String, PrimExpr> ComposeLayoutNode::Apply(PrimExpr coord) const {
  auto res = layout_B->Apply(coord);
  CHECK(res.size() == 1 && res.find("m") != res.end());
  auto m = res["m"];
  auto layout_A_res = layout_A->Apply(m);
  CHECK(layout_A_res.size() == 1 && layout_A_res.find("m") != layout_A_res.end());
  return layout_A_res;
}

TLayout ComposeLayoutNode::Normalize() const {
  auto layout_B_normalized = layout_B->Normalize().as<TileLayout>().value();
  if (layout_B_normalized->IsTrivial()) {
    return layout_A;
  }
  return ComposeLayout(layout_A, layout_B_normalized);
}

TLayout ComposeLayoutNode::Tile(const TileLayout& outer, const ffi::Array<PrimExpr>& outer_shape,
                                const ffi::Array<PrimExpr>& inner_shape) const {
  // layout_B is first tiled with `outer`, then compose with layout_A.
  auto tiled_B = layout_B->Tile(outer, outer_shape, inner_shape).as<TileLayout>().value();
  return ComposeLayout(layout_A, tiled_B);
}

ffi::Optional<TileLayout> ComposeLayoutNode::IsTileInner(
    const TLayout& tile_layout, const ffi::Array<PrimExpr>& tiled_shape,
    const ffi::Array<PrimExpr>& inner_shape) const {
  if (auto comp = tile_layout.as<ComposeLayout>()) {
    if (StructuralEqual()(comp.value()->layout_A, this->layout_A)) {
      return this->layout_B->IsTileInner(comp.value()->layout_B, tiled_shape, inner_shape);
    }
  }
  return std::nullopt;
}

ffi::Optional<TLayout> ComposeLayoutNode::IsTileOuter(
    const TLayout& tile_layout, const ffi::Array<PrimExpr>& tiled_shape,
    const ffi::Array<PrimExpr>& outer_shape) const {
  return std::nullopt;
}

}  // namespace tir
}  // namespace tvm
