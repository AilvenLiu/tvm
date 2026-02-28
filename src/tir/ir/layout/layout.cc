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

/**************** TLayout ****************/
ffi::Map<ffi::String, PrimExpr> TLayoutNode::Apply(const ffi::Array<PrimExpr>& coord,
                                                   const ffi::Array<PrimExpr>& shape) const {
  TVM_FFI_ICHECK_EQ(coord.size(), shape.size())
      << "ValueError: The size of coord and shape should be equal";
  return Apply(FlattenCoord(coord, shape));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  auto def = refl::GlobalDef();
  def.def("tir.TLayoutCompatibleWithShape",
          [](TLayout layout, Array<PrimExpr> shape) { return layout->CompatibleWithShape(shape); });
  def.def("tir.TLayoutVerifyWellFormed", [](TLayout layout) { return layout->VerifyWellFormed(); });
  def.def("tir.TLayoutGetSize", [](TLayout layout, ffi::Optional<ffi::String> axis_name) {
    return layout->GetSize(axis_name);
  });
  def.def("tir.TLayoutGetSpan", [](TLayout layout, ffi::Optional<ffi::String> axis_name) {
    return layout->GetSpan(axis_name);
  });
  def.def("tir.TLayoutApplyWithShape",
          [](TLayout layout, ffi::Array<PrimExpr> coord, ffi::Array<PrimExpr> shape) {
            return layout->Apply(coord, shape);
          });
  def.def("tir.TLayoutApply",
          [](TLayout layout, ffi::Array<PrimExpr> coord) { return layout->Apply(coord); });
  def.def("tir.TLayoutApplyLinear",
          [](TLayout layout, PrimExpr coord) { return layout->Apply(coord); });
  def.def("tir.TLayoutCanonicalize", [](TLayout layout) { return layout->Canonicalize(); });
  def.def("tir.TLayoutTile", [](TLayout layout, TileLayout outer, ffi::Array<PrimExpr> outer_shape,
                                ffi::Array<PrimExpr> inner_shape) {
    return layout->Tile(outer, outer_shape, inner_shape);
  });
  def.def("tir.TLayoutDirectSum",
          [](TLayout layout, TileLayout left, ffi::Array<PrimExpr> left_shape,
             ffi::Array<PrimExpr> right_shape) {
            return layout->DirectSum(left, left_shape, right_shape);
          });
  def.def("tir.TLayoutIsTileInner",
          [](TLayout layout, TLayout tile_layout, ffi::Array<PrimExpr> tiled_shape,
             ffi::Array<PrimExpr> inner_shape) {
            return layout->IsTileInner(tile_layout, tiled_shape, inner_shape);
          });
  def.def("tir.TLayoutIsTileOuter",
          [](TLayout layout, TLayout tile_layout, ffi::Array<PrimExpr> tiled_shape,
             ffi::Array<PrimExpr> outer_shape) {
            return layout->IsTileOuter(tile_layout, tiled_shape, outer_shape);
          });
  def.def("tir.TLayoutIsDirectSumRight",
          [](TLayout layout, TLayout sum_layout, ffi::Array<PrimExpr> interleaved_shape,
             ffi::Array<PrimExpr> right_shape) {
            return layout->IsDirectSumRight(sum_layout, interleaved_shape, right_shape);
          });
  def.def("tir.TLayoutIsDirectSumLeft",
          [](TLayout layout, TLayout sum_layout, ffi::Array<PrimExpr> interleaved_shape,
             ffi::Array<PrimExpr> left_shape) {
            return layout->IsDirectSumLeft(sum_layout, interleaved_shape, left_shape);
          });
  def.def("tir.TLayoutSlice",
          [](TLayout layout, ffi::Array<PrimExpr> shape, Region region) -> ffi::Optional<TLayout> {
            return layout->Slice(shape, region);
          });
}

}  // namespace tir
}  // namespace tvm
