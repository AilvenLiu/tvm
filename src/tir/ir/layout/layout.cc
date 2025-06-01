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
Map<String, PrimExpr> TLayoutNode::Apply(const Array<PrimExpr>& coord,
                                         const Array<PrimExpr>& shape) const {
  ICHECK_EQ(coord.size(), shape.size())
      << "ValueError: The size of coord and shape should be equal";
  return Apply(FlattenCoord(coord, shape));
}

TVM_REGISTER_GLOBAL("tir.TLayoutCompatibleWithShape")
    .set_body_typed([](TLayout layout, Array<PrimExpr> shape) {
      return layout->CompatibleWithShape(shape);
    });

TVM_REGISTER_GLOBAL("tir.TLayoutVerifyWellFormed").set_body_typed([](TLayout layout) {
  return layout->VerifyWellFormed();
});

TVM_REGISTER_GLOBAL("tir.TLayoutGetSize")
    .set_body_typed([](TLayout layout, Optional<String> axis_name) {
      return layout->GetSize(axis_name);
    });

TVM_REGISTER_GLOBAL("tir.TLayoutGetCosize")
    .set_body_typed([](TLayout layout, Optional<String> axis_name) {
      return layout->GetCosize(axis_name);
    });

TVM_REGISTER_GLOBAL("tir.TLayoutApplyWithShape")
    .set_body_typed([](TLayout layout, Array<PrimExpr> coord, Array<PrimExpr> shape) {
      return layout->Apply(coord, shape);
    });

TVM_REGISTER_GLOBAL("tir.TLayoutApply").set_body_typed([](TLayout layout, Array<PrimExpr> coord) {
  return layout->Apply(coord);
});

TVM_REGISTER_GLOBAL("tir.TLayoutApplyLinear").set_body_typed([](TLayout layout, PrimExpr coord) {
  return layout->Apply(coord);
});

TVM_REGISTER_GLOBAL("tir.TLayoutNormalize").set_body_typed([](TLayout layout) {
  return layout->Normalize();
});

TVM_REGISTER_GLOBAL("tir.TLayoutTile")
    .set_body_typed([](TLayout layout, TileLayout outer, Array<PrimExpr> outer_shape,
                       Array<PrimExpr> inner_shape) {
      return layout->Tile(outer, outer_shape, inner_shape);
    });

TVM_REGISTER_GLOBAL("tir.TLayoutIsTileInner")
    .set_body_typed([](TLayout layout, TLayout tile_layout, Array<PrimExpr> tiled_shape,
                       Array<PrimExpr> inner_shape) {
      return layout->IsTileInner(tile_layout, tiled_shape, inner_shape);
    });

TVM_REGISTER_GLOBAL("tir.TLayoutIsTileOuter")
    .set_body_typed([](TLayout layout, TLayout tile_layout, Array<PrimExpr> tiled_shape,
                       Array<PrimExpr> outer_shape) {
      return layout->IsTileOuter(tile_layout, tiled_shape, outer_shape);
    });

}  // namespace tir
}  // namespace tvm
