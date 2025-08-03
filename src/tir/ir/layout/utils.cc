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

Array<PrimExpr> SplitCoord(PrimExpr coord, const Array<PrimExpr>& shape) {
  Array<PrimExpr> result;
  for (int i = shape.size() - 1; i >= 0; --i) {
    if (i == 0) {
      result.push_back(coord);
    } else {
      result.push_back(floormod(coord, shape[i]));
      coord = floordiv(coord, shape[i]);
    }
  }
  return Array<PrimExpr>(result.rbegin(), result.rend());
}

PrimExpr FlattenCoord(const Array<PrimExpr>& coord, const Array<PrimExpr>& shape) {
  return std::accumulate(
      coord.begin(), coord.end(), PrimExpr(0),
      [&shape, i = 0](PrimExpr acc, const PrimExpr& c) mutable { return acc * shape[i++] + c; });
}

}  // namespace tir
}  // namespace tvm
