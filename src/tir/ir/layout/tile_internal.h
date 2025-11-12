/*
 * Internal helpers for TileLayout implementations.
 * This header is private to the layout implementation files.
 */

#ifndef TVM_TIR_IR_LAYOUT_TILE_INTERNAL_H_
#define TVM_TIR_IR_LAYOUT_TILE_INTERNAL_H_

#include "utils.h"

namespace tvm {
namespace tir {

// Group a tile layout's shard by a logical shape, returning the grouped layout and separators.
std::pair<TileLayout, std::vector<int64_t>> Group(TileLayout layout,
                                                  const ffi::Array<PrimExpr>& shape);

// Compute a tiled logical shape, either inner or outer tiling.
ffi::Array<PrimExpr> TileShape(ffi::Array<PrimExpr> shape, ffi::Array<PrimExpr> factor,
                               bool is_inner);

// Elementwise division of two shapes.
ffi::Array<PrimExpr> DivideShape(ffi::Array<PrimExpr> shape, ffi::Array<PrimExpr> factor);

// Extract the even indices from a vector of separators.
std::vector<int64_t> EvenSeparatorIndices(std::vector<int64_t> seps);

// Split axes according to a split scope on the target.
TileLayout SplitAxesByScope(TileLayout layout, const ffi::String& split_scope);

}  // namespace tir
}  // namespace tvm

#endif  // TVM_TIR_IR_LAYOUT_TILE_INTERNAL_H_
