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
#include <tvm/tirx/layout.h>

namespace tvm {
namespace tirx {

/******** Constructors ********/

// IterTreeSplit
IterTreeSplit::IterTreeSplit(Var parent, Array<Var> children, Array<PrimExpr> extents) {
  auto n = make_object<IterTreeSplitNode>();
  n->parent = std::move(parent);
  n->children = std::move(children);
  n->extents = std::move(extents);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(IterTreeSplitNode);

TVM_REGISTER_GLOBAL("tirx.IterTreeSplit")
    .set_body_typed([](Var parent, Array<Var> children, Array<PrimExpr> extents) {
      return IterTreeSplit(parent, children, extents);
    });

// IterTree
IterTree::IterTree(Var root, Array<IterTreeSplit> splits) {
  auto n = make_object<IterTreeNode>();
  n->root = std::move(root);
  n->splits = std::move(splits);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(IterTreeNode);

TVM_REGISTER_GLOBAL("tirx.IterTree").set_body_typed([](Var root, Array<IterTreeSplit> splits) {
  return IterTree(root, splits);
});

// CoordIterTree
CoordIterTree::CoordIterTree(Var root, Array<IterTreeSplit> splits, Array<PrimExpr> coeff) {
  auto n = make_object<CoordIterTreeNode>();
  n->root = std::move(root);
  n->splits = std::move(splits);
  n->coeff = std::move(coeff);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(CoordIterTreeNode);

TVM_REGISTER_GLOBAL("tirx.CoordIterTree")
    .set_body_typed([](Var root, Array<IterTreeSplit> splits, Array<PrimExpr> coeff) {
      return CoordIterTree(root, splits, coeff);
    });

// ScopeIdAttr
ScopeIdAttr::ScopeIdAttr(ScopeIdType type, Optional<Var> bound, Optional<PrimExpr> owner) {
  auto n = make_object<ScopeIdAttrNode>();
  n->type = type;
  n->bound = bound;
  n->owner = owner;
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(ScopeIdAttrNode);

TVM_REGISTER_GLOBAL("tirx.ScopeIdAttr")
    .set_body_typed([](int type, Optional<Var> bound, Optional<PrimExpr> owner) {
      return ScopeIdAttr(static_cast<ScopeIdType>(type), bound, owner);
    });

// ScopeIdIterTree
ScopeIdIterTree::ScopeIdIterTree(Var root, Array<IterTreeSplit> splits, Array<ScopeIdAttr> attrs) {
  auto n = make_object<ScopeIdIterTreeNode>();
  n->root = std::move(root);
  n->splits = std::move(splits);
  n->attrs = std::move(attrs);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(ScopeIdIterTreeNode);

TVM_REGISTER_GLOBAL("tirx.ScopeIdIterTree")
    .set_body_typed([](Var root, Array<IterTreeSplit> splits, Array<ScopeIdAttr> attrs) {
      return ScopeIdIterTree(root, splits, attrs);
    });

// TileLayout
TileLayout::TileLayout(Array<CoordIterTree> coord_iter_trees,
                       Array<ScopeIdIterTree> scope_id_iter_trees) {
  auto n = make_object<TileLayoutNode>();
  n->coord_iter_trees = std::move(coord_iter_trees);
  n->scope_id_iter_trees = std::move(scope_id_iter_trees);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(TileLayoutNode);

TVM_REGISTER_GLOBAL("tirx.TileLayout")
    .set_body_typed([](Array<CoordIterTree> coord_iter_trees,
                       Array<ScopeIdIterTree> scope_id_iter_trees) {
      return TileLayout(coord_iter_trees, scope_id_iter_trees);
    });

}  // namespace tirx
}  // namespace tvm
