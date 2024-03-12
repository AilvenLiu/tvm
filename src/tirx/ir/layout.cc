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

// DataIterTree
DataIterTree::DataIterTree(Var root, Array<IterTreeSplit> splits, Array<PrimExpr> coeff) {
  auto n = make_object<DataIterTreeNode>();
  n->root = std::move(root);
  n->splits = std::move(splits);
  n->coeff = std::move(coeff);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(DataIterTreeNode);

TVM_REGISTER_GLOBAL("tirx.DataIterTree")
    .set_body_typed([](Var root, Array<IterTreeSplit> splits, Array<PrimExpr> coeff) {
      return DataIterTree(root, splits, coeff);
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

// DeviceIterTree
DeviceIterTree::DeviceIterTree(Var root, Array<IterTreeSplit> splits, Array<ScopeIdAttr> attrs) {
  auto n = make_object<DeviceIterTreeNode>();
  n->root = std::move(root);
  n->splits = std::move(splits);
  n->attrs = std::move(attrs);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(DeviceIterTreeNode);

TVM_REGISTER_GLOBAL("tirx.DeviceIterTree")
    .set_body_typed([](Var root, Array<IterTreeSplit> splits, Array<ScopeIdAttr> attrs) {
      return DeviceIterTree(root, splits, attrs);
    });

// TileLayout
TileLayout::TileLayout(Array<DataIterTree> data_trees, Array<DeviceIterTree> device_trees) {
  auto n = make_object<TileLayoutNode>();
  n->data_trees = std::move(data_trees);
  n->device_trees = std::move(device_trees);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(TileLayoutNode);

TVM_REGISTER_GLOBAL("tirx.TileLayout")
    .set_body_typed([](Array<DataIterTree> data_trees, Array<DeviceIterTree> device_trees) {
      return TileLayout(data_trees, device_trees);
    });

}  // namespace tirxx
}  // namespace tvm
