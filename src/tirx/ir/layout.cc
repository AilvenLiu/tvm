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

/******** Utils ********/
PrimExpr ReduceMul(Array<PrimExpr> values) {
  PrimExpr result = values[0];
  for (size_t i = 1; i < values.size(); i++) {
    result = result * values[i];
  }
  return result;
}

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

std::tuple<DeviceIterTree, PrimExpr, std::vector<PrimExpr>> DeviceIterTreeFromTupleImpl(
    const ObjectRef& device) {
  Var root("");
  if (auto leaf_ptr = device.as<PrimExprNode>()) {
    auto leaf = GetRef<PrimExpr>(leaf_ptr);
    return {DeviceIterTree(root, {}, {DeviceIterAttr::Replicate()}), leaf, {leaf}};
  }
  auto tuple = device.as<ArrayNode>();
  ICHECK(tuple != nullptr) << "ValueError: Expect a tuple while get " << device->GetTypeKey();

  std::vector<IterTreeSplit> splits;
  std::vector<Var> children;
  std::vector<PrimExpr> extents;
  std::vector<DeviceIterAttr> attrs;
  std::vector<PrimExpr> leaf_extents;
  for (int i = tuple->size() - 1; i >= 0; i--) {
    auto d = tuple->at(i);
    auto [child, sub_extent, sub_leaf_extents] = DeviceIterTreeFromTupleImpl(d);
    splits.insert(splits.end(), child->splits.begin(), child->splits.end());
    children.push_back(child->root);
    extents.push_back(sub_extent);
    leaf_extents.insert(leaf_extents.end(), sub_leaf_extents.begin(), sub_leaf_extents.end());
    attrs.insert(attrs.end(), child->attrs.begin(), child->attrs.end());
  }
  splits.push_back(IterTreeSplit(root, children, extents));
  return {DeviceIterTree(root, splits, attrs), ReduceMul(extents), leaf_extents};
}

Array<ObjectRef> DeviceIterTree::FromTuple(const ObjectRef& device) {
  auto [tree, extent, leaf_extents] = DeviceIterTreeFromTupleImpl(device);
  return {tree, extent, Array<PrimExpr>(leaf_extents)};
}

TVM_REGISTER_GLOBAL("tirx.DeviceIterTreeFromTuple")
    .set_body_typed(DeviceIterTree::FromTuple);

// DeviceIterAttr
DeviceIterAttr::DeviceIterAttr(ScopeIdType type, Optional<Var> bound, Optional<PrimExpr> owner) {
  auto n = make_object<DeviceIterAttrNode>();
  n->type = type;
  n->bound = bound;
  n->owner = owner;
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(DeviceIterAttrNode);

TVM_REGISTER_GLOBAL("tirx.DeviceIterAttr")
    .set_body_typed([](int type, Optional<Var> bound, Optional<PrimExpr> owner) {
      return DeviceIterAttr(static_cast<ScopeIdType>(type), bound, owner);
    });

DeviceIterAttr DeviceIterAttr::Replicate() { return DeviceIterAttr(kReplicate, NullOpt, NullOpt); }

DeviceIterAttr DeviceIterAttr::Split(Var bound) { return DeviceIterAttr(kSplit, bound, NullOpt); }

// DeviceIterTree
DeviceIterTree::DeviceIterTree(Var root, Array<IterTreeSplit> splits, Array<DeviceIterAttr> attrs) {
  auto n = make_object<DeviceIterTreeNode>();
  n->root = std::move(root);
  n->splits = std::move(splits);
  n->attrs = std::move(attrs);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(DeviceIterTreeNode);

TVM_REGISTER_GLOBAL("tirx.DeviceIterTree")
    .set_body_typed([](Var root, Array<IterTreeSplit> splits, Array<DeviceIterAttr> attrs) {
      return DeviceIterTree(root, splits, attrs);
    });

// TileLayout
TileLayout::TileLayout(Array<DataIterTree> data_trees, Array<DeviceIterTree> device_trees,
                       Optional<ExecScope> from, Optional<ExecScope> to) {
  auto n = make_object<TileLayoutNode>();
  n->data_trees = std::move(data_trees);
  n->device_trees = std::move(device_trees);
  n->from = std::move(from);
  n->to = std::move(to);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(TileLayoutNode);

TVM_REGISTER_GLOBAL("tirx.TileLayout")
    .set_body_typed([](Array<DataIterTree> data_trees, Array<DeviceIterTree> device_trees,
                       Optional<ExecScope> from, Optional<ExecScope> to) {
      return TileLayout(data_trees, device_trees, from, to);
    });

TileLayout TileLayout::FromTile(const Array<PrimExpr>& shape, const TileLayout& inner,
                                const Optional<ObjectRef>& device,
                                const Optional<ObjectRef>& from_to) {
  // Create outer IterSplits
  
  if (device.defined()) {
    ICHECK(from_to.defined()) << "ValueError: from_to must be defined when device is defined";

  } else {
    ICHECK(!from_to.defined()) << "ValueError: from_to must not be defined when device is not defined";

  } 
}

TVM_REGISTER_GLOBAL("tirx.TileLayoutFromTile")
    .set_body_typed([](const Array<PrimExpr>& shape, const TileLayout& inner,
                       const Optional<ObjectRef>& device, const Optional<ObjectRef>& from_to) {
      return TileLayout::FromTile(shape, inner, device, from_to);
    });

}  // namespace tirx
}  // namespace tvm
