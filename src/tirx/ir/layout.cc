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
IterTreeSplit::IterTreeSplit(PrimExpr extent, Array<IterTreeBase> children) {
  auto n = make_object<IterTreeSplitNode>();
  n->extent = std::move(extent);
  n->children = std::move(children);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(IterTreeSplitNode);

TVM_REGISTER_GLOBAL("tirx.IterTreeSplit")
    .set_body_typed([](PrimExpr extent, Array<IterTreeBase> children) {
      return IterTreeSplit(extent, children);
    });

// IterTree
IterTree::IterTree(IterTreeSplit root) {
  auto n = make_object<IterTreeNode>();
  n->root = std::move(root);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(IterTreeNode);

TVM_REGISTER_GLOBAL("tirx.IterTree").set_body_typed([](IterTreeSplit root) {
  return IterTree(root);
});

std::pair<IterTree, std::vector<IterTreeBase>> IterTreeFromTupleImpl(const ObjectRef& device) {
  if (auto leaf_ptr = device.as<PrimExprNode>()) {
    auto leaf = GetRef<PrimExpr>(leaf_ptr);
    auto root = IterTreeSplit(leaf, {});
    return {IterTree({root}), {root}};
  }
  auto tuple = device.as<ArrayNode>();
  ICHECK(tuple != nullptr) << "ValueError: Expect a tuple while get " << device->GetTypeKey();

  std::vector<IterTreeBase> children;
  std::vector<IterTreeBase> leaves;
  std::vector<PrimExpr> extents;
  for (auto d : *tuple) {
    auto [child, child_leaves] = IterTreeFromTupleImpl(d);
    children.push_back(child->root);
    leaves.insert(leaves.end(), child_leaves.begin(), child_leaves.end());
    extents.push_back(child->root->extent);
  }
  auto root = IterTreeSplit(ReduceMul(extents), Array<IterTreeBase>(children));
  return {IterTree(root), leaves};
}

Array<ObjectRef> IterTree::FromTuple(const ObjectRef& device) {
  auto [tree, leaves] = IterTreeFromTupleImpl(device);
  return {tree, Array<IterTreeBase>(leaves)};
}

TVM_REGISTER_GLOBAL("tirx.IterTreeFromTuple").set_body_typed(IterTree::FromTuple);

Array<IterTreeBase> IterTree::GetLeaves() const {
  auto* n = operator->();
  std::vector<IterTreeBase> leaves;
  std::function<void(const IterTreeBase&)> visit = [&](const IterTreeBase& node) {
    if (auto* split = node.as<IterTreeSplitNode>()) {
      if (split->children.size() == 0) {
        leaves.push_back(node);
      }
      for (const auto& child : split->children) {
        visit(child);
      }
    } else {
      LOG(FATAL) << "InternalError: Expect IterTreeSplit but get " << node->GetTypeKey();
    }
  };
  visit(n->root);
  return Array<IterTreeBase>(leaves);
}

IterTree::LeafIndexMap IterTree::GetLeafToIndex() const {
  Array<IterTreeBase> leaves = GetLeaves();
  LeafIndexMap leaf_to_index;
  for (size_t i = 0; i < leaves.size(); i++) {
    leaf_to_index[leaves[i]] = i;
  }
  return std::move(leaf_to_index);
}

// DataIterTree
DataIterTree::DataIterTree(IterTreeSplit root, Array<PrimExpr> coeff) {
  auto n = make_object<DataIterTreeNode>();
  n->root = std::move(root);
  n->coeff = std::move(coeff);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(DataIterTreeNode);

TVM_REGISTER_GLOBAL("tirx.DataIterTree")
    .set_body_typed([](IterTreeSplit root, Array<PrimExpr> coeff) {
      return DataIterTree(root, coeff);
    });

// DeviceIterAttr
DeviceIterAttr::DeviceIterAttr(ScopeIdType type, Optional<PrimExpr> bound,
                               Optional<PrimExpr> owner) {
  auto n = make_object<DeviceIterAttrNode>();
  n->type = type;
  n->bound = bound;
  n->owner = owner;
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(DeviceIterAttrNode);

TVM_REGISTER_GLOBAL("tirx.DeviceIterAttr")
    .set_body_typed([](int type, Optional<PrimExpr> bound, Optional<PrimExpr> owner) {
      return DeviceIterAttr(static_cast<ScopeIdType>(type), bound, owner);
    });

DeviceIterAttr DeviceIterAttr::Replicate() { return DeviceIterAttr(kReplicate, NullOpt, NullOpt); }

DeviceIterAttr DeviceIterAttr::Split(PrimExpr bound) { return DeviceIterAttr(kSplit, bound); }

DeviceIterAttr DeviceIterAttr::Exclusive(PrimExpr owner) {
  return DeviceIterAttr(kExclusive, NullOpt, owner);
}

// DeviceIterTree
DeviceIterTree::DeviceIterTree(IterTreeSplit root, Array<DeviceIterAttr> attrs) {
  auto n = make_object<DeviceIterTreeNode>();
  n->root = std::move(root);
  n->attrs = std::move(attrs);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(DeviceIterTreeNode);

TVM_REGISTER_GLOBAL("tirx.DeviceIterTree")
    .set_body_typed([](IterTreeSplit root, Array<DeviceIterAttr> attrs) {
      return DeviceIterTree(root, attrs);
    });

// TileLayout
TileLayout::TileLayout(DataIterTree data_tree, DeviceIterTree device_tree, Optional<ExecScope> from,
                       Optional<ExecScope> to) {
  auto n = make_object<TileLayoutNode>();
  n->data_tree = std::move(data_tree);
  n->device_tree = std::move(device_tree);
  n->from = std::move(from);
  n->to = std::move(to);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(TileLayoutNode);

TVM_REGISTER_GLOBAL("tirx.TileLayout")
    .set_body_typed([](DataIterTree data_tree, DeviceIterTree device_tree, Optional<ExecScope> from,
                       Optional<ExecScope> to) {
      return TileLayout(data_tree, device_tree, from, to);
    });

TileLayout TileLayout::FromTile(const Array<PrimExpr>& shape, const TileLayout& inner,
                                const Optional<ObjectRef>& device,
                                const Optional<ObjectRef>& from_to) {
  // Create outer IterSplits
  if (device.defined()) {
    ICHECK(from_to.defined()) << "ValueError: from_to must be defined when device is defined";

  } else {
    ICHECK(!from_to.defined())
        << "ValueError: from_to must not be defined when device is not defined";
  }
}

TVM_REGISTER_GLOBAL("tirx.TileLayoutFromTile")
    .set_body_typed([](const Array<PrimExpr>& shape, const TileLayout& inner,
                       const Optional<ObjectRef>& device, const Optional<ObjectRef>& from_to) {
      return TileLayout::FromTile(shape, inner, device, from_to);
    });

}  // namespace tirxx
}  // namespace tvm
