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
#include <tvm/arith/analyzer.h>
#include <tvm/node/serialization.h>
#include <tvm/tirx/layout.h>
#include <tvm/tirx/op.h>

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

bool IterTreeSplit::IsLeaf() const { return this->get()->children.size() == 0; }

Array<IterTreeBase> IterTreeSplit::GetLeaves() const {
  std::vector<IterTreeBase> leaves;
  std::function<void(const IterTreeBase&)> visit = [&](const IterTreeBase& node) {
    if (auto* split = node.as<IterTreeSplitNode>()) {
      if (GetRef<IterTreeSplit>(split).IsLeaf()) {
        leaves.push_back(node);
      }
      for (const auto& child : split->children) {
        visit(child);
      }
    } else {
      LOG(FATAL) << "InternalError: Expect IterTreeSplit but get " << node->GetTypeKey();
    }
  };
  visit(*this);
  return Array<IterTreeBase>(leaves);
}

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

Array<IterTreeBase> IterTree::GetLeaves() const { return this->get()->root.GetLeaves(); }

IterTree::LeafIndexMap IterTree::GetLeafIndexMap(Optional<Array<IterTreeBase>> opt_leaves) const {
  Array<IterTreeBase> leaves = opt_leaves.defined() ? opt_leaves.value() : GetLeaves();
  LeafIndexMap leaf_to_index;
  for (size_t i = 0; i < leaves.size(); i++) {
    auto it = leaf_to_index.find(leaves[i]);
    ICHECK(it == leaf_to_index.end()) << "ValueError: Duplicate leaf node in the tree";
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

DataIterTree::CoeffMap DataIterTree::GetCoeffMap(Optional<Array<IterTreeBase>> opt_leaves) const {
  auto leaves = opt_leaves.defined() ? opt_leaves.value() : GetLeaves();
  ICHECK_EQ(leaves.size(), this->get()->coeff.size())
      << "ValueError: The number of leaves and coefficients must be the same";
  CoeffMap coeff_map;
  for (size_t i = 0; i < leaves.size(); i++) {
    auto it = coeff_map.find(leaves[i]);
    ICHECK(it == coeff_map.end()) << "ValueError: Duplicate leaf node in the tree";
    coeff_map[leaves[i]] = this->get()->coeff[i];
  }
  return std::move(coeff_map);
}

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

bool DeviceIterAttr::IsReplicate() const { return this->get()->type == kReplicate; }

bool DeviceIterAttr::IsSplit() const { return this->get()->type == kSplit; }

bool DeviceIterAttr::IsExclusive() const { return this->get()->type == kExclusive; }

size_t DeviceIterAttr::GetIntBound() const {
  ICHECK(this->get()->bound.defined()) << "ValueError: The bound is not defined";
  auto bound = this->get()->bound.value();
  const auto* n = bound.as<IntImmNode>();
  ICHECK(n != nullptr) << "ValueError: The bound is not an integer";
  ICHECK(n->value >= 0) << "ValueError: The bound must be non-negative";
  return n->value;
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

DeviceIterTree::AttrMap DeviceIterTree::GetAttrMap(Optional<Array<IterTreeBase>> opt_leaves) const {
  auto leaves = opt_leaves.defined() ? opt_leaves.value() : GetLeaves();
  ICHECK_EQ(leaves.size(), this->get()->attrs.size())
      << "ValueError: The number of leaves and attributes must be the same";
  AttrMap attr_map;
  for (size_t i = 0; i < leaves.size(); i++) {
    auto it = attr_map.find(leaves[i]);
    ICHECK(it == attr_map.end()) << "ValueError: Duplicate leaf node in the tree";
    attr_map[leaves[i]] = this->get()->attrs[i];
  }
  return std::move(attr_map);
}

// TileLayout
TileLayout::TileLayout(DataIterTree data_tree, Optional<DeviceIterTree> device_tree,
                       Optional<ExecScope> from, Optional<ExecScope> to) {
  auto n = make_object<TileLayoutNode>();
  n->data_tree = std::move(data_tree);
  n->device_tree = std::move(device_tree);
  n->from = std::move(from);
  n->to = std::move(to);
  if (n->from.defined()) {
    ICHECK(n->to.defined()) << "ValueError: The to scope must be defined if the from scope is";
    ICHECK(Higher(n->to.value(), n->from.value()))
        << "ValueError: The from scope must be higher than the to scope";
  }
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(TileLayoutNode);

TVM_REGISTER_GLOBAL("tirx.TileLayout")
    .set_body_typed([](DataIterTree data_tree, Optional<DeviceIterTree> device_tree,
                       Optional<ExecScope> from, Optional<ExecScope> to) {
      return TileLayout(data_tree, device_tree, from, to);
    });

TileLayout TileLayout::FromMaps(IterTreeSplit data_root, Optional<IterTreeSplit> device_root,
                                const DataIterTree::CoeffMap& coeff_map,
                                const DeviceIterTree::AttrMap& attr_map, const SplitMap& split_map,
                                Optional<ExecScope> from, Optional<ExecScope> to) {
  // Construct the coefficient array
  std::vector<PrimExpr> coeff;
  auto data_leaves = data_root.GetLeaves();
  ICHECK_EQ(data_leaves.size(), coeff_map.size())
      << "ValueError: The number of leaves and coefficients must be the same";
  for (const auto& leaf : data_leaves) {
    auto it = coeff_map.find(leaf);
    ICHECK(it != coeff_map.end()) << "ValueError: The coefficient of the leaf is not found";
    coeff.push_back(it->second);
  }
  DataIterTree data_tree(data_root, Array<PrimExpr>(coeff));
  if (!device_root.defined()) {
    ICHECK(attr_map.empty() && split_map.empty())
        << "ValueError: The attribute map and split map must be empty";
    // Only data tree
    return TileLayout(data_tree);
  } else {
    // Get the inverse split map
    SplitMap inv_split_map;
    for (const auto& kv : split_map) {
      ICHECK(inv_split_map.find(kv.second) == inv_split_map.end())
          << "ValueError: A device leaf is mapped to multiple data leaves";
      inv_split_map[kv.second] = kv.first;
    }
    // Construct the new device attrs array
    auto data_leaf2idx = data_tree.GetLeafIndexMap(data_leaves);
    std::vector<DeviceIterAttr> attrs;
    auto device_leaves = device_root.value().GetLeaves();
    ICHECK_EQ(device_leaves.size(), attr_map.size())
        << "InternalError: The number of leaves and attributes must be the same";
    for (const auto& leaf : device_leaves) {
      auto it = attr_map.find(leaf);
      ICHECK(it != attr_map.end()) << "ValueError: The attribute of the leaf is not found";
      if (it->second.IsSplit()) {
        auto split_it = inv_split_map.find(leaf);
        ICHECK(split_it != inv_split_map.end())
            << "InternalEror: A device leaf with Split attr is not mapped to any data leaf";
        auto data_it = data_leaf2idx.find(split_it->second);
        ICHECK(data_it != data_leaf2idx.end())
            << "InternalError: A data leaf is not found in the data tree";
        attrs.push_back(DeviceIterAttr::Split(data_it->second));
      } else {
        attrs.push_back(it->second);
      }
    }
    DeviceIterTree device_tree(device_root.value(), Array<DeviceIterAttr>(attrs));
    return TileLayout(data_tree, device_tree, from, to);
  }
}

TileLayout::SplitMap TileLayout::GetSplitMap(
    Optional<Array<IterTreeBase>> opt_data_leaves,
    Optional<Array<IterTreeBase>> opt_device_leaves) const {
  auto* n = operator->();
  auto data_leaves = opt_data_leaves.defined() ? opt_data_leaves.value() : n->data_tree.GetLeaves();
  if (!n->device_tree.defined()) {
    // No device tree is defined, return an empty map
    return {};
  }
  const auto& device_tree = n->device_tree.value();
  auto device_leaves =
      opt_device_leaves.defined() ? opt_device_leaves.value() : device_tree.GetLeaves();
  ICHECK_EQ(n->device_tree.value()->attrs.size(), device_leaves.size())
      << "ValueError: The number of leaves and attributes must be the same for the device tree";
  SplitMap split_map;
  for (size_t i = 0; i < device_leaves.size(); i++) {
    auto it = split_map.find(device_leaves[i]);
    ICHECK(it == split_map.end()) << "ValueError: Duplicate leaf node in the tree";
    auto attr = device_tree->attrs[i];
    if (attr.IsSplit()) {
      size_t bound = attr.GetIntBound();
      ICHECK(bound < data_leaves.size())
          << "ValueError: The bound of the split attribute is out of range";
      split_map[data_leaves[bound]] = device_leaves[i];
    }
  }
  return std::move(split_map);
}

/******** Normalization ********/
using NodeSet = std::unordered_set<IterTreeBase, ObjectPtrHash, ObjectPtrEqual>;

class IterTreeMutator {
 public:
  virtual ~IterTreeMutator() = default;

  virtual Optional<IterTreeBase> Visit(IterTreeBase node) {
    if (auto* split = node.as<IterTreeSplitNode>()) {
      return VisitIterSplit(split);
    } else {
      LOG(FATAL) << "InternalError: Expect IterTreeSplit but get " << node->GetTypeKey();
      return IterTreeBase();
    }
  }

  virtual Optional<IterTreeBase> VisitIterSplit(const IterTreeSplitNode* node) {
    bool changed = false;
    std::vector<IterTreeBase> new_children;
    for (const auto& child : node->children) {
      auto new_child = Visit(child);
      if (new_child.defined()) {
        new_children.push_back(new_child.value());
        changed |= (!new_child.same_as(child));
      } else {
        changed = true;
      }
    }
    if (changed) {
      auto* n = GetRef<IterTreeSplit>(node).CopyOnWrite();
      n->children = new_children;
      return GetRef<IterTreeBase>(n);
    } else {
      return GetRef<IterTreeBase>(node);
    }
  }
};

class DeepCopyMutator : public IterTreeMutator {
 public:
  static IterTreeBase DeepCopy(const IterTreeBase& node) {
    auto res = DeepCopyMutator().Visit(node);
    ICHECK(res.defined()) << "InternalError: The result should not be null";
    return res.value();
  }

  Optional<IterTreeBase> VisitIterSplit(const IterTreeSplitNode* node) final {
    std::vector<IterTreeBase> new_children;
    for (const auto& child : node->children) {
      auto new_child = Visit(child);
      ICHECK(new_child.defined()) << "InternalError: The child should not be null";
      new_children.push_back(new_child.value());
    }
    return IterTreeSplit(node->extent, Array<IterTreeBase>(new_children));
  }
};

class DeduplicateMutator : public IterTreeMutator {
 public:
  explicit DeduplicateMutator(NodeSet* node_set = nullptr) : node_set_(node_set) {}

  static IterTreeBase Deduplicate(const IterTreeBase& node, NodeSet* node_set) {
    auto res = DeduplicateMutator(node_set).Visit(node);
    ICHECK(res.defined()) << "InternalError: The result should not be null";
    return res.value();
  }

  Optional<IterTreeBase> VisitIterSplit(const IterTreeSplitNode* node) final {
    if (node_set_->count(GetRef<IterTreeBase>(node)) != 0) {
      return DeepCopyMutator::DeepCopy(GetRef<IterTreeBase>(node));
    }
    node_set_->insert(GetRef<IterTreeBase>(node));
    bool changed = false;
    std::vector<IterTreeBase> new_children;
    for (const auto& child : node->children) {
      auto new_child = Visit(child);
      ICHECK(new_child.defined()) << "InternalError: The child should not be null";
      new_children.push_back(new_child.value());
      changed |= (!new_children.back().same_as(child));
    }
    if (changed) {
      auto* n = GetRef<IterTreeSplit>(node).CopyOnWrite();
      n->children = new_children;
      return GetRef<IterTreeBase>(n);
    } else {
      return GetRef<IterTreeBase>(node);
    }
  }

 private:
  NodeSet* node_set_;
};

template <typename T>
T DeduplicateTree(T tree, NodeSet* visited) {
  auto new_root = DeduplicateMutator::Deduplicate(tree->root, visited);
  if (!new_root.same_as(tree->root)) {
    auto* n = tree.CopyOnWrite();
    n->root = Downcast<IterTreeSplit>(new_root);
    return GetRef<T>(n);
  } else {
    return tree;
  }
}

std::vector<ObjectRef> Deduplicate(std::vector<ObjectRef> inputs) {
  std::unordered_set<IterTreeBase, ObjectPtrHash, ObjectPtrEqual> visited;

  std::vector<ObjectRef> res;
  for (auto& input : inputs) {
    if (auto opt_layout = input.as<TileLayout>()) {
      auto layout = opt_layout.value();
      auto data_tree = DeduplicateTree<DataIterTree>(layout->data_tree, &visited);
      if (!layout->device_tree.defined()) {
        // Only data tree is defined
        if (data_tree.same_as(layout->data_tree)) {
          res.push_back(layout);
        } else {
          auto* n = layout.CopyOnWrite();
          n->data_tree = data_tree;
          res.push_back(GetRef<TileLayout>(n));
        }
      } else {
        // Both data tree and device tree are defined
        auto device_tree = DeduplicateTree<DeviceIterTree>(layout->device_tree.value(), &visited);
        if (data_tree.same_as(layout->data_tree) &&
            device_tree.same_as(layout->device_tree.value())) {
          res.push_back(layout);
        } else {
          auto* n = layout.CopyOnWrite();
          n->data_tree = Downcast<DataIterTree>(data_tree);
          n->device_tree = Downcast<DeviceIterTree>(device_tree);
          res.push_back(GetRef<TileLayout>(n));
        }
      }
    } else if (auto opt_iter_tree = input.as<IterTree>()) {
      res.push_back(DeduplicateTree<IterTree>(opt_iter_tree.value(), &visited));
    } else {
      LOG(FATAL) << "InternalError: Expect TileLayout or IterTree but get " << input->GetTypeKey();
    }
  }
  return std::move(res);
}

class UnitIterRemover : public IterTreeMutator {
 public:
  explicit UnitIterRemover(bool is_data, DataIterTree::CoeffMap* coeff_map,
                           DeviceIterTree::AttrMap* attr_map = nullptr,
                           TileLayout::SplitMap* split_map = nullptr)
      : is_data_(is_data), coeff_map_(coeff_map), attr_map_(attr_map), split_map_(split_map) {}

  static Optional<IterTreeSplit> RemoveUnitIter(IterTreeBase root, bool is_data,
                                                DataIterTree::CoeffMap* coeff_map,
                                                DeviceIterTree::AttrMap* attr_map = nullptr,
                                                TileLayout::SplitMap* split_map = nullptr) {
    auto res = UnitIterRemover(is_data, coeff_map, attr_map, split_map).Visit(root);
    return res.as<IterTreeSplit>();
  }

  Optional<IterTreeBase> VisitIterSplit(const IterTreeSplitNode* node) final {
    auto root = GetRef<IterTreeSplit>(node);
    if (root.IsLeaf()) {
      if (is_one(root->extent)) {
        if (is_data_) {
          // Clear the coeff in data tree
          coeff_map_->erase(root);
          if (attr_map_ != nullptr) {
            // Clear the attr in device tree
            ICHECK(split_map_ != nullptr) << "InternalError: The split map should be defined";
            auto dev_it = split_map_->find(root);
            if (dev_it != split_map_->end()) {
              attr_map_->erase(dev_it->second);
              split_map_->erase(dev_it);
            }
          }
        }
        // remove this iter
        return NullOpt;
      } else {
        return root;
      }
    } else {
      bool changed = false;
      std::vector<IterTreeBase> new_children;
      for (const auto& child : root->children) {
        bool is_root_cur = is_root_;
        is_root_ = false;
        auto new_child = Visit(child);
        is_root_ = is_root_cur;
        if (new_child.defined()) {
          new_children.push_back(new_child.value());
          changed |= (!new_child.same_as(child));
        } else {
          changed = true;
        }
      }
      // Case 1: all children are removed
      if (new_children.size() == 0) {
        ICHECK(is_one(root->extent)) << "InternalError: The extent of the root should be 1";
        if (!is_root_) {
          return NullOpt;
        } else {
          // return root -> 1
          auto unit = IterTreeSplit(1, {});
          if (is_data_) {
            coeff_map_->insert({unit, 1});
          } else {
            attr_map_->insert({unit, DeviceIterAttr::Replicate()});
          }
          auto* n = root.CopyOnWrite();
          n->children = {unit};
          return GetRef<IterTreeBase>(n);
        }
      }
      // Case 2: only one child is kept and the node is not root
      if (new_children.size() == 1 && !is_root_) {
        const auto& split = new_children[0].as<IterTreeSplit>();
        ICHECK(split != nullptr) << "InternalError: Expect IterTreeSplit but get "
                                 << new_children[0]->GetTypeKey();
        return split.value();
      }
      // Case 3: multiple children are kept or the node is root
      if (changed) {
        auto* n = root.CopyOnWrite();
        n->children = new_children;
        return GetRef<IterTreeBase>(n);
      } else {
        return root;
      }
    }
  }

 private:
  bool is_data_;
  bool is_root_{true};
  DataIterTree::CoeffMap* coeff_map_;
  DeviceIterTree::AttrMap* attr_map_;
  TileLayout::SplitMap* split_map_;
};

TileLayout RemoveUnitIter(TileLayout layout) {
  const auto& data_tree = layout->data_tree;
  const auto& data_leaves = data_tree.GetLeaves();
  DataIterTree::CoeffMap coeff_map = data_tree.GetCoeffMap(data_leaves);
  if (!layout->device_tree.defined()) {
    // Only data tree is defined
    auto new_data_root = UnitIterRemover::RemoveUnitIter(data_tree->root, true, &coeff_map);
    ICHECK(new_data_root.defined()) << "InternalError: The data tree should be defined";
    ICHECK_GT(new_data_root.value()->children.size(), 0)
        << "InternalError: The root of the data tree should have at least one child";
    return TileLayout::FromMaps(new_data_root.value(), NullOpt, coeff_map, {}, {});
  } else {
    // Both data tree and device tree are defined
    const auto& dev_tree = layout->device_tree.value();
    const auto& dev_leaves = dev_tree.GetLeaves();
    DeviceIterTree::AttrMap attr_map = dev_tree.GetAttrMap(dev_leaves);
    TileLayout::SplitMap split_map = layout.GetSplitMap(data_leaves, dev_leaves);

    auto new_data_root =
        UnitIterRemover::RemoveUnitIter(data_tree->root, true, &coeff_map, &attr_map, &split_map);
    auto new_dev_root =
        UnitIterRemover::RemoveUnitIter(dev_tree->root, false, &coeff_map, &attr_map, &split_map);
    ICHECK(new_data_root.defined() && new_dev_root.defined())
        << "InternalError: The data tree and device tree should be both defined";
    return TileLayout::FromMaps(new_data_root.value(), new_dev_root.value(), coeff_map, attr_map,
                                split_map, layout->from, layout->to);
  }
}

TileLayout NormalizeTileLayout(TileLayout layout) {
  TileLayout res = Downcast<TileLayout>(Deduplicate({layout})[0]);
  res = RemoveUnitIter(res);
  return std::move(res);
}

TVM_REGISTER_GLOBAL("tirx.NormalizeTileLayout").set_body_typed(NormalizeTileLayout);

/******** Tile ********/
TileLayout Tile(TileLayout outer, TileLayout inner) {
  auto dedup = Deduplicate({outer, inner});
  outer = Downcast<TileLayout>(dedup[0]);
  inner = Downcast<TileLayout>(dedup[1]);
  // check the data tree roots of inner and outer layouts have the same number of children
  auto* inner_n = inner.operator->();
  auto* outer_n = outer.operator->();
  const auto& inner_children = inner_n->data_tree->root->children;
  ICHECK_EQ(inner_children.size(), outer_n->data_tree->root->children.size())
      << "ValueError: The number of children of the data tree root must be the same";
  ICHECK(!outer->device_tree.defined() && !outer->from.defined() && !outer->to.defined())
      << "ValueError: The outer layout must not have device tree or scope";
  // Find the maximum stride in the inner layout
  arith::Analyzer analyzer;
  PrimExpr inner_extent = 1;
  PrimExpr inner_stride = 0;
  auto inner_leaves = inner_n->data_tree.GetLeaves();
  for (size_t i = 0; i < inner_leaves.size(); i++) {
    auto coeff = inner_n->data_tree->coeff[i];
    if (analyzer.CanProve(inner_stride < coeff)) {
      inner_stride = coeff;
      inner_extent = inner_leaves[i].as<IterTreeSplitNode>()->extent;
    }
  }
  inner_stride = analyzer.Simplify(inner_stride * inner_extent);
  // Update the coeff of the outer layout
  std::vector<PrimExpr> coeff;
  for (const auto& stride : outer_n->data_tree->coeff) {
    coeff.push_back(stride * inner_stride);
  }
  auto* outer_n_cow = outer.CopyOnWrite();
  outer_n_cow->data_tree = DataIterTree(outer_n->data_tree->root, Array<PrimExpr>(coeff));
  TileLayout new_outer = GetRef<TileLayout>(outer_n_cow);
  // Construct the tiled layout
  PrimExpr extent = 1;
  std::vector<IterTreeBase> children;
  for (size_t i = 0; i < inner_children.size(); i++) {
    const auto* inner = inner_children[i].as<IterTreeSplitNode>();
    const auto* outer = outer_n->data_tree->root->children[i].as<IterTreeSplitNode>();
    ICHECK(inner != nullptr) << "InternalError: Expect IterTreeSplit but get "
                             << inner_children[i]->GetTypeKey();
    ICHECK(outer != nullptr) << "InternalError: Expect IterTreeSplit but get "
                             << outer_n->data_tree->root->children[i]->GetTypeKey();
    PrimExpr child_extent = analyzer.Simplify(inner->extent * outer->extent);
    children.push_back(
        IterTreeSplit(child_extent, {GetRef<IterTreeBase>(outer), GetRef<IterTreeBase>(inner)}));
    extent = extent * child_extent;
  }
  IterTreeSplit root(analyzer.Simplify(extent), children);
  auto coeff_map = inner_n->data_tree.GetCoeffMap();
  auto coeff_map_outer = new_outer->data_tree.GetCoeffMap();
  coeff_map.insert(coeff_map_outer.begin(), coeff_map_outer.end());
  return TileLayout::FromMaps(root, NullOpt, coeff_map, {}, {}, inner_n->from, inner->to);
}

TVM_REGISTER_GLOBAL("tirx.TileLayoutTile").set_body_typed([](TileLayout outer, TileLayout inner) {
  return Tile(outer, inner);
});

/******** Shard ********/
std::vector<DeviceIterAttr> ParseStrategy(String strategy) {
  std::vector<DeviceIterAttr> attrs;
  size_t cur = 0;
  auto f_parse_number_with_bracket = [&cur, &strategy]() {
    std::string number;
    while ((++cur) < strategy.size() && isdigit(strategy.at(cur))) {
      number.push_back(strategy.at(cur));
    }
    ICHECK_GT(number.size(), 0) << "ValueError: Invalid strategy " << strategy;
    return atoi(number.c_str());
  };
  for (; cur < strategy.size();) {
    if (strategy.at(cur) == 'S') {
      attrs.push_back(DeviceIterAttr::Split(f_parse_number_with_bracket()));
    } else if (strategy.at(cur) == 'E') {
      attrs.push_back(DeviceIterAttr::Exclusive(f_parse_number_with_bracket()));
    } else if (strategy.at(cur) == 'R') {
      attrs.push_back(DeviceIterAttr::Replicate());
      ++cur;
    } else {
      LOG(FATAL) << "ValueError: Invalid strategy " << strategy;
    }
  }
  return std::move(attrs);
}

TileLayout Shard(Array<PrimExpr> shape, IterTree mesh, String strategy, TileLayout inner,
                 ExecScope from, ExecScope to) {
  auto res = Deduplicate({mesh, inner});
  mesh = Downcast<IterTree>(res[0]);
  inner = Downcast<TileLayout>(res[1]);
  arith::Analyzer analyzer;
  auto coeff_map = inner->data_tree.GetCoeffMap();
  auto attr_map = inner->device_tree.defined() ? inner->device_tree.value().GetAttrMap()
                                               : tvm::tirx::DeviceIterTree::AttrMap();
  auto split_map = inner.GetSplitMap();
  std::unordered_map<int, IterTreeSplit> outer_leaves;
  // Parse the strategy
  Array<IterTreeBase> mesh_leaves = mesh.GetLeaves();
  auto attrs = ParseStrategy(strategy);
  ICHECK_EQ(attrs.size(), mesh_leaves.size())
      << "ValueError: The number of attributes must be the same as the number of mesh leaves";
  // Construct the split map and attr map
  for (size_t i = 0; i < mesh_leaves.size(); i++) {
    auto leaf = mesh_leaves[i];
    auto attr = attrs[i];
    if (attr.IsSplit()) {
      auto bound = attr.GetIntBound();
      ICHECK_LT(bound, shape.size())
          << "ValueError: The bound of the split attribute is out of range";
      ICHECK(outer_leaves.find(bound) == outer_leaves.end())
          << "ValueError: The " << bound << "-th outer leaf is already mapped to a device leaf";
      auto outer_leaf = IterTreeSplit(Downcast<IterTreeSplit>(leaf)->extent, {});
      split_map[outer_leaf] = leaf;
      outer_leaves[bound] = outer_leaf;
    }
    attr_map[leaf] = attr;
  }
  // Create the new data tree
  ICHECK_EQ(shape.size(), inner->data_tree->root->children.size())
      << "ValueError: The number of shape dimensions must be the same as the number of mesh "
         "children";
  PrimExpr data_extent = 1;
  std::vector<IterTreeBase> new_roots;
  for (size_t i = 0; i < shape.size(); i++) {
    const auto& inner_leaf = inner->data_tree->root->children[i];
    auto extent = analyzer.Simplify(shape[i]);
    auto it = outer_leaves.find(i);
    if (it == outer_leaves.end()) {
      // The axis is not bound to any device leaf
      new_roots.push_back(inner_leaf);
      ICHECK(analyzer.CanProveEqual(extent, Downcast<IterTreeSplit>(inner_leaf)->extent))
          << "ValueError: Shape mismatch for the " << i << "-th axis";
    } else {
      // The axis is bound to a device leaf
      const auto& outer_leaf = it->second;
      ICHECK(analyzer.CanProveEqual(
          extent, outer_leaf->extent * Downcast<IterTreeSplit>(inner_leaf)->extent))
          << "ValueError: Shape mismatch for the " << i << "-th axis";
      auto new_root = IterTreeSplit(extent, {outer_leaf, inner_leaf});
      new_roots.push_back(new_root);
      coeff_map[outer_leaf] = -1;
    }
    data_extent = data_extent * extent;
  }
  auto data_root = IterTreeSplit(data_extent, new_roots);
  // Construct the new mesh tree
  auto mesh_root = mesh->root;
  if (inner->device_tree.defined()) {
    std::vector<IterTreeBase> new_children;
    new_children.insert(new_children.end(), mesh->root->children.begin(),
                        mesh->root->children.end());
    new_children.insert(new_children.end(), inner->device_tree.value()->root->children.begin(),
                        inner->device_tree.value()->root->children.end());
    mesh_root = IterTreeSplit(mesh->root->extent * inner->device_tree.value()->root->extent,
                              Array<IterTreeBase>(new_children));
  }
  // Construct the from and to scopes
  if (inner->device_tree.defined()) {
    ICHECK(inner->to.defined()) << "ValueError: The inner layout must have the to scope";
    ICHECK(Equal(inner->to.value(), from))
        << "ValueError: The from scope of the inner layout must be the same as the to scope of "
           "the outer layout";
  }
  return TileLayout::FromMaps(data_root, mesh_root, coeff_map, attr_map, split_map,
                              inner->device_tree.defined() ? inner->from : from, to);
}

TVM_REGISTER_GLOBAL("tirx.TileLayoutShard")
    .set_body_typed([](Array<PrimExpr> shape, IterTree mesh, String strategy, TileLayout inner,
                       ExecScope from,
                       ExecScope to) { return Shard(shape, mesh, strategy, inner, from, to); });

}  // namespace tirx
}  // namespace tvm
