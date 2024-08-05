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
#include <tvm/node/structural_equal.h>
#include <tvm/tirx/layout.h>
#include <tvm/tirx/op.h>

namespace tvm {
namespace tirx {

/**************** Utils ****************/
PrimExpr ReduceMul(Array<PrimExpr> values) {
  PrimExpr result = values[0];
  for (size_t i = 1; i < values.size(); i++) {
    result = result * values[i];
  }
  return result;
}

bool IsTrivialLayout(const TLayout& layout) {
  auto tile_layout = layout.as<TileLayoutNode>();
  if (tile_layout == nullptr) {
    return false;
  }
  if (tile_layout->device_tree.defined()) {
    return false;
  }
  ICHECK(tile_layout->data_tree.defined()) << "InternalError: The data tree should be defined";
  const auto& leaves = tile_layout->data_tree.GetLeaves();
  ICHECK_EQ(leaves.size(), tile_layout->data_tree->coeff.size())
      << "InternalError: The number of leaves and coefficients should be the same";
  arith::Analyzer ana;
  PrimExpr expected_stride = 1;
  for (int i = leaves.size() - 1; i >= 0; --i) {
    if (!ana.CanProveEqual(tile_layout->data_tree->coeff[i], expected_stride)) {
      return false;
    }
    expected_stride = expected_stride * Downcast<IterTreeSplit>(leaves[i])->extent;
  }
  return true;
}

TVM_REGISTER_GLOBAL("tirx.IsTrivialLayout").set_body_typed(IsTrivialLayout);

/**************** TLayout ****************/
TVM_REGISTER_GLOBAL("tirx.TLayoutGetSize").set_body_typed([](TLayout layout) {
  return layout->GetSize();
});

TVM_REGISTER_GLOBAL("tirx.TLayoutGetCosize").set_body_typed([](TLayout layout) {
  return layout->GetCosize();
});

TVM_REGISTER_GLOBAL("tirx.TLayoutApply").set_body_typed([](TLayout layout, Array<PrimExpr> coord) {
  return layout->Apply(coord);
});

/**************** IterTree ****************/
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

/**************** IterTree ****************/
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

class IterTreeVisitor {
 public:
  virtual ~IterTreeVisitor() = default;

  virtual void Visit(const IterTreeBase& node) {
    if (auto* split = node.as<IterTreeSplitNode>()) {
      VisitIterSplit(split);
    } else {
      LOG(FATAL) << "InternalError: Expect IterTreeSplit but get " << node->GetTypeKey();
    }
  }

  virtual void VisitIterSplit(const IterTreeSplitNode* node) = 0;
};

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

class IterTreeVerifier : public IterTreeVisitor {
 public:
  IterTreeVerifier() = default;

  static bool VerifyWellFormed(const IterTreeBase& node) {
    IterTreeVerifier verifier;
    verifier.Visit(node);
    return verifier.result_;
  }

  void VisitIterSplit(const IterTreeSplitNode* node) final {
    if (node->children.empty()) return;
    PrimExpr extent = 1;
    for (const auto& child : node->children) {
      this->Visit(child);
      extent = extent * Downcast<IterTreeSplit>(child)->extent;
    }
    if (!ana_.CanProveEqual(extent, node->extent)) {
      result_ = false;
    }
  }

 private:
  bool result_{true};
  arith::Analyzer ana_;
};

/**************** DataIterTree ****************/
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

/**************** DeviceIterAttr ****************/
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

/**************** DeviceIterTree ****************/
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

/**************** TileLayout ****************/
Array<PrimExpr> TileLayoutNode::GetDefaultShape() const {
  std::vector<PrimExpr> shape;
  for (const auto& child : this->data_tree->root->children) {
    shape.push_back(Downcast<IterTreeSplit>(child)->extent);
  }
  return Array<PrimExpr>(shape);
}

bool TileLayoutNode::CompatibleWithShape(const Array<PrimExpr>& shape) const {
  arith::Analyzer analyzer;
  return analyzer.CanProveEqual(FloorMod(ReduceMul(shape), this->GetSize()), 0);
}

PrimExpr TileLayoutNode::GetSize() const { return this->data_tree->root->extent; }

PrimExpr TileLayoutNode::GetCosize() const {
  auto data_leaves = this->data_tree.GetLeaves();
  auto split_map = GetRef<TileLayout>(this).GetSplitMap(data_leaves);
  arith::Analyzer analyzer;
  PrimExpr cosize = 1;
  for (size_t i = 0; i < data_leaves.size(); ++i) {
    if (split_map.find(data_leaves[i]) != split_map.end()) {
      continue;
    }
    // Check if the coefficient is non-negative
    ICHECK(analyzer.CanProveGreaterEqual(this->data_tree->coeff[i], 0))
        << "ValueError: The coefficient of a non-bound leaf must be non-negative";
    cosize += this->data_tree->coeff[i] * (Downcast<IterTreeSplit>(data_leaves[i])->extent - 1);
  }
  return cosize;
}

bool TileLayoutNode::VerifyWellFormed() const {
  arith::Analyzer analyzer;
  // Verify the data tree
  bool result = IterTreeVerifier::VerifyWellFormed(data_tree->root);
  if (device_tree.defined()) {
    // Verify the device tree if it is defined
    result &= IterTreeVerifier::VerifyWellFormed(device_tree.value()->root);
  }
  // Verify the split map
  auto data_leaves = data_tree.GetLeaves();
  const auto& split_map = GetRef<TileLayout>(this).GetSplitMap(data_leaves);
  for (const auto& kv : split_map) {
    ICHECK(analyzer.CanProveEqual(Downcast<IterTreeSplit>(kv.first)->extent,
                                  Downcast<IterTreeSplit>(kv.second)->extent))
        << "ValueError: The extents of the data and device leaves must be the same";
  }
  // Verify coefficients are non-negative for non-bound leaves in the data tree
  const auto& coeff_map = data_tree.GetCoeffMap(data_leaves);
  for (const auto& kv : coeff_map) {
    auto it = split_map.find(kv.first);
    if (it == split_map.end()) {
      ICHECK(analyzer.CanProveGreaterEqual(kv.second, 0))
          << "ValueError: The coefficient of a non-bound leaf must be non-negative";
    }
  }
  return result;
}

class TileLayoutApplier : public IterTreeVisitor {
 public:
  static Array<PrimExpr> Apply(PrimExpr input, IterTree data_tree) {
    TileLayoutApplier applier(input);
    applier.Visit(data_tree->root);
    std::reverse(applier.result_.begin(), applier.result_.end());
    return std::move(applier.result_);
  }

  explicit TileLayoutApplier(PrimExpr input) : cur_input_{input}, result_{} {}

  void VisitIterSplit(const IterTreeSplitNode* node) final {
    if (node->children.empty()) {
      result_.push_back(cur_input_);
      return;
    }
    for (int i = node->children.size() - 1; i >= 0; i--) {
      auto* child = node->children[i].as<IterTreeSplitNode>();
      ICHECK(child != nullptr) << "InternalError: Expect IterTreeSplit but get "
                               << node->children[i]->GetTypeKey();
      PrimExpr tmp = FloorDiv(cur_input_, child->extent);
      cur_input_ = FloorMod(cur_input_, child->extent);
      VisitIterSplit(child);
      cur_input_ = tmp;
    }
  }

 private:
  PrimExpr cur_input_;
  std::vector<PrimExpr> result_;
};

PrimExpr TileLayoutNode::Apply(const Array<PrimExpr>& coord) const {
  PrimExpr input;
  arith::Analyzer analyzer;
  if (coord.size() == 1) {
    input = coord[0];
  } else {
    const auto& shape = GetDefaultShape();
    ICHECK_EQ(coord.size(), shape.size())
        << "ValueError: The number of coordinates must be the same as the number of dimensions";
    input = 0;
    for (size_t i = 0; i < coord.size(); i++) {
      input = input * shape[i] + coord[i];
    }
  }
  input = analyzer.Simplify(input);
  const auto& size = GetSize();
  const auto& cosize = GetCosize();
  const auto& inner = analyzer.Simplify(floormod(input, size));
  const auto& outer = analyzer.Simplify(floordiv(input, size));

  auto data_leaves = this->data_tree.GetLeaves();
  auto leaf_coord = TileLayoutApplier::Apply(input, this->data_tree);

  ICHECK_EQ(leaf_coord.size(), data_leaves.size())
      << "ValueError: The number of coordinates must be the same as the number of leaves";
  auto split_map = GetRef<TileLayout>(this).GetSplitMap(data_leaves);
  PrimExpr result = 0;
  for (size_t i = 0; i < leaf_coord.size(); i++) {
    if (split_map.find(data_leaves[i]) != split_map.end()) {
      continue;
    }
    result += leaf_coord[i] * this->data_tree->coeff[i];
  }
  return analyzer.Simplify(outer * cosize + result);
}

/**************** TileLayout ****************/
TileLayout::TileLayout(DataIterTree data_tree, Optional<DeviceIterTree> device_tree,
                       Optional<ExecScope> from, Optional<ExecScope> to) {
  auto n = make_object<TileLayoutNode>();
  n->data_tree = std::move(data_tree);
  n->device_tree = std::move(device_tree);
  n->from = std::move(from);
  n->to = std::move(to);
  if (n->from.defined()) {
    ICHECK(n->to.defined()) << "ValueError: The to scope must be defined if the from scope is";
    ICHECK(n->to.value().Higher(n->from.value()))
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
    auto n = make_object<typename T::ContainerType>(*(tree.operator->()));
    n->root = Downcast<IterTreeSplit>(new_root);
    return T(n);
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
    } else if (auto opt_iter_tree = input.as<DataIterTree>()) {
      res.push_back(DeduplicateTree<DataIterTree>(opt_iter_tree.value(), &visited));
    } else if (auto opt_iter_tree = input.as<DeviceIterTree>()) {
      res.push_back(DeduplicateTree<DeviceIterTree>(opt_iter_tree.value(), &visited));
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
              split_map_->erase(dev_it);
            }
          }
        } else {
          attr_map_->erase(root);
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
            auto* n = root.CopyOnWrite();
            n->children = {unit};
            return GetRef<IterTreeBase>(n);
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

class EquivIterFuser : public IterTreeMutator {
 public:
  explicit EquivIterFuser(bool only_data, DataIterTree::CoeffMap* coeff_map,
                          DeviceIterTree::AttrMap* attr_map = nullptr,
                          TileLayout::SplitMap* split_map = nullptr,
                          const DeviceIterTree* dev_tree = nullptr)
      : only_data_(only_data),
        coeff_map_(coeff_map),
        attr_map_(attr_map),
        split_map_(split_map),
        dev_tree_(dev_tree),
        ana_() {}

  static Optional<IterTreeSplit> FuseEquivIter(IterTreeBase root, bool only_data,
                                               DataIterTree::CoeffMap* coeff_map,
                                               DeviceIterTree::AttrMap* attr_map = nullptr,
                                               TileLayout::SplitMap* split_map = nullptr,
                                               const DeviceIterTree* dev_tree = nullptr) {
    auto res = EquivIterFuser(only_data, coeff_map, attr_map, split_map, dev_tree).Visit(root);
    return res.as<IterTreeSplit>();
  }

  PrimExpr ComputeGCD(PrimExpr a, PrimExpr b, arith::Analyzer* ana) {
    while (!ana->CanProveEqual(b, 0)) {
      PrimExpr temp = b;
      b = floormod(a, b);
      a = temp;
    }
    return a;
  }

  bool CheckEquiv(PrimExpr a, PrimExpr b, arith::Analyzer* ana) { return ana->CanProveEqual(a, b); }

  IterTreeSplit* Check_device_adjacent(const IterTreeBase& device1, const IterTreeBase& device2,
                                       IterTreeSplit* dev_node) {
    if (dev_node->IsLeaf()) {
      return nullptr;
    }
    // add recursive check for device leaf nodes
    tvm::tirx::IterTreeSplit* curr_res = nullptr;
    std::vector<IterTreeBase> dev_children;
    for (const auto& child : (*dev_node)->children) {
      // recursively check all device leaf nodes
      auto* child_node = child.as<IterTreeSplitNode>();
      auto input_child_node = GetRef<IterTreeSplit>(child_node);
      auto new_curr_res = Check_device_adjacent(device1, device2, &input_child_node);
      ICHECK(!(curr_res != nullptr && new_curr_res != nullptr))
          << "InternalError: device nodes can only be fused once!";
      curr_res = new_curr_res == nullptr ? curr_res : new_curr_res;
      dev_children.push_back(((Optional<IterTreeBase>)input_child_node).value());
    }
    if (curr_res == nullptr) {
      for (int i = (int)dev_children.size() - 1; i >= 1; --i) {
        const auto& dev_child1 = Downcast<IterTreeSplit>(dev_children[i]);
        const auto& dev_child2 = Downcast<IterTreeSplit>(dev_children[i - 1]);
        if (dev_child1.defined() && dev_child2.defined() && dev_child1.IsLeaf() &&
            dev_child2.IsLeaf() && dev_child1.same_as(device1) && dev_child2.same_as(device2)) {
          // Fuse device 1 and device 2 to one device leaf node with the new extent equal to
          // the product of the two
          auto new_extent = dev_child1->extent * dev_child2->extent;
          curr_res = new IterTreeSplit();
          *curr_res = IterTreeSplit(new_extent, {});
          // Replace the old dev_child1 and dev_child2 with *curr_res
          dev_children.erase(dev_children.begin() + i);
          dev_children[i - 1] = GetRef<IterTreeBase>((*curr_res).get());
          auto* n = (*dev_node).CopyOnWrite();
          n->children = dev_children;
          return curr_res;
        }
      }
    } else {
      // If changed at the bottom, update children at the top recursively
      auto* n = (*dev_node).CopyOnWrite();
      n->children = dev_children;
      return curr_res;
    }
    return curr_res;
  }

  Optional<IterTreeBase> VisitIterSplit(const IterTreeSplitNode* node) final {
    auto root = GetRef<IterTreeSplit>(node);
    if (only_data_) {
      if (root.IsLeaf()) {
        // if it's a leaf, keep it
        return root;
      } else {
        // if it's a node, use bottom-up recursion
        std::vector<IterTreeBase> new_children;
        bool all_leaves = true;
        bool changed = false;
        bool fused = false;
        for (const auto& child : root->children) {
          // only normalize consider all-children-leaves case
          bool is_root_cur = is_root_;
          is_root_ = false;
          auto new_child = Visit(child);
          is_root_ = is_root_cur;
          auto new_child_node = GetRef<IterTreeSplit>(new_child.as<IterTreeSplitNode>());
          ICHECK(new_child.defined()) << "InternalError: New child must be valid";
          all_leaves = new_child_node.IsLeaf() ? all_leaves : false;
          new_children.push_back(new_child.value());
          changed |= (!new_child.same_as(child));
        }

        PrimExpr gcd_coeff;
        PrimExpr min_coeff;
        if (all_leaves) {
          // compute GCD of coefficients of all leaves
          auto it = coeff_map_->find(Downcast<IterTreeSplit>(new_children[0]));
          ICHECK(it != coeff_map_->end()) << "InternalError: Coefficient not found for child";
          gcd_coeff = it->second;
          min_coeff = it->second;
          for (size_t i = 1; i < new_children.size(); ++i) {
            it = coeff_map_->find(Downcast<IterTreeSplit>(new_children[i]));
            ICHECK(it != coeff_map_->end()) << "InternalError: Coefficient not found for child";
            gcd_coeff = ComputeGCD(gcd_coeff, it->second, &ana_);
            min_coeff = tvm::min(it->second, min_coeff);
          }
          // check if normalize requirement is satisfied by coeff and extents
          fused = CheckEquiv(min_coeff, gcd_coeff, &ana_);
          PrimExpr divider_product = gcd_coeff;
          for (int i = (int)new_children.size() - 1; i >= 0; --i) {
            // reversal order to check mod and coeff
            const auto& child = Downcast<IterTreeSplit>(new_children[i]);
            auto coeff_it = coeff_map_->find(child);
            ICHECK(coeff_it != coeff_map_->end())
                << "InternalError: Coefficient not found for child";
            if (!ana_.CanProveEqual(divider_product, coeff_it->second)) {
              fused = false;
              break;
            }
            divider_product *= Downcast<IterTreeSplit>(new_children[i])->extent;
          }
        }
        if (fused) {
          // fuse all children into the current node
          if (!is_root_) {
            for (int i = (int)new_children.size() - 1; i >= 0; --i) {
              coeff_map_->erase(GetRef<IterTreeSplit>(new_children[i].as<IterTreeSplitNode>()));
            }
            coeff_map_->erase(root);
            auto* n = root.CopyOnWrite();
            n->children = {};
            coeff_map_->insert({GetRef<IterTreeSplit>(n), gcd_coeff});
            return GetRef<IterTreeBase>(n);
          } else {
            // add a dummy leaf if normalizing root
            for (int i = (int)new_children.size() - 1; i >= 0; --i) {
              coeff_map_->erase(GetRef<IterTreeSplit>(new_children[i].as<IterTreeSplitNode>()));
            }
            coeff_map_->erase(root);
            auto* n = root.CopyOnWrite();
            auto unit = IterTreeSplit(root->extent, {});
            coeff_map_->insert({unit, gcd_coeff});
            n->children = {unit};
            return GetRef<IterTreeBase>(n);
          }
        } else {
          // no fuse happened
          if (changed) {
            auto* n = root.CopyOnWrite();
            n->children = new_children;
            return GetRef<IterTreeBase>(n);
          } else {
            return root;
          }
        }
      }
    } else {
      // Fuse the cases when both device tree and data tree are defined as follows:
      // For any two of data leaf nodes that share the same parent and adjacent to each other,
      // If two data nodes are mapped to two device nodes that are adjacent to each other,
      // then fuse those two data leaf nodes to one single data node with the extent of the new
      // node equal to the product of the extents of the two rest of data tree structure
      // unchanged, and fuse those two device nodes to one single node with the value being the
      // product of those two device nodes
      if (root.IsLeaf()) {
        // if it's a leaf, keep it
        return root;
      } else {
        // if it's a node, use bottom-up recursion
        std::vector<IterTreeBase> new_children;
        bool changed = false;
        for (const auto& child : root->children) {
          // only normalize consider all-children-leaves case
          bool is_root_cur = is_root_;
          is_root_ = false;
          auto new_child = Visit(child);
          is_root_ = is_root_cur;
          auto new_child_node = GetRef<IterTreeSplit>(new_child.as<IterTreeSplitNode>());
          ICHECK(new_child.defined()) << "InternalError: New child must be valid";
          new_children.push_back(new_child.value());
          changed |= (!new_child.same_as(child));
        }
        int children_cnt = (int)new_children.size();
        for (int i = children_cnt - 1; i >= 1; --i) {
          // reversal order to check mod and coeff
          const auto& child1 = Downcast<IterTreeSplit>(new_children[i]);
          const auto& child2 = Downcast<IterTreeSplit>(new_children[i - 1]);
          auto device1 = split_map_->find(child1);
          auto device2 = split_map_->find(child2);

          if (device1 != split_map_->end() && device2 != split_map_->end()) {
            const tvm::tirx::IterTreeSplit* fused_dev_node =
                Check_device_adjacent(device1->second, device2->second,
                                      const_cast<tvm::tirx::IterTreeSplit*>(&(*dev_tree_)->root));
            if (fused_dev_node != nullptr) {
              // fuse child1 and child2 in data tree to a new single node, modify coeff,
              // attr_map, split_map accordingly based on fused_dev_node
              // Fuse child1 and child2 in data tree to a new single node
              ICHECK(children_cnt >= 2)
                  << "InterError: Data nodes cannot fall below zero after fusing";
              PrimExpr new_extent = child1->extent * child2->extent;
              IterTreeSplit new_data_node = IterTreeSplit(new_extent, {});
              IterTreeSplit unit_data_node = IterTreeSplit(1, {});
              // Update coeff_map
              coeff_map_->erase(child1);
              coeff_map_->erase(child2);
              coeff_map_->insert({new_data_node, -1});
              coeff_map_->insert({unit_data_node, -1});
              // Update attr_map
              attr_map_->erase(device1->second);
              attr_map_->erase(device2->second);
              attr_map_->insert({*fused_dev_node, DeviceIterAttr::Split(new_extent)});
              // Update split_map
              split_map_->erase(child1);
              split_map_->erase(child2);
              split_map_->insert({new_data_node, *fused_dev_node});
              // Replace the old children with the new fused node
              // new_children.erase(new_children.begin() + i);
              new_children[i - 1] = new_data_node;
              new_children[i] = unit_data_node;
              children_cnt -= 1;
              changed = true;
            }
          }
        }
        if (changed) {
          auto* n = root.CopyOnWrite();
          n->children = new_children;
          return GetRef<IterTreeBase>(n);
        } else {
          return root;
        }
      }
    }
  }

 private:
  bool only_data_;  // if true, only data tree is defined;
                    // otherwise, both data&device trees are defined
  bool is_root_{true};
  DataIterTree::CoeffMap* coeff_map_;
  DeviceIterTree::AttrMap* attr_map_;
  TileLayout::SplitMap* split_map_;
  const DeviceIterTree* dev_tree_;
  arith::Analyzer ana_;  // Analyzer to simplify and check expressions
};

TileLayout FuseEquivIter(TileLayout layout) {
  const auto& data_tree = layout->data_tree;
  const auto& data_leaves = data_tree.GetLeaves();
  DataIterTree::CoeffMap coeff_map = data_tree.GetCoeffMap(data_leaves);
  if (!layout->device_tree.defined()) {
    auto new_data_root = EquivIterFuser::FuseEquivIter(data_tree->root, true, &coeff_map);
    ICHECK(new_data_root.defined()) << "InternalError: The data tree should be defined";
    return TileLayout::FromMaps(new_data_root.value(), NullOpt, coeff_map, {}, {});
  } else {
    // Both data tree and device tree are defined
    const auto& dev_tree = layout->device_tree.value();
    const auto& dev_leaves = dev_tree.GetLeaves();
    DeviceIterTree::AttrMap attr_map = dev_tree.GetAttrMap(dev_leaves);
    TileLayout::SplitMap split_map = layout.GetSplitMap(data_leaves, dev_leaves);
    auto new_data_root = EquivIterFuser::FuseEquivIter(data_tree->root, false, &coeff_map,
                                                       &attr_map, &split_map, &dev_tree);
    // auto new_dev_root = GetRef<IterTreeSplit>(dev_tree->root);
    ICHECK(new_data_root.defined() && dev_tree->root.defined())
        << "InternalError: The data tree and device tree should be both defined";
    return RemoveUnitIter(TileLayout::FromMaps(new_data_root.value(), dev_tree->root, coeff_map,
                                               attr_map, split_map, layout->from, layout->to));
  }
}

TileLayout NormalizeTileLayout(TileLayout layout) {
  TileLayout res = Downcast<TileLayout>(Deduplicate({layout})[0]);
  res = RemoveUnitIter(res);
  res = FuseEquivIter(res);
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
  // Get the stride in the inner layout
  arith::Analyzer analyzer;
  PrimExpr inner_stride = inner->GetCosize();
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
  auto attr_map = inner_n->device_tree.defined() ? inner_n->device_tree.value().GetAttrMap()
                                                 : tvm::tirx::DeviceIterTree::AttrMap();
  auto split_map = inner.GetSplitMap();
  return TileLayout::FromMaps(root,
                              inner_n->device_tree.defined() ? inner_n->device_tree.value()->root
                                                             : Optional<IterTreeSplit>(),
                              coeff_map, attr_map, split_map, inner_n->from, inner_n->to);
}

TVM_REGISTER_GLOBAL("tirx.TileLayoutTile").set_body_typed([](TileLayout outer, TileLayout inner) {
  return Tile(outer, inner);
});

/******** Tile - Check Inner Layout ********/
bool CompareSubStructureTrees(const IterTreeSplitNode* tiled_layout_node,
                              const IterTreeSplitNode* layout_node, arith::Analyzer& analyzer) {
  if (tiled_layout_node->children.size() != layout_node->children.size()) {
    return false;
  }

  for (size_t i = 0; i < tiled_layout_node->children.size(); i++) {
    const auto* tiled_layout_child = tiled_layout_node->children[i].as<IterTreeSplitNode>();
    const auto* layout_child = layout_node->children[i].as<IterTreeSplitNode>();

    if (!tiled_layout_child || !layout_child) {
      return false;
    }

    if (!analyzer.CanProveEqual(tiled_layout_child->extent, layout_child->extent)) {
      return false;
    }

    if (!CompareSubStructureTrees(tiled_layout_child, layout_child, analyzer)) {
      return false;
    }
  }

  return true;
}

bool CompareStride_Inner(const IterTreeSplitNode* tiled_layout_node,
                         const IterTreeSplitNode* layout_node,
                         const DataIterTree::CoeffMap& layout_coeff_map,
                         const DataIterTree::CoeffMap& tiled_layout_coeff_map,
                         arith::Analyzer& analyzer,
                         DeviceIterTree::AttrMap* layout_attr_map = nullptr,
                         DeviceIterTree::AttrMap* tiled_layout_attr_map = nullptr,
                         TileLayout::SplitMap* layout_split_map = nullptr,
                         TileLayout::SplitMap* tiled_layout_split_map = nullptr,
                         Optional<Array<IterTreeBase>> layout_dev_leaves = NullOpt,
                         Optional<Array<IterTreeBase>> tiled_layout_dev_leaves = NullOpt) {
  if (tiled_layout_node->children.size() != layout_node->children.size()) {
    return false;
  }
  if (tiled_layout_node->children.size() == 0) {
    auto tiled_layout_it = tiled_layout_coeff_map.find(GetRef<IterTreeSplit>(tiled_layout_node));
    auto layout_it = layout_coeff_map.find(GetRef<IterTreeSplit>(layout_node));
    if (tiled_layout_it == tiled_layout_coeff_map.end() || layout_it == layout_coeff_map.end()) {
      LOG(FATAL) << "InternalError: Coefficient not found for node";
    }
    // search tiled_layout_device and layout_device mapped by tiled_layout_node and
    // layout_node, respectively, from SplitMap and AttrMap and check two devices are essentially
    // the same device.
    if (layout_dev_leaves.defined() && tiled_layout_dev_leaves.defined()) {
      // device tree exists
      auto tiled_layout_device_it =
          tiled_layout_split_map->find(GetRef<IterTreeSplit>(tiled_layout_node));
      auto layout_device_it = layout_split_map->find(GetRef<IterTreeSplit>(layout_node));
      bool layout_found = tiled_layout_device_it != tiled_layout_split_map->end();
      bool tiled_layout_found = tiled_layout_device_it != tiled_layout_split_map->end();
      if (layout_found == tiled_layout_found && layout_found) {
        // check if it's the same device
        const auto& tiled_layout_device = tiled_layout_device_it->second;
        const auto& layout_device = layout_device_it->second;

        auto tiled_layout_attr_it = tiled_layout_attr_map->find(tiled_layout_device);
        auto layout_attr_it = layout_attr_map->find(layout_device);
        // 1. type check
        if (tiled_layout_attr_it->second->type != layout_attr_it->second->type) {
          return false;
        }
        // 2. extent check
        const auto& tiled_layout_device_extent =
            Downcast<IterTreeSplit>(tiled_layout_device)->extent;
        const auto& layout_device_extent = Downcast<IterTreeSplit>(layout_device)->extent;
        if (!analyzer.CanProveEqual(tiled_layout_device_extent, layout_device_extent)) {
          return false;
        }
        // 3. position check:
        // check if tiled_layout_device and layout_device are on the same index of
        // tiled_layout_dev_leaves and layout_dev_leaves, respectively
        // PS: since the device structure has been checked before, if device nodes have the same
        // index they are from the same position of each device tree
        for (size_t i = 0; i < tiled_layout_dev_leaves.value().size(); i++) {
          bool tiled_layout_dev_child_found =
              tiled_layout_dev_leaves.value()[i] == tiled_layout_device;
          bool layout_dev_child_found = layout_dev_leaves.value()[i] == layout_device;
          if (layout_dev_child_found != tiled_layout_dev_child_found) return false;
        }
      } else if (layout_found != tiled_layout_found) {
        // mismatch status
        return false;
      }
    }
    return analyzer.CanProveEqual(tiled_layout_it->second, layout_it->second);
  } else {
    for (size_t i = 0; i < tiled_layout_node->children.size(); i++) {
      const auto* tiled_layout_child = tiled_layout_node->children[i].as<IterTreeSplitNode>();
      const auto* layout_child = layout_node->children[i].as<IterTreeSplitNode>();
      if (!CompareStride_Inner(tiled_layout_child, layout_child, layout_coeff_map,
                               tiled_layout_coeff_map, analyzer, layout_attr_map,
                               tiled_layout_attr_map, layout_split_map, tiled_layout_split_map,
                               layout_dev_leaves, tiled_layout_dev_leaves)) {
        return false;
      }
    }
    return true;
  }
}

bool CompareExtentAndStride_Inner(const IterTreeSplitNode* tiled_layout_node,
                                  const IterTreeSplitNode* layout_node,
                                  const DataIterTree::CoeffMap& layout_coeff_map,
                                  const DataIterTree::CoeffMap& tiled_layout_coeff_map,
                                  arith::Analyzer& analyzer,
                                  DeviceIterTree::AttrMap* layout_attr_map = nullptr,
                                  DeviceIterTree::AttrMap* tiled_layout_attr_map = nullptr,
                                  TileLayout::SplitMap* layout_split_map = nullptr,
                                  TileLayout::SplitMap* tiled_layout_split_map = nullptr,
                                  Optional<Array<IterTreeBase>> layout_dev_leaves = NullOpt,
                                  Optional<Array<IterTreeBase>> tiled_layout_dev_leaves = NullOpt) {
  // Compare extents
  if (!analyzer.CanProveEqual(tiled_layout_node->extent, layout_node->extent)) {
    return false;
  }
  // Compare strides (coefficients)
  return CompareStride_Inner(tiled_layout_node, layout_node, layout_coeff_map,
                             tiled_layout_coeff_map, analyzer, layout_attr_map,
                             tiled_layout_attr_map, layout_split_map, tiled_layout_split_map,
                             layout_dev_leaves, tiled_layout_dev_leaves);
}

bool CheckDataSubtree_Inner(const IterTreeSplitNode* tiled_layout_node,
                            const IterTreeSplitNode* layout_node,
                            const DataIterTree::CoeffMap& layout_coeff_map,
                            const DataIterTree::CoeffMap& tiled_layout_coeff_map,
                            arith::Analyzer& analyzer,
                            DeviceIterTree::AttrMap* layout_attr_map = nullptr,
                            DeviceIterTree::AttrMap* tiled_layout_attr_map = nullptr,
                            TileLayout::SplitMap* layout_split_map = nullptr,
                            TileLayout::SplitMap* tiled_layout_split_map = nullptr,
                            Optional<Array<IterTreeBase>> layout_dev_leaves = NullOpt,
                            Optional<Array<IterTreeBase>> tiled_layout_dev_leaves = NullOpt) {
  if (tiled_layout_node->children.size() != layout_node->children.size()) {
    return false;
  }
  for (size_t i = 0; i < layout_node->children.size(); i++) {
    const auto* tiled_layout_child_root = tiled_layout_node->children[i].as<IterTreeSplitNode>();
    size_t sub_children_cnt = tiled_layout_child_root->children.size();
    if (sub_children_cnt < 2) return false;
    const auto* tiled_layout_child =
        tiled_layout_child_root->children[sub_children_cnt - 1].as<IterTreeSplitNode>();
    const auto* layout_child = layout_node->children[i].as<IterTreeSplitNode>();
    if (!tiled_layout_child || !layout_child ||
        !CompareExtentAndStride_Inner(
            tiled_layout_child, layout_child, layout_coeff_map, tiled_layout_coeff_map, analyzer,
            layout_attr_map, tiled_layout_attr_map, layout_split_map, tiled_layout_split_map,
            layout_dev_leaves, tiled_layout_dev_leaves)) {
      return false;
    }
  }
  return true;
}

bool CheckTileDataSubtree_Inner(const IterTreeBase& tiled_layout_node,
                                const IterTreeBase& layout_node,
                                const DataIterTree::CoeffMap& layout_coeff_map,
                                const DataIterTree::CoeffMap& tiled_layout_coeff_map,
                                arith::Analyzer& analyzer,
                                DeviceIterTree::AttrMap* layout_attr_map = nullptr,
                                DeviceIterTree::AttrMap* tiled_layout_attr_map = nullptr,
                                TileLayout::SplitMap* layout_split_map = nullptr,
                                TileLayout::SplitMap* tiled_layout_split_map = nullptr,
                                Optional<Array<IterTreeBase>> layout_dev_leaves = NullOpt,
                                Optional<Array<IterTreeBase>> tiled_layout_dev_leaves = NullOpt) {
  if (const auto* tiled_layout_split = tiled_layout_node.as<IterTreeSplitNode>()) {
    if (const auto* layout_split = layout_node.as<IterTreeSplitNode>()) {
      if (CheckDataSubtree_Inner(tiled_layout_split, layout_split, layout_coeff_map,
                                 tiled_layout_coeff_map, analyzer, layout_attr_map,
                                 tiled_layout_attr_map, layout_split_map, tiled_layout_split_map,
                                 layout_dev_leaves, tiled_layout_dev_leaves)) {
        return true;
      }
      for (const auto& child : tiled_layout_split->children) {
        if (CheckTileDataSubtree_Inner(child, layout_node, layout_coeff_map, tiled_layout_coeff_map,
                                       analyzer, layout_attr_map, tiled_layout_attr_map,
                                       layout_split_map, tiled_layout_split_map, layout_dev_leaves,
                                       tiled_layout_dev_leaves)) {
          return true;
        }
      }
    }
  }
  return false;
}

std::vector<int> PrimeFactorize(int n) {
  std::vector<int> factors;

  while (n % 2 == 0) {
    factors.push_back(2);
    n /= 2;
  }

  int sqrt_n = static_cast<int>(std::sqrt(n));  // Cast to int to ensure integer comparison

  for (int i = 3; i <= sqrt_n; i += 2) {
    while (n % i == 0) {
      factors.push_back(i);
      n /= i;
    }
  }

  if (n > 2) {
    factors.push_back(n);
  }

  return factors;
}

void partitionHelper(const std::vector<int>& factors, size_t n, size_t index,
                     std::vector<std::vector<int>>& current,
                     std::vector<std::vector<int>>& result) {
  if (index == factors.size()) {
    std::vector<int> partitionProducts;
    for (const auto& bucket : current) {
      int product = 1;
      for (const auto& factor : bucket) {
        product *= factor;
      }
      partitionProducts.push_back(product);
    }
    result.push_back(partitionProducts);
    return;
  }

  for (size_t i = 0; i < n; ++i) {
    current[i].push_back(factors[index]);
    partitionHelper(factors, n, index + 1, current, result);
    current[i].pop_back();
  }
}

std::vector<std::vector<int>> GetFactorPartitions(const std::vector<int>& factors, size_t n) {
  std::vector<std::vector<int>> result;
  std::vector<std::vector<int>> current(n);
  partitionHelper(factors, n, 0, current, result);
  return result;
}

void generate_strides(const std::vector<PrimExpr>& extent, PrimExpr init_stride_expr,
                      int current_index, std::vector<PrimExpr>& current_stride,
                      std::vector<bool>& used, std::vector<std::vector<PrimExpr>>& result) {
  if (std::all_of(used.begin(), used.end(), [](bool v) { return v; })) {
    result.push_back(current_stride);
    return;
  }

  // Calculate the product for the next stride
  PrimExpr product = current_stride[current_index] * extent[current_index];

  for (size_t i = 0; i <= extent.size() - 1; ++i) {
    if (!used[i]) {
      used[i] = true;
      current_stride[i] = product;
      generate_strides(extent, init_stride_expr, i, current_stride, used, result);
      used[i] = false;
    }
  }
}

std::vector<std::vector<PrimExpr>> generate_all_strides(const std::vector<PrimExpr>& extent,
                                                        PrimExpr init_stride_expr) {
  int n = extent.size();
  std::vector<std::vector<PrimExpr>> result;
  std::vector<PrimExpr> current_stride(n);
  std::vector<bool> used(n, false);
  for (size_t i = 0; i <= extent.size() - 1; ++i) {
    used[i] = true;
    current_stride[i] = init_stride_expr;
    generate_strides(extent, init_stride_expr, i, current_stride, used, result);
    used[i] = false;
  }
  return result;
}

bool Factorization_Inner(TileLayout tiled_layout, TileLayout inner, arith::Analyzer& analyzer,
                         size_t inner_children_size = 0, size_t tile_children_size = 0,
                         size_t unit_inner_size = 0) {
  const auto* tiled_layout_node = tiled_layout->data_tree->root.as<IterTreeSplitNode>();
  const auto* inner_node = inner->data_tree->root.as<IterTreeSplitNode>();
  PrimExpr cosize_inner = inner->GetCosize();
  auto tiled_extent_expr = tiled_layout_node->extent;
  auto inner_product_expr = inner_node->extent;

  if (tile_children_size == inner_children_size) {
    // fast check
    std::vector<PrimExpr> new_extents;
    std::vector<IterTreeBase> new_children;
    DataIterTree::CoeffMap new_coeff_map;
    auto tilelayout_coeff_map = tiled_layout->data_tree.GetCoeffMap();
    for (size_t i = 0; i < tiled_layout_node->children.size(); ++i) {
      auto* tile_child = tiled_layout_node->children[i].as<IterTreeSplitNode>();
      auto tile_child_root = GetRef<IterTreeSplit>(tile_child);
      auto* inner_child = inner_node->children[i].as<IterTreeSplitNode>();
      auto inner_child_root = GetRef<IterTreeSplit>(inner_child);
      PrimExpr inner_child_extent = inner_child_root->extent;
      PrimExpr tile_child_extent = tile_child_root->extent;
      PrimExpr factor_extent_expr = tile_child_extent / inner_child_extent;
      const int64_t* factor = as_const_int(factor_extent_expr);
      if (!factor) {
        return false;
      }
      auto new_tile_child = IterTreeSplit(factor_extent_expr, {});
      new_children.push_back(new_tile_child);
      new_extents.push_back(factor_extent_expr);
    }
    PrimExpr init_stride_expr = make_const(tiled_extent_expr->dtype, 1);  // must be continuous
    std::vector<std::vector<PrimExpr>> all_strides =
        generate_all_strides(new_extents, init_stride_expr);
    for (const auto& one_strides : all_strides) {
      PrimExpr combined_extent = 1;
      for (const auto& extent : new_extents) {
        combined_extent = combined_extent * extent;
      }
      auto root = IterTreeSplit(combined_extent, new_children);
      DataIterTree::CoeffMap new_coeff_map;
      for (size_t i = 0; i < new_children.size(); ++i) {
        new_coeff_map[new_children[i]] = one_strides[i];
      }
      DataIterTree new_data_tree(root, Array<PrimExpr>(one_strides.begin(), one_strides.end()));
      Optional<DeviceIterTree> device_tree = NullOpt;
      TileLayout new_outer =
          TileLayout::FromMaps(root, NullOpt, new_coeff_map, {}, {}, NullOpt, NullOpt);
      TileLayout new_tile_layout = Tile(new_outer, inner);
      new_tile_layout = NormalizeTileLayout(new_tile_layout);
      if (StructuralEqual()(new_tile_layout, tiled_layout)) {
        return true;
      }
    }
  } else {
    PrimExpr factor_expr = tiled_extent_expr / inner_product_expr;
    const int64_t* factor = as_const_int(factor_expr);
    if (!factor) {
      return false;
    }
    auto factors = PrimeFactorize(*factor);
    auto partitions = GetFactorPartitions(factors, inner_node->children.size());
    PrimExpr init_stride_expr = make_const(tiled_extent_expr->dtype, 1);  // must be continuous
    // assume outer must be continuous
    for (const auto& partition : partitions) {
      std::vector<PrimExpr> new_extents;
      for (const auto& part : partition) {
        new_extents.push_back(part);
      }
      std::vector<std::vector<PrimExpr>> all_strides =
          generate_all_strides(new_extents, init_stride_expr);
      for (const auto& one_strides : all_strides) {
        // reconstruct a layout based on one_strides and new_extents using
        // Create a vector to hold the new children nodes
        std::vector<IterTreeBase> new_children;
        for (size_t i = 0; i < new_extents.size(); ++i) {
          auto child = IterTreeSplit(new_extents[i], {});
          new_children.push_back(child);
        }
        PrimExpr combined_extent = 1;
        for (const auto& extent : new_extents) {
          combined_extent = combined_extent * extent;
        }
        auto root = IterTreeSplit(combined_extent, new_children);
        DataIterTree::CoeffMap new_coeff_map;
        for (size_t i = 0; i < new_children.size(); ++i) {
          new_coeff_map[new_children[i]] = one_strides[i];
        }
        DataIterTree new_data_tree(root, Array<PrimExpr>(one_strides.begin(), one_strides.end()));
        Optional<DeviceIterTree> device_tree = NullOpt;  // Adjust as necessary
        TileLayout new_outer =
            TileLayout::FromMaps(root, NullOpt, new_coeff_map, {}, {}, NullOpt, NullOpt);
        TileLayout new_tile_layout = Tile(new_outer, inner);
        new_tile_layout = NormalizeTileLayout(new_tile_layout);
        if (StructuralEqual()(new_tile_layout, tiled_layout)) {
          return true;
        }
      }
    }
  }
  return false;
}

bool CheckTileDataSubtree_Inner_Extended(TileLayout tiled_layout, TileLayout inner,
                                         arith::Analyzer& analyzer) {
  TileLayout inner_removed = RemoveUnitIter(inner);
  TileLayout tiled_layout_norm = NormalizeTileLayout(tiled_layout);
  const IterTreeBase& tiled_layout_norm_node = tiled_layout_norm->data_tree->root;
  const IterTreeBase& inner_node = inner->data_tree->root;
  const IterTreeBase& inner_removed_node = inner_removed->data_tree->root;
  if (const auto* tiled_layout_norm_node_split = tiled_layout_norm_node.as<IterTreeSplitNode>()) {
    if (const auto* inner_node_split = inner_node.as<IterTreeSplitNode>()) {
      if (const auto* inner_removed_node_split = inner_removed_node.as<IterTreeSplitNode>()) {
        size_t tile_children_size = tiled_layout_norm_node_split->children.size();
        size_t inner_children_size = inner_node_split->children.size();
        size_t unit_inner_size = inner_children_size - inner_removed_node_split->children.size();
        if (tile_children_size == 1 ||
            (tile_children_size >= inner_children_size - unit_inner_size &&
             tile_children_size <= inner_children_size)) {
          return Factorization_Inner(tiled_layout_norm, inner, analyzer, inner_children_size,
                                     tile_children_size, unit_inner_size);
        } else {
          return false;
        }
      }
    }
  }
  return false;
}

bool IsTileLayout_Inner(TileLayout tiled_layout, TileLayout layout) {
  auto* layout_n = layout.operator->();
  auto* tiled_layout_n = tiled_layout.operator->();
  auto layout_coeff_map = layout_n->data_tree.GetCoeffMap();
  auto tiled_layout_coeff_map = tiled_layout_n->data_tree.GetCoeffMap();
  arith::Analyzer analyzer;
  if (!tiled_layout->device_tree.defined() && !layout->device_tree.defined()) {
    // no device tree defined
    if (CheckTileDataSubtree_Inner(tiled_layout->data_tree->root, layout->data_tree->root,
                                   layout_coeff_map, tiled_layout_coeff_map, analyzer)) {
      return true;
    }
    return CheckTileDataSubtree_Inner_Extended(tiled_layout, layout, analyzer);
  } else if (tiled_layout->device_tree.defined() && layout->device_tree.defined()) {
    // device tree defined
    // 1. check device tree structure
    const auto& layout_dev_tree = layout->device_tree.value();
    const auto& layout_device_root = layout_dev_tree->root;
    const auto& layout_dev_leaves = layout_dev_tree.GetLeaves();
    const auto& layout_data_tree = layout->data_tree;
    const auto& layout_data_leaves = layout_data_tree.GetLeaves();
    const auto& tiled_layout_dev_tree = tiled_layout->device_tree.value();
    const auto& tiled_layout_device_root = tiled_layout_dev_tree->root;
    const auto& tiled_layout_dev_leaves = tiled_layout_dev_tree.GetLeaves();
    const auto& tiled_layout_data_tree = tiled_layout->data_tree;
    const auto& tiled_layout_data_leaves = tiled_layout_data_tree.GetLeaves();
    if (!CompareSubStructureTrees(tiled_layout_device_root.operator->(),
                                  layout_device_root.operator->(), analyzer)) {
      return false;
    }
    // 2. check data tree structure and mapping
    auto layout_split_map = layout.GetSplitMap(layout_data_leaves, layout_dev_leaves);
    auto tiled_layout_split_map =
        tiled_layout.GetSplitMap(tiled_layout_data_leaves, tiled_layout_dev_leaves);
    DeviceIterTree::AttrMap layout_attr_map = layout_dev_tree.GetAttrMap(layout_dev_leaves);
    DeviceIterTree::AttrMap tiled_layout_attr_map =
        tiled_layout_dev_tree.GetAttrMap(tiled_layout_dev_leaves);
    if (CheckTileDataSubtree_Inner(tiled_layout->data_tree->root, layout->data_tree->root,
                                   layout_coeff_map, tiled_layout_coeff_map, analyzer,
                                   &layout_attr_map, &tiled_layout_attr_map, &layout_split_map,
                                   &tiled_layout_split_map, layout_dev_leaves,
                                   tiled_layout_dev_leaves)) {
      return true;
    }
    return CheckTileDataSubtree_Inner_Extended(tiled_layout, layout, analyzer);
  } else {
    // device tree status mismatch
    return false;
  }
}

TVM_REGISTER_GLOBAL("tirx.IsTileLayout_Inner")
    .set_body_typed([](TileLayout tiled_layout, TileLayout layout) {
      return IsTileLayout_Inner(tiled_layout, layout);
    });

/******** Tile - Check Outer Layout ********/
bool CompareStride_Outer(const IterTreeSplitNode* tiled_layout_node,
                         const IterTreeSplitNode* layout_node,
                         const DataIterTree::CoeffMap& layout_coeff_map,
                         const DataIterTree::CoeffMap& tiled_layout_coeff_map,
                         PrimExpr cumulative_cosize, arith::Analyzer& analyzer,
                         TileLayout::SplitMap* tiled_layout_split_map = nullptr) {
  if (tiled_layout_node->children.size() != layout_node->children.size()) {
    return false;
  }

  if (tiled_layout_node->children.size() == 0) {
    auto tiled_layout_it = tiled_layout_coeff_map.find(GetRef<IterTreeSplit>(tiled_layout_node));
    auto layout_it = layout_coeff_map.find(GetRef<IterTreeSplit>(layout_node));
    if (tiled_layout_it == tiled_layout_coeff_map.end() || layout_it == layout_coeff_map.end()) {
      LOG(FATAL) << "InternalError: Coefficient not found for node";
    }
    return analyzer.CanProveEqual(tiled_layout_it->second / cumulative_cosize, layout_it->second);
  } else {
    for (size_t i = 0; i < tiled_layout_node->children.size(); i++) {
      const auto* tiled_layout_child = tiled_layout_node->children[i].as<IterTreeSplitNode>();
      const auto* layout_child = layout_node->children[i].as<IterTreeSplitNode>();
      if (!CompareStride_Outer(tiled_layout_child, layout_child, layout_coeff_map,
                               tiled_layout_coeff_map, cumulative_cosize, analyzer)) {
        return false;
      }
    }
    return true;
  }
}

bool CompareExtent_Outer(const IterTreeSplitNode* tiled_layout_node,
                         const IterTreeSplitNode* layout_node, arith::Analyzer& analyzer) {
  // Compare extents
  if (!analyzer.CanProveEqual(tiled_layout_node->extent, layout_node->extent)) {
    return false;
  }
  return true;
}

bool CheckDataSubtree_Outer(const IterTreeSplitNode* tiled_layout_node,
                            const IterTreeSplitNode* layout_node,
                            const DataIterTree::CoeffMap& layout_coeff_map,
                            const DataIterTree::CoeffMap& tiled_layout_coeff_map,
                            arith::Analyzer& analyzer,
                            TileLayout::SplitMap* tiled_layout_split_map = nullptr) {
  if (tiled_layout_node->children.size() != layout_node->children.size()) {
    return false;
  }
  PrimExpr cumulative_cosize = 1;
  for (size_t i = 0; i < layout_node->children.size(); i++) {
    const auto* tiled_layout_child_root = tiled_layout_node->children[i].as<IterTreeSplitNode>();
    size_t sub_children_cnt = tiled_layout_child_root->children.size();
    if (sub_children_cnt < 2) return false;
    const auto* last_tiled_layout_child =
        tiled_layout_child_root->children[sub_children_cnt - 1].as<IterTreeSplitNode>();
    // only the first sub-child will belong to the original outer layout
    const auto* tiled_layout_child = tiled_layout_child_root->children[0].as<IterTreeSplitNode>();
    const auto* layout_child = layout_node->children[i].as<IterTreeSplitNode>();
    if (!tiled_layout_child || !layout_child ||
        !CompareExtent_Outer(tiled_layout_child, layout_child, analyzer)) {
      return false;
    }

    // Retrieve all child leaves of last_tiled_layout_child and calculate cosize for each leaf
    auto last_child_leaves = GetRef<IterTreeSplit>(last_tiled_layout_child).GetLeaves();
    for (const auto& leaf : last_child_leaves) {
      if (tiled_layout_split_map &&
          tiled_layout_split_map->find(leaf) != tiled_layout_split_map->end()) {
        continue;
      }
      auto coeff_it = tiled_layout_coeff_map.find(Downcast<IterTreeSplit>(leaf));
      if (coeff_it != tiled_layout_coeff_map.end()) {
        cumulative_cosize += coeff_it->second * (Downcast<IterTreeSplit>(leaf)->extent - 1);
      }
    }
  }

  for (size_t i = 0; i < layout_node->children.size(); i++) {
    const auto* tiled_layout_child_root = tiled_layout_node->children[i].as<IterTreeSplitNode>();
    const auto* tiled_layout_child = tiled_layout_child_root->children[0].as<IterTreeSplitNode>();
    const auto* layout_child = layout_node->children[i].as<IterTreeSplitNode>();
    if (!CompareStride_Outer(tiled_layout_child, layout_child, layout_coeff_map,
                             tiled_layout_coeff_map, cumulative_cosize, analyzer,
                             tiled_layout_split_map)) {
      return false;
    }
  }
  return true;
}

bool CheckTileDataSubtree_Outer(const IterTreeBase& tiled_layout_node,
                                const IterTreeBase& layout_node,
                                const DataIterTree::CoeffMap& layout_coeff_map,
                                const DataIterTree::CoeffMap& tiled_layout_coeff_map,
                                arith::Analyzer& analyzer,
                                TileLayout::SplitMap* tiled_layout_split_map = nullptr) {
  if (const auto* tiled_layout_split = tiled_layout_node.as<IterTreeSplitNode>()) {
    if (const auto* layout_split = layout_node.as<IterTreeSplitNode>()) {
      if (CheckDataSubtree_Outer(tiled_layout_split, layout_split, layout_coeff_map,
                                 tiled_layout_coeff_map, analyzer, tiled_layout_split_map)) {
        return true;
      }
      for (const auto& child : tiled_layout_split->children) {
        if (CheckTileDataSubtree_Outer(child, layout_node, layout_coeff_map, tiled_layout_coeff_map,
                                       analyzer, tiled_layout_split_map)) {
          return true;
        }
      }
    }
  }
  return false;
}

bool IsTileLayout_Outer(TileLayout tiled_layout, TileLayout layout) {
  auto* layout_n = layout.operator->();
  auto* tiled_layout_n = tiled_layout.operator->();
  auto layout_coeff_map = layout_n->data_tree.GetCoeffMap();
  auto tiled_layout_coeff_map = tiled_layout_n->data_tree.GetCoeffMap();
  arith::Analyzer analyzer;
  if (!tiled_layout->device_tree.defined() && !layout->device_tree.defined()) {
    return CheckTileDataSubtree_Outer(tiled_layout_n->data_tree->root, layout_n->data_tree->root,
                                      layout_coeff_map, tiled_layout_coeff_map, analyzer);
  } else if (tiled_layout->device_tree.defined() && !layout->device_tree.defined()) {
    const auto& tiled_layout_dev_tree = tiled_layout->device_tree.value();
    const auto& tiled_layout_dev_leaves = tiled_layout_dev_tree.GetLeaves();
    const auto& tiled_layout_data_tree = tiled_layout->data_tree;
    const auto& tiled_layout_data_leaves = tiled_layout_data_tree.GetLeaves();
    auto tiled_layout_split_map =
        tiled_layout.GetSplitMap(tiled_layout_data_leaves, tiled_layout_dev_leaves);
    return CheckTileDataSubtree_Outer(tiled_layout->data_tree->root, layout->data_tree->root,
                                      layout_coeff_map, tiled_layout_coeff_map, analyzer,
                                      &tiled_layout_split_map);

  } else {
    // outer must not have device tree
    return false;
  }
}

TVM_REGISTER_GLOBAL("tirx.IsTileLayout_Outer")
    .set_body_typed([](TileLayout tiled_layout, TileLayout layout) {
      return IsTileLayout_Outer(tiled_layout, layout);
    });

/******* Vector Length ********/
PrimExpr ComputeGCD(PrimExpr a, PrimExpr b, arith::Analyzer* ana) {
  while (!ana->CanProveEqual(b, 0)) {
    PrimExpr temp = b;
    b = floormod(a, b);
    a = temp;
  }
  return a;
}

std::pair<int, int> GetMajorDimension(const Array<PrimExpr>& strides) {
  arith::Analyzer analyzer;
  int major_dim = 0;
  PrimExpr min_stride = strides[0];
  PrimExpr gcd = strides[0];
  PrimExpr argmin_stride = strides[0];
  for (size_t i = 0; i < strides.size(); ++i) {
    gcd = ComputeGCD(gcd, strides[i], &analyzer);
    if (as_const_int(strides[i])[0] < as_const_int(min_stride)[0]) {
      major_dim = static_cast<int>(i);
      min_stride = strides[i];
    }
  }
  int min_stride_int = as_const_int(min_stride) ? static_cast<int>(*as_const_int(min_stride)) : 0;
  return {(strides.size() - major_dim) - 1, min_stride_int};
}

std::tuple<int, int, PrimExpr> ExtractLayout(TileLayout layout) {
  arith::Analyzer analyzer;
  // Get the data dimensions and strides from the layout
  auto data_tree = layout->data_tree;
  auto data_leaves = data_tree.GetLeaves();
  Array<PrimExpr> strides = data_tree->coeff;

  // Get the major dimension (dimension with the largest stride)
  auto [major_dim, min_stride] = GetMajorDimension(strides);

  // Create new strides for the transposed layout
  std::vector<PrimExpr> transposed_strides(strides.size());
  transposed_strides[strides.size() - 1] = 1;  // innermost stride is min_stride
  for (int i = (int)strides.size() - 2; i >= 0; --i) {
    transposed_strides[i] =
        transposed_strides[i + 1] * Downcast<IterTreeSplit>(data_leaves[i + 1])->extent;
  }

  return {major_dim, min_stride,
          Downcast<IterTreeSplit>(data_leaves[strides.size() - major_dim - 1])->extent};
}

int Vec_Len(TileLayout Ls, TileLayout Ld) {
  // Constants initialization
  arith::Analyzer analyzer;
  std::vector<int> optimal_vec_len = {1, 2, 4, 8};
  auto Ls_node = Ls->data_tree->root;
  auto Ld_node = Ld->data_tree->root;
  PrimExpr Ls_extent = Ls_node->extent;
  PrimExpr Ld_extent = Ld_node->extent;

  // Remove unit dimensions and extract strides
  TileLayout Lsn = RemoveUnitIter(Ls);
  TileLayout Ldn = RemoveUnitIter(Ld);
  auto [major_s, min_stride_s, argmin_s] = ExtractLayout(Lsn);
  auto [major_d, min_stride_d, argmin_d] = ExtractLayout(Ldn);

  // Check if normalized layouts are structurally equal
  if (analyzer.CanProveEqual(Ls_extent, Ld_extent)) {
    if ((min_stride_s != 1) || (min_stride_d != 1) || (major_s != major_d)) {
      // Case 1 & 2: Layouts are not contiguous or min strides don't match; different major-dim
      return 1;
    } else {
      // Case 3: the same dim-major, and the same stride, return major dimension in terms of
      // optimal vector length based on extent
      return *std::lower_bound(
          optimal_vec_len.begin(), optimal_vec_len.end() - 1,
          static_cast<int>(as_const_int(ComputeGCD(argmin_s, argmin_d, &analyzer))[0]));
    }
  } else {
    // Case4: Structure doesn't match, then return the minimum of GCD stride, normally 1
    LOG(FATAL) << "Source and Destination extents don't match!";
  }
}

TVM_REGISTER_GLOBAL("tirx.Vec_Len").set_body_typed([](TileLayout tiled_layout, TileLayout layout) {
  return Vec_Len(tiled_layout, layout);
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
    ICHECK(inner->to.value().Is((from)))
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

/**************** SwizzleLayout ****************/
SwizzleLayout::SwizzleLayout(int per_element, int swizzle_len, int atom_len, bool swizzle_inner) {
  auto n = make_object<SwizzleLayoutNode>();
  n->per_element = per_element;
  n->swizzle_len = swizzle_len;
  n->atom_len = atom_len;
  n->swizzle_inner = swizzle_inner;
  ICHECK(n->VerifyWellFormed()) << "ValueError: The swizzle layout is not well-formed";
  int swizzle_mask = (1 << swizzle_len) - 1;
  n->inner_mask = swizzle_mask;
  n->outer_mask = swizzle_mask << atom_len;
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(SwizzleLayoutNode);

TVM_REGISTER_GLOBAL("tirx.SwizzleLayout")
    .set_body_typed([](int per_element, int swizzle_len, int atom_len, bool swizzle_inner) {
      return SwizzleLayout(per_element, swizzle_len, atom_len, swizzle_inner);
    });

Array<PrimExpr> SwizzleLayoutNode::GetDefaultShape() const { return {GetSize()}; }

bool SwizzleLayoutNode::CompatibleWithShape(const Array<PrimExpr>& shape) const {
  arith::Analyzer analyzer;
  return analyzer.CanProveEqual(FloorMod(ReduceMul(shape), GetSize()), 0);
}

bool SwizzleLayoutNode::VerifyWellFormed() const {
  return per_element >= 0 && swizzle_len >= 0 && atom_len >= swizzle_len;
}

PrimExpr SwizzleLayoutNode::GetSize() const { return 1 << (per_element + swizzle_len + atom_len); }

PrimExpr SwizzleLayoutNode::GetCosize() const { return GetSize(); }

PrimExpr SwizzleLayoutNode::Apply(const Array<PrimExpr>& coord) const {
  ICHECK_EQ(coord.size(), 1) << "ValueError: The number of coordinates must be 1";
  PrimExpr input = coord[0];
  auto f = [&](const PrimExpr& x) -> PrimExpr {
    if (swizzle_inner) {
      return x ^ ((x & outer_mask) >> atom_len);
    } else {
      return x ^ ((x & inner_mask) << atom_len);
    }
  };
  auto base = 1 << per_element;
  arith::Analyzer analyzer;
  // It takes more arithmetic operations to compute the result, but it is more friendly to the
  // vectorization
  return analyzer.Simplify((f(floordiv(input, base)) << per_element) + floormod(input, base));
}

}  // namespace tirxx

}  // namespace tvm