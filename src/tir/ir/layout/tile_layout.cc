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

/**************** TileLayout ****************/
TileLayout::TileLayout(Array<DataIterAttr> data_iter_array, Array<DeviceIterAttr> device_iter_array,
                       Optional<ExecScope> from, Optional<ExecScope> to) {
  auto n = make_object<TileLayoutNode>();
  n->data_iter_array = std::move(data_iter_array);
  n->device_iter_array = std::move(device_iter_array);
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

TVM_REGISTER_GLOBAL("tir.TileLayout")
    .set_body_typed([](Array<DataIterAttr> data_iter_array, Array<DeviceIterAttr> device_iter_array,
                       Optional<ExecScope> from, Optional<ExecScope> to) {
      return TileLayout(data_iter_array, device_iter_array, from, to);
    });

TileLayout::SplitMap TileLayout::GetSplitMap() const {
  auto* n = operator->();
  if (n->device_iter_array.empty()) {
    // No device array is defined, return an empty map
    return {};
  }
  SplitMap split_map;
  for (size_t i = 0; i < n->device_iter_array.size(); i++) {
    auto attr = n->device_iter_array[i];
    if (attr.IsSplit()) {
      size_t bound = attr.GetIntBound();
      ICHECK(bound < n->data_iter_array.size())
          << "ValueError: The bound of the split attribute is out of range";
      auto it = split_map.find(bound);
      ICHECK(it == split_map.end()) << "ValueError: Duplicate split bound for the same data iter";
      split_map[bound] = i;
    }
  }
  return std::move(split_map);
}

PrimExpr TileLayoutNode::GetSize() const {
  PrimExpr result = 1;
  for (size_t i = 0; i < data_iter_array.size(); i++) {
    result = result * data_iter_array[i]->extent;
  }
  return result;
}

PrimExpr TileLayoutNode::GetCosize() const {
  auto split_map = GetRef<TileLayout>(this).GetSplitMap();
  arith::Analyzer analyzer;
  PrimExpr cosize = 1;
  for (size_t i = 0; i < data_iter_array.size(); ++i) {
    if (split_map.find(i) != split_map.end()) {
      continue;
    }
    // Check if the coefficient is non-negative
    ICHECK(analyzer.CanProveGreaterEqual(data_iter_array[i]->stride, 0))
        << "ValueError: GetCosize cannot handle case where the coefficient of a non-bound leaf is "
           "negative";
    cosize += data_iter_array[i]->stride * (data_iter_array[i]->extent - 1);
  }
  return cosize;
}

bool TileLayoutNode::VerifyWellFormed() const {
  arith::Analyzer analyzer;
  // Verify the split map
  const auto& split_map = GetRef<TileLayout>(this).GetSplitMap();
  bool result = true;
  for (const auto& kv : split_map) {
    // The extents of the data and device iter array must be the same
    result &= analyzer.CanProveEqual(data_iter_array[kv.first]->extent,
                                     device_iter_array[kv.second]->extent);
  }
  return result;
}

Array<PrimExpr> TileLayoutNode::Apply(const PrimExpr& coord) const {
  arith::Analyzer analyzer;
  PrimExpr input = analyzer.Simplify(coord);
  const auto& size = GetSize();
  const auto& cosize = GetCosize();
  auto inner = analyzer.Simplify(floormod(input, size));
  const auto& outer = analyzer.Simplify(floordiv(input, size));

  auto split_map = GetRef<TileLayout>(this).GetSplitMap();
  PrimExpr result = 0;
  for (int i = static_cast<int>(data_iter_array.size()) - 1; i >= 0; i--) {
    if (split_map.find(i) != split_map.end()) {
      inner = floordiv(inner, data_iter_array[i]->extent);
      continue;
    }
    result += data_iter_array[i]->stride * floormod(inner, data_iter_array[i]->extent);
    inner = floordiv(inner, data_iter_array[i]->extent);
  }
  return {analyzer.Simplify(outer * cosize + result)};
}

/******* Vector Length ********/

PrimExpr ComputeGCD(PrimExpr a, PrimExpr b, arith::Analyzer* ana) {
  int max_iter = 100;
  int iter = 0;
  while (!ana->CanProveEqual(b, 0) && iter++ < max_iter) {
    PrimExpr temp = b;
    b = floormod(a, b);
    a = temp;
  }
  if (iter == max_iter) {
    return 1;
  }
  return a;
}

std::pair<int, int> GetMajorDimension(const Array<PrimExpr>& strides) {
  arith::Analyzer analyzer;
  int major_dim = 0;
  PrimExpr min_stride = strides[0];
  PrimExpr argmin_stride = strides[0];
  for (size_t i = 0; i < strides.size(); ++i) {
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
  Array<PrimExpr> strides;
  for (auto& iter : layout->data_iter_array) {
    strides.push_back(iter->stride);
  }

  // Get the major dimension (dimension with the largest stride)
  auto [major_dim, min_stride] = GetMajorDimension(strides);
  return {major_dim, min_stride, layout->data_iter_array[strides.size() - major_dim - 1]->extent};
}

int Vec_Len(TileLayout Ls, TileLayout Ld) {
  // Constants initialization
  arith::Analyzer analyzer;
  std::vector<int> optimal_vec_len = {1, 2, 4, 8};
  Ls = Ls->Normalize().as<TileLayout>().value();
  Ld = Ld->Normalize().as<TileLayout>().value();

  PrimExpr Ls_size = Ls->GetSize();
  PrimExpr Ld_size = Ld->GetSize();
  auto [major_s, min_stride_s, argmin_s] = ExtractLayout(Ls);
  auto [major_d, min_stride_d, argmin_d] = ExtractLayout(Ld);

  // Check if normalized layouts are structurally equal
  if (analyzer.CanProveEqual(Ls_size, Ld_size)) {
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

TVM_REGISTER_GLOBAL("tir.Vec_Len").set_body_typed([](TileLayout tiled_layout, TileLayout layout) {
  return Vec_Len(tiled_layout, layout);
});

/******** Shard ********/
std::vector<DeviceIterAttr> ParseStrategy(String strategy, Array<PrimExpr> mesh) {
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
  size_t mesh_dim = 0;
  for (; cur < strategy.size() && mesh_dim < mesh.size();) {
    if (strategy.at(cur) == 'S') {
      attrs.push_back(DeviceIterAttr::Split(mesh[mesh_dim++], f_parse_number_with_bracket()));
    } else if (strategy.at(cur) == 'E') {
      attrs.push_back(DeviceIterAttr::Exclusive(mesh[mesh_dim++], f_parse_number_with_bracket()));
    } else if (strategy.at(cur) == 'R') {
      attrs.push_back(DeviceIterAttr::Replicate(mesh[mesh_dim++]));
      ++cur;
    } else {
      LOG(FATAL) << "ValueError: Invalid strategy " << strategy;
    }
  }
  return std::move(attrs);
}

TileLayout Shard(Array<PrimExpr> shape, Array<PrimExpr> mesh, String strategy, TileLayout inner,
                 ExecScope from, ExecScope to) {
  // Parse the strategy
  auto attrs = ParseStrategy(strategy, mesh);
  ICHECK_EQ(attrs.size(), mesh.size())
      << "ValueError: The number of attributes must be the same as the number of mesh leaves";
  Array<PrimExpr> inner_shape(shape);
  for (int i = 0; i < static_cast<int>(mesh.size()); i++) {
    if (attrs[i].IsSplit()) {
      auto bound = attrs[i].GetIntBound();
      inner_shape.Set(bound, floordiv(inner_shape[bound], mesh[i]));
    }
  }
  auto [grouped_inner, inner_seps] = TileLayoutGroupByLogicalShape(inner, inner_shape, nullptr);
  inner = grouped_inner;
  ICHECK(!inner_seps.empty())
      << "ValueError: The inner layout must be able to transform from logical view shape only with "
         "split and reorder";

  Array<DataIterAttr> new_data_iter_array;
  std::unordered_map<int, int> index_map;
  auto split_map = inner.GetSplitMap();
  for (int i = 0; i < static_cast<int>(inner_seps.size()) - 1; i++) {
    for (size_t j = 0; j < attrs.size(); j++) {
      auto attr = attrs[j];
      if (attr.IsSplit() && static_cast<int>(attr.GetIntBound()) == i) {
        new_data_iter_array.push_back(DataIterAttr(attr->extent, -1));
        attrs[j] =
            DeviceIterAttr::Split(attr->extent, static_cast<int>(new_data_iter_array.size()) - 1);
      }
    }
    for (int j = inner_seps[i]; j < inner_seps[i + 1]; j++) {
      new_data_iter_array.push_back(inner->data_iter_array[j]);
      if (split_map.find(j) != split_map.end()) {
        index_map[j] = new_data_iter_array.size() - 1;
      }
    }
  }
  Array<DeviceIterAttr> new_device_iter_array;
  new_device_iter_array.insert(new_device_iter_array.end(), attrs.begin(), attrs.end());
  for (size_t i = 0; i < inner->device_iter_array.size(); i++) {
    auto device_iter = inner->device_iter_array[i];
    if (device_iter.IsSplit()) {
      new_device_iter_array.push_back(
          DeviceIterAttr::Split(device_iter->extent, index_map[device_iter.GetIntBound()]));
    } else {
      new_device_iter_array.push_back(device_iter);
    }
  }
  // Construct the from and to scopes
  if (!inner->device_iter_array.empty()) {
    ICHECK(inner->to.defined()) << "ValueError: The inner layout must have the to scope";
    ICHECK(inner->to.value().Is((from)))
        << "ValueError: The from scope of the inner layout must be the same as the to scope of "
           "the outer layout";
  }

  return TileLayout(new_data_iter_array, new_device_iter_array,
                    !inner->device_iter_array.empty() ? inner->from : from, to);
}

TVM_REGISTER_GLOBAL("tir.TileLayoutShard")
    .set_body_typed([](Array<PrimExpr> shape, Array<PrimExpr> mesh, String strategy,
                       TileLayout inner, ExecScope from,
                       ExecScope to) { return Shard(shape, mesh, strategy, inner, from, to); });

Array<PrimExpr> TileLayoutNode::Apply(const Array<PrimExpr>& coord, const Array<PrimExpr>& shape) const {
  if (shape.size() == 1) {
    ICHECK_EQ(coord.size(), 1) << "ValueError: Expected a single coordinate for single-dimension shape";
    return Apply(coord[0]);
  }
  auto prod_shape = ReduceMul(shape);
  arith::Analyzer analyzer;
  if(!analyzer.CanProveEqual(prod_shape, GetSize())) {
    return TLayoutNode::Apply(coord, shape);
  }
  auto pr = TileLayoutGroupByLogicalShape(GetRef<TileLayout>(this), shape, nullptr);
  auto grouped_layout = pr.first;
  auto seps = pr.second;
  if (!grouped_layout.defined()) {
    return TLayoutNode::Apply(coord, shape);
  }
  PrimExpr result = 0;
  auto split_map = grouped_layout.GetSplitMap();
  for (int i = 0; i < static_cast<int>(shape.size()); i++) {
    PrimExpr cur = coord[i];
    for (int j = seps[i + 1] - 1; j >= seps[i]; j--) {
      if (split_map.find(j) == split_map.end()) {
        result += grouped_layout->data_iter_array[j]->stride * floormod(cur, grouped_layout->data_iter_array[j]->extent);
      }
      cur = floordiv(cur, grouped_layout->data_iter_array[j]->extent);
    }
  }
  return {result};
}

}  // namespace tir
}  // namespace tvm
