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

/*!
 *  Lower TIR+ program to thread view TIR.
 * \file lower_tirp.cc
 */
#include <tvm/arith/analyzer.h>
#include <tvm/runtime/registry.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <queue>
#include <unordered_map>

#include "../../arith/ir_mutator_with_analyzer.h"

namespace tvm {
namespace tirx {

class LogicalTensorRemover : public StmtExprMutator {
 public:
  static Stmt Remove(const Stmt& stmt) { return LogicalTensorRemover()(stmt); }

 private:
  Stmt VisitStmt_(const BlockNode* op) override {
    for (const auto& alloc : op->alloc_buffers) {
      storage_map_[alloc] = alloc;
    }
    for (const auto& view : op->buffer_views) {
      auto it = storage_map_.find(view->src_buffer);
      ICHECK(it != storage_map_.end())
          << "Internal Error: Cannot find storage for " << view->src_buffer;
      storage_map_[view->dst_buffer] = it->second;
    }
    for (const auto& get : op->buffer_gets) {
      auto it = storage_map_.find(get->src_buffer);
      ICHECK(it != storage_map_.end())
          << "Internal Error: Cannot find storage for " << get->src_buffer;
      replace_map_[get->dst_buffer] = it->second;
    }
    Block block = Downcast<Block>(StmtExprMutator::VisitStmt_(op));
    auto* n = block.CopyOnWrite();
    n->buffer_views.clear();
    n->buffer_gets.clear();
    return std::move(block);
  }

  Stmt VisitStmt_(const BufferStoreNode* op) override {
    BufferStore store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    auto it = replace_map_.find(op->buffer);
    if (it != replace_map_.end()) {
      auto* n = store.CopyOnWrite();
      n->buffer = it->second;
      ICHECK_EQ(it->second->shape.size(), op->indices.size())
          << "Internal Error: Inconsistent shape for " << it->second;
    }
    return std::move(store);
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) override {
    BufferLoad load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
    auto it = replace_map_.find(op->buffer);
    if (it != replace_map_.end()) {
      auto* n = load.CopyOnWrite();
      n->buffer = it->second;
      ICHECK_EQ(it->second->shape.size(), op->indices.size())
          << "Internal Error: Inconsistent shape for " << it->second;
    }
    return std::move(load);
  }

  std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual> storage_map_;
  std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual> replace_map_;
};

class ScopeIdDefRemover : public StmtExprMutator {
 public:
  static Stmt Remove(const Stmt& stmt) { return ScopeIdDefRemover()(stmt); }

 private:
  using ScopeIdPrimExprMap =
      std::unordered_map<ScopeIdDef, PrimExpr, ScopeIdDef::ScopeHash, ScopeIdDef::ScopeEqual>;

  Stmt VisitStmt_(const BlockNode* op) override {
    Block block = Downcast<Block>(StmtExprMutator::VisitStmt_(op));
    ICHECK(op->exec_scope.defined()) << "Internal Error: exec_scope is not defined";
    const auto& scope = op->exec_scope.value();
    ICHECK(!scope.Is("world")) << "Internal Error: world scope is not supported at the moment";
    if (auto opt_kernel = scope.as<KernelScope>()) {
      auto kernel = opt_kernel.value();
      ICHECK(resolve_queue_.empty()) << "Internal Error: resolve_queue_ is not empty";
      id_fused_map_.clear();
      replace_map_.clear();

      // Replace ScopeId with PrimExprs using blockIdx / threadIdx
      Var blockIdx("blockIdx", DataType::Int(32)), threadIdx("threadIdx", DataType::Int(32));
      ScopeIdDef blockIdx_def{String("kernel"), String("cta")};
      ScopeIdDef threadIdx_def{String("cta"), String("thread")};
      resolve_queue_.push(blockIdx_def);
      resolve_queue_.push(threadIdx_def);
      id_fused_map_[blockIdx_def] = blockIdx;
      id_fused_map_[threadIdx_def] = threadIdx;
      id_extent_map_.clear();
      // Collect the extents of the ScopeIds
      auto get_extent = [&](const ScopeIdDef& id) {
        auto it = id_extent_map_.find(id);
        ICHECK(it != id_extent_map_.end()) << "Internal Error: Cannot find extent for " << id;
        return it->second;
      };
      {
        std::queue<ScopeIdDef> queue;
        auto insert_extent = [&](const ScopeIdDef& id, const PrimExpr& extent) {
          auto it = id_extent_map_.find(id);
          if (it == id_extent_map_.end()) {
            id_extent_map_[id] = extent;
            queue.push(id);
          } else {
            ICHECK(ana_.CanProveEqual(it->second, extent))
                << "Internal Error: Inconsistent extents for scope " << id;
          }
        };
        // TODO (@bohan): Extents of some ScopeIds are pre-defined by target.
        // Check if the extents are consistent with the pre-defined ones.
        // Or insert the pre-defined ones into id_extent_map_.
        // Such as for CUDA, insert_extent({"warp", "thread"}, 32);
        for (const auto& def : kernel->scope_id_def) {
          insert_extent(ScopeIdDef{def->parent, def->cur}, def.fused_extent());
        }
        while (!queue.empty()) {
          auto head = queue.front();
          queue.pop();
          PrimExpr extent = get_extent(head);

          std::vector<ScopeIdDef> ids;
          for (const auto& [k, v] : id_extent_map_) {
            ids.push_back(k);
          }
          for (const auto& def : ids) {
            auto def_extent = get_extent(def);
            if (auto composed = Compose(head, def)) {
              insert_extent(composed.value(), extent * def_extent);
            } else if (auto composed = Compose(def, head)) {
              insert_extent(composed.value(), extent * def_extent);
            } else if (auto compliment = Compliment(head, def)) {
              insert_extent(compliment.value(), FloorDiv(extent, def_extent));
            } else if (auto compliment = Compliment(def, head)) {
              insert_extent(compliment.value(), FloorDiv(def_extent, extent));
            }
          }
        }
      }
      {
        while (!resolve_queue_.empty()) {
          auto now = resolve_queue_.front();
          resolve_queue_.pop();

          auto it = id_fused_map_.find(now);
          ICHECK(it != id_fused_map_.end()) << "Internal Error: Cannot find PrimExpr for " << now;
          PrimExpr fused = it->second;
          PrimExpr extent = get_extent(now);

          // Find the corresponding ScopeIdDef
          for (const auto& def : kernel->scope_id_def) {
            if (def->parent == now->parent && def->cur == now->cur) {
              // Get the replacements for its scope ids
              std::vector<PrimExpr> replacements = GetReplacements(fused, def->extents);
              // Replace the ScopeId with the replacements
              for (size_t i = 0; i < def->def_ids.size(); i++) {
                replace_map_[def->def_ids[i]] = ana_.Simplify(replacements[i]);
              }
            }
          }

          // Push intermediate scopes to the queue
          for (auto [scope, _] : ScopeOrder) {
            if (Higher(now->parent, scope) && Higher(scope, now->cur)) {
              ScopeIdDef outer = ScopeIdDef{now->parent, scope},
                         inner = ScopeIdDef{scope, now->cur};
              auto it = id_extent_map_.find(inner);
              if (it == id_extent_map_.end()) {
                // Skip if the extent of the inner scope is not available
                continue;
              }
              if (id_fused_map_.find(outer) == id_fused_map_.end()) {
                resolve_queue_.push(outer);
                id_fused_map_[outer] = FloorDiv(fused, it->second);
              }
              if (id_fused_map_.find(inner) == id_fused_map_.end()) {
                resolve_queue_.push(inner);
                id_fused_map_[inner] = FloorMod(fused, it->second);
              }
            }
          }
        }
      }

      // Replace the ScopeIds with the PrimExprs
      auto* n = block.CopyOnWrite();
      n->body = Substitute(n->body, replace_map_);
      n->body = For(threadIdx, 0, get_extent(threadIdx_def), ForKind::kThreadBinding, n->body,
                    IterVar(Range(nullptr), Var(""), IterVarType::kThreadIndex, "threadIdx.x"));
      n->body = For(blockIdx, 0, get_extent(blockIdx_def), ForKind::kThreadBinding, n->body,
                    IterVar(Range(nullptr), Var(""), IterVarType::kThreadIndex, "blockIdx.x"));
      auto* n_scope = kernel.CopyOnWrite();
      n_scope->scope_id_def = {};
      n->exec_scope = KernelScope(kernel);
      return block;
    }
    return std::move(block);
  }

  std::vector<PrimExpr> GetReplacements(const PrimExpr& fused, const Array<PrimExpr>& extents) {
    std::vector<PrimExpr> res;
    PrimExpr mod = extents[0];
    res.push_back(FloorMod(fused, mod));
    for (size_t i = 1; i < extents.size(); i++) {
      res.push_back(FloorMod(FloorDiv(fused, mod), extents[i]));
      mod = mod * extents[i];
    }
    return std::move(res);
  }

  std::queue<ScopeIdDef> resolve_queue_;
  ScopeIdPrimExprMap id_fused_map_;
  ScopeIdPrimExprMap id_extent_map_;
  std::unordered_map<Var, PrimExpr, ObjectPtrHash, ObjectPtrEqual> replace_map_;
  arith::Analyzer ana_;
};

class StorageLower : public arith::IRMutatorWithAnalyzer {
 public:
  static Stmt Flatten(const Stmt& stmt, const Map<tirx::Var, Buffer> buffer_map) {
    arith::Analyzer ana;
    StorageLower storage_lower(&ana);
    for (const auto& kv : buffer_map) {
    }
    return storage_lower(stmt);
  }

 private:
  explicit StorageLower(arith::Analyzer* analyzer) : arith::IRMutatorWithAnalyzer(analyzer) {}

  Stmt VisitStmt_(const BlockNode* op) final {
    ICHECK_EQ(op->buffer_gets.size(), 0) << "Unexpected BufferGet found";
    ICHECK_EQ(op->buffer_views.size(), 0) << "Unexpected BufferView found";
    ICHECK_EQ(op->match_buffers.size(), 0) << "Unexpected MatchBufferRegion found";

    Block block = GetRef<Block>(op);

    Array<Buffer> alloc_buffers = op->alloc_buffers;
    alloc_buffers.MutateByApply([this](Buffer buf) { return GetFlattenedBuffer(buf); });
    if (!alloc_buffers.same_as(op->alloc_buffers)) {
      block.CopyOnWrite()->alloc_buffers = alloc_buffers;
    }

    return StmtExprMutator::VisitStmt_(block.get());
  }

  Buffer GetFlattenedBuffer(Buffer buf) {
    auto it = buffer_remap_.find(buf);
    if (it != buffer_remap_.end()) {
      return it->second;
    }
    auto flattened = buf.GetFlattenedBuffer();
    auto writer = flattened.CopyOnWrite();

    // TODO(Lunderberg): Move the handling of boolean into a
    // dedicated pass.
    if (flattened->dtype == DataType::Bool()) {
      writer->dtype = DataType::Int(8);
    }
    // canonicalize shape
    for (size_t i = 0; i < flattened->shape.size(); ++i) {
      writer->shape.Set(i, analyzer_->canonical_simplify(flattened->shape[i]));
    }
    writer->layout = NullOpt;

    buffer_remap_[buf] = flattened;
    return flattened;
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    BufferStore store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    bool store_returns_bool = (op->value.dtype() == DataType::Bool());
    store = VisitBufferAccess(store);

    // Handle casts from the value's dtype to the dtype of the
    // backing array.
    // TODO(Lunderberg): Move the handling of boolean into a
    // dedicated pass.
    if (store_returns_bool) {
      ICHECK_EQ(store->buffer->dtype, DataType::Int(8))
          << "Expected int8 backing array for boolean tensor";
      auto writer = store.CopyOnWrite();
      writer->value = tvm::cast(DataType::Int(8), store->value);
      return std::move(store);
    }
    return std::move(store);
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    bool load_returns_bool = (op->dtype == DataType::Bool());
    BufferLoad load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
    load = VisitBufferAccess(load);
    // Handle casts from dtype of the backing array to value's dtype.
    // TODO(Lunderberg): Move the handling of boolean into a
    // dedicated pass.
    if (load_returns_bool) {
      ICHECK_EQ(load->buffer->dtype, DataType::Int(8))
          << "Expected int8 backing array for boolean tensor";
      load.CopyOnWrite()->dtype = DataType::Int(8);
      return tvm::cast(DataType::Bool(), load);
    } else {
      return std::move(load);
    }
  }

  Array<PrimExpr> GetSimplifiedElemOffset(const Buffer& buffer, const Array<PrimExpr>& indices) {
    auto flattened_indices = buffer->ElemOffset(indices);
    flattened_indices = this->IterMapSimplifyWithContext(flattened_indices, false);
    ICHECK_EQ(flattened_indices.size(), 1) << "Expected a single element offset";
    if (buffer->layout.defined()) {
      return {analyzer_->Simplify(buffer->layout.value()->Apply({flattened_indices[0]}))};
    }
    return flattened_indices;
  }

  template <typename Node>
  Node VisitBufferAccess(Node node) {
    ICHECK(node->buffer.defined());
    auto flattened_indices = GetSimplifiedElemOffset(node->buffer, node->indices);
    Buffer flattened_buffer = GetFlattenedBuffer(node->buffer);

    auto writer = node.CopyOnWrite();
    writer->buffer = flattened_buffer;
    writer->indices = flattened_indices;
    return node;
  }

  /*! \brief Map of buffers being remapped. */
  std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual> buffer_remap_;
};

class BufferOffsetRemover : public StmtExprMutator {
 public:
  static Stmt Remove(const Stmt& stmt) { return BufferOffsetRemover()(stmt); }

 private:
  PrimExpr VisitExpr_(const tirx::CallNode* call) final {
    if (call->op.same_as(tirx::builtin::buffer_offset())) {
      auto buffer_load = Downcast<BufferLoad>(call->args[0]);
      ICHECK_EQ(buffer_load->indices.size(), 1) << "Expected a single index";
      return buffer_load->indices[0];
    }
    return StmtExprMutator::VisitExpr_(call);
  }
};

namespace transform {
Pass LowerTIRp() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    n->body = LogicalTensorRemover::Remove(n->body);
    n->body = ScopeIdDefRemover::Remove(n->body);
    n->body = StorageLower::Flatten(n->body, n->buffer_map);
    n->body = BufferOffsetRemover::Remove(n->body);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tirx.LowerTIRp", {});
}

TVM_REGISTER_GLOBAL("tirx.transform.LowerTIRp").set_body_typed(LowerTIRp);

}  // namespace transform
}  // namespace tirxx
}  // namespace tvm
