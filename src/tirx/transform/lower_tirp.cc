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
#include <tvm/tirx/tirp_op.h>
#include <tvm/tirx/transform.h>

#include <queue>
#include <unordered_map>

#include "../../arith/ir_mutator_with_analyzer.h"
#include "../ir/tir_visitor_with_path.h"

namespace tvm {
namespace tirx {

class ScopeIdDefResolver : public StmtExprMutator {
 public:
  static Stmt Resolve(const Stmt& stmt) { return ScopeIdDefResolver()(stmt); }

 private:
  using LaunchParams = ScopeIdResolveTable::LaunchParams;

  Stmt VisitStmt_(const BlockNode* op) override {
    Block block = Downcast<Block>(StmtExprMutator::VisitStmt_(op));
    ICHECK(op->exec_scope.defined()) << "Internal Error: exec_scope is not defined";
    const auto& scope = op->exec_scope.value();
    ICHECK(!scope.Is("world")) << "Internal Error: world scope is not supported at the moment";
    // Resolve under kernel exec scope
    auto opt_kernel = scope.as<KernelScope>();
    if (!opt_kernel) {
      return std::move(block);
    }
    auto kernel = opt_kernel.value();
    if (kernel->scope_id_def.empty()) {
      // No ScopeIdDef to resolve, return the block as is
      return std::move(block);
    }
    // Step 1: Verify the ScopeIdDef is well-formed
    ScopeIdDefVerifier verifier;
    CHECK(verifier.Verify(kernel->scope_id_def)) << "Inconsistent ScopeIdDef";

    // Step 2: Extract kernel launch parameters
    const auto& target = Target::Current(false);

    // Step 3: Extract kernel launch parameters
    LaunchParams launch_params;
    ExtractKernelLaunchParams(verifier.id_set, target, &launch_params);

    // Step 4: Resolve the ScopeIdDef and replace them
    auto* n = block.CopyOnWrite();

    std::unordered_map<Var, PrimExpr, ObjectPtrHash, ObjectPtrEqual> id_map;
    for (const auto& def : kernel->scope_id_def) {
      // Resolve the scope ids defined in the kernel
      auto resolved = ScopeIdResolveTable::Resolve(def, target->kind->name, launch_params);
      ICHECK_EQ(resolved.size(), def->extents.size())
          << "Internal Error: Inconsistent resolved size " << resolved.size() << " vs "
          << def->extents.size();
      for (size_t i = 0; i < def->def_ids.size(); i++) {
        id_map[def->def_ids[i]] = resolved[i];
      }
    }
    Stmt body = Substitute(n->body, id_map);

    // Step 5. Warp kernel launch parameters
    for (const auto& [tag, iv] : launch_params) {
      body = AttrStmt(iv, tir::attr::thread_extent, iv->dom->extent, body);
    }

    // Clear the scope_id_def
    auto* n_scope = kernel.CopyOnWrite();
    n_scope->scope_id_def = {};

    // Return the resolved block
    n->exec_scope = kernel;
    n->body = body;
    return block;
  }

  void ExtractKernelLaunchParams(const ScopeIdDefVerifier::ScopeIdSet& id_set, const Target& target,
                                 LaunchParams* launch_params) {
    auto add_launch_param = [&](const ScopePair& pair, const std::string& prefix) {
      auto it = id_set.find(pair);
      CHECK(it != id_set.end()) << "ValueError: Expected " << pair << " to be defined";
      const auto& def = (*it).second;
      CHECK_LE(def->extents.size(), 3) << "ValueError: Only up to 3 extents are supported";
      for (size_t i = 0; i < def->extents.size(); i++) {
        std::string thread_tag = prefix + static_cast<char>('x' + i);
        IterVar iv(Range::FromMinExtent(0, def->extents[i]), Var(thread_tag),
                   IterVarType::kThreadIndex, thread_tag);
        launch_params->insert({String(prefix + static_cast<char>('x' + i)), iv});
      }
    };
    // blockIdx.x, blockIdx.y, blockIdx.z
    auto it = id_set.find(ScopePair("cluster", "cta"));
    if (it == id_set.end() || is_one((*it).second.fused_extent())) {
      // no cluster
      add_launch_param(ScopePair("kernel", "cta"), "blockIdx.");
    } else {
      // use cluster
      // clusterCtaIdx.x, clusterCtaIdx.y, clusterCtaIdx.z
      CHECK(target->kind->name == "cuda") << "ValueError: cluster is only supported in CUDA";
      CHECK_EQ(target->kind->default_device_type, kDLCUDA)
          << "ValueError: cluster is only supported in CUDA";
      add_launch_param(ScopePair("cluster", "cta"), "clusterCtaIdx.");
      add_launch_param(ScopePair("kernel", "cta"), "blockIdx.");
    }
    // threadIdx.x, threadIdx.y, threadIdx.z
    add_launch_param(ScopePair("cta", "thread"), "threadIdx.");
  }

  /*! \brief The arithmetic analyzer */
  arith::Analyzer ana_;
};

class NoOpCallVerifier : public Verifier<NoOpCallVerifier> {
 public:
  using Verifier::Verifier;

 private:
  using Verifier::Visit;

  void VisitStmt_(const tirp::OpCallNode* obj, ObjectPath path) final {
    Verify(false) << "TIRpError: OpCall at " << path << " is not allowed in TIRp before lowering";
  }
};

class TIRpOpScheduler : public StmtExprMutator {
 public:
  static Stmt LowerOpCalls(const Stmt& stmt) { return TIRpOpScheduler()(stmt); }

 private:
  Stmt VisitStmt_(const BlockNode* op) final {
    Block block = GetRef<Block>(op);
    auto* n = block.CopyOnWrite();
    // Get the exec_scope
    if (op->exec_scope.defined()) {
      exec_scope_stack_.push_back(op->exec_scope.value());
    }
    n->body = VisitStmt(n->body);
    // Pop the exec_scope
    if (op->exec_scope.defined()) {
      exec_scope_stack_.pop_back();
    }
    return std::move(block);
  }

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    // Collect the launch parameters
    if (op->attr_key == tir::attr::thread_extent) {
      auto thread_extent = Downcast<IterVar>(op->node);
      ICHECK(thread_extent->thread_tag.defined())
          << "Internal Error: thread_extent without thread_tag";
      launch_params_[thread_extent->thread_tag] = op->value;
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const ForNode* op) final {
    // Collect the loop variables
    auto loop_var = Downcast<Var>(op->loop_var);
    ICHECK(!var_range_map_.count(loop_var)) << "Internal Error: Duplicate loop variable";
    var_range_map_.Set(loop_var, Range::FromMinExtent(op->min, op->extent));
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const tirp::OpCallNode* op) final {
    tirp::ScheduleContext sctx(Target::Current(false), exec_scope_stack_.back(), launch_params_);
    static auto f_op_scheduler_ = tvm::runtime::Registry::Get("tirp.f_op_scheduler");
    ICHECK(f_op_scheduler_ != nullptr) << "Internal Error: tirp.f_op_scheduler is not registered";
    PrimFunc res;
    res = (*f_op_scheduler_)(op->op, op->args, sctx, var_range_map_);
    ICHECK(res.defined()) << "Internal Error: tirp.f_op_scheduler returned an undefined PrimFunc";
    return res->body;
  }

  Map<Var, Range> var_range_map_;
  std::vector<ExecScope> exec_scope_stack_;
  std::unordered_map<String, PrimExpr, ObjectHash, ObjectEqual> launch_params_;
};

class ScopeMerger : public StmtExprMutator {
 public:
  static Stmt Merge(const Stmt& stmt) { return ScopeMerger()(stmt); }

 private:
  Stmt VisitStmt_(const SeqStmtNode* op) final {
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    if (auto* n = stmt.as<SeqStmtNode>()) {
      std::vector<Stmt> seq;
      for (size_t i = 0; i < n->seq.size();) {
        if (auto* realize = n->seq[i].as<BlockRealizeNode>()) {
          // Find a sequence of blocks with the same exec_scope
          auto block = realize->block;
          std::vector<Stmt> new_body{block->body};
          ICHECK(block->exec_scope.defined()) << "Internal Error: exec_scope is not defined";
          auto scope = block->exec_scope.value();
          for (i++; i < n->seq.size(); i++) {
            if (auto* next_realize = n->seq[i].as<BlockRealizeNode>()) {
              const auto& next_block = next_realize->block;
              ICHECK(next_block->exec_scope.defined())
                  << "Internal Error: exec_scope is not defined";
              if (scope.Is(next_block->exec_scope.value())) {
                new_body.push_back(next_block->body);
                continue;
              }
            }
            break;
          }
          auto* new_block = block.CopyOnWrite();
          new_block->body = SeqStmt::Flatten(new_body);
          seq.push_back(BlockRealize({}, Bool(true), block));
        } else {
          seq.push_back(n->seq[i]);
          i++;
        }
      }
      return SeqStmt::Flatten(seq);
    }
    return stmt;
  };
};

class ScheduleContextRemover : public StmtExprMutator {
 public:
  static Stmt Remove(const Stmt& stmt) { return ScheduleContextRemover()(stmt); }

 private:
  Stmt VisitStmt_(const BlockNode* op) final {
    Block block = GetRef<Block>(op);
    auto* n = block.CopyOnWrite();
    n->annotations.erase("scope_id_extent_map");
    n->annotations.erase("thread_var_map");
    n->body = VisitStmt(n->body);
    return std::move(block);
  }
};

class LogicalTensorRemover : public arith::IRMutatorWithAnalyzer {
 public:
  static Stmt Remove(const Stmt& stmt, const Map<tir::Var, Buffer> buffer_map) {
    arith::Analyzer ana;
    LogicalTensorRemover remover(&ana);
    for(const auto& kv : buffer_map) {
      remover.storage_map_[kv.second] = kv.second;
    }
    return remover(stmt);
  }

 private:
  explicit LogicalTensorRemover(arith::Analyzer* analyzer)
      : arith::IRMutatorWithAnalyzer(analyzer) {}
  Stmt VisitStmt_(const BlockNode* op) override {
    for (const auto& alloc : op->alloc_buffers) {
      storage_map_[alloc] = alloc;
    }
    for (const auto& view : op->buffer_views) {
      auto it = storage_map_.find(view->src_buffer);
      ICHECK(it != storage_map_.end())
          << "Internal Error: Cannot find storage for " << view->src_buffer;
      storage_map_[view->dst_buffer] = it->second;
      replace_map_[view->dst_buffer] = it->second;
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

  Array<PrimExpr> RewriteIndices(Array<PrimExpr> indices, Array<PrimExpr> old_shape, Array<PrimExpr> new_shape){
    PrimExpr indices_prod = 1;
    for(size_t i = 0; i < indices.size(); i++){
      indices_prod *= old_shape[i];
      indices_prod += indices[i];
    }
    std::vector<PrimExpr> new_indices;
    int new_shape_size = new_shape.size();
    for (int i = new_shape_size - 1; i>=0; i--) {
      new_indices.push_back(analyzer_->Simplify( floormod(indices_prod, new_shape[i])));
      indices_prod = floordiv(indices_prod, new_shape[i]);
    }
    std::reverse(new_indices.begin(), new_indices.end());
    return Array<PrimExpr>(new_indices.begin(), new_indices.end());
  }

  Stmt VisitStmt_(const BufferStoreNode* op) override {
    BufferStore store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    auto it = replace_map_.find(op->buffer);
    if (it != replace_map_.end()) {
      auto* n = store.CopyOnWrite();
      n->buffer = it->second;
      n->indices = RewriteIndices(op->indices, op->buffer->shape, it->second->shape);
      ICHECK_EQ(it->second->shape.size(), n->indices.size())
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
      n->indices = RewriteIndices(op->indices, op->buffer->shape, it->second->shape);
      ICHECK_EQ(it->second->shape.size(), n->indices.size())
          << "Internal Error: Inconsistent shape for " << it->second;
    }
    return std::move(load);
  }

  std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual> storage_map_;
  std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual> replace_map_;
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

 protected:
  using IRMutatorWithAnalyzer::VisitExpr_;
  using IRMutatorWithAnalyzer::VisitStmt_;

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

  virtual Buffer GetFlattenedBuffer(Buffer buf) {
    auto it = buffer_remap_.find(buf);
    if (it != buffer_remap_.end()) {
      return it->second;
    }
    auto trn_layout = buf->layout.as<TrainiumLayoutNode>();
    Buffer flattened;
    tir::BufferNode* writer;
    if (trn_layout) {
      auto new_shape = {trn_layout->GetPartitionSize(), trn_layout->GetCosize()};
      flattened = buf;
      writer = flattened.CopyOnWrite();
      writer->shape = new_shape;
      writer->strides = {};
      writer->axis_separators = {};
    } else {
      flattened = buf.GetFlattenedBuffer();
      writer = flattened.CopyOnWrite();
    }
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

  virtual Array<PrimExpr> GetSimplifiedElemOffset(const Buffer& buffer, const Array<PrimExpr>& indices) {
    if(buffer->layout.defined() && buffer->layout.value().as<TrainiumLayoutNode>()){
      auto simplified_indices = this->IterMapSimplifyWithContext(indices, false);
      if (buffer->layout.defined()) {
        simplified_indices = buffer->layout.value()->Apply(simplified_indices, buffer->shape);
        simplified_indices = this->IterMapSimplifyWithContext(simplified_indices, false);
        for (size_t i = 0; i < simplified_indices.size(); i++) {
          simplified_indices.Set(i, analyzer_->Simplify(simplified_indices[i]));
        }
      }
      return simplified_indices;
    }
    auto flattened_indices = buffer->ElemOffset(indices, true);
    flattened_indices = this->IterMapSimplifyWithContext(flattened_indices, false);
    ICHECK_EQ(flattened_indices.size(), 1) << "Expected a single element offset";
    if (buffer->layout.defined()) {
      return {analyzer_->Simplify(buffer->layout.value()->Apply(flattened_indices[0])[0])};
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
    const auto& target = Target::Current(false);

    auto* n = f.CopyOnWrite();
    // Lower ScopeIdDef, resolve the scope ids and replace them with target defined special
    // registers Collect the extents of the ScopeIds for OpCall scheduling
    n->body = ScopeIdDefResolver::Resolve(n->body);

    // Default Schedule: lower TIRp OpCalls
    while (!NoOpCallVerifier::Verify(n->body, false)) {
      n->body = TIRpOpScheduler::LowerOpCalls(n->body);
      n->body = ScopeMerger::Merge(n->body);
    }
    // Verify that there are no OpCalls in the TIRp
    NoOpCallVerifier::Verify(f, true);

    // Lower other TIRp aux data structures
    n->body = ScheduleContextRemover::Remove(n->body);
    n->body = LogicalTensorRemover::Remove(n->body, n->buffer_map);
    n->body = StorageLower::Flatten(n->body, n->buffer_map);
    n->body = BufferOffsetRemover::Remove(n->body);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tirx.LowerTIRp", {});
}

TVM_REGISTER_GLOBAL("tirx.transform.LowerTIRp").set_body_typed(LowerTIRp);

}  // namespace transform
}  // namespace tirxxx
}  // namespace tvm
