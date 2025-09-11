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
#include <tvm/tirx/function.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/tirp_op.h>
#include <tvm/tirx/transform.h>

#include <queue>
#include <unordered_map>

#include "../../arith/ir_mutator_with_analyzer.h"
#include "../ir/functor_common.h"
#include "../ir/tir_visitor_with_path.h"

namespace tvm {
namespace tirx {

class ScopeIdDefGather : public StmtExprVisitor {
 public:
  static std::vector<ScopeIdDef> Gather(const Stmt& stmt) {
    ScopeIdDefGather gather;
    gather(stmt);
    return gather.scope_id_def;
  }

  void VisitStmt_(const BlockNode* op) override {
    StmtExprVisitor::VisitStmt_(op);
    if (!op->exec_scope.has_value()) return;
    for (const auto& def : op->exec_scope.value()->scope_id_def) {
      scope_id_def.push_back(def);
    }
  }

  std::vector<ScopeIdDef> scope_id_def;
};

class ScopeIdDefRemover : public StmtExprMutator {
 public:
  static Stmt Remove(const Stmt& stmt) { return ScopeIdDefRemover()(stmt); }

  Stmt VisitStmt_(const BlockNode* op) override {
    Block block = ffi::GetRef<Block>(op);
    auto* n = block.CopyOnWrite();
    Stmt body = StmtExprMutator::VisitStmt(op->body);
    if (op->exec_scope.defined()) {
      if (const auto* slice = op->exec_scope.value().as<ExecScopeSliceNode>()) {
        auto n_scope = ffi::make_object<ExecScopeSliceNode>(*slice);
        n_scope->scope_id_def = {};
        n->exec_scope = ExecScopeSlice(n_scope);
      } else if (const auto* scope = op->exec_scope.value().as<ExecScopeNode>()) {
        auto n_scope = ffi::make_object<ExecScopeNode>(*scope);
        n_scope->scope_id_def = {};
        n->exec_scope = ExecScope(n_scope);
      } else {
        LOG(FATAL) << "Internal Error: unknown exec_scope type: " << op->exec_scope.value();
      }
    }
    n->body = body;
    return block;
  }
};

class ScopeIdDefResolver : public StmtExprMutator {
 public:
  ScopeIdDefResolver(const Target& target) : target_(target) {}

  static Stmt Resolve(const Stmt& stmt, const Target& target) {
    return ScopeIdDefResolver(target)(stmt);
  }

 private:
  using LaunchParams = ScopeIdResolveTable::LaunchParams;

  Stmt VisitStmt_(const BlockRealizeNode* op) override {
    auto n_realize = CopyOnWrite(op);
    if (!op->block->exec_scope.defined()) {
      // No exec_scope, return the block as is
      n_realize->block = Downcast<Block>(StmtExprMutator::VisitStmt_(op->block.get()));
      return Stmt(n_realize);
    }
    const auto& scope = op->block->exec_scope.value();
    ICHECK(!scope->Is("world")) << "TIRp Error: world scope is not supported at the moment";
    ICHECK(scope->Is("kernel") && !kernel_launch_params_.has_value())
        << "TIRp Error: a scope is not wrapped in a kernel scope";

    // Step 0: Gather the scope id defs from all the scope
    Array<ScopeIdDef> scope_id_def = std::move(ScopeIdDefGather::Gather(op->block));

    // Step 1: Verify the ScopeIdDef is well-formed
    ScopeIdDefVerifier verifier;
    CHECK(verifier.Verify(scope_id_def)) << "Inconsistent ScopeIdDef";

    // Step 2: Extract kernel launch parameters
    LaunchParams launch_params;
    ExtractKernelLaunchParams(verifier.id_set, target_, &launch_params);
    kernel_launch_params_ = launch_params;

    // Step 3: Visit the block and resolve the scope ids, replace them with kernel launch params
    auto block = op->block;
    auto* n = block.CopyOnWrite();

    std::unordered_map<Var, PrimExpr, ObjectPtrHash, ObjectPtrEqual> id_map;
    for (const auto& def : scope_id_def) {
      // Resolve the scope ids defined in the scope
      auto resolved =
          ScopeIdResolveTable::Resolve(def->scope, def->extents, def->extents.size(),
                                       target_->kind->name, kernel_launch_params_.value());
      ICHECK_EQ(resolved.size(), def->extents.size())
          << "Internal Error: Inconsistent resolved size " << resolved.size() << " vs "
          << def->extents.size();
      for (size_t i = 0; i < def->def_ids.size(); i++) {
        id_map[def->def_ids[i]] = resolved[i];
      }
    }
    n->body = Substitute(n->body, id_map);
    n_realize->block = block;

    // Step 4: Remove the scope_id_def inside the scope
    Stmt ret = ScopeIdDefRemover::Remove(Stmt(n_realize));

    // Step 5: Wrap the block with thread_extent attributes
    for (const auto& [tag, iv] : kernel_launch_params_.value()) {
      ret = AttrStmt(iv, tirx::attr::thread_extent, iv->dom->extent, ret);
    }
    // Clear the kernel launch params after the kernel scope is resolved
    kernel_launch_params_ = std::nullopt;
    return ret;
  }

  void ExtractKernelLaunchParams(const ScopeIdDefVerifier::ScopeIdSet& id_set, const Target& target,
                                 LaunchParams* launch_params) {
    auto add_launch_param = [&](const ScopePair& pair, const std::string& prefix) {
      auto it = id_set.find(pair);
      if (it == id_set.end()) {
        return;
      }
      const auto& def = (*it).second;
      CHECK_LE(def->extents.size(), 3) << "ValueError: Only up to 3 extents are supported";
      for (size_t i = 0; i < def->extents.size(); i++) {
        std::string thread_tag = prefix + static_cast<char>('x' + i);
        IterVar iv(Range::FromMinExtent(0, def->extents[i]), Var(thread_tag),
                   IterVarType::kThreadIndex, thread_tag);
        launch_params->insert({ffi::String(prefix + static_cast<char>('x' + i)), iv});
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

  /*! \brief The launch params of current kernel scope */
  ffi::Optional<LaunchParams> kernel_launch_params_{std::nullopt};
  /*! \brief The arithmetic analyzer */
  arith::Analyzer ana_;
  const Target& target_;
};

class NoOpCallVerifier : public Verifier<NoOpCallVerifier> {
 public:
  using Verifier::Verifier;

 private:
  using Verifier::Visit;

  void VisitStmt_(const tirp::OpCallNode* obj, ffi::reflection::AccessPath path) final {
    Verify(false) << "TIRpError: OpCall at " << path << " is not allowed in TIRp before lowering";
  }
};

class TIRpOpScheduler : public StmtExprMutator {
 public:
  TIRpOpScheduler(const Target& target) : target_(target) {}

  static Stmt LowerOpCalls(const Stmt& stmt, const Target& target) {
    return TIRpOpScheduler(target)(stmt);
  }

 private:
  class KernelReplacePointSearcher : public StmtExprMutator {
   public:
    KernelReplacePointSearcher(const Stmt& body) : body_(body) {}

    static Stmt Seek(const Stmt& stmt, const Stmt& body) {
      return KernelReplacePointSearcher(body)(stmt);
    }

   private:
    Stmt VisitStmt_(const tirp::OpCallNode* op) final {
      if (op->op == tirp::tvm_kernel_replace_point()) {
        return body_;
      }
      return StmtExprMutator::VisitStmt_(op);
    }

    Stmt body_;
  };

  Stmt VisitStmt_(const BlockRealizeNode* op) final {
    BlockRealize block_realize = ffi::GetRef<BlockRealize>(op);
    auto* n = block_realize.CopyOnWrite();
    Stmt body = VisitStmt(n->block);
    if (auto block = body.as<Block>()) {
      n->block = block.value();
      return std::move(block_realize);
    } else {
      return body;
    }
  }

  Stmt VisitStmt_(const BlockNode* op) final {
    bool is_first_block = false;
    std::swap(is_first_block, is_first_block_);
    Block block = ffi::GetRef<Block>(op);
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
    if (is_first_block) {
      // Insert device init stmts and alloc buffers
      for (const auto& stmt : device_init_stmts_) {
        n->body = KernelReplacePointSearcher::Seek(stmt, n->body);
      }
      n->alloc_buffers.insert(n->alloc_buffers.end(), alloc_buffers_.begin(), alloc_buffers_.end());
      Stmt res = BlockRealize({}, Bool(true), std::move(block));
      if (is_first_thread_attr_) {
        // Insert host init stmts outside the outermost thread binding or block
        for (const auto& stmt : host_init_stmts_) {
          res = KernelReplacePointSearcher::Seek(stmt, std::move(res));
        }
      }
      std::swap(is_first_block, is_first_block_);
      return res;
    }
    return std::move(block);
  }

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    // Collect the launch parameters
    if (op->attr_key == tirx::attr::thread_extent) {
      bool is_first_thread_attr = false;
      std::swap(is_first_thread_attr, is_first_thread_attr_);
      auto thread_extent = Downcast<IterVar>(op->node);
      launch_params_[thread_extent->thread_tag] = thread_extent;
      Stmt res = StmtExprMutator::VisitStmt_(op);
      if (is_first_thread_attr && is_first_block_) {
        // Insert host init stmts outside the outermost thread binding or block
        for (const auto& stmt : host_init_stmts_) {
          res = KernelReplacePointSearcher::Seek(stmt, std::move(res));
        }
      }
      std::swap(is_first_thread_attr, is_first_thread_attr_);
      return res;
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
    tirp::ScheduleContext sctx(target_, exec_scope_stack_.back(), launch_params_, var_range_map_);
    static auto f_op_scheduler_ = ffi::Function::GetGlobal("tirp.f_op_scheduler");
    ICHECK(f_op_scheduler_.has_value()) << "Internal Error: tirp.f_op_scheduler is not registered";
    PrimFunc res = f_op_scheduler_.value()(ffi::GetRef<tirp::OpCall>(op), sctx).cast<PrimFunc>();
    if (res.defined()) {
      // Implmentation found, handle callbacks
      if (auto bufs = sctx->callbacks.Get(tirp::callback::kPrivateAlloc)) {
        auto buf_list = bufs.value().as<Array<Buffer>>().value();
        alloc_buffers_.insert(alloc_buffers_.end(), buf_list.begin(), buf_list.end());
      }
      if (auto stmts = sctx->callbacks.Get(tirp::callback::kDeviceInitStmt)) {
        auto stmt_list = stmts.value().as<Array<Stmt>>().value();
        device_init_stmts_.insert(device_init_stmts_.end(), stmt_list.begin(), stmt_list.end());
      }
      if (auto stmts = sctx->callbacks.Get(tirp::callback::kHostInitStmt)) {
        auto stmt_list = stmts.value().as<Array<Stmt>>().value();
        host_init_stmts_.insert(host_init_stmts_.end(), stmt_list.begin(), stmt_list.end());
      }
      return res->body;
    } else {
      // No implementation found, it could be some deferred scheduling such as pipeline
      return StmtExprMutator::VisitStmt_(op);
    }
  }

  ffi::Map<Var, Range> var_range_map_;
  const Target& target_;
  std::vector<ExecScope> exec_scope_stack_;
  std::unordered_map<ffi::String, IterVar> launch_params_;
  std::vector<Buffer> alloc_buffers_;
  std::vector<Stmt> device_init_stmts_;
  std::vector<Stmt> host_init_stmts_;

  bool is_first_block_{true};
  bool is_first_thread_attr_{true};
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
          auto* new_block = block.CopyOnWrite();
          std::vector<Stmt> new_body{block->body};
          ICHECK(block->exec_scope.defined()) << "Internal Error: exec_scope is not defined";
          auto scope = block->exec_scope.value();
          for (i++; i < n->seq.size(); i++) {
            if (auto* next_realize = n->seq[i].as<BlockRealizeNode>()) {
              const auto& next_block = next_realize->block;
              ICHECK(next_block->exec_scope.defined())
                  << "Internal Error: exec_scope is not defined";
              if (scope->Is(next_block->exec_scope.value())) {
                new_body.push_back(next_block->body);
                continue;
              }
            }
            break;
          }
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

class ExecScopeSliceResolver : public StmtExprMutator {
 public:
  ExecScopeSliceResolver(const Target& target) : target_(target) {}

  static Stmt Resolve(const Stmt& stmt, const Target& target) {
    return ExecScopeSliceResolver(target)(stmt);
  }

 private:
  using LaunchParams = ScopeIdResolveTable::LaunchParams;

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == tirx::attr::thread_extent) {
      auto iv = op->node.as<IterVar>();
      ICHECK(iv.has_value()) << "Internal Error: thread_extent should annotate an IterVar";
      launch_params_[iv.value()->thread_tag] = iv.value();
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const BlockRealizeNode* op) final {
    Stmt block = this->VisitStmt(op->block);
    if (block.same_as(op->block)) {
      return ffi::GetRef<Stmt>(op);
    } else {
      if (auto* block_ptr = block.as<BlockNode>()) {
        auto n = CopyOnWrite(op);
        n->block = ffi::GetRef<Block>(block_ptr);
        return Stmt(n);
      } else {
        return block;
      }
    }
  }

  Stmt VisitStmt_(const BlockNode* op) final {
    Block block = ffi::GetRef<Block>(op);
    auto* n = block.CopyOnWrite();
    if (op->annotations.count("tirp.scope_partition")) {
      // scope partition is enabled, rewrite the body
      auto seq = block->body.as<SeqStmt>();
      if (!seq.has_value()) {
        CHECK(false) << "TIRpError: Block with scope partition at has invalid body " << block->body;
      }
      Array<Stmt> new_seq;
      for (const auto& stmt : seq.value()->seq) {
        new_seq.push_back(VisitStmt(stmt));
      }
      // Connect the IfThenElse stmts into a single IfThenElse
      Stmt body = new_seq[new_seq.size() - 1];
      for (int i = new_seq.size() - 2; i >= 0; i--) {
        auto if_then = new_seq[i].as<IfThenElse>();
        CHECK(if_then.has_value() && !if_then.value()->else_case.defined())
            << "TIRpError: Block with scope partition has invalid body " << block->body;
        body = IfThenElse(if_then.value()->condition, if_then.value()->then_case, body);
      }
      n->body = body;
      n->annotations.erase("tirp.scope_partition");
    } else {
      // no scope partition, visit the body
      n->body = VisitStmt(n->body);
    }
    // no scope partition, return the block as is
    if (!op->exec_scope.defined()) {
      return std::move(block);
    }
    auto exec_scope = op->exec_scope.value();
    auto scope_slice_opt = exec_scope.as<ExecScopeSlice>();
    if (!scope_slice_opt.has_value()) {
      // no scope slice, return the block as is
      return std::move(block);
    }
    auto scope_slice = scope_slice_opt.value();
    auto scope = ScopePair(scope_slice->parent, scope_slice->name);
    int out_dim = scope_slice->slices.as<PrimExpr>().has_value()
                      ? 1
                      : scope_slice->slices.as<Array<Range>>().value().size();
    auto resolved = ScopeIdResolveTable::Resolve(scope, scope_slice->extents, out_dim,
                                                 target_->kind->name, launch_params_);
    ICHECK_EQ(resolved.size(), out_dim);
    n->exec_scope = ExecScope::Create(scope_slice->name);
    if (auto select_cond = scope_slice->slices.as<PrimExpr>()) {
      return IfThenElse(select_cond.value(), BlockRealize({}, Bool(true), block));
    } else {
      auto slices = scope_slice->slices.as<Array<Range>>().value();
      PrimExpr cond = Bool(true);
      for (size_t i = 0; i < slices.size(); i++) {
        cond = cond && resolved[i] >= slices[i]->min &&
               resolved[i] < slices[i]->extent + slices[i]->min;
      }
      return IfThenElse(cond, BlockRealize({}, Bool(true), block));
    }
  }

  LaunchParams launch_params_;
  const Target& target_;
};

class ScheduleContextRemover : public StmtExprMutator {
 public:
  static Stmt Remove(const Stmt& stmt) { return ScheduleContextRemover()(stmt); }

 private:
  Stmt VisitStmt_(const BlockNode* op) final {
    Block block = ffi::GetRef<Block>(op);
    auto* n = block.CopyOnWrite();
    n->annotations.erase("scope_id_extent_map");
    n->annotations.erase("thread_var_map");
    n->body = VisitStmt(n->body);
    return std::move(block);
  }
};

class LayoutApplier : public arith::IRMutatorWithAnalyzer {
 public:
  static std::pair<Stmt, ffi::Map<Var, Buffer>> Flatten(const Stmt& stmt,
                                                        const ffi::Map<tirx::Var, Buffer> buffer_map,
                                                        const Target& target) {
    arith::Analyzer ana;
    LayoutApplier storage_lower(&ana, target);
    std::unordered_map<Var, Buffer> new_buffer_map;
    std::vector<Buffer> param_flattened_buffers;
    for (const auto& kv : buffer_map) {
      if (kv.second->layout.defined()) {
        param_flattened_buffers.push_back(storage_lower.GetFlattenedBuffer(kv.second));
        Buffer buffer = kv.second;
        auto* writer = buffer.CopyOnWrite();
        writer->layout = std::nullopt;
        new_buffer_map[kv.first] = buffer;
      } else {
        new_buffer_map[kv.first] = kv.second;
      }
    }
    auto new_stmt = storage_lower(stmt);
    for (const auto& buf : param_flattened_buffers) {
      new_stmt = DeclBuffer(buf, new_stmt);
    }
    return std::make_pair(new_stmt, ffi::Map<Var, Buffer>(new_buffer_map));
  }

 protected:
  using IRMutatorWithAnalyzer::VisitExpr_;
  using IRMutatorWithAnalyzer::VisitStmt_;

  explicit LayoutApplier(arith::Analyzer* analyzer, const Target& target)
      : arith::IRMutatorWithAnalyzer(analyzer), target_(target) {}

  ffi::Any VisitAny(const ffi::Any& any) {
    if (auto buffer = any.as<Buffer>()) {
      return GetFlattenedBuffer(buffer.value());
    } else if (auto prim_expr = any.as<PrimExpr>()) {
      return VisitExpr(prim_expr.value());
    } else if (auto stmt = any.as<Stmt>()) {
      return VisitStmt(stmt.value());
    }
    return any;
  }

  Stmt VisitStmt_(const AllocBufferNode* op) final {
    auto mutate = [this](Buffer buf) {
      if (target_->kind->name == "trn" && !buf->layout.defined()) {
        return buf;
      }
      return GetFlattenedBuffer(buf);
    };
    auto buffer = mutate(op->buffer);
    auto body = VisitStmt(op->body);
    if (buffer.same_as(op->buffer) && body.same_as(op->body)) {
      return ffi::GetRef<Stmt>(op);
    } else {
      auto n = CopyOnWrite(op);
      n->buffer = buffer;
      n->body = body;
      return Stmt(n);
    }
  }

  Stmt VisitStmt_(const AllocBulkGroupEventNode* op) final {
    auto mutate = ([this](BulkGroupEvent event) -> BulkGroupEvent {
      auto* n = event.CopyOnWrite();
      n->state.MutateByApply([this](ffi::Any state) -> ffi::Any { return VisitAny(state); });
      if (n->state.same_as(event->state)) {
        return event;
      } else {
        return ffi::GetRef<BulkGroupEvent>(n);
      }
    });
    auto event = mutate(op->bulk_group_event);
    auto body = VisitStmt(op->body);
    if (event.same_as(op->bulk_group_event) && body.same_as(op->body)) {
      return ffi::GetRef<Stmt>(op);
    } else {
      auto n = CopyOnWrite(op);
      n->bulk_group_event = event;
      n->body = body;
      return Stmt(n);
    }
  }

  Stmt VisitStmt_(const AllocSemaphoreEventTensorNode* op) final {
    auto mutate = ([this](SemaphoreEventTensor event) -> SemaphoreEventTensor {
      auto* n = event.CopyOnWrite();
      n->state.MutateByApply([this](ffi::Any state) -> ffi::Any { return VisitAny(state); });
      if (n->state.same_as(event->state)) {
        return event;
      } else {
        return ffi::GetRef<SemaphoreEventTensor>(n);
      }
    });
    auto event = mutate(op->sem_event_tensor);
    auto body = VisitStmt(op->body);
    if (event.same_as(op->sem_event_tensor) && body.same_as(op->body)) {
      return ffi::GetRef<Stmt>(op);
    } else {
      auto n = CopyOnWrite(op);
      n->sem_event_tensor = event;
      n->body = body;
      return Stmt(n);
    }
  }

  Stmt VisitStmt_(const DeclBufferNode* op) final {
    auto buffer = GetFlattenedBuffer(op->buffer);

    DeclBuffer decl_buffer = ffi::GetRef<DeclBuffer>(op);
    auto n = decl_buffer.CopyOnWrite();
    n->buffer = buffer;

    return StmtExprMutator::VisitStmt_(decl_buffer.get());
  }

  Buffer GetFlattenedBuffer(Buffer buf) {
    auto it = buffer_remap_.find(buf);
    if (it != buffer_remap_.end()) {
      return it->second;
    }
    auto trn_layout = buf->layout.as<TileLayoutNode>();
    Buffer flattened;
    tirx::BufferNode* writer;
    if (trn_layout && trn_layout->IsTrainium()) {
      ffi::Array<PrimExpr> new_shape =
          buf.scope() == "trn.psum"
              ? ffi::Array<PrimExpr>{trn_layout->GetCosize(ffi::String("Bank")),
                                     trn_layout->GetSize(ffi::String("P")),
                                     trn_layout->GetCosize(ffi::String("F"))}
              : ffi::Array<PrimExpr>{trn_layout->GetSize(ffi::String("P")),
                                     trn_layout->GetCosize(ffi::String("F"))};
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
    writer->layout = std::nullopt;
    writer->elem_offset = StmtExprMutator::VisitExpr(buf->elem_offset);

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

  Stmt VisitStmt_(const tirp::OpCallNode* op) final {
    ffi::Array<ffi::Any> args = op->args;
    args.MutateByApply([this](ffi::Any arg) -> ffi::Any { return VisitAny(arg); });
    if (args.same_as(op->args)) {
      return ffi::GetRef<Stmt>(op);
    } else {
      auto n = CopyOnWrite(op);
      n->args = std::move(args);
      return Stmt(n);
    }
  }

  ffi::Array<PrimExpr> GetSimplifiedElemOffset(const Buffer& buffer,
                                               const ffi::Array<PrimExpr>& indices) {
    if (buffer->layout.defined()) {
      auto tile_layout = buffer->layout.value().as<TileLayoutNode>();
      if (tile_layout && tile_layout->IsTrainium()) {
        auto coord = buffer->layout.value()->Apply(indices, buffer->shape);
        std::vector<PrimExpr> res;
        for (const auto& axis : buffer.scope() == "trn.psum"
                                    ? ffi::Array<ffi::String>{"Bank", "P", "F"}
                                    : ffi::Array<ffi::String>{"P", "F"}) {
          auto it = coord.find(ffi::String(axis));
          if (it != coord.end()) {
            res.push_back(analyzer_->Simplify((*it).second));
          } else {
            res.push_back(0);
          }
        }
        return res;
      }
      auto res = buffer->layout.value()->Normalize()->Apply(indices, buffer->shape);
      ICHECK_EQ(res.size(), 1) << "Expected a single element offset";
      return {analyzer_->Simplify((*res.begin()).second)};
    }
    auto flattened_indices = buffer->ElemOffset(indices, true);
    ICHECK_EQ(flattened_indices.size(), 1) << "Expected a single element offset";
    return {analyzer_->Simplify(flattened_indices[0])};
  }

  template <typename Node>
  Node VisitBufferAccess(Node node) {
    ICHECK(node->buffer.defined());
    if (target_->kind->name == "trn" && !node->buffer->layout.defined()) {
      return node;
    }
    auto flattened_indices = GetSimplifiedElemOffset(node->buffer, node->indices);
    Buffer flattened_buffer = GetFlattenedBuffer(node->buffer);
    auto writer = node.CopyOnWrite();
    writer->buffer = flattened_buffer;
    writer->indices = flattened_indices;
    return node;
  }

  /*! \brief Map of buffers being remapped. */
  std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual> buffer_remap_;
  const Target& target_;
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

  Stmt VisitStmt_(const DeclBufferNode* op) {
    auto buffer = op->buffer;
    auto elem_offset = this->VisitExpr(buffer->elem_offset);
    if (elem_offset.same_as(buffer->elem_offset)) {
      return StmtExprMutator::VisitStmt_(op);
    } else {
      auto n_buffer = buffer.CopyOnWrite();
      n_buffer->elem_offset = std::move(elem_offset);
      buffer_remap_[op->buffer] = buffer;
      auto n = CopyOnWrite(op);
      n->buffer = ffi::GetRef<Buffer>(n_buffer);
      n->body = StmtExprMutator::VisitStmt(op->body);
      return StmtExprMutator::VisitStmt_(n.get());
    }
  }

  using StmtExprMutator::VisitExpr_;
  using StmtExprMutator::VisitStmt_;

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    BufferStore store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    store = VisitBufferAccess(store);
    return std::move(store);
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    BufferLoad load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
    load = VisitBufferAccess(load);
    return std::move(load);
  }

  template <typename Node>
  Node VisitBufferAccess(Node node) {
    ICHECK(node->buffer.defined());
    auto it = buffer_remap_.find(node->buffer);
    if (it != buffer_remap_.end()) {
      auto writer = node.CopyOnWrite();
      writer->buffer = it->second;
      return node;
    }
    return node;
  }

  std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual> buffer_remap_;
};

class EventRemover : public StmtExprMutator {
 public:
  static Stmt Remove(const Stmt& stmt) { return EventRemover()(stmt); }

 private:
  Stmt VisitStmt_(const AllocBulkGroupEventNode* op) final { return VisitStmt(op->body); }

  Stmt VisitStmt_(const AllocSemaphoreEventTensorNode* op) final { return VisitStmt(op->body); }
};

namespace transform {
Pass LowerTIRp() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    auto target = f->GetAttr<Target>(tvm::attr::kTarget);
    if (!target.defined()) {
      target = Target::Current(false);
    }

    auto* n = f.CopyOnWrite();
    // Lower ScopeIdDef, resolve the scope ids and replace them with target defined special
    // registers
    // Collect the extents of the ScopeIds for OpCall scheduling
    n->body = ScopeIdDefResolver::Resolve(n->body, target.value());
    // TIRp OpCall Scheduling
    int max_try = 100;
    while (!NoOpCallVerifier::Verify(n->body, false)) {
      n->body = TIRpOpScheduler::LowerOpCalls(n->body, target.value());
      // n->body = ScopeMerger::Merge(n->body);
      if (max_try == 0) {
        LOG(FATAL) << "Failed to lower the TIRp program after " << 100 << " tries: " << f;
        break;
      }
      max_try--;
    }
    n->body = ExecScopeSliceResolver::Resolve(n->body, target.value());

    // Cleanup other TIRp aux data structures
    n->body = ScheduleContextRemover::Remove(n->body);
    std::tie(n->body, n->buffer_map) =
        LayoutApplier::Flatten(n->body, n->buffer_map, target.value());
    n->body = BufferOffsetRemover::Remove(n->body);
    n->body = EventRemover::Remove(n->body);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tirx.LowerTIRp", {});
}

TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.transform.LowerTIRp", LowerTIRp);
});

}  // namespace transform
}  // namespace tirxxxxxxxxxxxx
}  // namespace tvm
