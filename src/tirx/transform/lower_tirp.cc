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

class NoOpCallVerifier : public Verifier<NoOpCallVerifier> {
 public:
  using Verifier::Verifier;

 private:
  using Verifier::Visit;

  void VisitStmt_(const tirp::OpCallNode* obj, ObjectPath path) final {
    Verify(false) << "TIRpError: OpCall at " << path << " is not allowed in TIRp before lowering";
  }
};

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
            if (ExecScope::Create(now->parent).Higher(scope) &&
                ExecScope::Create(scope).Higher(now->cur)) {
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
      n->annotations.Set("scope_id_extent_map", tirp::ScopeExtentMap(id_extent_map_));
      n->annotations.Set("thread_var_map",
                         tirp::ThreadVarMap{{"blockIdx.x", blockIdx}, {"threadIdx.x", threadIdx}});
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

class BarrierToBarrierArray : public StmtExprMutator {
 public:
  static Stmt Convert(const Stmt& stmt) { return BarrierToBarrierArray()(stmt); }

 private:
  Stmt VisitStmt_(const BlockNode* op) final {
    Block block = GetRef<Block>(op);
    auto* n = block.CopyOnWrite();
    std::vector<BarrierArray> barrier_arrays(n->barrier_arrays.begin(), n->barrier_arrays.end());
    for (const auto& barrier : op->barriers) {
      ICHECK(barrier_map_.find(barrier) == barrier_map_.end())
          << "Internal Error: Duplicate barrier found";
      // Create a BarrierArray with a single barrier
      BarrierArray barrier_array(barrier->thread_scope, 1, barrier->name_hint);
      barrier_map_[barrier] = barrier_array;
      barrier_arrays.push_back(std::move(barrier_array));
    }
    n->body = VisitStmt(n->body);
    n->barriers = {};
    n->barrier_arrays = std::move(barrier_arrays);
    return std::move(block);
  }

  Stmt VisitStmt_(const tirp::OpCallNode* op) final {
    tirp::OpCall opcall = GetRef<tirp::OpCall>(op);
    auto* n = opcall.CopyOnWrite();
    n->args.MutateByApply([this](ObjectRef obj) -> ObjectRef {
      const auto* barrier = obj.as<BarrierNode>();
      if (barrier) {
        auto it = barrier_map_.find(GetRef<Barrier>(barrier));
        if (it != barrier_map_.end()) {
          return BarrierArrayElem(it->second, 0);
        }
      }
      return obj;
    });
    return std::move(opcall);
  }

  std::unordered_map<Barrier, BarrierArray, ObjectPtrHash, ObjectPtrEqual> barrier_map_;
};

class CUDABarrierArrayAllocator : public StmtExprMutator {
 public:
  static Stmt Allocate(const Stmt& stmt) { return CUDABarrierArrayAllocator()(stmt); }

 private:
  Stmt VisitStmt_(const BlockNode* op) final {
    Block block = GetRef<Block>(op);
    auto* n = block.CopyOnWrite();
    ICHECK(op->barriers.empty()) << "Internal Error: Barriers are not removed";
    if (!op->barrier_arrays.empty()) {
      std::vector<Stmt> body;
      for (const auto& barrier_array : op->barrier_arrays) {
        CHECK(barrier_array->thread_scope.Is("cta")) << "Internal Error: Unsupported thread scope";
        body.push_back(Evaluate(Call(DataType::Void(), builtin::cuda_barrier_create(),
                                     {StringImm(barrier_array->thread_scope->name),
                                      IntImm(DataType::Int(32), barrier_arr_id_),
                                      IntImm(DataType::Int(32), barrier_array->size)})));
        barrier_arr_map_[barrier_array] = barrier_arr_id_++;
      }
      body.push_back(StmtExprMutator::VisitStmt(n->body));
      n->body = SeqStmt::Flatten(std::move(body));
    } else {
      n->body = StmtExprMutator::VisitStmt(n->body);
    }
    return std::move(block);
  }

  Stmt VisitStmt_(const tirp::OpCallNode* opcall_ptr) final {
    tirp::OpCall opcall = GetRef<tirp::OpCall>(opcall_ptr);
    auto* n = opcall.CopyOnWrite();
    static const auto& barrier_op_map = Op::GetAttrMap<Bool>("TIsBarrierOp");
    if (bool(barrier_op_map.get(opcall_ptr->op, tvm::Bool(false)))) {
      auto barrier = opcall_ptr->args[0].as<BarrierArrayElemNode>();
      ICHECK(barrier) << "Internal Error: Expected BarrierArrayElem";
      auto it = barrier_arr_map_.find(barrier->arr);
      ICHECK(it != barrier_arr_map_.end()) << "Internal Error: Cannot find BarrierArray";
      n->args.push_back(IntImm(DataType::Int(32), it->second));
    }
    return std::move(opcall);
  }

  int barrier_arr_id_{0};
  std::unordered_map<BarrierArray, int, ObjectPtrHash, ObjectPtrEqual> barrier_arr_map_;
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
    // Get the scope_id_extent_map and thread_var_map
    auto it = n->annotations.find("scope_id_extent_map");
    if (it != n->annotations.end()) {
      ICHECK(scope_id_extent_stack_.empty())
          << "Internal Error: scope_id_extent_stack_ is not empty";
      scope_id_extent_stack_.push_back(Downcast<tirp::ScopeExtentMap>((*it).second));
      it = n->annotations.find("thread_var_map");
      ICHECK(it != n->annotations.end()) << "Internal Error: thread_var_map is not found";
      thread_var_stack_.push_back(Downcast<tirp::ThreadVarMap>((*it).second));
    }
    n->body = VisitStmt(n->body);
    // Pop the exec_scope
    if (op->exec_scope.defined()) {
      exec_scope_stack_.pop_back();
    }
    // Pop the scope_id_extent_map and thread_var_map
    if (it != n->annotations.end()) {
      scope_id_extent_stack_.pop_back();
      thread_var_stack_.pop_back();
    }
    return std::move(block);
  }

  Stmt VisitStmt_(const tirp::OpCallNode* op) final {
    static const auto& op_scheduler_map = Op::GetAttrMap<tirp::FOpScheduler>("FOpScheduler");
    ICHECK(op_scheduler_map.count(op->op)) << "Internal Error: Unsupported OpCall " << op->op;
    tirp::FOpScheduler scheduler = op_scheduler_map[op->op];
    ICHECK(!exec_scope_stack_.empty()) << "Internal Error: exec_scope_stack_ is empty";
    ICHECK(!scope_id_extent_stack_.empty()) << "Internal Error: scope_id_extent_stack_ is empty";
    ICHECK(!thread_var_stack_.empty()) << "Internal Error: thread_var_stack_ is empty";
    return scheduler(
        op->op, op->args,
        tirp::ScheduleContext(Target::Current(false), exec_scope_stack_.back(),
                              thread_var_stack_.back(), scope_id_extent_stack_.back()));
  }

  std::vector<ExecScope> exec_scope_stack_;
  std::vector<tirp::ScopeExtentMap> scope_id_extent_stack_;
  std::vector<tirp::ThreadVarMap> thread_var_stack_;
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

namespace transform {
Pass LowerTIRp() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    const auto& target = Target::Current(false);

    auto* n = f.CopyOnWrite();
    // Lower scope ids, resolve the scope ids and replace them with blockIdx / threadIdx
    n->body = ScopeIdDefRemover::Remove(n->body);

    if (target->kind->name == "cuda" && target->kind->default_device_type == kDLCUDA) {
      // CUDA specific lowering passes
      n->body = BarrierToBarrierArray::Convert(n->body);
      n->body = CUDABarrierArrayAllocator::Allocate(n->body);
    }

    // Default Schedule: lower TIRp OpCalls
    while (!NoOpCallVerifier::Verify(n->body, false)) {
      n->body = TIRpOpScheduler::LowerOpCalls(n->body);
      n->body = ScopeMerger::Merge(n->body);
    }
    // Verify that there are no OpCalls in the TIRp
    NoOpCallVerifier::Verify(f, true);

    // Lower other TIRp aux data structures
    n->body = ScheduleContextRemover::Remove(n->body);
    n->body = LogicalTensorRemover::Remove(n->body);
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
