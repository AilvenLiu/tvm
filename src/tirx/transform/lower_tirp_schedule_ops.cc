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
 * \file lower_tirp_schedule_ops.cc
 * \brief Lower TIRp OpCall nodes via registered schedulers.
 */

#include <tvm/runtime/logging.h>
#include <tvm/target/target.h>
#include <tvm/tirx/exec_scope.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/tirp_op.h>
#include <tvm/tirx/transform.h>

#include <unordered_map>
#include <utility>
#include <vector>

#include "../ir/functor_common.h"
#include "../ir/tir_visitor_with_path.h"

namespace tvm {
namespace tirx {

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
  explicit TIRpOpScheduler(const Target& target) : target_(target) {}

  static Stmt LowerOpCalls(const Stmt& stmt, const Target& target) {
    return TIRpOpScheduler(target)(stmt);
  }

 private:
  class KernelReplacePointSearcher : public StmtExprMutator {
   public:
    explicit KernelReplacePointSearcher(const Stmt& body) : body_(body) {}

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
      // Implementation found, handle callbacks
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

namespace {
Target ResolveTarget(const PrimFunc& f) {
  auto target = f->GetAttr<Target>(tvm::attr::kTarget);
  if (!target.defined()) {
    target = Target::Current(false);
  }
  return target.value();
}
}  // namespace

namespace transform {

Pass LowerTIRpScheduleOps() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    Target target = ResolveTarget(f);
    auto* n = f.CopyOnWrite();
    n->body = TIRpOpScheduler::LowerOpCalls(n->body, target);
    if (!NoOpCallVerifier::Verify(n->body, false)) {
      LOG(FATAL) << "Failed to lower the TIRp program: " << f;
    }
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tirx.LowerTIRpScheduleOps", {});
}

}  // namespace transform
}  // namespace tirx
}  // namespace tvm
