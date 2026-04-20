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
 * \file lower_tirx_dispatch_ops.cc
 * \brief Lower TIRx TilePrimitiveCall nodes via registered dispatchers.
 */

#include <tvm/runtime/logging.h>
#include <tvm/target/target.h>
#include <tvm/tirx/exec_scope.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/tirx_op.h>
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

  void VisitStmt_(const tirx::TilePrimitiveCallNode* obj, ffi::reflection::AccessPath path) final {
    Verify(false) << "TIRxError: TilePrimitiveCall at " << path
                  << " is not allowed in TIRx before lowering";
  }
};

class TIRxOpDispatcher : public StmtExprMutator {
 public:
  explicit TIRxOpDispatcher(const Target& target) : target_(target) {}

  static Stmt LowerOpCalls(const Stmt& stmt, const Target& target) {
    return TIRxOpDispatcher(target)(stmt);
  }

 private:
  class BufferRefRewriter : public StmtExprMutator {
   public:
    static Stmt Rewrite(const Stmt& stmt, const Buffer& src, const Buffer& dst) {
      if (src.same_as(dst)) {
        return stmt;
      }
      return BufferRefRewriter(src, dst)(stmt);
    }

   private:
    BufferRefRewriter(Buffer src, Buffer dst) : src_(std::move(src)), dst_(std::move(dst)) {}

    Buffer VisitBufferDef(const Buffer& buffer, bool alloc_data) final {
      Buffer new_buffer = StmtExprMutator::VisitBufferDef(buffer, alloc_data);
      if (new_buffer.same_as(src_)) {
        return dst_;
      }
      return new_buffer;
    }

    Buffer VisitBufferUse(const Buffer& buffer) final {
      if (buffer.same_as(src_)) {
        return dst_;
      }
      return StmtExprMutator::VisitBufferUse(buffer);
    }

    Buffer src_;
    Buffer dst_;
  };

  class KernelReplacePointSearcher : public StmtExprMutator {
   public:
    explicit KernelReplacePointSearcher(const Stmt& body) : body_(body) {}

    static Stmt Seek(const Stmt& stmt, const Stmt& body) {
      return KernelReplacePointSearcher(body)(stmt);
    }

   private:
    Stmt VisitStmt_(const tirx::TilePrimitiveCallNode* op) final {
      if (op->op == tirx::tvm_kernel_replace_point()) {
        return body_;
      }
      return StmtExprMutator::VisitStmt_(op);
    }

    Stmt body_;
  };

  Stmt VisitStmt_(const ExecScopeStmtNode* op) final {
    exec_scope_stack_.push_back(op->exec_scope);
    bool is_kernel = op->exec_scope->Is("kernel");
    bool is_first_block = false;
    if (is_kernel) {
      std::swap(is_first_block, is_first_block_);
    }
    Stmt body = VisitStmt(op->body);
    if (is_kernel && is_first_block) {
      // Insert device init stmts into kernel body
      for (const auto& stmt : device_init_stmts_) {
        body = KernelReplacePointSearcher::Seek(stmt, body);
      }
      // Insert alloc buffers at the beginning of the kernel body.
      if (!alloc_buffers_.empty()) {
        std::vector<Stmt> seq;
        seq.reserve(alloc_buffers_.size() + 1);
        for (const auto& buffer : alloc_buffers_) {
          seq.push_back(tvm::tirx::AllocBuffer(buffer));
        }
        seq.push_back(std::move(body));
        body = SeqStmt::Flatten(seq);
      }
      alloc_buffers_.clear();
      Stmt res = ExecScopeStmt(op->exec_scope, body);
      if (is_first_thread_attr_) {
        // Insert host init stmts outside the outermost thread binding or block
        for (const auto& stmt : host_init_stmts_) {
          res = KernelReplacePointSearcher::Seek(stmt, std::move(res));
        }
        host_init_stmts_.clear();
      }
      std::swap(is_first_block, is_first_block_);
      exec_scope_stack_.pop_back();
      return res;
    }
    exec_scope_stack_.pop_back();
    if (body.same_as(op->body)) {
      return ffi::GetRef<Stmt>(op);
    }
    return ExecScopeStmt(op->exec_scope, body);
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
        host_init_stmts_.clear();
      }
      std::swap(is_first_thread_attr, is_first_thread_attr_);
      return res;
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const SeqStmtNode* op) final {
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    if (post_buffer_def_stmts_.empty()) {
      return stmt;
    }
    const auto* seq = stmt.as<SeqStmtNode>();
    if (seq == nullptr) {
      return stmt;
    }

    std::vector<Stmt> rebuilt;
    rebuilt.reserve(seq->seq.size() + post_buffer_def_stmts_.size());
    bool changed = false;
    for (const Stmt& s : seq->seq) {
      rebuilt.push_back(s);
      if (const auto* alloc = s.as<AllocBufferNode>()) {
        changed |= AppendPostBufferDefStmts(&rebuilt, alloc->buffer, alloc->buffer);
      } else if (const auto* decl = s.as<DeclBufferNode>()) {
        changed |= AppendPostBufferDefStmts(&rebuilt, decl->buffer, decl->buffer);
      }
    }
    if (!changed) {
      return stmt;
    }
    return SeqStmt::Flatten(rebuilt);
  }

  Stmt VisitStmt_(const ForNode* op) final {
    // Collect the loop variables
    auto loop_var = Downcast<Var>(op->loop_var);
    TVM_FFI_ICHECK(!var_range_map_.count(loop_var)) << "Internal Error: Duplicate loop variable";
    var_range_map_.Set(loop_var, Range::FromMinExtent(op->min, op->extent));
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const AllocBufferNode* op) final {
    Buffer old_buffer = op->buffer;
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    op = stmt.as<AllocBufferNode>();
    TVM_FFI_ICHECK(op);

    std::vector<Stmt> seq{stmt};
    AppendPostBufferDefStmts(&seq, old_buffer, op->buffer);
    return SeqStmt::Flatten(seq);
  }

  Stmt VisitStmt_(const DeclBufferNode* op) final {
    Buffer old_buffer = op->buffer;
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    op = stmt.as<DeclBufferNode>();
    TVM_FFI_ICHECK(op);

    std::vector<Stmt> seq{stmt};
    AppendPostBufferDefStmts(&seq, old_buffer, op->buffer);
    return SeqStmt::Flatten(seq);
  }

  Stmt VisitStmt_(const tirx::TilePrimitiveCallNode* op) final {
    tirx::DispatchContext sctx(target_, exec_scope_stack_.back(), launch_params_, var_range_map_,
                               /*alloc_only=*/false, /*callbacks=*/{}, shared_state_);
    static auto f_op_dispatcher_ = ffi::Function::GetGlobal("tirx.f_op_dispatcher");
    TVM_FFI_ICHECK(f_op_dispatcher_.has_value())
        << "Internal Error: tirx.f_op_dispatcher is not registered";
    PrimFunc res =
        f_op_dispatcher_.value()(ffi::GetRef<tirx::TilePrimitiveCall>(op), sctx).cast<PrimFunc>();
    TVM_FFI_ICHECK(res.defined()) << "TIRx dispatcher did not return a PrimFunc";
    // Implementation found, handle callbacks
    if (auto bufs = sctx->callbacks.Get(tirx::callback::kPrivateAlloc)) {
      auto buf_list = bufs.value().as<Array<Buffer>>().value();
      alloc_buffers_.insert(alloc_buffers_.end(), buf_list.begin(), buf_list.end());
    }
    if (auto stmts = sctx->callbacks.Get(tirx::callback::kDeviceInitStmt)) {
      auto stmt_list = stmts.value().as<Array<Stmt>>().value();
      device_init_stmts_.insert(device_init_stmts_.end(), stmt_list.begin(), stmt_list.end());
    }
    if (auto stmts = sctx->callbacks.Get(tirx::callback::kHostInitStmt)) {
      auto stmt_list = stmts.value().as<Array<Stmt>>().value();
      host_init_stmts_.insert(host_init_stmts_.end(), stmt_list.begin(), stmt_list.end());
    }
    if (auto mapping = sctx->callbacks.Get(tirx::callback::kPostBufferDefStmt)) {
      auto map = Downcast<ffi::Map<Buffer, Array<Stmt>>>(mapping.value());
      for (const auto& [buffer, stmts] : map) {
        auto& vec = post_buffer_def_stmts_[buffer];
        vec.insert(vec.end(), stmts.begin(), stmts.end());
      }
    }
    // Propagate shared_state changes back (Map uses COW semantics)
    shared_state_ = sctx->shared_state;
    return res->body;
  }

  ffi::Map<Var, Range> var_range_map_;
  const Target& target_;
  std::vector<ExecScope> exec_scope_stack_;
  std::unordered_map<ffi::String, IterVar> launch_params_;
  std::vector<Buffer> alloc_buffers_;
  std::vector<Stmt> device_init_stmts_;
  std::vector<Stmt> host_init_stmts_;
  std::unordered_map<Buffer, std::vector<Stmt>, ObjectPtrHash, ObjectPtrEqual>
      post_buffer_def_stmts_;
  ffi::Map<ffi::String, ObjectRef> shared_state_;

  bool is_first_block_{true};
  bool is_first_thread_attr_{true};

  bool AppendPostBufferDefStmts(std::vector<Stmt>* seq, const Buffer& old_buffer,
                                const Buffer& new_buffer) {
    auto append_with_remap = [this, seq, &new_buffer](auto it) -> bool {
      Buffer src = it->first;
      for (const auto& stmt : it->second) {
        Stmt remapped = BufferRefRewriter::Rewrite(stmt, src, new_buffer);
        seq->push_back(KernelReplacePointSearcher::Seek(remapped, Evaluate(0)));
      }
      post_buffer_def_stmts_.erase(it);
      return true;
    };

    bool changed = false;
    if (auto it = post_buffer_def_stmts_.find(old_buffer); it != post_buffer_def_stmts_.end()) {
      changed |= append_with_remap(it);
    }
    if (!new_buffer.same_as(old_buffer)) {
      if (auto it = post_buffer_def_stmts_.find(new_buffer); it != post_buffer_def_stmts_.end()) {
        changed |= append_with_remap(it);
      }
    }
    return changed;
  }

  // No failure aggregation; pass surfaces per-op exceptions
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
        if (auto* exec_scope_stmt = n->seq[i].as<ExecScopeStmtNode>()) {
          // Find a sequence of ExecScopeStmts with the same exec_scope
          std::vector<Stmt> new_body{exec_scope_stmt->body};
          auto scope = exec_scope_stmt->exec_scope;
          for (i++; i < n->seq.size(); i++) {
            if (auto* next_exec_scope = n->seq[i].as<ExecScopeStmtNode>()) {
              if (scope->Is(next_exec_scope->exec_scope)) {
                new_body.push_back(next_exec_scope->body);
                continue;
              }
            }
            break;
          }
          seq.push_back(ExecScopeStmt(scope, SeqStmt::Flatten(new_body)));
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

Pass LowerTIRxDispatchOps() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    Target target = ResolveTarget(f);
    auto* n = f.CopyOnWrite();
    n->body = TIRxOpDispatcher::LowerOpCalls(n->body, target);
    if (!NoOpCallVerifier::Verify(n->body, false)) {
      LOG(FATAL) << "Failed to lower the TIRx program: " << f;
    }
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tirx.LowerTIRxDispatchOps", {});
}

}  // namespace transform
}  // namespace tirx
}  // namespace tvm
