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
 * \file tile_primitive_dispatch.cc
 * \brief Lower TilePrimitiveCall nodes via registered dispatchers (also resolves ScopeIdDef
 * declarations and emits launch params).
 */

#include <tvm/arith/analyzer.h>
#include <tvm/runtime/logging.h>
#include <tvm/target/target.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/exec_context.h>
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

namespace {

// Gather every ScopeIdDef declared anywhere under a given Stmt, paired with
// the name of the ExecScope that declared it (for implicit-eval routing).
struct ScopeIdDefWithSource {
  ScopeIdDef def;
  ffi::String source_scope;
};

class ScopeIdDefGather : public StmtExprVisitor {
 public:
  static std::vector<ScopeIdDefWithSource> Gather(const Stmt& stmt) {
    ScopeIdDefGather gather;
    gather(stmt);
    return std::move(gather.out_);
  }

  void VisitStmt_(const ExecScopeStmtNode* op) override {
    StmtExprVisitor::VisitStmt_(op);
    for (const auto& def : op->exec_scope->scope_id_def) {
      out_.push_back({def, op->exec_scope->name()});
    }
  }

 private:
  std::vector<ScopeIdDefWithSource> out_;
};

// Strip ``scope_id_def`` arrays off every nested ExecScopeStmt; the resolved
// values are bound at kernel scope via Bind statements emitted separately.
class ScopeIdDefRemover : public StmtExprMutator {
 public:
  static Stmt Remove(const Stmt& stmt) { return ScopeIdDefRemover()(stmt); }

  Stmt VisitStmt_(const ExecScopeStmtNode* op) override {
    Stmt body = StmtExprMutator::VisitStmt(op->body);
    auto n_scope = ffi::make_object<ExecScopeNode>(*op->exec_scope.as<ExecScopeNode>());
    n_scope->scope_id_def = {};
    return ExecScopeStmt(ExecScope(n_scope), body);
  }
};

// For implicitly-named ScopeIdDefs (parser-emitted Var("")), inject an
// Evaluate(var) at the source scope so the binding stays observably live in
// the IR even if user code never references it.
class ImplicitScopeIdEvalInjector : public StmtExprMutator {
 public:
  static Stmt Inject(const Stmt& stmt, const std::vector<std::pair<Var, ffi::String>>& eval_specs) {
    ImplicitScopeIdEvalInjector injector(eval_specs);
    return injector(stmt);
  }

 private:
  explicit ImplicitScopeIdEvalInjector(const std::vector<std::pair<Var, ffi::String>>& eval_specs) {
    for (const auto& [var, scope] : eval_specs) {
      eval_map_[scope.operator std::string()].push_back(var);
    }
  }

  Stmt VisitStmt_(const ExecScopeStmtNode* op) final {
    Stmt body = VisitStmt(op->body);
    auto it = eval_map_.find(op->exec_scope->name().operator std::string());
    if (it != eval_map_.end() && !it->second.empty()) {
      ffi::Array<Stmt> evals;
      evals.reserve(it->second.size());
      for (const Var& var : it->second) {
        evals.push_back(Evaluate(var));
      }
      body = SeqStmt::Flatten(evals, body);
      eval_map_.erase(it);
    }
    if (body.same_as(op->body)) return ffi::GetRef<Stmt>(op);
    return ExecScopeStmt(op->exec_scope, body);
  }

  std::unordered_map<std::string, std::vector<Var>> eval_map_;
};

}  // namespace

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

class TilePrimitiveDispatcher : public StmtExprMutator {
 public:
  explicit TilePrimitiveDispatcher(const Target& target) : target_(target) {}

  static Stmt LowerOpCalls(const Stmt& stmt, const Target& target) {
    return TilePrimitiveDispatcher(target)(stmt);
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
    bool is_kernel = op->exec_scope->kind == ScopeKind::kKernel;
    bool is_first_block = false;
    if (is_kernel) {
      std::swap(is_first_block, is_first_block_);
    }

    // Per-kernel scope-id resolution state. Populated at kernel entry,
    // consumed at kernel exit to emit Bind / thread_extent / implicit evals.
    std::vector<std::pair<Var, PrimExpr>> scope_binds;
    std::vector<std::pair<Var, ffi::String>> implicit_scope_id_evals;

    bool pushed_ctx = false;
    if (is_kernel) {
      // Resolve scope-ids: gather, verify, populate launch_params_, build
      // scope_binds. After this, launch_params_ has threadIdx / blockIdx /
      // clusterCtaIdx IterVars derivable from the user's ScopeIdDefs.
      // launch_params_ is cleared first since it accumulates across kernels.
      launch_params_.clear();
      ResolveKernelScopeIds(op, &scope_binds, &implicit_scope_id_evals);
      pushed_ctx = PushKernelEntryCtx();
    } else {
      pushed_ctx = PushScopeSwitchCtx(op->exec_scope->kind);
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

      // Strip scope_id_def from inner ExecScopeStmts -- their values are now
      // bound at kernel scope via the Bind statements below.
      res = ScopeIdDefRemover::Remove(res);

      // Prepend Bind(var, value) for every resolved scope id (and the derived
      // warp_id_in_cta var when threadIdx is present).
      ffi::Array<Stmt> bind_stmts;
      bind_stmts.reserve(scope_binds.size());
      for (const auto& [var, value] : scope_binds) {
        bind_stmts.push_back(Bind(var, value));
      }
      res = SeqStmt::Flatten(bind_stmts, res);

      // Wrap with thread_extent attrs (consumed by downstream codegen
      // passes that expect TVM-standard thread launch annotations).
      for (const auto& [tag, iv] : launch_params_) {
        if (tag == "warp_id_in_cta") continue;
        res = AttrStmt(iv, tirx::attr::thread_extent, iv->dom->extent, res);
      }
      // Inject implicit scope-id evals (parser-emitted unnamed Vars).
      res = ImplicitScopeIdEvalInjector::Inject(res, implicit_scope_id_evals);

      // Insert host init stmts outside the outermost thread binding or block.
      if (is_first_thread_attr_) {
        for (const auto& stmt : host_init_stmts_) {
          res = KernelReplacePointSearcher::Seek(stmt, std::move(res));
        }
        host_init_stmts_.clear();
      }
      std::swap(is_first_block, is_first_block_);
      exec_scope_stack_.pop_back();
      if (pushed_ctx) ctx_stack_.pop_back();
      return res;
    }
    exec_scope_stack_.pop_back();
    if (pushed_ctx) ctx_stack_.pop_back();
    if (body.same_as(op->body)) {
      return ffi::GetRef<Stmt>(op);
    }
    return ExecScopeStmt(op->exec_scope, body);
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

  Stmt VisitStmt_(const IfThenElseNode* op) final {
    // Handle `if Tx.filter(var, lo, hi): ...` / `if Tx.filter(var, cond): ...`:
    //  1. Narrow ExecContext on the then-branch (filter is one-sided -- the
    //     else-branch retains the outer active set).
    //  2. Rewrite the filter Call into a plain PrimExpr predicate so the
    //     downstream IR no longer carries the tirx.filter builtin.
    bool pushed_ctx = false;
    PrimExpr new_cond = op->condition;
    if (const auto* call = op->condition.as<CallNode>()) {
      if (call->op.same_as(tirx::builtin::filter())) {
        pushed_ctx = TryPushFilterCtx(call);
        TVM_FFI_ICHECK(call->args.size() == 2 || call->args.size() == 3)
            << "TIRxError: tirx.filter expects (var, lo, hi) or (var, cond); got "
            << call->args.size() << " args";
        PrimExpr var = call->args[0];
        new_cond = (call->args.size() == 3)
                       ? PrimExpr((var >= call->args[1]) && (var < call->args[2]))
                       : call->args[1];
      }
    }
    Stmt then_case = VisitStmt(op->then_case);
    if (pushed_ctx) ctx_stack_.pop_back();
    ffi::Optional<Stmt> else_case;
    if (op->else_case.defined()) {
      else_case = VisitStmt(op->else_case.value());
    }
    bool unchanged = new_cond.same_as(op->condition) && then_case.same_as(op->then_case) &&
                     ((!op->else_case.defined() && !else_case.defined()) ||
                      (op->else_case.defined() && else_case.defined() &&
                       else_case.value().same_as(op->else_case.value())));
    if (unchanged) return ffi::GetRef<Stmt>(op);
    return IfThenElse(new_cond, then_case, else_case);
  }

  Stmt VisitStmt_(const tirx::TilePrimitiveCallNode* op) final {
    ffi::Map<ffi::String, ffi::Array<IntImm>> inter_map, intra_map;
    // scope_kind always equals the current exec_scope name so dispatchers
    // can read sctx.scope_kind as a drop-in for sctx.exec_scope.name. When
    // ExecContext tracking is active the tracked scope_kind wins (identical for
    // legacy kinds and consistent once filters change the active set).
    ffi::String scope_kind = exec_scope_stack_.back()->name();
    if (!ctx_stack_.empty()) {
      const auto& ctx = ctx_stack_.back();
      inter_map = EncodeSplitSide(ctx.split.inter);
      intra_map = EncodeSplitSide(ctx.split.intra);
      scope_kind = ScopeKindToString(ctx.scope_kind);
    }
    tirx::DispatchContext sctx(target_, exec_scope_stack_.back(), launch_params_, var_range_map_,
                               /*alloc_only=*/false, /*callbacks=*/{}, shared_state_, inter_map,
                               intra_map, scope_kind);
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

  // --- Scope-id resolution at kernel scope ----------------------------------

  // Gather + verify ScopeIdDefs, build launch_params_ from the canonical
  // bindings, and append (Var, value) pairs to *scope_binds. Implicit
  // (unnamed) scope-id Vars are recorded for later evaluate-injection.
  void ResolveKernelScopeIds(const ExecScopeStmtNode* op,
                             std::vector<std::pair<Var, PrimExpr>>* scope_binds,
                             std::vector<std::pair<Var, ffi::String>>* implicit_scope_id_evals) {
    std::vector<ScopeIdDefWithSource> gathered = ScopeIdDefGather::Gather(ffi::GetRef<Stmt>(op));
    Array<ScopeIdDef> defs;
    defs.reserve(gathered.size());
    for (const auto& g : gathered) defs.push_back(g.def);

    ScopeIdDefVerifier verifier;
    TVM_FFI_ICHECK(verifier.Verify(defs)) << "Inconsistent ScopeIdDef";

    ExtractKernelLaunchParams(verifier.id_set);

    // Synthesize the warp_id_in_cta helper (CUDA only) when threadIdx is set.
    if (launch_params_.count("threadIdx.x") > 0) {
      PrimExpr shuffled = ScopeIdResolve::ComputeWarpIdInCta(launch_params_);
      Var warp_id_in_cta_var("warp_id_in_cta", shuffled.dtype());
      scope_binds->push_back({warp_id_in_cta_var, shuffled});
      IterVar warp_iv(Range::FromMinExtent(0, 1), warp_id_in_cta_var, kThreadIndex,
                      "warp_id_in_cta");
      launch_params_.insert({"warp_id_in_cta", warp_iv});
    }

    auto is_implicit = [](const Var& v) { return v->name_hint.empty(); };
    for (const auto& g : gathered) {
      const ScopeIdDef& def = g.def;
      auto resolved = ScopeIdResolve::Resolve(def->scope, def->extents, def->extents.size(),
                                              target_->kind->name, launch_params_);
      TVM_FFI_ICHECK_EQ(resolved.size(), def->extents.size())
          << "Internal Error: Inconsistent resolved size";
      for (size_t i = 0; i < def->def_ids.size(); i++) {
        // Reuse the original Var as the bind target -- no rename, no
        // substitution. The IR already references this Var directly, and
        // dispatch's filter resolution walks ExecScopeStmt::scope_id_def
        // to map Vars back to their ScopeBinding.
        Var bind_var = def->def_ids[i];
        PrimExpr value = resolved[i];
        if (bind_var->dtype != value.dtype()) {
          value = Cast(bind_var->dtype, value);
        }
        scope_binds->push_back({bind_var, value});
        if (is_implicit(bind_var)) {
          implicit_scope_id_evals->push_back({bind_var, g.source_scope});
        }
      }
    }
  }

  // Translate the canonical ScopeBinding -> launch param IterVars
  // (blockIdx.{x,y,z}, clusterCtaIdx.*, threadIdx.{x,y,z}, etc.).
  void ExtractKernelLaunchParams(const ScopeIdDefVerifier::ScopeIdSet& id_set) {
    auto add_launch_param = [&](ScopeBinding binding, const std::string& prefix) {
      auto it = id_set.find(binding);
      if (it == id_set.end()) return;
      const auto& def = (*it).second;
      TVM_FFI_ICHECK_LE(def->extents.size(), 3) << "ValueError: Only up to 3 extents are supported";
      for (size_t i = 0; i < def->extents.size(); i++) {
        std::string thread_tag = prefix + static_cast<char>('x' + i);
        IterVar iv(Range::FromMinExtent(0, def->extents[i]), Var(thread_tag),
                   IterVarType::kThreadIndex, thread_tag);
        launch_params_.insert({ffi::String(thread_tag), iv});
      }
    };
    auto cluster_cta_it = id_set.find(ScopeBinding::kClusterCta);
    if (cluster_cta_it == id_set.end() || is_one((*cluster_cta_it).second.fused_extent())) {
      // no cluster
      add_launch_param(ScopeBinding::kKernelCta, "blockIdx.");
    } else {
      // use cluster
      TVM_FFI_ICHECK(target_->kind->name == "cuda")
          << "ValueError: cluster is only supported in CUDA";
      TVM_FFI_ICHECK_EQ(target_->kind->default_device_type, kDLCUDA)
          << "ValueError: cluster is only supported in CUDA";
      add_launch_param(ScopeBinding::kClusterCta, "clusterCtaIdx.");
      // Preferred cluster size (CUDA 12.8+)
      const auto& cta_def = (*cluster_cta_it).second;
      if (cta_def->preferred_extents.defined()) {
        const auto& pref = cta_def->preferred_extents.value();
        for (size_t i = 0; i < pref.size(); i++) {
          std::string tag = "preferredClusterCtaIdx." + std::string(1, 'x' + i);
          IterVar iv(Range::FromMinExtent(0, pref[i]), Var(tag), IterVarType::kThreadIndex, tag);
          launch_params_.insert({ffi::String(tag), iv});
        }
      }
      add_launch_param(ScopeBinding::kKernelCta, "blockIdx.");
    }
    add_launch_param(ScopeBinding::kCtaThread, "threadIdx.");
    if (!id_set.empty()) {
      TVM_FFI_ICHECK(launch_params_.count("threadIdx.x") > 0)
          << "ValueError: kernel has no thread launch parameters. "
          << "At minimum, declare cta->thread extent (e.g., Tx.thread_id([128]))";
    }
  }

  // --- ExecContext tracking helpers -----------------------------------------

  bool PushKernelEntryCtx() {
    auto prod_extent = [&](std::initializer_list<const char*> keys) -> int64_t {
      int64_t n = 1;
      for (const char* k : keys) {
        auto it = launch_params_.find(ffi::String(k));
        if (it == launch_params_.end()) continue;
        const auto* imm = it->second->dom->extent.as<IntImmNode>();
        if (imm == nullptr) return 0;  // symbolic
        n *= imm->value;
      }
      return n;
    };
    int64_t thread_ext = prod_extent({"threadIdx.x", "threadIdx.y", "threadIdx.z"});
    if (thread_ext <= 0) {
      // launch params missing or symbolic; ExecContext tracking is not
      // available for this kernel. Dispatchers fall back to scope_kind only.
      LOG(WARNING) << "ExecContext tracking disabled: missing/symbolic threadIdx extents";
      return false;
    }
    int64_t warp_ext = thread_ext / 32;
    int64_t cta_ext = prod_extent({"clusterCtaIdx.x", "clusterCtaIdx.y", "clusterCtaIdx.z"});
    if (cta_ext == 0) cta_ext = 1;  // no cluster => single-CTA view
    ctx_stack_.push_back(ExecContext::AtKernelEntry(/*lane_ext=*/32, warp_ext, cta_ext));
    return true;
  }

  bool PushScopeSwitchCtx(ScopeKind new_scope_kind) {
    if (ctx_stack_.empty()) return false;
    ExecContext new_ctx;
    std::string err;
    if (!ctx_stack_.back().WithScopeSwitch(new_scope_kind, &new_ctx, &err)) {
      // Factoring failure (e.g. warpgroup case 3 / world scope_switch).
      // Pause tracking; dispatchers fall back to scope_kind. The verifier
      // (VerifyTIRxWellFormed) is responsible for catching this earlier.
      LOG(WARNING) << "ExecContext scope_switch failed: " << err;
      return false;
    }
    ctx_stack_.push_back(new_ctx);
    return true;
  }

  bool TryPushFilterCtx(const CallNode* call) {
    // Form 1: filter(var, lo, hi). Form 2: filter(var, cond_expr) -- cond_expr
    // is only resolvable when it's var==k. We support form 1 and var==k.
    if (ctx_stack_.empty()) return false;
    if (call->args.size() < 2 || call->args.size() > 3) return false;
    const auto* var_node = call->args[0].as<VarNode>();
    if (var_node == nullptr) return false;
    Var var = ffi::GetRef<Var>(var_node);
    // Look up Var -> ScopeBinding by walking enclosing ExecScopeStmts'
    // scope_id_def. The resolver kept the original Var identity (no rename),
    // so a Var-equality match is enough.
    ffi::Optional<ScopeBinding> binding_opt;
    for (auto it = exec_scope_stack_.rbegin();
         it != exec_scope_stack_.rend() && !binding_opt.has_value(); ++it) {
      for (const auto& def : (*it)->scope_id_def) {
        // Single-factor only -- multi-factor declarations like T.cta_id([3,4,5])
        // can't be narrowed by a single contiguous range.
        if (def->def_ids.size() != 1) continue;
        if (def->def_ids[0].same_as(var)) {
          binding_opt = def->scope;
          break;
        }
      }
    }
    if (!binding_opt.has_value()) return false;
    ScopeBinding binding = binding_opt.value();

    int64_t lo = 0, hi = 0;
    if (call->args.size() == 3) {
      const auto* lo_imm = call->args[1].as<IntImmNode>();
      const auto* hi_imm = call->args[2].as<IntImmNode>();
      if (lo_imm == nullptr || hi_imm == nullptr) return false;
      lo = lo_imm->value;
      hi = hi_imm->value;
    } else {
      // filter(var, cond). Only handle EQ(var, const).
      const auto* eq = call->args[1].as<EQNode>();
      if (eq == nullptr) return false;
      const IntImmNode* k = nullptr;
      if (eq->a.same_as(call->args[0])) {
        k = eq->b.as<IntImmNode>();
      } else if (eq->b.same_as(call->args[0])) {
        k = eq->a.as<IntImmNode>();
      }
      if (k == nullptr) return false;
      lo = k->value;
      hi = k->value + 1;
    }

    ExecContext new_ctx;
    std::string err;
    if (!ctx_stack_.back().WithFilter(binding, lo, hi, &new_ctx, &err)) {
      return false;
    }
    ctx_stack_.push_back(new_ctx);
    return true;
  }

  ffi::Map<Var, Range> var_range_map_;
  const Target& target_;
  std::vector<ExecScope> exec_scope_stack_;
  std::vector<ExecContext> ctx_stack_;
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
              if (scope->kind == next_exec_scope->exec_scope->kind) {
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

Pass TilePrimitiveDispatch() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    Target target = ResolveTarget(f);
    auto* n = f.CopyOnWrite();
    n->body = TilePrimitiveDispatcher::LowerOpCalls(n->body, target);
    if (!NoOpCallVerifier::Verify(n->body, false)) {
      LOG(FATAL) << "Failed to lower the TIRx program: " << f;
    }
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tirx.TilePrimitiveDispatch", {});
}

}  // namespace transform
}  // namespace tirx
}  // namespace tvm
