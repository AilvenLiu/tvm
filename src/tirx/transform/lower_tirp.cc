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
#include <tvm/runtime/registry.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <unordered_map>

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
    Block res = Downcast<Block>(StmtExprMutator::VisitStmt_(op));
    auto* n = res.CopyOnWrite();
    n->buffer_views.clear();
    n->buffer_gets.clear();
    return std::move(res);
  }

  Stmt VisitStmt_(const BufferStoreNode* op) override {
    BufferStore store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    auto it = replace_map_.find(op->buffer);
    if (it != replace_map_.end()) {
      auto* n = store.CopyOnWrite();
      n->buffer = it->second;
    }
    return std::move(store);
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) override {
    BufferLoad load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
    auto it = replace_map_.find(op->buffer);
    if (it != replace_map_.end()) {
      auto* n = load.CopyOnWrite();
      n->buffer = it->second;
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
  Stmt VisitStmt_(const AttrStmtNode* op) override {
    return StmtExprMutator::VisitStmt(op->body);
  }
};

namespace transform {

Pass LowerTIRp() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    n->body = LogicalTensorRemover::Remove(n->body);
    n->body = ScopeIdDefRemover::Remove(n->body);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tirx.LowerTIRp", {});
}

TVM_REGISTER_GLOBAL("tirx.transform.LowerTIRp").set_body_typed(LowerTIRp);

}  // namespace transform
}  // namespace tirx
}  // namespace tvm
