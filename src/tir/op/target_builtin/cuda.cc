
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
 * \file tir/op/target_builtin/cuda.cc
 *
 *  builtin intrinsic operators specific to CUDA target.
 */
#include <tvm/runtime/registry.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/op_attr_types.h>

namespace tvm {
namespace tir {
namespace builtin {

#define TIR_DEFINE_BUILTIN_FUNC(OpName)            \
  const Op& OpName() {                             \
    static const Op& op = Op::Get("tir." #OpName); \
    return op;                                     \
  }                                                \
  TVM_TIR_REGISTER_OP(#OpName)

TIR_DEFINE_BUILTIN_FUNC(tvm_load_matrix_sync)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kReadState));

TIR_DEFINE_BUILTIN_FUNC(tvm_mma_sync)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(tvm_bmma_sync)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(tvm_fill_fragment)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(tvm_store_matrix_sync)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(ptx_mma)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque))
    .set_attr<TScriptDtypePrintLocation>("TScriptDtypePrintLocation",
                                         Integer(ScriptDtypePrintLocation::kFirst));

TIR_DEFINE_BUILTIN_FUNC(ptx_ldg32).set_num_inputs(4).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kPure));

TIR_DEFINE_BUILTIN_FUNC(ptx_mma_sp)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque))
    .set_attr<TScriptDtypePrintLocation>("TScriptDtypePrintLocation",
                                         Integer(ScriptDtypePrintLocation::kFirst));

TIR_DEFINE_BUILTIN_FUNC(ptx_ldmatrix)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque))
    .set_attr<TScriptDtypePrintLocation>("TScriptDtypePrintLocation",
                                         Integer(ScriptDtypePrintLocation::kFirst));

TIR_DEFINE_BUILTIN_FUNC(ptx_cp_async)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque))
    .set_attr<TScriptDtypePrintLocation>("TScriptDtypePrintLocation",
                                         Integer(ScriptDtypePrintLocation::kFirst));

TIR_DEFINE_BUILTIN_FUNC(ptx_cp_async_bulk)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque))
    .set_attr<TScriptDtypePrintLocation>("TScriptDtypePrintLocation",
                                         Integer(ScriptDtypePrintLocation::kFirst));

TIR_DEFINE_BUILTIN_FUNC(ptx_cp_async_commit_group)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(ptx_cp_async_wait_group)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(ptx_cp_async_mbarrier_arrive)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(init_barrier_thread_count)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(arrive_barrier)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(arrive_barrier_expect_tx)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(wait_barrier)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(create_barriers)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(cuda_barrier_create)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(cuda_barrier_init)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(cuda_barrier_arrive)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(cuda_barrier_wait)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(cuda_barrier_arrive_and_wait)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(ptx_fence_proxy)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(ptx_mbarrier_init)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(ptx_mbarrier_arrive)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(ptx_mbarrier_arrive_expect_tx)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(ptx_mbarrier_try_wait)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(ptx_bar_arrive)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(ptx_bar_sync)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(ptx_cp_async_bulk_tensor_global_to_cluster)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(ptx_cp_async_bulk_tensor_shared_to_global)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(ptx_cp_async_bulk_commit_group)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(ptx_cp_async_bulk_wait_group)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(ptx_barrier_cluster_arrive)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(ptx_barrier_cluster_wait)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(ptx_elect_sync)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(ptx_fence_mbarrier_init_release_cluster)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(ptx_fetch_register)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kPure));

TIR_DEFINE_BUILTIN_FUNC(mma_store)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque))
    .set_attr<TScriptDtypePrintLocation>("TScriptDtypePrintLocation",
                                         Integer(ScriptDtypePrintLocation::kFirst));

TIR_DEFINE_BUILTIN_FUNC(mma_fill)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque))
    .set_attr<TScriptDtypePrintLocation>("TScriptDtypePrintLocation",
                                         Integer(ScriptDtypePrintLocation::kFirst));

TIR_DEFINE_BUILTIN_FUNC(ptx_encode_matrix_descriptor)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(ptx_wgmma_noop_barrier)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(ptx_wgmma_mma_async_ss)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(ptx_wgmma_mma_async_rs)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(ptx_wgmma_fence)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(ptx_wgmma_commit_group)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(ptx_wgmma_wait_group)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(ptx_stmatrix)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(ptx_setmaxnreg)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

}  // namespace builtin
}  // namespace tir
}  // namespace tvm
