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
 * \file src/runtime/vm/attn_utils.h
 * \brief Data structure and utilities for KV cache.
 */

#ifndef TVM_RUNTIME_VM_MEGAKERNEL_UTILS_H_
#define TVM_RUNTIME_VM_MEGAKERNEL_UTILS_H_

#include <tvm/runtime/tensor.h>

namespace tvm {
namespace runtime {
namespace vm {

inline int ceildiv(int a, int b) { return (a + b - 1) / b; }

namespace megakernel {

constexpr const int kSplitKReduceTileNUnit = 128;
constexpr const int kSplitKReduceTileNRepeat = 1;
constexpr const int kGemmTileBlkN = 128;
constexpr const int kGemmTileBlkK = 64;
constexpr const int kSplitQKVProject[] = {-1, 3, -1, -1, 4, -1, -1, -1, 4};
constexpr const int kSplitOProject[] = {-1, 3, -1, -1, 2, -1, -1, -1, 2};
constexpr const int kGateUpProjSplitKFactor[] = {-1, 1, -1, -1, 2, -1, -1, -1, 2};
constexpr const int kDownProjSplitKFactor[] = {-1, 10, -1, -1, 3, -1, -1, -1, 3};
constexpr const int kSiluMultiplyTileTileSize = 128;
constexpr const int kAllReduceTileMTile = 16;
constexpr const int kAllReduceTileNTile = 128;
constexpr const int kHiddenSize = 5120;
constexpr const int kNumAttentionHeadsTP1 = 64;
constexpr const int kIntermediateSizeTP1 = 25600;
constexpr const int kDecodeMergeHeadsPerTile = 1;

constexpr const int kNumSM = 148;
constexpr const int kNumWarpgroupPerBlock = 2;
constexpr const int kNumWarpPerWarpgroup = 4;
constexpr const int kStaticTileSchedulerMaxTasks = 128;
constexpr const int kDyanmicTileSchedulerMaxTasks = 8192;

constexpr const int kMaxTotalNumWorks = 65536;
constexpr const int kSemaphoreBase = (1 << 16);
constexpr const int kMaxSemaphore = 2147483647;

enum class JobType : int32_t {
  kVReduceAppend = 0,
  kKReduceNormRopeAppend = 1,
  kQReduceNormRope = 2,
  kBatchAttention = 3,
  kBatchMerge = 4,
  kGateUpProjReduce = 5,
  kDownProjAllReduce = 6,
  kOAllReduce = 7,
  kAttnAddRMSNorm = 8,
  kGemmOReduce = 9,
  kGemmOProj = 10,
  kGemmQKVProj = 11,
  kMLPAddRMSNorm = 12,
  kDownProjReduce = 13,
  kGemmDownProj = 14,
  kSplitSiluMultiply = 15,
  kGemmGateUpProj = 16,
  kGateUpSilu = 17,

  // end
  kEnd = 31,
};

// every task info in exec queue will be squashed into 32bit:
// task_type: [0:5], m_idx: [5:18], n_idx: [18:27], k_idx: [27:32]
constexpr const int kMaxTaskType = 1 << 5;
constexpr const int kMaxMIdx = 1 << 14;
constexpr const int kMaxNIdx = 1 << 9;
constexpr const int kMaxKIdx = 1 << 5;

inline int32_t PackInto32bit(int32_t task_type, int32_t m_idx, int32_t n_idx, int32_t k_idx) {
  CHECK_LT(task_type, kMaxTaskType);
  CHECK_LT(m_idx, kMaxMIdx);
  CHECK_LT(n_idx, kMaxNIdx);
  CHECK_LT(k_idx, kMaxKIdx);
  return task_type | (m_idx << 5) | (n_idx << 18) | (k_idx << 27);
}

ffi::Array<Tensor> GetEventTensorsOnLayer(ffi::Array<Tensor> etensors, int layer_id);

Tensor GenerateExecQueueStatic(int batch_size, int attn_task_num, int tp_size, int num_qo_heads,
                               int num_kv_heads, int head_dim, Device device,
                               Device preferred_host_device);

ffi::Array<ffi::Array<Tensor>> GenerateExecQueueDynamic(Tensor exec_queue_device_buf,
                                                        Tensor exec_queue_host_buf, int tp_size,
                                                        int num_qo_heads, int num_kv_heads,
                                                        int head_dim, int num_layers,
                                                        TVMStreamHandle copy_stream);

}  // namespace megakernel
}  // namespace vm
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_VM_MEGAKERNEL_UTILS_H_
