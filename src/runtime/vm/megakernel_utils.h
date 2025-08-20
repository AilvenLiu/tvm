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

#include <tvm/runtime/ndarray.h>

namespace tvm {
namespace runtime {
namespace vm {

inline int ceildiv(int a, int b) { return (a + b - 1) / b; }

namespace megakernel {

constexpr const int kSplitKReduceTileNUnit = 128;
constexpr const int kSplitKReduceTileNRepeat = 1;
constexpr const int kGemmTileBlkN = 128;
constexpr const int kGemmTileBlkK = 64;
constexpr const int kSplitQKVProject[] = {-1, 3, -1, -1, -1, -1, -1, -1, 4};
constexpr const int kSplitOProject[] = {-1, 3, -1, -1, -1, -1, -1, -1, 2};
constexpr const int kGateUpProjSplitKFactor[] = {-1, 1, -1, -1, -1, -1, -1, -1, 2};
constexpr const int kDownProjSplitKFactor[] = {-1, 10, -1, -1, -1, -1, -1, -1, 3};
constexpr const int kSiluMultiplyTileTileSize = 128;
constexpr const int kAllReduceTileMTile = 16;
constexpr const int kAllReduceTileNTile = 128;
constexpr const int kHiddenSize = 5120;
constexpr const int kNumAttentionHeadsTP1 = 64;
constexpr const int kIntermediateSizeTP1 = 25600;
constexpr const int kDecodeMergeHeadsPerTile = 1;

constexpr const int kNumSM = 148;
constexpr const int kStaticTileSchedulerMaxTasks = 128;
constexpr const int kTaskSize = 4;

enum class JobType : int32_t {
  kGemmGateUpProj = 0,
  kSplitSiluMultiply = 1,
  kGemmDownProj = 2,
  kDownProjReduce = 3,
  kMLPAddRMSNorm = 4,
  kGemmQKVProj = 5,
  kGemmQKVReduce = 6,
  kRMSNorm = 7,
  kRope = 8,
  kAppendKV = 9,
  kBatchDecodeNoSplit = 10,
  kBatchDecodeSplit = 11,
  kDecodeMerge = 12,
  kGemmOProj = 13,
  kGemmOReduce = 14,
  kAttnAddRMSNorm = 15,
  kKRMSNormRopeAppendKV = 16,
  kQRMSNormRope = 17,
  kVAppendKV = 18,
  kOAllReduce = 19,
  kDownProjAllReduce = 20,
  kGateUpProjReduce = 21,
  kEnd = 99,
};

Array<NDArray> GetEventTensorsOnLayer(Array<NDArray> etensors, int layer_id);

NDArray GenerateExecQueue(int batch_size, int new_batch_size, int tp_size, int num_qo_heads,
                          int num_kv_heads, int head_dim, Device device,
                          Device preferred_host_device);

}  // namespace megakernel
}  // namespace vm
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_VM_MEGAKERNEL_UTILS_H_
