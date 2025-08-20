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

#include "megakernel_utils.h"

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/shape.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/nvtx.h>

#include <utility>

namespace tvm {
namespace runtime {
namespace vm {
namespace megakernel {

using ffi::Array;
using ffi::Shape;
using runtime::NDArray;

Array<NDArray> GetEventTensorsOnLayer(Array<NDArray> etensors, int layer_id) {
  TVM_FFI_ICHECK_GE(layer_id, 0) << "Layer id must be non-negative, but got " << layer_id;
  TVM_FFI_ICHECK_EQ(etensors.size(), 17) << "Event tensors size must be 17";
  std::vector<NDArray> etensors_on_layer;
  etensors_on_layer.reserve(etensors.size());
  for (int i = 0; i < static_cast<int>(etensors.size()); i++) {
    Shape shape = etensors[i].shape();
    TVM_FFI_ICHECK_GE(shape.size(), 2) << "Event tensor must have at least 2 dimensions";
    TVM_FFI_ICHECK_LT(layer_id, shape[0])
        << "Layer id must be less than the first dimension of the event tensor, but got "
        << layer_id << " and " << shape[0];
    std::vector<int64_t> new_shape_vec{shape.begin() + 1, shape.end()};
    Shape new_shape(new_shape_vec);
    int64_t offset = layer_id * new_shape.Product() * DataType(etensors[i].dtype()).bytes();
    etensors_on_layer.push_back(etensors[i].CreateView(new_shape, etensors[i].dtype(), offset));
  }
  return Array<NDArray>(etensors_on_layer);
}

NDArray GenerateExecQueue(int batch_size, int new_batch_size, int tp_size, int num_qo_heads,
                          int num_kv_heads, int head_dim, Device device,
                          Device preferred_host_device) {
  NVTXScopedRange range("Generate execution queue");
  bool split_kv = batch_size != new_batch_size;

  NDArray exec_queue_host = NDArray::Empty({kNumSM, kStaticTileSchedulerMaxTasks, kTaskSize},
                                           DataType::Int(32), preferred_host_device);
  NDArray exec_queue_device =
      NDArray::Empty({kNumSM, kStaticTileSchedulerMaxTasks, kTaskSize}, DataType::Int(32), device);
  int32_t* exec_queue_host_data = static_cast<int32_t*>(exec_queue_host->data);

  // Generate round-robin static execution queue
  std::vector<int32_t> task_counts(kNumSM, 0);
  int cur_sm = 0;

  int split_qkv_project = kSplitQKVProject[tp_size];
  int split_o_project = kSplitOProject[tp_size];
  int down_proj_split_k_factor = kDownProjSplitKFactor[tp_size];
  int gate_up_proj_split_k_factor = kGateUpProjSplitKFactor[tp_size];
  TVM_FFI_ITVM_FFI_ICHECK_NE(split_qkv_project, -1);
  TVM_FFI_ITVM_FFI_ICHECK_NE(split_o_project, -1);
  TVM_FFI_ITVM_FFI_ICHECK_NE(down_proj_split_k_factor, -1);

  auto f_push_task = [&exec_queue_host_data, &task_counts, &cur_sm](int m_idx, int n_idx, int k_idx,
                                                                    JobType job_type) {
    int task_id = task_counts[cur_sm]++;
    // clang-format off
    exec_queue_host_data[cur_sm * kStaticTileSchedulerMaxTasks * kTaskSize + task_id * kTaskSize + 0] = m_idx;
    exec_queue_host_data[cur_sm * kStaticTileSchedulerMaxTasks * kTaskSize + task_id * kTaskSize + 1] = n_idx;
    exec_queue_host_data[cur_sm * kStaticTileSchedulerMaxTasks * kTaskSize + task_id * kTaskSize + 2] = k_idx;
    exec_queue_host_data[cur_sm * kStaticTileSchedulerMaxTasks * kTaskSize + task_id * kTaskSize + 3] = static_cast<int32_t>(job_type);
    // clang-format on
    cur_sm = (cur_sm + 1) % kNumSM;
  };

  for (int n_idx = 0; n_idx < ceildiv((num_qo_heads + 2 * num_kv_heads) * head_dim, kGemmTileBlkN);
       ++n_idx) {
    for (int k_idx = 0; k_idx < split_qkv_project; ++k_idx) {
      f_push_task(0, n_idx, k_idx, JobType::kGemmQKVProj);
    }
  }

  int32_t m_split = std::min(batch_size, ceildiv(kNumSM, (num_qo_heads + 2 * num_kv_heads) *
                                                             head_dim / kSplitKReduceTileNUnit));
  int32_t n_tile_qkv_proj_reduce =
      (ceildiv(kSplitKReduceTileNRepeat, ceildiv(batch_size, m_split)) * kSplitKReduceTileNUnit);
  for (int m_idx = 0; m_idx < m_split; ++m_idx) {
    for (int n_idx = 0;
         n_idx < ceildiv((num_qo_heads + 2 * num_kv_heads) * head_dim, n_tile_qkv_proj_reduce);
         ++n_idx) {
      f_push_task(m_idx, n_idx, 0, JobType::kGemmQKVReduce);
    }
  }

  for (int m_idx = 0; m_idx < m_split; ++m_idx) {
    for (int n_idx = 0; n_idx < num_qo_heads; ++n_idx) {
      f_push_task(m_idx, n_idx, -1, JobType::kQRMSNormRope);
    }
  }

  for (int m_idx = 0; m_idx < m_split; ++m_idx) {
    for (int n_idx = 0; n_idx < num_kv_heads; ++n_idx) {
      f_push_task(m_idx, n_idx, -1, JobType::kKRMSNormRopeAppendKV);
    }
  }

  for (int m_idx = 0; m_idx < m_split; ++m_idx) {
    for (int n_idx = 0; n_idx < num_kv_heads; ++n_idx) {
      f_push_task(m_idx, n_idx, -1, JobType::kVAppendKV);
    }
  }

  if (split_kv) {
    for (int n_idx = 0; n_idx < num_kv_heads; ++n_idx) {
      for (int m_idx = 0; m_idx < new_batch_size; ++m_idx) {
        f_push_task(m_idx, n_idx, -1, JobType::kBatchDecodeSplit);
      }
    }
  } else {
    for (int n_idx = 0; n_idx < num_kv_heads; ++n_idx) {
      for (int m_idx = 0; m_idx < batch_size; ++m_idx) {
        f_push_task(m_idx, n_idx, -1, JobType::kBatchDecodeNoSplit);
      }
    }
  }

  if (split_kv) {
    for (int n_idx = 0; n_idx < num_qo_heads / kDecodeMergeHeadsPerTile; ++n_idx) {
      for (int m_idx = 0; m_idx < batch_size; ++m_idx) {
        f_push_task(m_idx, n_idx, -1, JobType::kDecodeMerge);
      }
    }
  }

  for (int n_idx = 0; n_idx < ceildiv(kHiddenSize, kGemmTileBlkN); ++n_idx) {
    for (int k_idx = 0; k_idx < split_o_project; ++k_idx) {
      f_push_task(0, n_idx, k_idx, JobType::kGemmOProj);
    }
  }

  int32_t m_split_o_proj_reduce =
      std::min(batch_size, ceildiv(kNumSM, kHiddenSize / kSplitKReduceTileNUnit));
  int32_t n_tile_o_proj_reduce =
      (ceildiv(kSplitKReduceTileNRepeat, ceildiv(batch_size, m_split_o_proj_reduce)) *
       kSplitKReduceTileNUnit);
  for (int n_idx = 0; n_idx < ceildiv(kHiddenSize, n_tile_o_proj_reduce); ++n_idx) {
    for (int m_idx = 0; m_idx < m_split_o_proj_reduce; ++m_idx) {
      f_push_task(m_idx, n_idx, 0, JobType::kGemmOReduce);
    }
  }

  if (tp_size > 1) {
    for (int m_idx = 0; m_idx < ceildiv(batch_size, kAllReduceTileMTile); ++m_idx) {
      for (int n_idx = 0; n_idx < ceildiv(kHiddenSize / tp_size, kAllReduceTileNTile); ++n_idx) {
        f_push_task(m_idx, n_idx, 0, JobType::kOAllReduce);
      }
    }
  }

  for (int m_idx = 0; m_idx < batch_size; ++m_idx) {
    f_push_task(m_idx, -1, -1, JobType::kAttnAddRMSNorm);
  }

  for (int n_idx = 0; n_idx < ceildiv(kIntermediateSizeTP1 / tp_size * 2, kGemmTileBlkN); ++n_idx) {
    for (int k_idx = 0; k_idx < gate_up_proj_split_k_factor; ++k_idx) {
      f_push_task(0, n_idx, k_idx, JobType::kGemmGateUpProj);
    }
  }
  if (gate_up_proj_split_k_factor > 1) {
    int32_t m_split_gate_up_proj_reduce =
        std::min(batch_size, ceildiv(kNumSM, kIntermediateSizeTP1 / tp_size * 2 / kSplitKReduceTileNUnit));
    for (int m_idx = 0; m_idx < m_split_gate_up_proj_reduce; ++m_idx) {
      for (int n_idx = 0; n_idx < ceildiv(kIntermediateSizeTP1 / tp_size * 2, kGemmTileBlkN); ++n_idx) {
        f_push_task(m_idx, n_idx, 0, JobType::kGateUpProjReduce);
      }
    }
  }

  for (int n_idx = 0; n_idx < ceildiv(kIntermediateSizeTP1 / tp_size, kSiluMultiplyTileTileSize);
       ++n_idx) {
    f_push_task(n_idx, 0, 0, JobType::kSplitSiluMultiply);
  }

  for (int n_idx = 0; n_idx < ceildiv(kHiddenSize, kGemmTileBlkN); ++n_idx) {
    for (int k_idx = 0; k_idx < down_proj_split_k_factor; ++k_idx) {
      f_push_task(0, n_idx, k_idx, JobType::kGemmDownProj);
    }
  }

  int32_t m_split_down_proj_reduce =
      std::min(ceildiv(kNumSM, ceildiv(kHiddenSize, kSplitKReduceTileNUnit)), batch_size);
  int32_t n_tile_down_proj_reduce =
      (ceildiv(kSplitKReduceTileNRepeat, ceildiv(batch_size, m_split_down_proj_reduce)) *
       kSplitKReduceTileNUnit);
  for (int m_idx = 0; m_idx < m_split_down_proj_reduce; ++m_idx) {
    for (int n_idx = 0; n_idx < ceildiv(kHiddenSize, n_tile_down_proj_reduce); ++n_idx) {
      f_push_task(m_idx, n_idx, 0, JobType::kDownProjReduce);
    }
  }

  if (tp_size > 1) {
    for (int m_idx = 0; m_idx < ceildiv(batch_size, kAllReduceTileMTile); ++m_idx) {
      for (int n_idx = 0; n_idx < ceildiv(kHiddenSize / tp_size, kAllReduceTileNTile); ++n_idx) {
        f_push_task(m_idx, n_idx, 0, JobType::kDownProjAllReduce);
      }
    }
  }

  for (int m_idx = 0; m_idx < batch_size; ++m_idx) {
    f_push_task(m_idx, 0, 0, JobType::kMLPAddRMSNorm);
  }

  // Set the uninitialized queue to kEnd
  for (int sm_id = 0; sm_id < kNumSM; ++sm_id) {
    for (int task_id = task_counts[sm_id]; task_id < kStaticTileSchedulerMaxTasks; ++task_id) {
      // clang-format off
      exec_queue_host_data[sm_id * kStaticTileSchedulerMaxTasks * kTaskSize + task_id * kTaskSize + 0] = -1;
      exec_queue_host_data[sm_id * kStaticTileSchedulerMaxTasks * kTaskSize + task_id * kTaskSize + 1] = -1;
      exec_queue_host_data[sm_id * kStaticTileSchedulerMaxTasks * kTaskSize + task_id * kTaskSize + 2] = -1;
      exec_queue_host_data[sm_id * kStaticTileSchedulerMaxTasks * kTaskSize + task_id * kTaskSize + 3] = static_cast<int32_t>(JobType::kEnd);
      // clang-format on
    }
  }

  // Transfer to device
  DLTensor exec_queue_device_dl = *exec_queue_device.operator->();
  NDArray::CopyFromTo(exec_queue_host.operator->(), &exec_queue_device_dl);
  return exec_queue_device;
}

// RNN State methods
TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("megakernel.get_event_tensors_on_layer", GetEventTensorsOnLayer)
      .def("megakernel.generate_exec_queue", GenerateExecQueue);
});

}  // namespace megakernel
}  // namespace vm
}  // namespace runtime
}  // namespace tvm
