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
#include "attn_utils.h"

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/container/shape.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/nvtx.h>
#include <tvm/runtime/tensor.h>

#include <utility>

namespace tvm {
namespace runtime {
namespace vm {
namespace megakernel {

using ffi::Array;
using ffi::Shape;
using runtime::Tensor;

Array<Tensor> GetEventTensorsOnLayer(Array<Tensor> etensors, int layer_id) {
  TVM_FFI_ICHECK_GE(layer_id, 0) << "Layer id must be non-negative, but got " << layer_id;
  std::vector<Tensor> etensors_on_layer;
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
  return Array<Tensor>(etensors_on_layer);
}

Tensor GenerateExecQueueStatic(int batch_size, int attn_task_num, int tp_size,
                               std::string model_name, Device device,
                               Device preferred_host_device) {
  NVTXScopedRange range("Generate execution queue");
  const auto f_get_config =
      tvm::ffi::Function::GetGlobalRequired("tirp.megakernel.get_model_config");
  auto config = f_get_config(model_name).cast<ffi::Map<ffi::String, ffi::Any>>();
  int num_qo_heads = config["NUM_ATTENTION_HEADS"].cast<int>() / tp_size;
  int num_kv_heads = config["NUM_KEY_VALUE_HEADS"].cast<int>() / tp_size;
  int head_dim = config["HEAD_DIM"].cast<int>();
  int hidden_size = config["HIDDEN_SIZE"].cast<int>();
  int intermediate_size = config["INTERMEDIATE_SIZE"].cast<int>() / tp_size;
  int split_qkv_project = config["SPLIT_QKV_PROJECT_DICT"].cast<ffi::Map<int, int>>()[tp_size];
  int split_o_project = config["SPLIT_O_PROJECT_DICT"].cast<ffi::Map<int, int>>()[tp_size];
  TVM_FFI_ITVM_FFI_ICHECK_NE(split_qkv_project, -1);
  TVM_FFI_ITVM_FFI_ICHECK_NE(split_o_project, -1);
  bool is_moe = config.count("NUM_EXPERTS");
  bool split_kv = attn_task_num > num_kv_heads * batch_size;

  Tensor exec_queue_host = Tensor::Empty({kNumSM, kStaticTileSchedulerMaxTasks}, DataType::Int(32),
                                         preferred_host_device);
  Tensor exec_queue_device =
      Tensor::Empty({kNumSM, kStaticTileSchedulerMaxTasks}, DataType::Int(32), device);
  int32_t* exec_queue_host_data = static_cast<int32_t*>(exec_queue_host->data);

  // Generate round-robin static execution queue
  std::vector<int32_t> task_counts(kNumSM, 0);
  int cur_sm = 0;

  auto f_push_task = [&exec_queue_host_data, &task_counts, &cur_sm](int m_idx, int n_idx, int k_idx,
                                                                    JobType job_type) {
    int task_id = task_counts[cur_sm]++;
    exec_queue_host_data[cur_sm * kStaticTileSchedulerMaxTasks + task_id] =
        PackInto32bit(static_cast<int32_t>(job_type), m_idx, n_idx, k_idx);
    cur_sm = (cur_sm + 1) % kNumSM;
  };

  for (int n_idx = 0; n_idx < ceildiv((num_qo_heads + 2 * num_kv_heads) * head_dim, kGemmTileBlkN);
       ++n_idx) {
    for (int k_idx = 0; k_idx < split_qkv_project; ++k_idx) {
      f_push_task(0, n_idx, k_idx, JobType::kGemmQKVProj);
    }
  }

  int32_t m_split = std::min(batch_size, ceildiv(kNumSM, num_qo_heads + 2 * num_kv_heads));
  int32_t m_tile = ceildiv(batch_size, m_split);
  m_split = ceildiv(batch_size, m_tile);

  for (int m_idx = 0; m_idx < m_split; ++m_idx) {
    for (int n_idx = 0; n_idx < num_qo_heads; ++n_idx) {
      f_push_task(m_idx, n_idx, -1, JobType::kQReduceNormRope);
    }
  }

  for (int m_idx = 0; m_idx < m_split; ++m_idx) {
    for (int n_idx = 0; n_idx < num_kv_heads; ++n_idx) {
      f_push_task(m_idx, n_idx, -1, JobType::kKReduceNormRopeAppend);
    }
  }

  for (int m_idx = 0; m_idx < m_split; ++m_idx) {
    for (int n_idx = 0; n_idx < num_kv_heads; ++n_idx) {
      f_push_task(m_idx, n_idx, -1, JobType::kVReduceAppend);
    }
  }

  for (int m_idx = 0; m_idx < std::min(ceildiv(attn_task_num, kNumWarpgroupPerBlock), kNumSM);
       ++m_idx) {
    f_push_task(m_idx, -1, -1, JobType::kBatchAttention);
  }

  if (split_kv) {
    for (int m_idx = 0;
         m_idx < ceildiv(batch_size * num_qo_heads, kNumWarpgroupPerBlock * kNumWarpPerWarpgroup);
         ++m_idx) {
      f_push_task(m_idx, -1, -1, JobType::kBatchMerge);
    }
  }

  for (int n_idx = 0; n_idx < ceildiv(hidden_size, kGemmTileBlkN); ++n_idx) {
    for (int k_idx = 0; k_idx < split_o_project; ++k_idx) {
      f_push_task(0, n_idx, k_idx, JobType::kGemmOProj);
    }
  }

  if (tp_size > 1) {
    int32_t m_split_o_proj_reduce =
        std::min(batch_size, kNumSM / (hidden_size / kSplitKReduceTileNUnit));
    int32_t n_tile_o_proj_reduce =
        (ceildiv(kSplitKReduceTileNRepeat, ceildiv(batch_size, m_split_o_proj_reduce)) *
        kSplitKReduceTileNUnit);
    int32_t m_tile_o_reduce = ceildiv(batch_size, m_split_o_proj_reduce);
    m_split_o_proj_reduce = ceildiv(batch_size, m_tile_o_reduce);
    for (int n_idx = 0; n_idx < ceildiv(hidden_size, n_tile_o_proj_reduce); ++n_idx) {
      for (int m_idx = 0; m_idx < m_split_o_proj_reduce; ++m_idx) {
        f_push_task(m_idx, n_idx, 0, JobType::kGemmOReduce);
      }
    }

    for (int m_idx = 0; m_idx < ceildiv(batch_size, kAllReduceTileMTile); ++m_idx) {
      for (int n_idx = 0; n_idx < ceildiv(hidden_size / tp_size, kAllReduceTileNTile); ++n_idx) {
        f_push_task(m_idx, n_idx, 0, JobType::kOAllReduce);
      }
    }
  }

  for (int m_idx = 0; m_idx < batch_size; ++m_idx) {
    f_push_task(m_idx, -1, -1, JobType::kAttnAddRMSNorm);
  }

  if (is_moe) {
    int gating_split_k_factor = config["GATING_SPLIT_K_FACTOR"].cast<int>();
    int num_experts_per_tok = config["NUM_EXPERTS_PER_TOK"].cast<int>();
    int num_experts = config["NUM_EXPERTS"].cast<int>();
    for (int m_idx = 0; m_idx < ceildiv(batch_size, kGatingBlkM); ++m_idx) {
      for (int k_idx = 0; k_idx < gating_split_k_factor; ++k_idx) {
        f_push_task(m_idx, 0, k_idx, JobType::kMoeGating);
      }
    }
    for (int m_idx = 0; m_idx < kNumSM; ++m_idx) {
      f_push_task(m_idx, 0, 0, JobType::kMoeTopkSoftmax);
    }
    f_push_task(0, 0, 0, JobType::kMoeAlign);
    for (int m_idx = 0; m_idx < kNumSM; ++m_idx) {
      f_push_task(m_idx, 0, 0, JobType::kMoeCountAndSort);
    }
    const auto f_get_max_num_tokens_padded =
        ffi::Function::GetGlobalRequired("tirp.megakernel.get_max_num_tokens_padded");
    int max_num_tokens_padded =
        f_get_max_num_tokens_padded(batch_size, num_experts_per_tok, num_experts, kMoeBlkM)
            .cast<int>();
    for (int m_idx = 0; m_idx < max_num_tokens_padded / kMoeBlkM; ++m_idx) {
      for (int n_idx = 0; n_idx < (intermediate_size * 2) / kGroupGemmBlkN; ++n_idx) {
        f_push_task(m_idx, n_idx, 0, JobType::kMoeGroupGemmGateUpSilu);
      }
    }
    for (int m_idx = 0; m_idx < max_num_tokens_padded / kMoeBlkM; ++m_idx) {
      for (int n_idx = 0; n_idx < hidden_size / kGroupGemmBlkN; ++n_idx) {
        f_push_task(m_idx, n_idx, 0, JobType::kMoeGroupGemmDown);
      }
    }
  } else {
    int gate_up_proj_split_k_factor =
        config["GATE_UP_PROJ_SPLIT_K_FACTOR_DICT"].cast<ffi::Map<int, int>>()[tp_size];
    int down_proj_split_k_factor =
        config["DOWN_PROJ_SPLIT_K_FACTOR_DICT"].cast<ffi::Map<int, int>>()[tp_size];
    TVM_FFI_ITVM_FFI_ICHECK_NE(down_proj_split_k_factor, -1);
    TVM_FFI_ITVM_FFI_ICHECK_NE(gate_up_proj_split_k_factor, -1);
    if (gate_up_proj_split_k_factor == 1) {
      TVM_FFI_ICHECK_EQ(intermediate_size % kGemmTileBlkN, 0);
      for (int n_idx = 0; n_idx < 2 * intermediate_size / kGemmTileBlkN; ++n_idx) {
        f_push_task(0, n_idx, 0, JobType::kGateUpSilu);
      }
    } else {
      TVM_FFI_ICHECK_EQ(intermediate_size % kGemmTileBlkN, 0);
      for (int n_idx = 0; n_idx < intermediate_size / kGemmTileBlkN; ++n_idx) {
        for (int k_idx = 0; k_idx < gate_up_proj_split_k_factor; ++k_idx) {
          f_push_task(0, n_idx, k_idx, JobType::kGemmGateUpProj);
          f_push_task(0, n_idx + intermediate_size / kGemmTileBlkN, k_idx,
                      JobType::kGemmGateUpProj);
        }
      }
      int32_t m_split_gate_up_proj_reduce =
          std::min(batch_size, kNumSM / (intermediate_size * 2 / kSplitKReduceTileNUnit));
      int32_t m_tile_gate_up_proj_reduce = ceildiv(batch_size, m_split_gate_up_proj_reduce);
      m_split_gate_up_proj_reduce = ceildiv(batch_size, m_tile_gate_up_proj_reduce);
      for (int m_idx = 0; m_idx < m_split_gate_up_proj_reduce; ++m_idx) {
        for (int n_idx = 0; n_idx < ceildiv(intermediate_size * 2, kGemmTileBlkN); ++n_idx) {
          f_push_task(m_idx, n_idx, 0, JobType::kGateUpProjReduce);
        }
      }
      for (int n_idx = 0; n_idx < ceildiv(intermediate_size, kSiluMultiplyTileSize); ++n_idx) {
        f_push_task(n_idx, 0, 0, JobType::kSplitSiluMultiply);
      }
    }

    for (int n_idx = 0; n_idx < ceildiv(hidden_size, kGemmTileBlkN); ++n_idx) {
      for (int k_idx = 0; k_idx < down_proj_split_k_factor; ++k_idx) {
        f_push_task(0, n_idx, k_idx, JobType::kGemmDownProj);
      }
    }

    if (tp_size > 1) {
      int32_t m_split_down_proj_reduce =
          std::min(kNumSM / (hidden_size / kSplitKReduceTileNUnit), batch_size);
      int32_t n_tile_down_proj_reduce =
          (ceildiv(kSplitKReduceTileNRepeat, ceildiv(batch_size, m_split_down_proj_reduce)) *
          kSplitKReduceTileNUnit);
      int32_t m_tile_down_proj_reduce = ceildiv(batch_size, m_split_down_proj_reduce);
      m_split_down_proj_reduce = ceildiv(batch_size, m_tile_down_proj_reduce);
      for (int m_idx = 0; m_idx < m_split_down_proj_reduce; ++m_idx) {
        for (int n_idx = 0; n_idx < ceildiv(hidden_size, n_tile_down_proj_reduce); ++n_idx) {
          f_push_task(m_idx, n_idx, 0, JobType::kDownProjReduce);
        }
      }

      for (int m_idx = 0; m_idx < ceildiv(batch_size, kAllReduceTileMTile); ++m_idx) {
        for (int n_idx = 0; n_idx < ceildiv(hidden_size / tp_size, kAllReduceTileNTile); ++n_idx) {
          f_push_task(m_idx, n_idx, 0, JobType::kDownProjAllReduce);
        }
      }
    }
  }
  for (int m_idx = 0; m_idx < batch_size; ++m_idx) {
    f_push_task(m_idx, 0, 0, JobType::kMLPAddRMSNorm);
  }

  // Set the uninitialized queue to kEnd
  for (int sm_id = 0; sm_id < kNumSM; ++sm_id) {
    for (int task_id = task_counts[sm_id]; task_id < kStaticTileSchedulerMaxTasks; ++task_id) {
      exec_queue_host_data[sm_id * kStaticTileSchedulerMaxTasks + task_id] =
          PackInto32bit(static_cast<int32_t>(JobType::kEnd), -1, -1, -1);
    }
  }

  // Transfer to device
  DLTensor exec_queue_device_dl = *exec_queue_device.operator->();
  Tensor::CopyFromTo(exec_queue_host.operator->(), &exec_queue_device_dl);
  return exec_queue_device;
}

Array<Array<Tensor>> GenerateExecQueueDynamic(Tensor exec_queue_device_buf,
                                              Tensor exec_queue_host_buf, int tp_size,
                                              std::string model_name, int num_layers,
                                              TVMStreamHandle copy_stream) {
  NVTXScopedRange range("Generate execution queue");
  int elem_per_layer = kDyanmicTileSchedulerMaxTasks + 4;
  TVM_FFI_ICHECK(exec_queue_device_buf.dtype() == DataType::Int(32));
  TVM_FFI_ICHECK(exec_queue_host_buf.dtype() == DataType::Int(32));
  TVM_FFI_ICHECK(exec_queue_device_buf.Shape().Product() >= num_layers * elem_per_layer);
  TVM_FFI_ICHECK(exec_queue_host_buf.Shape().Product() >= num_layers * elem_per_layer);
  int32_t* exec_queue_host_data = static_cast<int32_t*>(exec_queue_host_buf->data);
  int num_tasks = 0;
  const auto f_get_config =
      tvm::ffi::Function::GetGlobalRequired("tirp.megakernel.get_model_config");
  auto config = f_get_config(model_name).cast<ffi::Map<ffi::String, ffi::Any>>();
  int num_qo_heads = config["NUM_ATTENTION_HEADS"].cast<int>() / tp_size;
  int num_kv_heads = config["NUM_KEY_VALUE_HEADS"].cast<int>() / tp_size;
  int head_dim = config["HEAD_DIM"].cast<int>();
  int split_qkv_project = config["SPLIT_QKV_PROJECT_DICT"].cast<ffi::Map<int, int>>()[tp_size];

  // fill the exec queue with -1
  for (int i = 0; i < num_layers * elem_per_layer; ++i) {
    exec_queue_host_data[i] = -1;
  }

  auto f_push_task = [&exec_queue_host_data, &num_tasks, &elem_per_layer](
                         int m_idx, int n_idx, int k_idx, JobType job_type, int layer_id) {
    exec_queue_host_data[layer_id * elem_per_layer + num_tasks] =
        PackInto32bit(static_cast<int32_t>(job_type), m_idx, n_idx, k_idx);
    num_tasks++;
  };

  for (int layer_id = 0; layer_id < num_layers; ++layer_id) {
    num_tasks = 0;
    // Push initial tasks.
    for (int n_idx = ceildiv((num_qo_heads + 2 * num_kv_heads) * head_dim, kGemmTileBlkN) - 1;
         n_idx >= 0; --n_idx) {
      for (int k_idx = 0; k_idx < split_qkv_project; ++k_idx) {
        f_push_task(0, n_idx, k_idx, JobType::kGemmQKVProj, layer_id);
      }
    }
    // Set head & tail.
    exec_queue_host_data[layer_id * elem_per_layer + kDyanmicTileSchedulerMaxTasks] = 0;
    exec_queue_host_data[layer_id * elem_per_layer + kDyanmicTileSchedulerMaxTasks + 1] = num_tasks;
  }

  // Transfer to device
  DLTensor exec_queue_device_dl = *exec_queue_device_buf.operator->();
  Tensor::CopyFromTo(exec_queue_host_buf.operator->(), &exec_queue_device_dl, copy_stream);

  // Slice the execution queue to get the dynamic execution queue.
  std::vector<Array<Tensor>> queue_by_layer;
  queue_by_layer.reserve(num_layers);
  for (int layer_id = 0; layer_id < num_layers; ++layer_id) {
    Tensor exec_queue = exec_queue_device_buf.CreateView(
        {elem_per_layer}, DataType::Int(32),
        /*relative_byte_offset=*/layer_id * elem_per_layer * DataType::Int(32).bytes());
    Tensor queue_tasks =
        exec_queue.CreateView({megakernel::kDyanmicTileSchedulerMaxTasks}, DataType::Int(32));
    Tensor queue_head =
        exec_queue.CreateView({1}, DataType::Int(32),
                              /*relative_byte_offset=*/megakernel::kDyanmicTileSchedulerMaxTasks *
                                  DataType::Int(32).bytes());
    Tensor queue_tail = exec_queue.CreateView(
        {1}, DataType::Int(32),
        /*relative_byte_offset=*/
        (megakernel::kDyanmicTileSchedulerMaxTasks + 1) * DataType::Int(32).bytes());

    queue_by_layer.push_back(Array<Tensor>({queue_tasks, queue_head, queue_tail}));
  }
  return queue_by_layer;
}

std::vector<HostMemoryVector> GenerateEventTensorHost(int batch_size, int attn_task_num, int max_batch_size, 
                                           std::string model_name, int tp_size, DLDataType dtype_aux, Device host_device){
  NVTXScopedRange range("Generate event tensor on host");
  const auto f_get_config =
      tvm::ffi::Function::GetGlobalRequired("tirp.megakernel.get_model_config");
  auto config = f_get_config(model_name).cast<ffi::Map<ffi::String, ffi::Any>>();
  int num_qo_heads = config["NUM_ATTENTION_HEADS"].cast<int>() / tp_size;
  int num_kv_heads = config["NUM_KEY_VALUE_HEADS"].cast<int>() / tp_size;
  int head_dim = config["HEAD_DIM"].cast<int>();
  int num_layers = config["NUM_HIDDEN_LAYERS"].cast<int>();
  bool split_kv = attn_task_num > num_kv_heads * batch_size;
  int split_qkv_project = config["SPLIT_QKV_PROJECT_DICT"].cast<ffi::Map<int, int>>()[tp_size];
  int split_o_project = config["SPLIT_O_PROJECT_DICT"].cast<ffi::Map<int, int>>()[tp_size];
  int hidden_size = config["HIDDEN_SIZE"].cast<int>();
  int intermediate_size = config["INTERMEDIATE_SIZE"].cast<int>() / tp_size;
  HostMemoryVector etensor_qkv_partial_host_ =
      HostMemoryVector(num_layers * ceildiv((num_qo_heads + 2 * num_kv_heads) * head_dim,
                                            megakernel::kSplitKReduceTileNUnit),
                       dtype_aux, host_device);
  HostMemoryVector etensor_notify_attn_host_ =
      HostMemoryVector(num_layers * megakernel::kNumSM, dtype_aux, host_device);
  HostMemoryVector etensor_o_partial_host_ =
      HostMemoryVector(num_layers * ceildiv(hidden_size, megakernel::kGemmTileBlkN), dtype_aux,
                       host_device);
  HostMemoryVector etensor_o_allreduce_host_ =
      HostMemoryVector(num_layers * ceildiv(hidden_size / tp_size, megakernel::kAllReduceTileNTile),
                       dtype_aux, host_device);
  HostMemoryVector etensor_attn_add_rms_host_ =
      HostMemoryVector(num_layers * max_batch_size, dtype_aux, host_device);
  HostMemoryVector etensor_attn_mlp_host_ = HostMemoryVector(num_layers, dtype_aux, host_device);
  HostMemoryVector etensor_end_host_ = HostMemoryVector(num_layers, dtype_aux, host_device);
  HostMemoryVector etensor_o_proj_host_ =
      HostMemoryVector(num_layers * split_o_project, dtype_aux, host_device);
  HostMemoryVector etensor_attn_merge_host_ = HostMemoryVector(num_layers * max_batch_size * num_kv_heads,
                                              dtype_aux, host_device);
  // 2. Initialize event tensors
  if (model_name == "qwen3_32b") {
    int gate_up_proj_factor =
        config["GATE_UP_PROJ_SPLIT_K_FACTOR_DICT"].cast<ffi::Map<int, int>>()[tp_size];
    int down_proj_split_k_factor =
        config["DOWN_PROJ_SPLIT_K_FACTOR_DICT"].cast<ffi::Map<int, int>>()[tp_size];
    ICHECK_NE(split_qkv_project, -1);
    ICHECK_NE(split_o_project, -1);
    ICHECK_NE(gate_up_proj_factor, -1);
    ICHECK_NE(down_proj_split_k_factor, -1);
    HostMemoryVector etensor_gate_up_proj_reduce_host_ =
        HostMemoryVector(num_layers * ceildiv(intermediate_size * 2, megakernel::kGemmTileBlkN),
                         dtype_aux, host_device);
    HostMemoryVector etensor_gate_up_proj_host_ =
        HostMemoryVector(num_layers * ceildiv(intermediate_size, megakernel::kGemmTileBlkN),
                         dtype_aux, host_device);
    HostMemoryVector etensor_down_proj_reduce_host_ =
        HostMemoryVector(num_layers * ceildiv(hidden_size, megakernel::kGemmTileBlkN), dtype_aux, host_device);
    HostMemoryVector etensor_down_proj_allreduce_host_ = HostMemoryVector(
        num_layers * ceildiv(hidden_size / tp_size, megakernel::kAllReduceTileNTile), dtype_aux, host_device);
    HostMemoryVector etensor_mlp_add_rms_host_ =
        HostMemoryVector(num_layers * max_batch_size, dtype_aux, host_device);
    HostMemoryVector etensor_down_proj_host_ =
        HostMemoryVector(num_layers * down_proj_split_k_factor, dtype_aux, host_device);
    // static zero etensors
    etensor_qkv_partial_host_.resize(num_layers *
                                      ceildiv((num_qo_heads + 2 * num_kv_heads) * head_dim,
                                              megakernel::kSplitKReduceTileNUnit));
    etensor_qkv_partial_host_.fill(0);
    etensor_o_partial_host_.resize(num_layers * ceildiv(hidden_size, megakernel::kGemmTileBlkN));
    etensor_o_partial_host_.fill(0);
    etensor_o_allreduce_host_.resize(
        num_layers * ceildiv(hidden_size / tp_size, megakernel::kAllReduceTileNTile));
    etensor_o_allreduce_host_.fill(0);
    etensor_attn_add_rms_host_.resize(num_layers * batch_size);
    etensor_attn_add_rms_host_.fill(0);
    etensor_attn_mlp_host_.resize(num_layers);
    etensor_attn_mlp_host_.fill(0);
    etensor_gate_up_proj_reduce_host_.resize(
        num_layers * ceildiv(intermediate_size * 2, megakernel::kGemmTileBlkN));
    etensor_gate_up_proj_reduce_host_.fill(0);
    etensor_gate_up_proj_host_.resize(num_layers *
                                      ceildiv(intermediate_size, megakernel::kGemmTileBlkN));
    etensor_gate_up_proj_host_.fill(0);
    etensor_down_proj_reduce_host_.resize(num_layers *
                                          ceildiv(hidden_size, megakernel::kGemmTileBlkN));
    etensor_down_proj_reduce_host_.fill(0);
    etensor_down_proj_allreduce_host_.resize(
        num_layers * ceildiv(hidden_size / tp_size, megakernel::kAllReduceTileNTile));
    etensor_down_proj_allreduce_host_.fill(0);
    etensor_mlp_add_rms_host_.resize(num_layers * batch_size);
    etensor_mlp_add_rms_host_.fill(0);
    etensor_end_host_.resize(num_layers);
    etensor_end_host_.fill(0);
    // dynamic etensors
    etensor_notify_attn_host_.resize(num_layers * megakernel::kNumSM);
    etensor_notify_attn_host_.fill(0);
    int attn_tile_num = ceildiv(attn_task_num, megakernel::kNumWarpgroupPerBlock);
    int unit = megakernel::kNumWarpgroupPerBlock * (2 + num_qo_heads / num_kv_heads);
    int num = unit * (attn_tile_num / megakernel::kNumSM);
    int remain = attn_tile_num % megakernel::kNumSM;
    for (int layer_id = 0; layer_id < num_layers; ++layer_id) {
      for (int i = 0; i < megakernel::kNumSM; ++i) {
        int cnt = (i < remain) ? unit + num : num;
        CHECK_LE(cnt, megakernel::kSemaphoreBase);
        CHECK_LT(cnt * (megakernel::kSemaphoreBase + 1), megakernel::kMaxSemaphore);
        etensor_notify_attn_host_.set(layer_id * megakernel::kNumSM + i,
                                      cnt * (megakernel::kSemaphoreBase + 1));
      }
    }

    etensor_o_proj_host_.resize(num_layers * split_o_project);
    etensor_o_proj_host_.fill(0);
    int o_proj_tile_k = (ceildiv(ceildiv(num_qo_heads * head_dim, split_o_project),
                                  megakernel::kGemmTileBlkK) *
                          megakernel::kGemmTileBlkK);
    if (split_kv) {
      for (int layer_id = 0; layer_id < num_layers; ++layer_id) {
        for (int m = 0;
              m < ceildiv(batch_size * num_qo_heads,
                          megakernel::kNumWarpgroupPerBlock * megakernel::kNumWarpPerWarpgroup);
              ++m) {
          int worker_id =
              m * megakernel::kNumWarpgroupPerBlock * megakernel::kNumWarpPerWarpgroup;
          int kv_idx = worker_id / (batch_size * (num_qo_heads / num_kv_heads));
          int range_start =
              (kv_idx * (num_qo_heads / num_kv_heads)) * head_dim / o_proj_tile_k;
          int range_end = (((kv_idx + 1) * (num_qo_heads / num_kv_heads)) * head_dim - 1) /
                          o_proj_tile_k;
          for (int i = range_start; i <= range_end; ++i) {
            CHECK_GE(i, 0) << "Index " << i << " is negative.";
            CHECK_LT(i, split_o_project) << "Index " << i << " out of bounds " << split_o_project;
            etensor_o_proj_host_.set(layer_id * split_o_project + i,
                                      etensor_o_proj_host_[layer_id * split_o_project + i] + 1);
          }
        }
      }
    } else {
      etensor_o_proj_host_.fill(std::min(megakernel::kNumSM, attn_tile_num));
    }
    for (int layer_id = 0; layer_id < num_layers; ++layer_id) {
      for (int i = 0; i < split_o_project; ++i) {
        CHECK_LE(etensor_o_proj_host_[layer_id * split_o_project + i],
                  megakernel::kSemaphoreBase);
        CHECK_LT(etensor_o_proj_host_[layer_id * split_o_project + i] *
                      (megakernel::kSemaphoreBase + 1),
                  megakernel::kMaxSemaphore);
        etensor_o_proj_host_.set(layer_id * split_o_project + i,
                                  etensor_o_proj_host_[layer_id * split_o_project + i] *
                                      (megakernel::kSemaphoreBase + 1));
      }
    }

    etensor_down_proj_host_.resize(num_layers * down_proj_split_k_factor);
    etensor_down_proj_host_.fill(0);
    int down_proj_tile_k = (ceildiv(ceildiv(intermediate_size, down_proj_split_k_factor),
                                    megakernel::kGemmTileBlkK) *
                            megakernel::kGemmTileBlkK);
    if (gate_up_proj_factor == 1) {
      for (int layer_id = 0; layer_id < num_layers; ++layer_id) {
        for (int m = 0; m < (intermediate_size) * 2 / megakernel::kGemmTileBlkN; ++m) {
          int range_start = m * megakernel::kGemmTileBlkN / 2 / down_proj_tile_k;
          int range_end = ((m + 1) * megakernel::kGemmTileBlkN / 2 - 1) / down_proj_tile_k;
          for (int i = range_start; i <= range_end; ++i) {
            CHECK_GE(i, 0) << "Index " << i << " is negative.";
            CHECK_LT(i, down_proj_split_k_factor)
                << "Index " << i << " out of bounds " << down_proj_split_k_factor;
            etensor_down_proj_host_.set(
                layer_id * down_proj_split_k_factor + i,
                etensor_down_proj_host_[layer_id * down_proj_split_k_factor + i] + 1);
          }
        }
      }
    } else {
      for (int layer_id = 0; layer_id < num_layers; ++layer_id) {
        for (int m = 0; m < intermediate_size / megakernel::kSiluMultiplyTileSize; ++m) {
          int range_start = m * megakernel::kSiluMultiplyTileSize / down_proj_tile_k;
          int range_end = ((m + 1) * megakernel::kSiluMultiplyTileSize - 1) / down_proj_tile_k;
          for (int i = range_start; i <= range_end; ++i) {
            CHECK_GE(i, 0) << "Index " << i << " is negative.";
            CHECK_LT(i, down_proj_split_k_factor)
                << "Index " << i << " out of bounds " << down_proj_split_k_factor;
            etensor_down_proj_host_.set(
                layer_id * down_proj_split_k_factor + i,
                etensor_down_proj_host_[layer_id * down_proj_split_k_factor + i] + 1);
          }
        }
      }
    }
    for (int layer_id = 0; layer_id < num_layers; ++layer_id) {
      for (int i = 0; i < down_proj_split_k_factor; ++i) {
        CHECK_LE(etensor_down_proj_host_[layer_id * down_proj_split_k_factor + i],
                  megakernel::kSemaphoreBase);
        CHECK_LT(etensor_down_proj_host_[layer_id * down_proj_split_k_factor + i] *
                      (megakernel::kSemaphoreBase + 1),
                  megakernel::kMaxSemaphore);
        etensor_down_proj_host_.set(
            layer_id * down_proj_split_k_factor + i,
            etensor_down_proj_host_[layer_id * down_proj_split_k_factor + i] *
                (megakernel::kSemaphoreBase + 1));
      }
    }

    etensor_attn_merge_host_.resize(num_layers * batch_size * num_kv_heads);
    etensor_attn_merge_host_.fill(std::min(megakernel::kNumSM, attn_tile_num) *
                                  (megakernel::kSemaphoreBase + 1));
    std::vector<HostMemoryVector> result;
    result.push_back(std::move(etensor_qkv_partial_host_));
    result.push_back(std::move(etensor_notify_attn_host_));
    result.push_back(std::move(etensor_o_partial_host_));
    result.push_back(std::move(etensor_o_allreduce_host_));
    result.push_back(std::move(etensor_attn_add_rms_host_));
    result.push_back(std::move(etensor_attn_mlp_host_));
    result.push_back(std::move(etensor_gate_up_proj_reduce_host_));
    result.push_back(std::move(etensor_gate_up_proj_host_));
    result.push_back(std::move(etensor_down_proj_reduce_host_));
    result.push_back(std::move(etensor_down_proj_allreduce_host_));
    result.push_back(std::move(etensor_mlp_add_rms_host_));
    result.push_back(std::move(etensor_end_host_));
    result.push_back(std::move(etensor_o_proj_host_));
    result.push_back(std::move(etensor_down_proj_host_));
    result.push_back(std::move(etensor_attn_merge_host_));
    return result;
  } else if (model_name == "qwen3_30b_a3b" || model_name == "qwen3_30b_a3b_unfused") {
    int gating_split_k_factor = config["GATING_SPLIT_K_FACTOR"].cast<int>();
    int num_experts = config["NUM_EXPERTS"].cast<int>();
    int num_experts_per_tok = config["NUM_EXPERTS_PER_TOK"].cast<int>();
    ICHECK_NE(split_qkv_project, -1);
    ICHECK_NE(split_o_project, -1);
    HostMemoryVector etensor_gating_host_ = HostMemoryVector(num_layers, dtype_aux, host_device);
    HostMemoryVector etensor_topk_softmax_host_ = HostMemoryVector(num_layers, dtype_aux, host_device);
    HostMemoryVector etensor_moe_align_host_ = HostMemoryVector(num_layers, dtype_aux, host_device);
    HostMemoryVector etensor_count_and_sort_host_ = HostMemoryVector(num_layers, dtype_aux, host_device);
    HostMemoryVector etensor_group_gemm_gate_up_host_;
    HostMemoryVector etensor_silu_mul_host_;
    if (model_name == "qwen3_30b_a3b_unfused") {
      etensor_group_gemm_gate_up_host_ =
          HostMemoryVector(num_layers, dtype_aux, host_device);
      etensor_silu_mul_host_ = HostMemoryVector(num_layers, dtype_aux, host_device);
    } else {
      const auto f_get_max_num_tokens_padded =
          tvm::ffi::Function::GetGlobalRequired("tirp.megakernel.get_max_num_tokens_padded");
      int max_num_tokens_padded =
          f_get_max_num_tokens_padded(max_batch_size, num_experts_per_tok, num_experts,
                                      megakernel::kMoeBlkM)
              .cast<int>();
      etensor_group_gemm_gate_up_host_ =
          HostMemoryVector(num_layers * (max_num_tokens_padded / megakernel::kMoeBlkM), dtype_aux,
                           host_device);
      etensor_silu_mul_host_ =
          HostMemoryVector(num_layers * (max_num_tokens_padded / megakernel::kMoeBlkM), dtype_aux,
                           host_device);
    }
    HostMemoryVector etensor_group_gemm_down_host_ = HostMemoryVector(num_layers, dtype_aux, host_device);
    // static zero etensors
    etensor_qkv_partial_host_.resize(num_layers *
                                      ceildiv((num_qo_heads + 2 * num_kv_heads) * head_dim,
                                              megakernel::kSplitKReduceTileNUnit));
    etensor_qkv_partial_host_.fill(0);
    etensor_o_partial_host_.resize(num_layers * ceildiv(hidden_size, megakernel::kGemmTileBlkN));
    etensor_o_partial_host_.fill(0);
    etensor_o_allreduce_host_.resize(
        num_layers * ceildiv(hidden_size / tp_size, megakernel::kAllReduceTileNTile));
    etensor_o_allreduce_host_.fill(0);
    etensor_attn_add_rms_host_.resize(num_layers * batch_size);
    etensor_attn_add_rms_host_.fill(0);
    etensor_attn_mlp_host_.resize(num_layers);
    etensor_attn_mlp_host_.fill(0);
    etensor_gating_host_.resize(num_layers);
    etensor_gating_host_.fill(megakernel::kSemaphoreFactor * gating_split_k_factor *
                              ceildiv(batch_size, megakernel::kGatingBlkM));
    etensor_topk_softmax_host_.resize(num_layers);
    etensor_topk_softmax_host_.fill(megakernel::kSemaphoreFactor * megakernel::kNumSM);
    etensor_moe_align_host_.resize(num_layers);
    etensor_moe_align_host_.fill(megakernel::kSemaphoreFactor);
    etensor_count_and_sort_host_.resize(num_layers);
    etensor_count_and_sort_host_.fill(megakernel::kSemaphoreFactor * megakernel::kNumSM);
    const auto f_get_max_num_tokens_padded =
        tvm::ffi::Function::GetGlobalRequired("tirp.megakernel.get_max_num_tokens_padded");
    int max_num_tokens_padded = f_get_max_num_tokens_padded(batch_size, num_experts_per_tok,
                                                            num_experts, megakernel::kMoeBlkM)
                                    .cast<int>();
    if (model_name == "qwen3_30b_a3b_unfused") {
      etensor_group_gemm_gate_up_host_.resize(num_layers);
      etensor_group_gemm_gate_up_host_.fill(megakernel::kSemaphoreFactor *
                                            (max_num_tokens_padded / megakernel::kMoeBlkM) *
                                            intermediate_size * 2 / megakernel::kGemmTileBlkN);
      etensor_silu_mul_host_.resize(num_layers);
      etensor_silu_mul_host_.fill(megakernel::kSemaphoreFactor *
                                  (max_num_tokens_padded / megakernel::kMoeBlkM) *
                                  intermediate_size / megakernel::kSiluMultiplyMoeTileSize);
    } else {
      etensor_group_gemm_gate_up_host_.resize(num_layers *
                                              (max_num_tokens_padded / megakernel::kMoeBlkM));
      etensor_group_gemm_gate_up_host_.fill(megakernel::kSemaphoreFactor * intermediate_size * 2 /
                                            megakernel::kGemmTileBlkN);
      etensor_silu_mul_host_.resize(num_layers * (max_num_tokens_padded / megakernel::kMoeBlkM));
      etensor_silu_mul_host_.fill(megakernel::kSemaphoreFactor * intermediate_size /
                                  megakernel::kSiluMultiplyMoeTileSize);
    }
    etensor_group_gemm_down_host_.resize(num_layers);
    etensor_group_gemm_down_host_.fill(megakernel::kSemaphoreFactor *
                                        (max_num_tokens_padded / megakernel::kMoeBlkM) *
                                        (hidden_size / megakernel::kGemmTileBlkN));
    etensor_end_host_.resize(num_layers);
    etensor_end_host_.fill(0);
    // dynamic etensors
    etensor_notify_attn_host_.resize(num_layers * megakernel::kNumSM);
    etensor_notify_attn_host_.fill(0);
    int attn_tile_num = ceildiv(attn_task_num, megakernel::kNumWarpgroupPerBlock);
    int unit = megakernel::kNumWarpgroupPerBlock * (2 + num_qo_heads / num_kv_heads);
    int num = unit * (attn_tile_num / megakernel::kNumSM);
    int remain = attn_tile_num % megakernel::kNumSM;
    for (int layer_id = 0; layer_id < num_layers; ++layer_id) {
      for (int i = 0; i < megakernel::kNumSM; ++i) {
        int cnt = (i < remain) ? unit + num : num;
        CHECK_LE(cnt, megakernel::kSemaphoreBase);
        CHECK_LT(cnt * (megakernel::kSemaphoreBase + 1), megakernel::kMaxSemaphore);
        etensor_notify_attn_host_.set(layer_id * megakernel::kNumSM + i,
                                      cnt * (megakernel::kSemaphoreBase + 1));
      }
    }

    etensor_o_proj_host_.resize(num_layers * split_o_project);
    etensor_o_proj_host_.fill(0);
    int o_proj_tile_k = (ceildiv(ceildiv(num_qo_heads * head_dim, split_o_project),
                                  megakernel::kGemmTileBlkK) *
                          megakernel::kGemmTileBlkK);
    if (split_kv) {
      // to simply, assume that one merge tile will not use two kv head
      CHECK_LE(megakernel::kNumWarpgroupPerBlock * megakernel::kNumWarpPerWarpgroup,
                num_qo_heads / num_kv_heads);
      for (int layer_id = 0; layer_id < num_layers; ++layer_id) {
        for (int m = 0;
              m < ceildiv(batch_size * num_qo_heads,
                          megakernel::kNumWarpgroupPerBlock * megakernel::kNumWarpPerWarpgroup);
              ++m) {
          int worker_id =
              m * megakernel::kNumWarpgroupPerBlock * megakernel::kNumWarpPerWarpgroup;
          int kv_idx = worker_id / (batch_size * (num_qo_heads / num_kv_heads));
          int qo_idx = worker_id % (num_qo_heads / num_kv_heads);
          int range_start =
              (kv_idx * (num_qo_heads / num_kv_heads) + qo_idx) * head_dim / o_proj_tile_k;
          int range_end =
              ((kv_idx * (num_qo_heads / num_kv_heads) + qo_idx +
                megakernel::kNumWarpgroupPerBlock * megakernel::kNumWarpPerWarpgroup) *
                    head_dim -
                1) /
              o_proj_tile_k;
          for (int i = range_start; i <= range_end; ++i) {
            CHECK_GE(i, 0) << "Index " << i << " is negative.";
            CHECK_LT(i, split_o_project) << "Index " << i << " out of bounds " << split_o_project;
            etensor_o_proj_host_.set(layer_id * split_o_project + i,
                                      etensor_o_proj_host_[layer_id * split_o_project + i] + 1);
          }
        }
      }
    } else {
      etensor_o_proj_host_.fill(std::min(megakernel::kNumSM, attn_tile_num));
    }
    for (int layer_id = 0; layer_id < num_layers; ++layer_id) {
      for (int i = 0; i < split_o_project; ++i) {
        CHECK_LE(etensor_o_proj_host_[layer_id * split_o_project + i],
                  megakernel::kSemaphoreBase);
        CHECK_LT(etensor_o_proj_host_[layer_id * split_o_project + i] *
                      (megakernel::kSemaphoreBase + 1),
                  megakernel::kMaxSemaphore);
        etensor_o_proj_host_.set(layer_id * split_o_project + i,
                                  etensor_o_proj_host_[layer_id * split_o_project + i] *
                                      (megakernel::kSemaphoreBase + 1));
      }
    }

    etensor_attn_merge_host_.resize(num_layers * batch_size * num_kv_heads);
    etensor_attn_merge_host_.fill(std::min(megakernel::kNumSM, attn_tile_num) *
                                  (megakernel::kSemaphoreBase + 1));
    std::vector<HostMemoryVector> result;
    result.push_back(std::move(etensor_qkv_partial_host_));
    result.push_back(std::move(etensor_notify_attn_host_));
    result.push_back(std::move(etensor_o_partial_host_));
    result.push_back(std::move(etensor_o_allreduce_host_));
    result.push_back(std::move(etensor_attn_add_rms_host_));
    result.push_back(std::move(etensor_attn_mlp_host_));
    result.push_back(std::move(etensor_end_host_));
    result.push_back(std::move(etensor_o_proj_host_));
    result.push_back(std::move(etensor_attn_merge_host_));
    result.push_back(std::move(etensor_gating_host_));
    result.push_back(std::move(etensor_topk_softmax_host_));
    result.push_back(std::move(etensor_moe_align_host_));
    result.push_back(std::move(etensor_count_and_sort_host_));
    result.push_back(std::move(etensor_group_gemm_gate_up_host_));
    result.push_back(std::move(etensor_silu_mul_host_));
    result.push_back(std::move(etensor_group_gemm_down_host_));
    return result;
  } else {
    LOG(FATAL) << "Megakernel does not support model " << model_name;
  }
}

    // RNN State methods
    TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("megakernel.get_event_tensors_on_layer", GetEventTensorsOnLayer)
      .def("megakernel.generate_exec_queue_static", GenerateExecQueueStatic)
      .def("megakernel.generate_exec_queue_dynamic", GenerateExecQueueDynamic);
}

}  // namespace megakernel
}  // namespace vm
}  // namespace runtime
}  // namespace tvm
