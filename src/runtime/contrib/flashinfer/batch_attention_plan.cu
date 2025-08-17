/*
 * Copyright (c) 2023 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <dlpack/dlpack.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/shape.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/int_tuple.h>
#include <tvm/runtime/ndarray.h>

#include <../../../../3rdparty/flashinfer/include/flashinfer/attention/decode.cuh>
#include <../../../../3rdparty/flashinfer/include/flashinfer/attention/variants.cuh>
#include <../../../../3rdparty/flashinfer/include/flashinfer/page.cuh>
#include <optional>

#include "../../../../3rdparty/flashinfer/include/flashinfer/attention/mask.cuh"
#include "../../../../3rdparty/flashinfer/include/flashinfer/attention/scheduler.cuh"
#include "../../../../3rdparty/flashinfer/include/flashinfer/pos_enc.cuh"

using IdType = int32_t;
using tvm::ffi::Array;
using tvm::runtime::DataType;
using tvm::runtime::IntTuple;
using tvm::runtime::NDArray;

namespace tvm {
namespace runtime {

using namespace flashinfer;

using DTypeQ = half;
using DTypeKV = half;
using DTypeO = half;

struct BatchDecodeParams {
  using DTypeQ = DTypeQ;
  using DTypeKV = half;
  using DTypeO = half;
  using IdType = IdType;

  DTypeQ* q;
  paged_kv_t<DTypeKV, IdType> paged_kv;
  DTypeO* o;
  float* lse;

  double sm_scale;
  double rope_rcp_scale;
  double rope_rcp_theta;

  IdType* decode_maybe_q_rope_offset;

  uint32_t padded_batch_size;
  uint32_t num_qo_heads;
  IdType q_stride_n;
  IdType q_stride_h;
  int32_t window_left;

  IdType* request_indices;
  IdType* kv_tile_indices;
  IdType* o_indptr;
  IdType* kv_chunk_size_ptr;
  bool* block_valid_mask;
  bool partition_kv;

  __host__ __device__ __forceinline__ int32_t get_qo_len(int32_t batch_idx) const { return 1; }

  __host__ __device__ __forceinline__ int32_t get_kv_len(int32_t batch_idx) const {
    return paged_kv.get_length(batch_idx);
  }
};

IntTuple BatchPrefillWithKVCachePlan(
    DLTensor* float_workspace_buffer, DLTensor* int_workspace_buffer,
    DLTensor* page_locked_int_workspace_buffer, DLTensor* qo_indptr, DLTensor* kv_indptr,
    IntTuple kv_len_arr, int64_t total_num_rows, int64_t batch_size, int64_t num_qo_heads,
    int64_t num_kv_heads, int64_t page_size, bool enable_cuda_graph, int64_t head_dim_qk,
    int64_t head_dim_vo, bool causal, TVMStreamHandle cuda_stream) {
  size_t float_workspace_size_in_bytes =
      float_workspace_buffer->shape[0] * DataType(float_workspace_buffer->dtype).bytes();
  size_t int_workspace_size_in_bytes =
      int_workspace_buffer->shape[0] * DataType(int_workspace_buffer->dtype).bytes();

  PrefillPlanInfo plan_info;

  cudaStream_t stream = static_cast<cudaStream_t>(cuda_stream);
  cudaError_t status = PrefillPlan<IdType>(
      static_cast<char*>(float_workspace_buffer->data) + float_workspace_buffer->byte_offset,
      float_workspace_size_in_bytes,
      static_cast<char*>(int_workspace_buffer->data) + int_workspace_buffer->byte_offset,
      static_cast<char*>(page_locked_int_workspace_buffer->data) +
          page_locked_int_workspace_buffer->byte_offset,
      int_workspace_size_in_bytes, plan_info,
      static_cast<IdType*>(qo_indptr->data) + qo_indptr->byte_offset / sizeof(IdType),
      static_cast<IdType*>(kv_indptr->data) + kv_indptr->byte_offset / sizeof(IdType),
      total_num_rows, batch_size, num_qo_heads, num_kv_heads, head_dim_qk, head_dim_vo, page_size,
      enable_cuda_graph,
      /*sizeof_dtype_o=*/2, stream);

  CHECK(status == cudaSuccess) << "Failed to plan prefill with error: "
                               << cudaGetErrorString(status);

  std::vector<int64_t> plan_info_vec = plan_info.ToVector();
  return IntTuple{plan_info_vec.begin(), plan_info_vec.end()};
}

Array<NDArray> BatchDecodeWithPagedKVCachePlan(
    NDArray float_workspace_buffer, NDArray int_workspace_buffer,
    NDArray page_locked_int_workspace_buffer, NDArray indptr, int64_t batch_size,
    int64_t num_qo_heads, int64_t num_kv_heads, int64_t page_size, bool enable_cuda_graph,
    int64_t pos_encoding_mode_code, int64_t window_left, int64_t head_dim_qk, int64_t head_dim_vo,
    DataType q_scalar_type, DataType kv_scalar_type, bool enforce_no_split_kv) {
  size_t float_workspace_size_in_bytes =
      float_workspace_buffer->shape[0] * DataType(float_workspace_buffer->dtype).bytes();
  size_t int_workspace_size_in_bytes =
      int_workspace_buffer->shape[0] * DataType(int_workspace_buffer->dtype).bytes();

  DecodePlanInfo plan_info;

  CHECK_EQ(head_dim_qk, head_dim_vo)
      << "CUDA cores template only supports equal head dim for QK and VO, please use tensor "
         "cores template for different head dim";

  const PosEncodingMode pos_encoding_mode = static_cast<PosEncodingMode>(pos_encoding_mode_code);

  Device device = float_workspace_buffer->device;
  const cudaStream_t stream =
      static_cast<cudaStream_t>(DeviceAPI::Get(device)->GetCurrentStream(device));
  constexpr int GROUP_SIZE = 8;
  constexpr int HEAD_DIM_QK = 128;
  constexpr PosEncodingMode POS_ENCODING_MODE = PosEncodingMode::kNone;
  using AttentionVariant = DefaultAttention<false, false, false, false>;
  using Params = BatchDecodeParams;

  auto work_estimation_func = BatchDecodeWithPagedKVCacheWorkEstimationDispatched<
      GROUP_SIZE, HEAD_DIM_QK, POS_ENCODING_MODE, AttentionVariant, Params>;
  cudaError_t status = DecodePlan<HEAD_DIM_QK, POS_ENCODING_MODE, AttentionVariant, Params>(
      static_cast<char*>(float_workspace_buffer->data) + float_workspace_buffer->byte_offset,
      float_workspace_size_in_bytes,
      static_cast<char*>(int_workspace_buffer->data) + int_workspace_buffer->byte_offset,
      static_cast<char*>(page_locked_int_workspace_buffer->data) +
          page_locked_int_workspace_buffer->byte_offset,
      int_workspace_size_in_bytes, plan_info,
      static_cast<IdType*>(indptr->data) + indptr->byte_offset / sizeof(IdType), batch_size,
      num_qo_heads, page_size, enable_cuda_graph,
      /*stream=*/stream, work_estimation_func, enforce_no_split_kv);

  CHECK(status == cudaSuccess) << "BatchDecodeWithPagedKVCache failed with error "
                               << cudaGetErrorString(status);

  std::vector<int64_t> plan_info_vec = plan_info.ToVector();
  int64_t padded_batch_size = plan_info_vec[0];
  int64_t request_indices_offset = plan_info_vec[3];
  int64_t request_indices_size = padded_batch_size;
  int64_t kv_tile_indices_offset = plan_info_vec[4];
  int64_t kv_tile_indices_size = padded_batch_size;
  int64_t kv_chunk_size_ptr_offset = plan_info_vec[7];
  int64_t kv_chunk_size_ptr_size = 1;

  DataType dtype = DataType::Int(32);
  NDArray request_indices =
      int_workspace_buffer.CreateView({request_indices_size}, dtype,
                                      /*relative_byte_offset=*/request_indices_offset);
  NDArray kv_tile_indices =
      int_workspace_buffer.CreateView({kv_tile_indices_size}, dtype,
                                      /*relative_byte_offset=*/kv_tile_indices_offset);
  NDArray kv_chunk_size_ptr =
      int_workspace_buffer.CreateView({kv_chunk_size_ptr_size}, dtype,
                                      /*relative_byte_offset=*/kv_chunk_size_ptr_offset);
  return {request_indices, kv_tile_indices, kv_chunk_size_ptr};
}

IntTuple BatchPagedAttentionPlan(DLTensor* float_workspace_buffer, DLTensor* int_workspace_buffer,
                                 DLTensor* page_locked_int_workspace_buffer, DLTensor* qo_indptr,
                                 DLTensor* kv_indptr, DLTensor* kv_len, int64_t batch_size,
                                 int64_t num_qo_heads, int64_t num_kv_heads, int64_t head_dim_o,
                                 bool causal) {
  size_t float_workspace_size_in_bytes =
      float_workspace_buffer->shape[0] * DataType(float_workspace_buffer->dtype).bytes();
  size_t int_workspace_size_in_bytes =
      int_workspace_buffer->shape[0] * DataType(int_workspace_buffer->dtype).bytes();

  HolisticPlanInfo<2> plan_info;
  Device device = float_workspace_buffer->device;
  const cudaStream_t stream =
      static_cast<cudaStream_t>(DeviceAPI::Get(device)->GetCurrentStream(device));

  cudaError_t status = TwoStageHolisticPlan<IdType>(
      static_cast<char*>(float_workspace_buffer->data) + float_workspace_buffer->byte_offset,
      float_workspace_size_in_bytes,
      static_cast<char*>(int_workspace_buffer->data) + int_workspace_buffer->byte_offset,
      static_cast<char*>(page_locked_int_workspace_buffer->data) +
          page_locked_int_workspace_buffer->byte_offset,
      int_workspace_size_in_bytes, plan_info,
      static_cast<IdType*>(qo_indptr->data) + qo_indptr->byte_offset / sizeof(IdType),
      static_cast<IdType*>(kv_indptr->data) + kv_indptr->byte_offset / sizeof(IdType),
      static_cast<IdType*>(kv_len->data) + kv_len->byte_offset / sizeof(IdType), batch_size,
      num_qo_heads, num_kv_heads, head_dim_o, causal, stream);

  CHECK(status == cudaSuccess) << "BatchPagedAttentionPlan failed with error "
                               << cudaGetErrorString(status);

  std::vector<int64_t> plan_info_vec = plan_info.ToVector();
  return IntTuple{plan_info_vec.begin(), plan_info_vec.end()};
}

TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("flashinfer.batch_prefill_with_kv_cache_plan", BatchPrefillWithKVCachePlan);
  refl::GlobalDef().def("flashinfer.batch_decode_with_paged_kv_cache_plan",
                        BatchDecodeWithPagedKVCachePlan);
  refl::GlobalDef().def("flashinfer.batch_paged_attention_plan", BatchPagedAttentionPlan);
});

}  // namespace runtime
}  // namespace tvm
