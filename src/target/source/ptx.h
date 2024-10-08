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
 * \file ptx.h
 * \brief Code generation with inlined PTX code.
 */
#ifndef TVM_TARGET_SOURCE_PTX_H_
#define TVM_TARGET_SOURCE_PTX_H_

#include <tvm/runtime/logging.h>

#include <string>
#include <tuple>

#include "codegen_cuda.h"

namespace tvm {
namespace codegen {

/*!
 * \brief Print MMA assembly string given parameters.
 * \param shape The shape string mMnNkK
 * \param A_layout The layout of multiplicand A, can be either "row" or "col".
 * \param B_layout The layout of multiplicand B, can be either "row" or "col".
 * \param A_dtype The data type of multiplicand A.
 * \param B_dtype The data type of multiplicand B.
 * \param C_dtype The data type of multiplicand C.
 * \param a_ptr Pointer to buffer A.
 * \param a_offset The offset of element in A.
 * \param b_ptr Pointer to buffer B.
 * \param b_offset The offset of element in B.
 * \param c_ptr Pointer to buffer C.
 * \param c_offset The offset of element in C.
 * \param metadata Pointer to metadata buffer (only used for sparse mma).
 * \param metadata_offset The offset of element in metadata.
 * \param sparsity_selector The sparsity selector in sparse mma.
 * \param bit_op The bit operator used in 1-bit mma, can be either "xor" or "and".
 * \param sparse Whether it's sparse mma or not.
 * \param saturate Whether saturate output or not.
 */
std::string PrintMMAAssembly(const std::string& shape, const std::string& A_layout,
                             const std::string& B_layout, const std::string& A_dtype,
                             const std::string& B_dtype, const std::string& C_dtype,
                             const std::string& a_ptr, const std::string& a_offset,
                             const std::string& b_ptr, const std::string& b_offset,
                             const std::string& c_ptr, const std::string& c_offset,
                             const std::string& metadata, const std::string& metadata_offset,
                             const std::string& sparsity_selector, const std::string& bit_op,
                             bool sparse, bool saturate);

/*!
 * \brief Print ldmatrix assembly string given parameters.
 * \param trans: whether the matrix is loaded in column major format or not.
 * \param num: number of matrices to load.
 * \param type: The data type in the matrix, .b16 is the only accepted data type.
 * \param local_ptr: pointer to local buffer.
 * \param local_elem_offset: The offset of the element to store in the local buffer.
 * \param smem_ptr: pointer to the shared memory buffer to load.
 * \param smem_elem_offset: The offset of the start element of the row to load in shared memory.
 */
std::string PrintLoadMatrixAssembly(bool trans, int num, const std::string& type,
                                    const std::string& local_ptr,
                                    const std::string& local_elem_offset,
                                    const std::string& smem_ptr,
                                    const std::string& smem_elem_offset);

/*!
 * \brief Print ptx cp.async assembly string given parameters.
 * \param shared_ptr: The pointer to the destination shared memory.
 * \param shared_elem_offset: The offset into the shared memory.
 * \param global_ptr: The pointer to the global memory.
 * \param global_elem_offset: The offset into the global memory.
 * \param bytes: The number of bytes to copy, valid values are 4, 8, and 16.
 */
std::string PrintCpAsyncAssembly(const std::string& shared_ptr,
                                 const std::string& shared_elem_offset,
                                 const std::string& global_ptr,
                                 const std::string& global_elem_offset, const std::string& bytes);

/*!
 * \brief Print predicated ptx cp.async assembly string given parameters.
 * \param shared_ptr: The pointer to the destination shared memory.
 * \param shared_elem_offset: The offset into the shared memory.
 * \param global_ptr: The pointer to the global memory.
 * \param global_elem_offset: The offset into the global memory.
 * \param bytes: The number of bytes to copy, valid values are 4, 8, and 16.
 * \param predicate_value: The value of predicate `@p`.
 */
std::string PrintPredicatedCpAsyncAssembly(const std::string& shared_ptr,
                                           const std::string& shared_elem_offset,
                                           const std::string& global_ptr,
                                           const std::string& global_elem_offset,
                                           const std::string& bytes,
                                           const std::string& predicate_value);

/*!
 * \brief Print ptx async copy from global to shared memory using cp.async.bulk
 * \param shared_ptr: The pointer to the destination shared memory.
 * \param shared_elem_offset: The offset into the shared memory.
 * \param global_ptr: The pointer to the global memory.
 * \param global_elem_offset: The offset into the global memory.
 * \param bytes: The number of bytes to copy.
 * \param barrier: The name of the barrier in shared memory.
 */
std::string PrintCpAsyncBulkAsm(const std::string& shared_ptr,
                                const std::string& shared_elem_offset,
                                const std::string& global_ptr,
                                const std::string& global_elem_offset, const std::string& bytes,
                                const std::string& barrier);

/*!
 * \brief Print ptx async copy barrier using cp.async.mbarrier.arrive
 * \param barrier: The name of the barrier in shared memory.
 */
std::string PrintCpAsyncBarrierAsm(const std::string& barrier);

/*!
 * \brief Print ptx barrier initialization of thread count using mbarrier.init
 * \param barrier: The name of the barrier in shared memory.
 * \param thread_count: The number of threads expected to arrive at the barrier.
 */
std::string PrintInitBarrierThreadCountAsm(const std::string& barrier,
                                           const std::string& thread_count);

/*!
 * \brief Print ptx barrier arrival using mbarrier.arrive
 * \param barrier: The name of the barrier in shared memory.
 */
std::string PrintArriveBarrierAsm(const std::string& barrier);

/*!
 * \brief Print ptx barrier arrival with expect tx operation using mbarrier.arrive.expect_tx
 * \param barrier: The name of the barrier in shared memory.
 * \param byte_count: Increases the tx count of the mbarrier object to track completion of
 * addtional async transactions.
 */
std::string PrintArriveBarrierExpectTxAsm(const std::string& barrier,
                                          const std::string& byte_count);

/*!
 * \brief Print ptx barrier wait using mbarrier.try_wait
 * \param barrier: The name of the barrier in shared memory.
 */
std::string PrintWaitBarrierAsm(const std::string& barrier);

/*!
 * \brief Print ptx fence.proxy.async.{global, shared::cta, shared::cluster}
 * \param scope: The scope of the fence.
 */
std::string PrintCudaFenceProxyAsyncAssembly(std::string scope);

/*!
 * \brief Print ptx mbarrier.init.shared.b64
 * \param barrier: The name of the barrier in shared memory.
 * \param thread_count: The number of threads expected to arrive at the barrier.
 */
std::string PrintMbarrierInitAssembly(const std::string& barrier, const std::string& thread_count);

/*!
 * \brief Print ptx mbarrier.arrive.shared.b64
 * \param barrier: The name of the barrier in shared memory.
 * \param remote: Whether the barrier is remote.
 * \param cta_id: The id of the target CTA.
 * \param pred: The predicate value.
 */
std::string PrintMbarrierArriveAssembly(const std::string& barrier, bool remote,
                                        const std::string& cta_id, const std::string& pred);

/*!
 * \brief Print ptx mbarrier.arrive.expect_tx.shared::cta.b64
 * \param barrier: The name of the barrier in shared memory.
 * \param byte_count: Increases the tx count of the mbarrier object to track completion of
 * addtional async transactions.
 * \param remote: Whether the barrier is remote.
 * \param cta_id: The id of the target CTA.
 * \param pred: The predicate value.
 */
std::string PrintMbarrierArriveExpectTxAssembly(const std::string& barrier,
                                                const std::string& byte_count, bool remote,
                                                const std::string& cta_id, const std::string& pred);

/*!
 * \brief Print ptx mbarrier.try_wait.parity repeatedly until it returns true
 * \param barrier: The name of the barrier in shared memory.
 * \param phase: The phase bit to wait for.
 */
std::string PrintMbarrierWaitAssembly(const std::string& barrier, const std::string& phase);

/*!
 * \brief Print bar.sync a, {b}
 * \param name_bar_id: The name of the barrier.
 * \param thread_count: The number of threads expected to arrive at the barrier.
 */
std::string PrintNamedBarrierSyncAssembly(const std::string& name_bar_id,
                                          const std::string& thread_count);

/*!
 * \brief Print ptx cp.async.bulk.tensor.{dim}.shared::cluster.global.mbarrier::complete_tx::bytes
 * \param dim: The dimension of the tensor.
 * \param dst: The pointer to the destination shared memory.
 * \param bar: The pointer to the barrier in shared memory.
 * \param tensormap: The pointer to the CUtensorMap object.
 * \param cta_mask: The mask for the CTA.
 * \param coords: The coordinates of the tensor.
 */
std::string PrintCpAsyncBulkTensorGlobalToClusterAssembly(int dim, const std::string& dst,
                                                          const std::string& bar,
                                                          const std::string& tensormap,
                                                          int cta_mask,
                                                          std::vector<std::string> coords);

/*!
 * \brief Print ptx cp.async.bulk.tensor.dim.global.shared::cta.tile。bulk_group
 * \param dim: The dimension of the tensor.
 * \param src: The pointer to the source shared memory.
 * \param tensormap: The pointer to the CUtensorMap object.
 * \param coords: The coordinates of the tensor.
 */
std::string PrintCpAsyncBulkTensorSharedToGlobalAssembly(int dim, const std::string& src,
                                                         const std::string& tensormap,
                                                         std::vector<std::string> coords);

/*!
 * \brief Print ptx cp.async.bulk.tensor.commit_group
 */
std::string PrintCpAsyncBulkTensorCommitGroupAssembly();

/*!
 * \brief Print ptx cp.async.bulk.tensor.wait_group{.read} N
 * \param N: The number of groups to wait for.
 * \param read: Whether to wait for read or write groups.
 */

std::string PrintCpAsyncBulkTensorWaitGroupAssembly(const std::string& N, bool read);

/*!
 * \brief Print predefined, read-only variables, which are visible as special registers in PTX.
 * \param bits: The number of bits of the register.
 * \param reg: The name of the register.
 */
std::string PrintPtxFetchRegisterAssembly(CodeGenCUDA* cg, int bits, const std::string& reg);

/*!
 * \brief Print "" : "+r"(reg) :: "memory"
 * \param reg: The register to print.
 */
std::string PrintWGMMAFenceOpearandAssembly(const std::string& reg);

/*!
 * \brief Print wgmma.mma.sync.aligned where both A and B are in shared memory.
 * \param M: The number of rows in the matrix.
 * \param N: The number of columns in the matrix.
 * \param K: The number of columns in the matrix.
 * \param in_dtype: The data type of the input matrix.
 * \param out_dtype: The data type of the output matrix.
 * \param transA: Whether the input matrix A is K major or M/N major.
 * \param transB: Whether the input matrix B is K major or M/N major.
 * \param scaleA: The scaling factor for matrix A.
 * \param scaleB: The scaling factor for matrix B.
 * \param scaleD: True: D = A * B + D, False: D = A * B.
 * \param descA: The descriptor for matrix A.
 * \param descB: The descriptor for matrix B.
 * \param accums: The accumulator matrix descriptors.
 */
std::string PrintWGMMAmmasyncSSAssembly(int M, int N, int K, const std::string& in_dtype,
                                        const std::string& out_dtype, bool transA, bool transB,
                                        float scaleA, float scaleB, bool scaleD,
                                        const std::string& descA, const std::string& descB,
                                        const std::vector<std::string>& accums);

/*!
 * \brief Print wgmma.fence.sync.aligned;
 */
std::string PrintWGMMAArriveAssembly();

/*!
 * \brief Print wgmma.commit_group.sync.aligned;
 */
std::string PrintWGMMACommitGroupAssembly();

/*!
 * \brief Print wgmma.wait_group.sync.aligned;
 * \param N: The number of groups to wait for.
 */
std::string PrintWGMMAWaitGroupAssembly(const std::string& N);

/*!
 * \brief Print shared memory matrix descriptor encoding.
 * \param desc: The pointer to the shared memory descriptor.
 * \param addr: The address of the matrix.
 * \param ldo: The leading dimension offset.
 * \param sdo: The stride dimension offset.
 * \param swizzle: The swizzle value (CUtensorMapSwizzle_enum).
 */
std::string PrintEncodeMatrixDescriptor(const std::string& desc, const std::string& addr,
                                        const std::string& ldo, const std::string& sdo,
                                        int swizzle);

/*!
 * \brief Print barrier.cluster.arrive{.sem}{.aligned};
 * \param sem: Either release or relaxed or empty string.
 * \param aligned: Whether all threads in the warp must execute the same instruction.
 */
std::string PrintBarrierClusterArriveAssembly(const std::string& sem, bool aligned);

/*!
 * \brief Print barrier.cluster.wait{.acquire}{.aligned};
 * \param acquire: The memory synchronization
 * \param aligned: Whether all threads in the warp must execute the same instruction.
 */
std::string PrintBarrierClusterWaitAssembly(bool acquire, bool aligned);

/*!
 * \brief Print elec.sync _|p membermask
 * \param membermask: The mask for the synchronization.
 */
std::string PrintElectSyncAssembly(CodeGenCUDA* cg, uint32_t membermask);

/*!
 * \brief Print fence_mbarrier_init_release_cluster
 */
std::string PrintFenceMbarrierInitReleaseClusterAssembly();

}  // namespace codegen
}  // namespace tvm
#endif  // TVM_TARGET_SOURCE_PTX_H_
