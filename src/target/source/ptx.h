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
std::string PrintCudaFenceProxyAsyncAssembly(CodeGenCUDA* cg, std::string scope);

/*!
 * \brief Print ptx mbarrier.init.shared.b64
 * \param barrier: The name of the barrier in shared memory.
 * \param thread_count: The number of threads expected to arrive at the barrier.
 */
std::string PrintMbarrierInitAssembly(CodeGenCUDA* cg, const std::string& barrier,
                                      const std::string& thread_count);

/*!
 * \brief Print ptx mbarrier.arrive.shared.b64
 * \param barrier: The name of the barrier in shared memory.
 * \param remote: Whether the barrier is remote.
 * \param cta_id: The id of the target CTA.
 * \param pred: The predicate value.
 */
std::string PrintMbarrierArriveAssembly(codegen::CodeGenCUDA* cg, const std::string& barrier,
                                        bool remote, const std::string& cta_id,
                                        const std::string& pred);

/*!
 * \brief Print ptx mbarrier.arrive.expect_tx.shared::cta.b64
 * \param barrier: The name of the barrier in shared memory.
 * \param byte_count: Increases the tx count of the mbarrier object to track completion of
 * addtional async transactions.
 * \param remote: Whether the barrier is remote.
 * \param cta_id: The id of the target CTA.
 * \param pred: The predicate value.
 */
std::string PrintMbarrierArriveExpectTxAssembly(CodeGenCUDA* cg, const std::string& barrier,
                                                const std::string& byte_count, bool remote,
                                                const std::string& cta_id, const std::string& pred);

/*!
 * \brief Print ptx mbarrier.try_wait.parity repeatedly until it returns true
 * \param barrier: The name of the barrier in shared memory.
 * \param phase: The phase bit to wait for.
 */
std::string PrintMbarrierWaitAssembly(codegen::CodeGenCUDA* cg, const std::string& barrier,
                                      const std::string& phase);

/*!
 * \brief Print bar.arrive a, b
 * \param name_bar_id: The name of the barrier.
 * \param thread_count The number of threads expected to arrive at the barrier.
 */
std::string PrintNamedBarrierArriveAssembly(const std::string& name_bar_id,
                                            const std::string& thread_count);

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
 * \param cta_group: The cta_group for the copy.
 * \param coords: The coordinates of the tensor.
 */
std::string PrintCpAsyncBulkTensorGlobalToClusterAssembly(
    CodeGenCUDA* cg, int dim, const std::string& dst, const std::string& bar,
    const std::string& tensormap, int cta_mask, int cta_group, std::vector<std::string> coords);

/*!
 * \brief Print ptx cp.async.bulk.tensor.dim.global.shared::cta.tile。bulk_group
 * \param dim: The dimension of the tensor.
 * \param src: The pointer to the source shared memory.
 * \param tensormap: The pointer to the CUtensorMap object.
 * \param coords: The coordinates of the tensor.
 */
std::string PrintCpAsyncBulkTensorSharedToGlobalAssembly(codegen::CodeGenCUDA* cg, int dim,
                                                         const std::string& src,
                                                         const std::string& tensormap,
                                                         std::vector<std::string> coords);

/*!
 * \brief Print ptx cp.async.bulk.tensor.commit_group
 */
std::string PrintCpAsyncBulkTensorCommitGroupAssembly(codegen::CodeGenCUDA* cg);

/*!
 * \brief Print ptx cp.async.bulk.tensor.wait_group{.read} N
 * \param N: The number of groups to wait for.
 * \param read: Whether to wait for read or write groups.
 */
std::string PrintCpAsyncBulkTensorWaitGroupAssembly(codegen::CodeGenCUDA* cg, const std::string& N,
                                                    bool read);

/*!
 * \brief Print predefined, read-only variables, which are visible as special registers in PTX.
 * \param bits: The number of bits of the register.
 * \param reg: The name of the register.
 */
std::string PrintPtxFetchRegisterAssembly(CodeGenCUDA* cg, int bits, const std::string& reg);

/*!
 * \brief Print "" : "+r"(reg) :: "memory"
 * \param reg: The register to print.
 * \param dtype: The data type of the register.
 */
std::string PrintWGMMAFenceOpearandAssembly(CodeGenCUDA* cg, const std::string& reg,
                                            tvm::DataType dtype);

/*!
 * \brief Print wgmma.mma_async.sync.aligned where both A and B are in shared memory.
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
 * \param descA: The SMEM descriptor for matrix A.
 * \param descB: The SMEM descriptor for matrix B.
 * \param accums: The registers to store the accumulators.
 */
std::string PrintWGMMAasyncSSAssembly(int M, int N, int K, const std::string& in_dtype,
                                      const std::string& out_dtype, bool transA, bool transB,
                                      float scaleA, float scaleB, const std::string& scaleD,
                                      const std::string& descA, const std::string& descB,
                                      const std::vector<std::string>& accums);

/*!
 * \brief Print wgmma.mma_async.sync.aligned where A is in register and B is in shared memory.
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
 * \param A_regs: The registers to store matrix A.
 * \param descB: The descriptor for matrix B.
 * \param accums: The registers to store the accumulators.
 */
std::string PrintWGMMAasyncRSAssembly(int M, int N, int K, const std::string& in_dtype,
                                      const std::string& out_dtype, bool transA, bool transB,
                                      float scaleA, float scaleB, const std::string& scaleD,
                                      const std::vector<std::string>& A_regs,
                                      const std::string& descB,
                                      const std::vector<std::string>& accums);

/*!
 * \brief Print wgmma.fence.sync.aligned;
 */
std::string PrintWGMMAArriveAssembly(CodeGenCUDA* cg);

/*!
 * \brief Print wgmma.commit_group.sync.aligned;
 */
std::string PrintWGMMACommitGroupAssembly(CodeGenCUDA* cg);

/*!
 * \brief Print wgmma.wait_group.sync.aligned;
 * \param N: The number of groups to wait for.
 */
std::string PrintWGMMAWaitGroupAssembly(CodeGenCUDA* cg, const std::string& N);

/*!
 * \brief Print encoding matrix descriptor for wgmma instructions.
 * \param desc: The pointer to the shared memory descriptor.
 * \param addr: The address of the matrix.
 * \param ldo: The leading dimension offset.
 * \param sdo: The stride dimension offset.
 * \param swizzle: The swizzle value (CUtensorMapSwizzle_enum).
 */
std::string PrintEncodeWgmmaMatrixDescriptor(codegen::CodeGenCUDA* cg, const std::string& desc,
                                             const std::string& addr, const std::string& ldo,
                                             const std::string& sdo, int swizzle);

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
 * \brief Print ptx_fence_mbarrier_init_release_cluster
 */
std::string PrintFenceMbarrierInitReleaseClusterAssembly(codegen::CodeGenCUDA* cg);

/*!
 * \brief Print stmatrix.sync.aligned.m8n8.num{.trans}.shared.b16 [p], r;
 * \param num: The number of 8x8 matrices to store.
 * \param trans: true if the matrix is stored in col-major format.
 * \param ptr: The pointer to the destination shared memory.
 * \param vars: The registers to store.
 */
std::string PrintStmatrixSyncAlignedAssembly(int num, bool trans, const std::string& ptr,
                                             const std::vector<std::string>& vars);

/*!
 * \brief Print setmaxnreg.action.sync.aligned.u32 imm-reg-count
 * \param inc: true if the register count should be incremented.
 * \param reg_count: The number of registers to set.
 */
std::string PrintSetMaxNRegAssembly(bool inc, int reg_count);

/*!
 * \brief Print tcgen05.alloc{.cta_group::1, .cta_group::2}.sync.aligned{.shared::cta}.b32;
 * \param dst_shared_ptr: The pointer to the destination shared memory.
 * \param n_cols: The number of columns to allocate in tensor memory.
 * \param n_cta_group: The number of CTA groups involved in the allocation.
 */
std::string PrintTcgen05AllocAssembly(CodeGenCUDA* cg, const std::string& dst_shared_ptr,
                                      const std::string& n_cols, int n_cta_group);

/*!
 * \brief Print tcgen05.dealloc{.cta_group::1, .cta_group::2}.sync.aligned{.shared::cta}.b32;
 * \param taddr: The address of previously allocated tensor memory.
 * \param n_cols: The number of columns to deallocate in tensor memory.
 * \param n_cta_group: The number of CTA groups involved in the deallocation.
 */
std::string PrintTcgen05DeallocAssembly(CodeGenCUDA* cg, const std::string& taddr,
                                        const std::string& n_cols, int n_cta_group);

/*!
 * \brief Print tcgen05.relinquish_alloc_permit{.cta_group::1, .cta_group::2}.sync.aligned;
 * \param n_cta_group: The number of CTA groups involved in relinquishing.
 */
std::string PrintTcgen05RelinquishAllocPermitAssembly(CodeGenCUDA* cg, int n_cta_group);

/*!
 * \brief Print tcgen05.fence::before_thread_sync;
 */
std::string PrintTcgen05FenceBeforeThreadSyncAssembly(CodeGenCUDA* cg);

/*!
 * \brief Print tcgen05.fence::after_thread_sync;
 */
std::string PrintTcgen05FenceAfterThreadSyncAssembly(CodeGenCUDA* cg);

/*!
 * \brief Print tcgen05.ld.sync.aligned
 * \param src_addr: The address of the source matrix in tensor memory, should be uint32_t.
 * \param row_offset: The row offset of the source matrix in tensor memory.
 * \param col_offset: The column offset of the source matrix in tensor memory.
 * \param regs: The destination registers to copy into.
 * \param shape: The data movement shape, should be lane x size.
 * \param num: The repeat factor along the columns of tensor memory.
 * \param pack: Whether to pack two 16-bit chunks into a single 32-bit chunk in the register.
 */
std::string PrintTcgen05LoadAssembly(CodeGenCUDA* cg, const std::string& src_addr,
                                     const std::string& row_offset, const std::string& col_offset,
                                     const std::vector<std::string>& regs, const std::string& shape,
                                     int num, bool pack);

/*!
 * \brief Print tcgen05.st.sync.aligned
 * \param dst_addr: The address of the destination matrix in tensor memory, should be uint32_t.
 * \param row_offset: The row offset of the destination matrix in tensor memory.
 * \param col_offset: The column offset of the destination matrix in tensor memory.
 * \param regs: The source registers to copy from.
 * \param shape: The data movement shape, should be lane x size.
 * \param num: The repeat factor along the columns of tensor memory.
 * \param unpack: Whether to unpack a single 32-bit chunk into two 16-bit chunks in the register.
 */
std::string PrintTcgen05StoreAssembly(CodeGenCUDA* cg, const std::string& dst_addr,
                                      const std::string& row_offset, const std::string& col_offset,
                                      const std::vector<std::string>& regs,
                                      const std::string& shape, int num, bool unpack);

/*!
 * \brief Print tcgen05.wait::ld.sync.aligned;
 */
std::string PrintTcgen05WaitLdSyncAssembly(CodeGenCUDA* cg);

/*!
 * \brief Print tcgen05.wait::st.sync.aligned;
 */
std::string PrintTcgen05WaitStSyncAssembly(CodeGenCUDA* cg);

/*!
 * \brief Print encoding matrix descriptor for tcgen05 instructions.
 * \param desc: The pointer to the shared memory descriptor.
 * \param addr: The address of the matrix.
 * \param ldo: The leading dimension offset.
 * \param sdo: The stride dimension offset.
 * \param swizzle: The swizzle value (CUtensorMapSwizzle_enum).
 */
std::string PrintEncodeTcgen05MatrixDescriptor(codegen::CodeGenCUDA* cg, const std::string& desc,
                                               const std::string& addr, const std::string& ldo,
                                               const std::string& sdo, int swizzle);

/*!
 * \brief Print encoding instruction descriptor for tcgen05 MMA.
 * \param desc: The pointer to the instruction descriptor.
 * \param d_dtype: The datatype of resultant matrix D.
 * \param a_dtype: The datatype of multiplicand matrix A.
 * \param b_dtype: The datatype of multiplicand matrix B.
 * \param M: Size of non-reduction dimension of Matrix A.
 * \param N: Size of non-reduction dimension of Matrix B.
 * \param K: Size of reduction dimension of Matrix A/B.
 * \param trans_a: Whether the multiplicand matrix A is transposed.
 * \param trans_b: Whether the multiplicand matrix B is transposed.
 * \param n_cta_group: The number of CTA groups involved in the MMA operation.
 * \param neg_a: Whether to negate the multiplicand matrix A.
 * \param neg_b: Whether to negate the multiplicand matrix B.
 * \param sat_d: Whether to saturate the output matrix D.
 * \param is_sparse: Whether the MMA operation is sparse.
 */
std::string PrintEncodeTcgen05InstrDescriptor(codegen::CodeGenCUDA* cg, const std::string& desc,
                                              const std::string& d_dtype,
                                              const std::string& a_dtype,
                                              const std::string& b_dtype, int M, int N, int K,
                                              bool trans_a, bool trans_b, int n_cta_group,
                                              bool neg_a, bool neg_b, bool sat_d, bool is_sparse);

/*!
 * \brief Print encoding instruction descriptor for tcgen05 MMA block scaled
 * \param desc: The pointer to the instruction descriptor.
 * \param d_dtype: The datatype of resultant matrix D.
 * \param a_dtype: The datatype of multiplicand matrix A.
 * \param b_dtype: The datatype of multiplicand matrix B.
 * \param sfa_dtype: The datatype of scale factor matrix A.
 * \param sfb_dtype: The datatype of scale factor matrix B.
 * \param sfa_tmem_addr: The address of the scale factor matrix A in tensor memory.
 * \param sfb_tmem_addr: The address of the scale factor matrix B in tensor memory.
 * \param M: Size of non-reduction dimension of Matrix A.
 * \param N: Size of non-reduction dimension of Matrix B.
 * \param K: Size of reduction dimension of Matrix A/B.
 * \param trans_a: Whether the multiplicand matrix A is transposed.
 * \param trans_b: Whether the multiplicand matrix B is transposed.
 * \param n_cta_group: The number of CTA groups involved in the MMA operation.
 * \param neg_a: Whether to negate the multiplicand matrix A.
 * \param neg_b: Whether to negate the multiplicand matrix B.
 * \param is_sparse: Whether the MMA operation is sparse.
 */
std::string PrintEncodeTcgen05InstrDescriptorBlockScaled(
    codegen::CodeGenCUDA* cg, const std::string& desc, const std::string& d_dtype,
    const std::string& a_dtype, const std::string& b_dtype, const std::string& sfa_dtype,
    const std::string& sfb_dtype, const std::string& sfa_tmem_addr,
    const std::string& sfb_tmem_addr, int M, int N, int K, bool trans_a, bool trans_b,
    int n_cta_group, bool neg_a, bool neg_b, bool is_sparse);

/*!
 * \brief Print tcgen05.mma.cta_group.kind without block scaling.
 * \param d_dtype: The datatype of resultant matrix D.
 * \param a_dtype: The datatype of multiplicand matrix A.
 * \param b_dtype: The datatype of multiplicand matrix B.
 * \param d_tmem_addr: The address of the resultant matrix D in tensor memory.
 * \param a_operand: Either the matrix descriptor of multiplicand matrix A in shared memory,
 *                   or the address of the multiplicand matrix A in tensor memory.
 * \param b_desc: The matrix descriptor of multiplicand matrix B in shared memory.
 * \param i_desc: The instruction descriptor of the MMA operation.
 * \param use_a_tmem: Whether the multiplicand matrix A is in tensor memory.
 * \param cta_group: The number of CTA groups involved in the MMA operation.
 * \param disable_output_lane: The lanes that should not be updated in the resultant matrix D.
 * \param enable_input_d: Whether to accum results into the resultant matrix D or not.
 *                        If enabled, D = A*B + D; else, D = A*B.
 * \param scale_input_d: The optional scaling factor to scale input matrix D.
 *                       D = A*B+D * (2 ^ - scale-input-d)
 * \param sparse: Whether the MMA operation is on sparse matrix or not.
 * \param sp_tmem_addr: The address of the metadata of sparse matrix in tensor memory.
 */
std::string PrintTcgen05MMAAssembly(CodeGenCUDA* cg, const std::string& d_dtype,
                                    const std::string& a_dtype, const std::string& b_dtype,
                                    const std::string& d_tmem_addr, const std::string& a_operand,
                                    const std::string& b_desc, const std::string& i_desc,
                                    bool use_a_tmem, int cta_group,
                                    const std::vector<std::string>& disable_output_lane,
                                    bool enable_input_d, int scale_input_d, bool sparse,
                                    const std::string& sp_tmem_addr = "");

/*!
 * \brief Print tcgen05.mma.cta_group.kind.block_scale{.scale_vec_size}
 * \param d_dtype: The datatype of resultant matrix D.
 * \param a_dtype: The datatype of multiplicand matrix A.
 * \param b_dtype: The datatype of multiplicand matrix B.
 * \param sfa_dtype: The datatype of scale factor matrix A.
 * \param sfb_dtype: The datatype of scale factor matrix B.
 * \param d_tmem_addr: The address of the resultant matrix D in tensor memory.
 * \param a_operand: Either the matrix descriptor of multiplicand matrix A in shared memory,
 *                   or the address of the multiplicand matrix A in tensor memory.
 * \param b_desc: The matrix descriptor of multiplicand matrix B in shared memory.
 * \param sfa_tmem_addr: The address of the scale factor matrix A in tensor memory.
 * \param sfb_tmem_addr: The address of the scale factor matrix B in tensor memory.
 * \param i_desc: The instruction descriptor of the MMA operation.
 * \param use_a_tmem: Whether the multiplicand matrix A is in tensor memory.
 * \param cta_group: The number of CTA groups involved in the MMA operation.
 * \param enable_input_d: Whether to accum results into the resultant matrix D or not.
 *                        If enabled, D = (A * scale_A) * (B * scale_B) + D;
 *                        else, D = (A * scale_A) * (B * scale_B).
 * \param sparse: Whether the MMA operation is on sparse matrix or not.
 * \param sp_tmem_addr: The address of the metadata of sparse matrix in tensor memory.
 */
std::string PrintTcgen05MMABlockScaleAssembly(
    CodeGenCUDA* cg, const std::string& d_dtype, const std::string& a_dtype,
    const std::string& b_dtype, const std::string& sfa_dtype, const std::string& sfb_dtype,
    const std::string& d_tmem_addr, const std::string& a_operand, const std::string& b_desc,
    const std::string& sfa_tmem_addr, const std::string& sfb_tmem_addr, const std::string& i_desc,
    bool use_a_tmem, int cta_group, bool enable_input_d, bool sparse,
    const std::string& sp_tmem_addr = "");

/*!
 * \brief Print tcgen05.commit.cta_group
 * \param bar: The pointer to mbarrier variable.
 * \param cta_group: The number of CTA groups involved in previous tcgen05 operations.
 * \param cta_mask: The mask of the CTAs in the cluster, used for multicast.
 */
std::string PrintTcgen05CommitAssembly(CodeGenCUDA* cg, const std::string& bar, int cta_group,
                                       int cta_mask);

/*!
 * \brief Print tcgen05.cp.cta_group
 * \param dst_addr: The address of the destination in tensor memory, should be uint32_t.
 * \param row_offset: The row offset of the destination in tensor memory.
 * \param col_offset: The column offset of the destination in tensor memory.
 * \param src_desc: The matrix descriptor of the source in shared memory.
 * \param shape: The data movement shape, should be lane x size.
 * \param dst_dtype: The data type of the destination.
 * \param src_dtype: The data type of the source.
 * \param cta_group: The number of CTA groups involved in the copy.
 * \param multicast: Specify how to multicast the data being copied across warps.
 */
std::string PrintTcgen05CopyAssembly(CodeGenCUDA* cg, const std::string& dst_addr,
                                     const std::string& row_offset, const std::string& col_offset,
                                     const std::string& src_desc, const std::string& shape,
                                     const std::string& dst_dtype, const std::string& src_dtype,
                                     int cta_group, const std::string& multicast);

/*!
 * \brief Print tcgen05.shift.cta_group.down
 * \param taddr: The address of matrix in tensor memory, should be uint32_t.
 * \param n_cta_group: The number of CTA groups involved in the shift.
 */
std::string PrintTcgen05ShiftAssembly(CodeGenCUDA* cg, const std::string& taddr, int n_cta_group);

/*!
 * \brief Print mov.u32 %0, %globaltimer_lo.
 */
std::string PrintGetTimestampAssembly(CodeGenCUDA* cg);

/*!
 * \brief Print ld.global.acquire.gpu.{type} %0, [%1].
 * \param res: The register to store the result.
 * \param addr: The address of the global memory.
 * \param dtype: The data type of the global memory.
 */
std::string PrintLdGlobalAcquireAssembly(codegen::CodeGenCUDA* cg, const std::string& res,
                                         const std::string& addr, DataType dtype);
}  // namespace codegen
}  // namespace tvm
#endif  // TVM_TARGET_SOURCE_PTX_H_
