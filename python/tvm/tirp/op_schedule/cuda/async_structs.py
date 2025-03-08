# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Implementation of async structs schedules."""

from typing import Optional
from enum import Enum
import tvm
from tvm.script import tir as T
from tvm.script import tirp as Tp
from tvm.tirp.op_schedule import ScheduleContext
from tvm.tir import Buffer, PrimFunc, BufferRegion
from tvm.tir.async_structs import CopyPipeline
from tvm.tir.stmt import OpCall
from tvm.arith import Analyzer
from .common import InstType, copy_g2s_s2g_cta_vec_load_impl, target_cuda
from ..registry import register_schedule


class PipelineOp(Enum):
    """The pipeline operation."""

    PRODUCER_COMMIT = 0
    CONSUMER_WAIT = 1
    COPY = 2
    INIT = 3


TENSORMAP_COLLECTOR = {}


def copy_g2s_cta_tma_impl(
    pipeline: CopyPipeline,
    dst_buffer_region: BufferRegion,
    src_buffer_region: BufferRegion,
    sctx: ScheduleContext,
) -> Optional[PrimFunc]:
    """Schedule copy operation between global and shared memory on CUDA."""
    if not (sctx.is_cuda() and sctx.exec_scope.name == "cta"):
        return None

    src: Buffer = src_buffer_region.buffer
    dst: Buffer = dst_buffer_region.buffer
    if not all(
        [
            src.layout and dst.layout,
            src.dtype == dst.dtype,
            src.layout.is_trivial(),
            src.scope() == "global" and dst.scope().startswith("shared"),
        ]
    ):
        return None

    # Extract regions and validate dimensions
    analyzer = Analyzer()
    src_region, dst_region = src_buffer_region.region, dst_buffer_region.region
    src_st = [r.min for r in src_region]
    dst_st = [r.min for r in dst_region]
    src_extent = [r.extent for r in src_region]
    dst_extent = [r.extent for r in dst_region]
    src_dtype = tvm.DataType(src.dtype)
    src_shape = src.shape
    # Derive strides from shape
    src_strides = []
    stride = src_dtype.bits // 8
    for dim in reversed(src_shape):
        src_strides.append(stride)
        stride *= dim
    src_strides = list(reversed(src_strides))

    # Validate non-unit dimensions match
    src_extent_ = [e for e in src_extent if e != 1]
    dst_extent_ = [e for e in dst_extent if e != 1]
    if not (
        len(src_extent_) == len(dst_extent_)
        and all(analyzer.can_prove_equal(s, d) for s, d in zip(src_extent_, dst_extent_))
    ):
        return None

    if dst.layout.is_trivial():
        swizzle_mode = None
    else:
        # TODO(@bohan): support other swizzle modes
        return

    tx = sctx.launch_params["threadIdx.x"]
    assert "threadIdx.y" not in sctx.launch_params and "threadIdx.z" not in sctx.launch_params

    tensor_map = T.Var(src.data.name + "_tensormap", dtype=T.handle("tensormap").type_annotation)
    mbarrier: Optional[Buffer] = pipeline.workspace.get("mbarrier", None)

    if swizzle_mode is None:

        @T.prim_func(tirp=True, check_well_formed=False)
        def impl():
            """Implement copy operation with TMA."""
            for tid_x in T.thread_binding(tx, "threadIdx.x"):
                with T.thread()[tid_x == 0]:
                    T.cp_async_bulk_tensor_global_to_cluster(
                        len(src.shape),
                        dst.access_ptr("w", offset=dst.offset_of_p(dst_st)),
                        mbarrier.access_ptr("rw", offset=0),
                        tensor_map,
                        *reversed(src_st),  # inner-most dimension first
                    )

        element_strides = [1 for _ in range(len(src.shape))]

        @T.prim_func(tirp=True, check_well_formed=False)
        def create_tensor_map():
            """Create the tensor map."""
            with T.LetStmt(T.tvm_stack_alloca("tensormap", 1), var=tensor_map):
                T.call_packed(
                    "runtime.cuTensorMapEncodeTiled",
                    tensor_map,  # pointer to the tensor map
                    src.dtype,  # tensor_dtype
                    len(src.shape),  # global_rank
                    src.data,  # global_ptr
                    *reversed(src_shape),  # global_shape
                    *reversed(src_strides[:-1]),  # global_strides
                    *reversed(src_extent),  # boxDim
                    *element_strides,  # element_strides
                    0,  # CU_TENSOR_MAP_INTERLEAVE_NONE
                    0,  # CU_TENSOR_MAP_SWIZZLE_NONE
                    0,  # CU_TENSOR_MAP_L2PROMOTION_NONE
                    0,  # CU_TENSOR_MAP_FLOAT_OOBFILL_NONE
                )
                Tp.tvm_kernel_replace_point()

        # insert these codes to host code before the kernel
        sctx.add_init_stmt(create_tensor_map.body, host=True)
        return impl
    else:
        # TODO(@bohan): support other swizzle modes
        return None


def copy_pipeline_cta_impl(
    op_type: PipelineOp,
    sctx: ScheduleContext,
    *args,
) -> Optional[PrimFunc]:
    """Copy pipeline implementation."""
    pipeline = args[0]

    impl = pipeline.schedule_config.get(CopyPipeline.StrategyKind.IMPL, None)
    assert impl is not None, "Copy pipeline implementation is not found"

    if impl == CopyPipeline.Impl.VEC_LOAD:
        # async-group-like completion mechanism
        assert pipeline.depth == 0 and not pipeline.separate_pc
        if op_type == PipelineOp.PRODUCER_COMMIT:
            # cp.async.commit_group()
            @T.prim_func
            def func():
                T.evaluate(T.ptx_commit_group())

            return func

        if op_type == PipelineOp.CONSUMER_WAIT:
            # cp.async.wait_group(num_stages)
            num_stages = args[1]

            @T.prim_func
            def func():  # pylint: disable=function-redefined
                T.evaluate(T.ptx_wait_group(num_stages))
                T.tvm_storage_sync("shared")

            return func

        if op_type == PipelineOp.COPY:
            dst, src = args[1], args[2]
            # copy the data from src to dst
            for schedule in [copy_g2s_s2g_cta_vec_load_impl]:
                res = schedule(dst, src, sctx, InstType.CP_ASYNC)
                if res is not None:
                    return res
            raise ValueError(
                f"No valid implementation found for copy pipeline with strategy={pipeline.strategy}"
            )

        if op_type == PipelineOp.INIT:
            # no-op
            @T.prim_func
            def func():  # pylint: disable=function-redefined
                T.evaluate(0)

            return func
        # other ops are not supported
        raise ValueError(
            f"Copy pipeline {op_type} is not supported for strategy={pipeline.strategy}"
        )

    if impl == CopyPipeline.Impl.TMA:
        # TODO(@bohan): support private memory allocation
        mbarrier: Optional[Buffer] = pipeline.workspace.get("mbarrier", None)
        assert mbarrier is not None, "mbarrier is not found in the workspace"
        # TODO(@bohan): support other pipeline configurations
        assert pipeline.depth == 0 and not pipeline.separate_pc
        # TODO(@bohan): consider cluster multicasting cases
        # TODO(@bohan): support other launch parameters
        tx = sctx.launch_params["threadIdx.x"]
        assert "threadIdx.y" not in sctx.launch_params and "threadIdx.z" not in sctx.launch_params

        # TMA-based copy pipeline
        if op_type == PipelineOp.PRODUCER_COMMIT:
            # mbarrier.arrive_expect_tx.shared::cta.b64
            tma_bytes = args[1]

            @T.prim_func(check_well_formed=False, tirp=True)
            def func():  # pylint: disable=function-redefined
                # TODO(@bohan): fetch env variables from the launch parameters
                for tid in T.thread_binding(tx, "threadIdx.x"):
                    with T.thread()[tid == 0]:
                        T.mbarrier_arrive_expect_tx(mbarrier.access_ptr("rw"), tma_bytes)

            return func
        if op_type == PipelineOp.CONSUMER_WAIT:
            # wait mbarrier to flip
            @T.prim_func(check_well_formed=False, tirp=True)
            def func():  # pylint: disable=function-redefined
                # TODO(@bohan): fix the phase issue
                T.mbarrier_wait(mbarrier.access_ptr("rw"), 0)
                T.tvm_storage_sync("shared")

            return func
        if op_type == PipelineOp.COPY:
            dst, src = args[1], args[2]
            for schedule in [copy_g2s_cta_tma_impl]:
                res = schedule(pipeline, dst, src, sctx)
                if res is not None:
                    return res
            raise ValueError(
                f"No valid implementation found for copy pipeline with strategy={pipeline.strategy}"
            )
        if op_type == PipelineOp.INIT:
            # initialize the mbarrier, make sure the initialization is visible to all threads
            @T.prim_func(check_well_formed=False, tirp=True)
            def func():  # pylint: disable=function-redefined
                for tid in T.thread_binding(tx, "threadIdx.x"):
                    with T.thread()[tid == 0]:
                        T.mbarrier_init(mbarrier.access_ptr("rw"), 1)
                        T.cuda_fence_proxy_async("shared")
                T.tvm_storage_sync("shared")

            return func
        # other ops are not supported
        raise ValueError(
            f"Copy pipeline {op_type} is not supported for strategy={pipeline.strategy}"
        )


@register_schedule("pipeline_init", "cuda")
@target_cuda
def pipeline_init(op: OpCall, sctx: ScheduleContext) -> Optional[PrimFunc]:
    """Schedule pipeline initialization."""
    if sctx.exec_scope.name != "cta":
        return None
    pipeline = op.args[0]
    if isinstance(pipeline, CopyPipeline):
        return copy_pipeline_cta_impl(PipelineOp.INIT, sctx, *op.args)

    return None


@register_schedule("pipeline_producer_commit", "cuda")
@target_cuda
def pipeline_producer_commit(op: OpCall, sctx: ScheduleContext) -> Optional[PrimFunc]:
    """Schedule pipeline producer commit."""
    if sctx.exec_scope.name != "cta":
        return None
    pipeline = op.args[0]
    if isinstance(pipeline, CopyPipeline):
        return copy_pipeline_cta_impl(PipelineOp.PRODUCER_COMMIT, sctx, *op.args)

    return None


@register_schedule("pipeline_consumer_wait", "cuda")
@target_cuda
def pipeline_consumer_wait(op: OpCall, sctx: ScheduleContext) -> Optional[PrimFunc]:
    """Schedule pipeline consumer wait."""
    if sctx.exec_scope.name != "cta":
        return None
    pipeline = op.args[0]
    if isinstance(pipeline, CopyPipeline):
        return copy_pipeline_cta_impl(PipelineOp.CONSUMER_WAIT, sctx, *op.args)

    return None


@register_schedule("pipeline_copy", "cuda")
@target_cuda
def pipeline_copy(op: OpCall, sctx: ScheduleContext) -> Optional[PrimFunc]:
    """Schedule pipeline copy."""
    if sctx.exec_scope.name != "cta":
        return None
    pipeline = op.args[0]
    if isinstance(pipeline, CopyPipeline):
        return copy_pipeline_cta_impl(PipelineOp.COPY, sctx, *op.args)
    return None
