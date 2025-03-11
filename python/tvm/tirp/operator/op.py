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

"""Implementation of TIR operator."""

from typing import List, Tuple

from tvm.tir.stmt import OpCall
from tvm.tir import PrimExpr, BufferRegion, FloatImm
from tvm.ir import Op
from tvm.tir.async_structs import Pipeline
from tvm.tir.predicate import Predicate


def get_tirp_op(op_name: str):
    assert isinstance(op_name, str)
    return Op.get("tirp." + op_name)


### Base Operator Classes ###
class UnaryOp(OpCall):
    """Base class for unary operators: unary(output, input).

    Unary operators take a single input tensor and produce a single output tensor.
    """

    scalar_input = False

    @property
    def srcs(self) -> List[PrimExpr]:
        """Get the source expression (input) of the operator."""
        return [self.args[1]]

    @property
    def dsts(self) -> List[PrimExpr]:
        """Get the destination expression (output) of the operator."""
        return [self.args[0]]

    def validate(self) -> None:
        """Validate that the operator has the correct number and types of arguments."""
        assert len(self.args) == 2, f"{self} expects 2 arguments, got {len(self.args)}"
        assert isinstance(
            self.args[0], BufferRegion
        ), f"{self} expects BufferRegion as output, got {self.args[0]}"
        if self.scalar_input:
            assert isinstance(
                self.args[1], FloatImm
            ), f"{self} expects FloatImm as value, got {self.args[1]}"
        else:
            assert isinstance(
                self.args[1], BufferRegion
            ), f"{self} expects BufferRegion as value, got {self.args[1]}"


class UnaryOpWithBiasScale(UnaryOp):
    """Extended unary operator with bias and scale parameters: unary_with_bias_scale(output, input, bias, scale).

    These operators support additional bias and scale parameters for more complex operations (only on trn).
    output = unary(input * scale + bias)
    """

    @property
    def srcs(self) -> List[PrimExpr]:
        """Get the source expressions (inputs) of the operator."""
        return [self.args[i] for i in range(1, 4)]

    def validate(self) -> None:
        """Validate that the operator has the correct number and types of arguments."""
        assert len(self.args) == 4, f"{self} expects 4 arguments, got {len(self.args)}"
        assert all(
            isinstance(arg, BufferRegion) for arg in self.args[:2]
        ), f"{self} expects BufferRegion arguments as input and output, got {self.args[:2]}"


class BinaryOp(OpCall):
    """Base class for binary operators: binary(output, input0, input1).

    Binary operators take two input tensors and produce a single output tensor.
    """

    @property
    def srcs(self) -> List[PrimExpr]:
        """Get the source expressions (inputs) of the operator."""
        return [self.args[1], self.args[2]]

    @property
    def dsts(self) -> List[PrimExpr]:
        """Get the destination expression (output) of the operator."""
        return [self.args[0]]

    def validate(self) -> None:
        """Validate that the operator has the correct number and types of arguments."""
        assert len(self.args) == 3, f"{self} expects 3 arguments, got {len(self.args)}"
        assert isinstance(
            self.dsts[0], BufferRegion
        ), f"{self} expects BufferRegion as output, got {self.dsts[0]}"
        assert all(
            isinstance(arg, (BufferRegion, FloatImm)) for arg in self.srcs
        ), f"{self} expects BufferRegion or FloatImm arguments as inputs, got {self.srcs}"


class ReduceOp(OpCall):
    """Base class for reduction operators: reduce(output, input, reduce_axes, accum).

    Reduction operators reduce one or more dimensions of the input tensor.
    """

    @property
    def srcs(self) -> List[PrimExpr]:
        """Get the source expression (input) of the operator."""
        return [self.args[1]]

    @property
    def dsts(self) -> List[PrimExpr]:
        """Get the destination expression (output) of the operator."""
        return [self.args[0]]

    @property
    def reduce_axes(self) -> Tuple[int]:
        """Get the axes to reduce."""
        return self.args[2]

    @property
    def accum(self) -> bool:
        """Whether to accumulate the result into the output."""
        return self.args[3]

    def validate(self) -> None:
        """Validate that the operator has the correct number and types of arguments."""
        assert len(self.args) == 4, f"{self} expects 4 arguments, got {len(self.args)}"
        assert isinstance(
            self.args[0], BufferRegion
        ), f"{self} expects BufferRegion as output, got {self.args[0]}"
        assert isinstance(
            self.args[1], BufferRegion
        ), f"{self} expects BufferRegion as input, got {self.args[1]}"
        assert isinstance(self.accum, bool), f"{self} expects bool as accum, got {self.accum}"


class PipelineOp(OpCall):
    """Base class for pipeline operators.

    Pipeline operators manage the execution pipeline for tensor operations.
    """

    @property
    def srcs(self) -> List[PrimExpr]:
        """Get the source expressions (inputs) of the operator."""
        return []

    @property
    def dsts(self) -> List[PrimExpr]:
        """Get the destination expressions (outputs) of the operator."""
        return []

    def validate(self) -> None:
        """Validate that the operator has the correct number and types of arguments."""
        assert isinstance(
            self.args[0], Pipeline
        ), f"{self} expects Pipeline as argument, got {self.args[0]}"


### Schedule Operators ###
class Zero(UnaryOp):
    """Zero out all elements in src and store to dst."""

    op = get_tirp_op("zero")


class Sqrt(UnaryOpWithBiasScale):
    """Compute square root of all elements in src and store to dst.

    If bias and scale are provided: dst = sqrt(src * scale + bias)
    """

    op = get_tirp_op("sqrt")


class Copy(UnaryOp):
    """Copy all elements from src to dst."""

    op = get_tirp_op("copy")


class Fill(UnaryOp):
    """Fill dst with a scalar value."""

    op = get_tirp_op("fill")
    scalar_input = True


class Add(BinaryOp):
    """Add src1 and src2 element-wise and store to dst."""

    op = get_tirp_op("add")


class Sub(BinaryOp):
    """Subtract src2 from src1 element-wise and store to dst."""

    op = get_tirp_op("sub")


class Mul(BinaryOp):
    """Multiply src1 and src2 element-wise and store to dst."""

    op = get_tirp_op("mul")


class FDiv(BinaryOp):
    """Divide src1 by src2 element-wise using floating point division and store to dst."""

    op = get_tirp_op("fdiv")


class Gemm(OpCall):
    """General matrix multiplication: D = A * B * alpha + C * beta.

    Args:
        D: Output matrix
        A: First input matrix
        B: Second input matrix
        C: Third input matrix (for bias)
        transpose_A: Whether to transpose A
        transpose_B: Whether to transpose B
        alpha: Scalar multiplier for A*B
        beta: Scalar multiplier for C
    """

    op = get_tirp_op("gemm")

    @property
    def srcs(self) -> List[PrimExpr]:
        """Get the source matrices."""
        return [self.args[1], self.args[2], self.args[3]]

    @property
    def dsts(self) -> List[PrimExpr]:
        """Get the destination matrix."""
        return [self.args[0]]

    @property
    def transpose_A(self) -> bool:
        """Whether to transpose matrix A."""
        return self.args[4]

    @property
    def transpose_B(self) -> bool:
        """Whether to transpose matrix B."""
        return self.args[5]

    @property
    def alpha(self) -> PrimExpr:
        """Scalar multiplier for A*B."""
        return self.args[6]

    @property
    def beta(self) -> PrimExpr:
        """Scalar multiplier for C."""
        return self.args[7]

    def validate(self) -> None:
        """Validate that the operator has the correct number and types of arguments."""
        assert len(self.args) == 8, f"{self} expects 8 arguments, got {len(self.args)}"
        assert all(
            isinstance(arg, BufferRegion) for arg in self.args[:3]
        ), f"{self} expects BufferRegion arguments as A, B and D, got {self.args[:3]}"
        assert isinstance(
            self.args[3], (BufferRegion, FloatImm)
        ), f"{self} expects BufferRegion or FloatImm arguments as C, got {self.args[3]}"
        assert isinstance(
            self.transpose_A, bool
        ), f"{self} expects bool arguments as transpose_A, got {self.transpose_A}"
        assert isinstance(
            self.transpose_B, bool
        ), f"{self} expects bool arguments as transpose_B, got {self.transpose_B}"
        assert isinstance(
            self.alpha, FloatImm
        ), f"{self} expects FloatImm as alpha, got {self.alpha}"
        assert isinstance(self.beta, FloatImm), f"{self} expects FloatImm as beta, got {self.beta}"


class Sum(ReduceOp):
    """Sum elements in src along specified axes and store in dst."""

    op = get_tirp_op("sum")


class Max(ReduceOp):
    """Compute maximum value in src along specified axes and store in dst."""

    op = get_tirp_op("max")


class Min(ReduceOp):
    """Compute minimum value in src along specified axes and store in dst."""

    op = get_tirp_op("min")


class Reciprocal(UnaryOp):
    """Compute reciprocal (1/x) for all elements in src and store to dst."""

    op = get_tirp_op("reciprocal")


class Memset(UnaryOp):
    """Set all elements in dst to a specified value."""

    op = get_tirp_op("memset")
    scalar_input = True


class Maximum(BinaryOp):
    """Compute element-wise maximum of src1 and src2 and store to dst."""

    op = get_tirp_op("maximum")


class Minimum(BinaryOp):
    """Compute element-wise minimum of src1 and src2 and store to dst."""

    op = get_tirp_op("minimum")


class Exp(UnaryOpWithBiasScale):
    """Compute exponential (e^x) of all elements in src and store to dst.

    If bias and scale are provided: dst = exp(src * scale + bias)
    """

    op = get_tirp_op("exp")


class Select(BinaryOp):
    """Select elements from src1 or src2 based on the predicate.

    select(dst, src1, src2, predicate)
    """

    op = get_tirp_op("select")

    @property
    def predicate(self) -> Predicate:
        """Get the predicate."""
        return self.args[3]

    def validate(self) -> None:
        """Validate that the operator has the correct number and types of arguments."""
        assert len(self.args) == 4, f"{self} expects 4 arguments, got {len(self.args)}"
        assert isinstance(
            self.dsts[0], BufferRegion
        ), f"{self} expects BufferRegion as output, got {self.dsts[0]}"
        assert all(
            isinstance(arg, (BufferRegion, FloatImm)) for arg in self.srcs
        ), f"{self} expects BufferRegion or FloatImm arguments as inputs, got {self.srcs}"
        assert isinstance(
            self.args[3], Predicate
        ), f"{self} expects Predicate as predicate, got {self.args[3]}"


### Pipeline Ops ###
class PipelineInit(PipelineOp):
    """Initialize a pipeline."""

    op = get_tirp_op("pipeline_init")


class PipelineProducerAcquire(PipelineOp):
    """Acquire a producer slot in the pipeline."""

    op = get_tirp_op("pipeline_producer_acquire")


class PipelineCopy(PipelineOp):
    """Copy data through the pipeline from src to dst."""

    op = get_tirp_op("pipeline_copy")

    @property
    def srcs(self) -> List[PrimExpr]:
        """Get the source buffer region."""
        return [self.args[2]]

    @property
    def dsts(self) -> List[PrimExpr]:
        """Get the destination buffer region."""
        return [self.args[1]]

    def validate(self) -> None:
        """Validate that the operator has the correct number and types of arguments."""
        super().validate()
        assert len(self.args) == 3, f"{self} expects 3 arguments, got {len(self.args)}"
        assert isinstance(
            self.dsts[0], BufferRegion
        ), f"{self} expects BufferRegion as output, got {self.dsts[0]}"
        assert isinstance(
            self.srcs[0], BufferRegion
        ), f"{self} expects BufferRegion as input, got {self.srcs[0]}"


class PipelineProducerCommit(PipelineOp):
    """Commit a producer operation in the pipeline."""

    op = get_tirp_op("pipeline_producer_commit")


class PipelineConsumerWait(PipelineOp):
    """Wait for data to be available for consumption in the pipeline."""

    op = get_tirp_op("pipeline_consumer_wait")


class PipelineConsumerRelease(PipelineOp):
    """Release a consumer slot after data has been consumed."""

    op = get_tirp_op("pipeline_consumer_release")


class KernelReplacePoint(OpCall):
    """A placeholder for kernel replacement points in TIR scheduling."""

    op = get_tirp_op("tvm_kernel_replace_point")

    @property
    def srcs(self) -> List[PrimExpr]:
        """Get the source expressions (inputs) of the operator."""
        return []

    @property
    def dsts(self) -> List[PrimExpr]:
        """Get the destination expressions (outputs) of the operator."""
        return []

    def validate(self) -> None:
        """Validate that the operator has the correct number and types of arguments."""
        assert len(self.args) == 0, f"{self} expects 0 arguments, got {len(self.args)}"


### Compose Ops ###
class BinaryReduce(OpCall):
    """Combine a binary operation with a reduction operation.

    binary_reduce(binary_output, reduce_output, binary_input1, binary_input2, binary_op, reduce_op, reduce_axes, )
    """

    op = get_tirp_op("binary_reduce")

    @property
    def srcs(self) -> List[PrimExpr]:
        """Get the source expressions (inputs) of the operator."""
        return [self.args[2], self.args[3]]

    @property
    def dsts(self) -> List[PrimExpr]:
        """Get the destination expressions (outputs) of the operator."""
        return [self.args[0], self.args[1]]

    @property
    def reduce_axes(self) -> Tuple[int]:
        """Get the axes to reduce."""
        return self.args[6]

    @property
    def binary_op(self) -> Op:
        """Get the binary operation."""
        return self.args[4]

    @property
    def reduce_op(self) -> Op:
        """Get the reduction operation."""
        return self.args[5]

    def validate(self) -> None:
        assert len(self.args) == 7, f"{self} expects 7 arguments, got {len(self.args)}"
        assert all(
            isinstance(arg, BufferRegion) for arg in self.dsts
        ), f"{self} expects BufferRegion arguments as binary_output and reduce_output, got {self.dsts}"
        assert all(
            isinstance(arg, (BufferRegion, FloatImm)) for arg in self.srcs
        ), f"{self} expects BufferRegion or FloatImm arguments as binary_input1 and binary_input2, got {self.srcs}"
        assert isinstance(
            self.binary_op, Op
        ), f"{self} expects Op as binary_op, got {self.binary_op}"
        assert isinstance(
            self.reduce_op, Op
        ), f"{self} expects Op as reduce_op, got {self.reduce_op}"


class UnaryReduce(OpCall):
    """Combine a unary operation with a reduction operation.

    unary_reduce(unary_output, reduce_output, unary_input, unary_op, reduce_op, bias, scale, reduce_axes)
    """

    op = get_tirp_op("unary_reduce")

    @property
    def srcs(self) -> List[PrimExpr]:
        """Get the source expressions (inputs) of the operator."""
        return [self.args[2], self.args[5], self.args[6]]

    @property
    def dsts(self) -> List[PrimExpr]:
        """Get the destination expressions (outputs) of the operator."""
        return [self.args[0], self.args[1]]

    @property
    def reduce_axes(self) -> Tuple[int]:
        """Get the axes to reduce."""
        return self.args[7]

    @property
    def unary_op(self) -> Op:
        """Get the unary operation."""
        return self.args[3]

    @property
    def reduce_op(self) -> Op:
        """Get the reduction operation."""
        return self.args[4]

    def validate(self) -> None:
        """Validate that the operator has the correct number and types of arguments."""
        assert len(self.args) == 8, f"{self} expects 8 arguments, got {len(self.args)}"
        assert all(
            isinstance(arg, BufferRegion) for arg in self.dsts
        ), f"{self} expects BufferRegion arguments as unary_output and reduce_output, got {self.dsts}"
        assert isinstance(
            self.srcs[0], (BufferRegion, FloatImm)
        ), f"{self} expects BufferRegion or FloatImm arguments as unary_input, got {self.srcs[0]}"
        assert isinstance(self.unary_op, Op), f"{self} expects Op as unary_op, got {self.unary_op}"
        assert isinstance(
            self.reduce_op, Op
        ), f"{self} expects Op as reduce_op, got {self.reduce_op}"


class BinaryChain(OpCall):
    """Chain multiple binary operations together.

    binary_chain(output, data, operand0, operand1, op0, op1, reverse1)

    if not reverse1:
        output = (operand0 op0 data) op1 operand1
    else:
        output = operand1 op1 (operand0 op0 data)
    """

    op = get_tirp_op("binary_chain")

    @property
    def srcs(self) -> List[PrimExpr]:
        """Get the source expressions (inputs) of the operator."""
        return [self.args[i] for i in range(1, 4)]

    @property
    def dsts(self) -> List[PrimExpr]:
        """Get the destination expressions (outputs) of the operator."""
        return [self.args[0]]

    @property
    def reverse1(self) -> bool:
        """Whether to reverse the order of the second binary operation."""
        return self.args[6]

    @property
    def op0(self) -> Op:
        """Get the first binary operation."""
        return self.args[4]

    @property
    def op1(self) -> Op:
        """Get the second binary operation."""
        return self.args[5]

    def validate(self) -> None:
        """Validate that the operator has the correct number and types of arguments."""
        assert len(self.args) == 7, f"{self} expects 7 arguments, got {len(self.args)}"
        assert isinstance(
            self.dsts[0], BufferRegion
        ), f"{self} expects BufferRegion as output, got {self.dsts[0]}"
        assert all(
            isinstance(arg, (BufferRegion, FloatImm)) for arg in self.srcs
        ), f"{self} expects BufferRegion or FloatImm arguments as data, operand0 and operand1, got {self.srcs}"
        assert isinstance(self.op0, Op), f"{self} expects Op as op0, got {self.op0}"
        assert isinstance(self.op1, Op), f"{self} expects Op as op1, got {self.op1}"


class ReduceNegate(ReduceOp):
    """
    Negate the result of a reduction operation.

    reduce_negate(output, input, reduce_axes, accum, reduce_op)
    """

    op = get_tirp_op("reduce_negate")

    @property
    def reduce_op(self) -> Op:
        """Get the reduction operation."""
        return self.args[4]

    def validate(self) -> None:
        """Validate that the operator has the correct number and types of arguments."""
        assert len(self.args) == 5, f"{self} expects 5 arguments, got {len(self.args)}"
        assert isinstance(
            self.dsts[0], BufferRegion
        ), f"{self} expects BufferRegion as output, got {self.dsts[0]}"
        assert isinstance(
            self.srcs[0], BufferRegion
        ), f"{self} expects BufferRegion as input, got {self.srcs[0]}"
        assert isinstance(
            self.reduce_op, Op
        ), f"{self} expects Op as reduce_op, got {self.reduce_op}"


class ComposeOp(OpCall):
    """Generic operator for composition of multiple operations.

    Must be lowered to specific compose operations before operator-level passes.
    """

    # TODO: add a pass to lower generic compose_op to specific compose ops

    op = get_tirp_op("compose_op")

    @property
    def srcs(self) -> List[PrimExpr]:
        """Get the source expressions (inputs) of the operator."""
        raise NotImplementedError(
            "Generic compose_op must be lowered to specific compose ops before operator-level passes"
        )

    @property
    def dsts(self) -> List[PrimExpr]:
        """Get the destination expressions (outputs) of the operator."""
        raise NotImplementedError(
            "Generic compose_op must be lowered to specific compose ops before operator-level passes"
        )
