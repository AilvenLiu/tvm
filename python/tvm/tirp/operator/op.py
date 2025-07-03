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

from typing import List, Tuple, Dict, Any, Optional

import tvm
from tvm.tir.stmt import OpCall
from tvm.tir import PrimExpr, BufferRegion, FloatImm, Buffer, Stmt, IntImm, BufferLoad
from tvm.ir import Op
from tvm.tir.async_structs import Pipeline
from tvm.tir.predicate import Predicate
from tvm.script import tir as T
from tvm.target import Target


def get_tirp_op(op_name: str):
    assert isinstance(op_name, str)
    return Op.get("tirp." + op_name)


class ArgProperty:
    def __init__(self, index):
        self.index = index

    def __get__(self, obj, objtype=None):
        assert obj is not None, "OpCall cannot be None"
        return obj.args[self.index]


### Base Operator Classes ###
class UnaryOp(OpCall):
    """Base class for unary operators: unary(output, input).

    Unary operators take a single input tensor and produce a single output tensor.
    """

    scalar_input = False
    output = ArgProperty(0)
    input = ArgProperty(1)

    @property
    def srcs(self) -> List[PrimExpr]:
        """Get the source expression (input) of the operator."""
        return [self.input]

    @property
    def dsts(self) -> List[PrimExpr]:
        """Get the destination expression (output) of the operator."""
        return [self.output]

    def validate(self) -> None:
        """Validate that the operator has the correct number and types of arguments."""
        assert len(self.args) == 2, f"{self} expects 2 arguments, got {len(self.args)}"
        assert isinstance(
            self.output, BufferRegion
        ), f"{self} expects BufferRegion as output, got {self.output}"
        if self.scalar_input:
            assert isinstance(
                self.input, (FloatImm, IntImm)
            ), f"{self} expects FloatImm or IntImm as value, got {self.input}"
        else:
            assert isinstance(
                self.input, BufferRegion
            ), f"{self} expects BufferRegion as value, got {self.input}"


class UnaryOpWithBiasScale(UnaryOp):
    """Extended unary operator with bias and scale parameters: unary_with_bias_scale(output, input, bias, scale).

    These operators support additional bias and scale parameters for more complex operations (only on trn).
    output = unary(input * scale + bias)
    """

    bias = ArgProperty(2)
    scale = ArgProperty(3)

    @property
    def srcs(self) -> List[PrimExpr]:
        """Get the source expressions (inputs) of the operator."""
        return [self.input, self.bias, self.scale]

    def validate(self) -> None:
        """Validate that the operator has the correct number and types of arguments."""
        assert len(self.args) == 4, f"{self} expects 4 arguments, got {len(self.args)}"
        assert all(
            isinstance(arg, BufferRegion) for arg in (self.output, self.input)
        ), f"{self} expects BufferRegion arguments as input and output, got {self.args[:2]}"


class BinaryOp(OpCall):
    """Base class for binary operators: binary(output, input0, input1).

    Binary operators take two input tensors and produce a single output tensor.
    """

    lhs = ArgProperty(1)
    rhs = ArgProperty(2)
    output = ArgProperty(0)

    @property
    def srcs(self) -> List[PrimExpr]:
        """Get the source expressions (inputs) of the operator."""
        return [self.lhs, self.rhs]

    @property
    def dsts(self) -> List[PrimExpr]:
        """Get the destination expression (output) of the operator."""
        return [self.output]

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

    input = ArgProperty(1)
    output = ArgProperty(0)
    reduce_axes = ArgProperty(2)
    accum = ArgProperty(3)

    @property
    def srcs(self) -> List[PrimExpr]:
        """Get the source expression (input) of the operator."""
        return [self.input]

    @property
    def dsts(self) -> List[PrimExpr]:
        """Get the destination expression (output) of the operator."""
        return [self.output]

    def validate(self) -> None:
        """Validate that the operator has the correct number and types of arguments."""
        assert len(self.args) == 4, f"{self} expects 4 arguments, got {len(self.args)}"
        assert isinstance(
            self.output, BufferRegion
        ), f"{self} expects BufferRegion as output, got {self.output}"
        assert isinstance(
            self.input, BufferRegion
        ), f"{self} expects BufferRegion as input, got {self.input}"
        assert isinstance(
            self.accum, (bool, IntImm)
        ), f"{self} expects bool or IntImm as accum, got {self.accum}"


class PipelineOp(OpCall):
    """Base class for pipeline operators.

    Pipeline operators manage the execution pipeline for tensor operations.
    """

    pipeline = ArgProperty(0)

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
            self.pipeline, Pipeline
        ), f"{self} expects Pipeline as argument, got {self.pipeline}"


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
    output = ArgProperty(0)
    lhs = ArgProperty(1)
    rhs = ArgProperty(2)
    bias = ArgProperty(3)
    transpose_A = ArgProperty(4)
    transpose_B = ArgProperty(5)
    alpha = ArgProperty(6)
    beta = ArgProperty(7)

    @property
    def srcs(self) -> List[PrimExpr]:
        """Get the source matrices."""
        return [self.lhs, self.rhs, self.bias]

    @property
    def dsts(self) -> List[PrimExpr]:
        """Get the destination matrix."""
        return [self.output]

    def validate(self) -> None:
        """Validate that the operator has the correct number and types of arguments."""
        assert len(self.args) == 8, f"{self} expects 8 arguments, got {len(self.args)}"
        assert all(
            isinstance(arg, BufferRegion) for arg in (self.output, self.lhs, self.rhs)
        ), f"{self} expects BufferRegion arguments as output, lhs and rhs, got {(self.output, self.lhs, self.rhs)}"
        assert isinstance(
            self.bias, (BufferRegion, FloatImm)
        ), f"{self} expects BufferRegion or FloatImm arguments as bias, got {self.bias}"
        assert isinstance(
            self.transpose_A, (bool, IntImm)
        ), f"{self} expects bool or IntImm arguments as transpose_A, got {self.transpose_A}"
        assert isinstance(
            self.transpose_B, (bool, IntImm)
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
    predicate = ArgProperty(3)

    def validate(self) -> None:
        """Validate that the operator has the correct number and types of arguments."""
        assert len(self.args) == 4, f"{self} expects 4 arguments, got {len(self.args)}"
        assert isinstance(
            self.output, BufferRegion
        ), f"{self} expects BufferRegion as output, got {self.output}"
        assert all(
            isinstance(arg, (BufferRegion, FloatImm)) for arg in (self.lhs, self.rhs)
        ), f"{self} expects BufferRegion or FloatImm arguments as inputs, got {(self.lhs, self.rhs)}"
        assert isinstance(
            self.predicate, Predicate
        ), f"{self} expects Predicate as predicate, got {self.predicate}"


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
    input = ArgProperty(2)
    output = ArgProperty(1)

    @property
    def srcs(self) -> List[PrimExpr]:
        """Get the source buffer region."""
        return [self.input]

    @property
    def dsts(self) -> List[PrimExpr]:
        """Get the destination buffer region."""
        return [self.output]

    def validate(self) -> None:
        """Validate that the operator has the correct number and types of arguments."""
        super().validate()
        assert len(self.args) == 3, f"{self} expects 3 arguments, got {len(self.args)}"
        assert isinstance(
            self.output, BufferRegion
        ), f"{self} expects BufferRegion as output, got {self.output}"
        assert isinstance(
            self.input, BufferRegion
        ), f"{self} expects BufferRegion as input, got {self.input}"


class PipelineProducerCommit(PipelineOp):
    """Commit a producer operation in the pipeline."""

    op = get_tirp_op("pipeline_producer_commit")


class PipelineConsumerWait(PipelineOp):
    """Wait for data to be available for consumption in the pipeline."""

    op = get_tirp_op("pipeline_consumer_wait")


class PipelineConsumerRelease(PipelineOp):
    """Release a consumer slot after data has been consumed."""

    op = get_tirp_op("pipeline_consumer_release")


class EventTensorInit(OpCall):
    """Initialize an event tensor."""

    op = get_tirp_op("event_tensor_init")

    event_tensor = ArgProperty(0)
    init_value = ArgProperty(1)

    def validate(self) -> None:
        """Validate that the operator has the correct number and types of arguments."""
        assert len(self.args) == 2, f"{self} expects 2 arguments, got {len(self.args)}"
        assert isinstance(
            self.event_tensor, Buffer
        ), f"{self} expects Buffer as event_tensor, got {self.event_tensor}"
        assert self.event_tensor.is_event_tensor(), f"{self.event_tensor} is not an event tensor"
        assert (
            isinstance(self.init_value, IntImm) or self.init_value is None
        ), f"{self} expects IntImm as init_value, got {self.init_value}"


class EventCommit(OpCall):
    """Commit an event."""

    op = get_tirp_op("event_commit")

    event = ArgProperty(0)
    tx_count = ArgProperty(1)

    def validate(self) -> None:
        """Validate that the operator has the correct number and types of arguments."""
        assert len(self.args) == 2, f"{self} expects 2 arguments, got {len(self.args)}"
        assert isinstance(
            self.event, BufferLoad
        ), f"{self} expects BufferLoad as event, got {self.event}"
        assert self.event.is_event(), f"{self.event} is not an event"


class EventWait(OpCall):
    """Wait for an event to be committed."""

    op = get_tirp_op("event_wait")

    event = ArgProperty(0)

    def validate(self) -> None:
        """Validate that the operator has the correct number and types of arguments."""
        assert len(self.args) == 1, f"{self} expects 1 argument, got {len(self.args)}"
        assert isinstance(
            self.event, BufferLoad
        ), f"{self} expects BufferLoad as event, got {self.event}"
        assert self.event.is_event(), f"{self.event} is not an event"


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

    binary_output = ArgProperty(0)
    reduce_output = ArgProperty(1)
    binary_input1 = ArgProperty(2)
    binary_input2 = ArgProperty(3)
    binary_op = ArgProperty(4)
    reduce_op = ArgProperty(5)
    reduce_axes = ArgProperty(6)

    @property
    def srcs(self) -> List[PrimExpr]:
        """Get the source expressions (inputs) of the operator."""
        return [self.binary_input1, self.binary_input2]

    @property
    def dsts(self) -> List[PrimExpr]:
        """Get the destination expressions (outputs) of the operator."""
        return [self.binary_output, self.reduce_output]

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

    unary_output = ArgProperty(0)
    reduce_output = ArgProperty(1)
    unary_input = ArgProperty(2)
    unary_op = ArgProperty(3)
    reduce_op = ArgProperty(4)
    bias = ArgProperty(5)
    scale = ArgProperty(6)
    reduce_axes = ArgProperty(7)

    @property
    def srcs(self) -> List[PrimExpr]:
        """Get the source expressions (inputs) of the operator."""
        return [self.unary_input, self.bias, self.scale]

    @property
    def dsts(self) -> List[PrimExpr]:
        """Get the destination expressions (outputs) of the operator."""
        return [self.unary_output, self.reduce_output]

    def validate(self) -> None:
        """Validate that the operator has the correct number and types of arguments."""
        assert len(self.args) == 8, f"{self} expects 8 arguments, got {len(self.args)}"
        assert all(
            isinstance(arg, BufferRegion) for arg in self.dsts
        ), f"{self} expects BufferRegion arguments as unary_output and reduce_output, got {self.dsts}"
        assert isinstance(
            self.unary_input, (BufferRegion, FloatImm)
        ), f"{self} expects BufferRegion or FloatImm arguments as unary_input, got {self.unary_input}"
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

    output = ArgProperty(0)
    data = ArgProperty(1)
    operand0 = ArgProperty(2)
    operand1 = ArgProperty(3)
    op0 = ArgProperty(4)
    op1 = ArgProperty(5)
    reverse1 = ArgProperty(6)

    @property
    def srcs(self) -> List[PrimExpr]:
        """Get the source expressions (inputs) of the operator."""
        return [self.data, self.operand0, self.operand1]

    @property
    def dsts(self) -> List[PrimExpr]:
        """Get the destination expressions (outputs) of the operator."""
        return [self.output]

    def validate(self) -> None:
        """Validate that the operator has the correct number and types of arguments."""
        assert len(self.args) == 7, f"{self} expects 7 arguments, got {len(self.args)}"
        assert isinstance(
            self.output, BufferRegion
        ), f"{self} expects BufferRegion as output, got {self.output}"
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

    reduce_op = ArgProperty(4)

    def validate(self) -> None:
        """Validate that the operator has the correct number and types of arguments."""
        assert len(self.args) == 5, f"{self} expects 5 arguments, got {len(self.args)}"
        assert isinstance(
            self.output, BufferRegion
        ), f"{self} expects BufferRegion as output, got {self.output}"
        assert isinstance(
            self.input, BufferRegion
        ), f"{self} expects BufferRegion as input, got {self.input}"
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
