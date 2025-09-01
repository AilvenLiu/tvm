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
"""IRBuilder for TIR"""

from tvm_ffi import register_object as _register_object

from tvm.ir.expr import Range
from tvm.tirx import Buffer, PrimExpr, Var
from tvm.tirx.event import BulkGroupEvent, SemaphoreEventTensor

from ..base import IRBuilderFrame
from . import _ffi_api


@_register_object("script.ir_builder.tirx.TIRFrame")
class TIRFrame(IRBuilderFrame): ...


@_register_object("script.ir_builder.tirx.PrimFuncFrame")
class PrimFuncFrame(TIRFrame): ...


@_register_object("script.ir_builder.tir.SSBlockFrame")
class SBlockFrame(TIRFrame):
    ...

    def __getitem__(self, slices) -> "BlockFrame":
        """Slice operator for block frame.

        Parameters
        ----------
        slices : Union[Range, Tuple[Range, ...]]
            The slices to apply to the block frame.

        Returns
        -------
        BlockFrame
            A new block frame with the slices applied.
        """
        if not isinstance(slices, tuple):
            slices = (slices,)
        if len(slices) == 1 and isinstance(slices[0], PrimExpr):
            # If the slice is a single PrimExpr, it is a select condition
            return _ffi_api.BlockFrameSlice(self, slices[0])  # pylint: disable=no-member
        # Otherwise, the slices are a list of ranges
        slices_t = []
        for s in slices:
            if isinstance(s, slice):
                assert s.step is None, "Slice step is not supported"
                slices_t.append(Range(s.start, s.stop))
            elif isinstance(s, Range):
                slices_t.append(s)
            else:
                assert False, f"Slice must be a slice or Range, got {s} of type {type(s)}"
        return _ffi_api.BlockFrameSlice(self, slices_t)  # pylint: disable=no-member


@_register_object("script.ir_builder.tirx.SBlockInitFrame")
class BlockInitFrame(TIRFrame): ...


@_register_object("script.ir_builder.tirx.ForFrame")
class ForFrame(TIRFrame):
    def __enter__(self) -> Var | list[Var]:  # type: ignore[override]
        super().__enter__()
        return self.vars if len(self.vars) > 1 else self.vars[0]


@_register_object("script.ir_builder.tirx.AssertFrame")
class AssertFrame(TIRFrame): ...


@_register_object("script.ir_builder.tir.LetFrame")
class LetFrame(TIRFrame):
    def __enter__(self) -> Var:
        super().__enter__()
        return self.var


@_register_object("script.ir_builder.tir.RealizeFrame")
class RealizeFrame(TIRFrame): ...


@_register_object("script.ir_builder.tir.AllocateFrame")
class AllocateFrame(TIRFrame):
    def __enter__(self) -> Buffer:
        super().__enter__()
        return self.buffer_var


@_register_object("script.ir_builder.tir.AttrFrame")
class AttrFrame(TIRFrame): ...


@_register_object("script.ir_builder.tirx.WhileFrame")
class WhileFrame(TIRFrame): ...


@_register_object("script.ir_builder.tirx.IfFrame")
class IfFrame(TIRFrame): ...


@_register_object("script.ir_builder.tirx.ThenFrame")
class ThenFrame(TIRFrame): ...


@_register_object("script.ir_builder.tirx.ElseFrame")
class ElseFrame(TIRFrame): ...


@_register_object("script.ir_builder.tir.DeclBufferFrame")
class DeclBufferFrame(TIRFrame):
    def __enter__(self) -> Buffer:
        super().__enter__()
        return self.buffer


@_register_object("script.ir_builder.tir.LaunchThreadFrame")
class LaunchThreadFrame(TIRFrame):
    def __enter__(self) -> Var:
        super().__enter__()
        return self.iter_var.var


@_register_object("script.ir_builder.tir.ComposeOpFrame")
class ComposeOpFrame(TIRFrame): ...


@_register_object("script.ir_builder.tir.AllocBufferFrame")
class AllocBufferFrame(TIRFrame):
    def __enter__(self) -> Buffer:
        super().__enter__()
        return self.buffer


@_register_object("script.ir_builder.tir.AllocBulkGroupEventFrame")
class AllocBulkGroupEventFrame(TIRFrame):
    def __enter__(self) -> BulkGroupEvent:
        super().__enter__()
        return self.bulk_group_event


@_register_object("script.ir_builder.tir.AllocSemaphoreEventTensorFrame")
class AllocSemaphoreEventTensorFrame(TIRFrame):
    def __enter__(self) -> SemaphoreEventTensor:
        super().__enter__()
        return self.sem_event_tensor
