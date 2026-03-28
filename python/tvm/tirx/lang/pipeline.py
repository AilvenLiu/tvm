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
"""Reusable pipeline state and mbarrier helpers for SM100 kernels.

These classes emit TIR via @Tx.inline. Decorate with @Tx.meta_class so that
instances are automatically treated as meta values inside @Tx.prim_func.
"""

from tvm.script import tirx as Tx


@Tx.meta_class
class PipelineState:
    """Tracks pipeline stage and phase for software-pipelined loops.

    Parameters
    ----------
    prefix : str
        Name prefix for the generated local cells (stage/phase).
    pipe_depth : int
        Number of pipeline stages.
    """

    def __init__(self, prefix: str, pipe_depth: int):
        self.stage = Tx.local_scalar("int32", name=prefix + "_stage")
        self.phase = Tx.local_scalar("int32", name=prefix + "_phase")
        self.pipe_depth = pipe_depth

    @Tx.inline
    def init(self, is_producer):
        self.stage = 0
        if is_producer:
            self.phase = 1
        else:
            self.phase = 0

    @Tx.inline
    def move_to_next_stage(self):
        if self.pipe_depth > 1:
            self.stage = self.stage + 1
            if self.stage == self.pipe_depth:
                self.stage = 0
                self.phase = self.phase ^ 1
        else:
            self.phase = self.phase ^ 1


@Tx.meta_class
class MBarrier:
    """Mbarrier wrapper with regular mbarrier.arrive.

    Parameters
    ----------
    pool : PoolAllocator
        Shared memory pool allocator.
    depth : int
        Number of barrier slots (one per pipeline stage).
    name : str
        Descriptive name (unused at runtime, for readability).
    """

    def __init__(self, pool, depth, name="mbar", phase_offset=0):
        self.buf = pool.alloc((depth,), "uint64", align=8)
        self.depth = depth
        self.phase_offset = phase_offset

    @Tx.inline
    def init(self, count):
        with Tx.thread()[0:1]:
            for i in Tx.unroll(self.depth):
                Tx.ptx.mbarrier.init(self.buf.ptr_to([i]), count)

    @Tx.inline
    def wait(self, stage, phase):
        Tx.ptx.mbarrier.try_wait(self.buf.ptr_to([stage]), phase ^ self.phase_offset)

    @Tx.inline
    def arrive(self, stage, cta_id=0, pred=True):
        Tx.ptx.mbarrier.arrive(self.buf.ptr_to([stage]), cta_id=cta_id, pred=pred)

    def ptr_to(self, idx):
        return self.buf.ptr_to(idx)

    def remote_view(self, rank):
        """Create a view of this barrier mapped to another CTA's shared memory."""
        from tvm.ir import PointerType, PrimType
        from tvm.tirx import Var as TIRVar

        expr = Tx.reinterpret("handle", Tx.ptx.map_shared_rank(self.buf.ptr_to([0]), rank))
        ptr = TIRVar("remote_mbar_ptr", PointerType(PrimType("uint64")))
        Tx.Bind(expr, var=ptr)
        buf = Tx.decl_buffer([self.depth], "uint64", data=ptr, scope="shared", name="remote_mbar")
        remote = object.__new__(type(self))
        remote.buf = buf
        remote.depth = self.depth
        remote.phase_offset = self.phase_offset
        return remote


class TMABar(MBarrier):
    """Barrier signaled by TMA (mbarrier.arrive.expect_tx).

    When ``tx_count`` is None, falls back to a remote mbarrier.arrive
    (matching MBarrier.arrive defaults).
    """

    @Tx.inline
    def arrive(self, stage, tx_count=None):
        if tx_count is not None:
            Tx.ptx.mbarrier.arrive.expect_tx(self.buf.ptr_to([stage]), tx_count)
        else:
            Tx.ptx.mbarrier.arrive(self.buf.ptr_to([stage]), cta_id=0, pred=True)


class TCGen05Bar(MBarrier):
    """Barrier signaled by tcgen05 commit.

    The caller is responsible for ensuring only one thread issues the
    commit (e.g. by wrapping the call in ``Tx.elected()`` or
    ``if Tx.ptx.elect_sync():``).
    """

    @Tx.inline
    def arrive(self, stage, cta_group=1, cta_mask=None):
        if cta_mask is None and cta_group == 1:
            Tx.ptx.tcgen05.commit(self.buf.ptr_to([stage]))
        else:
            Tx.ptx.tcgen05.commit(self.buf.ptr_to([stage]), cta_group=cta_group, cta_mask=cta_mask)
