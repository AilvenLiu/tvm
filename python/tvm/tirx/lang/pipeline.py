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
        Descriptive name. Propagated to the underlying buffer so that
        debug tools (e.g. synccheck) can surface it in reports.
    """

    def __init__(self, pool, depth, name="mbar", phase_offset=0):
        self.buf = pool.alloc((depth,), "uint64", align=8, name=name)
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


@Tx.meta_class
class Pipe:
    """Full+empty barrier pair for a software-pipelined data flow.

    Wraps a full barrier (signaled when data is ready) and an optional
    empty barrier (signaled when a slot is consumed) into a single object.
    Provides factory methods for common barrier type combinations.

    Parameters
    ----------
    pool : SMEMPool
        Shared memory pool allocator.
    stages : int
        Number of pipeline stages (barrier slots).
    full_type : type
        Barrier class for the full signal (TMABar, TCGen05Bar, or MBarrier).
    empty_type : type or None
        Barrier class for the empty signal, or None for one-way pipes.
    init_full : int
        Expected arrival count for the full barrier.
    init_empty : int or None
        Expected arrival count for the empty barrier.
    name : str
        Descriptive name prefix for generated barriers.
    """

    def __init__(
        self,
        pool,
        stages,
        *,
        full_type=MBarrier,
        empty_type=None,
        init_full=1,
        init_empty=1,
        empty_phase_offset=0,
        name="pipe",
    ):
        self.full = full_type(pool, stages, name=f"{name}_full")
        if empty_type is not None:
            self.empty = empty_type(
                pool, stages, name=f"{name}_empty", phase_offset=empty_phase_offset
            )
        else:
            self.empty = None
        self.stages = stages
        self.name = name
        self.full.init(init_full)
        if self.empty is not None:
            self.empty.init(init_empty)

    @classmethod
    def tma(cls, pool, stages, *, empty_count=1, empty_phase_offset=0, name="pipe"):
        """TMA -> consumer: full=TMABar, empty=TCGen05Bar."""
        return cls(
            pool,
            stages,
            full_type=TMABar,
            empty_type=TCGen05Bar,
            init_full=1,
            init_empty=empty_count,
            empty_phase_offset=empty_phase_offset,
            name=name,
        )

    @classmethod
    def tcgen05(cls, pool, stages, *, empty_count=None, empty_phase_offset=0, name="pipe"):
        """TCGen05 -> consumer: full=TCGen05Bar, empty=MBarrier (if empty_count given)."""
        return cls(
            pool,
            stages,
            full_type=TCGen05Bar,
            empty_type=MBarrier if empty_count is not None else None,
            init_full=1,
            init_empty=empty_count,
            empty_phase_offset=empty_phase_offset,
            name=name,
        )

    @classmethod
    def mbar(cls, pool, stages, *, full_count, empty_count=None, empty_phase_offset=0, name="pipe"):
        """Thread -> thread: full=MBarrier, empty=MBarrier (if empty_count given)."""
        return cls(
            pool,
            stages,
            full_type=MBarrier,
            empty_type=MBarrier if empty_count is not None else None,
            init_full=full_count,
            init_empty=empty_count,
            empty_phase_offset=empty_phase_offset,
            name=name,
        )

    def cursor(self, role="consumer"):
        """Create a PipeCursor for this pipe."""
        return PipeCursor(self, role)


@Tx.meta_class
class PipeCursor:
    """Automatic phase/stage tracking for a Pipe.

    Wraps a Pipe and a PipelineState to provide wait/signal/advance
    operations that automatically manage stage and phase progression.

    Parameters
    ----------
    pipe : Pipe
        The pipe to track.
    role : str
        Either ``"producer"`` or ``"consumer"``.
    """

    def __init__(self, pipe, role):
        self.pipe = pipe
        self.role = role
        self._state = PipelineState(f"{pipe.name}_{role}", pipe.stages)
        self._state.init(is_producer=(role == "producer"))

    @property
    def stage(self):
        return self._state.stage

    @property
    def phase(self):
        return self._state.phase

    @Tx.inline
    def wait(self):
        """Producer: wait for empty slot. Consumer: wait for full data."""
        if self.role == "producer":
            self.pipe.empty.wait(self.stage, self.phase)
        else:
            self.pipe.full.wait(self.stage, self.phase)

    @Tx.inline
    def signal(self, **kwargs):
        """Producer: signal full. Consumer: signal empty."""
        if self.role == "producer":
            self.pipe.full.arrive(self.stage, **kwargs)
        else:
            self.pipe.empty.arrive(self.stage, **kwargs)

    @Tx.inline
    def advance(self):
        """Move to the next pipeline stage."""
        self._state.move_to_next_stage()

    def snapshot(self):
        """Freeze current (stage, phase) for deferred use."""
        return (self.stage, self.phase)
