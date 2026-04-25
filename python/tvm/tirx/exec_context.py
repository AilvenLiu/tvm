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
"""ExecContext: per-program-point active-thread state.

The compiler maintains two orthogonal pieces of state at each program point:

- A   -- the active thread set, a box on native axes (laneid, warpid, cta_id)
- scope_kind -- the current scope kind (kernel/cluster/cta/warpgroup/warp/thread)

Plus a derived (inter, intra) split for the current scope_kind. This module is the
compile-time analysis that threads these through an IR walker; filter narrows A
and scope_switch produces (inter, intra).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

WG_SIZE = 4

# Scope kinds, from widest to narrowest.
KERNEL = "kernel"
CLUSTER = "cluster"
CTA = "cta"
WARPGROUP = "warpgroup"
WARP = "warp"
THREAD = "thread"

SCOPE_KINDS = (KERNEL, CLUSTER, CTA, WARPGROUP, WARP, THREAD)

# Lane kinds for ScopeIdDef var binding. "flat" means the full native axis;
# "wg_outer" / "w_inner" are the two factor lanes of warpid.
LANE_FLAT = "flat"
LANE_WG_OUTER = "wg_outer"
LANE_W_INNER = "w_inner"


class ExecContextError(Exception):
    """Raised on structural violations of the ExecContext model
    (empty active set, factoring failure, filter on a non-narrowable lane,
    etc.)."""


@dataclass(frozen=True)
class Lane:
    """An integer range [offset, offset+extent) on one axis."""

    extent: int
    offset: int = 0

    def intersect(self, lo: int, hi: int) -> Lane:
        """Intersect with [lo, hi); raises if empty."""
        new_off = max(self.offset, lo)
        new_end = min(self.offset + self.extent, hi)
        if new_end <= new_off:
            raise ExecContextError(
                f"filter produces empty lane: current=[{self.offset},"
                f" {self.offset + self.extent}) ∩ [{lo}, {hi})"
            )
        return Lane(extent=new_end - new_off, offset=new_off)


@dataclass(frozen=True)
class ActiveSet:
    """A = box on (laneid, warpid, cta_id). Each dim is kept even when ext=1."""

    laneid: Lane
    warpid: Lane
    cta_id: Lane

    @property
    def size(self) -> int:
        return self.laneid.extent * self.warpid.extent * self.cta_id.extent

    def replace_axis(self, axis: str, lane: Lane) -> ActiveSet:
        if axis == "laneid":
            return replace(self, laneid=lane)
        if axis == "warpid":
            return replace(self, warpid=lane)
        if axis == "cta_id":
            return replace(self, cta_id=lane)
        raise ValueError(f"unknown axis: {axis!r}")

    def axis(self, name: str) -> Lane:
        return getattr(self, name)


@dataclass(frozen=True)
class LaneBinding:
    """Resolution of a user-declared ScopeIdDef Var to a lane of A.

    A Var declared via ``T.warp_id`` binds to (axis='warpid', kind='flat').
    ``T.warpgroup_id`` -> ('warpid', 'wg_outer'). ``T.warp_id_in_wg`` ->
    ('warpid', 'w_inner'). ``T.lane_id`` -> ('laneid', 'flat'). ``T.cta_id``
    and ``T.cta_id_in_cluster`` -> ('cta_id', 'flat'). ``T.cluster_id`` is
    not a filter target (cluster is the unique outermost bucket).

    ``declared_extent`` is the product of ``extents`` supplied at the API
    (e.g. ``T.cta_id([4, 4])`` -> declared_extent=16). For factored bindings
    this is the factor size visible to the user (e.g. ``T.warpgroup_id([4])``
    -> declared_extent=4, meaning 4 warpgroups per CTA).
    """

    axis: str
    kind: str
    declared_extent: int


# ---------------------------------------------------------------------------
# Initial construction
# ---------------------------------------------------------------------------


def initial_A(*, lane_ext: int = 32, warp_ext: int, cta_ext: int = 1) -> ActiveSet:
    """Build A at T.kernel() entry: all threads active, offsets all zero."""
    return ActiveSet(
        laneid=Lane(extent=lane_ext, offset=0),
        warpid=Lane(extent=warp_ext, offset=0),
        cta_id=Lane(extent=cta_ext, offset=0),
    )


# ---------------------------------------------------------------------------
# filter: narrow one lane of A by range intersection
# ---------------------------------------------------------------------------


def filter_narrow(A: ActiveSet, binding: LaneBinding, lo: int, hi: int) -> ActiveSet:
    """Intersect A's binding lane with [lo, hi). Raises on empty result or
    factor-lane narrowing that would break A's box shape."""
    if lo >= hi:
        raise ExecContextError(f"filter range [{lo}, {hi}) is empty or inverted")

    if binding.kind == LANE_FLAT:
        new_lane = A.axis(binding.axis).intersect(lo, hi)
        return A.replace_axis(binding.axis, new_lane)

    # Factor lanes are only defined on warpid.
    if binding.axis != "warpid":
        raise ExecContextError(
            f"kind={binding.kind!r} only valid for axis='warpid'; got {binding.axis!r}"
        )

    wp = A.warpid
    if binding.kind == LANE_WG_OUTER:
        # Current warpid must be WG_SIZE-aligned to talk about the outer lane.
        if wp.offset % WG_SIZE != 0 or wp.extent % WG_SIZE != 0:
            raise ExecContextError(
                f"filter on wg_outer requires warpid lane aligned to WG_SIZE={WG_SIZE};"
                f" got extent={wp.extent}, offset={wp.offset}"
            )
        cur_outer = Lane(extent=wp.extent // WG_SIZE, offset=wp.offset // WG_SIZE)
        new_outer = cur_outer.intersect(lo, hi)
        return A.replace_axis(
            "warpid",
            Lane(extent=new_outer.extent * WG_SIZE, offset=new_outer.offset * WG_SIZE),
        )

    if binding.kind == LANE_W_INNER:
        # Narrowing the inner lane only preserves a box when warpid currently
        # sits inside a single warpgroup (extent <= WG_SIZE - off%WG_SIZE).
        cur_inner_off = wp.offset % WG_SIZE
        if wp.extent > WG_SIZE - cur_inner_off:
            raise ExecContextError(
                "filter on w_inner would break A's box: warpid spans multiple"
                f" warpgroups (extent={wp.extent}, offset={wp.offset})"
            )
        cur_inner = Lane(extent=wp.extent, offset=cur_inner_off)
        new_inner = cur_inner.intersect(lo, hi)
        # Lift back to warpid: outer part is fixed by wp.offset // WG_SIZE.
        outer_base = (wp.offset // WG_SIZE) * WG_SIZE
        return A.replace_axis(
            "warpid",
            Lane(extent=new_inner.extent, offset=outer_base + new_inner.offset),
        )

    raise ValueError(f"unknown lane kind: {binding.kind!r}")


# ---------------------------------------------------------------------------
# scope_switch: factor A into (inter, intra) for a target scope_kind
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Split:
    """A scope_switch split of A. Each field is a dict {axis_name: Lane}.

    Keys are a subset of {"laneid", "warpid", "cta_id", "wid_in_wg", "wgid"}.
    An empty dict denotes the empty layout (e.g. intra under scope_kind=thread).
    """

    inter: dict[str, Lane]
    intra: dict[str, Lane]


def _factor_warpid(warp: Lane) -> tuple[Lane, Lane] | None:
    """Factor warpid into (wid_in_wg, wgid). Returns None on case 3 (fail)."""
    off = warp.offset
    ext = warp.extent
    wid_off = off % WG_SIZE
    wgid_off = off // WG_SIZE

    # case 1: aligned
    if wid_off == 0 and ext % WG_SIZE == 0:
        return (
            Lane(extent=WG_SIZE, offset=0),
            Lane(extent=ext // WG_SIZE, offset=wgid_off),
        )
    # case 2: fits in one warpgroup
    if ext <= WG_SIZE - wid_off:
        return (
            Lane(extent=ext, offset=wid_off),
            Lane(extent=1, offset=wgid_off),
        )
    # case 3: crosses boundary, unaligned
    return None


def scope_switch(A: ActiveSet, scope_kind: str) -> Split:
    """Split A into (inter, intra) for the target scope kind. Raises on failure."""
    if scope_kind == THREAD:
        return Split(
            inter={
                "laneid": A.laneid,
                "warpid": A.warpid,
                "cta_id": A.cta_id,
            },
            intra={},
        )
    if scope_kind == WARP:
        return Split(
            inter={"warpid": A.warpid, "cta_id": A.cta_id},
            intra={"laneid": A.laneid},
        )
    if scope_kind == CTA:
        return Split(
            inter={"cta_id": A.cta_id},
            intra={"laneid": A.laneid, "warpid": A.warpid},
        )
    if scope_kind == CLUSTER:
        return Split(
            inter={},
            intra={
                "laneid": A.laneid,
                "warpid": A.warpid,
                "cta_id": A.cta_id,
            },
        )
    if scope_kind == WARPGROUP:
        factored = _factor_warpid(A.warpid)
        if factored is None:
            raise ExecContextError(
                "scope_switch(warpgroup) failed: warpid lane"
                f" (extent={A.warpid.extent}, offset={A.warpid.offset})"
                " crosses warpgroup boundary and is not aligned"
            )
        wid_in_wg, wgid = factored
        return Split(
            inter={"wgid": wgid, "cta_id": A.cta_id},
            intra={"laneid": A.laneid, "wid_in_wg": wid_in_wg},
        )
    if scope_kind == KERNEL:
        # scope_kind=kernel is the initial state -- inter is A, intra is empty.
        return Split(
            inter={
                "laneid": A.laneid,
                "warpid": A.warpid,
                "cta_id": A.cta_id,
            },
            intra={},
        )
    raise ValueError(f"unknown scope kind: {scope_kind!r}")


# ---------------------------------------------------------------------------
# ExecContext -- bundle {A, scope_kind, current_inter, current_intra}
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExecContext:
    """Per-program-point compiler state: active set + scope kind + split."""

    A: ActiveSet
    scope_kind: str
    inter: dict[str, Lane]
    intra: dict[str, Lane]

    @staticmethod
    def at_kernel_entry(*, lane_ext: int = 32, warp_ext: int, cta_ext: int = 1) -> ExecContext:
        A = initial_A(lane_ext=lane_ext, warp_ext=warp_ext, cta_ext=cta_ext)
        split = scope_switch(A, KERNEL)
        return ExecContext(A=A, scope_kind=KERNEL, inter=split.inter, intra=split.intra)

    def with_filter(self, binding: LaneBinding, lo: int, hi: int) -> ExecContext:
        new_A = filter_narrow(self.A, binding, lo, hi)
        split = scope_switch(new_A, self.scope_kind)
        return ExecContext(
            A=new_A, scope_kind=self.scope_kind, inter=split.inter, intra=split.intra
        )

    def with_scope_switch(self, scope_kind: str) -> ExecContext:
        split = scope_switch(self.A, scope_kind)
        return ExecContext(A=self.A, scope_kind=scope_kind, inter=split.inter, intra=split.intra)
