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

from typing import Literal

import tvm
from tvm.script import tirx as Tx
from tvm.tir.layout import S, TCol, TileLayout, TLane
from tvm.tir.layout import tid_in_wg as axis_tid_in_wg
from tvm.tirx.bench.utils import CudaProfiler
from tvm.tirx.megakernel.utils.base import Barriers, SmemManager, Tile
from tvm.tirx.megakernel.utils.config import F16_BYTES, F32_BYTES, KernelConfig, ProfileEventType
from tvm.tirx.megakernel.utils.utils import ceildiv, mbarrier_try_wait
from tvm.tirx.op_schedule.cuda.tma_utils import SwizzleMode, tma_shared_layout


class BarTMA2MMA(Barriers):
    @Tx.inline
    def arrive(self, idx, expected_bytes):
        Tx.ptx.mbarrier.arrive.expect_tx(self.mbar.ptr_to([idx]), expected_bytes)

    @Tx.inline
    def arrive_only(self, idx):
        Tx.ptx.mbarrier.arrive(self.mbar.ptr_to([idx]))


class BarMMA2LD(Barriers):
    @Tx.inline
    def arrive(self, idx):
        Tx.ptx.tcgen05.commit(self.mbar.ptr_to([idx]), cta_group=KernelConfig.CTA_GROUP)


class BarMMA2TMA(Barriers):
    @Tx.inline
    def arrive(self, idx):
        Tx.ptx.tcgen05.commit(self.mbar.ptr_to([idx]), cta_group=KernelConfig.CTA_GROUP)


class BarLD2MMA(Barriers):
    @Tx.inline
    def arrive(self, idx):
        Tx.ptx.mbarrier.arrive(self.mbar.ptr_to([idx]), cta_id=0, pred=True)


########################################################################
# F16-lowM GEMM Megakernel-Tile
########################################################################


class GemmTile(Tile):
    SMEM_PIPE_DEPTH = 6
    TMEM_PIPE_DEPTH = 2
    MAX_BLK_M, BLK_N, BLK_K = 128, 128, 64
    MMA_N, MMA_K = 128, 16
    EPI_TILE = 32
    TMEM_LD_SIZE = 8
    N_COLS = 512
    SWIZZLE = 3
    SMEM_SIZE = (
        SMEM_PIPE_DEPTH * MAX_BLK_M * BLK_K * F16_BYTES
        + SMEM_PIPE_DEPTH * BLK_N * BLK_K * F16_BYTES
        + TMEM_PIPE_DEPTH * EPI_TILE * MMA_N * F32_BYTES
        + 1024
    )

    assert SMEM_SIZE <= 232448
    assert TMEM_PIPE_DEPTH * MMA_N <= 512

    # idx of current gemm tile (no matter which shape it is)
    tile_idx = None

    def __init__(
        self,
        N,
        K,
        a_type,
        b_type,
        split_k_factor,
        BLK_M,
        MMA_M,
        out_type=None,
        use_tma_reduce=False,
        low_batch=True,
        prefetch_on=False,
        profiler_on=False,
    ):
        super().__init__()
        self.BLK_M = BLK_M
        self.MMA_M = MMA_M
        self.N = N
        self.K = K
        self.a_type = a_type
        self.b_type = b_type
        assert a_type == "float16", "only float16 is supported for now"
        assert b_type == "float16", "only float16 is supported for now"
        assert not (use_tma_reduce and split_k_factor == 1), (
            "use_tma_reduce when split_k_factor == 1 is not supported"
        )
        if out_type is None:
            self.out_type = "float32" if split_k_factor > 1 or use_tma_reduce else "float16"
        else:
            self.out_type = out_type
        self.split_k_factor = split_k_factor
        self.use_tma_reduce = use_tma_reduce
        self.low_batch = low_batch
        self.prefetch_on = prefetch_on
        self.profiler_on = profiler_on
        self.TILE_K = ceildiv(ceildiv(self.K, self.split_k_factor), self.BLK_K) * self.BLK_K
        self.PIPE_CIRCLE_NUM = (self.TILE_K // self.BLK_K) // self.SMEM_PIPE_DEPTH
        self.PIPE_REMAIN_NUM = (self.TILE_K // self.BLK_K) % self.SMEM_PIPE_DEPTH
        self.M_pad_size = BLK_M
        self.A_layout = tma_shared_layout(
            a_type,
            SwizzleMode.SWIZZLE_128B_ATOM,
            (self.SMEM_PIPE_DEPTH, self.MAX_BLK_M, self.BLK_K),
        )
        self.B_layout = tma_shared_layout(
            b_type, SwizzleMode.SWIZZLE_128B_ATOM, (self.SMEM_PIPE_DEPTH, self.BLK_N, self.BLK_K)
        )
        self.D_layout = Tx.TileLayout(
            Tx.S[
                (self.TMEM_PIPE_DEPTH, self.EPI_TILE, self.MMA_N) : (
                    self.EPI_TILE * self.MMA_N,
                    self.MMA_N,
                    1,
                )
            ]
        )

    def _alloc_buffer(self, smem_manager: SmemManager):
        self.smem_manager = smem_manager
        # alloc shared memory
        self.A_smem = smem_manager.alloc(
            (self.SMEM_PIPE_DEPTH, self.MAX_BLK_M, self.BLK_K),
            self.a_type,
            layout=self.A_layout,
            align=1024,
            split=self.SMEM_PIPE_DEPTH,
            name="A_smem",
            method="exclusive",
        )
        self.B_smem = smem_manager.alloc(
            (self.SMEM_PIPE_DEPTH, self.BLK_N, self.BLK_K),
            self.b_type,
            layout=self.B_layout,
            align=1024,
            split=self.SMEM_PIPE_DEPTH,
            name="B_smem",
            method="exclusive",
        )
        self.output_smem = smem_manager.alloc(
            (self.TMEM_PIPE_DEPTH, self.EPI_TILE, self.MMA_N),
            self.out_type,
            layout=self.D_layout,
            align=1024,
            name="output_smem",
            method="exclusive",
        )

    def _alloc_local(self, m_idx):
        # alloc local memory
        self.reg = Tx.alloc_buffer((self.TMEM_LD_SIZE,), "float32", scope="local", name="reg")
        if self.out_type == "float16":
            self.reg_fp16 = Tx.alloc_buffer(
                (self.TMEM_LD_SIZE,), self.out_type, scope="local", name="reg_fp16"
            )
        self.tmem_idx = Tx.local_scalar("int32", name="tmem_idx")
        self.tmem_phase = Tx.local_scalar("int32", name="tmem_phase")
        self.stage = Tx.local_scalar("int32", name="stage")
        self.wait_complete = Tx.local_scalar("bool", name="wait_complete")

    @classmethod
    def _alloc_buffer_class_member(cls, smem_manager: SmemManager):
        # alloc shared memory
        # use GemmTile instead of cls to avoid re-allocating memory for different subclasses
        # TODO: this cannot be generalized if there are multiple subclasses of GemmTile
        #       we need to delete these members in class_finalize, and only alloc when there are no members  # noqa: E501
        GemmTile.tmem_addr = smem_manager.alloc(
            [1], "uint32", name="tmem_addr", method="persistent"
        )
        GemmTile.tma2mma_bar = BarTMA2MMA(smem_manager, cls.SMEM_PIPE_DEPTH, True)
        GemmTile.mma2tma_bar = BarMMA2TMA(smem_manager, cls.SMEM_PIPE_DEPTH, False)
        GemmTile.mma2ld_bar = BarMMA2LD(smem_manager, cls.TMEM_PIPE_DEPTH, True)
        GemmTile.ld2mma_bar = BarLD2MMA(smem_manager, cls.TMEM_PIPE_DEPTH, False)
        # alloc local memory
        GemmTile.tile_idx = Tx.local_scalar("int32", "tile_idx")
        GemmTile.phase = Tx.alloc_buffer((1,), "int32", scope="local", name="phase")
        GemmTile.tmem = Tx.decl_buffer(
            (128, 512),
            "float32",
            scope="tmem",
            allocated_addr=0,
            layout=TileLayout(S[(128, 512) : (1 @ TLane, 1 @ TCol)]),
            name="tmem",
        )

    @classmethod
    @Tx.inline
    def class_init(cls, smem_manager: SmemManager):
        cls._alloc_buffer_class_member(smem_manager)
        cls.tile_idx = 0
        # alloc TMEM
        with Tx.warp()[0:1]:
            Tx.ptx.tcgen05.alloc(Tx.address_of(cls.tmem_addr[0]), n_cols=cls.N_COLS, cta_group=1)
            Tx.cuda.warp_sync()
        # init mbarrier and phase
        cls.tma2mma_bar.init(1)
        cls.mma2ld_bar.init(1)
        cls.mma2tma_bar.init(1)
        cls.ld2mma_bar.init(KernelConfig.CTA_GROUP * 128)
        cls.phase[0] = 0

        # sync
        Tx.ptx.fence.proxy_async("shared::cta")
        Tx.ptx.fence.mbarrier_init()
        Tx.tvm_storage_sync("shared")

    @classmethod
    @Tx.inline
    def class_finalize(cls):
        Tx.tvm_storage_sync("shared")
        # dealloc TMEM
        with Tx.warp()[0:1]:
            Tx.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
            Tx.ptx.tcgen05.dealloc(cls.tmem_addr[0], n_cols=cls.N_COLS, cta_group=1)
        Tx.tvm_storage_sync("shared")

    @Tx.inline
    def init(self, smem_manager: SmemManager):
        self._alloc_buffer(smem_manager)

    @Tx.inline
    def host_init(self):
        # Notes: cuTensorMap initialization will be insert when lowering Tx.gemm_async
        pass

    @Tx.inline
    def _tma(self, ks, buf, buf_name: Literal["A", "B"], mn_st, k_st, tma_config, predicate=True):
        if predicate:
            if buf_name == "A":
                Tx.copy_async(
                    self.A_smem[ks, 0 : self.BLK_M, :],
                    buf[mn_st : mn_st + self.BLK_M, k_st : k_st + self.BLK_K],
                    **tma_config,
                )
            elif buf_name == "B":
                Tx.copy_async(
                    self.B_smem[ks, 0 : self.BLK_N, :],
                    buf[mn_st : mn_st + self.BLK_N, k_st : k_st + self.BLK_K],
                    **tma_config,
                )
            else:
                Tx.cuda.trap_when_assert_failed(False)

    @Tx.inline
    def _consumer_wg(self, m_idx, n_idx, k_idx, A, B, output, profiler: CudaProfiler):
        with Tx.cta():
            tid_in_wg = Tx.thread_id([128], parent="warpgroup")
            warp_id = Tx.warp_id([KernelConfig.WARP_NUMBER], parent="warpgroup")
            lane_id = Tx.thread_id([32], parent="warp")
            Tx.cuda.trap_when_assert_failed(self.tmem_addr[0] == 0)
            if warp_id == 0:
                self.smem_manager.wait_specific(lane_id, self.output_smem, 0)
            Tx.cuda.warpgroup_sync(10)
            self.phase[0] = 0
            self.tmem_idx = self.tile_idx % self.TMEM_PIPE_DEPTH
            self.tmem_phase = (self.tile_idx // self.TMEM_PIPE_DEPTH) & 1

            # flush previous tma
            # wait for the completion of all the mma of the same tile
            self.mma2ld_bar.wait(self.tmem_idx, self.tmem_phase)
            Tx.ptx.tcgen05.fence.after_thread_sync()

            for ko in Tx.unroll(self.MMA_M // self.EPI_TILE):
                self.stage = (
                    self.tile_idx * self.MMA_M // self.EPI_TILE + ko
                ) % self.TMEM_PIPE_DEPTH
                # wait the smem to be free
                if ko >= self.TMEM_PIPE_DEPTH:
                    if lane_id == 0 and warp_id == 0:
                        Tx.ptx.cp_async.bulk.wait_group(self.TMEM_PIPE_DEPTH - 1)
                    Tx.cuda.warpgroup_sync(10)

                # tmem -> rf (ld) -> smem
                for ki in Tx.unroll(self.EPI_TILE // self.TMEM_LD_SIZE):
                    with Tx.warpgroup():
                        reg_wg = self.reg.view(
                            128,
                            self.TMEM_LD_SIZE,
                            layout=TileLayout(
                                S[(128, self.TMEM_LD_SIZE) : (1 @ axis_tid_in_wg, 1)]
                            ),
                        )
                        col_st = Tx.meta_var(
                            self.tmem_idx * self.M_pad_size
                            + ko * self.EPI_TILE
                            + ki * self.TMEM_LD_SIZE
                        )
                        Tx.copy(reg_wg[:, :], self.tmem[:, col_st : col_st + self.TMEM_LD_SIZE])
                    with Tx.thread():
                        st = Tx.meta_var(ki * self.TMEM_LD_SIZE)
                        if self.out_type == "float16":
                            Tx.cast(self.reg_fp16[:], self.reg[:])
                            Tx.copy(
                                self.output_smem[
                                    self.stage, st : st + self.TMEM_LD_SIZE, tid_in_wg
                                ],
                                self.reg_fp16[:],
                            )
                        else:
                            Tx.copy(
                                self.output_smem[
                                    self.stage, st : st + self.TMEM_LD_SIZE, tid_in_wg
                                ],
                                self.reg[:],
                            )
                # the tmem can be overwritten
                if ko == self.MMA_M // self.EPI_TILE - 1:
                    Tx.ptx.tcgen05.fence.before_thread_sync()
                    self.ld2mma_bar.arrive(self.tmem_idx)

                Tx.ptx.fence.proxy_async("shared::cta")
                Tx.cuda.warpgroup_sync(10)
                # smem -> gmem
                with Tx.thread(parent="warpgroup")[tid_in_wg == 0]:
                    m_st = Tx.meta_var(m_idx * self.M_pad_size + ko * self.EPI_TILE)
                    n_st = Tx.meta_var(n_idx * self.BLK_N)
                    tma_config = Tx.meta_var(
                        {"dispatch": "tma", "cta_group": KernelConfig.CTA_GROUP}
                        | (
                            {"cache_hint": "evict_last" if self.low_batch else ""}
                            if self.split_k_factor > 1
                            else {}
                        )
                        | ({"use_tma_reduce": "add"} if self.use_tma_reduce else {})
                    )
                    if self.split_k_factor > 1 and not self.use_tma_reduce:
                        Tx.copy_async(
                            output[k_idx, m_st : m_st + self.EPI_TILE, n_st : n_st + self.BLK_N],
                            self.output_smem[self.stage, :, :],
                            **tma_config,
                        )
                    else:
                        Tx.copy_async(
                            output[m_st : m_st + self.EPI_TILE, n_st : n_st + self.BLK_N],
                            self.output_smem[self.stage, :, :],
                            **tma_config,
                        )
                    Tx.ptx.cp_async.bulk.commit_group()
            if tid_in_wg:
                Tx.ptx.cp_async.bulk.wait_group(0)
            Tx.cuda.warpgroup_sync(10)
            self.tile_idx += 1
            if warp_id == 0:
                self.smem_manager.arrive_specific(lane_id, self.output_smem, 0)

    @Tx.inline
    def _run(self, m_idx, n_idx, k_idx, A, B, output, profiler: CudaProfiler):
        with Tx.cta():
            Tx.warpgroup_id([KernelConfig.WG_NUMBER], parent="cta")
            warp_id = Tx.warp_id([KernelConfig.WARP_NUMBER], parent="warpgroup")
            lane_id = Tx.thread_id([32], parent="warp")
            Tx.thread_id([KernelConfig.NUM_THREADS], parent="cta")
            with Tx.cta():
                Tx.attr({"tirx.scope_partition": True})
                with Tx.warpgroup()[1:2]:
                    if warp_id == 3:
                        # GMEM -> SMEM  (tma)
                        @Tx.inline
                        def tma_stage(ks, k_st, first_stage):
                            self.mma2tma_bar.wait(ks, self.phase[0])
                            B_tma_config = Tx.meta_var(
                                {
                                    "dispatch": "tma",
                                    "cta_group": KernelConfig.CTA_GROUP,
                                    "mbar": self.tma2mma_bar.mbar.ptr_to([ks]),
                                    "cache_hint": "evict_first" if self.low_batch else "",
                                }
                            )
                            A_tma_config = Tx.meta_var(
                                {
                                    "dispatch": "tma",
                                    "cta_group": KernelConfig.CTA_GROUP,
                                    "mbar": self.tma2mma_bar.mbar.ptr_to([ks]),
                                    "cache_hint": "evict_last" if self.low_batch else "",
                                }
                            )
                            if self.profiler_on:
                                profiler.start(ProfileEventType.TMA, lane_id == 0)
                            if first_stage:
                                self.smem_manager.wait_specific_one_thread(self.A_smem, ks)
                            self._tma(ks, A, "A", m_idx * self.M_pad_size, k_st, A_tma_config)
                            if not self.prefetch_on and first_stage:
                                self.smem_manager.wait_specific_one_thread(self.B_smem, ks)
                            self._tma(
                                ks,
                                B,
                                "B",
                                n_idx * self.BLK_N,
                                k_st,
                                B_tma_config,
                                predicate=tvm.tir.Not(self.prefetch_on and first_stage),
                            )
                            if self.profiler_on:
                                profiler.end(ProfileEventType.TMA, lane_id == 0)
                            self.tma2mma_bar.arrive(
                                ks,
                                KernelConfig.CTA_GROUP
                                * self.BLK_K
                                * (self.BLK_M + self.BLK_N)
                                * F16_BYTES,
                            )

                        with Tx.thread(parent="warp")[Tx.ptx.elect_sync()]:
                            k_offset = k_idx * self.TILE_K
                            for ko in Tx.serial(self.PIPE_CIRCLE_NUM):
                                for ks in Tx.unroll(self.SMEM_PIPE_DEPTH):
                                    tma_stage(
                                        ks,
                                        (ko * self.SMEM_PIPE_DEPTH + ks) * self.BLK_K + k_offset,
                                        ko == 0,
                                    )
                                self.phase[0] = self.phase[0] ^ 1
                            if self.PIPE_REMAIN_NUM > 0:
                                for ks in Tx.unroll(self.PIPE_REMAIN_NUM):
                                    tma_stage(
                                        ks,
                                        (self.PIPE_CIRCLE_NUM * self.SMEM_PIPE_DEPTH + ks)
                                        * self.BLK_K
                                        + k_offset,
                                        self.PIPE_CIRCLE_NUM == 0,
                                    )
                                # for unaligned cases
                                for ks in Tx.unroll(self.PIPE_REMAIN_NUM, self.SMEM_PIPE_DEPTH):
                                    self.mma2tma_bar.wait(ks, self.phase[0])
                                    self.tma2mma_bar.arrive_only(ks)
                                self.phase[0] = self.phase[0] ^ 1

                        Tx.ptx.bar.sync(13, 64)  # notify warp 6 to release smem chunks

                    elif warp_id == 0:
                        # MMA

                        descI: Tx.uint32
                        Tx.ptx.tcgen05.encode_instr_descriptor(
                            Tx.address_of(descI),  # noqa: F821
                            "float32",
                            self.a_type,
                            self.b_type,
                            self.MMA_N,
                            self.MMA_M,
                            self.MMA_K,
                            False,
                            False,
                            KernelConfig.CTA_GROUP,
                        )

                        @Tx.inline
                        def mbar_try_wait(idx, phase):
                            self.wait_complete = mbarrier_try_wait(
                                self.tma2mma_bar.mbar.ptr_to([idx]),
                                self.tma2mma_bar.init_phase ^ phase,
                            )

                        @Tx.inline
                        def mma_stage(ks, acc):
                            if self.profiler_on:
                                profiler.start(ProfileEventType.MMA, lane_id == 0)
                            Tx.gemm_async(
                                self.tmem[
                                    :,
                                    self.tmem_idx * self.M_pad_size : self.tmem_idx
                                    * self.M_pad_size
                                    + self.BLK_N,
                                ],
                                self.B_smem[ks, :, :],
                                self.A_smem[ks, :, :],
                                accum=acc,
                                dispatch="tcgen05",
                                cta_group=KernelConfig.CTA_GROUP,
                                descI=descI,  # noqa: F821
                            )
                            if self.profiler_on:
                                profiler.end(ProfileEventType.MMA, lane_id == 0)
                            self.mma2tma_bar.arrive(ks)

                        with Tx.thread(parent="warp")[Tx.ptx.elect_sync()]:
                            self.tmem_idx = self.tile_idx % self.TMEM_PIPE_DEPTH
                            self.tmem_phase = (self.tile_idx // self.TMEM_PIPE_DEPTH) & 1

                            # wait for the tmem result to be consumed
                            self.ld2mma_bar.wait(self.tmem_idx, self.tmem_phase)
                            Tx.ptx.tcgen05.fence.after_thread_sync()
                            mbar_try_wait(0, self.phase[0])

                            for ko in Tx.serial(self.PIPE_CIRCLE_NUM):
                                for ks in Tx.unroll(self.SMEM_PIPE_DEPTH):
                                    # wait tma
                                    if not self.wait_complete:
                                        self.tma2mma_bar.wait(ks, self.phase[0])
                                        # Tx.ptx.tcgen05.fence.after_thread_sync()
                                    if (
                                        self.PIPE_REMAIN_NUM > 0
                                        or ko != self.PIPE_REMAIN_NUM - 1
                                        or ks != self.SMEM_PIPE_DEPTH - 1
                                    ):
                                        mbar_try_wait(
                                            (ks + 1) % self.SMEM_PIPE_DEPTH,
                                            self.phase[0]
                                            ^ (1 if ks == self.SMEM_PIPE_DEPTH - 1 else 0),
                                        )
                                    mma_stage(ks, not (ko == 0 and ks == 0))

                                self.phase[0] = self.phase[0] ^ 1

                            if self.PIPE_REMAIN_NUM > 0:
                                for ks in Tx.unroll(self.PIPE_REMAIN_NUM):
                                    if not self.wait_complete:
                                        self.tma2mma_bar.wait(ks, self.phase[0])
                                        # Tx.ptx.tcgen05.fence.after_thread_sync()
                                    if ks != self.PIPE_REMAIN_NUM - 1:
                                        mbar_try_wait(
                                            (ks + 1) % self.SMEM_PIPE_DEPTH, self.phase[0]
                                        )
                                    mma_stage(ks, not (self.PIPE_CIRCLE_NUM == 0 and ks == 0))

                                # ensure that all mma is issued
                                self.mma2ld_bar.arrive(self.tmem_idx)

                                # for unaligned cases
                                for ks in Tx.unroll(self.PIPE_REMAIN_NUM, self.SMEM_PIPE_DEPTH):
                                    self.tma2mma_bar.wait(ks, self.phase[0])
                                    self.mma2tma_bar.arrive(ks)

                                self.phase[0] = self.phase[0] ^ 1
                            else:
                                # ensure that all mma is issued
                                self.mma2ld_bar.arrive(self.tmem_idx)
                        self.tile_idx += 1

                    elif warp_id == 1:
                        self.smem_manager.wait_unused(lane_id, self)
                        self.smem_manager.arrive_unused(lane_id, self)
                    elif warp_id == 2:
                        self.phase[0] = self.phase[0] ^ (self.PIPE_CIRCLE_NUM & 1)
                        if self.PIPE_REMAIN_NUM > 0:
                            self.phase[0] = self.phase[0] ^ 1
                        Tx.ptx.bar.sync(13, 64)  # wait warp 7 to finish
                        for ks in Tx.unroll(self.SMEM_PIPE_DEPTH):
                            self.mma2tma_bar.wait(ks, self.phase[0])
                            self.smem_manager.arrive_specific(lane_id, self.B_smem, ks)
                            self.smem_manager.arrive_specific(lane_id, self.A_smem, ks)

                with Tx.warpgroup()[0:1]:
                    self._consumer_wg(m_idx, n_idx, k_idx, A, B, output, profiler)

    # call by warp 7 (tmp load warp)
    @Tx.inline
    def prefetch(self, m_idx, n_idx, k_idx, A, B, output, profiler: CudaProfiler):
        self._alloc_local(m_idx)
        with Tx.cta():
            wg_id = Tx.warpgroup_id([KernelConfig.WG_NUMBER], parent="cta")
            warp_id = Tx.warp_id([KernelConfig.WARP_NUMBER], parent="warpgroup")
            lane_id = Tx.thread_id([32], parent="warp")
            if wg_id == 1 and warp_id == 3:
                k_offset = k_idx * self.TILE_K
                if self.PIPE_CIRCLE_NUM > 0:
                    for ks in Tx.unroll(self.SMEM_PIPE_DEPTH):
                        # GMEM -> SMEM  (tma)
                        self.stage = ks
                        self.smem_manager.wait_specific(lane_id, self.B_smem, ks)
                        if self.profiler_on:
                            profiler.start(ProfileEventType.TMA, lane_id == 0)
                        with Tx.thread(parent="warp")[Tx.ptx.elect_sync()]:
                            tma_config = Tx.meta_var(
                                {
                                    "dispatch": "tma",
                                    "cta_group": KernelConfig.CTA_GROUP,
                                    "mbar": self.tma2mma_bar.mbar.ptr_to([ks]),
                                    "cache_hint": "evict_first" if self.low_batch else "",
                                }
                            )
                            self._tma(
                                ks,
                                B,
                                "B",
                                n_idx * self.BLK_N,
                                self.stage * self.BLK_K + k_offset,
                                tma_config,
                            )
                        if self.profiler_on:
                            profiler.end(ProfileEventType.TMA, lane_id == 0)

    @Tx.inline
    def run(self, m_idx, n_idx, k_idx, A, B, output, profiler: CudaProfiler = None):
        self._alloc_local(m_idx)
        self._run(m_idx, n_idx, k_idx, A, B, output, profiler)
        self.smem_manager.advance()
