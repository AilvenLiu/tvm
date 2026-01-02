import argparse
import math
import ml_dtypes
import numpy as np
from enum import Enum
import pytest

import tvm
from tvm.script import tir as T, tirx as Tx, ir as I, relax as R
import tvm.testing
from tvm import relax as rx

import flashinfer
import torch

from tvm.tirx.megakernel.relax_compatible.gemm import FuseGemmTile, FuseGateUpSiluTile
from tvm.tirx.megakernel.relax_compatible.gemm_splitk_reduce import FuseSplitKReduceTile
from tvm.tirx.megakernel.relax_compatible.reduce_rms_rope_append import FuseSplitKReduceRMSnormRopeQTile, FuseSplitKReduceRMSnormRopeAppendKTile, FuseSplitKReduceAppendVTile
from tvm.tirx.megakernel.relax_compatible.batch_attn import FuseBatchAttnTile
from tvm.tirx.megakernel.relax_compatible.batch_merge import FuseBatchMergeTile
from tvm.tirx.megakernel.relax_compatible.add_rmsnorm import FuseAddRMSNormTile, FuseRMSNormTile

from tvm.tirx.megakernel.common import KernelConfig, ceildiv, JobType, event_type_names
from tvm.tirx.bench.utils import ProtonContext, bench, export_to_perfetto_trace
from tvm.tirx.megakernel.support import get_inverse_plan_info
import tvm.tirx.megakernel.static_scheduler as static_scheduler
import tvm.tirx.megakernel.dynamic_scheduler as dynamic_scheduler

class MegaKernel:

    # model configs
    VOCAB_SIZE = 151936
    MAX_POSITION_EMBEDDINGS = 40960
    HIDDEN_SIZE = 5120
    INTERMEDIATE_SIZE = 25600
    NUM_HIDDEN_LAYERS = 64
    NUM_ATTENTION_HEADS = 64
    NUM_KEY_VALUE_HEADS = 8
    HEAD_DIM = 128
    RMS_NORM_EPS = 1e-6
    ROPE_THETA = 1000000
    MAX_PAGE_NUM = 8192
    PAGE_SIZE = 16

    EVT_WORKSPACE_SIZE = 1024 * 1024 * 128

    SPLIT_QKV_PROJECT = 3
    SPLIT_O_PROJECT = 3
    DOWN_PROJ_SPLIT_K_FACTOR = 10
    GATE_UP_PROJ_SPLIT_K_FACTOR = 1
    NUM_TASK_ARGS = 10
    MAX_TOTAL_NUM_WORKERS = 1025
    MAX_NUM_KV_SPLITS = 4 * KernelConfig.SM_NUMBER * 2 * 16

    def __init__(self):
        self.world_size = 1


    def _qwen3_layer_inner(self, bb: rx.BlockBuilder, max_batch_size, blk_m, profile_on):

        def f_init_unmatched_dim(dim_len, in_par_size, out_par_size):
            def f_init(i):
                start_out_par = i * out_par_size
                end_out_par = T.min(dim_len, (i + 1) * out_par_size)
                return (end_out_par - 1) // in_par_size - start_out_par // in_par_size + 1
            return f_init

        batch_size = T.var("int64", name="batch_size")
        x = rx.Var("x", rx.TensorStructInfo([batch_size, self.HIDDEN_SIZE], "float16"))
        residual = rx.Var("residual", rx.TensorStructInfo([batch_size, self.HIDDEN_SIZE], "float32"))
        packed_info = rx.Var(
            "packed_info",
            rx.TupleStructInfo(
                [rx.TensorStructInfo(None, dtype="float16")] +
                [rx.TensorStructInfo(None, dtype="int32")] * 9 +
                [rx.TensorStructInfo([self.MAX_TOTAL_NUM_WORKERS], dtype="int32")] * 10 +
                [rx.TensorStructInfo(None, dtype="float32")] +
                [rx.TensorStructInfo([self.MAX_TOTAL_NUM_WORKERS], dtype="int32")] * 2 +
                [R.Shape()]
            ),
        )
        packed_weights = rx.Var(
            "packed_weights",
            rx.TupleStructInfo(
                [
                    rx.TensorStructInfo([(self.NUM_ATTENTION_HEADS + self.NUM_KEY_VALUE_HEADS * 2) * self.HEAD_DIM, self.HIDDEN_SIZE], "float16"),
                    rx.TensorStructInfo([self.HEAD_DIM], "float16"),
                    rx.TensorStructInfo([self.HEAD_DIM], "float16"),
                    rx.TensorStructInfo([self.HIDDEN_SIZE, self.NUM_ATTENTION_HEADS * self.HEAD_DIM], "float16"),
                    rx.TensorStructInfo([self.HIDDEN_SIZE], "float16"),
                    rx.TensorStructInfo([self.INTERMEDIATE_SIZE * 2, self.HIDDEN_SIZE], "float16"),
                    rx.TensorStructInfo([self.HIDDEN_SIZE, self.INTERMEDIATE_SIZE], "float16"),
                    rx.TensorStructInfo([self.HIDDEN_SIZE], "float16"),
                ]
            ),
        )
        workspace = rx.Var("workspace", rx.TensorStructInfo([self.EVT_WORKSPACE_SIZE], "int32"))

        with bb.function(f"megakernel_blkm{blk_m}", [x, residual, packed_info, packed_weights, workspace]):
            # with bb.dataflow():
                # unpack info
                kv_data_ = bb.emit(R.TupleGetItem(packed_info, 0))
                page_kv_indptr = bb.emit(R.TupleGetItem(packed_info, 1))
                page_kv_indices_ = bb.emit(R.TupleGetItem(packed_info, 2))
                page_kv_last_page_len = bb.emit(R.TupleGetItem(packed_info, 3))
                append_pos = bb.emit(R.TupleGetItem(packed_info, 4))
                rope_pos = bb.emit(R.TupleGetItem(packed_info, 5))
                len_kv_chunk_ = bb.emit(R.TupleGetItem(packed_info, 6))
                merge_indptr_ = bb.emit(R.TupleGetItem(packed_info, 7))
                merge_o_indices_ = bb.emit(R.TupleGetItem(packed_info, 8))
                num_qo_len_ = bb.emit(R.TupleGetItem(packed_info, 9))
                q_indptr_ = bb.emit(R.TupleGetItem(packed_info, 10))
                kv_indptr_ = bb.emit(R.TupleGetItem(packed_info, 11))
                partial_indptr_ = bb.emit(R.TupleGetItem(packed_info, 12))
                q_len_ = bb.emit(R.TupleGetItem(packed_info, 13))
                kv_len_ = bb.emit(R.TupleGetItem(packed_info, 14))
                q_start_ = bb.emit(R.TupleGetItem(packed_info, 15))
                kv_start_ = bb.emit(R.TupleGetItem(packed_info, 16))
                kv_end_ = bb.emit(R.TupleGetItem(packed_info, 17))
                kv_head_idx_ = bb.emit(R.TupleGetItem(packed_info, 18))
                work_indptr_ = bb.emit(R.TupleGetItem(packed_info, 19))
                cos_sin_cache = bb.emit(R.TupleGetItem(packed_info, 20))
                inverse_indptr = bb.emit(R.TupleGetItem(packed_info, 21))
                inverse_indices = bb.emit(R.TupleGetItem(packed_info, 22))
                attn_task_num = T.var("int64", name="attn_task_num")
                _ = bb.match_cast(R.TupleGetItem(packed_info, 23), R.Shape([attn_task_num]))
                max_page_num = T.var("int64", name="max_page_num")
                kv_data_ = bb.match_cast(kv_data_, rx.TensorStructInfo([max_page_num, 2, self.NUM_KEY_VALUE_HEADS, self.PAGE_SIZE, self.HEAD_DIM], "float16"))

                # unpack weights
                qkv_proj_weight = bb.emit(R.TupleGetItem(packed_weights, 0))
                q_rms_weight = bb.emit(R.TupleGetItem(packed_weights, 1))
                k_rms_weight = bb.emit(R.TupleGetItem(packed_weights, 2))
                o_proj_weight = bb.emit(R.TupleGetItem(packed_weights, 3))
                attn_add_rms_weight = bb.emit(R.TupleGetItem(packed_weights, 4))
                gate_up_weight = bb.emit(R.TupleGetItem(packed_weights, 5))
                down_weight = bb.emit(R.TupleGetItem(packed_weights, 6))
                mlp_add_rms_weight = bb.emit(R.TupleGetItem(packed_weights, 7))

                # get tile kernels
                qkv_proj_func, qkv_proj_num_tiles, _ = FuseGemmTile.get_func(
                    (self.NUM_ATTENTION_HEADS + self.NUM_KEY_VALUE_HEADS * 2) * self.HEAD_DIM,
                    self.HIDDEN_SIZE,
                    "float16",
                    "float32",
                    blk_m,
                    split_k_factor=self.SPLIT_QKV_PROJECT,
                    prefetch_on=False,
                )
                reduce_rms_rope_q_func, reduce_rms_rope_q_num_tiles, reduce_rms_rope_q_size = FuseSplitKReduceRMSnormRopeQTile.get_func(
                    batch_size,
                    self.RMS_NORM_EPS,
                    self.NUM_ATTENTION_HEADS,
                    self.NUM_KEY_VALUE_HEADS,
                    self.HEAD_DIM,
                    self.SPLIT_QKV_PROJECT,
                )
                reduce_rms_rope_q_size_0, _, _ = reduce_rms_rope_q_size
                reduce_rms_rope_append_k_func, reduce_rms_rope_append_k_num_tiles, _ = FuseSplitKReduceRMSnormRopeAppendKTile.get_func(
                    batch_size,
                    self.RMS_NORM_EPS,
                    self.NUM_ATTENTION_HEADS,
                    self.NUM_KEY_VALUE_HEADS,
                    self.HEAD_DIM,
                    self.SPLIT_QKV_PROJECT,
                    self.PAGE_SIZE,
                )
                reduce_append_v_func, reduce_append_v_num_tiles, _ = FuseSplitKReduceAppendVTile.get_func(
                    batch_size,
                    self.NUM_ATTENTION_HEADS,
                    self.NUM_KEY_VALUE_HEADS,
                    self.HEAD_DIM,
                    self.SPLIT_QKV_PROJECT,
                    self.PAGE_SIZE,
                )
                batch_attn_func, batch_attn_num_tiles, _ = FuseBatchAttnTile.get_func(
                    self.PAGE_SIZE,
                    self.NUM_ATTENTION_HEADS,
                    self.NUM_KEY_VALUE_HEADS,
                    self.HEAD_DIM,
                    attn_task_num,
                    prefetch_on=True,
                )
                batch_merge_func, batch_merge_num_tiles, _ = FuseBatchMergeTile.get_func(
                    self.HEAD_DIM,
                    attn_task_num,
                    self.NUM_KEY_VALUE_HEADS,
                    self.NUM_ATTENTION_HEADS,
                    batch_size,
                )
                o_proj_func, o_proj_num_tiles, o_proj_size = FuseGemmTile.get_func(
                    self.HIDDEN_SIZE,
                    self.NUM_ATTENTION_HEADS * self.HEAD_DIM,
                    "float16",
                    "float32",
                    blk_m,
                    split_k_factor=self.SPLIT_O_PROJECT,
                    prefetch_on=True,
                    A_dim=3,
                    use_tma_reduce=True,
                )
                _, o_proj_size_1, o_proj_size_2 = o_proj_size
                attn_add_rms_func, attn_add_rms_num_tiles, _ = FuseRMSNormTile.get_func(
                    batch_size,
                    self.RMS_NORM_EPS,
                    self.HIDDEN_SIZE,
                )
                gate_up_silu_func, gate_up_silu_num_tiles, gate_up_silu_size = FuseGateUpSiluTile.get_func(
                    self.INTERMEDIATE_SIZE * 2,
                    self.HIDDEN_SIZE,
                    "float16",
                    "float16",
                    blk_m,
                    prefetch_on=True,
                )
                _, gate_up_silu_size_1, _ = gate_up_silu_size
                down_proj_func, down_proj_num_tiles, down_proj_size = FuseGemmTile.get_func(
                    self.HIDDEN_SIZE,
                    self.INTERMEDIATE_SIZE,
                    "float16",
                    "float16",
                    blk_m,
                    split_k_factor=self.DOWN_PROJ_SPLIT_K_FACTOR,
                    prefetch_on=True,
                    use_tma_reduce=True,
                )
                _, down_proj_size_1, down_proj_size_2 = down_proj_size
                mlp_add_rms_func, mlp_add_rms_num_tiles, _ = FuseRMSNormTile.get_func(
                    batch_size,
                    self.RMS_NORM_EPS,
                    self.HIDDEN_SIZE,
                )

                # add tile kernels to bb
                qkv_proj_gv = bb.add_func(qkv_proj_func, f"qkv_proj_blk{blk_m}")
                reduce_rms_rope_q_gv = bb.add_func(reduce_rms_rope_q_func, "reduce_rms_rope_q")
                reduce_rms_rope_append_k_gv = bb.add_func(reduce_rms_rope_append_k_func, "reduce_rms_rope_append_k")
                reduce_append_v_gv = bb.add_func(reduce_append_v_func, "reduce_append_v")
                batch_attn_gv = bb.add_func(batch_attn_func, "batch_attn")
                batch_merge_gv = bb.add_func(batch_merge_func, "batch_merge")
                o_proj_gv = bb.add_func(o_proj_func, f"o_proj_blk{blk_m}")
                attn_add_rms_gv = bb.add_func(attn_add_rms_func, "attn_add_rms")
                gate_up_silu_gv = bb.add_func(gate_up_silu_func, f"gate_up_silu_blk{blk_m}")
                down_proj_gv = bb.add_func(down_proj_func, f"down_proj_blk{blk_m}")
                mlp_add_rms_gv = bb.add_func(mlp_add_rms_func, "mlp_add_rms")

                # alloc and init events
                etensor_qkv_partial = bb.emit(
                    R.alloc_event_tensor(
                        workspace,
                        [ceildiv((self.NUM_ATTENTION_HEADS + 2 * self.NUM_KEY_VALUE_HEADS) * self.HEAD_DIM, FuseGemmTile.BLK_N)],
                        f_init=self.SPLIT_QKV_PROJECT
                    )
                )
                attn_tile_num = T.int64(ceildiv(attn_task_num, KernelConfig.WG_NUMBER))
                remain = T.int64(attn_tile_num % KernelConfig.SM_NUMBER)
                num = T.int64(attn_tile_num // KernelConfig.SM_NUMBER)
                unit = T.int64(KernelConfig.WG_NUMBER * (2 + self.NUM_ATTENTION_HEADS // self.NUM_KEY_VALUE_HEADS))
                etensor_notify_attn = bb.emit(
                    R.alloc_event_tensor(
                        workspace,
                        [KernelConfig.SM_NUMBER],
                        f_init=lambda i: T.if_then_else(i < remain, (num + 1) * unit, num * unit),
                    )
                )
                etensor_attn_merge = bb.emit(
                    R.alloc_event_tensor(
                        workspace,
                        [max_batch_size * self.NUM_KEY_VALUE_HEADS],
                        f_init=T.min(KernelConfig.SM_NUMBER, attn_tile_num),
                    )
                )
                etensor_o_proj = bb.emit(
                    R.alloc_event_tensor(
                        workspace,
                        [self.SPLIT_O_PROJECT],
                        f_init=lambda i:
                            T.if_then_else(
                                attn_task_num > self.NUM_KEY_VALUE_HEADS * batch_size,
                                f_init_unmatched_dim(
                                    self.NUM_ATTENTION_HEADS * self.HEAD_DIM,
                                    (self.NUM_ATTENTION_HEADS // self.NUM_KEY_VALUE_HEADS) * self.HEAD_DIM,
                                    o_proj_size_2,
                                )(i) * batch_size,
                                T.min(KernelConfig.SM_NUMBER, attn_tile_num),
                            ),
                    )
                )
                etensor_attn_add_rms = bb.emit(
                    R.alloc_event_tensor(
                        workspace,
                        [max_batch_size],
                        f_init=self.SPLIT_O_PROJECT * (self.HIDDEN_SIZE // o_proj_size_1),
                    )
                )
                etensor_attn_mlp = bb.emit(
                    R.alloc_event_tensor(
                        workspace,
                        [1],
                        f_init=lambda i: batch_size,
                    )
                )
                etensor_down_proj = bb.emit(
                    R.alloc_event_tensor(
                        workspace,
                        [self.DOWN_PROJ_SPLIT_K_FACTOR],
                        f_init=lambda i: f_init_unmatched_dim(self.INTERMEDIATE_SIZE, gate_up_silu_size_1 // 2, down_proj_size_2)(i),
                    )
                )
                etensor_mlp_add_rms = bb.emit(
                    R.alloc_event_tensor(
                        workspace,
                        [max_batch_size],
                        f_init=self.DOWN_PROJ_SPLIT_K_FACTOR * (self.HIDDEN_SIZE // down_proj_size_1),
                    )
                )
                etensor_end = bb.emit(
                    R.alloc_event_tensor(
                        workspace,
                        [1],
                        f_init=batch_size,
                    )
                )

                # data flow
                # QKV_GEMM
                qkv_partial = bb.emit(
                    R.call_tir_device(
                        qkv_proj_gv,
                        [x, qkv_proj_weight],
                        rx.TensorStructInfo([self.SPLIT_QKV_PROJECT, batch_size, (self.NUM_ATTENTION_HEADS + self.NUM_KEY_VALUE_HEADS * 2) * self.HEAD_DIM], "float32"),
                        job_id=JobType.GEMM_QKV_PROJ.value,
                        tile_num=qkv_proj_num_tiles,
                        out_deps=rx.utils.Dependency(
                            event=etensor_qkv_partial,
                            dep=lambda i, j, k, notify_idx: (1, -1, j),
                        ),
                    )
                )

                # Q_REDUCE_RMS_ROPE
                def reduce_rms_rope_q_dep_notify(i, j, k, notify_idx):
                    beg_idx = j // (self.NUM_ATTENTION_HEADS // self.NUM_KEY_VALUE_HEADS) * batch_size + i * reduce_rms_rope_q_size_0
                    end_idx = j // (self.NUM_ATTENTION_HEADS // self.NUM_KEY_VALUE_HEADS) * batch_size + T.min((i + 1) * reduce_rms_rope_q_size_0, batch_size)
                    beg = inverse_indptr[beg_idx]
                    end = inverse_indptr[end_idx]
                    return end - beg, -1, inverse_indices[beg + notify_idx]

                reduce_rms_rope_q_num_tiles_0 = reduce_rms_rope_q_num_tiles[0]
                q = bb.emit(
                    R.call_tir_device(
                        reduce_rms_rope_q_gv,
                        [qkv_partial, q_rms_weight, rope_pos, cos_sin_cache],
                        rx.TensorStructInfo([batch_size, self.NUM_ATTENTION_HEADS + 2 * self.NUM_KEY_VALUE_HEADS, self.HEAD_DIM], "float16"),
                        job_id=JobType.Q_REDUCE_RMS_ROPE.value,
                        tile_num=reduce_rms_rope_q_num_tiles,
                        in_deps=rx.utils.Dependency(
                            event=etensor_qkv_partial,
                            dep=lambda i, j, k, wait_idx: (1, -1, j),
                        ),
                        inverse_in_deps=rx.utils.Dependency(
                            event=etensor_qkv_partial,
                            dep=lambda rank, x, inv_idx: (T.if_then_else(x < self.NUM_ATTENTION_HEADS, reduce_rms_rope_q_num_tiles_0, 0), inv_idx, x, 0),
                        ),
                        out_deps=rx.utils.Dependency(
                            event=etensor_notify_attn,
                            dep=reduce_rms_rope_q_dep_notify,
                        ),
                    )
                )

                # K_RMS_ROPE_APPEND
                def reduce_rms_rope_append_k_dep(i, j, k, notify_idx):
                    beg_idx = j * batch_size + i * reduce_rms_rope_q_size_0
                    end_idx = j * batch_size + T.min((i + 1) * reduce_rms_rope_q_size_0, batch_size)
                    beg = inverse_indptr[beg_idx]
                    end = inverse_indptr[end_idx]
                    return end - beg, -1, inverse_indices[beg + notify_idx]

                reduce_rms_rope_append_k_num_tiles_0 = reduce_rms_rope_append_k_num_tiles[0]
                kv_data_ = bb.emit(
                    R.call_tir_device(
                        reduce_rms_rope_append_k_gv,
                        [qkv_partial, k_rms_weight, rope_pos, cos_sin_cache, append_pos, kv_data_],
                        rx.TensorStructInfo([max_page_num, 2, self.NUM_KEY_VALUE_HEADS, self.PAGE_SIZE, self.HEAD_DIM], "float16"),
                        tile_num=reduce_rms_rope_append_k_num_tiles,
                        job_id=JobType.K_REDUCE_RMS_ROPE_APPEND.value,
                        in_deps=rx.utils.Dependency(
                            event=etensor_qkv_partial,
                            dep=lambda i, j, k, wait_idx: (1, -1, j + self.NUM_ATTENTION_HEADS),
                        ),
                        inverse_in_deps=rx.utils.Dependency(
                            event=etensor_qkv_partial,
                            dep=lambda rank, x, inv_idx: (
                                T.if_then_else(tvm.tir.all(x >= self.NUM_ATTENTION_HEADS, x < self.NUM_ATTENTION_HEADS + self.NUM_KEY_VALUE_HEADS), reduce_rms_rope_append_k_num_tiles_0, 0),
                                inv_idx, x - self.NUM_ATTENTION_HEADS, 0
                            ),
                        ),
                        out_deps=rx.utils.Dependency(
                            event=etensor_notify_attn,
                            dep=reduce_rms_rope_append_k_dep,
                        ),
                        inplace_indices=5,
                    )
                )

                # V_REDUCE_APPEND
                def reduce_append_v_dep_notify(i, j, k, notify_idx):
                    beg_idx = j * batch_size + i * reduce_rms_rope_q_size_0
                    end_idx = j * batch_size + T.min((i + 1) * reduce_rms_rope_q_size_0, batch_size)
                    beg = inverse_indptr[beg_idx]
                    end = inverse_indptr[end_idx]
                    return end - beg, -1, inverse_indices[beg + notify_idx]

                reduce_append_v_num_tiles_0 = reduce_append_v_num_tiles[0]
                kv_data_ = bb.emit(
                    R.call_tir_device(
                        reduce_append_v_gv,
                        [qkv_partial, kv_data_, append_pos],
                        rx.TensorStructInfo([max_page_num, 2, self.NUM_KEY_VALUE_HEADS, self.PAGE_SIZE, self.HEAD_DIM], "float16"),
                        job_id=JobType.V_REDUCE_APPEND.value,
                        tile_num=reduce_append_v_num_tiles,
                        in_deps=rx.utils.Dependency(
                            event=etensor_qkv_partial,
                            dep=lambda i, j, k, wait_idx: (1, -1, j + self.NUM_ATTENTION_HEADS + self.NUM_KEY_VALUE_HEADS),
                        ),
                        inverse_in_deps=rx.utils.Dependency(
                            event=etensor_qkv_partial,
                            dep=lambda rank, x, inv_idx: (
                                T.if_then_else(x >= self.NUM_ATTENTION_HEADS + self.NUM_KEY_VALUE_HEADS, reduce_append_v_num_tiles_0, 0),
                                inv_idx, x - self.NUM_ATTENTION_HEADS - self.NUM_KEY_VALUE_HEADS, 0
                            ),
                        ),
                        out_deps=rx.utils.Dependency(
                            event=etensor_notify_attn,
                            dep=reduce_append_v_dep_notify,
                        ),
                        inplace_indices=1,
                    )
                )

                # ATTENTION
                batch_attn_packed = bb.emit(
                    R.call_tir_device(
                        batch_attn_gv,
                        [
                            q, kv_data_, q_indptr_, kv_indptr_, partial_indptr_, page_kv_indices_, q_len_, kv_len_,
                            q_start_, kv_start_, kv_end_, kv_head_idx_, work_indptr_, len_kv_chunk_
                        ],
                        out_sinfo=[
                            rx.TensorStructInfo([batch_size, self.NUM_ATTENTION_HEADS, self.HEAD_DIM], "float16"),
                            rx.TensorStructInfo([FuseBatchAttnTile.max_num_kv_splits * self.NUM_KEY_VALUE_HEADS * self.HEAD_DIM], "float32"),
                            rx.TensorStructInfo([FuseBatchAttnTile.max_num_kv_splits * self.NUM_KEY_VALUE_HEADS], "float32"),
                        ],
                        job_id=JobType.BATCH_ATTENTION.value,
                        tile_num=batch_attn_num_tiles,
                        in_deps=rx.utils.Dependency(
                            event=etensor_notify_attn,
                            dep=lambda i, j, k, wait_idx: (1, -1, i),
                        ),
                        inverse_in_deps=rx.utils.Dependency(
                            event=etensor_notify_attn,
                            dep=lambda rank, x, inv_idx: (1, x, 0, 0),
                        ),
                        out_deps=[
                            rx.utils.Dependency(
                                event=etensor_attn_merge,
                                dep=lambda i, j, k, notify_idx: (
                                    T.if_then_else(attn_task_num > batch_size * self.NUM_KEY_VALUE_HEADS, 1, 0), -1, 0
                                ),
                            ),
                            rx.utils.Dependency(
                                event=etensor_o_proj,
                                dep=lambda i, j, k, notify_idx: (
                                    T.if_then_else(attn_task_num <= batch_size * self.NUM_KEY_VALUE_HEADS, self.SPLIT_O_PROJECT, 0), -1, notify_idx
                                ),
                            )
                        ],
                        handle_config={"wait_scope": "warp"}
                    )
                )
                o, partial_splitkv_o, partial_lse = (
                    bb.emit(R.TupleGetItem(batch_attn_packed, 0)),
                    bb.emit(R.TupleGetItem(batch_attn_packed, 1)),
                    bb.emit(R.TupleGetItem(batch_attn_packed, 2)),
                )

                # ATTN_MERGE
                def batch_merge_dep_notify(i, j, k, notify_idx):
                    worker_id = i * KernelConfig.WG_NUMBER * KernelConfig.WARP_NUMBER
                    kv_idx = T.truncdiv(worker_id, batch_size * (self.NUM_ATTENTION_HEADS // self.NUM_KEY_VALUE_HEADS))
                    range_start = T.truncdiv((kv_idx * (self.NUM_ATTENTION_HEADS // self.NUM_KEY_VALUE_HEADS)) * self.HEAD_DIM, o_proj_size_2)
                    range_end = T.truncdiv(((kv_idx + 1) * (self.NUM_ATTENTION_HEADS // self.NUM_KEY_VALUE_HEADS)) * self.HEAD_DIM - 1, o_proj_size_2)
                    return range_end - range_start + 1, -1, range_start + notify_idx

                batch_merge_num_tiles_0 = batch_merge_num_tiles[0]
                o = bb.emit(
                    R.call_tir_device(
                        batch_merge_gv,
                        [
                            partial_splitkv_o,
                            partial_lse,
                            num_qo_len_,
                            merge_indptr_,
                            merge_o_indices_,
                            o,
                        ],
                        out_sinfo=rx.TensorStructInfo([batch_size, self.NUM_ATTENTION_HEADS, self.HEAD_DIM], "float16"),
                        job_id=JobType.BATCH_ATTENTION_MERGE.value,
                        tile_num=batch_merge_num_tiles,
                        in_deps=rx.utils.Dependency(
                            event=etensor_attn_merge,
                            dep=lambda i, j, k, wait_idx: (1, -1, 0),
                        ),
                        inverse_in_deps=rx.utils.Dependency(
                            event=etensor_attn_merge,
                            dep=lambda rank, x, inv_idx: (batch_merge_num_tiles_0, inv_idx, 0, 0),
                        ),
                        out_deps=rx.utils.Dependency(
                            event=etensor_o_proj,
                            dep=batch_merge_dep_notify,
                        ),
                        inplace_indices=[5],
                    )
                )

                # O_PROJ
                o_proj_num_tiles_1 = o_proj_num_tiles[1]
                residual = bb.emit(
                    R.call_tir_device(
                        o_proj_gv,
                        [o, o_proj_weight, residual],
                        out_sinfo=rx.TensorStructInfo([batch_size, self.HIDDEN_SIZE], "float32"),
                        tile_num=o_proj_num_tiles,
                        job_id=JobType.GEMM_O_PROJ.value,
                        in_deps=rx.utils.Dependency(
                            event=etensor_o_proj,
                            dep=lambda i, j, k, wait_idx: (1, -1, k),
                        ),
                        inverse_in_deps=rx.utils.Dependency(
                            event=etensor_o_proj,
                            dep=lambda rank, x, inv_idx: (o_proj_num_tiles_1, 0, inv_idx, x),
                        ),
                        out_deps=rx.utils.Dependency(
                            event=etensor_attn_add_rms,
                            dep=lambda i, j, k, notify_idx: (1, -1, 0),
                        ),
                        inplace_indices=[2],
                    )
                )

                # ATTN_ADD_RMS
                mlp_hidden_state = bb.emit(
                    R.call_tir_device(
                        attn_add_rms_gv,
                        [residual, attn_add_rms_weight],
                        out_sinfo=rx.TensorStructInfo([batch_size, self.HIDDEN_SIZE], "float16"),
                        job_id=JobType.ATTN_ADD_RMS_NORM.value,
                        tile_num=attn_add_rms_num_tiles,
                        in_deps=rx.utils.Dependency(
                            event=etensor_attn_add_rms,
                            dep=lambda i, j, k, wait_idx: (1, -1, 0),
                        ),
                        inverse_in_deps=rx.utils.Dependency(
                            event=etensor_attn_add_rms,
                            dep=lambda rank, x, inv_idx: (batch_size, inv_idx, 0, 0),
                        ),
                        out_deps=rx.utils.Dependency(
                            event=etensor_attn_mlp,
                            dep=lambda i, j, k, notify_idx: (1, -1, 0),
                        )
                    )
                )

                # GATE_UP_SILU
                def gate_up_silu_dep_notify(i, j, k, notify_idx):
                    range_start = j * gate_up_silu_size_1 // 2 // down_proj_size_2
                    range_end = ((j + 1) * gate_up_silu_size_1 // 2 - 1) // down_proj_size_2
                    return range_end - range_start + 1, -1, range_start + notify_idx

                gate_up_silu_num_tiles_1 = gate_up_silu_num_tiles[1]
                out_silu_mul = bb.emit(
                    R.call_tir_device(
                        gate_up_silu_gv,
                        [mlp_hidden_state, gate_up_weight],
                        out_sinfo=rx.TensorStructInfo([batch_size, self.INTERMEDIATE_SIZE], "float16"),
                        job_id=JobType.GATE_UP_SILU.value,
                        tile_num=gate_up_silu_num_tiles,
                        in_deps=rx.utils.Dependency(
                            event=etensor_attn_mlp,
                            dep=lambda i, j, k, wait_idx: (1, -1, 0),
                        ),
                        inverse_in_deps=rx.utils.Dependency(
                            event=etensor_attn_mlp,
                            dep=lambda rank, x, inv_idx: (gate_up_silu_num_tiles_1, 0, inv_idx, 0),
                        ),
                        out_deps=rx.utils.Dependency(
                            event=etensor_down_proj,
                            dep=gate_up_silu_dep_notify,
                        )
                    )
                )

                # DOWN_PROJ
                residual = bb.emit(
                    R.call_tir_device(
                        down_proj_gv,
                        [out_silu_mul, down_weight, residual],
                        out_sinfo=rx.TensorStructInfo([batch_size, self.HIDDEN_SIZE], "float32"),
                        job_id=JobType.GEMM_DOWN_PROJ.value,
                        tile_num=down_proj_num_tiles,
                        in_deps=rx.utils.Dependency(
                            event=etensor_down_proj,
                            dep=lambda i, j, k, wait_idx: (1, -1, k),
                        ),
                        inverse_in_deps=rx.utils.Dependency(
                            event=etensor_down_proj,
                            dep=lambda rank, x, inv_idx: (down_proj_num_tiles[1], 0, inv_idx, x),
                        ),
                        out_deps=rx.utils.Dependency(
                            event=etensor_mlp_add_rms,
                            dep=lambda i, j, k, notify_idx: (1, -1, 0),
                        ),
                        inplace_indices=[2],
                    )
                )

                # MLP_ADD_RMS
                output_hidden_state = bb.emit(
                    R.call_tir_device(
                        mlp_add_rms_gv,
                        [residual, mlp_add_rms_weight],
                        out_sinfo=rx.TensorStructInfo([batch_size, self.HIDDEN_SIZE], "float16"),
                        job_id=JobType.MLP_ADD_RMS_NORM.value,
                        tile_num=mlp_add_rms_num_tiles,
                        in_deps=rx.utils.Dependency(
                            event=etensor_mlp_add_rms,
                            dep=lambda i, j, k, wait_idx: (1, -1, 0),
                        ),
                        inverse_in_deps=rx.utils.Dependency(
                            event=etensor_mlp_add_rms,
                            dep=lambda rank, x, inv_idx: (batch_size, inv_idx, 0, 0),
                        ),
                        out_deps=rx.utils.Dependency(
                            event=etensor_end,
                            dep=lambda i, j, k, notify_idx: (1, -1, 0),
                        ),
                    )
                )

                output = rx.Tuple([output_hidden_state, residual])
                # out = bb.emit_output(mlp_add_rmsnorm_packed)

            # output
                bb.emit_func_output(output)

        func_gv = bb.add_func(bb.get()[f"megakernel_blkm{blk_m}"], f"megakernel_blkm{blk_m}")

        return func_gv

    def get_mod(self, max_batch_size, profile_on=False):
        bb = rx.BlockBuilder()

        inner_blkm32 = self._qwen3_layer_inner(bb, max_batch_size, blk_m=32, profile_on=profile_on)
        inner_blkm64 = self._qwen3_layer_inner(bb, max_batch_size, blk_m=64, profile_on=profile_on)
        inner_blkm128 = self._qwen3_layer_inner(bb, max_batch_size, blk_m=128, profile_on=profile_on)

        # dispatcher function
        batch_size = T.var("int64", name="batch_size")
        x = rx.Var("x", rx.TensorStructInfo([batch_size, self.HIDDEN_SIZE], "float16"))
        residual = rx.Var("residual", rx.TensorStructInfo([batch_size, self.HIDDEN_SIZE], "float32"))
        packed_info = rx.Var(
            "packed_info",
            rx.TupleStructInfo(
                [rx.TensorStructInfo(None, dtype="float16")] +
                [rx.TensorStructInfo(None, dtype="int32")] * 9 +
                [rx.TensorStructInfo([self.MAX_TOTAL_NUM_WORKERS], dtype="int32")] * 10 +
                [rx.TensorStructInfo(None, dtype="float32")] +
                [rx.TensorStructInfo([self.MAX_TOTAL_NUM_WORKERS], dtype="int32")] * 2 +
                [R.Shape()]
            ),
        )
        packed_weights = rx.Var(
            "packed_weights",
            rx.TupleStructInfo(
                [
                    rx.TensorStructInfo([(self.NUM_ATTENTION_HEADS + self.NUM_KEY_VALUE_HEADS * 2) * self.HEAD_DIM, self.HIDDEN_SIZE], "float16"),
                    rx.TensorStructInfo([self.HEAD_DIM], "float16"),
                    rx.TensorStructInfo([self.HEAD_DIM], "float16"),
                    rx.TensorStructInfo([self.HIDDEN_SIZE, self.NUM_ATTENTION_HEADS * self.HEAD_DIM], "float16"),
                    rx.TensorStructInfo([self.HIDDEN_SIZE], "float16"),
                    rx.TensorStructInfo([self.INTERMEDIATE_SIZE * 2, self.HIDDEN_SIZE], "float16"),
                    rx.TensorStructInfo([self.HIDDEN_SIZE, self.INTERMEDIATE_SIZE], "float16"),
                    rx.TensorStructInfo([self.HIDDEN_SIZE], "float16"),
                ]
            ),
        )
        workspace = rx.Var("workspace", rx.TensorStructInfo([self.EVT_WORKSPACE_SIZE], "int32"))

        with bb.function("megakernel", [x, residual, packed_info, packed_weights, workspace]):
            with bb.dataflow():
                runtime_batch_size = T.var("int64", name="runtime_batch_size")
                _ = bb.match_cast(R.shape_of(x), R.Shape([runtime_batch_size, self.HIDDEN_SIZE]))
                if profile_on:
                    output_sinfo = [
                        rx.TensorStructInfo([batch_size, self.HIDDEN_SIZE], "float16"),
                        rx.TensorStructInfo([batch_size, self.HIDDEN_SIZE], "float32"),
                        rx.TensorStructInfo([T.int64(1e7)], "uint64")
                    ]
                    output_shape = R.Tuple(R.Tensor([batch_size, self.HIDDEN_SIZE], "float16"),
                                           R.Tensor([batch_size, self.HIDDEN_SIZE], "float32"),
                                           R.Tensor([T.int64(1e7)], "uint64"))
                else:
                    output_sinfo = [
                        rx.TensorStructInfo([batch_size, self.HIDDEN_SIZE], "float16"),
                        rx.TensorStructInfo([batch_size, self.HIDDEN_SIZE], "float32"),
                    ]
                    output_shape = R.Tuple(R.Tensor([batch_size, self.HIDDEN_SIZE], "float16"),
                                           R.Tensor([batch_size, self.HIDDEN_SIZE], "float32"))
                def bind_branch_with_r_func(r_func, blk_m):
                    out = rx.Var(f"out_blkm{blk_m}", output_shape)
                    bindings = [rx.VarBinding(out, rx.Call(r_func, [x, residual, packed_info, packed_weights, workspace], sinfo_args=output_sinfo))]
                    bindings_blocks = [rx.BindingBlock(bindings)]
                    bindings_seq_expr = rx.SeqExpr(bindings_blocks, bindings_blocks[-1].bindings[-1].var)
                    return bindings_seq_expr

                def bind_branch_with_if_else(cond, then_seq, else_seq):
                    out = rx.Var("out", output_shape)
                    bindings = [rx.VarBinding(out, rx.If(R.prim_value(cond), then_seq, else_seq))]
                    bindings_blocks = [rx.BindingBlock(bindings)]
                    bindings_seq_expr = rx.SeqExpr(bindings_blocks, bindings_blocks[-1].bindings[-1].var)
                    return bindings_seq_expr

                output_tuple = rx.If(
                    R.prim_value(runtime_batch_size <= 32),
                    bind_branch_with_r_func(inner_blkm32, 32),
                    bind_branch_with_if_else(runtime_batch_size <= 64, bind_branch_with_r_func(inner_blkm64, 64), bind_branch_with_r_func(inner_blkm128, 128))
                )
                out = bb.emit_output(output_tuple)
            bb.emit_func_output(out)
        return bb.get()


arg_dict = {}
def prepare_data(batch_size, seq_len, mk: MegaKernel):
    global arg_dict
    import torch

    def _correct_weight_tensor_view(tensor):
        if mk.world_size == 1:
            return tensor.view(*tensor.shape[1:])
        return tensor

    torch.manual_seed(42)

    # input
    arg_dict["hidden_state"] = torch.randn(
        (batch_size, mk.HIDDEN_SIZE), dtype=torch.float16
    )
    arg_dict["residual"] = torch.randn(
        (batch_size, mk.HIDDEN_SIZE), dtype=torch.float16
    )

    # rms
    arg_dict["q_rms_wight"] = torch.randn((mk.HEAD_DIM), dtype=torch.float16)
    arg_dict["k_rms_wight"] = torch.randn((mk.HEAD_DIM), dtype=torch.float16)

    # qkv
    arg_dict["qkv"] = torch.randn(
        (
            batch_size,
            mk.NUM_KEY_VALUE_HEADS * 2 + mk.NUM_ATTENTION_HEADS,
            mk.HEAD_DIM,
        ),
        dtype=torch.float16,
    )

    # rope cos_sin_cache
    inv_freq = 1.0 / (
        mk.ROPE_THETA
        ** (
            torch.arange(0, mk.HEAD_DIM, 2, dtype=torch.float, device="cuda")
            / mk.HEAD_DIM
        )
    )
    pos = seq_len - 1
    assert pos < 4096  # for faster test
    arg_dict["rope_pos"] = torch.full((batch_size,), pos, dtype=torch.int32)
    t = torch.arange(4096, dtype=torch.float, device="cuda")
    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()
    arg_dict["cos_sin_cache"] = (
        torch.cat((cos, sin), dim=-1).reshape(-1, mk.HEAD_DIM).cpu()
    )

    # paged kv-cache
    page_last_len = mk.PAGE_SIZE if seq_len % mk.PAGE_SIZE == 0 else seq_len % mk.PAGE_SIZE
    page_num = ceildiv(seq_len, mk.PAGE_SIZE)
    total_page_num = page_num * batch_size
    assert total_page_num <= mk.MAX_PAGE_NUM
    kv_indptr = torch.empty(batch_size + 1, dtype=torch.int32).int()
    for i in range(batch_size + 1):
        kv_indptr[i] = i * page_num
    kv_last_page_len = torch.empty(batch_size, dtype=torch.int32).int()
    for i in range(batch_size):
        kv_last_page_len[i] = page_last_len
    kv_indices = torch.arange(mk.MAX_PAGE_NUM, dtype=torch.int32).int()
    kv_indices = kv_indices[torch.randperm(mk.MAX_PAGE_NUM)]
    kv_indices = kv_indices[:total_page_num]
    append_pos = torch.empty(batch_size, dtype=torch.int32).int()
    for i in range(batch_size):
        append_pos[i] = seq_len - 1
    arg_dict["page_kv_indptr"] = kv_indptr.cpu()
    arg_dict["page_kv_last_page_len"] = kv_last_page_len.cpu()
    arg_dict["page_kv_indices"] = kv_indices.cpu()
    arg_dict["append_pos"] = append_pos.cpu()

    # output
    arg_dict["o"] = torch.zeros(
        (batch_size, mk.NUM_ATTENTION_HEADS, mk.HEAD_DIM), dtype=torch.float16
    )
    arg_dict["lse"] = torch.zeros(
        (batch_size, mk.NUM_ATTENTION_HEADS), dtype=torch.float32
    )
    arg_dict["hidden_state_attn_mlp"] = torch.zeros(
        (batch_size, mk.HIDDEN_SIZE), dtype=torch.float16
    )

    # add rms
    arg_dict["attn_add_rms_weight"] = torch.randn(
        (mk.HIDDEN_SIZE), dtype=torch.float16
    )

    # mlp
    arg_dict["out_gate_up_proj"] = torch.zeros(
        (batch_size, mk.INTERMEDIATE_SIZE * 2), dtype=torch.float16
    )
    arg_dict["out_silu_multiply"] = torch.zeros(
        (batch_size, mk.INTERMEDIATE_SIZE), dtype=torch.float16
    )
    arg_dict["partial_sum_down_proj"] = torch.zeros(
        (mk.DOWN_PROJ_SPLIT_K_FACTOR, batch_size, mk.HIDDEN_SIZE),
        dtype=torch.float32,
    )
    arg_dict["output"] = torch.zeros((batch_size, mk.HIDDEN_SIZE), dtype=torch.float16)
    arg_dict["mlp_add_rms_weight"] = torch.randn(
        (mk.HIDDEN_SIZE,), dtype=torch.float16
    )

    # plan info
    wrapper = flashinfer.BatchAttention("HND")
    wrapper.plan(
        torch.arange(0, batch_size + 1, dtype=torch.int32).to(0),
        arg_dict["page_kv_indptr"].to(0),
        arg_dict["page_kv_indices"].to(0),
        torch.tensor([seq_len] * batch_size, dtype=torch.int32).to(0),
        mk.NUM_ATTENTION_HEADS,
        mk.NUM_KEY_VALUE_HEADS,
        mk.HEAD_DIM,
        mk.HEAD_DIM,
        mk.PAGE_SIZE,
        kv_data_type=torch.float16,
        q_data_type=torch.float16,
    )
    plan_info = wrapper._plan_info

    def get_id(i):
        return plan_info[i].item()

    def tensor_from_bytes(
        byte_tensor: torch.Tensor, offset: int, shape, data_type: torch.dtype
    ) -> torch.Tensor:
        if byte_tensor.dtype != torch.uint8 or byte_tensor.dim() != 1:
            raise ValueError("Input must be a 1D torch.uint8 tensor.")

        num_elements = shape
        element_byte_size = torch.tensor([], dtype=data_type).element_size()
        required_bytes = num_elements * element_byte_size

        if offset + required_bytes > byte_tensor.numel():
            raise ValueError("The requested offset and shape are out of bounds.")

        byte_slice = byte_tensor[offset : offset + required_bytes]

        return byte_slice.view(data_type)

    def get_tensor(offset, shape, data_type):
        if data_type == torch.int32:
            return tensor_from_bytes(wrapper.int_workspace_buffer, offset, shape, data_type)
        elif data_type in [torch.float16, torch.float32]:
            return tensor_from_bytes(wrapper.float_workspace_buffer, offset, shape, data_type)
        else:
            raise ValueError(f"Unsupported data type: {data_type}")

    arg_dict["q_indptr"] = get_tensor(
        get_id(mk.NUM_TASK_ARGS + 2), mk.MAX_TOTAL_NUM_WORKERS, torch.int32
    ).cpu()
    arg_dict["kv_indptr"] = get_tensor(
        get_id(mk.NUM_TASK_ARGS + 3), mk.MAX_TOTAL_NUM_WORKERS, torch.int32
    ).cpu()
    arg_dict["partial_indptr"] = get_tensor(
        get_id(mk.NUM_TASK_ARGS + 4), mk.MAX_TOTAL_NUM_WORKERS, torch.int32
    ).cpu()
    arg_dict["q_len"] = get_tensor(
        get_id(mk.NUM_TASK_ARGS + 5), mk.MAX_TOTAL_NUM_WORKERS, torch.int32
    ).cpu()
    arg_dict["kv_len"] = get_tensor(
        get_id(mk.NUM_TASK_ARGS + 6), mk.MAX_TOTAL_NUM_WORKERS, torch.int32
    ).cpu()
    arg_dict["q_start"] = get_tensor(
        get_id(mk.NUM_TASK_ARGS + 7), mk.MAX_TOTAL_NUM_WORKERS, torch.int32
    ).cpu()
    arg_dict["kv_start"] = get_tensor(
        get_id(mk.NUM_TASK_ARGS + 8), mk.MAX_TOTAL_NUM_WORKERS, torch.int32
    ).cpu()
    arg_dict["kv_end"] = get_tensor(
        get_id(mk.NUM_TASK_ARGS + 9), mk.MAX_TOTAL_NUM_WORKERS, torch.int32
    ).cpu()
    arg_dict["kv_head_idx"] = get_tensor(
        get_id(mk.NUM_TASK_ARGS + 10), mk.MAX_TOTAL_NUM_WORKERS, torch.int32
    ).cpu()
    arg_dict["work_indptr"] = get_tensor(
        get_id(mk.NUM_TASK_ARGS + 11), mk.MAX_TOTAL_NUM_WORKERS, torch.int32
    ).cpu()
    arg_dict["attn_task_num"] = get_tensor(
        get_id(mk.NUM_TASK_ARGS + 11), mk.MAX_TOTAL_NUM_WORKERS, torch.int32
    )[2 * KernelConfig.SM_NUMBER].cpu()
    arg_dict["len_kv_chunk"] = get_tensor(
        get_id(mk.NUM_TASK_ARGS + 12), 2, torch.int32
    ).cpu()
    arg_dict["merge_indptr"] = get_tensor(
        get_id(mk.NUM_TASK_ARGS + 15), mk.MAX_NUM_KV_SPLITS, torch.int32
    ).cpu()
    arg_dict["merge_o_indices"] = get_tensor(
        get_id(mk.NUM_TASK_ARGS + 16), mk.MAX_NUM_KV_SPLITS, torch.int32
    ).cpu()
    arg_dict["num_qo_len"] = get_tensor(
        get_id(mk.NUM_TASK_ARGS + 17), 1, torch.int32
    ).cpu()

    # weight initialization
    if not hasattr(prepare_data, "weight_initialized"):
        prepare_data.weight_initialized = True
    else:
        return arg_dict
    arg_dict["kv_cache"] = _correct_weight_tensor_view(torch.randn(
        (
            mk.world_size,
            mk.MAX_PAGE_NUM,
            2,
            mk.NUM_KEY_VALUE_HEADS,
            mk.PAGE_SIZE,
            mk.HEAD_DIM,
        ),
        dtype=torch.float16,
    )).cpu()
    arg_dict["qkv_proj_weight"] = _correct_weight_tensor_view(torch.randn(
        (
            mk.world_size,
            (mk.NUM_ATTENTION_HEADS + 2 * mk.NUM_KEY_VALUE_HEADS)
            * mk.HEAD_DIM,
            mk.HIDDEN_SIZE,
        ),
        dtype=torch.float16,
    ))
    torch.nn.init.xavier_normal_(arg_dict["qkv_proj_weight"], gain=1.0)
    arg_dict["o_proj_weight"] = _correct_weight_tensor_view(torch.randn(
        (
            mk.world_size,
            mk.HIDDEN_SIZE,
            mk.NUM_ATTENTION_HEADS * mk.HEAD_DIM
        ),
        dtype=torch.float16,
    ))
    torch.nn.init.xavier_normal_(arg_dict["o_proj_weight"], gain=1.0)
    arg_dict["gate_up_weight"] = _correct_weight_tensor_view(
        torch.zeros((mk.world_size, mk.INTERMEDIATE_SIZE * 2, mk.HIDDEN_SIZE), dtype=torch.float16)
    )
    torch.nn.init.xavier_normal_(arg_dict["gate_up_weight"], gain=1.0)
    arg_dict["down_weight"] = _correct_weight_tensor_view(torch.zeros(
        (
            mk.world_size,
            mk.HIDDEN_SIZE,
            mk.INTERMEDIATE_SIZE
        ),
        dtype=torch.float16
    ))
    torch.nn.init.xavier_normal_(arg_dict["down_weight"], gain=1.0)
    return arg_dict

def get_packed_info(arg_dict, mk: MegaKernel):
    append_pos = arg_dict["append_pos"].clone()
    for b in range(batch_size):
        append_pos[b] = (
            arg_dict["page_kv_indices"][(arg_dict["page_kv_indptr"][b] * mk.PAGE_SIZE + append_pos[b]) // mk.PAGE_SIZE]
            * mk.PAGE_SIZE + append_pos[b] % mk.PAGE_SIZE
        )
    inverse_info = get_inverse_plan_info(batch_size, mk.NUM_KEY_VALUE_HEADS, arg_dict["q_indptr"], arg_dict["kv_head_idx"], arg_dict["attn_task_num"].item())
    info = [
        arg_dict["kv_cache"],
        arg_dict["page_kv_indptr"],
        arg_dict["page_kv_indices"],
        arg_dict["page_kv_last_page_len"],
        append_pos,
        arg_dict["rope_pos"],
        arg_dict["len_kv_chunk"],
        arg_dict["merge_indptr"],
        arg_dict["merge_o_indices"],
        arg_dict["num_qo_len"],
        arg_dict["q_indptr"],
        arg_dict["kv_indptr"],
        arg_dict["partial_indptr"],
        arg_dict["q_len"],
        arg_dict["kv_len"],
        arg_dict["q_start"],
        arg_dict["kv_start"],
        arg_dict["kv_end"],
        arg_dict["kv_head_idx"],
        arg_dict["work_indptr"],
        arg_dict["cos_sin_cache"],
        inverse_info[0],
        inverse_info[1],
    ]
    info = [tvm.runtime.tensor(info, device=tvm.cuda()) for info in info]
    info.append(tvm.runtime.ShapeTuple([arg_dict["attn_task_num"].item()]))
    return info

@tvm.testing.requires_cuda_compute_version(10, exact=True)
@pytest.mark.skip
def test(batch_size, seq_len, vm, mega_kernel_wrapper, profile_on=False):
    arg_dict = prepare_data(batch_size, seq_len, mega_kernel_wrapper)

    def tir(vm, arg_dict, batch_size, mk: MegaKernel):
        dev = tvm.cuda()
        hidden_state = tvm.runtime.tensor(arg_dict["hidden_state"], device=dev)
        # residual = tvm.runtime.tensor(arg_dict["residual"].to(torch.float32), device=dev)

        if mk.GATE_UP_PROJ_SPLIT_K_FACTOR == 1:
            # reorder the gate_up_weight for the fusion of gate_up projection and silu
            new_order_indices = np.stack(
                (
                    np.arange(mk.INTERMEDIATE_SIZE).reshape(-1, 16),
                    np.arange(mk.INTERMEDIATE_SIZE, mk.INTERMEDIATE_SIZE * 2).reshape(-1, 16)
                ), axis=1
            ).reshape(-1)
            if mk.world_size > 1:
                gate_up_weight = arg_dict["gate_up_weight"][:, new_order_indices, :]
            else:
                gate_up_weight = arg_dict["gate_up_weight"][new_order_indices, :]

        weights = [
            arg_dict["qkv_proj_weight"],
            arg_dict["q_rms_wight"],
            arg_dict["k_rms_wight"],
            arg_dict["o_proj_weight"],
            arg_dict["attn_add_rms_weight"],
            gate_up_weight,
            arg_dict["down_weight"],
            arg_dict["mlp_add_rms_weight"],
        ]
        weights = [tvm.runtime.tensor(weight, device=dev) for weight in weights]
        packed_info = get_packed_info(arg_dict, mk)
        # events = prepare_events(arg_dict, batch_size, max_batch_size=128, mk=mk)
        workspace = tvm.runtime.tensor(np.zeros((mk.EVT_WORKSPACE_SIZE,), dtype=np.int32), device=dev)
        iter = 0
        def func():
            nonlocal iter
            iter += 1
            residual = tvm.runtime.tensor(arg_dict["residual"].to(torch.float32), device=dev)
            res = vm["megakernel"](hidden_state, residual, packed_info, weights, workspace)
            return res
        res = bench(func, warmup=3, repeat=10, proton_name="tir")
        res = func()
        if profile_on:
            export_to_perfetto_trace(res[2].numpy(), f"layer.perfetto-trace", event_type_names)
        return res[0].numpy(), res[1].numpy().astype(np.float16)

    def std(arg_dict, batch_size, use_prefill, mk: MegaKernel):
        import flashinfer
        import torch

        FULL_INTERMEDIATE_SIZE = mk.INTERMEDIATE_SIZE * mk.world_size
        FULL_NUM_ATTENTION_HEADS = mk.NUM_ATTENTION_HEADS * mk.world_size
        FULL_NUM_KEY_VALUE_HEADS = mk.NUM_KEY_VALUE_HEADS * mk.world_size

        torch_dev = torch.device("cuda")
        std_arg_dict = {}

        def func():
            for key, value in arg_dict.items():
                if mk.world_size > 1:
                    if key == "qkv_proj_weight":
                        split_sizes = [mk.NUM_ATTENTION_HEADS, mk.NUM_KEY_VALUE_HEADS, mk.NUM_KEY_VALUE_HEADS]
                        value = value.reshape(mk.world_size, -1, mk.HEAD_DIM, mk.HIDDEN_SIZE)
                        q_weight, k_weight, v_weight = torch.split(value, split_sizes, dim=1)
                        q_weight = q_weight.reshape(-1, mk.HIDDEN_SIZE)
                        k_weight = k_weight.reshape(-1, mk.HIDDEN_SIZE)
                        v_weight = v_weight.reshape(-1, mk.HIDDEN_SIZE)
                        value = torch.cat([q_weight, k_weight, v_weight], dim=0)
                    elif key == "gate_up_weight":
                        value = value.reshape(-1, *value.shape[2:])
                    elif key == "o_proj_weight" or key == "down_weight":
                        value = value.transpose(0, 1)
                        value = value.reshape(value.shape[0], -1)
                    elif key == "kv_cache":
                        value = value.movedim(0, 2)
                        value = value.reshape(value.shape[0], value.shape[1], -1, *value.shape[4:])
                std_arg_dict[key] = value.clone().to(torch_dev)
            out_f = torch.zeros(
                batch_size, FULL_NUM_ATTENTION_HEADS, mk.HEAD_DIM, dtype=torch.float16, device="cuda"
            )
            lse_f = torch.zeros(batch_size, FULL_NUM_ATTENTION_HEADS, dtype=torch.float32, device="cuda")
            if use_prefill:
                workspace_buffer = torch.empty(1024 * 1024 * 1024, dtype=torch.int8).to(0)
                wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace_buffer, "HND")
                wrapper.plan(
                    torch.arange(0, batch_size + 1, dtype=torch.int32).to(0),
                    std_arg_dict["page_kv_indptr"],
                    std_arg_dict["page_kv_indices"],
                    std_arg_dict["page_kv_last_page_len"],
                    FULL_NUM_ATTENTION_HEADS,
                    FULL_NUM_KEY_VALUE_HEADS,
                    mk.HEAD_DIM,
                    mk.PAGE_SIZE,
                    pos_encoding_mode="NONE",
                    kv_data_type=torch.float16,
                    q_data_type=torch.float16,
                )
            else:
                wrapper = flashinfer.BatchAttention("HND")
                wrapper.plan(
                    torch.arange(0, batch_size + 1, dtype=torch.int32).to(0),
                    std_arg_dict["page_kv_indptr"],
                    std_arg_dict["page_kv_indices"],
                    torch.tensor([seq_len] * batch_size, dtype=torch.int32).to(0),
                    FULL_NUM_ATTENTION_HEADS,
                    FULL_NUM_KEY_VALUE_HEADS,
                    mk.HEAD_DIM,
                    mk.HEAD_DIM,
                    mk.PAGE_SIZE,
                    kv_data_type=torch.float16,
                    q_data_type=torch.float16,
                )
            qkv = torch.matmul(
                std_arg_dict["hidden_state"], std_arg_dict["qkv_proj_weight"].T
            ).reshape(batch_size, -1, mk.HEAD_DIM)
            q, k, v = torch.split(
                qkv, [FULL_NUM_ATTENTION_HEADS, FULL_NUM_KEY_VALUE_HEADS, FULL_NUM_KEY_VALUE_HEADS], dim=1
            )
            q = flashinfer.norm.rmsnorm(
                input=q.reshape(-1, mk.HEAD_DIM),
                weight=std_arg_dict["q_rms_wight"],
                eps=mk.RMS_NORM_EPS,
                enable_pdl=False,
            ).reshape(batch_size, FULL_NUM_ATTENTION_HEADS, mk.HEAD_DIM)
            k = flashinfer.norm.rmsnorm(
                input=k.reshape(-1, mk.HEAD_DIM),
                weight=std_arg_dict["k_rms_wight"],
                eps=mk.RMS_NORM_EPS,
                enable_pdl=False,
            ).reshape(batch_size, FULL_NUM_KEY_VALUE_HEADS, mk.HEAD_DIM)
            q, k = flashinfer.rope.apply_rope_with_cos_sin_cache(
                positions=std_arg_dict["rope_pos"],
                query=q.reshape(batch_size, -1),
                key=k.reshape(batch_size, -1),
                head_size=mk.HEAD_DIM,
                cos_sin_cache=std_arg_dict["cos_sin_cache"],
                is_neox=True,
            )
            flashinfer.page.append_paged_kv_cache(
                append_key=k.reshape(batch_size, FULL_NUM_KEY_VALUE_HEADS, mk.HEAD_DIM),
                append_value=v,
                batch_indices=torch.arange(batch_size, dtype=torch.int32, device=torch_dev),
                positions=std_arg_dict["append_pos"],
                paged_kv_cache=std_arg_dict["kv_cache"],
                kv_indices=std_arg_dict["page_kv_indices"],
                kv_indptr=std_arg_dict["page_kv_indptr"],
                kv_last_page_len=std_arg_dict["page_kv_last_page_len"],
                kv_layout="HND",
            )
            if use_prefill:
                out_f = wrapper.run(
                    q.reshape(batch_size, FULL_NUM_ATTENTION_HEADS, mk.HEAD_DIM), std_arg_dict["kv_cache"]
                )
            else:
                wrapper.run(
                    q.reshape(batch_size, FULL_NUM_ATTENTION_HEADS, mk.HEAD_DIM),
                    std_arg_dict["kv_cache"],
                    out_f,
                    lse_f,
                )
            hidden_state_attn_mlp = torch.matmul(
                out_f.reshape(batch_size, FULL_NUM_ATTENTION_HEADS * mk.HEAD_DIM),
                std_arg_dict["o_proj_weight"].T,
            )
            flashinfer.norm.fused_add_rmsnorm(
                input=hidden_state_attn_mlp,
                residual=std_arg_dict["residual"],
                weight=std_arg_dict["attn_add_rms_weight"],
                eps=mk.RMS_NORM_EPS,
                enable_pdl=False,
            )
            out_gate_up_proj = torch.matmul(hidden_state_attn_mlp, std_arg_dict["gate_up_weight"].T)
            out_silu_multiply = flashinfer.activation.silu_and_mul(
                input=out_gate_up_proj,
            )
            output = torch.matmul(out_silu_multiply, std_arg_dict["down_weight"].T)
            flashinfer.norm.fused_add_rmsnorm(
                input=output,
                residual=std_arg_dict["residual"],
                weight=std_arg_dict["mlp_add_rms_weight"],
                eps=mk.RMS_NORM_EPS,
                enable_pdl=False,
            )
            return output.cpu().numpy(), std_arg_dict["residual"].cpu().numpy()

        output = func()
        ms = bench(func, warmup=10, repeat=30, proton_name=f"std-use_prefill={use_prefill}")
        print(f"std time: {ms:.3f} ms")
        return output


    with ProtonContext("blackwell_attn"):
        fused_res, fused_residual = tir(vm, arg_dict, batch_size, mega_kernel_wrapper)
        std_res, std_residual = std(arg_dict, batch_size, True, mega_kernel_wrapper)

    np.testing.assert_allclose(fused_res, std_res, atol=1e-2, rtol=1e-3)
    np.testing.assert_allclose(fused_residual, std_residual, atol=1e-2, rtol=1e-3)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="MegaKernel testing script.")
    parser.add_argument("--scheduler", type=str, nargs='+', default=["static"],
                        choices=["static", "dynamic"], help="A list of test methods to run.")
    parser.add_argument("--world-size", type=int, default=1, choices=[1],
                        help="The number of devices for the world size, now only support 1.")
    parser.add_argument("--batch-size", type=int, nargs='+',
                        default=[1, 3, 7, 15, 31, 63, 127, 128],
                        help="A list of batch sizes to test.")
    parser.add_argument("--seq-len", type=int, nargs='+', default=[512],
                        help="A list of sequence lengths to test.")
    parser.add_argument("--profiler-on", action="store_true", help="Enable the profiler.")
    args = parser.parse_args()

    tile_scheduler_class_map = {
        "static": static_scheduler.StaticTileScheduler,
        "dynamic": dynamic_scheduler.DynamicTileScheduler,
    }
    semaphore_class_map = {
        "static": static_scheduler.Semaphore,
        "dynamic": dynamic_scheduler.Semaphore,
    }
    for scheduler in args.scheduler:
        print(f"Testing with {scheduler} tile scheduler...", flush=True)
        megakernel_wrapper = MegaKernel()
        mod = megakernel_wrapper.get_mod(max_batch_size=128, profile_on=args.profiler_on)
        mod.show()
        mod = rx.transform.StaticHorizontalFusion(
            ["megakernel_blkm32", "megakernel_blkm64", "megakernel_blkm128"],
            strategy=scheduler, tile_scheduler_class=tile_scheduler_class_map[scheduler],
            semaphore_class=semaphore_class_map[scheduler], profiler_on=args.profiler_on
        )(mod)
        mod.show()
        ex = rx.build(mod, target="cuda", tir_pipeline="tirx")
        src = ex.mod.imports[0].imports[0].inspect_source()
        print(src)
        vm = rx.VirtualMachine(ex, tvm.cuda())
        for batch_size in args.batch_size:
            print(f"batch_size: {batch_size}", flush=True)
            for seq_len in args.seq_len:
                print(f"seq_len: {seq_len}", flush=True)
                test(batch_size, seq_len, vm, megakernel_wrapper, profile_on=args.profiler_on)
