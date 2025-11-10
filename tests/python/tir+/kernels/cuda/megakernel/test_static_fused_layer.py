import argparse
import math
import ml_dtypes
import numpy as np
from enum import Enum
import pytest

import tvm
from tvm.script import tir as T, tirp as Tp, ir as I, relax as R
import tvm.testing
from tvm import relax as rx

import flashinfer
import torch

from tvm.tirp.megakernel.relax_compatible.gemm import FuseGemmTile, FuseGateUpSiluTile
from tvm.tirp.megakernel.relax_compatible.gemm_splitk_reduce import FuseSplitKReduceTile
from tvm.tirp.megakernel.relax_compatible.reduce_rms_rope_append import FuseSplitKReduceRMSnormRopeQTile, FuseSplitKReduceRMSnormRopeAppendKTile, FuseSplitKReduceAppendVTile
from tvm.tirp.megakernel.relax_compatible.batch_attn import FuseBatchAttnTile
from tvm.tirp.megakernel.relax_compatible.batch_merge import FuseBatchMergeTile
from tvm.tirp.megakernel.relax_compatible.add_rmsnorm import FuseAddRMSNormTile, FuseRMSNormTile
from tvm.tirp.megakernel.relax_compatible.end import FuseEndTile

from tvm.tirp.megakernel.common import KernelConfig, ceildiv, JobType
from tvm.tirp.bench.utils import ProtonContext, bench, export_to_perfetto_trace
from tvm.tirp.megakernel.support import get_inverse_plan_info
import tvm.tirp.megakernel.static_scheduler as static_scheduler
import tvm.tirp.megakernel.dynamic_scheduler as dynamic_scheduler

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


    SPLIT_QKV_PROJECT = 3
    SPLIT_O_PROJECT = 3
    DOWN_PROJ_SPLIT_K_FACTOR = 10
    GATE_UP_PROJ_SPLIT_K_FACTOR = 1
    NUM_TASK_ARGS = 10
    MAX_TOTAL_NUM_WORKERS = 1025
    MAX_NUM_KV_SPLITS = 4 * KernelConfig.SM_NUMBER * 2 * 16
    
    def __init__(self):
        self.world_size = 1


    def _qwen3_layer_inner(self, bb: rx.BlockBuilder, max_batch_size, blk_m):
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
        packed_events = rx.Var(
            "packed_events",
            rx.TupleStructInfo(
                [
                    rx.TensorStructInfo([ceildiv((self.NUM_ATTENTION_HEADS + 2 * self.NUM_KEY_VALUE_HEADS) * self.HEAD_DIM, FuseGemmTile.BLK_N)], "int32"),  # etensor_qkv_partial
                    rx.TensorStructInfo([KernelConfig.SM_NUMBER], "int32"),  # etensor_notify_attn
                    rx.TensorStructInfo([max_batch_size * self.NUM_KEY_VALUE_HEADS], "int32"),  # etensor_attn_merge
                    rx.TensorStructInfo([self.SPLIT_O_PROJECT], "int32"),  # etensor_o_proj
                    rx.TensorStructInfo([self.HIDDEN_SIZE // FuseGemmTile.BLK_N], "int32"),  # etensor_o_partial
                    rx.TensorStructInfo([max_batch_size], "int32"),  # etensor_attn_add_rmsnorm
                    rx.TensorStructInfo([1], "int32"),  # etensor_attn_mlp
                    rx.TensorStructInfo([self.DOWN_PROJ_SPLIT_K_FACTOR], "int32"),  # etensor_down_proj
                    rx.TensorStructInfo([self.HIDDEN_SIZE // FuseGemmTile.BLK_N], "int32"),  # etensor_down_proj_reduce
                    rx.TensorStructInfo([max_batch_size], "int32"),  # etensor_mlp_add_rmsnorm
                    rx.TensorStructInfo([1], "int32"),  # etensor_end
                ]
            ),
        )

        with bb.function(f"megakernel_blkm{blk_m}", [x, residual, packed_info, packed_weights, packed_events]):
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
                
                # unpack events
                etensor_qkv_partial = bb.emit(R.TupleGetItem(packed_events, 0))
                etensor_notify_attn = bb.emit(R.TupleGetItem(packed_events, 1))
                etensor_attn_merge = bb.emit(R.TupleGetItem(packed_events, 2))
                etensor_o_proj = bb.emit(R.TupleGetItem(packed_events, 3))
                etensor_o_partial = bb.emit(R.TupleGetItem(packed_events, 4))
                etensor_attn_add_rms = bb.emit(R.TupleGetItem(packed_events, 5))
                etensor_attn_mlp = bb.emit(R.TupleGetItem(packed_events, 6))
                etensor_down_proj = bb.emit(R.TupleGetItem(packed_events, 7))
                etensor_down_proj_reduce = bb.emit(R.TupleGetItem(packed_events, 8))
                etensor_mlp_add_rms = bb.emit(R.TupleGetItem(packed_events, 9))
                etensor_end = bb.emit(R.TupleGetItem(packed_events, 10))

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
                mlp_add_rms_func, mlp_add_rms_num_tiles, _ = FuseRMSNormTile.get_func(
                    batch_size,
                    self.RMS_NORM_EPS,
                    self.HIDDEN_SIZE,
                )
                end_func, end_num_tiles, _ = FuseEndTile.get_func()
                
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
                end_gv = bb.add_func(end_func, "end")

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
                            dep=lambda i, j, k, notify_idx: (-1, j),
                            num=1,
                        ),
                    )
                )
                
                # Q_REDUCE_RMS_ROPE 
                def reduce_rms_rope_q_num_notify(i, j, k, inverse_indptr_buf, inverse_indices_buf, batch_size, m_tile):
                    beg_idx = j // (self.NUM_ATTENTION_HEADS // self.NUM_KEY_VALUE_HEADS) * batch_size + i * m_tile
                    end_idx = j // (self.NUM_ATTENTION_HEADS // self.NUM_KEY_VALUE_HEADS) * batch_size + T.min((i + 1) * m_tile, batch_size)
                    beg = inverse_indptr_buf[beg_idx]
                    end = inverse_indptr_buf[end_idx]
                    return end - beg
                def reduce_rms_rope_q_dep_notify(i, j, k, notify_idx, inverse_indptr_buf, inverse_indices_buf, batch_size, m_tile):
                    beg_idx = j // (self.NUM_ATTENTION_HEADS // self.NUM_KEY_VALUE_HEADS) * batch_size + i * m_tile
                    beg = inverse_indptr_buf[beg_idx]
                    return (-1, inverse_indices_buf[beg + notify_idx])
                q = bb.emit(
                    R.call_tir_device(
                        reduce_rms_rope_q_gv,
                        [qkv_partial, q_rms_weight, rope_pos, cos_sin_cache],
                        rx.TensorStructInfo([batch_size, self.NUM_ATTENTION_HEADS + 2 * self.NUM_KEY_VALUE_HEADS, self.HEAD_DIM], "float16"),
                        job_id=JobType.Q_REDUCE_RMS_ROPE.value,
                        tile_num=reduce_rms_rope_q_num_tiles,
                        in_deps=rx.utils.Dependency(
                            event=etensor_qkv_partial,
                            dep=lambda i, j, k, wait_idx: (-1, j),
                            num=1,
                        ),
                        inverse_in_deps=rx.utils.Dependency(
                            event=etensor_qkv_partial,
                            dep=lambda rank, x, inv_idx, m_split: (inv_idx, x, 0),
                            num=lambda rank, x, m_split: T.if_then_else(x < self.NUM_ATTENTION_HEADS, m_split, 0),
                            extra_args=[reduce_rms_rope_q_num_tiles[0]],
                        ),
                        out_deps=rx.utils.Dependency(
                            event=etensor_notify_attn,
                            dep=reduce_rms_rope_q_dep_notify,
                            num=reduce_rms_rope_q_num_notify,
                            extra_args=[inverse_indptr, inverse_indices, batch_size, reduce_rms_rope_q_size[0]],
                        ),
                    )
                )

                # K_RMS_ROPE_APPEND
                def reduce_rms_rope_append_k_num_notify(i, j, k, inverse_indptr_buf, inverse_indices_buf, batch_size, m_tile):
                    beg_idx = j * batch_size + i * m_tile
                    end_idx = j * batch_size + T.min((i + 1) * m_tile, batch_size)
                    beg = inverse_indptr_buf[beg_idx]
                    end = inverse_indptr_buf[end_idx]
                    return end - beg
                def reduce_rms_rope_append_k_dep_notify(i, j, k, notify_idx, inverse_indptr_buf, inverse_indices_buf, batch_size, m_tile):
                    beg_idx = j * batch_size + i * m_tile
                    beg = inverse_indptr_buf[beg_idx]
                    return (-1, inverse_indices_buf[beg + notify_idx])        
                kv_data_ = bb.emit(
                    R.call_tir_device(
                        reduce_rms_rope_append_k_gv,
                        [qkv_partial, k_rms_weight, rope_pos, cos_sin_cache, append_pos, kv_data_],
                        rx.TensorStructInfo([max_page_num, 2, self.NUM_KEY_VALUE_HEADS, self.PAGE_SIZE, self.HEAD_DIM], "float16"),
                        tile_num=reduce_rms_rope_append_k_num_tiles,
                        job_id=JobType.K_REDUCE_RMS_ROPE_APPEND.value,
                        in_deps=rx.utils.Dependency(
                            event=etensor_qkv_partial,
                            dep=lambda i, j, k, wait_idx: (-1, j + self.NUM_ATTENTION_HEADS),
                            num=1,
                        ),
                        inverse_in_deps=rx.utils.Dependency(
                            event=etensor_qkv_partial,
                            dep=lambda rank, x, inv_idx, m_split: (inv_idx, x - self.NUM_ATTENTION_HEADS, 0),
                            num=lambda rank, x, m_split: T.if_then_else(tvm.tir.all(x >= self.NUM_ATTENTION_HEADS, x < self.NUM_ATTENTION_HEADS + self.NUM_KEY_VALUE_HEADS), m_split, 0),
                            extra_args=[reduce_rms_rope_append_k_num_tiles[0]],
                        ),
                        out_deps=rx.utils.Dependency(
                            event=etensor_notify_attn,
                            dep=reduce_rms_rope_append_k_dep_notify,
                            num=reduce_rms_rope_append_k_num_notify,
                            extra_args=[inverse_indptr, inverse_indices, batch_size, reduce_rms_rope_q_size[0]],
                        ),
                        inplace_indices=5,
                    )
                )

                # V_REDUCE_APPEND
                def reduce_append_v_num_notify(i, j, k, inverse_indptr_buf, inverse_indices_buf, batch_size, m_tile):
                    beg_idx = j * batch_size + i * m_tile
                    end_idx = j * batch_size + T.min((i + 1) * m_tile, batch_size)
                    beg = inverse_indptr_buf[beg_idx]
                    end = inverse_indptr_buf[end_idx]
                    return end - beg
                def reduce_append_v_dep_notify(i, j, k, notify_idx, inverse_indptr_buf, inverse_indices_buf, batch_size, m_tile):
                    beg_idx = j * batch_size + i * m_tile
                    beg = inverse_indptr_buf[beg_idx]
                    return (-1, inverse_indices_buf[beg + notify_idx])        
                kv_data_ = bb.emit(
                    R.call_tir_device(
                        reduce_append_v_gv,
                        [qkv_partial, kv_data_, append_pos],
                        rx.TensorStructInfo([max_page_num, 2, self.NUM_KEY_VALUE_HEADS, self.PAGE_SIZE, self.HEAD_DIM], "float16"),
                        job_id=JobType.V_REDUCE_APPEND.value,
                        tile_num=reduce_append_v_num_tiles,
                        in_deps=rx.utils.Dependency(
                            event=etensor_qkv_partial,
                            dep=lambda i, j, k, wait_idx: (-1, j + self.NUM_ATTENTION_HEADS + self.NUM_KEY_VALUE_HEADS),
                            num=1,
                        ),
                        inverse_in_deps=rx.utils.Dependency(
                            event=etensor_qkv_partial,
                            dep=lambda rank, x, inv_idx, m_split: (inv_idx, x - self.NUM_ATTENTION_HEADS - self.NUM_KEY_VALUE_HEADS, 0),
                            num=lambda rank, x, m_split: T.if_then_else(x >= self.NUM_ATTENTION_HEADS + self.NUM_KEY_VALUE_HEADS, m_split, 0),
                            extra_args=[reduce_append_v_num_tiles[0]],
                        ),
                        out_deps=rx.utils.Dependency(
                            event=etensor_notify_attn,
                            dep=reduce_append_v_dep_notify,
                            num=reduce_append_v_num_notify,
                            extra_args=[inverse_indptr, inverse_indices, batch_size, reduce_rms_rope_q_size[0]],
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
                            dep=lambda i, j, k, wait_idx: (-1, i),
                            num=1,
                        ),
                        inverse_in_deps=rx.utils.Dependency(
                            event=etensor_notify_attn,
                            dep=lambda rank, x, inv_idx: (x, 0, 0),
                            num=1,
                        ),
                        out_deps=[
                            rx.utils.Dependency(
                                event=etensor_attn_merge,
                                dep=lambda i, j, k, notify_idx, attn_task_num, batch_size: (-1, 0),
                                num=lambda i, j, k, attn_task_num, batch_size: T.if_then_else(attn_task_num > batch_size * self.NUM_KEY_VALUE_HEADS, 1, 0),
                                extra_args=[attn_task_num, batch_size],
                            ),
                            rx.utils.Dependency(
                                event=etensor_o_proj,
                                dep=lambda i, j, k, notify_idx, attn_task_num, batch_size: (-1, notify_idx),
                                num=lambda i, j, k, attn_task_num, batch_size: T.if_then_else(attn_task_num <= batch_size * self.NUM_KEY_VALUE_HEADS, self.SPLIT_O_PROJECT, 0),
                                extra_args=[attn_task_num, batch_size],
                            )
                        ]
                    )
                )
                o, partial_splitkv_o, partial_lse = (
                    bb.emit(R.TupleGetItem(batch_attn_packed, 0)),
                    bb.emit(R.TupleGetItem(batch_attn_packed, 1)),
                    bb.emit(R.TupleGetItem(batch_attn_packed, 2)),
                )
                
                # ATTN_MERGE
                def batch_merge_num_notify(i, j, k, batch_size, k_tile):
                    worker_id = i * KernelConfig.WG_NUMBER * KernelConfig.WARP_NUMBER
                    kv_idx = worker_id // (batch_size * (self.NUM_ATTENTION_HEADS // self.NUM_KEY_VALUE_HEADS))
                    range_start = (kv_idx * (self.NUM_ATTENTION_HEADS // self.NUM_KEY_VALUE_HEADS)) * self.HEAD_DIM // k_tile
                    range_end = (((kv_idx + 1) * (self.NUM_ATTENTION_HEADS // self.NUM_KEY_VALUE_HEADS)) * self.HEAD_DIM - 1) // k_tile
                    return range_end - range_start + 1
                def batch_merge_dep_notify(i, j, k, notify_idx, batch_size, k_tile):
                    worker_id = i * KernelConfig.WG_NUMBER * KernelConfig.WARP_NUMBER
                    kv_idx = worker_id // (batch_size * (self.NUM_ATTENTION_HEADS // self.NUM_KEY_VALUE_HEADS))
                    range_start = (kv_idx * (self.NUM_ATTENTION_HEADS // self.NUM_KEY_VALUE_HEADS)) * self.HEAD_DIM // k_tile
                    return (-1, range_start + notify_idx)
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
                            dep=lambda i, j, k, wait_idx: (-1, 0),
                            num=1,
                        ),
                        inverse_in_deps=rx.utils.Dependency(
                            event=etensor_attn_merge,
                            dep=lambda rank, x, inv_idx, num_tile: (inv_idx, 0, 0),
                            num=lambda rank, x, num_tile: num_tile,
                            extra_args=[batch_merge_num_tiles[0]],
                        ),
                        out_deps=rx.utils.Dependency(
                            event=etensor_o_proj,
                            dep=batch_merge_dep_notify,
                            num=batch_merge_num_notify,
                            extra_args=[batch_size, o_proj_size[2]],
                        ),
                        inplace_indices=[5],
                    )
                )
                
                # O_PROJ
                residual = bb.emit(
                    R.call_tir_device(
                        o_proj_gv,
                        [o, o_proj_weight, residual],
                        out_sinfo=rx.TensorStructInfo([batch_size, self.HIDDEN_SIZE], "float32"),
                        tile_num=o_proj_num_tiles,
                        job_id=JobType.GEMM_O_PROJ.value,
                        in_deps=rx.utils.Dependency(
                            event=etensor_o_proj,
                            dep=lambda i, j, k, wait_idx: (-1, k),
                            num=1,
                        ),
                        inverse_in_deps=rx.utils.Dependency(
                            event=etensor_o_proj,
                            dep=lambda rank, x, inv_idx, split_n: (0, inv_idx, x),
                            num=lambda rank, x, split_n: split_n,
                            extra_args=[o_proj_num_tiles[1]],
                        ),
                        out_deps=rx.utils.Dependency(
                            event=etensor_attn_add_rms,
                            dep=lambda i, j, k, notify_idx: (-1, 0),
                            num=1,
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
                            dep=lambda i, j, k, wait_idx: (-1, 0),
                            num=1,
                        ),
                        inverse_in_deps=rx.utils.Dependency(
                            event=etensor_attn_add_rms,
                            dep=lambda rank, x, inv_idx, batch_size: (inv_idx, 0, 0),
                            num=lambda rank, x, batch_size: batch_size,
                            extra_args=[batch_size],
                        ),
                        out_deps=rx.utils.Dependency(
                            event=etensor_attn_mlp,
                            dep=lambda i, j, k, notify_idx: (-1, 0),
                            num=1,
                        )
                    )
                )

                # GATE_UP_SILU
                def gate_up_silu_num_notify(i, j, k, n_tile, k_tile):
                    range_start = j * n_tile // 2 // k_tile
                    range_end = ((j + 1) * n_tile // 2 - 1) // k_tile
                    return range_end - range_start + 1
                def gate_up_silu_dep_notify(i, j, k, notify_idx, n_tile, k_tile):
                    range_start = j * n_tile // 2 // k_tile
                    return (-1, range_start + notify_idx) 
                out_silu_mul = bb.emit(
                    R.call_tir_device(
                        gate_up_silu_gv,
                        [mlp_hidden_state, gate_up_weight],
                        out_sinfo=rx.TensorStructInfo([batch_size, self.INTERMEDIATE_SIZE], "float16"),
                        job_id=JobType.GATE_UP_SILU.value,
                        tile_num=gate_up_silu_num_tiles,
                        in_deps=rx.utils.Dependency(
                            event=etensor_attn_mlp,
                            dep=lambda i, j, k, wait_idx: (-1, 0),
                            num=1,
                        ),
                        inverse_in_deps=rx.utils.Dependency(
                            event=etensor_attn_mlp,
                            dep=lambda rank, x, inv_idx, split_n: (0, inv_idx, 0),
                            num=lambda rank, x, split_n: split_n,
                            extra_args=[gate_up_silu_num_tiles[1]],
                        ),
                        out_deps=rx.utils.Dependency(
                            event=etensor_down_proj,
                            dep=gate_up_silu_dep_notify,
                            num=gate_up_silu_num_notify,
                            extra_args=[gate_up_silu_size[1], down_proj_size[2]],
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
                            dep=lambda i, j, k, wait_idx: (-1, k),
                            num=1,
                        ),
                        inverse_in_deps=rx.utils.Dependency(
                            event=etensor_down_proj,
                            dep=lambda rank, x, inv_idx, split_n: (0, inv_idx, x),
                            num=lambda rank, x, split_n: split_n,
                            extra_args=[down_proj_num_tiles[1]],
                        ),
                        out_deps=rx.utils.Dependency(
                            event=etensor_mlp_add_rms,
                            dep=lambda i, j, k, notify_idx: (-1, 0),
                            num=1,
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
                            dep=lambda i, j, k, wait_idx: (-1, 0),
                            num=1,
                        ),
                        inverse_in_deps=rx.utils.Dependency(
                            event=etensor_mlp_add_rms,
                            dep=lambda rank, x, inv_idx, batch_size: (inv_idx, 0, 0),
                            num=lambda rank, x, batch_size: batch_size,
                            extra_args=[batch_size],
                        ),
                        out_deps=rx.utils.Dependency(
                            event=etensor_end,
                            dep=lambda i, j, k, notify_idx: (-1, 0),
                            num=1,
                        ),
                    )
                )
                
                # END
                _ = bb.emit(
                    R.call_tir_device(
                        end_gv,
                        [],
                        out_sinfo=[],
                        job_id=JobType.END.value,
                        tile_num=end_num_tiles,
                        in_deps=rx.utils.Dependency(
                            event=etensor_end,
                            dep=lambda i, j, k, wait_idx: (-1, 0),
                            num=1,
                        ),
                        inverse_in_deps=rx.utils.Dependency(
                            event=etensor_end,
                            dep=lambda rank, x, inv_idx: (inv_idx, 0, 0),
                            num=KernelConfig.SM_NUMBER,
                        ),
                    )
                )
                
                output = rx.Tuple([output_hidden_state, residual])
                # out = bb.emit_output(mlp_add_rmsnorm_packed)

            # output
                bb.emit_func_output(output)

        func_gv = bb.add_func(bb.get()[f"megakernel_blkm{blk_m}"], f"megakernel_blkm{blk_m}")

        return func_gv
    
    def get_mod(self, max_batch_size):
        bb = rx.BlockBuilder()
        
        inner_blkm32 = self._qwen3_layer_inner(bb, max_batch_size, blk_m=32)
        inner_blkm64 = self._qwen3_layer_inner(bb, max_batch_size, blk_m=64)
        inner_blkm128 = self._qwen3_layer_inner(bb, max_batch_size, blk_m=128)
        
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
        packed_events = rx.Var(
            "packed_events",
            rx.TupleStructInfo(
                [
                    rx.TensorStructInfo([ceildiv((self.NUM_ATTENTION_HEADS + 2 * self.NUM_KEY_VALUE_HEADS) * self.HEAD_DIM, FuseGemmTile.BLK_N)], "int32"),  # etensor_qkv_partial
                    rx.TensorStructInfo([KernelConfig.SM_NUMBER], "int32"),  # etensor_notify_attn
                    rx.TensorStructInfo([max_batch_size * self.NUM_KEY_VALUE_HEADS], "int32"),  # etensor_attn_merge
                    rx.TensorStructInfo([self.SPLIT_O_PROJECT], "int32"),  # etensor_o_proj
                    rx.TensorStructInfo([self.HIDDEN_SIZE // FuseGemmTile.BLK_N], "int32"),  # etensor_o_partial
                    rx.TensorStructInfo([max_batch_size], "int32"),  # etensor_attn_add_rmsnorm
                    rx.TensorStructInfo([1], "int32"),  # etensor_attn_mlp
                    rx.TensorStructInfo([self.DOWN_PROJ_SPLIT_K_FACTOR], "int32"),  # etensor_down_proj
                    rx.TensorStructInfo([self.HIDDEN_SIZE // FuseGemmTile.BLK_N], "int32"),  # etensor_down_proj_reduce
                    rx.TensorStructInfo([max_batch_size], "int32"),  # etensor_mlp_add_rmsnorm
                    rx.TensorStructInfo([1], "int32"),  # etensor_end
                ]
            ),
        )

        with bb.function("megakernel", [x, residual, packed_info, packed_weights, packed_events]):
            with bb.dataflow():
                runtime_batch_size = T.var("int64", name="runtime_batch_size")
                _ = bb.match_cast(R.shape_of(x), R.Shape([runtime_batch_size, self.HIDDEN_SIZE]))
                output_sinfo = [
                    rx.TensorStructInfo([batch_size, self.HIDDEN_SIZE], "float16"),
                    rx.TensorStructInfo([batch_size, self.HIDDEN_SIZE], "float32"),
                ]
                output_shape = R.Tuple(R.Tensor([batch_size, self.HIDDEN_SIZE], "float16"), R.Tensor([batch_size, self.HIDDEN_SIZE], "float32"))
                def bind_branch_with_r_func(r_func, blk_m):
                    out = rx.Var(f"out_blkm{blk_m}", output_shape)
                    bindings = [rx.VarBinding(out, rx.Call(r_func, [x, residual, packed_info, packed_weights, packed_events], sinfo_args=output_sinfo))]
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

def prepare_events(arg_dict, batch_size, max_batch_size, mk: MegaKernel, repeat=100):
    DEV = tvm.cuda()
    base = 1 << 16
    all_event_list = []
    attn_task_num = arg_dict["attn_task_num"].item()
    for _ in range(repeat):
        event_list = []
        
        # etensor_qkv_partial
        event_list.append(tvm.runtime.tensor(np.full(ceildiv((mk.NUM_ATTENTION_HEADS + 2 * mk.NUM_KEY_VALUE_HEADS) * mk.HEAD_DIM, FuseGemmTile.BLK_N), mk.SPLIT_QKV_PROJECT * (base + 1), dtype=np.int32), device=DEV))
        
        # etensor_notify_attn
        etensor_notify_attn = np.zeros((KernelConfig.SM_NUMBER), dtype=np.int32)
        attn_tile_num = ceildiv(attn_task_num, KernelConfig.WG_NUMBER)
        unit = KernelConfig.WG_NUMBER * (2 + mk.NUM_ATTENTION_HEADS // mk.NUM_KEY_VALUE_HEADS)
        num = unit * (attn_tile_num // KernelConfig.SM_NUMBER)
        remain = attn_tile_num % KernelConfig.SM_NUMBER
        for m in range(KernelConfig.SM_NUMBER):
            if m < remain:
                etensor_notify_attn[m] = num + unit
            else:
                etensor_notify_attn[m] = num
        etensor_notify_attn *= (base + 1)
        etensor_notify_attn = tvm.runtime.tensor(etensor_notify_attn, device=DEV)
        event_list.append(etensor_notify_attn)
        
        # etensor_attn_merge
        event_list.append(tvm.runtime.tensor(np.full((max_batch_size * mk.NUM_KEY_VALUE_HEADS), min(KernelConfig.SM_NUMBER, attn_tile_num) * (base + 1), dtype=np.int32), device=DEV))
        
        # etensor_o_proj
        etensor_o_proj = np.zeros(mk.SPLIT_O_PROJECT, dtype=np.int32)
        o_proj_tile_k = ceildiv(ceildiv(mk.NUM_ATTENTION_HEADS * mk.HEAD_DIM, mk.SPLIT_O_PROJECT), FuseGemmTile.BLK_K) * FuseGemmTile.BLK_K
        if attn_task_num > mk.NUM_KEY_VALUE_HEADS * batch_size:
            # to simply, assume that one merge tile will not use two kv head
            assert KernelConfig.WG_NUMBER * KernelConfig.WARP_NUMBER <= mk.NUM_ATTENTION_HEADS // mk.NUM_KEY_VALUE_HEADS
            for m in range(ceildiv(batch_size * mk.NUM_ATTENTION_HEADS, KernelConfig.WARP_NUMBER * KernelConfig.WG_NUMBER)):
                worker_id = m * KernelConfig.WG_NUMBER * KernelConfig.WARP_NUMBER
                kv_idx = worker_id // (batch_size * (mk.NUM_ATTENTION_HEADS // mk.NUM_KEY_VALUE_HEADS))
                qo_idx = worker_id % (mk.NUM_ATTENTION_HEADS // mk.NUM_KEY_VALUE_HEADS)
                range_start = (kv_idx * (mk.NUM_ATTENTION_HEADS // mk.NUM_KEY_VALUE_HEADS) + qo_idx) * mk.HEAD_DIM // o_proj_tile_k
                range_end = ((kv_idx * (mk.NUM_ATTENTION_HEADS // mk.NUM_KEY_VALUE_HEADS) + qo_idx + KernelConfig.WG_NUMBER * KernelConfig.WARP_NUMBER) * mk.HEAD_DIM - 1) // o_proj_tile_k
                for i in range(range_start, range_end + 1):
                    etensor_o_proj[i] += 1
        else:
            etensor_o_proj += min(KernelConfig.SM_NUMBER, attn_tile_num)
        etensor_o_proj *= (base + 1)
        etensor_o_proj = tvm.runtime.tensor(etensor_o_proj, device=DEV)
        event_list.append(etensor_o_proj)
        
        # etensor_attn_o_partial
        event_list.append(tvm.runtime.tensor(np.full(mk.HIDDEN_SIZE // FuseGemmTile.BLK_N, mk.SPLIT_O_PROJECT * (base + 1), dtype=np.int32), device=DEV))
        
        # etensor_attn_add_rmsnorm
        event_list.append(tvm.runtime.tensor(np.full(max_batch_size, mk.SPLIT_O_PROJECT * mk.HIDDEN_SIZE // FuseSplitKReduceTile.N_UNIT * (base + 1), dtype=np.int32), device=DEV))
        
        # etensor_attn_mlp
        event_list.append(tvm.runtime.tensor(np.full(1, batch_size * (base + 1), dtype=np.int32), device=DEV))
        
        # etensor_down_proj
        etensor_down_proj = np.zeros(mk.DOWN_PROJ_SPLIT_K_FACTOR, dtype=np.int32)
        down_proj_tile_k = (ceildiv(ceildiv(mk.INTERMEDIATE_SIZE, mk.DOWN_PROJ_SPLIT_K_FACTOR), FuseGemmTile.BLK_K) * FuseGemmTile.BLK_K)
        for m in range(mk.INTERMEDIATE_SIZE * 2 // FuseGateUpSiluTile.BLK_N):
            range_start = m * FuseGateUpSiluTile.BLK_N // 2 // down_proj_tile_k
            range_end = ((m + 1) * FuseGateUpSiluTile.BLK_N // 2 - 1) // down_proj_tile_k
            for i in range(range_start, range_end + 1):
                etensor_down_proj[i] += 1
        etensor_down_proj *= (base + 1)
        etensor_down_proj = tvm.runtime.tensor(etensor_down_proj, device=DEV)
        event_list.append(etensor_down_proj)
          
        # etensor_down_proj_reduce
        event_list.append(tvm.runtime.tensor(np.full(mk.HIDDEN_SIZE // FuseGemmTile.BLK_N, mk.DOWN_PROJ_SPLIT_K_FACTOR * (base + 1), dtype=np.int32), device=DEV))
        
        # etensor_mlp_add_rmsnorm
        event_list.append(tvm.runtime.tensor(np.full(max_batch_size, mk.DOWN_PROJ_SPLIT_K_FACTOR * mk.HIDDEN_SIZE // FuseSplitKReduceTile.N_UNIT * (base + 1), dtype=np.int32), device=DEV))
        
        # etensor_end
        event_list.append(tvm.runtime.tensor(np.array([batch_size * (base + 1)], dtype=np.int32), device=DEV))
        
        all_event_list.append(event_list)
    return all_event_list


@tvm.testing.requires_cuda_compute_version(10, exact=True)
@pytest.mark.skip
def test(batch_size, seq_len, vm, mega_kernel_wrapper):
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
        events = prepare_events(arg_dict, batch_size, max_batch_size=128, mk=mk)
        iter = 0
        def func():
            nonlocal iter
            iter += 1
            residual = tvm.runtime.tensor(arg_dict["residual"].to(torch.float32), device=dev)
            res = vm["megakernel"](hidden_state, residual, packed_info, weights, events[iter])
            return res
        res = bench(func, warmup=3, repeat=10, proton_name="tir")
        res = func()
        # export_to_perfetto_trace(res, "blackwell_attn.json")
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
        mod = megakernel_wrapper.get_mod(max_batch_size=128)
        mod = rx.transform.StaticHorizontalFusion(
            ["megakernel_blkm32", "megakernel_blkm64", "megakernel_blkm128"], 
            strategy=scheduler, tile_scheduler_class=tile_scheduler_class_map[scheduler], semaphore_class=semaphore_class_map[scheduler]
        )(mod)
        # mod.show()
        ex = rx.build(mod, target="cuda", tir_pipeline="tirp")
        src = ex.mod.imports[0].imports[0].inspect_source()
        print(src)
        vm = rx.VirtualMachine(ex, tvm.cuda())
        for batch_size in args.batch_size:
            print(f"batch_size: {batch_size}", flush=True)
            for seq_len in args.seq_len:
                print(f"seq_len: {seq_len}", flush=True)
                test(batch_size, seq_len, vm, megakernel_wrapper)

