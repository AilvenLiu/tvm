from typing import List
from pathlib import Path
import numpy as np
import tvm
from tqdm import tqdm

from tvm import dlight, relax, te, tir, target
from tvm.contrib import tvmjs
from tvm.relax.frontend import nn
from tvm.relax import register_pipeline
from mlc_llm.model.qwen3.qwen3_model import Qwen3Config, Qwen3LMHeadModel
from mlc_llm.nn.kv_cache import PagedKVCache
from mlc_llm.compiler_pass.dispatch_kv_cache_creation import DispatchKVCacheCreation
from mlc_llm.compiler_pass.blas_dispatch import BLASDispatch
from mlc_llm.compiler_pass.fuse_add_norm import FuseAddRMSNorm
from mlc_llm.compiler_pass.pipeline import _DebugDump
from tvm.runtime import ShapeTuple

from tvm.script import relax as R
from tvm.script import ir as I
from tvm.script import tir as T
from .test_rmsnorm import get_rmsnorm_kernel
from .test_hgemm_1consumer_1cta_swap_splitk import get_hgemm_kernel
from .test_rope import get_cos_sin_cache_kernel
from ..megakernel.common import ceildiv
from ..megakernel.gemm import GemmTile
from ..megakernel.gemm_splitk_reduce import SplitKReduceTile
from ..megakernel.test_layer import MegaKernel, generate_exec_queue, generate_event_tensor

dev = tvm.cuda()
target = tvm.target.Target("cuda")

config = Qwen3Config(
    hidden_act="silu",
    hidden_size=5120,
    intermediate_size=25600,
    attention_bias=False,
    num_attention_heads=64,
    num_hidden_layers=1,  # 64,
    num_key_value_heads=8,
    rms_norm_eps=1e-06,
    rope_theta=1000000,
    vocab_size=151936,
    tie_word_embeddings=False,
    context_window_size=40960,
    prefill_chunk_size=2048,
    tensor_parallel_shards=1,
    head_dim=128,
    dtype="float32",
    max_batch_size=128,
    weight_block_size=None,
    kwargs={},
)

problem_config = {
    "vocab_size": config.vocab_size,
    "hidden_size": config.hidden_size,
    "intermediate_size": config.intermediate_size,
    "num_hidden_layers": config.num_hidden_layers,
    "num_attention_heads": config.num_attention_heads,
    "num_key_value_heads": config.num_key_value_heads,
    "head_dim": config.head_dim,
    "rms_norm_eps": config.rms_norm_eps,
    "rope_theta": config.rope_theta,
    "page_size": 16,
}

SPLIT_QKV_PROJECT = 3
SPLIT_O_PROJRCT = 3
DOWN_PROJ_SPLIT_K_FACTOR = 10


def get_default_spec(model):
    mod_spec = {
        "embed": {
            "input_ids": nn.spec.Tensor(["seq_len"], "int32"),
            "$": {
                "param_mode": "packed",
                "effect_mode": "none",
            },
        },
        "batch_decode": {
            "input_embeds": nn.spec.Tensor(["batch_size", 1, model.hidden_size], model.dtype),
            "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
            "$": {
                "param_mode": "packed",
                "effect_mode": "none",
            },
        },
        "create_paged_kv_cache": {
            "max_batch_size": int,
            "max_total_seq_len": int,
            "prefill_chunk_size": int,
            "page_size": int,
            "support_sliding_window": int,
            "$": {
                "param_mode": "none",
                "effect_mode": "none",
            },
        },
    }
    return nn.spec.ModuleSpec.from_raw(mod_spec, model)


def _craft_pipeline(ext_mods: List[nn.ExternModule], dump_file_prefix: Path):
    ext_mods = ext_mods or []
    debug_dir = Path("/home/guanjiew/qwen3-mg-debug")

    @tvm.transform.module_pass(opt_level=0)
    def _pipeline(mod: tvm.ir.IRModule, _ctx: tvm.transform.PassContext) -> tvm.ir.IRModule:
        seq = tvm.transform.Sequential(
            [
                _DebugDump(f"{dump_file_prefix}-phase0.py", debug_dir, show_meta=False),
                tvm.tir.transform.BindTarget(target),
                DispatchKVCacheCreation(target, True, dict()),
                BLASDispatch(target),
                FuseAddRMSNorm(target=target),
                # Phase 1. Passes on high-level operator graph
                # We can enable cublas for further optimization
                relax.transform.FuseTransposeMatmul(),
                # Phase 2. Lowering to TIR, inherited TVM Relax's official "zero" pipeline
                relax.transform.LegalizeOps(),
                relax.transform.AnnotateTIROpPattern(),
                relax.transform.FoldConstant(),
                relax.transform.FuseOps(),
                relax.transform.FuseTIR(),
                # Phase 3. Passes on TIR
                relax.transform.DeadCodeElimination(),
                _DebugDump(f"{dump_file_prefix}-phase1.py", debug_dir, show_meta=False),
                # Phase 4. Low-level Optimizations
                dlight.ApplyDefaultSchedule(
                    dlight.gpu.Matmul(),
                    dlight.gpu.GEMV(),
                    dlight.gpu.Reduction(),
                    dlight.gpu.GeneralReduction(),
                    dlight.gpu.Fallback(),
                ),
                # Phase 5. Lowering to VM bytecode
                relax.transform.RewriteDataflowReshape(),
                relax.transform.ToNonDataflow(),
                relax.transform.RemovePurityChecking(),
                relax.transform.CallTIRRewrite(),
                relax.transform.StaticPlanBlockMemory(),
                _DebugDump(f"{dump_file_prefix}-phase2.py", debug_dir, show_meta=False),
                relax.transform.RewriteCUDAGraph(),
                relax.transform.LowerAllocTensor(),
                relax.transform.KillAfterLastUse(),
                relax.transform.LowerRuntimeBuiltin(),
                relax.transform.VMShapeLower(),
                relax.transform.AttachGlobalSymbol(),
                relax.transform.AttachExternModules(ext_mods),
            ]
        )
        mod = seq(mod)
        return mod

    return _pipeline


@register_pipeline("opt_llm_1")
def _pipeline(  # pylint: disable=too-many-arguments
    ext_mods: List[nn.ExternModule] = None,
):
    return _craft_pipeline(ext_mods, "opt_llm_1")


@register_pipeline("opt_llm_2")
def _pipeline(  # pylint: disable=too-many-arguments
    ext_mods: List[nn.ExternModule] = None,
):
    return _craft_pipeline(ext_mods, "opt_llm_2")


@register_pipeline("opt_llm_mg")
def _pipeline(  # pylint: disable=too-many-arguments
    ext_mods: List[nn.ExternModule] = None,
):
    return _craft_pipeline(ext_mods, "opt_llm_mg")


tvm.register_func("megakernel.generate_exec_queue", generate_exec_queue)
tvm.register_func("megakernel.generate_event_tensor", generate_event_tensor)


def get_params(named_params):
    import torch
    import torch.nn as nn

    torch.manual_seed(0)
    result = list()
    for k, param in tqdm(named_params):
        torch_tensor = torch.empty(tuple(param.shape)).to(getattr(torch, param.dtype)).to("cuda")
        if k.endswith("norm.weight"):
            nn.init.ones_(torch_tensor)
        elif k.endswith("embed_tokens.weight"):
            nn.init.normal_(torch_tensor, mean=0.0, std=0.02)
        else:
            nn.init.xavier_uniform_(torch_tensor, gain=1.0)
        torch_tensor = torch_tensor
        result.append(tvm.runtime.ndarray.from_dlpack(torch.to_dlpack(torch_tensor)))
    return result


def sample_token(logits):
    return np.argmax(logits, axis=-1)


model = Qwen3LMHeadModel(config)
model.to("float16")
mod, named_params = model.export_tvm(get_default_spec(model))
params = get_params(named_params)
with target:
    ex = tvm.compile(mod, target, relax_pipeline=relax.get_pipeline("opt_llm_1"))
    vm = relax.VirtualMachine(ex, dev)

MAX_BATCH_SIZE = 128
MAX_SEQ_LEN = 4096
MAX_TOTAL_SEQ_LEN = MAX_BATCH_SIZE * MAX_SEQ_LEN
PAGE_SIZE = 16
ROPE_THETA = 1000000


def test_qwen3_layer(batch_decode_func, is_megakernel=False, cos_sin_cache_func=None):
    kv_cache = vm["create_flashinfer_paged_kv_cache"](
        ShapeTuple([MAX_BATCH_SIZE]),  # max_batch_size
        ShapeTuple([MAX_TOTAL_SEQ_LEN]),  # max_total_seq_len
        ShapeTuple([MAX_SEQ_LEN]),  # prefill_chunk_size
        ShapeTuple([PAGE_SIZE]),  # page_size
        ShapeTuple([0]),  # support_sliding_window
    )

    if is_megakernel:
        cos_sin_cache = tvm.nd.array(np.zeros((MAX_SEQ_LEN, 128), dtype="float32"), device=dev)
        cos_sin_cache_func(cos_sin_cache)

    nd_view_func = tvm.get_global_func("vm.builtin.reshape")

    def embed(tokens, params):
        _embed = vm["embed"](tokens, params)
        _embed = nd_view_func(_embed, ShapeTuple([_embed.shape[0], 1, _embed.shape[1]]))
        return _embed

    add_sequence_func = tvm.get_global_func("vm.builtin.kv_state_add_sequence")
    begin_forward_func = tvm.get_global_func("vm.builtin.kv_state_begin_forward")
    end_forward_func = tvm.get_global_func("vm.builtin.kv_state_end_forward")

    batch_size = 32
    seq_len = 16
    seq_ids = []

    for i in range(batch_size):
        seq_ids.append(i)
        add_sequence_func(kv_cache, i)

    np.random.seed(0)
    logits_arr = list()
    last_tokens = np.random.randint(0, 100, size=(batch_size,))
    for i in tqdm(range(seq_len)):
        tokens = tvm.nd.array(last_tokens.astype("int32"), device=dev)
        hidden_states = embed(tokens, params)
        begin_forward_func(kv_cache, ShapeTuple(seq_ids), ShapeTuple([1] * batch_size))
        if is_megakernel:
            logits = batch_decode_func(
                hidden_states,
                kv_cache,
                cos_sin_cache,
                params,
            )
        else:
            logits, kv_cache = batch_decode_func(hidden_states, kv_cache, params)

        end_forward_func(kv_cache)
        last_tokens = sample_token(logits.numpy()).flatten()
        logits_arr.append(logits)

    return logits_arr


# res0 = test_qwen3_layer(vm["batch_decode"])


def attach_attr(func, name):
    func = func.without_attr("global_symbol")
    func = func.with_attr("global_symbol", name)
    func = func.with_attr("tir.is_scheduled", True)
    func = func.with_attr("tir.noalias", True)
    return func


def get_qwen3_megakernel_mod():
    rms_norm = get_rmsnorm_kernel(5120)
    layer_kernel = MegaKernel(problem_config).get_func()
    hgemm, reduce, tile_k_num = get_hgemm_kernel(dim_n=151936, dim_k=5120)
    cos_sin_cache = get_cos_sin_cache_kernel(128, ROPE_THETA)

    # fmt: off
    @I.ir_module
    class Module:
        @T.prim_func(private=True)
        def rms_norm(input_ptr: T.handle, weight_ptr: T.handle, out_ptr: T.handle):
            pass

        @T.prim_func(private=True)
        def cast(var_lv4: T.handle, var_compute: T.handle):
            T.func_attr({"op_pattern": 0, "tir.noalias": True})
            batch_size = T.int64()
            lv4 = T.match_buffer(var_lv4, (batch_size, T.int64(1), T.int64(151936)), "float16")
            compute = T.match_buffer(var_compute, (batch_size, T.int64(1), T.int64(151936)))
            # with T.block("root"):
            for i0, i1, i2 in T.grid(batch_size, T.int64(1), T.int64(151936)):
                with T.block("compute"):
                    v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                    T.reads(lv4[v_i0, v_i1, v_i2])
                    T.writes(compute[v_i0, v_i1, v_i2])
                    compute[v_i0, v_i1, v_i2] = T.Cast("float32", lv4[v_i0, v_i1, v_i2])

        @T.prim_func(private=True)
        def hgemm(A_ptr: T.handle, b_ptr: T.handle, partial_sum_ptr: T.handle):
            pass
        
        @T.prim_func(private=True)
        def reduce(partial_sum_ptr: T.handle, D_ptr: T.handle):
            pass

        @T.prim_func(private=True)
        def cos_sin_cache(cos_sin_cache: T.handle):
            pass
        
        @R.function
        def cos_sin_cache_func(cache: R.Tensor((MAX_SEQ_LEN, 128), dtype="float32")):
            cls = Module
            with R.dataflow():
                cache: R.Tensor((MAX_SEQ_LEN, 128), dtype="float32") = R.call_tir_inplace(cls.cos_sin_cache, (cache), [0], out_sinfo=R.Tensor((MAX_SEQ_LEN, 128), dtype="float32") )
                R.output(cache)
            return cache
        
        @T.prim_func(private=True)
        def layer_kernel(
                # input and output
                hidden_state_ptr: T.handle, # input: read-only
                residual_ptr: T.handle, # input & output: inplace update
                output_ptr: T.handle, # output

                # weight
                qkv_proj_weight_ptr: T.handle, # read-only
                o_proj_weight_ptr: T.handle, # read-only
                q_rms_weight_ptr: T.handle, # read-only
                k_rms_weight_ptr: T.handle, # read-only
                gate_up_weight_ptr: T.handle, # read-only
                down_weight_ptr: T.handle, # read-only
                attn_add_rms_weight_ptr: T.handle, # read-only
                mlp_add_rms_weight_ptr: T.handle, # read-only

                # page cache, cos_sin cache and plan info
                cos_sin_cache_ptr: T.handle, # read-only
                rope_pos_ptr: T.handle, # read-only
                kv_cache_ptr: T.handle, # inplace update
                kv_indptr_ptr: T.handle, # read-only
                kv_indices_ptr: T.handle, # read-only
                kv_last_page_len_ptr: T.handle, # read-only
                append_pos_ptr: T.handle, # read-only
                request_indices_ptr: T.handle, # read-only
                kv_tile_indices_ptr: T.handle, # read-only
                max_chunk_size_ptr: T.handle, # read-only
                o_indptr_ptr: T.handle, # read-only

                # intermediate buffer
                partital_qkv_ptr: T.handle, # intermediate
                qkv_ptr: T.handle,  # intermediate
                o_ptr: T.handle, # intermediate
                lse_ptr: T.handle, # intermediate
                o_tmp_ptr: T.handle, # intermediate
                lse_tmp_ptr: T.handle, # intermediate
                partial_o_ptr: T.handle, # intermediate
                hidden_state_attn_mlp_ptr: T.handle, # intermediate
                out_gate_up_proj_ptr: T.handle, # intermediate
                out_silu_multiply_ptr: T.handle, # intermediate
                partial_sum_down_proj_ptr: T.handle, # intermediate
                
                # event tensor
                etensor_qkv_partial_ptr: T.handle, 
                etensor_q_reduce_ptr: T.handle, 
                etensor_k_reduce_ptr: T.handle, 
                etensor_v_reduce_ptr: T.handle, 
                etensor_rms_rope_ptr: T.handle, 
                etensor_q_rope_decode_ptr: T.handle, 
                etensor_k_rope_append_ptr: T.handle, 
                etensor_append_decode_ptr: T.handle, 
                etensor_decode_merge_ptr: T.handle,
                etensor_o_proj_ptr: T.handle, 
                etensor_o_partial_ptr: T.handle, 
                etensor_attn_add_rms_ptr: T.handle,
                etensor_attn_mlp_ptr: T.handle,
                etensor_gate_up_proj_ptr: T.handle,
                etensor_down_proj_ptr: T.handle,
                etensor_down_proj_reduce_ptr: T.handle,
                etensor_mlp_add_rms_ptr: T.handle,

                # execution queue
                exec_queue_ptr: T.handle,
        ):
            pass

        @R.function
        def batch_decode(
            input_embeds: R.Tensor(("batch_size", 1, 5120), dtype="float16"),
            paged_kv_cache: R.Object,
            cos_sin_cache: R.Tensor((MAX_SEQ_LEN, 128), dtype="float32"),
            packed_params: R.Tuple(
                R.Tensor((151936, 5120), dtype="float16"),
                R.Tensor((10240, 5120), dtype="float16"),
                R.Tensor((5120, 8192), dtype="float16"),
                R.Tensor((128,), dtype="float16"),
                R.Tensor((128,), dtype="float16"),
                R.Tensor((51200, 5120), dtype="float16"),
                R.Tensor((5120, 25600), dtype="float16"),
                R.Tensor((5120,), dtype="float16"),
                R.Tensor((5120,), dtype="float16"),
                R.Tensor((5120,), dtype="float16"),
                R.Tensor((151936, 5120), dtype="float16"),
            ),
        ):    
            batch_size = T.int64()
            new_batch_size = T.int64()
            total_page_num = T.int64()
            max_page_num = T.int64()
            page_size = T.int64()
            
            R.func_attr({"num_input": 2})
            cls = Module
            with R.dataflow():
                res0 = R.call_pure_packed(
                    "vm.builtin.paged_attention_kv_cache_tensor_retrieve",
                    paged_kv_cache, R.prim_value(1),
                    sinfo_args=[
                        R.Tuple([R.Tensor(None, dtype="float16")]),
                        R.Tensor((batch_size + 1,), dtype="int32"),
                        R.Tensor((batch_size + 1,), dtype="int32"),
                        R.Tensor(None, dtype="int32"),
                        R.Tensor((batch_size,), dtype="int32"),
                        R.Tensor((batch_size,), dtype="int32"),
                        R.Tensor((batch_size,), dtype="int32"),
                        R.Tensor(None, dtype="uint8"),
                        R.Tensor(None, dtype="uint8"),
                        R.Tensor(None, dtype="uint8"),
                    ],
                )
                kv_data_, kv_indptr, kv_indptr_host, kv_indices_, kv_last_page_len, append_pos, rope_pos, temp_float_attn_workspace, temp_int_attn_workspace, temp_int_pinned_attn_workspace = (
                    res0[0],
                    res0[1],
                    res0[2],
                    res0[3],
                    res0[4],
                    res0[5],
                    res0[6],
                    res0[7],
                    res0[8],
                    res0[9],
                )
                kv_data = R.match_cast(kv_data_, R.Tuple([R.Tensor((max_page_num, 2, 8, page_size, 128), dtype="float16")]))
                kv_indices = R.match_cast(kv_indices_, R.Tensor((total_page_num,), dtype="int32"))
                res1 = R.call_pure_packed(
                    "flashinfer.batch_decode_with_paged_kv_cache_plan",
                    temp_float_attn_workspace,
                    temp_int_attn_workspace,
                    temp_int_pinned_attn_workspace,
                    kv_indptr_host,
                    R.prim_value(batch_size),
                    R.prim_value(64),
                    R.prim_value(8),
                    R.prim_value(page_size),
                    R.prim_value(False),
                    R.prim_value(0),
                    R.prim_value(-1),
                    R.prim_value(128),
                    R.prim_value(128),
                    R.dtype("float16"),
                    R.dtype("float16"),
                    R.prim_value(True), # TODO: remove this flag
                    sinfo_args=[
                        R.Tensor(None, dtype="int32"),
                        R.Tensor(None, dtype="int32"),
                        R.Tensor((1,), dtype="int32"),
                        R.Tensor(None, dtype="int32"),
                    ],
                )
                plan_request_indices_, plan_kv_tile_indices_, plan_max_chunk_size, plan_o_indptr_ = res1[0], res1[1], res1[2], res1[3]
                plan_request_indices = R.match_cast(plan_request_indices_, R.Tensor((new_batch_size,), dtype="int32"))
                plan_kv_tile_indices = R.match_cast(plan_kv_tile_indices_, R.Tensor((new_batch_size,), dtype="int32"))
                plan_o_indptr = R.match_cast(plan_o_indptr_, R.Tensor((new_batch_size + 1,), dtype="int32"))

                res2 = R.call_pure_packed(
                    "megakernel.generate_exec_queue",
                    R.prim_value(batch_size),
                    R.prim_value(new_batch_size),
                    sinfo_args=[
                        R.Tensor((144, 128, 4), dtype="int32"),
                    ]
                )
                exec_queue = res2

                res3 = R.call_pure_packed(
                    "megakernel.generate_event_tensor",
                    R.prim_value(batch_size),
                    plan_o_indptr,
                    sinfo_args=[
                        R.Tensor((ceildiv((config.num_attention_heads + 2 * config.num_key_value_heads) * config.head_dim, SplitKReduceTile.N_UNIT),), dtype="int32"),
                        R.Tensor(None, dtype="int32"),
                        R.Tensor(None, dtype="int32"),
                        R.Tensor(None, dtype="int32"),
                        R.Tensor(None, dtype="int32"),
                        R.Tensor(None, dtype="int32"),
                        R.Tensor(None, dtype="int32"),
                        R.Tensor(None, dtype="int32"),
                        R.Tensor(None, dtype="int32"),
                        R.Tensor((SPLIT_O_PROJRCT,), dtype="int32"),
                        R.Tensor((ceildiv(config.hidden_size, GemmTile.BLK_N),), dtype="int32"),
                        R.Tensor(None, dtype="int32"),
                        R.Tensor((1,), dtype="int32"),
                        R.Tensor((config.intermediate_size // GemmTile.BLK_N,), dtype="int32"),
                        R.Tensor((DOWN_PROJ_SPLIT_K_FACTOR,), dtype="int32"),
                        R.Tensor((config.hidden_size // GemmTile.BLK_N,), dtype="int32"),
                        R.Tensor(None, dtype="int32"),
                    ]
                )
                (
                    etensor_qkv_partial,
                    etensor_q_reduce_,
                    etensor_k_reduce_,
                    etensor_v_reduce_,
                    etensor_rms_rope_,
                    etensor_q_rope_decode_,
                    etensor_k_rope_append_,
                    etensor_append_decode_,
                    etensor_decode_merge_,
                    etensor_o_proj,
                    etensor_o_partial,
                    etensor_attn_add_rms_norm_,
                    etensor_attn_mlp,
                    etensor_gate_up_proj,
                    etensor_down_proj,
                    etensor_down_proj_reduce,
                    etensor_mlp_add_rms_norm_
                ) = (
                    res3[0],
                    res3[1],
                    res3[2],
                    res3[3],
                    res3[4],
                    res3[5],
                    res3[6],
                    res3[7],
                    res3[8],
                    res3[9],
                    res3[10],
                    res3[11],
                    res3[12],
                    res3[13],
                    res3[14],
                    res3[15],
                    res3[16],
                )

                etensor_q_reduce = R.match_cast(etensor_q_reduce_, R.Tensor((batch_size, ceildiv(config.num_attention_heads * config.head_dim, SplitKReduceTile.N_UNIT)), dtype="int32"))
                etensor_k_reduce = R.match_cast(etensor_k_reduce_, R.Tensor((batch_size, ceildiv(config.num_key_value_heads * config.head_dim, SplitKReduceTile.N_UNIT)), dtype="int32"))
                etensor_v_reduce = R.match_cast(etensor_v_reduce_, R.Tensor((batch_size, ceildiv(config.num_key_value_heads * config.head_dim, SplitKReduceTile.N_UNIT)), dtype="int32"))
                etensor_rms_rope = R.match_cast(etensor_rms_rope_, R.Tensor((batch_size, config.num_attention_heads + config.num_key_value_heads), dtype="int32"))
                etensor_q_rope_decode = R.match_cast(etensor_q_rope_decode_, R.Tensor((batch_size, config.num_key_value_heads), dtype="int32"))
                etensor_k_rope_append = R.match_cast(etensor_k_rope_append_, R.Tensor((batch_size, config.num_key_value_heads), dtype="int32"))
                etensor_append_decode = R.match_cast(etensor_append_decode_, R.Tensor((batch_size, config.num_key_value_heads), dtype="int32"))
                etensor_decode_merge = R.match_cast(etensor_decode_merge_, R.Tensor((batch_size, config.num_key_value_heads), dtype="int32"))
                etensor_attn_add_rms_norm = R.match_cast(etensor_attn_add_rms_norm_, R.Tensor((batch_size,), dtype="int32"))
                etensor_mlp_add_rms_norm = R.match_cast(etensor_mlp_add_rms_norm_, R.Tensor((batch_size,), dtype="int32"))

                model_layers_0_self_attn_c_attn_weight1: R.Tensor((10240, 5120), dtype="float16") = packed_params[1]
                model_layers_0_self_attn_o_proj_weight1: R.Tensor((5120, 8192), dtype="float16") = packed_params[2]
                model_layers_0_self_attn_q_norm_weight1: R.Tensor((128,), dtype="float16") = packed_params[3]
                model_layers_0_self_attn_k_norm_weight1: R.Tensor((128,), dtype="float16") = packed_params[4]
                model_layers_0_mlp_gate_up_proj_weight1: R.Tensor((51200, 5120), dtype="float16") = packed_params[5]
                model_layers_0_mlp_down_proj_weight1: R.Tensor((5120, 25600), dtype="float16") = packed_params[6]
                model_layers_0_input_layernorm_weight1: R.Tensor((5120,), dtype="float16") = packed_params[7]
                model_layers_0_post_attention_layernorm_weight1: R.Tensor((5120,), dtype="float16") = packed_params[8]
                model_norm_weight1: R.Tensor((5120,), dtype="float16") = packed_params[9]
                lm_head_weight1: R.Tensor((151936, 5120), dtype="float16") = packed_params[10]

                partital_qkv = R.builtin.alloc_tensor(R.shape([SPLIT_QKV_PROJECT, batch_size, 10240]), dtype="float32", runtime_device_index=0)
                qkv = R.builtin.alloc_tensor(R.shape([batch_size, 80, 128]), dtype="float16", runtime_device_index=0)
                o = R.builtin.alloc_tensor(R.shape([batch_size, 64, 128]), dtype="float16", runtime_device_index=0)
                lse = R.builtin.alloc_tensor(R.shape([batch_size, 64]), dtype="float32", runtime_device_index=0)
                o_tmp = R.builtin.alloc_tensor(R.shape([batch_size, 64, 128]), dtype="float32", runtime_device_index=0)
                lse_tmp = R.builtin.alloc_tensor(R.shape([batch_size, 64]), dtype="float32", runtime_device_index=0)
                partial_o = R.builtin.alloc_tensor(R.shape([SPLIT_O_PROJRCT, batch_size, 5120]), dtype="float32", runtime_device_index=0)
                hidden_state_attn_mlp = R.builtin.alloc_tensor(R.shape([batch_size, 5120]), dtype="float16", runtime_device_index=0)
                out_gate_up_proj = R.builtin.alloc_tensor(R.shape([batch_size, 51200]), dtype="float16", runtime_device_index=0)
                out_silu_multiply = R.builtin.alloc_tensor(R.shape([batch_size, 25600]), dtype="float16", runtime_device_index=0)
                partial_sum_down_proj = R.builtin.alloc_tensor(R.shape([DOWN_PROJ_SPLIT_K_FACTOR, batch_size, 5120]), dtype="float32", runtime_device_index=0)

                output_tensor = R.builtin.alloc_tensor(R.shape([batch_size, 5120]), dtype="float16", runtime_device_index=0)

                rs0 = R.reshape(input_embeds, (batch_size, 5120))
                rms_norm = R.call_tir(cls.rms_norm, (rs0, model_layers_0_input_layernorm_weight1), out_sinfo=R.Tensor((batch_size, 5120), dtype="float16"))
                layer_res = R.call_tir_inplace(cls.layer_kernel, (
                                            # input and output
                                            rms_norm, rs0, output_tensor,
                                            # weights
                                            model_layers_0_self_attn_c_attn_weight1, model_layers_0_self_attn_o_proj_weight1, 
                                            model_layers_0_self_attn_q_norm_weight1, model_layers_0_self_attn_k_norm_weight1,
                                            model_layers_0_mlp_gate_up_proj_weight1, model_layers_0_mlp_down_proj_weight1,
                                            model_layers_0_post_attention_layernorm_weight1, model_norm_weight1,
                                            # page cache, cos_sin cache and plan info
                                            cos_sin_cache, rope_pos, kv_data[0], kv_indptr, kv_indices, kv_last_page_len, 
                                            append_pos, plan_request_indices, plan_kv_tile_indices, plan_max_chunk_size, plan_o_indptr,
                                            # intermediate buffer
                                            partital_qkv, qkv, o, lse, o_tmp, lse_tmp, partial_o, hidden_state_attn_mlp,
                                            out_gate_up_proj, out_silu_multiply, partial_sum_down_proj,
                                            # event tensor
                                            etensor_qkv_partial, etensor_q_reduce, etensor_k_reduce, etensor_v_reduce, etensor_rms_rope,
                                            etensor_q_rope_decode, etensor_k_rope_append, etensor_append_decode, etensor_decode_merge, etensor_o_proj,
                                            etensor_o_partial, etensor_attn_add_rms_norm, etensor_attn_mlp, etensor_gate_up_proj, etensor_down_proj,
                                            etensor_down_proj_reduce, etensor_mlp_add_rms_norm,
                                            # execution queue
                                            exec_queue),
                                            [1, 2, 13],
                                            out_sinfo=[
                                                R.Tensor((batch_size, 5120), dtype="float16"), # residual
                                                R.Tensor((batch_size, 5120), dtype="float16"), # output
                                                R.Tensor((max_page_num, 2, 8, page_size, 128), dtype="float16"), # kv_cache
                                            ]
                )
                output: R.Tensor((batch_size, 5120), dtype="float16") = layer_res[1]

                rs = R.call_tir(cls.hgemm, (output, lm_head_weight1), out_sinfo=R.Tensor((tile_k_num, batch_size, 151936), dtype="float32"))
                lv = R.call_tir(cls.reduce, (rs,), out_sinfo=R.Tensor((batch_size, 151936), dtype="float16"))

                lv_rs = R.reshape(lv, (batch_size, 1, 151936))
                astype = R.call_tir(cls.cast, (lv_rs,), out_sinfo=R.Tensor((batch_size, 1, 151936), dtype="float32"))
                R.output(astype)
            return astype
    # fmt: on
    mod = Module

    mod.update_func(mod.get_global_var("rms_norm"), attach_attr(rms_norm, "rms_norm"))
    mod.update_func(mod.get_global_var("layer_kernel"), attach_attr(layer_kernel, "layer_kernel"))
    mod.update_func(mod.get_global_var("hgemm"), attach_attr(hgemm, "hgemm"))
    mod.update_func(mod.get_global_var("reduce"), attach_attr(reduce, "reduce"))
    mod.update_func(
        mod.get_global_var("cos_sin_cache"), attach_attr(cos_sin_cache, "cos_sin_cache")
    )
    return mod


def get_qwen3_megakernel_batch_decode_func():
    mg_model = get_qwen3_megakernel_mod()

    with target:
        ex = tvm.compile(
            mg_model, target, relax_pipeline=relax.get_pipeline("opt_llm_mg"), tir_pipeline="tirp"
        )
        vm = relax.VirtualMachine(ex, dev)
    return vm["batch_decode"], vm["cos_sin_cache_func"]


batch_decode_func, cos_sin_cache_func = get_qwen3_megakernel_batch_decode_func()
res1 = test_qwen3_layer(
    batch_decode_func,
    is_megakernel=True,
    cos_sin_cache_func=cos_sin_cache_func,
)

# fmt: off
@I.ir_module
class Module2:
    @R.function
    def batch_decode(input_embeds: R.Tensor(("batch_size", 1, 5120), dtype="float16"), paged_kv_cache: R.Object, packed_params: R.Tuple(R.Tensor((151936, 5120), dtype="float16"), R.Tensor((10240, 5120), dtype="float16"), R.Tensor((5120, 8192), dtype="float16"), R.Tensor((128,), dtype="float16"), R.Tensor((128,), dtype="float16"), R.Tensor((51200, 5120), dtype="float16"), R.Tensor((5120, 25600), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((151936, 5120), dtype="float16"))) -> R.Tuple(R.Tensor(("batch_size", 1, 151936), dtype="float32"), R.Object):
        batch_size = T.int64()
        R.func_attr({"num_input": 2})
        with R.dataflow():
            model_embed_tokens_weight1: R.Tensor((151936, 5120), dtype="float16") = packed_params[0]
            model_layers_0_self_attn_c_attn_weight1: R.Tensor((10240, 5120), dtype="float16") = packed_params[1]
            model_layers_0_self_attn_o_proj_weight1: R.Tensor((5120, 8192), dtype="float16") = packed_params[2]
            model_layers_0_self_attn_q_norm_weight1: R.Tensor((128,), dtype="float16") = packed_params[3]
            model_layers_0_self_attn_k_norm_weight1: R.Tensor((128,), dtype="float16") = packed_params[4]
            model_layers_0_mlp_gate_up_proj_weight1: R.Tensor((51200, 5120), dtype="float16") = packed_params[5]
            model_layers_0_mlp_down_proj_weight1: R.Tensor((5120, 25600), dtype="float16") = packed_params[6]
            model_layers_0_input_layernorm_weight1: R.Tensor((5120,), dtype="float16") = packed_params[7]
            model_layers_0_post_attention_layernorm_weight1: R.Tensor((5120,), dtype="float16") = packed_params[8]
            model_norm_weight1: R.Tensor((5120,), dtype="float16") = packed_params[9]
            lm_head_weight1: R.Tensor((151936, 5120), dtype="float16") = packed_params[10]
            rms_norm: R.Tensor((batch_size, 1, 5120), dtype="float16") = R.nn.rms_norm(input_embeds, model_layers_0_input_layernorm_weight1, axes=[-1], epsilon=9.9999999999999995e-07)
            permute_dims: R.Tensor((5120, 10240), dtype="float16") = R.permute_dims(model_layers_0_self_attn_c_attn_weight1, axes=None)
            matmul: R.Tensor((batch_size, 1, 10240), dtype="float16") = R.matmul(rms_norm, permute_dims, out_dtype="void")
            reshape: R.Tensor((batch_size, 1, 80, 128), dtype="float16") = R.reshape(matmul, R.shape([batch_size, 1, 80, 128]))
            split: R.Tuple(R.Tensor((batch_size, 1, 64, 128), dtype="float16"), R.Tensor((batch_size, 1, 8, 128), dtype="float16"), R.Tensor((batch_size, 1, 8, 128), dtype="float16")) = R.split(reshape, indices_or_sections=[64, 72], axis=2)
            split_0: R.Tensor((batch_size, 1, 64, 128), dtype="float16") = split[0]
            split_1: R.Tensor((batch_size, 1, 8, 128), dtype="float16") = split[1]
            split_2: R.Tensor((batch_size, 1, 8, 128), dtype="float16") = split[2]
            rms_norm1: R.Tensor((batch_size, 1, 64, 128), dtype="float16") = R.nn.rms_norm(split_0, model_layers_0_self_attn_q_norm_weight1, axes=[-1], epsilon=9.9999999999999995e-07)
            rms_norm2: R.Tensor((batch_size, 1, 8, 128), dtype="float16") = R.nn.rms_norm(split_1, model_layers_0_self_attn_k_norm_weight1, axes=[-1], epsilon=9.9999999999999995e-07)
            concat: R.Tensor((batch_size, 1, 80, 128), dtype="float16") = R.concat((rms_norm1, rms_norm2, split_2), axis=2)
            reshape1: R.Tensor((batch_size, 80, 128), dtype="float16") = R.reshape(concat, R.shape([batch_size, 80, 128]))
            lv = R.call_dps_packed("vm.builtin.attention_kv_cache_attention_with_fused_qkv", (paged_kv_cache, R.prim_value(0), R.prim_value(T.float32(0.088388347648318447)), reshape1), out_sinfo=R.Tensor((batch_size, 64, 128), dtype="float16"))
            reshape2: R.Tensor((batch_size, 1, 64, 128), dtype="float16") = R.reshape(lv, R.shape([batch_size, 1, 64, 128]))
            reshape3: R.Tensor((batch_size, 1, 8192), dtype="float16") = R.reshape(reshape2, R.shape([batch_size, 1, 8192]))
            permute_dims1: R.Tensor((8192, 5120), dtype="float16") = R.permute_dims(model_layers_0_self_attn_o_proj_weight1, axes=None)
            matmul1: R.Tensor((batch_size, 1, 5120), dtype="float16") = R.matmul(reshape3, permute_dims1, out_dtype="void")
            add: R.Tensor((batch_size, 1, 5120), dtype="float16") = R.add(matmul1, input_embeds)
            rms_norm3: R.Tensor((batch_size, 1, 5120), dtype="float16") = R.nn.rms_norm(add, model_layers_0_post_attention_layernorm_weight1, axes=[-1], epsilon=9.9999999999999995e-07)
            permute_dims2: R.Tensor((5120, 51200), dtype="float16") = R.permute_dims(model_layers_0_mlp_gate_up_proj_weight1, axes=None)
            matmul2: R.Tensor((batch_size, 1, 51200), dtype="float16") = R.matmul(rms_norm3, permute_dims2, out_dtype="void")
            split1: R.Tuple(R.Tensor((batch_size, 1, 25600), dtype="float16"), R.Tensor((batch_size, 1, 25600), dtype="float16")) = R.split(matmul2, indices_or_sections=2, axis=-1)
            split_01: R.Tensor((batch_size, 1, 25600), dtype="float16") = split1[0]
            split_11: R.Tensor((batch_size, 1, 25600), dtype="float16") = split1[1]
            silu: R.Tensor((batch_size, 1, 25600), dtype="float16") = R.nn.silu(split_01)
            mul: R.Tensor((batch_size, 1, 25600), dtype="float16") = R.multiply(silu, split_11)
            permute_dims3: R.Tensor((25600, 5120), dtype="float16") = R.permute_dims(model_layers_0_mlp_down_proj_weight1, axes=None)
            matmul3: R.Tensor((batch_size, 1, 5120), dtype="float16") = R.matmul(mul, permute_dims3, out_dtype="void")
            add1: R.Tensor((batch_size, 1, 5120), dtype="float16") = R.add(matmul3, add)
            rms_norm4: R.Tensor((batch_size, 1, 5120), dtype="float16") = R.nn.rms_norm(add1, model_norm_weight1, axes=[-1], epsilon=9.9999999999999995e-07)
            permute_dims4: R.Tensor((5120, 151936), dtype="float16") = R.permute_dims(lm_head_weight1, axes=None)
            matmul4: R.Tensor((batch_size, 1, 151936), dtype="float16") = R.matmul(rms_norm4, permute_dims4, out_dtype="void")
            astype: R.Tensor((batch_size, 1, 151936), dtype="float32") = R.astype(matmul4, dtype="float32")
            gv1: R.Tuple(R.Tensor((batch_size, 1, 151936), dtype="float32"), R.Object) = astype, paged_kv_cache
            R.output(gv1)
        return gv1
# fmt: on


def get_ref_batch_decode_func():
    with target:
        ex = tvm.compile(
            Module2,
            target,
            relax_pipeline=relax.get_pipeline("opt_llm_2"),
        )
        vm = relax.VirtualMachine(ex, dev)
    return vm["batch_decode"]


res2 = test_qwen3_layer(get_ref_batch_decode_func())

import numpy as np

for i, (mg, ref) in enumerate(zip(res1, res2)):
    print(f"batch {i}")
    mg = mg.numpy()
    ref = ref.numpy().reshape(mg.shape)
    tokens_mg = sample_token(mg)
    tokens_ref = sample_token(ref)
    try:
        np.testing.assert_equal(tokens_mg, tokens_ref)
    except Exception as e:
        print(e)