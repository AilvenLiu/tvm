from typing import List
from pathlib import Path
import numpy as np
import torch
import tvm
from tqdm import tqdm

from tvm import dlight, relax, te, tir, target
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
from .test_fused_add_rms_norm import get_fused_add_rmsnorm_kernel
from .test_fused_split_silu_multiply import get_fused_split_silu_multiply_kernel
from .test_hgemm_1consumer_1cta_swap_splitk import get_hgemm_kernel
from .test_rope import get_rope_kernel
from .test_append_paged_kv_cache import get_append_paged_kv_cache_kernel
from .test_batch_decode import PlanInfo, get_decode_kernel

# pyright: reportInvalidTypeForm=false

dev = tvm.cuda()
target = tvm.target.Target("cuda")

NUM_HIDDEN_LAYERS = 64
LOAD_WEIGHTS = "/raid/user_data/bohanhou/Qwen3-32B-q0f16-MLC/"  # load weights from real model
# LOAD_WEIGHTS = None # generate weights
MAX_BATCH_SIZE = 32
MAX_SEQ_LEN = 1024
MAX_TOTAL_SEQ_LEN = MAX_BATCH_SIZE * MAX_SEQ_LEN
PAGE_SIZE = 16
ROPE_THETA = 1000000

config = Qwen3Config(
    hidden_act="silu",
    hidden_size=5120,
    intermediate_size=25600,
    attention_bias=False,
    num_attention_heads=64,
    num_hidden_layers=NUM_HIDDEN_LAYERS,  # 64,
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
    debug_dir = Path("/home/bohanhou/qwen3-mg-debug")

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


def get_params(named_params):
    if LOAD_WEIGHTS is None:
        print("Generating weights")
        import torch
        import torch.nn as nn

        torch.manual_seed(0)
        result = list()
        for k, param in tqdm(named_params):
            torch_tensor = (
                torch.empty(tuple(param.shape)).to(getattr(torch, param.dtype)).to("cuda")
            )
            if k.endswith("norm.weight"):
                nn.init.ones_(torch_tensor)
            elif k.endswith("embed_tokens.weight"):
                nn.init.normal_(torch_tensor, mean=0.0, std=0.02)
            else:
                nn.init.xavier_uniform_(torch_tensor, gain=1.0)
            torch_tensor = torch_tensor
            result.append(tvm.runtime.ndarray.from_dlpack(torch.to_dlpack(torch_tensor)))
    else:
        from tvm.contrib import tvmjs

        print("Loading weights from", LOAD_WEIGHTS)
        params, _ = tvmjs.load_ndarray_cache(LOAD_WEIGHTS, device=dev)
        print("Loaded", len(params), "weights")
        result = [params[k] for k, v in named_params]
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


def test_qwen3_model(batch_decode_func, is_megakernel=False):
    func = tvm.get_global_func("vm.builtin.paged_attention_kv_cache_tensor_retrieve")

    kv_cache = vm["create_flashinfer_paged_kv_cache"](
        ShapeTuple([MAX_BATCH_SIZE]),  # max_batch_size
        ShapeTuple([MAX_TOTAL_SEQ_LEN]),  # max_total_seq_len
        ShapeTuple([MAX_SEQ_LEN]),  # prefill_chunk_size
        ShapeTuple([PAGE_SIZE]),  # page_size
        ShapeTuple([0]),  # support_sliding_window
    )

    if is_megakernel:
        _, sin_cos_cache = get_rope_kernel(MAX_SEQ_LEN, 128, ROPE_THETA)
        sin_cos_cache = tvm.runtime.ndarray.from_dlpack(torch.to_dlpack(sin_cos_cache))
        plan_info = PlanInfo(qo_heads=64, kv_heads=8, head_dim=128, enforce_no_split_kv=True)
        assert not plan_info.split_kv

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
            [
                *kv_data,
                kv_indptr,
                kv_indices,
                kv_last_page_len,
                append_position_map,
                q_rope_position_map,
            ] = func(kv_cache, NUM_HIDDEN_LAYERS)
            plan_info.plan(batch_size, kv_indptr.numpy().tolist(), PAGE_SIZE, kv_data[0].shape[0])
            logits = batch_decode_func(
                hidden_states,
                #########################################################
                kv_cache,
                #########################################################
                kv_data,
                kv_indptr,
                kv_indices,
                kv_last_page_len,
                # append
                append_position_map,
                q_rope_position_map,
                # rope
                sin_cos_cache,
                # attn
                plan_info.lse_tvm,
                plan_info.request_indices_tvm,
                plan_info.kv_tile_indices_tvm,
                plan_info.max_chunk_size,
                # #########################################################
                # params
                params,
            )
        else:
            logits, kv_cache = batch_decode_func(hidden_states, kv_cache, params)

        end_forward_func(kv_cache)
        last_tokens = sample_token(logits.numpy()).flatten()
        logits_arr.append(logits)

    return logits_arr


res0 = test_qwen3_model(vm["batch_decode"])


def attach_attr(func, name):
    func = func.without_attr("global_symbol")
    func = func.with_attr("global_symbol", name)
    func = func.with_attr("tir.is_scheduled", True)
    func = func.with_attr("tir.noalias", True)
    return func


def get_qwen3_megakernel_mod():
    rms_norm = get_rmsnorm_kernel(5120)
    fused_add_rmsnorm = get_fused_add_rmsnorm_kernel(5120)
    rms_norm1 = get_rmsnorm_kernel(128)
    fused_split_silu_multiply = get_fused_split_silu_multiply_kernel(25600)
    hgemm1, reduce1, tile_k_num1 = get_hgemm_kernel(dim_n=10240, dim_k=5120)
    hgemm2, reduce2, tile_k_num2 = get_hgemm_kernel(dim_n=5120, dim_k=8192)
    hgemm3, reduce3, tile_k_num3 = get_hgemm_kernel(dim_n=51200, dim_k=5120)
    hgemm4, reduce4, tile_k_num4 = get_hgemm_kernel(dim_n=5120, dim_k=25600)
    hgemm5, reduce5, tile_k_num5 = get_hgemm_kernel(dim_n=151936, dim_k=5120)
    rope, sin_cos_cache = get_rope_kernel(MAX_SEQ_LEN, 128, ROPE_THETA)
    append_paged_kv_cache = get_append_paged_kv_cache_kernel(8, 1, 128)
    decode, merge = get_decode_kernel(PlanInfo(64, 8, 128, enforce_no_split_kv=True), PAGE_SIZE)

    # fmt: off
    @R.macro(hygienic=False)
    def call_qwen3_layer(input0, input1, layer_id, tile_k_num1, tile_k_num2, tile_k_num3, tile_k_num4):
        with R.dataflow():
            # 8i+1, 8i+2, 8i+3, 8i+4, 8i+5, 8i+6, 8i+8, 8i+15 if i<num_hidden_layers-1 else 8i+9
            model_layers_0_self_attn_c_attn_weight1: R.Tensor((10240, 5120), dtype="float16") = packed_params[8*layer_id+1]
            model_layers_0_self_attn_o_proj_weight1: R.Tensor((5120, 8192), dtype="float16") = packed_params[8*layer_id+2]
            model_layers_0_self_attn_q_norm_weight1: R.Tensor((128,), dtype="float16") = packed_params[8*layer_id+3]
            model_layers_0_self_attn_k_norm_weight1: R.Tensor((128,), dtype="float16") = packed_params[8*layer_id+4]
            model_layers_0_mlp_gate_up_proj_weight1: R.Tensor((51200, 5120), dtype="float16") = packed_params[8*layer_id+5]
            model_layers_0_mlp_down_proj_weight1: R.Tensor((5120, 25600), dtype="float16") = packed_params[8*layer_id+6]
            model_layers_0_post_attention_layernorm_weight1: R.Tensor((5120,), dtype="float16") = packed_params[8*layer_id+8]
            model_norm_weight1: R.Tensor((5120,), dtype="float16") = packed_params[8*layer_id+15 if layer_id < NUM_HIDDEN_LAYERS-1 else 8*layer_id+9]

            # permute_dims0 = R.permute_dims(model_layers_0_self_attn_c_attn_weight1, axes=None)
            # lv = R.matmul(rms_norm, permute_dims0, out_dtype="void")
            ps0 = R.call_tir(cls.hgemm1, (input0, model_layers_0_self_attn_c_attn_weight1), out_sinfo=R.Tensor((tile_k_num1, batch_size, 10240), dtype="float32"))
            lv = R.call_tir(cls.reduce1, (ps0,), out_sinfo=R.Tensor((batch_size, 10240), dtype="float16"))

            reshape = R.reshape(lv, (batch_size, 1, 80, 128))
            split = R.call_tir(cls.split, (reshape,), out_sinfo=[R.Tensor((batch_size, 1, 64, 128), dtype="float16"), R.Tensor((batch_size, 1, 8, 128), dtype="float16"), R.Tensor((batch_size, 1, 8, 128), dtype="float16")])
            split_0: R.Tensor((batch_size, 1, 64, 128), dtype="float16") = split[0]
            split_1: R.Tensor((batch_size, 1, 8, 128), dtype="float16") = split[1]
            rs1 = R.reshape(split_0, (batch_size * 64, 128))
            rs2 = R.reshape(split_1, (batch_size * 8, 128))
            rms_norm1 = R.call_tir(cls.rms_norm1, (rs1, model_layers_0_self_attn_q_norm_weight1), out_sinfo=R.Tensor((batch_size * 64, 128), dtype="float16"))
            rms_norm2 = R.call_tir(cls.rms_norm1, (rs2, model_layers_0_self_attn_k_norm_weight1), out_sinfo=R.Tensor((batch_size * 8, 128), dtype="float16"))
            rms_norm1_rs = R.reshape(rms_norm1, (batch_size, 1, 64, 128)) # q
            rms_norm2_rs = R.reshape(rms_norm2, (batch_size, 1, 8, 128)) # k
            v: R.Tensor((batch_size, 1, 8, 128), dtype="float16") = split[2] # v

            ######################################################### Attention #########################################################
            #########################################################
            # lv1 = R.call_tir(cls.fused_concatenate, (v, rms_norm1_rs, rms_norm2_rs), out_sinfo=R.Tensor((batch_size, 1, 80, 128), dtype="float16"))
            # reshape1 = R.reshape(lv1, (batch_size, 80, 128))
            # lv_2 = R.call_dps_packed("vm.builtin.attention_kv_cache_attention_with_fused_qkv", (paged_kv_cache, R.prim_value(0), R.prim_value(T.float32(0.088388347648318447)), reshape1), out_sinfo=R.Tensor((batch_size, 64, 128), dtype="float16"))
            # reshape2 = R.reshape(lv_2, (batch_size, 1, 64, 128))
            # reshape3 = R.reshape(reshape2, (batch_size, 8192))
            #########################################################
            # rope
            q_rs = R.reshape(rms_norm1_rs, (batch_size, 64, 128))
            q_rope = R.call_tir(cls.rope, (q_rs, sin_cos_cache, q_rope_position_map), out_sinfo=R.Tensor((batch_size, 64, 128), dtype="float16"))
            k_rs = R.reshape(rms_norm2_rs, (batch_size, 8, 128))
            k = R.call_tir(cls.rope, (k_rs, sin_cos_cache, q_rope_position_map), out_sinfo=R.Tensor((batch_size, 8, 128), dtype="float16"))
            k_rope = R.reshape(k, (batch_size, 1, 8, 128))
            # append
            append_position_map_rs = R.reshape(append_position_map, (batch_size, 1))
            new_kv_data = R.call_tir_inplace(cls.append_paged_kv_cache, (kv_data[layer_id], k_rope, v, append_position_map_rs), [0], 
                                                out_sinfo=R.Tensor((max_page_num, 2, 8, page_size, 128), dtype="float16"))
            # attention
            o = R.call_tir(cls.decode, 
                            (q_rope, new_kv_data, plan_lse_tvm, kv_indptr, kv_last_page_len, kv_indices, plan_request_indices, plan_kv_tile_indices, plan_max_chunk_size), out_sinfo=R.Tensor((batch_size, 64, 128), dtype="float16"))
            reshape3 = R.reshape(o, (batch_size, 8192))
            #########################################################
            ######################################################### Attention #########################################################

            # permute_dims1 = R.permute_dims(model_layers_0_self_attn_o_proj_weight1, axes=None)
            # lv1_1: R.Tensor((batch_size, 5120), dtype="float16") = R.matmul(reshape3, permute_dims1, out_dtype="void")
            rs1 = R.call_tir(cls.hgemm2, (reshape3, model_layers_0_self_attn_o_proj_weight1), out_sinfo=R.Tensor((tile_k_num2, batch_size, 5120), dtype="float32"))
            lv1_1 = R.call_tir(cls.reduce2, (rs1,), out_sinfo=R.Tensor((batch_size, 5120), dtype="float16"))

            lv_3 = R.call_tir_inplace(cls.fused_add_rmsnorm, (lv1_1, input1, model_layers_0_post_attention_layernorm_weight1), [0, 1], out_sinfo=[R.Tensor((batch_size, 5120), dtype="float16"), R.Tensor((batch_size, 5120), dtype="float16")])
            lv1_2: R.Tensor((batch_size, 5120), dtype="float16") = lv_3[1]
            rms_norm3: R.Tensor((batch_size, 5120), dtype="float16") = lv_3[0]

            # permute_dims2 = R.permute_dims(model_layers_0_mlp_gate_up_proj_weight1, axes=None)
            # lv2: R.Tensor((batch_size, 51200), dtype="float16") = R.matmul(rms_norm3, permute_dims2, out_dtype="void")
            rs2 = R.call_tir(cls.hgemm3, (rms_norm3, model_layers_0_mlp_gate_up_proj_weight1), out_sinfo=R.Tensor((tile_k_num3, batch_size, 51200), dtype="float32"))
            lv2 = R.call_tir(cls.reduce3, (rs2,), out_sinfo=R.Tensor((batch_size, 51200), dtype="float16"))

            lv2_1 = R.call_tir(cls.fused_split_silu_multiply, (lv2,), out_sinfo=R.Tensor((batch_size, 25600), dtype="float16"))

            # permute_dims3 = R.permute_dims(model_layers_0_mlp_down_proj_weight1, axes=None)
            # lv3: R.Tensor((batch_size, 5120), dtype="float16") = R.matmul(lv2_1, permute_dims3, out_dtype="void")
            rs3 = R.call_tir(cls.hgemm4, (lv2_1, model_layers_0_mlp_down_proj_weight1), out_sinfo=R.Tensor((tile_k_num4, batch_size, 5120), dtype="float32"))
            lv3 = R.call_tir(cls.reduce4, (rs3,), out_sinfo=R.Tensor((batch_size, 5120), dtype="float16"))

            lv2_2 = R.call_tir_inplace(cls.fused_add_rmsnorm, (lv3, lv1_2, model_norm_weight1), [0, 1], out_sinfo=[R.Tensor((batch_size, 5120), dtype="float16"), R.Tensor((batch_size, 5120), dtype="float16")])
            R.output(lv2_2)
        return lv2_2


    @I.ir_module
    class Module:
        @T.prim_func
        def rms_norm(input_ptr: T.handle, weight_ptr: T.handle, out_ptr: T.handle):
            pass

        @T.prim_func
        def rms_norm1(input_ptr: T.handle, weight_ptr: T.handle, out_ptr: T.handle):
            pass

        @T.prim_func
        def fused_add_rmsnorm(input_ptr: T.handle, residual_ptr: T.handle, weight_ptr: T.handle):
            pass

        @T.prim_func
        def fused_split_silu_multiply(input_ptr: T.handle, output_ptr: T.handle):
            pass

        @T.prim_func(private=True)
        def split(var_reshape: T.handle, var_T_split: T.handle, var_T_split_1: T.handle, var_T_split_2: T.handle):
            T.func_attr({"tir.noalias": True})
            batch_size = T.int64()
            reshape = T.match_buffer(var_reshape, (batch_size, T.int64(1), T.int64(80), T.int64(128)), "float16")
            T_split = T.match_buffer(var_T_split, (batch_size, T.int64(1), T.int64(64), T.int64(128)), "float16")
            T_split_1 = T.match_buffer(var_T_split_1, (batch_size, T.int64(1), T.int64(8), T.int64(128)), "float16")
            T_split_2 = T.match_buffer(var_T_split_2, (batch_size, T.int64(1), T.int64(8), T.int64(128)), "float16")
            # with T.block("root"):
            for ax0, ax1, ax2, ax3 in T.grid(batch_size, T.int64(1), T.int64(64), T.int64(128)):
                with T.block("T_split"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T.reads(reshape[v_ax0, v_ax1, v_ax2, v_ax3])
                    T.writes(T_split[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_split[v_ax0, v_ax1, v_ax2, v_ax3] = reshape[v_ax0, v_ax1, v_ax2, v_ax3]
            for ax0, ax1, ax2, ax3 in T.grid(batch_size, T.int64(1), T.int64(8), T.int64(128)):
                with T.block("T_split_1"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T.reads(reshape[v_ax0, v_ax1, v_ax2 + T.int64(64), v_ax3])
                    T.writes(T_split_1[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_split_1[v_ax0, v_ax1, v_ax2, v_ax3] = reshape[v_ax0, v_ax1, v_ax2 + T.int64(64), v_ax3]
            for ax0, ax1, ax2, ax3 in T.grid(batch_size, T.int64(1), T.int64(8), T.int64(128)):
                with T.block("T_split_2"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T.reads(reshape[v_ax0, v_ax1, v_ax2 + T.int64(72), v_ax3])
                    T.writes(T_split_2[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_split_2[v_ax0, v_ax1, v_ax2, v_ax3] = reshape[v_ax0, v_ax1, v_ax2 + T.int64(72), v_ax3]

        @T.prim_func(private=True)
        def fused_concatenate(p_split_2: T.handle, p_rms_norm1: T.handle, p_rms_norm2: T.handle, p_output0: T.handle):
            T.func_attr({"tir.noalias": True})
            batch_size = T.int64()
            split_2 = T.match_buffer(p_split_2, (batch_size, T.int64(1), T.int64(8), T.int64(128)), "float16")
            rms_norm1 = T.match_buffer(p_rms_norm1, (batch_size, T.int64(1), T.int64(64), T.int64(128)), "float16")
            rms_norm2 = T.match_buffer(p_rms_norm2, (batch_size, T.int64(1), T.int64(8), T.int64(128)), "float16")
            T_concat_intermediate = T.match_buffer(p_output0, (batch_size, T.int64(1), T.int64(80), T.int64(128)), "float16")
            # with T.block("root"):
            for ax0, ax1, ax2, ax3 in T.grid(batch_size, T.int64(1), T.int64(80), T.int64(128)):
                with T.block("T_concat"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T.reads(split_2[v_ax0, v_ax1, v_ax2 - T.int64(72), v_ax3], rms_norm2[v_ax0, v_ax1, v_ax2 - T.int64(64), v_ax3], rms_norm1[v_ax0, v_ax1, v_ax2, v_ax3])
                    T.writes(T_concat_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_concat_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = T.if_then_else(T.int64(72) <= v_ax2, split_2[v_ax0, v_ax1, v_ax2 - T.int64(72), v_ax3], T.if_then_else(T.int64(64) <= v_ax2, rms_norm2[v_ax0, v_ax1, v_ax2 - T.int64(64), v_ax3], rms_norm1[v_ax0, v_ax1, v_ax2, v_ax3]))

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
        def hgemm1(A_ptr: T.handle, b_ptr: T.handle, partial_sum_ptr: T.handle):
            pass

        @T.prim_func(private=True)
        def reduce1(partial_sum_ptr: T.handle, D_ptr: T.handle):
            pass

        @T.prim_func(private=True)
        def hgemm2(A_ptr: T.handle, b_ptr: T.handle, partial_sum_ptr: T.handle):
            pass

        @T.prim_func(private=True)
        def reduce2(partial_sum_ptr: T.handle, D_ptr: T.handle):
            pass

        @T.prim_func(private=True)
        def hgemm3(A_ptr: T.handle, b_ptr: T.handle, partial_sum_ptr: T.handle):
            pass

        @T.prim_func(private=True)
        def reduce3(partial_sum_ptr: T.handle, D_ptr: T.handle):
            pass

        @T.prim_func(private=True)
        def hgemm4(A_ptr: T.handle, b_ptr: T.handle, partial_sum_ptr: T.handle):
            pass

        @T.prim_func(private=True)
        def reduce4(partial_sum_ptr: T.handle, D_ptr: T.handle):
            pass

        @T.prim_func(private=True)
        def hgemm5(A_ptr: T.handle, b_ptr: T.handle, partial_sum_ptr: T.handle):
            pass

        @T.prim_func(private=True)
        def reduce5(partial_sum_ptr: T.handle, D_ptr: T.handle):
            pass

        @T.prim_func(private=True)
        def rope(q: T.handle, sin_cos_cache: T.handle, pos_ids: T.handle, q_rope: T.handle):
            pass

        @T.prim_func(private=True)
        def append_paged_kv_cache(cache_ptr: T.handle, k_ptr: T.handle, v_ptr: T.handle, pos_map_ptr: T.handle):
            pass

        @T.prim_func(private=True)
        def decode(q: T.handle, kv: T.handle, lse: T.handle, kv_indptr: T.handle, kv_last_page_len: T.handle, kv_indices: T.handle, request_indices: T.handle, kv_tile_indices: T.handle, max_chunk_size: T.handle, o: T.handle):
            pass

        @R.function
        def batch_decode(
            input_embeds: R.Tensor(("batch_size", 1, 5120), dtype="float16"),
            #########################################################
            paged_kv_cache: R.Object,
            #########################################################
            kv_data: R.Tuple(
                [R.Tensor(("max_page_num", 2, 8, "page_size", 128), dtype="float16")] * NUM_HIDDEN_LAYERS
            ),
            kv_indptr: R.Tensor(("batch_size + 1",), dtype="int32"),
            kv_indices: R.Tensor(("total_page_num",), dtype="int32"),
            kv_last_page_len: R.Tensor(("batch_size",), dtype="int32"),
            # append
            append_position_map: R.Tensor(("batch_size",), dtype="int32"),
            q_rope_position_map: R.Tensor(("batch_size",), dtype="int32"),
            # rope
            sin_cos_cache: R.Tensor((MAX_SEQ_LEN, 128), dtype="float32"),
            # attn
            plan_lse_tvm: R.Tensor(("new_batch_size", 64), dtype="float32"),
            plan_request_indices: R.Tensor(("new_batch_size",), dtype="int32"),
            plan_kv_tile_indices: R.Tensor(("new_batch_size",), dtype="int32"),
            plan_max_chunk_size: R.Tensor((1,), dtype="int32"),
            #########################################################
            packed_params: R.Tuple(
                R.Tensor((151936, 5120), dtype="float16"),
                #
                *([R.Tensor((10240, 5120), dtype="float16"),
                R.Tensor((5120, 8192), dtype="float16"),
                R.Tensor((128,), dtype="float16"),
                R.Tensor((128,), dtype="float16"),
                R.Tensor((51200, 5120), dtype="float16"),
                R.Tensor((5120, 25600), dtype="float16"),
                R.Tensor((5120,), dtype="float16"),
                R.Tensor((5120,), dtype="float16"),] * NUM_HIDDEN_LAYERS),
                #                
                R.Tensor((5120,), dtype="float16"),
                R.Tensor((151936, 5120), dtype="float16"),
            ),
        ):
            batch_size = T.int64()
            max_page_num = T.int64()
            page_size = T.int64()

            cls = Module
            with R.dataflow():
                # num_hidden_layers=2
                model_layers_0_input_layernorm_weight1: R.Tensor((5120,), dtype="float16") = packed_params[7]
                lm_head_weight1: R.Tensor((151936, 5120), dtype="float16") = packed_params[NUM_HIDDEN_LAYERS*8+2] # num_hidden_layers*8+2

                rs0 = R.reshape(input_embeds, (batch_size, 5120))
                rms_norm = R.call_tir(cls.rms_norm, (rs0, model_layers_0_input_layernorm_weight1), out_sinfo=R.Tensor((batch_size, 5120), dtype="float16"))

                o_layer0 = call_qwen3_layer(rms_norm, rs0, 0, tile_k_num1, tile_k_num2, tile_k_num3, tile_k_num4)
                o_layer1 = call_qwen3_layer(o_layer0[0], o_layer0[1], 1, tile_k_num1, tile_k_num2, tile_k_num3, tile_k_num4)
                o_layer2 = call_qwen3_layer(o_layer1[0], o_layer1[1], 2, tile_k_num1, tile_k_num2, tile_k_num3, tile_k_num4)
                o_layer3 = call_qwen3_layer(o_layer2[0], o_layer2[1], 3, tile_k_num1, tile_k_num2, tile_k_num3, tile_k_num4)
                o_layer4 = call_qwen3_layer(o_layer3[0], o_layer3[1], 4, tile_k_num1, tile_k_num2, tile_k_num3, tile_k_num4)
                o_layer5 = call_qwen3_layer(o_layer4[0], o_layer4[1], 5, tile_k_num1, tile_k_num2, tile_k_num3, tile_k_num4)
                o_layer6 = call_qwen3_layer(o_layer5[0], o_layer5[1], 6, tile_k_num1, tile_k_num2, tile_k_num3, tile_k_num4)
                o_layer7 = call_qwen3_layer(o_layer6[0], o_layer6[1], 7, tile_k_num1, tile_k_num2, tile_k_num3, tile_k_num4)
                o_layer8 = call_qwen3_layer(o_layer7[0], o_layer7[1], 8, tile_k_num1, tile_k_num2, tile_k_num3, tile_k_num4)
                o_layer9 = call_qwen3_layer(o_layer8[0], o_layer8[1], 9, tile_k_num1, tile_k_num2, tile_k_num3, tile_k_num4)
                o_layer10 = call_qwen3_layer(o_layer9[0], o_layer9[1], 10, tile_k_num1, tile_k_num2, tile_k_num3, tile_k_num4)
                o_layer11 = call_qwen3_layer(o_layer10[0], o_layer10[1], 11, tile_k_num1, tile_k_num2, tile_k_num3, tile_k_num4)
                o_layer12 = call_qwen3_layer(o_layer11[0], o_layer11[1], 12, tile_k_num1, tile_k_num2, tile_k_num3, tile_k_num4)
                o_layer13 = call_qwen3_layer(o_layer12[0], o_layer12[1], 13, tile_k_num1, tile_k_num2, tile_k_num3, tile_k_num4)
                o_layer14 = call_qwen3_layer(o_layer13[0], o_layer13[1], 14, tile_k_num1, tile_k_num2, tile_k_num3, tile_k_num4)
                o_layer15 = call_qwen3_layer(o_layer14[0], o_layer14[1], 15, tile_k_num1, tile_k_num2, tile_k_num3, tile_k_num4)
                o_layer16 = call_qwen3_layer(o_layer15[0], o_layer15[1], 16, tile_k_num1, tile_k_num2, tile_k_num3, tile_k_num4)
                o_layer17 = call_qwen3_layer(o_layer16[0], o_layer16[1], 17, tile_k_num1, tile_k_num2, tile_k_num3, tile_k_num4)
                o_layer18 = call_qwen3_layer(o_layer17[0], o_layer17[1], 18, tile_k_num1, tile_k_num2, tile_k_num3, tile_k_num4)
                o_layer19 = call_qwen3_layer(o_layer18[0], o_layer18[1], 19, tile_k_num1, tile_k_num2, tile_k_num3, tile_k_num4)
                o_layer20 = call_qwen3_layer(o_layer19[0], o_layer19[1], 20, tile_k_num1, tile_k_num2, tile_k_num3, tile_k_num4)
                o_layer21 = call_qwen3_layer(o_layer20[0], o_layer20[1], 21, tile_k_num1, tile_k_num2, tile_k_num3, tile_k_num4)
                o_layer22 = call_qwen3_layer(o_layer21[0], o_layer21[1], 22, tile_k_num1, tile_k_num2, tile_k_num3, tile_k_num4)
                o_layer23 = call_qwen3_layer(o_layer22[0], o_layer22[1], 23, tile_k_num1, tile_k_num2, tile_k_num3, tile_k_num4)
                o_layer24 = call_qwen3_layer(o_layer23[0], o_layer23[1], 24, tile_k_num1, tile_k_num2, tile_k_num3, tile_k_num4)
                o_layer25 = call_qwen3_layer(o_layer24[0], o_layer24[1], 25, tile_k_num1, tile_k_num2, tile_k_num3, tile_k_num4)
                o_layer26 = call_qwen3_layer(o_layer25[0], o_layer25[1], 26, tile_k_num1, tile_k_num2, tile_k_num3, tile_k_num4)
                o_layer27 = call_qwen3_layer(o_layer26[0], o_layer26[1], 27, tile_k_num1, tile_k_num2, tile_k_num3, tile_k_num4)
                o_layer28 = call_qwen3_layer(o_layer27[0], o_layer27[1], 28, tile_k_num1, tile_k_num2, tile_k_num3, tile_k_num4)
                o_layer29 = call_qwen3_layer(o_layer28[0], o_layer28[1], 29, tile_k_num1, tile_k_num2, tile_k_num3, tile_k_num4)
                o_layer30 = call_qwen3_layer(o_layer29[0], o_layer29[1], 30, tile_k_num1, tile_k_num2, tile_k_num3, tile_k_num4)
                o_layer31 = call_qwen3_layer(o_layer30[0], o_layer30[1], 31, tile_k_num1, tile_k_num2, tile_k_num3, tile_k_num4)
                o_layer32 = call_qwen3_layer(o_layer31[0], o_layer31[1], 32, tile_k_num1, tile_k_num2, tile_k_num3, tile_k_num4)
                o_layer33 = call_qwen3_layer(o_layer32[0], o_layer32[1], 33, tile_k_num1, tile_k_num2, tile_k_num3, tile_k_num4)
                o_layer34 = call_qwen3_layer(o_layer33[0], o_layer33[1], 34, tile_k_num1, tile_k_num2, tile_k_num3, tile_k_num4)
                o_layer35 = call_qwen3_layer(o_layer34[0], o_layer34[1], 35, tile_k_num1, tile_k_num2, tile_k_num3, tile_k_num4)
                o_layer36 = call_qwen3_layer(o_layer35[0], o_layer35[1], 36, tile_k_num1, tile_k_num2, tile_k_num3, tile_k_num4)
                o_layer37 = call_qwen3_layer(o_layer36[0], o_layer36[1], 37, tile_k_num1, tile_k_num2, tile_k_num3, tile_k_num4)
                o_layer38 = call_qwen3_layer(o_layer37[0], o_layer37[1], 38, tile_k_num1, tile_k_num2, tile_k_num3, tile_k_num4)
                o_layer39 = call_qwen3_layer(o_layer38[0], o_layer38[1], 39, tile_k_num1, tile_k_num2, tile_k_num3, tile_k_num4)
                o_layer40 = call_qwen3_layer(o_layer39[0], o_layer39[1], 40, tile_k_num1, tile_k_num2, tile_k_num3, tile_k_num4)
                o_layer41 = call_qwen3_layer(o_layer40[0], o_layer40[1], 41, tile_k_num1, tile_k_num2, tile_k_num3, tile_k_num4)
                o_layer42 = call_qwen3_layer(o_layer41[0], o_layer41[1], 42, tile_k_num1, tile_k_num2, tile_k_num3, tile_k_num4)
                o_layer43 = call_qwen3_layer(o_layer42[0], o_layer42[1], 43, tile_k_num1, tile_k_num2, tile_k_num3, tile_k_num4)
                o_layer44 = call_qwen3_layer(o_layer43[0], o_layer43[1], 44, tile_k_num1, tile_k_num2, tile_k_num3, tile_k_num4)
                o_layer45 = call_qwen3_layer(o_layer44[0], o_layer44[1], 45, tile_k_num1, tile_k_num2, tile_k_num3, tile_k_num4)
                o_layer46 = call_qwen3_layer(o_layer45[0], o_layer45[1], 46, tile_k_num1, tile_k_num2, tile_k_num3, tile_k_num4)
                o_layer47 = call_qwen3_layer(o_layer46[0], o_layer46[1], 47, tile_k_num1, tile_k_num2, tile_k_num3, tile_k_num4)
                o_layer48 = call_qwen3_layer(o_layer47[0], o_layer47[1], 48, tile_k_num1, tile_k_num2, tile_k_num3, tile_k_num4)
                o_layer49 = call_qwen3_layer(o_layer48[0], o_layer48[1], 49, tile_k_num1, tile_k_num2, tile_k_num3, tile_k_num4)
                o_layer50 = call_qwen3_layer(o_layer49[0], o_layer49[1], 50, tile_k_num1, tile_k_num2, tile_k_num3, tile_k_num4)
                o_layer51 = call_qwen3_layer(o_layer50[0], o_layer50[1], 51, tile_k_num1, tile_k_num2, tile_k_num3, tile_k_num4)
                o_layer52 = call_qwen3_layer(o_layer51[0], o_layer51[1], 52, tile_k_num1, tile_k_num2, tile_k_num3, tile_k_num4)
                o_layer53 = call_qwen3_layer(o_layer52[0], o_layer52[1], 53, tile_k_num1, tile_k_num2, tile_k_num3, tile_k_num4)
                o_layer54 = call_qwen3_layer(o_layer53[0], o_layer53[1], 54, tile_k_num1, tile_k_num2, tile_k_num3, tile_k_num4)
                o_layer55 = call_qwen3_layer(o_layer54[0], o_layer54[1], 55, tile_k_num1, tile_k_num2, tile_k_num3, tile_k_num4)
                o_layer56 = call_qwen3_layer(o_layer55[0], o_layer55[1], 56, tile_k_num1, tile_k_num2, tile_k_num3, tile_k_num4)
                o_layer57 = call_qwen3_layer(o_layer56[0], o_layer56[1], 57, tile_k_num1, tile_k_num2, tile_k_num3, tile_k_num4)
                o_layer58 = call_qwen3_layer(o_layer57[0], o_layer57[1], 58, tile_k_num1, tile_k_num2, tile_k_num3, tile_k_num4)
                o_layer59 = call_qwen3_layer(o_layer58[0], o_layer58[1], 59, tile_k_num1, tile_k_num2, tile_k_num3, tile_k_num4)
                o_layer60 = call_qwen3_layer(o_layer59[0], o_layer59[1], 60, tile_k_num1, tile_k_num2, tile_k_num3, tile_k_num4)
                o_layer61 = call_qwen3_layer(o_layer60[0], o_layer60[1], 61, tile_k_num1, tile_k_num2, tile_k_num3, tile_k_num4)
                o_layer62 = call_qwen3_layer(o_layer61[0], o_layer61[1], 62, tile_k_num1, tile_k_num2, tile_k_num3, tile_k_num4)
                o_layer63 = call_qwen3_layer(o_layer62[0], o_layer62[1], 63, tile_k_num1, tile_k_num2, tile_k_num3, tile_k_num4)

                # permute_dims4 = R.permute_dims(lm_head_weight1, axes=None)
                # lv4: R.Tensor((batch_size, 151936), dtype="float16") = R.matmul(rms_norm4, permute_dims4, out_dtype="void")
                rs4 = R.call_tir(cls.hgemm5, (o_layer63[0], lm_head_weight1), out_sinfo=R.Tensor((tile_k_num5, batch_size, 151936), dtype="float32"))
                lv4 = R.call_tir(cls.reduce5, (rs4,), out_sinfo=R.Tensor((batch_size, 151936), dtype="float16"))

                lv4_rs = R.reshape(lv4, (batch_size, 1, 151936))
                astype = R.call_tir(cls.cast, (lv4_rs,), out_sinfo=R.Tensor((batch_size, 1, 151936), dtype="float32"))
                R.output(astype)
            return astype
    # fmt: on
    mod = Module

    mod.update_func(mod.get_global_var("rms_norm"), attach_attr(rms_norm, "rms_norm"))
    mod.update_func(
        mod.get_global_var("fused_add_rmsnorm"), attach_attr(fused_add_rmsnorm, "fused_add_rmsnorm")
    )
    mod.update_func(mod.get_global_var("rms_norm1"), attach_attr(rms_norm1, "rms_norm1"))
    mod.update_func(
        mod.get_global_var("fused_split_silu_multiply"),
        attach_attr(fused_split_silu_multiply, "fused_split_silu_multiply"),
    )
    mod.update_func(mod.get_global_var("hgemm1"), attach_attr(hgemm1, "hgemm1"))
    mod.update_func(mod.get_global_var("reduce1"), attach_attr(reduce1, "reduce1"))
    mod.update_func(mod.get_global_var("hgemm2"), attach_attr(hgemm2, "hgemm2"))
    mod.update_func(mod.get_global_var("reduce2"), attach_attr(reduce2, "reduce2"))
    mod.update_func(mod.get_global_var("hgemm3"), attach_attr(hgemm3, "hgemm3"))
    mod.update_func(mod.get_global_var("reduce3"), attach_attr(reduce3, "reduce3"))
    mod.update_func(mod.get_global_var("hgemm4"), attach_attr(hgemm4, "hgemm4"))
    mod.update_func(mod.get_global_var("reduce4"), attach_attr(reduce4, "reduce4"))
    mod.update_func(mod.get_global_var("hgemm5"), attach_attr(hgemm5, "hgemm5"))
    mod.update_func(mod.get_global_var("reduce5"), attach_attr(reduce5, "reduce5"))
    mod.update_func(mod.get_global_var("rope"), attach_attr(rope, "rope"))
    mod.update_func(
        mod.get_global_var("append_paged_kv_cache"),
        attach_attr(append_paged_kv_cache, "append_paged_kv_cache"),
    )
    mod.update_func(mod.get_global_var("decode"), attach_attr(decode, "decode"))
    return mod


def get_qwen3_megakernel_batch_decode_func():
    mg_model = get_qwen3_megakernel_mod()

    with target:
        ex = tvm.compile(
            mg_model, target, relax_pipeline=relax.get_pipeline("opt_llm_mg"), tir_pipeline="tirp"
        )
        vm = relax.VirtualMachine(ex, dev)
    return vm["batch_decode"]


res1 = test_qwen3_model(get_qwen3_megakernel_batch_decode_func(), is_megakernel=True)


import numpy as np

for i, (ref, mg) in enumerate(zip(res0, res1)):
    print(f"batch {i}")
    mg = mg.numpy()
    ref = ref.numpy().reshape(mg.shape)
    print(mg.flatten()[:10])
    print(ref.flatten()[:10])
    print("--------------------------------")
    try:
        np.testing.assert_allclose(ref, mg, atol=1e-2, rtol=1e-2)
    except Exception as e:
        print(e)
