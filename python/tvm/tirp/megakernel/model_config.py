from typing import Any, Dict
import tvm


qwen3_30b_a3b_config = {
    "MODEL_NAME": "qwen3_30b_a3b",
    "TIE_WORD_EMBEDDINGS": False,
    "VOCAB_SIZE": 151936,
    "MAX_POSITION_EMBEDDINGS": 40960,
    "HIDDEN_SIZE": 2048,
    "INTERMEDIATE_SIZE": 768,
    "NUM_HIDDEN_LAYERS": 48,
    "NUM_ATTENTION_HEADS": 32,
    "NUM_KEY_VALUE_HEADS": 4,
    "HEAD_DIM": 128,
    "RMS_NORM_EPS": 1e-6,
    "ROPE_THETA": 1000000,
    "NUM_EXPERTS": 128,
    "NUM_EXPERTS_PER_TOK": 8,
    "GATING_SPLIT_K_FACTOR": 4,
    "SPLIT_QKV_PROJECT_DICT": {1: 3, 4: 4, 8: 4},
    "SPLIT_O_PROJECT_DICT": {1: 5, 4: 3, 8: 2},
}

qwen3_32b_config = {
    "MODEL_NAME": "qwen3_32b",
    "TIE_WORD_EMBEDDINGS": False,
    "VOCAB_SIZE": 151936,
    "MAX_POSITION_EMBEDDINGS": 40960,
    "HIDDEN_SIZE": 5120,
    "INTERMEDIATE_SIZE": 25600,
    "NUM_HIDDEN_LAYERS": 64,
    "NUM_ATTENTION_HEADS": 64,
    "NUM_KEY_VALUE_HEADS": 8,
    "HEAD_DIM": 128,
    "RMS_NORM_EPS": 1e-6,
    "ROPE_THETA": 1000000,

    "SPLIT_QKV_PROJECT_DICT": {1: 3, 4: 4, 8: 4},
    "SPLIT_O_PROJECT_DICT": {1: 3, 4: 3, 8: 2},
    "GATE_UP_PROJ_SPLIT_K_FACTOR_DICT": {1: 1, 4: 1, 8: 2},
    "DOWN_PROJ_SPLIT_K_FACTOR_DICT": {1: 10, 4: 3, 8: 3},
}

llama3_1b_config = {
    "MODEL_NAME": "llama3_1b",
    "TIE_WORD_EMBEDDINGS": True,
    "VOCAB_SIZE": 128256,
    "MAX_POSITION_EMBEDDINGS": 131072,
    "HIDDEN_SIZE": 2048,
    "INTERMEDIATE_SIZE": 8192,
    "NUM_HIDDEN_LAYERS": 16,
    "NUM_ATTENTION_HEADS": 32,
    "NUM_KEY_VALUE_HEADS": 8,
    "HEAD_DIM": 64,
    "RMS_NORM_EPS": 1e-5,
    "ROPE_THETA": 500000.0,
    "ROPE_SCALING": {
        "FACTOR": 32.0,
        "HIGH_FREQ_FACTOR": 4.0,
        "LOW_FREQ_FACTOR": 1.0,
        "ORIGINAL_MAX_POSITION_EMBEDDINGS": 8192,
        "ROPE_TYPE": "llama3"
    },
    "SPLIT_QKV_PROJECT_DICT": {1: 8, 4: -1, 8: -1},
    "SPLIT_O_PROJECT_DICT": {1: 8, 4: -1, 8: -1},
    "GATE_UP_PROJ_SPLIT_K_FACTOR_DICT": {1: 1, 4: -1, 8: -1},
    "DOWN_PROJ_SPLIT_K_FACTOR_DICT": {1: 9, 4: -1, 8: -1},
}

@tvm.register_global_func("tirp.megakernel.get_model_config")
def get_model_config(model_name: str) -> Dict[str, Any]:
    """Get model config by model name.

    Parameters
    ----------
    model_name : str
        The model name.

    Returns
    -------
    Dict[str, Any]
        The model config.
    """
    if model_name == "qwen3_30b_a3b" or model_name == "qwen3_30b_a3b_unfused":
        return qwen3_30b_a3b_config
    elif model_name == "qwen3_32b":
        return qwen3_32b_config
    elif model_name == "llama3_1b":
        return llama3_1b_config
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
