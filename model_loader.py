# model_loader.py

import os
import sys
import threading
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from huggingface_hub import hf_hub_download
from transformers import GPT2Tokenizer


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

USE_HALF = False


# ============================================================
# MODEL REGISTRY
# ============================================================

MODEL_REGISTRY = {
    "gouda0.0.1": {
        "repo_id": "Erbium08/gouda0.0.1",
        "filename": "model.pt",
    },

    "gouda-g1-xs-r2": {
        "repo_id": "Erbium08/gouda-g1-xs-r2",
        "filename": "model.pt",
    },

    "gouda-g1-xs-r4": {
        "repo_id": "Erbium08/gouda-g1-xs-r4",
        "filename": "model_10000.pt",
    },

    "gouda-g1-s-r1": {
            "repo_id": "Erbium08/gouda-g1-s-r1",
            "filename": "model_30517_50m.pt",
    },

    "Gruyere-1.0-r1": {
        "repo_id": "Erbium08/Gruyere-1.0-r1",
        "filename": "model_19999.pt",
    }
}

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 8
    n_embd: int = 512


class CausalSelfAttention(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()

        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.c_attn = nn.Linear(
            config.n_embd,
            3 * config.n_embd
        )

        self.c_proj = nn.Linear(
            config.n_embd,
            config.n_embd
        )

        self.register_buffer(
            "bias",
            torch.tril(
                torch.ones(
                    config.block_size,
                    config.block_size
                )
            ).view(
                1,
                1,
                config.block_size,
                config.block_size
            )
        )

    def forward(self, x):

        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(
            self.n_embd,
            dim=2
        )

        head_dim = C // self.n_head

        q = q.view(
            B,
            T,
            self.n_head,
            head_dim
        ).transpose(1, 2)

        k = k.view(
            B,
            T,
            self.n_head,
            head_dim
        ).transpose(1, 2)

        v = v.view(
            B,
            T,
            self.n_head,
            head_dim
        ).transpose(1, 2)

        # Fast PyTorch attention implementation
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True
        )

        y = (
            y.transpose(1, 2)
             .contiguous()
             .view(B, T, C)
        )

        y = self.c_proj(y)

        return y


class MLP(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()

        self.c_fc = nn.Linear(
            config.n_embd,
            4 * config.n_embd
        )

        self.gelu = nn.GELU()

        self.c_proj = nn.Linear(
            4 * config.n_embd,
            config.n_embd
        )

    def forward(self, x):

        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)

        return x


class Block(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()

        self.ln_1 = nn.LayerNorm(
            config.n_embd
        )

        self.attn = CausalSelfAttention(
            config
        )

        self.ln_2 = nn.LayerNorm(
            config.n_embd
        )

        self.mlp = MLP(
            config
        )

    def forward(self, x):

        x = x + self.attn(
            self.ln_1(x)
        )

        x = x + self.mlp(
            self.ln_2(x)
        )

        return x



class GPT(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()

        self.config = config

        self.transformer = nn.ModuleDict({

            "wte": nn.Embedding(
                config.vocab_size,
                config.n_embd
            ),

            "wpe": nn.Embedding(
                config.block_size,
                config.n_embd
            ),

            "h": nn.ModuleList([
                Block(config)
                for _ in range(config.n_layer)
            ]),

            "ln_f": nn.LayerNorm(
                config.n_embd
            ),
        })

        self.lm_head = nn.Linear(
            config.n_embd,
            config.vocab_size,
            bias=False
        )

        # Weight tying
        self.transformer.wte.weight = (
            self.lm_head.weight
        )

        self.apply(
            self._init_weights
        )

    @staticmethod
    def _init_weights(module):

        if isinstance(module, nn.Linear):

            torch.nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

            if module.bias is not None:
                torch.nn.init.zeros_(
                    module.bias
                )

        elif isinstance(module, nn.Embedding):

            torch.nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

    def forward(self, idx):

        B, T = idx.size()

        if T > self.config.block_size:

            raise ValueError(
                f"Sequence length {T} exceeds "
                f"model context length "
                f"{self.config.block_size}"
            )

        pos = torch.arange(
            0,
            T,
            dtype=torch.long,
            device=idx.device
        )

        tok_emb = self.transformer.wte(idx)

        pos_emb = self.transformer.wpe(pos)

        x = tok_emb + pos_emb

        for block in self.transformer.h:

            x = block(x)

        x = self.transformer.ln_f(x)

        logits = self.lm_head(x)

        return logits

    @torch.inference_mode()
    def generate(
        self,
        idx,
        max_new_tokens=256,
        temperature=0.7,
        top_k=40,
    ):

        for _ in range(max_new_tokens):

            idx_cond = idx[
                :,
                -self.config.block_size:
            ]

            logits = self(
                idx_cond
            )

            logits = logits[:, -1, :]

            if temperature != 1.0:

                logits = (
                    logits / temperature
                )

            if top_k is not None:

                v, _ = torch.topk(
                    logits,
                    min(
                        top_k,
                        logits.size(-1)
                    )
                )

                logits[
                    logits < v[:, [-1]]
                ] = float("-inf")

            probs = F.softmax(
                logits,
                dim=-1
            )

            next_token = torch.multinomial(
                probs,
                num_samples=1
            )

            idx = torch.cat(
                (idx, next_token),
                dim=1
            )

        return idx



sys.modules["__main__"].GPTConfig = GPTConfig


_tokenizer = None
_tokenizer_lock = threading.Lock()


def get_tokenizer():

    global _tokenizer

    if _tokenizer is None:

        with _tokenizer_lock:

            if _tokenizer is None:

                print(
                    "[MODEL] Loading GPT-2 tokenizer..."
                )

                _tokenizer = (
                    GPT2Tokenizer.from_pretrained(
                        "gpt2"
                    )
                )

                print(
                    "[MODEL] Tokenizer loaded."
                )

    return _tokenizer



model_cache = {}

_model_lock = threading.Lock()


def list_models() -> list[str]:

    return list(
        MODEL_REGISTRY.keys()
    )



def is_model_available(
    model_name: str
) -> bool:

    # IMPORTANT:
    # This checks whether the model is registered,
    # NOT whether it has already been downloaded.

    return model_name in MODEL_REGISTRY


def download_model(
    model_name: str
) -> str:

    if not is_model_available(
        model_name
    ):

        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: "
            f"{list_models()}"
        )

    info = MODEL_REGISTRY[
        model_name
    ]

    repo_id = info["repo_id"]

    filename = info["filename"]

    model_dir = os.path.join(
        "models",
        model_name
    )

    os.makedirs(
        model_dir,
        exist_ok=True
    )

    local_path = os.path.join(
        model_dir,
        filename
    )


    if os.path.isfile(
        local_path
    ):

        print(
            f"[MODEL] Using local checkpoint: "
            f"{local_path}"
        )

        return local_path


    print(
        f"[MODEL] Downloading {model_name} "
        f"from {repo_id}..."
    )

    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=model_dir,
    )

    print(
        f"[MODEL] Download complete: {path}"
    )

    return path



def normalize_config(
    raw_config: Any
) -> GPTConfig:

    if isinstance(
        raw_config,
        GPTConfig
    ):

        return raw_config

    if isinstance(
        raw_config,
        dict
    ):

        return GPTConfig(
            block_size=raw_config.get(
                "block_size",
                1024
            ),

            vocab_size=raw_config.get(
                "vocab_size",
                50257
            ),

            n_layer=raw_config.get(
                "n_layer",
                12
            ),

            n_head=raw_config.get(
                "n_head",
                8
            ),

            n_embd=raw_config.get(
                "n_embd",
                512
            ),
        )

    required = [
        "block_size",
        "vocab_size",
        "n_layer",
        "n_head",
        "n_embd",
    ]

    if all(
        hasattr(
            raw_config,
            key
        )
        for key in required
    ):

        return GPTConfig(
            block_size=raw_config.block_size,
            vocab_size=raw_config.vocab_size,
            n_layer=raw_config.n_layer,
            n_head=raw_config.n_head,
            n_embd=raw_config.n_embd,
        )

    raise TypeError(
        "Unable to interpret model config "
        f"of type {type(raw_config)}"
    )


def load_model(
    model_name: str
):

    checkpoint_path = download_model(
        model_name
    )

    print(
        f"[MODEL] Loading checkpoint: "
        f"{checkpoint_path}"
    )

    # Legacy checkpoints use pickle and therefore require
    # weights_only=False.
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )


    if (
        isinstance(checkpoint, dict)
        and "config" in checkpoint
    ):

        config = normalize_config(
            checkpoint["config"]
        )

    else:

        raise RuntimeError(
            f"Checkpoint for '{model_name}' "
            "does not contain a recognised "
            "'config'."
        )

    print(
        "[MODEL] Config: "
        f"block_size={config.block_size}, "
        f"vocab_size={config.vocab_size}, "
        f"n_layer={config.n_layer}, "
        f"n_head={config.n_head}, "
        f"n_embd={config.n_embd}"
    )


    model = GPT(
        config
    )

    if "model" in checkpoint:

        state_dict = checkpoint[
            "model"
        ]

    elif "state_dict" in checkpoint:

        state_dict = checkpoint[
            "state_dict"
        ]

    else:

        raise RuntimeError(
            f"Checkpoint for '{model_name}' "
            "does not contain model weights."
        )


    try:

        model.load_state_dict(
            state_dict,
            strict=True
        )

    except RuntimeError as e:

        raise RuntimeError(
            f"Checkpoint architecture does not "
            f"match the current GPT implementation "
            f"for '{model_name}'.\n\n{e}"
        ) from e


    model = model.to(
        DEVICE
    )

    if (
        USE_HALF
        and DEVICE.type == "cuda"
    ):

        model = model.half()

    model.eval()

    del checkpoint
    del state_dict

    if DEVICE.type == "cuda":

        torch.cuda.empty_cache()

    print(
        f"[MODEL] {model_name} loaded on "
        f"{DEVICE}"
    )

    return model



def get_model(
    model_name: str
):

    if not is_model_available(
        model_name
    ):

        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: "
            f"{list_models()}"
        )


    if model_name in model_cache:

        return model_cache[
            model_name
        ]


    with _model_lock:

        # Another request may have loaded it
        # while this thread was waiting.

        if model_name in model_cache:

            return model_cache[
                model_name
            ]

        print(
            f"[MODEL] Initialising "
            f"{model_name}..."
        )

        model = load_model(
            model_name
        )

        model_cache[
            model_name
        ] = model

        return model


def generate_response(
    model,
    message: str,
    max_tokens: int = 256,
    temperature: float = 0.7,
    top_k: int = 40,
):

    tokenizer = get_tokenizer()

    input_ids = tokenizer.encode(
        message,
        return_tensors="pt"
    )

    input_ids = input_ids.to(
        DEVICE
    )

    input_length = (
        input_ids.shape[1]
    )


    with torch.inference_mode():

        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
        )


    generated_ids = (
        output_ids[
            :,
            input_length:
        ]
    )

    response = tokenizer.decode(
        generated_ids[0].tolist(),
        skip_special_tokens=True
    )

    return response.strip()