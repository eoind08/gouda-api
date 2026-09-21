import threading

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    LogitsProcessor,
    LogitsProcessorList,
)


# ============================================================
# Model registry
# ============================================================

MODELS = {
    "Gruyere-1.1": "Erbium08/Gruyere-1.1",
    # "Gruyere-1.2": "Erbium08/Gruyere-1.2",
    # "Ricotta-1.0": "Erbium08/Ricotta-1.0",
}


# ============================================================
# Runtime configuration
# ============================================================

DEVICE = torch.device("cpu")
DTYPE = torch.float32

_models = {}
_tokenizers = {}
_load_locks = {}


# ============================================================
# Custom logits processors
# ============================================================

class FrequencyPenaltyLogitsProcessor(LogitsProcessor):

    def __init__(self, penalty: float):
        self.penalty = penalty

    def __call__(self, input_ids, scores):
        if self.penalty == 0:
            return scores

        for batch_idx in range(input_ids.shape[0]):
            token_counts = torch.bincount(
                input_ids[batch_idx],
                minlength=scores.shape[-1],
            ).to(scores.device)

            scores[batch_idx] -= self.penalty * token_counts

        return scores


class PresencePenaltyLogitsProcessor(LogitsProcessor):

    def __init__(self, penalty: float):
        self.penalty = penalty

    def __call__(self, input_ids, scores):
        if self.penalty == 0:
            return scores

        for batch_idx in range(input_ids.shape[0]):
            unique_tokens = torch.unique(input_ids[batch_idx])

            scores[batch_idx, unique_tokens] -= self.penalty

        return scores


# ============================================================
# Model loading
# ============================================================

def load_model(model_name: str):

    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}")

    if model_name in _models:
        return _models[model_name], _tokenizers[model_name]

    lock = _load_locks.setdefault(
        model_name,
        threading.Lock(),
    )

    with lock:

        if model_name in _models:
            return _models[model_name], _tokenizers[model_name]

        repo_id = MODELS[model_name]

        print(
            f"[MODEL] Loading {model_name} "
            f"from {repo_id}..."
        )

        tokenizer = AutoTokenizer.from_pretrained(
            repo_id,
            trust_remote_code=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            repo_id,
            trust_remote_code=True,
            torch_dtype=DTYPE,
        )

        model.to(DEVICE)
        model.eval()

        _models[model_name] = model
        _tokenizers[model_name] = tokenizer

        print(
            f"[MODEL] {model_name} loaded "
            f"on {DEVICE}"
        )

        return model, tokenizer


def get_model(model_name: str):
    model, _ = load_model(model_name)
    return model


def get_tokenizer(model_name: str):
    _, tokenizer = load_model(model_name)
    return tokenizer


def list_models():
    return list(MODELS.keys())


def is_model_available(model_name: str):
    return model_name in MODELS


# ============================================================
# Generation helpers
# ============================================================

def _build_generation_kwargs(
    tokenizer,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int | None,
    repetition_penalty: float,
    frequency_penalty: float,
    presence_penalty: float,
    seed: int | None,
    stop: list[str] | None,
):

    do_sample = temperature > 0

    kwargs = {
        "max_new_tokens": max_tokens,
        "use_cache": True,
        "do_sample": do_sample,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.eos_token_id,
        "repetition_penalty": repetition_penalty,
    }

    if do_sample:
        kwargs["temperature"] = temperature
        kwargs["top_p"] = top_p

        if top_k is not None and top_k > 0:
            kwargs["top_k"] = top_k

    if frequency_penalty != 0 or presence_penalty != 0:

        processors = LogitsProcessorList()

        if frequency_penalty != 0:
            processors.append(
                FrequencyPenaltyLogitsProcessor(
                    frequency_penalty
                )
            )

        if presence_penalty != 0:
            processors.append(
                PresencePenaltyLogitsProcessor(
                    presence_penalty
                )
            )

        kwargs["logits_processor"] = processors

    if stop:
        kwargs["stop_strings"] = stop
        kwargs["tokenizer"] = tokenizer

    if seed is not None:
        torch.manual_seed(seed)

    return kwargs


def _prepare_inputs(
    tokenizer,
    message: str,
):

    inputs = tokenizer(
        message,
        return_tensors="pt",
        add_special_tokens=False,
    )

    return {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }


def _validate_context_length(
    model,
    prompt_tokens: int,
    max_tokens: int,
):

    config = model.config

    block_size = getattr(
        config,
        "max_position_embeddings",
        getattr(config, "block_size", None),
    )

    if block_size is None:
        return

    if prompt_tokens >= block_size:
        raise ValueError(
            f"Input is too long. "
            f"Maximum context length is {block_size} tokens."
        )

    available_tokens = block_size - prompt_tokens

    if max_tokens > available_tokens:
        raise ValueError(
            f"max_tokens is too large for this prompt. "
            f"Prompt uses {prompt_tokens} tokens and the model "
            f"has a maximum context of {block_size} tokens, "
            f"leaving {available_tokens} tokens available."
        )


# ============================================================
# Non-streaming generation
# ============================================================

@torch.inference_mode()
def generate_response(
    model_name: str,
    message: str,
    max_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 1.0,
    top_k: int | None = 40,
    repetition_penalty: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    seed: int | None = None,
    stop: list[str] | None = None,
):

    model, tokenizer = load_model(model_name)

    inputs = _prepare_inputs(
        tokenizer,
        message,
    )

    input_ids = inputs["input_ids"]

    prompt_tokens = input_ids.shape[1]

    _validate_context_length(
        model,
        prompt_tokens,
        max_tokens,
    )

    generation_kwargs = _build_generation_kwargs(
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        seed=seed,
        stop=stop,
    )

    output = model.generate(
        **inputs,
        **generation_kwargs,
    )

    generated_ids = output[
        0,
        prompt_tokens:
    ]

    completion_tokens = generated_ids.shape[0]

    text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )

    return {
        "text": text,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": (
                prompt_tokens +
                completion_tokens
            ),
        },
    }


# ============================================================
# Streaming generation
# ============================================================

def generate_response_stream(
    model_name: str,
    message: str,
    max_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 1.0,
    top_k: int | None = 40,
    repetition_penalty: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    seed: int | None = None,
    stop: list[str] | None = None,
):

    model, tokenizer = load_model(model_name)

    inputs = _prepare_inputs(
        tokenizer,
        message,
    )

    prompt_tokens = inputs["input_ids"].shape[1]

    _validate_context_length(
        model,
        prompt_tokens,
        max_tokens,
    )

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
        timeout=300.0,
    )

    generation_kwargs = _build_generation_kwargs(
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        seed=seed,
        stop=stop,
    )

    generation_kwargs["streamer"] = streamer

    result = {}

    def generate():

        try:

            with torch.inference_mode():

                result["output"] = model.generate(
                    **inputs,
                    **generation_kwargs,
                )

        except Exception as e:

            result["error"] = e

            # Force the streamer to finish so that
            # the consuming request does not hang.
            streamer.end()

    thread = threading.Thread(
        target=generate,
        daemon=True,
    )

    thread.start()

    for text in streamer:
        yield text

    thread.join()

    if "error" in result:
        raise result["error"]