import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from transformers import GPT2Tokenizer
import os
import sys
from huggingface_hub import hf_hub_download

# === GPTConfig ===
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

# === MLP, Attention, Block ===
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

# === Full GPT Model ===
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        B, T = idx.size()
        assert T <= self.config.block_size
        pos = torch.arange(0, T, device=idx.device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx

# Make GPTConfig available to the pickle loader
sys.modules['__main__'].GPTConfig = GPTConfig


# Map model_name from request to Hugging Face repo ID
HF_REPO_MAP = {
    "gouda0.0.1": "Erbium08/gouda0.0.1",
    # Add more here if needed:
    # "cheddar1.0": "Erbium08/cheddar1.0"
}

def download_model(model_name):
    if model_name not in HF_REPO_MAP:
        raise ValueError(f"Unknown model: {model_name}")

    repo_id = HF_REPO_MAP[model_name]

    # Create local cache directory
    model_dir = f"models/{model_name}"
    os.makedirs(model_dir, exist_ok=True)

    # Download model.pt from Hugging Face Hub
    model_path = hf_hub_download(
        repo_id=repo_id,
        filename="model.pt",
        cache_dir=model_dir,
        local_dir=model_dir,
        force_filename="model.pt"
    )

    return model_path


# === Load model + tokenizer ===
model_cache = {}

def load_model(model_name):
    model_path = download_model(model_name)
    
    try:
        # Try to add GPTConfig to safe globals (for newer PyTorch)
        import torch.serialization
        torch.serialization.add_safe_globals([GPTConfig])
    except (ImportError, AttributeError):
        # Skip if not supported
        pass
    
    # Load checkpoint with weights_only=False
    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    except TypeError:
        # Fallback for older PyTorch versions
        checkpoint = torch.load(model_path, map_location="cpu")
    
    config = checkpoint["config"]
    model = GPT(config)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    return tokenizer, model

def generate_response(model_pair, message, max_tokens=100):
    tokenizer, model = model_pair
    input_ids = tokenizer.encode(message, return_tensors="pt")
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=max_tokens)
    output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output[len(message):].strip()