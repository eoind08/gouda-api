from dataclasses import dataclass
import torch.nn as nn

@dataclass
class ModelConfig:
    """Base class for all model configurations"""
    pass

class BaseModel(nn.Module):
    """Base class for all models"""
    pass

# gpt_model.py
from .base import ModelConfig, BaseModel
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class GPTConfig(ModelConfig):
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class GPT(BaseModel):
    # Your GPT implementation
    pass