"""
Base class for all frontier architectures.

Every architecture in the frontier system must inherit from FrontierModel
and implement the required interface. This ensures all architectures can
be trained, evaluated, and compared using the same infrastructure.
"""

import torch
import torch.nn as nn
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple


@dataclass
class FrontierConfig:
    """Base config shared by all frontier architectures."""
    # Architecture
    d_model: int = 512
    n_layers: int = 22
    d_ff: int = 2048
    vocab_size: int = 49152
    max_seq_len: int = 2048
    dropout: float = 0.0
    tie_weights: bool = True

    # Training (inherited from ablation system)
    compile_model: bool = True
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    train_tokens: int = 6000000
    muon_lr: float = 0.024
    muon_momentum: float = 0.95
    adamw_lr: float = 0.006
    warmup_ratio: float = 0.02
    schedule_type: str = "linear"
    weight_decay: float = 0.2
    grad_clip: float = 1.0
    use_amp: bool = True

    # Evaluation
    eval_every: Optional[int] = None
    eval_steps: int = 100
    eval_milestones: Optional[Tuple[int, ...]] = None

    # Logging
    log_milestones: Tuple[int, ...] = (100, 500, 1000)

    # Architecture-specific (override in subclass configs)
    arch_family: str = "base"
    arch_config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.d_k = self.d_model // max(getattr(self, 'n_heads', 8), 1)

    def param_budget(self) -> int:
        """Target parameter count (~88M)."""
        return 88_000_000

    def to_dict(self) -> Dict[str, Any]:
        """Serialize config for metrics recording."""
        result = {}
        for k, v in self.__dict__.items():
            if isinstance(v, (int, float, str, bool, type(None))):
                result[k] = v
            elif isinstance(v, (list, tuple)):
                result[k] = list(v)
            elif isinstance(v, dict):
                result[k] = v
        return result


class FrontierModel(nn.Module, ABC):
    """
    Base class for all frontier architecture models.

    Subclasses must implement:
    - forward(x) -> logits of shape (batch, seq_len, vocab_size)
    - arch_family() -> string identifying the architecture family
    - describe() -> human-readable description of the architecture
    """

    def __init__(self, config: FrontierConfig):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: input token ids, shape (batch, seq_len)

        Returns:
            logits: shape (batch, seq_len, vocab_size)
        """
        ...

    @classmethod
    @abstractmethod
    def arch_family(cls) -> str:
        """Return the architecture family name (e.g., 'state_space', 'linear_attention')."""
        ...

    @abstractmethod
    def describe(self) -> str:
        """Return a human-readable one-line description of this architecture variant."""
        ...

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameter_breakdown(self) -> Dict[str, int]:
        """Return parameter counts by module group."""
        breakdown = {}
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            group = name.split('.')[0]
            breakdown[group] = breakdown.get(group, 0) + param.numel()
        return breakdown

    def estimate_flops_per_token(self) -> int:
        """
        Estimate FLOPs per token for a forward pass.
        Override in subclasses for accurate estimates.
        Default: rough estimate based on parameter count.
        """
        return self.count_parameters() * 2  # rough: 2 FLOPs per param per token

    def supports_recurrent_inference(self) -> bool:
        """Whether this architecture supports O(1)-per-token recurrent inference."""
        return False

    def sequence_mixing_complexity(self) -> str:
        """Return the sequence mixing complexity class."""
        return "O(n^2)"  # default for attention-based

    def get_optimizer_groups(self) -> Tuple[list, list]:
        """
        Split parameters into Muon-eligible (2D matrices) and AdamW (rest).
        Override if the architecture needs different optimizer assignment.
        """
        muon_params = []
        adamw_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if (param.ndim == 2
                    and 'token_embedding' not in name
                    and 'embedding' not in name
                    and 'norm' not in name):
                muon_params.append(param)
            else:
                adamw_params.append(param)
        return muon_params, adamw_params


class EmbeddingWithScale(nn.Module):
    """Token embedding with optional sqrt(d_model) scaling, shared across all architectures."""

    def __init__(self, vocab_size: int, d_model: int, scale: bool = True, dropout: float = 0.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.scale = math.sqrt(d_model) if scale else 1.0
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.embedding(x) * self.scale)


class LMHead(nn.Module):
    """Language model head, optionally tied to embedding weights."""

    def __init__(self, d_model: int, vocab_size: int, embedding_weight: Optional[nn.Parameter] = None):
        super().__init__()
        if embedding_weight is not None:
            self.proj = nn.Linear(d_model, vocab_size, bias=False)
            self.proj.weight = embedding_weight
        else:
            self.proj = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)
