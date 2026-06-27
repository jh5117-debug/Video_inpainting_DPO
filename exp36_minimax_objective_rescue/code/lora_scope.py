"""Small LoRA module used by Exp36 scope tests."""

from __future__ import annotations

import torch
from torch import nn


class LoRALinear(nn.Module):
    """Frozen base linear layer plus trainable low-rank residual."""

    def __init__(self, base: nn.Linear, rank: int = 2, alpha: int = 4) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.base = base
        for param in self.base.parameters():
            param.requires_grad_(False)
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        self.scale = float(alpha) / float(rank)
        nn.init.normal_(self.lora_a.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_b(self.lora_a(x)) * self.scale


def trainable_parameter_names(module: nn.Module) -> list[str]:
    return [name for name, param in module.named_parameters() if param.requires_grad]

