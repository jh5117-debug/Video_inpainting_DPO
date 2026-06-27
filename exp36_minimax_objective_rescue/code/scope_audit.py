"""Helpers for Exp36 MiniMax trainable-scope audits."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable


LORA_MARKERS = ("lora", "adapter", "peft")


@dataclass(frozen=True)
class TensorScope:
    name: str
    shape: tuple[int, ...]

    @property
    def numel(self) -> int:
        total = 1
        for dim in self.shape:
            total *= int(dim)
        return total


@dataclass(frozen=True)
class ScopePlan:
    name: str
    description: str
    train_full_transformer: bool
    include_families: tuple[str, ...]
    rank: int | None = None
    alpha: int | None = None
    dropout: float = 0.0
    locked_reason: str = ""


def has_lora_marker(name: str) -> bool:
    lowered = name.lower()
    return any(marker in lowered for marker in LORA_MARKERS)


def module_family(name: str) -> str:
    lowered = name.lower()
    if has_lora_marker(lowered):
        return "lora_or_adapter"
    if "attn" in lowered or "attention" in lowered:
        if any(part in lowered for part in (".q", "to_q", "q_proj", "query")):
            return "attention_q"
        if any(part in lowered for part in (".k", "to_k", "k_proj", "key")):
            return "attention_k"
        if any(part in lowered for part in (".v", "to_v", "v_proj", "value")):
            return "attention_v"
        if any(part in lowered for part in ("out", "to_out", "o_proj")):
            return "attention_out"
        return "attention_other"
    if "mlp" in lowered or "ffn" in lowered or "feed_forward" in lowered or "feedforward" in lowered:
        return "mlp"
    if "norm" in lowered:
        return "normalization"
    if "embed" in lowered or "patch" in lowered or "pos" in lowered:
        return "embedding_or_positional"
    if "proj" in lowered or "projection" in lowered:
        return "projection_other"
    return "other"


def module_group(name: str) -> str:
    parts = name.split(".")
    if len(parts) >= 3 and parts[0] in {"blocks", "transformer_blocks"}:
        return ".".join(parts[:2])
    if len(parts) >= 2:
        return ".".join(parts[:2])
    return parts[0]


def summarize_tensors(tensors: Iterable[TensorScope]) -> dict[str, object]:
    rows = list(tensors)
    family_counts: Counter[str] = Counter()
    family_numel: Counter[str] = Counter()
    group_counts: Counter[str] = Counter()
    group_numel: Counter[str] = Counter()
    lora_count = 0
    total_numel = 0
    for tensor in rows:
        family = module_family(tensor.name)
        group = module_group(tensor.name)
        family_counts[family] += 1
        family_numel[family] += tensor.numel
        group_counts[group] += 1
        group_numel[group] += tensor.numel
        total_numel += tensor.numel
        if has_lora_marker(tensor.name):
            lora_count += 1
    return {
        "tensor_count": len(rows),
        "total_numel": total_numel,
        "lora_or_adapter_tensor_count": lora_count,
        "family_counts": dict(sorted(family_counts.items())),
        "family_numel": dict(sorted(family_numel.items())),
        "top_groups_by_numel": [
            {"module_group": name, "tensor_count": group_counts[name], "numel": numel}
            for name, numel in group_numel.most_common(32)
        ],
    }


def exp36_scope_plans() -> dict[str, ScopePlan]:
    return {
        "S0": ScopePlan(
            name="S0",
            description="Current Exp30/Exp35 full MiniMax transformer scope.",
            train_full_transformer=True,
            include_families=("all_transformer_parameters",),
        ),
        "S1": ScopePlan(
            name="S1",
            description="LoRA on DiT self-attention q/k/v/out and output projections.",
            train_full_transformer=False,
            include_families=("attention_q", "attention_k", "attention_v", "attention_out", "projection_other"),
            rank=8,
            alpha=16,
            dropout=0.0,
        ),
        "S2": ScopePlan(
            name="S2",
            description="S1 plus last-four-block MLP LoRA; locked until S1 positive-control.",
            train_full_transformer=False,
            include_families=("attention_q", "attention_k", "attention_v", "attention_out", "projection_other", "mlp"),
            rank=8,
            alpha=16,
            dropout=0.0,
            locked_reason="Only unlock if S1 has weak positive-control evidence.",
        ),
    }


def scope_selects_tensor(plan: ScopePlan, tensor_name: str, block_index: int | None = None) -> bool:
    if plan.train_full_transformer:
        return True
    family = module_family(tensor_name)
    if plan.name == "S2" and family == "mlp":
        return block_index is not None and block_index >= 26
    return family in plan.include_families

