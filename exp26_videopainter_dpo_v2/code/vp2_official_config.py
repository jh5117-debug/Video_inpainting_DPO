#!/usr/bin/env python3
"""Parse VideoPainter official training defaults for Exp26 parity gates."""

from __future__ import annotations

import ast
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class OfficialOptimizerSchedulerConfig:
    learning_rate: float | None
    adam_beta1: float | None
    adam_beta2: float | None
    adam_epsilon: float | None
    weight_decay: float | None
    lr_scheduler: str | None
    lr_warmup_steps: int | None
    max_grad_norm: float | None
    mixed_precision: str | None


ARG_ALIASES = {
    "learning_rate": ["learning_rate", "lr"],
    "adam_beta1": ["adam_beta1", "adam_betas_0", "beta1"],
    "adam_beta2": ["adam_beta2", "adam_betas_1", "beta2"],
    "adam_epsilon": ["adam_epsilon", "adam_eps", "epsilon", "eps"],
    "weight_decay": ["weight_decay", "adam_weight_decay"],
    "lr_scheduler": ["lr_scheduler", "scheduler"],
    "lr_warmup_steps": ["lr_warmup_steps", "warmup_steps"],
    "max_grad_norm": ["max_grad_norm", "gradient_clip", "gradient_clipping"],
    "mixed_precision": ["mixed_precision", "precision"],
}


def _literal(node: ast.AST):
    try:
        return ast.literal_eval(node)
    except Exception:  # noqa: BLE001
        return None


def _arg_name(call: ast.Call) -> str | None:
    if not call.args:
        return None
    raw = _literal(call.args[0])
    if not isinstance(raw, str) or not raw.startswith("--"):
        return None
    return raw.lstrip("-").replace("-", "_")


def parse_argparse_defaults(train_file: Path) -> dict[str, object]:
    tree = ast.parse(train_file.read_text())
    defaults: dict[str, object] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        attr = getattr(func, "attr", "")
        if attr != "add_argument":
            continue
        name = _arg_name(node)
        if not name:
            continue
        for kw in node.keywords:
            if kw.arg == "default":
                defaults[name] = _literal(kw.value)
                break
        else:
            # store_true/store_false action defaults are useful parity metadata.
            for kw in node.keywords:
                if kw.arg == "action":
                    action = _literal(kw.value)
                    if action == "store_true":
                        defaults[name] = False
                    elif action == "store_false":
                        defaults[name] = True
    return defaults


def _lookup(defaults: dict[str, object], canonical: str):
    for key in ARG_ALIASES[canonical]:
        if key in defaults:
            return defaults[key]
    return None


def parse_official_optimizer_scheduler_config(train_file: Path) -> OfficialOptimizerSchedulerConfig:
    defaults = parse_argparse_defaults(train_file)
    return OfficialOptimizerSchedulerConfig(
        learning_rate=_lookup(defaults, "learning_rate"),
        adam_beta1=_lookup(defaults, "adam_beta1"),
        adam_beta2=_lookup(defaults, "adam_beta2"),
        adam_epsilon=_lookup(defaults, "adam_epsilon"),
        weight_decay=_lookup(defaults, "weight_decay"),
        lr_scheduler=_lookup(defaults, "lr_scheduler"),
        lr_warmup_steps=_lookup(defaults, "lr_warmup_steps"),
        max_grad_norm=_lookup(defaults, "max_grad_norm"),
        mixed_precision=_lookup(defaults, "mixed_precision"),
    )


def write_locked_config(train_file: Path, output_json: Path) -> OfficialOptimizerSchedulerConfig:
    cfg = parse_official_optimizer_scheduler_config(train_file)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(asdict(cfg), indent=2, sort_keys=True) + "\n")
    return cfg
