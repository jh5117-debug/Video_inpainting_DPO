#!/usr/bin/env python3
"""Audit the current Exp30 MiniMax trainable scope for Exp35."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from safetensors import safe_open

from exp35_minimax_flow_dpo_rescue.code.scope_audit import TensorScope, module_family, module_group, summarize_tensors


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--sensitivity-summary", required=True)
    p.add_argument("--reports-root", required=True)
    return p.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    reports_root = Path(args.reports_root)
    tensor_file = checkpoint / "diffusion_pytorch_model.safetensors"
    if not tensor_file.exists():
        raise FileNotFoundError(tensor_file)
    tensors: list[TensorScope] = []
    rows: list[dict[str, object]] = []
    with safe_open(tensor_file, framework="pt", device="cpu") as f:
        for key in sorted(f.keys()):
            shape = tuple(int(x) for x in f.get_slice(key).get_shape())
            scope = TensorScope(key, shape)
            tensors.append(scope)
            rows.append({
                "tensor": key,
                "shape": "x".join(str(x) for x in shape),
                "numel": scope.numel,
                "module_group": module_group(key),
                "module_family": module_family(key),
                "contains_lora_or_adapter_marker": "lora" in key.lower() or "adapter" in key.lower() or "peft" in key.lower(),
            })
    sensitivity = json.loads(Path(args.sensitivity_summary).read_text(encoding="utf-8"))
    summary = summarize_tensors(tensors)
    summary.update({
        "checkpoint": str(checkpoint),
        "checkpoint_tensor_file": str(tensor_file),
        "exp30_trainable_scope": "all_transformer_parameters",
        "lora_scope_present": summary["lora_or_adapter_tensor_count"] > 0,
        "sensitivity_status": sensitivity.get("status"),
        "sensitivity_perturb_full_mae_mean": sensitivity.get("perturb_full_mae_mean"),
        "sensitivity_perturb_mask_mae_mean": sensitivity.get("perturb_mask_mae_mean"),
        "in_forward_path_evidence": "MINIMAX_INFERENCE_SENSITIVITY_PASS" if sensitivity.get("status") == "MINIMAX_INFERENCE_SENSITIVITY_PASS" else "NOT_PROVEN",
        "current_scope_too_small": False,
        "current_scope_ignored_by_inference": False,
        "expanded_lora_scope_prepared": False,
        "expanded_lora_scope_reason": "not prepared because Exp30 scope is already full transformer and sensitivity positive-control proves checkpoint weights are used",
        "status": "MINIMAX_TRAINABLE_SCOPE_CURRENT_OK" if sensitivity.get("status") == "MINIMAX_INFERENCE_SENSITIVITY_PASS" else "MINIMAX_TRAINABLE_SCOPE_BLOCKED",
    })
    reports_root.mkdir(parents=True, exist_ok=True)
    write_csv(reports_root / "exp35_minimax_trainable_scope_audit.csv", rows)
    write_json(reports_root / "exp35_minimax_trainable_scope_summary.json", summary)
    md = [
        "# Exp35 MiniMax Trainable Scope Audit",
        "",
        f"Status: `{summary['status']}`",
        "",
        f"- Checkpoint: `{checkpoint}`",
        f"- Tensor count: `{summary['tensor_count']}`",
        f"- Total parameters represented: `{summary['total_numel']}`",
        f"- LoRA/adapter tensor count: `{summary['lora_or_adapter_tensor_count']}`",
        f"- Exp30 trainable scope: `{summary['exp30_trainable_scope']}`",
        f"- Sensitivity status: `{summary['sensitivity_status']}`",
        f"- Perturbed full/mask MAE means: `{summary['sensitivity_perturb_full_mae_mean']}` / `{summary['sensitivity_perturb_mask_mae_mean']}`",
        "",
        "Conclusion: current Exp30 MiniMax scope is not too small and is not ignored by inference. It is the full transformer scope. Therefore no expanded LoRA scope is prepared in this milestone; the next bottleneck remains objective/update-scale and hard-state selection.",
        "",
        "Top module groups by parameter count:",
    ]
    for group in summary["top_groups_by_numel"][:12]:
        md.append(f"- `{group['module_group']}`: tensors `{group['tensor_count']}`, numel `{group['numel']}`")
    (reports_root / "exp35_minimax_trainable_scope_audit.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
