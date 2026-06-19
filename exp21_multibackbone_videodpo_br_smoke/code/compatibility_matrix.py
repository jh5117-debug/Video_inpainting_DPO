"""Generate Exp21 VideoDPO/Diff-DPO plumbing compatibility matrix."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


FIELDS = [
    "model",
    "backend_path",
    "native_target",
    "policy_forward",
    "reference_forward",
    "zero_init_parity",
    "finite_dpo_loss",
    "finite_grad",
    "save_reload",
    "inference_with_adapter",
    "status",
    "blocker",
    "next_action",
]


DEFAULT_ROWS = [
    {
        "model": "DiffuEraser",
        "backend_path": "exp21_multibackbone_videodpo_br_smoke/backends/diffueraser",
        "native_target": "epsilon/noise residual as used by project trainer",
        "policy_forward": "existing backend; wrapper pending",
        "reference_forward": "existing frozen ref; wrapper pending",
        "zero_init_parity": "pending",
        "finite_dpo_loss": "pending",
        "finite_grad": "pending",
        "save_reload": "pending",
        "inference_with_adapter": "pending",
        "status": "PENDING_REAL_SMOKE",
        "blocker": "",
        "next_action": "wrap existing DiffuEraser trainer interface without changing shared code",
    },
    {"model": "ProPainter", "status": "NOT_APPLICABLE_NON_DIFFUSION", "blocker": "non-diffusion propagation baseline"},
    {"model": "EffectErase", "status": "WAITING_AUTH", "blocker": "VOR data authorization pending; code/weight readiness only"},
    {"model": "FloED", "status": "PENDING_AUDIT", "blocker": "training forward / public checkpoint not yet verified"},
    {"model": "CoCoCo", "status": "PENDING_AUDIT", "blocker": "SD inpainting dependency and trainable modules need isolated env"},
    {"model": "VideoComposer", "status": "PENDING_AUDIT", "blocker": "exact repo/checkpoint/inference entry must be disambiguated"},
    {"model": "VACE", "status": "PENDING_AUDIT", "blocker": "Wan/VACE flow-matching forward and LoRA path need audit"},
    {"model": "MiniMax-Remover", "status": "PENDING_AUDIT", "blocker": "independent env and transformer forward need audit"},
    {"model": "VideoPainter", "status": "PENDING_AUDIT", "blocker": "Exp14 adapter was negative; new smoke only, no quality claim"},
]


def write(rows: list[dict[str, str]], csv_path: Path, md_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    norm_rows = [{field: row.get(field, "") for field in FIELDS} for row in rows]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(norm_rows)
    lines = ["# Exp21 multibackbone DPO compatibility", "", "| " + " | ".join(FIELDS) + " |", "| " + " | ".join(["---"] * len(FIELDS)) + " |"]
    for row in norm_rows:
        lines.append("| " + " | ".join(row[field] for field in FIELDS) + " |")
    md_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="reports/exp21_multibackbone_dpo_compatibility.csv")
    parser.add_argument("--md", default="reports/exp21_multibackbone_dpo_compatibility.md")
    args = parser.parse_args()
    write(DEFAULT_ROWS, Path(args.csv), Path(args.md))
    print(f"wrote {args.csv} and {args.md}")


if __name__ == "__main__":
    main()
