"""Model asset scanner for Exp22.

This is a conservative scanner: it records existing repos, weights, and env
clues. It does not download, delete, or mutate assets.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import subprocess
from pathlib import Path
from typing import Iterable


MODELS = [
    "DiffuEraser",
    "FloED",
    "CoCoCo",
    "VideoComposer",
    "VACE",
    "MiniMax-Remover",
    "EffectErase",
    "ProPainter",
    "VideoPainter",
]

SEARCH_ROOTS = [
    "/home/hj",
    "/home/hj/dpo-2-1-exp",
    "/mnt/workspace/hj/nas_hj",
    "/mnt/nas/hj",
]

OFFICIAL = {
    "DiffuEraser": "",
    "FloED": "",
    "CoCoCo": "",
    "VideoComposer": "",
    "VACE": "",
    "MiniMax-Remover": "https://github.com/zibojia/MiniMax-Remover",
    "EffectErase": "",
    "ProPainter": "https://github.com/sczhou/ProPainter",
    "VideoPainter": "https://huggingface.co/TencentARC/VideoPainter",
}

KNOWN_ASSETS = {
    "DiffuEraser": {
        "repo": [
            "/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp20_autoresearch",
            "/home/hj/H20_Video_inpainting_DPO_exp20_autoresearch",
        ],
        "weight": [
            "/mnt/workspace/hj/nas_hj/weights/diffuEraser/converted_weights_step48000",
            "/mnt/nas/hj/weights/diffuEraser/converted_weights_step48000",
            "/home/hj/weights/diffuEraser/converted_weights_step48000",
        ],
        "inference_entry": "tools/run_davis50_framewise_protocol_eval.py",
        "BR_supported": "yes",
        "raw_output_supported": "yes",
        "training_forward_available": "yes",
    },
    "VideoPainter": {
        "repo": [
            "/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp14_videopainter_gate/third_party/VideoPainter",
            "/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp20_autoresearch/third_party/VideoPainter",
        ],
        "weight": [
            "/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp14_videopainter_gate/third_party/VideoPainter/ckpt/VideoPainter/checkpoints/branch",
        ],
    },
    "MiniMax-Remover": {
        "repo": ["/mnt/nas/hj/official_repos/MiniMax-Remover"],
        "weight": ["/mnt/nas/hj/H20_Video_inpainting_DPO/weights/minimax_remover/current"],
    },
}


FIELDS = [
    "model",
    "official_repo",
    "commit",
    "architecture",
    "native_prediction_target",
    "HAL_repo",
    "PAI_repo",
    "HAL_weight",
    "PAI_weight",
    "weight_sha256",
    "env",
    "inference_entry",
    "BR_supported",
    "OR_supported",
    "raw_output_supported",
    "training_forward_available",
    "VideoDPO_smoke_status",
    "blocker",
    "next_action",
]


def run_git_commit(path: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True, timeout=10).strip()
    except Exception:
        return ""


SKIP_DIR_NAMES = {
    ".git",
    "__pycache__",
    "JPEGImages",
    "Annotations",
    "raw_frames",
    "comp_frames",
    "videos",
    "frame_by_frame",
    "contact_sheets",
}


def find_candidates(name: str, roots: Iterable[str], max_depth: int = 5) -> list[Path]:
    lowered = name.lower().replace("-", "").replace("_", "")
    hits = []
    for root in roots:
        rp = Path(root)
        if not rp.exists():
            continue
        base_depth = len(rp.parts)
        for current, dirs, _files in os.walk(rp):
            current_path = Path(current)
            depth = len(current_path.parts) - base_depth
            if depth > max_depth:
                dirs[:] = []
                continue
            dirs[:] = [d for d in dirs if d not in SKIP_DIR_NAMES and not d.startswith(".cache")]
            path = current_path
            if len(hits) > 30:
                return hits
            key = path.name.lower().replace("-", "").replace("_", "")
            if lowered in key or key in lowered:
                hits.append(path)
    return hits


def first_existing(paths: Iterable[Path]) -> str:
    for p in paths:
        if p.exists():
            return str(p)
    return ""


def sha256_small(path: str) -> str:
    if not path:
        return ""
    p = Path(path)
    if not p.is_file() or p.stat().st_size > 2 * 1024**3:
        return ""
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def infer_row(model: str, roots: list[str], max_depth: int) -> dict[str, str]:
    hits = find_candidates(model, roots, max_depth=max_depth)
    known = KNOWN_ASSETS.get(model, {})
    repo = first_existing([Path(p) for p in known.get("repo", [])])
    if not repo:
        repo = first_existing([p for p in hits if (p / ".git").exists()])
    commit = run_git_commit(Path(repo)) if repo else ""
    weight = first_existing([Path(p) for p in known.get("weight", [])])
    if not weight:
        weight = first_existing(
        [
            p
            for p in hits
            if any(x in p.name.lower() for x in ["ckpt", "weight", "checkpoint", "current", "converted"])
        ]
        )
    if model == "EffectErase":
        status = "WAITING_AUTH"
        blocker = "VOR/VOR-Eval/VOR-Wild authorization pending"
    elif model == "ProPainter":
        status = "NOT_APPLICABLE_NON_DIFFUSION"
        blocker = "non-diffusion baseline; no Diff-DPO plumbing"
    elif repo and weight:
        status = "READY_ASSET_PENDING_SMOKE"
        blocker = ""
    elif repo:
        status = "BLOCKED_NO_WEIGHT"
        blocker = "repo found; complete checkpoint not verified"
    else:
        status = "BLOCKED_NO_REPO"
        blocker = "repo not found in scanned roots"
    return {
        "model": model,
        "official_repo": OFFICIAL.get(model, ""),
        "commit": commit,
        "architecture": "",
        "native_prediction_target": "",
        "HAL_repo": repo if repo.startswith("/home") else "",
        "PAI_repo": repo if repo.startswith("/mnt") else "",
        "HAL_weight": weight if weight.startswith("/home") else "",
        "PAI_weight": weight if weight.startswith("/mnt") else "",
        "weight_sha256": sha256_small(weight),
        "env": "",
        "inference_entry": known.get("inference_entry", ""),
        "BR_supported": known.get("BR_supported", "unknown"),
        "OR_supported": "unknown",
        "raw_output_supported": known.get("raw_output_supported", "unknown"),
        "training_forward_available": known.get("training_forward_available", "unknown"),
        "VideoDPO_smoke_status": status,
        "blocker": blocker,
        "next_action": (
            "run official real-weight inference smoke"
            if status in {"PENDING_REAL_SMOKE", "READY_ASSET_PENDING_SMOKE"}
            else "resolve blocker"
        ),
    }


def write_outputs(rows: list[dict[str, str]], csv_path: Path, md_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Exp22 model asset matrix", "", "| " + " | ".join(FIELDS) + " |", "| " + " | ".join(["---"] * len(FIELDS)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")).replace("\n", " ") for field in FIELDS) + " |")
    md_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="reports/exp22_model_asset_matrix.csv")
    parser.add_argument("--md", default="reports/exp22_model_asset_matrix.md")
    parser.add_argument("--roots", nargs="*", default=SEARCH_ROOTS)
    parser.add_argument("--max-depth", type=int, default=5)
    args = parser.parse_args()
    rows = [infer_row(model, args.roots, args.max_depth) for model in MODELS]
    write_outputs(rows, Path(args.csv), Path(args.md))
    print(f"wrote {args.csv} and {args.md}")


if __name__ == "__main__":
    main()
