#!/usr/bin/env python
"""Select the best VideoDPO checkpoint from a VBench sweep."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path


def load_manifest(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                raise ValueError(f"Bad manifest line in {path}: {line}")
            rows.append({"label": parts[0], "checkpoint": parts[1]})
    return rows


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "symlink":
        os.symlink(src.resolve(), dst)
    elif mode == "hardlink":
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unknown link mode: {mode}")


def load_score(summary_path: Path, score_key: str) -> dict[str, object]:
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    score = summary.get(score_key)
    if score is None:
        raise ValueError(f"{summary_path} does not contain score key {score_key!r}")
    return {
        "summary_path": str(summary_path),
        "score": float(score),
        "quality_score": summary.get("quality_score"),
        "semantic_score": summary.get("semantic_score"),
        "total_score": summary.get("total_score"),
        "quality_score_percent": summary.get("quality_score_percent"),
        "semantic_score_percent": summary.get("semantic_score_percent"),
        "total_score_percent": summary.get("total_score_percent"),
        "mean_score": summary.get("mean_score"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_root", required=True, type=Path)
    parser.add_argument("--checkpoint_manifest", required=True, type=Path)
    parser.add_argument("--train_run_dir", required=True, type=Path)
    parser.add_argument("--selection_dir", required=True, type=Path)
    parser.add_argument("--score_key", default="total_score")
    parser.add_argument("--link_mode", choices=["hardlink", "copy", "symlink"], default="hardlink")
    parser.add_argument("--prune_trainstep_checkpoints", action="store_true")
    args = parser.parse_args()

    entries = load_manifest(args.checkpoint_manifest)
    if not entries:
        raise RuntimeError(f"No checkpoints found in manifest: {args.checkpoint_manifest}")

    ranked = []
    for entry in entries:
        label = entry["label"]
        ckpt = Path(entry["checkpoint"])
        summary_path = args.sweep_root / label / "vbench_eval" / "summary.json"
        score_info = load_score(summary_path, args.score_key)
        ranked.append({**entry, **score_info, "checkpoint": str(ckpt)})

    ranked.sort(key=lambda row: row["score"], reverse=True)
    best = ranked[0]
    best_ckpt = Path(best["checkpoint"])
    last_ckpt = args.train_run_dir / "checkpoints" / "last.ckpt"

    args.selection_dir.mkdir(parents=True, exist_ok=True)
    link_or_copy(best_ckpt, args.selection_dir / "best_vbench_total.ckpt", args.link_mode)
    shutil.copy2(Path(best["summary_path"]), args.selection_dir / "best_vbench_summary.json")
    if last_ckpt.exists():
        link_or_copy(last_ckpt, args.selection_dir / "last.ckpt", args.link_mode)

    ranking_csv = args.selection_dir / "vbench_ranking.csv"
    with ranking_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "label",
            "checkpoint",
            "score",
            "total_score",
            "total_score_percent",
            "quality_score",
            "quality_score_percent",
            "semantic_score",
            "semantic_score_percent",
            "mean_score",
            "summary_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in ranked:
            writer.writerow({k: row.get(k) for k in fieldnames})

    selection = {
        "score_key": args.score_key,
        "best": best,
        "last_checkpoint": str(last_ckpt) if last_ckpt.exists() else None,
        "selection_dir": str(args.selection_dir),
        "ranking_csv": str(ranking_csv),
        "pruned_trainstep_checkpoints": False,
    }

    if args.prune_trainstep_checkpoints:
        best_resolved = best_ckpt.resolve()
        pruned = []
        for entry in entries:
            ckpt = Path(entry["checkpoint"])
            if not ckpt.exists():
                continue
            if ckpt.resolve() == best_resolved:
                continue
            ckpt.unlink()
            pruned.append(str(ckpt))
        selection["pruned_trainstep_checkpoints"] = True
        selection["pruned"] = pruned

    selection_path = args.selection_dir / "selection.json"
    with selection_path.open("w", encoding="utf-8") as f:
        json.dump(selection, f, indent=2, ensure_ascii=False)

    print(json.dumps(selection, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
