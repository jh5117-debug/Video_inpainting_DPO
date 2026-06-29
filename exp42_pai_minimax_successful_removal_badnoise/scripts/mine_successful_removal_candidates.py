#!/usr/bin/env python3
"""Mine MiniMax successful-removal and failure candidates for Exp42.

This runner is inference-only. It reuses the existing project MiniMax
`run_pipeline` helper so the executable protocol matches Exp30/Exp40 and the
shared metric code remains untouched.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--source-manifest", action="append", required=True, help="NAME=PATH; may be repeated")
    parser.add_argument("--limit-sources", type=int, default=64)
    parser.add_argument("--seeds", default="20260629,20260630,20260631,20260632")
    parser.add_argument("--num-inference-steps", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--reports-root", required=True)
    parser.add_argument("--repo-manifest-root", required=True)
    parser.add_argument("--heartbeat", default="")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def heartbeat(path: Path | None, message: str) -> None:
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{time.time():.0f}\t{message}\n", encoding="utf-8")


def torch_dtype(name: str) -> torch.dtype:
    return torch.float16 if name == "float16" else torch.bfloat16


def parse_manifest_args(values: list[str]) -> list[tuple[str, Path]]:
    parsed: list[tuple[str, Path]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"source manifest must be NAME=PATH, got {value!r}")
        name, path = value.split("=", 1)
        parsed.append((name.strip(), Path(path).resolve()))
    return parsed


def parse_seeds(value: str) -> list[int]:
    seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not seeds:
        raise ValueError("at least one seed is required")
    return seeds


def row_group(row: dict[str, object]) -> str:
    for key in ("scene_group", "source_group", "group_id"):
        val = row.get(key)
        if val:
            return str(val)
    sample = str(row.get("sample_id", ""))
    return sample.rsplit("_", 1)[0] if "_" in sample else sample


def load_source_rows(manifest_args: list[tuple[str, Path]], limit: int) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    manifest_records: list[dict[str, object]] = []
    seen_groups: set[str] = set()
    for name, path in manifest_args:
        manifest_rows = read_jsonl(path)
        manifest_records.append(
            {
                "name": name,
                "path": str(path),
                "rows": len(manifest_rows),
                "sha256": sha256_file(path),
            }
        )
        for idx, row in enumerate(manifest_rows):
            group = row_group(row)
            if not group or group in seen_groups:
                continue
            if str(row.get("split", "")).lower() == "vor_eval" or "VOR-Eval" in str(row):
                continue
            item = dict(row)
            item["exp42_source_manifest"] = name
            item["exp42_source_manifest_path"] = str(path)
            item["exp42_source_index"] = idx
            item["exp42_scene_group"] = group
            rows.append(item)
            seen_groups.add(group)
            if len(rows) >= limit:
                return rows, manifest_records
    return rows, manifest_records


def finite_float(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def auto_classify(metrics: dict[str, object]) -> tuple[str, str]:
    full = finite_float(metrics.get("full_psnr"))
    mask = finite_float(metrics.get("mask_psnr"))
    boundary = finite_float(metrics.get("boundary_psnr"))
    outside = finite_float(metrics.get("outside_psnr"))
    outside_mae = finite_float(metrics.get("outside_mae"))
    temporal = finite_float(metrics.get("temporal_diff_mae"))

    if not all(math.isfinite(v) for v in (full, mask, boundary, outside)):
        return "TECHNICAL_INVALID", "non-finite metric"
    if outside < 20 or outside_mae > 18 or temporal > 35:
        return "TRIVIAL_BAD", "global/outside/temporal metric failure"
    if full >= 30 and mask >= 28 and boundary >= 28:
        return "TOO_CLOSE", "near-GT metric regime"
    if full >= 27.5 and mask >= 23.0 and boundary >= 23.0 and outside >= 28.0 and temporal <= 8.0:
        return "SUCCESSFUL_REMOVAL_CANDIDATE", "high local and outside metrics"
    if mask >= 19.0 and boundary >= 19.0 and outside >= 25.0 and temporal <= 14.0:
        return "MEDIUM_HARD_REMOVAL", "usable local defect regime"
    if mask >= 17.0 and outside >= 24.0:
        return "BOUNDARY_BAD", "local quality borderline with outside preserved"
    if outside < 25.0:
        return "OUTSIDE_BAD", "outside preservation below mining threshold"
    return "FOGGING_OVER_ERASURE", "metric profile suggests diffuse or over-erased output"


def metric_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    out: dict[str, object] = {"rows": len(rows)}
    for key in ("full_psnr", "mask_psnr", "boundary_psnr", "outside_psnr", "outside_mae", "temporal_diff_mae"):
        vals = [finite_float(row.get(key)) for row in rows]
        vals = [v for v in vals if math.isfinite(v)]
        out[f"{key}_mean"] = sum(vals) / len(vals) if vals else None
        out[f"{key}_min"] = min(vals) if vals else None
        out[f"{key}_max"] = max(vals) if vals else None
    classes: dict[str, int] = {}
    for row in rows:
        cls = str(row.get("auto_classification", ""))
        classes[cls] = classes.get(cls, 0) + 1
    out["classification_counts"] = classes
    out["technical_valid"] = len([row for row in rows if row.get("auto_classification") != "TECHNICAL_INVALID"])
    return out


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(Path(args.repo_dir).resolve()))
    sys.path.insert(0, str(Path(args.project_root).resolve()))

    from diffusers.models import AutoencoderKLWan  # noqa: WPS433
    from diffusers.schedulers import UniPCMultistepScheduler  # noqa: WPS433
    from transformer_minimax_remover import Transformer3DModel  # noqa: WPS433
    from exp30_vor_or_multimodel_minimax.scripts.run_minimax_gate64_adapter_gate_v3 import (  # noqa: WPS433
        BatchCache,
        run_pipeline,
    )

    seeds = parse_seeds(args.seeds)
    output_root = Path(args.output_root).resolve()
    reports_root = Path(args.reports_root).resolve()
    manifest_root = Path(args.repo_manifest_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    reports_root.mkdir(parents=True, exist_ok=True)
    manifest_root.mkdir(parents=True, exist_ok=True)
    hb = Path(args.heartbeat).resolve() if args.heartbeat else output_root / "mine_successful_removal.heartbeat"

    sources, manifest_records = load_source_rows(parse_manifest_args(args.source_manifest), args.limit_sources)
    if not sources:
        raise RuntimeError("no source rows selected")

    model_dir = Path(args.model_dir).resolve()
    for child in ("vae", "transformer", "scheduler"):
        if not (model_dir / child).exists():
            raise FileNotFoundError(f"missing MiniMax model component: {model_dir / child}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch_dtype(args.dtype)
    torch.manual_seed(min(seeds))

    heartbeat(hb, "loading_model")
    vae = AutoencoderKLWan.from_pretrained(model_dir / "vae", torch_dtype=dtype).to(device).eval()
    for param in vae.parameters():
        param.requires_grad_(False)
    transformer = Transformer3DModel.from_pretrained(model_dir / "transformer", torch_dtype=dtype).to(device).eval()
    for param in transformer.parameters():
        param.requires_grad_(False)
    latents_mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(device, dtype)
    latents_std = (1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1)).to(device, dtype)
    cache = BatchCache(vae, latents_mean, latents_std, device, dtype)

    all_rows: list[dict[str, object]] = []
    failed_rows: list[dict[str, object]] = []
    start_time = time.time()
    total = len(sources) * len(seeds)
    counter = 0
    for source_idx, row in enumerate(sources, 1):
        sample_id = str(row["sample_id"])
        for seed_idx, seed in enumerate(seeds):
            counter += 1
            candidate_id = f"{sample_id}__seed{seed_idx}_{seed}"
            heartbeat(hb, f"{counter}/{total}:{candidate_id}")
            candidate_root = output_root / "candidates" / sample_id / f"seed{seed_idx}_{seed}"
            metrics_path = candidate_root / "metrics.json"
            try:
                if metrics_path.exists():
                    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                else:
                    metrics = run_pipeline(
                        transformer,
                        vae,
                        UniPCMultistepScheduler,
                        model_dir,
                        cache,
                        row,
                        candidate_root,
                        seed,
                        args.num_inference_steps,
                        args.iterations,
                    )
                    write_json(metrics_path, metrics)
                auto_class, reason = auto_classify(metrics)
                all_rows.append(
                    {
                        "candidate_id": candidate_id,
                        "sample_id": sample_id,
                        "source_index": source_idx - 1,
                        "seed_index": seed_idx,
                        "seed": seed,
                        "source_manifest": row.get("exp42_source_manifest", ""),
                        "source_manifest_path": row.get("exp42_source_manifest_path", ""),
                        "scene_group": row.get("exp42_scene_group", row_group(row)),
                        "source_type": row.get("source_type", ""),
                        "profile": row.get("profile", ""),
                        "condition_path": row.get("condition_path", ""),
                        "winner_path": row.get("winner_path", ""),
                        "mask_path": row.get("mask_path", ""),
                        "raw_output_mp4": metrics.get("raw_output_mp4", ""),
                        "side_by_side_mp4": metrics.get("side_by_side_mp4", ""),
                        "temporal_strip_16": metrics.get("temporal_strip_16", ""),
                        "review_sheet": metrics.get("review_sheet", ""),
                        "frames_dir": metrics.get("frames_dir", ""),
                        "full_psnr": metrics.get("full_psnr", ""),
                        "mask_psnr": metrics.get("mask_psnr", ""),
                        "boundary_psnr": metrics.get("boundary_psnr", ""),
                        "outside_psnr": metrics.get("outside_psnr", ""),
                        "outside_mae": metrics.get("outside_mae", ""),
                        "temporal_diff_mae": metrics.get("temporal_diff_mae", ""),
                        "auto_classification": auto_class,
                        "auto_reason": reason,
                        "codex_visual_review": "PENDING",
                        "raw_output_primary": True,
                        "diagnostic_comp_used": False,
                        "vor_eval_used": False,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                failed = {
                    "candidate_id": candidate_id,
                    "sample_id": sample_id,
                    "seed": seed,
                    "source_manifest": row.get("exp42_source_manifest", ""),
                    "scene_group": row.get("exp42_scene_group", row_group(row)),
                    "auto_classification": "TECHNICAL_INVALID",
                    "error": repr(exc),
                }
                failed_rows.append(failed)
                all_rows.append(failed)

    successes = [row for row in all_rows if row.get("auto_classification") == "SUCCESSFUL_REMOVAL_CANDIDATE"]
    failures = [row for row in all_rows if row.get("auto_classification") == "MEDIUM_HARD_REMOVAL"]

    write_jsonl(manifest_root / "exp42_minimax_successful_candidates_all.jsonl", all_rows)
    write_jsonl(manifest_root / "exp42_minimax_successful_candidates_selected.jsonl", successes)
    write_jsonl(manifest_root / "exp42_minimax_failure_candidates_selected.jsonl", failures)
    write_csv(reports_root / "exp42_minimax_official_successful_removal_mining.csv", all_rows)

    summary = {
        "status": "MINIMAX_SUCCESSFUL_REMOVAL_POOL_READY"
        if len(successes) >= 24 and len(failures) >= 24 and len(failed_rows) / max(1, len(all_rows)) <= 0.05
        else "MINIMAX_SUCCESSFUL_REMOVAL_POOL_WEAK",
        "runtime_seconds": time.time() - start_time,
        "device": str(device),
        "dtype": args.dtype,
        "num_sources": len(sources),
        "seeds": seeds,
        "num_candidates": len(all_rows),
        "num_successful_candidates": len(successes),
        "num_failure_candidates": len(failures),
        "num_failed_candidates": len(failed_rows),
        "source_manifests": manifest_records,
        "protocol": {
            "scheduler": "UniPCMultistepScheduler",
            "num_inference_steps": args.num_inference_steps,
            "iterations": args.iterations,
            "cfg": "none",
            "raw_output_primary": True,
        },
        "metrics": metric_summary(all_rows),
        "selected_success_manifest": str(manifest_root / "exp42_minimax_successful_candidates_selected.jsonl"),
        "selected_failure_manifest": str(manifest_root / "exp42_minimax_failure_candidates_selected.jsonl"),
        "visual_review_status": "PENDING_CODEX_REVIEW",
    }
    write_json(reports_root / "exp42_minimax_successful_removal_summary.json", summary)

    report = [
        "# Exp42 MiniMax Official Successful-Removal Mining",
        "",
        f"Status: `{summary['status']}`",
        "",
        "This milestone ran official MiniMax raw inference only. It did not train,",
        "did not use VOR-Eval, did not use hard comp, and did not modify the",
        "official MiniMax repository or shared metric code.",
        "",
        "## Protocol",
        "",
        f"- Sources: `{len(sources)}`",
        f"- Seeds per source: `{len(seeds)}`",
        f"- Candidates: `{len(all_rows)}`",
        "- Scheduler: `UniPCMultistepScheduler`",
        f"- num_inference_steps: `{args.num_inference_steps}`",
        f"- iterations: `{args.iterations}`",
        f"- dtype: `{args.dtype}`",
        "- raw output primary: `true`",
        "- VOR-Eval used: `false`",
        "",
        "## Automatic Mining Counts",
        "",
        f"- Successful candidates: `{len(successes)}`",
        f"- Medium-hard failure candidates: `{len(failures)}`",
        f"- Technical-invalid candidates: `{len(failed_rows)}`",
        "",
        "Automatic labels are provisional. Codex visual review must inspect the",
        "selected candidate videos before any promotion to Stage2 data.",
        "",
        "## Outputs",
        "",
        "- `exp42_pai_minimax_successful_removal_badnoise/manifests/exp42_minimax_successful_candidates_all.jsonl`",
        "- `exp42_pai_minimax_successful_removal_badnoise/manifests/exp42_minimax_successful_candidates_selected.jsonl`",
        "- `exp42_pai_minimax_successful_removal_badnoise/manifests/exp42_minimax_failure_candidates_selected.jsonl`",
        "- `reports/exp42_minimax_official_successful_removal_mining.csv`",
        "- `reports/exp42_minimax_successful_removal_summary.json`",
        "",
        f"NAS evidence root: `{output_root}`",
        "",
    ]
    (reports_root / "exp42_minimax_official_successful_removal_mining.md").write_text("\n".join(report), encoding="utf-8")
    heartbeat(hb, f"done:{summary['status']}")


if __name__ == "__main__":
    main()
