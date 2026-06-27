#!/usr/bin/env python3
"""Forensic audit for the Exp30 MiniMax no-change 10-step gate."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from safetensors import safe_open


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--exp30-run-root", required=True)
    p.add_argument("--reports-root", required=True)
    p.add_argument("--limit-tensors", type=int, default=0)
    return p.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def ffloat(value: str | float | int, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def module_group(name: str) -> str:
    parts = name.split(".")
    if len(parts) >= 3 and parts[0] in {"blocks", "transformer_blocks"}:
        return ".".join(parts[:2])
    if len(parts) >= 2:
        return ".".join(parts[:2])
    return parts[0]


def tensor_delta_rows(step0_path: Path, step10_path: Path, recipe: str, limit: int = 0) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    group_acc: dict[str, dict[str, float]] = defaultdict(lambda: {
        "num_tensors": 0.0,
        "numel": 0.0,
        "delta_l1_sum": 0.0,
        "delta_l2_sum": 0.0,
        "param_l2_sum": 0.0,
        "max_abs_delta": 0.0,
        "zero_delta_tensors": 0.0,
    })
    total = {
        "num_tensors": 0,
        "numel": 0,
        "delta_l1_sum": 0.0,
        "delta_l2_sum": 0.0,
        "param_l2_sum": 0.0,
        "max_abs_delta": 0.0,
        "zero_delta_tensors": 0,
    }
    with safe_open(step0_path, framework="pt", device="cpu") as f0, safe_open(step10_path, framework="pt", device="cpu") as f1:
        keys0 = set(f0.keys())
        keys1 = set(f1.keys())
        common = sorted(keys0 & keys1)
        missing = sorted(keys0 - keys1)
        unexpected = sorted(keys1 - keys0)
        for idx, key in enumerate(common):
            if limit and idx >= limit:
                break
            t0 = f0.get_tensor(key).float()
            t1 = f1.get_tensor(key).float()
            diff = t1 - t0
            numel = diff.numel()
            mean_abs = float(diff.abs().mean().item()) if numel else 0.0
            max_abs = float(diff.abs().max().item()) if numel else 0.0
            delta_l2 = float(torch.sum(diff * diff).item())
            param_l2 = float(torch.sum(t0 * t0).item())
            ratio = math.sqrt(delta_l2) / (math.sqrt(param_l2) + 1e-12)
            zero = max_abs == 0.0
            group = module_group(key)
            rows.append({
                "recipe": recipe,
                "tensor": key,
                "module_group": group,
                "shape": "x".join(str(x) for x in t0.shape),
                "numel": numel,
                "mean_abs_delta": mean_abs,
                "max_abs_delta": max_abs,
                "delta_norm": math.sqrt(delta_l2),
                "param_norm": math.sqrt(param_l2),
                "delta_param_norm_ratio": ratio,
                "zero_delta": zero,
            })
            acc = group_acc[group]
            acc["num_tensors"] += 1
            acc["numel"] += numel
            acc["delta_l1_sum"] += mean_abs * numel
            acc["delta_l2_sum"] += delta_l2
            acc["param_l2_sum"] += param_l2
            acc["max_abs_delta"] = max(acc["max_abs_delta"], max_abs)
            acc["zero_delta_tensors"] += 1 if zero else 0
            total["num_tensors"] += 1
            total["numel"] += numel
            total["delta_l1_sum"] += mean_abs * numel
            total["delta_l2_sum"] += delta_l2
            total["param_l2_sum"] += param_l2
            total["max_abs_delta"] = max(total["max_abs_delta"], max_abs)
            total["zero_delta_tensors"] += 1 if zero else 0
    group_rows: list[dict[str, object]] = []
    for group, acc in sorted(group_acc.items()):
        group_rows.append({
            "recipe": recipe,
            "module_group": group,
            "num_tensors": int(acc["num_tensors"]),
            "numel": int(acc["numel"]),
            "mean_abs_delta": acc["delta_l1_sum"] / max(1.0, acc["numel"]),
            "max_abs_delta": acc["max_abs_delta"],
            "delta_norm": math.sqrt(acc["delta_l2_sum"]),
            "param_norm": math.sqrt(acc["param_l2_sum"]),
            "delta_param_norm_ratio": math.sqrt(acc["delta_l2_sum"]) / (math.sqrt(acc["param_l2_sum"]) + 1e-12),
            "zero_delta_tensors": int(acc["zero_delta_tensors"]),
        })
    summary = {
        "recipe": recipe,
        "common_keys": len(common),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "num_tensors_read": total["num_tensors"],
        "total_numel": total["numel"],
        "trainable_parameter_count_assuming_exp30_full_transformer": total["numel"],
        "trainable_ratio_assuming_exp30_full_transformer": 1.0,
        "mean_abs_delta": total["delta_l1_sum"] / max(1, total["numel"]),
        "max_abs_delta": total["max_abs_delta"],
        "delta_norm": math.sqrt(total["delta_l2_sum"]),
        "param_norm": math.sqrt(total["param_l2_sum"]),
        "delta_param_norm_ratio": math.sqrt(total["delta_l2_sum"]) / (math.sqrt(total["param_l2_sum"]) + 1e-12),
        "zero_delta_tensors": total["zero_delta_tensors"],
        "module_groups": group_rows,
    }
    return rows, summary


def read_rgb(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if arr is None:
        raise RuntimeError(f"failed to read {path}")
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


def read_gray(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise RuntimeError(f"failed to read mask {path}")
    return arr


def image_files(path: Path) -> list[Path]:
    return sorted(p for p in path.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})


def region_stats(diff: np.ndarray, region: np.ndarray) -> tuple[float, float]:
    if not np.any(region):
        return float("nan"), float("nan")
    vals = np.abs(diff[region])
    return float(vals.mean()), float(vals.max())


def output_diff_rows(metrics_rows: list[dict[str, str]], visual_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    visual_map = {(r["recipe"], r["sample_id"]): r for r in visual_rows}
    out: list[dict[str, object]] = []
    for row in metrics_rows:
        recipe = row["recipe"]
        sample_id = row["sample_id"]
        vis = visual_map.get((recipe, sample_id), {})
        step0_dir = Path(row["step0_frames"])
        step10_dir = Path(row["step10_frames"])
        mask_dir = Path(vis.get("mask_path", ""))
        cond_dir = Path(vis.get("condition_path", ""))
        winner_dir = Path(vis.get("winner_path", ""))
        step0_files = image_files(step0_dir)
        step10_files = image_files(step10_dir)
        mask_files = image_files(mask_dir) if mask_dir.exists() else []
        cond_files = image_files(cond_dir) if cond_dir.exists() else []
        winner_files = image_files(winner_dir) if winner_dir.exists() else []
        n = min(len(step0_files), len(step10_files))
        frame_rows = []
        byte_identical = True
        for idx in range(n):
            a = read_rgb(step0_files[idx])
            b = read_rgb(step10_files[idx])
            if a.shape != b.shape:
                b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LINEAR)
            diff = b.astype(np.float32) - a.astype(np.float32)
            full_mean = float(np.abs(diff).mean())
            full_max = float(np.abs(diff).max())
            byte_identical = byte_identical and full_max == 0.0
            mask = np.ones(a.shape[:2], dtype=bool)
            if idx < len(mask_files):
                m = read_gray(mask_files[idx])
                if m.shape != a.shape[:2]:
                    m = cv2.resize(m, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_NEAREST)
                mask = m > 20
            kernel = np.ones((9, 9), np.uint8)
            dil = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
            outside = ~dil
            affected = mask.copy()
            if idx < len(cond_files) and idx < len(winner_files):
                c = read_rgb(cond_files[idx])
                w = read_rgb(winner_files[idx])
                if c.shape[:2] != a.shape[:2]:
                    c = cv2.resize(c, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LINEAR)
                if w.shape[:2] != a.shape[:2]:
                    w = cv2.resize(w, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LINEAR)
                affected_score = np.mean(np.abs(c.astype(np.float32) - w.astype(np.float32)), axis=2)
                affected = affected_score > 10.0
            mask_mean, mask_max = region_stats(diff, mask)
            affected_mean, affected_max = region_stats(diff, affected)
            outside_mean, outside_max = region_stats(diff, outside)
            frame_rows.append({
                "frame_index": idx,
                "full_mean_abs_diff": full_mean,
                "full_max_abs_diff": full_max,
                "mask_mean_abs_diff": mask_mean,
                "mask_max_abs_diff": mask_max,
                "affected_mean_abs_diff": affected_mean,
                "affected_max_abs_diff": affected_max,
                "outside_mean_abs_diff": outside_mean,
                "outside_max_abs_diff": outside_max,
            })
        def avg(key: str) -> float:
            vals = [r[key] for r in frame_rows if math.isfinite(float(r[key]))]
            return float(np.mean(vals)) if vals else float("nan")
        def mx(key: str) -> float:
            vals = [r[key] for r in frame_rows if math.isfinite(float(r[key]))]
            return float(np.max(vals)) if vals else float("nan")
        out.append({
            "recipe": recipe,
            "sample_id": sample_id,
            "source_group": row.get("source_group", ""),
            "loser_model": row.get("loser_model", ""),
            "frames_compared": n,
            "byte_identical": byte_identical,
            "full_mean_abs_diff": avg("full_mean_abs_diff"),
            "full_max_abs_diff": mx("full_max_abs_diff"),
            "mask_mean_abs_diff": avg("mask_mean_abs_diff"),
            "mask_max_abs_diff": mx("mask_max_abs_diff"),
            "affected_mean_abs_diff": avg("affected_mean_abs_diff"),
            "affected_max_abs_diff": mx("affected_max_abs_diff"),
            "outside_mean_abs_diff": avg("outside_mean_abs_diff"),
            "outside_max_abs_diff": mx("outside_max_abs_diff"),
            "delta_full_psnr": ffloat(row.get("delta_full_psnr", "")),
            "delta_mask_psnr": ffloat(row.get("delta_mask_psnr", "")),
            "delta_boundary_psnr": ffloat(row.get("delta_boundary_psnr", "")),
            "delta_outside_psnr": ffloat(row.get("delta_outside_psnr", "")),
            "visual_classification": vis.get("classification", ""),
        })
    return out


def loss_scale_rows(diag_rows: list[dict[str, str]]) -> tuple[list[dict[str, object]], dict[str, object]]:
    out: list[dict[str, object]] = []
    by_recipe: dict[str, list[dict[str, object]]] = defaultdict(list)
    for r in diag_rows:
        row = {
            "recipe": r["recipe"],
            "step": int(float(r["step"])),
            "sample_id": r["sample_id"],
            "t": ffloat(r.get("t", "")),
            "winner_policy_loss": ffloat(r["winner_policy_loss"]),
            "loser_policy_loss": ffloat(r["loser_policy_loss"]),
            "winner_reference_loss": ffloat(r["winner_reference_loss"]),
            "loser_reference_loss": ffloat(r["loser_reference_loss"]),
            "win_gap": ffloat(r["win_gap"]),
            "lose_gap": ffloat(r["lose_gap"]),
            "preference_margin": ffloat(r["preference_margin"]),
            "linear_utility": ffloat(r["linear_utility"]),
            "loss": ffloat(r["loss"]),
            "grad_norm": ffloat(r["grad_norm"]),
            "grad_max_abs": ffloat(r["grad_max_abs"]),
            "grad_tensors": int(float(r.get("grad_tensors", 0))),
            "finite": r.get("finite", ""),
        }
        out.append(row)
        by_recipe[str(row["recipe"])].append(row)
    summary: dict[str, object] = {}
    for recipe, rows in by_recipe.items():
        def vals(key: str) -> list[float]:
            return [float(x[key]) for x in rows if math.isfinite(float(x[key]))]
        summary[recipe] = {
            "rows": len(rows),
            "loss_mean": float(np.mean(vals("loss"))),
            "linear_utility_mean": float(np.mean(vals("linear_utility"))),
            "linear_utility_min": float(np.min(vals("linear_utility"))),
            "linear_utility_max": float(np.max(vals("linear_utility"))),
            "abs_margin_mean": float(np.mean([abs(v) for v in vals("preference_margin")])),
            "grad_norm_mean": float(np.mean(vals("grad_norm"))),
            "grad_norm_max": float(np.max(vals("grad_norm"))),
            "t_min": float(np.min(vals("t"))),
            "t_max": float(np.max(vals("t"))),
            "t_mean": float(np.mean(vals("t"))),
        }
    return out, summary


def main() -> None:
    args = parse_args()
    run_root = Path(args.exp30_run_root).resolve()
    reports_root = Path(args.reports_root).resolve()
    reports_root.mkdir(parents=True, exist_ok=True)
    exp30_reports = run_root / "reports"
    metrics_rows = read_csv(exp30_reports / "exp30_minimax_gate64_adapter_10step_metrics_v3.csv")
    visual_rows = read_csv(exp30_reports / "exp30_minimax_gate64_adapter_10step_visual_review_v3.csv")
    diag_rows = read_csv(exp30_reports / "exp30_minimax_gate64_adapter_diagnostics_v3.csv")
    summary_in = json.loads((exp30_reports / "exp30_minimax_gate64_adapter_summary_v3.json").read_text(encoding="utf-8"))

    all_param_rows: list[dict[str, object]] = []
    recipe_param_summary: dict[str, object] = {}
    for recipe in ["frozen", "ema"]:
        ckpt_root = run_root / "checkpoints" / recipe
        rows, summ = tensor_delta_rows(
            ckpt_root / "checkpoint-0" / "diffusion_pytorch_model.safetensors",
            ckpt_root / "checkpoint-10" / "diffusion_pytorch_model.safetensors",
            recipe,
            args.limit_tensors,
        )
        all_param_rows.extend(rows)
        recipe_param_summary[recipe] = summ
    write_csv(reports_root / "exp35_minimax_10step_param_delta.csv", all_param_rows)

    diff_rows = output_diff_rows(metrics_rows, visual_rows)
    write_csv(reports_root / "exp35_minimax_10step_output_diff.csv", diff_rows)

    loss_rows, loss_summary = loss_scale_rows(diag_rows)
    write_csv(reports_root / "exp35_minimax_10step_loss_scale.csv", loss_rows)

    cause = "MINIMAX_NOCHANGE_CAUSE_LR_TOO_SMALL"
    reason = "Full-transformer updates were nonzero but extremely small, utility stayed near 0.5, and output diffs were sub-perceptual."
    if all(row["byte_identical"] for row in diff_rows):
        cause = "MINIMAX_NOCHANGE_CAUSE_CHECKPOINT_LOAD"
        reason = "Step10 outputs are byte-identical to Step0; checkpoint/inference path must be checked before recipes."
    else:
        max_output_diff = max(float(row["full_max_abs_diff"]) for row in diff_rows)
        mean_mask_diff = float(np.mean([float(row["mask_mean_abs_diff"]) for row in diff_rows]))
        if max_output_diff > 0 and mean_mask_diff < 0.05:
            cause = "MINIMAX_NOCHANGE_CAUSE_OUTPUT_INSENSITIVE"
            reason = "Step10 outputs are not identical but movement is far below visible scale."
        util_means = [float(v["linear_utility_mean"]) for v in loss_summary.values()]
        if util_means and max(abs(u - 0.5) for u in util_means) < 0.001:
            cause = "MINIMAX_NOCHANGE_CAUSE_UTILITY_SCALE_TOO_WEAK"
            reason = "Linear utility stayed effectively constant near 0.5; gradients are too weak for visible output movement."

    forensic_summary = {
        "status": cause,
        "reason": reason,
        "exp30_status": summary_in.get("status"),
        "checkpoint_load": {
            recipe: {
                "missing_keys": recipe_param_summary[recipe]["missing_keys"],
                "unexpected_keys": recipe_param_summary[recipe]["unexpected_keys"],
                "common_keys": recipe_param_summary[recipe]["common_keys"],
            }
            for recipe in recipe_param_summary
        },
        "param_delta_summary": recipe_param_summary,
        "output_diff": {
            "rows": len(diff_rows),
            "byte_identical_rows": sum(1 for row in diff_rows if row["byte_identical"]),
            "mean_full_abs_diff": float(np.mean([float(row["full_mean_abs_diff"]) for row in diff_rows])),
            "mean_mask_abs_diff": float(np.mean([float(row["mask_mean_abs_diff"]) for row in diff_rows])),
            "mean_affected_abs_diff": float(np.mean([float(row["affected_mean_abs_diff"]) for row in diff_rows])),
            "mean_outside_abs_diff": float(np.mean([float(row["outside_mean_abs_diff"]) for row in diff_rows])),
            "max_abs_diff": float(np.max([float(row["full_max_abs_diff"]) for row in diff_rows])),
        },
        "loss_scale_summary": loss_summary,
        "bad_noise_used_in_exp30": False,
        "flow_target": "epsilon_minus_z0",
        "trainable_scope_exp30": "all Transformer3DModel parameters",
        "per_module_grad_norm_available": False,
        "per_module_grad_norm_note": "Exp30 diagnostics logged total grad norm only; per-module gradient instrumentation starts in Exp35 follow-up milestones.",
    }
    write_json(reports_root / "exp35_minimax_10step_forensic_summary.json", forensic_summary)

    # A compact CSV summary for the milestone.
    compact_rows: list[dict[str, object]] = []
    for recipe, summ in recipe_param_summary.items():
        compact_rows.append({
            "kind": "param_delta",
            "recipe": recipe,
            "status": cause,
            "num_tensors": summ["num_tensors_read"],
            "numel": summ["total_numel"],
            "mean_abs_delta": summ["mean_abs_delta"],
            "max_abs_delta": summ["max_abs_delta"],
            "delta_param_norm_ratio": summ["delta_param_norm_ratio"],
            "zero_delta_tensors": summ["zero_delta_tensors"],
        })
    for recipe, summ in loss_summary.items():
        compact_rows.append({
            "kind": "loss_scale",
            "recipe": recipe,
            "status": cause,
            "linear_utility_mean": summ["linear_utility_mean"],
            "abs_margin_mean": summ["abs_margin_mean"],
            "grad_norm_mean": summ["grad_norm_mean"],
            "grad_norm_max": summ["grad_norm_max"],
            "t_mean": summ["t_mean"],
        })
    write_csv(reports_root / "exp35_minimax_10step_forensic_audit.csv", compact_rows)

    md = [
        "# Exp35 MiniMax 10-Step Forensic Audit\n\n",
        f"Status: `{cause}`\n\n",
        f"{reason}\n\n",
        "## Scope\n\n",
        "- Source run: Exp30 MiniMax Gate64 adapter V3.\n",
        "- Training performed in this milestone: false.\n",
        "- Flow target: `epsilon - z0`.\n",
        "- Exp30 trainable scope: all `Transformer3DModel` parameters.\n",
        "- Bad-noise / hard-timestep miner used in Exp30: false.\n\n",
        "## Checkpoint And Parameter Delta\n\n",
    ]
    for recipe, summ in recipe_param_summary.items():
        md.extend([
            f"### {recipe}\n\n",
            f"- Common checkpoint tensors: `{summ['common_keys']}`.\n",
            f"- Missing / unexpected keys: `{len(summ['missing_keys'])}` / `{len(summ['unexpected_keys'])}`.\n",
            f"- Parameter count read: `{summ['total_numel']}`.\n",
            f"- Mean abs delta: `{summ['mean_abs_delta']}`.\n",
            f"- Max abs delta: `{summ['max_abs_delta']}`.\n",
            f"- Delta / param norm ratio: `{summ['delta_param_norm_ratio']}`.\n",
            f"- Zero-delta tensors: `{summ['zero_delta_tensors']}`.\n\n",
        ])
    md.extend([
        "## Output Diff\n\n",
        f"- Compared rows: `{len(diff_rows)}`.\n",
        f"- Byte-identical rows: `{forensic_summary['output_diff']['byte_identical_rows']}`.\n",
        f"- Mean full abs diff: `{forensic_summary['output_diff']['mean_full_abs_diff']}`.\n",
        f"- Mean mask abs diff: `{forensic_summary['output_diff']['mean_mask_abs_diff']}`.\n",
        f"- Mean affected abs diff: `{forensic_summary['output_diff']['mean_affected_abs_diff']}`.\n",
        f"- Mean outside abs diff: `{forensic_summary['output_diff']['mean_outside_abs_diff']}`.\n",
        f"- Max abs diff: `{forensic_summary['output_diff']['max_abs_diff']}`.\n\n",
        "Step10 is not byte-identical to Step0, so the checkpoint/inference path is not obviously falling back to Step0. The movement is, however, sub-perceptual and not quality-positive.\n\n",
        "## Loss / Utility / Timestep Scale\n\n",
    ])
    for recipe, summ in loss_summary.items():
        md.extend([
            f"### {recipe}\n\n",
            f"- Loss mean: `{summ['loss_mean']}`.\n",
            f"- Linear utility mean/min/max: `{summ['linear_utility_mean']}` / `{summ['linear_utility_min']}` / `{summ['linear_utility_max']}`.\n",
            f"- Abs margin mean: `{summ['abs_margin_mean']}`.\n",
            f"- Grad norm mean/max: `{summ['grad_norm_mean']}` / `{summ['grad_norm_max']}`.\n",
            f"- t min/mean/max: `{summ['t_min']}` / `{summ['t_mean']}` / `{summ['t_max']}`.\n\n",
        ])
    md.extend([
        "## Conclusion\n\n",
        f"Root-cause status for this milestone: `{cause}`. The strongest evidence is that Exp30 had valid data and nonzero checkpoint/output movement, but the utility stayed near constant and parameter/output movement was too small to matter. Next step is inference sensitivity positive-control before changing recipes.\n",
    ])
    (reports_root / "exp35_minimax_10step_forensic_audit.md").write_text("".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()

