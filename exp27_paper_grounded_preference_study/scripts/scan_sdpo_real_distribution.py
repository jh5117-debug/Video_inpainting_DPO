#!/usr/bin/env python3
"""Scan SDPO safe-lambda on real video residual distributions.

This is not a full DiffuEraser policy-forward training pass.  It uses real
preference rows and real condition/candidate/winner frames to avoid synthetic
gradient construction while keeping the gate lightweight.  Reports are marked
as residual-proxy scans and must not be used to promote RC-FPO by themselves.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exp27_paper_grounded_preference_study.code.official_parity import exp27_sdpo_safe_lambda  # noqa: E402


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--candidate-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model", default="diffueraser")
    p.add_argument("--rows", type=int, default=32)
    p.add_argument("--timesteps-per-row", type=int, default=4)
    p.add_argument("--height", type=int, default=96)
    p.add_argument("--width", type=int, default=160)
    p.add_argument("--mu", type=float, default=0.37)
    p.add_argument("--tiny-step-lr", type=float, default=1e-3)
    return p.parse_args()


def read_jsonl(path: Path, limit: int) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


def list_images(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(p for p in path.iterdir() if p.suffix.lower() in IMG_EXTS)


def load_frame(path: Path, idx: int, size: tuple[int, int]) -> torch.Tensor:
    files = list_images(path)
    img = Image.open(files[idx]).convert("RGB").resize(size, Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def grad_stats(pred: torch.Tensor, target: torch.Tensor, mu: float, tiny_lr: float) -> dict:
    pred = pred.detach().clone().requires_grad_(True)
    target = target.detach().clone()
    lam = exp27_sdpo_safe_lambda(pred, target, mu=mu, eps=1e-8, max_lambda=1.0).float()
    pred_w, pred_l = pred.chunk(2, dim=0)
    target_w, target_l = target.chunk(2, dim=0)
    win_loss = ((pred_w.float() - target_w.float()) ** 2).mean()
    lose_loss = ((pred_l.float() - target_l.float()) ** 2).mean()
    grad_w = torch.autograd.grad(win_loss, pred_w, retain_graph=True)[0]
    grad_l = torch.autograd.grad(lose_loss, pred_l, retain_graph=True)[0]
    cos = torch.nn.functional.cosine_similarity(grad_w.flatten(), grad_l.flatten(), dim=0)
    objective = win_loss - lam * lose_loss
    objective.backward()
    with torch.no_grad():
        before = win_loss.detach().float()
        updated_w = pred_w - tiny_lr * pred.grad[: pred_w.shape[0]]
        after = ((updated_w.float() - target_w.float()) ** 2).mean()
    return {
        "lambda_safe": float(lam.detach().cpu()),
        "lambda_lt_1": float(lam.detach().cpu()) < 1.0,
        "gradient_cosine": float(cos.detach().cpu()),
        "unsafe_rate": float(after.cpu() > before.cpu()),
        "winner_loss_before": float(before.cpu()),
        "winner_loss_after_tiny_step": float(after.cpu()),
        "winner_predicted_change": float((after - before).cpu()),
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.manifest, args.rows)
    out: list[dict] = []
    for row in rows:
        sid = row["sample_id"]
        pred_dir = args.candidate_root / args.model / "raw_frames" / sid
        if not pred_dir.exists():
            continue
        files = list_images(pred_dir)
        if not files:
            continue
        idxs = np.linspace(0, min(len(files), 24) - 1, args.timesteps_per_row).round().astype(int).tolist()
        for idx in idxs:
            try:
                winner = load_frame(Path(row["winner_video_path"]), idx, (args.width, args.height))
                condition = load_frame(Path(row["condition_video_path"]), idx, (args.width, args.height))
                loser = load_frame(pred_dir, idx, (args.width, args.height))
            except Exception as exc:  # noqa: BLE001
                out.append({"sample_id": sid, "frame_index": idx, "status": "failed", "error": repr(exc)})
                continue
            # Winner-side proxy is the task condition residual; loser-side proxy
            # is the generated candidate residual. Both are compared to V_bg.
            pred = torch.stack([condition, loser], dim=0)
            target = torch.stack([winner, winner], dim=0)
            stats = grad_stats(pred, target, args.mu, args.tiny_step_lr)
            out.append({"sample_id": sid, "frame_index": idx, "status": "ok", **stats})
    csv_path = args.output_dir / "sdpo_real_distribution_scan.csv"
    fields = sorted({k for r in out for k in r.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out)
    ok = [r for r in out if r.get("status") == "ok"]
    lambdas = [float(r["lambda_safe"]) for r in ok]
    lt1 = [r for r in ok if r.get("lambda_lt_1") is True]
    unsafe = [r for r in ok if float(r.get("unsafe_rate", 0.0)) > 0.0]
    summary = {
        "status": "completed_residual_proxy_scan",
        "distribution_source": "real_video_residual_proxy_not_policy_forward",
        "rows_requested": args.rows,
        "num_records": len(ok),
        "lambda_lt_1_ratio": float(len(lt1) / len(ok)) if ok else 0.0,
        "unsafe_rate": float(len(unsafe) / len(ok)) if ok else 0.0,
        "lambda_min": min(lambdas) if lambdas else None,
        "lambda_mean": float(np.mean(lambdas)) if lambdas else None,
        "lambda_max": max(lambdas) if lambdas else None,
        "csv": str(csv_path),
    }
    (args.output_dir / "sdpo_real_distribution_scan_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
