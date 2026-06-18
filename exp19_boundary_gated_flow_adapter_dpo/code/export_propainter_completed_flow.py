#!/usr/bin/env python3
"""Export ProPainter completed bidirectional flow for Exp19.

This exporter intentionally computes RAFT flow on masked input frames, not on
unmasked GT frames. It reuses ProPainter's RAFT and recurrent flow-completion
modules, then writes adjacent forward/backward completed flows plus
forward-backward confidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from propainter.inference import Propainter  # noqa: E402
from propainter.core.utils import to_tensors  # noqa: E402
from training.dpo.dataset.generated_loser_manifest_dataset import list_image_frames  # noqa: E402

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
from flow_confidence import flow_to_color, forward_backward_confidence, save_flow_npy  # noqa: E402


@dataclass
class FlowStats:
    sample_id: str
    status: str
    num_frames: int = 0
    flow_conf_mean: float | None = None
    flow_conf_p10: float | None = None
    flow_conf_p50: float | None = None
    flow_conf_p90: float | None = None
    valid_flow_ratio: float | None = None
    mean_flow_magnitude: float | None = None
    forward_backward_error: float | None = None
    failure_reason: str = ""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_id(text: object) -> str:
    raw = str(text or "sample")
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in raw)


def load_frames(path: str | Path, nframes: int, size: tuple[int, int]) -> list[Image.Image]:
    files = list_image_frames(path)
    if len(files) < nframes:
        raise ValueError(f"expected at least {nframes} frames under {path}, found {len(files)}")
    return [Image.open(p).convert("RGB").resize(size, Image.BILINEAR) for p in files[:nframes]]


def load_masks(path: str | Path, nframes: int, size: tuple[int, int]) -> list[Image.Image]:
    files = list_image_frames(path)
    if len(files) < nframes:
        raise ValueError(f"expected at least {nframes} masks under {path}, found {len(files)}")
    masks = []
    for p in files[:nframes]:
        arr = np.asarray(Image.open(p).convert("L").resize(size, Image.NEAREST), dtype=np.uint8)
        masks.append(Image.fromarray(((arr > 127).astype(np.uint8) * 255), mode="L"))
    return masks


def save_vis(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB").save(path)


def compute_completed_flow(
    propainter: Propainter,
    frames: list[Image.Image],
    masks: list[Image.Image],
    raft_iter: int,
    fp16: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = propainter.device
    frame_t = to_tensors()(frames).unsqueeze(0) * 2 - 1
    mask_t = to_tensors()(masks).unsqueeze(0)
    frame_t = frame_t.to(device)
    mask_t = mask_t.to(device)
    masked_frames = frame_t * (1.0 - mask_t)
    with torch.no_grad():
        raw_flows = propainter.fix_raft(masked_frames.float(), iters=raft_iter)
        if fp16:
            raw_flows = (raw_flows[0].half(), raw_flows[1].half())
            mask_t = mask_t.half()
            propainter.fix_flow_complete = propainter.fix_flow_complete.half()
        completed, _ = propainter.fix_flow_complete.forward_bidirect_flow(raw_flows, mask_t)
        completed = propainter.fix_flow_complete.combine_flow(raw_flows, completed, mask_t)
        fwd = completed[0].float().detach().cpu()
        bwd = completed[1].float().detach().cpu()
        source_valid = (1.0 - mask_t[:, 1:]).float().detach().cpu()
        conf_pack = forward_backward_confidence(fwd, bwd, tau_flow=1.0, source_valid=source_valid)
    return fwd[0], bwd[0], conf_pack["confidence"][0], conf_pack["fb_error"][0]


def process_row(row: dict[str, Any], args: argparse.Namespace, propainter: Propainter, output_root: Path) -> tuple[dict[str, Any] | None, FlowStats]:
    sid = safe_id(row.get("sample_id") or row.get("video_name") or row.get("pair_index"))
    sample_root = output_root / "samples" / sid
    done_flag = sample_root / ".done"
    fwd_path = sample_root / "completed_forward_flow.npy"
    bwd_path = sample_root / "completed_backward_flow.npy"
    conf_path = sample_root / "flow_confidence.npy"
    fb_path = sample_root / "forward_backward_error.npy"
    meta_path = sample_root / "metadata.json"

    if args.resume and done_flag.exists() and fwd_path.exists() and bwd_path.exists() and conf_path.exists():
        new_row = dict(row)
        new_row.update(
            completed_forward_flow_path=str(fwd_path),
            completed_backward_flow_path=str(bwd_path),
            flow_confidence_path=str(conf_path),
            forward_backward_error_path=str(fb_path),
        )
        return new_row, FlowStats(sample_id=sid, status="reused")

    frames = load_frames(row["win_video_path"], args.nframes, (args.width, args.height))
    masks = load_masks(row["mask_path"], args.nframes, (args.width, args.height))
    fwd, bwd, conf, fb_error = compute_completed_flow(propainter, frames, masks, args.raft_iter, args.fp16)
    sample_root.mkdir(parents=True, exist_ok=True)
    save_flow_npy(fwd_path, fwd.numpy())
    save_flow_npy(bwd_path, bwd.numpy())
    save_flow_npy(conf_path, conf.numpy())
    save_flow_npy(fb_path, fb_error.numpy())

    conf_np = conf.numpy()
    fmag = np.sqrt((fwd.numpy() ** 2).sum(axis=1))
    stats = FlowStats(
        sample_id=sid,
        status="ok",
        num_frames=args.nframes,
        flow_conf_mean=float(conf_np.mean()),
        flow_conf_p10=float(np.quantile(conf_np, 0.10)),
        flow_conf_p50=float(np.quantile(conf_np, 0.50)),
        flow_conf_p90=float(np.quantile(conf_np, 0.90)),
        valid_flow_ratio=float((conf_np > 0.05).mean()),
        mean_flow_magnitude=float(fmag.mean()),
        forward_backward_error=float(fb_error.numpy().mean()),
    )
    if args.save_visuals:
        vis_root = sample_root / "visuals"
        rgb0 = np.asarray(frames[0], dtype=np.uint8)
        mask0 = np.asarray(masks[0], dtype=np.uint8)
        flow_rgb = flow_to_color(fwd[0].permute(1, 2, 0).numpy())
        conf_rgb = np.repeat((conf[0, 0].numpy() * 255).astype(np.uint8)[..., None], 3, axis=2)
        overlay = rgb0.copy()
        overlay[mask0 > 127] = (0.45 * overlay[mask0 > 127] + np.array([0, 255, 0]) * 0.55).astype(np.uint8)
        save_vis(vis_root / "frame0_input.png", rgb0)
        save_vis(vis_root / "frame0_mask_overlay.png", overlay)
        save_vis(vis_root / "flow_forward_0.png", flow_rgb)
        save_vis(vis_root / "confidence_0.png", conf_rgb)
    meta_path.write_text(json.dumps(stats.__dict__, indent=2), encoding="utf-8")
    done_flag.write_text("ok\n", encoding="utf-8")

    new_row = dict(row)
    new_row.update(
        completed_forward_flow_path=str(fwd_path),
        completed_backward_flow_path=str(bwd_path),
        flow_confidence_path=str(conf_path),
        forward_backward_error_path=str(fb_path),
        flow_cache_method="propainter_completed_flow_masked_input",
        flow_confidence_method="forward_backward_consistency_no_gt",
    )
    return new_row, stats


def write_reports(output_root: Path, stats: list[FlowStats], args: argparse.Namespace, manifest_path: Path) -> None:
    reports = output_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    csv_path = reports / "flow_cache_quality_limit100.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FlowStats.__dataclass_fields__.keys()))
        writer.writeheader()
        for stat in stats:
            writer.writerow(stat.__dict__)
    ok = [s for s in stats if s.status in {"ok", "reused"}]

    def mean(name: str) -> float:
        vals = [getattr(s, name) for s in ok if getattr(s, name) is not None]
        return float(np.mean(vals)) if vals else float("nan")

    md = reports / "flow_cache_quality_limit100.md"
    md.write_text(
        "\n".join(
            [
                "# Exp19 ProPainter Completed-Flow Cache Quality",
                "",
                f"- input_manifest: `{args.input_manifest}`",
                f"- output_manifest: `{manifest_path}`",
                f"- output_root: `{output_root}`",
                f"- limit: `{args.limit}`",
                f"- flow_input: `masked winner frames; mask interior zeroed before RAFT`",
                f"- confidence: `forward-backward consistency, no GT-error`",
                "",
                "## Summary",
                "",
                f"- ok/reused: `{len(ok)}` / `{len(stats)}`",
                f"- mean flow_conf_mean: `{mean('flow_conf_mean'):.6f}`",
                f"- mean valid_flow_ratio: `{mean('valid_flow_ratio'):.6f}`",
                f"- mean flow magnitude: `{mean('mean_flow_magnitude'):.6f}`",
                f"- mean forward_backward_error: `{mean('forward_backward_error'):.6f}`",
                "",
                "Training must not start if confidence collapses to zero, valid_flow_ratio is below 0.2,",
                "or visualized flow direction/scale is obviously wrong.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_manifest", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--propainter_model_dir", required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--nframes", type=int, default=16)
    parser.add_argument("--width", type=int, default=432)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--raft_iter", type=int, default=20)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save_visuals", action="store_true")
    args = parser.parse_args()

    rows = read_jsonl(Path(args.input_manifest).expanduser())
    if args.limit > 0:
        rows = rows[: args.limit]
    output_root = Path(args.output_root).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    propainter = Propainter(args.propainter_model_dir, device)

    out_rows: list[dict[str, Any]] = []
    stats: list[FlowStats] = []
    failed: list[dict[str, str]] = []
    for idx, row in enumerate(rows, start=1):
        sid = safe_id(row.get("sample_id") or idx)
        try:
            new_row, stat = process_row(row, args, propainter, output_root)
            if new_row is not None:
                out_rows.append(new_row)
            stats.append(stat)
            print(f"[{idx}/{len(rows)}] {sid}: {stat.status} conf={stat.flow_conf_mean}")
        except Exception as exc:
            msg = str(exc)
            print(f"[{idx}/{len(rows)}] {sid}: FAILED {msg}", file=sys.stderr)
            stats.append(FlowStats(sample_id=sid, status="failed", failure_reason=msg))
            failed.append({"sample_id": sid, "failure_reason": msg})

    manifest_path = output_root / "manifests" / "exp19_train_with_completed_flow_limit100.jsonl"
    write_jsonl(manifest_path, out_rows)
    if failed:
        with (output_root / "failed_cases.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["sample_id", "failure_reason"])
            writer.writeheader()
            writer.writerows(failed)
    write_reports(output_root, stats, args, manifest_path)
    if not out_rows:
        raise SystemExit("No usable Exp19 completed-flow rows were generated")


if __name__ == "__main__":
    main()
