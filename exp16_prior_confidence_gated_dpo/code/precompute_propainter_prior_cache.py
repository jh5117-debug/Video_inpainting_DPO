#!/usr/bin/env python3
"""Precompute real ProPainter prior frames for Exp16.

The script is resumable and only writes a new manifest with prior paths. It does
not run DiffuEraser generation and does not modify the original manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INFER = PROJECT_ROOT / "DPO_finetune" / "infer_propainter_candidate.py"


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def prior_field(row: dict[str, Any]) -> str | None:
    for key in ("prior_frame_dir", "propainter_frame_dir", "prior_video_path", "propainter_video_path", "propainter_mp4"):
        if row.get(key):
            return str(row[key])
    return None


def list_frames(path: Path) -> list[Path]:
    if not path.exists() or not path.is_dir():
        return []
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    direct = sorted(p for p in path.iterdir() if p.suffix.lower() in exts)
    return direct or sorted(p for p in path.rglob("*") if p.suffix.lower() in exts)


def validate_prior_dir(path: Path, nframes: int) -> bool:
    return len(list_frames(path)) >= int(nframes)


def run_propainter(row: dict[str, Any], out_dir: Path, args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        str(args.infer_script),
        "--video_dir",
        str(row["win_video_path"]),
        "--mask_dir",
        str(row["mask_path"]),
        "--output_dir",
        str(out_dir),
        "--model_dir",
        str(args.propainter_model_dir),
        "--num_frames",
        str(args.nframes),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--ref_stride",
        str(args.ref_stride),
        "--neighbor_length",
        str(args.neighbor_length),
        "--subvideo_length",
        str(args.subvideo_length),
        "--mask_dilation",
        str(args.mask_dilation),
    ]
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_manifest", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--propainter_model_dir", default="weights/propainter")
    parser.add_argument("--infer_script", default=str(DEFAULT_INFER))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--nframes", type=int, default=16)
    parser.add_argument("--width", type=int, default=432)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--ref_stride", type=int, default=3)
    parser.add_argument("--neighbor_length", type=int, default=25)
    parser.add_argument("--subvideo_length", type=int, default=80)
    parser.add_argument("--mask_dilation", type=int, default=0)
    args = parser.parse_args()

    input_manifest = Path(args.input_manifest).expanduser()
    output_root = Path(args.output_root).expanduser()
    frames_root = output_root / "frames"
    manifest_dir = output_root / "manifests"
    report_dir = output_root / "reports"
    failed_path = report_dir / "failed_cases.csv"
    out_manifest = manifest_dir / "exp16_train_with_prior.jsonl"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    frames_root.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    reused = 0
    generated = 0
    failed: list[dict[str, Any]] = []

    with out_manifest.open("w", encoding="utf-8") as out:
        for idx, row in enumerate(read_jsonl(input_manifest)):
            if args.num_shards > 1 and idx % args.num_shards != args.shard_index:
                continue
            if args.limit and rows_written >= args.limit:
                break
            sample_id = str(row.get("sample_id") or row.get("video_name") or f"row_{idx:06d}")
            existing = prior_field(row)
            if existing and validate_prior_dir(Path(existing), args.nframes):
                row["prior_frame_dir"] = existing
                reused += 1
            else:
                prior_dir = frames_root / sample_id
                if args.resume and validate_prior_dir(prior_dir, args.nframes):
                    row["prior_frame_dir"] = str(prior_dir)
                    reused += 1
                elif args.dry_run:
                    row["prior_frame_dir"] = str(prior_dir)
                    generated += 1
                else:
                    try:
                        run_propainter(row, prior_dir, args)
                        if not validate_prior_dir(prior_dir, args.nframes):
                            raise RuntimeError(f"prior frame count under {prior_dir} is too small")
                        row["prior_frame_dir"] = str(prior_dir)
                        generated += 1
                    except Exception as exc:  # noqa: BLE001
                        failed.append({"sample_id": sample_id, "error": repr(exc)})
                        continue
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows_written += 1

    if failed:
        with failed_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["sample_id", "error"])
            writer.writeheader()
            writer.writerows(failed)

    report = [
        "# Exp16 ProPainter Prior Cache Report",
        "",
        f"input_manifest: `{input_manifest}`",
        f"output_manifest: `{out_manifest}`",
        f"rows_written: {rows_written}",
        f"reused_existing_prior: {reused}",
        f"generated_prior: {generated}",
        f"failed: {len(failed)}",
        f"dry_run: {args.dry_run}",
        "",
        "This cache stores real ProPainter prior frames only. It must not be",
        "replaced by generated losers or frozen-reference epsilon proxies.",
    ]
    (report_dir / "prior_cache_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())

