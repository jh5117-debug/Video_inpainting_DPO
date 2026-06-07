#!/usr/bin/env python3
"""Prepare Exp8c manifests with original YouTube-VOS GT as the winner.

Exp8c keeps each selected D3 comp loser and mask path unchanged, but replaces
``win_video_path`` with a generated cache of symlinks to the original
YouTube-VOS GT frames aligned by ``canonical_frame_indices``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
INDEX_KEYS = (
    "canonical_frame_indices",
    "frame_indices",
    "selected_frame_indices",
    "source_frame_indices",
)
VIDEO_ID_KEYS = (
    "source_video_id",
    "video_id",
    "youtubevos_video_id",
    "ytb_video_id",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an Exp8c GT-winner manifest from a D3 comp manifest."
    )
    parser.add_argument("--source_manifest", required=True, help="D3 selected-primary-comp JSONL manifest.")
    parser.add_argument(
        "--youtubevos_train_root",
        required=True,
        help=(
            "YouTube-VOS train root. Either the train directory containing "
            "JPEGImages/ or JPEGImages itself."
        ),
    )
    parser.add_argument("--output_root", required=True, help="Root for the Exp8c generated-loser asset.")
    parser.add_argument("--output_manifest", required=True, help="Output JSONL manifest path.")
    parser.add_argument("--cache_root", default=None, help="GT winner frame cache root. Defaults to output_root/gt_win_cache.")
    parser.add_argument("--link_mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing cached frame links/files.")
    parser.add_argument("--strict", action="store_true", help="Fail if any row cannot be aligned.")
    parser.add_argument("--max_rows", type=int, default=0, help="Optional debug limit.")
    parser.add_argument("--report_path", default=None, help="Optional markdown report path.")
    return parser.parse_args()


def read_jsonl(path: Path, max_rows: int = 0) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_rows and len(rows) >= max_rows:
                break
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def image_files(path: Path) -> List[Path]:
    if not path.exists() or not path.is_dir():
        return []
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)


def numeric_stem(path: Path) -> int | None:
    try:
        return int(path.stem)
    except ValueError:
        return None


def video_id_from_row(row: Mapping[str, Any]) -> str:
    for key in VIDEO_ID_KEYS:
        value = row.get(key)
        if value:
            return str(value)
    sample_id = str(row.get("sample_id", ""))
    if sample_id:
        return sample_id.split("_")[-1]
    raise ValueError(f"row has no usable video id: {row}")


def frame_indices_from_row(row: Mapping[str, Any], fallback_len: int) -> List[int]:
    for key in INDEX_KEYS:
        value = row.get(key)
        if value is None or value == "":
            continue
        if isinstance(value, str):
            value = json.loads(value)
        if isinstance(value, Sequence):
            return [int(x) for x in value]
    return list(range(fallback_len))


def youtubevos_jpeg_root(train_root: Path) -> Path:
    candidates = [train_root / "JPEGImages", train_root]
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            if any(p.is_dir() for p in candidate.iterdir()):
                return candidate
    return train_root / "JPEGImages"


def build_video_index(jpeg_root: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    if not jpeg_root.exists():
        return index
    for item in jpeg_root.iterdir():
        if item.is_dir():
            index[item.name] = item
    return index


def selected_gt_frames(gt_dir: Path, indices: Sequence[int]) -> List[Path]:
    files = image_files(gt_dir)
    by_number = {num: path for path in files if (num := numeric_stem(path)) is not None}
    selected: List[Path] = []
    for idx in indices:
        if idx in by_number:
            selected.append(by_number[idx])
        elif 0 <= idx < len(files):
            selected.append(files[idx])
        else:
            raise IndexError(f"frame index {idx} missing in {gt_dir} with {len(files)} files")
    return selected


def replace_path(dst: Path, src: Path, mode: str, overwrite: bool) -> None:
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
    else:
        os.symlink(src, dst)


def prepare_cache(
    row: Mapping[str, Any],
    gt_dir: Path,
    cache_root: Path,
    link_mode: str,
    overwrite: bool,
) -> Path:
    sample_id = str(row.get("sample_id") or video_id_from_row(row))
    nframes = int(row.get("num_frames") or row.get("canonical_num_frames") or 16)
    indices = frame_indices_from_row(row, fallback_len=nframes)
    frames = selected_gt_frames(gt_dir, indices)
    cache_dir = cache_root / sample_id / "win"
    for out_idx, src in enumerate(frames):
        dst = cache_dir / f"{out_idx:05d}{src.suffix.lower()}"
        replace_path(dst, src.resolve(), link_mode, overwrite)
    return cache_dir


def markdown_report(args: argparse.Namespace, stats: Mapping[str, Any]) -> str:
    lines = [
        "# Exp8c GT-Win Manifest Preparation",
        "",
        f"- source_manifest: `{args.source_manifest}`",
        f"- youtubevos_train_root: `{args.youtubevos_train_root}`",
        f"- output_manifest: `{args.output_manifest}`",
        f"- cache_root: `{args.cache_root}`",
        f"- link_mode: `{args.link_mode}`",
        f"- rows_read: `{stats['rows_read']}`",
        f"- rows_written: `{stats['rows_written']}`",
        f"- missing_rows: `{stats['missing_rows']}`",
        f"- strict: `{args.strict}`",
        "",
        "Invariant:",
        "",
        "- `win_video_path` is replaced with aligned YouTube-VOS GT cache.",
        "- `final_loser_video_path` is preserved from the source D3 manifest.",
        "- `mask_path` is preserved from the source D3 manifest.",
    ]
    if stats["issues"]:
        lines.extend(["", "## Issues", ""])
        for issue in stats["issues"][:200]:
            lines.append(f"- {issue}")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    source_manifest = Path(args.source_manifest)
    output_root = Path(args.output_root)
    output_manifest = Path(args.output_manifest)
    cache_root = Path(args.cache_root) if args.cache_root else output_root / "gt_win_cache"
    args.cache_root = str(cache_root)

    rows = read_jsonl(source_manifest, max_rows=args.max_rows)
    jpeg_root = youtubevos_jpeg_root(Path(args.youtubevos_train_root))
    video_index = build_video_index(jpeg_root)

    output_rows: List[Dict[str, Any]] = []
    issues: List[str] = []

    for row in rows:
        out = dict(row)
        try:
            video_id = video_id_from_row(row)
            gt_dir = video_index.get(video_id)
            if gt_dir is None:
                raise FileNotFoundError(f"missing YouTube-VOS GT dir for video_id={video_id}")
            cache_dir = prepare_cache(row, gt_dir, cache_root, args.link_mode, args.overwrite)
            out["win_video_path_original_d3"] = row.get("win_video_path")
            out["win_video_path"] = str(cache_dir.resolve())
            out["win_source"] = "youtubevos_gt_aligned_by_canonical_frame_indices"
            out["exp8c_gtwin_video_id"] = video_id
            output_rows.append(out)
        except Exception as exc:
            sample_id = row.get("sample_id", "<missing-sample-id>")
            message = f"{sample_id}: {exc}"
            issues.append(message)
            if args.strict:
                print(f"[prepare-exp8c][ERROR] {message}", file=sys.stderr)
                return 2

    write_jsonl(output_manifest, output_rows)
    stats = {
        "rows_read": len(rows),
        "rows_written": len(output_rows),
        "missing_rows": len(rows) - len(output_rows),
        "issues": issues,
    }
    if args.report_path:
        report = Path(args.report_path)
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(markdown_report(args, stats), encoding="utf-8")

    print(
        "[prepare-exp8c] "
        f"rows_read={stats['rows_read']} rows_written={stats['rows_written']} "
        f"missing_rows={stats['missing_rows']} output={output_manifest}"
    )
    if issues:
        print(f"[prepare-exp8c] issues={len(issues)} first={issues[0]}", file=sys.stderr)
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
