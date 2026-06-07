#!/usr/bin/env python3
"""Prepare Exp7c GT/winner manifests for small-D2 partial-mask DPO.

The Exp7a generated-loser manifest contains a cached ``win_video_path`` created
while generating losers. Exp7c rebuilds that winner cache from the original
VideoDPO pair source so the positive side is tracked as an experiment artifact,
while preserving the Exp7a loser and mask paths.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.pai_videodpo_single_sample_generation_smoke import (  # noqa: E402
    choose_frame_indices,
    load_yaml,
    raw_video_info,
    read_canonical_frames,
    read_json,
    resolve_videodpo_roots,
    save_rgb_frames,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Exp7c GT/winner manifest.")
    parser.add_argument("--source_manifest", required=True)
    parser.add_argument("--train_data_yaml", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--output_manifest", required=True)
    parser.add_argument("--cache_root", default=None)
    parser.add_argument("--official_config", default="DPO_finetune/configs/official_diffueraser_stage1.yaml")
    parser.add_argument("--seed", type=int, default=20260524)
    parser.add_argument("--frame_selection", choices=["seeded_random", "first"], default="seeded_random")
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--nframes", type=int, default=16)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--report_path", default=None)
    return parser.parse_args()


def read_jsonl(path: Path, max_rows: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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


def video_path(root: Path, metadata_item: Mapping[str, Any]) -> Path:
    raw = str(metadata_item["basic"]["clip_path"])
    path = Path(raw)
    return path if path.is_absolute() else root / path


def prepare_cache(
    row: Mapping[str, Any],
    pair: Mapping[str, Any],
    metadata: list[Mapping[str, Any]],
    root: Path,
    cache_root: Path,
    args: argparse.Namespace,
) -> tuple[Path, list[int], Path]:
    pair_index = int(row["pair_index"])
    winner_idx = int(pair["video1"])
    winner = metadata[winner_idx]
    winner_path = video_path(root, winner).resolve()
    raw_w, raw_h, raw_fps, raw_count = raw_video_info(winner_path)
    indices = choose_frame_indices(
        raw_count,
        int(args.nframes),
        int(args.stride),
        int(args.seed) + pair_index,
        str(args.frame_selection),
    )
    frames = read_canonical_frames(winner_path, indices, int(args.width), int(args.height))
    sample_id = str(row.get("sample_id") or f"videodpo_pair{pair_index:06d}")
    cache_dir = cache_root / sample_id / "win"
    if args.overwrite or not cache_dir.exists() or not any(cache_dir.iterdir()):
        save_rgb_frames(frames, cache_dir)
    return cache_dir, indices, winner_path


def report_text(args: argparse.Namespace, stats: Mapping[str, Any]) -> str:
    lines = [
        "# Exp7c GT-Win Manifest Preparation",
        "",
        f"- source_manifest: `{args.source_manifest}`",
        f"- train_data_yaml: `{args.train_data_yaml}`",
        f"- output_manifest: `{args.output_manifest}`",
        f"- cache_root: `{args.cache_root}`",
        f"- seed: `{args.seed}`",
        f"- frame_selection: `{args.frame_selection}`",
        f"- rows_read: `{stats['rows_read']}`",
        f"- rows_written: `{stats['rows_written']}`",
        f"- missing_rows: `{stats['missing_rows']}`",
        "",
        "Invariant:",
        "",
        "- `win_video_path` is replaced with a cache rebuilt from the original VideoDPO pair winner source.",
        "- `final_loser_video_path` and `mask_path` are preserved from Exp7a.",
        "- Eval must use VideoDPO small-D2 partial-mask metrics, not DAVIS.",
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

    # Parse the official config as a cheap guard that this tool is being run in
    # the same VideoDPO/DiffuEraser repo context.
    load_yaml(Path(args.official_config))

    rows = read_jsonl(source_manifest, max_rows=args.max_rows)
    roots = resolve_videodpo_roots(Path(args.train_data_yaml))
    root = roots[0]
    metadata = read_json(root / "metadata.json")
    pairs = read_json(root / "pair.json")
    if not isinstance(metadata, list) or not isinstance(pairs, list):
        raise SystemExit(f"unexpected VideoDPO metadata/pair format under {root}")

    output_rows: list[dict[str, Any]] = []
    issues: list[str] = []
    for row in rows:
        out = dict(row)
        sample_id = str(row.get("sample_id") or "<missing-sample-id>")
        try:
            pair_index = int(row["pair_index"])
            pair = pairs[pair_index]
            cache_dir, indices, source_path = prepare_cache(row, pair, metadata, root, cache_root, args)
            out["win_video_path_original_exp7a"] = row.get("win_video_path")
            out["win_video_path"] = str(cache_dir.resolve())
            out["win_source"] = "videodpo_raw_winner_rebuilt_by_pair_index"
            out["exp7c_gtwin_source_video"] = str(source_path)
            out["exp7c_gtwin_frame_indices"] = indices
            out["exp7c_gtwin_train_data_yaml"] = str(Path(args.train_data_yaml).resolve())
            output_rows.append(out)
        except Exception as exc:
            message = f"{sample_id}: {exc}"
            issues.append(message)
            if args.strict:
                print(f"[prepare-exp7c][ERROR] {message}", file=sys.stderr)
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
        report.write_text(report_text(args, stats), encoding="utf-8")

    print(
        "[prepare-exp7c] "
        f"rows_read={stats['rows_read']} rows_written={stats['rows_written']} "
        f"missing_rows={stats['missing_rows']} output={output_manifest}"
    )
    if issues:
        print(f"[prepare-exp7c] issues={len(issues)} first={issues[0]}", file=sys.stderr)
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())

