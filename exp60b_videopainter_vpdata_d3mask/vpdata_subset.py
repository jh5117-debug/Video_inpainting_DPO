#!/usr/bin/env python3
"""Select and optionally download a bounded VPData subset.

This utility is intentionally conservative:

- It never clones the full Hugging Face dataset.
- It defaults to metadata/plan generation only.
- Raw video download requires ``--download``.
- It refuses requests larger than Exp60B's locked train1000/test100 bounds.

The first Exp60B pass uses Pexels rows only because official Pexels raw videos
are exposed as row-level URLs in ``pexels.csv``. VideoVo raw videos are bundled
in multi-GB zip shards and are therefore excluded unless a later experiment
explicitly authorizes shard-level handling.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def hf_base() -> str:
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    return f"{endpoint}/datasets/TencentARC/VPData/resolve/main"


HF_BASE = hf_base()
TRAIN_CSV = f"{HF_BASE}/pexels_videovo_train_dataset.csv"
VAL_CSV = f"{HF_BASE}/pexels_videovo_val_dataset.csv"
TEST_CSV = f"{HF_BASE}/pexels_videovo_test_dataset.csv"
PEXELS_CSV = f"{HF_BASE}/pexels.csv"
VIDEOVO_CSV = f"{HF_BASE}/videovo.csv"

PEXELS_RE = re.compile(r"^(?P<index>\d{12})_(?P<video_id>\d+)\.mp4$")
VIDEOVO_RE = re.compile(r"^(?P<stem>\d{12})\.0\.mp4$")


@dataclass(frozen=True)
class SourceRow:
    split: str
    vpdata_id: str
    source_kind: str
    source_index: int | None
    source_video_id: str
    path: str
    start_frame: int
    end_frame: int
    fps: float
    native_mask_id: str
    caption: str
    source_url: str | None
    native_mask_path: str | None
    planned_video_path: str
    file_size_bytes: int | None = None
    sha256: str | None = None
    download_status: str = "PLANNED"
    decode_status: str = "NOT_CHECKED"
    frame_count: int | None = None
    resolution: str | None = None
    duration_sec: float | None = None
    license_note: str = "TencentARC/VPData row; Pexels raw video URL where source_kind=pexels"


def open_text_url(url: str, retries: int = 3):
    headers = {"User-Agent": "Mozilla/5.0 exp60b-vpdata-subset/1.0"}
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            return urllib.request.urlopen(req, timeout=120)
        except Exception as exc:
            last_exc = exc
            if attempt >= retries:
                raise
            time.sleep(5 * attempt)
    raise RuntimeError(f"unreachable retry state for {url}: {last_exc}")


def classify_path(path: str) -> tuple[str, int | None, str]:
    m = PEXELS_RE.match(path)
    if m:
        return "pexels", int(m.group("index")), m.group("video_id")
    m = VIDEOVO_RE.match(path)
    if m:
        stem = m.group("stem")
        return "videovo", int(stem), stem
    return "unknown", None, path


def read_pexels_url_map() -> dict[int, tuple[str, str]]:
    mapping: dict[int, tuple[str, str]] = {}
    with open_text_url(PEXELS_CSV) as raw:
        text = (line.decode("utf-8", errors="replace") for line in raw)
        reader = csv.DictReader(text)
        for idx, row in enumerate(reader):
            mapping[idx] = (row["videoId"], row["link"])
    return mapping


def iter_vpdata_csv(url: str) -> Iterable[dict[str, str]]:
    with open_text_url(url) as raw:
        text = (line.decode("utf-8", errors="replace") for line in raw)
        yield from csv.DictReader(text)


def reservoir_sample_unique(
    url: str,
    *,
    split: str,
    limit: int,
    seed: int,
    source_filter: str,
    exclude_source_ids: set[str] | None = None,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    rng = random.Random(seed)
    exclude_source_ids = exclude_source_ids or set()
    selected: list[dict[str, str]] = []
    seen_paths: set[str] = set()
    eligible = 0
    excluded = 0
    total = 0
    counts = {"pexels": 0, "videovo": 0, "unknown": 0}
    for row in iter_vpdata_csv(url):
        total += 1
        kind, _, source_id = classify_path(row["path"])
        counts[kind] = counts.get(kind, 0) + 1
        if source_filter == "pexels_only" and kind != "pexels":
            continue
        if source_filter == "videovo_only" and kind != "videovo":
            continue
        if source_filter == "all" and kind == "unknown":
            continue
        if source_id in exclude_source_ids:
            excluded += 1
            continue
        if row["path"] in seen_paths:
            continue
        seen_paths.add(row["path"])
        eligible += 1
        enriched = dict(row)
        enriched["_source_kind"] = kind
        enriched["_split"] = split
        if len(selected) < limit:
            selected.append(enriched)
        else:
            j = rng.randrange(eligible)
            if j < limit:
                selected[j] = enriched
    stats = {
        "total_rows": total,
        "eligible_unique_rows": eligible,
        "selected_rows": len(selected),
        "pexels_rows_seen": counts["pexels"],
        "videovo_rows_seen": counts["videovo"],
        "unknown_rows_seen": counts["unknown"],
        "excluded_source_overlap_rows": excluded,
    }
    return selected, stats


def source_row_from_csv(
    row: dict[str, str],
    *,
    split: str,
    output_root: Path,
    pexels_map: dict[int, tuple[str, str]],
) -> SourceRow:
    kind, index, source_id = classify_path(row["path"])
    source_url: str | None = None
    if kind == "pexels":
        if index is None or index not in pexels_map:
            raise RuntimeError(f"Pexels path has no URL mapping: {row['path']}")
        mapped_video_id, source_url = pexels_map[index]
        if mapped_video_id != source_id:
            raise RuntimeError(f"Pexels video id mismatch for {row['path']}: {mapped_video_id} != {source_id}")
        rel_video = Path("raw_video") / "pexels" / f"{index:012d}"[:9] / row["path"]
    elif kind == "videovo":
        # VideoVo raw videos require zip-shard handling; keep planned path only.
        rel_video = Path("raw_video") / "videovo" / source_id[:9] / row["path"]
    else:
        rel_video = Path("raw_video") / "unknown" / row["path"]

    native_mask_path = None
    if kind == "pexels" and index is not None:
        shard_start = (index // 100) * 100
        native_mask_path = f"pexels_masks/pexels-{shard_start}-{shard_start + 99}.zip::pexels/{index:012d}/all_masks.npz"
    elif kind == "videovo" and index is not None:
        shard_start = (index // 100) * 100
        native_mask_path = f"videovo_masks/videovo-{shard_start}-{shard_start + 99}.zip::videovo/{source_id}/all_masks.npz"

    vpdata_id = f"{split}_{kind}_{row['path'].replace('/', '_').replace('.mp4', '')}_mask{row['mask_id']}"
    return SourceRow(
        split=split,
        vpdata_id=vpdata_id,
        source_kind=kind,
        source_index=index,
        source_video_id=source_id,
        path=row["path"],
        start_frame=int(float(row["start_frame"])),
        end_frame=int(float(row["end_frame"])),
        fps=float(row["fps"]),
        native_mask_id=str(row["mask_id"]),
        caption=row.get("caption", ""),
        source_url=source_url,
        native_mask_path=native_mask_path,
        planned_video_path=str(output_root / rel_video),
    )


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ffprobe(path: Path) -> tuple[str, int | None, str | None, float | None]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_frames,width,height,duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        proc = subprocess.run(cmd, check=False, text=True, capture_output=True, timeout=30)
    except Exception as exc:
        return f"FFPROBE_ERROR:{type(exc).__name__}:{exc}", None, None, None
    if proc.returncode != 0:
        return f"FFPROBE_ERROR:{proc.stderr.strip()[:200]}", None, None, None
    try:
        data = json.loads(proc.stdout)
        stream = data.get("streams", [{}])[0]
        frames = stream.get("nb_frames")
        width = stream.get("width")
        height = stream.get("height")
        duration = stream.get("duration")
        return (
            "OK",
            int(frames) if frames not in (None, "N/A") else None,
            f"{width}x{height}" if width and height else None,
            float(duration) if duration not in (None, "N/A") else None,
        )
    except Exception as exc:
        return f"FFPROBE_PARSE_ERROR:{type(exc).__name__}:{exc}", None, None, None


def download_url(url: str, dest: Path, retries: int = 3) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            headers = {"User-Agent": "Mozilla/5.0 exp60b-vpdata-subset/1.0"}
            mode = "wb"
            start = tmp.stat().st_size if tmp.exists() else 0
            if start > 0:
                headers["Range"] = f"bytes={start}-"
                mode = "ab"
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as r:
                if start > 0 and getattr(r, "status", None) != 206:
                    # Server ignored Range; restart cleanly instead of corrupting.
                    mode = "wb"
                with tmp.open(mode) as f:
                    while True:
                        chunk = r.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
            if tmp.stat().st_size <= 0:
                raise RuntimeError("download produced an empty file")
            tmp.replace(dest)
            return
        except Exception:
            if attempt >= retries:
                raise
            time.sleep(5 * attempt)

def materialize(rows: list[SourceRow], *, download: bool, max_downloads: int) -> list[dict]:
    out: list[dict] = []
    downloaded = 0
    for row in rows:
        data = asdict(row)
        path = Path(row.planned_video_path)
        if download and row.source_kind != "pexels":
            data["download_status"] = "SKIPPED_NON_ROW_LEVEL_SOURCE"
        elif download and row.source_url:
            if downloaded >= max_downloads:
                data["download_status"] = "SKIPPED_MAX_DOWNLOADS_REACHED"
            else:
                try:
                    if not path.exists():
                        download_url(row.source_url, path)
                    data["file_size_bytes"] = path.stat().st_size
                    data["sha256"] = sha256_file(path)
                    status, frames, resolution, duration = ffprobe(path)
                    data["decode_status"] = status
                    data["frame_count"] = frames
                    data["resolution"] = resolution
                    data["duration_sec"] = duration
                    data["download_status"] = "DOWNLOADED"
                    downloaded += 1
                except Exception as exc:
                    data["download_status"] = f"DOWNLOAD_FAILED:{type(exc).__name__}:{exc}"
        out.append(data)
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_root", required=True, help="H20/PAI raw subset root")
    parser.add_argument("--manifest_dir", required=True)
    parser.add_argument("--reports_dir", required=True)
    parser.add_argument("--seed", type=int, default=20260702)
    parser.add_argument("--max_train", type=int, default=1000)
    parser.add_argument("--max_test", type=int, default=100)
    parser.add_argument("--source_filter", choices=["pexels_only", "videovo_only", "all"], default="pexels_only")
    parser.add_argument("--test_split", choices=["test", "val"], default="test")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--max_downloads", type=int, default=1100)
    parser.add_argument("--report_prefix", default="exp60b_vpdata_subset")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.max_train > 1000 or args.max_test > 100:
        raise SystemExit("[guard] Exp60B max is train1000/test100")
    output_root = Path(args.output_root)
    manifest_dir = Path(args.manifest_dir)
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    pexels_map = read_pexels_url_map()
    train_raw, train_stats = reservoir_sample_unique(
        TRAIN_CSV,
        split="train",
        limit=args.max_train,
        seed=args.seed,
        source_filter=args.source_filter,
    )
    train_source_ids = {classify_path(r["path"])[2] for r in train_raw}
    test_url = TEST_CSV if args.test_split == "test" else VAL_CSV
    test_raw, test_stats = reservoir_sample_unique(
        test_url,
        split=args.test_split,
        limit=args.max_test,
        seed=args.seed + 17,
        source_filter=args.source_filter,
        exclude_source_ids=train_source_ids,
    )
    train_rows = [source_row_from_csv(r, split="train", output_root=output_root, pexels_map=pexels_map) for r in train_raw]
    test_rows = [source_row_from_csv(r, split=args.test_split, output_root=output_root, pexels_map=pexels_map) for r in test_raw]

    train_ids = {r.source_video_id for r in train_rows}
    test_ids = {r.source_video_id for r in test_rows}
    overlap = sorted(train_ids & test_ids)
    if overlap:
        raise SystemExit(f"[guard] train/test source_video_id overlap: {overlap[:10]}")

    max_downloads = min(args.max_downloads, args.max_train + args.max_test)
    train_out = materialize(train_rows, download=args.download, max_downloads=max_downloads)
    remaining = max(0, max_downloads - sum(1 for r in train_out if r.get("download_status") == "DOWNLOADED"))
    test_out = materialize(test_rows, download=args.download, max_downloads=remaining)

    suffix = "h20" if "/home/nvme01" in str(output_root) else "pai"
    train_manifest = manifest_dir / f"exp60b_vpdata_train{args.max_train}_sources_{suffix}.jsonl"
    test_manifest = manifest_dir / f"exp60b_vpdata_test{args.max_test}_sources_{suffix}.jsonl"
    write_jsonl(train_manifest, train_out)
    write_jsonl(test_manifest, test_out)

    summary = {
        "status": "EXP60B_H20_VPDATA_SUBSET_READY" if args.download else "EXP60B_VPDATA_SUBSET_PLAN_READY",
        "download": bool(args.download),
        "hf_endpoint": os.environ.get("HF_ENDPOINT"),
        "source_filter": args.source_filter,
        "seed": args.seed,
        "train_manifest": str(train_manifest),
        "test_manifest": str(test_manifest),
        "train_stats": train_stats,
        "test_stats": test_stats,
        "train_selected": len(train_out),
        "test_selected": len(test_out),
        "downloaded": sum(1 for r in train_out + test_out if r.get("download_status") == "DOWNLOADED"),
        "train_test_overlap_count": 0,
        "native_masks": "audit-only planned shard references; not downloaded by default",
        "full_vpdata_downloaded": False,
    }
    summary_name = f"{args.report_prefix}_{'download' if args.download else 'plan'}_summary.json"
    csv_name = f"{args.report_prefix}_{'download' if args.download else 'plan'}.csv"
    (reports_dir / summary_name).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    # Preserve the original plan filenames used by the first Exp60B milestone.
    if not args.download and args.report_prefix == "exp60b_vpdata_subset":
        (reports_dir / "exp60b_vpdata_subset_plan_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (reports_dir / csv_name).open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "split",
            "vpdata_id",
            "source_kind",
            "source_index",
            "source_video_id",
            "path",
            "start_frame",
            "end_frame",
            "fps",
            "native_mask_id",
            "planned_video_path",
            "download_status",
            "decode_status",
            "file_size_bytes",
            "sha256",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in train_out + test_out:
            writer.writerow({k: row.get(k) for k in fieldnames})
    if not args.download and args.report_prefix == "exp60b_vpdata_subset":
        # Keep the original plan CSV path stable for already-written reports.
        src = reports_dir / csv_name
        dst = reports_dir / "exp60b_vpdata_subset_plan.csv"
        if src != dst:
            dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    failed = [
        row for row in train_out + test_out
        if row.get("source_url") and row.get("download_status") not in {"DOWNLOADED", "PLANNED"}
    ]
    with (reports_dir / f"{args.report_prefix}_failed_urls.txt").open("w", encoding="utf-8") as f:
        for row in failed:
            f.write(f"{row.get('split')}\t{row.get('vpdata_id')}\t{row.get('download_status')}\t{row.get('source_url')}\n")
    with (reports_dir / f"{args.report_prefix}_sha256.txt").open("w", encoding="utf-8") as f:
        for row in train_out + test_out:
            if row.get("sha256"):
                f.write(f"{row['sha256']}  {row['planned_video_path']}\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
