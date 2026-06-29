#!/usr/bin/env python3
"""Prepare and validate Exp43 H20 MiniMax Stage2 SFT data splits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
REQUIRED_KEYS = ("condition_path", "winner_path", "mask_path", "loser_path")
OPTIONAL_KEYS = ("raw_output_mp4", "review_sheet", "temporal_strip_16", "side_by_side_mp4")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def image_files(path: Path) -> list[Path]:
    return sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def decode_probe(path: Path) -> str:
    try:
        import imageio.v3 as iio  # noqa: WPS433

        arr = iio.imread(path)
        shape = getattr(arr, "shape", None)
        return f"PASS:{shape}"
    except Exception as exc:  # noqa: BLE001
        return f"FAIL:{type(exc).__name__}:{exc}"


def validate_path(sample_id: str, split: str, key: str, raw: Any, expected_frames: int) -> dict[str, Any]:
    row = {
        "sample_id": sample_id,
        "split": split,
        "key": key,
        "path": str(raw or ""),
        "exists": False,
        "kind": "",
        "entries": "",
        "decode_probe": "",
        "status": "MISSING",
    }
    if not raw:
        return row
    path = Path(str(raw))
    row["exists"] = path.exists()
    if not path.exists():
        return row
    if path.is_dir():
        files = image_files(path)
        row.update(kind="dir", entries=len(files))
        if len(files) < expected_frames:
            row["status"] = "TOO_FEW_FRAMES"
            return row
        row["decode_probe"] = decode_probe(files[0]) if files else "EMPTY_DIR"
        row["status"] = "PASS" if str(row["decode_probe"]).startswith("PASS:") else "DECODE_FAIL"
        return row
    row.update(kind="file", entries=path.stat().st_size)
    row["decode_probe"] = "not_decoded_file"
    row["status"] = "PASS"
    return row


def normalize_rows(rows: list[dict[str, Any]], split: str, source_path: Path) -> list[dict[str, Any]]:
    out = []
    for idx, row in enumerate(rows):
        item = dict(row)
        item["split"] = split
        item["exp43_split"] = split
        item["exp43_manifest_source"] = str(source_path)
        item["exp43_stage2_sft_candidate"] = True
        item["exp43_vor_eval_used"] = False
        item["exp43_hard_comp_used"] = bool(item.get("hard_comp", False))
        item["exp43_row_index"] = idx
        item["condition_role"] = item.get("condition_source_role", "V_obj")
        item["target_role"] = item.get("winner_source_role", "V_bg")
        item["mask_role"] = item.get("mask_source_role", "foreground_object_mask")
        out.append(item)
    return out


def split_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "scene_groups": len({str(r.get("scene_group", r.get("sample_id", ""))) for r in rows}),
        "source_type": dict(Counter(str(r.get("source_type", "unknown")) for r in rows)),
        "mask_bucket": dict(Counter(str(r.get("mask_bucket", "unknown")) for r in rows)),
        "classification": dict(Counter(str(r.get("classification", "unknown")) for r in rows)),
        "profiles": dict(Counter(str(r.get("profile", "unknown")) for r in rows)),
        "vor_eval_used_rows": sum(1 for r in rows if bool(r.get("vor_eval_used")) or bool(r.get("exp43_vor_eval_used"))),
        "hard_comp_rows": sum(1 for r in rows if bool(r.get("hard_comp")) or bool(r.get("exp43_hard_comp_used"))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--search", type=Path, required=True)
    parser.add_argument("--shadow", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--reports-dir", type=Path, required=True)
    parser.add_argument("--expected-frames", type=int, default=17)
    args = parser.parse_args()

    sources = {"train": args.train, "search": args.search, "shadow": args.shadow}
    splits = {name: normalize_rows(read_jsonl(path), name, path) for name, path in sources.items()}
    out_paths = {
        "train": args.manifest_dir / "exp43_stage2_sft_train.jsonl",
        "search": args.manifest_dir / "exp43_stage2_sft_search.jsonl",
        "shadow": args.manifest_dir / "exp43_stage2_sft_shadow.jsonl",
    }
    for split, rows in splits.items():
        write_jsonl(out_paths[split], rows)

    validation_rows: list[dict[str, Any]] = []
    for split, rows in splits.items():
        for row in rows:
            sample_id = str(row.get("sample_id", ""))
            for key in REQUIRED_KEYS + OPTIONAL_KEYS:
                if key in row or key in REQUIRED_KEYS:
                    validation_rows.append(validate_path(sample_id, split, key, row.get(key), args.expected_frames))
    write_csv(args.reports_dir / "exp43_h20_data_manifest_validation.csv", validation_rows)

    scene_sets = {
        split: {str(r.get("scene_group", r.get("sample_id", ""))) for r in rows}
        for split, rows in splits.items()
    }
    overlaps = {
        "train_search": sorted(scene_sets["train"] & scene_sets["search"]),
        "train_shadow": sorted(scene_sets["train"] & scene_sets["shadow"]),
        "search_shadow": sorted(scene_sets["search"] & scene_sets["shadow"]),
    }
    missing = [r for r in validation_rows if r["status"] != "PASS" and r["key"] in REQUIRED_KEYS]
    optional_missing = [r for r in validation_rows if r["status"] != "PASS" and r["key"] in OPTIONAL_KEYS]
    summaries = {split: split_summary(rows) for split, rows in splits.items()}
    minimum_ready = (
        summaries["train"]["rows"] >= 64
        and summaries["search"]["rows"] >= 24
        and summaries["shadow"]["rows"] >= 24
        and not any(overlaps.values())
        and not missing
        and all(v["vor_eval_used_rows"] == 0 for v in summaries.values())
        and all(v["hard_comp_rows"] == 0 for v in summaries.values())
    )
    full_target_ready = summaries["train"]["rows"] >= 96 and summaries["search"]["rows"] >= 32 and summaries["shadow"]["rows"] >= 32
    status = "H20_EXP43_DATA_READY" if minimum_ready else "H20_EXP43_DATA_BLOCKED"
    summary = {
        "status": status,
        "minimum_train64_search24_shadow24_ready": minimum_ready,
        "full_train96_search32_shadow32_ready": full_target_ready,
        "source_manifests": {k: str(v) for k, v in sources.items()},
        "source_manifest_sha256": {k: sha256_file(v) for k, v in sources.items()},
        "output_manifests": {k: str(v) for k, v in out_paths.items()},
        "output_manifest_sha256": {k: sha256_file(v) for k, v in out_paths.items()},
        "summaries": summaries,
        "scene_overlap_counts": {k: len(v) for k, v in overlaps.items()},
        "required_path_failures": len(missing),
        "optional_path_failures": len(optional_missing),
        "vor_eval_excluded": all(v["vor_eval_used_rows"] == 0 for v in summaries.values()),
        "hard_comp_excluded": all(v["hard_comp_rows"] == 0 for v in summaries.values()),
    }
    write_json(args.reports_dir / "exp43_h20_data_summary.json", summary)
    write_csv(
        args.reports_dir / "exp43_h20_data_readiness.csv",
        [
            {
                "split": split,
                "rows": data["rows"],
                "scene_groups": data["scene_groups"],
                "vor_eval_used_rows": data["vor_eval_used_rows"],
                "hard_comp_rows": data["hard_comp_rows"],
                "source_type": json.dumps(data["source_type"], sort_keys=True),
                "mask_bucket": json.dumps(data["mask_bucket"], sort_keys=True),
            }
            for split, data in summaries.items()
        ],
    )
    md = [
        "# Exp43 H20 Data Readiness",
        "",
        f"Status: `{status}`",
        "",
        f"- Train rows: `{summaries['train']['rows']}`",
        f"- Search rows: `{summaries['search']['rows']}`",
        f"- Shadow rows: `{summaries['shadow']['rows']}`",
        f"- Required path failures: `{len(missing)}`",
        f"- Optional path failures: `{len(optional_missing)}`",
        f"- Scene overlap train/search: `{len(overlaps['train_search'])}`",
        f"- Scene overlap train/shadow: `{len(overlaps['train_shadow'])}`",
        f"- Scene overlap search/shadow: `{len(overlaps['search_shadow'])}`",
        f"- VOR-Eval excluded: `{summary['vor_eval_excluded']}`",
        f"- Hard comp excluded: `{summary['hard_comp_excluded']}`",
        f"- Full train96/search32/shadow32 target ready: `{full_target_ready}`",
        "",
        "This is data readiness only. It does not claim MiniMax quality improvement.",
    ]
    (args.reports_dir / "exp43_h20_data_readiness.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
