#!/usr/bin/env python3
"""Exp47 manifest/path/frame alignment audit for Exp46 pseudo-success SFT."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

SPLITS = ("train", "search", "shadow")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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
        writer.writerows(rows)


def frame_files(path: Path) -> list[Path]:
    return sorted([p for p in path.glob("*.png")] + [p for p in path.glob("*.jpg")] + [p for p in path.glob("*.jpeg")])


def read_rgb_image(path: Path) -> np.ndarray | None:
    arr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if arr is None:
        return None
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


def read_probe_dir(path: str) -> dict[str, Any]:
    p = Path(path)
    files = frame_files(p) if p.exists() and p.is_dir() else []
    out: dict[str, Any] = {
        "exists": p.exists(),
        "is_dir": p.is_dir(),
        "frame_count": len(files),
        "first_name": files[0].name if files else "",
        "mid_name": files[len(files)//2].name if files else "",
        "last_name": files[-1].name if files else "",
        "width": "",
        "height": "",
        "decode_ok": False,
    }
    if files:
        arr = read_rgb_image(files[len(files)//2])
        if arr is not None:
            out.update({"decode_ok": True, "height": int(arr.shape[0]), "width": int(arr.shape[1])})
    return out


def read_mp4_probe(path: str) -> dict[str, Any]:
    p = Path(path)
    out: dict[str, Any] = {"exists": p.exists(), "is_file": p.is_file(), "frame_count": 0, "width": "", "height": "", "decode_ok": False}
    if not p.exists() or not p.is_file():
        return out
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        return out
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    out["frame_count"] = n
    out["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or ""
    out["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or ""
    idx = max(0, min(n - 1, n // 2)) if n else 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    out["decode_ok"] = bool(ok and frame is not None)
    return out


def mid_frame_dir(path: str) -> np.ndarray | None:
    files = frame_files(Path(path))
    if not files:
        return None
    return read_rgb_image(files[len(files)//2])


def mid_frame_mp4(path: str) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    idx = max(0, min(n - 1, n // 2)) if n else 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def l1(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return float("nan")
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))


def mask_stats(path: str) -> dict[str, Any]:
    files = frame_files(Path(path))
    if not files:
        return {"mask_decode_ok": False, "mask_nonempty_frames": 0, "mask_area_mean": float("nan"), "mask_polarity_ok": False}
    areas = []
    ok_count = 0
    for f in files:
        arr = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if arr is None:
            continue
        ok_count += 1
        areas.append(float(np.mean(arr > 20)))
    mean = float(np.mean(areas)) if areas else float("nan")
    nonempty = sum(a > 1e-5 for a in areas)
    return {
        "mask_decode_ok": ok_count == len(files),
        "mask_nonempty_frames": nonempty,
        "mask_area_mean": mean,
        "mask_polarity_ok": bool(math.isfinite(mean) and 0.0001 < mean < 0.5 and nonempty == len(files)),
    }


def path_is_h20_local(path: str) -> bool:
    return path.startswith("/home/nvme01/")


def path_is_pai_abs(path: str) -> bool:
    return path.startswith("/mnt/nas/") or path.startswith("/mnt/workspace/")


def audit(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.worktree)
    rows: list[dict[str, Any]] = []
    split_groups: dict[str, set[str]] = {s: set() for s in SPLITS}
    split_source_ids: dict[str, set[str]] = {s: set() for s in SPLITS}
    for split in SPLITS:
        runner_path = root / f"manifests/exp46_runner_pseudosuccess_{split}.jsonl"
        original_path = root / f"manifests/exp45_h20_pseudosuccess_{split}.jsonl"
        runner_rows = read_jsonl(runner_path)
        original_by_id = {str(r.get("row_id") or r.get("sample_id")): r for r in read_jsonl(original_path)}
        for idx, row in enumerate(runner_rows):
            row_id = str(row.get("row_id") or row.get("sample_id"))
            orig = original_by_id.get(row_id, {})
            condition_path = str(row.get("condition_path", ""))
            mask_path = str(row.get("mask_path", ""))
            target_frames = str(row.get("winner_path", ""))
            pseudo_mp4 = str(row.get("pseudo_success_mp4") or orig.get("target_path", ""))
            gt_path = str(row.get("loser_path") or orig.get("gt_background_path", ""))
            source_group = str(row.get("source_group", ""))
            source_id = str(row.get("source_id", ""))
            split_groups[split].add(source_group)
            split_source_ids[split].add(source_id)
            c_probe = read_probe_dir(condition_path)
            m_probe = read_probe_dir(mask_path)
            t_probe = read_probe_dir(target_frames)
            g_probe = read_probe_dir(gt_path)
            mp4_probe = read_mp4_probe(pseudo_mp4)
            orig_frames_probe = read_probe_dir(str(orig.get("target_frames_dir", ""))) if orig else {"frame_count": "", "exists": False}
            target_mid = mid_frame_dir(target_frames)
            mp4_mid = mid_frame_mp4(pseudo_mp4)
            cond_mid = mid_frame_dir(condition_path)
            gt_mid = mid_frame_dir(gt_path)
            mask = mask_stats(mask_path)
            target_mp4_l1 = l1(target_mid, mp4_mid)
            target_condition_l1 = l1(target_mid, cond_mid)
            target_gt_l1 = l1(target_mid, gt_mid)
            rgb_l1 = target_mp4_l1
            bgr_l1 = l1(target_mid, None if mp4_mid is None else mp4_mid[..., ::-1])
            active_paths = [condition_path, mask_path, target_frames, gt_path]
            active_exist = all(Path(p).exists() for p in active_paths)
            active_h20 = all(path_is_h20_local(p) for p in active_paths)
            active_not_pai_abs = all(not path_is_pai_abs(p) for p in active_paths)
            no_hal = all("/home/hj" not in p and "hal-9000" not in p for p in active_paths)
            counts = [c_probe["frame_count"], m_probe["frame_count"], t_probe["frame_count"], g_probe["frame_count"]]
            frame_count_consistent = all(v == int(row.get("num_frames", 17)) for v in counts)
            res_consistent = len({(c_probe["width"], c_probe["height"]), (t_probe["width"], t_probe["height"]), (g_probe["width"], g_probe["height"])}) == 1
            source_match = source_id in condition_path and source_id in mask_path and source_id in gt_path and source_group in pseudo_mp4
            target_is_pseudo = row.get("target_type") == "visually_approved_minimax_pseudo_success" and "pseudosuccess_target_frames" in target_frames and bool(row.get("pseudo_success_label"))
            target_not_gt_or_condition = target_frames != gt_path and target_frames != condition_path and target_gt_l1 > 0.01 and target_condition_l1 > 0.01
            target_matches_mp4 = bool(math.isfinite(target_mp4_l1) and target_mp4_l1 < 3.0)
            rgb_bgr_ok = bool(math.isfinite(rgb_l1) and (not math.isfinite(bgr_l1) or rgb_l1 <= bgr_l1))
            no_vor_eval = not bool(row.get("vor_eval_used")) and not bool(orig.get("vor_eval_used"))
            no_hard_comp = not bool(row.get("hard_comp_used")) and not bool(orig.get("hard_comp_used"))
            checks = {
                "active_paths_exist": active_exist,
                "active_paths_h20_local": active_h20,
                "active_paths_not_pai_abs": active_not_pai_abs,
                "active_paths_not_hal": no_hal,
                "frame_count_consistent": frame_count_consistent,
                "resolution_consistent": res_consistent,
                "source_match": source_match,
                "target_is_pseudo_success": target_is_pseudo,
                "target_not_gt_condition": target_not_gt_or_condition,
                "target_frames_match_mp4": target_matches_mp4,
                "rgb_bgr_decode_ok": rgb_bgr_ok,
                "mask_polarity_ok": bool(mask["mask_polarity_ok"]),
                "no_vor_eval": no_vor_eval,
                "no_hard_comp": no_hard_comp,
            }
            status = "PASS" if all(checks.values()) else "FAIL"
            rows.append({
                "split": split,
                "row_index": idx,
                "sample_id": row_id,
                "source_group": source_group,
                "source_id": source_id,
                "condition_path": condition_path,
                "mask_path": mask_path,
                "active_target_frames_dir": target_frames,
                "pseudo_success_path": pseudo_mp4,
                "gt_winner_path": gt_path,
                "original_target_frames_dir": orig.get("target_frames_dir", ""),
                "original_target_frames_count": orig_frames_probe.get("frame_count", ""),
                "condition_frames": c_probe["frame_count"],
                "mask_frames": m_probe["frame_count"],
                "target_frames": t_probe["frame_count"],
                "gt_frames": g_probe["frame_count"],
                "mp4_frames": mp4_probe["frame_count"],
                "width": t_probe["width"],
                "height": t_probe["height"],
                "mask_area_mean": mask["mask_area_mean"],
                "mask_nonempty_frames": mask["mask_nonempty_frames"],
                "target_mid_l1_vs_mp4": target_mp4_l1,
                "target_mid_l1_vs_condition": target_condition_l1,
                "target_mid_l1_vs_gt": target_gt_l1,
                "rgb_l1": rgb_l1,
                "bgr_l1": bgr_l1,
                **checks,
                "row_status": status,
                "failure_reasons": ";".join(k for k, v in checks.items() if not v),
            })
    overlaps = {
        "train_search_group_overlap": len(split_groups["train"] & split_groups["search"]),
        "train_shadow_group_overlap": len(split_groups["train"] & split_groups["shadow"]),
        "search_shadow_group_overlap": len(split_groups["search"] & split_groups["shadow"]),
        "train_search_source_overlap": len(split_source_ids["train"] & split_source_ids["search"]),
        "train_shadow_source_overlap": len(split_source_ids["train"] & split_source_ids["shadow"]),
        "search_shadow_source_overlap": len(split_source_ids["search"] & split_source_ids["shadow"]),
    }
    fail_rows = [r for r in rows if r["row_status"] != "PASS"]
    status = "EXP47_MANIFEST_ALIGNMENT_PASS" if not fail_rows and all(v == 0 for v in overlaps.values()) else "EXP47_MANIFEST_ALIGNMENT_BUG_FOUND"
    summary = {
        "status": status,
        "rows": len(rows),
        "fail_rows": len(fail_rows),
        "overlaps": overlaps,
        "active_target": "winner_path from exp46_runner_pseudosuccess_* manifests",
        "original_target_frames_note": "Exp45 original target_frames_dir may be empty or partial; Exp46 active runner target was extracted from pseudo_success_mp4 into winner_path.",
        "checks": {k: sum(1 for r in rows if r.get(k) is True) for k in ["active_paths_exist", "active_paths_h20_local", "active_paths_not_pai_abs", "active_paths_not_hal", "frame_count_consistent", "resolution_consistent", "source_match", "target_is_pseudo_success", "target_not_gt_condition", "target_frames_match_mp4", "rgb_bgr_decode_ok", "mask_polarity_ok", "no_vor_eval", "no_hard_comp"]},
        "worst_failures": fail_rows[:10],
    }
    reports = Path(args.reports_dir)
    write_csv(reports / "exp47_manifest_path_frame_alignment_audit.csv", rows)
    (reports / "exp47_manifest_path_frame_alignment_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Exp47 Manifest / Path / Frame Alignment Audit",
        "",
        f"Status: `{status}`",
        "",
        f"- Rows audited: `{len(rows)}`",
        f"- Failed rows: `{len(fail_rows)}`",
        f"- Active target field: `winner_path` in `exp46_runner_pseudosuccess_*` manifests",
        "- Original Exp45 `target_path` / `target_frames_dir` were preserved for traceability; Exp46 training used extracted H20-local pseudo-success frame dirs.",
        "",
        "## Split Overlap",
        "",
        "| overlap | count |",
        "| --- | ---: |",
    ]
    for k, v in overlaps.items():
        lines.append(f"| {k} | {v} |")
    lines += ["", "## Check Summary", "", "| check | pass rows | total |", "| --- | ---: | ---: |"]
    for k, v in summary["checks"].items():
        lines.append(f"| {k} | {v} | {len(rows)} |")
    lines += ["", "## Interpretation", ""]
    if status == "EXP47_MANIFEST_ALIGNMENT_PASS":
        lines += [
            "No manifest/path/frame alignment bug was found in the active Exp46 runner manifests.",
            "Active paths are H20-local absolute paths, targets are pseudo-success extracted frame directories, masks are non-empty with expected polarity, frame counts/resolution are consistent, no VOR-Eval or hard-comp rows are active, and split overlap is zero.",
        ]
    else:
        lines += ["Potential alignment bug found; see CSV failure_reasons before any further training."]
    lines += ["", "## Outputs", "", "- `reports/exp47_manifest_path_frame_alignment_audit.csv`", "- `reports/exp47_manifest_path_frame_alignment_summary.json`"]
    (reports / "exp47_manifest_path_frame_alignment_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worktree", default="/home/nvme01/H20_Video_inpainting_DPO_exp47_minimax_pseudosuccess_forensic")
    parser.add_argument("--reports-dir", default="/home/nvme01/H20_Video_inpainting_DPO_exp47_minimax_pseudosuccess_forensic/reports")
    args = parser.parse_args()
    print(json.dumps(audit(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
