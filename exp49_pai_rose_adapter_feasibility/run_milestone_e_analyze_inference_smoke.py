#!/usr/bin/env python3
"""Analyze Exp49 ROSE official inference smoke outputs."""

from __future__ import annotations

import csv
import json
import math
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity


PROJECT_ROOT = Path("/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp49_rose_adapter")
BASE = Path("/mnt/nas/hj/H20_Video_inpainting_DPO")
REPORT_DIR = PROJECT_ROOT / "reports"
REGISTRY_DIR = PROJECT_ROOT / "experiment_registry/exp49_pai_rose_adapter_feasibility"
OUTPUT_ROOT = BASE / "experiments/dpo/exp49_pai_rose_adapter_feasibility"
LOG_ROOT = BASE / "logs/autoresearch/exp49_pai_rose_adapter_feasibility"
VOR_ROOT = BASE / "data/external/effecterase_vor/extracted/vor_gate128_exact_20260623"

DEMO_OUT = OUTPUT_ROOT / "official_demo_smoke_default_20260630_083542"
DEMO_LOG = LOG_ROOT / "milestone_e_official_demo_default_20260630_083542.log"
VOR_OUT = OUTPUT_ROOT / "vor_or_smoke6_20260630_083835"
VOR_LOG = LOG_ROOT / "milestone_e_vor_or_smoke6_20260630_083835.log"

IDS = [
    "BLENDER_CON001_00332",
    "BLENDER_FOREST039_00117",
    "BLENDER_FOREST039_00530",
    "REAL_ENV024_00002_008_01",
    "REAL_ENV114_00004_004_02",
    "REAL_ENV159_00010_003_05",
]

VISUAL_REVIEW = {
    "BLENDER_CON001_00332": (
        "ROSE_OUTPUT_USABLE",
        "Clean removal in the inspected start/mid/end sheet; output is close to BG and outside region is stable.",
    ),
    "BLENDER_FOREST039_00117": (
        "SIDE_EFFECT_LEFT",
        "Object region still leaves visible local residual/boundary mismatch against the rocky background.",
    ),
    "BLENDER_FOREST039_00530": (
        "SIDE_EFFECT_LEFT",
        "Object region remains visually inconsistent with BG; useful as a failure/loser candidate, not clean baseline.",
    ),
    "REAL_ENV024_00002_008_01": (
        "MEDIUM_HARD_ELIGIBLE",
        "Person is removed, but the local fence/grass region has a clear bounded smear; outside is not collapsed.",
    ),
    "REAL_ENV114_00004_004_02": (
        "ROSE_OUTPUT_USABLE",
        "Person is removed and inspected frames look close to BG; no obvious black frame or global collapse.",
    ),
    "REAL_ENV159_00010_003_05": (
        "MEDIUM_HARD_ELIGIBLE",
        "Cropped person is removed, but a visible local wall ghost remains; bounded and useful as medium-hard loser.",
    ),
}


def run(cmd: list[str], cwd: Path | None = None) -> str:
    return subprocess.check_output(cmd, cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()


def append_once(path: Path, marker: str, block: str) -> None:
    text = path.read_text()
    if marker not in text:
        path.write_text(text.rstrip() + "\n\n" + block.strip() + "\n")


def read_video(path: Path, limit: int = 17, size: tuple[int, int] | None = None) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    while len(frames) < limit:
        ok, frame = cap.read()
        if not ok:
            break
        if size is not None:
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        return np.zeros((0, 0, 0, 3), dtype=np.uint8)
    return np.stack(frames, axis=0)


def video_meta(path: Path) -> dict[str, object]:
    cap = cv2.VideoCapture(str(path))
    meta = {
        "exists": path.exists(),
        "decode_ok": cap.isOpened(),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if cap.isOpened() else 0,
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if cap.isOpened() else 0,
        "fps": float(cap.get(cv2.CAP_PROP_FPS)) if cap.isOpened() else 0.0,
        "bytes": path.stat().st_size if path.exists() else 0,
    }
    cap.release()
    return meta


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    if mse <= 1e-12:
        return 99.0
    return 20.0 * math.log10(255.0 / math.sqrt(float(mse)))


def l1(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))


def ssim_video(a: np.ndarray, b: np.ndarray) -> float:
    vals = []
    for fa, fb in zip(a, b):
        try:
            vals.append(structural_similarity(fa, fb, channel_axis=2, data_range=255))
        except Exception:
            pass
    return float(np.mean(vals)) if vals else float("nan")


def masked_values(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    mask3 = np.repeat(mask[..., None], 3, axis=-1)
    return arr[mask3]


def region_psnr(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    av = masked_values(a, mask)
    bv = masked_values(b, mask)
    if av.size == 0:
        return float("nan")
    return psnr(av, bv)


def region_l1(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    av = masked_values(a, mask)
    bv = masked_values(b, mask)
    if av.size == 0:
        return float("nan")
    return l1(av, bv)


def mask_to_bool(mask_rgb: np.ndarray) -> np.ndarray:
    gray = mask_rgb.mean(axis=-1)
    return gray >= 128


def boundary_mask(mask: np.ndarray) -> np.ndarray:
    out = []
    kernel = np.ones((9, 9), np.uint8)
    for m in mask.astype(np.uint8):
        dil = cv2.dilate(m, kernel, iterations=1).astype(bool)
        ero = cv2.erode(m, kernel, iterations=1).astype(bool)
        out.append(np.logical_xor(dil, ero))
    return np.stack(out, axis=0)


def temporal_flicker(frames: np.ndarray) -> float:
    if len(frames) < 2:
        return float("nan")
    diff = np.abs(frames[1:].astype(np.float32) - frames[:-1].astype(np.float32))
    return float(np.mean(diff))


def make_sheet(condition: np.ndarray, mask: np.ndarray, output: np.ndarray, bg: np.ndarray, out_path: Path) -> None:
    idxs = [0, min(len(output) - 1, len(output) // 2), len(output) - 1]
    rows = []
    for idx in idxs:
        mask_vis = np.repeat(mask[idx][..., None], 3, axis=-1).astype(np.uint8) * 255
        row = np.concatenate([condition[idx], mask_vis, output[idx], bg[idx]], axis=1)
        rows.append(row)
    sheet = np.concatenate(rows, axis=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))


def make_temporal_strip(output: np.ndarray, out_path: Path, n: int = 16) -> None:
    if len(output) == 0:
        return
    idxs = np.linspace(0, len(output) - 1, min(n, len(output))).round().astype(int)
    thumbs = []
    for idx in idxs:
        frame = cv2.resize(output[idx], (160, 106), interpolation=cv2.INTER_AREA)
        thumbs.append(frame)
    strip = np.concatenate(thumbs, axis=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))


def classify(row: dict[str, object]) -> str:
    if not row["decode_ok"]:
        return "TECHNICAL_INVALID"
    outside = float(row["outside_l1"])
    mask_score = float(row["mask_psnr"]) if row["mask_psnr"] == row["mask_psnr"] else -999
    full = float(row["full_psnr"])
    if outside > 30:
        return "OUTSIDE_DAMAGE"
    if mask_score < 18:
        return "SIDE_EFFECT_LEFT"
    if full > 35 and mask_score > 30:
        return "TOO_CLOSE"
    return "MEDIUM_HARD_ELIGIBLE"


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    evidence_dir = REPORT_DIR / "exp49_rose_vor_or_smoke_evidence"
    branch = run(["git", "branch", "--show-current"], PROJECT_ROOT)
    commit = run(["git", "rev-parse", "HEAD"], PROJECT_ROOT)
    generated = datetime.now(timezone.utc).astimezone().isoformat()
    host = socket.gethostname()

    demo_meta = video_meta(DEMO_OUT / "example-1.mp4")

    rows: list[dict[str, object]] = []
    visual_rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    for idx, sample_id in enumerate(IDS, start=1):
        condition_path = VOR_ROOT / f"VOR-Train/VOR-Train/FG_BG/{sample_id}.mp4"
        bg_path = VOR_ROOT / f"VOR-Train/VOR-Train/BG/{sample_id}.mp4"
        mask_path = VOR_ROOT / f"VOR-Train-MASK/MASK/{sample_id}.mp4"
        output_path = VOR_OUT / f"example-{idx}.mp4"
        meta = video_meta(output_path)
        output = read_video(output_path, 17)
        if len(output):
            size = (output.shape[2], output.shape[1])
        else:
            size = (720, 480)
        condition = read_video(condition_path, 17, size=size)
        bg = read_video(bg_path, 17, size=size)
        mask_rgb = read_video(mask_path, 17, size=size)
        n = min(len(output), len(condition), len(bg), len(mask_rgb))
        output, condition, bg, mask_rgb = output[:n], condition[:n], bg[:n], mask_rgb[:n]
        mask = mask_to_bool(mask_rgb)
        boundary = boundary_mask(mask) if n else mask
        affected = np.mean(np.abs(condition.astype(np.float32) - bg.astype(np.float32)), axis=-1) > 15 if n else mask
        outside = np.logical_not(mask)

        sheet_path = evidence_dir / f"{sample_id}_sheet.jpg"
        strip_path = evidence_dir / f"{sample_id}_temporal_strip.jpg"
        if n:
            make_sheet(condition, mask, output, bg, sheet_path)
            make_temporal_strip(output, strip_path)

        row = {
            "sample_id": sample_id,
            "output_path": str(output_path),
            "condition_path": str(condition_path),
            "bg_path": str(bg_path),
            "mask_path": str(mask_path),
            "decode_ok": bool(meta["decode_ok"]) and n >= 1,
            "frame_count": meta["frame_count"],
            "width": meta["width"],
            "height": meta["height"],
            "bytes": meta["bytes"],
            "frames_used": n,
            "full_psnr": psnr(output, bg),
            "full_ssim": ssim_video(output, bg),
            "full_l1": l1(output, bg),
            "mask_psnr": region_psnr(output, bg, mask),
            "boundary_psnr": region_psnr(output, bg, boundary),
            "affected_psnr": region_psnr(output, bg, affected),
            "outside_psnr": region_psnr(output, bg, outside),
            "outside_l1": region_l1(output, bg, outside),
            "condition_to_bg_psnr": psnr(condition, bg),
            "condition_to_bg_l1": l1(condition, bg),
            "output_flicker": temporal_flicker(output),
            "condition_flicker": temporal_flicker(condition),
            "sheet_path": str(sheet_path),
            "temporal_strip_path": str(strip_path),
        }
        row["auto_classification"] = classify(row)
        rows.append(row)
        visual_label, visual_note = VISUAL_REVIEW[sample_id]
        visual_rows.append({
            "sample_id": sample_id,
            "reviewed": "YES",
            "visual_label": visual_label,
            "notes": visual_note,
            "sheet_path": str(sheet_path),
            "temporal_strip_path": str(strip_path),
        })
        manifest_rows.append({
            "sample_id": sample_id,
            "source": "VOR-Train",
            "condition_path": str(condition_path),
            "mask_path": str(mask_path),
            "gt_bg_path": str(bg_path),
            "rose_output_path": str(output_path),
            "evidence_sheet": str(sheet_path),
            "temporal_strip": str(strip_path),
        })

    metrics_path = REPORT_DIR / "exp49_rose_vor_or_smoke_metrics.csv"
    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    visual_path = REPORT_DIR / "exp49_rose_vor_or_smoke_visual_review.csv"
    with visual_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(visual_rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(visual_rows)

    manifest_dir = PROJECT_ROOT / "manifests"
    manifest_dir.mkdir(exist_ok=True)
    with (manifest_dir / "exp49_rose_vor_or_smoke6_manifest.jsonl").open("w") as f:
        for row in manifest_rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    technical_valid = sum(1 for row in rows if row["decode_ok"])
    output_count = sum(1 for row in rows if Path(str(row["output_path"])).exists())
    status = "ROSE_INFERENCE_SMOKE_PASS" if demo_meta["decode_ok"] and technical_valid == len(rows) else "ROSE_INFERENCE_SMOKE_BLOCKED"

    summary = {
        "generated": generated,
        "hostname": host,
        "branch": branch,
        "commit": commit,
        "status": status,
        "e1_demo_output": str(DEMO_OUT / "example-1.mp4"),
        "e1_demo_meta": demo_meta,
        "e1_first_attempt": {
            "status": "blocked_expected_shape_mismatch",
            "log": str(LOG_ROOT / "milestone_e_official_demo_20260630_082754.log"),
            "note": "A reduced 256x384 attempt reached transformer forward but failed seq_len assertion because official inference.py does not pass height/width through to the pipeline; default 480x720 run passed.",
        },
        "e2_vor_output_dir": str(VOR_OUT),
        "e2_outputs": output_count,
        "technical_valid": technical_valid,
        "total_vor_rows": len(rows),
        "demo_log": str(DEMO_LOG),
        "vor_log": str(VOR_LOG),
        "metrics_csv": str(metrics_path),
        "visual_review_csv": str(visual_path),
        "manifest": str(manifest_dir / "exp49_rose_vor_or_smoke6_manifest.jsonl"),
        "visual_counts": {
            label: sum(1 for row in visual_rows if row["visual_label"] == label)
            for label in sorted({row["visual_label"] for row in visual_rows})
        },
        "note": "Codex inspected all six start/mid/end evidence sheets and representative temporal strips.",
    }
    (REPORT_DIR / "exp49_rose_official_inference_summary.json").write_text(json.dumps(summary, indent=2))

    md = [
        "# Exp49 ROSE Official Inference Smoke",
        "",
        f"Status: `{status}`",
        "",
        f"Generated: {generated}",
        f"Host: `{host}`",
        f"Branch: `{branch}`",
        f"Commit: `{commit}`",
        "",
        "## E1 Official Demo",
        "",
        f"- Output: `{DEMO_OUT / 'example-1.mp4'}`",
        f"- Decode ok: `{demo_meta['decode_ok']}`",
        f"- Frames: `{demo_meta['frame_count']}`",
        f"- Resolution: `{demo_meta['width']}x{demo_meta['height']}`",
        f"- Log: `{DEMO_LOG}`",
        "- Reduced 256x384 probe failed with a transformer `seq_len` assertion because official `inference.py` does not propagate reduced `height/width`; the default 480x720 smoke passed without modifying official source.",
        "",
        "## E2 VOR-Train Smoke6",
        "",
        f"- Output dir: `{VOR_OUT}`",
        f"- Technical valid: `{technical_valid}/{len(rows)}`",
        f"- Metrics: `reports/exp49_rose_vor_or_smoke_metrics.csv`",
        f"- Visual review: `reports/exp49_rose_vor_or_smoke_visual_review.csv`",
        f"- Evidence dir: `{evidence_dir}`",
        "- Codex visual inspection: `ROSE_OUTPUT_USABLE=2`, `MEDIUM_HARD_ELIGIBLE=2`, `SIDE_EFFECT_LEFT=2`.",
        "",
        "## Safety",
        "",
        "No training, optimizer step, checkpoint update, VOR-Eval use, H20 action, hard comp, shared trainer change, metrics code change, or official ROSE source modification was performed.",
    ]
    (REPORT_DIR / "exp49_rose_official_inference_smoke.md").write_text("\n".join(md) + "\n")

    append_once(
        PROJECT_ROOT / "PRD/00_current_status.md",
        "2026-06-30 Exp49 ROSE Official Inference Smoke",
        f"""## 2026-06-30 Exp49 ROSE Official Inference Smoke

Status: `{status}`.

ROSE official `inference.py` ran on PAI GPU0 for one official demo sample and six VOR-Train smoke rows. It produced `{technical_valid}/{len(rows)}` technically valid VOR outputs. The reduced-size probe failed at transformer `seq_len` because official `inference.py` does not pass reduced height/width into the pipeline; default 480x720 inference passed. No training, optimizer step, H20 action, VOR-Eval use, hard comp, or official source modification was performed.
""",
    )
    append_once(
        PROJECT_ROOT / "PRD/01_experiment_matrix.md",
        "2026-06-30 Exp49 ROSE Inference Smoke",
        f"""## 2026-06-30 Exp49 ROSE Inference Smoke

| Experiment | Milestone | Status | Notes |
| --- | --- | --- | --- |
| `exp49_pai_rose_adapter_feasibility` | E official inference smoke | `{status}` | Official demo plus six VOR-Train rows generated `{technical_valid}/{len(rows)}` decodable outputs on PAI GPU0; no training. |
""",
    )
    append_once(
        PROJECT_ROOT / "PRD/46_exp49_pai_rose_adapter_feasibility.md",
        "## Milestone E Update - 2026-06-30",
        f"""## Milestone E Update - 2026-06-30

Status: `{status}`.

Official ROSE `inference.py` loaded the downloaded Wan2.1-Fun base and Kunbyte/ROSE transformer weights via isolated runtime symlinks. E1 default-size demo produced a decodable mp4. E2 VOR-Train smoke6 produced `{technical_valid}/{len(rows)}` decodable outputs. A reduced 256x384 probe is recorded as blocked because official `inference.py` does not pass reduced `height/width` into the pipeline, causing a transformer sequence-length assertion; official default 480x720 passed without modifying official source.

No training, optimizer step, checkpoint update, VOR-Eval use, H20 action, hard comp, shared trainer change, metrics code change, or official ROSE source modification was performed.
""",
    )

    (REGISTRY_DIR / "metric_summary.md").write_text(
        "# Exp49 Metric Summary\n\n"
        f"Milestone E official ROSE inference smoke status: `{status}`. "
        f"VOR-Train technical-valid outputs: `{technical_valid}/{len(rows)}`. "
        "See `reports/exp49_rose_vor_or_smoke_metrics.csv`.\n"
    )
    append_once(
        REGISTRY_DIR / "status.md",
        "Official inference smoke:",
        f"Official inference smoke: `{status}` with `{technical_valid}/{len(rows)}` VOR-Train outputs technically valid.",
    )
    result_line = f"E_inference_smoke\t{status}\tOfficial demo plus six VOR-Train rows; technical valid {technical_valid}/{len(rows)}; no training.\n"
    results_path = REGISTRY_DIR / "results.tsv"
    if result_line.strip() not in results_path.read_text():
        with results_path.open("a") as f:
            f.write(result_line)
    (REGISTRY_DIR / "qualitative_summary.md").write_text(
        "# Exp49 Qualitative Summary\n\n"
        "Milestone E generated evidence sheets under "
        f"`{evidence_dir}`. Codex inspected all six start/mid/end sheets plus representative temporal strips. "
        "Visual outcome: `ROSE_OUTPUT_USABLE=2`, `MEDIUM_HARD_ELIGIBLE=2`, `SIDE_EFFECT_LEFT=2`. "
        "No black-frame collapse was observed; this is baseline/loser-generator evidence only, not adapter-positive evidence.\n"
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
