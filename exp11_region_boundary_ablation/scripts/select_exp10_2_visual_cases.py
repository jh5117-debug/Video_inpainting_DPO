#!/usr/bin/env python3
"""Build Exp10-2 visual case candidates and contact sheets.

This script intentionally does not score videos with a new metric. It reuses
existing side-by-side outputs and existing metric summaries where available.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
from pathlib import Path


def find_exp10_2_roots(search_roots: list[Path]) -> list[Path]:
    matches: list[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            text = str(path).lower()
            if not path.is_dir():
                continue
            if "exp10" in text and ("dpo-s1_dpo-s2" in text or "exp10_2" in text):
                if list(path.rglob("*.mp4")):
                    matches.append(path)
    return sorted(set(matches), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)


def load_summary(root: Path) -> dict[str, str]:
    for candidate in [root / "metrics" / "summary.json", root.parent / "metrics" / "summary.json"]:
        if candidate.exists():
            try:
                return json.loads(candidate.read_text(encoding="utf-8"))
            except Exception:
                return {}
    return {}


def frame_extract(video: Path, out_jpg: Path) -> bool:
    out_jpg.parent.mkdir(parents=True, exist_ok=True)
    try:
        import cv2

        cap = cv2.VideoCapture(str(video))
        if cap.isOpened():
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if frame_count > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_count // 2))
            ok, frame = cap.read()
            cap.release()
            if ok and frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                from PIL import Image

                Image.fromarray(frame).save(out_jpg, quality=92)
                return out_jpg.exists()
    except Exception:
        pass

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        "1",
        "-i",
        str(video),
        "-frames:v",
        "1",
        str(out_jpg),
    ]
    try:
        subprocess.run(cmd, check=True)
        return out_jpg.exists()
    except Exception:
        return False


def make_contact_sheet(images: list[Path], out_path: Path) -> bool:
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return False
    if not images:
        return False
    thumbs = []
    for img_path in images:
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue
        img.thumbnail((900, 260))
        canvas = Image.new("RGB", (920, 300), (18, 24, 32))
        canvas.paste(img, (10, 10))
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 272), img_path.stem[:120], fill=(235, 240, 248))
        thumbs.append(canvas)
    if not thumbs:
        return False
    sheet = Image.new("RGB", (920, 300 * len(thumbs)), (12, 16, 22))
    for idx, thumb in enumerate(thumbs):
        sheet.paste(thumb, (0, idx * 300))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=92)
    return True


def main() -> int:
    project_root = Path(os.environ.get("PROJECT_ROOT", ".")).resolve()
    output_root = Path(os.environ.get("OUTPUT_ROOT", "/mnt/nas/hj/H20_Video_inpainting_DPO"))
    report_dir = project_root / "reports"
    contact_dir = report_dir / "exp10_2_visual_contact_sheets"
    report_dir.mkdir(parents=True, exist_ok=True)
    contact_dir.mkdir(parents=True, exist_ok=True)

    search_roots = [
        project_root / "logs" / "target_eval",
        output_root / "logs" / "target_eval",
        output_root / "reports",
        project_root / "reports",
    ]
    roots = find_exp10_2_roots(search_roots)
    if not roots:
        md = report_dir / "exp10_2_visual_success_tie_failure_cases.md"
        csv_path = report_dir / "exp10_2_visual_success_tie_failure_cases.csv"
        md.write_text("# Exp10-2 Visual Cases\n\nstatus: no side-by-side videos found\n", encoding="utf-8")
        csv_path.write_text(
            "category,video_name,video_path,mask_position,baseline_issue,exp10_improvement,remaining_issue,artifacts,metric_support\n",
            encoding="utf-8",
        )
        print(f"[exp10-visual] no roots found; wrote {md}")
        return 1

    root = roots[0]
    videos = sorted(root.rglob("*.mp4"))
    summary = load_summary(root)
    selected = {
        "success": videos[:5],
        "tie": videos[5:10],
        "failure": videos[-5:] if len(videos) >= 5 else videos,
    }

    rows = []
    extracted: list[Path] = []
    for category, vids in selected.items():
        for video in vids:
            jpg = contact_dir / category / f"{video.stem}.jpg"
            if frame_extract(video, jpg):
                extracted.append(jpg)
            rows.append(
                {
                    "category": category,
                    "video_name": video.stem,
                    "video_path": str(video),
                    "mask_position": "needs visual review from contact sheet",
                    "baseline_issue": "candidate selected from existing Exp10-2 side-by-side set",
                    "exp10_improvement": "needs manual confirmation on contact sheet",
                    "remaining_issue": "needs manual confirmation on contact sheet",
                    "artifacts": "check purple haze / white block / grid / patching",
                    "metric_support": f"global_summary_available={bool(summary)}",
                }
            )

    make_contact_sheet(extracted, contact_dir / "exp10_2_candidates_contact_sheet.jpg")

    csv_path = report_dir / "exp10_2_visual_success_tie_failure_cases.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "category",
                "video_name",
                "video_path",
                "mask_position",
                "baseline_issue",
                "exp10_improvement",
                "remaining_issue",
                "artifacts",
                "metric_support",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    md_lines = [
        "# Exp10-2 Visual Success / Tie / Failure Candidates",
        "",
        f"- selected_root: `{root}`",
        f"- videos_found: `{len(videos)}`",
        f"- contact_sheet: `{contact_dir / 'exp10_2_candidates_contact_sheet.jpg'}`",
        f"- csv: `{csv_path}`",
        "",
        "These are candidates generated from the existing side-by-side output.",
        "Final success/tie/failure labels should be confirmed from the contact sheet and per-video metrics if available.",
        "",
        "| category | video | path |",
        "|---|---|---|",
    ]
    for row in rows:
        md_lines.append(f"| {row['category']} | `{row['video_name']}` | `{row['video_path']}` |")
    (report_dir / "exp10_2_visual_success_tie_failure_cases.md").write_text(
        "\n".join(md_lines) + "\n", encoding="utf-8"
    )
    print(f"[exp10-visual] root={root}")
    print(f"[exp10-visual] wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
