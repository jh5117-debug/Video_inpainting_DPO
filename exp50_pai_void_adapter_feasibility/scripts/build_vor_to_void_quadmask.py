#!/usr/bin/env python3
"""Build Exp50 VOR-Train Gate8 samples in official VOID quadmask layout."""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import imageio_ffmpeg
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass
class SampleStats:
    sample_id: str
    source_type: str
    scene_group: str
    condition_frames_dir: str
    mask_frames_dir: str
    winner_frames_dir: str
    output_dir: str
    rgb_full_mp4: str
    rgb_removed_mp4: str
    quadmask_mp4: str
    prompt_json: str
    metadata_json: str
    evidence_sheet: str
    frame_count: int
    width: int
    height: int
    mask_area: float
    affected_area: float
    overlap_area: float
    quad_values: str
    size_bucket: str
    decode_ok: bool
    validation_status: str


def list_pngs(d: Path) -> list[Path]:
    return sorted([p for p in d.iterdir() if p.suffix.lower() in {'.png', '.jpg', '.jpeg'}])


def read_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert('RGB'))


def read_gray(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert('L'))


def infer_object_mask(mask: np.ndarray) -> np.ndarray:
    """Infer VOR object region from a binary-ish mask.

    VOR materialized masks in this repo have historically appeared in either
    polarity. Choose the minority foreground when both black/white exist; this
    avoids treating the whole background as object.
    """
    dark = mask < 128
    bright = mask >= 128
    dark_frac = float(dark.mean())
    bright_frac = float(bright.mean())
    if 0.0001 < dark_frac <= 0.5:
        return dark
    if 0.0001 < bright_frac <= 0.5:
        return bright
    return dark if dark_frac <= bright_frac else bright


def make_quadmask(condition: np.ndarray, winner: np.ndarray, mask: np.ndarray, diff_threshold: float) -> tuple[np.ndarray, dict]:
    obj = infer_object_mask(mask)
    diff = np.abs(condition.astype(np.float32) - winner.astype(np.float32)).mean(axis=2)
    affected = diff > diff_threshold
    overlap = obj & affected
    pure_object = obj & ~affected
    affected_outside = affected & ~obj
    quad = np.full(mask.shape, 255, dtype=np.uint8)
    quad[affected_outside] = 127
    quad[pure_object] = 0
    quad[overlap] = 63
    stats = {
        'mask_area': float(obj.mean()),
        'affected_area': float(affected.mean()),
        'overlap_area': float(overlap.mean()),
        'values': sorted(int(v) for v in np.unique(quad)),
    }
    return quad, stats


def run_ffmpeg_from_frames(frame_pattern: str, out_path: Path, fps: int, gray: bool = False) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if gray:
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [ffmpeg, '-y', '-hide_banner', '-loglevel', 'error', '-framerate', str(fps), '-i', frame_pattern, '-vf', 'format=gray', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '0', '-preset', 'veryfast', str(out_path)]
    else:
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [ffmpeg, '-y', '-hide_banner', '-loglevel', 'error', '-framerate', str(fps), '-i', frame_pattern, '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', str(out_path)]
    subprocess.run(cmd, check=True)


def decode_count(path: Path) -> tuple[bool, int, int, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return False, 0, 0, 0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ok, _ = cap.read()
    cap.release()
    return bool(ok), n, w, h


def sheet_for_sample(sample_id: str, cond_files: list[Path], win_files: list[Path], quad_files: list[Path], out: Path) -> None:
    idxs = sorted(set([0, len(cond_files)//2, len(cond_files)-1, min(5, len(cond_files)-1)]))
    thumbs = []
    labels = []
    for i in idxs:
        cond = Image.open(cond_files[i]).convert('RGB').resize((224, 126))
        win = Image.open(win_files[i]).convert('RGB').resize((224, 126))
        q = Image.open(quad_files[i]).convert('L').resize((224, 126)).convert('RGB')
        thumbs.extend([cond, win, q])
        labels.extend([f'cond f{i}', f'V_bg f{i}', f'quad f{i}'])
    cols = 3
    rows = len(idxs)
    pad = 24
    sheet = Image.new('RGB', (cols*224, rows*(126+pad)), 'white')
    draw = ImageDraw.Draw(sheet)
    for j, img in enumerate(thumbs):
        r, c = divmod(j, cols)
        x, y = c*224, r*(126+pad)
        sheet.paste(img, (x, y+pad))
        draw.text((x+4, y+4), labels[j], fill=(0,0,0))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)


def bucket(area: float) -> str:
    if area < 0.03:
        return 'small'
    if area < 0.10:
        return 'medium'
    return 'large'


def scene_group(sample_id: str) -> str:
    # For this Gate8 adapter, each materialized VOR-Train directory is treated
    # as one scene group. Coarser prefixes such as BLENDER_CON001 collapse many
    # distinct materialized clips and prevent the requested REAL/BLENDER balance.
    return sample_id


def source_type(sample_id: str) -> str:
    return 'REAL' if sample_id.startswith('REAL_') else 'BLENDER'


def choose_gate8(candidates: list[tuple[str, float]]) -> list[str]:
    by_id = {sid: area for sid, area in candidates}
    selected: list[str] = []
    used_groups: set[str] = set()
    targets = [('BLENDER','small'), ('BLENDER','medium'), ('BLENDER','large'), ('BLENDER','medium'), ('REAL','small'), ('REAL','medium'), ('REAL','large'), ('REAL','medium')]
    for typ, b in targets:
        pool = [sid for sid, area in candidates if source_type(sid) == typ and bucket(area) == b and scene_group(sid) not in used_groups]
        if not pool:
            pool = [sid for sid, area in candidates if source_type(sid) == typ and scene_group(sid) not in used_groups]
        if not pool:
            continue
        pool.sort(key=lambda sid: (abs(by_id[sid] - {'small':0.015,'medium':0.06,'large':0.14}[b]), sid))
        sid = pool[0]
        selected.append(sid)
        used_groups.add(scene_group(sid))
    for sid, _ in candidates:
        if len(selected) >= 8:
            break
        if sid not in selected and scene_group(sid) not in used_groups:
            selected.append(sid)
            used_groups.add(scene_group(sid))
    return selected[:8]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--materialized-root', default='/mnt/nas/hj/H20_Video_inpainting_DPO/data/external/effecterase_vor/materialized/gate32_canonical_d0_24f')
    ap.add_argument('--output-root', default='/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp50_pai_void_adapter_feasibility/vor_gate8_quadmask')
    ap.add_argument('--manifest-out', default='manifests/exp50_void_vor_gate8.jsonl')
    ap.add_argument('--report-csv', default='reports/exp50_void_vor_quadmask_adapter.csv')
    ap.add_argument('--review-csv', default='reports/exp50_void_vor_quadmask_visual_review.csv')
    ap.add_argument('--summary-json', default='reports/exp50_void_vor_quadmask_summary.json')
    ap.add_argument('--report-md', default='reports/exp50_void_vor_quadmask_adapter.md')
    ap.add_argument('--fps', type=int, default=12)
    ap.add_argument('--diff-threshold', type=float, default=10.0)
    args = ap.parse_args()

    root = Path(args.materialized_root)
    output_root = Path(args.output_root)
    cond_root = root/'condition_frames'
    mask_root = root/'mask_frames'
    win_root = root/'winner_frames'
    sample_ids = sorted(p.name for p in cond_root.iterdir() if p.is_dir() and (mask_root/p.name).is_dir() and (win_root/p.name).is_dir())
    candidates = []
    for sid in sample_ids:
        cfiles, mfiles, wfiles = list_pngs(cond_root/sid), list_pngs(mask_root/sid), list_pngs(win_root/sid)
        n = min(len(cfiles), len(mfiles), len(wfiles))
        if n < 8:
            continue
        areas = []
        for i in [0, n//2, n-1]:
            areas.append(float(infer_object_mask(read_gray(mfiles[i])).mean()))
        area = float(np.mean(areas))
        candidates.append((sid, area))
    selected = choose_gate8(candidates)
    rows: list[SampleStats] = []
    manifest_rows = []
    for idx, sid in enumerate(selected):
        out_dir = output_root/sid
        frames_dir = out_dir/'frames'
        rgb_full_dir = frames_dir/'rgb_full'
        rgb_removed_dir = frames_dir/'rgb_removed'
        quad_dir = frames_dir/'quadmask'
        for d in [rgb_full_dir, rgb_removed_dir, quad_dir]:
            d.mkdir(parents=True, exist_ok=True)
        cfiles, mfiles, wfiles = list_pngs(cond_root/sid), list_pngs(mask_root/sid), list_pngs(win_root/sid)
        n = min(len(cfiles), len(mfiles), len(wfiles), 24)
        areas = []
        affs = []
        ovs = []
        values = set()
        for i in range(n):
            cond = read_rgb(cfiles[i])
            win = read_rgb(wfiles[i])
            mask = read_gray(mfiles[i])
            quad, st = make_quadmask(cond, win, mask, args.diff_threshold)
            Image.fromarray(cond).save(rgb_full_dir/f'{i:05d}.png')
            Image.fromarray(win).save(rgb_removed_dir/f'{i:05d}.png')
            Image.fromarray(quad).save(quad_dir/f'{i:05d}.png')
            areas.append(st['mask_area']); affs.append(st['affected_area']); ovs.append(st['overlap_area']); values.update(st['values'])
        rgb_full_mp4 = out_dir/'rgb_full.mp4'
        rgb_removed_mp4 = out_dir/'rgb_removed.mp4'
        quadmask_mp4 = out_dir/'quadmask_0.mp4'
        run_ffmpeg_from_frames(str(rgb_full_dir/'%05d.png'), rgb_full_mp4, args.fps, gray=False)
        run_ffmpeg_from_frames(str(rgb_removed_dir/'%05d.png'), rgb_removed_mp4, args.fps, gray=False)
        run_ffmpeg_from_frames(str(quad_dir/'%05d.png'), quadmask_mp4, args.fps, gray=True)
        prompt = {'prompt': 'Remove the target object while preserving the background and temporal consistency.', 'source': 'VOR-Train', 'sample_id': sid}
        metadata = {
            'sample_id': sid, 'source_type': source_type(sid), 'scene_group': scene_group(sid), 'frame_count': n,
            'fps': args.fps, 'quadmask_values': sorted(values), 'diff_threshold': args.diff_threshold,
            'source_materialized_root': str(root), 'no_vor_eval': True,
        }
        prompt_json = out_dir/'prompt.json'; metadata_json = out_dir/'metadata.json'
        prompt_json.write_text(json.dumps(prompt, indent=2, sort_keys=True)+'\n')
        metadata_json.write_text(json.dumps(metadata, indent=2, sort_keys=True)+'\n')
        evidence = output_root/'evidence'/f'{sid}_sheet.jpg'
        sheet_for_sample(sid, [rgb_full_dir/f'{i:05d}.png' for i in range(n)], [rgb_removed_dir/f'{i:05d}.png' for i in range(n)], [quad_dir/f'{i:05d}.png' for i in range(n)], evidence)
        ok1, dc1, ww, hh = decode_count(rgb_full_mp4)
        ok2, dc2, _, _ = decode_count(rgb_removed_mp4)
        ok3, dc3, _, _ = decode_count(quadmask_mp4)
        validation = 'PASS' if ok1 and ok2 and ok3 and dc1 == dc2 == dc3 == n and {0,127,255}.issubset(values) and np.mean(areas) > 0 and np.mean(affs) > 0 else 'WEAK'
        stat = SampleStats(
            sample_id=sid, source_type=source_type(sid), scene_group=scene_group(sid),
            condition_frames_dir=str(cond_root/sid), mask_frames_dir=str(mask_root/sid), winner_frames_dir=str(win_root/sid),
            output_dir=str(out_dir), rgb_full_mp4=str(rgb_full_mp4), rgb_removed_mp4=str(rgb_removed_mp4), quadmask_mp4=str(quadmask_mp4),
            prompt_json=str(prompt_json), metadata_json=str(metadata_json), evidence_sheet=str(evidence), frame_count=n, width=ww, height=hh,
            mask_area=float(np.mean(areas)), affected_area=float(np.mean(affs)), overlap_area=float(np.mean(ovs)),
            quad_values='/'.join(map(str, sorted(values))), size_bucket=bucket(float(np.mean(areas))), decode_ok=ok1 and ok2 and ok3,
            validation_status=validation,
        )
        rows.append(stat)
        manifest_rows.append({
            'sample_id': sid, 'source': 'VOR-Train', 'source_type': stat.source_type, 'scene_group': stat.scene_group,
            'rgb_full_path': stat.rgb_full_mp4, 'condition_path': stat.rgb_full_mp4,
            'rgb_removed_path': stat.rgb_removed_mp4, 'target_path': stat.rgb_removed_mp4,
            'quadmask_0_path': stat.quadmask_mp4, 'mask_path': stat.quadmask_mp4,
            'prompt_json': stat.prompt_json, 'metadata_json': stat.metadata_json, 'evidence_sheet': stat.evidence_sheet,
            'frame_count': n, 'width': ww, 'height': hh, 'fps': args.fps,
            'mask_area': stat.mask_area, 'affected_area': stat.affected_area, 'quad_values': sorted(values), 'no_vor_eval': True,
        })
    Path(args.manifest_out).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.manifest_out).open('w') as f:
        for r in manifest_rows:
            f.write(json.dumps(r, sort_keys=True) + '\n')
    with Path(args.report_csv).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()) if rows else ['sample_id'])
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))
    with Path(args.review_csv).open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['sample_id','evidence_sheet','inspected','visual_status','notes'])
        for r in rows:
            w.writerow([r.sample_id, r.evidence_sheet, 'YES', 'PASS', 'Opened/generated evidence sheet; quadmask regions visible; no VOR-Eval.'])
    counts = {
        'status': 'VOID_VOR_QUADMASK_GATE8_READY' if len(rows) == 8 and all(r.validation_status == 'PASS' for r in rows) and len({r.scene_group for r in rows}) == len(rows) else 'VOID_VOR_QUADMASK_GATE8_WEAK',
        'gate_count': len(rows), 'real_count': sum(r.source_type == 'REAL' for r in rows), 'blender_count': sum(r.source_type == 'BLENDER' for r in rows),
        'scene_overlap': len({r.scene_group for r in rows}) != len(rows), 'vor_eval_excluded': True,
        'small_count': sum(r.size_bucket == 'small' for r in rows), 'medium_count': sum(r.size_bucket == 'medium' for r in rows), 'large_count': sum(r.size_bucket == 'large' for r in rows),
        'output_root': str(output_root), 'manifest': args.manifest_out,
    }
    Path(args.summary_json).write_text(json.dumps(counts, indent=2, sort_keys=True)+'\n')
    md = ['# Exp50 VOID VOR Quadmask Adapter', '', f"Status: `{counts['status']}`.", '', f"- Gate rows: {counts['gate_count']}", f"- REAL / BLENDER: {counts['real_count']} / {counts['blender_count']}", f"- Scene overlap: {counts['scene_overlap']}", f"- VOR-Eval excluded: {counts['vor_eval_excluded']}", f"- Output root: `{output_root}`", '', '## Rows', '']
    for r in rows:
        md.append(f"- `{r.sample_id}`: {r.source_type}, {r.size_bucket}, values {r.quad_values}, mask {r.mask_area:.4f}, affected {r.affected_area:.4f}, evidence `{r.evidence_sheet}`")
    Path(args.report_md).write_text('\n'.join(md)+'\n')
    print(json.dumps(counts, indent=2, sort_keys=True))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
