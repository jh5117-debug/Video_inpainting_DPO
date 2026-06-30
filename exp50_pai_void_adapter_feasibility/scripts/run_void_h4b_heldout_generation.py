#!/usr/bin/env python
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import cv2
import numpy as np
import torch
from safetensors.torch import load_file, save_file

ROOT = Path(os.environ.get('EXP50_ROOT', '/home/hj/H20_Video_inpainting_DPO_exp50_void_adapter_feasibility'))
ENV_PY = Path(os.environ.get('EXP50_ENV_PY', '/home/hj/conda_envs/void_exp50_official_v2/bin/python'))
ASSET_ROOT = Path(os.environ.get('EXP50_ASSET_ROOT', '/mnt/nas/hj/H20_Video_inpainting_DPO'))
VOID_REPO = Path(os.environ.get('EXP50_VOID_REPO', str(ASSET_ROOT / 'third_party/VOID/Netflix_void-model')))
BASE = Path(os.environ.get('EXP50_BASE_MODEL', str(ASSET_ROOT / 'weights/void/CogVideoX-Fun-V1.5-5b-InP')))
VOID_WEIGHTS = Path(os.environ.get('EXP50_VOID_WEIGHTS', str(ASSET_ROOT / 'weights/void/netflix_void-model')))
STEP0_CKPT = VOID_WEIGHTS / 'void_pass1.safetensors'
ADAPTER = Path(os.environ.get('EXP50_ADAPTER', str(ASSET_ROOT / 'experiments/dpo/exp50_pai_void_adapter_feasibility/one_step_v2/adapter_proj_out_step1.pt')))
MANIFEST = Path(os.environ.get('EXP50_HELDOUT_MANIFEST', str(ROOT / 'manifests/exp50_void_adapter_heldout4.jsonl')))
DATA_VIEW = Path(os.environ.get('EXP50_DATA_VIEW', str(ASSET_ROOT / 'experiments/dpo/exp50_pai_void_adapter_feasibility/vor_gate8_quadmask_official_prompt_view')))
F2_STEP0 = Path(os.environ.get('EXP50_F2_STEP0', str(ASSET_ROOT / 'experiments/dpo/exp50_pai_void_adapter_feasibility/f2_vor_gate8_pass1')))
OUT = Path(os.environ.get('EXP50_H4B_OUT', str(ASSET_ROOT / 'logs/autoresearch/exp50_pai_void_adapter_feasibility/one_step_heldout_evidence_v2')))
REPORTS = ROOT / 'reports'
RUNTIME = Path(os.environ.get('EXP50_RUNTIME', str(ASSET_ROOT / 'runtime/exp50_pai_void_adapter_feasibility')))
FFMPEG_DIR = RUNTIME / 'ffmpeg_bin'
SH_TZ = timezone(timedelta(hours=8))


def now() -> str:
    return datetime.now(SH_TZ).replace(microsecond=0).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_manifest() -> list[dict]:
    return [json.loads(line) for line in MANIFEST.read_text().splitlines() if line.strip()]


def gpu_snapshot() -> str:
    return subprocess.check_output(['bash', '-lc', 'nvidia-smi --query-gpu=index,memory.used,memory.free,memory.total,utilization.gpu --format=csv'], text=True)


def make_step1_checkpoint() -> Path:
    ckpt_dir = OUT / 'checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    step1 = ckpt_dir / 'void_pass1_step1_proj_out.safetensors'
    meta = ckpt_dir / 'void_pass1_step1_proj_out.json'
    adapter_sha = sha256(ADAPTER)
    if step1.exists() and meta.exists():
        try:
            m = json.loads(meta.read_text())
            if m.get('adapter_sha256') == adapter_sha:
                return step1
        except Exception:
            pass
    base_state = load_file(str(STEP0_CKPT), device='cpu')
    adapter_obj = torch.load(ADAPTER, map_location='cpu')
    adapter_state = adapter_obj['adapter_state']
    replaced = []
    for k, v in adapter_state.items():
        if k not in base_state:
            raise KeyError(f'adapter key not in base checkpoint: {k}')
        base_state[k] = v.detach().cpu().to(dtype=base_state[k].dtype).contiguous()
        replaced.append(k)
    save_file(base_state, str(step1), metadata={'source': 'exp50_h4b_one_step_adapter_merge'})
    meta.write_text(json.dumps({
        'created': now(),
        'step0_checkpoint': str(STEP0_CKPT),
        'step0_sha256': sha256(STEP0_CKPT),
        'adapter_checkpoint': str(ADAPTER),
        'adapter_sha256': adapter_sha,
        'step1_checkpoint': str(step1),
        'step1_sha256': sha256(step1),
        'replaced_keys': replaced,
        'hard_comp_used': False,
    }, indent=2, sort_keys=True) + '\n')
    return step1


def run_group(gpu: int, seqs: list[str], step1: Path) -> dict:
    save_path = OUT / f'step1_gpu{gpu}'
    save_path.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(ENV_PY), 'inference/cogvideox_fun/predict_v2v.py',
        '--config', 'config/quadmask_cogvideox.py',
        f'--config.data.data_rootdir={DATA_VIEW}',
        f'--config.experiment.run_seqs={",".join(seqs)}',
        f'--config.experiment.save_path={save_path}',
        f'--config.video_model.model_name={BASE}',
        f'--config.video_model.transformer_path={step1}',
    ]
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    env['PYTHONPATH'] = str(VOID_REPO) + ':' + env.get('PYTHONPATH', '')
    env['PATH'] = str(FFMPEG_DIR) + ':' + env.get('PATH', '')
    log = OUT / f'gpu{gpu}_runtime_log.txt'
    start = now(); t0 = time.time()
    with log.open('w') as f:
        f.write('start=' + start + '\n')
        f.write('gpu=' + str(gpu) + '\n')
        f.write('seqs=' + ','.join(seqs) + '\n')
        f.write('cmd=' + ' '.join(cmd) + '\n')
        f.write('gpu_before:\n' + gpu_snapshot() + '\n')
        f.flush()
        proc = subprocess.run(cmd, cwd=str(VOID_REPO), env=env, stdout=f, stderr=subprocess.STDOUT, text=True)
        f.write('end=' + now() + '\n')
        f.write('returncode=' + str(proc.returncode) + '\n')
        f.write('runtime_sec=' + str(time.time() - t0) + '\n')
        f.write('gpu_after:\n' + gpu_snapshot() + '\n')
    return {'gpu': gpu, 'seqs': seqs, 'returncode': proc.returncode, 'log': str(log), 'save_path': str(save_path), 'runtime_sec': time.time() - t0}


def find_step1_output(sample_id: str, gpu_ids: list[int]) -> Path | None:
    candidates: list[Path] = []
    search_dirs = [OUT / f'step1_gpu{gpu}' for gpu in gpu_ids]
    search_dirs.extend(sorted(OUT.glob('step1_gpu*')))
    seen: set[Path] = set()
    for save_dir in search_dirs:
        if save_dir in seen:
            continue
        seen.add(save_dir)
        candidates.extend(save_dir.glob(f'{sample_id}-fg=-1-*.mp4'))
    candidates = [p for p in candidates if p.exists() and not p.name.endswith('_tuple.mp4')]
    return sorted(candidates)[0] if candidates else None


def decode_frames(path: Path, max_frames: int | None = None) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    return frames


def write_video(path: Path, frames: list[np.ndarray], fps: float = 12.0) -> None:
    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for fr in frames:
        writer.write(fr)
    writer.release()


def resize_like(fr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    return cv2.resize(fr, (w, h), interpolation=cv2.INTER_NEAREST if fr.ndim == 2 else cv2.INTER_LINEAR)


def quad_vis(fr: np.ndarray) -> np.ndarray:
    gray = fr[:, :, 0] if fr.ndim == 3 else fr
    out = np.zeros((gray.shape[0], gray.shape[1], 3), np.uint8)
    out[gray <= 31] = (0, 0, 255)
    out[(gray > 31) & (gray <= 95)] = (0, 255, 255)
    out[(gray > 95) & (gray <= 191)] = (0, 255, 0)
    out[gray > 191] = (160, 160, 160)
    return out


def label(fr: np.ndarray, text: str) -> np.ndarray:
    out = fr.copy()
    cv2.rectangle(out, (0, 0), (min(out.shape[1], 240), 28), (0, 0, 0), -1)
    cv2.putText(out, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
    return out


def make_evidence(row: dict, step0: Path, step1: Path, sample_dir: Path) -> dict:
    condition = Path(row['condition_path']); winner = Path(row['winner_path']); quad = Path(row['quadmask_0_path'])
    for name, src in [('step0_raw.mp4', step0), ('step1_raw.mp4', step1), ('condition.mp4', condition), ('winner.mp4', winner), ('quadmask.mp4', quad)]:
        dst = sample_dir / name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src)
    f0 = decode_frames(step0)
    f1 = decode_frames(step1)
    fc = decode_frames(condition, len(f1))
    fw = decode_frames(winner, len(f1))
    fq = decode_frames(quad, len(f1))
    n = min(len(f0), len(f1), len(fc), len(fw), len(fq))
    shape = f1[0].shape[:2]
    side = []
    for i in range(n):
        c = resize_like(fc[i], shape)
        w = resize_like(fw[i], shape)
        q = resize_like(quad_vis(fq[i]), shape)
        rowimg = np.concatenate([label(c, 'condition'), label(q, 'quadmask'), label(f0[i], 'step0'), label(f1[i], 'step1'), label(w, 'winner')], axis=1)
        side.append(rowimg)
    write_video(sample_dir / 'side_by_side.mp4', side, 12.0)
    idxs = np.linspace(0, n-1, min(16, n)).astype(int).tolist()
    strips = []
    for title, frames in [('step0', f0), ('step1', f1), ('winner', [resize_like(x, shape) for x in fw])]:
        imgs = [label(resize_like(frames[i], shape), title) for i in idxs]
        strips.append(np.concatenate(imgs, axis=1))
    cv2.imwrite(str(sample_dir / 'temporal_strip_16f.jpg'), np.concatenate(strips, axis=0))
    qimgs = [label(resize_like(quad_vis(fq[i]), shape), f'quad {i}') for i in [0, n//2, n-1]]
    cv2.imwrite(str(sample_dir / 'quadmask_visualization.jpg'), np.concatenate(qimgs, axis=1))
    # crop sheets around object/affected bbox.
    qgray = resize_like(fq[n//2][:,:,0], shape)
    mask = qgray <= 191
    ys, xs = np.where(mask)
    if len(xs):
        x0,x1=max(0,xs.min()-32),min(shape[1],xs.max()+33); y0,y1=max(0,ys.min()-32),min(shape[0],ys.max()+33)
    else:
        x0,y0,x1,y1=0,0,shape[1],shape[0]
    crop_rows=[]
    for title, frames in [('condition',[resize_like(x, shape) for x in fc]),('step0',f0),('step1',f1),('winner',[resize_like(x, shape) for x in fw])]:
        imgs=[label(frames[i][y0:y1,x0:x1], title) for i in [0,n//2,n-1]]
        crop_rows.append(np.concatenate(imgs, axis=1))
    crop=np.concatenate(crop_rows, axis=0)
    cv2.imwrite(str(sample_dir / 'object_crop_sheet.jpg'), crop)
    cv2.imwrite(str(sample_dir / 'affected_crop_sheet.jpg'), crop)
    # Outside crop: top-left quarter, fallback simple.
    outside_rows=[]
    ox1, oy1 = shape[1]//2, shape[0]//2
    for title, frames in [('condition',[resize_like(x, shape) for x in fc]),('step0',f0),('step1',f1),('winner',[resize_like(x, shape) for x in fw])]:
        imgs=[label(frames[i][0:oy1,0:ox1], title) for i in [0,n//2,n-1]]
        outside_rows.append(np.concatenate(imgs, axis=1))
    cv2.imwrite(str(sample_dir / 'outside_crop_sheet.jpg'), np.concatenate(outside_rows, axis=0))
    diffs=[cv2.applyColorMap(cv2.convertScaleAbs(cv2.absdiff(f0[i], f1[i]), alpha=4), cv2.COLORMAP_INFERNO) for i in idxs]
    cv2.imwrite(str(sample_dir / 'temporal_diff_heatmap.jpg'), np.concatenate(diffs, axis=1))
    return {'frame_count': n, 'resolution': f'{shape[1]}x{shape[0]}', 'decode_status': 'ok' if n else 'failed'}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'evidence').mkdir(exist_ok=True)
    rows = read_manifest()
    heldout = [r['sample_id'] for r in rows]
    start = now()
    step1_ckpt = make_step1_checkpoint()
    gpu_ids = [int(x) for x in os.environ.get('EXP50_H4B_GPUS', '0,1').split(',') if x.strip()]
    if not gpu_ids:
        raise ValueError('EXP50_H4B_GPUS must contain at least one GPU id')
    groups = [(gpu, heldout[i::len(gpu_ids)]) for i, gpu in enumerate(gpu_ids)]
    skip_inference = os.environ.get('EXP50_H4B_SKIP_INFERENCE', '0') == '1'
    if skip_inference:
        runs = [
            {
                'gpu': gpu,
                'seqs': seqs,
                'returncode': 0,
                'log': str(OUT / f'gpu{gpu}_runtime_log.txt'),
                'save_path': str(OUT / f'step1_gpu{gpu}'),
                'runtime_sec': 0.0,
                'skipped_existing_outputs': True,
            }
            for gpu, seqs in groups
        ]
    else:
        runs = [run_group(gpu, seqs, step1_ckpt) for gpu, seqs in groups]
    ok = all(r['returncode'] == 0 for r in runs)
    records = []
    for row in rows:
        sid = row['sample_id']
        sample_dir = OUT / 'evidence' / sid
        sample_dir.mkdir(parents=True, exist_ok=True)
        step0 = F2_STEP0 / f'{sid}-fg=-1-0001.mp4'
        step1 = find_step1_output(sid, gpu_ids)
        rec = {'sample_id': sid, 'step0_raw': str(step0), 'step1_raw': str(step1) if step1 else '', 'status': 'missing_step1'}
        if step0.exists() and step1 and step1.exists():
            ev = make_evidence(row, step0, step1, sample_dir)
            rec.update(ev); rec['status'] = 'generated'; rec['evidence_dir'] = str(sample_dir)
        records.append(rec)
    status = 'VOID_ONE_STEP_HELDOUT_GENERATION_READY' if ok and all(r['status']=='generated' for r in records) else 'VOID_ONE_STEP_HELDOUT_GENERATION_BLOCKED'
    summary = {
        'status': status,
        'start': start,
        'end': now(),
        'gpus_requested': gpu_ids,
        'skip_inference': skip_inference,
        'root_processes_killed': [],
        'step1_checkpoint': str(step1_ckpt),
        'step1_checkpoint_sha256': sha256(step1_ckpt),
        'adapter_checkpoint': str(ADAPTER),
        'adapter_sha256': sha256(ADAPTER),
        'output_root': str(OUT),
        'runs': runs,
        'records': records,
        'step0_outputs': sum(1 for r in records if Path(r['step0_raw']).exists()),
        'step1_outputs': sum(1 for r in records if r.get('step1_raw') and Path(r['step1_raw']).exists()),
        'videos_generated': all(r['status']=='generated' for r in records),
        'vor_eval_used': False,
        'hard_comp_used': False,
        'training_run': False,
        'optimizer_step': False,
        'gpu_after': gpu_snapshot().strip().splitlines(),
    }
    (REPORTS / 'exp50_void_one_step_heldout_generation_summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    with (REPORTS / 'exp50_void_one_step_heldout_generation.csv').open('w', newline='') as f:
        fields=['sample_id','status','step0_raw','step1_raw','frame_count','resolution','decode_status','evidence_dir']
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader();
        for r in records: w.writerow({k:r.get(k,'') for k in fields})
    md = ['# Exp50 VOID One-Step Heldout Generation', '', f'Time: {now()}', '', f'Status: `{status}`', '', '## Protocol', '', '- Used existing F2 Step0 official pass1 outputs.', '- Created Step1 checkpoint by replacing only `proj_out.weight` and `proj_out.bias` from the one-step adapter into a temporary pass1 safetensors checkpoint.', '- Ran official `inference/cogvideox_fun/predict_v2v.py` on heldout4, split over GPU0/GPU1.', '- No VOR-Eval, no hard comp, no training, no optimizer step.', '', '## Runs', '']
    for r in runs:
        md.append(f"- GPU{r['gpu']}: seqs={r['seqs']} returncode={r['returncode']} log=`{r['log']}`")
    md += ['', '## Outputs', '']
    for r in records:
        md.append(f"- {r['sample_id']}: {r['status']} frames={r.get('frame_count','')} resolution={r.get('resolution','')} evidence=`{r.get('evidence_dir','')}`")
    md += ['', '## Safety', '', 'No root process was killed. Existing root GPU processes were left untouched; generation used available free memory on GPU0/GPU1.']
    (REPORTS / 'exp50_void_one_step_heldout_generation.md').write_text('\n'.join(md) + '\n')
    print(status)
    if status != 'VOID_ONE_STEP_HELDOUT_GENERATION_READY':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
