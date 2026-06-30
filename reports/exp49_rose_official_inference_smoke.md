# Exp49 ROSE Official Inference Smoke

Status: `ROSE_INFERENCE_SMOKE_PASS`

Generated: 2026-06-30T08:45:38.057346+08:00
Host: `dsw-753014-85f54df947-bkp7h`
Branch: `research/exp49-pai-rose-adapter-feasibility-20260629`
Commit: `1420a17bf67b3d6377470cb57e139b7a167e4792`

## E1 Official Demo

- Output: `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp49_pai_rose_adapter_feasibility/official_demo_smoke_default_20260630_083542/example-1.mp4`
- Decode ok: `True`
- Frames: `17`
- Resolution: `720x480`
- Log: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp49_pai_rose_adapter_feasibility/milestone_e_official_demo_default_20260630_083542.log`
- Reduced 256x384 probe failed with a transformer `seq_len` assertion because official `inference.py` does not propagate reduced `height/width`; the default 480x720 smoke passed without modifying official source.

## E2 VOR-Train Smoke6

- Output dir: `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp49_pai_rose_adapter_feasibility/vor_or_smoke6_20260630_083835`
- Technical valid: `6/6`
- Metrics: `reports/exp49_rose_vor_or_smoke_metrics.csv`
- Visual review: `reports/exp49_rose_vor_or_smoke_visual_review.csv`
- Evidence dir: `/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp49_rose_adapter/reports/exp49_rose_vor_or_smoke_evidence`
- Codex visual inspection: `ROSE_OUTPUT_USABLE=2`, `MEDIUM_HARD_ELIGIBLE=2`, `SIDE_EFFECT_LEFT=2`.

## Safety

No training, optimizer step, checkpoint update, VOR-Eval use, H20 action, hard comp, shared trainer change, metrics code change, or official ROSE source modification was performed.
