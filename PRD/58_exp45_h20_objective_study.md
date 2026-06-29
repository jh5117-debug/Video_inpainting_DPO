# Exp45 H20 Objective Study

Status: `EXP45_READBACK_COMPLETE_ASSETS_PENDING_ENV_PENDING_GPU_BLOCKED_EXTERNAL`

Created: `2026-06-29T16:01:55,415804093+08:00` on `instance-afs92r3e`.
Branch: `research/exp45-h20-diffueraser-videopainter-objective-study-20260629`.
Base: `origin/research/exp27-paper-grounded-preference-study` at `17b99a421fbf1bb79a446713a1f30ef6c8ecc769`.
Fallback used: none.

## Goal

Run an isolated H20 objective study for DiffuEraser first, then a selected VideoPainter transfer only after DiffuEraser identifies a top method. The main method remains LoVI-DPO. Linear-DPO, Diffusion-SDPO, and LocalDPO-style variants are external baselines, ablations, or integration studies.

## Hard Boundaries

- PAI is read-only for inventory and minimal file transfer only.
- No PAI GPU, PAI training, PAI signal, or PAI output/checkpoint overwrite.
- Do not modify `inference/metrics.py`, shared trainer code, MiniMax official source, or Exp1-Exp44 historical results.
- Do not use VOR-Eval for training, selection, or tuning.
- Do not write `UNIVERSAL_ADAPTER`, `FINAL_SOTA`, or `TOP_CONFERENCE_NOVELTY_CONFIRMED`.

## H20 Paths

- Worktree: `/home/nvme01/H20_Video_inpainting_DPO_exp45_objective_study`
- Output root: `/home/nvme01/H20_Video_inpainting_DPO/experiments/dpo/exp45_h20_objective_study`
- Log root: `/home/nvme01/H20_Video_inpainting_DPO/logs/autoresearch/exp45_h20_objective_study`
- Runtime root: `/home/nvme01/H20_Video_inpainting_DPO/runtime/exp45_h20_objective_study`
- Mirror root: `/home/nvme01/H20_Video_inpainting_DPO/data/h20_mirror/exp45_objective_study`

## Current Gate State

- Readback: complete.
- Asset inventory: pending.
- Minimal transfer: pending.
- Environment smoke: pending.
- DiffuEraser O0-O6 objective study: pending.
- VideoPainter selected transfer: pending.
- GPU state: blocked for training right now because all H20 GPUs are occupied by external root `experiments/libero/eval_libero_single.py` jobs. These are not Video_inpainting_DPO jobs and were not touched.

## Objective Set

- O0 Vanilla Diffusion-DPO.
- O1 Current LoVI-DPO anchor.
- O2 Exact Diffusion-SDPO.
- O3 LoVI + SDPO safeguard.
- O4 Linear-DPO Frozen.
- O5 Linear-DPO EMA.
- O6 LocalDPO-style 24F adaptation.

## Promotion Rule

No positive or best status may be written from metrics alone. Any positive, improves, best, or paper-ready conclusion requires actual raw videos, per-video metrics, temporal strips, mask/boundary/outside crops, visual review CSV, and explicit review of the evidence pages.
