# Exp49 ROSE Environment Smoke

Status: `ROSE_ENV_PARTIAL`

Generated: 2026-06-30T07:35:30,817270147+08:00
Host: dsw-753014-85f54df947-bkp7h
Branch: research/exp49-pai-rose-adapter-feasibility-20260629
Commit: ccda728a1a9f6cd66f1b555f7cf62460d0814281

## Environment

- Env path: `/home/hj/venvs/rose_exp49_py310`
- Python used: `Python 3.10.19`
- Official requested Python: ROSE README says Python 3.12.
- PAI Python 3.12: `Python 3.12.3`
- Python 3.12 pip/torch: unavailable in this image.
- Torch/CUDA: `2.6.0+cu126 12.6 True`

## Install

ROSE core requirements were installed into the isolated venv. Base/global Python was not modified.

- pip install rc: `0`
- smoke rc: `0`

## Smoke Checks

- CSV: `reports/exp49_rose_env_smoke.csv`
- Summary JSON: `reports/exp49_rose_env_smoke_summary.json`
- Import log: `/mnt/nas/hj/H20_Video_inpainting_DPO/runtime/exp49_pai_rose_adapter_feasibility/exp49_rose_env_import_smoke.log`

## GPU

A tiny CUDA matmul smoke was run on GPU0 only. No model inference, training, optimizer step, checkpoint update, or H20 action was performed.
