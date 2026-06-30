# Exp49 ROSE Environment Smoke

Status: `ROSE_ENV_READY`

Generated: 2026-06-30T08:23:03.012636+08:00
Host: `dsw-753014-85f54df947-bkp7h`
Branch: `research/exp49-pai-rose-adapter-feasibility-20260629`
Commit: `e19b912af23e031b5dc97e411c037059e76dd007`

## Environment

- Env path: `/home/hj/venvs/rose_exp49_py312`
- Python: `3.12.3`
- Torch: `2.6.0+cu124 cuda=12.4 available=True`
- Install method: Python 3.12 venv via `virtualenv`; PyTorch/torchvision and ROSE requirements installed from public mirrors into isolated venv.

## Smoke

- Failed checks: `0`
- CSV: `reports/exp49_rose_env_smoke.csv`
- Summary JSON: `reports/exp49_rose_env_smoke_summary.json`

## Safety

Only dependency installation plus import/CUDA matmul smoke was run. No model inference, training, optimizer step, checkpoint update, VOR-Eval use, H20 action, or official ROSE source modification was performed.
