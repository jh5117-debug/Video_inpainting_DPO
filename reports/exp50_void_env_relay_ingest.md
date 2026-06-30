# Exp50 VOID Env Relay Ingest

Timestamp: 2026-06-30T16:44:02+08:00

Status: `VOID_ENV_READY`

## Relay Wheelhouse

- Machine: HAL / `hal-9000`
- Wheelhouse status: `VOID_ENV_WHEELHOUSE_READY`
- Files: 145
- Size: 3.3G
- Torch wheel: `torch-2.7.1+cu126-cp310-cp310-manylinux_2_28_x86_64.whl`
- Torchvision wheel: `torchvision-0.22.1+cu126-cp310-cp310-manylinux_2_28_x86_64.whl`
- Fonttools: relay package available; PAI import reports `4.63.0`
- Deepspeed: `DEEPSPEED_TRAIN_ONLY_NOT_INSTALLED_NO_DEPS_SDIST_AVAILABLE`; intentionally not installed because it is training-only and must not pull wrong torch/CUDA.

## Transfer Verification

- PAI target: `/mnt/nas/hj/H20_Video_inpainting_DPO/weights/void/env_wheelhouse`
- Status: `VOID_ENV_WHEELHOUSE_TRANSFER_VERIFIED`
- Source files: 145
- PAI files: 145
- Hash match: True
- Missing files: 0
- Mismatches: 0

## PAI Isolated Env

- Env: `/home/hj/conda_envs/void_exp50_official_v2`
- Python: `3.10.19 (main, Nov 28 2025, 15:21:53) [GCC 13.2.0]`
- Torch: `2.7.1+cu126`
- Torch CUDA runtime: `12.6`
- CUDA available: True
- BF16 supported: True
- Import failures: 0
- CUDA tiny smoke: `CUDA_BF16_BACKWARD_OK` on GPU `0`
- CUDA tiny max memory allocated: 67146240

## Safety

- Training run: no
- Inference run: no
- Optimizer step: no
- PAI base/system env modified: no
- VOID official source modified: no
- Torch 2.12 / CUDA 13 installed: no

## Decision

`VOID_ENV_READY` is restored from the HAL/H20 wheel relay. Exp50 can proceed to F0 component load smoke. Deepspeed remains a controlled training-only caveat for later G gates.
