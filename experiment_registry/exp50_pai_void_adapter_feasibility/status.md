# Exp50 PAI VOID Adapter Feasibility Status

Last updated: 2026-06-30T16:50:03+08:00

Current status: `VOID_OFFICIAL_SAMPLE_INFERENCE_PASS`

- Permission recovery: `VOID_ASSET_PERMISSION_RECOVERED`
- Official repo: `VOID_REPO_READY`
- HF relay ingest: `VOID_WEIGHTS_READY`
- Environment/import smoke: `VOID_ENV_READY`
- Trainable-forward audit: `VOID_TRAINING_FORWARD_HEAVY_BUT_POSSIBLE`
- VOR-to-VOID quadmask Gate8: `VOID_VOR_QUADMASK_GATE8_READY`
- Training: not run
- Inference: not run
- VOID positive claim: not made

Official inference smoke has not run yet; F0 component load smoke is now unblocked by `VOID_ENV_READY`.

## Environment repair C2


- Environment repair C2: `VOID_ENV_PARTIAL`; exact blockers: `VOID_ENV_BLOCKED_TORCH, VOID_ENV_BLOCKED_DEEPSPEED`.
- F0/F1/F2/G gates: not run because `VOID_ENV_READY` was not reached.

## Environment relay ingest C3

- Environment relay ingest C3: `VOID_ENV_READY`.
- Wheelhouse transfer: `VOID_ENV_WHEELHOUSE_TRANSFER_VERIFIED`; hash match True; missing 0; mismatch 0.
- Env: `/home/hj/conda_envs/void_exp50_official_v2`; torch `2.7.1+cu126`; CUDA runtime `12.6`.
- CUDA tiny smoke: `CUDA_BF16_BACKWARD_OK` on GPU `0`.
- F0 component load smoke: next.

- Component load smoke: `VOID_COMPONENT_LOAD_PASS`

## Component load smoke F0

- Component load smoke F0: `VOID_COMPONENT_LOAD_PASS`.
- Pass1/pass2 checkpoint headers and base model component headers loaded.
- Full GPU model load: not attempted; no inference/training.

- Official sample inference: `VOID_OFFICIAL_SAMPLE_INFERENCE_PASS`

## Official sample inference F1

- Official sample inference F1: `VOID_OFFICIAL_SAMPLE_INFERENCE_PASS`.
- Sample: `lime`; raw frames 85; return code `0`.
- Bundled ffmpeg shim used under Exp50 runtime; system env and official source unchanged.
