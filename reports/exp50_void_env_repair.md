# Exp50 VOID Environment Repair

Status: `VOID_ENV_PARTIAL`.

C2 attempted to repair the VOID environment in an isolated venv without modifying the PAI base env or the official VOID repo source. The old `void_exp50` venv remains available and import-capable, but it is not an official-pinned environment. The new `void_exp50_official` venv could not be made ready on PAI during this run.

## Paths

- Old partial env: `/home/hj/conda_envs/void_exp50`
- New repair env: `/home/hj/conda_envs/void_exp50_official`
- Official requirements: `/mnt/nas/hj/H20_Video_inpainting_DPO/third_party/VOID/Netflix_void-model/requirements.txt`
- Official repo: `/mnt/nas/hj/H20_Video_inpainting_DPO/third_party/VOID/Netflix_void-model`

## What Was Tried

1. Full `requirements.txt` install in the isolated repair env.
2. Fallback install without `deepspeed`.
3. Direct `fonttools` probe.
4. Targeted official core pins: `torch==2.7.1`, `torchvision==0.22.1`, `numpy==1.26.4`, `diffusers==0.33.1`, `transformers==4.57.1`, `accelerate==1.12.0`, `safetensors==0.6.2`, `peft==0.17.1`.
5. Separate `deepspeed==0.17.6` train-only attempt.

## Exact Blockers

- `VOID_ENV_BLOCKED_TORCH`: PAI could resolve `torch==2.7.1`, but the 821MB wheel repeatedly timed out; only a few MB were observed after several minutes, so the install was stopped rather than leaving an uncontrolled long-running pip process.
- `VOID_ENV_BLOCKED_DEEPSPEED`: full requirements entered a slow deepspeed build/metadata path; separate deepspeed install attempted to pull `torch 2.12` and CUDA 13 packages, so it was stopped to avoid off-protocol environment drift.

## Secondary Blocker

- Full non-deepspeed requirements also hit `matplotlib -> fonttools>=4.22.0`, but the configured/direct index did not provide a usable `fonttools` wheel in this PAI session. This is not expected to block inference by itself, but it blocks a literal full requirements install.

## Gate Decision

`VOID_ENV_READY` was **not** reached. Therefore Milestone F0 component load, F1 official sample inference, F2 VOR Gate8 inference, and G zero-gap/one-step were not run.

## Safety

- Base env modified: no
- VOID official repo source modified: no
- Full 5B model loaded: no
- Inference run: no
- Training run: no
- Optimizer step: no
