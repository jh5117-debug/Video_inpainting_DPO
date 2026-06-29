# Exp49 PAI ROSE Adapter Feasibility

Date: 2026-06-29

Branch: `research/exp49-pai-rose-adapter-feasibility-20260629`

Status: `EXP49_PAI_ACCESS_BLOCKED_BEFORE_DOWNLOAD`

## Objective

Evaluate ROSE as a possible third video-inpainting adapter candidate:

- audit public ROSE code/demo/dataset/model assets;
- prepare PAI asset download plan;
- run ROSE VOR-OR inference baseline on PAI only if assets and environment are ready;
- audit whether ROSE exposes a true trainable forward;
- only if the forward gate passes, run zero-gap, one-step, strict reload, and then at most a 10-step micro gate.

## Safety Boundaries

- Do not modify `inference/metrics.py`.
- Do not modify shared trainer code.
- Do not modify official ROSE source.
- Do not touch MiniMax Exp42-Exp48 outputs.
- Do not use VOR-Eval for training, filtering, or threshold design.
- Do not write ROSE adapter positive unless quantitative, qualitative, and heldout micro gates really pass.
- Do not run 50/100/300/500/1000/2000-step training.

## Current Environment Finding

The current Codex shell is on `hal-9000`, not PAI.

Observed:

```text
hostname = hal-9000
/mnt/nas/hj/H20_Video_inpainting_DPO = missing
/mnt/workspace/hj/nas_hj = missing
```

PAI endpoint probing did not yield a verified login:

- `dsw-753014-85f54df947-bkp7h` does not resolve from HAL.
- Historical host `47.103.26.60` has TCP/22 open, but candidate public-key SSH commands timed out before returning `hostname`.
- No command returned a verified `dsw-*` host with `/mnt/nas` and `/mnt/workspace` mounted.

Therefore no PAI download, inference, GPU smoke, or training-forward work was started in this milestone.

## Public Asset Readback

ROSE public assets exist:

- Project page: `https://rose2025-inpaint.github.io/`
- Code: `https://github.com/Kunbyte-AI/ROSE`
- HF model: `https://huggingface.co/Kunbyte/ROSE`
- HF demo space: `https://huggingface.co/spaces/Kunbyte/ROSE`
- HF dataset: `https://huggingface.co/datasets/Kunbyte/ROSE-Dataset`

Readback facts from public metadata:

- GitHub `main` HEAD: `6be41c5420bf331c6d491277d5a6feaf9b3a779a`
- Model repo SHA: `8a5e57c9f25e73a4b62d324719cdf3367c13df59`
- Dataset repo SHA: `94159b52c62c914dbf86ab6be4cbae6039cae9aa`
- HF Space repo SHA: `0ea1fc65605d8734bd85df2c12d8198687cc4229`
- Model license: Apache-2.0
- Code license: Apache-2.0
- Dataset license: CC-BY-NC-4.0
- Model repo is public and not gated in HF API metadata.
- Dataset repo is public and not gated in HF API metadata.
- Space repo is public and not gated in HF API metadata.

## Why ROSE Is The Next Candidate

DiffuEraser and VideoPainter remain the positive adapter evidence. MiniMax is plumbing-positive and trainability-positive, but repeated objective/data recipes did not become quality-positive. ROSE is better aligned with the VOR-OR problem because it targets object removal with side effects, uses video context, includes side-effect-oriented masks, and is based on a Wan-style diffusion transformer family that may expose native training code.

## Promotion Gate

1. `ROSE_ASSETS_READY`: code/demo/model/base/dataset sample inventory must be present on PAI with checksums.
2. `ROSE_ENV_READY`: isolated environment imports official code and passes CUDA smoke.
3. `ROSE_TRUE_ADAPTER_FEASIBLE`: official code exposes trainable forward, target parameterization, checkpoint save/reload, and trainable module scope.
4. `ROSE_INFERENCE_SMOKE_PASS` or technically valid weak smoke: official inference must produce decodable raw VOR-OR outputs.
5. `ROSE_VOR_OR_GATE16_PASS` or technically valid weak gate: Gate16 baseline/loser-yield review must pass technical validity.
6. `ROSE_ONE_STEP_PASS`: zero-gap, one-step, strict reload, finite gradients, and small nonzero output movement.
7. `ROSE_ADAPTER_10STEP_PROMISING` or `ROSE_ADAPTER_10STEP_POSITIVE`: only after all previous gates.

Inference smoke alone can only support `ROSE_BASELINE_READY` or loser-generator language. It cannot support adapter success.

## Next Action

Resume on a verified PAI shell or provide a reachable PAI SSH endpoint. The first PAI command must confirm:

```bash
hostname
date -Ins
ls -ld /mnt/nas/hj/H20_Video_inpainting_DPO /mnt/workspace/hj/nas_hj
```

Only after that should Milestone B start staged asset download.
