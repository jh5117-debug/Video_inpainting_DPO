# Exp50 VOID Preference-Wrapper Readback

Status: `VOID_PREFERENCE_WRAPPER_REQUIRED_CONFIRMED`

Time: 2026-06-30T22:59:39+08:00

## Source State

- Branch: `research/exp50-pai-void-adapter-feasibility-20260630`
- Start HEAD for H0: `39b1e005438bcd19622e81bf262ad65696a65749`
- Environment: `/home/hj/conda_envs/void_exp50_official_v2`
- Official repo: `/mnt/nas/hj/H20_Video_inpainting_DPO/third_party/VOID/Netflix_void-model`
- Weights: relayed official `netflix/void-model` and `alibaba-pai/CogVideoX-Fun-V1.5-5b-InP`
- Micro data: `manifests/exp50_void_adapter_train4.jsonl`, `manifests/exp50_void_adapter_heldout4.jsonl`

## Required Answers

1. Why official `train.py` cannot directly serve as DPO forward:
   It is an SFT-style trainer. It loads one target video per sample, builds a noisy latent, predicts noise or velocity with one trainable transformer, and minimizes MSE. It does not expose a frozen reference model, paired winner/loser targets, same-noise/same-timestep paired losses, or a DPO/preference margin.

2. Current VOID official training loss:
   `F.mse_loss(noise_pred.float(), target.float(), reduction="mean")`, optionally mixed with a temporal motion sub-loss. The target is selected from the scheduler prediction type.

3. Inputs needed by policy/reference forward:
   `rgb_full.mp4` condition, `quadmask_0.mp4`/official `mask.mp4`, prompt text, winner target (`rgb_removed.mp4`), loser target (same-model VOID medium-hard output or controlled local loser), scheduler state, timestep, shared noise, policy transformer, frozen reference transformer, VAE, text encoder/tokenizer, and region masks derived from the quadmask.

4. How winner/loser target should enter VOID-native loss:
   Winner and loser videos should be encoded through the same VOID/CogVideoX VAE path to latents. For a shared timestep and shared noise, each target latent gets its scheduler target (`epsilon` or velocity). Policy and reference predict against each target under the same condition and quadmask. Preference loss is formed from winner and loser loss gaps, not from raw pixels or hard composition.

5. Target parameterization:
   VOID follows the scheduler `prediction_type`: official code uses `epsilon` when `noise_scheduler.config.prediction_type == "epsilon"`, or `v_prediction` via `noise_scheduler.get_velocity(...)` when configured. The base scheduler config must be read at runtime and recorded.

6. Can single-process forward avoid deepspeed:
   Yes for tiny forward if memory allows. The Python `train.py` has non-deepspeed branches and `Accelerator` can run single-process. The official shell defaults to 8-process deepspeed, but a wrapper can call modules directly in one process.

7. If deepspeed is only launcher, can it be bypassed:
   For isolated wrapper forward, yes. For official shell training exactly as released, no: the shell runner uses `--use_deepspeed`. This continuation should bypass shell training and implement an isolated wrapper.

8. If deepspeed becomes essential, minimal controlled install:
   Extend the HAL/H20 wheelhouse with a deepspeed artifact compatible with torch `2.7.1+cu126`, install with `--no-index --find-links`, and prevent dependency resolution from pulling torch 2.12/CUDA 13. Do not install from PAI internet.

9. Why 10-step cannot directly run:
   Zero-gap and one-step have not passed. Running 10 optimizer steps without SFT parity, policy/reference preference forward, and zero-gap would create off-protocol evidence.

10. Current VOID role:
   VOID is usable for VOR-OR inference on PAI and is a baseline / loser-generator candidate. It is not adapter evidence yet.

## Safety

No training, optimizer step, VOR-Eval, hard comp, deepspeed install, base env modification, official source modification, VOID positive claim, universal adapter claim, or final SOTA claim was made in H0.
