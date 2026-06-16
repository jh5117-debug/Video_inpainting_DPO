# PRD 29: VideoPainter DPO Adapter Trainer

Date: 2026-06-15

## Summary

VideoPainter can support direct Diff-DPO in principle because its official
training loop is diffusion / denoising based. An isolated first-pass DPO
adapter trainer has now been implemented under Exp14. The PAI preflight passed
with real VideoPainter weights, and gate2000 completed 2000 steps.

## Why Direct Diff-DPO Is Structurally Possible

The upstream training loop exposes:

- VAE latent encoding of clean video;
- noise sampling;
- random diffusion timesteps;
- scheduler `add_noise`;
- branch + transformer forward;
- `model_pred = scheduler.get_velocity(...)`;
- denoising target `target = model_input`;
- mask tensor downsampled to latent resolution;
- weighted denoising MSE and mask inpainting MSE.

Therefore the following DPO quantities can be defined if we add a pair
dataloader and frozen reference forward:

```text
m_w     = policy winner region loss
m_l     = policy loser region loss
m_w_ref = reference winner region loss
m_l_ref = reference loser region loss
```

## Required Exp14 Loss

Use the current best method setting from Exp11 outer b0.75 S2:

```text
boundary_mode = outer
mask_weight = 1.0
boundary_weight = 0.75
outside_weight = 0.05
beta_dpo = 10
lose_gap_weight = 0.25
lose_gap_clip_tau = 1.0
winner_abs_reg_weight = 0.05
winner_gap_reg_weight = 1.0
winner_gap_reg_margin = 0.0
```

Formula:

```text
g_w = log((m_w + eps) / (m_w_ref + eps))
g_l = log((m_l + eps) / (m_l_ref + eps))
g_l_clip = clip(g_l, max=1.0)

L_DPO = mean[-logsigmoid(-0.5 * beta_dpo * (g_w - lose_gap_weight * g_l_clip))]
L_total = L_DPO + 0.05 * m_w + ReLU(g_w)
```

## Implemented Isolated Trainer

Implemented file:

```text
exp14_adapter_videopainter/code/train_videopainter_dpo_adapter.py
```

This trainer is isolated from old Exp9 / Exp10 / Exp11 code and does not modify
shared `training/dpo`.

It implements:

- a JSONL pair dataloader for `win_video_path`, `final_loser_video_path`, and
  `mask_path`;
- VideoPainter policy branch and frozen reference branch;
- winner / loser forward passes on shared noise and timestep;
- region-local MSE with `boundary_mode=outer`, `mask=1.0`,
  `boundary=0.75`, `outside=0.05`;
- normalized-gap clipped-loser-gap winner-anchored DPO;
- `dpo_diagnostics.csv`;
- `--preflight_only`;
- checkpoint / `last_weights` saving for the policy branch.

The upstream training code is still insufficient by itself because it:

- reads one clean video and one mask, not GT winner + generated loser pairs;
- uses VideoPainter CSV + `all_masks.npz`, not our frame-directory manifest;
- has one policy branch, not policy + frozen reference;
- does not run reference forward;
- does not compute normalized DPO gaps;
- does not record `dpo_diagnostics.csv`;
- does not run the project fixed metric / four-column eval.

## Trainer Design

The trainer remains only under:

```text
exp14_adapter_videopainter/code/
```

Implemented modules:

1. `VideoPainterPairDataset`
   - reads JSONL manifest;
   - loads `win_video_path`, `final_loser_video_path`, `mask_path` frame dirs;
   - converts mask convention `255=inpaint, 0=keep` into `mask=1` hole tensor;
   - produces winner, loser, masked conditioning, mask, prompt.

2. `VideoPainterDPOForward`
   - shares timestep/noise across winner and loser;
   - runs policy winner / loser forward;
   - runs reference winner / loser forward under `torch.no_grad()`;
   - computes region-local MSE.

3. Region map utilities
   - nearest-interpolate mask to latent resolution;
   - compute `boundary_outer = dilate(mask) - mask`;
   - compute `weight_map = 1.0 * mask + 0.75 * boundary_outer + 0.05 * outside`;
   - normalize by `sum(weight_map) + eps`.

4. Diagnostics writer
   - writes all required DPO columns every 10 steps.

5. Checkpoint writer
   - saves checkpoints every 500 steps and `last_weights`.

6. Eval adapter
   - uses VideoPainter baseline and adapter output;
   - writes project-standard metrics and four-column visualizations.

## Current Gate Blocker

Resolved. Gate2000 was previously blocked because PAI did not have the required
VideoPainter / CogVideoX weights, and PAI could not download them from Hugging
Face in the current network environment.

Required PAI paths:

```text
VIDEO_PAINTER_ROOT
VIDEO_PAINTER_BASE_MODEL
VIDEO_PAINTER_CHECKPOINT_ROOT
VIDEO_PAINTER_REFERENCE_CHECKPOINT_ROOT
```

The weights were downloaded on HAL and transferred to PAI. The updated gate
launcher runs preflight before training:

```text
exp14_adapter_videopainter/scripts/launch_videopainter_adapter_gate2000_pai.sh
```

If preflight fails, it must report blocked and must not launch upstream
VideoPainter training as a substitute.

PAI preflight status:

```text
status = passed
loss = 0.7026171684
dpo_loss = 0.6931471825
m_w = 0.1893996745
m_l = 0.2312944233
m_w_ref = 0.1893996745
m_l_ref = 0.2312944233
reference_has_grad = false
```

Implementation fixes needed for the PAI preflight:

- add a Transformers compatibility shim for the vendored VideoPainter /
  Diffusers import path;
- use VideoPainter's native `480x720` resolution rather than the earlier
  `320x512` DPO default;
- install the missing lightweight `ffmpeg-python` import dependency on PAI.

## Latest PAI Run

Date: 2026-06-16 CST

```text
sync_strategy = clean_worktree
clean_repo = /mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp14_videopainter_gate
source_commit = 2e187ee
status = completed_2000_steps
davis_eval = completed_davis50_with_thin_eval_adapter
```

What passed:

- Exp14 clean worktree created without touching the dirty priority repo.
- Exp14 trainer and launcher are present.
- Static checks passed.
- VideoPainter code repo was synced from HAL.
- YouTube-VOS, DAVIS, and DPO manifest paths are present.
- The manifest is PAI-safe.
- GPUs are available.
- Missing weights were resolved by downloading on HAL and transferring to PAI.
- Trainer preflight passed.
- Gate2000 completed 2000 steps.
- `checkpoint-500`, `checkpoint-1000`, `checkpoint-1500`, `checkpoint-2000`,
  and `last_weights` exist.

Resolved weight blocker:

```text
third_party/VideoPainter/ckpt/CogVideoX-5b-I2V
third_party/VideoPainter/ckpt/VideoPainter/checkpoints/branch
```

Historical PAI download failure:

```text
huggingface-cli download TencentARC/VideoPainter
  -> deprecated / exits failure

hf download TencentARC/VideoPainter
  -> httpx.ConnectError: [Errno 101] Network is unreachable
```

Resolution:

```text
HAL downloaded TencentARC/VideoPainter and THUDM/CogVideoX-5b-I2V.
HAL transferred weights to PAI via rsync --partial --append-verify.
```

Required final layout:

```text
third_party/VideoPainter/ckpt/CogVideoX-5b-I2V/model_index.json
third_party/VideoPainter/ckpt/CogVideoX-5b-I2V/transformer/
third_party/VideoPainter/ckpt/CogVideoX-5b-I2V/vae/
third_party/VideoPainter/ckpt/CogVideoX-5b-I2V/tokenizer/
third_party/VideoPainter/ckpt/CogVideoX-5b-I2V/text_encoder/
third_party/VideoPainter/ckpt/VideoPainter/checkpoints/branch/config.json
third_party/VideoPainter/ckpt/VideoPainter/checkpoints/branch/diffusion_pytorch_model.safetensors
```

## Preflight Requirement

The user does not want smoke experiments, but the trainer must pass a minimum
preflight before gate2000:

```text
load model -> load one batch -> policy forward -> reference forward ->
compute DPO loss -> backward once -> verify diagnostics
```

This preflight is not an experiment result.

## Current Status

```text
adapter_type = direct_diff_dpo_isolated_trainer
gate2000 = completed_2000_steps
preflight = passed_on_pai
trainer = implemented_locally
davis_eval = completed_davis50_with_thin_eval_adapter
```

Limitations:

- This is a branch-adapter DPO trainer, not full VideoPainter model finetuning.
- Multi-GPU sharding is not implemented in the isolated trainer.
- DAVIS four-column eval integration is now implemented by an Exp14 thin eval
  adapter. It does not use upstream VideoPainter eval as the final metric path.
- Full DAVIS50 eval is negative for the adapter: VideoPainter baseline scores
  31.6124 PSNR / 0.9608 SSIM, while the DPO adapter scores 29.8028 PSNR /
  0.9580 SSIM.

## DAVIS Eval Adapter

Implemented file:

```text
exp14_adapter_videopainter/code/eval_videopainter_adapter_davis.py
```

Launch script:

```text
exp14_adapter_videopainter/scripts/run_videopainter_adapter_davis_eval_pai.sh
```

Responsibilities:

- load VideoPainter official baseline branch;
- load Exp14 gate2000 `last_weights`;
- verify adapter weights differ from baseline;
- generate baseline and adapter outputs on DAVIS50;
- apply hard comp with GT outside the mask;
- avoid mask dilation / Gaussian blur / VBench;
- call the project metric backend from `inference/metrics.py`;
- save four-column videos, contact sheets, frame-by-frame images, and metrics.

Eval output:

```text
/mnt/nas/hj/H20_Video_inpainting_DPO_exp14_videopainter_gate/logs/target_eval/exp14_videopainter_adapter_gate2000_davis
```

Conclusion:

The trainer is technically functional, but the current Exp11-style DPO adapter
does not transfer successfully to VideoPainter. Do not continue longer training
without redesigning the adapter objective / data pairing.
