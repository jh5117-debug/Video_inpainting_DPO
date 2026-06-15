# PRD 29: VideoPainter DPO Adapter Trainer

Date: 2026-06-15

## Summary

VideoPainter can support direct Diff-DPO in principle because its official
training loop is diffusion / denoising based. However, the actual isolated DPO
adapter trainer is not implemented yet, so gate2000 remains blocked.

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

## Why Current Gate Is Blocked

The missing file is:

```text
exp14_adapter_videopainter/code/train_videopainter_dpo_adapter.py
```

The upstream training code is insufficient because it:

- reads one clean video and one mask, not GT winner + generated loser pairs;
- uses VideoPainter CSV + `all_masks.npz`, not our frame-directory manifest;
- has one policy branch, not policy + frozen reference;
- does not run reference forward;
- does not compute normalized DPO gaps;
- does not record `dpo_diagnostics.csv`;
- does not run the project fixed metric / four-column eval.

## Required Implementation Plan

The trainer must be implemented only under:

```text
exp14_adapter_videopainter/code/
```

Required modules:

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
adapter_type = direct_diff_dpo_blocked_pending_isolated_trainer
gate2000 = not_launched
preflight = not_run
```

