# VideoPainter DPO Adapter Trainer Implementation

Date: 2026-06-15

## Status

Implemented locally; PAI preflight pending.

## Implemented File

```text
exp14_adapter_videopainter/code/train_videopainter_dpo_adapter.py
```

The implementation is isolated under Exp14 and does not modify:

- `training/dpo/train_stage1.py`
- `training/dpo/train_stage2.py`
- Exp9 / Exp10 / Exp11 / Exp12 code
- upstream VideoPainter code

## Adapter Type

```text
adapter_type = direct_diff_dpo_isolated_trainer
```

This is a VideoPainter branch-adapter DPO trainer. It uses the upstream
VideoPainter CogVideoX denoising structure but adds the missing DPO pieces:

- GT winner / generated loser pair dataloader;
- trainable policy branch;
- frozen reference branch;
- same-timestep/same-noise policy and reference loss;
- region-local normalized-gap DPO;
- DPO diagnostics.

## Loss

The trainer implements the current best setting from Exp11 outer b0.75 S2:

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

Definitions:

```text
m_w     = policy winner region-local denoising MSE
m_l     = policy loser region-local denoising MSE
m_w_ref = frozen-reference winner region-local denoising MSE
m_l_ref = frozen-reference loser region-local denoising MSE

g_w = log((m_w + eps) / (m_w_ref + eps))
g_l = log((m_l + eps) / (m_l_ref + eps))
g_l_clip = clip(g_l, max=lose_gap_clip_tau)

L_DPO = mean[-logsigmoid(-0.5 * beta_dpo * (g_w - lose_gap_weight * g_l_clip))]
L_total = L_DPO + 0.05 * m_w + ReLU(g_w)
```

## Data Interface

The trainer reads the current project DPO manifest directly:

```text
win_video_path
final_loser_video_path
mask_path
```

Each path is expected to be a frame directory. Masks use:

```text
255 = inpaint / hole
0   = keep / known
```

The trainer converts this to:

```text
mask = 1.0 for hole
mask = 0.0 for known outside
```

## Diagnostics

The trainer writes:

```text
exp14_adapter_videopainter/dpo_diag/dpo_diagnostics.csv
```

Preflight writes:

```text
exp14_adapter_videopainter/dpo_diag/preflight_dpo_diagnostics.csv
exp14_adapter_videopainter/runs/preflight/preflight_report.json
```

Required columns include:

- loss
- dpo_loss
- implicit_acc
- `m_w`, `m_l`, `m_w_ref`, `m_l_ref`
- raw / normalized gaps
- clipped loser gap
- winner regularizers
- loser dominant ratio
- grad norm
- mask / boundary / outside area ratios
- region weights

## Gate Launcher Change

Updated:

```text
exp14_adapter_videopainter/scripts/launch_videopainter_adapter_gate2000_pai.sh
```

The launcher now:

1. prechecks VideoPainter repo, base model, branch checkpoints, data, and GPU;
2. compiles the isolated trainer;
3. runs `--preflight_only`;
4. launches 2000-step training only if preflight succeeds.

## What Is Not Yet Done

- PAI preflight has not been run.
- Gate2000 has not been launched.
- DAVIS eval after gate2000 is not implemented in this trainer yet.
- Multi-GPU sharding is not implemented; memory feasibility must be checked on
  PAI during preflight.
- This is not upstream VideoPainter official training and should not be
  compared as such.

## Decision

Do not run upstream VideoPainter training as a substitute. Sync this Exp14 code
to PAI and run the updated launcher. If preflight fails, keep Exp14 blocked and
record the exact failure.
