# PRD 26: VideoPainter Adapter Feasibility

Date: 2026-06-15

## Goal

Check whether the current best method:

```text
Exp11 outer b0.75 S2
```

can be adapted to VideoPainter.

## Current Stage

```text
feasibility + smoke planning only
```

No 2000-step training is allowed yet.

## Current Best Loss Idea

```text
region-local normalized-gap clipped-loser-gap winner-anchored DPO
boundary_mode = outer
boundary_weight = 0.75
outside_weight = 0.05
```

## VideoPainter Status

Repo:

```text
https://github.com/TencentARC/VideoPainter
/home/hj/dpo-2-1-exp/third_party_baselines/VideoPainter
```

Commit checked on HAL:

```text
bbab6cd5cd5cb89f0e2444305c32fd74a010ae0a
```

VideoPainter has official training code and is CogVideoX / DiT diffusion-based.

## Adapter Decision

```text
adapter_type = direct_diff_dpo_design_feasible_not_implemented
```

It is feasible because VideoPainter exposes diffusion timesteps, noise, latent
targets, masks, and inpainting loss. It is not ready because the upstream code
does not implement winner/loser pair loading, frozen reference forward, DPO loss,
or diagnostics.

## Not Allowed Yet

- 2000-step gate
- full training
- paper result claim
- modifying old Exp9 / Exp10 / Exp11 code
- modifying shared `training/dpo`

## Reports

```text
exp14_adapter_videopainter/reports/videopainter_repo_audit.md
exp14_adapter_videopainter/reports/videopainter_training_entry_audit.md
exp14_adapter_videopainter/reports/videopainter_loss_interface_audit.md
exp14_adapter_videopainter/reports/videopainter_reference_model_audit.md
```

