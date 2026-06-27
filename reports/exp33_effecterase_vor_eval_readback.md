# Exp33 EffectErase VOR-Eval Baseline Readback

Date: 2026-06-27

Status: `EXP33_READBACK_COMPLETED_VOREVAL_PENDING`

## Git Readback

- branch: `research/exp33-effecterase-vor-eval-baseline-20260627`
- base branch: `origin/research/exp29-minimax-effecterase-adapter-feasibility-20260626`
- HEAD: `6bc6c67c60b5cf2fe8d937ffd1e1d88a4684991c`
- worktree: `/home/hj/H20_Video_inpainting_DPO_exp33_effecterase_eval`
- `git status --short`: clean before this readback
- `git diff --stat`: empty before this readback
- `git diff --check`: passed before this readback

Recent base log:

```text
6bc6c67 Audit EffectErase adapter feasibility
afb1561 Run EffectErase official 81-frame inference smoke
e16cac5 Validate EffectErase official 81-frame command
35ece20 Materialize EffectErase official 81-frame inputs
9103c4e Preregister EffectErase official 81-frame smoke
2a596aa Record Exp29 continuation v5 readback
c06958c Review MiniMax expanded source-pool candidates v2
372e5f2 Audit MiniMax full VOR source candidates
```

## Source Files Read

- `PRD/49_exp29_or_adapter_feasibility.md`
- `experiment_registry/exp29_or_adapter_feasibility/status.md`
- `experiment_registry/exp29_or_adapter_feasibility/results.tsv`
- `reports/exp29_effecterase_official81_preregistration.md`
- `reports/exp29_effecterase_official81_input_materialization.md`
- `reports/exp29_effecterase_official81_command_validation.md`
- `reports/exp29_effecterase_official81_inference_smoke.md`
- `reports/exp29_effecterase_official81_aggregate_metrics.csv`
- `reports/exp29_effecterase_official81_inference_visual_review.csv`
- `reports/exp29_effecterase_trainable_forward_audit.md`
- `reports/exp29_minimax_effecterase_adapter_summary.md`

## Prior EffectErase Baseline State

Exp29 official 81-frame diagnostic smoke passed:

- `EFFECTERASE_OFFICIAL81_PREREGISTERED`
- `EFFECTERASE_OFFICIAL81_INPUTS_READY`
- `EFFECTERASE_OFFICIAL81_COMMAND_READY`
- `EFFECTERASE_OR_BASELINE_READY`
- `EFFECTERASE_BASELINE_ONLY_FOR_NOW`

Official 81-frame diagnostic metrics on 8 rows:

| metric | mean |
| --- | ---: |
| whole_video_psnr | 27.416948 |
| whole_video_ssim | 0.840580 |
| whole_video_lpips | 0.085822 |
| mask_region_psnr | 25.778614 |
| mask_region_ssim | 0.760667 |
| boundary_psnr | 25.696018 |
| boundary_ssim | 0.768534 |
| ewarp_mask_region | 1.766501 |
| outside_region_diff_mean | 8.210687 |

Visual review:

- technical-valid outputs: 8/8.
- object/effect removal: 8/8.
- black/purple/global collapse: 0/8.
- baseline-ready diagnostic rows: 8/8.
- medium-hard loser rows: 0/8.

Interpretation: EffectErase is strong as an OR baseline/diagnostic, but not a
training adapter and not a primary DPO loser source.

## VOR-Eval Baseline Decision

The prior official 81-frame run is not yet VOR-Eval. Exp33 must audit VOR-Eval
43 rows and then run the official EffectErase pipeline only if the rows satisfy
the pre-registered 81-frame input protocol or a smaller subset is explicitly
pre-registered before inference.

Training and adapter work remain forbidden in Exp33.

## Right-Side Protection

Read-only PAI checks:

- `2026-06-27T12:56:38+08:00`: GPUs 0-7 idle, no compute apps.
- `2026-06-27T12:58:03+08:00`: GPUs 0-7 idle, no compute apps.
- `2026-06-27T13:04:42+08:00`: no active Exp30/MiniMax process and no compute
  apps.

Protected state:

- Exp30 worktree exists locally at
  `/home/hj/H20_Video_inpainting_DPO_exp30_vor_or_minimax`.
- Exp30/MiniMax outputs exist on PAI.
- stale MiniMax locks reserve GPU0 and GPU5.

Exp33 planned GPU is GPU3 for baseline inference only. GPU0 and GPU5 remain
avoided.

No signal was sent and no right-side file was modified.

## Decision

Readback passes. Exp33 may proceed to a VOR-Eval 81-frame compatibility audit
and preregistration. It must not train or adapt EffectErase.

