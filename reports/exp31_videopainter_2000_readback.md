# Exp31 VideoPainter 2000-Step Readback

Date: 2026-06-27

Status: `EXP31_READBACK_COMPLETED`

## Git Readback

- branch: `research/exp31-videopainter-2000step-longrun-20260627`
- base branch: `origin/research/exp26-videopainter-dpo-v2`
- HEAD: `568a7dfb48bcdfce893176a1dd48c653414a13a8`
- worktree: `/home/hj/H20_Video_inpainting_DPO_exp31_vp2000`
- `git status --short`: clean before this readback
- `git diff --stat`: empty before this readback
- `git diff --check`: passed before this readback

Recent base log:

```text
568a7df Audit third-model adapter compatibility
4983bfd Build VideoPainter result evidence pack
e6cc099 Review VideoPainter external validation videos
29a7cd2 Evaluate VideoPainter external validation metrics
32da83e Run VideoPainter external validation checkpoint trajectory
eca6844 Preregister external VideoPainter validation protocol
9af47b7 Inventory external 49-frame VideoPainter validation sources
a6ea7d9 Audit VideoPainter post-confirmation evidence
```

## Source Files Read

- `PRD/00_current_status.md`
- `PRD/01_experiment_matrix.md`
- `PRD/48_exp26_videopainter_dpo_v2.md`
- `experiment_registry/exp26_videopainter_dpo_v2/status.md`
- `experiment_registry/exp26_videopainter_dpo_v2/paths.yaml`
- `experiment_registry/exp26_videopainter_dpo_v2/config.yaml`
- `experiment_registry/exp26_videopainter_dpo_v2/results.tsv`
- `experiment_registry/exp26_videopainter_dpo_v2/metric_summary.md`
- `experiment_registry/exp26_videopainter_dpo_v2/qualitative_summary.md`
- `reports/exp26_vp_50step_final.md`
- `reports/exp26_vp_shadowdev_final_decision.md`
- `reports/exp26_vp_shadowdev_metrics_and_statistics.md`
- `reports/exp26_vp_shadowdev_visual_review.md`
- `reports/exp26_vp_50step_dynamics_audit.md`
- `reports/exp26_gate64_primary32_final.md`
- `reports/exp26_gate64_manifest_identity.json`
- `reports/exp26_external_validation_visual_review.md`
- `reports/exp26_videopainter_result_pack.md`

## Previous Step50 Identity

- run: `vp_primary32_50step_20260625_171032`
- primary32 SHA256:
  `82f661f2f30a581a213972533817624217eabb97eba7aaeedc00ee2109e4e716`
- search-dev SHA256:
  `41c6571d26e4a5130818dd50fbbe1314c9d953284561a3cd20f630572f7c2a71`
- shadow-dev SHA256:
  `0338dba1513cfe0e5dd85cbf793b3782902b981ac9610b0e472c6a048f738c02`
- checkpoint trajectory: Step0, Step10, Step30, Step50 only.

## Previous 50-Step Outcome

Search-dev Step50 minus Step0:

- whole PSNR: `+4.816168`
- strict mask PSNR: `+4.942246`
- boundary PSNR: `+12.111889`
- LPIPS: `-0.044059`
- Ewarp: `-7.055122`
- NaN/Inf count: `0`
- status: `VIDEOPAINTER_ADAPTER_POSITIVE` for the fixed 50-step search-dev
  micro gate only.

Shadow-dev Step50 minus Step0:

- whole PSNR: `+5.160739`
- strict mask PSNR: `+5.186942`
- boundary PSNR: `+12.175098`
- LPIPS: `-0.040142`
- Ewarp: `-8.378847`
- visual review: Step50 better or slightly better on `25/32`, tie on `3/32`,
  Step0 better or new Step50 artifact on `4/32`.
- status: `VIDEOPAINTER_SHADOWDEV_CONFIRMED`.

External DAVIS-derived validation:

- whole PSNR delta: `-2.563047`
- strict mask PSNR delta: `-2.610576`
- boundary PSNR delta: `+0.662358`
- LPIPS delta: `+0.002466`
- Ewarp delta: `-3.602171`
- status: `EXP26_EXTERNAL_VALIDATION_NOT_CONFIRMED`.

## Why 2000 Steps Are Needed

Exp26 proves a fixed 50-step VideoPainter VOR-BG micro result and confirms it
on shadow-dev. It does not prove a stable long-run trajectory. Exp31 is needed
to test whether the VideoPainter adapter remains beneficial at the intended
long-run scale, with Step2000 pre-registered as the primary endpoint.

## What Will Not Change

- no writes to Exp26 output roots;
- no checkpoint reselection on shadow-dev;
- no VOR-Eval training, loser mining, threshold tuning, or checkpoint choice;
- no MiniMax recipe or MiniMax adapter training;
- no `inference/metrics.py` change;
- no shared trainer change;
- no universal-adapter or final-SOTA claim.

## Right-Side MiniMax Protection State

Read-only PAI checks:

- `2026-06-27T12:56:38+08:00`: all GPUs 0-7 had `0 MiB` used memory, `0%`
  utilization, and no compute apps.
- `2026-06-27T12:58:03+08:00`: same result.
- `2026-06-27T13:04:42+08:00`: targeted Exp30/MiniMax process query found no
  active Exp30/MiniMax process and no compute apps.

Detected protected state:

- local Exp30 worktree exists at
  `/home/hj/H20_Video_inpainting_DPO_exp30_vor_or_minimax`.
- PAI Exp30/MiniMax output directories exist under
  `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp30_vor_or_multimodel_minimax`.
- stale MiniMax candidate locks exist for GPU0 and GPU5.

Decision:

- reserve GPU0 and GPU5 conservatively;
- eligible idle GPUs for left-side work: GPU1, GPU2, GPU3, GPU4, GPU6, GPU7;
- Exp31 planned GPU: GPU1.

No signal was sent and no right-side file was modified.

## Optimizer/Scheduler Resume State

Not yet audited in this readback. The next required milestone is
`reports/exp31_vp_resume_policy_audit.md` and
`reports/exp31_vp_resume_policy_audit.json`.

If full optimizer and scheduler state exists, Exp31 will continue from Step50
to total Step2000 and label the run
`VIDEOPAINTER_2000_CONTINUED_FROM_STEP50`.

If it does not exist, Exp31 will start a fresh total-2000 run from Step0 and
label the run `VIDEOPAINTER_2000_FRESH_FROM_STEP0`.

## Decision

Readback passes. Training remains blocked until resume-policy audit and L0/L1
preflight pass.

