# Exp40 MiniMax PSNR-Safe Readback

Status: `EXP40_READBACK_COMPLETED`

Branch: `research/exp40-minimax-psnr-safe-rescue-20260628`

Start HEAD: `06b17c0a4be2cb82d1ffbdf7b6c93406f37a3ff8`

## Files Read

PRD and registry:

- `PRD/00_current_status.md`
- `PRD/01_experiment_matrix.md`
- `PRD/54_exp38_minimax_full_adapter_breakthrough.md`
- `experiment_registry/exp38_minimax_full_adapter_breakthrough/status.md`
- `experiment_registry/exp38_minimax_full_adapter_breakthrough/results.tsv`
- `experiment_registry/exp38_minimax_full_adapter_breakthrough/metric_summary.md`
- `experiment_registry/exp38_minimax_full_adapter_breakthrough/qualitative_summary.md`

Reports:

- `reports/exp38_minimax_full_readback.md`
- `reports/exp38_minimax_failure_taxonomy.md`
- `reports/exp38_minimax_decision_tree.json`
- `reports/exp38_minimax_train_overfit_diagnosis.md`
- `reports/exp38_localdpo_v2_pool.md`
- `reports/exp38_minimax_badnoise_v2_diagnostic_scan.md`
- `reports/exp38_minimax_sft_dpo_rescue_10step.md`
- `reports/exp38_minimax_sft_dpo_rescue_10step_codex_review.md`
- `reports/exp38_minimax_sft_dpo_rescue_10step_metrics.csv`
- `reports/exp38_minimax_sft_dpo_rescue_10step_visual_review.csv`
- `reports/exp38_minimax_sft_dpo_rescue_10step_summary.json`

## GPU0/GPU1 Audit

PAI hostname: `dsw-753014-85f54df947-bkp7h`

GPU0/GPU1 were free by `nvidia-smi` before Exp40. A stale old Exp30 process
group was found:

- PID: `1715136`
- PGID: `1715134`
- user: `hj`
- cwd: `/mnt/nas/hj/H20_Video_inpainting_DPO/runtime/exp30_vor_or_minimax_worktree`
- command: old Exp30 gate64 launcher containing a MiniMax GPU0 heartbeat
- compute PID: none

The process group was terminated with `TERM -- -1715134`, waited for 30 seconds,
and no `KILL` was required. GPU0/GPU1 remained `0 MiB`, `0%`, and no compute
PID after cleanup. GPU2-GPU7 were not used or signaled.

Audit path:

`/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp40_minimax_psnr_safe_rescue/gpu0_1_preclean_exp30_stale_20260629_030722.txt`

## R1 Positive-Signal Audit

Exp38 R1 heldout13 aggregate:

- full PSNR: `+0.102167`
- mask PSNR: `+0.117230`
- boundary PSNR: `-0.141510`
- outside PSNR: `-0.037262`

Sample-level counts:

- full-positive rows: `7/13`
- mask-positive rows: `7/13`
- boundary-negative rows: `9/13`
- outside-negative rows: `8/13`
- outside-MAE-worse rows: `10/13`

Visual classification from Exp38 reviewed CSV:

- clear better: `0/13`
- worse/tradeoff: `9/13`
- tie/local numeric gain: `4/13`

Rows with positive full and mask PSNR:

- `REAL_ENV103_00001_001_01`: full `+0.564`, mask `+0.615`, boundary `-0.581`, outside `+0.185`, visual `STEP10_LOCAL_TRADEOFF_NOT_BETTER`.
- `REAL_ENV089_00001_001_01`: full `+0.193`, mask `+0.358`, boundary `-0.132`, outside `-0.123`, visual `STEP10_LOCAL_TRADEOFF_NOT_BETTER`.
- `REAL_ENV093_00001_001_01`: full `+0.156`, mask `+0.043`, boundary `+0.202`, outside `+0.293`, visual `TIE_LOCAL_NUMERIC_GAIN`.
- `REAL_ENV103_00004_001_01`: full `+0.081`, mask `+0.179`, boundary `-0.354`, outside `-0.244`, visual `STEP10_LOCAL_TRADEOFF_NOT_BETTER`.
- `REAL_ENV095_00002_001_01`: full `+0.073`, mask `+0.010`, boundary `+0.036`, outside `+0.093`, visual `TIE_LOCAL_NUMERIC_GAIN`.
- `REAL_ENV104_00001_001_01`: full `+1.023`, mask `+1.028`, boundary `+0.364`, outside `+0.853`, visual `STEP10_LOCAL_TRADEOFF_NOT_BETTER` due fogging/over-erasure.
- `REAL_ENV105_00004_001_01`: full `+0.031`, mask `+0.100`, boundary `+0.442`, outside `+0.014`, visual `TIE_NO_MEANINGFUL_CHANGE`.

## Required Answers

1. R1 had `+0.102 dB` full PSNR because a subset of rows moved raw pixels in the
   mask toward the winner. It failed visual gate because that movement is often
   fogging, over-erasure, or boundary/outside tradeoff.
2. Numerically improved rows are listed above; reliable clear visual wins are
   zero.
3. Train-overfit evidence from Exp38 did not show a larger clean train win:
   Exp37 R1 train32 had local movement but worse full/outside behavior.
4. Boundary loss is a main blocker.
5. Outside damage is also a main blocker.
6. R1 is not simply too much SFT; it is local DPO pressure without enough
   boundary/outside preservation and without a PSNR-safe SFT anchor.
7. Keep: local Linear-DPO, hard-state bookkeeping, and raw-output PSNR signal.
8. Do not repeat: R2 SDPO-safe as configured or R3 SFT-warmup as configured,
   because both worsen boundary/outside aggregates.
9. Exp40 success target is shadow raw full PSNR `> +0.2 dB` with safe mask,
   boundary, outside, LPIPS, Ewarp, and clean visual review.
10. PRD/report files read are listed above.

No GPU training or inference was launched in this milestone.
