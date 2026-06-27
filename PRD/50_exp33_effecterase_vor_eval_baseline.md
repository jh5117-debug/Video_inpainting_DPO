# PRD 50: Exp33 EffectErase VOR-Eval Baseline

Date: 2026-06-27

## Objective

Evaluate EffectErase as a held-out VOR-Eval OR baseline only.

Exp33 does not train, adapt, run zero-gap, run one-step, run DPO, or mine
preference losers. It only evaluates the official EffectErase removal pipeline
on held-out VOR-Eval if the official 81-frame protocol supports the rows.

## Isolation

- Branch: `research/exp33-effecterase-vor-eval-baseline-20260627`
- Base: `origin/research/exp29-minimax-effecterase-adapter-feasibility-20260626`
- Base HEAD: `6bc6c67c60b5cf2fe8d937ffd1e1d88a4684991c`
- HAL worktree: `/home/hj/H20_Video_inpainting_DPO_exp33_effecterase_eval`
- PAI runtime root:
  `/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp33_effecterase_eval`
- Experiment output root:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp33_effecterase_vor_eval_baseline`
- Log root:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp33_effecterase_vor_eval_baseline`

Exp33 must not write Exp29, Exp30, or MiniMax outputs. It must not modify
`inference/metrics.py` or shared trainer code.

## Prior EffectErase Evidence

Exp29 official 81-frame diagnostic smoke:

- status: `EFFECTERASE_OR_BASELINE_READY`
- rows: `8`
- technical valid: `8/8`
- manifest SHA256:
  `706cb09286fd8528d7efbbb91eb89673a9ec7ce61b0047e6b3b2e8ea4c9b1fb3`
- output root:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp29_or_adapter_feasibility/effecterase_inference_smoke_official81_20260626`

Project metric wrapper on the 8 diagnostic rows:

- whole PSNR: `27.416948`
- whole SSIM: `0.840580`
- whole LPIPS: `0.085822`
- mask PSNR: `25.778614`
- mask SSIM: `0.760667`
- boundary PSNR: `25.696018`
- boundary SSIM: `0.768534`
- Ewarp mask region: `1.766501`
- outside diff mean: `8.210687`

Visual review found 8/8 object/effect removal and no global collapse, but the
rows are VOR-confounded and too strong/clean for primary DPO loser use.

## Exp33 Protocol

VOR-Eval rules:

- held-out only;
- no tuning;
- no threshold selection;
- no loser mining;
- no training;
- no checkpoint selection.

Planned gate:

1. Audit VOR-Eval 43 rows for official 81-frame compatibility.
2. If all or a pre-registered subset passes, materialize 81-frame inputs in
   Exp33 output roots.
3. Run official EffectErase inference as baseline only.
4. Save raw output, diagnostic comp, condition, winner, mask, side-by-side
   videos, temporal strips, affected crops, and metric CSVs.
5. Visually review all 43 if feasible, otherwise all failures plus a stratified
   pass set.

## Right-Side Protection

Read-only PAI checks found no active compute process. Exp30/MiniMax outputs
exist and stale MiniMax candidate locks reserve GPU0 and GPU5, so Exp33 may use
GPU3 by plan and must avoid GPU0 and GPU5.

## Status

Current status: `EXP33_READBACK_COMPLETED_VOREVAL_PENDING`

Final-status family: `EFFECTERASE_BASELINE_ONLY_FOR_NOW`

