# Exp37 MiniMax Train-vs-Heldout Diagnosis

Status: `MINIMAX_OBJECTIVE_SIGNAL_TOO_WEAK`

Evaluated Exp36 best S1 winner-SFT/control checkpoint on locked Gate64 `train16` and `heldout16`. Codex reviewed all 32 Step0-vs-Step10 temporal strips plus individual full-resolution strips for best/worst metric rows.

## Quantitative Summary

### train16

- Rows: `16`
- Mean delta full PSNR: `0.029083`
- Mean delta mask PSNR: `-0.008362`
- Mean delta boundary PSNR: `0.001128`
- Mean delta outside PSNR: `0.048654`
- Positive mask rows: `9/16`
- Positive boundary rows: `7/16`
- Visual better: `0/16`
- Visual tie/no visible change: `16/16`
- Visual worse/new artifact: `0/16`

### heldout16

- Rows: `16`
- Mean delta full PSNR: `-0.010218`
- Mean delta mask PSNR: `-0.008293`
- Mean delta boundary PSNR: `-0.010939`
- Mean delta outside PSNR: `-0.014499`
- Positive mask rows: `6/16`
- Positive boundary rows: `5/16`
- Visual better: `0/16`
- Visual tie/no visible change: `16/16`
- Visual worse/new artifact: `0/16`

## Visual Review

All 32 Step0-vs-Step10 strips were reviewed. The Step10 column is visually indistinguishable from Step0 across train and heldout rows; the diff column remains near black. No row showed a clear local object/effect improvement. No collapse or new visible artifact was introduced, but the update is sub-perceptual and not useful.

Representative checked rows:

- Train best mask metric: `BLENDER_GRASS001_00001`, visually tie/no visible change.
- Train worst mask metric: `BLENDER_MOUNTAIN008_00001`, visually tie/no visible change.
- Heldout best mask metric: `REAL_ENV087_00002_001_01`, visually tie/no visible change.
- Heldout worst mask metric: `REAL_ENV105_00004_001_01`, visually tie/no visible change.

## Diagnosis

`MINIMAX_OBJECTIVE_SIGNAL_TOO_WEAK`: the selected Exp36 S1 checkpoint does not meaningfully improve train-side MiniMax outputs, so the failure is not merely train-positive/heldout-negative generalization. Cleaner LocalDPO-style local corruption pairs and explicit bad-noise state selection remain the next allowed direction. No 30-step or long training is unlocked.

Artifacts:

- Metrics: `reports/exp37_minimax_train_vs_heldout_metrics.csv`
- Visual review: `reports/exp37_minimax_train_vs_heldout_visual_review.csv`
- Summary: `reports/exp37_minimax_train_vs_heldout_summary.json`
- NAS output root: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp37_minimax_localdpo_badnoise_rescue/train_vs_heldout_20260628/outputs`
