# Exp45 H20 Handoff Instructions

Status: `MINIMAX_STAGE2_FORMAL_DATA_PARTIAL`

Do not execute this from PAI. A later H20 session may use it only after mirroring and validating the listed files.

## Required First Step

Mirror the files from `reports/exp45_h20_required_filelist.txt` using the template in `reports/exp45_h20_handoff_package.md`, then validate every absolute path in the Exp45 manifests.

## Training Guidance

- Current split is only `24/8/8`; training remains `TRAINING_NOT_UNLOCKED`.
- If H20 runs a debug preflight anyway, first experiment must be pseudo-success SFT 30-step.
- Do not start GT-only SFT first.
- Do not run DPO from this partial package.
- Raw output must remain primary; no hard comp.

## Manifest Paths

- `exp45_pai_minimax_pair_scaleup/manifests/exp45_stage2_pseudosuccess_train.jsonl`
- `exp45_pai_minimax_pair_scaleup/manifests/exp45_stage2_pseudosuccess_search.jsonl`
- `exp45_pai_minimax_pair_scaleup/manifests/exp45_stage2_pseudosuccess_shadow.jsonl`
- `exp45_pai_minimax_pair_scaleup/manifests/exp45_stage2_gt_distill_train.jsonl`
- `exp45_pai_minimax_pair_scaleup/manifests/exp45_stage2_gt_distill_search.jsonl`
- `exp45_pai_minimax_pair_scaleup/manifests/exp45_stage2_gt_distill_shadow.jsonl`
- `exp45_pai_minimax_pair_scaleup/manifests/exp45_stage2_preference_train.jsonl`
- `exp45_pai_minimax_pair_scaleup/manifests/exp45_stage2_preference_search.jsonl`
- `exp45_pai_minimax_pair_scaleup/manifests/exp45_stage2_preference_shadow.jsonl`
