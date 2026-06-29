# Exp45 H20 Handoff Instructions

Status: `MINIMAX_STAGE2_FORMAL_DATA_READY`

Do not execute this from PAI. A later H20 session should mirror and validate the package, then run the first training preflight.

## Required First Step

Mirror the files from `reports/exp45_h20_required_filelist.txt` using the template in `reports/exp45_h20_handoff_package.md`, then validate every absolute path in the Exp45 manifests.

## Handoff Package

- PAI source root: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp45_pai_minimax_pair_scaleup/scaleup_mining_20260629_190229`
- expected H20 target root: `/home/hj/H20_Video_inpainting_DPO_h20_mirror/exp45_pai_minimax_pair_scaleup/scaleup_mining_20260629_190229`
- filelist: `reports/exp45_h20_required_filelist.txt`
- sha256/status: `reports/exp45_h20_required_sha256.txt`
- filelist CSV: `reports/exp45_h20_required_filelist.csv`
- package summary: `reports/exp45_h20_handoff_package.json`
- required paths: `326`
- missing paths on PAI: `0`

## Training Guidance

- Formal split is `64/24/24`; training is unlocked for H20 handoff only.
- First H20 experiment must be pseudo-success SFT 30-step.
- Do not start GT-only SFT first.
- Do not run DPO before pseudo-success SFT has a positive 30-step gate.
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
