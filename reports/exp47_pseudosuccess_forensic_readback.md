# Exp47 Pseudo-Success Forensic Readback

Status: `EXP47_READBACK_READY`

- Branch: `research/exp47-h20-minimax-pseudosuccess-forensic-20260629`
- Start HEAD: `94d2531a6782914e91bd4629fb477e154cfba98b`
- Base Exp46 final HEAD: `94d2531a6782914e91bd4629fb477e154cfba98b`

## What Failed In Exp46

Exp46 pseudo-success SFT30 completed technically (`TRAIN_DONE`, checkpoint-30, finite loss/gradients) but failed quality gates. Step30 regressed raw search/shadow outputs and visual review marked all rows worse.

| split | dFull PSNR | dMask PSNR | dBoundary PSNR | dOutside PSNR | dEwarp | visual worse |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| search | -4.612642 | -0.548113 | -1.591353 | -4.812891 | -0.019463 | 24/24 |
| shadow | -3.366753 | -5.674479 | -3.636023 | -3.029058 | 0.021337 | 24/24 |

## Manifests And Target Fields

Exp46 training used the runner-compatible pseudo-success manifest, not GT-only and not DPO preference manifests. The active target field consumed by the Exp43 runner is `winner_path`, which Exp46 rewrote to H20-local pseudo-success frame directories. The original pseudo-success mp4 path is preserved in `pseudo_success_mp4` and PAI fields where available.

- train: `manifests/exp46_runner_pseudosuccess_train.jsonl`
- search: `manifests/exp46_runner_pseudosuccess_search.jsonl`
- shadow: `manifests/exp46_runner_pseudosuccess_shadow.jsonl`
- search_shards: `manifests/exp46_step0_shards/search_shard{0..7}.jsonl`
- shadow_shards: `manifests/exp46_step0_shards/shadow_shard{0..7}.jsonl`

Active per-row fields to audit in Exp47: `condition_path`, `mask_path`, `winner_path`, `pseudo_success_mp4`, `loser_path`, `source_group`, `source_id`, `split`, `target_type`, `hard_comp_used`, `vor_eval_used`.

## Checkpoints And Outputs

- Checkpoints found: `5`
- Step0 raw mp4 count: `48`
- Step30 raw mp4 count: `48`

- `/home/nvme01/H20_Video_inpainting_DPO/experiments/dpo/exp46_h20_minimax_pseudosuccess_sft/sft_ladder/PSEUDO-SFT-A_lr3em5_step30/checkpoints/checkpoint-0`
- `/home/nvme01/H20_Video_inpainting_DPO/experiments/dpo/exp46_h20_minimax_pseudosuccess_sft/sft_ladder/PSEUDO-SFT-A_lr3em5_step30/checkpoints/checkpoint-1`
- `/home/nvme01/H20_Video_inpainting_DPO/experiments/dpo/exp46_h20_minimax_pseudosuccess_sft/sft_ladder/PSEUDO-SFT-A_lr3em5_step30/checkpoints/checkpoint-10`
- `/home/nvme01/H20_Video_inpainting_DPO/experiments/dpo/exp46_h20_minimax_pseudosuccess_sft/sft_ladder/PSEUDO-SFT-A_lr3em5_step30/checkpoints/checkpoint-20`
- `/home/nvme01/H20_Video_inpainting_DPO/experiments/dpo/exp46_h20_minimax_pseudosuccess_sft/sft_ladder/PSEUDO-SFT-A_lr3em5_step30/checkpoints/checkpoint-30`

## Largest Damage

The largest aggregate damage is full/outside PSNR on search and mask/boundary/outside on shadow. Worst full-PSNR rows include:

- `search` `pseudo_search_0003_BLENDER_MOUNTAIN002__exp45_pair003__s6__f3` dFull=-5.198774049614141 dMask=-1.9415188167248445 dBoundary=-2.5944632229391367 dOutside=-5.338959691411688
- `search` `pseudo_search_0004_BLENDER_MOUNTAIN002__exp45_pair004__s6__f5` dFull=-5.198774049614141 dMask=-1.9415188167248445 dBoundary=-2.5944632229391367 dOutside=-5.338959691411688
- `search` `pseudo_search_0005_BLENDER_MOUNTAIN002__exp45_pair005__s6__f7` dFull=-5.198774049614141 dMask=-1.9415188167248445 dBoundary=-2.5944632229391367 dOutside=-5.338959691411688
- `search` `pseudo_search_0000_BLENDER_MOUNTAIN002__exp45_pair000__s0__f3` dFull=-5.0737064032751675 dMask=-0.2224927705888362 dBoundary=-1.565414641647525 dOutside=-5.314273318060373
- `search` `pseudo_search_0001_BLENDER_MOUNTAIN002__exp45_pair001__s0__f5` dFull=-5.0737064032751675 dMask=-0.2224927705888362 dBoundary=-1.565414641647525 dOutside=-5.314273318060373

## Visual Artifacts Observed

Exp46 visual review reported search global tone/haze/outside drift and shadow visible global brightness/color drift with mask/boundary degradation. This makes metric-only failure unlikely; visual artifacts are consistent with the metric regression.

## Hypotheses To Test

A. pseudo-success teacher has global drift or outside/background mismatch.
B. manifest/path/frame/mask alignment bug.
C. region loss or mask polarity/weight map implementation bug.
D. runner did not actually train against pseudo-success target.
E. MiniMax flow target/data objective is poorly matched to this global pseudo-success SFT setup.

## Forbidden In Exp47

No training, no optimizer step, no 100-step, no DPO, no GT-only SFT, no PAI write/GPU, no Exp46 output deletion/overwrite, no shared trainer or metrics edits, no MiniMax official source edits, no positive/third-backbone/universal/final-SOTA claim.

## Files Read

| path | exists | bytes |
| --- | --- | ---: |
| `PRD/00_current_status.md` | True | 158974 |
| `PRD/01_experiment_matrix.md` | True | 90532 |
| `PRD/60_exp46_h20_minimax_exp45_pseudosuccess_sft.md` | True | 5565 |
| `experiment_registry/exp46_h20_minimax_pseudosuccess_sft/status.md` | True | 676 |
| `experiment_registry/exp46_h20_minimax_pseudosuccess_sft/results.tsv` | True | 1015 |
| `experiment_registry/exp46_h20_minimax_pseudosuccess_sft/metric_summary.md` | True | 670 |
| `experiment_registry/exp46_h20_minimax_pseudosuccess_sft/qualitative_summary.md` | True | 664 |
| `experiment_registry/exp46_h20_minimax_pseudosuccess_sft/paths.yaml` | True | 2475 |
| `experiment_registry/exp46_h20_minimax_pseudosuccess_sft/config.yaml` | True | 1165 |
| `reports/exp46_exp45_pseudosuccess_readback.md` | True | 5378 |
| `reports/exp46_exp45_file_mirror.md` | True | 1358 |
| `reports/exp46_exp45_manifest_rewrite_validation.md` | True | 1592 |
| `reports/exp46_bf16_pseudosuccess_preflight.md` | True | 2239 |
| `reports/exp46_exp45_step0_baseline.md` | True | 1413 |
| `reports/exp46_pseudosuccess_sft30.md` | True | 2817 |
| `reports/exp46_pseudosuccess_sft30_metrics.csv` | True | 141043 |
| `reports/exp46_pseudosuccess_sft30_visual_review.csv` | True | 114774 |
| `reports/exp46_pseudosuccess_sft30_diagnostics.csv` | True | 8363 |
| `reports/exp46_pseudosuccess_sft30_summary.json` | True | 11217 |
| `reports/exp46_minimax_pseudosuccess_decision.md` | True | 2322 |
| `reports/exp46_minimax_pseudosuccess_paper_positioning.md` | True | 1060 |
