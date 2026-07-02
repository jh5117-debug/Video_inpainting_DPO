# Exp60B VPData Subset Plan

Status: `EXP60B_VPDATA_SUBSET_PLAN_READY`

No raw video was downloaded in this step.

## Selection

- Seed: `20260702`
- Source filter: `pexels_only`
- Train split: official `pexels_videovo_train_dataset.csv`
- Test split: official `pexels_videovo_test_dataset.csv`
- Train selected: 1,000 unique Pexels source videos
- Test selected: 100 unique Pexels source videos
- Train/test `source_video_id` overlap: 0

The first full 1000/100 plan attempt found one train/test overlap, so the
selector now locks the train set first and excludes its source ids before
sampling the test set. This is deterministic and not manual cherry-picking.

## Why Pexels-Only First

VPData contains Pexels and VideoVo rows. Pexels raw videos are row-level URLs
through `pexels.csv`, so they can be downloaded without cloning VPData. VideoVo
raw videos are grouped in multi-GB zip shards. To respect the no-full-download
boundary, Exp60B starts with the Pexels-only subset and records VideoVo as a
future shard-handling extension.

## Files

- `manifests/exp60b_vpdata_train1000_sources_h20.jsonl`
- `manifests/exp60b_vpdata_test100_sources_h20.jsonl`
- `reports/exp60b_vpdata_subset_plan.csv`
- `reports/exp60b_vpdata_subset_plan_summary.json`

## Download Guard

The downloader defaults to plan-only mode. Raw video download requires the
explicit `--download` flag and refuses requests above train1000/test100.

