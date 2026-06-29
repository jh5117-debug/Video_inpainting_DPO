# Exp46 Exp45 Manifest Rewrite Validation

Status: EXP45_H20_MANIFESTS_READY

## Scope

Exp45 formal Stage2 manifests were rewritten for H20-local paths under `/home/nvme01/H20_Video_inpainting_DPO/data/h20_mirror/minimax/pai_abs`. Original PAI paths are preserved in `pai_*` fields only. No training, no optimizer step, no PAI write/GPU, no GT-only SFT, and no DPO occurred.

## Rewritten Manifests

- Pseudo-success: `manifests/exp45_h20_pseudosuccess_train.jsonl`, `manifests/exp45_h20_pseudosuccess_search.jsonl`, `manifests/exp45_h20_pseudosuccess_shadow.jsonl`
- GT distillation: `manifests/exp45_h20_gt_distill_train.jsonl`, `manifests/exp45_h20_gt_distill_search.jsonl`, `manifests/exp45_h20_gt_distill_shadow.jsonl`
- Preference: `manifests/exp45_h20_preference_train.jsonl`, `manifests/exp45_h20_preference_search.jsonl`, `manifests/exp45_h20_preference_shadow.jsonl`
- Same-source traceability: `manifests/exp45_h20_same_source_pairs_train.jsonl`, `manifests/exp45_h20_same_source_pairs_search.jsonl`, `manifests/exp45_h20_same_source_pairs_shadow.jsonl`

## Counts

- Pseudo-success split: `64/24/24`
- GT distillation split: `64/24/24`
- Preference split: `64/24/24`

## Validation

- Validation rows: `336`
- Failed rows: `0`
- Scene overlap OK: `True`
- Pseudo-success split count OK: `True`
- VOR-Eval rows: `0`
- Hard-comp rows: `0`
- Bad-noise states required by manifest rows: `0`
- Bad-noise unmatched rows: `0`
- Rows using mp4 fallback for pseudo/failure video decode: `128`

## Decision

Exp45 H20 manifests are ready for BF16/environment preflight and Step0 baseline.
