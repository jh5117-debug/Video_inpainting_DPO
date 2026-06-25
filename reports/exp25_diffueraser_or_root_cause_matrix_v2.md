# Exp25 DiffuEraser OR Root-Cause Matrix v2

- run_root: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp25_vor_or_preference_data/diffueraser_root_cause_matrix_v2_20260625_131650`
- source_manifest: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp25_vor_or_preference_data/root_cause_matrix_20260625/root_cause_sample12_manifest.jsonl`
- metric_pair_manifest: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp25_vor_or_preference_data/diffueraser_root_cause_matrix_v2_20260625_131650/review_v2/root_cause_metric_pairs.csv`
- metric_backend: `tools/run_inpainting_metric_eval.py` -> `inference/metrics.py`
- final_decision: `DIFFUSERASER_NATIVE_OR_STACK_USABLE`

## Stack Summary

| stack | ok | medium-hard | hard-plausible | trivial-bad | too-close | technical-invalid | blocked | mean mask PSNR | mean outside PSNR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DE-A_sft_canonical_raw6_d0_propainter | 12 | 8 | 4 | 0 | 0 | 0 | 0 | 20.6544 | 29.1165 |
| DE-B_sft_raw6_d8_propainter | 12 | 9 | 3 | 0 | 0 | 0 | 0 | 21.9773 | 28.8082 |
| DE-C_sft_official_pcm2_d8_propainter | 0 | 0 | 0 | 0 | 0 | 1 | 0 | nan | nan |
| DE-D_official_core_official_pcm2_d8_propainter | 0 | 0 | 0 | 0 | 0 | 0 | 1 | nan | nan |
| DE-E_official_core_canonical_raw6_d0_propainter | 0 | 0 | 0 | 0 | 0 | 0 | 1 | nan | nan |
| DE-F_sft_native_high_quality_no_prior | 0 | 0 | 0 | 0 | 0 | 0 | 1 | nan | nan |
| DE-G_official_pcm2_alias | 0 | 0 | 0 | 0 | 0 | 0 | 1 | nan | nan |

## Root-Cause Answers

- 6-step impact: DE-A is the canonical 6-step raw baseline; quality is judged from the per-sample review rows.
- Dilation impact: compare DE-B against DE-A; no model regeneration was hidden in this review.
- ProPainter prior impact: not isolated in this run because verified Exp25 wrapper only exposes the ProPainter-prior path.
- PCM impact: DE-C failed before inference because the active UNetMotionModel lacks `load_lora_adapter`; this is a technical stack incompatibility, not a quality sample.
- Official core vs SFT: official core was not evaluated because no strict-load-identifiable official core checkpoint path was available.
- BR->OR domain shift: can only be inferred from SFT DE-A/DE-B quality; official-core comparison remains blocked.

Hard-comp outputs were not used as loser quality evidence.
