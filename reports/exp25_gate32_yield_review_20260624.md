# Exp25 Gate32 Yield Review - 2026-06-24

Status: `GATE32_YIELD_REVIEW_COMPLETED`

Controller run:

`/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/overnight_exp25_26_27_20260624_three_lane_retry2`

Inputs:

- Manifest: `/mnt/nas/hj/H20_Video_inpainting_DPO/data/external/effecterase_vor/materialized/gate32_canonical_d0_24f/gate32_materialized.jsonl`
- DiffuEraser raw OR candidates: `/mnt/nas/hj/H20_Video_inpainting_DPO/data/external/effecterase_vor/preference_candidates/gate32_canonical_raw6_d0`
- Protocol: raw OR loser, no hard comp, 24 frames, canonical d0 mask.

Results:

| bucket | count |
| --- | ---: |
| medium-hard | 11 |
| too-close | 0 |
| trivial-bad | 21 |

Aggregate diagnostics:

| metric | mean | min | max |
| --- | ---: | ---: | ---: |
| full PSNR | 23.2438 | 12.9490 | 32.5300 |
| mask PSNR | 17.2707 | 7.0612 | 27.5980 |
| outside PSNR | 25.1892 | 15.7134 | 36.3461 |
| black frame ratio | 0.0000 | 0.0000 | 0.0000 |

Visual review:

- All 32 generated contact sheets were reviewed via the run-level index image.
- The 21 `trivial-bad` samples are dominated by obvious black/purple/raw artifact outputs or severe region mismatch.
- The 11 `medium-hard` samples retain enough structure to be useful for preference-yield analysis.
- No `too-close` samples were found, so seed2 supplementation was not launched.

Outputs:

- Metrics CSV: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/overnight_exp25_26_27_20260624_three_lane_retry2/exp25_gate32_yield_review/gate32_yield_metrics.csv`
- Summary JSON: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/overnight_exp25_26_27_20260624_three_lane_retry2/exp25_gate32_yield_review/gate32_yield_summary.json`
- Contact sheets: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/overnight_exp25_26_27_20260624_three_lane_retry2/exp25_gate32_yield_review/contact_sheets`
- Visual index: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/overnight_exp25_26_27_20260624_three_lane_retry2/exp25_contact_sheet_index.jpg`

Decision:

Gate32 yield calibration is complete. Do not expand to Gate128 or training from this result alone; the medium-hard yield is only `11/32`, and full VOR preference construction still needs a stronger candidate-quality policy.
