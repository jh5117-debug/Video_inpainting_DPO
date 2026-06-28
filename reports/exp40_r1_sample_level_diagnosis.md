# Exp40 R1 Sample-Level Diagnosis

Status: `MINIMAX_R1_SIGNAL_AUDITED`.

No new training and no new GPU inference were launched. This milestone reads existing Exp38 R1 evidence and separates the available heldout13 SFT/DPO rescue outputs from the earlier train-overfit Exp37 R1 train32/heldout16 outputs.

## Availability Caveat

Exp38 SFT/DPO rescue R1 wrote heldout outputs only; no per-train R1 outputs exist under the R1 rescue output root. For train-side diagnosis, this report uses the existing Exp38 train-overfit audit of the Exp37 R1 checkpoint on train32/heldout16. Missing metrics from prior reports are explicitly marked `NOT_AVAILABLE`, not invented.

## Aggregate Findings

- exp38_sft_dpo_rescue_R1_heldout13/heldout13: rows `13`, full/mask/boundary/outside mean `+0.102167` / `+0.117230` / `-0.141510` / `-0.037262`.
- exp38_train_overfit_exp37_R1_existing_outputs/train: rows `32`, full/mask/boundary/outside mean `-0.586255` / `+0.152062` / `+0.069123` / `-0.895018`.
- exp38_train_overfit_exp37_R1_existing_outputs/heldout: rows `16`, full/mask/boundary/outside mean `+0.200826` / `+0.161946` / `-0.049755` / `+0.028198`.

## Diagnosis Counts

- `R1_WORSE`: `9`
- `R1_GOOD_BUT_BOUNDARY_BAD`: `9`
- `R1_GOOD_LOCAL_IMPROVEMENT_NOT_DECISIVE`: `10`
- `R1_TIE`: `4`
- `R1_FOGGING_OVER_ERASURE`: `1`
- `R1_GOOD_BUT_OUTSIDE_BAD`: `28`

## Decision

R1 failures are dominated by boundary/outside cost and local over-erasure/fogging risk; Exp40 recipes must add boundary/outside preservation and use PSNR-safe SFT before DPO.

Recipe narrowing:

- keep local Linear-DPO only as a later stage after PSNR-safe SFT;
- increase boundary/outside preservation weights;
- reduce fogging/over-erasure by reducing DPO pressure and corruption severity;
- expand data scale and scene-disjoint splits before any positive claim;
- require new Exp40 baseline/eval to compute affected PSNR, LPIPS, Ewarp, object/effect residuals, because Exp38 R1 reports did not include those fields.

Reports:

- `reports/exp40_r1_sample_level_diagnosis.csv`
- `reports/exp40_r1_visual_review.csv`
- `reports/exp40_r1_diagnosis_summary.json`
