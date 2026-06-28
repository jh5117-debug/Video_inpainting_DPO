# Exp33 EffectErase VOR-Eval Baseline Status

Current status: `EFFECTERASE_VOREVAL_BASELINE_WEAK_OR_FAILED`

- branch: `research/exp33-effecterase-vor-eval-baseline-20260627`
- base: `origin/research/exp29-minimax-effecterase-adapter-feasibility-20260626`
- worktree: `/home/hj/H20_Video_inpainting_DPO_exp33_effecterase_eval`
- training: forbidden.
- adapter / DPO / loser mining: forbidden.
- VOR-Eval official81 compatibility: 43/43 ready, 0 rejected.
- input materialization: 43/43 rows ready, 129 condition/winner/mask MP4 files.
- command validation: `EXP33_VOREVAL_OFFICIAL81_COMMAND_READY`.
- inference: `EXP33_VOREVAL_EFFECTERASE_INFERENCE_COMPLETED`, 43/43 rows, failed rows `[]`.
- metrics and visual review: `EFFECTERASE_VOREVAL_BASELINE_WEAK_OR_FAILED`.
- full visual contact-sheet review: `43/43` review sheets and `43/43` crop
  sheets opened under
  `reports/exp33_effecterase_voreval_review_contact_sheets/`.

Metric summary:

- full PSNR mean: `21.9229`
- full SSIM mean: `0.7349`
- mask PSNR mean: `19.3942`
- mask SSIM mean: `0.5889`
- boundary PSNR mean: `20.0981`
- outside L1 mean: `16.4051`
- Ewarp proxy mean: `6.4370`
- LPIPS: unavailable in this milestone; no proxy LPIPS is reported as real.

Classification:

- `BASELINE_USABLE`: 9
- `BASELINE_MIXED`: 17
- `BASELINE_WEAK`: 17

Evidence:

- `reports/exp33_effecterase_voreval_output_integrity.md`
- `reports/exp33_effecterase_voreval_metrics.md`
- `reports/exp33_effecterase_voreval_visual_review.md`
- `reports/exp33_effecterase_voreval_full_contact_sheet_review.md`
- `reports/exp33_effecterase_voreval_final_baseline.md`
- `reports/exp33_effecterase_vor_eval_official81_metrics_summary.md`
- `reports/exp33_effecterase_vor_eval_official81_metrics.csv`
- `reports/exp33_effecterase_vor_eval_official81_visual_review.csv`
- `reports/exp33_effecterase_vor_eval_official81_visual_audit_notes.md`
- `reports/exp33_effecterase_vor_eval_official81_final_report.md`

Final-status family: `EFFECTERASE_BASELINE_ONLY_FOR_NOW`

No strong baseline, scientific positive, final SOTA, top-conference novelty, or
universal-adapter claim is made from this result.
