# Exp33 EffectErase VOR-Eval Final Baseline

Final status: `EFFECTERASE_VOREVAL_BASELINE_WEAK_OR_FAILED`

EffectErase completed held-out VOR-Eval baseline evaluation on all 43 rows. The run is technically valid, but quality is weak/mixed overall, so it is not a strong OR baseline and not adapter evidence.

## Metrics

- technical-valid outputs: `43/43`
- full PSNR mean: `21.9229`
- full SSIM mean: `0.7349`
- mask PSNR mean: `19.3942`
- mask SSIM mean: `0.5889`
- boundary PSNR mean: `20.0981`
- outside L1 mean: `16.4051`
- Ewarp proxy mean: `6.4370`
- LPIPS: unavailable; no proxy LPIPS is reported as real LPIPS.

## Visual Review

- `BASELINE_USABLE`: 9
- `BASELINE_MIXED`: 17
- `BASELINE_WEAK`: 17

Opened all `43/43` review sheets and all `43/43` crop sheets through contact
sheet pages under `reports/exp33_effecterase_voreval_review_contact_sheets/`.
The observed pattern is outside exposure/shadow drift in weak cases,
reflection/background residuals in mixed cases, and residual fine texture errors
even in usable cases.

## Paper Role

- Use as held-out EffectErase baseline evidence only.
- Do not use as DPO loser source in this prompt.
- Do not call it a strong baseline, adapter evidence, universal adapter evidence, final SOTA, or top-conference novelty confirmation.
