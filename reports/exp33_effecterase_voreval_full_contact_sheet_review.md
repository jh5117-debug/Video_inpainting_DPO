# Exp33 EffectErase VOR-Eval Full Contact-Sheet Review

Date: 2026-06-28

Status: `EFFECTERASE_VOREVAL_ALL43_CONTACT_SHEETS_REVIEWED`

Reviewed material:

- `43/43` review sheets, grouped in
  `reports/exp33_effecterase_voreval_review_contact_sheets/exp33_review_sheets_all43_page*.jpg`
- `43/43` crop sheets, grouped in
  `reports/exp33_effecterase_voreval_review_contact_sheets/exp33_crop_sheets_all43_page*.jpg`
- source cache:
  `reports/exp33_effecterase_vor_eval_official81_visual_review_assets/`
  remains local/untracked because it is a 203M raw image cache.

The full contact-sheet review does not change the baseline decision:
`EFFECTERASE_VOREVAL_BASELINE_WEAK_OR_FAILED`.

## Observed Pattern

- Some rows remove the foreground object/effect cleanly enough to be usable as
  baseline examples.
- Water, reflection, lamp/light, and high-gloss rows frequently retain residual
  effect structure or create strong exposure/color drift.
- Several person/object rows remove the foreground object but leave local
  texture breaks, shadows, reflections, or edge seams.
- Outside preservation is inconsistent; weak rows show visible global or
  outside-region brightness/color shifts.
- Temporal evidence is technically decodable and complete, but flicker and
  residual motion artifacts are visible in weak/mixed rows.

## Decision

The all-43 visual pass supports the metric classification:

- `BASELINE_USABLE`: 9
- `BASELINE_MIXED`: 17
- `BASELINE_WEAK`: 17

EffectErase is a technically valid held-out VOR-Eval baseline, but it is not a
strong baseline, not adapter evidence, and not a DPO loser source in this
prompt.
