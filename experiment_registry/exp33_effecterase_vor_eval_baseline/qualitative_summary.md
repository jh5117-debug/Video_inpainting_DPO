# Exp33 Qualitative Summary

Status: `EXP33_EFFECTERASE_BASELINE_WEAK`

The 43-row visual review evidence pack has been generated for the completed
EffectErase VOR-Eval baseline. Representative weak/mixed/usable review sheets
were opened after metric generation.

Observed pattern:

- Weak cases can introduce large outside-region exposure/shadow drift.
- Mixed cases often remove the foreground object but retain visible reflection
  or background residuals.
- Usable cases show coherent removal, but still have texture errors in darker
  or fine-detail regions.

Opened representative sheets:

- `REAL_ENV900_00044_002_04`: weak; global darkening / shadow drift.
- `REAL_ENV900_00020_002_01`: mixed; person removed with reflection/background residual.
- `REAL_ENV900_00017_001_01`: usable; mostly coherent removal with residual texture error.

Evidence:

- `reports/exp33_effecterase_vor_eval_official81_visual_review.csv`
- `reports/exp33_effecterase_vor_eval_official81_visual_review.md`
- `reports/exp33_effecterase_vor_eval_official81_visual_audit_notes.md`

Conclusion: EffectErase provides technically valid baseline evidence, but the
held-out VOR-Eval official81 result is weak/mixed overall and is not a strong
baseline or promotion result.
