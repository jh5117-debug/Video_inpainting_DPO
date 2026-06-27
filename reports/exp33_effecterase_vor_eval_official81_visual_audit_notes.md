# Exp33 EffectErase VOR-Eval Official81 Visual Audit Notes

Status: `EXP33_EFFECTERASE_BASELINE_WEAK`

The full 43-row visual review evidence pack was generated under:

`/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp33_effecterase_vor_eval_baseline/vor_eval_official81_compat_20260627_134945/reports/exp33_effecterase_vor_eval_official81_visual_review_assets`

Representative review sheets were opened after metric generation:

| sample | metric class | visual observation |
| --- | --- | --- |
| `REAL_ENV900_00044_002_04` | `BASELINE_WEAK` | Raw output shows global darkening / shadow drift outside the mask, consistent with high outside L1. |
| `REAL_ENV900_00020_002_01` | `BASELINE_MIXED` | Person is removed, but mirror/reflection/background residuals remain visible. |
| `REAL_ENV900_00017_001_01` | `BASELINE_USABLE` | Subject removal is mostly coherent, but tree/dark-region texture errors remain. |

Conclusion: the 43-row VOR-Eval baseline is technically valid and useful as
paper evidence, but the visual evidence supports a weak/mixed baseline label.
No strong baseline, scientific positive, final SOTA, top-conference novelty, or
universal-adapter claim is made.
