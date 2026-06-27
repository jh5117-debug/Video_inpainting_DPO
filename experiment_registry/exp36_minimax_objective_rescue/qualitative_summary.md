# Exp36 Qualitative Summary

No Exp36 videos have been generated yet.

Prior qualitative source-of-truth:

- Exp35 reviewed `48/48` R1/R2/R3 Step0-vs-Step10 heldout strips.
- Visual better rows: `0`.
- Collapse / black-purple artifact rows: `0`.
- Failure mode: mostly tie or slight degradation, not catastrophic collapse.

## 2026-06-27 No-Change Forensic Audit

No new videos were generated. The audit preserves the previous visual
source-of-truth: MiniMax movement is real but not quality-positive. Current
failure is not black/purple collapse; it is tie/slight degradation and local
metric harm.

## 2026-06-27 Inference Sensitivity Test

Codex opened `4/4` Exp36 sensitivity comparison strips:

- Identity controls: visually identical `4/4`.
- Perturbed outputs: subtle nonzero response `4/4`.
- Collapse / black-purple / new artifact: `0/4`.

This confirms inference sensitivity, not heldout quality improvement.

## 2026-06-27 Trainable Scope Audit

No videos were generated or reviewed. The qualitative source-of-truth remains
unchanged: MiniMax has measurable weight sensitivity but no heldout
quality-positive recipe yet. S1 scope readiness only prepares the next
positive-control test.

## 2026-06-27 Winner-SFT Positive-Control

Codex opened `24/24` Step0-vs-Step10 heldout strips from the fixed winner-SFT positive-control run.

- `CLEARLY_WORSE_NEW_ARTIFACT`: 4 rows, all S0 LR `1e-4`.
- `TIE_METRIC_WORSE`: 6 rows.
- `TIE_METRIC_MIXED`: 6 rows.
- `TIE_NO_VISIBLE_CHANGE`: 8 rows.
- Visual better: 0 rows.

S1 LoRA is stable but visually tied; S0 high LR proves output sensitivity by producing artifacts. There is no heldout quality-positive positive-control.

## 2026-06-27 Paper Positioning

Final qualitative stance: MiniMax has `0` visually better heldout rows across Exp36 winner-SFT and prior Exp30/Exp35 preference attempts. The safe paper language is two-backbone adapter evidence plus MiniMax plumbing-only candidate evidence.
