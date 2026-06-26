# Exp29 Qualitative Summary

No Exp29 video evidence has been generated yet. Any future visual pass must be
based on per-sample video evidence, not a run-level montage alone.

MiniMax is the only audited model currently eligible for a future Exp29 visual
smoke. EffectErase cannot produce visual evidence until the official LoRA and
Wan base assets are installed.

## 2026-06-26 MiniMax Inference Smoke

- `davis_bear`: medium-hard candidate; object mostly removed, mild smoothing.
- `davis_bus`: trivial-bad; bus remains.
- `davis_mallard-water`: trivial-bad; duck remains with blue/black artifact.
- `davis_elephant`: trivial-bad; elephant remains with haze/smoothing.

MiniMax inference is technically runnable but not yet a robust OR baseline.
EffectErase visual review is blocked because official weights are missing.

## 2026-06-26 MiniMax 10-Step Heldout Review

- `davis_hockey`: Step10 and Step0 are visually tied; target remains.
- `davis_koala`: Step10 and Step0 are visually tied; fur texture remains
  smoothed/residual.

No heldout visual improvement was found. This is a technical adapter-plumbing
pass, not a third-backbone scientific positive.

## 2026-06-26 MiniMax 10-Step Failure Analysis

The heldout tie is now attributed to three interacting issues:

- the stable recovery optimizer was too conservative to move outputs;
- the previous train set was dominated by trivial-bad OR losers;
- the heldout set had only two rows and cannot support a quality-positive
  claim.

Next MiniMax quality work must start with per-video reviewed medium-hard
preferences, not with a longer version of the same 10-step recipe.

## 2026-06-26 MiniMax Preference Data Quality Gate

Codex opened all 24 temporal review pages copied under
`reports/exp29_minimax_micro_review_pages/`. The evidence shows a few plausible
human/indoor OR losers, but most sources are either target-preserving,
trivial-bad, too close, or technically invalid. Since usable candidates are
clustered in only 9 source groups, the MiniMax micro data gate is not ready for
training.

## 2026-06-26 EffectErase Weight Recovery

No EffectErase video evidence has been generated yet after weight recovery.
Recovered weights only unblock the next inference-smoke milestone. They do not
support `EFFECTERASE_OR_BASELINE_READY`, adapter feasibility, or scientific
positive language until actual videos and metrics are produced and reviewed.

## 2026-06-26 Architecture Family Audit

Scientific wording is constrained to model-specific backend adapters. The
project has cross-backbone evidence from DiffuEraser and VideoPainter, while
MiniMax and EffectErase remain candidates with different Wan/DiT flow-style
semantics. This is not a universal adapter result.

## 2026-06-26 MiniMax Preference Data Quality Gate

All 24 low-resolution review pages were opened and checked. MiniMax can produce
some finite local OR defects, but the useful cases cluster in a small number of
scenes. Many rejected cases retain the object/effect, are too close to the
winner, have empty/invalid masks, show temporal instability, or damage outside
regions.

Final qualitative decision: `MINIMAX_DATA_YIELD_INSUFFICIENT`. No visual
quality-positive adapter claim is supported, and no recipe or 30-step gate
should run from this candidate set.

## 2026-06-26 EffectErase Smoke Pre-Registration

EffectErase smoke inputs are locked before output review. The locked rows are
VOR diagnostics only and cannot be used to claim scientific adapter success.
Visual review remains pending until the smoke is actually run and the generated
videos are opened per sample.

## 2026-06-26 Continuation V3 Readback

No new visual evidence was generated. EffectErase visual review remains
pending. MiniMax remains blocked by data-yield, not by a visual-quality
positive result.
