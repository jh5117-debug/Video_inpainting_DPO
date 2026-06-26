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

## 2026-06-26 EffectErase Smoke Input Materialization

No EffectErase output video was generated. The preregistered smoke is blocked
before inference because one locked row has an empty task mask. The row was not
silently replaced, preserving preregistration integrity.

## 2026-06-26 EffectErase Command Dry-Run

No EffectErase output video was generated. The command path and environment are
now ready, but the official smoke remains blocked by the locked empty-mask row.
No OR baseline quality judgment is supported yet.

## 2026-06-26 MiniMax Expanded Source-Pool Plan

No new MiniMax video evidence was generated. The expanded data-yield first pass
is blocked before inference because the current audited source inventory is too
small after excluding the previous 32 rows. MiniMax remains plumbing-positive
but data-yield-limited.

## 2026-06-26 Continuation V4 Readback

No new video evidence was generated. EffectErase remains pending smoke v2
preregistration with a non-empty replacement mask row. MiniMax remains pending
a full-VOR source audit before any expanded candidate generation. The accepted
language remains cross-backbone evidence from DiffuEraser plus VideoPainter;
universal-adapter and final-SOTA language remains forbidden.

## 2026-06-26 EffectErase Smoke V2 Pre-Registration

Codex inspected 6/6 generated preview sheets. The replacement row
`REAL_ENV248_00118_005_03` shows a toy car in the condition, a clean winner,
and a non-empty small mask following the target. The retained rows show
non-empty masks and visible condition/winner differences. This supports
input-valid preregistration only; EffectErase output quality remains unknown
until official inference outputs are generated and reviewed.

## 2026-06-26 EffectErase Smoke V2 Input Materialization

No EffectErase model output has been generated yet. The v2 input videos now
decode correctly and preserve non-empty masks, so the next allowed step is the
official EffectErase inference smoke. Baseline readiness and visual quality
remain pending actual model outputs.

## 2026-06-26 EffectErase Official Inference Smoke V2

No EffectErase output video was produced. The official model assets loaded
after the PYTHONPATH fix, but the 17-frame diagnostic protocol is incompatible
with the official pipeline call as currently wired because `num_frames` is not
forwarded into `WanRemovePipeline`. EffectErase remains blocked before OR
baseline visual judgment.

## 2026-06-26 MiniMax Full-VOR Source Audit

No MiniMax video evidence was generated by this milestone. The audit fixes the
prior source-pool-size blocker by selecting 192 scene-disjoint full-VOR source
groups from the existing Exp25 metadata index, balanced REAL/BLENDER = 96/96.

This is not a medium-hard quality pass. Mask non-emptiness, 17-frame decoding,
defect taxonomy, and visual usefulness must be checked during the next
first-pass generation review before any train16/heldout16 split, recipe gate,
30-step micro, or third-backbone-positive language is allowed.

## 2026-06-26 MiniMax Expanded Data-Yield Review V2

Codex opened all per-sample evidence pages generated for the expanded MiniMax
data-yield gate: 24 seed-A pages and 8 conditional seed-B near-miss pages. The
visual pattern matches the CSV classification: a limited number of useful
medium-hard local defects, but many trivial-bad outputs, several too-close
outputs, and technical-invalid rows. Seed B rescued only one extra
medium-hard sample and did not change the scientific decision.

Final qualitative decision: `MINIMAX_EXPANDED_DATA_YIELD_INSUFFICIENT`. The
current expanded source pool cannot support a scene-disjoint train16+heldout16
MiniMax micro gate, so MiniMax remains plumbing-positive but data-yield-limited.

## 2026-06-27 Continuation V5 Readback

No new visual evidence was generated. The readback confirms that EffectErase
must move from the blocked 17-frame diagnostic smoke to a preregistered official
81-frame smoke before any OR baseline visual claim. MiniMax remains short of the
scene-disjoint train16+heldout16 requirement and may only proceed through
top-up data-yield review before any 10-step recipe gate.
