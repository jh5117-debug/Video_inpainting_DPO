# Exp40 Qualitative Summary

No Exp40 videos have been generated yet.

Readback imported Exp38 visual conclusion:

- R1: `0/13` clear visual better, `9/13` worse/tradeoff, `4/13` tie or local
  numeric gain.
- R1 failure mode: over-erasure, fogging/soft local fill, boundary cost, and
  outside/background drift.
- R2/R3 failure mode: stronger harmful drift and worse boundary/outside
  aggregates.

Any Exp40 pass requires raw-output videos, temporal strips, crops, per-video
metrics, and Codex visual review. No metric-only promotion is allowed.

## 2026-06-28 R1 Sample-Level Diagnosis

R1 sample-level review preserves the prior visual evidence:

- no clear heldout13 visual win from Exp38 SFT/DPO R1;
- train-overfit Exp37 R1 rows move pixels but often damage outside/background
  tone;
- the most promising rows are still classified as local numeric gains rather
  than clear visual improvements;
- fogging/over-erasure is a real blocker on the high-delta row
  `REAL_ENV104_00001_001_01`.

Conclusion: a recipe that only increases DPO pressure is likely to amplify the
wrong behavior. Exp40 should first build cleaner local corruption data and run
PSNR-safe SFT.

## 2026-06-29 LocalDPO v3 Pool Visual Review

Codex opened all 19 temporal-strip review pages covering the selected
`train64/search24/shadow24` rows.

Observed:

- local object/affected-region corruptions are visible;
- no global black/purple collapse was seen;
- no run-wide fogging or full-frame degradation was seen;
- far outside regions are visually stable in the review pages;
- boundary defects are mostly mild and bounded by metric gates.

This review only supports using the pool for the next Step0/SFT diagnostics. It
does not promote MiniMax model quality and does not unlock any adapter-positive
claim.
