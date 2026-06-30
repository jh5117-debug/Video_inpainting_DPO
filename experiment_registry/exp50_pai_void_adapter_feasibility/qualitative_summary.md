# Exp50 Qualitative Summary

Last updated: 2026-06-30T16:44:02+08:00

Gate8 evidence pages were manually inspected. Condition, V_bg, and quadmask are frame-aligned, object masks are present, and quadmask regions are visible. Several REAL examples have broad/speckled affected regions from real-scene texture differences; they are acceptable for Gate8 smoke only and are not quality evidence. VOID remains not positive.

## C2 Environment Repair (2026-06-30T15:25:36+08:00)

No visual output was generated. Official inference and adapter gates were intentionally skipped because the env gate did not pass.

## C3 Environment Relay Ingest (2026-06-30T16:44:02+08:00)

No visual output was generated. The isolated PAI VOID env is now `VOID_ENV_READY`; F0/F1/F2 remain not run. VOID remains not positive.

## F0 Component Load Smoke (2026-06-30T16:50:03+08:00)

No visual output was generated. F0 is metadata/config/header only and does not support any quality claim. VOID remains not positive.

## F1 Official Sample Inference (2026-06-30T17:02:21+08:00)

Codex inspected raw and tuple quick sheets. The official sample output is nonblank and aligned. This is technical smoke evidence only; VOID remains not positive until VOR and adapter gates pass.

## F2 VOR Gate8 Visual Review - 2026-06-30T17:24:15+08:00

- Status: `VOID_INFERENCE_SMOKE_PASS`
- Classification counts: {'MEDIUM_HARD_LOSER': 2, 'TOO_CLOSE': 2, 'VOID_OUTPUT_USABLE': 4}
- Codex opened all 8 Gate8 evidence sheets.
- Visual conclusion: no systematic outside collapse; 4 outputs usable, 2 bounded medium-hard loser candidates, 2 too-close samples.
- Scientific boundary: this is official inference smoke only and does not establish VOID as adapter evidence.

## G0 Adapter Micro Data - 2026-06-30T17:29:15+08:00

- Status: `VOID_ADAPTER_MICRO_DATA_READY`
- Split is scene-disjoint and VOR-Train-only.
- Same-model loser rows: 2 medium-hard VOID outputs.
- Controlled local-corruption loser rows: 6, explicitly marked as synthetic local losers and not used as hard-comp promotion.
- No VOID adapter evidence claim is made.

## G1 Zero-Gap / One-Step - 2026-06-30T17:37:17+08:00

- Status: `VOID_TRAINABLE_FORWARD_BLOCKED`
- Visual outputs: none, because no one-step output was generated.
- Interpretation: VOID is currently validated as official inference and data-signal source on Gate8, not as a proven adapter training path.

## H0 Preference-Wrapper Readback - 2026-06-30T22:59:39+08:00

- Status: `VOID_PREFERENCE_WRAPPER_REQUIRED_CONFIRMED`
- VOID role remains baseline / loser-generator candidate.
- No adapter evidence claim is made.

## H1 SFT Forward Parity - 2026-06-30T23:08:07+08:00

- Status: `VOID_SFT_FORWARD_PARITY_EXPLAINED`
- No visual output was generated; this is a forward/loss parity gate.
- No adapter evidence claim is made.

## H2 Preference Forward - 2026-06-30T23:15:48+08:00

- Status: `VOID_PREFERENCE_FORWARD_PASS`
- No visual output was generated; this is a preference-forward loss/gradient gate.
- No adapter evidence claim is made.

## H3 Zero-Gap V2 - 2026-06-30T23:17:43+08:00

- Status: `VOID_ZERO_GAP_PASS`
- No visual output was generated; zero-gap is a loss/gradient consistency gate.
- No adapter evidence claim is made.

## H4 One-Step V2 - 2026-06-30T23:24:57+08:00

- Status: `VOID_ONE_STEP_PARETO_MIXED`
- No video visual evidence was generated; one-step produced finite model forward diagnostics only.
- 10-step remains locked and no VOID adapter evidence claim is made.


## H6 Preference Wrapper Decision - 2026-06-30T23:30:58+08:00

- Status: `VOID_TRUE_ADAPTER_FEASIBLE_NEEDS_MICRO_TRAINING`
- Qualitative interpretation: VOID remains a strong inference baseline / loser-generator candidate and a preference-wrapper engineering candidate.
- No new video evidence was generated in H6.
- One-step lacks heldout video visual review, so 10-step remains locked and no adapter-positive claim is made.


## H4b-0 One-Step Evidence Readback - 2026-07-01T00:05:40+08:00

- Status: `VOID_ONE_STEP_EVIDENCE_READBACK_DONE`
- No visual output generated. H4b must generate and inspect heldout4 Step0/Step1 evidence before 10-step can unlock.


## H4b-1 One-Step Checkpoint Audit - 2026-07-01T00:08:01+08:00

- Status: `VOID_ONE_STEP_CHECKPOINT_READY`
- The one-step adapter artifact is present and matches the `proj_out` trainable subset. No visual evidence generated yet.


## H4b-2 One-Step Heldout Generation - 2026-07-01T00:09:39+08:00

- Status: `VOID_ONE_STEP_HELDOUT_GENERATION_BLOCKED`
- No visual evidence generated. H4b remains blocked by GPU availability; no 10-step unlock.


## H6-v2 One-Step Evidence Decision - 2026-07-01T00:11:20+08:00

- Status: `VOID_TRUE_ADAPTER_FEASIBLE_NEEDS_MICRO_TRAINING`
- No visual review was possible because H4b video generation was blocked by unavailable GPUs.
- VOID remains baseline / loser generator / adapter engineering candidate, not third adapter evidence.


## H4b-2 Resumed Heldout Generation - 2026-07-01T01:00:46+08:00

- Status: `VOID_ONE_STEP_HELDOUT_GENERATION_READY`
- Heldout4 evidence files were generated for all samples. Visual review remains pending for H4b-3.
