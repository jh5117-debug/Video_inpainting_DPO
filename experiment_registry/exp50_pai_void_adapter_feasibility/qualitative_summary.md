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
