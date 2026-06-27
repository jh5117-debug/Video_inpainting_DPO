# Exp30 Metric Summary

No Exp30 metric-producing inference or training has run yet.

Readback imported the following source-of-truth metric conclusions:

- Exp29 EffectErase official 81F baseline diagnostic:
  whole PSNR `27.416948`, LPIPS `0.085822`, mask PSNR `25.778614`,
  boundary PSNR `25.696018`, Ewarp `1.766501`.
- Exp29 MiniMax expanded data-yield: 128 attempts, 26 eligible unique scene
  groups, insufficient for train16+heldout16.
- Exp26 VideoPainter shadow-dev confirmed, but external DAVIS-derived
  validation was not confirmed.

## 2026-06-27 Three-Backbone Positioning

No new metrics were produced. This milestone locks interpretation only.

## 2026-06-27 VOR-OR Source Pool Audit

- Discovered extracted triplets: 192.
- Candidate scene groups after exclusions: 80.
- Source rows: 80.
- Reserve rows: 0.
- Source type counts: REAL 71, BLENDER 9.
- Mask bucket counts: small 22, medium 40, large 18.
- Source manifest SHA256:
  `58696bc504e79eec1342f00cbbb93d244b96d8311f128cf14156c3c6283cb595`.

Metric decision: the source pool is blocked before model generation because it
does not meet the 128 source + 128 reserve requirement and is not balanced.

## 2026-06-27 Continuation V2

No new model metrics were produced. The readback located the full VOR metadata
index with 57,751 rows and confirmed the previous 80-row pool was an extracted
cache subset.

## 2026-06-27 Full VOR Index Recovery

No model metrics were produced. Data-index metrics: 57,751 metadata rows,
57,750 valid triplets after one known-bad quarantine, 1,449 scene groups,
BLENDER 21,495 rows, REAL 36,256 rows.

## 2026-06-27 Source-Pool V2

No model metrics were produced. Source-pool metrics: primary 128 rows
(BLENDER 64, REAL 64), reserve 128 rows (BLENDER 20, REAL 108), reserve2 128
rows (REAL 128). Mask/effect labels are metadata-unavailable and remain
unknown.

## 2026-06-27 Smoke16 V2 Preregistration

- Metric status: not applicable yet; no model outputs or candidate metrics were
  generated.
- Locked rows: 16.
- Source type balance: BLENDER 8; REAL 8.
- Manifest SHA256:
  `1871f8e1aa23579425a87661040f91a992e934492aaa98c196f924ff21990ca3`.

## 2026-06-27 Smoke16 V2 Pre-Inference Repair

- Metric status: not applicable yet; no model outputs or candidate metrics were
  generated.
- Technical source repair: 3 rows replaced before inference.
- Final locked rows: 16.
- Final scene groups: 16.
- Final source type balance: BLENDER 8; REAL 8.
- Final manifest SHA256:
  `7e8cfd1b672b17b131476c9dd82804841d22d7450adf26301cf9ae8ff83f7f76`.

## 2026-06-27 Smoke16 V2 Final Materialization

- Model metric status: not generated yet.
- Materialization rows: 16/16.
- Failed rows: 0.
- Frames: 17.
- Resolution: 512 x 512.
- Source type balance: BLENDER 8; REAL 8.
- Materialized manifest SHA256:
  `72be9884335fef61926c307c66878fdc05dec85e9be4da28ab1547db98f8c26d`.
- All final rows passed the non-empty-mask guard.

## 2026-06-27 Multi-Model OR Smoke16 V2

- Non-EffectErase candidates: 32.
- Technical valid: 32.
- Usable total: 9.
- Classification totals: MEDIUM_HARD_ELIGIBLE 6; HARD_BUT_PLAUSIBLE 3;
  TRIVIAL_BAD 23.
- Controlled corruption: 16 technical-valid, 5 usable.
- MiniMax official: 16 technical-valid, 4 usable.
- Gate failure: controlled-corruption usable fallback is below the required
  6/16 threshold.
- Gate64 unlocked: no.

## 2026-06-27 Continuation V3 Readback

No new model metrics were produced. The readback confirmed the smoke16 v2
metric state: 32/32 non-EffectErase candidates were technical-valid, 9/32 were
usable, controlled corruption was 5/16 usable against a required 6/16, and
MiniMax was 4/16 usable. The next metric-producing task is smoke16 v3 only
after failure analysis, calibration planning, generator stack audit, and
preregistration.
