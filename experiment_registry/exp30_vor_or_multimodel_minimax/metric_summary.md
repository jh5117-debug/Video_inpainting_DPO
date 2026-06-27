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
