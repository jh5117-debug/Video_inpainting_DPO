# Exp26 Gate16 Reclassification

Status: `GATE16_METRIC_GATE_PASS_VISUAL_REVIEW_PENDING`

Run root:

`/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp25_26_27_followup_20260624/exp26_gate16_reclassification_v2`

Scope:

- Existing Gate16 outputs only.
- The failed row `vp2_gate16_BLENDER_CON001_00742` was not replaced.
- Gate64 was not launched.
- No VideoPainter DPO training was launched.

Reclassification:

| class | count |
| --- | ---: |
| medium-hard | 15 |
| trivial-bad | 1 |
| technical-invalid | 0 |

Pre-registered gate check:

| criterion | value | status |
| --- | ---: | --- |
| technical valid >= 15/16 | 16/16 | pass |
| systematic failure = 0 | 0 | pass |
| trivial bad <= 2/16 | 1/16 | pass |
| medium-hard/hard-plausible >= 8/16 | 15/16 | pass |
| visual review complete | headless frame/crop only | pending |

Failed row:

- sample: `vp2_gate16_BLENDER_CON001_00742`
- classification: `trivial-bad`
- technical valid: true
- frames: 49
- mask area mean: `0.194941`
- edge-touch frames: `20`
- motion proxy: `358.444162`
- PSNR: `8.797221`
- mask PSNR: `1.349967`
- failure mode: `model_failure`
- reason: extreme mask-region failure.

Decision:

Metric gate would pass with one rejected true model failure, but the hard visual-review rule is not satisfied because this run used headless frame/crop audit rather than interactive mp4 playback. Therefore:

- `GATE16_PASSED_WITH_REJECTION` is not set.
- `Gate64` is not launched.
- `DPO training` is not launched.

Outputs:

- CSV: `reports/exp26_gate16_reclassification.csv`
- Summary JSON: `reports/exp26_gate16_reclassification_summary.json`
- PAI audit evidence: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp25_26_27_followup_20260624/exp26_gate16_reclassification_v2`
