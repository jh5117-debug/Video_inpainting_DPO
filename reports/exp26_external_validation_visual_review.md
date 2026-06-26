# Exp26 External Validation Visual Review

Status: `EXP26_EXTERNAL_VIDEO_REVIEW_COMPLETE`

Conclusion: the external DAVIS-derived validation videos do **not** confirm the shadow-dev Step50 improvement. The fixed Step50 checkpoint is occasionally better than Step0, but most external rows show Step50-specific local artifacts in the masked/affected region. This visual result agrees with the preregistered external metric failure and does not authorize checkpoint reselection, new training, or a 100-step continuation.

## Scope

- Split: preregistered external 32-row exact-49F DAVIS-derived validation split.
- Primary comparison: fixed Step50 vs fixed Step0 from the same 50-step trajectory.
- Review inputs: anonymous A/B pages, informed Step0/Step50 pages, crop pages, 16-frame temporal strips, start/middle/end frames, local crop sheets, and previously generated leakage/metric reports.
- Full PAI output root: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp26_videopainter_dpo_v2/postconfirmation_20260626/external_validation/`
- Git-friendly review assets: `reports/exp26_external_validation_visual_review_assets/`

## Counts

| Class | Count |
| --- | ---: |
| STEP50_CLEARLY_BETTER | 0 |
| STEP50_SLIGHTLY_BETTER | 3 |
| TIE | 5 |
| STEP0_SLIGHTLY_BETTER | 7 |
| STEP0_CLEARLY_BETTER | 17 |
| Step50 better total | 3 |
| Step0 better total | 24 |
| Step50 new artifact rows | 29 |

## Main Failure Modes

- Dark/green masked-region blobs in animals, vehicles, people, water, and foliage.
- Color mismatch on water/grass/foliage scenes.
- Texture break and local smear in high-motion scenes.
- Thin-structure boundary tinting or line artifacts.
- Temporal instability in mask-region residuals; no evidence of first-frame or frame-order failure.

No unexpected GT/winner leakage was observed in the earlier leakage audit. Comp outside-winner copy remains expected by protocol; the visible failures are concentrated in the generated local region.

## Best Step50 Cases

- `davis_bus`: STEP50_SLIGHTLY_BETTER; strict mask PSNR delta `3.906294`; Step50 slightly cleans the bus-side residual without a new obvious artifact.
- `davis_hockey`: STEP50_SLIGHTLY_BETTER; strict mask PSNR delta `2.291667`; Step50 reduces a Step0 ghost/patch, though a local dark residual remains.
- `davis_boxing-fisheye`: STEP50_SLIGHTLY_BETTER; strict mask PSNR delta `1.734928`; Step50 reduces the beige Step0 patch and looks locally more plausible.
- `davis_disc-jockey`: TIE; strict mask PSNR delta `16.897785`; Both are visibly wrong; neither checkpoint is clearly usable.
- `davis_flamingo`: TIE; strict mask PSNR delta `7.470058`; Both leave thin-line artifacts; Step50 is not reliably better.

## Worst Visual Cases

- `davis_kid-football`: STEP0_CLEARLY_BETTER; strict mask PSNR delta `-11.733536`; Step50 creates an obvious green patch in a human-sports case.
- `davis_motocross-bumps`: STEP0_CLEARLY_BETTER; strict mask PSNR delta `-10.601354`; Step50 damages high-motion motocross content.
- `davis_dog-gooses`: STEP0_CLEARLY_BETTER; strict mask PSNR delta `-9.285406`; Step50 creates a strong dark/green local blob around moving animals.
- `davis_color-run`: STEP0_CLEARLY_BETTER; strict mask PSNR delta `-8.060196`; Step50 adds a green cloth-like patch in a high-motion colorful scene.
- `davis_paragliding`: STEP0_CLEARLY_BETTER; strict mask PSNR delta `-7.380814`; Step50 introduces a green blob over distant landscape.

## Worst Strict-Mask Metric Cases

- `davis_kid-football`: strict mask PSNR delta `-11.733536`, LPIPS delta `0.048141`, class `STEP0_CLEARLY_BETTER`.
- `davis_motocross-bumps`: strict mask PSNR delta `-10.601354`, LPIPS delta `0.048935`, class `STEP0_CLEARLY_BETTER`.
- `davis_dog-gooses`: strict mask PSNR delta `-9.285406`, LPIPS delta `0.018375`, class `STEP0_CLEARLY_BETTER`.
- `davis_color-run`: strict mask PSNR delta `-8.060196`, LPIPS delta `0.012025`, class `STEP0_CLEARLY_BETTER`.
- `davis_paragliding`: strict mask PSNR delta `-7.380814`, LPIPS delta `0.004459`, class `STEP0_CLEARLY_BETTER`.

## Decision

This milestone preserves the prior metric status `EXP26_EXTERNAL_VALIDATION_NOT_CONFIRMED`. External video review is complete, but it strengthens the negative external conclusion rather than rescuing the result. The confirmed statement remains: VideoPainter Step50 is strong on search-dev and independent shadow-dev in the VOR-BG distribution, but this DAVIS-derived external split exposes poor generalization and visible local artifacts.

Forbidden follow-ups remain unchanged: no Step30 replacement based on this external split, no 100-step run, no retraining, and no universal-adapter claim.
