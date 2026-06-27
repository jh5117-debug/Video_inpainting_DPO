# Exp30 New Generators Smoke2

Status: `NEW_GENERATORS_SMOKE2_PARTIAL_PASS`

Generated raw/no-comp outputs for ProPainter and DiffuEraser no-PCM on two locked Smoke16 rows. Metrics and review pages are generated; Codex opened all review sheets and recorded final visual classifications.

Output root: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp30_vor_or_multimodel_minimax/smoke2_verified_generators_20260627`

Metrics: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp30_vor_or_multimodel_minimax/smoke2_verified_generators_20260627/reports/exp30_new_generators_smoke2_metrics.csv`

Visual review CSV: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp30_vor_or_multimodel_minimax/smoke2_verified_generators_20260627/reports/exp30_new_generators_smoke2_visual_review.csv`

## Codex Visual Review

Codex opened 4/4 review sheets:

- ProPainter `BLENDER_FOREST006_00001`: `TOO_CLOSE`.
- DiffuEraser `BLENDER_FOREST006_00001`: `TOO_CLOSE`.
- ProPainter `BLENDER_FOREST007_00001`: `HARD_BUT_PLAUSIBLE`.
- DiffuEraser `BLENDER_FOREST007_00001`: `MEDIUM_HARD_ELIGIBLE`.

Technical result:

- ProPainter: 2/2 generated, 17/17 frames each.
- DiffuEraser no-PCM: 2/2 generated, 17/17 frames each.

Quality result:

- Usable: 2/4.
- Too-close: 2/4.
- Trivial-bad by final visual review: 0/4.

The smoke validates the Exp30 generator wiring and supports including
ProPainter/DiffuEraser in Smoke16 v3, but it is not a data-ready gate and does
not unlock Gate64, Smoke32, adapter training, or any scientific claim.
