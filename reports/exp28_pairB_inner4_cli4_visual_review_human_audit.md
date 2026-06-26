# Exp28 Pair B Visual Audit Note

pair_id: `pairB_inner4_cli4`
comparison: `candidate_s2_2000` vs `fresh_s2_2000`
status: `VISUAL_ASSETS_GENERATED_PARTIAL_HUMAN_REVIEW_MIXED`

## Asset Generation

- Side-by-side MP4: 50/50 generated with `mp4_status=ok`.
- 16-frame temporal review PNG: 50/50 generated.
- Mask-bbox crop review PNG: 50/50 generated.
- Candidate temporal top3 heatmap PNG: 50/50 generated.
- Manifest: `visual_review_manifest.csv`.

## Human-Viewed Samples

Viewed representative samples selected from per-video PSNR/LPIPS/Ewarp deltas:

- `goat`: strongest negative PSNR delta. No global outside collapse visible in the 16-frame sheet, but differences are concentrated around the object/local boundary.
- `cows`: high LPIPS degradation. The crop review shows differences around the lower rail/object-adjacent structure, matching the perceptual risk.
- `lucia`: strongest positive PSNR delta. Differences are mostly on the person silhouette/nearby grass; no sampled global collapse.
- `surf`: strong positive PSNR but Ewarp risk. The temporal top3 heatmap shows motion concentrated on the sail and horizontal sea texture, so temporal risk remains plausible.

## Interpretation

The main Stage2 2000 metrics are encouraging, but visual evidence is mixed and only partially reviewed. The generated pack is suitable for full 50-video review, but this note does not certify all videos as visually clean.

Do not mark `INNER_RADIUS_POSITIVE` or `SCIENTIFIC_POSITIVE` from Pair B alone because:

- VFID and TC were skipped by optional-metric guards due missing model assets.
- Stage1-hybrid 2000 is metric-negative relative to its fresh control.
- Human visual review is partial, not 50/50 complete.
- At least one sampled positive case (`surf`) has temporal-risk evidence.
