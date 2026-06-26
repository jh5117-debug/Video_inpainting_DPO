# Exp26 External Validation Preregistration

Status: `EXP26_EXTERNAL_VALIDATION_PREREGISTERED`

This milestone locks the external source rows, exact 49-frame materialization, mixed masks, prompt/seed rule, and raw/comp semantics before any Step0/Step50 output exists.

## Identity

- source manifest: `exp26_videopainter_dpo_v2/manifests/vp2_external_49f_validation_16_or_32.jsonl`
- source SHA256: `be118a7ce7d462bda6c339053d0c112994c8da7fab6cf00a4ee5dae87b628e5a`
- preregistered manifest: `exp26_videopainter_dpo_v2/manifests/vp2_external_validation_preregistered.jsonl`
- preregistered SHA256: `69ecd96d4b25da702229df2d45bf1343ad5e7ef5385cbd32d24ce61644e4bc2c`
- mask manifest: `exp26_videopainter_dpo_v2/manifests/vp2_external_validation_masks.jsonl`
- mask manifest SHA256: `f646792469f53a8122fe341be5988344ba7b32d33b3a53593d558e227aed138b`
- rows: `32`

## Locked Inference Protocol

- primary comparison: `Step50 - Step0`
- secondary checkpoints: `Step10, Step30`
- first-frame GT: `True`
- formal frames: `49`
- inference seed: `20260619`
- mask seed: `20260623`
- inference resolution: `720x480`
- steps/guidance/dtype: `20` / `6.0` / `bf16`

## Distribution

- source datasets: `{'DAVIS': 32}`
- mask profiles: `{'edge_touch_freeform': 8, 'ellipse_circle_subset': 5, 'irregular_freeform': 5, 'object_like_polygon': 5, 'soft_blob': 4, 'thin_structure_freeform': 5}`
- area buckets: `{'large': 8, 'medium': 16, 'small': 8}`
- motion buckets: `{'high': 8, 'low': 6, 'medium': 18}`

## Hard Constraints

- No model outputs were generated during preregistration.
- Shadow-dev/search-dev/primary32 were not changed.
- Step10/Step30 remain trajectory-only and cannot replace Step50.
- External validation cannot be used for tuning, source replacement, seed replacement, or checkpoint selection.
- Comp keeps winner outside the mask only; primary local metrics use frame1-48.

## Row Status

- `davis_bear`: OK, frames=49, mask_profile=edge_touch_freeform, area=small, motion=low
- `davis_bmx-bumps`: OK, frames=49, mask_profile=object_like_polygon, area=medium, motion=high
- `davis_boat`: OK, frames=49, mask_profile=soft_blob, area=medium, motion=low
- `davis_boxing-fisheye`: OK, frames=49, mask_profile=edge_touch_freeform, area=large, motion=medium
- `davis_breakdance-flare`: OK, frames=49, mask_profile=ellipse_circle_subset, area=small, motion=high
- `davis_bus`: OK, frames=49, mask_profile=thin_structure_freeform, area=medium, motion=medium
- `davis_car-turn`: OK, frames=49, mask_profile=irregular_freeform, area=large, motion=medium
- `davis_cat-girl`: OK, frames=49, mask_profile=object_like_polygon, area=medium, motion=medium
- `davis_classic-car`: OK, frames=49, mask_profile=soft_blob, area=small, motion=low
- `davis_color-run`: OK, frames=49, mask_profile=edge_touch_freeform, area=medium, motion=medium
- `davis_crossing`: OK, frames=49, mask_profile=ellipse_circle_subset, area=medium, motion=low
- `davis_dance-jump`: OK, frames=49, mask_profile=thin_structure_freeform, area=large, motion=medium
- `davis_disc-jockey`: OK, frames=49, mask_profile=irregular_freeform, area=small, motion=medium
- `davis_dog-gooses`: OK, frames=49, mask_profile=edge_touch_freeform, area=medium, motion=medium
- `davis_drift-turn`: OK, frames=49, mask_profile=soft_blob, area=large, motion=high
- `davis_drone`: OK, frames=49, mask_profile=edge_touch_freeform, area=medium, motion=medium
- `davis_elephant`: OK, frames=49, mask_profile=ellipse_circle_subset, area=small, motion=medium
- `davis_flamingo`: OK, frames=49, mask_profile=thin_structure_freeform, area=medium, motion=medium
- `davis_hike`: OK, frames=49, mask_profile=irregular_freeform, area=medium, motion=medium
- `davis_hockey`: OK, frames=49, mask_profile=object_like_polygon, area=large, motion=high
- `davis_horsejump-low`: OK, frames=49, mask_profile=soft_blob, area=small, motion=medium
- `davis_kid-football`: OK, frames=49, mask_profile=edge_touch_freeform, area=medium, motion=medium
- `davis_kite-walk`: OK, frames=49, mask_profile=ellipse_circle_subset, area=large, motion=medium
- `davis_koala`: OK, frames=49, mask_profile=thin_structure_freeform, area=medium, motion=low
- `davis_lady-running`: OK, frames=49, mask_profile=irregular_freeform, area=small, motion=medium
- `davis_mallard-water`: OK, frames=49, mask_profile=object_like_polygon, area=medium, motion=medium
- `davis_miami-surf`: OK, frames=49, mask_profile=edge_touch_freeform, area=medium, motion=high
- `davis_motocross-bumps`: OK, frames=49, mask_profile=edge_touch_freeform, area=large, motion=high
- `davis_paragliding`: OK, frames=49, mask_profile=ellipse_circle_subset, area=small, motion=medium
- `davis_rhino`: OK, frames=49, mask_profile=thin_structure_freeform, area=medium, motion=low
- `davis_scooter-board`: OK, frames=49, mask_profile=irregular_freeform, area=large, motion=high
- `davis_surf`: OK, frames=49, mask_profile=object_like_polygon, area=medium, motion=high
