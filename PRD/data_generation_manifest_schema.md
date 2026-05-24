# Data Generation Manifest Schema

Every generated loser manifest is JSONL. The first-generation VideoDPO
generated-loser pipeline writes candidate manifests first, then selected
primary/secondary manifests.

## Candidate Manifest

`candidates_all.jsonl` keeps every generated candidate. Required fields:

- `sample_id`
- `source_video_id`
- `pair_index`
- `mask_id`
- `prompt`
- `prompt_input_mode`: `text_conditioned` or `ignored_by_model`
- `prompt_used_by_model`: boolean
- `win_video_path`
- `raw_loser_video_path`
- `comp_loser_video_path`
- `mask_path`
- `mask_mode`: `full` or `partial`
- `mask_convention`
- `mask_policy`
- `mask_area_ratio`
- `mask_bbox`
- `mask_motion_type`
- `mask_velocity`
- `generation_model`: `diffueraser`, `propainter`, `cococo`, or `minimax_remover`
- `source_dataset`: `videodpo` or `youtubevos`
- `raw_metrics`
- `comp_metrics`
- `quality_score`
- `defect_bucket`
- `status`
- `error_message`

## Selected Manifests

Selected manifests keep the final DPO pair view. Required fields:

- `sample_id`
- `source_video_id`
- `pair_index`
- `prompt`
- `win_video_path`
- `final_loser_video_path`
- `raw_loser_video_path`
- `comp_loser_video_path`
- `final_loser_type`: `raw` or `comp`
- `selected_role`: `primary` or `secondary`
- `mask_id`
- `mask_path`
- `mask_policy`
- `generation_model`
- `quality_score`
- `defect_bucket`
- `selection_policy`
- `selection_reason`
- `candidate_rank`
- `source_counts_snapshot`
- `seed`
- `fps`
- `num_frames`
- `height`
- `width`
- `status`

Full-mask loser generation uses `num_masks_per_video = 1`.
Partial-mask loser generation uses offline `num_masks_per_video = 4` by default
with policy `videodpo_partialmask_policy_v1_medium_hard_k4`.

For comp manifests:

```text
final_loser_video_path = comp_loser_video_path
```

For no-comp manifests:

```text
final_loser_video_path = raw_loser_video_path
```

Comp and no-comp selected manifests must reference the same selected candidate;
only `final_loser_video_path` and `final_loser_type` differ.

Primary is the default training manifest. Secondary is retained for diagnosis
and future ablations, but is not mixed into first-version DPO training by default.
