# Data Generation Manifest Schema

Every generated loser manifest is JSONL. Each line describes one final DPO
pair candidate.

Required fields:

- `sample_id`
- `source_video_id`
- `mask_id`
- `prompt`
- `win_video_path`
- `raw_loser_video_path`
- `comp_loser_video_path`
- `final_loser_video_path`
- `mask_path`
- `mask_mode`: `full` or `partial`
- `mask_convention`
- `comp`: `true` or `false`
- `generation_model`: `diffueraser`, `propainter`, `cococo`, or `minimax_remover`
- `source_dataset`: `videodpo` or `youtubevos`
- `seed`
- `fps`
- `num_frames`
- `height`
- `width`
- `mask_area_ratio`
- `mask_bbox`
- `status`

Full-mask loser generation uses `num_masks_per_video = 1`.
Partial-mask loser generation uses offline `num_masks_per_video = 4` by default.

For comp manifests:

```text
final_loser_video_path = comp_loser_video_path
```

For no-comp manifests:

```text
final_loser_video_path = raw_loser_video_path
```
