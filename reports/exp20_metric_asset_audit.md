# Exp20 Metric Asset Audit

## Purpose

Restore the optional VFID/FVD-style metric and TC metric for the locked Exp20
dev evaluator without changing the metric implementation or re-running
DiffuEraser inference.

## Existing Metric Implementation

- VFID is implemented in `inference/metrics.py` with the project-local
  Inception-I3D feature extractor and Frechet distance. This is the same
  implementation that previous DAVIS50 / YouTubeVOS100 reports call `VFID`.
- No separate FVD implementation was found in the active Exp20 evaluator.
  Therefore the restored column is reported as `VFID/FVD` only to acknowledge
  naming ambiguity; it is not a second independent metric.
- TC is implemented in `inference/metrics.py` as adjacent-frame CLIP ViT-H/14
  cosine similarity through `TemporalConsistencyMetric`.

## Asset Search Result

HAL has the required metric assets:

| Asset | HAL path | Size | SHA256 |
| --- | --- | ---: | --- |
| I3D RGB ImageNet | `/home/hj/Video_inpainting_DPO/weights/i3d_rgb_imagenet.pt` | 49M | `2609088c2e8c868187c9921c50bc225329a9057ed75e76120e0b4a397a2c7538` |
| OpenCLIP ViT-H/14 | `/home/hj/.tmp/open_clip_vit_h14/open_clip_pytorch_model.bin` | 3.7G | `9a78ef8e8c73fd0df621682e7a8e8eb36c6916cb3c16b291a082ecd52ab79cc4` |
| RAFT Things | `/home/hj/Video_inpainting_DPO/weights/propainter/raft-things.pth` | 21M | `fcfa4125d6418f4de95d84aec20a3c5f4e205101715a79f193243c186ac9a7e1` |

PAI already had the RAFT weight at
`/mnt/nas/hj/data/third_party_video_inpainting/weights/propainter/raft-things.pth`
with matching SHA256 `fcfa4125d6418f4de95d84aec20a3c5f4e205101715a79f193243c186ac9a7e1`.

PAI was missing:

- `/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp20_autoresearch/weights/i3d_rgb_imagenet.pt`
- `/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp20_autoresearch/weights/open_clip_vit_h14/open_clip_pytorch_model.bin`

## Transfer Plan

The missing I3D and OpenCLIP assets are being copied from HAL to the PAI
Exp20 worktree `weights/` directory, which is the path expected by
`exp20_autoresearch_scale_adaptive_region_dpo/scripts/run_dev_baselines_pai.sh`.

Weights are not committed to Git.

## Backfill Protocol

`exp20_autoresearch_scale_adaptive_region_dpo/code/backfill_existing_eval_metrics.py`
recomputes optional metrics from already-saved hard-comp frame outputs:

- no DiffuEraser re-inference;
- no evaluator protocol change;
- no overwrite of original `summary.csv`;
- writes `backfill_per_video_metrics.csv`, `backfill_summary.csv`, and the
  aggregate first-wave full metric reports.
