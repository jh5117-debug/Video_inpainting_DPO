# PAI Asset Readiness Report

- generated_at: 2026-05-24T01:19:58+08:00
- repo_path: /mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO
- local_patch: .tmp/codex_asset_prepare/local_changes_before_fast_asset_fix_20260524_011954.patch

## Detected Env

```bash
export VIDEO_DPO_OFFICIAL_ROOT=/mnt/nas/hj/official_repos/VideoDPO_official_1febdb4
export VIDEO_DPO_DATA_ROOT=/mnt/nas/hj/official_repos/VideoDPO_official_1febdb4/data

export YOUTUBE_VOS_ROOT=/mnt/workspace/hj/nas_hj/data/external/ytbv_2019_full_resolution/train
export GENERATED_LOSER_ROOT=/mnt/nas/hj/H20_Video_inpainting_DPO/data/generated_losers
export DIFFUERASER_WEIGHT_ROOT=/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/weights/diffueraser/Orign_Diffueraser
export PROPAINTER_WEIGHT_ROOT=/mnt/nas/hj/data/third_party_video_inpainting/weights/propainter
export COCOCO_WEIGHT_ROOT=/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/weights/COCOCO_weight
export MINIMAX_REMOVER_WEIGHT_ROOT=/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/weights/minimax
export OFFICIAL_VIDEODPO_WEIGHT_ROOT=/mnt/nas/hj/official_repos/VideoDPO_official_1febdb4/checkpoints
export VC2_WEIGHT_ROOT=/mnt/nas/hj/official_repos/VideoDPO_official_1febdb4/checkpoints/vc2
export EXP_OUTPUT_ROOT=/mnt/nas/hj/H20_Video_inpainting_DPO/outputs
```

## Symlinks

```text
  6870942      0 lrwxrwxrwx   1 root     root           57 May 24 01:19 data/videodpo/current -> /mnt/nas/hj/official_repos/VideoDPO_official_1febdb4/data
  6870943      0 lrwxrwxrwx   1 root     root           58 May 24 01:19 data/generated_losers/current -> /mnt/nas/hj/H20_Video_inpainting_DPO/data/generated_losers
  6870944      0 lrwxrwxrwx   1 root     root          130 May 24 01:19 weights/diffueraser/current -> /mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/weights/diffueraser/Orign_Diffueraser
  6870945      0 lrwxrwxrwx   1 root     root           64 May 24 01:19 weights/propainter/current -> /mnt/nas/hj/data/third_party_video_inpainting/weights/propainter
  6870946      0 lrwxrwxrwx   1 root     root          114 May 24 01:19 weights/cococo/current -> /mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/weights/COCOCO_weight
  6870947      0 lrwxrwxrwx   1 root     root          108 May 24 01:19 weights/minimax_remover/current -> /mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/weights/minimax
  6870948      0 lrwxrwxrwx   1 root     root           64 May 24 01:19 weights/official_videodpo/current -> /mnt/nas/hj/official_repos/VideoDPO_official_1febdb4/checkpoints
  6870949      0 lrwxrwxrwx   1 root     root           68 May 24 01:19 weights/vc2/current -> /mnt/nas/hj/official_repos/VideoDPO_official_1febdb4/checkpoints/vc2
```

## Existence Check

```text
FOUND VIDEO_DPO_DATA_ROOT=/mnt/nas/hj/official_repos/VideoDPO_official_1febdb4/data
MISSING_OR_UNCONFIRMED YOUTUBE_VOS_ROOT=
FOUND GENERATED_LOSER_ROOT=/mnt/nas/hj/H20_Video_inpainting_DPO/data/generated_losers
FOUND DIFFUERASER_WEIGHT_ROOT=/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/weights/diffueraser/Orign_Diffueraser
FOUND PROPAINTER_WEIGHT_ROOT=/mnt/nas/hj/data/third_party_video_inpainting/weights/propainter
FOUND COCOCO_WEIGHT_ROOT=/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/weights/COCOCO_weight
FOUND MINIMAX_REMOVER_WEIGHT_ROOT=/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/weights/minimax
FOUND OFFICIAL_VIDEODPO_WEIGHT_ROOT=/mnt/nas/hj/official_repos/VideoDPO_official_1febdb4/checkpoints
FOUND VC2_WEIGHT_ROOT=/mnt/nas/hj/official_repos/VideoDPO_official_1febdb4/checkpoints/vc2
```

## Notes

- This fast report avoids broad NAS scans.
- Full generation has not been started.
- DPO training has not been started.
