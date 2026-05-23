# Data And Weight Assets

This document records the asset layout used before launching new ablations.
Large files stay on NAS/PAI storage; the repository only keeps env files,
README files, manifests, and `current` symlinks.

## Standard Roots

- `VIDEO_DPO_DATA_ROOT`: original VideoDPO data, including winner/chosen videos and prompt metadata.
- `VIDEO_DPO_TRAIN_DATA_YAML`: the PAI train-data YAML used by the completed official VC2 and official DiffuEraser runs.
- `YOUTUBE_VOS_ROOT`: real YouTube-VOS root. Do not treat one old `ytbv_*` generated candidate directory as the dataset root.
- `YOUTUBE_VOS_FRAMES_ROOT`, `YOUTUBE_VOS_MASKS_ROOT`: frame and annotation sub-roots.
- `GENERATED_LOSER_ROOT`: `data/generated_losers`.
- `DIFFUERASER_WEIGHT_ROOT`, `PROPAINTER_WEIGHT_ROOT`, `COCOCO_WEIGHT_ROOT`, `MINIMAX_REMOVER_WEIGHT_ROOT`: model-specific weights.
- `OFFICIAL_VIDEODPO_WEIGHT_ROOT`, `VC2_WEIGHT_ROOT`: official VideoDPO / VC2 checkpoints.

Run on PAI:

```bash
bash scripts/pai_audit_and_prepare_assets.sh
source configs/paths/pai.detected.env
```

## Generated Loser Roots

- `data/generated_losers/official_videodpo_diffueraser_data_fullmask_loser`
- `data/generated_losers/official_videodpo_diffueraser_data_partialmask_loser_k4`
- `data/generated_losers/official_videodpo_diffueraser_youtubevos_partialmask_loser_k4`

`partialmask_loser_k4` stores raw loser, comp loser, masks, and two manifest views:
comp uses `comp_loser_video_path` as `final_loser_video_path`; no-comp uses
`raw_loser_video_path`.

## PAI Status From 2026-05-24 Probe

| Asset | Status | Path / Evidence |
| --- | --- | --- |
| VideoDPO data root | FOUND | `/mnt/nas/hj/data/VideoDPO` |
| VideoDPO train YAML | FOUND | `/mnt/nas/hj/data/VideoDPO/configs/vc2_dpo/vidpro/train_data.pai.yaml` |
| VideoDPO extracted VC2 root | FOUND | `/mnt/nas/hj/data/external/hf/vidpro10k-vc2-dataset/_extracted/home/liurt/liurt_data/haoyu/dataset/vidpro10k-vc2-dataset` |
| VideoDPO pair count | CONFIRMED | completed logs print `DPO dataset has 10000 pairs` |
| YouTube-VOS root | FOUND | `/mnt/workspace/hj/nas_hj/data/external/ytbv_2019_full_resolution/train` |
| YouTube-VOS frames | FOUND | `$YOUTUBE_VOS_ROOT/JPEGImages` |
| YouTube-VOS masks | FOUND | `$YOUTUBE_VOS_ROOT/Annotations` |
| Generated loser root | FOUND | `/mnt/nas/hj/H20_Video_inpainting_DPO/data/generated_losers` |
| DiffuEraser weights | FOUND | backup `third_party_video_inpainting/weights/diffueraser/Orign_Diffueraser` |
| ProPainter weights | FOUND | `/mnt/nas/hj/data/third_party_video_inpainting/weights/propainter` |
| CoCoCo weights | FOUND | backup `third_party_video_inpainting/weights/COCOCO_weight` |
| MiniMax-Remover weights | FOUND | backup `third_party_video_inpainting/weights/minimax` |
| Real one-sample generation smoke | NOT RUN | required before full data generation |
| Full offline generated loser data | NOT GENERATED | wait for real generation smoke |
| Partial K=4 raw/comp loser data | NOT GENERATED | wait for real generation smoke |

Current conclusion: asset paths are ready enough to run one-sample generation
smoke tests. Full offline generation should not start until each selected model
passes real full-mask and partial-mask smoke with decode/fps/frame-count checks.
