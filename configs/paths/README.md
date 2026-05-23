# Path Configuration

This project keeps code separate from data, generated losers, model weights, and run outputs. Prefer environment variables over hard-coded absolute paths in new scripts.

## Environment Variables

| Variable | Meaning |
| --- | --- |
| `VIDEO_DPO_DATA_ROOT` | Original VideoDPO preference dataset root. |
| `VIDEO_DPO_OFFICIAL_ROOT` | Official VideoDPO source checkout used for VC2 config and inference assets. |
| `VIDEO_DPO_TRAIN_DATA_YAML` | PAI train-data YAML used by the completed VC2/DiffuEraser DPO runs. |
| `VIDEO_DPO_PAIR_MANIFEST` | Optional resolved pair manifest, if exposed separately by the data YAML. |
| `VIDEO_DPO_WINNER_ROOT` | Optional resolved winner/chosen video root. |
| `VIDEO_DPO_REJECTED_ROOT` | Optional resolved rejected/loser video root. |
| `VIDEO_DPO_PROMPT_FILE` | Optional resolved prompt metadata file. |
| `YOUTUBE_VOS_ROOT` | YouTube-VOS / YouTube-VOS-derived frame dataset root. |
| `YOUTUBE_VOS_FRAMES_ROOT` | YouTube-VOS frame root, usually `$YOUTUBE_VOS_ROOT/JPEGImages`. |
| `YOUTUBE_VOS_MASKS_ROOT` | YouTube-VOS annotation/mask root, usually `$YOUTUBE_VOS_ROOT/Annotations`. |
| `GENERATED_LOSER_ROOT` | Root for offline generated loser videos and manifests. |
| `DIFFUERASER_WEIGHT_ROOT` | DiffuEraser checkpoint root. |
| `PROPAINTER_WEIGHT_ROOT` | ProPainter checkpoint root. |
| `COCOCO_WEIGHT_ROOT` | CoCoCo checkpoint/root config path. |
| `MINIMAX_REMOVER_WEIGHT_ROOT` | MiniMax-Remover checkpoint/cache root. |
| `OFFICIAL_VIDEODPO_ROOT` | Official VideoDPO source checkout. |
| `OFFICIAL_VIDEODPO_WEIGHT_ROOT` | Official VideoDPO checkpoint root. |
| `VC2_WEIGHT_ROOT` | VC2 baseline checkpoint root. |
| `EXP_OUTPUT_ROOT` | Experiment output root. |
| `LINGBOT_PROCESS_NAME` | Process/job display name; new scripts default to `lingbot-phy`. |

Use `configs/paths/pai.example.env` as a template. Do not commit private machine-specific `.env` files.

## PAI Paths Confirmed On 2026-05-24

These paths were confirmed from the PAI shell transcript and should be copied
into `configs/paths/pai.detected.env` on PAI, not hard-coded into experiment
code:

```bash
export VIDEO_DPO_OFFICIAL_ROOT=/mnt/nas/hj/official_repos/VideoDPO_official_1febdb4
export VIDEO_DPO_DATA_ROOT=/mnt/nas/hj/data/VideoDPO
export VIDEO_DPO_TRAIN_DATA_YAML=/mnt/nas/hj/data/VideoDPO/configs/vc2_dpo/vidpro/train_data.pai.yaml
export YOUTUBE_VOS_ROOT=/mnt/workspace/hj/nas_hj/data/external/ytbv_2019_full_resolution/train
export YOUTUBE_VOS_FRAMES_ROOT=/mnt/workspace/hj/nas_hj/data/external/ytbv_2019_full_resolution/train/JPEGImages
export YOUTUBE_VOS_MASKS_ROOT=/mnt/workspace/hj/nas_hj/data/external/ytbv_2019_full_resolution/train/Annotations
```

The VideoDPO train-data YAML resolves to the extracted VidPro10K/VC2 dataset
root and completed PAI logs report `DPO dataset has 10000 pairs`.
