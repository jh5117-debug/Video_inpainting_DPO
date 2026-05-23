# Path Configuration

This project keeps code separate from data, generated losers, model weights, and run outputs. Prefer environment variables over hard-coded absolute paths in new scripts.

## Environment Variables

| Variable | Meaning |
| --- | --- |
| `VIDEO_DPO_DATA_ROOT` | Original VideoDPO preference dataset root. |
| `YOUTUBE_VOS_ROOT` | YouTube-VOS / YouTube-VOS-derived frame dataset root. |
| `GENERATED_LOSER_ROOT` | Root for offline generated loser videos and manifests. |
| `DIFFUERASER_WEIGHT_ROOT` | DiffuEraser checkpoint root. |
| `PROPAINTER_WEIGHT_ROOT` | ProPainter checkpoint root. |
| `COCOCO_WEIGHT_ROOT` | CoCoCo checkpoint/root config path. |
| `MINIMAX_REMOVER_WEIGHT_ROOT` | MiniMax-Remover checkpoint/cache root. |
| `OFFICIAL_VIDEODPO_ROOT` | Official VideoDPO source checkout. |
| `VC2_WEIGHT_ROOT` | VC2 baseline checkpoint root. |
| `EXP_OUTPUT_ROOT` | Experiment output root. |
| `LINGBOT_PROCESS_NAME` | Process/job display name; new scripts default to `lingbot-phy`. |

Use `configs/paths/pai.example.env` as a template. Do not commit private machine-specific `.env` files.
