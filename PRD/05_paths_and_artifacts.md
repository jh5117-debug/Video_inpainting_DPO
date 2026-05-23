# Paths And Artifacts

## Policy

Code lives in experiment-named directories. Data, weights, generated losers, logs, and outputs must remain outside code or be represented by symlinks/placeholders.

## Tracked Lightweight Directories

- `data/README.md`
- `data/videodpo/.gitkeep`
- `data/youtubevos/.gitkeep`
- `data/generated_losers/README.md`
- `weights/README.md`
- `outputs/README.md`
- `experiments/README.md`
- `configs/paths/README.md`
- `configs/paths/pai.example.env`

## Environment Variables

| Variable | Meaning |
| --- | --- |
| `VIDEO_DPO_DATA_ROOT` | Original VideoDPO preference data. |
| `YOUTUBE_VOS_ROOT` | YouTube-VOS data. |
| `GENERATED_LOSER_ROOT` | Offline generated loser data root. |
| `DIFFUERASER_WEIGHT_ROOT` | DiffuEraser weights. |
| `PROPAINTER_WEIGHT_ROOT` | ProPainter weights. |
| `COCOCO_WEIGHT_ROOT` | CoCoCo weights/config root. |
| `MINIMAX_REMOVER_WEIGHT_ROOT` | MiniMax-Remover weights/cache root. |
| `OFFICIAL_VIDEODPO_ROOT` | Official VideoDPO checkout. |
| `VC2_WEIGHT_ROOT` | VC2 checkpoint root. |
| `EXP_OUTPUT_ROOT` | Run outputs/logs. |
| `LINGBOT_PROCESS_NAME` | Process/job name, default `lingbot-phy`. |

## PAI Recorded Artifact Roots

These must be verified on PAI before destructive cleanup:

- `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/videodpo_vc2_dpo_official_clean/pai-vc2-dpo-official-full-gpu0-3-gb8-step3000-20260521_061414`
- `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/vbench_full/pai-vc2-official-step3000-full-vbench-20260521_141824`
- `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/official_diffueraser_stage1/pai-official-diffueraser-stage1-full-gpu4-7-gb8-step3000-20260521_072559`
- `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/official_diffueraser_stage2/pai-official-diffueraser-stage2-full-gpu4-7-gb8-step3000-20260521_150540`
- `/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO/logs/vbench_fullmask/pai-official-diffueraser-stage2-vs-base-full-vbench-20260522_002926`

## Git Ignore Rules

The repository ignores real data, weights, outputs, checkpoints, logs, W&B, and large model files. Only README/placeholders in data/weight/output directories should be tracked.
