# Data And Weight Assets

This document records the asset layout used before launching new ablations.
Large files stay on NAS/PAI storage; the repository only keeps env files,
README files, manifests, and `current` symlinks.

## Standard Roots

- `VIDEO_DPO_DATA_ROOT`: original VideoDPO data, including winner/chosen videos and prompt metadata.
- `YOUTUBE_VOS_ROOT`: real YouTube-VOS root. Do not treat one old `ytbv_*` generated candidate directory as the dataset root.
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
