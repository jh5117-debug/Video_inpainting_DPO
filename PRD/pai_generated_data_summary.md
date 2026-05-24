# PAI Generated Data Summary

Updated from the 2026-05-24 PAI probe. This is the checked-in summary; the PAI
node may also keep timestamped audit logs under `.tmp/codex_asset_prepare/`.

## Asset Status

| Asset | Status | Path / Evidence |
| --- | --- | --- |
| VideoDPO data | FOUND | `/mnt/nas/hj/data/VideoDPO` |
| VideoDPO train YAML | FOUND | `/mnt/nas/hj/data/VideoDPO/configs/vc2_dpo/vidpro/train_data.pai.yaml` |
| VideoDPO extracted VC2 data | FOUND | `/mnt/nas/hj/data/external/hf/vidpro10k-vc2-dataset/_extracted/home/liurt/liurt_data/haoyu/dataset/vidpro10k-vc2-dataset` |
| VideoDPO pair count | CONFIRMED | completed PAI logs report `DPO dataset has 10000 pairs` |
| YouTube-VOS train root | FOUND | `/mnt/workspace/hj/nas_hj/data/external/ytbv_2019_full_resolution/train` |
| YouTube-VOS frames | FOUND | `$YOUTUBE_VOS_ROOT/JPEGImages` |
| YouTube-VOS masks | FOUND | `$YOUTUBE_VOS_ROOT/Annotations` |
| Generated loser root | FOUND | `/mnt/nas/hj/H20_Video_inpainting_DPO/data/generated_losers` |
| DiffuEraser weights | FOUND | backup `third_party_video_inpainting/weights/diffueraser/Orign_Diffueraser` |
| ProPainter weights | FOUND | `/mnt/nas/hj/data/third_party_video_inpainting/weights/propainter` |
| CoCoCo weights | FOUND | backup `third_party_video_inpainting/weights/COCOCO_weight` |
| MiniMax-Remover weights | FOUND | backup `third_party_video_inpainting/weights/minimax` |
| DiffuEraser PCM weights | FOUND | `/mnt/nas/hj/weights/PCM_Weights/sd15/pcm_sd15_smallcfg_2step_converted.safetensors` |
| CoCoCo SD inpainting root | FOUND | backup `third_party_video_inpainting/downloads/sd_inpaint_hf_extract/stable-diffusion-inpainting` |

## Current Readiness

Ready:

- path/env documentation for data, weights, outputs, and generated losers;
- PAI symlink plan for `data/videodpo/current`, `data/youtubevos/current`, and model weight `current` links;
- manifest schema scaffolds for full-mask losers and partial-mask K=4 raw/comp losers;
- canonical VideoDPO setting verified from the completed official DiffuEraser runs: 320x512, 16 frames, frame stride 1;
- real one-sample full-mask and partial-mask generation smoke passed for DiffuEraser, ProPainter, CoCoCo, and MiniMax-Remover;
- output decode, frame count, resolution, and partial-mask comp outside-region checks passed for those smoke runs;
- DPO training remains untouched.

Still not done:

- full offline generated-loser data has not been launched;
- DPO training has not been launched;
- before full generation, run the disk/capacity preflight and choose the exact model set, sample range, and output root.

## Canonical Smoke Results

| Model | Full Mask | Partial Mask | Evidence |
| --- | --- | --- | --- |
| DiffuEraser | OK | OK | `outputs/asset_smoke_tests/parallel_generation_smoke_20260524_085008/diffueraser/report.md` |
| ProPainter | OK | OK | `outputs/asset_smoke_tests/parallel_generation_smoke_20260524_063024/propainter/report.md` |
| CoCoCo | OK | OK | `outputs/asset_smoke_tests/parallel_generation_smoke_20260524_070827/cococo/report.md` |
| MiniMax-Remover | OK | OK | `outputs/asset_smoke_tests/parallel_generation_smoke_20260524_070018/minimax_remover/report.md` |

All passing rows decoded 16 frames at 320x512. Partial-mask comp rows reported
outside-mask max absolute diff `0.000000`.

## Decision

The asset smoke gate is passed for all four generation models. The next safe
step is an explicit full/offline data generation launch plan with disk estimate
and manifest validation. Training must remain paused until generated data is
complete and verified.
