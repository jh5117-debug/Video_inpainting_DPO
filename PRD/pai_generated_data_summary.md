# PAI Generated Data Summary

Updated from the 2026-05-24 PAI probe. This is the checked-in summary; the PAI
node may also keep timestamped audit logs under `.tmp/codex_asset_prepare/`.

## 2026-05-30 D2 Training Status

D2 is ready and should not be regenerated for the beta10 reruns.

| Manifest | Rows | Use |
| --- | ---: | --- |
| `selected_primary_comp.repaired.jsonl` | 10000 | Exp5 beta10 rerun |
| `selected_primary_nocomp.repaired.jsonl` | 10000 | Exp6 beta10 rerun |
| `selected_secondary_comp.repaired.jsonl` | 10000 | reserve / later diagnostic |
| `selected_secondary_nocomp.repaired.jsonl` | 10000 | reserve / later diagnostic |

Old Exp5 beta500 / 10000-step Stage1+Stage2 is failed/collapsed and diagnostic
only. The failure is attributed to D2 generated losers plus full-mask full-loss
DPO with `beta_dpo=500`, no SFT regularization, and long training. The rerun
policy is `beta_dpo=10`, 4000 Stage1 steps, 4000 Stage2 steps, no in-training
validation, followed by qual30 and full VBench.

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

## Current Readiness

Ready:

- path/env documentation for data, weights, outputs, and generated losers;
- PAI symlink plan for `data/videodpo/current`, `data/youtubevos/current`, and model weight `current` links;
- manifest schema scaffolds for full-mask losers and partial-mask K=4 raw/comp losers;
- DPO training remains untouched.

Not ready for full offline generation yet:

- real one-sample full-mask generation smoke has not run for the four generator models;
- real one-sample partial-mask generation smoke has not run for the four generator models;
- output video decode, fps, frame count, resolution, and comp outside-mask checks are still pending;
- `tools/offline_loser_generation.py` currently writes plans/schemas and intentionally does not dispatch real inference.

Use `tools/pai_videodpo_single_sample_generation_smoke.py --run_generation` for
the real one-sample gate. It uses the canonical VideoDPO setting from
`DPO_finetune/configs/official_diffueraser_stage1.yaml` and
`VIDEO_DPO_TRAIN_DATA_YAML`.

## Decision

The next safe step is one-sample real generation smoke, not full data generation.
After a model passes smoke, it can be enabled for full-mask generation and then
partial-mask K=4 offline generation.
