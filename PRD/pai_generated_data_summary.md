# PAI Generated Data Summary

Updated from the 2026-05-24 PAI probe and the 2026-05-25 generated-loser launch
debugging session. This is the checked-in summary; the PAI node may also keep
timestamped audit logs under `.tmp/codex_asset_prepare/`.

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

- accepted D1 fullmask generated-loser data has not been found; old OR/fullmask
  is invalid and BR/no-prior failed the limit=100 quality gate;
- DPO training has not been launched;
- the active production model set is now DiffuEraser-only (`MODELS=diffueraser`, `generation_source=diffueraser_only`);
- D2 partialmask K4 full generation is the active main data asset for
  experiments 5/6/7/8 and must continue without stopping.

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

The historical all-model smoke gate passed, but the current production path is
DiffuEraser-only for speed and operational simplicity. All PRD, README,
manifest, and report language should write:

```text
generation_source = diffueraser_only
```

D1 fullmask decision:

- old OR/fullmask root is invalid and retained only for failure audit;
- BR/no-prior root passed technical checks at limit=100 but failed quality;
- do not full-generate D1 and do not train experiment 4 from the current D1
  roots unless it is explicitly reframed as a failure ablation.

D2 partialmask decision:

- keep the PAI D2 partialmask K4 run active;
- D2 remains OR + ProPainter prior because partial masks have visible context;
- D2 is the active main data asset for experiments 5/6/7/8.

Training must remain paused until generated data is complete and verified.

## 2026-05-25 Launch Notes

- Four-model generation is runnable but too slow for the immediate full-data goal.
- DiffuEraser-only generation means each D2 winner writes four candidate rows, one per K=4 mask, and each row must record `generation_source=diffueraser_only`.
- A high-concurrency probe with very large worker count overloaded the host: GPU memory was occupied, GPU util stayed near zero, and load average rose above 3000.
- The sharded launcher now caps OpenMP/MKL/OpenBLAS/NumExpr/OpenCV threads to one by default to prevent CPU fan-out.
- Do not reuse exploratory `_shards`; archive them before changing `MODELS`, `WORKERS_PER_GPU`, or `SHARD_SIZE`.

## D1 H20 Quality Gate

Root:

```text
/home/nvme01/H20_Video_inpainting_DPO/data/generated_losers/official_videodpo_diffueraser_data_fullmask_loser_br_noise
```

Limit=100 result:

```text
rows = 100
status = OK: 100
generation_source = diffueraser_only: 100
diffueraser_inference_stack = br: 100
diffueraser_prior_mode = noise: 100
propainter.mp4 count = 0
diffueraser.mp4 count = 100
selected_primary_fullmask.jsonl = 100
selected_secondary_fullmask.jsonl = 100
quality buckets = too_bad: 95, texture_or_structure_shift: 5
quality_score median = 0.1947
quality_score mean = 0.1884
quality_score max = 0.3315
```

Preview artifacts:

- `/home/nvme01/H20_Video_inpainting_DPO/data/generated_losers/official_videodpo_diffueraser_data_fullmask_loser_br_noise/reports/previews/d1_br_noise_limit20_win_raw_first_frame.jpg`
- `/home/nvme01/H20_Video_inpainting_DPO/data/generated_losers/official_videodpo_diffueraser_data_fullmask_loser_br_noise/reports/previews/d1_br_noise_limit100_win_raw_first_frame.jpg`

Conclusion: the D1 BR/no-prior path is technically valid but the generated
losers are mostly too poor to serve as meaningful hard negatives.

## PAI D2 Audit

PAI D2 output root:

```text
/mnt/nas/hj/H20_Video_inpainting_DPO/data/generated_losers/official_videodpo_diffueraser_data_partialmask_loser_k4
```

Expected identity:

```text
generation_source = diffueraser_only
generation_model = diffueraser
mask_mode = partial
num_masks_per_video = 4
diffueraser_inference_stack = or
diffueraser_prior_mode = propainter
```

Before pulling H20 changes onto PAI, preserve local state:

```bash
git status --short
git diff
git diff --stat
git fetch --no-tags h20 main:refs/remotes/h20/main
git merge --ff-only h20/main
```

If the merge cannot fast-forward, do not reset or overwrite PAI-local changes.
