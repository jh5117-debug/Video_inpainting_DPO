# Exp59 VOID Official Inference On Kubric Gate8

Date: 2026-07-02

Status: `VOID_KUBRIC_OFFICIAL_INFERENCE_PASS`

## Run Summary

Official VOID pass1 inference was run on exactly 8 Exp58B Kubric Gate8 samples.

- Environment: `/home/hj/conda_envs/void_exp50_official_v2`
- Repo: `/mnt/nas/hj/H20_Video_inpainting_DPO/third_party/VOID/Netflix_void-model`
- Script: `inference/cogvideox_fun/predict_v2v.py`
- Base model: `/mnt/nas/hj/H20_Video_inpainting_DPO/weights/void/CogVideoX-Fun-V1.5-5b-InP`
- Pass1 checkpoint: `/mnt/nas/hj/H20_Video_inpainting_DPO/weights/void/netflix_void-model/void_pass1.safetensors`
- Input root: `/mnt/nas/hj/H20_Video_inpainting_DPO/runtime/exp59_void_kubric_gate8_inference/official_inputs`
- Output root used: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp59_void_kubric_gate8_inference/official_pass1_outputs`
- Evidence root: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp59_void_kubric_gate8_inference/official_pass1_evidence`
- GPU: PAI GPU0
- Start: `2026-07-02T13:13:44,515089030+08:00`
- End: `2026-07-02T13:22:39,142316431+08:00`

## Outputs

- Raw pass1 outputs: 8/8
- Official `_tuple.mp4` outputs: 8/8
- Per-sample evidence directories: 8/8

Each evidence directory contains:

- `raw_output.mp4`
- `official_tuple.mp4`
- `condition.mp4`
- `quadmask.mp4`
- `winner_rgb_removed.mp4`
- `side_by_side.mp4`
- `temporal_strip_16f.jpg`
- `object_crop_sheet.jpg`
- `overlap_crop_sheet.jpg`
- `affected_crop_sheet.jpg`
- `boundary_crop_sheet.jpg`
- `outside_crop_sheet.jpg`
- `temporal_diff_heatmap.jpg`
- `mask_value_histogram.jpg`
- `resolved_config.json`
- `checkpoint_sha.json`
- `gpu_runtime.json`
- `stdout.log`
- `stderr.log`
- `runtime_log.txt`

## Controlled Runtime Fix

The first official run attempt loaded the model successfully but failed at video decode with:

`/usr/bin/ffmpeg: error while loading shared libraries: libblas.so.3: cannot open shared object file`

This is an ffmpeg runtime path issue. The retry used the `imageio-ffmpeg` bundled static ffmpeg binary by placing a run-local symlink at the front of `PATH`:

`ffmpeg version 7.0.2-static`

No system/base environment was modified. VOID official source was not modified.

## Protocol Caveats

- Source videos are native 128x128 and 24 frames.
- Official inference resized to `384x672`.
- Official inference padded clips to the 85-frame temporal window.
- Metrics must compare at a common resolution and frame window.
- All samples remain `target_hit=false`, so this is an inference diagnostic, not adapter evidence.

## Evidence

- Runtime CSV: `reports/exp59_kubric_official_inference_runtime.csv`
- Summary JSON: `reports/exp59_kubric_official_inference_summary.json`

## Safety

- Training run: no
- Preference forward: no
- Zero-gap: no
- One-step: no
- 10-step: no
- Official VOID source modified: no
- `inference/metrics.py` modified: no
- Shared trainer modified: no
