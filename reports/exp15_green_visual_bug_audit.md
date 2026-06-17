# Exp15 Green Visual Bug Audit

## Symptom

User reported that HAL visual outputs under:

```text
/home/hj/dpo-2-1-exp/exp15_or_benchmark_davis50_visuals
```

look green in playback.

## Debug Performed

Sampled three videos (`bear`, `boat`, `kite-surf`) and three frames each. For each sample, exported:

- input frame;
- mask overlay;
- ProPainter raw output;
- DiffuEraser SFT-48000 raw output;
- Ours Exp11 outer b0.75 S2 raw output;
- old contact sheet / old mp4 frame stats.

Debug frames and stats:

```text
reports/exp15_green_visual_debug_frames/
reports/exp15_green_visual_debug_frames/pixel_stats.csv
reports/exp15_green_visual_fixed_pixel_stats.csv
```

## Findings

- Raw RGB input/mask/method frames are normal; they are not all green and are not masks accidentally loaded as RGB frames.
- Old contact sheets are also normal by pixel statistics.
- Old mp4s show low green-dominant ratios in OpenCV decode, but were encoded with OpenCV `mp4v`, which is less portable and can show green-cast playback in some viewers/platforms.
- PAI system `/usr/bin/ffmpeg` exists but fails with `libblas.so.3` missing. The first fixed attempt fell back to OpenCV and was discarded.

## Fix

`exp15_or_benchmark_davis50/code/make_or_visual_grid.py` now:

1. prefers `EXP15_FFMPEG_BIN` if provided;
2. then tries Exp15 vendored `imageio-ffmpeg` static ffmpeg;
3. only falls back to system ffmpeg / OpenCV if needed.

On PAI, `imageio-ffmpeg` was installed into an Exp15-local vendor directory:

```text
/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO/exp15_or_benchmark_davis50/vendor/imageio_ffmpeg_pkg
```

The fixed MP4s are H.264 / yuv420p:

```text
Video: h264 (High), yuv420p(progressive), 1920x360, libx264
```

## Fixed Output

PAI:

```text
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/target_eval/exp15_or_benchmark_davis50_fixed/visual_grids
```

HAL:

```text
/home/hj/dpo-2-1-exp/exp15_or_benchmark_davis50_visuals_fixed
```

The fixed directory contains 50 mp4 files and 50 contact sheets.

## Conclusion

The green issue was not caused by the raw method outputs or mask/path mapping. It was a visual evidence encoding/playback robustness issue caused by OpenCV `mp4v` outputs and unavailable system ffmpeg dependencies. Fixed visual grids now use static ffmpeg H.264/yuv420p encoding.
