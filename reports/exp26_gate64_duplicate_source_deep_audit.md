# Exp26 Gate64 Deep Duplicate Source Audit

Run root: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp26_videopainter_dpo_v2/gate64_official_43597cf_20260625_031155`

This audit treats 49 unique decoded indices with unique monotonic PTS/time as formal-valid even when two static frames are pixel-identical.

## Summary

- audited failures: `8`
- formal-valid after timestamp/index audit: `8`
- `F_STATIC_PIXEL_DUPLICATE_BUT_UNIQUE_TIMESTAMPS`: `8`

## Rows

| sample_id | classification | formal_valid | unique_hashes | unique_pts | pixel dup groups | ffprobe | framemd5 | recommendation |
| --- | --- | --- | ---: | ---: | --- | --- | --- | --- |
| vp2_gate64_002_REAL_ENV158_00005_001_04 | F_STATIC_PIXEL_DUPLICATE_BUT_UNIQUE_TIMESTAMPS | True | 48 | 49 | 0,1 | False | False | guard_misclassified_static_pixels_as_invalid |
| vp2_gate64_005_REAL_ENV233_00101_005_05 | F_STATIC_PIXEL_DUPLICATE_BUT_UNIQUE_TIMESTAMPS | True | 48 | 49 | 0,1 | False | False | guard_misclassified_static_pixels_as_invalid |
| vp2_gate64_013_REAL_ENV280_00103_004_05 | F_STATIC_PIXEL_DUPLICATE_BUT_UNIQUE_TIMESTAMPS | True | 48 | 49 | 0,1 | False | False | guard_misclassified_static_pixels_as_invalid |
| vp2_gate64_026_REAL_ENV243_00003_002_05 | F_STATIC_PIXEL_DUPLICATE_BUT_UNIQUE_TIMESTAMPS | True | 48 | 49 | 0,1 | False | False | guard_misclassified_static_pixels_as_invalid |
| vp2_gate64_039_REAL_ENV280_00102_007_02 | F_STATIC_PIXEL_DUPLICATE_BUT_UNIQUE_TIMESTAMPS | True | 44 | 49 | 2,3;4,5,6,7,8 | False | False | guard_misclassified_static_pixels_as_invalid |
| vp2_gate64_043_REAL_ENV185_00009_005_05 | F_STATIC_PIXEL_DUPLICATE_BUT_UNIQUE_TIMESTAMPS | True | 48 | 49 | 0,1 | False | False | guard_misclassified_static_pixels_as_invalid |
| vp2_gate64_048_REAL_ENV202_00005_003_04 | F_STATIC_PIXEL_DUPLICATE_BUT_UNIQUE_TIMESTAMPS | True | 48 | 49 | 0,1 | False | False | guard_misclassified_static_pixels_as_invalid |
| vp2_gate64_056_REAL_ENV198_00003_006_02 | F_STATIC_PIXEL_DUPLICATE_BUT_UNIQUE_TIMESTAMPS | True | 48 | 49 | 7,8 | False | False | guard_misclassified_static_pixels_as_invalid |

## Dependency Note

If `ffprobe_available` or `ffmpeg_framemd5_available` is false, the row still has PyAV timestamp/index evidence and OpenCV seek evidence; the missing CLI tool is tracked as an environment issue, not as source invalidity.
