# Exp26 Gate64 Duplicate Source Audit

Run root: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp26_videopainter_dpo_v2/gate64_official_43597cf_20260625_031155`

## Summary

- `SOURCE_PIXEL_DUPLICATE_IN_SELECTED_49F`: 8

## Failed Source Rows

| sample_id | classification | seq unique | seek unique | duplicate groups | recommendation |
| --- | --- | ---: | ---: | --- | --- |
| vp2_gate64_002_REAL_ENV158_00005_001_04 | SOURCE_PIXEL_DUPLICATE_IN_SELECTED_49F | 48 | 48 | 0,1 | treat_as_source_static_duplicate; decide whether formal mode permits pixel-identical real frames |
| vp2_gate64_005_REAL_ENV233_00101_005_05 | SOURCE_PIXEL_DUPLICATE_IN_SELECTED_49F | 48 | 48 | 0,1 | treat_as_source_static_duplicate; decide whether formal mode permits pixel-identical real frames |
| vp2_gate64_013_REAL_ENV280_00103_004_05 | SOURCE_PIXEL_DUPLICATE_IN_SELECTED_49F | 48 | 48 | 0,1 | treat_as_source_static_duplicate; decide whether formal mode permits pixel-identical real frames |
| vp2_gate64_026_REAL_ENV243_00003_002_05 | SOURCE_PIXEL_DUPLICATE_IN_SELECTED_49F | 48 | 48 | 0,1 | treat_as_source_static_duplicate; decide whether formal mode permits pixel-identical real frames |
| vp2_gate64_039_REAL_ENV280_00102_007_02 | SOURCE_PIXEL_DUPLICATE_IN_SELECTED_49F | 44 | 44 | 2,3;4,5,6,7,8 | treat_as_source_static_duplicate; decide whether formal mode permits pixel-identical real frames |
| vp2_gate64_043_REAL_ENV185_00009_005_05 | SOURCE_PIXEL_DUPLICATE_IN_SELECTED_49F | 48 | 48 | 0,1 | treat_as_source_static_duplicate; decide whether formal mode permits pixel-identical real frames |
| vp2_gate64_048_REAL_ENV202_00005_003_04 | SOURCE_PIXEL_DUPLICATE_IN_SELECTED_49F | 48 | 48 | 0,1 | treat_as_source_static_duplicate; decide whether formal mode permits pixel-identical real frames |
| vp2_gate64_056_REAL_ENV198_00003_006_02 | SOURCE_PIXEL_DUPLICATE_IN_SELECTED_49F | 48 | 48 | 7,8 | treat_as_source_static_duplicate; decide whether formal mode permits pixel-identical real frames |

## Interpretation

- `SOURCE_PIXEL_DUPLICATE_IN_SELECTED_49F` means sequential decoding of the first 49 source frames already contains pixel-identical frames.
- `OPENCV_RANDOM_SEEK_DUPLICATE` means sequential decode is unique but OpenCV random seeking repeated frames; this is a materializer implementation issue.
- This audit does not alter Gate64 outputs and does not start VideoPainter DPO.

## Environment Note

PAI `ffprobe` was present but failed to load `libblas.so.3` in the active shell,
so metadata columns were unavailable in this run. The classification is still
valid because it is based on OpenCV sequential decode and target-index seek
decode, which both reproduced the same duplicate-frame groups.
