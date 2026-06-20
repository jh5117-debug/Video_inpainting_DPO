# Exp20 Adaptive Radius Calibration

- Manifest: `/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO/data/generated_losers/exp09_10_11_youtubevos_gtwin_d3comp_pai/manifests/selected_primary_comp.gtwin.pai_paths.jsonl`
- Clips scanned: `3327`
- Valid non-empty clips: `3327`
- Runtime seconds: `311.2`
- Mask convention: `png_255_inpaint_region_0_keep_region`

## Base Radius Statistics

| base | mean | median | std | p10 | p25 | p75 | p90 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ap | 46.9357 | 46.9880 | 3.8494 | 42.1198 | 44.2448 | 49.6466 | 51.6626 |
| sqrt | 115.0066 | 115.4991 | 7.4388 | 105.1126 | 109.6177 | 120.2247 | 123.1612 |

## Calibrated k Candidates

| mode | target median | k | radius mean | radius median | std | p10 | p90 | clamp min | clamp max | valid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| adaptive_area_perimeter | 12 | 0.255385 | 11.987 | 12.000 | 0.983 | 10.757 | 13.194 | 0.000 | 0.000 | True |
| adaptive_area_perimeter | 16 | 0.340513 | 15.982 | 16.000 | 1.311 | 14.342 | 17.592 | 0.000 | 0.000 | True |
| adaptive_area_perimeter | 20 | 0.425641 | 19.978 | 20.000 | 1.638 | 17.928 | 21.990 | 0.000 | 0.000 | True |
| adaptive_area_perimeter | 24 | 0.510769 | 23.973 | 24.000 | 1.966 | 21.513 | 26.388 | 0.000 | 0.000 | True |
| adaptive_sqrt_area | 12 | 0.103897 | 11.949 | 12.000 | 0.773 | 10.921 | 12.796 | 0.000 | 0.000 | True |
| adaptive_sqrt_area | 16 | 0.138529 | 15.932 | 16.000 | 1.030 | 14.561 | 17.061 | 0.000 | 0.000 | True |
| adaptive_sqrt_area | 20 | 0.173161 | 19.915 | 20.000 | 1.288 | 18.201 | 21.327 | 0.000 | 0.000 | True |
| adaptive_sqrt_area | 24 | 0.207794 | 23.898 | 24.000 | 1.546 | 21.842 | 25.592 | 0.000 | 0.000 | True |
