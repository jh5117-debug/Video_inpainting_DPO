# Exp47 Manifest / Path / Frame Alignment Audit

Status: `EXP47_MANIFEST_ALIGNMENT_PASS`

- Rows audited: `112`
- Failed rows: `0`
- Active target field: `winner_path` in `exp46_runner_pseudosuccess_*` manifests
- Original Exp45 `target_path` / `target_frames_dir` were preserved for traceability; Exp46 training used extracted H20-local pseudo-success frame dirs.

## Split Overlap

| overlap | count |
| --- | ---: |
| train_search_group_overlap | 0 |
| train_shadow_group_overlap | 0 |
| search_shadow_group_overlap | 0 |
| train_search_source_overlap | 0 |
| train_shadow_source_overlap | 0 |
| search_shadow_source_overlap | 0 |

## Check Summary

| check | pass rows | total |
| --- | ---: | ---: |
| active_paths_exist | 112 | 112 |
| active_paths_h20_local | 112 | 112 |
| active_paths_not_pai_abs | 112 | 112 |
| active_paths_not_hal | 112 | 112 |
| frame_count_consistent | 112 | 112 |
| resolution_consistent | 112 | 112 |
| source_match | 112 | 112 |
| target_is_pseudo_success | 112 | 112 |
| target_not_gt_condition | 112 | 112 |
| target_frames_match_mp4 | 112 | 112 |
| rgb_bgr_decode_ok | 112 | 112 |
| mask_polarity_ok | 112 | 112 |
| no_vor_eval | 112 | 112 |
| no_hard_comp | 112 | 112 |

## Interpretation

No manifest/path/frame alignment bug was found in the active Exp46 runner manifests.
Active paths are H20-local absolute paths, targets are pseudo-success extracted frame directories, masks are non-empty with expected polarity, frame counts/resolution are consistent, no VOR-Eval or hard-comp rows are active, and split overlap is zero.

## Outputs

- `reports/exp47_manifest_path_frame_alignment_audit.csv`
- `reports/exp47_manifest_path_frame_alignment_summary.json`
