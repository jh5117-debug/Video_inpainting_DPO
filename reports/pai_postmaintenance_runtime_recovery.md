# PAI Post-Maintenance Runtime Recovery

Date: 2026-06-25

## Durable Runtime Recovery

The volatile `/home/hj` outputs needed by Exp25 and Exp26 were copied to NAS before continuing post-maintenance experiments.

| track | source | NAS destination | files | bytes | inventory | sha256 |
| --- | --- | --- | ---: | ---: | --- | --- |
| Exp26 | `/home/hj/exp26_gate64_runs/gate64_official_43597cf_20260625_031155` | `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp26_videopainter_dpo_v2/gate64_official_43597cf_20260625_031155` | 14408 | 8405904095 | OK | OK |
| Exp25 | `/home/hj/exp25_gate32_dense_review_runs` | `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp25_vor_or_preference_data/gate32_dense_review_20260625` | 99 | 66982608 | OK | OK |

Markers:

- `/mnt/nas/hj/H20_Video_inpainting_DPO/runtime/EXP26_GATE64_PERSISTED_TO_NAS`
- `/mnt/nas/hj/H20_Video_inpainting_DPO/runtime/EXP25_GATE32_REVIEW_PERSISTED_TO_NAS`

## Remaining Permission Blockers

See `reports/pai_postmaintenance_asset_permission_matrix.csv` and `reports/runtime/pai_postmaintenance_root_permission_fix.sh`.
