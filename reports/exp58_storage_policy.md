# Exp58 Storage Policy

Status: `EXP58_STORAGE_PAI_NAS_PREFERRED`

Large videos, Kubric generated data, inference outputs, metrics cache, and visual evidence should not accumulate on H20 local storage. H20 `/home/nvme01` is allowed for git worktree, isolated env, and short-lived tiny Gate8 scratch only.

## Audit

- HAL control host `/`: 1.8T total, 238G free, 86% used. HAL should not host Exp58 video/cache outputs.
- PAI `/home`: 5.3T total, 5.0T free.
- PAI `/mnt/nas`: 10P scale and mounted.
- PAI `/mnt/workspace`: 70T total, 1.9T free, 98% used; avoid large new writes.
- H20 `/home/nvme01`: 3.4T total, 1.2T free, 66% used. Safe for tiny smoke only, not expansion.

## Permission Finding

The requested PAI output root:

`/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp58_void_native_data_diagnostic`

could not be created by `hj` because the parent is not writable. PAI log and runtime roots are writable:

- `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp58_void_native_data_diagnostic`
- `/mnt/nas/hj/H20_Video_inpainting_DPO/runtime/exp58_void_native_data_diagnostic`

If Kubric generation proceeds on PAI before the experiment output root is fixed, use PAI local fallback `/home/hj/exp58_void_native_data_diagnostic_outputs` and rsync final reports to the writable repo branch. Do not use H20 local for persistent large data.
