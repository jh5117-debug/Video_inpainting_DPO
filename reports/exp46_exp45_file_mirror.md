# Exp46 Exp45 H20 File Mirror

Status: EXP45_H20_MIRROR_READY

## Scope

This milestone validates the PAI Exp45 formal Stage2 handoff package after mirroring it to H20. PAI was used read-only; no PAI GPU, no PAI write, no H20 training, and no optimizer step occurred.

## Source

- Exp45 branch: `origin/research/exp45-pai-minimax-pair-scaleup-20260629`
- Known Exp45 HEAD: `d0c8430a5ba35f37415ed52d53040829ef1123d6`
- PAI source root: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp45_pai_minimax_pair_scaleup/scaleup_mining_20260629_190229`
- H20 mirror root: `/home/nvme01/H20_Video_inpainting_DPO/data/h20_mirror/minimax/pai_abs`

## Transfer Note

H20 direct PAI authentication failed with resident H20 keys. The completed transfer used SSH agent forwarding from HAL to H20 (`ssh -A`) so H20 could read PAI/NAS via rsync. This was a read-only mirror operation; PAI outputs and processes were not modified.

## Validation

- Required paths: 326
- Required files: 232
- Required directories: 94
- Ready paths: 326
- Missing paths: 0
- SHA mismatches: 0
- Required-path mirrored bytes: 359462035

## Outputs

- `reports/exp46_exp45_file_mirror.csv`
- `reports/exp46_exp45_file_mirror.json`
- `reports/exp46_exp45_h20_sha256.txt`
- `reports/exp46_exp45_missing_files.csv`

## Decision

Exp45 mirror is ready for H20 manifest rewrite/validation.
