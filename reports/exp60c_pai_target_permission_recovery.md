# Exp60C PAI Target Permission Recovery

Status: `EXP60C_PAI_TARGET_PERMISSION_RECOVERED`

The root-side permission fix was already completed before this milestone. Codex did not run any root permission script, chmod, or chown.

## Verified Directories

| path | owner | group | mode | read | write | execute | write_probe |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `/mnt/nas/hj/H20_Video_inpainting_DPO/data/external/vpdata_exp60b` | `hj` | `hj` | `0o770` | `True` | `True` | `True` | `True` |
| `/mnt/nas/hj/H20_Video_inpainting_DPO/data/external/vpdata_exp60b/raw_subset` | `hj` | `hj` | `0o770` | `True` | `True` | `True` | `True` |
| `/mnt/nas/hj/H20_Video_inpainting_DPO/data/external/vpdata_exp60b/reports` | `hj` | `hj` | `0o770` | `True` | `True` | `True` | `True` |
| `/mnt/nas/hj/H20_Video_inpainting_DPO/data/external/vpdata_exp60b/sha256` | `hj` | `hj` | `0o770` | `True` | `True` | `True` | `True` |
| `/mnt/nas/hj/H20_Video_inpainting_DPO/data/external/vpdata_exp60b/manifests` | `hj` | `hj` | `0o770` | `True` | `True` | `True` | `True` |

All target directories passed exists/read/write/execute/write_probe checks for user `hj`. The next allowed milestone is transfer and verification only. No mask generation, loser generation, DPO, training, GPU use, or VPData redownload was run.
