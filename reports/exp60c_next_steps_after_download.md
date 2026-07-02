# Exp60C Next Steps After Download

Status: `EXP60C_WAITING_FOR_PAI_NAS_PERMISSION`

The H20 subset is complete and verified, but PAI/NAS transfer is blocked.

Next minimal step:

1. Fix PAI write permission for:
   `/mnt/nas/hj/H20_Video_inpainting_DPO/data/external`
2. Rerun only the PAI transfer and verification milestone:
   - rsync H20 `raw_subset` to PAI/NAS
   - copy manifests / reports / sha256
   - verify train1000/test100 file counts
   - verify sha256 match
   - run PAI no-GPU decode check
   - generate PAI path manifests
3. Only after `EXP60C_PAI_VPDATA_SUBSET_READY`, start a separate D3 mask
   generation milestone.

Do not run mask generation, loser generation, inference, DPO, or training while
the PAI/NAS data root is missing.
