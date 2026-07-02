# Exp60C H20 Final VPData Subset Validation

Status: `EXP60C_H20_VPDATA_SUBSET_READY`

- Train videos: `1000`
- Test videos: `100`
- Total videos: `1100`
- Replacement videos: `11`
- Full SHA256 rows: `1100` available, `0` missing for final subset
- Train/test source overlap: `0`
- Original failed URLs remaining: `0`
- H20 raw file count observed after replacement download: `1100`
- H20 decode note: H20 lacks `ffprobe`; replacement download therefore recorded `FFPROBE_ERROR`. Decode verification is required on PAI/NAS before `EXP60C_PAI_VPDATA_SUBSET_READY`.

No masks, losers, inference, DPO, training, GPU use, or full VPData download ran.

## Decode Repair Addendum

- Initial H20 OpenCV decode: `1094 / 1100` pass, `6` decode-failed incomplete MP4 files.
- Targeted repair: `6 / 6` decode-failed files redownloaded from the same VPData source URLs; corrupt originals retained with `.corrupt_*` suffix.
- Final H20 OpenCV decode after repair: `1100 / 1100` pass.
- Full final SHA256 rows after replacement and repair: `1100`.
