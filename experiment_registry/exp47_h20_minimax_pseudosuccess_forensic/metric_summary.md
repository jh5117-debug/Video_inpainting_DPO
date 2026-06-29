# Exp47 Metric Summary

Status: EXP47_READBACK_READY

Exp46 SFT30 aggregate deltas used as forensic starting point:

| split | dFull PSNR | dMask PSNR | dBoundary PSNR | dOutside PSNR | dEwarp |
| --- | ---: | ---: | ---: | ---: | ---: |
| search | -4.612642 | -0.548113 | -1.591353 | -4.812891 | -0.019463 |
| shadow | -3.366753 | -5.674479 | -3.636023 | -3.029058 | 0.021337 |

## Milestone B Manifest Alignment

- Status: `EXP47_MANIFEST_ALIGNMENT_PASS`
- Rows audited: `112`
- Failed rows: `0`
- Split overlap total: `0`
- Full-pass checks: active paths exist/H20-local/not PAI/not HAL, target pseudo-success identity, target frame/mp4 match, frame count, resolution, RGB sanity, mask polarity, no VOR-Eval, no hard comp.

## Milestone C Teacher Quality

- Status: `EXP47_TEACHER_GLOBAL_DRIFT_CONFIRMED`
- Rows audited: `48`
- Strict clean / local-only / global drift: `0/22/26`
- Mean pseudo target vs V_bg full/mask/boundary/outside PSNR: `32.370202` / `28.258355` / `25.815867` / `32.921862`
- Mean outside L1: `0.017477`
- Mean abs brightness delta: `0.014741`
- Mean low-frequency drift proxy: `0.014869`
- Mean mask removal PSNR gain over condition: `18.313552`

## Milestone D Step30 Movement

- Status: `EXP47_STEP30_MOVEMENT_AUDITED`
- Root signal: `SFT_LOSS_OR_TARGET_PATH_BUG`
- Rows audited: `48`
- Step30 closer to pseudo target / V_bg: `0/0`
- Mean Step30-to-pseudo full L1 delta vs Step0: `0.005742`
- Mean Step30-to-V_bg full/mask/outside L1 delta vs Step0: `0.011993` / `0.017009` / `0.012223`
- Full cosine train/teacher, train/V_bg, teacher/V_bg: `0.262656` / `-0.140454` / `0.025774`
