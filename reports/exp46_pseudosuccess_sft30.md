# Exp46 Pseudo-Success SFT 30-Step Gate

Status: `EXP46_PSEUDOSUCCESS_SFT30_NEGATIVE`

## Scope

- H20-only pseudo-success SFT was run.
- GT-only SFT was not run.
- DPO was not run.
- No PAI write/GPU was used.
- Raw output is primary; no hard comp or VOR-Eval was used.

## Training

- Run: `PSEUDO-SFT-A_lr3em5_step30`
- Runner recipe: `SFT-B` matching Exp46 pseudo-success weights (`mask=0.75`, `boundary=1.50`, `affected=0.75`, `outside=0.20`, `far_outside=0.03`)
- LR: `3e-5`
- Steps: `30`
- Precision: `bf16_safe`
- DDP world size: `8`
- Checkpoints: `checkpoint-0`, `checkpoint-1`, `checkpoint-10`, `checkpoint-20`, `checkpoint-30`
- Peak rank0 VRAM: `66414.964844 MiB`
- Training status: `TRAIN_DONE`; losses/gradients finite through step30

## Metrics

| split | rows | dFull PSNR | dMask PSNR | dBoundary PSNR | dOutside PSNR | dSSIM | dLPIPS | dEwarp | dTemporal MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| search | 24 | -4.612642 | -0.548113 | -1.591353 | -4.812891 | -0.012066 | null | -0.019463 | 0.076098 |
| shadow | 24 | -3.366753 | -5.674479 | -3.636023 | -3.029058 | -0.014788 | null | 0.021337 | 0.180932 |

LPIPS remains unavailable/null in this evaluator path, so no positive claim can depend on LPIPS. Ewarp is computed and worsens on shadow.

## Visual Review

Codex opened and inspected:

- `reports/exp46_sft30_visual_montages/exp46_sft30_search_midframe_review.jpg`
- `reports/exp46_sft30_visual_montages/exp46_sft30_shadow_midframe_review.jpg`

Findings:

- Search: Step30 appears close at a glance but introduces subtle global tone/haze drift in outside background, matching large full/outside PSNR drops.
- Shadow: Step30 visibly shifts global brightness/color and harms mask/boundary regions; mask/boundary/outside metrics are strongly negative.
- Visual better rows: `0/48`.
- Visual worse rows: `48/48` (including `24/24` shadow).

## Gate Decision

30-step gate result: `EXP46_PSEUDOSUCCESS_SFT30_NEGATIVE`

Reasons:

- shadow full PSNR delta -3.366753 < -0.02
- shadow mask PSNR delta -5.674479 is strongly negative
- shadow boundary PSNR delta -3.636023 < -0.02
- shadow outside PSNR delta -3.029058 < -0.02
- shadow Ewarp delta 0.021337 > safe threshold direction
- visual worse 24/24 on shadow contact sheet; no visual better/tie gate
- systematic global tone/outside drift and boundary/mask degradation observed

`EXP46_PSEUDOSUCCESS_SFT100` is not unlocked. The next step is not longer training; it is data/target relabeling or a safer objective change.

## Outputs

- Metrics CSV: `reports/exp46_pseudosuccess_sft30_metrics.csv`
- Visual review CSV: `reports/exp46_pseudosuccess_sft30_visual_review.csv`
- Diagnostics CSV: `reports/exp46_pseudosuccess_sft30_diagnostics.csv`
- Summary JSON: `reports/exp46_pseudosuccess_sft30_summary.json`

