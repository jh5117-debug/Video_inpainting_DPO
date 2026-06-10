# Exp9/10/11 DAVIS-50 Frame-wise raw6 Comparison

Status: complete for PSNR/SSIM under the fixed frame-wise protocol.

Output root:

```text
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/target_eval/exp09_10_11_20260610_framewise_raw6_davis50_v2
```

Summary table:

```text
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/target_eval/exp09_10_11_20260610_framewise_raw6_davis50_v2/metrics/summary_all.csv
```

Protocol:

- DAVIS-50
- 24 validation frames
- raw6
- no PCM
- D+G off
- hard comp
- frame-wise metrics
- metric backend: `inference/metrics.py`

## Current Table

| Method | Weight | Step | D+G | PSNR | SSIM | LPIPS | VFID | TC |
|---|---|---|---|---:|---:|---:|---:|---:|
| DiffuEraser | step48000 | raw6 | off | 32.7018 | 0.9710 | TBD | TBD | TBD |
| Exp9-1 | DPO-S1 + SFT-S2 | raw6 | off | 32.7402 | 0.9719 | TBD | TBD | TBD |
| Exp9-2 | DPO-S1 + DPO-S2 | raw6 | off | 32.8080 | 0.9717 | TBD | TBD | TBD |
| Exp10-1 | DPO-S1 + SFT-S2 | raw6 | off | 32.9083 | 0.9716 | TBD | TBD | TBD |
| Exp10-2 | DPO-S1 + DPO-S2 | raw6 | off | 32.8791 | 0.9715 | TBD | TBD | TBD |
| Exp11-1 | DPO-S1 + SFT-S2 | raw6 | off | 32.8726 | 0.9716 | TBD | TBD | TBD |
| Exp11-2 | DPO-S1 + DPO-S2 | raw6 | off | 32.9034 | 0.9716 | TBD | TBD | TBD |

TBD means not computed in the verified run because the full LPIPS/VFID/TC asset
chain was not present on PAI. The metric code supports these columns through
`inference/metrics.py`; the fixed protocol wrapper exposes explicit switches
for them.

## Conclusion

The best current mean PSNR is Exp10-1:

```text
Exp10-1 DPO-S1 + SFT-S2
PSNR = 32.9083
mask PSNR = 24.0618
```

Exp11-2 is very close:

```text
Exp11-2 DPO-S1 + DPO-S2
PSNR = 32.9034
mask PSNR = 24.0569
```

All Exp9/10/11 combinations beat the SFT48000 baseline in PSNR under the fixed
frame-wise protocol, but the improvement is modest. Current dpo-diag evidence
does not support blindly extending Stage1/Stage2 beyond 2000 steps. Prefer a
checkpoint sweep under this same protocol before any longer run.
