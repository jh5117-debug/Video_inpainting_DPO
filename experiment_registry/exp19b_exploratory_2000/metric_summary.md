# Metric Summary

DAVIS50 completed.

Important label note: the evaluator row is still named `Exp19b_stage2_500`,
but the eval script loaded the exploratory 2000 adapter from:

```text
/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage2/exp19b_boundary_flow_adapter_s2_2000_exploratory_from500_limit100/last_weights/flow_adapter.pt
```

| Method | PSNR | SSIM | LPIPS | Ewarp | strict mask PSNR | boundary PSNR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SFT-48000 | 32.665330 | 0.971062 | 0.016222 | 7.214799 | 21.021880 | 26.194571 |
| Exp11 outer b0.75 S2 | 32.840213 | 0.971818 | 0.015339 | 7.181782 | 21.196763 | 26.441316 |
| Exp19b exploratory 2000 | 32.840122 | 0.971818 | 0.015340 | 7.181850 | 21.196671 | 26.441224 |

Delta versus Exp11:

| Metric | Delta |
| --- | ---: |
| PSNR | -0.000092 |
| SSIM | -0.000000 |
| LPIPS | +0.000001 |
| Ewarp | +0.000069 |
| strict mask PSNR | -0.000092 |
| boundary PSNR | -0.000092 |

Conclusion: longer adapter-only training did not amplify the tiny DAVIS10
Ewarp trend. It remains a no-op/slightly negative ablation.
