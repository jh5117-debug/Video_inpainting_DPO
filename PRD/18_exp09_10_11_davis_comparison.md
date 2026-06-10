# Exp9/10/11 DAVIS-50 Frame-wise raw6 Comparison

Protocol: DAVIS-50, 24 frames, raw6, no PCM, D+G off, hard comp, frame-wise in-memory metrics from `inference/metrics.py`.

All five metric columns below were computed in the same PAI all-metric pass, not merged from mp4 outputs.

Output root:

```text
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/target_eval/exp09_10_11_20260611_050013_framewise_raw6_davis50_allmetrics
```

| Method | Weight | Step | D+G | PSNR | SSIM | LPIPS | VFID | TC |
|---|---|---|---|---:|---:|---:|---:|---:|
| DiffuEraser | step48000 | raw6 | off | 32.6755 | 0.9702 | 0.0168 | 0.1940 | 0.9708 |
| Exp9-1 | DPO-S1 + SFT-S2 | raw6 | off | 32.7511 | 0.9715 | 0.0162 | 0.1794 | 0.9712 |
| Exp9-2 | DPO-S1 + DPO-S2 | raw6 | off | 32.7161 | 0.9714 | 0.0163 | 0.1981 | 0.9712 |
| Exp10-1 | DPO-S1 + SFT-S2 | raw6 | off | 32.8410 | 0.9715 | 0.0154 | 0.1894 | 0.9715 |
| Exp10-2 | DPO-S1 + DPO-S2 | raw6 | off | 32.9491 | 0.9723 | 0.0154 | 0.1912 | 0.9714 |
| Exp11-1 | DPO-S1 + SFT-S2 | raw6 | off | 32.8914 | 0.9716 | 0.0157 | 0.1930 | 0.9712 |
| Exp11-2 | DPO-S1 + DPO-S2 | raw6 | off | 32.8839 | 0.9719 | 0.0157 | 0.1829 | 0.9711 |

## Quick Read

- Best PSNR: Exp10-2 (32.9491).
- Best SSIM: Exp10-2 (0.9723).
- Best LPIPS (lower is better): Exp10-1 (0.0154, nearly tied with Exp10-2).
- Best VFID (lower is better): Exp9-1 (0.1794).
- Best TC: Exp10-1 (0.9715, very close to Exp10-2).
- All Exp9/10/11 variants improve PSNR over the SFT48000 baseline under this all-metric run.

## Conclusion

The fixed all-metric pass confirms that Exp10 is the strongest quantitative family. Exp10-2 is best on PSNR/SSIM, while Exp10-1 is best or tied on LPIPS/TC. Exp11 improves over baseline but does not beat Exp10 on PSNR/SSIM in this run. Exp9 improves over baseline but is weaker than Exp10/11 on distortion metrics.

The earlier PSNR-only run remains useful for reproducibility, but this 20260611 all-metric pass should be treated as the current table because PSNR/SSIM/LPIPS/VFID/TC were computed from the same in-memory outputs.
