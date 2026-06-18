# Exp19 Metric Summary

DAVIS10 legal Exp19 inference completed on PAI.

| method | PSNR | SSIM | LPIPS | Ewarp | strict mask PSNR | boundary PSNR |
|---|---:|---:|---:|---:|---:|---:|
| SFT-48000 | 29.6181 | 0.9620 | 0.02204 | 8.3724 | 18.3203 | 24.2735 |
| Exp11 outer b0.75 S2 | 29.8295 | 0.9633 | 0.02065 | 8.3307 | 18.5317 | 24.6577 |
| Exp19b Stage2-500 | 29.8291 | 0.9633 | 0.02065 | 8.3306 | 18.5313 | 24.6574 |

Delta Exp19b - Exp11:

- PSNR: `-0.00038 dB`
- SSIM: `-0.0000024`
- LPIPS: `-0.0000013` (tiny better)
- Ewarp: `-0.000080` (tiny better, far below 2% positive gate)
- strict mask PSNR: `-0.00038 dB`
- boundary PSNR: `-0.00028 dB`

TC was not computed in this run because the TC backend tried to fetch an
OpenCLIP checkpoint from Hugging Face and PAI network access failed. Ewarp was
computed with the local RAFT backend.

Decision: negative / neutral gate. Exp19b does not exceed Exp11.
