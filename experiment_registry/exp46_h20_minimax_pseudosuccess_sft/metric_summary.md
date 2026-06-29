# Metric Summary

Status: MINIMAX_PSEUDOSUCCESS_STAGE2_NEGATIVE

Step0 baseline and Step30 use raw MiniMax output on H20 Exp45 pseudo-success search/shadow splits. LPIPS is unavailable in this evaluator path and remains null.

| split | rows | dFull PSNR | dMask PSNR | dBoundary PSNR | dOutside PSNR | dSSIM | dEwarp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| search | 24 | -4.612642 | -0.548113 | -1.591353 | -4.812891 | -0.012066 | -0.019463 |
| shadow | 24 | -3.366753 | -5.674479 | -3.636023 | -3.029058 | -0.014788 | 0.021337 |

Pseudo-success SFT30 strongly regressed full, mask, boundary, and outside metrics on shadow; SFT100 is not unlocked.
