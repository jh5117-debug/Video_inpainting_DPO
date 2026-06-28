# Exp31 VideoPainter 2000 LPIPS/Ewarp Metrics

Status: `VIDEOPAINTER_2000_POSITIVE`

- eval run root: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp31_videopainter_2000step_longrun/exp31_vp2000_eval_step0_50_2000_20260628_032700`
- metric output dir: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp31_videopainter_2000step_longrun/exp31_vp2000_lpips_ewarp_20260628_095055`
- aggregate csv: `reports/exp31_vp_2000_lpips_ewarp_metrics.csv`
- per-video csv: `reports/exp31_vp_2000_lpips_ewarp_per_video.csv`
- paired csv: `reports/exp31_vp_2000_lpips_ewarp_paired_deltas.csv`
- summary json: `reports/exp31_vp_2000_lpips_ewarp_summary.json`
- protocol: 49 frames, 720x480, boundary pixels `4`, LPIPS region bbox min size `64`
- metric backend: `inference/metrics.py` via Exp31 wrapper; backend file was not modified.
- Ewarp: backend `EwarpMetric`, using RAFT if local weights exist, otherwise DIS fallback.
- TC: computed only when a local TC model path is supplied; otherwise recorded as unavailable.

## Aggregate Comp Metrics

| split | step | PSNR | SSIM | LPIPS | mask PSNR | mask SSIM | mask LPIPS | boundary PSNR | boundary SSIM | boundary LPIPS | outside PSNR | outside L1 | Ewarp mask | TC status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| search | 0 | 24.115362 | 0.871482 | 0.071498 | 16.257492 | 0.616091 | 0.246716 | 15.676742 | 0.611520 | 0.241083 | nan | 0.000000 | 9.401874 | `TC_BACKEND_NOT_LOCAL` |
| search | 50 | 32.546294 | 0.968865 | 0.025269 | 24.688424 | 0.872860 | 0.094564 | 27.237397 | 0.873731 | 0.093392 | nan | 0.000000 | 0.897926 | `TC_BACKEND_NOT_LOCAL` |
| search | 2000 | 34.579363 | 0.976316 | 0.016601 | 26.721493 | 0.918970 | 0.056539 | 30.575066 | 0.919636 | 0.055777 | nan | 0.000000 | 0.645174 | `TC_BACKEND_NOT_LOCAL` |
| shadow | 0 | 22.992627 | 0.867663 | 0.072446 | 15.384463 | 0.568330 | 0.271744 | 15.088279 | 0.564134 | 0.265378 | nan | 0.000000 | 11.911291 | `TC_BACKEND_NOT_LOCAL` |
| shadow | 50 | 32.127458 | 0.967396 | 0.024419 | 24.519294 | 0.869166 | 0.092108 | 26.694114 | 0.869889 | 0.090944 | nan | 0.000000 | 0.998177 | `TC_BACKEND_NOT_LOCAL` |
| shadow | 2000 | 34.433188 | 0.981436 | 0.015606 | 26.825024 | 0.928328 | 0.058025 | 30.331172 | 0.928783 | 0.057677 | nan | 0.000000 | 0.739641 | `TC_BACKEND_NOT_LOCAL` |

## Primary Paired Deltas

| split | comparison | metric | mean delta | win rate | prob improved | 95% CI | leave-one-out |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| search | step2000-step0 | full_psnr | +10.464001 | 1.0000 | 1.0000 | [+8.845316, +12.117878] | [+10.063558, +10.726744] |
| search | step2000-step0 | full_lpips | -0.054897 | 1.0000 | 1.0000 | [-0.070977, -0.040608] | [-0.056267, -0.050785] |
| search | step2000-step0 | mask_psnr | +10.464001 | 1.0000 | 1.0000 | [+8.846702, +12.111835] | [+10.063558, +10.726744] |
| search | step2000-step0 | mask_lpips | -0.190177 | 1.0000 | 1.0000 | [-0.231090, -0.154323] | [-0.194743, -0.180724] |
| search | step2000-step0 | boundary_psnr | +14.898324 | 1.0000 | 1.0000 | [+13.416758, +16.474399] | [+14.481559, +15.252742] |
| search | step2000-step0 | boundary_lpips | -0.185306 | 1.0000 | 1.0000 | [-0.225104, -0.149882] | [-0.189758, -0.176336] |
| search | step2000-step0 | outside_l1 | +0.000000 | 0.0000 | 0.0000 | [+0.000000, +0.000000] | [+0.000000, +0.000000] |
| search | step2000-step0 | ewarp_mask_region | -8.756700 | 1.0000 | 1.0000 | [-12.777289, -5.576532] | [-9.006167, -7.601393] |
| search | step2000-step50 | full_psnr | +2.033069 | 0.8438 | 0.8438 | [+0.773539, +3.206759] | [+1.805572, +2.382270] |
| search | step2000-step50 | full_lpips | -0.008668 | 0.9375 | 0.9375 | [-0.011420, -0.006036] | [-0.009263, -0.007765] |
| search | step2000-step50 | mask_psnr | +2.033069 | 0.8438 | 0.8438 | [+0.810382, +3.223212] | [+1.805572, +2.382270] |
| search | step2000-step50 | mask_lpips | -0.038025 | 0.9375 | 0.9375 | [-0.049017, -0.028362] | [-0.039524, -0.035482] |
| search | step2000-step50 | boundary_psnr | +3.337669 | 0.9688 | 0.9688 | [+2.506893, +4.143365] | [+3.190212, +3.545757] |
| search | step2000-step50 | boundary_lpips | -0.037615 | 0.9375 | 0.9375 | [-0.048619, -0.027208] | [-0.039171, -0.035015] |
| search | step2000-step50 | outside_l1 | +0.000000 | 0.0000 | 0.0000 | [+0.000000, +0.000000] | [+0.000000, +0.000000] |
| search | step2000-step50 | ewarp_mask_region | -0.252752 | 0.8438 | 0.8438 | [-0.418103, -0.103262] | [-0.288840, -0.200096] |
| shadow | step2000-step0 | full_psnr | +11.440561 | 1.0000 | 1.0000 | [+9.212271, +14.109938] | [+10.549038, +11.673019] |
| shadow | step2000-step0 | full_lpips | -0.056840 | 1.0000 | 1.0000 | [-0.071430, -0.044501] | [-0.058560, -0.053375] |
| shadow | step2000-step0 | mask_psnr | +11.440561 | 1.0000 | 1.0000 | [+9.227450, +13.937645] | [+10.549038, +11.673019] |
| shadow | step2000-step0 | mask_lpips | -0.213718 | 1.0000 | 1.0000 | [-0.266090, -0.162436] | [-0.219791, -0.198714] |
| shadow | step2000-step0 | boundary_psnr | +15.242894 | 1.0000 | 1.0000 | [+13.255022, +17.575313] | [+14.512615, +15.515291] |
| shadow | step2000-step0 | boundary_lpips | -0.207700 | 1.0000 | 1.0000 | [-0.261969, -0.159793] | [-0.213594, -0.193286] |
| shadow | step2000-step0 | outside_l1 | +0.000000 | 0.0000 | 0.0000 | [+0.000000, +0.000000] | [+0.000000, +0.000000] |
| shadow | step2000-step0 | ewarp_mask_region | -11.171650 | 1.0000 | 1.0000 | [-17.759618, -6.484490] | [-11.491337, -8.701826] |
| shadow | step2000-step50 | full_psnr | +2.305730 | 0.9062 | 0.9062 | [+1.580904, +3.137667] | [+1.978904, +2.423504] |
| shadow | step2000-step50 | full_lpips | -0.008813 | 0.9375 | 0.9375 | [-0.012091, -0.006124] | [-0.009257, -0.007503] |
| shadow | step2000-step50 | mask_psnr | +2.305730 | 0.9062 | 0.9062 | [+1.572980, +3.220398] | [+1.978904, +2.423504] |
| shadow | step2000-step50 | mask_lpips | -0.034082 | 0.9688 | 0.9688 | [-0.048306, -0.023689] | [-0.035408, -0.028244] |
| shadow | step2000-step50 | boundary_psnr | +3.637059 | 0.9375 | 0.9375 | [+2.729173, +4.683092] | [+3.283746, +3.794741] |
| shadow | step2000-step50 | boundary_lpips | -0.033266 | 0.9688 | 0.9688 | [-0.047849, -0.023352] | [-0.034678, -0.027444] |
| shadow | step2000-step50 | outside_l1 | +0.000000 | 0.0000 | 0.0000 | [+0.000000, +0.000000] | [+0.000000, +0.000000] |
| shadow | step2000-step50 | ewarp_mask_region | -0.258536 | 0.9375 | 0.9375 | [-0.348117, -0.170928] | [-0.282745, -0.240951] |

## Decision

- formal LPIPS/Ewarp gate satisfied on shadow-dev comp metrics

The status above is a VideoPainter-only long-run decision. It is not a universal adapter, final SOTA, or top-conference novelty claim.
