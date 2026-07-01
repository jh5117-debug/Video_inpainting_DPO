# Exp50 VOID One-Step Heldout Metrics V2

Time: 2026-07-01T06:54:51+08:00

Status: `VOID_ONE_STEP_PASS`

- Better/tie/worse by conservative review: 0/3/1
- Mean full PSNR delta: 0.035484
- Mean outside PSNR delta: 0.061493
- Mean mask PSNR delta: -0.078046
- Mean affected PSNR delta: -0.006237
- Mean boundary PSNR delta: -0.035899
- Mean pixel diff L1: 2.310697

## Per-sample

- BLENDER_CON001_00742: ONE_STEP_SAFE_TIE; d_full=0.009783; d_out=0.011096; d_mask=-0.039166; diff_l1=1.337507
- BLENDER_CON001_00744: ONE_STEP_SAFE_TIE_LOCAL_SIGNAL; d_full=-0.010325; d_out=-0.015819; d_mask=0.055536; diff_l1=1.488257
- REAL_ENV102_00001_002_02: ONE_STEP_SAFE_TIE; d_full=0.203918; d_out=0.303363; d_mask=-0.136005; diff_l1=1.708145
- REAL_ENV200_00001_006_02: ONE_STEP_METRIC_WORSE; d_full=-0.061440; d_out=-0.052666; d_mask=-0.192549; diff_l1=4.708880

## Safety

No VOR-Eval, no hard comp, no training, no optimizer step in H4b metrics. LPIPS/Ewarp/TC unavailable and not used for promotion.
