# Exp50 VOID One-Step Heldout Metrics V2

Time: 2026-07-01T01:04:09+08:00

Status: `VOID_ONE_STEP_PARETO_MIXED`

- Better/tie/worse by metric pre-review: 0/2/2
- Mean full PSNR delta: -0.025049
- Mean outside PSNR delta: 0.028255
- Mean mask PSNR delta: -0.513424
- Mean affected PSNR delta: -0.224311
- Mean boundary PSNR delta: -0.310957
- Mean pixel diff L1: 2.233569

## Per-sample

- BLENDER_CON001_00742: ONE_STEP_SAFE_TIE_LOCAL_SIGNAL; d_full=-0.004430; d_out=0.000285; d_mask=0.175087; diff_l1=1.053309
- BLENDER_CON001_00744: ONE_STEP_METRIC_WORSE; d_full=-0.052780; d_out=-0.007564; d_mask=-0.362738; diff_l1=1.502495
- REAL_ENV102_00001_002_02: ONE_STEP_SAFE_TIE; d_full=0.013580; d_out=0.226052; d_mask=-0.755522; diff_l1=1.651581
- REAL_ENV200_00001_006_02: ONE_STEP_METRIC_WORSE; d_full=-0.056567; d_out=-0.105755; d_mask=-1.110523; diff_l1=4.726890

## Safety

No VOR-Eval, no hard comp, no training, no optimizer step in H4b metrics. LPIPS/Ewarp/TC unavailable and not used for promotion.
