# Status

Status: completed.

Best variant:

```text
exp11_boundary_outer_b075_o005_s1s2_2000
```

The best Stage2 combination is `DPO-S1 + DPO-S2` with:

- `boundary_mode=outer`
- `mask_weight=1.0`
- `boundary_weight=0.75`
- `outside_weight=0.05`

Under the fixed DAVIS50 raw6 hard-comp / D+G off / no PCM / frame-wise in-memory metric protocol, this variant is the current best result among Exp9 / Exp10 / Exp11 / Exp12.

Selected visual evidence has been verified and copied to HAL:

```text
/home/hj/dpo-2-1-exp/this_week_exp11_exp12/visual_evidence/exp11_outer_b075_s2_selected_visuals
```

Primary positive case: `boat`.

Usable positives: `rhino`, `dog-agility`, `lucia`, `blackswan`.

Caution / failure cases: `dance-jump`, `soccerball`.

YouTubeVOS100 extension completed on PAI with the same raw6 hard-comp protocol.
Exp11 outer b0.75 S2 improves over SFT-48000:

```text
YouTubeVOS100:
  SFT-48000: PSNR 33.3968, SSIM 0.9701, LPIPS 0.0176, VFID 0.2007, TC 0.9819
  Exp11 S2:  PSNR 33.7238, SSIM 0.9711, LPIPS 0.0168, VFID 0.1925, TC 0.9821
```

Final 20 paper/PPT visual cases are archived at:

```text
/home/hj/dpo-2-1-exp/final_20_visual_cases_for_paper
```
