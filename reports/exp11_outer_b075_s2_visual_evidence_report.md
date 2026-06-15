# Exp11 Outer B0.75 S2 Visual Evidence Report

Date: 2026-06-15

## Status

The selected DAVIS visual evidence rerun completed on PAI and was copied back to HAL.

PAI root:

```text
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/target_eval/20260615_exp11_outer_b075_s2_selected_visuals
```

HAL archive:

```text
/home/hj/dpo-2-1-exp/this_week_exp11_exp12/visual_evidence/exp11_outer_b075_s2_selected_visuals
```

The archive includes:

- per-video frame-wise metrics for SFT-48000 and Exp11 outer b0.75 S2
- selected frame JPG comparisons
- vertical frame contact sheets
- side-by-side MP4s

## Metric Delta On Selected Cases

Positive means Exp11 outer b0.75 S2 is better than SFT-48000 in this selected rerun.

| video | delta whole PSNR | delta mask PSNR | delta whole SSIM | delta mask SSIM | interpretation |
|---|---:|---:|---:|---:|---|
| boat | +1.4362 | +1.4362 | +0.0208 | +0.0925 | strong positive |
| rhino | +0.9862 | +0.9862 | +0.0052 | +0.0133 | positive |
| dog-agility | +0.7001 | +0.7001 | +0.0023 | -0.0167 | useful visual positive, mask SSIM mixed |
| blackswan | +0.4102 | +0.4102 | +0.0017 | -0.0048 | mild positive, mask SSIM mixed |
| lucia | +0.3976 | +0.3976 | +0.0008 | +0.0167 | positive but visually subtle |
| flamingo | +0.1811 | +0.1811 | +0.0003 | -0.0000 | weak positive |
| dance-jump | -1.0214 | -1.0214 | -0.0039 | -0.0253 | failure / cautionary |
| soccerball | -0.8039 | -0.8039 | -0.0090 | -0.0772 | failure / cautionary |

## Qualitative Notes

`boat` is the best current visual evidence case. SFT-48000 produces a visible white fog / patch over the boat wake and hull, while Exp11 outer b0.75 S2 keeps the water texture and hull boundary cleaner.

`rhino` is useful because the mask crosses the foreground animal boundary. Exp11 reduces the smearing around the head/body boundary compared with SFT-48000.

`dog-agility` is useful for motion and thin structures. The mask crosses the moving dog and poles; Exp11 improves whole-frame and mask PSNR, though mask SSIM is mixed.

`lucia` is a subtle positive case and is better suited as a supplementary example rather than the lead figure.

`dance-jump` and `soccerball` should not be used as positive evidence from this rerun; they are useful as failure/caution cases.

## Recommended Figure Pool

Use these first:

1. `boat`
2. `rhino`
3. `dog-agility`
4. `lucia`
5. `blackswan`

Keep as failure/caution:

1. `dance-jump`
2. `soccerball`

## Important Protocol

This rerun used the fixed DAVIS frame-wise protocol:

- raw6
- hard comp
- D+G off
- no PCM
- no mask dilation
- no Gaussian blur
- frame-wise in-memory metric

