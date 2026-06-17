# Exp16 Stage1-500 Visual Case Judgement

Date: 2026-06-17

Visual source:

```text
/home/hj/dpo-2-1-exp/exp16_stage1_500_visual_sanity_davis10
```

Compared methods:

```text
SFT-48000 DiffuEraser
Exp11 outer b0.75 S2
Exp16 Stage1-500 limit100, evaluated as DPO-S1 + SFT-S2 hybrid
```

Protocol:

```text
raw6, no PCM, no mask dilation, no Gaussian blur, hard comp, frame-wise metrics, no VBench
```

## Metric Summary

| Method | PSNR | SSIM | strict mask PSNR | boundary PSNR | bbox PSNR | bbox SSIM |
|---|---:|---:|---:|---:|---:|---:|
| SFT-48000 | 29.8193 | 0.9625 | 18.2894 | 24.2926 | 21.3016 | 0.7380 |
| Exp11 outer b0.75 S2 | 30.1736 | 0.9644 | 18.6437 | 24.5907 | 21.6559 | 0.7513 |
| Exp16 Stage1-500 | 29.9460 | 0.9642 | 18.4161 | 24.5280 | 21.4284 | 0.7562 |

Exp16 improves over SFT on most DAVIS10 metrics, but it does not beat Exp11 on
the main PSNR / strict-mask / boundary-PSNR measures. Bbox SSIM is slightly
higher for Exp16, but the visual evidence is mixed.

## A. Exp16 Better Than Exp11

These are positive sanity signals, not final evidence.

| Video | Judgement | Metric support | Visual note |
|---|---|---|---|
| lucia | Exp16 slightly better | Exp16-Exp11 PSNR +0.6092, SSIM +0.0015, strict mask PSNR +0.6092 | Grass and person boundary look a little more coherent; no obvious new purple fog or grid artifact. |
| dance-jump | Exp16 slightly better | Exp16-Exp11 PSNR +0.2259, SSIM +0.0016, strict mask PSNR +0.2259 | Pavement/seam area is marginally smoother. Improvement is modest and not paper-ready by itself. |
| soccerball | Exp16 better than weak Exp11 result | Exp16-Exp11 PSNR +1.2508, SSIM +0.0169, strict mask PSNR +1.2508 | Background vegetation/fence region is more stable than Exp11 on this subset, but the case is not a strong visual win because all methods remain close and motion blur dominates. |

## B. Exp16 Roughly Tied With Exp11

| Video | Judgement | Metric support | Visual note |
|---|---|---|---|
| bear | near tie, Exp11 slightly safer | Exp16-Exp11 PSNR -0.4407, SSIM -0.0005 | Small mask / simple background; all methods look similar, so this mainly confirms Exp16 does not catastrophically break output. |
| kite-surf | near tie, Exp11 slightly safer | Exp16-Exp11 PSNR -0.1748, SSIM -0.0005 | Water texture and spray remain plausible, but Exp11 has marginally better numeric score. |

## C. Exp16 Worse Than Exp11

| Video | Judgement | Metric support | Visual note |
|---|---|---|---|
| boat | Exp16 slightly worse | Exp16-Exp11 PSNR -0.1644, SSIM +0.0009, strict mask PSNR -0.1644 | Exp16 is not broken, but the boat wake / water edge is less stable than Exp11. |
| rhino | Exp16 clearly worse | Exp16-Exp11 PSNR -2.1822, SSIM -0.0153, strict mask PSNR -2.1822 | Large object / structured foreground remains harder; Exp16 introduces a softer, less faithful region near the mask. |
| dog-agility | Exp16 worse than Exp11 but better than SFT | Exp16-Exp11 PSNR -0.4021, SSIM -0.0024, strict mask PSNR -0.4021 | Fast motion and thin poles are handled better by Exp11; Exp16 remains usable but not better. |
| blackswan | Exp16 worse | Exp16-Exp11 PSNR -0.5336, SSIM -0.0020, strict mask PSNR -0.5336 | Water texture / bird body region is less stable than Exp11. |
| breakdance | Exp16 clearly worse | Exp16-Exp11 PSNR -0.4633, SSIM -0.0020, strict mask PSNR -0.4633 | Pink/paste-like residual around the dancer/crowd is more visible; this is a concrete caution case for the current loss weights. |

## Final Visual Judgement

```text
Exp16 Stage1-500 has weak positive visual signal but does not beat Exp11.
```

The run is useful as implementation validation: real ProPainter prior cache,
confidence gating, latent-x0 prior/gen/boundary losses, checkpointing, and eval
loading all work. But the visual and metric results are not strong enough to
justify immediate full prior cache or Stage1 2000.

Recommended decision:

```text
Do not launch full Exp16 training yet.
Pause and adjust lambda_prior / lambda_gen / confidence_alpha or add a loss schedule if continuing.
```
