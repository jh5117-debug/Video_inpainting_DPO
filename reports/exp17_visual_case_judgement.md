# Exp17 Visual Case Judgement

Date: 2026-06-17

Visual evidence was inspected from contact sheets copied to HAL:

```text
/home/hj/dpo-2-1-exp/exp17_saturation_positive_dpo_davis10_visuals/
```

Each sheet compares:

```text
GT / mask overlay / SFT-48000 / Exp11 outer b0.75 S2 / Exp17 variant
```

## Overall Judgement

```text
Exp17 Stage1-1000 has no positive visual signal strong enough to continue.
```

Exp17b is the best of the three variants by DAVIS10 metrics, but it still does
not beat Exp11 outer b0.75 S2. Visual inspection agrees with the metrics:
there are a few weak/tie cases, but no stable improvement across the selected
videos.

## Positive Or Near-Positive Cases

- `boat` with Exp17a: metric improves over Exp11 and some boat/water detail
  looks closer to GT. However the shoreline / wake texture remains unstable,
  so this is a weak positive rather than a reliable method gain.
- `breakdance` with Exp17b: slight metric gain and visually close to Exp11.
  The improvement is small and boundary PSNR is still lower.
- `lucia` with Exp17b: near tie to slight positive. The figure and grass look
  comparable to Exp11, but the visual difference is not decisive.
- `kite-surf` with Exp17a: tiny numerical gain, visually near tie.

## Failure Cases

- `rhino`: Exp17b is much worse than Exp11 in PSNR, SSIM, strict-mask PSNR, and
  visual quality. The animal head/body and foreground transition are blurrier.
- `dog-agility`: fast-motion dog structure and pole boundaries are not improved.
- `dance-jump`: body/motion region degrades relative to Exp11.
- `soccerball`: no visible improvement; boundary and object region remain less
  stable than Exp11.
- `blackswan` with Exp17c: feather/water texture becomes flatter and blurrier.

## Artifact Notes

Exp17 does not introduce a dramatic new purple-fog or grid artifact in the
inspected cases, but it also does not reduce the existing difficult-case
artifacts more reliably than Exp11. The dominant failure is softer / less
faithful mask-region reconstruction and weaker boundary consistency.

## Decision

Do not run Exp17 Stage1-2000 or Stage2. Keep Exp17 as a negative ablation:

```text
positive preservation and the current saturation gate were implemented, but the
current formulation did not solve loser-dominant diagnostics and did not improve
DAVIS10 metrics or visuals over Exp11.
```
