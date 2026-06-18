# Exp18 Visual Case Judgement

Date: 2026-06-18

Visual source:

```text
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/target_eval/exp18_multiframe_propagation_gated_dpo_davis10/visual_cases/all_methods
```

Local review copies were inspected from:

```text
/home/hj/tmp_exp18_visual_review/contact_sheets
```

Columns:

```text
GT / mask overlay / SFT / Exp11 / Exp18a / Exp18b / Exp18c-oracle
```

## Overall Judgement

```text
Exp18 Stage1-500 has no positive visual signal over Exp11 outer b0.75 S2.
```

Exp18a is the best Exp18 variant and occasionally looks close to Exp11, but it does not clearly improve mask texture, boundary seams, or motion structure. Exp18b and Exp18c often soften details or introduce small local artifacts. The oracle upper bound not winning visually matches the metric result.

## Case Groups

### Exp18 Clearly Better Than Exp11

None observed in DAVIS10.

### Exp18 Close / Tie-Like But Exp11 Preferred

- `rhino`: Exp18a is close, but Exp11 has more stable edge and stone/ground texture.
- `dog-agility`: Exp18a is close on motion, but no clear advantage over Exp11.
- `blackswan`: all methods are close; Exp18 does not improve the swan/body-water region.
- `lucia`: Exp18a is close but slightly softer around grass/person boundary.
- `kite-surf`: water texture is comparable, but Exp18 has no decisive boundary or mask gain.
- `bear`: near-tie; Exp18a does not improve fur/ground consistency.

### Exp18 Worse / Negative

- `boat`: Exp11 remains best for water/wake continuity. Exp18b/Exp18c show more local texture artifacts.
- `soccerball`: Exp18 does not improve fast ball/foliage structure.
- `dance-jump`: no stable improvement on transparent motion/cloth; Exp18 is tie-or-worse.
- `breakdance`: no reliable gain on motion/limb structure.

## Metric-Visual Agreement

The metric ranking agrees with visual inspection:

```text
Exp11 > Exp18a > SFT > Exp18c > Exp18b on primary DAVIS10 PSNR,
and no Exp18 variant beats Exp11 on strict-mask or boundary PSNR.
```

Exp18a being close but not better is visible in several cases. Exp18b/Exp18c being worse despite the oracle diagnostic indicates that the current propagation/generation extra losses disturb the existing Exp11 balance rather than improving it.

## Decision

Do not expand Exp18 to Stage1 1000, full cache, Stage1 2000, or Stage2 under the current formulation. Keep it as an exploratory negative ablation.
