# Qualitative Summary

HAL visual evidence path:

```text
/home/hj/dpo-2-1-exp/exp17_saturation_positive_dpo_davis10_visuals/
```

PAI visual evidence path:

```text
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/target_eval/exp17_saturation_positive_dpo_20260617_171347_exp17_saturation_positive_davis10/visual_cases/
```

Judgement:

```text
No Exp17 variant has a stable positive visual signal over Exp11.
```

Weak / near-positive cases:

- Exp17a on `boat`: visually and metrically better in some frames, but not
  stable enough.
- Exp17b on `breakdance` and `lucia`: small local gains / near ties.

Failure cases:

- `rhino`, `dog-agility`, `dance-jump`, `soccerball`, and Exp17c on
  `blackswan` are worse than Exp11.

Evidence:

```text
reports/exp17_visual_case_judgement.md
reports/exp17_visual_case_judgement.csv
```
