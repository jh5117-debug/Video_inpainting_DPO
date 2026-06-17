# Qualitative Summary

DAVIS10 visual sanity completed.

HAL output:

```text
/home/hj/dpo-2-1-exp/exp16_stage1_500_visual_sanity_davis10
```

Columns:

1. GT
2. mask overlay
3. SFT-48000 baseline
4. Exp11 outer b0.75 S2
5. Exp16 Stage1-500

Judgement:

- Positive / weak signal: `lucia`, `dance-jump`, `soccerball`.
- Roughly tied: `bear`, `kite-surf`.
- Worse than Exp11: `boat`, `rhino`, `dog-agility`, `blackswan`, `breakdance`.

Final qualitative conclusion:

```text
Exp16 Stage1-500 has weak positive signal but does not beat Exp11.
```

Report:

```text
reports/exp16_stage1_500_visual_case_judgement.md
```
