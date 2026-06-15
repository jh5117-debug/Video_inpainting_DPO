# DPO Diagnostics

Best variant:

```text
exp11_boundary_outer_b075_o005_s1s2_2000
```

Detailed report:

```text
reports/exp11_outer_b075_s2_dpo_diag_summary.md
```

## Paths

Stage1:

```text
/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage1/20260613_042729_exp11_boundary_exp11_boundary_outer_b075_o005_s1s2_2000_s1_2000_davis_pai/dpo_diagnostics.csv
```

Stage2:

```text
/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage2/20260613_042729_exp11_boundary_exp11_boundary_outer_b075_o005_s1s2_2000_s2_2000_davis_pai/dpo_diagnostics.csv
```

HAL snapshots:

```text
/home/hj/dpo-2-1-exp/this_week_exp11_exp12/dpo_diag/exp11_boundary_outer_b075_s1_dpo_diagnostics.csv
/home/hj/dpo-2-1-exp/this_week_exp11_exp12/dpo_diag/exp11_boundary_outer_b075_s2_dpo_diagnostics.csv
```

## Summary

| stage | rows | steps | dpo_loss last20 | implicit_acc last20 | norm_win_gap last20 | norm_lose_gap_clipped last20 | winner_gap_reg last20 | loser_dominant last20 | label |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| Stage1 | 201 | 1-2000 | 0.310162 | 0.987500 | -0.015532 | 0.829626 | 0.008638 | 1.000000 | LOSER_DOMINANT |
| Stage2 | 201 | 1-2000 | 0.338639 | 1.000000 | -0.000614 | 0.787290 | 0.006708 | 1.000000 | LOSER_DOMINANT |

Interpretation:

- The run is much more stable than old raw-DPO experiments: no sustained winner-gap explosion appears in the final diagnostics.
- The loser branch remains fully dominant, so final claims should combine metric and qualitative evidence with this residual dpo-diag caveat.
- The key method setting is `boundary_mode=outer`, `boundary_weight=0.75`, `outside_weight=0.05`.
