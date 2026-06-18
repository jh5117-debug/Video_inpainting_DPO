# Exp19c DPO / Warp Diagnostic Summary

Runs:

```text
exp19c_light_warp_dpo/dpo_diag/lambda000_dpo_diagnostics.csv
exp19c_light_warp_dpo/dpo_diag/lambda005_dpo_diagnostics.csv
exp19c_light_warp_dpo/dpo_diag/lambda010_dpo_diagnostics.csv
exp19c_light_warp_dpo/dpo_diag/lambda020_dpo_diagnostics.csv
```

Training outcome:

- all four continuations reached 500 steps;
- `base_grad_norm` stayed at `0.0`;
- `adapter_grad_norm` stayed non-zero;
- `warp_loss` stayed finite;
- no NaN/OOM/Traceback was observed.

Final logged rows:

| Variant | total_loss | dpo_loss | warp_loss | adapter_grad_norm | base_grad_norm | residual_scale | confidence_exp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lambda000 | 0.694622 | 0.693231 | 0.244141 | 0.0002088 | 0.0 | 0.5 | 2.0 |
| lambda005 | 0.695349 | 0.692895 | 0.244141 | 0.0002110 | 0.0 | 0.5 | 2.0 |
| lambda010 | 0.698138 | 0.694234 | 0.244141 | 0.0002142 | 0.0 | 0.5 | 2.0 |
| lambda020 | 0.698942 | 0.692780 | 0.244141 | 0.0002019 | 0.0 | 0.5 | 2.0 |

Diagnostic tag:

```text
IMPLEMENTATION_VALIDATED_BUT_NEGATIVE_ABLATION
```

The warp loss is connected and stable, but the DAVIS10 metric/visual gate shows
no meaningful gain over the lambda=0 continuation control.
