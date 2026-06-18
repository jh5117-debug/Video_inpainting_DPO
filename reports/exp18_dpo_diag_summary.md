# Exp18 DPO Diagnostic Summary

Date: 2026-06-18

## Inputs

- Exp18a: `exp18_multiframe_propagation_gated_dpo/dpo_diag/exp18a_prop_only_stage1_500_dpo_diagnostics.csv`
- Exp18b: `exp18_multiframe_propagation_gated_dpo/dpo_diag/exp18b_prop_gen_stage1_500_dpo_diagnostics.csv`
- Exp18c: `exp18_multiframe_propagation_gated_dpo/dpo_diag/exp18c_oracle_stage1_500_dpo_diagnostics.csv`

All three Stage1-500 gates completed on PAI with finite losses and checkpoints.

## Final-Step Snapshot

| Variant | loss | dpo_loss | L_prop | L_gen | L_boundary | loser_dominant | prop coverage | prop conf mean | prop PSNR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Exp18a prop-only | 0.5519 | 0.4656 | 0.6251 | 0.3164 | 0.1725 | 1.0000 | 0.0047 | 0.0062 | 14.9414 |
| Exp18b prop+gen | 0.5911 | 0.5170 | 0.3182 | 0.2351 | 0.1385 | 1.0000 | 0.0300 | 0.0355 | 25.5840 |
| Exp18c oracle | 0.4954 | 0.4173 | 0.4025 | 0.2820 | 0.2301 | 1.0000 | 0.9648 | 0.9538 | 31.5914 |

## Mean Diagnostic Reading

| Variant | mean loss | mean dpo_loss | mean loser_dominant | mean prop coverage | mean prop conf | mean prop PSNR |
|---|---:|---:|---:|---:|---:|---:|
| Exp18a prop-only | 0.6409 | 0.5712 | 0.9412 | 0.0277 | 0.0316 | 23.4680 |
| Exp18b prop+gen | 0.6571 | 0.5726 | 0.9216 | 0.0228 | 0.0273 | 24.2600 |
| Exp18c oracle | 0.6642 | 0.5826 | 0.9608 | 0.9463 | 0.9400 | 23.7120 |

## Labels

- Exp18a: `NON_ORACLE_SPARSE_CONFIDENCE`, `LOSER_DOMINANT`, `NEGATIVE_ABLATION`
- Exp18b: `NON_ORACLE_SPARSE_CONFIDENCE`, `LOSER_DOMINANT`, `NEGATIVE_ABLATION`
- Exp18c: `ORACLE_UPPER_BOUND_NEGATIVE`, `LOSER_DOMINANT`, `DIAGNOSTIC_ONLY`

## Interpretation

The non-oracle propagation cache is usable but sparse. Exp18a/Exp18b see only about 2%-3% average hard propagation coverage during training diagnostics, with many batches having near-zero reliable propagation pixels.

The oracle diagnostic is more important: Exp18c has high oracle coverage and high confidence, but it still does not beat Exp11 on DAVIS10. That means the current formulation should not be expanded automatically. The bottleneck is not only confidence estimation; the added propagation/generation latent losses also appear to disturb the current best Exp11 balance.

Current decision:

```text
Do not run Exp18 Stage1 1000, full cache, Stage1 2000, or Stage2.
Keep Exp18 as an exploratory / negative ablation unless the loss design is changed.
```
