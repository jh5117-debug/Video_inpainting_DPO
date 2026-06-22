# Exp23 Pair001 DPO Diagnostic Analysis

pair_id: `phaseA_scale1_pair001_outer2_corrected_outer_control_seed20260619_gpus2456`

| model | stage | rows | max grad | p95 grad | p99 grad | loser-dom mean/final | winner improvement mean | loser degradation mean | NaN/Inf |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fresh_exp11_outer_b075 | stage1 | 201 | 1043.013367 | 177.391317 | 545.185564 | 0.991294/1.000000 | 0.013979 | 0.506900 | 0 |
| fresh_exp11_outer_b075 | stage2 | 201 | 31.822927 | 14.024547 | 26.749936 | 1.000000/1.000000 | 0.005799 | 0.744613 | 0 |
| candidate_scale1_outer2_b075 | stage1 | 201 | 607.323425 | 91.772963 | 421.471628 | 0.997512/1.000000 | 0.013589 | 0.510799 | 0 |
| candidate_scale1_outer2_b075 | stage2 | 201 | 35.114224 | 15.194437 | 24.139323 | 0.998756/1.000000 | 0.005411 | 0.749645 | 0 |

Interpretation: all four stages remain heavily loser-dominant by the Exp11-aligned definition. Stage1 has large but sparse gradient spikes; no NaN/Inf tokens were recorded. Candidate does not clearly reduce loser dominance.
