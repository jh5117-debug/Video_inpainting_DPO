# Exp28 Pair C Inner8 CLI4 Failed-Final Report

Status: `FAILED_FINAL_RESOURCE_IOWAIT_RECURRED`

Pair: `pairC_inner8_cli4` (`fresh_control_C` vs `inner8_candidate`)

Training completed for both models:

- `fresh_control_C`: Stage1 2000, Stage2 2000
- `inner8_candidate`: Stage1 2000, Stage2 2000

Auto-eval was paused once for NAS iowait, resumed once, then stopped on the second iowait recurrence while GPU0 had an external task. Per CLI4 rules, no further resume is allowed.

Completed eval labels:

- `sft48000_baseline`
- `fresh_s2_1000`
- `candidate_s2_1000`

Partial label left incomplete: `fresh_stage1_1000_sft_s2`

Main reduced comparison available: `candidate_s2_1000 - fresh_s2_1000`. This is not the promotion comparison and cannot support `INNER_RADIUS_POSITIVE`.

| Metric | Mean delta | Median delta | Directional win rate |
| --- | ---: | ---: | ---: |
| PSNR | -0.125200281163 | -0.0713954147229 | 0.460000 |
| SSIM | 2.631230156e-05 | -0.000225337000632 | 0.440000 |
| strict mask PSNR | -0.125200281163 | -0.0713954147229 | 0.460000 |
| boundary PSNR | 0.00667817833553 | 0.0179115211432 | 0.520000 |
| LPIPS | 7.49545484238e-05 | 4.5352790039e-05 | 0.440000 |
| Ewarp | -0.0244557231401 | 0.00301995010966 | 0.480000 |

Interpretation:

- The completed 1000-step Stage2 comparison is negative/mixed: PSNR and strict-mask PSNR are both -0.1252 dB on average, with PSNR directional win rate 0.46.
- Boundary PSNR is nearly neutral (+0.00668 dB, win rate 0.52).
- LPIPS is slightly worse on average (+0.00007495; lower is better).
- Ewarp is slightly improved on average (-0.02446; lower is better), but this is not enough to overcome the PSNR/mask loss and incomplete full protocol.
- Outside diff remains 0.0 for completed labels.

Decision:

```text
INNER8_REDUCED_EVAL_NEGATIVE_MIXED
PAIR_C_FULL_EVAL_FAILED_FINAL_RESOURCE_IOWAIT_RECURRED
NO_INNER_RADIUS_POSITIVE
NO_SCIENTIFIC_POSITIVE
```

Evidence files:

- `reports/exp28_pairC_inner8_cli4_reduced_eval_summary.csv`
- `reports/exp28_pairC_inner8_cli4_reduced_eval_paired_deltas.csv`
- `reports/exp28_pairC_inner8_cli4_failed_final_iowait_recurred.json`
- `reports/exp28_pairC_inner8_cli4_metrics/`
