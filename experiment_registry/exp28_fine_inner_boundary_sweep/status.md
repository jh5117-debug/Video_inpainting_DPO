# Exp28 Registry Status

Updated: 2026-06-27

Status: `CLI4_WAVE_STOPPED_PAIRC_FAILED_FINAL_IOWAIT_RECURRED_NO_POSITIVE`

Pairs:

| Pair | Control | Candidate | Status |
| --- | --- | --- | --- |
| A | `fresh_control_A` | `inner2_candidate` | training complete; eval `FAILED_FINAL`; no radius decision |
| B | `fresh_control_B` | `inner4_candidate` | training/eval complete; reduced-metric mixed; visual assets 50/50; sampled review mixed |
| C | `fresh_control_C` | `inner8_candidate` | training complete; reduced 1000-step eval negative/mixed; full eval `FAILED_FINAL_RESOURCE_IOWAIT_RECURRED` |

Pair B main Stage2-2000 metric result:

| Metric | Delta |
| --- | ---: |
| PSNR | +0.103389 |
| win rate | 0.62 |
| P(delta>0) | 0.9091 |
| LPIPS | +0.000181 |
| Ewarp | -0.033830 |

Promotion status:

```text
NO_INNER_RADIUS_POSITIVE
NO_SCIENTIFIC_POSITIVE
```

Pair C reduced Stage2-1000 result:

| Metric | Delta |
| --- | ---: |
| PSNR | -0.125200 |
| win rate | 0.46 |
| LPIPS | +0.000075 |
| Ewarp | -0.024456 |
| boundary PSNR | +0.006678 |

Pair C full eval was stopped after NAS iowait recurred following the allowed one resume. Completed labels are `sft48000_baseline`, `fresh_s2_1000`, and `candidate_s2_1000`; `fresh_stage1_1000_sft_s2` is partial.

Reasons: VFID/TC unavailable, Pair B Stage1-hybrid 2000 negative/mixed, Pair B visual review is partial, Pair C reduced Stage2-1000 is negative/mixed, and Pair C full eval is failed-final due resource iowait recurrence.
