# Exp28 Registry Status

Updated: 2026-06-26

Status: `CLI4_WAVE_RUNNING_PAIRB_REDUCED_METRIC_MIXED_NO_POSITIVE`

Pairs:

| Pair | Control | Candidate | Status |
| --- | --- | --- | --- |
| A | `fresh_control_A` | `inner2_candidate` | training complete; eval `FAILED_FINAL`; no radius decision |
| B | `fresh_control_B` | `inner4_candidate` | training/eval complete; reduced-metric mixed; visual assets 50/50; sampled review mixed |
| C | `fresh_control_C` | `inner8_candidate` | running; control Stage1 complete, Stage2 running |

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

Reasons: VFID/TC unavailable, Stage1-hybrid 2000 negative/mixed, visual review is partial, and Pair C is still running.
