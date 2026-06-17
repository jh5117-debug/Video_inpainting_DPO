# Exp18 Propagation Confidence Audit

Current status:

```text
PENDING_REAL_CACHE_RUN
```

Implemented non-oracle confidence:

```text
C_prop = source_valid * forward_backward_consistency * multi_source_agreement * source_count_score
```

Definitions:

- `source_valid`: source pixel is outside source mask.
- `forward_backward_consistency`: target->source flow plus sampled source->target flow should be close to zero.
- `multi_source_agreement`: warped RGB values from multiple source frames should agree.
- `source_count_score`: confidence increases when at least two source frames agree.

Default hard split:

```text
M_prop_hard = M * 1[C_prop > 0.5]
M_gen = M * (1 - 1[C_prop > 0.5])
```

Expected report after PAI run:

```text
reports/exp18_propagation_cache_quality_limit100.md
reports/exp18_propagation_cache_quality_limit100.csv
```

Training should not start if:

- propagation coverage `< 5%`; or
- propagated-region PSNR is low; or
- failure rate is high.

