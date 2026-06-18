# Exp19c Motion-Bin Summary

Motion bins are inherited from the Exp19-R0 DAVIS10 motion-score audit.

Key observation: Exp19c does not produce a reliable high-motion gain. The
highest-motion subset has the lowest Ewarp for `lambda000`, not for a positive
warp lambda.

| Motion bin | Best Ewarp label | Exp11 Ewarp | Best Ewarp | Note |
| --- | --- | ---: | ---: | --- |
| low | exp19b | 6.492636 | 6.492350 | best is pre-warp Exp19b, not Exp19c |
| medium | exp19b | 4.233873 | 4.233863 | tiny tie-level movement |
| high | lambda000 | 14.125456 | 14.125176 | lambda>0 does not beat continuation control |

Positive-gate conclusion:

```text
FAIL
```

The high-motion subset does not support moving to Exp19d motion-aware sampling.
