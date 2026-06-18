# Status

```text
BLOCKED_AT_ARCHITECTURE_PREFLIGHT
```

Training was not launched. The shared `UNetMotionModel` path is unsafe for the
requested multi-scale flow adapter because down residuals can be double-added
when combined with a mid residual. Current best remains Exp11 outer b0.75 S2.
