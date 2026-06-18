# Exp19-R0 Flow Adapter Calibration

R0 calibrates the Exp19 inference path before any further flow-adapter training.

Primary gate:

```text
adapter disabled wrapper output ~= original Exp11 evaluator output
```

If parity does not reach the configured tolerance, Exp19c warp-loss training and
Exp19d motion-aware sampling remain blocked.
