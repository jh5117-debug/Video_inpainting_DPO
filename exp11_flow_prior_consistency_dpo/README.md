# Exp11 Flow-Prior Consistency DPO

Experiment name: `exp11_flow_prior_consistency_dpo_s1s2_2000_davis_pai`

Purpose: build on Exp10 and add flow / ProPainter prior / boundary consistency to address purple haze, patches, boundary discontinuity, and temporal flicker.

Current status: valid as `Exp11-proxy` only; blocked as real flow-prior consistency.

Audited label:

```text
Exp11-proxy: frozen-ref prior + boundary + temporal residual proxy DPO
```

The 2026-06-11 existing-run audit found that Stage1 and Stage2 completed and
both have `dpo_diagnostics.csv`. The proxy code did not use train-time
ProPainter prior frames/tensors for `L_prior`, and did not use optical-flow /
warp targets for `L_flow`, so the result must not be called real flow-prior
consistency DPO.

The launcher writes `reports/exp11_flow_prior_implementation_audit.md` and
stops before new training. Do not retrain unless explicitly requested.
