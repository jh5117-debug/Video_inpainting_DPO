# Registry: Exp11 Flow-Prior Consistency DPO

Canonical experiment folder: `exp11_flow_prior_consistency_dpo/`

Status: valid as `Exp11-proxy` only; blocked as real flow-prior consistency.

Audited label:

```text
Exp11-proxy: frozen-ref prior + boundary + temporal residual proxy DPO
```

The existing Stage1/Stage2 run is complete and has dpo_diag, so no retraining is
needed for the proxy result. It is not a valid real flow-prior consistency DPO
method result because the audit found no train-time ProPainter-prior target and
no optical-flow warp consistency target.
