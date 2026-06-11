# Exp11 Flow-Prior Consistency DPO

Experiment name: `exp11_flow_prior_consistency_dpo_s1s2_2000_davis_pai`

Purpose: build on Exp10 and add flow / ProPainter prior / boundary consistency to address purple haze, patches, boundary discontinuity, and temporal flicker.

Current status: invalid / mislabeled / blocked.

The 2026-06-11 truth audit found that old Exp11 proxy outputs are not valid
flow-prior consistency DPO results. The proxy code did not use train-time
ProPainter prior frames/tensors for `L_prior`, and did not use optical-flow /
warp targets for `L_flow`.

The launcher writes `reports/exp11_flow_prior_implementation_audit.md` and
stops before training. Do not use old Exp11 rows as method evidence.
